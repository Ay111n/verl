"""Rebuild a hardware decision from raw update/telemetry/CPU+CUDA trace files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvr.hardware import hardware_gate
from rlvr.profiling import analyze_trace


def read_jsonl(path, missing=None):
    if not path.exists():
        if missing is not None:
            missing.append(f"Raw JSONL evidence missing: {path}")
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")
                records.append(record)
            except ValueError as error:
                if missing is None:
                    raise
                missing.append(f"Invalid raw JSONL evidence {path}:{number}: {error}")
    return records


def normalize_updates(records, stage_records):
    updates = [dict(row) for row in records if row.get("event") in (None, "update_complete", "update_end")]
    durations = {}
    starts = {}
    for row in stage_records:
        aliases = {"old_log_prob": "old_logprob", "ref_log_prob": "ref_logprob",
                   "update_actor": "actor_update", "update_weights": "weight_sync", "adv": "advantage"}
        stage = aliases.get(row.get("stage"), row.get("stage"))
        key = (row.get("update_id"), stage, row.get("rank"), row.get("pid"))
        if row.get("event") == "start":
            starts[key] = row
        elif row.get("event") == "end" and row.get("status") == "complete":
            value = row.get("duration_s")
            if value is None and key in starts:
                value = (row["monotonic_ns"]-starts[key]["monotonic_ns"])/1e9
            if value is not None:
                update_stage = (key[0], key[1])
                durations.setdefault(update_stage, []).append(value)
    for update in updates:
        stages = update.setdefault("stage_seconds", {})
        for (identifier, stage), values in durations.items():
            if identifier == update.get("update_id") and stages.get(stage) is None:
                # Parallel workers' durations are never added. This is the
                # max observed rank duration, not summed device-time.
                stages[stage] = max(values)
    return updates


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_json(path, missing):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload
    except (OSError, ValueError, TypeError) as error:
        missing.append(f"Missing or invalid binding evidence {path}: {error}")
        return None


def _run_binding(directory, gpu_ids, missing, diagnostic=False):
    """Bind measurements to completed runs, actual GPUs and immutable data artifacts."""
    artifacts = {name: _required_json(directory / f"{name}.json", missing) for name in
                 ("status", "environment", "gpu_selection", "frozen_algorithm", "resolved_config")}
    if any(value is None for value in artifacts.values()):
        return artifacts
    if artifacts["status"].get("status") != "complete":
        missing.append(f"Run did not complete: {directory}")
    selection = artifacts["gpu_selection"].get("physical_gpu_ids")
    if selection != list(gpu_ids):
        missing.append(f"Run GPU binding differs from requested ordered pair: {directory}")
    environment = artifacts["environment"]
    for key in ("python", "executable", "verl_commit", "packages", "custom_source_hashes"):
        if key not in environment:
            missing.append(f"Environment evidence missing {key}: {directory}")
    samples = read_jsonl(directory / "gpu_samples.jsonl", missing)
    uuids = {}
    for index in gpu_ids:
        observed = {row.get("gpu_uuid") for row in samples if row.get("gpu_index") == index}
        if len(observed) != 1 or None in observed or not observed:
            missing.append(f"GPU {index} has no stable UUID telemetry: {directory}")
        else:
            uuids[str(index)] = next(iter(observed))
    artifacts["gpu_uuids"] = uuids
    config = artifacts["resolved_config"]
    try:
        data, rlvr = config["data"], config["rlvr"]
        required_data = ("train_batch_size", "gen_batch_size", "max_prompt_length", "max_response_length",
                         "shuffle", "seed", "apply_chat_template_kwargs")
        required_rlvr = ("prompts_per_update", "num_generations", "base_seed", "reward_profile", "component_flags")
        artifacts["data_fields"] = {key: data[key] for key in required_data}
        artifacts["rlvr_fields"] = {key: rlvr[key] for key in required_rlvr}
        runtime_env = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
        # Run paths and profiling controls differ; execution-affecting flags do not.
        artifacts["execution_env"] = {key: value for key, value in runtime_env.items()
                                      if key.startswith(("VLLM_", "NCCL_", "CUDA_", "TORCH_"))}
        manifest_path, plan_path = Path(rlvr["data_manifest_path"]), Path(rlvr["manifest_path"])
        manifest = _required_json(manifest_path, missing)
        plan = _required_json(plan_path, missing)
        if manifest is None or plan is None:
            return artifacts
        artifacts["data_source_contract"] = {key: manifest[key] for key in
                                             ("schema_version", "prompt_version", "source_files")}
        artifacts["data_manifest_sha256"] = _sha256(manifest_path)
        phase = "diagnostic" if diagnostic else "benchmark"
        role = manifest["roles"][f"{phase}_train"]
        if list(map(str, data["train_files"])) != [role["path"]] or _sha256(role["path"]) != role["sha256"]:
            missing.append(f"Prepared {phase} data differs from its frozen role: {directory}")
        artifacts["training_source_sha256"] = role["source_sha256"]
        expected_plan = manifest["plans"][phase]
        plan_hash = _sha256(plan_path)
        if str(plan_path) != expected_plan["path"] or plan_hash != expected_plan["sha256"]:
            missing.append(f"Frozen {phase} training plan mismatch: {directory}")
        artifacts["plan_fields"] = {key: plan[key] for key in
                                     ("schema_version", "seed_version", "base_seed", "dataset_order_seed",
                                      "prompts_per_update", "num_generations", "shuffle")}
        updates = normalize_updates(read_jsonl(directory / "updates.jsonl", missing), [])
        budget = rlvr["train_updates"]
        if budget != plan["train_updates"] or sorted(row.get("update_id") for row in updates) != list(range(1, budget+1)):
            missing.append(f"Run update IDs/budget differ from frozen plan: {directory}")
        for row in updates:
            if row.get("manifest_sha256") != plan_hash or row.get("status") != "complete" \
                    or row.get("prompt_count") != 32 or row.get("response_count") != 256 \
                    or row.get("group_count") != 32 or row.get("generation_count") != 8 \
                    or row.get("finite_update") is not True or row.get("oom") is not False \
                    or row.get("optimizer_steps_by_rank") != {"0": 1, "1": 1}:
                missing.append(f"Run has an unverified committed update {row.get('update_id')}: {directory}")
        artifacts["committed_update_ids"] = {row["update_id"] for row in updates}
    except (KeyError, OSError, ValueError, TypeError) as error:
        missing.append(f"Incomplete configuration/data binding for {directory}: {error}")
    return artifacts


def build_report(run_dir, gpu_ids, diagnostic_dirs=()):
    directory = Path(run_dir)
    binding_missing = []
    update_path = directory / "updates.jsonl"
    samples = read_jsonl(directory / "gpu_samples.jsonl", binding_missing)
    stages = []
    for path in sorted((directory / "stage_events").glob("*.jsonl")):
        stages.extend(read_jsonl(path, binding_missing))
    updates = normalize_updates(read_jsonl(update_path, binding_missing), stages)
    diagnostics = []
    bindings = {}
    bindings[str(directory)] = _run_binding(directory, gpu_ids, binding_missing)
    trace_directories = [directory, *map(Path, diagnostic_dirs)]
    for diagnostic in trace_directories[1:]:
        left = bindings[str(directory)]
        right = bindings[str(diagnostic)] = _run_binding(diagnostic, gpu_ids, binding_missing, diagnostic=True)
        for key in ("environment", "gpu_selection", "gpu_uuids", "frozen_algorithm", "data_fields",
                    "rlvr_fields", "execution_env", "data_source_contract", "data_manifest_sha256",
                    "training_source_sha256", "plan_fields"):
            if key not in left or key not in right or left[key] != right[key]:
                binding_missing.append(f"Diagnostic/candidate {key} mismatch: {diagnostic}")
        for key in ("actor_rollout_ref", "algorithm"):
            if (left.get("resolved_config") or {}).get(key) != (right.get("resolved_config") or {}).get(key):
                binding_missing.append(f"Diagnostic/candidate {key} mismatch: {diagnostic}")
    for folder in trace_directories:
        for path in sorted((folder / "traces").glob("*.offload.json")):
            metadata = _required_json(path, binding_missing)
            if not metadata:
                continue
            try:
                trace = Path(metadata["trace_path"])
                if not trace.exists():
                    trace = path.parent / trace.name
                if not trace.exists():
                    binding_missing.append(f"Diagnostic raw trace missing: {trace}")
                    continue
                if metadata["update_id"] not in bindings[str(folder)].get("committed_update_ids", set()):
                    binding_missing.append(f"Diagnostic trace is not bound to a committed update: {path}")
                diagnostics.append(analyze_trace(trace, metadata["update_id"], metadata["rank"],
                                                  metadata.get("offload_policy")))
            except (OSError, ValueError, TypeError, KeyError) as error:
                binding_missing.append(f"Invalid diagnostic trace evidence {path}: {error}")
    report = hardware_gate(updates, samples, diagnostics, gpu_ids, stages)
    report["missing_evidence"].extend(binding_missing)
    if binding_missing:
        report.update(decision="insufficient_evidence", r0_authorized=False)
    report["binding_evidence"] = {folder: {key: value for key, value in binding.items()
                                           if key not in ("resolved_config", "committed_update_ids")}
                                  for folder, binding in bindings.items()}
    report["run_dir"] = str(directory.resolve())
    report["input_paths"] = {"updates": str(update_path),
                             "gpu_samples": str(directory / "gpu_samples.jsonl")}
    report["diagnostics"] = diagnostics
    report['diagnostic_run_dirs'] = [str(Path(folder).resolve()) for folder in diagnostic_dirs]
    report["stage_timings_definition"] = "Reported trainer elapsed stages, else max observed parallel-rank elapsed stage"
    return report


def summarize_candidates(reports):
    """Only actually accepted candidates participate in throughput ranking."""
    suitable = [report for report in reports if report.get("r0_authorized")]
    chosen = min(suitable, key=lambda report: report["update_median_seconds"]) if suitable else None
    return {"schema": "rlvr-hardware-candidates-v1",
            "decision": "candidate_available" if chosen else "no_accepted_candidate",
            "selected_candidate_run_dir": chosen["run_dir"] if chosen else None,
            "selection_rule": "Lowest median complete-update time among independently gate-passed candidates",
            "selection_requires_frozen_algorithm_contract": True,
            "candidates": [{"run_dir": r["run_dir"], "decision": r["decision"],
                            "r0_authorized": r["r0_authorized"],
                            "update_median_seconds": r["update_median_seconds"],
                            "update_p95_seconds": r["update_p95_seconds"],
                            "p95_median_ratio": r["p95_median_ratio"],
                            "rollout_tokens_per_second": r["rollout_tokens_per_second"],
                            "rollout_responses_per_second": r["rollout_responses_per_second"],
                            "average_response_tokens": r["average_response_tokens"],
                            "two_gpu_mean_active_time_percent": r["two_gpu_mean_active_time_percent"],
                            "peak_memory_used_mib_by_gpu": r["peak_memory_used_mib_by_gpu"],
                            "missing_evidence": r["missing_evidence"],
                            "gate_failures": r["gate_failures"]} for r in reports]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="+")
    parser.add_argument("--gpu-ids", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument('--diagnostic-dir', action='append', type=Path, default=[])
    args = parser.parse_args()
    if args.diagnostic_dir and len(args.run_dir) != 1:
        parser.error('--diagnostic-dir requires exactly one throughput candidate')
    reports = [build_report(directory, args.gpu_ids, args.diagnostic_dir) for directory in args.run_dir]
    for directory, candidate in zip(args.run_dir, reports):
        (directory / "hardware_gate.json").write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = reports[0] if len(reports) == 1 else summarize_candidates(reports)
    destination = args.output or args.run_dir[0] / ("hardware_gate.json" if len(reports) == 1
                                                 else "hardware_candidates.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    authorized = report.get("r0_authorized", report.get("selected_candidate_run_dir") is not None)
    print(json.dumps({"decision": report["decision"], "r0_authorized": authorized,
                      "missing_evidence": report.get("missing_evidence", []),
                      "gate_failures": report.get("gate_failures", []),
                      "report": str(destination.resolve())}, ensure_ascii=False))
    return 0 if authorized else 2


if __name__ == "__main__":
    raise SystemExit(main())
