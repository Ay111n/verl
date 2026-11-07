"""Qualify two-card execution using the first committed updates of a fresh R0.

The user's direct-run instruction changes the order of qualification, not the
algorithm or exposure contract. No update is replayed or omitted.
"""
from __future__ import annotations

import json
from pathlib import Path

from rlvr.contracts import atomic_json, sha256_file

POLICY = {'schema': 'rlvr-direct-r0-user-authorized-v3', 'training_mode': 'lora', 'mode': 'explicit_user_start_after_benchmark', 'warmup_updates': [1], 'diagnostic_updates': [], 'steady_updates': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], 'decision_after_update': None, 'stop_before_update': None, 'preexisting_hardware_gate_complete': False, 'all_committed_updates_count_towards_r0': True, 'sampling_repeat_exact_tokens': 'not_required', 'sampling_seed_order_parameters': 'strictly_frozen', 'authorization_instruction': '直接启动 50-step R0', 'unresolved_evidence': ['Full actor-update offload fraction not certified by sampled microbatches', 'Previous steady GPU telemetry coverage 94.66%, below 95% script requirement'], 'full_update_profiler_enabled': False, 'formal_16k_authorized': False}


def validate_direct_start(config, resume_from=None, hardware_gate_path=None):
    if (str(config.rlvr.phase) != "reward_dev" or config.rlvr.reward_profile != "R0"
            or config.rlvr.get("training_mode") != "lora"
            or int(config.rlvr.train_updates) != 50
            or int(config.trainer.total_training_steps) != 50):
        raise ValueError("Direct mode requires fresh LoRA reward-dev R0 with exactly 50 updates")
    if resume_from or hardware_gate_path:
        raise ValueError("Direct R0 starts from original Mixed-SFT; no resume or prior-gate override")
    if config.rlvr.measurement is not None:
        raise ValueError("Direct R0 assigns per-update measurement labels")
    from rlvr.paths import load_paths
    if str(config.actor_rollout_ref.model.path) != str(load_paths()['model_path']):
        raise ValueError("Direct R0 requires the original frozen Mixed-SFT model")


def measurement_label(update_id):
    if update_id in POLICY["warmup_updates"]:
        return "warmup"
    if update_id in POLICY["diagnostic_updates"]:
        return "diagnostic"
    return "steady"


def _snapshot_jsonl(path, output):
    raw = Path(path).read_bytes()
    # GPU telemetry can be appending concurrently. Only complete lines belong
    # to the snapshot; malformed complete lines fail rather than being skipped.
    complete = raw[:raw.rfind(b"\n") + 1]
    Path(output).write_bytes(complete)
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Raw hardware evidence must contain JSON objects")
    return rows


def qualify_committed_prefix(run_dir, completed_update):
    """Stop before update 15 unless actual R0 prefix satisfies the original gate."""
    if completed_update != POLICY["decision_after_update"]:
        return None
    from rlvr.hardware import hardware_gate
    from rlvr.launch import source_hashes, verify_data
    from omegaconf import OmegaConf

    directory = Path(run_dir).resolve()
    output = directory / "qualification"
    output.mkdir(exist_ok=True)
    report_path = directory / "hardware_gate.json"
    try:
        config = OmegaConf.create(json.loads((directory / "resolved_config.json").read_text()))
        policy = json.loads((directory / "direct_r0_policy.json").read_text())
        if policy != POLICY or config.rlvr.get("direct_r0") is not True:
            raise ValueError("Direct R0 policy changed during execution")
        validate_direct_start(config)
        verify_data(config)
        environment = json.loads((directory / "environment.json").read_text())
        if source_hashes() != environment["custom_source_hashes"]:
            raise ValueError("Extension source changed during R0")
        updates = _snapshot_jsonl(directory / "updates.jsonl", output / "updates.jsonl")
        samples = _snapshot_jsonl(directory / "gpu_samples.jsonl", output / "gpu_samples.jsonl")
        if [row["update_id"] for row in updates] != list(range(1, completed_update + 1)):
            raise ValueError("R0 qualification requires exactly the first 14 committed updates")
        plan_sha = sha256_file(config.rlvr.manifest_path)
        for row in updates:
            if (row["phase"] != "reward_dev" or row["measurement"] != measurement_label(row["update_id"])
                    or row["manifest_sha256"] != plan_sha or row["status"] != "complete"
                    or row["optimizer_steps_by_rank"] != {"0": 1, "1": 1}
                    or row["finite_update"] is not True or row["oom"] is not False
                    or row["prompt_count"] != 32 or row["group_count"] != 32
                    or row["response_count"] != 256 or row["generation_count"] != 8):
                raise ValueError("Actual R0 prefix violates frozen complete-update contract")
        pair = json.loads((directory / "gpu_selection.json").read_text())["physical_gpu_ids"]
        for index in pair:
            uuids = {row.get("gpu_uuid") for row in samples if row.get("gpu_index") == index}
            if len(uuids) != 1 or None in uuids:
                raise ValueError("GPU telemetry does not identify a stable physical GPU")
        diagnostics, trace_hashes = [], {}
        for path in sorted((directory / "traces").glob("*.offload.json")):
            row = json.loads(path.read_text())
            if row.get("update_id") not in POLICY["diagnostic_updates"]:
                raise ValueError("Unexpected diagnostic update in direct R0")
            trace = Path(row["trace_path"]).resolve()
            if trace.parent != (directory / "traces").resolve():
                raise ValueError("Diagnostic trace belongs to another run")
            diagnostics.append(row)
            trace_hashes[str(path.relative_to(directory))] = sha256_file(path)
        report = hardware_gate(updates, samples, diagnostics, pair, qualification_phase="reward_dev")
        report.update(run_dir=str(directory), qualification_mode=POLICY["mode"],
                      committed_r0_updates=completed_update,
                      continuation_from_update=15,
                      all_updates_count_towards_r0=True,
                      independent_benchmark_complete=False,
                      tp1_dp2_comparison_complete=False,
                      raw_snapshot_sha256={str(path.relative_to(directory)): sha256_file(path)
                                           for path in output.glob("*.jsonl")},
                      binding_sha256={name: sha256_file(directory / name) for name in
                                     ("environment.json", "resolved_config.json", "frozen_algorithm.json",
                                      "invocation.json", "gpu_selection.json", "direct_r0_policy.json")},
                      training_plan_sha256=plan_sha, diagnostic_metadata_sha256=trace_hashes)
    except Exception as exc:
        report = {"schema": "rlvr-hardware-gate-v1", "decision": "insufficient_evidence",
                  "r0_authorized": False, "qualification_phase": "reward_dev",
                  "qualification_mode": POLICY["mode"], "committed_r0_updates": completed_update,
                  "run_dir": str(directory), "gate_failures": [],
                  "missing_evidence": [f"{type(exc).__name__}: {exc}"]}
    atomic_json(report_path, report)
    if report["decision"] != "suitable_for_first_round":
        raise RuntimeError(f"Direct R0 hardware gate stopped before update 15: {report['decision']}; {report_path}")
    return report
