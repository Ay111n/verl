"""Read frozen RLVR sources and derive role-specific verl datasets without source edits."""
from __future__ import annotations
import json
from pathlib import Path
import os
import tempfile
from rlvr.contracts import (FROZEN_HASHES, SCHEMA_VERSION, SEED_VERSION, atomic_json,
                            build_train_plan, ids_hash, sha256_file, stable_seed,
                            unique_ids, validate_role_relations)
from rlvr.verifier import Candidate, make_prompt, PROMPT_VERSION


def read_rows(path):
    rows = []
    with Path(path).open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"Blank or incomplete JSONL line {path}:{line}")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row object {path}:{line}")
            rows.append(row)
    unique_ids(rows)
    return rows


def adapt_row(row, split_role, difficulty=None, source_sha256=None):
    sid = str(row["id"])
    domain = row["domain"]
    if row.get("translation_valid") is not True:
        raise ValueError(f"Translation did not pass: {sid}")
    if domain not in ("knowledge", "math"):
        raise ValueError(f"Unknown domain: {domain}")
    if not str(row.get("question_bo", "")).strip() or not str(row.get("canonical_answer", "")).strip():
        raise ValueError(f"Missing Tibetan question or gold answer: {sid}")
    options = row.get("options_bo", {})
    if domain == "knowledge" and (not options or row["canonical_answer"] not in options):
        raise ValueError(f"Invalid option mapping: {sid}")
    if domain == "math" and row["answer_type"] not in ("integer", "decimal", "fraction"):
        raise ValueError(f"Unsupported numeric answer type: {sid}")
    probe = difficulty or row.get("probe_info", {})
    extra = {"sample_id": sid, "domain": domain, "split_role": split_role,
             "difficulty_bucket": probe.get("difficulty_bucket"),
             "initial_bucket": probe.get("initial_bucket"), "answer_type": row["answer_type"],
             "option_count": len(options) if domain == "knowledge" else None,
             "option_labels": list(options), "source": row["source"],
             "gold_answer": row["canonical_answer"],
             "source_sha256": source_sha256,
             "original_reward_dev_split": row.get("reward_dev_split"),
             "eval_seeds": [stable_seed(sid, i) for i in range(8)]}
    return {"sample_id": sid, "split_role": split_role, "data_source": domain,
            "prompt": [{"role": "user", "content": make_prompt(Candidate(domain, row, b""))}],
            "ability": domain, "reward_model": {"style": "rule", "ground_truth": row["canonical_answer"]},
            "extra_info": extra}


def _balanced_prefix(rows, size):
    if size % 2:
        raise ValueError("Balanced workload requires an even prompt count")
    selected = [row for domain in ("knowledge", "math")
                for row in [r for r in rows if r["domain"] == domain][:size // 2]]
    unique_ids(selected, size)
    return selected


def _write_parquet_frozen(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    fd, tmp = tempfile.mkstemp(prefix=".adapter-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        if path.exists():
            # Content equality allows harmless Arrow serialization-version differences.
            if pq.read_table(path).to_pylist() != table.to_pylist():
                raise ValueError(f"Frozen adapter output differs: {path}")
        else:
            os.replace(tmp, path)
        if pq.read_table(path).to_pylist() != table.to_pylist():
            raise ValueError(f"Parquet readback differs: {path}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def prepare_data(source_root, output_root, *, evaluation_source_root, evaluation_output_root):
    """source_root is rl/processed_data; only derived output_root files are written."""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Derived outputs must be outside the frozen source tree")
    if evaluation_source_root is None or evaluation_output_root is None:
        raise ValueError('Configure explicit benchmark-owned evaluation source/output roots')
    evaluation_source_root, evaluation_output_root = Path(evaluation_source_root).resolve(), Path(evaluation_output_root).resolve()
    if evaluation_source_root == source_root or evaluation_output_root == output_root:
        raise ValueError('Training and evaluation roots must be separate')
    evaluation_names = {'rl_validation.jsonl', 'reward_dev/reward_dev_eval.jsonl'}
    source_paths = {name: (evaluation_source_root if name in evaluation_names else source_root) / name for name in FROZEN_HASHES}
    sources = {}
    hashes = {}
    for name, expected in FROZEN_HASHES.items():
        path = source_paths[name]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
        hashes[name] = actual
        sources[name] = read_rows(path)
    selection = {row["id"]: row for row in sources["rl_train_final_selection.jsonl"]}
    formal_ids = set(unique_ids(sources["rl_train_final.jsonl"]))
    if set(selection) != formal_ids:
        raise ValueError("Final difficulty sidecar coverage differs")
    role_rows = {
        "reward_dev_train": sources["reward_dev/reward_dev_train.jsonl"],
        "reward_dev_eval": sources["reward_dev/reward_dev_eval.jsonl"],
        "formal_train": sources["rl_train_final.jsonl"],
        "formal_validation": sources["rl_validation.jsonl"],
    }
    validation = validate_role_relations(role_rows)
    for role, count in (("reward_dev_train", 800), ("reward_dev_eval", 200),
                        ("formal_train", 8000), ("formal_validation", 1000)):
        for domain in ("knowledge", "math"):
            if sum(row["domain"] == domain for row in role_rows[role]) != count:
                raise ValueError(f"Domain quota changed: {role}/{domain}")
    role_rows.update({"smoke_train": _balanced_prefix(role_rows["reward_dev_train"], 32),
                      "smoke_eval": _balanced_prefix(role_rows["reward_dev_eval"], 32),
                      "benchmark_train": _balanced_prefix(role_rows["reward_dev_train"], 352),
                      "diagnostic_train": _balanced_prefix(role_rows["reward_dev_train"], 96)})
    source_names = {"reward_dev_train": "reward_dev/reward_dev_train.jsonl",
                    "reward_dev_eval": "reward_dev/reward_dev_eval.jsonl",
                    "formal_train": "rl_train_final.jsonl", "formal_validation": "rl_validation.jsonl",
                    "smoke_train": "reward_dev/reward_dev_train.jsonl",
                    "smoke_eval": "reward_dev/reward_dev_eval.jsonl",
                    "benchmark_train": "reward_dev/reward_dev_train.jsonl",
                    "diagnostic_train": "reward_dev/reward_dev_train.jsonl"}
    plans = {}
    for phase, role in (("reward_dev", "reward_dev_train"), ("smoke", "smoke_train"),
                         ("benchmark", "benchmark_train"), ("diagnostic", "diagnostic_train")):
        plan = build_train_plan(role_rows[role], phase=phase,
                                train_updates=3 if phase == "smoke" else None,
                                allow_repetition=phase == "smoke")
        path = output_root / ("train_plan.json" if phase == "reward_dev" else f"{phase}_train_plan.json")
        atomic_json(path, plan, frozen=True)
        plans[phase] = {"path": str(path), "sha256": sha256_file(path), "updates": plan["train_updates"]}
        by_id = {row["id"]: row for row in role_rows[role]}
        ordered_ids = list(dict.fromkeys(sample["sample_id"] for update in plan["updates"]
                                        for sample in update["samples"]))
        role_rows[role] = [by_id[sid] for sid in ordered_ids]
    role_manifest = {}
    for role, rows in role_rows.items():
        source_name = source_names[role]
        adapted = [adapt_row(row, role, selection.get(row["id"]), hashes[source_name]) for row in rows]
        role_root = evaluation_output_root if role in {"reward_dev_eval", "formal_validation", "smoke_eval"} else output_root
        path = role_root / f"{role}.parquet"
        _write_parquet_frozen(path, adapted)
        role_manifest[role] = {"path": str(path), "sha256": sha256_file(path), "records": len(rows),
                               "source_path": str(source_paths[source_name]),
                               "source_sha256": hashes[source_name],
                               "ids_sha256": ids_hash(unique_ids(rows)),
                               "order": "fixed_train_plan" if role in ("reward_dev_train", "smoke_train", "benchmark_train", "diagnostic_train") else "source_order",
                               "participates_in_parameter_updates": role.endswith("train")}
    eval_contract = {"seed_version": SEED_VERSION, "base_seed": 42,
                     "namespace": "rlvr-eval-seed-v1", "sampled_generations": 8,
                     "greedy_generations": 1, "greedy_do_sample": False,
                     "sampled_sampling": {"temperature": .6, "top_p": .95, "top_k": 20,
                                          "max_new_tokens": 4096},
                     "seed_formula": "canonical JSON [namespace,42,str(sample_id),index], SHA256 first8 BE & ((1<<63)-1)"}
    atomic_json(evaluation_output_root / "eval_seed_contract.json", eval_contract, frozen=True)
    manifest = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
                "roles": role_manifest, "plans": plans, "verification": validation,
                "formal_training_budget": None, "source_files": hashes,
                "source_files_unchanged": True, "source_roles_annotated": False}
    for name, before in hashes.items():
        if sha256_file(source_paths[name]) != before:
            raise ValueError(f"Source changed during adaptation: {name}")
    atomic_json(output_root / "manifest.json", manifest, frozen=True)
    return manifest
