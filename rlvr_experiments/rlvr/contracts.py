"""Frozen roles, deterministic group seeds, and auditable exposure contracts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from collections import Counter

SCHEMA_VERSION = "rlvr-experiments-v1"
SEED_VERSION = "sha256-canonical-json-first8-be-mask63-v1"
ROLES = ("reward_dev_train", "reward_dev_eval", "formal_train", "formal_validation",
         "smoke_train", "smoke_eval", "benchmark_train", "diagnostic_train")
FROZEN_HASHES = {
    "rl_train_final.jsonl": "f447f51feeed347908a2aa53cba82d49d49f8f4d176e76be6025839276892614",
    "rl_train_final_selection.jsonl": "6cf09cb563800c3c6cf0d12d88efb6f3f5cd5ce19a6ee5a9c3b229386098d045",
    "rl_validation.jsonl": "5f86e140eed67f9d5ccbb7276b388be5a2e8d9f434d812a1394427b9ee894e7a",
    "reward_dev/reward_dev_train.jsonl": "5278e1ea4af86e9e2591fceb6dc7c5ab8b26db11db44dc9a0b51e7f89bfbeea0",
    "reward_dev/reward_dev_eval.jsonl": "fd00691d4c86d44090a901ba73402abaa472a35dbcdaf585adc1ab0610890d95",
    "reward_dev/verifier_edge_cases.jsonl": "b615cc07e065cb5df7da82a106d52651ca03aea6cda6899c7c528a31ef6d1b0a",
}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256("".join(f"{v}\n" for v in sorted(map(str, ids))).encode()).hexdigest()


def stable_seed(sample_id, rollout_index, base_seed=42, namespace="rlvr-eval-seed-v1"):
    if not isinstance(rollout_index, int) or rollout_index < 0:
        raise ValueError("rollout_index must be a nonnegative integer")
    raw = canonical_json([namespace, base_seed, str(sample_id), rollout_index]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


derive_seed = stable_seed


def stable_uid(sample_id, update_id=None, sequence_position=None):
    payload = ["rlvr-group-v1", str(sample_id), update_id, sequence_position]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_json(path, payload, *, frozen=False):
    path = Path(path)
    def native(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Unsupported audit JSON value: {type(value).__name__}")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                       default=native, allow_nan=False) + "\n").encode("utf-8")
    if frozen and path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".rlvr-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class RunManifest:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def updates(self):
        return self.load()["updates"]

    def update(self, **fields):
        """Only mutable run-state manifests may be updated; plans stay immutable."""
        payload = self.load()
        if "updates" in payload:
            raise ValueError("An immutable training plan cannot be updated")
        payload.update(fields)
        atomic_json(self.path, payload)
        return payload


def unique_ids(rows, expected_count=None):
    values = [row.get("sample_id", row.get("id")) for row in rows]
    if any(v is None or str(v) == "" for v in values):
        raise ValueError("Missing sample IDs")
    ids = list(map(str, values))
    if len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate sample IDs")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} records, got {len(ids)}")
    return ids


def validate_role_relations(role_rows):
    counts = {"reward_dev_train": 1600, "reward_dev_eval": 400,
              "formal_train": 16000, "formal_validation": 2000}
    ids = {role: set(unique_ids(role_rows[role], count)) for role, count in counts.items()}
    if ids["reward_dev_train"] & ids["reward_dev_eval"]:
        raise ValueError("Development train/eval overlap")
    if ids["formal_train"] & ids["formal_validation"]:
        raise ValueError("Formal train/validation overlap")
    if not (ids["reward_dev_train"] | ids["reward_dev_eval"]) <= ids["formal_train"]:
        raise ValueError("Development split must remain a subset of frozen formal training")
    for role, rows in role_rows.items():
        if role not in ROLES:
            raise ValueError(f"Unknown split_role: {role}")
        if any(row.get("translation_valid") is not True for row in rows):
            raise ValueError(f"Non-passing translation in {role}")
    return {"dev_train_eval_overlap": 0, "formal_validation_overlap": 0,
            "development_subset_of_formal_train": True}


def build_train_plan(rows, prompts_per_update=32, num_generations=8, base_seed=42,
                     train_updates=None, phase="reward_dev", allow_repetition=False):
    ids = unique_ids(rows)
    # Dataset ordering stays fixed even when a later ablation varies the model/generation seed.
    random.Random(42).shuffle(ids)
    available = len(ids) // prompts_per_update
    train_updates = available if train_updates is None else train_updates
    if train_updates <= 0 or (train_updates > available and not allow_repetition) or available == 0:
        raise ValueError("Training plan must use complete batches without unplanned repetition")
    updates = []
    for i in range(train_updates):
        samples = []
        batch_index = i % available
        for offset, sid in enumerate(ids[batch_index * prompts_per_update:(batch_index + 1) * prompts_per_update]):
            position = i * prompts_per_update + offset
            samples.append({"sample_id": sid, "group_id": stable_uid(sid, i + 1, position),
                            "sequence_position": position,
                            "seeds": [stable_seed(sid, j, base_seed, "rlvr-train-seed-v1")
                                      for j in range(num_generations)]})
        updates.append({"update_id": i + 1, "samples": samples})
    return {"schema_version": SCHEMA_VERSION, "phase": phase, "seed_version": SEED_VERSION,
            "base_seed": base_seed, "dataset_order_seed": 42,
            "shuffle": "random.Random(42).shuffle(source_order)",
            "train_updates": train_updates, "prompts_per_update": prompts_per_update,
            "num_generations": num_generations, "source_ids_sha256": ids_hash(ids),
            "source_records": len(ids), "allow_repetition": allow_repetition, "updates": updates}


def formal_budget_guard(phase, train_updates):
    if phase in ("formal", "formal_16k") and (not isinstance(train_updates, int) or train_updates <= 0):
        raise ValueError("Formal training budget is unset; formal training is not authorized")


def check_exposure(records, plan, successful_updates=None):
    """Verify actual committed rollouts against the immutable plan, never planned CSVs."""
    expected = {}
    for update in plan["updates"]:
        for sample in update["samples"]:
            for index, seed in enumerate(sample["seeds"]):
                expected[(update["update_id"], sample["sample_id"], index)] = (sample, seed)
    actual = {}
    committed = set()
    for row in records:
        if row.get("update_committed") is not True:
            raise ValueError("Exposure records must be from committed successful updates")
        key = (int(row["update_id"]), str(row["sample_id"]), int(row["rollout_index"]))
        if key in actual:
            raise ValueError(f"Duplicate actual rollout exposure: {key}")
        if key not in expected:
            raise ValueError(f"Unplanned exposure: {key}")
        sample, seed = expected[key]
        if row.get("seed") != seed or row.get("group_id") != sample["group_id"]:
            raise ValueError(f"Seed or group mismatch: {key}")
        if row.get("sequence_position") != sample["sequence_position"]:
            raise ValueError(f"Prompt order mismatch: {key}")
        actual[key] = row
        committed.add(key[0])
    if set(actual) != set(expected):
        raise ValueError(f"Incomplete actual exposure: {len(actual)}/{len(expected)}")
    wanted = {update["update_id"] for update in plan["updates"]}
    if committed != wanted or (successful_updates is not None and set(successful_updates) != wanted):
        raise ValueError("Successful update coverage differs from plan")
    count = Counter(row["sample_id"] for row in records)
    return {"successful_updates": len(committed), "unique_prompts": len(count),
            "committed_responses": len(actual), "generations_per_prompt": plan["num_generations"],
            "actual_matches_plan": True}
