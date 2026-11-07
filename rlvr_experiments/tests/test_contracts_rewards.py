"""CPU contract tests; no model inference, provider calls, or source-file writes."""
from __future__ import annotations
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlvr.contracts import (RunManifest, atomic_json, build_train_plan, check_exposure,
                            formal_budget_guard, stable_seed, stable_uid, unique_ids)
from rlvr.data import adapt_row, prepare_data, read_rows
from rlvr.metrics import group_metrics, summarise_records
from rlvr.reward import RLVRRewardManager, score_response
from rlvr.verifier import Candidate, parse_answer, parse_protocol, verify

from rlvr.paths import load_paths
SOURCE = Path(os.environ.get("RLVR_SOURCE_ROOT", str(load_paths()["source_root"])))
PROBE = ROOT / "tests" / "reference" / "probe_rl_candidates.py"


def test_atomic_audit_json_handles_numpy_metrics_and_rejects_nonfinite(tmp_path):
    import numpy as np
    path = tmp_path / "audit.json"
    atomic_json(path, {"count": np.int64(32), "values": np.array([1, 2]), "mean": np.float32(.5)})
    assert json.loads(path.read_text()) == {"count": 32, "values": [1, 2], "mean": .5}
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "invalid.json", {"mean": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


@pytest.fixture(scope="module")
def legacy():
    expected = PROBE.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(PROBE.read_bytes()).hexdigest() == expected, "Frozen oracle was modified"
    spec = importlib.util.spec_from_file_location("frozen_probe_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prompt,completion", [
    ("<|im_start|>assistant\n", "ལན་ A"),
    ("<|im_start|>assistant\n", "<think></think>42"),
    ("<|im_start|>assistant\n", "<think>work</think>42<|im_end|>"),
    ("<|im_start|>assistant\n<think>\n", "</think>42"),
    ("<|im_start|>assistant\n<think>\n", "work</think>42"),
    ("<|im_start|>assistant\n<think>\n", "work 42"),
    ("<|im_start|>assistant\n", "</think>A"),
    ("<|im_start|>assistant\n", "</think></think>A"),
    ("<|im_start|>assistant\n", "<think>work</think></think>42"),
    ("<|im_start|>assistant\n", "<think>a<think>b</think>42"),
])
def test_probe_protocol_parity(legacy, prompt, completion):
    assert parse_protocol(prompt, completion) == legacy.parse_protocol(prompt, completion)


def test_all_35_real_verifier_edges_and_saved_answers(legacy):
    edges = read_rows(SOURCE / "reward_dev" / "verifier_edge_cases.jsonl")
    assert len(edges) == 35
    assert sum(row["domain"] == "math" and row["answer_type"] == "fraction" for row in edges) == 19
    assert sum(row["domain"] == "knowledge" and len(row["options_bo"]) not in (4, 10) for row in edges) == 16
    saved = 0
    for row in edges:
        candidate = Candidate(row["domain"], row, b"")
        labels = tuple(row.get("options_bo", {}))
        response = f"ལན་ {row['canonical_answer']}"
        result = verify("<|im_start|>assistant\n<think>\n", f"work</think>{response}",
                        row["domain"], row["canonical_answer"], labels)
        assert result["correctness"] == 1, row["id"]
        assert result["answer_extractable"]
        for rollout in row["probe_info"]["rollouts"]:
            actual = parse_answer(candidate, rollout["final_region"])
            expected = legacy.parse_answer(legacy.Candidate(row["domain"], row, b""), rollout["final_region"])
            assert actual == expected
            assert actual[0] == rollout["answer_extractable"]
            assert actual[2] == rollout["answer_correct"]
            saved += 1
    assert saved >= 140


@pytest.mark.parametrize("answer,gold", [("0.50", "1/2"), ("-2/4", "-0.5"),
                                        ("+0.2500", "1/4"), ("5/2", "2.5")])
def test_numeric_fraction_equivalence(answer, gold):
    assert verify("assistant", answer, "math", gold)["correctness"] == 1




def test_numeric_not_float_rounded_and_zero_denominator():
    assert verify("assistant", "0.3333333333333333", "math", "1/3")["correctness"] == 0
    r = verify("assistant", "1/0", "math", "1")
    assert r["answer_extractable"] and r["correctness"] == 0
    r = verify("assistant", "J", "knowledge", "A", ("A", "B"))
    assert r["answer_extractable"] and r["correctness"] == 0


def scored(completion, **kwargs):
    kwargs.setdefault("output_tokens", 16)
    return score_response(prompt="<|im_start|>assistant\n<think>\n", completion=completion,
                          domain="math", gold_answer="42", **kwargs)






@pytest.mark.parametrize("completion,expected_reward", [("</think>42", 1.0), ("</think>41", 0.0)])
def test_r0_only_correctness_and_rejects_reserved_components(completion, expected_reward):
    result = scored(completion)
    info = result["reward_extra_info"]
    components = json.loads(info["components_json"])
    assert components == {
        "correctness": {"enabled": True, "raw": expected_reward, "contribution": expected_reward}
    }
    assert info["reward_profile"] == "R0"
    assert result["reward_score"] == info["total_reward"] == expected_reward
    for key in ("compute_cost", "prefix_value", "consistency", "structure"):
        assert scored(completion, component_flags={key: False})["reward_score"] == expected_reward
        with pytest.raises(ValueError, match="Reserved"):
            scored(completion, component_flags={key: True})
    with pytest.raises(ValueError):
        scored(completion, profile="R3")


def test_eval_seed_canonical_bytes_and_namespace():
    raw = '["rlvr-eval-seed-v1",42,"བོད",0]'.encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)
    assert stable_seed("བོད", 0) == expected
    assert stable_seed("བོད", 0, namespace="rlvr-train-seed-v1") != expected
    assert stable_seed("བོད", 1) != expected
    assert "_" not in stable_uid("བོད", 1, 0)


def actual_records(plan):
    return [{"update_id": u["update_id"], "sample_id": s["sample_id"], "group_id": s["group_id"],
             "sequence_position": s["sequence_position"], "rollout_index": j, "seed": seed,
             "update_committed": True}
            for u in plan["updates"] for s in u["samples"] for j, seed in enumerate(s["seeds"])]


def test_frozen_1600_50_32_g8_exposure_and_shuffle():
    rows = read_rows(SOURCE / "reward_dev" / "reward_dev_train.jsonl")
    plan = build_train_plan(rows)
    assert len(plan["updates"]) == 50
    assert all(len(u["samples"]) == 32 for u in plan["updates"])
    ids = [row["id"] for row in rows]
    random.Random(42).shuffle(ids)
    assert [s["sample_id"] for u in plan["updates"] for s in u["samples"]] == ids
    changed_seed = build_train_plan(rows, base_seed=43)
    assert [s["sample_id"] for u in changed_seed["updates"] for s in u["samples"]] == ids
    assert changed_seed["updates"][0]["samples"][0]["seeds"] != plan["updates"][0]["samples"][0]["seeds"]
    records = actual_records(plan)
    assert check_exposure(records, plan, range(1, 51)) == {
        "successful_updates": 50, "unique_prompts": 1600, "committed_responses": 12800,
        "generations_per_prompt": 8, "actual_matches_plan": True}
    for bad in (records[:-1], records + [records[-1]]):
        with pytest.raises(ValueError):
            check_exposure(bad, plan)
    wrong = copy.deepcopy(records)
    wrong[0]["update_committed"] = False
    with pytest.raises(ValueError, match="committed"):
        check_exposure(wrong, plan)
    wrong[0]["update_committed"] = True
    wrong[0]["seed"] += 1
    with pytest.raises(ValueError, match="Seed"):
        check_exposure(wrong, plan)


def test_missing_duplicate_ids_and_formal_guard():
    for rows in ([{}], [{"id": None}], [{"id": "a"}, {"id": "a"}]):
        with pytest.raises(ValueError):
            unique_ids(rows)
    with pytest.raises(ValueError, match="budget"):
        formal_budget_guard("formal_16k", None)
    formal_budget_guard("reward_dev", 50)


def metric_group(correct_count, gid="g", advantages=None):
    return [{"group_id": gid, "rollout_index": j, "correctness": int(j < correct_count),
             "total_reward": float(j < correct_count), "domain": "knowledge", "thinking_tokens": j,
             **({"advantages": advantages, "response_mask": [1] * len(advantages)} if advantages is not None else {})}
            for j in range(8)]


def test_nine_histogram_bins_and_actual_advantage_zero():
    rows = [r for correct in range(9) for r in metric_group(correct, f"g{correct}", [0.0] if correct in (0, 8) else [-.5, .5])]
    m = group_metrics(rows)
    assert m["correct_count_histogram"] == [1] * 9
    assert m["zero_advantage_group_rate"] == 2 / 9
    assert group_metrics(metric_group(8))["zero_advantage_group_rate"] is None
    # Equal outcome rewards do not prove zero actual advantages when KL/masking changes them.
    assert group_metrics(metric_group(8, advantages=[.2]))["zero_advantage_group_rate"] == 0
    with pytest.raises(ValueError, match="Missing/duplicate"):
        group_metrics(rows[:-1])
    summary = summarise_records(rows)
    assert summary["knowledge/responses"] == 72
    assert summary["thinking_tokens/median"] == 3.5
    assert summary["grpo/correct_count_8"] == 1


def test_frozen_plan_rejects_mutation_and_smoke_resume_prefix():
    plan = build_train_plan([{"id": str(i)} for i in range(32)], train_updates=3, phase="smoke", allow_repetition=True)
    assert len(plan["updates"]) == 3
    assert [s["sample_id"] for s in plan["updates"][0]["samples"]] == [s["sample_id"] for s in plan["updates"][2]["samples"]]
    assert plan["updates"][0]["samples"][0]["group_id"] != plan["updates"][2]["samples"][0]["group_id"]
    with tempfile.TemporaryDirectory(prefix="rlvr-test-") as d:
        path = Path(d) / "plan.json"
        atomic_json(path, plan, frozen=True)
        assert RunManifest(path).load() == plan
        with pytest.raises(ValueError, match="immutable"):
            RunManifest(path).update(train_updates=2)


def test_prepare_actual_frozen_sources_parquet_roles_and_order():
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory(prefix="rlvr-adapter-test-") as d:
        m = prepare_data(SOURCE, Path(d)/"train", evaluation_source_root=Path(os.environ.get("RLVR_EVALUATION_SOURCE_ROOT", str(load_paths()["evaluation_source_root"]))), evaluation_output_root=Path(d)/"eval")
        assert m["verification"]["development_subset_of_formal_train"]
        assert m["roles"]["formal_train"]["records"] == 16000
        assert m["roles"]["formal_validation"]["records"] == 2000
        assert m["plans"]["smoke"]["updates"] == 3
        assert m["plans"]["benchmark"]["updates"] == 11
        assert m["plans"]["diagnostic"]["updates"] == 3
        assert m["roles"]["benchmark_train"]["records"] == 352
        assert m["roles"]["diagnostic_train"]["records"] == 96
        for phase, role in (("reward_dev", "reward_dev_train"), ("smoke", "smoke_train"), ("benchmark", "benchmark_train"), ("diagnostic", "diagnostic_train")):
            plan = RunManifest(m["plans"][phase]["path"]).load()
            ids = list(dict.fromkeys(s["sample_id"] for u in plan["updates"] for s in u["samples"]))
            actual = pq.read_table(m["roles"][role]["path"]).to_pylist()
            assert [r["sample_id"] for r in actual] == ids
            assert all(r["extra_info"]["split_role"] == role for r in actual)
        # A second preparation must accept identical immutable artifacts.
        assert prepare_data(SOURCE, Path(d)/"train", evaluation_source_root=Path(os.environ.get("RLVR_EVALUATION_SOURCE_ROOT", str(load_paths()["evaluation_source_root"]))), evaluation_output_root=Path(d)/"eval") == m


