"""CPU contracts: deterministic order, observer isolation and actual step audit."""
import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch
import types
import sys
from pathlib import Path
from types import SimpleNamespace

from rlvr.replay import ordered_trajectory_keys, load_plan
from rlvr.rollout import EngineObserver, RequestClient, _capture, request_identity, observe_kv_cache
from rlvr.trainer import (optimizer_snapshot, verify_optimizer_events, finalize_exposure,
                          verify_eval_contract, compare_repeat_evaluation)
from rlvr.worker_hooks import record_optimizer_batch, summarize_optimizer_state, optimizer_audit_metrics


class IntegrationContracts(unittest.TestCase):
    def test_optimizer_metrics_survive_verl_dp_flatten(self):
        from itertools import chain
        from verl.utils.torch_functional import allgather_dict_into_dict
        rank_metrics = [optimizer_audit_metrics(1, 10.0 + rank) for rank in range(2)]
        def gather(destination, data, group=None):
            destination[:] = rank_metrics
        with patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.all_gather_object", side_effect=gather):
            gathered = allgather_dict_into_dict(rank_metrics[0])
        # TrainingWorker.train_mini_batch uses this exact flatten after DP gather.
        flattened = {key: list(chain.from_iterable(value)) for key, value in gathered.items()}
        self.assertEqual(flattened["rlvr_optimizer_successful_steps"], [1, 1])
        self.assertEqual(flattened["rlvr_train_batch_seconds"], [10.0, 11.0])

    def test_per_request_eval_sampling_ignores_stale_worker_config(self):
        from rlvr.rollout import request_sampling_params, sampling_contract_matches
        original = {"temperature": .6, "top_p": .95, "top_k": 20, "logprobs": False}
        config = SimpleNamespace(temperature=.6, top_p=.95, top_k=20)
        greedy = request_sampling_params(original, "greedy", config)
        self.assertEqual((greedy["temperature"], greedy["top_p"], greedy["top_k"]), (0, 1, -1))
        self.assertEqual(request_sampling_params(greedy, "sampled_g8", config), original)
        self.assertEqual(request_sampling_params(original, None, config), original)
        self.assertEqual(original["temperature"], .6)
        self.assertTrue(sampling_contract_matches({**greedy, "top_k": 0}, greedy))
        self.assertFalse(sampling_contract_matches({**original, "top_k": 0}, original))
        with self.assertRaises(ValueError):
            request_sampling_params(original, "invalid", config)

    def test_order_independent_of_arrival(self):
        samples = [{"group_id": "a"}, {"group_id": "b"}]
        snapshot = {f"{uid}_{i}_0": {} for uid in ["b", "a"] for i in reversed(range(8))}
        keys = ordered_trajectory_keys(samples, snapshot, 8)
        self.assertEqual(keys, [f"{uid}_{i}_0" for uid in ["a", "b"] for i in range(8)])
        del snapshot["a_7_0"]
        with self.assertRaises(RuntimeError):
            ordered_trajectory_keys(samples, snapshot, 8)

    def test_extra_output_rejected(self):
        snapshot = {"a_0_0": {}, "a_0_1": {}}
        with self.assertRaises(RuntimeError):
            ordered_trajectory_keys([{"group_id": "a"}], snapshot, 1)

    def test_manifest_rejects_nonconsecutive_updates_and_bad_group_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan = {"updates": [{"update_id": 2, "samples": []}]}
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_plan(path)
            plan = {"updates": [{"update_id": 1, "samples": [
                {"group_id": "bad_uid", "seeds": list(range(8))}]}]}
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_plan(path)

    def test_optimizer_actual_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = optimizer_snapshot(tmp)
            for rank in range(2):
                record_optimizer_batch(Path(tmp) / f"{rank}.jsonl", rank=rank,
                                       attempted=1, successful=1, grad_norm=1)
            self.assertEqual(len(verify_optimizer_events(tmp, previous, 2)), 2)
            previous = optimizer_snapshot(tmp)
            record_optimizer_batch(Path(tmp) / "0.jsonl", rank=0, attempted=1, successful=0, grad_norm=0)
            with self.assertRaises(RuntimeError):
                verify_optimizer_events(tmp, previous, 1)

    def test_optimizer_state_step_range(self):
        optimizer = SimpleNamespace(state={"parameter_a": {"step": 2}, "parameter_b": {"step": 2}})
        summary = summarize_optimizer_state(optimizer)
        self.assertEqual(summary["step_min"], 2)
        self.assertEqual(summary["step_max"], 2)
        self.assertEqual(summary["step_distribution"], {"2": 2})

    def test_smoke_partial_then_resume_keeps_prior_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"phase": "smoke", "num_generations": 8, "updates": []}
            for update_id in range(1, 4):
                plan["updates"].append({"update_id": update_id, "samples": [{
                    "sample_id": "same-smoke-sample", "group_id": f"group{update_id}",
                    "sequence_position": update_id - 1, "seeds": list(range(8))}]})
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            for update_id in range(1, 4):
                with (root / "updates.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps({"update_id": update_id}) + "\n")
                with (root / "rollouts.jsonl").open("a", encoding="utf-8") as file:
                    for generation in range(8):
                        file.write(json.dumps({"update_id": update_id, "sample_id": "same-smoke-sample",
                            "group_id": f"group{update_id}", "sequence_position": update_id - 1,
                            "rollout_index": generation, "seed": generation, "update_committed": True}) + "\n")
                if update_id == 2:
                    partial = finalize_exposure(root, root / "plan.json", 2)
                    self.assertEqual(partial["successful_updates"], 2)
                    self.assertTrue(partial["partial_smoke"])
            complete = finalize_exposure(root, root / "plan.json", 3)
            self.assertEqual(complete["committed_responses"], 24)
            self.assertFalse(complete["partial_smoke"])
            import csv
            with (root / "train_exposure_manifest.csv").open(newline="") as file:
                groups = list(csv.DictReader(file))
            self.assertEqual(len(groups), 3)
            self.assertEqual([int(row["generation_count"]) for row in groups], [8, 8, 8])
            self.assertEqual([int(row["update_id"]) for row in groups], [1, 2, 3])
            with (root / "train_rollout_exposure.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 24)

    def test_unique_namespaces(self):
        train = request_identity("train", "sample", 0, 1)
        self.assertNotEqual(train, request_identity("train", "sample", 1, 1))
        self.assertNotEqual(train, request_identity("eval", "sample", 0, 1))

    def test_evaluation_seed_contract_and_repeat_comparison(self):
        from rlvr.contracts import stable_seed
        records = [{"sample_id": "s", "rollout_index": i,
                    "seed": stable_seed("s", i), "seed_namespace": "rlvr-eval-seed-v1",
                    "response_token_ids": [i], "native_finish_reason": "stop",
                    "native_stop_reason": None, "total_reward": 1.0} for i in range(8)]
        self.assertTrue(verify_eval_contract(records, [{"sample_id": "s"}], 8)["sample_generation_seed_contract_passed"])
        self.assertTrue(compare_repeat_evaluation(records, records)["exact_token_seed_termination_reward_match"])
        changed = [dict(row) for row in records]
        changed[0]["response_token_ids"] = [999]
        self.assertFalse(compare_repeat_evaluation(records, changed)["exact_token_seed_termination_reward_match"])
        changed[0]["seed"] += 1
        with self.assertRaises(RuntimeError):
            verify_eval_contract(changed, [{"sample_id": "s"}], 8)

    def test_native_observation_preserves_yields(self):
        class Engine:
            async def generate(self, **kwargs):
                await asyncio.sleep(0)
                yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length", stop_reason=None,
                                                                 token_ids=[kwargs["request_id"]])])
        async def request(name):
            capture = {}
            token = _capture.set(capture)
            try:
                outputs = [item async for item in EngineObserver(Engine()).generate(
                    request_id=name, sampling_params=SimpleNamespace(max_tokens=4096))]
                self.assertEqual(len(outputs), 1)
                self.assertEqual(capture["raw_response_token_ids"], [name])
                self.assertEqual(capture["native_finish_reason"], "length")
                self.assertEqual(capture["effective_max_tokens"], 4096)
            finally:
                _capture.reset(token)
        async def main():
            await asyncio.gather(request("first"), request("second"))
        asyncio.run(main())

    def test_missing_native_kv_metrics_records_error_and_fails(self):
        reader = types.ModuleType("vllm.v1.metrics.reader")
        reader.get_metrics_snapshot = lambda: []
        modules = {name: types.ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.metrics")}
        modules["vllm.v1.metrics.reader"] = reader
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, modules), patch.dict("os.environ", {"RLVR_RUN_DIR": tmp}):
                with self.assertRaisesRegex(RuntimeError, "statistics unavailable"):
                    observe_kv_cache(SimpleNamespace())
            paths = list((Path(tmp) / "kv_metrics").glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            record = json.loads(paths[0].read_text().strip())
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["metrics"], {})

    def test_seed_injection_does_not_mutate_shared_dict(self):
        class Client:
            async def generate(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(extra_fields={})
        params = {"temperature": .6}
        client = Client()
        asyncio.run(RequestClient(client, "unique", 73, 8).generate(sampling_params=params))
        self.assertNotIn("seed", params)
        self.assertEqual(client.kwargs["sampling_params"]["seed"], 73)
        self.assertEqual(client.kwargs["request_id"], "unique")


if __name__ == "__main__":
    unittest.main()
