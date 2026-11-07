"""CPU checks of the current policy and explicitly scoped historical prefix rules."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf
from rlvr.direct_r0 import POLICY, measurement_label, qualify_committed_prefix, validate_direct_start, _snapshot_jsonl
from rlvr.contracts import sha256_file


def config():
    return OmegaConf.create({
        "rlvr": {"phase": "reward_dev", "reward_profile": "R0", "train_updates": 50,
                 "measurement": None, "training_mode": "lora", "direct_r0": True, "manifest_path": "set-in-fixture"},
        "trainer": {"total_training_steps": 50},
        "actor_rollout_ref": {"model": {"path": "/home/nlp/data/lyl/model/qwen3-8bbo-sft-new2-mixed-checkpoint6000-merged"}}})


class DirectR0Tests(unittest.TestCase):
    def test_fresh_exact_budget_and_model_required(self):
        validate_direct_start(config())
        for key, value in (("rlvr.phase", "formal_16k"), ("rlvr.reward_profile", "unsupported"),
                           ("rlvr.train_updates", 14), ("trainer.total_training_steps", 14),
                           ("actor_rollout_ref.model.path", "/other-model"), ("rlvr.measurement", "steady")):
            candidate = config()
            OmegaConf.update(candidate, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_direct_start(candidate)
        for resume, gate in (("checkpoint", None), (None, "old_gate.json")):
            with self.assertRaises(ValueError):
                validate_direct_start(config(), resume, gate)

    def test_current_measurements_and_disabled_prefix_gate(self):
        labels = [measurement_label(step) for step in range(1, 51)]
        self.assertEqual(labels.count("warmup"), 1)
        self.assertEqual(labels.count("diagnostic"), 0)
        self.assertEqual(labels.count("steady"), 49)
        self.assertIsNone(POLICY['decision_after_update'])
        self.assertIsNone(POLICY['stop_before_update'])
        for step in range(1, 51):
            self.assertIsNone(qualify_committed_prefix("nonexistent", step))

    def test_live_snapshot_excludes_only_incomplete_last_line(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"live.jsonl", Path(folder)/"snapshot.jsonl"
            source.write_bytes(b'{"a": 1}\n{"a":')
            self.assertEqual(_snapshot_jsonl(source, output), [{"a": 1}])
            self.assertEqual(output.read_bytes(), b'{"a": 1}\n')
            source.write_bytes(b'{"a": 1}\ninvalid\n')
            with self.assertRaises(ValueError):
                _snapshot_jsonl(source, output)

    def fixture(self, directory):
        # Exercise the retained historical verifier under its original prefix policy.
        self.enterContext(mock.patch.dict(POLICY, {
            'mode': 'historical_prefix_cpu_fixture', 'warmup_updates': [1],
            'diagnostic_updates': [2, 3, 4], 'steady_updates': list(range(5, 15)),
            'decision_after_update': 14, 'stop_before_update': 15,
        }))
        plan = directory/"plan.json"
        plan.write_text('{"test": "synthetic CPU fixture"}')
        candidate = config()
        candidate.rlvr.manifest_path = str(plan)
        artifacts = {"resolved_config.json": OmegaConf.to_container(candidate),
                     "direct_r0_policy.json": POLICY,
                     "environment.json": {"custom_source_hashes": {"synthetic.py": "same"}},
                     "gpu_selection.json": {"physical_gpu_ids": [3, 7]},
                     "frozen_algorithm.json": {"synthetic": True}, "invocation.json": {"direct_r0": True}}
        for name, value in artifacts.items():
            (directory/name).write_text(json.dumps(value))
        updates = [{"update_id": step, "phase": "reward_dev", "measurement": measurement_label(step),
                    "manifest_sha256": sha256_file(plan), "status": "complete",
                    "optimizer_steps_by_rank": {"0": 1, "1": 1}, "finite_update": True, "oom": False,
                    "prompt_count": 32, "group_count": 32, "response_count": 256, "generation_count": 8}
                   for step in range(1, 15)]
        (directory/"updates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in updates))
        (directory/"gpu_samples.jsonl").write_text("".join(json.dumps({"gpu_index": i, "gpu_uuid": f"synthetic-{i}"})+"\n" for i in (3, 7)))
        return updates

    def evaluate(self, directory, decision="suitable_for_first_round"):
        with mock.patch("rlvr.launch.source_hashes", return_value={"synthetic.py": "same"}), \
             mock.patch("rlvr.launch.verify_data") as verify, \
             mock.patch("rlvr.hardware.hardware_gate", return_value={"decision": decision, "r0_authorized": decision=="suitable_for_first_round"}) as gate:
            result = qualify_committed_prefix(directory, 14)
            verify.assert_called_once()
            self.assertEqual(gate.call_args.kwargs["qualification_phase"], "reward_dev")
            self.assertEqual(len(gate.call_args.args[0]), 14)
            return result

    def test_historical_prefix_pass_records_phase_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.fixture(directory)
            report = self.evaluate(directory)
            self.assertEqual(report["committed_r0_updates"], 14)
            self.assertEqual(report["continuation_from_update"], 15)
            self.assertFalse(report["independent_benchmark_complete"])
            self.assertEqual(len(report["raw_snapshot_sha256"]), 2)
            self.assertEqual(json.loads((directory/"hardware_gate.json").read_text()), report)

    def test_historical_failed_or_unknown_hardware_decision_stops(self):
        for decision in ("technically_runs_only", "insufficient_evidence"):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                self.fixture(directory)
                with self.assertRaises(RuntimeError):
                    self.evaluate(directory, decision)
                self.assertEqual(json.loads((directory/"hardware_gate.json").read_text())["decision"], decision)

    def test_historical_corrupt_rank_manifest_label_or_duplicate_commit_stops(self):
        for fault in ("rank", "manifest", "measurement", "duplicate"):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                updates = self.fixture(directory)
                if fault == "rank": updates[0]["optimizer_steps_by_rank"] = {"0": 1, "1": 0}
                elif fault == "manifest": updates[0]["manifest_sha256"] = "wrong"
                elif fault == "measurement": updates[1]["measurement"] = "steady"
                else: updates.append(updates[-1])
                (directory/"updates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in updates))
                with self.assertRaises(RuntimeError):
                    self.evaluate(directory)
                self.assertEqual(json.loads((directory/"hardware_gate.json").read_text())["decision"], "insufficient_evidence")

    def test_historical_source_change_stops_and_gate_is_not_called(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.fixture(directory)
            with mock.patch("rlvr.launch.source_hashes", return_value={"synthetic.py": "changed"}), \
                 mock.patch("rlvr.launch.verify_data"), mock.patch("rlvr.hardware.hardware_gate") as gate:
                with self.assertRaises(RuntimeError):
                    qualify_committed_prefix(directory, 14)
                gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
