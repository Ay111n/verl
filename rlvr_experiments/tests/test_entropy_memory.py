"""Actual native PPO equivalence for removing a zero-weight entropy graph."""
import unittest

from rlvr.trainer import verify_entropy_observation


class EntropyMemoryContracts(unittest.TestCase):
    def test_final_checkpoint_uses_completed_step_and_restores_next_counter(self):
        from types import SimpleNamespace
        from rlvr.trainer import save_completed_checkpoint_after_eval
        order = ["train", "exposure", "post_eval"]
        trainer = SimpleNamespace(global_steps=51)
        def save():
            self.assertEqual(trainer.global_steps, 50)
            order.append("checkpoint")
        trainer._save_checkpoint = save
        save_completed_checkpoint_after_eval(trainer, 50)
        self.assertEqual(trainer.global_steps, 51)
        self.assertEqual(order, ["train", "exposure", "post_eval", "checkpoint"])
        trainer.global_steps = 49
        with self.assertRaises(RuntimeError):
            save_completed_checkpoint_after_eval(trainer, 50)

    def test_final_checkpoint_failure_restores_counter(self):
        from types import SimpleNamespace
        from rlvr.trainer import save_completed_checkpoint_after_eval
        def fail():
            raise OSError("controlled checkpoint failure")
        trainer = SimpleNamespace(global_steps=51, _save_checkpoint=fail)
        with self.assertRaises(OSError):
            save_completed_checkpoint_after_eval(trainer, 50)
        self.assertEqual(trainer.global_steps, 51)

    def test_entropy_guard_preserves_old_policy_stage_events(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
        from rlvr.trainer import build_trainer_class
        trainer_type = build_trainer_class()
        trainer = trainer_type.__new__(trainer_type)
        trainer.config = SimpleNamespace(rlvr=SimpleNamespace(phase="smoke"))
        trainer.global_steps = 1
        def native(self, batch, metrics):
            metrics["actor/entropy"] = 0.3
            return batch
        with tempfile.TemporaryDirectory() as directory:
            trainer.run_dir = Path(directory)
            metrics = {}
            batch = object()
            with patch.object(PPOTrainerSync, "_compute_old_log_prob", new=native):
                self.assertIs(trainer._compute_old_log_prob(batch, metrics), batch)
            events = [json.loads(line) for path in (trainer.run_dir / "stage_events").glob("*.jsonl")
                      for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in events], ["start", "end"])
            self.assertTrue(all(row["stage"] == "old_log_prob" for row in events))
            self.assertEqual(events[-1]["status"], "complete")
            self.assertEqual(metrics["rlvr/entropy_pre_update_observed"], 1)

    def test_pre_update_entropy_is_required_and_preserved(self):
        metrics = {"actor/entropy": 0.3}
        verify_entropy_observation(metrics)
        self.assertEqual(metrics["actor/entropy"], 0.3)
        self.assertEqual(metrics["rlvr/entropy_pre_update_observed"], 1)
        for invalid in ({}, {"actor/entropy": float("nan")}, {"actor/entropy": float("inf")}):
            with self.assertRaises(RuntimeError):
                verify_entropy_observation(invalid)

    def test_native_ppo_loss_gradient_and_adamw_equal_without_zero_entropy_graph(self):
        import torch
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu
        from verl.utils.torch_functional import entropy_from_logits
        from verl.workers.config import ActorConfig
        from verl.workers.utils.losses import ppo_loss

        initial = torch.randn(10, 11, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
        outcomes = []
        for include_entropy in (True, False):
            logits = torch.nn.Parameter(initial.clone())
            optimizer = torch.optim.AdamW([logits], lr=1e-6, weight_decay=0.01)
            data = TensorDict({
                "prompts": torch.tensor([[1, 2], [3, 4]]),
                "responses": torch.tensor([[2, 3, 4], [5, 6, 7]]),
                "attention_mask": torch.ones(2, 5, dtype=torch.int64),
                "response_mask": torch.ones(2, 3, dtype=torch.int64),
                "old_log_probs": torch.tensor([[-2.0, -2.1, -2.2], [-2.2, -2.3, -2.4]], dtype=torch.float64),
                "ref_log_prob": torch.tensor([[-2.4, -2.3, -2.2], [-2.0, -2.1, -2.2]], dtype=torch.float64),
                "advantages": torch.tensor([[1.0, -1.0, 0.0], [-0.5, 0.5, 0.0]], dtype=torch.float64),
            }, batch_size=[2])
            tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=6, global_batch_size=2)
            labels = torch.arange(10).unsqueeze(1)
            output = {"log_probs": logits.log_softmax(-1).gather(-1, labels).squeeze(-1)}
            if include_entropy:
                output["entropy"] = entropy_from_logits(logits)
            config = ActorConfig(strategy="fsdp2", rollout_n=8, ppo_micro_batch_size_per_gpu=1,
                                 entropy_coeff=0.0, use_kl_loss=True,
                                 kl_loss_coef=0.001, kl_loss_type="low_var_kl", clip_ratio=0.2,
                                 clip_ratio_low=0.2, clip_ratio_high=0.2, loss_agg_mode="token-mean")
            loss, metrics = ppo_loss(config, output, data)
            loss.backward()
            gradient = logits.grad.clone()
            norm = torch.nn.utils.clip_grad_norm_([logits], 1.0)
            optimizer.step()
            outcomes.append((loss.detach(), gradient, norm, logits.detach().clone(),
                             optimizer.state[logits]["step"].clone(), metrics))
        for before, after in zip(outcomes[0][:5], outcomes[1][:5]):
            self.assertTrue(torch.equal(before, after))
        self.assertIn("actor/entropy_loss", outcomes[0][5])
        self.assertNotIn("actor/entropy_loss", outcomes[1][5])
        self.assertIn("kl_loss", outcomes[1][5])


if __name__ == "__main__":
    unittest.main()
