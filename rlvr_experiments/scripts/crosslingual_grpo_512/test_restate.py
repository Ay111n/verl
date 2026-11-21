"""Exercise the two visits without GPUs or a language model."""
from unittest import TestCase, main
from unittest.mock import patch

import torch

import restate


class RestatementLossTest(TestCase):
    def run_visit(self, visit):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        ids = [str(i) for i in range(32)]
        bo = {q: {"prompt_token_ids": [12]} for q in ids}
        targets = {q: ([12, 13], [7] * (1 + int(q) % 5)) for q in ids}
        preds = [
            dict(id=q, rollout_index=rep, prompt_token_ids=[12],
                 completion=str(rep), finish_reason="stop",
                 response_token_ids=[rep + 1], raw_logprobs=[0.0])
            for q in ids for rep in range(4)
        ]
        calls = []
        def likelihood(model, prompts, responses, chunk, temperature, raw=False):
            calls.append((prompts[0], responses[0], temperature, raw))
            if raw:
                lp = parameter * (responses[0][0] - 2) * torch.ones(1)
                return [lp], [torch.zeros(1)]
            return [-parameter * torch.ones(len(responses[0]))]

        def step(opt, params, cfg, version):
            grad = parameter.grad.item()
            opt.step()
            return dict(grad_norm=abs(grad), grad_clipped=False)

        def judge(view, text, *args):
            return {"reward": int(text == "0")}

        cfg = dict(rollouts_per_question=4, head_chunk=128,
                   likelihood_temperature=0.6, restate_temperature=1.0,
                   restate_weight=0.1, clip=0.2)
        with patch.object(restate, "response_logps_batch", side_effect=likelihood), \
             patch.object(restate, "judge", side_effect=judge), \
             patch.object(restate, "finish_update", side_effect=step) as finish:
            event = restate.update(None, optimizer, [parameter], cfg, {"policy": 0},
                                   ids, preds, bo, targets, visit)
        finish.assert_called_once()
        self.assertEqual(event["group_histogram"], [0, 32, 0, 0, 0])
        self.assertEqual(event["active_grpo_rollouts"], 128)
        self.assertTrue(all(p == [12] and t == 0.6 for p, _, t, raw in calls if raw))
        supervised = [(p, r, t) for p, r, t, raw in calls if not raw]
        return event, supervised, parameter.item()

    def test_first_visit_joint_gradient_and_causal_targets(self):
        evt, supervised, weight = self.run_visit(1)
        self.assertEqual(len(supervised), 32)
        self.assertTrue(all(p == [12, 13] and t == 1.0 and set(r) == {7}
                            for p, r, t in supervised))
        self.assertEqual(evt["target_tokens"], sum(len(r) for _, r, _ in supervised))
        self.assertAlmostEqual(evt["english_body_nll"], 1.0)
        self.assertAlmostEqual(evt["loss"], 0.1, places=6)
        self.assertAlmostEqual(weight, 1.0 - 0.375 - 0.1, places=5)

    def test_second_visit_online_only(self):
        evt, supervised, weight = self.run_visit(2)
        self.assertEqual(supervised, [])
        self.assertEqual(evt["target_tokens"], 0)
        self.assertEqual(evt["english_body_nll"], 0)
        self.assertTrue(all("target_tokens" not in g for g in evt["groups"]))
        self.assertAlmostEqual(weight, 1.0 - 0.375, places=6)


if __name__ == "__main__":
    main()
