import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/same_source_dpo_316"))
from continuation_diagnostics import (
    accumulate_components, cycle_lr, gradient_summary, optimizer_digest
)
from paired_dpo import terms


def test_lr_cycles_preserve_original_budget():
    plan = dict(optimizer_steps=40, warmup_steps=4, learning_rate=1e-5)
    assert [cycle_lr(s, plan) for s in range(41, 81)] == [cycle_lr(s, plan) for s in range(1, 41)]
    assert [cycle_lr(s, plan) for s in range(81, 121)] == [cycle_lr(s, plan) for s in range(1, 41)]
    assert cycle_lr(44, plan) == 1e-5
    assert cycle_lr(80, plan) == 1e-5 * (1 / 36)


def test_adam_resume_preserves_moments_and_next_update():
    p = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=.001)
    for i in range(40):
        opt.zero_grad(); ((p - i / 100) ** 2).sum().backward(); opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = torch.optim.AdamW([q], lr=.001)
    saved = copy.deepcopy(opt.state_dict())
    resumed.load_state_dict(saved)
    assert optimizer_digest(resumed.state_dict()) == optimizer_digest(opt.state_dict())
    assert int(resumed.state[q]["step"]) == 40
    for param, optimizer in [(p, opt), (q, resumed)]:
        optimizer.zero_grad(); (param.sin().sum()).backward(); optimizer.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert optimizer_digest(resumed.state_dict()) == optimizer_digest(opt.state_dict())


def test_component_probe_does_not_modify_combined_update():
    p = torch.nn.Parameter(torch.tensor([-10., -20.], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    op = torch.optim.AdamW([p], lr=.001)
    oq = torch.optim.AdamW([q], lr=.001)
    b, a, d = terms(p[0], p[1], -10., -20., 10, 10, 2., 1.)
    buffers = {name: [torch.zeros_like(p)] for name in ("dpo", "nll", "aux")}
    accumulate_components({"dpo": d["dpo"], "nll": d["nll"], "aux": a}, buffers, [p], 4)
    assert p.grad is None
    ((b + 7 * a) / 4).backward()
    result = gradient_summary(buffers, [p.grad], 7.)
    assert result["component_reconstruction_relative_error"] < 1e-12
    bq, aq, _ = terms(q[0], q[1], -10., -20., 10, 10, 2., 1.)
    ((bq + 7 * aq) / 4).backward()
    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    op.step(); oq.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_component_zero_gate_and_uneven_last_batch_scaling():
    # Two ranks handle two pairs each from the final four-pair global batch.
    all_grads = []
    for rank in range(2):
        p = torch.nn.Parameter(torch.tensor([-10., -20.], dtype=torch.float64))
        buffers = {name: [torch.zeros_like(p)] for name in ("dpo", "nll", "aux")}
        for _ in range(2):
            b, a, d = terms(p[0], p[1], -10., -20., 10, 10, 2., 0.)
            accumulate_components({"dpo": d["dpo"], "nll": d["nll"], "aux": a}, buffers, [p], 4)
            ((b + 7 * a) / 4).backward()
        assert buffers["aux"][0].abs().sum() == 0
        all_grads.append(p.grad)
    q = torch.nn.Parameter(torch.tensor([-10., -20.], dtype=torch.float64))
    b, a, _ = terms(q[0], q[1], -10., -20., 10, 10, 2., 0.)
    (b + 7 * a).backward()
    torch.testing.assert_close(sum(all_grads), q.grad, rtol=0, atol=0)
