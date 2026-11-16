"""Verify full vocabulary loss and its gradients against native autograd."""
import json
import torch
import torch.nn.functional as F
from common import reverse_kl

def check(dtype, device):
    torch.manual_seed(19)
    h = torch.randn(9, 17, dtype=dtype, device=device, requires_grad=True)
    teacher = torch.randn(9, 17, dtype=dtype, device=device)
    weight = torch.randn(73, 17, dtype=dtype, device=device) / 5
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    lp = F.log_softmax(F.linear(h, weight).to(acc), -1)
    lq = F.log_softmax(F.linear(teacher, weight).to(acc), -1)
    reference = (lp.exp() * (lp - lq)).sum(-1).mean()
    expected = torch.autograd.grad(reference * .37, h)[0]
    actual = reverse_kl(h, teacher, weight, 4)
    observed = torch.autograd.grad(actual * .37, h)[0]
    loss_error = float((actual - reference).abs())
    gradient_error = float((observed - expected).abs().max())
    if dtype == torch.float64:
        assert loss_error < 1e-12 and gradient_error < 1e-12
        assert torch.autograd.gradcheck(
            lambda x: reverse_kl(x, teacher, weight, 4), (h,), atol=1e-6, rtol=1e-4)
    else:
        # BF16 rounding occurs at the same head/gradient casting boundaries.
        assert loss_error < 1e-5 and gradient_error < 8e-5
    zero = reverse_kl(h, h.detach(), weight, 3)
    zg = torch.autograd.grad(zero, h)[0]
    assert abs(float(zero)) < 1e-8 and float(zg.abs().max()) < 1e-8
    assert weight.grad is None and teacher.grad is None
    return {"dtype": str(dtype), "device": device, "loss_abs_error": loss_error,
            "gradient_max_abs_error": gradient_error, "identical_distribution_zero": True}

if __name__ == "__main__":
    torch.set_num_threads(2)
    results = [check(torch.float64, "cpu"), check(torch.float32, "cpu")]
    if torch.cuda.is_available():
        results.append(check(torch.bfloat16, "cuda"))
    print(json.dumps(results), flush=True)
