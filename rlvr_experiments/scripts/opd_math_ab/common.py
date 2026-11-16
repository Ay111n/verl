"""Exact response-position reverse KL with a shared frozen SFT teacher."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE.parent / "same_source_dpo_316"))
from paired_dpo import load_model, adapter_hash, read_rows, save, sha, runtime
sys.path.remove(str(HERE.parent / "same_source_dpo_316"))
from rlvr.paths import load_paths
from rlvr.verifier import Candidate, make_prompt
from rlvr.answer_verifier import verify_answer


def paths():
    return load_paths("opd_math_ab_l20")


def prompt_text(row):
    return make_prompt(Candidate(row["domain"], row, json.dumps(row, ensure_ascii=False)))


def render(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)


def teacher_text(row, arm):
    assert arm in ("answer", "solution")
    reference = row["reference_reasoning"] if arm == "solution" else ""
    return (
        "Use the supplied reference information to solve the Tibetan problem. "
        "Reason in English and give the final answer in Tibetan. "
        "The English solution field may be empty. Do not repeat the problem.\n\n"
        f"<english_question>\n{row['question_en']}\n</english_question>\n"
        f"<verified_final_value>\n{row['canonical_answer']}\n</verified_final_value>\n"
        f"<english_solution>\n{reference}\n</english_solution>\n\n"
        + prompt_text(row)
    )


def response_hidden(model, prompt_ids, response_ids):
    assert prompt_ids and response_ids
    base = model.get_base_model()
    token_ids = torch.tensor([prompt_ids + response_ids[:-1]],
                             device=next(base.parameters()).device)
    h = base.model(input_ids=token_ids, use_cache=False).last_hidden_state
    h = h[0, len(prompt_ids) - 1:]
    assert h.shape[0] == len(response_ids)
    return h


class ReverseKL(torch.autograd.Function):
    """Exact full vocabulary KL(p||q), chunking positions, frozen output head.

    Forward and backward recompute BF16 head projections in bounded chunks.
    Reduction is FP32 (FP64 in the double-precision correctness test).
    Teacher and output head are constant; only student hidden states get gradients.
    """
    @staticmethod
    def forward(ctx, student, teacher, weight, chunk):
        assert student.shape == teacher.shape
        assert student.ndim == 2 and student.shape[0] > 0
        assert not weight.requires_grad and not teacher.requires_grad
        ctx.save_for_backward(student, teacher, weight)
        ctx.chunk = chunk
        acc = torch.float64 if student.dtype == torch.float64 else torch.float32
        total = torch.zeros((), device=student.device, dtype=acc)
        for i in range(0, len(student), chunk):
            lp = F.log_softmax(F.linear(student[i:i+chunk], weight).to(acc), -1)
            lq = F.log_softmax(F.linear(teacher[i:i+chunk], weight).to(acc), -1)
            total += (lp.exp() * (lp - lq)).sum()
        return total / len(student)

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher, weight = ctx.saved_tensors
        acc = torch.float64 if student.dtype == torch.float64 else torch.float32
        gradient = torch.empty_like(student)
        scale = grad_output.to(acc) / len(student)
        for i in range(0, len(student), ctx.chunk):
            lp = F.log_softmax(F.linear(student[i:i+ctx.chunk], weight).to(acc), -1)
            lq = F.log_softmax(F.linear(teacher[i:i+ctx.chunk], weight).to(acc), -1)
            p = lp.exp()
            divergence = (p * (lp - lq)).sum(-1, keepdim=True)
            g = (p * (lp - lq - divergence) * scale).to(weight.dtype)
            gradient[i:i+ctx.chunk] = g @ weight
        return gradient, None, None, None


def reverse_kl(student, teacher, weight, chunk=64):
    return ReverseKL.apply(student, teacher.detach(), weight, chunk)


def score_verification(row, completion, finish_reason, output_tokens, budget):
    return verify_answer(row, prompt=prompt_text(row), completion=completion,
                         finish_reason=finish_reason, output_tokens=output_tokens,
                         max_new_tokens=budget)


def rpc(root, request, timeout=7200):
    """One synchronous request; each response includes loaded adapter identity."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = request["request_id"]
    assert "/" not in name
    request_file = root / (name + ".request.json")
    response_file = root / (name + ".response.json")
    if not request_file.exists():
        save(request_file, request)
    else:
        assert json.loads(request_file.read_text()) == request, "Changed replay request"
    deadline = time.monotonic() + timeout
    while not response_file.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(name)
        time.sleep(1)
    result = json.loads(response_file.read_text())
    if result.get("error"):
        raise RuntimeError(result["error"])
    assert result["request_id"] == name
    return result


def tensor_manifest(model):
    # Small immutable fingerprints plus exhaustive requires_grad checks.
    selected = {}
    for name, p in model.named_parameters():
        if not p.requires_grad and (
            "embed_tokens" in name or "lm_head" in name or
            "layers.0.self_attn.q_proj.base_layer.weight" in name or
            "layers.35.mlp.down_proj.base_layer.weight" in name):
            selected[name] = hashlib.sha256(
                p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert selected
    return selected


def current_identity(kl_chunk=64):
    metadata = runtime()
    metadata["chunk_size"] = kl_chunk
    return {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime": metadata}
