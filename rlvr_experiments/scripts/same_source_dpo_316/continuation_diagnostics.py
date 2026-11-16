"""Read-only gradient and full-pair diagnostics for a continued DPO run."""
import hashlib
import math
import re
import torch
import torch.distributed as dist
from paired_dpo import adapter_hash, pair_scores, save, terms


def cycle_lr(step, plan):
    """Repeat the exact original cycle; Adam's update counter does not restart."""
    local = (step - 1) % plan["optimizer_steps"] + 1
    warm = plan["warmup_steps"]
    factor = local / warm if local <= warm else (
        plan["optimizer_steps"] - local + 1
    ) / (plan["optimizer_steps"] - warm)
    return plan["learning_rate"] * factor


def optimizer_digest(state):
    h = hashlib.sha256()
    for index, values in sorted(state["state"].items()):
        h.update(str(index).encode())
        for name, value in sorted(values.items()):
            h.update(name.encode())
            if torch.is_tensor(value):
                h.update(str(value.dtype).encode())
                h.update(str(tuple(value.shape)).encode())
                h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def accumulate_components(components, buffers, params, denominator):
    for name, component in components.items():
        grads = torch.autograd.grad(
            component / denominator, params, retain_graph=True, allow_unused=True
        )
        for total, gradient in zip(buffers[name], grads):
            if gradient is not None:
                total.add_(gradient.detach())


def vector_dot(xs, ys):
    return float(sum((x.float() * y.float()).sum() for x, y in zip(xs, ys)))


def vector_norm(xs):
    return math.sqrt(max(0.0, vector_dot(xs, xs)))


def gradient_summary(buffers, actual, weight):
    base = [d + n for d, n in zip(buffers["dpo"], buffers["nll"])]
    aux = buffers["aux"]
    dn, nn, bn, an = (vector_norm(v) for v in (
        buffers["dpo"], buffers["nll"], base, aux
    ))
    total_norm = vector_norm(actual)
    residual = [g - b - weight * a for g, b, a in zip(actual, base, aux)]
    return {
        "dpo_norm": dn, "alpha_nll_norm": nn, "base_norm": bn,
        "raw_aux_norm": an, "weighted_aux_norm": abs(weight) * an,
        "weighted_aux_to_base": abs(weight) * an / bn if bn else None,
        "base_aux_cosine": vector_dot(base, aux) / (bn * an) if bn * an else None,
        "dpo_nll_cosine": vector_dot(buffers["dpo"], buffers["nll"]) / (dn * nn) if dn * nn else None,
        "actual_combined_norm": total_norm,
        "component_reconstruction_relative_error": vector_norm(residual) / total_norm if total_norm else None,
    }


def trace_group(row):
    nonempty = []
    for side in ("chosen", "rejected"):
        match = re.search(r"<think>(.*?)</think>", row[side]["text"], re.S)
        nonempty.append(bool(match and match.group(1).strip()))
    return { (True, True): "both_nonempty", (True, False): "chosen_only",
             (False, True): "rejected_only", (False, False): "both_empty"}[tuple(nonempty)]


def summarize_scores(rows):
    groups = {"all": rows, "initial_true_active": [r for r in rows if r["initial_true_active"]]}
    for key in ("domain", "trace_group"):
        for value in sorted({r[key] for r in rows}):
            groups[key + ":" + value] = [r for r in rows if r[key] == value]
    result = {}
    for name, subset in groups.items():
        if not subset:
            continue
        entry = {"count": len(subset)}
        for key in ("dpo", "nll", "margin", "chosen_mean", "rejected_mean",
                    "margin_delta", "true_aux", "own_aux", "true_gap", "initial_true_gap"):
            entry[key + "_mean"] = sum(r[key] for r in subset) / len(subset)
        for key in ("true_active", "own_active", "margin_increased"):
            entry[key + "_count"] = sum(r[key] for r in subset)
        gap0 = sum(r["initial_true_gap"] for r in subset)
        entry["true_gap_closure_fraction"] = 1 - sum(r["true_gap"] for r in subset) / gap0 if gap0 else None
        result[name] = entry
    return result


@torch.no_grad()
def score_epoch(model, records, targets, plan, target_key, step, out, rank, world):
    before = adapter_hash(model)
    values = []
    for i in range(rank, len(records), world):
        r, t = records[i], targets[i]
        c, j = pair_scores(model, r)
        _, aux, d = terms(c, j, t["bo_chosen"], t["bo_rejected"], t["nc"], t["nr"],
                         t[target_key], t["gate"], plan["beta"], plan["alpha"])
        c, j, margin = float(c), float(j), float(d["margin"])
        initial_gap = max(0., t["target"] - t["initial_bo_margin"]) if t["gate"] else 0.
        true_gap = max(0., t["target"] - margin) if t["gate"] else 0.
        values.append({
            "index": i, "pair_id": r["pair_id"], "id": r["id"], "domain": r["domain"],
            "trace_group": trace_group(r), "chosen_sum": c, "rejected_sum": j,
            "chosen_mean": c / t["nc"], "rejected_mean": j / t["nr"],
            "margin": margin, "margin_delta": margin - t["initial_bo_margin"],
            "margin_increased": margin > t["initial_bo_margin"],
            "dpo": float(d["dpo"]), "nll": float(d["nll"]),
            "true_aux": true_gap ** 2, "own_aux": float(aux),
            "true_gap": true_gap, "initial_true_gap": initial_gap,
            "initial_true_active": initial_gap > 0, "true_active": true_gap > 0,
            "own_active": bool(d["active"]),
        })
        if len(values) % 40 == 0 and rank == 0:
            print(f"score step={step} rank0_pairs={len(values)}", flush=True)
    assert adapter_hash(model) == before, "Read-only scoring changed adapter"
    save(out / f"scores.step{step}.rank{rank}.json", values)
    gathered = [None] * world
    dist.all_gather_object(gathered, values)
    values = sorted([r for shard in gathered for r in shard], key=lambda r: r["index"])
    assert [r["index"] for r in values] == list(range(len(records)))
    if rank == 0:
        save(out / f"scores.step{step}.json", {
            "step": step, "adapter_hash": before, "summary": summarize_scores(values), "rows": values
        })
    return values
