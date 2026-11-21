"""One conditional-prefix GRPO step with an unhinted on-policy distillation term."""

from collections import Counter
import math
import time

import torch

from contract import advantages
from paired import judge
from scripts.crosslingual_grpo_512.objective import thinking_span
from scripts.knowledge_grpo_intermittent_nll_v3.train import finish_update
from scripts.luffy_math_pilot.common import online_loss, response_logps_batch
from scripts.opd_math_ab.common import response_hidden, reverse_kl


def score_batch(ids, selected, plain, initial, rescue_ids, data, teacher_prompts, cfg):
    """v3 scores the selected GRPO group; KD eligibility uses both real groups."""
    n = cfg["rollouts_per_question"]
    assert n == 4 and len(ids) == len(set(ids)) == cfg["batch_size"]
    assert len(selected) == len(plain) == len(ids) * n and set(ids) == set(initial)
    rescued = set(rescue_ids)
    assert len(rescued) == len(rescue_ids) and rescued.issubset(ids)
    groups, items = [], []
    for index, uid in enumerate(ids):
        native = data[uid]
        start = index * n
        samples = selected[start:start+n]
        assert [r["rep"] for r in initial[uid]] == list(range(n))
        native_rewards = [r["reward"] for r in initial[uid]]
        assert all(r in (0, 1) for r in native_rewards)
        is_rescued = uid in rescued
        if is_rescued:
            assert not any(native_rewards) and uid in teacher_prompts
            assert teacher_prompts[uid] != native["prompt_token_ids"]
        judgments, rewards = [], []
        for rep, (pred, independent) in enumerate(zip(samples, plain[start:start+n], strict=True)):
            assert (uid, rep) == (pred["id"], pred["rollout_index"]) == (independent["id"], independent["rollout_index"])
            assert independent["prompt_token_ids"] == native["prompt_token_ids"]
            assert (pred["prompt_token_ids"] != independent["prompt_token_ids"]) == is_rescued
            if is_rescued:
                assert pred["prompt_token_ids"] != teacher_prompts[uid]
            verdict = judge(native, pred["completion"], pred["finish_reason"], len(pred["response_token_ids"]), cfg)
            rewards.append(verdict.pop("reward"))
            judgments.append(dict(rep=rep, seed=pred["seed"], **verdict))
        if not is_rescued:
            assert rewards == native_rewards
        advs = advantages(rewards, n)
        groups.append(dict(id=uid, rewards=rewards, advantages=advs, judgments=judgments,
                           verifier=cfg["verifier_version"], assisted=is_rescued,
                           kd_eligible=bool(is_rescued and any(rewards)), kd_tokens_by_rep=[0] * n,
                           kd_kl_by_rep=[None] * n))
        items.extend(zip(samples, advs, strict=True))
    return groups, items


def accumulate_grpo(model, items, cfg, first, tag):
    """Match the prefix baseline's response means, group denominator and policy audit."""
    loss, clipped, active_tokens = 0.0, 0, 0
    errors = []
    for pred, adv in sorted(items, key=lambda item: len(item[0]["response_token_ids"])):
        if not adv and not first:
            continue
        with torch.set_grad_enabled(bool(adv)):
            values, raw = response_logps_batch(model, [pred["prompt_token_ids"]], [pred["response_token_ids"]],
                                              cfg["head_chunk"], cfg["likelihood_temperature"], return_raw=True)
            lp = values[0]
            mae = float((raw[0].cpu() - torch.as_tensor(pred["raw_logprobs"])).abs().mean())
            if not mae < .05:
                raise ValueError(f"Sampling policy differs from the actor on {tag} {pred['id']}: {mae}")
            errors.append(mae)
            if adv:
                old_lp = lp.detach()
                ratio = (lp.detach() - old_lp).exp()
                clipped += int(((ratio - 1).abs() > cfg["clip"]).sum())
                active_tokens += lp.numel()
                term = online_loss(lp, old_lp, adv, cfg["clip"]) / len(items)
                term.backward()
                loss += term.detach().item()
        del values, raw, lp
    return dict(grpo_loss=loss, active_grpo_tokens=active_tokens,
                raw_logprob_mae_max=max(errors, default=None), likelihood_responses=len(errors),
                ppo_clip_fraction=clipped / max(1, active_tokens))


def accumulate_distillation(model, groups, plain, teacher_prompts, think_open, think_close, terminal, cfg):
    """Teacher and student score exactly the same unhinted student history at each position."""
    n = cfg["rollouts_per_question"]
    w = cfg["kd_weight"]
    assert w == .1 and cfg["kd_temperature"] == 1 and cfg["kd_chunk"] > 0
    head = model.get_base_model().lm_head
    assert not head.weight.requires_grad and head.bias is None
    started = time.monotonic()
    total_kl = 0.
    active_tokens = active_reps = 0
    for index, group in enumerate(groups):
        if not group["kd_eligible"]:
            continue
        note = teacher_prompts[group["id"]]
        for rep, row in enumerate(plain[index*n:(index+1)*n]):
            tokens = row["response_token_ids"]
            span = thinking_span(tokens, row["finish_reason"], think_open, think_close, terminal)
            if span is None:
                continue
            lo, hi = span
            # The transformer receives the exact original student history, including
            # its opening marker; the marker, closing marker and answer are never targets.
            prefix = tokens[:lo]
            targets = tokens[lo:hi]
            assert len(note) + hi <= cfg["max_model_len"]
            was_training = model.training
            model.eval()
            try:
                # Keep the current adapter enabled. No teacher graph or second model.
                with torch.no_grad():
                    teacher = response_hidden(model, note + prefix, targets)
            finally:
                model.train(was_training)
            student = response_hidden(model, row["prompt_token_ids"] + prefix, targets)
            mean_kl = reverse_kl(student, teacher, head.weight, cfg["kd_chunk"])
            value = mean_kl.detach().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite reverse KL on {group['id']}/{rep}")
            (mean_kl * (w / len(plain))).backward()
            group["kd_tokens_by_rep"][rep] = hi - lo
            group["kd_kl_by_rep"][rep] = value
            total_kl += value
            active_tokens += hi - lo
            active_reps += 1
            del mean_kl, student, teacher
    return dict(kd_loss=total_kl / len(plain), kd_weight=w, kd_tokens=active_tokens,
                kd_eligible=sum(g["kd_eligible"] for g in groups), kd_responses=active_reps,
                kd_questions=sum(any(g["kd_tokens_by_rep"]) for g in groups),
                kd_seconds=time.monotonic() - started)


def online(model, opt, params, cfg, state, ids, preds, data, tag, first, times, *,
           plain, initial, rescue_ids, teacher_prompts, think_open, think_close, terminal):
    """The only optimizer step follows both gradient accumulations at one policy version."""
    started = time.monotonic()
    groups, items = score_batch(ids, preds, plain, initial, rescue_ids, data, teacher_prompts, cfg)
    opt.zero_grad(set_to_none=True)
    grpo = accumulate_grpo(model, items, cfg, first, tag)
    kd = accumulate_distillation(model, groups, plain, teacher_prompts, think_open, think_close, terminal, cfg)
    active = bool(grpo["active_grpo_tokens"] or kd["kd_tokens"])
    stats = finish_update(opt, params, cfg, state["policy"]) if active else dict(grad_norm=0., grad_clipped=False)
    histogram = Counter(sum(g["rewards"]) for g in groups)
    return dict(kind="online", groups=groups, question_count=len(ids), online_responses=len(preds),
                correct_responses=sum(k*v for k,v in histogram.items()),
                group_histogram=[histogram[k] for k in range(cfg["rollouts_per_question"]+1)],
                output_tokens=sum(len(p["response_token_ids"]) for p in preds),
                prompt_tokens=sum(len(p["prompt_token_ids"]) for p in preds),
                truncated=sum(p["finish_reason"] == "length" for p in preds), optimizer_updated=active,
                loss=grpo["grpo_loss"] + kd["kd_weight"] * kd["kd_loss"],
                request_tag=tag, update_seconds=time.monotonic() - started, **grpo, **kd, **stats, **times)
