"""One GRPO plus gated full-vocabulary reverse-KL update at the rollout policy."""
from collections import Counter
import time

import torch

from paired import judge
from contract import advantages
from scripts.knowledge_grpo_intermittent_nll_v3.train import finish_update
from scripts.luffy_knowledge_388.batched_likelihood import response_token_logps
from scripts.luffy_math_pilot.common import online_loss
from scripts.opd_math_ab.common import response_hidden, reverse_kl


def thinking_span(tokens, finish, opening, closing, terminal):
    starts = [i for i,t in enumerate(tokens) if t == opening]
    ends = [i for i,t in enumerate(tokens) if t == closing]
    if len(starts) != 1 or len(ends) > 1:
        return None
    lo = starts[0] + 1
    if ends:
        hi = ends[0]
    elif finish == "length":
        hi = len(tokens)
    else:
        return None
    if hi <= lo or any(t in terminal for t in tokens[:hi]):
        return None
    return lo, hi


def score_groups(ids, preds, data, english, cfg):
    groups, items = [], []
    n = cfg["rollouts_per_question"]
    if len(preds) != n*len(ids):
        raise ValueError("Missing student rollout slots")
    for i,uid in enumerate(ids):
        view = data[uid]["bo"]
        samples = preds[i*n:(i+1)*n]
        judgments = []
        for rep, p in enumerate(samples):
            assert (p["id"],p["rollout_index"]) == (uid,rep) and p["prompt_token_ids"] == view["prompt_token_ids"]
            v = judge(view,p["completion"],p["finish_reason"],len(p["response_token_ids"]),cfg)
            judgments.append(dict(rep=rep,seed=p["seed"],**v))
        rewards = [v["reward"] for v in judgments]
        advs = advantages(rewards,n)
        k_en, k_bo = sum(english[uid]), sum(rewards)
        gate = k_en > k_bo
        groups.append(dict(id=uid,rewards=rewards,advantages=advs,judgments=judgments,
            verifier=cfg["verifier_version"],english_rewards=english[uid],english_correct=k_en,
            tibetan_correct=k_bo,kd_eligible=gate,kd_tokens_by_rep=[0]*n))
        items.extend((p,adv,groups[-1]) for p,adv in zip(samples,advs,strict=True))
    return groups,items


def update(model, opt, params, cfg, state, ids, preds, data, english, audit, terminal, tag, times):
    start = time.monotonic()
    groups,items = score_groups(ids,preds,data,english,cfg)
    size = len(items)
    w = cfg["kd_weight"]
    grpo_loss = kd_loss = 0.
    max_mae = 0.
    checked = active_tokens = clipped_tokens = 0
    head = model.get_base_model().lm_head
    assert not head.weight.requires_grad and head.bias is None and cfg["kd_temperature"] == 1.
    opt.zero_grad(set_to_none=True)
    for pred,adv,group in sorted(items,key=lambda x:len(x[0]["response_token_ids"])):
        uid, rep = pred["id"], pred["rollout_index"]
        tokens = pred["response_token_ids"]
        span = None
        if group["kd_eligible"] and w:
            span = thinking_span(tokens,pred["finish_reason"],audit["think_open"],audit["think_close"],terminal)
        audit_policy = bool(adv) or state["cursor"] == 0
        if not span and not audit_policy:
            continue
        if span:
            lo,hi = span
            en = data[uid]["en"]["prompt_token_ids"]
            model.eval()
            try:
                with model.disable_adapter(), torch.no_grad():
                    teacher = response_hidden(model,en+tokens[:lo],tokens[lo:hi]).detach()
            finally:
                model.train()
        with torch.set_grad_enabled(bool(adv or span)):
            if audit_policy:
                hidden = response_hidden(model,pred["prompt_token_ids"],tokens)
                target = torch.as_tensor(tokens,device=hidden.device)
                lp,raw = response_token_logps(head,hidden,target,cfg["head_chunk"],cfg["likelihood_temperature"],True)
                mae = (raw.detach().cpu()-torch.as_tensor(pred["raw_logprobs"])).abs().mean().item()
                if not mae < .05:
                    raise ValueError(f"Actor versus rollout policy mismatch {tag} {uid}/{rep}: {mae}")
                max_mae = max(mae,max_mae)
                checked += 1
            else:
                hidden = response_hidden(model,pred["prompt_token_ids"]+tokens[:lo],tokens[lo:hi])
            terms = []
            if adv:
                # A single update uses the exact pre-update likelihood for every rollout.
                old_lp = lp.detach()
                ratio = (lp.detach() - old_lp).exp()
                clipped_tokens += ((ratio - 1).abs() > cfg["clip"]).sum().item()
                term = online_loss(lp,old_lp,adv,cfg["clip"])/size
                grpo_loss += term.detach().item()
                active_tokens += len(tokens)
                terms.append(term)
            if span:
                aligned = hidden[lo:hi] if audit_policy else hidden
                term = reverse_kl(aligned,teacher,head.weight,cfg["kd_chunk"])/size
                if not torch.isfinite(term):
                    raise FloatingPointError(f"Non-finite distillation on {uid}/{rep}")
                kd_loss += term.detach().item()
                terms.append(w*term)
                group["kd_tokens_by_rep"][rep] = hi-lo
                del aligned,teacher
            if terms:
                sum(terms).backward()
            del hidden,terms
            if audit_policy:
                del lp,raw,target
    kd_tokens = sum(sum(g["kd_tokens_by_rep"]) for g in groups)
    active = bool(active_tokens or kd_tokens)
    stats = finish_update(opt,params,cfg,state["policy"]) if active else dict(grad_norm=0.,grad_clipped=False)
    correct = Counter(g["tibetan_correct"] for g in groups)
    return dict(kind="online", request_tag=tag, groups=groups, question_count=len(ids), online_responses=size,
        group_histogram=[correct[k] for k in range(cfg["rollouts_per_question"]+1)],
        correct_responses=sum(k*v for k,v in correct.items()), truncated=sum(p["finish_reason"]=="length" for p in preds),
        output_tokens=sum(len(p["response_token_ids"]) for p in preds),
        prompt_tokens=sum(len(p["prompt_token_ids"]) for p in preds), optimizer_updated=active,
        loss=grpo_loss+w*kd_loss,grpo_loss=grpo_loss,kd_loss=kd_loss,kd_tokens=kd_tokens,
        kd_eligible=sum(g["kd_eligible"] for g in groups), raw_logprob_mae_max=max_mae,
        likelihood_responses=checked, ppo_clip_fraction=clipped_tokens/max(active_tokens,1), update_seconds=time.monotonic()-start, **stats, **times)
