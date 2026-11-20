"""Eight on-policy responses with question- and distractor-specific English weights."""
from collections import Counter
import time

import torch
import torch.nn.functional as F

from objective import score_groups
from scripts.knowledge_grpo_intermittent_nll_v3.train import finish_update
from scripts.luffy_math_pilot.common import online_loss, response_logps_batch


def option_pairs(gold, english, tibetan, labels="ABCD"):
    n = len(english)
    assert n == len(tibetan) == 8 and gold in labels
    assert all(v is None or v in labels for v in [*english,*tibetan])
    en, bo = Counter(english), Counter(tibetan)
    question_gap = en[gold] - bo[gold]
    pairs = []
    for wrong in labels:
        if wrong == gold or not bo[wrong]:
            continue
        margin = min(question_gap,en[gold]-en[wrong])
        weight = bo[wrong]*max(margin,0)/(n*n)
        pairs.append(dict(wrong=wrong, student_votes=bo[wrong],english_votes=en[wrong],
                          option_gap=en[gold]-en[wrong],weight=weight))
    return pairs


def choice_loss(scores, labels, gold, pairs):
    return sum(p["weight"]*F.softplus(scores[labels.index(p["wrong"])]-scores[labels.index(gold)])
               for p in pairs if p["weight"]>0)


def choice_scores(model, prefix, label_ids):
    base = model.get_base_model()
    head = base.lm_head
    assert head.bias is None
    tokens = torch.tensor([prefix],device=head.weight.device)
    hidden = base.model(input_ids=tokens,use_cache=False).last_hidden_state[0,-1]
    return F.linear(hidden.float(),head.weight.index_select(0,label_ids).float())


def prepare_readout(model,tok,data,cfg,audit):
    device = next(model.parameters()).device
    labels = torch.tensor([audit["rank_label_ids"][label] for label in "ABCD"],device=device)
    prefixes = {q:tok.encode(row["bo"]["prompt"]+cfg["rank_prefix"],add_special_tokens=False)
                for q,row in data.items()}
    return prefixes, labels


def update(model,opt,params,cfg,state,ids,preds,data,teacher,readout,tag,times):
    started = time.monotonic()
    groups,items = score_groups(ids,preds,data,teacher["rewards"],cfg)
    prefixes,label_ids = readout
    opt.zero_grad(set_to_none=True)
    grpo_loss = rank_loss = weight_sum = max_mae = 0.
    checked = active_rollouts = active_questions = positive_pairs = zero_questions = 0
    for pred,adv,group in sorted(items,key=lambda item:len(item[0]["response_token_ids"])):
        if not adv:
            continue
        ts = pred["response_token_ids"]
        lp,raw = response_logps_batch(model,[pred["prompt_token_ids"]],[ts],cfg["head_chunk"],
                                      cfg["likelihood_temperature"],True)
        mae = (raw[0].cpu()-torch.as_tensor(pred["raw_logprobs"])).abs().mean().item()
        if not mae<.05:
            raise RuntimeError(f"Actor and sampling policy differ on {tag}: {mae}")
        max_mae = max(max_mae,mae)
        checked += 1
        term = online_loss(lp[0],lp[0].detach(),adv,cfg["clip"])/len(items)
        grpo_loss += term.detach().item()
        term.backward()
        active_rollouts += 1
        del lp,raw,term
    for group in groups:
        group.pop("kd_eligible")
        group.pop("kd_tokens_by_rep")
        uid = group["id"]
        gold = data[uid]["bo"]["canonical_answer"]
        student = [j["selected"] if j["finish_reason"]=="stop" else None for j in group["judgments"]]
        english = teacher["votes"][uid]
        assert group["english_correct"] == english.count(gold)
        assert group["tibetan_correct"] == student.count(gold)
        pairs = option_pairs(gold,english,student)
        group.update(english_votes=english,student_votes=student,gold=gold,rank_pairs=pairs)
        active = [p for p in pairs if p["weight"]>0]
        if not active:
            continue
        scores = choice_scores(model,prefixes[uid],label_ids)
        term = choice_loss(scores,"ABCD",gold,active)/len(groups)
        rank_loss += term.detach().item()
        (cfg["rank_weight"]*term).backward()
        group["option_scores"] = scores.detach().tolist()
        active_questions += 1
        positive_pairs += len(active)
        zero_questions += int(group["tibetan_correct"]==0)
        weight_sum += sum(p["weight"] for p in active)
        del scores,term
    updated = bool(active_rollouts or active_questions)
    stats = finish_update(opt,params,cfg,state["policy"]) if updated else dict(grad_norm=0.,grad_clipped=False)
    hist = Counter(g["tibetan_correct"] for g in groups)
    return dict(kind="online",request_tag=tag,groups=groups,question_count=len(ids),online_responses=len(items),
                optimizer_updated=updated,group_histogram=[hist[i] for i in range(9)],
                correct_responses=sum(k*v for k,v in hist.items()),truncated=sum(p["finish_reason"]=="length" for p in preds),
                output_tokens=sum(len(p["response_token_ids"]) for p in preds),
                prompt_tokens=sum(len(p["prompt_token_ids"]) for p in preds),rank_questions=active_questions,
                rank_zero=zero_questions,rank_pairs=positive_pairs,weight_sum=weight_sum,grpo_loss=grpo_loss,
                rank_loss=rank_loss,loss=grpo_loss+cfg["rank_weight"]*rank_loss,active_grpo_rollouts=active_rollouts,
                raw_logprob_mae_max=max_mae,likelihood_responses=checked,ppo_clip_fraction=0.,
                update_seconds=time.monotonic()-started,**stats,**times)
