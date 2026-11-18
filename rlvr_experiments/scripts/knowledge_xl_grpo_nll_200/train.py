"""One optimizer, two distinct losses; advance the queue in the checkpoint transaction."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from contract import (CONFIG, MODEL_OUT, ROOT, RUN, advantages, choose_refs, epoch_batches,
                      pilot_ids, read, reference_tokens, refresh_pool, student_reward)
from checkpointing import commit, restore, snapshots
from sampling import generate, sync
from rlvr.answer_verifier import verify_answer
from scripts.luffy_math_pilot.common import (adapter_hash, load_model, online_loss, read_rows,
                                            response_logps_batch, save)


def finish_update(opt, params, cfg, policy):
    grad = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"], error_if_nonfinite=True)
    lr = cfg["learning_rate"] * min(1.0, (policy + 1) / cfg["warmup_updates"])
    for group in opt.param_groups:
        group["lr"] = lr
    opt.step()
    torch.cuda.synchronize()
    return dict(grad_norm=float(grad), grad_clipped=bool(grad > cfg["max_grad_norm"]), learning_rate=lr)


def online(model, opt, params, cfg, state, ids, preds, data, tag, first, times):
    started = time.monotonic()
    groups, items = [], []
    correct = Counter()
    for i, qid in enumerate(ids):
        row = data[qid]
        samples = preds[3*i:3*i+3]
        assert len(samples) == 3 and all((p["id"],p["rollout_index"]) == (qid,j) for j,p in enumerate(samples))
        rewards = []
        for p in samples:
            verdict = verify_answer(row, prompt=row["prompt"], completion=p["completion"], finish_reason=p["finish_reason"],
                                    output_tokens=len(p["response_token_ids"]), max_new_tokens=cfg["max_new_tokens"])
            rewards.append(student_reward(verdict))
        advs = advantages(rewards)
        groups.append(dict(id=qid, rewards=rewards, advantages=advs))
        correct[sum(rewards)] += 1
        items.extend(zip(samples, advs, strict=True))
    opt.zero_grad(set_to_none=True)
    loss, clipped_tokens, checked_tokens = 0.0, 0, 0
    raw_maes = []
    for pred, advantage in sorted(items, key=lambda item: len(item[0]["response_token_ids"])):
        if not advantage and not first:
            continue
        with torch.set_grad_enabled(bool(advantage)):
            values, raw = response_logps_batch(model, [pred["prompt_token_ids"]], [pred["response_token_ids"]],
                                              cfg["head_chunk"], cfg["likelihood_temperature"], return_raw=True)
            lp = values[0]
            mae = float((raw[0].cpu() - torch.as_tensor(pred["raw_logprobs"])).abs().mean())
            raw_maes.append(mae)
            if not mae < .05:
                raise ValueError(f"Sampling policy and actor raw log probabilities differ: {mae} ({tag})")
            if advantage:
                # One gradient accumulation at the exact frozen rollout policy. No optimizer
                # update occurs until all three-response groups have contributed.
                old_lp = lp.detach()
                ratio = (lp.detach() - old_lp).exp()
                clipped_tokens += int(((ratio - 1).abs() > cfg["clip"]).sum())
                checked_tokens += lp.numel()
                term = online_loss(lp, old_lp, advantage, cfg["clip"]) / (3*len(ids))
                term.backward()
                loss += float(term.detach())
        del values, raw, lp
    active = bool(correct[1] + correct[2])
    stats = finish_update(opt, params, cfg, state["policy"]) if active else dict(grad_norm=0.,grad_clipped=False)
    return dict(kind="online", groups=groups, question_count=len(ids), online_responses=len(preds),
                correct_responses=sum(k*v for k,v in correct.items()), group_histogram=[correct[k] for k in range(4)],
                output_tokens=sum(len(p["response_token_ids"]) for p in preds),
                prompt_tokens=sum(len(p["prompt_token_ids"]) for p in preds),
                truncated=sum(p["finish_reason"] == "length" for p in preds), optimizer_updated=active,
                loss=loss, ppo_clip_fraction=clipped_tokens/max(checked_tokens,1),
                raw_logprob_mae_max=max(raw_maes,default=None), likelihood_responses=len(raw_maes),
                request_tag=tag, update_seconds=time.monotonic()-started, **stats, **times)


def supervised(model, opt, params, cfg, state, data, eos, selection):
    before = time.monotonic()
    assert len(selection) == len({qid for qid,ref in selection}) == cfg["nll_batch"]
    opt.zero_grad(set_to_none=True)
    records = []
    for qid, ref in selection:
        prompt = data[qid]["prompt_token_ids"]
        target = reference_tokens(ref, eos)
        assert len(prompt) + len(target) <= cfg["max_model_len"]
        lp = response_logps_batch(model, [prompt], [target], cfg["head_chunk"], cfg["nll_temperature"])[0]
        assert lp.numel() == len(target) and target[-1] == eos
        loss = -lp.mean()
        (loss / len(selection)).backward()
        records.append(dict(id=qid, reference_id=ref["rollout_id"], completion_sha256=ref["completion_sha256"],
                            prompt_tokens=len(prompt), target_tokens=len(target), loss=float(loss.detach())))
        del lp, loss
    stats = finish_update(opt, params, cfg, state["policy"])
    return dict(kind="nll", references=records, target_tokens=sum(r["target_tokens"] for r in records),
                prompt_tokens=sum(r["prompt_tokens"] for r in records),
                loss=sum(r["loss"] for r in records)/len(records), update_seconds=time.monotonic()-before, **stats)


def check_mask(model, row, eos, cfg):
    from torch.nn import functional as F
    p = row["prompt_token_ids"][-12:]
    targets = row["english_references"][0]["response_token_ids"][:4] + [eos]
    prompts, responses = [p,p[:-2]], [targets,targets[-2:]]
    with torch.no_grad():
        values = response_logps_batch(model,prompts,responses,cfg["head_chunk"],1)
        for prompt, ref, lp in zip(prompts,responses,values,strict=True):
            tokens = torch.tensor([prompt + ref[:-1]], device=lp.device)
            dense = model(input_ids=tokens,use_cache=False).logits[0,len(prompt)-1:].float()
            expected = -F.cross_entropy(dense,torch.tensor(ref,device=lp.device),reduction="none")
            assert len(lp) == len(ref) and (lp-expected).abs().max().item() < .15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["pilot","formal"])
    args = parser.parse_args()
    pilot = args.stage == "pilot"
    cfg = read(RUN / ("pilot_config.json" if pilot else "effective_training_config.json"))
    run = RUN / "pilot" if pilot else RUN
    out = RUN / "pilot" / "temporary_model" if pilot else MODEL_OUT
    run.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    preflight = read(RUN / "preflight.json")
    assert cfg["train_sha256"] == preflight["data_sha256"]
    data = {r["id"]:r for r in read_rows(ROOT / cfg["training_data"])}
    assert list(data) == preflight["fixed_ids"]
    model_path = ROOT / cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    eos = tok.eos_token_id
    for row in data.values():
        row["prompt"] = tok.decode(row["prompt_token_ids"],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    batches = [pilot_ids(data)] if pilot else epoch_batches(data, cfg, cfg["epochs"])
    workers = [RUN/"rpc_workers"/f"worker{i}" for i in range(len(cfg["rollout_gpus"]))]
    plan = dict(ordered_batches=batches, train_sha256=cfg["train_sha256"], stage=args.stage)
    if (run/"training_plan.json").exists():
        assert read(run/"training_plan.json") == plan
    else:
        save(run/"training_plan.json",plan)
    existing = snapshots(out)
    last = existing[-1] if existing else None
    model = load_model(str(model_path),cfg["initial_seed"],adapter=str(last) if last else None,trainable_adapter=bool(last))
    params = [p for _,p in model.named_parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    assert model.get_base_model().config.attention_dropout == 0
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                            betas=(.9,.999),eps=1e-8)
    if last:
        state, digest = restore(last,model,opt,cfg)
        for path in existing:
            dest = run/"steps"/(path.name+".json")
            if not dest.exists():
                save(dest,read(path/"event.json"))
        save(run/f'resume-{state["tx"]:04d}-{os.getpid()}.json',dict(checkpoint=str(last),adapter_sha256=digest,
            optimizer_rng_restored=True, policy_version=state["policy"], queue_size=len(state["pending"])))
    else:
        state = dict(tx=0,policy=0,cursor=0,online=0,nll=0,pending=[],due=[],uses={})
        initial = adapter_hash(model)
        if not pilot:
            assert initial == read(RUN/"pilot/initial.json")["trainable_sha256"]
        save(run/"initial.json",dict(trainable_sha256=initial, model=str(model_path), phase=args.stage))
        last,digest,_ = commit(out,run,model,opt,state,dict(kind="initial",initial_hash=initial),cfg,model_path)
    model_load_seconds = time.monotonic()-started
    if pilot:
        check_mask(model, data[batches[0][0]], eos, cfg)
    while state["due"] or state["cursor"] < len(batches):
        before = time.monotonic()
        if state["due"]:
            qids = state["due"]
            selection = choose_refs(qids,data,state["uses"])
            if pilot:
                # Cover the actual longest full prompt+target in the timing and gradient test.
                qid, longest = max(((q, r) for q in qids for r in data[q]["english_references"]),
                                   key=lambda e:e[1]["training_context_tokens"])
                selection = [(q,longest if q == qid else ref) for q,ref in selection]
            event = supervised(model,opt,params,cfg,state,data,eos,selection)
            state["nll"] += 1
            state["policy"] += 1
            state["pending"] = [q for q in state["pending"] if q not in qids]
            for q in qids:
                state["uses"][q] = state["uses"].get(q,0)+1
            state["due"] = []
        else:
            cursor = state["cursor"]
            batch = batches[cursor]
            tag = f'{args.stage}-b{cursor+1:03d}-v{state["policy"]:03d}-{digest[:16]}'
            preds,times = generate(batch,data,cfg,workers,tag,last,digest,cursor+1,args.stage)
            event = online(model,opt,params,cfg,state,batch,preds,data,tag,cursor==0,times)
            del preds
            state["cursor"] += 1
            state["pending"] = refresh_pool(state["pending"],event["groups"],data)
            if event["optimizer_updated"]:
                state["online"] += 1
                state["policy"] += 1
                if state["online"] % cfg["check_every_online"] == 0:
                    event["queue_at_check"] = len(state["pending"])
                    if len(state["pending"]) >= cfg["nll_batch"] and state["nll"] < cfg["nll_limit"]:
                        state["due"] = state["pending"][:cfg["nll_batch"]]
                    event["scheduled_nll_ids"] = list(state["due"])
            if pilot:
                assert event["optimizer_updated"]
                state["due"] = list(batch)
                event["timing_only_forced_nll"] = True
        state["tx"] += 1
        event.update(input_adapter_sha256=digest, online_step=state["online"], nll_step=state["nll"],
                     cursor=state["cursor"], queue_size=len(state["pending"]), epoch=(state["cursor"]-1)//13+1)
        last,digest,written = commit(out,run,model,opt,state,event,cfg,model_path)
        if pilot and event["kind"] == "online":
            restored, verified = restore(last,model,opt,cfg)
            assert restored == state and verified == digest
        save(run/"timings"/f'{state["tx"]:04d}.json',dict(kind=event["kind"],checkpoint_seconds=written,
             total_seconds=time.monotonic()-before, end_unix=time.time()))
        if not pilot and not state["due"] and state["cursor"] % 13 == 0:
            save(out/f'epoch-{state["cursor"]//13}.json',dict(checkpoint=str(last),adapter_sha256=digest,**state))
        print(f'{args.stage} {event["kind"]} cursor={state["cursor"]} online={state["online"]} '
              f'nll={state["nll"]} pending={len(state["pending"])} policy={state["policy"]}',flush=True)
    sync_seconds = sync(workers,f'{args.stage}-final-v{state["policy"]}-{digest[:12]}',last,digest)
    final = dict(state, adapter=str(last), adapter_sha256=digest, model_load_seconds=model_load_seconds,
                 final_sync_seconds=sync_seconds, stage=args.stage, epochs=cfg["epochs"],
                 question_visits=sum(map(len,batches)), online_responses=3*sum(map(len,batches)),
                 train_sha256=cfg["train_sha256"], model_path=str(model_path))
    if not pilot:
        for epoch in range(1,cfg["epochs"]+1):
            eligible = [p for p in snapshots(out) if read(p/"state.json")["cursor"] == epoch*13]
            ep = eligible[-1]
            assert not read(ep/"state.json")["due"]
            save(out/f"epoch-{epoch}.json",dict(checkpoint=str(ep),adapter_sha256=read(ep/"manifest.json")["adapter_sha256"]))
        save(out/"final.json",final)
    save(run/"complete.json",final)


if __name__ == "__main__":
    main()
