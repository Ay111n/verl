"""One trainable Tibetan GRPO student, one frozen English S0 teacher and native DEV."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from paired import CONFIG, OUT, ROOT, RUN, load_inputs, preflight, read, save, sha
from objective import update
from checkpointing import commit, restore, snapshots
from sampling import generate, sync
from run import supervise
from scripts.luffy_math_pilot.common import adapter_hash, load_model
from scripts.opd_math_ab.common import response_hidden, reverse_kl
from teacher import supervise as collect_teacher


def save_responses(cursor, version, digest, predictions, groups, data, run=RUN):
    directory = run/"responses"
    directory.mkdir(exist_ok=True)
    dest = directory/f"batch-{cursor:04d}.jsonl.gz"
    tmp = dest.with_suffix(".tmp")
    indexed = {g["id"]:g for g in groups}
    with gzip.open(tmp,"wt",encoding="utf-8",compresslevel=2) as f:
        for pred in predictions:
            uid,rep = pred["id"],pred["rollout_index"]
            group = indexed[uid]
            row = {k:pred[k] for k in ("id","seed","response_token_ids","completion","finish_reason","stop_reason")}
            row.update(rep=rep,policy_version=version,adapter_sha256=digest,prompt_sha256=data[uid]["bo"]["prompt_sha256"],
                output_tokens=len(pred["response_token_ids"]),judgment=group["judgments"][rep],
                reward=group["rewards"][rep])
            if "kd_tokens_by_rep" in group:
                row["kd_tokens"] = group["kd_tokens_by_rep"][rep]
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    tmp.replace(dest)
    return sha(dest)


def check_teacher(model,pair,audit,cfg,params):
    # A short causal prefix exercises the production projections and LoRA gradient on the assigned actor.
    suffix = [audit["think_open"],*pair["en"]["prompt_token_ids"][1:5]]
    bo = pair["bo"]["prompt_token_ids"]
    en = pair["en"]["prompt_token_ids"]
    with model.disable_adapter(), torch.no_grad():
        model.eval()
        h = response_hidden(model,en,suffix)[1:].detach()
    model.train()
    frozen_hash = hashlib.sha256(h.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if not (RUN/"actor_check.json").exists():
        s = response_hidden(model,bo,suffix)[1:]
        v = reverse_kl(s,h,model.get_base_model().lm_head.weight,cfg["kd_chunk"])
        grad = torch.autograd.grad(v,params,allow_unused=True)
        norm = sum(x.detach().square().sum().item() for x in grad if x is not None)**.5
        if not torch.isfinite(v) or norm <= 0:
            raise RuntimeError("Non-finite or inactive actor distillation gradient")
        save(RUN/"actor_check.json",dict(loss=v.item(),gradient_norm=norm,frozen_probe_sha256=frozen_hash,
              adapter_disabled_for_teacher=True,eligible_positions=len(s),temperature=1))
    return frozen_hash


def train():
    cfg,audit,data = load_inputs()
    ranking = "rank_weight" in cfg
    expected_size = 8 if ranking else 4
    assert (cfg["epochs"],cfg["batch_size"],cfg["rollouts_per_question"],cfg["teacher_rollouts"]) == (1,16,expected_size,expected_size)
    devices = read(RUN/"effective_training_config.json")
    assert len(devices["rollout_gpus"]) == 3 and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    teacher = read(RUN/"teacher_complete.json")
    assert teacher["config_sha256"] == audit["config_sha256"] and teacher["prompt_sha256"] == audit["prompt_sha256"]
    assert teacher["model"] == cfg["model_path"] and teacher["responses"] == len(data)*expected_size and list(teacher["rewards"]) == list(data)
    assert all(len(r)==expected_size and all(v in (0,1) for v in r) for r in teacher["rewards"].values())
    bo = {uid:pair["bo"] for uid,pair in data.items()}
    ids = list(bo)
    batches = [ids[i:i+cfg["batch_size"]] for i in range(0,len(ids),cfg["batch_size"])]
    assert len(batches) == 32 and all(len(b)==16 for b in batches)
    save(RUN/"training_plan.json",dict(ordered_batches=batches,teacher_sha256=sha(RUN/"teacher_complete.json")))
    base = ROOT/cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(base,local_files_only=True)
    terminal = {tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>"),tok.convert_tokens_to_ids("<|endoftext|>")}
    OUT.mkdir(parents=True,exist_ok=True)
    existing = snapshots(OUT)
    last = existing[-1] if existing else None
    if ranking:
        assert devices["actor_gpu"] in cfg["shared_actor_gpus"]
        free,total = torch.cuda.mem_get_info()
        capacity = free-cfg["actor_reserve_mib"]*1024**2
        if free<cfg["actor_min_free_mib"]*1024**2:
            raise RuntimeError(f"Shared actor has only {free//1024**2} MiB available")
        torch.cuda.set_per_process_memory_fraction(capacity/total)
        save(RUN/f"actor_memory_{os.getpid()}.json",dict(gpu=devices["actor_gpu"],free=free,cap=capacity,total=total))
    model = load_model(str(base),cfg["initial_seed"],adapter=str(last) if last else None,trainable_adapter=bool(last))
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all("lora_" in name for name,p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    if last:
        state,digest = restore(last,model,opt,cfg)
        for cp in existing:
            if not (RUN/"steps"/(cp.name+".json")).exists():
                save(RUN/"steps"/(cp.name+".json"),read(cp/"event.json"))
    else:
        state = dict(tx=0,cursor=0,policy=0,online=0,nll=0,pending=[],due=[])
        last,digest,_ = commit(OUT,RUN,model,opt,state,
            dict(kind="initial",s0=str(base),initial_lora_sha256=adapter_hash(model)),cfg,base)
    if ranking:
        from weighted_rank import prepare_readout, choice_scores, choice_loss, update as weighted_update
        readout = prepare_readout(model,tok,data,cfg,audit)
        if not (RUN/"actor_check.json").exists():
            gold = data[ids[0]]["bo"]["canonical_answer"]
            wrong = next(c for c in "ABCD" if c!=gold)
            scores = choice_scores(model,readout[0][ids[0]],readout[1])
            value = choice_loss(scores,"ABCD",gold,[dict(wrong=wrong,weight=1.)])
            grads = torch.autograd.grad(value,params,allow_unused=True)
            norm = sum(g.detach().square().sum().item() for g in grads if g is not None)**.5
            assert torch.isfinite(value) and norm>0
            save(RUN/"actor_check.json",dict(rank_loss=value.item(),gradient_norm=norm,label_ids=audit["rank_label_ids"]))
            del scores,value,grads
    else:
        teacher_probe = check_teacher(model,data[ids[0]],audit,cfg,params)
    workers = [RUN/"rpc_workers"/f"worker{i}" for i in range(len(devices["rollout_gpus"]))]
    while state["cursor"] < len(batches):
        started = time.monotonic()
        cursor = state["cursor"] + 1
        batch = batches[cursor-1]
        tag = f'formal-b{cursor:03d}-v{state["policy"]:03d}-{digest[:16]}'
        preds,times = generate(batch,bo,cfg,workers,tag,last,digest,cursor,"formal")
        if ranking:
            event = weighted_update(model,opt,params,cfg,state,batch,preds,data,teacher,readout,tag,times)
        else:
            event = update(model,opt,params,cfg,state,batch,preds,data,teacher["rewards"],audit,terminal,tag,times)
        event["responses_sha256"] = save_responses(cursor,state["policy"],digest,preds,event["groups"],data)
        event.update(input_adapter_sha256=digest,cursor=cursor,epoch=1)
        del preds
        state["policy"] += int(event["optimizer_updated"])
        state["online"] = state["policy"]
        state["cursor"] = state["tx"] = cursor
        last,digest,written = commit(OUT,RUN,model,opt,state,event,cfg,base)
        save(RUN/"timings"/f"{cursor:04d}.json",dict(checkpoint_seconds=written,
             total_seconds=time.monotonic()-started,end_unix=time.time()))
        active = event["rank_questions"] if ranking else event["kd_eligible"]
        print(f'Batch {cursor}/{len(batches)}: {event["correct_responses"]}/{event["online_responses"]} correct, '
              f'auxiliary questions {active}, gradient {event["grad_norm"]:.4f}',flush=True)
    if not ranking:
        assert check_teacher(model,data[ids[0]],audit,cfg,params) == teacher_probe
    final_sync = sync(workers,f'formal-final-v{state["policy"]}-{digest[:12]}',last,digest)
    events = [read(RUN/"steps"/f"checkpoint-{i:04d}.json") for i in range(1,len(batches)+1)]
    groups = [g for e in events for g in e["groups"]]
    assert [g["id"] for g in groups] == ids
    matrix = [[0]*(expected_size+1) for _ in range(expected_size+1)]
    for g in groups:
        matrix[g["english_correct"]][g["tibetan_correct"]] += 1
    final = dict(state,epochs=1,adapter=str(last),adapter_sha256=digest,model_path=str(base),
        train_sha256=cfg["train_sha256"],question_visits=len(groups),teacher_responses=teacher["responses"],
        teacher_correct=teacher["correct"],teacher_output_tokens=teacher["output_tokens"],teacher_truncated=teacher["truncated"],
        english_by_tibetan_correct=matrix,final_sync_seconds=final_sync,
        **{k:sum(e[k] for e in events) for k in ("online_responses","correct_responses","truncated","output_tokens","prompt_tokens")})
    if ranking:
        for key in ("rank_questions","rank_zero","rank_pairs","weight_sum","active_grpo_rollouts"):
            final[key]=sum(e[key] for e in events)
        final["teacher_reused"]=len(ids)*cfg["teacher_reused_slots"]
        final["teacher_new"]=teacher["responses"]-final["teacher_reused"]
    else:
        assert all(g["kd_eligible"] == (g["english_correct"]>g["tibetan_correct"]) for g in groups)
        assert all(not any(g["kd_tokens_by_rep"]) or g["kd_eligible"] for g in groups)
        final.update(kd_eligible=sum(g["kd_eligible"] for g in groups),
            kd_applied=sum(any(g["kd_tokens_by_rep"]) for g in groups),
            kd_applied_bo_zero=sum(any(g["kd_tokens_by_rep"]) and g["tibetan_correct"]==0 for g in groups),
            kd_eligible_bo_zero=sum(g["kd_eligible"] and g["tibetan_correct"]==0 for g in groups),
            kd_tokens=sum(e["kd_tokens"] for e in events),frozen_teacher_probe_sha256=teacher_probe)
    assert final["online_responses"] == len(ids)*expected_size and sum(map(sum,matrix)) == 512
    save(OUT/"final.json",final)
    save(RUN/"complete.json",final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preflight","run","train"))
    stage = parser.parse_args().stage
    if stage == "preflight":
        preflight()
    elif stage == "run":
        if not (RUN/"preflight.json").is_file():
            preflight()
        collect_teacher()
        supervise(CONFIG,RUN,online_actor=Path(__file__).resolve())
        report = ROOT/"benchmark/eval/analysis"/RUN.name/"summarize.py"
        subprocess.run([sys.executable,str(report)], cwd=ROOT, check=True)
    else:
        train()
