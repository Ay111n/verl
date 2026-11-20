"""One epoch of native Tibetan GRPO, substituting an assisted group only after 0/4."""
from collections import Counter
import gzip
import json
import os
import sys
import time

from prepare import CONFIG, OUT, PROJECT, ROOT, RUN, read, sha
sys.path.insert(0,str(PROJECT/"scripts/crosslingual_grpo_512"))
import torch
from contract import student_reward
from checkpointing import commit, restore, snapshots
from sampling import generate, sync
from train import online
from run import supervise
from rlvr.answer_verifier import verify_answer
from scripts.luffy_math_pilot.common import adapter_hash, load_model, save


def cached_group(ids, prompts, cfg, workers, tag, cp, digest, cursor, stage):
    path=RUN/"rollouts"/f"{tag}.json.gz"
    if path.exists():
        with gzip.open(path,"rt",encoding="utf8") as f:
            entry=json.load(f)
        assert entry["digest"]==digest and entry["ids"]==ids
        for i,row in enumerate(entry["rows"]):
            assert (row["id"],row["rollout_index"],row["prompt_token_ids"])==(ids[i//4],i%4,prompts[ids[i//4]]["prompt_token_ids"])
        assert len(entry["rows"])==4*len(ids)
        return entry["rows"],entry["times"]
    rows,times=generate(ids,prompts,cfg,workers,tag,cp,digest,cursor,stage)
    path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix(".tmp")
    with gzip.open(temp,"wt",encoding="utf8",compresslevel=1) as f:
        json.dump(dict(ids=ids,digest=digest,rows=rows,times=times),f,ensure_ascii=False,separators=(",",":"))
    os.replace(temp,path)
    return rows,times


def train():
    cfg=read(CONFIG)
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="5"
    gpu=read(RUN/"effective_training_config.json")
    assert gpu["actor_gpu"]==5 and len(set(gpu["rollout_gpus"]))==2 and 5 not in gpu["rollout_gpus"]
    audit=read(RUN/"preflight.json")
    assert sha(CONFIG)==audit["config_sha256"] and sha(RUN/"prefixes.json")==audit["prefixes_sha256"]
    assert sha(RUN/"training_prompts.json")==audit["prompt_file_sha256"]
    data=read(RUN/"training_prompts.json")
    refs=read(RUN/"prefixes.json")
    keys=list(data)
    batches=[keys[i:i+16] for i in range(0,len(keys),16)]
    assert len(batches)==32 and all(len(b)==16 for b in batches)
    workers=[RUN/"rpc_workers"/f"worker{i}" for i in range(2)]
    free,total=torch.cuda.mem_get_info()
    assert free>=cfg["actor_min_free_mib"]*1024**2
    cap=free-cfg["actor_reserve_mib"]*1024**2
    torch.cuda.set_per_process_memory_fraction(cap/total)
    save(RUN/f"actor_memory_{os.getpid()}.json",dict(gpu=5,free=free,cap=cap))
    OUT.mkdir(parents=True,exist_ok=True)
    cps=snapshots(OUT)
    cp=cps[-1] if cps else None
    modelpath=ROOT/cfg["model_path"]
    model=load_model(str(modelpath),cfg["initial_seed"],adapter=str(cp) if cp else None,trainable_adapter=bool(cp))
    params=[p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype==torch.float32 for p in params)
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    if cp:
        state,digest=restore(cp,model,opt,cfg)
        for c in cps:
            if not (RUN/"steps"/(c.name+".json")).exists():
                save(RUN/"steps"/(c.name+".json"),read(c/"event.json"))
    else:
        state=dict(tx=0,cursor=0,policy=0,online=0,nll=0,pending=[],due=[])
        cp,digest,_=commit(OUT,RUN,model,opt,state,dict(kind="initial",s0=str(modelpath),initial_lora_sha256=adapter_hash(model)),cfg,modelpath)
    assisted={uid:dict(data[uid],prompt_token_ids=v["prompt_token_ids"]) for uid,v in refs.items()}
    while state["cursor"]<len(batches):
        start=time.monotonic()
        index=state["cursor"];batch=batches[index];step=index+1
        tag=f"formal-b{step:03d}-v{state['policy']:03d}-{digest[:16]}"
        plain,times=cached_group(batch,data,cfg,workers,tag,cp,digest,step,"formal")
        initial={}
        for pred in plain:
            uid=pred["id"];q=data[uid]
            verdict=verify_answer(q,prompt=q["prompt"],completion=pred["completion"],finish_reason=pred["finish_reason"],
                                  output_tokens=len(pred["response_token_ids"]),max_new_tokens=cfg["max_new_tokens"])
            assert verdict["verifier_version"]==cfg["verifier_version"]
            initial.setdefault(uid,[]).append(dict(rep=pred["rollout_index"],reward=student_reward(verdict),
                selected=verdict["extracted_answer"],finish=pred["finish_reason"],tokens=len(pred["response_token_ids"])))
        rescue=[uid for uid in batch if uid in refs and not any(r["reward"] for r in initial[uid])]
        chosen={(r["id"],r["rollout_index"]):r for r in plain}
        extra=[]
        if rescue:
            extra,extra_times=cached_group(rescue,assisted,cfg,workers,tag+"-rescue",cp,digest,step,"prefix-rescue")
            times=dict(times,rescue_timing=extra_times)
            for r in extra:
                uid=r["id"]
                # Only the text supplied to v3 is reconstructed. The policy loss uses
                # the real prefilled prompt IDs and the newly generated suffix IDs.
                chosen[uid,r["rollout_index"]]=dict(r,completion=refs[uid]["forced"]+r["completion"])
        selected=[chosen[uid,i] for uid in batch for i in range(4)]
        event=online(model,opt,params,cfg,state,batch,selected,data,tag,index==0,times)
        for group in event["groups"]:
            uid=group["id"]; group["assisted"]=uid in rescue
            if uid not in rescue:
                assert group["rewards"]==[r["reward"] for r in initial[uid]]
        event.update(initial=initial,rescue_ids=rescue,independent_responses=len(plain),assisted_responses=len(extra),
            sampled_output_tokens=sum(len(r["response_token_ids"]) for r in plain+extra),input_adapter_sha256=digest,
            peak_actor_mib=torch.cuda.max_memory_reserved()/1024**2)
        state["cursor"]+=1;state["tx"]+=1
        if event["optimizer_updated"]:
            state["online"]+=1;state["policy"]+=1
        event.update(cursor=state["cursor"],online_step=state["online"],epoch=1)
        cp,digest,write_sec=commit(OUT,RUN,model,opt,state,event,cfg,modelpath)
        save(RUN/"timings"/f"{step:04d}.json",dict(checkpoint_seconds=write_sec,total_seconds=time.monotonic()-start,end_unix=time.time()))
        print(f"Committed batch {step}/32; assisted {len(rescue)} of 16 questions.",flush=True)
    sync(workers,"final-"+digest[:16],cp,digest)
    events=[read(RUN/"steps"/f"checkpoint-{i:04d}.json") for i in range(1,33)]
    assert [g["id"] for e in events for g in e["groups"]]==keys
    rescued=sum(len(e["rescue_ids"]) for e in events)
    final=dict(state,epochs=1,adapter=str(cp),adapter_sha256=digest,question_visits=len(keys),
        online_responses=sum(e["independent_responses"] for e in events),assisted_responses=4*rescued,
        rescue_questions=rescued,eligible=len(refs),all_generated_responses=2048+4*rescued,
        output_tokens=sum(e["sampled_output_tokens"] for e in events))
    save(OUT/"final.json",final);save(RUN/"complete.json",final)


if __name__=="__main__":
    if sys.argv[1]=="train":
        train()
    elif sys.argv[1]=="run":
        from pathlib import Path
        supervise(CONFIG,RUN,online_actor=Path(__file__).resolve())
    else:
        raise ValueError(sys.argv)
