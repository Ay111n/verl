"""Two independent visits per question; supervise the English question body only on visit one."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import sys
import time

os.environ["RLVR_PAIRED_EXPERIMENT"] = "formateval_priority512_restate_20261121"
import torch
from transformers import AutoTokenizer
from paired import CONFIG, OUT, ROOT, RUN, load_inputs, judge, read, save, sha
from checkpointing import commit, restore, snapshots
from contract import advantages
from sampling import generate, sync
from run import supervise
from train import finish_update
from scripts.luffy_math_pilot.common import adapter_hash, load_model, online_loss, response_logps_batch


def prepare(data, tok, cfg):
    targets = {}
    head = "<|im_start|>user\nPlease answer the question.\n\n"
    tail = "\n\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
    encode = lambda s: tok.encode(s,add_special_tokens=False)
    for uid,pair in data.items():
        en,bo=pair["en"],pair["bo"]
        assert en["canonical_answer"]==bo["canonical_answer"] and en["option_labels"]==bo["option_labels"]==list("ABCD")
        prompt=en["prompt"]
        assert prompt.startswith(head) and prompt.endswith(tail)
        body=prompt[len(head):-len(tail)]
        assert body.startswith("Question: ") and body.endswith("\n"+"\n".join(k+". "+en["options_en"][k] for k in "ABCD"))
        prefix=bo["prompt"]+"<think>\n"
        x=encode(prefix);xy=encode(prefix+body)
        assert x[:len(bo["prompt_token_ids"])]==bo["prompt_token_ids"] and xy[:len(x)]==x
        y=xy[len(x):]
        assert y and len(xy)<=cfg["max_model_len"] and not set(y).intersection(tok.all_special_ids)
        targets[uid]=(x,y)
    assert sum(len(y) for x,y in targets.values())==45186
    order=list(data);second=order.copy();random.Random(cfg["second_order_seed"]).shuffle(second)
    assert len(order)==len(set(second))==512 and set(order)==set(second) and order!=second
    return targets,[q[i:i+32] for q in (order,second) for i in range(0,len(q),32)]


def update(model,opt,params,cfg,state,batch,preds,bo,targets,visit):
    start=time.monotonic(); n=cfg["rollouts_per_question"]; size=len(preds)
    assert size==len(batch)*n==128 and n==4
    groups=[];items=[];hist=[0]*5
    for k,q in enumerate(batch):
        view=bo[q]; samples=preds[k*n:(k+1)*n]
        verdicts=[]
        for j,r in enumerate(samples):
            assert r["id"]==q and r["rollout_index"]==j and r["prompt_token_ids"]==view["prompt_token_ids"]
            verdicts.append(judge(view,r["completion"],r["finish_reason"],len(r["response_token_ids"]),cfg))
        rewards=[v["reward"] for v in verdicts];advs=advantages(rewards,n);hist[sum(rewards)]+=1
        groups.append(dict(id=q,judgments=verdicts,rewards=rewards,advantages=advs))
        items.extend(zip(samples,advs,strict=True))
    opt.zero_grad(set_to_none=True)
    grpo=nll=max_mae=0.;target_tokens=active_rollouts=0
    for r,a in sorted(items,key=lambda pair:len(pair[0]["response_token_ids"])):
        if a==0:continue
        lp,raw=response_logps_batch(model,[r["prompt_token_ids"]],[r["response_token_ids"]],cfg["head_chunk"],cfg["likelihood_temperature"],True)
        mae=(raw[0].cpu()-torch.as_tensor(r["raw_logprobs"])).abs().mean().item()
        if not mae<.05:raise ValueError(f"Actor and sampler log probabilities differ: {mae}")
        max_mae=max(mae,max_mae)
        loss=online_loss(lp[0],lp[0].detach(),a,cfg["clip"])/size
        loss.backward();grpo+=loss.detach().item();active_rollouts+=1
        del lp,raw,loss
    if visit==1:
        for g in groups:
            x,y=targets[g["id"]]
            logps=response_logps_batch(model,[x],[y],cfg["head_chunk"],cfg["restate_temperature"])[0]
            assert len(logps)==len(y)
            per_question=-logps.mean()
            (cfg["restate_weight"]*per_question/len(batch)).backward()
            g["english_body_nll"]=per_question.detach().item();g["target_tokens"]=len(y)
            nll+=g["english_body_nll"]/len(batch);target_tokens+=len(y)
            del logps,per_question
    active=bool(active_rollouts or target_tokens)
    stats=finish_update(opt,params,cfg,state["policy"]) if active else dict(grad_norm=0.,grad_clipped=False)
    return dict(groups=groups,group_histogram=hist,question_count=len(batch),online_responses=size,active_grpo_rollouts=active_rollouts,
                correct_responses=sum(i*v for i,v in enumerate(hist)),output_tokens=sum(len(r["response_token_ids"]) for r in preds),
                truncated=sum(r["finish_reason"]=="length" for r in preds),grpo_loss=grpo,english_body_nll=nll,target_tokens=target_tokens,
                loss=grpo+cfg["restate_weight"]*nll,optimizer_updated=active,raw_logprob_mae_max=max_mae,
                update_seconds=time.monotonic()-start,**stats)


def train():
    from candidate import save_responses
    cfg,audit,data=load_inputs()
    hw=read(RUN/"effective_training_config.json")
    assert (hw["actor_gpu"] in [4,5,6] and len(hw["rollout_gpus"])==2 and
            hw["actor_gpu"] not in hw["rollout_gpus"] and hw["epochs"]==2)
    free,total=torch.cuda.mem_get_info()
    assert free>=cfg["actor_min_free_mib"]*1024**2
    cap=free-cfg["actor_reserve_mib"]*1024**2
    torch.cuda.set_per_process_memory_fraction(cap/total)
    save(RUN/f"actor_memory_{os.getpid()}.json",dict(gpu=hw["actor_gpu"],free=free,cap=cap,total=total))
    base=ROOT/cfg["model_path"]
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    targets,batches=prepare(data,tok,cfg)
    save(RUN/"training_plan.json",dict(batches=batches,targets={q:dict(count=len(y),sha256=hashlib.sha256(json.dumps(y).encode()).hexdigest()) for q,(_,y) in targets.items()}))
    OUT.mkdir(parents=True,exist_ok=True)
    cps=snapshots(OUT);last=cps[-1] if cps else None
    model=load_model(str(base),cfg["initial_seed"],adapter=str(last) if last else None,trainable_adapter=bool(last))
    params=[p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype==torch.float32 and "lora_" in k for k,p in model.named_parameters() if p.requires_grad)
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    if last:
        state,digest=restore(last,model,opt,cfg)
        for path in cps:
            if not (RUN/"steps"/(path.name+".json")).exists():save(RUN/"steps"/(path.name+".json"),read(path/"event.json"))
    else:
        state=dict(tx=0,cursor=0,policy=0,online=0,nll=0,pending=[],due=[])
        last,digest,_=commit(OUT,RUN,model,opt,state,dict(kind="initial",s0=str(base),initial_lora_sha256=adapter_hash(model)),cfg,base)
    workers=[RUN/"rpc_workers"/f"worker{i}" for i in range(2)]
    bo={q:p["bo"] for q,p in data.items()}
    while state["cursor"]<len(batches):
        start=time.monotonic();i=state["cursor"];step=i+1;visit=i//16+1;batch=batches[i]
        version=state["policy"];tag=f"formal-b{step:03d}-v{version:03d}-{digest[:16]}"
        preds,times=generate(batch,bo,cfg,workers,tag,last,digest,step,f"native-visit-{visit}")
        event=update(model,opt,params,cfg,state,batch,preds,bo,targets,visit)
        event.update(kind="online",visit=visit,cursor=step,input_adapter_sha256=digest,request_tag=tag,**times)
        event["responses_sha256"]=save_responses(step,version,digest,preds,event["groups"],data)
        state["cursor"]=state["tx"]=step
        state["policy"]+=int(event["optimizer_updated"]);state["online"]=state["policy"]
        last,digest,secs=commit(OUT,RUN,model,opt,state,event,cfg,base)
        save(RUN/"timings"/f"{step:04d}.json",dict(checkpoint_seconds=secs,total_seconds=time.monotonic()-start,end_unix=time.time()))
        print(f"Step {step}/32, visit {visit}, correct {event['correct_responses']}/128, target tokens {event['target_tokens']}",flush=True)
    sync(workers,f"final-{digest[:16]}",last,digest)
    events=[read(RUN/"steps"/f"checkpoint-{i:04d}.json") for i in range(1,33)]
    byvisit=[{k:sum(e[k] for e in events if e["visit"]==v) for k in ("online_responses","correct_responses","question_count","output_tokens","truncated","target_tokens","active_grpo_rollouts")} for v in (1,2)]
    assert [s["target_tokens"] for s in byvisit]==[45186,0] and all(s["online_responses"]==2048 and s["question_count"]==512 for s in byvisit)
    final=dict(state,epochs=2,question_visits=1024,online_responses=4096,visits=byvisit,adapter=str(last),adapter_sha256=digest,model_path=str(base))
    save(OUT/"final.json",final);save(RUN/"complete.json",final)



def configure_runtime():
    """Use the data volume for compiled artifacts and a short directory for Unix sockets."""
    cache = ROOT / ".runtime" / RUN.name
    ipc = ROOT / ".runtime/rest512-1121"
    paths = {
        "TMP": ipc, "TEMP": ipc, "TMPDIR": ipc, "VLLM_RPC_BASE_PATH": ipc,
        "VLLM_CACHE_ROOT": cache / "vllm", "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor", "CUDA_CACHE_PATH": cache / "cuda",
    }
    for key, path in paths.items():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = str(ipc)



if __name__=="__main__":
    if sys.argv[1]=="train":train()
    elif sys.argv[1]=="run":
        configure_runtime()
        supervise(CONFIG,RUN,online_actor=Path(__file__).resolve())
        subprocess.run([sys.executable,str(ROOT/"benchmark/eval/analysis"/RUN.name/"summarize.py")],check=True,cwd=ROOT)
    else:raise ValueError(sys.argv[1])
