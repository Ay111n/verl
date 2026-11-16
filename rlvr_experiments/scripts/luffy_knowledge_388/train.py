"""Three matched online arms; manual data-parallel LoRA gradient summation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from common import (
    paths, read_rows, save, sha, prompts, verify, plan_indices, object_sha,
    load_model, response_logps, adapter_hash, runtime, advantages, online_terms,
    demonstration_loss, lr_for_update, setup_dist, barrier, reduce_gradients,
    grad_copy, grad_norm, grad_summary, gather, rpc, generate, frozen_manifest,
)

def adapter_write(model, directory):
    directory.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(directory,safe_serialization=True)
    return sha(directory/"adapter_model.safetensors")

def save_checkpoint(model,opt,out,step,cfg,initial,frozen,rank,world):
    digest=adapter_hash(model)
    assert len(set(gather(digest,world)))==1,"Data-parallel replicas diverged"
    dest=out/f"checkpoint-{step}"
    rng={"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"python":random.getstate()}
    states=gather(rng,world)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        if (dest/"adapter_model.safetensors").exists():
            # This only happens after a crash between writing the adapter and recovery.
            from safetensors.torch import load_file
            existing=load_file(str(dest/"adapter_model.safetensors"))
            from peft import get_peft_model_state_dict
            current=get_peft_model_state_dict(model)
            assert set(existing)==set(current) and all(torch.equal(existing[k],current[k].detach().cpu()) for k in existing)
        file_hash=adapter_write(model,dest)
        state={"step":step,"optimizer":opt.state_dict(),"config":cfg,"initial_adapter_hash":initial,
               "frozen_hashes":frozen,"adapter":str(dest),"adapter_sha256":file_hash,
               "adapter_trainable_hash":digest,"rng_by_rank":states}
        temp=out/"recovery.pt.tmp";torch.save(state,temp);temp.replace(out/"recovery.pt")
        if step in (0,25,50,75): torch.save(state,dest/"training_state.pt")
        save(dest/"manifest.json",{"step":step,"adapter_sha256":file_hash,"adapter_trainable_hash":digest,
            "initial_adapter_hash":initial,"config_sha256":object_sha(cfg),
            "optimizer_steps":sorted({int(v["step"]) for v in opt.state.values()}),
            "full_optimizer_checkpoint":step in (0,25,50,75)})
        save(out/"latest.json",{"step":step,"adapter":str(dest),"adapter_sha256":file_hash})
    barrier()
    return dest

def create_model(p,cfg,rank,state=None):
    model=load_model(str(p["model_path"]),cfg["initial_seed"],
        adapter=state["adapter"] if state else None,trainable_adapter=bool(state))
    params=[v for v in model.parameters() if v.requires_grad]
    assert len(params)==504 and all(v.dtype==torch.float32 for v in params)
    assert model.get_base_model().config.attention_dropout==0
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],betas=(.9,.999),eps=1e-8,weight_decay=cfg["weight_decay"])
    if state:
        opt.load_state_dict(state["optimizer"])
        rng=state["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"]);torch.cuda.set_rng_state_all(rng["cuda"]);random.setstate(rng["python"])
    return model,opt,params

def get_rollouts(batch,adapter,name,epoch,cfg,p,rank):
    if rank==0:
        value=generate(batch,adapter,name,epoch,cfg,p["run_root"]/"rpc")
        assert len(value["records"])==len(batch)*4
    barrier()
    value=json.loads((p["run_root"]/"rpc"/(name+".response.json")).read_text())
    assert value["adapter_sha256"]==sha(adapter/"adapter_model.safetensors")
    assert len(value["records"])==len(batch)*4
    return value

def compact(v):
    keys=("correctness","parse_status","termination_status","protocol_valid","truncated",
          "answer_extractable","final_present","thinking_closed")
    return {k:v[k] for k in keys}

def update(model,opt,params,batch,generated,arm,cfg,rank,world,number):
    opt.zero_grad(set_to_none=True)
    outputs=generated["records"]
    stats=[]; rewards=[]; demo_stats=[]
    started=time.monotonic()
    for i,row in enumerate(batch):
        group=outputs[4*i:4*i+4]
        assert all(o["id"]==row["id"] and o["rollout_index"]==j for j,o in enumerate(group))
        verifications=[verify(row,o,cfg["max_new_tokens"]) for o in group]
        rs=[v["correctness"] for v in verifications];rewards.append(rs)
        if i%world!=rank:continue
        adv,_=advantages(rs,arm)
        for j,(o,v) in enumerate(zip(group,verifications,strict=True)):
            ids=o["response_token_ids"];prompt=row["prompt_token_ids"]
            assert o["prompt_token_ids"]==prompt and 0<len(ids)<=cfg["max_new_tokens"]
            assert len(prompt)+len(ids)<=cfg["max_model_len"]
            model.eval()
            with model.disable_adapter(),torch.no_grad():
                ref=response_logps(model,prompt,ids,cfg["chunk_size"]).detach()
            model.train()
            logp=response_logps(model,prompt,ids,cfg["chunk_size"])
            # Exactly one update per fresh batch, with zero dropout and no preceding
            # optimizer step: the old HF policy is the current unchanged parameters.
            # Detach only the denominator; ratio value=1 but numerator gradient remains.
            old=logp.detach()
            on,kl=online_terms(logp,old,ref,adv[j],cfg["clip"])
            loss=(on+cfg["kl_coef"]*kl)/(len(batch)*4)
            assert torch.isfinite(loss)
            loss.backward()
            backend=(logp.detach().cpu()-torch.tensor(o["raw_logprobs"])).abs()
            stats.append({"id":row["id"],"rollout_index":j,"tokens":len(ids),
                "advantage":float(adv[j]),"policy_loss":float(on.detach()),"sampled_kl":float(kl.detach()),
                "mean_logprob":float(logp.detach().mean()),"hf_vllm_raw_logprob_mae":float(backend.mean()),
                "verification":compact(v),"old_bo_correct":sum(row["old_bo_rewards"]),
                "stratum":row["probe_selection"]["stratum"]})
            if rank==0:
                print(json.dumps({"event":"online_backward","arm":arm,"step":number,"question":i,"rollout":j,
                    "tokens":len(ids),"reward":v["correctness"]}),flush=True)
            del logp,ref,on,kl,loss
    reduce_gradients(params)
    online=grad_copy(params)
    opt.zero_grad(set_to_none=True)
    if arm in ("B","C"):
        for i,row in enumerate(batch):
            if i%world!=rank:continue
            _,ad=advantages(rewards[i],"C")
            model.train()
            logp=response_logps(model,row["prompt_token_ids"],row["demo"]["response_token_ids"],cfg["chunk_size"])
            nll=-logp.mean()
            shaped=demonstration_loss(logp,ad,cfg["gamma"])
            loss=(cfg["nll_alpha"]*nll if arm=="B" else cfg["demo_scale"]*shaped)/len(batch)
            assert torch.isfinite(loss)
            loss.backward()
            demo_stats.append({"id":row["id"],"tokens":len(logp),"nll":float(nll.detach()),
                "mixed_demo_advantage":float(ad),"unweighted_shaped_loss":float(shaped.detach()),
                "weighted_demo_loss":float(loss.detach())*len(batch)})
            del logp,nll,shaped,loss
    reduce_gradients(params)
    demo=grad_copy(params)
    diagnostics=grad_summary(online,demo)
    for parameter,go,gd in zip(params,online,demo):parameter.grad=go+gd
    assert all(x.grad is None for x in model.parameters() if not x.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"])
    assert torch.isfinite(norm)
    lr=lr_for_update(number,cfg)
    for group in opt.param_groups:group["lr"]=lr
    opt.step()
    torch.cuda.synchronize()
    all_stats=sum(gather(stats,world),[])
    all_demo=sum(gather(demo_stats,world),[])
    peaks=gather({"allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                  "reserved_gib":torch.cuda.max_memory_reserved()/2**30},world)
    assert len(all_stats)==len(batch)*4
    assert len(all_demo)==(len(batch) if arm in ("B","C") else 0)
    return {"arm":arm,"step":number,"question_ids":[r["id"] for r in batch],"question_count":len(batch),
        "rollout_count":len(all_stats),"rewards":rewards,"correct":sum(map(sum,rewards)),
        "zero_correct_groups":sum(sum(x)==0 for x in rewards),"all_correct_groups":sum(sum(x)==4 for x in rewards),
        "mixed_groups":sum(0<sum(x)<4 for x in rewards),"online":all_stats,"demonstrations":all_demo,
        "grad_norm_before_clip":float(norm),"learning_rate":lr,**diagnostics,
        "peak_memory_by_rank":peaks,"training_seconds":time.monotonic()-started,
        "rollout_seconds":generated["elapsed_seconds"],"policy_adapter_sha256":generated["adapter_sha256"]}

def calibrate(p,cfg,rows,rank,world):
    model,opt,params=create_model(p,cfg,rank)
    initial=adapter_hash(model)
    plan=plan_indices(len(rows),cfg)
    indices=plan[0]["indices"]+plan[1]["indices"]
    assert len(indices)==32
    records=[]
    for k in range(2):
        batch=[rows[i] for i in indices[16*k:16*k+16]]
        norms={}
        for kind in ("nll","shaped"):
            opt.zero_grad(set_to_none=True)
            for i,row in enumerate(batch):
                if i%world!=rank:continue
                lp=response_logps(model,row["prompt_token_ids"],row["demo"]["response_token_ids"],cfg["chunk_size"])
                _,ad=advantages(row["old_bo_rewards"],"C")
                loss=-lp.mean() if kind=="nll" else cfg["demo_scale"]*demonstration_loss(lp,ad,cfg["gamma"])
                (loss/len(batch)).backward()
                del lp,loss
            reduce_gradients(params)
            norms[kind]=grad_norm(grad_copy(params))
            if rank==0: print(json.dumps({"event":"calibration","batch":k, "kind":kind,"norm":norms[kind]}),flush=True)
        records.append(norms)
    alpha=(sum(r["shaped"]**2 for r in records)/sum(r["nll"]**2 for r in records))**.5
    assert 0<alpha<1 and initial==adapter_hash(model) and len(opt.state)==0
    if rank==0:
        save(p["run_root"]/"calibration.json",{"passed":True,"alpha":alpha,"batch_norms":records,
            "question_ids":[rows[i]["id"] for i in indices],"optimizer_steps":0,
            "initial_adapter_hash":initial,"config":cfg,"train_sha256":sha(p["train_data"]),
            "method":"sqrt(sum two batch shaped-grad norms squared / sum NLL-grad norms squared); C coefficient 0.25 included"})
    barrier()

def backend_check(model,row,adapter,name,cfg,p,rank):
    ids=row["demo"]["response_token_ids"][:64];prompt=row["prompt_token_ids"]
    if rank==0:
        vl=rpc(p["run_root"]/"rpc",{"request_id":name,"mode":"score","adapter":str(adapter),
            "adapter_sha256":sha(adapter/"adapter_model.safetensors"),
            "prompts":[{"id":row["id"],"prompt_token_ids":prompt+ids,"score_start":len(prompt),"seed":42}]})
        model.eval()
        with torch.no_grad():hf=response_logps(model,prompt,ids,cfg["chunk_size"]).cpu()
        vals=vl["records"][0]["logprobs"];mae=float((hf-torch.tensor(vals)).abs().mean())
        assert mae<.08,(mae,name)
        value={"mae":mae,"hf":hf.tolist(),"vllm":vals}
    else:value=None
    barrier()
    return value

def smoke(p,cfg,rows,rank,world,attempt="smoke"):
    assert attempt.replace("-", "").replace("_", "").isalnum()
    preflight=json.loads((p["run_root"]/"preflight.json").read_text())
    batch=[rows[i] for i in preflight["smoke_indices"]]
    model,opt,params=create_model(p,cfg,rank)
    initial=adapter_hash(model);frozen=frozen_manifest(model)
    out=p["model_output"]/attempt;run=p["run_root"]/attempt
    if rank==0:
        assert not (out/"recovery.pt").exists(), "Use a new smoke-id for an independent acceptance attempt"
        run.mkdir(parents=True,exist_ok=True)
    barrier()
    adapter=save_checkpoint(model,opt,out,0,cfg,initial,frozen,rank,world)
    before=backend_check(model,batch[0],adapter,f"{attempt}-score-zero",cfg,p,rank)
    model.eval()
    with model.disable_adapter(),torch.no_grad():
        reference_before=response_logps(model,batch[0]["prompt_token_ids"],batch[0]["demo"]["response_token_ids"][:64]).cpu()
    records=[]
    torch.cuda.reset_peak_memory_stats()
    for number in (1,2):
        generated=get_rollouts(batch,adapter,f"{attempt}-C-step-{number}",number,cfg,p,rank)
        record=update(model,opt,params,batch,generated,"C",cfg,rank,world,number)
        adapter=save_checkpoint(model,opt,out,number,cfg,initial,frozen,rank,world)
        if rank==0:save(run/f"step-{number}.json",record)
        records.append(record)
    after=backend_check(model,batch[0],adapter,f"{attempt}-score-updated",cfg,p,rank)
    model.eval()
    with model.disable_adapter(),torch.no_grad():
        reference_after=response_logps(model,batch[0]["prompt_token_ids"],batch[0]["demo"]["response_token_ids"][:64]).cpu()
    delta=float((reference_before-reference_after).abs().max())
    assert delta==0 and frozen_manifest(model)==frozen and adapter_hash(model)!=initial
    # Verify saved optimizer and adapter independently in this same process.
    state=torch.load(out/"recovery.pt",map_location="cpu",weights_only=False)
    assert state["step"]==2 and state["adapter_trainable_hash"]==adapter_hash(model)
    assert {int(v["step"]) for v in state["optimizer"]["state"].values()}=={2}
    restored=torch.optim.AdamW(params,lr=cfg["learning_rate"])
    restored.load_state_dict(state["optimizer"])
    assert restored.param_groups[0]["lr"]==opt.param_groups[0]["lr"]
    for parameter in params:
        for key in ("exp_avg","exp_avg_sq","step"):
            assert torch.equal(restored.state[parameter][key],opt.state[parameter][key])
    del restored
    if rank==0:
        effect=max(abs(a-b) for a,b in zip(before["vllm"],after["vllm"]))
        assert effect>1e-7
        save(run/"summary.json",{"passed":True,"config":cfg,"runtime":runtime(),
            "question_ids":[r["id"] for r in batch],"fresh_rollouts":32,"optimizer_steps":2,
            "maximum_demo_context":max(len(r["prompt_token_ids"])+len(r["demo"]["response_token_ids"]) for r in batch),
            "initial_adapter_hash":initial,"updated_adapter_hash":adapter_hash(model),
            "frozen_reference_logprob_max_change":delta,"frozen_base_fingerprints_unchanged":True,
            "initial_backend":before,"updated_backend":after,"vllm_adapter_effect":effect,
            "optimizer_restore_state_verified":True,
            "max_gpu_allocated_gib":max(x["allocated_gib"] for r in records for x in r["peak_memory_by_rank"]),
            "attempt":attempt,"scope":"Two mixed-policy GPU updates on four training questions; discarded before A/B/C."})
        if run != p["run_root"]/"smoke":
            save(p["run_root"]/"smoke/summary.json",json.loads((run/"summary.json").read_text()))
    barrier()

def diagnose_demonstrations(model,rows,p,cfg,arm,step,rank,world):
    plan=plan_indices(len(rows),cfg)
    indices=plan[0]["indices"]+plan[1]["indices"]
    values=[]
    model.eval()
    with torch.no_grad():
        for position,index in enumerate(indices):
            if position%world!=rank:continue
            row=rows[index]
            lp=response_logps(model,row["prompt_token_ids"],row["demo"]["response_token_ids"],cfg["chunk_size"])
            values.append({"id":row["id"],"nll":float(-lp.mean()),"tokens":len(lp)})
    values=sum(gather(values,world),[])
    assert len(values)==32
    if rank==0:save(p["run_root"]/arm/("demonstration-diagnostic-%03d.json"%step),
        {"step":step,"scope":"Same fixed 32 training questions, teacher forcing only; not a validation set",
         "mean_nll":sum(x["nll"] for x in values)/len(values),"rows":values})
    barrier()


def train(p,cfg,rows,arm,rank,world):
    smoke_result=json.loads((p["run_root"]/"smoke/summary.json").read_text())
    assert smoke_result["passed"] and smoke_result["config"]==cfg
    out=p["model_output"]/arm;run=p["run_root"]/arm;run.mkdir(parents=True,exist_ok=True)
    recovery=out/"recovery.pt"
    state=torch.load(recovery,map_location="cpu",weights_only=False) if recovery.exists() else None
    if state:
        assert state["config"]==cfg and sha(Path(state["adapter"])/"adapter_model.safetensors")==state["adapter_sha256"]
    model,opt,params=create_model(p,cfg,rank,state)
    if state:
        initial=state["initial_adapter_hash"];frozen=state["frozen_hashes"];step=state["step"];adapter=Path(state["adapter"])
        assert adapter_hash(model)==state["adapter_trainable_hash"]
    else:
        initial=adapter_hash(model);frozen=frozen_manifest(model);step=0
        adapter=save_checkpoint(model,opt,out,0,cfg,initial,frozen,rank,world)
    assert initial==smoke_result["initial_adapter_hash"]
    if rank==0:
        save(run/"initialization.json",{"arm":arm,"initial_adapter_hash":initial,"config":cfg,"runtime":runtime(),
            "train_sha256":sha(p["train_data"]),"trainable_parameters":sum(v.numel() for v in params),
            "trainable_names":[n for n,v in model.named_parameters() if v.requires_grad]})
    plan=plan_indices(len(rows),cfg)
    assert len(plan)==75
    diagnose_demonstrations(model,rows,p,cfg,arm,step,rank,world)
    torch.cuda.reset_peak_memory_stats()
    for index,item in enumerate(plan):
        if index<step:continue
        number=index+1;batch=[rows[i] for i in item["indices"]]
        generated=get_rollouts(batch,adapter,f"{arm}-step-{number:03d}",item["epoch"],cfg,p,rank)
        record=update(model,opt,params,batch,generated,arm,cfg,rank,world,number)
        record["epoch"]=item["epoch"]
        adapter=save_checkpoint(model,opt,out,number,cfg,initial,frozen,rank,world)
        if rank==0:
            record["updated_adapter_sha256"]=sha(adapter/"adapter_model.safetensors")
            save(run/f"step-{number:03d}.json",record)
            save(run/"progress.json",{k:v for k,v in record.items() if k not in ("online","demonstrations","rewards","question_ids")})
            print(json.dumps({"event":"update_complete",**{k:record[k] for k in
                ("arm","step","epoch","correct","rollout_count","training_seconds","rollout_seconds","grad_norm_before_clip")}}),flush=True)
        barrier()
        if number in (25,50,75):diagnose_demonstrations(model,rows,p,cfg,arm,number,rank,world)
    assert frozen_manifest(model)==frozen
    if rank==0:
        records=[json.loads((run/f"step-{i:03d}.json").read_text()) for i in range(1,76)]
        assert sum(r["question_count"] for r in records)==1164 and sum(r["rollout_count"] for r in records)==4656
        save(run/"complete.json",{"arm":arm,"steps":75,"question_visits":1164,"online_rollouts":4656,
            "demonstration_visits":1164 if arm in ("B","C") else 0,"initial_adapter_hash":initial,
            "final_adapter_hash":adapter_hash(model),"final_adapter":str(adapter),
            "final_adapter_sha256":sha(adapter/"adapter_model.safetensors"),"config":cfg,
            "frozen_base_fingerprints_unchanged":True})
    barrier()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("calibrate","smoke","train"),required=True)
    parser.add_argument("--arm",choices=("A","B","C"),default="C")
    parser.add_argument("--smoke-id",default="smoke")
    args=parser.parse_args()
    rank,world=setup_dist()
    assert world==2
    p=paths();cfg=json.loads((p["project_dir"]/"configs/luffy_knowledge_388_20261116.json").read_text())
    preflight=json.loads((p["run_root"]/"preflight.json").read_text())
    assert preflight["passed"] and preflight["config"]==cfg
    assert sha(p["train_data"])==cfg["train_sha256"]
    tok=AutoTokenizer.from_pretrained(str(p["model_path"]),local_files_only=True)
    rows=read_rows(p["train_data"])
    for row in rows:row["prompt_token_ids"]=prompts(tok,row)[1]
    if args.mode=="calibrate":calibrate(p,cfg,rows,rank,world)
    else:
        calibration=json.loads((p["run_root"]/"calibration.json").read_text())
        assert calibration["passed"] and calibration["config"]==cfg
        cfg={**cfg,"nll_alpha":calibration["alpha"],"calibration_sha256":sha(p["run_root"]/"calibration.json")}
        if args.mode=="smoke":smoke(p,cfg,rows,rank,world,args.smoke_id)
        else:train(p,cfg,rows,args.arm,rank,world)
    if dist.is_initialized():dist.destroy_process_group()

if __name__=="__main__":main()
