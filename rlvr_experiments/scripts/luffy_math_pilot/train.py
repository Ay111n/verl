"""One small mixed-policy Math run; exactly one optimizer update per fresh batch."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import torch
from common import (ROOT,PROJECT,load_model,read_rows,save,sha,adapter_hash,runtime,response_logps_batch,
                    verify,advantages,online_loss,demo_loss,rpc,rpc_parallel,norm)
def checkpoint(model,opt,out,step,cfg):
    dest=out/f"checkpoint-{step:04d}";assert not dest.exists(),f"Refusing to overwrite {dest}"
    staging=out/f".checkpoint-{step:04d}.pending";staging.mkdir()
    model.save_pretrained(staging,safe_serialization=True)
    for f in staging.iterdir():
        if f.is_file():
            with f.open("rb") as stream:os.fsync(stream.fileno())
    staging.rename(dest)
    digest=sha(dest/"adapter_model.safetensors")
    state={"step":step,"adapter":str(dest),"adapter_sha256":digest,"optimizer":opt.state_dict(),
           "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
           "python_rng":random.getstate(),"config":cfg}
    tmp=out/"recovery.pt.tmp";torch.save(state,tmp);tmp.replace(out/"recovery.pt")
    save(dest/"manifest.json",{"step":step,"adapter_sha256":digest,"optimizer_steps":sorted({int(v["step"]) for v in opt.state.values()})})
    save(out/"latest.json",{"step":step,"adapter":str(dest),"adapter_sha256":digest})
    return dest,digest
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);ap.add_argument("--resume",action="store_true");a=ap.parse_args()
    cfg=json.loads(a.config.read_text());cfg["model_path"]=str(ROOT/cfg["model_path"]);assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==1
    run=PROJECT/"runs"/cfg["name"];out=ROOT/"finetuning/model/rl"/cfg["name"]
    run.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    state=None
    if a.resume:
        state=torch.load(out/"recovery.pt",map_location="cpu",weights_only=False)
        assert state["config"]==cfg,"Resume configuration changed"
        latest=json.loads((out/"latest.json").read_text())
        assert latest["step"]==state["step"] and latest["adapter_sha256"]==state["adapter_sha256"]
        assert sha(Path(state["adapter"])/"adapter_model.safetensors")==state["adapter_sha256"]
    else:
        assert not (out/"latest.json").exists(),"Fresh pilot only; use explicit --resume"
    start=time.monotonic()
    model=load_model(cfg["model_path"],cfg["initial_seed"],adapter=state["adapter"] if state else None,trainable_adapter=bool(state))
    params=[p for p in model.parameters() if p.requires_grad]
    assert params and all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype==torch.float32 for p in params)
    assert model.get_base_model().config.attention_dropout==0
    opt=torch.optim.AdamW(params,lr=cfg["learning_rate"],betas=(.9,.999),eps=1e-8,weight_decay=0)
    if state:
        opt.load_state_dict(state["optimizer"])
        adapter=Path(state["adapter"]);digest=state["adapter_sha256"]
        initial=json.loads((run/"actor_ready.json").read_text())["initial_trainable_hash"]
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict
        disk=load_file(str(adapter/"adapter_model.safetensors"))
        live=get_peft_model_state_dict(model)
        assert set(disk)==set(live) and all(torch.equal(disk[k],live[k].detach().cpu()) for k in disk)
        assert all(int(v["step"])==state["step"] for v in opt.state.values())
        torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        save(run/f"resume-{state['step']:04d}.json",{"pid":os.getpid(),"from_step":state["step"],
             "adapter_sha256":digest,"optimizer_restored":True,"rng_restored":True,
             "weights_equal_checkpoint":True,"loading_seconds":time.monotonic()-start})
    else:
        adapter,digest=checkpoint(model,opt,out,0,cfg);initial=adapter_hash(model)
        save(run/"actor_ready.json",{"pid":os.getpid(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"],"runtime":runtime(),
            "initial_adapter_sha256":digest,"initial_trainable_hash":initial,"loading_seconds":time.monotonic()-start})
    data=ROOT/cfg.get("training_data",f'finetuning/rl/final_data/{cfg["name"]}/train.jsonl')
    while not (data.parent/"manifest.json").exists():
        if time.time()>cfg["train_deadline_unix"]:raise TimeoutError("Data preparation exceeded training budget")
        time.sleep(1)
    rows=read_rows(data);manifest=json.loads((data.parent/"manifest.json").read_text())
    assert sha(data)==manifest["train_sha256"]
    assert len(rows)==cfg.get("strict_question_count",len(rows))
    random.Random(cfg["order_seed"]).shuffle(rows)
    batches=[rows[i:i+cfg["batch_size"]] for i in range(0,len(rows),cfg["batch_size"])]
    assert cfg["epochs"]==1
    plan_path=run/"training_plan.json"
    if state:
        assert json.loads(plan_path.read_text())["ordered_ids"]==[r["id"] for r in rows]
    else:save(plan_path,{"train_sha256":sha(data),"question_count":len(rows),
        "ordered_ids":[r["id"] for r in rows],"planned_updates":len(batches),"online_per_question":3,
        "demo_per_question":1,"config":cfg,"config_sha256":sha(a.config)})
    completed=state["step"] if state else 0
    history=[json.loads((run/"steps"/f"{i:04d}.json").read_text()) for i in range(1,completed+1)]
    stop_reason="completed_fixed_plan"
    for step,batch in enumerate(batches,1):
        if step<=completed:continue
        remaining=cfg["train_deadline_unix"]-time.time()
        estimated=1.25*max((s["step_seconds"] for s in history[-2:]),default=900)+60
        if cfg.get("strict_wall_budget",True) and remaining<estimated:
            stop_reason="predeclared_wall_budget_before_next_batch";break
        if not cfg.get("strict_wall_budget",True):remaining=max(3600,remaining)
        before=time.monotonic();prompts=[]
        for row in batch:
            for j in range(3):
                seed=int.from_bytes(hashlib.sha256(f'{cfg["sampling_seed"]}|{step}|{row["id"]}|{j}'.encode()).digest()[:4],"big")%(2**31-1)
                prompts.append({"id":row["id"],"rollout_index":j,"prompt_token_ids":row["prompt_token_ids"],"seed":seed})
        request={"request_id":f"train-{step:04d}","mode":"generate","adapter":str(adapter),
            "adapter_sha256":digest,"prompts":prompts,"return_logprobs":True,
            **{k:cfg[k] for k in ["max_new_tokens","temperature","top_p","top_k"]}}
        if cfg.get("rollout_gpus"):
            workers=[run/"rpc_workers"/f"worker{i+1}" for i in range(len(cfg["rollout_gpus"]))]
            generated=rpc_parallel(run/"rpc",request,workers,timeout=min(3600,max(60,remaining)))
        else:
            generated=rpc(run/"rpc",request,timeout=min(3600,max(60,remaining)))
        assert generated["adapter_sha256"]==digest and len(generated["records"])==3*len(batch)
        items=[];groups=[];demonstrations=[]
        for i,row in enumerate(batch):
            responses=generated["records"][3*i:3*i+3]
            assert all(x["id"]==row["id"] and x["rollout_index"]==j for j,x in enumerate(responses))
            vv=[verify(row,x,cfg["max_new_tokens"]) for x in responses]
            assert all(v["execution_valid"] and v["correctness"] in (0,1) for v in vv)
            rs=[v["correctness"] for v in vv];adv,ad=advantages(rs)
            groups.append({"id":row["id"],"rewards":rs,"advantages":adv.tolist(),"demo_advantage":float(ad)})
            for j,(pred,v) in enumerate(zip(responses,vv)):
                assert pred["prompt_token_ids"]==row["prompt_token_ids"]
                items.append({"row":row,"pred":pred,"verification":v,"advantage":float(adv[j])})
            demonstrations.append({"row":row,"advantage":float(ad)})
        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        stats=[];demos=[];checks=[];update_start=time.monotonic()
        for idx,item in enumerate(sorted(items,key=lambda x:len(x["pred"]["response_token_ids"]))):
            row,pred=item["row"],item["pred"]
            values,raw=response_logps_batch(model,[row["prompt_token_ids"]],[pred["response_token_ids"]],
                cfg["head_chunk"],cfg["likelihood_temperature"],True)
            lp,raw_lp=values[0],raw[0]
            term=online_loss(lp,lp.detach(),item["advantage"],cfg["clip"])
            backend=float((raw_lp.cpu()-torch.tensor(pred["raw_logprobs"])).abs().mean())
            checks.append(backend)
            if step==1:
                assert backend<.05,f"HF/vLLM likelihood discrepancy {backend}"
            (term/(4*len(batch))).backward()
            stats.append({"id":row["id"],"rollout_index":pred["rollout_index"],"tokens":len(lp),
                "advantage":item["advantage"],"policy_loss":float(term.detach()),
                "mean_logprob":float(lp.detach().mean()),"hf_vllm_raw_mae":backend,
                "verification":item["verification"]})
            del values,raw,lp,raw_lp,term
        online=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        for item in demonstrations:
            row=item["row"];ids=row["demo"]["response_token_ids"]
            if item["advantage"]==0:
                demos.append({"id":row["id"],"advantage":0,"tokens":len(ids),"loss":0,"forward_skipped_zero_advantage":True})
                continue
            lp=response_logps_batch(model,[row["prompt_token_ids"]],[ids],
                cfg["head_chunk"],cfg["likelihood_temperature"])[0]
            term=(-cfg["reference_nll_weight"]*item["advantage"]*lp.mean() if cfg.get("reference_loss","shaped")=="advantage_nll" else demo_loss(lp,item["advantage"],cfg["gamma"]))
            (term/(4*len(batch))).backward()
            demos.append({"id":row["id"],"advantage":item["advantage"],"tokens":len(ids),
                "loss":float(term.detach()),"nll":float(-lp.detach().mean()),"gold_provided":row["demo"]["gold_provided"]})
            del lp,term
        demo=[p.grad.detach()-g if p.grad is not None else -g for p,g in zip(params,online)]
        on_norm,demo_norm=norm(online),norm(demo)
        dot=float(torch.stack([(g*d).sum() for g,d in zip(online,demo)]).sum())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        grad=torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"]);assert torch.isfinite(grad)
        lr=cfg["learning_rate"]*min(1,step/cfg["warmup_updates"])
        for g in opt.param_groups:g["lr"]=lr
        opt.step();torch.cuda.synchronize()
        assert adapter_hash(model)!=initial
        adapter,digest=checkpoint(model,opt,out,step,cfg)
        value={"step":step,"questions":len(batch),"online_responses":len(stats),
            "online_correct":sum(sum(g["rewards"]) for g in groups),
            "all_wrong_groups":sum(sum(g["rewards"])==0 for g in groups),
            "all_correct_groups":sum(sum(g["rewards"])==3 for g in groups),
            "groups":groups,"online":stats,"demonstrations":demos,
            "online_grad_norm":on_norm,"demo_grad_norm":demo_norm,
            "gradient_cosine":dot/(on_norm*demo_norm) if on_norm*demo_norm else None,
            "grad_norm_before_clip":float(grad),"learning_rate":lr,
            "hf_vllm_mae_mean":sum(checks)/len(checks),"hf_vllm_mae_max":max(checks),
            "rollout_seconds":generated["elapsed_seconds"],"update_seconds":time.monotonic()-update_start,
            "step_seconds":time.monotonic()-before,"policy_adapter_sha256":generated["adapter_sha256"],
            "output_adapter_sha256":digest,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
        save(run/"steps"/f"{step:04d}.json",value);history.append(value);completed=step
        print(json.dumps({k:value[k] for k in ["step","questions","online_correct","all_wrong_groups",
             "all_correct_groups","online_grad_norm","demo_grad_norm","step_seconds","peak_allocated_gib"]}),flush=True)
        del online,demo
    assert completed>0,"No optimizer updates completed"
    final={"status":"complete" if completed==len(batches) else "budget_limited",
        "stop_reason":stop_reason,"steps":completed,"planned_steps":len(batches),
        "training_questions":len(rows),"question_visits":sum(x["questions"] for x in history),
        "online_responses":sum(x["online_responses"] for x in history),
        "adapter":str(adapter),"adapter_sha256":digest,"initial_adapter_hash":initial,
        "final_adapter_hash":adapter_hash(model),"train_sha256":sha(data),
        "wall_seconds":time.monotonic()-start}
    save(run/"complete.json",final);print(json.dumps(final),flush=True)
if __name__=="__main__":main()
