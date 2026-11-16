"""One fixed epoch on the identical 316 pairs for three offline DPO arms."""
import argparse,json,math,time,os
import torch.distributed as dist
from pathlib import Path
import torch
from paired_dpo import *

def load_inputs(data):
    records=read_rows(data/"pairs.jsonl");targets=read_rows(data/"targets.jsonl")
    plan=json.loads((data/"training_plan.json").read_text())
    assert sha(data/"pairs.jsonl")==plan["data_sha256"]
    assert sha(data/"targets.jsonl")==plan["targets_sha256"]
    assert len(records)==len(targets)==316
    for r,t in zip(records,targets):assert r["pair_id"]==t["pair_id"]
    return records,targets,plan

def losses(model,r,t,target_key="target",plan=None):
    c,j=pair_scores(model,r)
    b,a,d=terms(c,j,t["bo_chosen"],t["bo_rejected"],t["nc"],t["nr"],t[target_key],
        t["gate"],beta=plan["beta"],alpha=plan["alpha"])
    return b,a,d,c,j

def calibrate(a):
    records,targets,plan=load_inputs(a.data);model=load_model(a.model)
    init=adapter_hash(model);params=[p for p in model.parameters() if p.requires_grad]
    accum={s:[torch.zeros_like(p) for p in params] for s in ["base","true","shuffled"]}
    checks=[];started=time.time()
    for n,i in enumerate(plan["calibration_indices"]):
        r=records[i];t=targets[i]
        b,aux,d,c,j=losses(model,r,t,plan=plan)
        _,sh,_=terms(c,j,t["bo_chosen"],t["bo_rejected"],t["nc"],t["nr"],t["shuffled_target"],t["gate"],plan["beta"],plan["alpha"])
        dc=float(c.detach())-t["bo_chosen"];dj=float(j.detach())-t["bo_rejected"]
        assert max(abs(dc),abs(dj))<=.05,("reference/gradient backend drift",r["id"],dc,dj)
        if n==0:
            with model.disable_adapter(),torch.no_grad():
                c0,j0=pair_scores(model,r)
            assert max(abs(float(c0-c.detach())),abs(float(j0-j.detach())))<=.01,"Zero LoRA differs from complete SFT"
        for name,loss in [("base",b),("true",aux),("shuffled",sh)]:
            grads=torch.autograd.grad(loss/len(plan["calibration_indices"]),params,retain_graph=name!="shuffled",allow_unused=True)
            for total,g in zip(accum[name],grads):
                if g is not None:total.add_(g)
            del grads
        checks.append({"id":r["id"],"chosen_sum_difference":dc,"rejected_sum_difference":dj,
            "initial_dpo":float(d["dpo"].detach()),"true_aux":float(aux.detach()),"shuffled_aux":float(sh.detach()),"active":d["active"]})
        print(json.dumps({"calibration_completed":n+1,"total":len(plan["calibration_indices"]),"elapsed":time.time()-started}),flush=True)
        del b,aux,sh,d,c,j
    norms={name:math.sqrt(sum(float(v.float().square().sum()) for v in values)) for name,values in accum.items()}
    assert norms["true"]>1e-12 and all(math.isfinite(v) for v in norms.values())
    weight=.1*norms["base"]/norms["true"]
    result={"lambda":weight,"target_gradient_ratio":.1,"gradient_norms":norms,
        "shuffled_weighted_gradient_ratio":weight*norms["shuffled"]/norms["base"],
        "adapter_initial_sha256":init,"trainable_parameters":sum(p.numel() for p in params),
        "reference_checks":checks,"plan_sha256":sha(a.data/"training_plan.json"),"runtime":runtime(),
        "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,"elapsed_seconds":time.time()-started,
        "scope":"Calibration only; no optimizer updates; no DEV feedback; fresh model for every training arm"}
    assert adapter_hash(model)==init
    save(a.run/"calibration.json",result);print(json.dumps(result),flush=True)

def train(a):
    world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
    if world>1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]));dist.init_process_group("nccl")
    records,targets,plan=load_inputs(a.data)
    cal=json.loads((a.run/"calibration.json").read_text());assert cal["plan_sha256"]==sha(a.data/"training_plan.json")
    weight=0. if a.arm=="base" else cal["lambda"]
    target_key="shuffled_target" if a.arm=="shuffled" else "target"
    model=load_model(a.model);init=adapter_hash(model)
    assert init==cal["adapter_initial_sha256"],"Different adapter initialization"
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=plan["learning_rate"],betas=tuple(plan["betas"]),eps=plan["eps"],weight_decay=plan["weight_decay"])
    out=a.run/a.arm;out.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/"adapter_model.safetensors").exists()
    config={**plan,"arm":a.arm,"lambda":weight,"target_key":target_key,"model":a.model,
        "adapter_initial_sha256":init,"runtime":runtime(),"calibration_sha256":sha(a.run/"calibration.json"),"data_parallel_world_size":world,
        "code_sha256":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}}
    if rank==0:save(out/"config.json",config)
    started=time.time();tokens=0;exposure=[];step=0
    with (out/f"steps.rank{rank}.jsonl").open("w",encoding="utf-8") as stream:
        for start in range(0,len(records),plan["effective_batch"]):
            indices=plan["order"][start:start+plan["effective_batch"]];step+=1
            warm=plan["warmup_steps"]
            factor=step/warm if step<=warm else (plan["optimizer_steps"]-step+1)/(plan["optimizer_steps"]-warm)
            lr=plan["learning_rate"]*factor
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True);values=[]
            for i in indices[rank::world]:
                r=records[i];t=targets[i]
                b,aux,d,c,j=losses(model,r,t,target_key,plan)
                loss=b+weight*aux;assert torch.isfinite(loss)
                (loss/len(indices)).backward()
                values.append({"base":float(b.detach()),"aux":float(aux.detach()),"dpo":float(d["dpo"].detach()),
                    "nll":float(d["nll"].detach()),"margin":float(d["margin"].detach()),"active":d["active"],
                    "chosen_delta":float(c.detach())-t["bo_chosen"],"rejected_delta":float(j.detach())-t["bo_rejected"]})
                del b,aux,d,c,j,loss
            if world>1:
                for param in params:
                    if param.grad is None:param.grad=torch.zeros_like(param)
                    dist.all_reduce(param.grad,op=dist.ReduceOp.SUM)
                packed=torch.tensor([sum(float(v[k]) for v in values) for k in values[0]],device="cuda",dtype=torch.float64)
                dist.all_reduce(packed,op=dist.ReduceOp.SUM)
                averaged={k:float(v)/len(indices) for k,v in zip(values[0],packed)}
            else:averaged={k:sum(float(v[k]) for v in values)/len(values) for k in values[0]}
            for i in indices:
                r=records[i];t=targets[i]
                tokens+=len(r["prompt_bo_token_ids"])*2+t["nc"]+t["nr"];exposure.append(r["pair_id"])
            norm=torch.nn.utils.clip_grad_norm_(params,plan["max_grad_norm"],error_if_nonfinite=True)
            opt.step()
            record={"step":step,"pairs":len(exposure),"batch_pairs":len(indices),"lr":lr,"gradient_norm":float(norm),
                "elapsed_seconds":time.time()-started,"input_tokens":tokens,
                "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,
                **averaged}
            stream.write(json.dumps(record)+"\n");stream.flush()
            if rank==0:save(out/"status.json",record)
            if rank==0:print(json.dumps(record),flush=True)
    assert step==40 and len(exposure)==316 and object_sha(exposure)==plan["order_sha256"]
    final_hash=adapter_hash(model)
    if world>1:
        hashes=[None]*world;dist.all_gather_object(hashes,final_hash)
        assert len(set(hashes))==1,hashes
    if rank!=0:
        dist.barrier();dist.destroy_process_group();return
    model.save_pretrained(a.output)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.model,local_files_only=True).save_pretrained(a.output)
    torch.save({"optimizer":opt.state_dict(),"torch_rng_state":torch.get_rng_state(),
        "cuda_rng_state":torch.cuda.get_rng_state_all(),"step":step,"exposure":exposure,"config":config},a.output/"training_state.pt")
    result={"status":"completed","arm":a.arm,"optimizer_steps":step,"pairs_seen":len(exposure),
        "order_sha256":object_sha(exposure),"input_tokens":tokens,"elapsed_seconds":time.time()-started,
        "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9,"model":a.model,"adapter":str(a.output),
        "adapter_initial_sha256":init,"adapter_final_sha256":final_hash,"data_parallel_world_size":world,
        "adapter_files":{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},
        "config_sha256":sha(out/"config.json")}
    save(out/"completion.json",result);print(json.dumps(result),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["calibrate","train"])
    p.add_argument("--data",type=Path,required=True);p.add_argument("--model",required=True)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--arm",choices=["base","true","shuffled"])
    a=p.parse_args()
    if a.mode=="calibrate":calibrate(a)
    else:
        assert a.output and a.arm;train(a)
if __name__=="__main__":main()
