"""Gradient-only GPU benchmark on one immutable existing on-policy batch."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from common import paths,configuration,read_rows,load_model,adapter_hash,save,sha
import train
import batched_update


class NoUpdateOptimizer:
    """Exercise the real update routines while leaving every parameter unchanged."""
    def __init__(self,model):self.model=model;self.param_groups=[{"lr":0.}];self.calls=0
    def zero_grad(self,set_to_none=True):self.model.zero_grad(set_to_none=set_to_none)
    def step(self):self.calls+=1


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",choices=["A","B","C"],default="C")
    parser.add_argument("--source-step",type=int,default=51);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists() or not any(args.out.iterdir()), "Use an empty output directory for a new benchmark attempt"
    args.out.mkdir(parents=True,exist_ok=True)
    p=paths();cfg=configuration(p);cfg["nll_alpha"]=json.loads((p["run_root"]/"calibration.json").read_text())["alpha"]
    request=json.loads((p["run_root"]/"rpc"/f"A-step-{args.source_step:03d}.request.json").read_text())
    generated=json.loads((p["run_root"]/"rpc"/f"A-step-{args.source_step:03d}.response.json").read_text())
    assert generated["adapter_sha256"]==request["adapter_sha256"]==sha(Path(request["adapter"])/"adapter_model.safetensors")
    lookup={r["id"]:r for r in read_rows(p["train_data"])}
    question_ids=list(dict.fromkeys(x["id"] for x in request["prompts"]))
    assert len(question_ids)==16 and len(generated["records"])==64
    batch=[]
    for qid in question_ids:
        row=dict(lookup[qid]);row["prompt_token_ids"]=next(x["prompt_token_ids"] for x in request["prompts"] if x["id"]==qid);batch.append(row)
    torch.cuda.set_device(0)
    model=load_model(str(p["model_path"]),cfg["initial_seed"],adapter=request["adapter"],trainable_adapter=True)
    params=[v for v in model.parameters() if v.requires_grad];initial=adapter_hash(model)
    opt=NoUpdateOptimizer(model);results=[];baseline_gradient=None;baseline_probabilities=None
    original_single=train.response_logps;original_batch=batched_update.response_logps_batch
    captured={}
    def key(prompt,response):return (torch.is_grad_enabled(),tuple(prompt),tuple(response))
    def capture_single(model,prompt,response,*a,**kw):
        result=original_single(model,prompt,response,*a,**kw)
        lp=result[0] if isinstance(result,tuple) else result
        captured[key(prompt,response)]=lp.detach().cpu()
        return result
    def capture_batch(model,prompts,responses,*a,**kw):
        result=original_batch(model,prompts,responses,*a,**kw)
        values=result[0] if isinstance(result,tuple) else result
        for prompt,response,lp in zip(prompts,responses,values):captured[key(prompt,response)]=lp.detach().cpu()
        return result
    train.response_logps=capture_single;batched_update.response_logps_batch=capture_batch
    options=[("legacy_mb1_chunk128",1,128),("batched_mb1_chunk512",1,512),("batched_mb2_chunk512",2,512),("batched_mb4_chunk512",4,512)]
    for name,micro,chunk in options:
        record=None;stage="warmup"
        try:
            model.zero_grad(set_to_none=True)
            warm=sorted(generated["records"],key=lambda x:len(x["response_token_ids"]))[:micro]
            model.train()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                if name.startswith("legacy"):
                    values=[original_single(model,warm[0]["prompt_token_ids"],warm[0]["response_token_ids"],128,temperature=.6)]
                else:
                    values=original_batch(model,[x["prompt_token_ids"] for x in warm],[x["response_token_ids"] for x in warm],chunk,temperature=.6)
                (-sum(x.mean() for x in values)/len(values)).backward();del values
            attention_operators=sorted({x.key for x in profile.key_averages()
                if any(label in x.key for label in ("scaled_dot_product","flash_attention","efficient_attention","cudnn_attention"))})
            del profile
            torch.cuda.synchronize();captured.clear();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            started=time.monotonic();stage="measured_update"
            if name.startswith("legacy"):
                record=train.update(model,opt,params,batch,generated,args.arm,cfg,0,1,1)
            else:
                record=batched_update.update(model,opt,params,batch,generated,args.arm,cfg,0,1,1,micro_batch=micro,head_chunk=chunk)
            elapsed=time.monotonic()-started
            assert adapter_hash(model)==initial
            gradient=torch.cat([v.grad.detach().cpu().flatten() for v in params])
            item={"name":name,"passed":True,"arm":args.arm,"seconds":elapsed,
                "peak_allocated_gib":record["peak_memory_by_rank"][0]["allocated_gib"],
                "gradient_norm":record["grad_norm_before_clip"],"online_responses":64,"warmup_attention_operators":attention_operators,
                "online_tokens":sum(len(x["response_token_ids"]) for x in generated["records"]),
                "demo_tokens":sum(len(r["demo"]["response_token_ids"]) for r in batch) if args.arm in ("B","C") else 0}
            if baseline_gradient is None:
                baseline_gradient=gradient.clone();baseline_probabilities={k:v.clone() for k,v in captured.items()}
            else:
                assert set(captured)==set(baseline_probabilities)
                left=baseline_gradient.double();right=gradient.double()
                cosine=float(torch.dot(left,right)/(left.norm()*right.norm()))
                norm_change=float(abs(right.norm()/left.norm()-1))
                difference=float((right-left).norm()/left.norm())
                errors=torch.cat([(captured[k]-baseline_probabilities[k]).abs() for k in captured])
                item.update(gradient_cosine=cosine,relative_gradient_norm_change=norm_change,
                    relative_gradient_difference=difference,logprob_mae=float(errors.mean()),logprob_max_error=float(errors.max()),
                    speedup_vs_legacy=results[0]["seconds"]/elapsed)
                item["passed"]=cosine>=.999 and norm_change<=.02 and difference<=.05 and float(errors.mean())<=.03
                del left,right,errors
            save(args.out/(name+".json"),record)
            del gradient
        except torch.cuda.OutOfMemoryError as error:
            item={"name":name,"passed":False,"oom":True,"stage":stage,"error":str(error)}
            model.zero_grad(set_to_none=True);captured.clear();gc.collect();torch.cuda.empty_cache()
            if name.startswith("legacy"):raise RuntimeError("Legacy baseline did not fit; no speedup comparison is valid") from error
        results.append(item);print(json.dumps(item),flush=True)
        save(args.out/"progress.json",{"results":results,"active_model_unchanged":adapter_hash(model)==initial})
    assert adapter_hash(model)==initial
    result={"complete":True,"parameter_updates_performed":False,"no_update_optimizer_step_calls":opt.calls,
        "source_policy_adapter":request["adapter"],"source_adapter_sha256":request["adapter_sha256"],
        "source_step":args.source_step,"arm":args.arm,"gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "results":results,"all_candidate_numerics_passed":all(x["passed"] for x in results),
        "input_budget":{"questions":16,"responses_per_question":4,"max_model_len":32768,"max_new_tokens":8192},
        "likelihood_temperature":cfg["likelihood_temperature"],"optimizer_state_allocated":False,
        "additional_real_adam_moments_gib":sum(v.numel() for v in params)*8/2**30,
        "scope":"One-GPU gradient-only benchmark, including reference, policy, and demonstration gradients; not a formal training run."}
    save(args.out/"summary.json",result)


if __name__=="__main__":main()
