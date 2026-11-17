"""Same objective and normalization as train.update, with length-bucketed micro-batches."""
import time
import torch
from common import (verify,advantages,online_terms,demonstration_loss,reduce_gradients,
                    grad_copy,grad_summary,lr_for_update,gather)
from train import compact
from batched_likelihood import response_logps_batch


def batches(items,micro_batch,max_padded_tokens=32768):
    ordered=sorted(items,key=lambda x:len(x["prompt"])+len(x["ids"]))
    group=[]
    for item in ordered:
        length=len(item["prompt"])+len(item["ids"])-1
        if group and (len(group)==micro_batch or length*(len(group)+1)>max_padded_tokens):
            yield group;group=[]
        group.append(item)
    if group:yield group


def update(model,opt,params,batch,generated,arm,cfg,rank,world,number,micro_batch=1,head_chunk=512):
    started=time.monotonic();opt.zero_grad(set_to_none=True)
    rewards=[];items=[];stats=[];demo_stats=[]
    for i,row in enumerate(batch):
        group=generated["records"][4*i:4*i+4]
        assert all(o["id"]==row["id"] and o["rollout_index"]==j for j,o in enumerate(group))
        verified=[verify(row,o,cfg["max_new_tokens"]) for o in group]
        rs=[v["correctness"] for v in verified];rewards.append(rs)
        if i%world!=rank:continue
        adv,_=advantages(rs,arm)
        for j,(o,v) in enumerate(zip(group,verified)):
            assert o["prompt_token_ids"]==row["prompt_token_ids"]
            assert 0<len(o["response_token_ids"])<=cfg["max_new_tokens"]
            assert len(row["prompt_token_ids"])+len(o["response_token_ids"])<=cfg["max_model_len"]
            items.append({"row":row,"output":o,"verification":v,"advantage":float(adv[j]),
                          "prompt":row["prompt_token_ids"],"ids":o["response_token_ids"],"index":j})
    for group in batches(items,micro_batch):
        prompts=[x["prompt"] for x in group];responses=[x["ids"] for x in group]
        model.eval()
        with model.disable_adapter(),torch.no_grad():
            refs=response_logps_batch(model,prompts,responses,head_chunk,cfg["likelihood_temperature"])
        model.train()
        current,raw=response_logps_batch(model,prompts,responses,head_chunk,cfg["likelihood_temperature"],True)
        terms=[online_terms(lp,lp.detach(),ref,x["advantage"],cfg["clip"]) for x,lp,ref in zip(group,current,refs)]
        loss=sum(on+cfg["kl_coef"]*kl for on,kl in terms)/(len(batch)*4)
        assert torch.isfinite(loss)
        loss.backward()
        for x,lp,raw_lp,(on,kl) in zip(group,current,raw,terms):
            row=x["row"];o=x["output"];backend=(raw_lp.cpu()-torch.tensor(o["raw_logprobs"])).abs()
            stats.append({"id":row["id"],"rollout_index":x["index"],"tokens":len(x["ids"]),
                "advantage":x["advantage"],"policy_loss":float(on.detach()),"sampled_kl":float(kl.detach()),
                "mean_logprob":float(lp.detach().mean()),"mean_raw_logprob":float(raw_lp.mean()),
                "likelihood_temperature":cfg["likelihood_temperature"],"hf_vllm_raw_logprob_mae":float(backend.mean()),
                "verification":compact(x["verification"]),"old_bo_correct":sum(row["old_bo_rewards"]),
                "stratum":row["probe_selection"]["stratum"]})
        del current,raw,refs,terms,loss
    reduce_gradients(params);online=grad_copy(params);opt.zero_grad(set_to_none=True)
    if arm in ("B","C"):
        demonstrations=[]
        for i,row in enumerate(batch):
            if i%world!=rank:continue
            _,ad=advantages(rewards[i],"C")
            demonstrations.append({"row":row,"advantage":float(ad),"prompt":row["prompt_token_ids"],
                                   "ids":row["demo"]["response_token_ids"]})
        for group in batches(demonstrations,micro_batch):
            values=response_logps_batch(model,[x["prompt"] for x in group],[x["ids"] for x in group],head_chunk,cfg["likelihood_temperature"])
            terms=[]
            for x,lp in zip(group,values):
                nll=-lp.mean();shaped=demonstration_loss(lp,x["advantage"],cfg["gamma"])
                term=cfg["nll_alpha"]*nll if arm=="B" else cfg["demo_scale"]*shaped
                terms.append(term)
                demo_stats.append({"id":x["row"]["id"],"tokens":len(lp),"nll":float(nll.detach()),
                    "mixed_demo_advantage":x["advantage"],"unweighted_shaped_loss":float(shaped.detach()),
                    "weighted_demo_loss":float(term.detach())})
            loss=sum(terms)/len(batch);assert torch.isfinite(loss);loss.backward()
            del values,terms,loss
        reduce_gradients(params);demo=grad_copy(params)
    else:
        # Every rank's demonstration contribution is identically zero in A.
        demo=[torch.zeros_like(x) for x in params]
    diagnostics=grad_summary(online,demo)
    for parameter,left,right in zip(params,online,demo):parameter.grad=left+right
    assert all(x.grad is None for x in model.parameters() if not x.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"]);assert torch.isfinite(norm)
    lr=lr_for_update(number,cfg)
    for group in opt.param_groups:group["lr"]=lr
    opt.step();torch.cuda.synchronize()
    all_stats=sum(gather(stats,world),[]);all_demo=sum(gather(demo_stats,world),[])
    peaks=gather({"allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                  "reserved_gib":torch.cuda.max_memory_reserved()/2**30},world)
    assert len(all_stats)==len(batch)*4 and len(all_demo)==(len(batch) if arm in ("B","C") else 0)
    return {"arm":arm,"step":number,"question_ids":[r["id"] for r in batch],"question_count":len(batch),
        "rollout_count":len(all_stats),"rewards":rewards,"correct":sum(map(sum,rewards)),
        "zero_correct_groups":sum(sum(x)==0 for x in rewards),"all_correct_groups":sum(sum(x)==4 for x in rewards),
        "mixed_groups":sum(0<sum(x)<4 for x in rewards),"online":all_stats,"demonstrations":all_demo,
        "grad_norm_before_clip":float(norm),"learning_rate":lr,**diagnostics,"peak_memory_by_rank":peaks,
        "training_seconds":time.monotonic()-started,"rollout_seconds":generated["elapsed_seconds"],
        "policy_adapter_sha256":generated["adapter_sha256"],
        "execution":{"micro_batch":micro_batch,"head_chunk":head_chunk,"length_bucketed":True}}
