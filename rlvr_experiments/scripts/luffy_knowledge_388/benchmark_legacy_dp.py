"""Measure the original two-actor C update on an already-generated benchmark batch."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from common import paths,configuration,read_rows,load_model,adapter_hash,save,sha,setup_dist,barrier,gather,response_logps
from benchmark_actor import NoUpdateOptimizer
import train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--response",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--start-gate",type=Path)
    args=parser.parse_args();rank,world=setup_dist();assert world==2
    if rank==0:
        assert not args.out.exists() or not any(args.out.iterdir()), "Use a new empty benchmark directory"
        args.out.mkdir(parents=True,exist_ok=True)
    barrier()
    p=paths();cfg=configuration(p)
    cfg["nll_alpha"]=json.loads((p["run_root"]/"calibration.json").read_text())["alpha"]
    request=json.loads(args.request.read_text());generated=json.loads(args.response.read_text())
    assert generated["adapter_sha256"]==request["adapter_sha256"]==sha(Path(request["adapter"])/"adapter_model.safetensors")
    ids=list(dict.fromkeys(x["id"] for x in request["prompts"]));assert len(ids)==16 and len(generated["records"])==64
    lookup={x["id"]:x for x in read_rows(p["train_data"])}
    batch=[]
    for qid in ids:
        row=dict(lookup[qid]);row["prompt_token_ids"]=next(x["prompt_token_ids"] for x in request["prompts"] if x["id"]==qid);batch.append(row)
    model=load_model(str(p["model_path"]),cfg["initial_seed"],adapter=request["adapter"],trainable_adapter=True)
    initial=adapter_hash(model);params=[x for x in model.parameters() if x.requires_grad];opt=NoUpdateOptimizer(model)
    warm=min(generated["records"][rank::world],key=lambda x:len(x["response_token_ids"]))
    model.train();lp=response_logps(model,warm["prompt_token_ids"],warm["response_token_ids"],cfg["chunk_size"],temperature=cfg["likelihood_temperature"])
    (-lp.mean()).backward();del lp
    model.zero_grad(set_to_none=True);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    if args.start_gate:
        deadline=time.monotonic()+1800
        while not args.start_gate.exists():
            assert time.monotonic()<deadline,"Timed out waiting for isolated timed interval"
            time.sleep(.2)
    barrier();started=time.monotonic()
    result=train.update(model,opt,params,batch,generated,"C",cfg,rank,world,1)
    barrier();elapsed=time.monotonic()-started
    assert adapter_hash(model)==initial
    digest=hashlib.sha256()
    for parameter in params:digest.update(parameter.grad.detach().cpu().contiguous().numpy().tobytes())
    hashes=gather(digest.hexdigest(),world);assert len(set(hashes))==1
    seconds=gather(elapsed,world)
    if rank==0:
        save(args.out/"update_record.json",result)
        save(args.out/"summary.json",{
            "passed":True,"parameter_updates_performed":False,"arm":"C","actor_gpus":2,
            "questions":16,"responses":64,"seconds":max(seconds),"seconds_by_rank":seconds,
            "online_tokens":sum(len(x["response_token_ids"]) for x in generated["records"]),
            "demo_tokens":sum(len(x["demo"]["response_token_ids"]) for x in batch),
            "gradient_identical_on_ranks":True,"gradient_sha256":hashes[0],
            "source_policy_adapter":request["adapter"],"source_adapter_sha256":request["adapter_sha256"],
            "source_request_sha256":sha(args.request),"source_response_sha256":sha(args.response),
            "peak_memory_by_rank":result["peak_memory_by_rank"],
            "optimizer_state_allocated":False,"additional_adam_moments_gib_per_rank":sum(x.numel() for x in params)*8/2**30,
            "scope":"Original two-actor update on the same outputs as the measured single-sampler benchmark; no additional generation or parameter updates."})
    barrier();torch.distributed.destroy_process_group()


if __name__=="__main__":main()
