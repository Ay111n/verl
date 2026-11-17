"""Three-GPU proof: two samplers feed one gradient-only actor without policy lag."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from common import paths,configuration,read_rows,load_model,adapter_hash,save,sha
from benchmark_actor import NoUpdateOptimizer
import batched_update


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rpc",type=Path,nargs=2,required=True)
    parser.add_argument("--micro-batch",type=int,default=1);parser.add_argument("--head-chunk",type=int,default=512)
    parser.add_argument("--readiness-check",type=Path)
    args=parser.parse_args()
    assert not args.out.exists() or not any(args.out.iterdir()), "Use an empty output directory for a new benchmark attempt"
    args.out.mkdir(parents=True,exist_ok=True)
    p=paths();cfg=configuration(p);cfg["nll_alpha"]=json.loads((p["run_root"]/"calibration.json").read_text())["alpha"]
    source=json.loads((p["run_root"]/"rpc/A-step-051.request.json").read_text())
    ids=list(dict.fromkeys(x["id"] for x in source["prompts"]));assert len(ids)==16
    by_id={row["id"]:row for row in read_rows(p["train_data"])}
    batch=[]
    for qid in ids:
        row=dict(by_id[qid]);row["prompt_token_ids"]=next(x["prompt_token_ids"] for x in source["prompts"] if x["id"]==qid);batch.append(row)
    torch.cuda.set_device(0)
    model=load_model(str(p["model_path"]),cfg["initial_seed"],adapter=source["adapter"],trainable_adapter=True)
    initial=adapter_hash(model);params=[v for v in model.parameters() if v.requires_grad];opt=NoUpdateOptimizer(model)
    warm=batched_update.response_logps_batch(model,[x["prompt_token_ids"] for x in batch[:args.micro_batch]],
        [x["demo"]["response_token_ids"][:512] for x in batch[:args.micro_batch]],args.head_chunk,temperature=.6)
    (-sum(x.mean() for x in warm)/len(warm)).backward();del warm
    model.zero_grad(set_to_none=True);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    accumulated=[torch.zeros_like(v) for v in params];local_cfg={**cfg,"max_grad_norm":float("inf")}
    if args.readiness_check:
        ready_deadline=time.monotonic()+1200
        while not args.readiness_check.exists():
            assert time.monotonic()<ready_deadline,"Workers did not pass their streaming interface check"
            time.sleep(.2)
        assert json.loads(args.readiness_check.read_text())["passed"]
    # Exclude any preceding diagnostic request from the timed interval.
    idle_deadline=time.monotonic()+900
    while any(not f.with_name(f.name.replace(".request.",".response.")).exists()
              for rpc in args.rpc for f in rpc.glob("*.request.json")):
        assert time.monotonic()<idle_deadline,"Sampler queue did not become idle"
        time.sleep(.1)
    torch.cuda.reset_peak_memory_stats();requests=[];started=time.monotonic();started_unix=time.time()
    for shard,rpc in enumerate(args.rpc):
        req={**source,"request_id":f"perf-stream-{shard}","stream_groups":True,
             "prompts":[{**x,"question_index":ids.index(x["id"])} for x in source["prompts"] if ids.index(x["id"])%2==shard]}
        assert len(req["prompts"])==32
        save(rpc/(req["request_id"]+".request.json"),req);requests.append(req)
    seen=set();events=[];all_records={};deadline=time.monotonic()+1800
    while len(seen)<16:
        ready=[]
        for i in range(16):
            if i in seen:continue
            rpc=args.rpc[i%2];file=rpc/(f"perf-stream-{i%2}.groups")/(f"group-{i:03d}.json")
            if file.exists():ready.append((file.stat().st_mtime,i,file))
        if not ready:
            for shard,rpc in enumerate(args.rpc):
                response=rpc/(f"perf-stream-{shard}.response.json")
                if response.exists():assert not json.loads(response.read_text()).get("error"),response.read_text()
            assert time.monotonic()<deadline,"Timed out waiting for complete question groups"
            time.sleep(.05);continue
        _,i,file=min(ready);value=json.loads(file.read_text());assert value["id"]==ids[i]
        assert value["adapter_sha256"]==source["adapter_sha256"] and len(value["records"])==4
        begin=time.monotonic();generated={"records":value["records"],"adapter_sha256":value["adapter_sha256"],"elapsed_seconds":0.}
        record=batched_update.update(model,opt,params,[batch[i]],generated,"C",local_cfg,0,1,1,micro_batch=args.micro_batch,head_chunk=args.head_chunk)
        for total,parameter in zip(accumulated,params):total.add_(parameter.grad,alpha=1/16)
        all_records[i]=value["records"];seen.add(i)
        events.append({"question_index":i,"sampling_group_ready_unix":file.stat().st_mtime,
            "backward_start_seconds":begin-started,"backward_end_seconds":time.monotonic()-started,
            "tokens":sum(len(x["response_token_ids"]) for x in value["records"]),"correct":record["correct"]})
        print(json.dumps({"event":"group_backward_complete","questions":len(seen),"elapsed":time.monotonic()-started}),flush=True)
    for parameter,total in zip(params,accumulated):parameter.grad=total
    total_norm=torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"]);torch.cuda.synchronize()
    elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**30
    stream_gradient=torch.cat([v.grad.detach().cpu().flatten() for v in params]);assert adapter_hash(model)==initial
    responses=[]
    for shard,rpc in enumerate(args.rpc):
        file=rpc/(f"perf-stream-{shard}.response.json")
        while not file.exists():assert time.monotonic()<deadline;time.sleep(.1)
        response=json.loads(file.read_text());assert not response.get("error"),response
        responses.append(response)
        for output in response["records"]:
            i=ids.index(output["id"]);assert output==all_records[i][output["rollout_index"]]
    final_generated={"records":[x for i in range(16) for x in all_records[i]],
        "adapter_sha256":source["adapter_sha256"],"elapsed_seconds":max(x["elapsed_seconds"] for x in responses)}
    save(args.out/"combined_streamed_outputs.json",final_generated)
    save(args.out/"stream_phase_complete.json",{"end_to_end_seconds":elapsed,"completed_unix":time.time(),"parameters_unchanged":True})
    # Replay exactly the same outputs without streaming to verify the full-batch gradient.
    del accumulated;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    replay_start=time.monotonic()
    replay=batched_update.update(model,opt,params,batch,final_generated,"C",cfg,0,1,1,micro_batch=args.micro_batch,head_chunk=args.head_chunk)
    replay_seconds=time.monotonic()-replay_start
    replay_gradient=torch.cat([v.grad.detach().cpu().flatten() for v in params]);left=stream_gradient.double();right=replay_gradient.double()
    cosine=float(torch.dot(left,right)/(left.norm()*right.norm()));change=float(abs(left.norm()/right.norm()-1))
    difference=float((left-right).norm()/right.norm());passed=cosine>=.999 and change<=.02 and difference<=.05
    assert adapter_hash(model)==initial
    report={"complete":True,"passed":passed,"parameter_updates_performed":False,
        "model_unchanged":True,"source_policy_adapter":source["adapter"],"source_adapter_sha256":source["adapter_sha256"],
        "micro_batch":args.micro_batch,"head_chunk":args.head_chunk,"actor_gpus":1,"sampling_gpus":2,"questions":16,"responses_per_question":4,
        "max_model_len":32768,"max_new_tokens":8192,"likelihood_temperature":.6,"arm":"C",
        "end_to_end_seconds":elapsed,"generation_max_worker_seconds":final_generated["elapsed_seconds"],
        "offline_batched_replay_training_seconds":replay_seconds,
        "component_sum_without_overlap_seconds":final_generated["elapsed_seconds"]+replay_seconds,
        "component_sum_is_counterfactual_not_a_separate_full_run":True,
        "gradient_cosine":cosine,"relative_gradient_norm_change":change,"relative_gradient_difference":difference,
        "stream_total_gradient_norm_before_clip":float(total_norm),"peak_actor_allocated_gib":peak,
        "output_tokens":sum(len(x["response_token_ids"]) for x in final_generated["records"]),
        "correct":sum(x["correct"] for x in events),"events":events,
        "started_unix":started_unix,"completed_unix":time.time(),
        "note":"Each complete four-response group is scored while the samplers finish remaining groups. Model parameters remain fixed until all gradients have been accumulated; no stale-policy pipeline is introduced."}
    save(args.out/"summary.json",report);save(args.out/"offline_replay_record.json",replay)
    print(json.dumps({k:v for k,v in report.items() if k!="events"}),flush=True)


if __name__=="__main__":main()
