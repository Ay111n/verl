"""Read-only cost audit of completed LUFFY pilot updates; no model loading."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def quantile(values, q):
    values=sorted(values);position=(len(values)-1)*q
    left=int(position);right=min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def generation_tails(log, last_step):
    progress=[];batches=[]
    for line in log.replace("\r","\n").splitlines():
        match=re.search(r"Processed prompts:.*?\b(\d+)/(\d+) \[([\d:]+)<",line)
        if match:
            n,total=int(match[1]),int(match[2]);seconds=0
            for value in match[3].split(":"):seconds=seconds*60+int(value)
            progress.append((n,total,seconds))
        if line.startswith('{"event": "completed"'):
            event=json.loads(line)
            match=re.fullmatch(r"A-step-(\d+)",event["request"])
            if match and int(match[1])<=last_step and event["count"]==64:
                checkpoints={str(k):next((s for n,total,s in progress if total==64 and n>=k),None)
                             for k in (16,32,48,58,64)}
                assert all(v is not None for v in checkpoints.values())
                batches.append({"step":int(match[1]),"elapsed_seconds":event["seconds"],
                                "completion_seconds":checkpoints})
            progress=[]
    return batches


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--max-step",type=int,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();run=args.run_root
    files=[run/"A"/f"step-{step:03d}.json" for step in range(1,args.max_step+1)]
    records=[read(path) for path in files]
    assert [r["step"] for r in records]==list(range(1,args.max_step+1))
    outputs=[o for record in records for o in record["online"]]
    assert len(outputs)==sum(r["rollout_count"] for r in records)
    assert all(len(r["online"])==r["question_count"]*4 for r in records)
    config=read(run/"preflight.json")["config"]
    rollout=sum(r["rollout_seconds"] for r in records)
    training=sum(r["training_seconds"] for r in records)
    process=next(x for x in read(run/"pipeline_processes.json") if x["label"]=="train-A")
    elapsed=files[-1].stat().st_mtime-process["started_unix"]
    visits=sum(r["question_count"] for r in records)
    lengths=[o["tokens"] for o in outputs];tokens=sum(lengths)
    full=[r for r in records if r["question_count"]==16]
    worker_bytes=(run/"worker.log").read_bytes()
    tails=generation_tails(worker_bytes.decode(errors="replace"),args.max_step)
    assert len(tails)==len(full)
    source={str(p.relative_to(run)):{"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())} for p in files}
    source["worker.log"]={"prefix_bytes":len(worker_bytes),"prefix_sha256":digest(worker_bytes)}
    last25=sum(b["elapsed_seconds"]-b["completion_seconds"]["48"] for b in tails)
    last10=sum(b["elapsed_seconds"]-b["completion_seconds"]["58"] for b in tails)
    generation_time=sum(b["elapsed_seconds"] for b in tails)
    per_visit=elapsed/visits
    share=rollout/(rollout+training)
    report={
        "as_of_beijing":datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "run":run.name,"arm":"A","completed_steps":args.max_step,"question_visits":visits,
        "online_outputs":len(outputs),"output_tokens":tokens,
        "timing_boundaries":{"train_started_unix":process["started_unix"],"last_record_unix":files[-1].stat().st_mtime},
        "phase_seconds":{"generation":rollout,"training_including_reference_and_diagnostics":training,
                         "startup_checkpoints_diagnostics_and_other":elapsed-rollout-training,"total":elapsed},
        "phase_share_total":{"generation":rollout/elapsed,"training":training/elapsed,
                             "other":(elapsed-rollout-training)/elapsed},
        "full_batch_mean_seconds":{"generation":statistics.mean(r["rollout_seconds"] for r in full),
                                   "training":statistics.mean(r["training_seconds"] for r in full)},
        "aggregate_generation_tokens_per_second":tokens/rollout,
        "lengths":{"mean":tokens/len(outputs),"median":quantile(lengths,.5),"p90":quantile(lengths,.9),
                   "p95":quantile(lengths,.95),"truncated":sum(o["verification"]["truncated"] for o in outputs),
                   "over4096":sum(n>4096 for n in lengths),"tokens_above4096":sum(max(0,n-4096) for n in lengths)},
        "tails":{"full_batches":len(tails),"after75percent_complete_mean_seconds":last25/len(tails),
                 "after75percent_complete_time_fraction":last25/generation_time,
                 "after90percent_complete_mean_seconds":last10/len(tails),
                 "after90percent_complete_time_fraction":last10/generation_time,
                 "note":"Approximate completion times from second-rounded tqdm logs; not active KV-cache sequence counts."},
        "groups":{"all_correct":sum(r["all_correct_groups"] for r in records),
                  "all_wrong":sum(r["zero_correct_groups"] for r in records),
                  "mixed":sum(r["mixed_groups"] for r in records)},
        "budget":{"unique_questions":388,"epochs":config["epochs"],"online_samples_per_visit":4,
                  "arms":3,"online_outputs_per_arm":388*config["epochs"]*4,
                  "online_outputs_three_arms":388*config["epochs"]*4*3,
                  "projected_output_tokens_per_arm":388*config["epochs"]*4*tokens/len(outputs),
                  "projected_output_tokens_three_arms":388*config["epochs"]*4*3*tokens/len(outputs)},
        "fixed_epoch_scaling_days_per_arm":{str(n):n*config["epochs"]*per_visit/86400 for n in (388,1000,5000,10000)},
        "scaling_assumption":"Linear projection of measured A cost per question visit at three epochs, four outputs and similar lengths; not a measured larger-data run. B/C demonstration cost is additional.",
        "illustrative_speedup_not_benchmarked":{"training_infinite_upper_bound":1/share,
            "generation_2x_only":1/(share/2+(1-share)),
            "generation_2x_and_training_2x":2.0},
        "config":config,"sources":source,"audit_script_sha256":digest(Path(__file__).read_bytes()),
        "gpu_probe_performed":False,"new_rollouts_performed":False,"optimizer_updates_performed":False}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"worker_log_prefix.log").write_bytes(worker_bytes)
    (args.out/"runtime_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.out/"generation_tails.json").write_text(json.dumps(tails,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("sources","config")},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
