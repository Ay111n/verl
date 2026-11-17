"""Record GPU ownership by process group throughout the bounded experiment."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.config.read_text());root=Path(__file__).resolve().parents[4]
run=root/"verl/rlvr_experiments/runs"/cfg["name"]
previous=json.loads((run/"gpu_usage_summary.json").read_text()) if (run/"gpu_usage_summary.json").exists() else {}
peak=previous.get("max_simultaneous_owned_gpus",0);observed=set(previous.get("observed_gpus",[]));samples=previous.get("samples",0)
while not cfg.get("strict_wall_budget",True) or time.time()<cfg["overall_deadline_unix"]:
    owners={w["pid"] for w in json.loads((run/"workers.json").read_text())}
    if (run/"owned_processes.json").exists():owners.update(json.loads((run/"owned_processes.json").read_text())["pids"])
    lookup=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    indices={v.strip():int(k) for k,v in (line.split(",") for line in lookup.splitlines())}
    raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader,nounits"],text=True)
    processes=[]
    for line in raw.splitlines():
        pid,uuid,memory=[x.strip() for x in line.split(",")]
        stat=Path(f"/proc/{pid}/stat")
        if not stat.exists():continue
        fields=stat.read_text().rsplit(")",1)[1].split();pgid=int(fields[2])
        if pgid in owners:processes.append({"pid":int(pid),"process_group":pgid,"gpu":indices[uuid],"memory_mib":memory})
    devices=sorted({x["gpu"] for x in processes});peak=max(peak,len(devices));observed.update(devices);samples+=1
    event={"time_unix":time.time(),"owned_gpu_count":len(devices),"devices":devices,"processes":processes}
    with (run/"gpu_usage.jsonl").open("a") as f:f.write(json.dumps(event)+"\n")
    summary={"max_simultaneous_owned_gpus":peak,"observed_gpus":sorted(observed),"samples":samples,
             "limit":cfg["maximum_concurrent_gpus"],"within_limit":peak<=cfg["maximum_concurrent_gpus"]}
    (run/"gpu_usage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    assert summary["within_limit"],summary
    main_done=(run/"pipeline_complete.json").exists() or (run/"pipeline_failure.json").exists()
    background_done=not (run/"background_sampling_contract.json").exists() or (run/"background_sampling_complete.json").exists()
    if main_done and background_done:break
    time.sleep(30)
