"""Freeze HF references, positive English targets, and a domain/length shuffled control."""
import argparse,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
from paired_dpo import read_rows,save,sha,object_sha

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True);a=p.parse_args()
    records=read_rows(a.data);scored={}
    for s in range(4):
        comp=json.loads((a.reference/f"complete.shard{s}.json").read_text())
        path=a.reference/f"reference.shard{s}.jsonl"
        assert comp["sha256"]==sha(path) and comp["data_sha256"]==sha(a.data)
        for r in read_rows(path):
            assert r["key"] not in scored;scored[r["key"]]=r
    assert len(scored)==4*len(records)==1264
    targets=[];diffs=defaultdict(list)
    for r in records:
        ref={}
        for lang in ["bo","en"]:
            for side in ["chosen","rejected"]:
                s=scored[f'{r["pair_id"]}|{lang}|{side}'];n=len(r[side]["response_token_ids"])
                assert s["response_tokens"]==n
                assert abs(s["mean_logprob"]*n-s["sum_logprob"])<1e-6
                assert np.isfinite(s["sum_logprob"]) and s["sum_logprob"]<=0
                ref[lang+"_"+side]=s["sum_logprob"]
                diffs[lang].append(s["mean_logprob"]-r[side]["vllm_reference"][lang]["mean_logprob"])
        nc=len(r["chosen"]["response_token_ids"]);nr=len(r["rejected"]["response_token_ids"])
        en=ref["en_chosen"]/nc-ref["en_rejected"]/nr;bo=ref["bo_chosen"]/nc-ref["bo_rejected"]/nr
        targets.append({"id":r["id"],"pair_id":r["pair_id"],"domain":r["domain"],"nc":nc,"nr":nr,
            **ref,"target":en,"gate":int(en>0),"initial_bo_margin":bo})
    rng=random.Random(20261116)
    groups=defaultdict(list)
    for domain in ["knowledge","math"]:
        part=sorted((r for r in targets if r["domain"]==domain and r["gate"]),key=lambda r:(r["nc"]+r["nr"],r["id"]))
        for i,r in enumerate(part):groups[(domain,min(3,4*i//len(part)))].append(r)
    for key,group in groups.items():
        assert len(group)>1
        donor=group.copy();rng.shuffle(donor)
        for i,r in enumerate(donor):
            other=donor[(i+1)%len(donor)]
            r["shuffled_target"]=other["target"];r["shuffled_donor_id"]=other["id"];r["shuffle_stratum"]=list(key)
    for r in targets:
        if not r["gate"]:r["shuffled_target"]=r["target"];r["shuffled_donor_id"]=None
    assert sorted(r["target"] for r in targets if r["gate"])==sorted(r["shuffled_target"] for r in targets if r["gate"])
    with (a.data.parent/"targets.jsonl").open("w",encoding="utf-8") as f:
        for r in targets:f.write(json.dumps(r)+"\n")
    order=list(range(len(records)));random.Random(42).shuffle(order)
    calibration=[]
    for domain in ["knowledge","math"]:
        group=sorted([i for i,r in enumerate(records) if r["domain"]==domain],key=lambda i:sum(len(records[i][s]["response_token_ids"]) for s in ["chosen","rejected"]))
        for b in range(4):
            bucket=group[b*len(group)//4:(b+1)*len(group)//4]
            calibration.extend(sorted(bucket,key=lambda i:object_sha(["calibration",42,records[i]["id"]]))[:2])
    plan={"data_sha256":sha(a.data),"targets_sha256":sha(a.data.parent/"targets.jsonl"),
        "order":order,"order_sha256":object_sha([records[i]["pair_id"] for i in order]),
        "calibration_indices":calibration,"seed":42,"epochs":1,"effective_batch":8,"micro_batch":1,
        "optimizer_steps":40,"last_batch":4,"learning_rate":1e-5,"warmup_steps":4,
        "scheduler":"linear decay after warmup","optimizer":"AdamW","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,
        "max_grad_norm":1.,"beta":.1,"alpha":1.,"lora_rank":16,"lora_alpha":32,"lora_dropout":0.,
        "max_sequence_length":8192,"lambda_calibration":"0.1 * aggregate base gradient norm / aggregate true auxiliary gradient norm; 16 fixed stratified training pairs; same lambda for true/shuffled",
        "teacher_reference":"frozen full merged SFT; HF training backend; both BO-origin response sequences scored under both inputs",
        "gate":"frozen 1[English margin > 0]; same gate in shuffled control",
        "shuffle":"derangement within domain and length quartile of gate-positive items",
        "dev_selection":"evaluate final one-epoch adapter once per arm; no checkpoint/lambda selection on DEV"}
    save(a.data.parent/"training_plan.json",plan)
    summary={"pairs":len(targets),"english_positive":sum(r["gate"] for r in targets),
        "true_initial_active":sum(r["gate"] and r["target"]>r["initial_bo_margin"] for r in targets),
        "shuffled_initial_active":sum(r["gate"] and r["shuffled_target"]>r["initial_bo_margin"] for r in targets),
        "calibration_pairs":len(calibration),"calibration_gate_positive":sum(targets[i]["gate"] for i in calibration),
        "vllm_hf_mean_logprob_differences":{k:{"mean":float(np.mean(v)),"max_abs":float(np.max(np.abs(v))),"p95_abs":float(np.quantile(np.abs(v),.95))} for k,v in diffs.items()},
        "reference_sources":{p.name:sha(p) for p in a.reference.glob("reference.shard*.jsonl")},
        "plan_sha256":sha(a.data.parent/"training_plan.json"),"targets_sha256":sha(a.data.parent/"targets.jsonl")}
    save(a.reference/"validation.json",summary);print(json.dumps(summary))
if __name__=="__main__":main()
