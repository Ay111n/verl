"""Cache complete-model reference log probabilities using the training backend."""
import argparse,json,time
from pathlib import Path
import torch
from paired_dpo import load_model,read_rows,save,sha,response_logps,adapter_hash,runtime

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True);p.add_argument("--model",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1);p.add_argument("--adapter")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=read_rows(a.data);tasks=[(r,l,s) for r in data for l in (["bo"] if a.adapter else ["bo","en"]) for s in ["chosen","rejected"]]
    tasks=tasks[a.shard::a.shards];path=a.output/f"reference.shard{a.shard}.jsonl"
    done={r["key"] for r in read_rows(path)} if path.exists() else set()
    model=load_model(a.model,adapter=a.adapter)
    meta={"data_sha256":sha(a.data),"model":a.model,"adapter":a.adapter,"runtime":runtime(),
        "adapter_initial_sha256":adapter_hash(model),"total":len(tasks)}
    save(a.output/f"manifest.shard{a.shard}.json",meta)
    t=time.time()
    with path.open("a",encoding="utf-8") as f,torch.no_grad():
        for row,lang,side in tasks:
            key=f'{row["pair_id"]}|{lang}|{side}'
            if key in done:continue
            ids=row[side]["response_token_ids"]
            v=response_logps(model,row["prompt_"+lang+"_token_ids"],ids)
            assert len(v)==len(ids) and torch.isfinite(v).all()
            rec={"key":key,"pair_id":row["pair_id"],"id":row["id"],"language":lang,"side":side,
                "sum_logprob":float(v.sum()),"mean_logprob":float(v.sum())/len(v),
                "response_tokens":len(ids),"first_logprob":float(v[0]),"last_logprob":float(v[-1])}
            f.write(json.dumps(rec)+"\n");f.flush();done.add(key)
            if len(done)%8==0 or len(done)==len(tasks):
                status={"completed":len(done),"total":len(tasks),"elapsed_seconds":time.time()-t,
                    "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9}
                save(a.output/f"status.shard{a.shard}.json",status);print(json.dumps(status),flush=True)
    save(a.output/f"complete.shard{a.shard}.json",{**meta,"rows":len(done),"sha256":sha(path),"elapsed_seconds":time.time()-t})
if __name__=="__main__":main()
