"""Dedicated vLLM actor; one completed request per immutable policy version."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from common import paths, save, sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rpc",type=Path,required=True)
    args=ap.parse_args()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==1
    args.rpc.mkdir(parents=True,exist_ok=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    cfg=paths()
    llm=LLM(model=str(cfg["model_path"]),dtype="bfloat16",tensor_parallel_size=1,
        max_model_len=32768,gpu_memory_utilization=.85,max_num_seqs=64,
        max_num_batched_tokens=2048,enable_prefix_caching=False,enable_chunked_prefill=True,
        safetensors_load_strategy="eager",generation_config="vllm",
        enable_lora=True,max_lora_rank=16,max_cpu_loras=2,logprobs_mode="raw_logprobs")
    save(args.rpc/"ready.json",{"pid":os.getpid(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"],
        "prefix_cache":False,"max_model_len":32768,"max_num_seqs":64,"max_num_batched_tokens":2048,
        "model":str(cfg["model_path"])})
    active=None;version=1000
    while True:
        pending=[p for p in sorted(args.rpc.glob("*.request.json"),key=lambda p:p.stat().st_mtime)
                 if not p.with_name(p.name.replace(".request.",".response.")).exists()]
        if not pending: time.sleep(.5);continue
        for file in pending:
            request=json.loads(file.read_text());start=time.monotonic()
            dest=file.with_name(file.name.replace(".request.",".response."))
            try:
                if request["mode"]=="shutdown":
                    save(dest,{"request_id":request["request_id"],"stopped":True});return
                if active is not None:
                    assert llm.llm_engine.remove_lora(active);active=None
                adapter=request.get("adapter");lora=None;digest=None
                if adapter:
                    digest=sha(Path(adapter)/"adapter_model.safetensors")
                    assert digest==request["adapter_sha256"]
                    version+=1;active=version
                    lora=LoRARequest(f"luffy_policy_{version}",version,adapter)
                params=[]
                for row in request["prompts"]:
                    if request["mode"]=="score":
                        params.append(SamplingParams(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,
                            seed=row.get("seed",42),prompt_logprobs=0,detokenize=False))
                    else:
                        assert len(row["prompt_token_ids"])+request["max_new_tokens"]<=32768
                        params.append(SamplingParams(max_tokens=request["max_new_tokens"],
                            temperature=request["temperature"],top_p=request["top_p"],top_k=request["top_k"],
                            seed=row["seed"],skip_special_tokens=False,logprobs=0))
                generated=llm.generate([{"prompt_token_ids":r["prompt_token_ids"]} for r in request["prompts"]],
                    params,lora_request=lora,use_tqdm=True)
                records=[]
                for row,out in zip(request["prompts"],generated,strict=True):
                    c=out.outputs[0]
                    assert out.prompt_token_ids==row["prompt_token_ids"]
                    item={"id":row["id"],"rollout_index":row.get("rollout_index"),
                        "seed":row.get("seed"),"prompt_token_ids":out.prompt_token_ids,
                        "response_token_ids":list(c.token_ids),"completion":c.text,
                        "finish_reason":c.finish_reason,"stop_reason":c.stop_reason}
                    if request["mode"]=="score":
                        item["logprobs"]=[out.prompt_logprobs[i][out.prompt_token_ids[i]].logprob
                            for i in range(row["score_start"],len(out.prompt_token_ids))]
                    else:
                        item["raw_logprobs"]=[v[token].logprob for token,v in zip(c.token_ids,c.logprobs,strict=True)]
                    records.append(item)
                save(dest,{"request_id":request["request_id"],"records":records,"adapter_sha256":digest,
                    "lora_id":active,"elapsed_seconds":time.monotonic()-start})
                print(json.dumps({"event":"completed","request":request["request_id"],"count":len(records),
                    "seconds":time.monotonic()-start}),flush=True)
            except Exception:
                save(dest,{"request_id":request["request_id"],"error":traceback.format_exc()});raise

if __name__=="__main__":main()
