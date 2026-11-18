"""Single dedicated GPU sampler with immutable adapters and per-row seeds."""
import argparse,json,os,time,traceback
from pathlib import Path
from common import save,sha,verified_adapter
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True);ap.add_argument("--rpc",type=Path,required=True)
    ap.add_argument("--max-num-seqs",type=int,default=16)
    a=ap.parse_args();assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==1
    a.rpc.mkdir(parents=True,exist_ok=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=a.model,dtype="bfloat16",tensor_parallel_size=1,max_model_len=32768,
        gpu_memory_utilization=.90,max_num_seqs=a.max_num_seqs,max_num_batched_tokens=2048,
        enable_prefix_caching=False,enable_chunked_prefill=True,safetensors_load_strategy="eager",
        generation_config="vllm",enable_lora=True,max_lora_rank=16,max_cpu_loras=2,logprobs_mode="raw_logprobs")
    save(a.rpc/"ready.json",{"pid":os.getpid(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"],
        "model":a.model,"max_num_seqs":a.max_num_seqs,"max_model_len":32768})
    active=None;version=1000
    loaded_request=None;loaded_digest=None
    while True:
        pending=[p for p in sorted(a.rpc.glob("*.request.json"),key=lambda p:p.stat().st_mtime)
                 if not p.with_name(p.name.replace(".request.",".response.")).exists()]
        if not pending:time.sleep(.5);continue
        for file in pending:
            r=json.loads(file.read_text());dest=file.with_name(file.name.replace(".request.",".response."));start=time.monotonic()
            try:
                if r["mode"]=="shutdown":save(dest,{"request_id":r["request_id"],"stopped":True});return
                adapter=r.get("adapter");lora=None;digest=None
                eager=r.get("preload_adapter",False)
                if adapter:
                    digest=verified_adapter(Path(adapter)/"adapter_model.safetensors",r["adapter_sha256"])
                if eager and digest and digest==loaded_digest:
                    lora=loaded_request
                else:
                    if active is not None:
                        assert llm.llm_engine.remove_lora(active)
                    active=None;loaded_digest=None;loaded_request=None
                    if adapter:
                        version+=1;active=version;lora=LoRARequest(f"math_mix_{version}",version,adapter)
                        if eager:
                            assert llm.llm_engine.add_lora(lora)
                            assert active in llm.llm_engine.list_loras()
                            loaded_digest=digest;loaded_request=lora
                if r["mode"]=="sync":
                    assert eager and digest and lora
                    save(dest,{"request_id":r["request_id"],"adapter_sha256":digest,"lora_id":active,
                               "elapsed_seconds":time.monotonic()-start})
                    continue
                params=[]
                for row in r["prompts"]:
                    assert len(row["prompt_token_ids"])+r["max_new_tokens"]<=32768
                    params.append(SamplingParams(max_tokens=r["max_new_tokens"],temperature=r["temperature"],
                        top_p=r["top_p"],top_k=r["top_k"],seed=row["seed"],skip_special_tokens=False,
                        logprobs=0 if r.get("return_logprobs",False) else None))
                outs=llm.generate([{"prompt_token_ids":x["prompt_token_ids"]} for x in r["prompts"]],
                    params,lora_request=lora,use_tqdm=True)
                records=[]
                for row,out in zip(r["prompts"],outs,strict=True):
                    c=out.outputs[0];assert out.prompt_token_ids==row["prompt_token_ids"]
                    v={"id":row["id"],"rollout_index":row.get("rollout_index"),"seed":row["seed"],
                       "prompt_token_ids":out.prompt_token_ids,"response_token_ids":list(c.token_ids),
                       "completion":c.text,"finish_reason":c.finish_reason,"stop_reason":c.stop_reason}
                    if r.get("return_logprobs",False):
                        v["raw_logprobs"]=[x[token].logprob for token,x in zip(c.token_ids,c.logprobs,strict=True)]
                    records.append(v)
                save(dest,{"request_id":r["request_id"],"adapter_sha256":digest,"lora_id":active,
                           "records":records,"elapsed_seconds":time.monotonic()-start})
                print(json.dumps({"event":"complete","request":r["request_id"],"count":len(records),"seconds":time.monotonic()-start}),flush=True)
            except Exception:
                save(dest,{"request_id":r["request_id"],"error":traceback.format_exc()});raise
if __name__=="__main__":main()
