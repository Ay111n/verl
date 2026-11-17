"""vLLM worker that can publish complete four-response groups before the batch ends."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback
from common import paths,save,sha


def record(row,out,scoring=False):
    choice=out.outputs[0]
    assert out.prompt_token_ids==row["prompt_token_ids"]
    item={"id":row["id"],"rollout_index":row.get("rollout_index"),"question_index":row.get("question_index"),
          "seed":row.get("seed"),"prompt_token_ids":out.prompt_token_ids,
          "response_token_ids":list(choice.token_ids),"completion":choice.text,
          "finish_reason":choice.finish_reason,"stop_reason":choice.stop_reason}
    if scoring:
        item["logprobs"]=[out.prompt_logprobs[i][out.prompt_token_ids[i]].logprob for i in range(row["score_start"],len(out.prompt_token_ids))]
    else:
        item["raw_logprobs"]=[v[token].logprob for token,v in zip(choice.token_ids,choice.logprobs,strict=True)]
    return item


def stream_generate(llm,prompts,params,lora,on_complete):
    ids=llm.enqueue(prompts,sampling_params=params,lora_request=lora,use_tqdm=False)
    assert len(ids)==len(prompts) and len(set(ids))==len(ids)
    order={request_id:i for i,request_id in enumerate(ids)};outputs={}
    # Same stepping loop as installed vLLM 0.19.1 LLM._run_engine; callbacks
    # publish finished responses only. All prompts are enqueued before stepping.
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if not output.finished:continue
            assert output.request_id in order and output.request_id not in outputs
            outputs[output.request_id]=output
            on_complete(order[output.request_id],output)
    assert set(outputs)==set(ids)
    return [outputs[request_id] for request_id in ids]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--rpc",type=Path,required=True)
    args=parser.parse_args();args.rpc.mkdir(parents=True,exist_ok=True)
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))==1
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    p=paths()
    llm=LLM(model=str(p["model_path"]),dtype="bfloat16",tensor_parallel_size=1,max_model_len=32768,
        gpu_memory_utilization=.85,max_num_seqs=64,max_num_batched_tokens=2048,
        enable_prefix_caching=False,enable_chunked_prefill=True,safetensors_load_strategy="eager",
        generation_config="vllm",enable_lora=True,max_lora_rank=16,max_cpu_loras=2,logprobs_mode="raw_logprobs")
    save(args.rpc/"ready.json",{"pid":os.getpid(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"],
        "model":str(p["model_path"]),"max_model_len":32768,"max_num_seqs":64,
        "max_num_batched_tokens":2048,"prefix_cache":False,"parameter_updates":False})
    active=None;version=1000
    while True:
        pending=[f for f in sorted(args.rpc.glob("*.request.json"),key=lambda f:f.stat().st_mtime)
                 if not f.with_name(f.name.replace(".request.",".response.")).exists()]
        if not pending:time.sleep(.1);continue
        for file in pending:
            req=json.loads(file.read_text());start=time.monotonic();dest=file.with_name(file.name.replace(".request.",".response."))
            try:
                if req["mode"]=="shutdown":save(dest,{"request_id":req["request_id"],"stopped":True});return
                if active is not None:assert llm.llm_engine.remove_lora(active);active=None
                adapter=req.get("adapter");lora=None;digest=None
                if adapter:
                    digest=sha(Path(adapter)/"adapter_model.safetensors");assert digest==req["adapter_sha256"]
                    version+=1;active=version;lora=LoRARequest(f"luffy_perf_{version}",version,adapter)
                scoring=req["mode"]=="score";params=[]
                for row in req["prompts"]:
                    if scoring:
                        params.append(SamplingParams(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,
                            seed=row.get("seed",42),prompt_logprobs=0,detokenize=False))
                    else:
                        assert len(row["prompt_token_ids"])+req["max_new_tokens"]<=32768
                        params.append(SamplingParams(max_tokens=req["max_new_tokens"],temperature=req["temperature"],
                            top_p=req["top_p"],top_k=req["top_k"],seed=row["seed"],skip_special_tokens=False,logprobs=0))
                prompts=[{"prompt_token_ids":x["prompt_token_ids"]} for x in req["prompts"]]
                completions=[]
                if req.get("stream_groups") and not scoring:
                    assert set(Counter(x["id"] for x in req["prompts"]).values())=={4}
                    completed={};groups={};progress=args.rpc/(req["request_id"]+".groups")
                    progress.mkdir(exist_ok=True)
                    def on_complete(index,output):
                        row=req["prompts"][index];value=record(row,output)
                        completed[index]=value;items=groups.setdefault(row["id"],[]);items.append(value)
                        completions.append({"id":row["id"],"rollout_index":row["rollout_index"],
                            "question_index":row["question_index"],"elapsed_seconds":time.monotonic()-start})
                        if len(items)==4:
                            assert {x["rollout_index"] for x in items}=={0,1,2,3}
                            save(progress/("group-%03d.json"%row["question_index"]),
                                {"id":row["id"],"question_index":row["question_index"],"adapter_sha256":digest,
                                 "records":sorted(items,key=lambda x:x["rollout_index"]),"elapsed_seconds":time.monotonic()-start})
                    stream_generate(llm,prompts,params,lora,on_complete)
                    records=[completed[i] for i in range(len(req["prompts"]))]
                else:
                    outs=llm.generate(prompts,params,lora_request=lora,use_tqdm=True)
                    records=[record(row,out,scoring) for row,out in zip(req["prompts"],outs,strict=True)]
                save(dest,{"request_id":req["request_id"],"records":records,"adapter_sha256":digest,
                    "lora_id":active,"elapsed_seconds":time.monotonic()-start,"completion_events":completions})
                print(json.dumps({"request":req["request_id"],"responses":len(records),"elapsed_seconds":time.monotonic()-start}),flush=True)
            except Exception:
                save(dest,{"request_id":req["request_id"],"error":traceback.format_exc()});raise


if __name__=="__main__":main()
