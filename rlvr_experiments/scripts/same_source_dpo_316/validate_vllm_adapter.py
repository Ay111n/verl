"""Check native LoRA loading using fixed sequences, without collecting new samples."""
import argparse,json,shutil
from pathlib import Path
import torch
from paired_dpo import read_rows,save,sha
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    root=a.root;name="same_source_dpo_316_20261116"
    data=root/"finetuning/rl/final_data"/name
    run=root/"verl/rlvr_experiments/runs"/name
    base=root/"finetuning/model/sft/en_cot_49800/merged/checkpoint-9000"
    adapter=root/"finetuning/model/rl"/name/"base/adapter"
    zero=root/"finetuning/model/rl"/name/"validation_zero_adapter";zero.mkdir(exist_ok=True)
    from safetensors.torch import load_file,save as serialize
    state=load_file(str(adapter/"adapter_model.safetensors"),device="cpu")
    zeroed=0
    for k in state:
        if "lora_B" in k:state[k].zero_();zeroed+=1
    assert zeroed>0
    (zero/"adapter_model.safetensors").write_bytes(serialize(state,metadata={"format":"pt"}))
    shutil.copy2(adapter/"adapter_config.json",zero/"adapter_config.json")
    del state
    rows=read_rows(data/"pairs.jsonl");plan=json.loads((data/"training_plan.json").read_text())
    chosen=[rows[i] for i in plan["calibration_indices"][::2]]
    examples=[(r,s) for r in chosen for s in ["chosen","rejected"]]
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    llm=LLM(model=str(base),dtype="bfloat16",tensor_parallel_size=1,max_model_len=32768,
        gpu_memory_utilization=.85,max_num_seqs=16,max_num_batched_tokens=2048,
        enable_prefix_caching=False,enable_chunked_prefill=True,safetensors_load_strategy="eager",
        generation_config="vllm",enable_lora=True,max_lora_rank=16,max_cpu_loras=2,
        logprobs_mode="raw_logprobs")
    prompts=[{"prompt_token_ids":r["prompt_bo_token_ids"]+r[s]["response_token_ids"]} for r,s in examples]
    params=SamplingParams(max_tokens=1,temperature=1.,top_p=1.,top_k=-1,seed=42,prompt_logprobs=0,detokenize=False)
    scores={}
    for name,request in [("none",None),("zero",LoRARequest("zero",71,str(zero))),("trained",LoRARequest("trained",72,str(adapter)))]:
        output=llm.generate(prompts,params,lora_request=request)
        values=[]
        for (r,side),o in zip(examples,output,strict=True):
            ids=r[side]["response_token_ids"];start=len(r["prompt_bo_token_ids"])
            logps=[o.prompt_logprobs[start+i][t].logprob for i,t in enumerate(ids)]
            values.append({"pair_id":r["pair_id"],"side":side,"tokens":len(ids),"sum":sum(logps),"next_token":o.outputs[0].token_ids[0]})
        scores[name]=values
    zero_diff=[abs(x["sum"]-y["sum"])/x["tokens"] for x,y in zip(scores["none"],scores["zero"])]
    trained_diff=[abs(x["sum"]-y["sum"])/x["tokens"] for x,y in zip(scores["none"],scores["trained"])]
    result={"sequences":len(examples),"zero_B_tensors":zeroed,
        "zero_vs_no_adapter_max_abs_mean_logprob":max(zero_diff),
        "zero_vs_no_adapter_next_token_matches":sum(x["next_token"]==y["next_token"] for x,y in zip(scores["none"],scores["zero"])),
        "trained_vs_no_adapter_max_abs_mean_logprob":max(trained_diff),
        "trained_adapter_sha256":sha(adapter/"adapter_model.safetensors"),
        "raw_fixed_sequence_scores":scores,
        "scope":"Teacher forcing fixed training sequences; one unused next token per request; no generated training data"}
    save(run/"vllm_adapter_validation.json",result)
    assert max(zero_diff)<1e-4,result
    assert max(trained_diff)>1e-7,"Trained adapter has no observed effect"
    print(json.dumps({k:v for k,v in result.items() if k!="raw_fixed_sequence_scores"}),flush=True)
if __name__=="__main__":main()
