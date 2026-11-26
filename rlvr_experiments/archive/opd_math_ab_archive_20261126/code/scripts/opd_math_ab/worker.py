"""Persistent single-GPU vLLM worker; reload a unique LoRA for each policy step."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from common import paths, read_rows, save, sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", type=Path, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    args = ap.parse_args()
    root = args.rpc
    root.mkdir(parents=True, exist_ok=True)
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    cfg = paths()
    llm = LLM(
        model=str(cfg["model_path"]), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=32768, gpu_memory_utilization=.90, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=2048, enable_prefix_caching=False,
        enable_chunked_prefill=True, safetensors_load_strategy="eager",
        generation_config="vllm", enable_lora=True, max_lora_rank=16,
        max_cpu_loras=2, logprobs_mode="raw_logprobs",
    )
    save(root / "ready.json", {"pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                              "base": str(cfg["model_path"]), "prefix_cache": False, "max_num_seqs": args.max_num_seqs})
    active_lora = None
    next_lora = 1000
    while True:
        pending = [p for p in sorted(root.glob("*.request.json"),
                                   key=lambda x: (x.name.startswith("eval-"), x.stat().st_mtime))
                   if not p.with_name(p.name.replace(".request.", ".response.")).exists()]
        if not pending:
            time.sleep(.5)
            continue
        for file in pending:
            request = json.loads(file.read_text())
            dest = file.with_name(file.name.replace(".request.", ".response."))
            start = time.monotonic()
            try:
                if request.get("mode") == "shutdown":
                    save(dest, {"request_id": request["request_id"], "stopped": True})
                    return
                adapter = request.get("adapter")
                if active_lora is not None:
                    assert llm.llm_engine.remove_lora(active_lora)
                    active_lora = None
                lora = None
                digest = None
                if adapter:
                    digest = sha(Path(adapter) / "adapter_model.safetensors")
                    assert digest == request["adapter_sha256"], "Adapter changed before generation"
                    next_lora += 1
                    lora = LoRARequest(f"policy_{next_lora}", next_lora, adapter)
                    active_lora = next_lora
                rows = request["prompts"]
                mode = request.get("mode", "generate")
                params = []
                for i, row in enumerate(rows):
                    if mode == "score":
                        params.append(SamplingParams(
                            max_tokens=1, temperature=1., top_p=1., top_k=-1,
                            seed=request["seed"] + i, prompt_logprobs=0, detokenize=False))
                    else:
                        assert len(row["prompt_token_ids"]) + request["max_new_tokens"] <= 32768
                        params.append(SamplingParams(
                            max_tokens=request["max_new_tokens"],
                            temperature=request["temperature"], top_p=request["top_p"],
                            top_k=request["top_k"], seed=request["seed"] + i,
                            skip_special_tokens=False))
                outputs = llm.generate(
                    [{"prompt_token_ids": r["prompt_token_ids"]} for r in rows],
                    params, lora_request=lora, use_tqdm=True)
                records = []
                for row, output in zip(rows, outputs, strict=True):
                    c = output.outputs[0]
                    value = {"id": row["id"], "prompt_token_ids": output.prompt_token_ids,
                             "response_token_ids": list(c.token_ids), "completion": c.text,
                             "finish_reason": c.finish_reason, "stop_reason": c.stop_reason}
                    assert output.prompt_token_ids == row["prompt_token_ids"]
                    if mode == "score":
                        begin = row["score_start"]
                        value["logprobs"] = [
                            output.prompt_logprobs[j][output.prompt_token_ids[j]].logprob
                            for j in range(begin, len(output.prompt_token_ids))]
                    records.append(value)
                save(dest, {"request_id": request["request_id"], "records": records,
                            "elapsed_seconds": time.monotonic() - start,
                            "adapter_sha256": digest, "lora_id": active_lora,
                            "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
                print(json.dumps({"event": "completed", "request_id": request["request_id"],
                                  "items": len(records), "seconds": time.monotonic()-start}), flush=True)
            except Exception:
                error = traceback.format_exc()
                save(dest, {"request_id": request["request_id"], "error": error})
                raise

if __name__ == "__main__":
    main()
