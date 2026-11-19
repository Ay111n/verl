"""Distribute whole native rollouts on identical, explicitly loaded policies."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import time

from contract import PROJECT
from scripts.luffy_math_pilot.common import rpc


def request_lane(directory, request, mode):
    """Retry only the failed replica. Never score incomplete output as an error reward."""
    last = None
    for attempt in range(4):
        part = dict(request, request_id=f'{request["request_id"]}-r{attempt}')
        try:
            result = rpc(directory, part, timeout=1500)
            if result["adapter_sha256"] != request["adapter_sha256"]:
                raise RuntimeError("Worker returned another policy")
            if mode == "generate":
                for pred, item in zip(result["records"], part["prompts"], strict=True):
                    for key in ("id", "rollout_index", "prompt_token_ids", "seed"):
                        if pred[key] != item[key]:
                            raise RuntimeError("Rollout key or prompt differs")
                    ids = pred["response_token_ids"]
                    if (pred["finish_reason"] not in ("stop", "length") or not ids or
                            len(pred["raw_logprobs"]) != len(ids)):
                        raise RuntimeError("Incomplete worker response")
            return result
        except (RuntimeError, TimeoutError, ValueError) as exc:
            last = exc
            print(f'{part["request_id"]} failed: {exc}', flush=True)
    raise RuntimeError(f"Replica failed for {request['request_id']}") from last


def all_lanes(workers, requests, mode):
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        jobs = [pool.submit(request_lane, worker, req, mode) for worker, req in zip(workers, requests, strict=True)]
        return [job.result() for job in jobs]


def sync(workers, tag, adapter, digest):
    before = time.monotonic()
    requests = [dict(request_id=f"{tag}-sync-w{i}",mode="sync",preload_adapter=True,
                     adapter=str(adapter),adapter_sha256=digest) for i in range(len(workers))]
    all_lanes(workers, requests, "sync")
    return time.monotonic() - before


def generate(batch, data, cfg, workers, tag, adapter, digest, cursor, stage):
    synced = sync(workers, tag, adapter, digest)
    prompts = []
    for qid in batch:
        for rep in range(cfg.get("rollouts_per_question", 3)):
            seed_text = f'{cfg["sampling_seed"]}|{stage}|{cursor}|{qid}|{rep}'
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "big") % (2**31-1)
            prompts.append(dict(id=qid, rollout_index=rep, seed=seed, prompt_token_ids=data[qid]["prompt_token_ids"]))
    before = time.monotonic()
    params = {k:cfg[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")}
    requests = [dict(request_id=f"{tag}-gen-w{i}", mode="generate", preload_adapter=True,
                     adapter=str(adapter), adapter_sha256=digest, prompts=prompts[i::len(workers)],
                     return_logprobs=True, **params) for i in range(len(workers))]
    results = all_lanes(workers, requests, "generate")
    by_key = {(r["id"],r["rollout_index"]):r for result in results for r in result["records"]}
    assert len(by_key) == len(prompts)
    ordered = [by_key[r["id"], r["rollout_index"]] for r in prompts]
    return ordered, dict(sync_seconds=synced, generation_seconds=time.monotonic()-before,
                         worker_seconds=[v["elapsed_seconds"] for v in results])
