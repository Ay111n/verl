"""Generate and persist each independent English S0 sample, with replayable seeds."""
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from paired import ROOT, RUN, judge, load_inputs, read, save
from run import available_eval_gpus


def seed_for(cfg, qid, rep):
    key = f'{cfg["sampling_seed"]}|{cfg["teacher_seed_namespace"]}|{qid}|{rep}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (2**31-1)


def slot(qid, rep):
    assert Path(qid).name == qid
    return RUN / "teacher_responses" / qid / f"{rep}.json.gz"


def validate(row, view, rep, cfg, audit):
    assert row["id"] == view["id"] and row["rep"] == rep
    assert row["seed"] == seed_for(cfg, view["id"], rep)
    assert row["config_sha256"] == audit["config_sha256"] and row["prompt_sha256"] == view["prompt_sha256"]
    assert row["output_tokens"] == len(row["response_token_ids"])
    assert row["judgment"] == judge(view, row["completion"], row["finish_reason"], row["output_tokens"], cfg)


def load_slot(view, rep, cfg, audit):
    with gzip.open(slot(view["id"], rep), "rt", encoding="utf-8") as f:
        row = json.load(f)
    validate(row, view, rep, cfg, audit)
    return row


async def worker(rank, world):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    cfg, audit, data = load_inputs()
    jobs = []
    for uid in list(data)[rank::world]:
        for rep in range(cfg["teacher_rollouts"]):
            view = data[uid]["en"]
            if slot(uid, rep).exists():
                load_slot(view, rep, cfg, audit)
            else:
                jobs.append((view, rep))
    if not jobs:
        return
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(ROOT / cfg["teacher_model"]), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=cfg["max_model_len"], gpu_memory_utilization=.90, seed=cfg["sampling_seed"],
        max_num_seqs=cfg["teacher_max_num_seqs"], max_num_batched_tokens=2048, enable_prefix_caching=True,
        enable_chunked_prefill=True, safetensors_load_strategy="eager", generation_config="vllm"))
    limit = asyncio.Semaphore(cfg["teacher_max_num_seqs"]*2)
    count = 0
    async def sample(view, rep):
        nonlocal count
        async with limit:
            uid = view["id"]
            seed = seed_for(cfg, uid, rep)
            params = SamplingParams(n=1, temperature=cfg["temperature"], top_p=cfg["top_p"],
                top_k=cfg["top_k"], max_tokens=cfg["max_new_tokens"], seed=seed, skip_special_tokens=False,
                output_kind=RequestOutputKind.FINAL_ONLY)
            final = None
            async for output in engine.generate({"prompt_token_ids":view["prompt_token_ids"]}, params, f"{uid}-{rep}"):
                final = output
            if final is None or not final.finished or len(final.outputs) != 1:
                raise RuntimeError(f"Incomplete request {uid}-{rep}")
            assert list(final.prompt_token_ids) == view["prompt_token_ids"]
            answer = final.outputs[0]
            tokens = list(answer.token_ids)
            row = dict(id=uid, rep=rep, seed=seed, prompt_sha256=view["prompt_sha256"], config_sha256=audit["config_sha256"],
                completion=answer.text, response_token_ids=tokens, output_tokens=len(tokens), finish_reason=answer.finish_reason,
                stop_reason=answer.stop_reason, judgment=judge(view, answer.text, answer.finish_reason, len(tokens), cfg))
            dest = slot(uid, rep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=2) as f:
                json.dump(row, f, ensure_ascii=False, separators=(",", ":"))
            tmp.replace(dest)
            count += 1
            if count % 32 == 0:
                print(f"English replica {rank}: {count}/{len(jobs)} new responses", flush=True)
    try:
        await asyncio.gather(*(sample(view, rep) for view,rep in jobs))
    finally:
        engine.shutdown()


def aggregate(cfg, audit, data):
    groups, total_tokens, truncated = {}, 0, 0
    for uid, pair in data.items():
        rewards = []
        for rep in range(cfg["teacher_rollouts"]):
            row = load_slot(pair["en"], rep, cfg, audit)
            rewards.append(row["judgment"]["reward"])
            total_tokens += row["output_tokens"]
            truncated += row["finish_reason"] == "length"
        groups[uid] = rewards
    save(RUN/"teacher_complete.json", dict(config_sha256=audit["config_sha256"],
        prompt_sha256=audit["prompt_sha256"], model=cfg["teacher_model"], rewards=groups,
        responses=sum(map(len, groups.values())), correct=sum(map(sum, groups.values())), truncated=truncated,
        output_tokens=total_tokens, completed_unix=time.time()))


def supervise():
    cfg, audit, data = load_inputs()
    with (RUN/"teacher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (RUN/"teacher_complete.json").exists():
            saved = read(RUN/"teacher_complete.json")
            assert saved["config_sha256"] == audit["config_sha256"] and saved["prompt_sha256"] == audit["prompt_sha256"]
            assert list(saved["rewards"]) == list(data) and saved["responses"] == len(data)*cfg["teacher_rollouts"]
            return
        if not (RUN/"launch_request.json").exists():
            save(RUN/"launch_request.json", dict(unix=time.time(), config_sha256=audit["config_sha256"]))
        for attempt in range(4):
            gpus = available_eval_gpus(range(cfg["maximum_concurrent_gpus"]))
            save(RUN/f"teacher_assignment_{time.time_ns()}.json", dict(gpus=gpus, attempt=attempt, unix=time.time()))
            children = []
            try:
                for rank,gpu in enumerate(gpus):
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD="spawn",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    with (RUN/f"teacher{rank}.log").open("a") as log:
                        children.append(subprocess.Popen([sys.executable,"-u",__file__,str(rank),str(len(gpus))],
                            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                codes = [p.wait() for p in children]
            finally:
                for proc in children:
                    if proc.poll() is None:
                        proc.terminate()
                for proc in children:
                    proc.wait()
            if not any(codes):
                aggregate(cfg, audit, data)
                print(f"Verified {len(data)*cfg['teacher_rollouts']} frozen English responses.", flush=True)
                return
            print(f"English worker return codes {codes}; retrying missing slots.", flush=True)
        raise RuntimeError("English workers exhausted retries; all completed slots were retained")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        asyncio.run(worker(int(sys.argv[1]), int(sys.argv[2])))
    else:
        supervise()
