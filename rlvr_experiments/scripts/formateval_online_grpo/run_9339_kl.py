"""Two-pass Tibetan outcome GRPO with a frozen merged SFT KL reference."""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import tempfile
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "scripts/knowledge_grpo_intermittent_nll_v3"))
from checkpointing import commit, restore
from contract import read, text_sha, advantages, student_reward
from sampling import generate, sync
from train import finish_update
from scripts.luffy_math_pilot.common import adapter_hash, load_model, online_loss, response_logps_batch, save, sha
from scripts.crosslingual_grpo_512.weighted_hardware import inventory
from rlvr.answer_verifier import ANSWER_EXTRACTION_VERSION, verify_answer

NAME = "formateval_9339_priority512_grpo_kl_20261126"
CONFIG = PROJECT / "configs" / f"{NAME}.json"
RUN = PROJECT / "runs" / NAME
OUT = ROOT / "finetuning/model/rl" / NAME


def preflight(cfg):
    from difficulty8_generation import ANSWER_REQUEST, PROMPT_PROTOCOL
    from transformers import AutoTokenizer
    assert cfg["prompt_protocol"] == PROMPT_PROTOCOL
    ds = ROOT / cfg["training_data"]
    mf = read(ds.parent / "manifest.json")
    assert sha(ds.parent / "manifest.json") == cfg["dataset_manifest_sha256"]
    assert sha(ds) == mf["train_sha256"] == cfg["train_sha256"]
    assert mf["verification"]["protected_matches"] == 0
    base = ROOT / cfg["model_path"]
    published = read(base / "publication.json")["source_identity"]
    weights = read(base / "merge_manifest.json")
    assert published["step"] == 9339 and published["epoch"] == 3.0 and published["dataset_rows"] == 49800
    assert weights["adapter_files_sha256"]["adapter_model.safetensors"] == published["sha256"]["adapter_model.safetensors"]
    for key, digest in weights["merged_output_files_sha256"].items():
        assert sha(base / key) == digest, key
    assert ANSWER_EXTRACTION_VERSION == cfg["verifier_version"]
    assert sha(ROOT / "benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py") == cfg["choice_verifier_sha256"]
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    prompts = {}
    for line in ds.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        uid = row["id"]
        assert uid not in prompts and row["answer"] in range(4)
        options = dict(zip("ABCD", row["choices"], strict=True))
        content = "\n\n".join((row["question"].strip(), "\n".join(
            f"{label}. {text.strip()}" for label, text in options.items()), ANSWER_REQUEST["bo"]))
        prompt = tok.apply_chat_template([dict(role="user", content=content)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        tokens = tok(prompt, add_special_tokens=False)["input_ids"]
        assert len(tokens) + cfg["max_new_tokens"] <= cfg["max_model_len"]
        prompts[uid] = dict(id=uid, domain="knowledge", options_bo=options, prompt=prompt,
                            prompt_token_ids=tokens, canonical_answer="ABCD"[row["answer"]],
                            source=row["source_dataset"], subject=row["source_subject"])
    assert len(prompts) == cfg["training_questions"] == 512
    assert text_sha("\n".join(prompts) + "\n") == cfg["id_order_sha256"]
    assert Counter(p["source"] for p in prompts.values()) == mf["source_counts"]
    registry = read(ROOT / "benchmark/eval/scripts/multilingual_mcqa_data_registry.json")
    for split, expected in [("dev", 666), ("test", 1418)]:
        protected = registry["files"][split]["bo"]["4"]
        path = ROOT / "benchmark/eval" / protected["path"]
        assert sha(path) == protected["sha256"]
        ids = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}
        assert len(ids) == expected and not ids.intersection(prompts)
    save(RUN / "training_prompts.json", prompts)
    save(RUN / "preflight.json", dict(config_sha256=sha(CONFIG), prompts_sha256=sha(RUN/"training_prompts.json"),
         model=cfg["model_path"], model_step=9339, model_hashes=weights["merged_output_files_sha256"],
         data_sha256=sha(ds), python=sys.executable,
         packages={k:importlib.metadata.version(k) for k in ("torch", "vllm", "transformers", "peft")}))


def update(model, optimizer, params, cfg, state, ids, pred, data, times):
    import torch
    started = time.monotonic()
    size = cfg["rollouts_per_question"]
    groups, items = [], []
    hist = Counter()
    for i, uid in enumerate(ids):
        row = data[uid]
        responses = pred[i*size:(i+1)*size]
        assert len(responses) == size
        rewards, judgments = [], []
        for j, p in enumerate(responses):
            assert p["id"] == uid and p["rollout_index"] == j
            v = verify_answer(row, prompt=row["prompt"], completion=p["completion"], finish_reason=p["finish_reason"],
                              output_tokens=len(p["response_token_ids"]), max_new_tokens=cfg["max_new_tokens"])
            assert v["verifier_version"] == cfg["verifier_version"]
            rewards.append(student_reward(v))
            judgments.append(dict(rep=j, seed=p["seed"], selected=v["extracted_answer"], gold=row["canonical_answer"],
                                  status=v["parse_status"], method=v["choice_method"], finish_reason=p["finish_reason"]))
        adv = advantages(rewards, size)
        hist[sum(rewards)] += 1
        groups.append(dict(id=uid, rewards=rewards, advantages=adv, judgments=judgments))
        items.extend(zip(responses, adv, strict=True))
    optimizer.zero_grad(set_to_none=True)
    mae, pg_mean, kl_mean = 0., 0., 0.
    denom = len(items)
    assert denom == len(ids)*size
    for p, adv in sorted(items, key=lambda item:len(item[0]["response_token_ids"])):
        args = ([p["prompt_token_ids"]], [p["response_token_ids"]], cfg["head_chunk"], cfg["likelihood_temperature"])
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            reference = response_logps_batch(model, *args)[0]
        model.train()
        values, raw = response_logps_batch(model, *args, return_raw=True)
        lp = values[0]
        err = float((raw[0].cpu() - torch.as_tensor(p["raw_logprobs"])).abs().mean())
        if err >= .05:
            raise RuntimeError(f"Sampler/actor logits mismatch: {err:.6f}, {p['id']}/{p['rollout_index']}")
        mae = max(mae, err)
        delta = reference.float() - lp.float()
        kl = (torch.expm1(delta) - delta).mean()
        pg = online_loss(lp, lp.detach(), adv, cfg["clip"])
        loss = (pg + cfg["kl_coef"] * kl) / denom
        assert torch.isfinite(loss)
        loss.backward()
        pg_mean += float(pg.detach()) / denom
        kl_mean += float(kl.detach()) / denom
        del values, raw, lp, reference, delta, kl, pg, loss
    stats = finish_update(optimizer, params, cfg, state["policy"])
    return dict(kind="online", groups=groups, question_count=len(ids), online_responses=denom,
                group_histogram=[hist[k] for k in range(size+1)], correct_responses=sum(k*v for k,v in hist.items()),
                truncated=sum(p["finish_reason"] == "length" for p in pred),
                output_tokens=sum(len(p["response_token_ids"]) for p in pred),
                grpo_loss=pg_mean, sampled_kl=kl_mean, kl_coef=cfg["kl_coef"], raw_logprob_mae_max=mae,
                update_seconds=time.monotonic()-started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                **stats, **times)


def recovery_points(out):
    return sorted(p for p in out.glob("checkpoint-[0-9][0-9][0-9][0-9]") if p.is_dir())


def publish_rollout_adapter(model, out):
    target = out / "rollout_adapter"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".rollout-") as staging:
        staged = Path(staging)
        model.save_pretrained(staged, safe_serialization=True)
        digest = sha(staged / "adapter_model.safetensors")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (staged / name).replace(target / name)
    return target, digest


def sampling_lane(session, index):
    return RUN / "rpc_workers" / session / f"worker{index}"


def train(cfg, actor_gpu, session):
    import torch
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(actor_gpu)
    free, total = torch.cuda.mem_get_info()
    cap = free - 3072 * 1024**2
    assert cap >= 24000 * 1024**2, f"GPU {actor_gpu} free memory changed: {free/1024**2:.0f} MiB"
    torch.cuda.set_per_process_memory_fraction(cap / total)
    pre = read(RUN / "preflight.json")
    assert sha(CONFIG) == pre["config_sha256"] and sha(RUN/"training_prompts.json") == pre["prompts_sha256"]
    data = read(RUN / "training_prompts.json")
    first = list(data)
    second = first.copy()
    random.Random(cfg["order_seed"] + 1).shuffle(second)
    assert set(first) == set(second) and first != second
    orders = (first, second)
    size = cfg["batch_size"]
    batches = [order[i:i+size] for order in orders for i in range(0, len(order), size)]
    assert len(batches) == 64 and all(len(batch) == 16 for batch in batches)
    cadence, keep = cfg["checkpoint_every_updates"], cfg["checkpoints_to_keep"]
    assert cadence > 0 and keep > 0 and len(batches) % cadence == 0
    save(RUN/"training_plan.json", dict(ordered_batches=batches, train_sha256=cfg["train_sha256"]))
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT/cfg["model_path"]
    existing = recovery_points(OUT)
    recovery = existing[-1] if existing else None
    model = load_model(str(base), cfg["initial_seed"], adapter=str(recovery) if recovery else None, trainable_adapter=bool(recovery),
                       lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"])
    lora = model.peft_config["default"]
    assert (lora.r, lora.lora_alpha) == (32,64)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    if recovery:
        state, digest = restore(recovery, model, opt, cfg)
        assert state["tx"] == state["cursor"] and state["tx"] % cadence == 0
        if not (RUN/"steps"/f"{recovery.name}.json").exists():
            save(RUN/"steps"/f"{recovery.name}.json", read(recovery/"event.json"))
        last = recovery
    else:
        state = dict(tx=0, cursor=0, policy=0, online=0, nll=0, pending=[], due=[])
        last, digest = publish_rollout_adapter(model, OUT)
        save(RUN/"initialization.json", dict(base_step=9339, training_updates=0, seed=cfg["initial_seed"],
            initial_lora_sha256=adapter_hash(model), sampler_adapter_sha256=digest,
            parameters=sum(p.numel() for p in params), actor_gpu=actor_gpu, actor_mem_cap=cap))
    for event_file in (RUN/"steps").glob("checkpoint-[0-9][0-9][0-9][0-9].json"):
        if int(event_file.stem.removeprefix("checkpoint-")) > state["cursor"]:
            event_file.unlink()
    save(RUN/"progress.json", dict(updates=state["cursor"], recovery=str(recovery) if recovery else None,
         adapter_sha256=digest, actor_gpu=actor_gpu, sampling_session=session))
    lanes = [sampling_lane(session, i) for i in range(2)]
    while state["cursor"] < len(batches):
        cursor = state["cursor"]
        tag = f"b{cursor+1:03d}-v{state['policy']:03d}-{digest[:16]}"
        p, timing = generate(batches[cursor],data,cfg,lanes,tag,last,digest,cursor+1,"formal")
        event = update(model,opt,params,cfg,state,batches[cursor],p,data,timing)
        assert event["grad_norm"] > 0
        del p
        for key in ("cursor","tx","policy","online"):
            state[key] += 1
        event.update(epoch=cursor//32+1, batch=cursor+1, input_adapter_sha256=digest, sampling_session=session)
        if state["tx"] % cadence == 0:
            last, digest, _ = commit(OUT, RUN, model, opt, state, event, cfg, base)
            recovery = last
            for outdated in recovery_points(OUT)[:-keep]:
                shutil.rmtree(outdated)
        else:
            last, digest = publish_rollout_adapter(model, OUT)
            save(RUN/"steps"/f'checkpoint-{state["tx"]:04d}.json',
                 dict(event, tx=state["tx"], policy_version=state["policy"], output_adapter_sha256=digest))
        assert digest != event["input_adapter_sha256"]
        save(RUN/"progress.json", dict(updates=state["cursor"], recovery=str(recovery) if recovery else None,
             adapter_sha256=digest, actor_gpu=actor_gpu, sampling_session=session))
        print(f"Batch {cursor+1}/64: {event['group_histogram']}, KL={event['sampled_kl']:.7g}, "
              f"peak_reserved={event['peak_reserved_bytes']/1024**3:.2f} GiB", flush=True)
    assert recovery == last and last.name == "checkpoint-0064"
    sync(lanes, f"final-{digest[:16]}", last, digest)
    working_adapter = OUT / "rollout_adapter"
    if working_adapter.exists():
        shutil.rmtree(working_adapter)
    events = [read(RUN/"steps"/f"checkpoint-{i:04d}.json") for i in range(1,65)]
    assert [g["id"] for e in events for g in e["groups"]] == first + second
    assert sum(e["online_responses"] for e in events) == 8192
    assert len({(g["id"], j["seed"]) for e in events for g in e["groups"] for j in g["judgments"]}) == 8192
    save(RUN/"complete.json",dict(state,epochs=2,batches=64,question_visits=1024,online_responses=8192,
         train_sha256=cfg["train_sha256"],adapter=str(last),adapter_sha256=digest,
         checkpoint_every_updates=cadence, retained_checkpoints=[p.name for p in recovery_points(OUT)],
         output_tokens=sum(e["output_tokens"] for e in events),truncated=sum(e["truncated"] for e in events)))


def devices(cfg, actor_gpu, sampling_gpus, training):
    while True:
        first = inventory()
        time.sleep(3)
        second = inventory()
        idle = [i for i in sorted(first) if all(m[i][1]<128 and m[i][3]<5 for m in (first,second))]
        if not training:
            eval_pool = sampling_gpus if sampling_gpus else idle
            if len(eval_pool) >= cfg["eval_replicas"] and all(i in idle for i in eval_pool[:cfg["eval_replicas"]]):
                return eval_pool[:cfg["eval_replicas"]]
        else:
            lanes = sampling_gpus if sampling_gpus else [i for i in idle if i != actor_gpu]
            if len(lanes) >= 2 and all(i in idle for i in lanes[:2]) and min(first[actor_gpu][2],second[actor_gpu][2]) >= 27072:
                return [actor_gpu, *lanes[:2]]
        time.sleep(20)


def pipeline(cfg, actor_gpu, sampling_gpus):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.luffy_math_pilot.common import rpc
    active = []
    def stage(name, **info):
        save(RUN/"status.json",dict(stage=name,pid=os.getpid(),unix=time.time(),**info))
    def launch(args, gpus, log, background=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str,gpus)), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTHONPATH=os.pathsep.join(map(str,(PROJECT, ROOT/"finetuning/rl/scripts", ROOT/"benchmark/eval/scripts"))),
                   VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
        with (RUN/log).open("a") as f:
            proc = subprocess.Popen([sys.executable,"-u",*map(str,args)],cwd=ROOT,env=env,
                                    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        active.append(proc)
        save(RUN/"owned_processes.json",dict(controller=os.getpid(),pids=[p.pid for p in active if p.poll() is None]))
        if not background and proc.wait():
            raise RuntimeError(f"{log}: exit {proc.returncode}")
        return proc
    try:
        if not (RUN/"preflight.json").exists():
            stage("preflight")
            preflight(cfg)
        if not (RUN/"complete.json").exists():
            stage("waiting_for_training_gpus", actor_gpu=actor_gpu, sampling_gpus=sampling_gpus, min_free_mib=27072)
            gpus = devices(cfg, actor_gpu, sampling_gpus, True)
            stage("loading_training", gpus=gpus)
            save(RUN/"effective_training_config.json", cfg)
            session = f"attempt-{time.time_ns()}"
            assigned = RUN/"training_gpus.json"
            history = RUN/"training_gpu_history.json"
            previous = read(history) if history.exists() else [read(assigned)] if assigned.exists() else []
            latest = RUN/"latest.json"
            checkpoint = read(latest) if latest.exists() else None
            assignment = dict(actor=gpus[0], sampling=gpus[1:], unix=time.time(), sampling_session=session,
                              from_checkpoint=checkpoint["checkpoint"] if checkpoint else None,
                              from_adapter_sha256=checkpoint["adapter_sha256"] if checkpoint else None)
            save(history, [*previous, assignment])
            save(assigned, assignment)
            workers = []
            for i,gpu in enumerate(gpus[1:]):
                lane = sampling_lane(session, i)
                worker = launch([PROJECT/"scripts/luffy_math_pilot/worker.py","--model",ROOT/cfg["model_path"],
                    "--rpc",lane,"--max-lora-rank",32],[gpu],f"worker{i}.log",True)
                workers.append((worker,lane))
            actor = launch([Path(__file__), "train", "--actor-gpu", actor_gpu, "--sampling-session", session],
                           gpus[:1], "train.log", True)
            stage("training",gpus=gpus,batches=64)
            while actor.poll() is None:
                if any(worker.poll() is not None for worker,_ in workers):
                    raise RuntimeError("A sampling process exited")
                time.sleep(4)
            if actor.returncode:
                raise RuntimeError(f"Training actor exited {actor.returncode}")
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = [pool.submit(rpc,lane,dict(request_id=f"shutdown-{os.getpid()}-{i}",mode="shutdown"),90)
                           for i,(_,lane) in enumerate(workers)]
                for response in pending:
                    response.result()
            for worker,_ in workers:
                worker.wait(timeout=90)
        final = read(RUN/"complete.json")
        adapter = Path(final["adapter"])
        assert sha(adapter/"adapter_model.safetensors") == final["adapter_sha256"]
        merged = OUT/"merged"
        if not (merged/"merge_manifest.json").exists():
            stage("merging")
            staging = OUT/f".merge-{os.getpid()}"
            launch([ROOT/"finetuning/scripts/model/merge_same_source_dpo_adapter.py","--base",ROOT/cfg["model_path"],
                    "--adapter",adapter,"--output",staging],[],"merge.log")
            manifest = read(staging/"merge_manifest.json")
            assert manifest["adapter_sha256"] == final["adapter_sha256"]
            for name,item in manifest["files"].items():
                assert sha(staging/name) == item["sha256"]
            manifest["output"] = str(merged)
            save(staging/"merge_manifest.json",manifest)
            staging.rename(merged)
        assert read(merged/"merge_manifest.json")["adapter_sha256"] == final["adapter_sha256"]
        analysis = ROOT/"benchmark/eval/analysis"/NAME
        if not (analysis/"candidate_result.json").exists():
            stage("waiting_for_eval_gpus", sampling_gpus=sampling_gpus)
            eval_cards = devices(cfg, actor_gpu, sampling_gpus, False)
            save(RUN/"eval_gpu_assignment.json",dict(gpus=eval_cards,unix=time.time()))
            stage("evaluating_666DEV",gpus=eval_cards)
            launch([ROOT/cfg["eval_entrypoint"],"--model",merged,"--arm","candidate",
                    "--replicas",len(eval_cards),"--run-name",NAME],eval_cards,"eval.log")
        evidence = read(analysis/"candidate_result.json")
        result = Path(evidence["run_dir"])
        assert sum(1 for _ in (result/"predictions.jsonl").open("rb")) == 666
        save(RUN/"pipeline_complete.json",dict(questions=666,run_dir=str(result),merged=str(merged),unix=time.time()))
        stage("complete",run_dir=str(result))
    except Exception as exc:
        stage("failed",error=str(exc))
        raise
    finally:
        for proc in active:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in active:
            if proc.poll() is None:
                proc.wait(timeout=60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("preflight", "run", "train"))
    ap.add_argument("--actor-gpu", type=int, help="Physical GPU index for updates")
    ap.add_argument("--sampling-gpus", type=int, nargs=2, help="Reuse these GPUs for sampling and final evaluation")
    ap.add_argument("--sampling-session", help=argparse.SUPPRESS)
    a = ap.parse_args()
    cfg = read(CONFIG)
    actor_gpu = cfg["actor_gpu"] if a.actor_gpu is None else a.actor_gpu
    sampling_gpus = a.sampling_gpus or cfg["sampling_gpus"]
    if len(set([actor_gpu, *sampling_gpus])) != 3:
        ap.error("the actor and the two sampling GPUs must be distinct")
    RUN.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        if not a.sampling_session or not a.sampling_session.removeprefix("attempt-").isdigit():
            ap.error("launch training through the run stage")
        train(cfg, actor_gpu, a.sampling_session)
    else:
        with (RUN/"controller.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if a.stage == "preflight":
                preflight(cfg)
            elif not (RUN/"pipeline_complete.json").exists():
                pipeline(cfg, actor_gpu, sampling_gpus)


if __name__ == "__main__":
    main()
