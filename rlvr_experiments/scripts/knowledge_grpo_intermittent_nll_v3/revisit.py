"""Continue a verified NLL checkpoint with one unassisted GRPO revisit per NLL batch."""
import argparse
import os
import time
from pathlib import Path

from contract import PROJECT, ROOT, choose_refs, epoch_batches, preflight, read, refresh_pool
from scripts.same_source_dpo_316.paired_dpo import save, sha

NAME = "knowledge_nll_revisit_v3_200_fork7_20261119"
CONFIG = PROJECT / "configs" / (NAME + ".json")
RUN = PROJECT / "runs" / NAME
MODEL_OUT = ROOT / "finetuning/model/rl" / NAME


def fork_paths(cfg):
    name = cfg["fork"]["parent_run"]
    parent_run = PROJECT / "runs" / name
    checkpoint = ROOT / "finetuning/model/rl" / name / cfg["fork"]["checkpoint"]
    return parent_run, checkpoint


def verify_fork(cfg):
    parent_run, checkpoint = fork_paths(cfg)
    original = read(parent_run / "effective_training_config.json")
    for key, value in original.items():
        if key not in {"name", "actor_gpu", "rollout_gpus", "eval_gpus"}:
            assert cfg[key] == value, key
    hashes = cfg["fork"]["files_sha256"]
    for filename, digest in hashes.items():
        assert sha(checkpoint / filename) == digest, filename
    assert read(checkpoint / "training_config.json") == original
    state = read(checkpoint / "state.json")
    event = read(checkpoint / "event.json")
    assert state["tx"] == event["tx"] == state["policy"] == 7
    assert state["online"] == state["cursor"] == 6 and state["nll"] == 1 and not state["due"]
    assert event == read(parent_run / "steps" / (checkpoint.name + ".json"))
    ref_ids = [r["id"] for r in event["references"]]
    assert event["kind"] == "nll" and len(set(ref_ids)) == len(ref_ids) == cfg["nll_batch"]
    assert state["uses"] == dict.fromkeys(ref_ids, 1)
    assert not set(ref_ids).intersection(state["pending"]) and len(state["pending"]) == 10
    assert read(parent_run / "steps/checkpoint-0006.json")["scheduled_nll_ids"] == ref_ids
    assert sha(parent_run / "training_plan.json") == cfg["fork"]["training_plan_sha256"]
    return original, checkpoint, ref_ids


def check_inputs():
    cfg = read(CONFIG)
    original, parent, ids = verify_fork(cfg)
    preflight(CONFIG, RUN)
    proof = read(RUN / "preflight.json")
    old_proof = read(fork_paths(cfg)[0] / "preflight.json")
    for key in ("model_hashes", "data_sha256", "id_order_sha256", "choice_verifier_sha256", "protected_matches"):
        assert proof[key] == old_proof[key], key
    baseline = cfg["fork"]["baselines"]
    for row in baseline.values():
        assert sha(ROOT / row["predictions"]) == row["sha256"]
    save(RUN / "fork_contract.json", dict(parent_checkpoint=str(parent), original_gpus=[original["actor_gpu"], *original["rollout_gpus"]],
         parent_hashes=cfg["fork"]["files_sha256"], first_revisit_ids=ids, inherited_regular_questions=96,
         inherited_regular_responses=288, inherited_regular_updates=6, inherited_nll_updates=1,
         regular_seeds="formal", revisit_seeds="nll_revisit", baselines=baseline))


def record_event(state, event, data, cfg):
    kind = event["kind"]
    if kind == "revisit":
        assert [g["id"] for g in event["groups"]] == state["revisit_due"] and not state["due"]
        state["revisit_due"] = []
        state["revisit_batches"] += 1
        state["revisit_updates"] += int(event["optimizer_updated"])
    elif kind == "nll":
        ids = [r["id"] for r in event["references"]]
        assert ids == state["due"] and not state["revisit_due"]
        state["pending"] = [qid for qid in state["pending"] if qid not in ids]
        for qid in ids:
            state["uses"][qid] = state["uses"].get(qid, 0) + 1
        state["due"] = []
        state["nll"] += 1
        state["revisit_due"] = list(ids)
    else:
        assert kind == "online" and not state["revisit_due"] and not state["due"]
        state["cursor"] += 1
        state["pending"] = refresh_pool(state["pending"], event["groups"], data)
        if event["optimizer_updated"]:
            state["online"] += 1
            if state["online"] % cfg["check_every_online"] == 0:
                event["queue_at_check"] = len(state["pending"])
                if len(state["pending"]) >= cfg["nll_batch"] and state["nll"] < cfg["nll_limit"]:
                    state["due"] = state["pending"][:cfg["nll_batch"]]
                event["scheduled_nll_ids"] = list(state["due"])
    if kind == "nll" or event["optimizer_updated"]:
        state["policy"] += 1
    state["tx"] += 1
    assert state["policy"] == state["online"] + state["nll"] + state["revisit_updates"]
    assert state["nll"] == state["revisit_batches"] + int(bool(state["revisit_due"]))
    event.update(online_step=state["online"], nll_step=state["nll"], cursor=state["cursor"],
                 revisit_batches=state["revisit_batches"], revisit_updates=state["revisit_updates"],
                 queue_size=len(state["pending"]), epoch=1)


def train():
    import torch
    from transformers import AutoTokenizer

    from checkpointing import commit, restore
    from sampling import generate, sync
    from train import online, supervised
    from scripts.luffy_math_pilot.common import load_model, read_rows

    cfg = read(RUN / "effective_training_config.json")
    original, parent, first_ids = verify_fork(cfg)
    parent_run, _ = fork_paths(cfg)
    start_time = time.monotonic()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1 and len(cfg["rollout_gpus"]) == 3
    data = {r["id"]: r for r in read_rows(ROOT / cfg["training_data"])}
    assert list(data) == read(RUN / "preflight.json")["fixed_ids"]
    model_path = ROOT / cfg["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    for row in data.values():
        row["prompt"] = tok.decode(row["prompt_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    plan = read(parent_run / "training_plan.json")
    batches = plan["ordered_batches"]
    assert batches == epoch_batches(data, cfg, 1) and list(map(len,batches)) == [16]*12+[8]
    save(RUN / "training_plan.json", plan)
    workers = [RUN / "rpc_workers" / f"worker{i}" for i in range(len(cfg["rollout_gpus"]))]
    MODEL_OUT.mkdir(exist_ok=True, parents=True)
    fork_tx = read(parent / "state.json")["tx"]
    existing = sorted(MODEL_OUT.glob("checkpoint-[0-9][0-9][0-9][0-9]"))
    assert [p.name for p in existing] == [f"checkpoint-{i:04d}" for i in range(fork_tx+1,fork_tx+1+len(existing))]
    last = existing[-1] if existing else parent
    model = load_model(str(model_path), cfg["initial_seed"], adapter=str(last), trainable_adapter=True)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], betas=(.9,.999),eps=1e-8)
    restore_cfg = original
    if existing:
        restore_cfg = read(last / "training_config.json")
        runtime_keys = {"actor_gpu","rollout_gpus","eval_gpus"}
        assert {k:v for k,v in restore_cfg.items() if k not in runtime_keys} == {k:v for k,v in cfg.items() if k not in runtime_keys}
    state, digest = restore(last, model, opt, restore_cfg)
    if existing:
        for p in existing:
            target = RUN / "steps" / (p.name + ".json")
            if not target.exists():
                save(target,read(p / "event.json"))
    else:
        state.update(revisit_due=first_ids, revisit_batches=0, revisit_updates=0)
    save(RUN / f"restore-{state['tx']:04d}-{os.getpid()}.json",
         dict(checkpoint=str(last), adapter_sha256=digest, optimizer_rng_restored=True, state=state))
    load_seconds = time.monotonic()-start_time
    while state["revisit_due"] or state["due"] or state["cursor"] < len(batches):
        started = time.monotonic()
        if state["due"]:
            assert not state["revisit_due"]
            selection = choose_refs(state["due"], data, state["uses"])
            event = supervised(model, opt, params, cfg, state, data, tok.eos_token_id, selection)
        else:
            revisit = bool(state["revisit_due"])
            ids = state["revisit_due"] if revisit else batches[state["cursor"]]
            stage = "nll_revisit" if revisit else "formal"
            index = state["nll"] if revisit else state["cursor"] + 1
            tag = f"{stage}-b{index:03d}-v{state['policy']:03d}-{digest[:16]}"
            preds, times = generate(ids, data, cfg, workers, tag, last, digest, index, stage)
            event = online(model, opt, params, cfg, state, ids, preds, data, tag, state["tx"]==fork_tx, times)
            del preds
            event.update(seed_stage=stage, seed_index=index)
            if revisit:
                event.update(kind="revisit", source_nll_tx=state["tx"])
        event["input_adapter_sha256"] = digest
        record_event(state,event,data,cfg)
        last,digest,written = commit(MODEL_OUT, RUN, model,opt,state,event,cfg,model_path)
        save(RUN / "timings" / f"{state['tx']:04d}.json",dict(kind=event["kind"], checkpoint_seconds=written,
             total_seconds=time.monotonic()-started,end_unix=time.time()))
        print(f"{event['kind']} tx={state['tx']} regular={state['cursor']}/13 nll={state['nll']} "
              f"revisit_batches={state['revisit_batches']} revisit_updates={state['revisit_updates']}",flush=True)
    synced = sync(workers, f"fork-final-v{state['policy']}-{digest[:12]}", last,digest)
    visits = 16 * state["revisit_batches"]
    assert state["nll"]==state["revisit_batches"]<=3 and state["cursor"]==13
    final=dict(state,adapter=str(last),adapter_sha256=digest,model_path=str(model_path),train_sha256=cfg["train_sha256"],
               parent_checkpoint=str(parent),epochs=1,question_visits=200,online_responses=600,regular_branch_responses=312,
               revisit_visits=visits,revisit_responses=3*visits,branch_responses=312+3*visits,
               model_load_seconds=load_seconds,final_sync_seconds=synced)
    save(MODEL_OUT/"epoch-1.json",dict(checkpoint=str(last),adapter_sha256=digest))
    save(MODEL_OUT/"final.json",final)
    save(RUN/"complete.json",final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["preflight","run","train"])
    action=parser.parse_args().action
    if action=="preflight":
        check_inputs()
    elif action=="run":
        from run import supervise
        supervise(CONFIG, RUN, fork_actor=Path(__file__).resolve())
    else:
        train()
