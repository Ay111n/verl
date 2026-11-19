"""Outcome-only training from S0 on frozen, subject-stratified FormatEval TRAIN."""
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
PRIMITIVES = PROJECT / "scripts/knowledge_grpo_intermittent_nll_v3"
sys.path.insert(0, str(PRIMITIVES))
from contract import read, text_sha
from sampling import generate, sync
from train import online
from checkpointing import commit, restore, snapshots
from run import supervise
from scripts.luffy_math_pilot.common import adapter_hash, load_model, read_rows, save, sha
from rlvr.answer_verifier import ANSWER_EXTRACTION_VERSION

NAME = "formateval_grpo_v3_928_20261120"
CONFIG = PROJECT / "configs" / (NAME + ".json")
RUN = PROJECT / "runs" / NAME
OUT = ROOT / "finetuning/model/rl" / NAME


def preflight():
    from transformers import AutoTokenizer
    from difficulty8_generation import PROMPT_PROTOCOL, prepare_prompts

    cfg = read(CONFIG)
    ds = ROOT / cfg["training_data"]
    manifest = read(ds.parent / "manifest.json")
    assert sha(ds.parent / "manifest.json") == cfg["dataset_manifest_sha256"]
    assert sha(ds) == manifest["train_sha256"] == cfg["train_sha256"]
    assert ANSWER_EXTRACTION_VERSION == cfg["verifier_version"]
    assert sha(ROOT / "benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py") == cfg["choice_verifier_sha256"]
    assert PROMPT_PROTOCOL == cfg["prompt_protocol"]
    assert (cfg["training_questions"], cfg["rollouts_per_question"], cfg["batch_size"], cfg["epochs"]) == (928, 4, 16, 1)
    assert cfg["kl_coef"] == cfg["entropy_coef"] == 0
    eligible_inputs = manifest["sources"]
    for item in [*eligible_inputs.values(), *manifest["protected_inputs"].values()]:
        assert sha(ROOT / (item.get("local_path") or item["file"])) == item["sha256"]
    native = read(ROOT / "finetuning/rl/processed_data/difficulty8_paired_full_20261118/protocol.json")
    base = ROOT / cfg["model_path"]
    assert Path(native["model"]) == base
    for filename, digest in native["model_hashes"].items():
        assert sha(base / filename) == digest, filename
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    bo_full, en_full = [{r["id"]: r for r in read_rows(ROOT / eligible_inputs[key]["local_path"])}
                        for key in ("data_010", "data_025")]
    rows = read_rows(ds)
    prepared = {}
    for row in rows:
        uid = row["id"]
        assert row == bo_full[uid] and uid not in prepared
        en = en_full[uid]
        assert row["answer"] == en["answer"] and len(row["choices"]) == len(en["choices"]) == 4
        example = dict(id=uid, domain="knowledge", question_bo=row["question"], question_en=en["question"],
                       options_bo=dict(zip("ABCD", row["choices"], strict=True)),
                       options_en=dict(zip("ABCD", en["choices"], strict=True)),
                       canonical_answer="ABCD"[row["answer"]])
        prompt = prepare_prompts(example, tok, PROMPT_PROTOCOL)[uid, "bo"]
        # Native DEV uses the four displayed Tibetan options when resolving text answers.
        prepared[uid] = dict(id=uid, domain="knowledge", options_bo=example["options_bo"],
                             canonical_answer=example["canonical_answer"], prompt=prompt["prompt"],
                             prompt_token_ids=prompt["token_ids"], source=row["source_dataset"],
                             subject=row["source_subject"])
        assert prompt["prompt"].endswith("<|im_start|>assistant\n"), "S0 native Thinking ON prefix differs"
        assert len(prompt["token_ids"]) + cfg["max_new_tokens"] <= cfg["max_model_len"]
    assert text_sha("\n".join(prepared) + "\n") == cfg["id_order_sha256"] == manifest["id_order_sha256"]
    assert len(prepared) == cfg["training_questions"] and len(manifest["subject_strata"]) == 111
    assert Counter(q["source"] for q in prepared.values()) == manifest["source_counts"]
    RUN.mkdir(parents=True, exist_ok=True)
    save(RUN / "training_prompts.json", prepared)
    save(RUN / "preflight.json", dict(questions=len(prepared), id_order_sha256=cfg["id_order_sha256"],
         config_sha256=sha(CONFIG), train_sha256=sha(ds), prompt_file_sha256=sha(RUN / "training_prompts.json"),
         prompt_protocol=PROMPT_PROTOCOL, source_counts=manifest["source_counts"], model_path=str(base),
         model_hashes=native["model_hashes"], verifier=ANSWER_EXTRACTION_VERSION, python=sys.executable,
         packages={k:importlib.metadata.version(k) for k in ("torch","vllm","transformers","peft","safetensors")}))
    print(f"Verified {len(prepared)} Tibetan prompts, frozen data, S0 shards, and v3.", flush=True)


def write_responses(tx, policy, digest, predictions, groups):
    directory = RUN / "responses"
    directory.mkdir(exist_ok=True)
    dest = directory / f"batch-{tx:04d}.jsonl.gz"
    tmp = dest.with_suffix(".tmp")
    judgments = {(g["id"], j["rep"]): (j, r) for g in groups for j, r in zip(g["judgments"], g["rewards"], strict=True)}
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=2) as f:
        for p in predictions:
            j, reward = judgments[p["id"], p["rollout_index"]]
            r = dict(id=p["id"], rep=p["rollout_index"], seed=p["seed"], policy_version=policy,
                     adapter_sha256=digest, completion=p["completion"], finish_reason=p["finish_reason"],
                     stop_reason=p["stop_reason"], output_tokens=len(p["response_token_ids"]),
                     response_token_ids=p["response_token_ids"], judgment=j, reward=reward)
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, dest)
    return sha(dest)


def train():
    import torch
    cfg = read(CONFIG)
    devices = read(RUN / "effective_training_config.json")
    assert len(devices["rollout_gpus"]) == 3 and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    pre = read(RUN / "preflight.json")
    assert sha(CONFIG) == pre["config_sha256"] and sha(RUN / "training_prompts.json") == pre["prompt_file_sha256"]
    data = read(RUN / "training_prompts.json")
    ids = list(data)
    assert text_sha("\n".join(ids) + "\n") == cfg["id_order_sha256"]
    size = cfg["batch_size"]
    batches = [ids[i:i + size] for i in range(0, len(ids), size)]
    assert list(map(len, batches)) == [16]*58
    save(RUN / "training_plan.json", dict(ordered_batches=batches, train_sha256=cfg["train_sha256"]))
    OUT.mkdir(parents=True, exist_ok=True)
    before = time.monotonic()
    existing = snapshots(OUT)
    last = existing[-1] if existing else None
    base = ROOT / cfg["model_path"]
    model = load_model(str(base), cfg["initial_seed"], adapter=str(last) if last else None, trainable_adapter=bool(last))
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    if last:
        state, digest = restore(last, model, opt, cfg)
        for cp in existing:
            path = RUN / "steps" / (cp.name + ".json")
            if not path.exists():
                save(path, read(cp / "event.json"))
    else:
        state = dict(tx=0, cursor=0, policy=0, online=0, nll=0, pending=[], due=[])
        last, digest, _ = commit(OUT, RUN, model, opt, state,
            dict(kind="initial", s0=str(base), initial_lora_sha256=adapter_hash(model)), cfg, base)
    load_seconds = time.monotonic()-before
    workers = [RUN / "rpc_workers" / f"worker{i}" for i in range(len(devices["rollout_gpus"]))]
    while state["cursor"] < len(batches):
        started = time.monotonic()
        cursor = state["cursor"]
        batch = batches[cursor]
        tag = f'formal-b{cursor+1:03d}-v{state["policy"]:03d}-{digest[:16]}'
        pred, times = generate(batch, data, cfg, workers, tag, last, digest, cursor+1, "formal")
        event = online(model,opt,params,cfg,state,batch,pred,data,tag,cursor==0,times)
        event["responses_sha256"] = write_responses(cursor+1, state["policy"], digest, pred, event["groups"])
        del pred
        if event["optimizer_updated"]:
            state["online"] += 1
            state["policy"] += 1
        state["cursor"] += 1
        state["tx"] += 1
        event.update(input_adapter_sha256=digest, cursor=state["cursor"], online_step=state["online"], epoch=1)
        last,digest,written = commit(OUT,RUN,model,opt,state,event,cfg,base)
        save(RUN/"timings"/f'{state["tx"]:04d}.json', dict(checkpoint_seconds=written,
             total_seconds=time.monotonic()-started, end_unix=time.time()))
        print(f'Batch {state["cursor"]}/{len(batches)}: {event["group_histogram"]}, correct '
              f'{event["correct_responses"]}/{event["online_responses"]}, grad {event["grad_norm"]:.4f}', flush=True)
    sync_time = sync(workers, f'formal-final-v{state["policy"]}-{digest[:12]}', last, digest)
    events = [read(RUN/"steps"/f"checkpoint-{i:04d}.json") for i in range(1, len(batches)+1)]
    assert [g["id"] for e in events for g in e["groups"]] == ids
    assert sum(e["online_responses"] for e in events) == 3712
    assert all(conf["rewards"] == [int(y["selected"]==y["gold"] and y["finish_reason"]=="stop")
                                    for y in conf["judgments"]] for e in events for conf in e["groups"])
    final = dict(state, epochs=1, adapter=str(last), adapter_sha256=digest, model_path=str(base),
                 train_sha256=cfg["train_sha256"], question_visits=len(ids), online_responses=3712,
                 correct_responses=sum(e["correct_responses"] for e in events), truncated=sum(e["truncated"] for e in events),
                 group_histogram=[sum(e["group_histogram"][i] for e in events) for i in range(5)],
                 output_tokens=sum(e["output_tokens"] for e in events), prompt_tokens=sum(e["prompt_tokens"] for e in events),
                 load_seconds=load_seconds, final_sync_seconds=sync_time)
    save(OUT/"final.json",final)
    save(RUN/"complete.json",final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "train"))
    stage = parser.parse_args().stage
    if stage == "preflight":
        preflight()
    elif stage == "run":
        if not (RUN / "preflight.json").is_file():
            preflight()
        supervise(CONFIG, RUN, online_actor=Path(__file__).resolve())
    else:
        train()
