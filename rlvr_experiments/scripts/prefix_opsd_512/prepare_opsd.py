"""Prepare teacher-only notes from the exact verified hints of the prefix GRPO control."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
NAME = "formateval_priority512_prefix_opsd_20261121"
CONFIG = PROJECT / "configs" / (NAME + ".json")
RUN = PROJECT / "runs" / NAME
OUT = ROOT / "finetuning/model/rl" / NAME


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(path, content):
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen file changed: {path}")
    else:
        temp = path.with_suffix(".tmp")
        temp.write_bytes(content)
        os.replace(temp, path)


def prepare():
    cfg = read(CONFIG)
    old = PROJECT / "runs" / cfg["previous"]
    previous = read(old / "preflight.json")
    model = ROOT / cfg["model_path"]
    source = ROOT / cfg["training_data"]
    assert sha(source) == cfg["train_sha256"] == previous["train_sha256"]
    assert sha(source.parent / "manifest.json") == cfg["dataset_manifest_sha256"]
    assert read(source.parent / "manifest.json")["verification"]["protected_matches"] == 0
    assert sha(ROOT / "benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py") == cfg["choice_verifier_sha256"]
    for f, digest in previous["model_hashes"].items():
        assert sha(model / f) == digest, f
    RUN.mkdir(parents=True, exist_ok=True)
    for filename, key, label in [("training_prompts.json", "frozen_student_prompts_sha256", "prompt_file_sha256"),
                                 ("prefixes.json", "fixed_prefixes_sha256", "prefixes_sha256")]:
        assert sha(old / filename) == cfg[key] == previous[label]
        freeze(RUN / filename, (old / filename).read_bytes())
    data = read(RUN / "training_prompts.json")
    hints = read(RUN / "prefixes.json")
    order = "\n".join(data) + "\n"
    assert hashlib.sha256(order.encode()).hexdigest() == cfg["id_order_sha256"]
    assert len(data) == cfg["training_questions"] == 512 and len(hints) == previous["eligible"] == 417
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    end_user = "<|im_end|>\n<|im_start|>assistant\n"
    notes = {}
    for uid, record in hints.items():
        question = data[uid]["prompt"]
        forced = record["forced"]
        assert question.endswith(end_user) and question.count(end_user) == 1
        assert forced.startswith("<think>\n") and forced.endswith("\n")
        hint = forced[len("<think>\n"):-1]
        assert hint and all(m not in hint for m in ["<think>","</think>","<|im_end|>","<|im_start|>"])
        assert tok.encode(question + forced, add_special_tokens=False).ids == record["prompt_token_ids"]
        user_with_note = question[:-len(end_user)] + "\n\nReference note:\n" + hint + end_user
        ids = tok.encode(user_with_note, add_special_tokens=False).ids
        assert ids != data[uid]["prompt_token_ids"] and len(ids) + cfg["max_new_tokens"] <= cfg["max_model_len"]
        assert ids.count(tok.token_to_id("<|im_start|>")) == 2 and ids.count(tok.token_to_id("<|im_end|>")) == 1
        notes[uid] = ids
    enc = lambda v: (json.dumps(v, ensure_ascii=False, indent=2) + "\n").encode()
    freeze(RUN / "teacher_prompts.json", enc(notes))
    audit = dict(previous, config_sha256=sha(CONFIG), teacher_prompts_sha256=sha(RUN / "teacher_prompts.json"),
                 think_open=tok.token_to_id("<think>"), think_close=tok.token_to_id("</think>"),
                 terminal=[tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")],
                 python=sys.executable, packages={k:importlib.metadata.version(k) for k in ["torch","peft","transformers","vllm"]},
                 previous_checkpoint=read(old / "complete.json")["adapter_sha256"])
    assert audit["previous_checkpoint"] == cfg["previous_final_sha256"]
    assert all(isinstance(v,int) for v in [audit["think_open"], audit["think_close"], *audit["terminal"]])
    freeze(RUN / "preflight.json", enc(audit))
    print(f"Verified {len(data)} TRAIN prompts, {len(notes)} teacher notes, v3 and S0 shards.", flush=True)


if __name__ == "__main__":
    prepare()
