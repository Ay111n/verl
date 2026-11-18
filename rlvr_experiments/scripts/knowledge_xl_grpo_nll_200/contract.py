"""Fixed data, reward and queue contracts for Tibetan GRPO with intermittent NLL."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
NAME = "knowledge_xl_grpo_nll_200_20261119"
CONFIG = PROJECT / "configs" / (NAME + ".json")
RUN = PROJECT / "runs" / NAME
MODEL_OUT = ROOT / "finetuning/model/rl" / NAME
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "finetuning/rl/scripts"))


def read(path):
    return json.loads(Path(path).read_bytes())


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def advantages(rewards):
    if len(rewards) != 3 or any(r not in (0, 1) for r in rewards):
        raise ValueError("Expected three binary student rewards")
    mean = sum(rewards) / 3
    return [r - mean for r in rewards]


def student_reward(verification):
    if not verification["execution_valid"]:
        raise ValueError("Incomplete backend result")
    return int(verification["answer_correct"] and verification["answer_extractable"] and
               not verification["truncated"])


def refresh_pool(pending, groups, data):
    queue = dict.fromkeys(pending)
    for group in groups:
        qid = group["id"]
        if sum(group["rewards"]):
            queue.pop(qid, None)
        elif data[qid]["english_references"]:
            queue.setdefault(qid, None)
    return list(queue)


def choose_refs(qids, data, uses):
    chosen = []
    for qid in qids:
        refs = sorted(data[qid]["english_references"], key=lambda r: (r["rep"], r["rollout_id"]))
        chosen.append((qid, refs[uses.get(qid, 0) % len(refs)]))
    return chosen


def reference_tokens(ref, eos):
    ids = ref["response_token_ids"]
    if not ids or eos in ids[:-1]:
        raise ValueError("Empty or prematurely terminated reference " + ref["rollout_id"])
    return ids if ids[-1] == eos else ids + [eos]


def epoch_batches(rows, cfg, count):
    batches = []
    for epoch in range(count):
        ids = list(rows)
        random.Random(cfg["order_seed"] + epoch).shuffle(ids)
        batches.extend([ids[i:i + cfg["batch_size"]] for i in range(0, len(ids), cfg["batch_size"])])
    return batches


def pilot_ids(data):
    strata = defaultdict(list)
    for qid, row in data.items():
        strata[row["selection_category"], row["option_count"]].append(qid)
    result = []
    for (category, size), group in sorted(strata.items()):
        count = {"class_1": 5, "class_2": 2, "class_3": 1}[category]
        result.extend(sorted(group, key=lambda q: text_sha("pilot|" + q))[:count])
    longest = max(data, key=lambda q: max(r["training_context_tokens"] for r in data[q]["english_references"]))
    if longest not in result:
        longest_row = data[longest]
        replace_at = next(i for i, q in enumerate(result) if all(
            data[q][k] == longest_row[k] for k in ("selection_category", "option_count")))
        result[replace_at] = longest
    assert len(result) == len(set(result)) == 16
    return result


def preflight():
    import importlib.metadata as md
    from prepare_knowledge_enref_grpo_nll_200 import inputs, inspect_question, classify
    from difficulty8_generation import PROMPT_PROTOCOL, prepare_prompts
    from prepare_mixed_rlvr_2000_200 import read_rows, sha
    from scripts.same_source_dpo_316.paired_dpo import save
    cfg = read(CONFIG)
    data_file = ROOT / cfg["training_data"]
    manifest = read(data_file.parent / "manifest.json")
    assert sha(data_file) == manifest["train_sha256"] == cfg["train_sha256"]
    rows = read_rows(data_file)
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(rows) == 200 and ids == manifest["id_order"]
    assert text_sha("\n".join(ids) + "\n") == manifest["id_order_sha256"] == cfg["id_order_sha256"]
    snapshot = ROOT / "finetuning/rl/processed_data/difficulty8_paired_full_20261118"
    base, protected, tokenizer, evidence = inputs(ROOT, snapshot)
    assert evidence["protected_sha256"] == manifest["protected_sha256"] and len(evidence["protected_sha256"]) == 25
    assert evidence["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    assert evidence["sampling_protocol_sha256"] == manifest["sampling_protocol_sha256"]
    assert all(sha(ROOT / k) == v for k, v in manifest["source_files_sha256"].items())
    model = ROOT / cfg["model_path"]
    assert str(model) == evidence["sampling_protocol"]["model"]
    assert all(sha(model / k) == v for k, v in evidence["sampling_protocol"]["model_hashes"].items())
    eos = tokenizer.eos_token_id
    count = Counter()
    contexts = []
    strata = Counter()
    for row in rows:
        qid = row["id"]
        assert not protected.match(row) and row["domain"] == "knowledge"
        assert all(row[k] == base[qid][k] for k in ("question_bo","question_en","options_bo","options_en","canonical_answer"))
        records = {(r["language"], r["rep"]): r for r in row["rollouts"]}
        prompt, eligible, _ = inspect_question(row, records, tokenizer)
        assert prompt == row["prompt_token_ids"] and row["has_nll_reference"] and len(row["rollouts"]) == 8
        assert classify(row, True) == row["selection_category"]
        render = prepare_prompts(row, tokenizer, PROMPT_PROTOCOL)[qid, "bo"]["prompt"]
        assert tokenizer.decode(prompt, skip_special_tokens=False, clean_up_tokenization_spaces=False) == render
        refs = row["english_references"]
        assert sorted(r["rep"] for r in refs) == sorted(eligible)
        for ref in refs:
            raw = row["rollouts"][ref["rollout_index"]]
            assert raw == records["en", ref["rep"]]
            assert ref["rollout_id"] == f'{qid}:en:{ref["rep"]}' and ref["token_origin"] == "offline_encoded_original_text"
            assert text_sha(raw["completion"]) == ref["completion_sha256"] and eligible[ref["rep"]] == ref["response_token_ids"]
            tokens = reference_tokens(ref, eos)
            contexts.append(len(tokens) + len(prompt))
            count["added_eos"] += int(len(tokens) > len(ref["response_token_ids"]))
        strata[row["selection_category"], row["option_count"]] += 1
        count["references"] += len(refs)
        count["raw_responses"] += len(row["rollouts"])
    assert sorted(strata.values()) == [10,10,30,30,60,60]
    assert count["references"] == 520 and count["raw_responses"] == 1600 and max(contexts) <= cfg["max_model_len"]
    save(RUN / "preflight.json", dict(count, max_reference_context_with_eos=max(contexts), eos=eos,
        data_sha256=sha(data_file), id_order_sha256=cfg["id_order_sha256"], fixed_ids=ids,
        protected_files=evidence["protected_sha256"], protected_matches=0, base_model=str(model),
        model_hashes=evidence["sampling_protocol"]["model_hashes"], tokenizer_hashes=evidence["tokenizer_sha256"],
        strata={f"{c}/{s}":v for (c,s),v in strata.items()}, python=sys.executable,
        packages={k:md.version(k) for k in ("torch","vllm","peft","transformers","safetensors")}))
    print("Preflight passed", count, "maximum context", max(contexts), flush=True)


if __name__ == "__main__":
    preflight()
