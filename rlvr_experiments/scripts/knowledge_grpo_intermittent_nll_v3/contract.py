"""Fixed data, reward and queue contracts for Tibetan GRPO with intermittent NLL."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
NAME = "knowledge_grpo_intermittent_nll_v3_200_20261119"
CONFIG = PROJECT / "configs" / (NAME + ".json")
RUN = PROJECT / "runs" / NAME
MODEL_OUT = ROOT / "finetuning/model/rl" / NAME
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "finetuning/rl/scripts"))


def read(path):
    return json.loads(Path(path).read_bytes())


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def advantages(rewards, group_size=3):
    if group_size < 2 or len(rewards) != group_size or any(r not in (0, 1) for r in rewards):
        raise ValueError(f"Expected {group_size} binary student rewards")
    mean = sum(rewards) / group_size
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


def preflight(config_path=CONFIG, run_dir=RUN):
    import importlib.metadata as md
    from freeze_knowledge_enref_v3_200 import load_inputs, reference, category
    from difficulty8_generation import PROMPT_PROTOCOL, SLOTS, prepare_prompts, sample_seed
    from prepare_mixed_rlvr_2000_200 import read_rows, sha
    from rlvr.answer_verifier import ANSWER_EXTRACTION_VERSION, verify_answer
    from scripts.same_source_dpo_316.paired_dpo import save

    cfg = read(config_path)
    d = ROOT / cfg["training_data"]
    m = read(d.parent / "manifest.json")
    assert sha(d.parent / "manifest.json") == cfg["dataset_manifest_sha256"]
    assert sha(d) == m["file_sha256"]["train.jsonl"] == cfg["train_sha256"]
    assert m["scorer"] == cfg["verifier_version"] == ANSWER_EXTRACTION_VERSION
    scorer = ROOT / "benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py"
    assert sha(scorer) == m["evidence"]["choice_verifier_sha256"] == cfg["choice_verifier_sha256"]
    assert sha(PROJECT / "rlvr/answer_verifier.py") == m["evidence"]["grpo_verifier_sha256"]
    rows = read_rows(d)
    ids = [r["id"] for r in rows]
    selection = read_rows(d.parent / "selection.jsonl")
    assert sha(d.parent / "selection.jsonl") == m["file_sha256"]["selection.jsonl"]
    assert len(rows) == len(set(ids)) == 200 and ids == [s["id"] for s in selection]
    assert text_sha("\n".join(ids) + "\n") == m["id_order_sha256"] == cfg["id_order_sha256"]
    tok, base, cache, published, protected, evidence = load_inputs()
    assert evidence["protected_inputs"] == m["evidence"]["protected_inputs"] and len(evidence["protected_inputs"]) == 25
    model = ROOT / cfg["model_path"]
    assert str(model) == evidence["model"] == m["evidence"]["model"]
    native = read(ROOT/"finetuning/rl/processed_data/difficulty8_paired_full_20261118/protocol.json")
    assert all(sha(model / name) == digest for name, digest in native["model_hashes"].items())
    counts, strata = Counter(), Counter()
    contexts = []
    for row in rows:
        uid = row["id"]
        assert uid in cache and not protected.match(row) and all(row[k] == v for k, v in base[uid].items())
        assert row["domain"] == "knowledge" and row["verifier_version"] == cfg["verifier_version"]
        prompts = prepare_prompts(row, tok, PROMPT_PROTOCOL)
        bo = prompts[uid, "bo"]["token_ids"]
        assert bo == row["prompt_token_ids"]
        assert tok.decode(bo, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prompts[uid,"bo"]["prompt"]
        recs = row["rollouts"]
        refs = {v["rep"]: v for v in row["english_references"]}
        assert len(recs) == 8 and refs and len(refs) == len(row["english_references"])
        scores = Counter()
        eligible = []
        for i, (lang, rep) in enumerate(SLOTS):
            pred = recs[i]
            prompt = prompts[uid, lang]["prompt"]
            assert (pred["id"], pred["language"], pred["rep"], pred["seed"]) == (uid, lang, rep, sample_seed(uid, lang, rep))
            assert pred["prompt_sha256"] == text_sha(prompt) and pred["completion_sha256"] == text_sha(pred["completion"])
            verdict = verify_answer(row, prompt=prompt, completion=pred["completion"], finish_reason=pred["finish_reason"],
                                    output_tokens=pred["output_tokens"], max_new_tokens=cfg["max_new_tokens"])
            assert verdict["verifier_version"] == ANSWER_EXTRACTION_VERSION
            assert (student_reward(verdict), verdict["extracted_answer"], verdict["parse_status"]) == (
                pred["answer_correct"], pred["selected"], pred["parse_status"])
            scores[lang] += student_reward(verdict)
            if lang != "en":
                continue
            metadata, _ = reference(pred, verdict, tok, len(bo))
            if metadata is None:
                continue
            eligible.append(rep)
            ref = refs[rep]
            assert all(ref[k] == v for k,v in metadata.items())
            assert ref["rollout_index"] == i
            assert ref["response_token_ids"] == tok.encode(pred["completion"], add_special_tokens=False)
            target = reference_tokens(ref, tok.eos_token_id)
            assert len(bo) + len(target) == ref["training_context_tokens"] <= cfg["max_model_len"]
            assert target.count(tok.eos_token_id) == 1
            contexts.append(len(bo) + len(target))
        assert sorted(refs) == eligible
        assert category(scores["bo"], scores["en"], bool(refs)) == row["selection_category"]
        strata[(row["selection_category"], row["option_count"])] += 1
        counts["references"] += len(refs)
        counts["raw_responses"] += len(recs)
    expected = {(k, n): m["statistics"]["gaps_by_category_options"][f"{k}/{n}"]["target"]
                for k in m["quotas"] for n in (4,10)}
    assert dict(strata) == expected and counts == {"references":581, "raw_responses":1600}
    assert max(contexts) == m["statistics"]["max_context_with_eos"]
    save(run_dir/"preflight.json", dict(count=counts, max_reference_context_with_eos=max(contexts),
        eos=tok.eos_token_id, data_sha256=sha(d), id_order_sha256=cfg["id_order_sha256"], fixed_ids=ids,
        base_model=str(model), model_hashes=native["model_hashes"], protected_matches=0,
        dataset_manifest_sha256=cfg["dataset_manifest_sha256"], verifier_version=ANSWER_EXTRACTION_VERSION,
        choice_verifier_sha256=sha(scorer), python=sys.executable,
        strata={f"{c}/{n}":v for (c,n),v in strata.items()},
        packages={k:md.version(k) for k in ("torch","vllm","peft","transformers","safetensors")}))


if __name__ == "__main__":
    preflight()
