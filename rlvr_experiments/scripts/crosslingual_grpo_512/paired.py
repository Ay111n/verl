"""Verified native-language views of the frozen 512-question training cohort."""
from collections import Counter
import importlib.metadata
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
NAME = os.environ.get("RLVR_PAIRED_EXPERIMENT", "formateval_priority512_xlang_kd_20261120")
if NAME not in {"formateval_priority512_xlang_kd_20261120","formateval_priority512_xlang_rank8_20261121",
                "formateval_priority512_restate_20261121"}:
    raise ValueError(f"Unknown paired experiment: {NAME}")
CONFIG = PROJECT / "configs" / f"{NAME}.json"
RUN = PROJECT / "runs" / NAME
OUT = ROOT / "finetuning/model/rl" / NAME
PRIMITIVES = PROJECT / "scripts/knowledge_grpo_intermittent_nll_v3"
sys.path[:0] = [str(PRIMITIVES), str(PROJECT), str(ROOT / "benchmark/eval/scripts")]
from contract import read, student_reward, text_sha
from rlvr.answer_verifier import ANSWER_EXTRACTION_VERSION, verify_answer
from scripts.same_source_dpo_316.paired_dpo import read_rows, save, sha


def training_content(row, language):
    from difficulty8_generation import ANSWER_REQUEST
    from eval_multilingual_mcqa_vllm import QUESTION_LABELS, TASK_INSTRUCTIONS, render_user_prompt
    labels = "ABCD"
    official = render_user_prompt(row, len(labels), language)
    header = TASK_INSTRUCTIONS[language].format(letters_tibetan="、".join(labels),
        letters_chinese="、".join(labels), letters_english=", ".join(labels))
    assert official.startswith(header + "\n\n")
    content = ANSWER_REQUEST[language] + official[len(header):]
    question, choices, answer = QUESTION_LABELS[language]
    expected = [ANSWER_REQUEST[language], "", f"{question} {row['question']}", choices]
    expected.extend(f"{label}. {value}" for label, value in zip(labels, row["choices"], strict=True))
    assert content == "\n".join([*expected, "", answer])
    return content


def language_view(row, language, tokenizer, cfg):
    from eval_multilingual_mcqa_vllm import render_chat_prompt
    content = training_content(row, language)
    prompt, prefilled = render_chat_prompt(tokenizer, content, "on")
    assert not prefilled and prompt.endswith("<|im_start|>assistant\n")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert len(token_ids) + cfg["max_new_tokens"] <= cfg["max_model_len"]
    return dict(id=row["id"], domain="knowledge", language=language,
        option_labels=list("ABCD"), **{"options_" + language: dict(zip("ABCD", row["choices"], strict=True))},
        canonical_answer="ABCD"[row["answer"]], prompt=prompt, prompt_token_ids=token_ids,
        prompt_sha256=text_sha(prompt))


def preflight():
    from transformers import AutoTokenizer
    cfg = read(CONFIG)
    ds = ROOT / cfg["training_data"]
    manifest = read(ds.parent / "manifest.json")
    assert sha(ds) == cfg["train_sha256"] == manifest["train_sha256"]
    assert sha(ds.parent / "manifest.json") == cfg["dataset_manifest_sha256"]
    assert sha(ds.parent / "ordered_ids.txt") == manifest["id_order_sha256"] == cfg["id_order_sha256"]
    assert sha(ds.parent / manifest["audit_file"]) == manifest["audit_sha256"]
    assert manifest["verification"]["protected_matches"] == 0
    parent = ds.parent.parent / manifest["parent"]
    sources = read(parent / "manifest.json")
    assert sha(parent / "train.jsonl") == manifest["parent_train_sha256"] == sources["train_sha256"]
    for item in [*sources["sources"].values(), *sources["protected_inputs"].values()]:
        assert sha(ROOT / (item.get("local_path") or item["file"])) == item["sha256"]
    assert ANSWER_EXTRACTION_VERSION == cfg["verifier_version"]
    assert sha(ROOT / "benchmark/eval/scripts/mcqa_answer_extraction_multilingual.py") == cfg["choice_verifier_sha256"]
    assert cfg["model_path"] == cfg["teacher_model"] and cfg["name"] == NAME
    native = read(ROOT / "finetuning/rl/processed_data/difficulty8_paired_full_20261118/protocol.json")
    base = ROOT / cfg["model_path"]
    assert base == Path(native["model"])
    for filename, digest in native["model_hashes"].items():
        assert sha(base / filename) == digest, filename
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    ids = {s:tok.encode(s, add_special_tokens=False) for s in ("<think>", "</think>")}
    assert all(len(v) == 1 and v[0] in tok.all_special_ids + list(tok.added_tokens_decoder) for v in ids.values())
    full = [{r["id"]:r for r in read_rows(ROOT / sources["sources"][k]["local_path"])}
            for k in ("data_010", "data_025")]
    paired = {}
    for row in read_rows(ds):
        uid = row["id"]
        en = full[1][uid]
        assert uid not in paired and row == full[0][uid]
        assert all(row[k] == en[k] for k in ("id", "source_dataset", "source_subject", "answer"))
        assert all(len(r["choices"]) == 4 and r["question"].strip() and
                   all(c.strip() for c in r["choices"]) for r in (row, en))
        paired[uid] = {"bo":language_view(row, "bo", tok, cfg), "en":language_view(en, "en", tok, cfg),
                       "source":row["source_dataset"], "subject":row["source_subject"]}
    assert len(paired) == cfg["training_questions"] == 512
    assert text_sha("\n".join(paired) + "\n") == cfg["id_order_sha256"]
    assert Counter(r["source"] for r in paired.values()) == manifest["source_counts"]
    save(RUN / "training_prompts.json", paired)
    result = dict(questions=len(paired), views=2*len(paired), config_sha256=sha(CONFIG),
        train_sha256=sha(ds), id_order_sha256=cfg["id_order_sha256"], model_path=str(base),
        model_hashes=native["model_hashes"], prompt_sha256=sha(RUN/"training_prompts.json"),
        prompt_protocol=cfg["prompt_protocol"], verifier=ANSWER_EXTRACTION_VERSION,
        think_open=ids["<think>"][0], think_close=ids["</think>"][0], eos=tok.eos_token_id,
        python=sys.executable, packages={k:importlib.metadata.version(k) for k in ("torch","vllm","transformers","peft")})
    if "teacher_source" in cfg:
        source_cfg = PROJECT/"configs"/(cfg["teacher_source"]+".json")
        prior = read(source_cfg)
        original = read(PROJECT/"runs"/cfg["teacher_source"]/"preflight.json")
        assert sha(source_cfg) == cfg["teacher_source_sha256"] == original["config_sha256"]
        settings = ("teacher_model","sampling_seed","teacher_seed_namespace","temperature","top_p","top_k",
                    "max_new_tokens","max_model_len","choice_verifier_sha256","prompt_protocol","train_sha256")
        assert all(cfg[k] == prior[k] for k in settings)
        assert result["model_hashes"] == original["model_hashes"] and result["prompt_sha256"] == original["prompt_sha256"]
        label_ids = {}
        for uid,pair in paired.items():
            prefix = pair["bo"]["prompt"] + cfg["rank_prefix"]
            context = tok.encode(prefix,add_special_tokens=False)
            assert context[:len(pair["bo"]["prompt_token_ids"])] == pair["bo"]["prompt_token_ids"]
            for letter in "ABCD":
                completed = tok.encode(prefix+letter,add_special_tokens=False)
                assert completed[:-1] == context and label_ids.setdefault(letter,completed[-1]) == completed[-1]
        assert len(set(label_ids.values())) == 4
        result["rank_label_ids"] = label_ids
    save(RUN/"preflight.json", result)
    print(f"Verified {len(paired)} paired TRAIN questions, prompts, verifier and S0 shards.", flush=True)


def load_inputs():
    audit, cfg = read(RUN/"preflight.json"), read(CONFIG)
    assert sha(CONFIG) == audit["config_sha256"] and sha(RUN/"training_prompts.json") == audit["prompt_sha256"]
    return cfg, audit, read(RUN/"training_prompts.json")


def judge(view, text, finish, length, cfg):
    if finish not in ("stop", "length") or length < 1:
        raise ValueError(f"Incomplete backend result for {view['id']}: {finish}, {length}")
    verdict = verify_answer(view, prompt=view["prompt"], completion=text, finish_reason=finish,
        output_tokens=length, max_new_tokens=cfg["max_new_tokens"])
    assert verdict["verifier_version"] == cfg["verifier_version"]
    return dict(reward=student_reward(verdict), selected=verdict["extracted_answer"],
                gold=view["canonical_answer"], method=verdict["choice_method"],
                status=verdict["parse_status"], finish_reason=finish)


if __name__ == "__main__":
    preflight()
