"""Read-only teacher-forced scoring of the references consumed by the formal run."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import time

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer

from contract import MODEL_OUT, PROJECT, ROOT, RUN, read, reference_tokens, student_reward, text_sha
from difficulty8_generation import PROMPT_PROTOCOL, prepare_prompts
from rlvr.answer_verifier import verify_answer
from scripts.luffy_math_pilot.common import load_model, read_rows, response_logps_batch, save, sha

OUT = RUN / "signal_diagnostic"


def verify_base(path, expected):
    for name, digest in expected.items():
        if sha(path / name) != digest:
            raise ValueError(f"S0 file changed: {name}")
    return expected


def collect_examples(events, rows, tok, cfg):
    examples = {}
    for e in events:
        if e["kind"] != "nll":
            continue
        assert len(e["references"]) == cfg["nll_batch"]
        for record in e["references"]:
            qid = record["id"]
            row = rows[qid]
            ref = next(r for r in row["english_references"] if r["rollout_id"] == record["reference_id"])
            raw = row["rollouts"][ref["rollout_index"]]
            assert qid not in examples and raw["id"] == qid and raw["language"] == "en"
            assert ref["rep"] == raw["rep"] and ref["finish_reason"] == raw["finish_reason"] == "stop"
            assert raw["answer_correct"] and raw["answer_extractable"] and not raw["truncated"]
            assert text_sha(raw["completion"]) == ref["completion_sha256"] == record["completion_sha256"]
            source_ids = tok.encode(raw["completion"], add_special_tokens=False)
            assert source_ids == ref["response_token_ids"]
            target = reference_tokens(ref, tok.eos_token_id)
            prepared = prepare_prompts(row, tok, PROMPT_PROTOCOL)
            bo = prepared[qid, "bo"]
            assert bo["token_ids"] == row["prompt_token_ids"]
            assert text_sha(prepared[qid, "en"]["prompt"]) == raw["prompt_sha256"]
            assert len(bo["token_ids"]) == record["prompt_tokens"] and len(target) == record["target_tokens"]
            assert len(target) + len(bo["token_ids"]) <= cfg["max_model_len"]
            for language in ("en", "bo"):
                verdict = verify_answer(row, prompt=prepared[qid,language]["prompt"], completion=raw["completion"],
                    finish_reason=raw["finish_reason"], output_tokens=raw["output_tokens"], max_new_tokens=cfg["max_new_tokens"])
                assert student_reward(verdict) == 1 and verdict["extracted_answer"] == row["canonical_answer"]
            assert events[e["tx"]+1]["kind"] == "online"
            examples[qid] = dict(id=qid, reference_id=ref["rollout_id"], reference_sha256=ref["completion_sha256"],
                option_count=row["option_count"], gold=row["canonical_answer"], tx=e["tx"], online_step=e["online_step"],
                prompt=bo["token_ids"], target=target, expected_pre_nll=record["loss"], scores={})
    assert len(examples) == 48 and len({x["reference_id"] for x in examples.values()}) == 48
    return examples


def replay_zero_groups(events, rows, tok, cfg):
    failures = []
    for event in events:
        if event["kind"] != "online":
            continue
        zero = {g["id"] for g in event["groups"] if sum(g["rewards"]) == 0}
        prompts = {q:prepare_prompts(rows[q],tok,PROMPT_PROTOCOL)[q,"bo"] for q in zero}
        seen = set()
        for lane in range(len(cfg["rollout_gpus"])):
            base = RUN / "rpc_workers" / f"worker{lane}"
            prefix = event["request_tag"] + f"-gen-w{lane}"
            # The frozen native verification recorded zero retried formal requests.
            files = list(base.glob(prefix+"-r*.response.json"))
            assert len(files) == 1
            reply = read(files[0])
            assert reply["adapter_sha256"] == event["input_adapter_sha256"]
            for pred in reply["records"]:
                q = pred["id"]
                if q not in zero:
                    continue
                slot = (q,pred["rollout_index"])
                assert slot not in seen and pred["prompt_token_ids"] == prompts[q]["token_ids"]
                seen.add(slot)
                verdict = verify_answer(rows[q],prompt=prompts[q]["prompt"],completion=pred["completion"],
                    finish_reason=pred["finish_reason"],output_tokens=len(pred["response_token_ids"]),max_new_tokens=cfg["max_new_tokens"])
                assert student_reward(verdict) == 0
                if verdict["truncated"]:
                    cause = "truncated"
                elif not verdict["answer_extractable"]:
                    cause = "stop_unextractable"
                else:
                    cause = "stop_extracted_wrong"
                failures.append(dict(id=q,rep=slot[1],online_step=event["online_step"],reason=cause,
                    finish_reason=pred["finish_reason"],parse_status=verdict["parse_status"],
                    extracted=verdict["extracted_answer"],gold=rows[q]["canonical_answer"],
                    native_correct=verdict["answer_correct"],output_tokens=len(pred["response_token_ids"])))
        assert seen == {(q,j) for q in zero for j in range(3)}
    return failures


def load_snapshot(model, tx, event):
    directory = MODEL_OUT / f"checkpoint-{tx:04d}"
    path = directory / "adapter_model.safetensors"
    digest = sha(path)
    assert digest == read(directory/"manifest.json")["adapter_sha256"] == event["output_adapter_sha256"]
    weights = load_file(str(path), device="cpu")
    if tx == 0:
        zero_b = [v for k,v in weights.items() if ".lora_B." in k]
        assert zero_b and all(not torch.count_nonzero(v) for v in zero_b)
    set_peft_model_state_dict(model, weights, adapter_name="default")
    live = get_peft_model_state_dict(model, adapter_name="default")
    assert weights.keys() == live.keys() and all(torch.equal(w,live[k].detach().cpu()) for k,w in weights.items())
    return digest


def score(model, x, cfg):
    lp = response_logps_batch(model, [x["prompt"]], [x["target"]], cfg["head_chunk"], temperature=1.0)[0]
    assert lp.numel() == len(x["target"]) and torch.isfinite(lp).all()
    return float(-lp.mean())


def main():
    import importlib.metadata as md
    elapsed = time.monotonic()
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1 and torch.cuda.device_count() == 1
    cfg = read(RUN/"effective_training_config.json")
    frozen = read(RUN/"preflight.json")
    done = read(RUN/"complete.json")
    prior_replay = read(RUN/"training_native_verification.json")
    assert not prior_replay["formal_retry_artifacts"] and prior_replay["responses"] == 600
    assert cfg["nll_temperature"] == 1 and cfg["epochs"] == done["epochs"] == 1
    src = ROOT / cfg["training_data"]
    assert sha(src) == frozen["data_sha256"] == done["train_sha256"]
    rows = {r["id"]:r for r in read_rows(src)}
    assert list(rows) == frozen["fixed_ids"]
    events = [read(p) for p in sorted((RUN/"steps").glob("checkpoint-*.json"))]
    assert [e["tx"] for e in events] == list(range(len(events)))
    final = events[-1]["tx"]
    assert events[-1]["output_adapter_sha256"] == done["adapter_sha256"]
    for a,b in zip(events, events[1:]):
        assert a["output_adapter_sha256"] == b["input_adapter_sha256"]
    model_path = ROOT / cfg["model_path"]
    assert str(model_path) == frozen["base_model"]
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as io:
        base_verification = io.submit(verify_base, model_path, frozen["model_hashes"])
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        examples = collect_examples(events,rows,tok,cfg)
        failures = replay_zero_groups(events,rows,tok,cfg)
        assert len(failures) == 3*sum(e["group_histogram"][0] for e in events if e["kind"] == "online")
        print(f"Verified {len(examples)} references and classified {len(failures)} all-wrong-group responses",flush=True)
        model = load_model(str(model_path), cfg["initial_seed"], adapter=str(MODEL_OUT/"checkpoint-0000"))
        model.eval()
        model.requires_grad_(False)
        model.gradient_checkpointing_disable()
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert model.get_base_model().config.attention_dropout == 0
        version_digests = {}
        plan = defaultdict(list)
        for x in examples.values():
            for tx in {x["tx"]-1, x["tx"], x["tx"]+1, final}:
                plan[tx].append(x)
        with torch.inference_mode():
            version_digests[0] = load_snapshot(model,0,events[0])
            with model.disable_adapter():
                for x in examples.values():
                    x["s0"] = score(model,x,cfg)
            checked = {}
            for tx in sorted({x["tx"] for x in examples.values()}):
                x = next(x for x in examples.values() if x["tx"] == tx)
                checked[x["id"]] = abs(x["s0"] - score(model,x,cfg))
            assert max(checked.values()) <= 1e-6, checked
            print("Scored genuine S0 with adapters disabled; neutral adapter check",checked,flush=True)
            for tx, group in sorted(plan.items()):
                version_digests[tx] = load_snapshot(model,tx,events[tx])
                for x in group:
                    x["scores"][tx] = score(model,x,cfg)
                print(f"Scored {len(group)} used references at checkpoint-{tx:04d}",flush=True)
        base_hashes = base_verification.result()
    records = []
    for x in examples.values():
        tx = x["tx"]
        records.append(dict(id=x["id"],reference_id=x["reference_id"],completion_sha256=x["reference_sha256"],
            option_count=x["option_count"],gold=x["gold"],nll_tx=tx,online_step=x["online_step"],
            prompt_tokens=len(x["prompt"]),target_tokens=len(x["target"]),s0_nll=x["s0"],
            pre_nll=x["scores"][tx-1],post_nll=x["scores"][tx],next_grpo_nll=x["scores"][tx+1],
            final_nll=x["scores"][final],logged_pre_nll=x["expected_pre_nll"]))
    assert sum(r["target_tokens"] for r in records) == prior_replay["nll_target_tokens"]
    provenance = dict(base_model=str(model_path),final_adapter=done["adapter"],base_model_hashes=base_hashes,
        adapter_sha256=version_digests,training_data_sha256=frozen["data_sha256"],
        tokenizer_sha256=frozen["tokenizer_hashes"],source_sha256=sha(Path(__file__)),
        likelihood_source_sha256=sha(PROJECT/"scripts/luffy_knowledge_388/batched_likelihood.py"),
        python=sys.executable,packages={k:md.version(k) for k in ("torch","transformers","peft","safetensors")},
        device=torch.cuda.get_device_name(),physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],temperature=1,
        dtype="bf16 base, fp32 adapter and logits",reduction="one mean over response targets including EOS, equal question weights",
        neutral_control_abs_errors=checked,elapsed_seconds=time.monotonic()-elapsed)
    save(OUT/"reference_fit_raw.json",dict(records=records,provenance=provenance))
    save(OUT/"zero_group_responses.json",dict(records=failures,original_600_replay=prior_replay["responses"],
        verifier_sha256=sha(PROJECT/"rlvr/answer_verifier.py"),training_output_budget=cfg["max_new_tokens"],
        scored_reference_ids=[x["reference_id"] for x in records]))
    print("Completed read-only diagnostic",round(provenance["elapsed_seconds"],1),"seconds",flush=True)


if __name__ == "__main__":
    main()
