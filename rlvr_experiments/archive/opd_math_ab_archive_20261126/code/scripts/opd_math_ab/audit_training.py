"""Read-only audit of Math A/B trajectory provenance, updates and measured cost."""
import argparse
import json
import math
from pathlib import Path
from transformers import AutoTokenizer
from common import paths, read_rows, save, sha, render, prompt_text, teacher_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    p = paths()
    run = p["run_root"]
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/opd_math_ab_20261116.json").read_text())
    rows = read_rows(p["train_data"])
    assert len(rows) == len({r["id"] for r in rows}) == 342
    assert sha(p["train_data"]) == config["train_sha256"]
    assert sha(p["eval_data"]) == config["eval_sha256"]
    tok = AutoTokenizer.from_pretrained(str(p["model_path"]), local_files_only=True)
    student_ids = {r["id"]: tok.encode(render(tok, prompt_text(r)), add_special_tokens=False) for r in rows}
    arms = {}
    file_hashes = {}
    def adapter_sha(directory):
        path = Path(directory) / "adapter_model.safetensors"
        if str(path) not in file_hashes:
            file_hashes[str(path)] = sha(path)
        return file_hashes[str(path)]
    initial_hashes = []
    initial_files = []
    for arm in ("answer", "solution"):
        arun = run / arm
        start_path = arun / "initialization.json"
        if not start_path.exists():
            assert args.allow_partial
            arms[arm] = {"status": "not_started", "steps": 0}
            continue
        start = json.loads(start_path.read_text())
        assert start["config"] == config
        assert start["train_sha256"] == config["train_sha256"]
        assert start["runtime"]["cuda_visible_devices"] == "0"
        assert all("lora_" in name for name in start["trainable_names"])
        initial_hashes.append(start["initial_adapter_hash"])
        initial_files.append(adapter_sha(p["model_output"] / arm / "checkpoint-0000"))
        steps = sorted(arun.glob("step-*.json"))
        assert [x.name for x in steps] == [f"step-{i:04d}.json" for i in range(1, len(steps)+1)]
        complete_path = arun / "complete.json"
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
        if not args.allow_partial:
            assert complete and len(steps) == 43
        samples = []
        all_ids = []
        rollout_seconds = step_seconds = 0.
        for number, path in enumerate(steps):
            step = json.loads(path.read_text())
            assert step["step"] == number + 1 and step["arm"] == arm
            batch = rows[number*8:(number+1)*8]
            expected_ids = [r["id"] for r in batch]
            request = json.loads((run / "rpc" / f"{arm}-step-{number:04d}.request.json").read_text())
            response = json.loads((run / "rpc" / f"{arm}-step-{number:04d}.response.json").read_text())
            assert "error" not in response
            for field in ["seed", "max_new_tokens", "temperature", "top_p", "top_k"]:
                assert request[field] == config[field]
            assert request["mode"] == "generate" and response["gpu"] == "2"
            assert Path(request["adapter"]) == p["model_output"] / arm / f"checkpoint-{number:04d}"
            assert adapter_sha(request["adapter"]) == request["adapter_sha256"] == response["adapter_sha256"]
            assert [x["id"] for x in request["prompts"]] == expected_ids
            assert [x["id"] for x in response["records"]] == expected_ids
            assert [x["id"] for x in step["samples"]] == expected_ids
            assert len(response["records"]) == len(batch)
            assert adapter_sha(step["adapter"]) == step["adapter_sha256"]
            assert Path(step["adapter"]) == p["model_output"] / arm / f"checkpoint-{number+1:04d}"
            for row, req, pred, stat in zip(batch, request["prompts"], response["records"], step["samples"], strict=True):
                assert req["prompt_token_ids"] == pred["prompt_token_ids"] == student_ids[row["id"]]
                assert stat["student_prompt_tokens"] == len(student_ids[row["id"]])
                assert stat["teacher_prompt_tokens"] == len(tok.encode(render(tok, teacher_text(row, arm)), add_special_tokens=False))
                assert stat["response_tokens"] == len(pred["response_token_ids"])
                assert 0 < stat["response_tokens"] <= config["max_new_tokens"]
                assert stat["teacher_prompt_tokens"] + stat["response_tokens"] <= 32768
                if pred["finish_reason"] == "stop":
                    assert pred["response_token_ids"][-1] == tok.eos_token_id
                assert math.isfinite(stat["loss"])
                assert math.isfinite(stat["teacher_seconds"]) and math.isfinite(stat["student_forward_backward_seconds"])
            assert math.isfinite(step["mean_sequence_kl"]) and math.isfinite(step["grad_norm"])
            assert abs(step["mean_sequence_kl"] - sum(s["loss"] for s in step["samples"])/len(batch)) < 1e-8
            all_ids.extend(expected_ids)
            samples.extend(step["samples"])
            rollout_seconds += step["rollout_seconds"]
            step_seconds += step["elapsed_seconds"]
        if complete:
            assert complete["steps"] == 43 and complete["examples"] == len(rows) == len(all_ids)
            assert all_ids == [r["id"] for r in rows]
            assert complete["config"] == config
            assert complete["initial_adapter_hash"] == start["initial_adapter_hash"]
            assert complete["initial_adapter_hash"] != complete["final_adapter_hash"]
            assert complete["frozen_parameter_checksums_unchanged"]
            assert adapter_sha(complete["adapter"]) == complete["adapter_sha256"]
            latest = json.loads((p["model_output"] / arm / "latest.json").read_text())
            assert latest["step"] == 43 and latest["adapter_sha256"] == complete["adapter_sha256"]
        sums = {key: sum(x[key] for x in samples) for key in
                ["student_prompt_tokens", "teacher_prompt_tokens", "response_tokens",
                 "teacher_seconds", "student_forward_backward_seconds"]}
        arms[arm] = {
            "status": "complete" if complete else "running", "steps": len(steps), "examples": len(samples),
            "finite_losses_and_grad_norms": True, "current_policy_adapter_chain_verified": True,
            "tibetan_only_student_prompts_verified": True, "teacher_prompt_lengths_verified": True,
            "native_eos_verified": True, "totals": sums,
            "rollout_seconds": rollout_seconds, "step_seconds_including_queue_and_checkpoints": step_seconds,
            "peak_allocated_gib": max((x["peak_allocated_gib"] for x in samples), default=0.),
            "peak_reserved_gib": max((x["peak_reserved_gib"] for x in samples), default=0.),
        }
    assert len(set(initial_hashes)) <= 1 and len(set(initial_files)) <= 1
    result = {
        "status": "partial" if args.allow_partial else "complete",
        "audit_script_sha256": sha(Path(__file__)), "arms": arms,
        "initial_tensor_hashes_identical": True if len(initial_hashes) == 2 else None,
        "initial_adapter_files_identical": True if len(initial_files) == 2 else None,
        "verified_adapter_files": len(file_hashes),
        "frozen_base_check_scope": "All trainable names are LoRA; trainer checks no gradients on any frozen parameter and compares selected immutable tensor hashes, not every in-memory base tensor.",
        "timing_scope": "Teacher forward and student forward/backward are CUDA-synchronized. Step wall time includes RPC queue waits, optimizer and checkpoint I/O. Unequal teacher context and student trajectories mean unequal compute.",
        "gpu_accounting": "Both trainers CUDA=0 and every training rollout CUDA=2; final evaluation may reuse these two physical devices."
    }
    save(run / ("audit_partial.json" if args.allow_partial else "audit_training.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
