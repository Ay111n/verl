"""Small V1 subclass; all training, checkpoint and validation loops remain upstream."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import csv
from pathlib import Path

from rlvr.replay import load_plan


def verify_entropy_observation(metrics):
    """Require the native pre-update actor entropy even without entropy bonus."""
    value = metrics.get("actor/entropy")
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError("Pre-update actor entropy observation is missing or non-finite")
    metrics["rlvr/entropy_pre_update_observed"] = 1


def save_completed_checkpoint_after_eval(trainer, completed_updates):
    """Save the completed update, preserving the upstream next-step counter."""
    completed_updates = int(completed_updates)
    previous = int(trainer.global_steps)
    if previous not in (completed_updates, completed_updates + 1):
        raise RuntimeError("Final checkpoint counter does not match completed training")
    try:
        trainer.global_steps = completed_updates
        trainer._save_checkpoint()
    finally:
        trainer.global_steps = previous


def append_jsonl(path, value):
    append_jsonl_batch(path, [value])


def append_jsonl_batch(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        def native(item):
            if hasattr(item, "item"):
                return item.item()
            if hasattr(item, "tolist"):
                return item.tolist()
            raise TypeError(f"Cannot serialize audit value {type(item)}")
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=native) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def optimizer_snapshot(directory):
    result = {}
    for path in Path(directory).glob("*.jsonl"):
        result[str(path)] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def verify_optimizer_events(directory, previous, expected_ranks):
    records = []
    for path in Path(directory).glob("*.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines()
        records.extend(json.loads(line) for line in lines[previous.get(str(path), 0):])
    if len(records) != expected_ranks or len({r["rank"] for r in records}) != expected_ranks:
        raise RuntimeError(f"Expected one optimizer event on each of {expected_ranks} ranks, got {records}")
    if any(r["attempted_steps"] != 1 or r["successful_steps"] != 1
           or not r.get("grad_norm_finite", False) for r in records):
        raise RuntimeError("An optimizer update was skipped or duplicated")
    return records


def checkpoint_events(directory, checkpoint_path, event):
    by_rank = {}
    for path in Path(directory).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["event"] == event and row["checkpoint_path"] == str(checkpoint_path):
                rank = str(row["rank"])
                if rank not in by_rank or row["timestamp_ns"] > by_rank[rank]["timestamp_ns"]:
                    by_rank[rank] = row
    return by_rank


def data_progress_summary(state):
    result = {}
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (int, bool)) and any(word in path for word in ("yield", "sample", "epoch", "step", "finished")):
            result[path] = value
    visit(state, "data")
    return result


def finalize_exposure(run_dir, manifest_path, successful_updates=None, *, allow_benchmark_prefix=False):
    from rlvr.contracts import check_exposure, atomic_json
    directory = Path(run_dir)
    plan = load_plan(manifest_path)
    commits = [json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [int(row["update_id"]) for row in commits]
    count = max(completed, default=0) if successful_updates is None else int(successful_updates)
    if sorted(completed) != list(range(1, count + 1)):
        raise RuntimeError("Actual optimizer commits are missing or duplicated")
    full_length = len(plan["updates"])
    benchmark_prefix = False
    if allow_benchmark_prefix:
        config = json.loads((directory / 'resolved_config.json').read_text())
        benchmark_prefix = (config['rlvr']['phase'] == 'benchmark' and
                            config['trainer']['total_training_steps'] == count and 0 < count < full_length)
        if not benchmark_prefix:
            raise RuntimeError('Benchmark prefix permission requires matching benchmark config/budget')
    if count != full_length and plan.get("phase") != "smoke" and not benchmark_prefix:
        raise RuntimeError("Only smoke permits an intentional partial manifest run")
    partial_plan = {**plan, "updates": plan["updates"][:count], "train_updates": count}
    records = [json.loads(line) for line in (directory / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    result = check_exposure(records, partial_plan, successful_updates=completed)
    result.update(manifest_updates=full_length, completed_updates=count,
                  partial_smoke=count != full_length and not benchmark_prefix,
                  manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest())
    if benchmark_prefix:
        result.update(partial_benchmark=True, scope='configured benchmark prefix; not a complete reward-dev epoch')
    atomic_json(directory / "exposure_summary.json", result)
    columns = ["update_id", "sample_id", "group_id", "sequence_position", "rollout_index", "seed",
               "update_committed", "domain", "difficulty_bucket", "correctness", "total_reward",
               "component_r0", "truncated", "thinking_tokens"]
    temporary = directory / "train_rollout_exposure.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, directory / "train_rollout_exposure.csv")
    # Aggregate only verified actual committed trajectories, never the planned sample list.
    grouped = {}
    for row in records:
        key = (row["update_id"], row["sample_id"], row["group_id"], row["sequence_position"])
        grouped.setdefault(key, []).append(row)
    group_columns = ["sample_id", "update_id", "group_id", "generation_count", "sequence_position",
                     "rollout_indices", "seeds", "update_committed", "domain", "difficulty_bucket"]
    temporary = directory / "train_exposure_manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_columns)
        writer.writeheader()
        for (update_id, sample_id, group_id, position), members in sorted(grouped.items(), key=lambda item: item[0][3]):
            members.sort(key=lambda row: row["rollout_index"])
            if len(members) != plan["num_generations"]:
                raise RuntimeError("Actual group exposure has the wrong generation count")
            writer.writerow({"sample_id": sample_id, "update_id": update_id, "group_id": group_id,
                "generation_count": len(members), "sequence_position": position,
                "rollout_indices": json.dumps([row["rollout_index"] for row in members]),
                "seeds": json.dumps([row["seed"] for row in members]), "update_committed": True,
                "domain": members[0].get("domain"), "difficulty_bucket": members[0].get("difficulty_bucket")})
    os.replace(temporary, directory / "train_exposure_manifest.csv")
    return result


def verify_eval_contract(records, samples, generations, base_seed=42):
    from rlvr.contracts import stable_seed
    expected = {(str(sample["sample_id"]), generation)
                for sample in samples for generation in range(generations)}
    actual = [(str(row["sample_id"]), int(row["rollout_index"])) for row in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Evaluation sample/generation exposure differs from its frozen contract")
    for row in records:
        seed = stable_seed(row["sample_id"], row["rollout_index"], base_seed, "rlvr-eval-seed-v1")
        if row["seed"] != seed or row["seed_namespace"] != "rlvr-eval-seed-v1":
            raise RuntimeError("Evaluation seed differs from the 400-sample seed manifest")
        if "native_sampling_seed" in row and row["native_sampling_seed"] != seed:
            raise RuntimeError("Native backend evaluation seed differs from its frozen contract")
    return {"samples": len(samples), "responses": len(records), "generations": generations,
            "sample_generation_seed_contract_passed": True}


def compare_repeat_evaluation(first, second):
    def mapping(rows):
        return {(r["sample_id"], r["rollout_index"]): r for r in rows}
    left, right = mapping(first), mapping(second)
    if set(left) != set(right):
        raise RuntimeError("Repeated evaluation changed sample/generation coverage")
    fields = ("seed", "response_token_ids", "native_finish_reason", "native_stop_reason", "total_reward")
    mismatches = [{"sample_id": key[0], "rollout_index": key[1],
                   "fields": [field for field in fields if left[key].get(field) != right[key].get(field)]}
                  for key in left if any(left[key].get(field) != right[key].get(field) for field in fields)]
    return {"compared_responses": len(left), "matching_responses": len(left)-len(mismatches),
            "exact_token_seed_termination_reward_match": not mismatches, "mismatches": mismatches}


def repeat_comparison_gate(comparison):
    """Sampled numerical differences are observations; seeds and Greedy stay strict."""
    failures = []
    if set(comparison) != {'greedy', 'sampled_g8'}:
        failures.append('Missing repeated Greedy or sampled mode')
    for mode, result in comparison.items():
        if mode == 'greedy' and not result.get('exact_token_seed_termination_reward_match'):
            failures.append('Greedy repeat differs')
        if any('seed' in row.get('fields', []) for row in result.get('mismatches', [])):
            failures.append(f'{mode} repeat seed differs')
    return {'passed': not failures, 'failures': failures,
            'greedy_exact_required': True, 'sampled_exact_tokens_required': False,
            'sampled_mismatching_responses_observed': len(comparison.get('sampled_g8', {}).get('mismatches', []))}


def build_trainer_class():
    import numpy as np
    import transfer_queue as tq
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from verl.utils import tensordict_utils as tu
    from rlvr.replay import OrderedReplayBuffer

    class RLVRTrainer(PPOTrainerSync):
        def __init__(self, config):
            self.plan = load_plan(config.rlvr.manifest_path)
            self.manifest_sha = hashlib.sha256(Path(config.rlvr.manifest_path).read_bytes()).hexdigest()
            self.run_dir = Path(config.rlvr.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._last_records_step = None
            self._records = None
            if config.trainer.v1.trainer_mode != "sync":
                raise ValueError("RLVR requires V1 synchronous mode")
            if config.algorithm.adv_estimator != "grpo":
                raise ValueError("RLVR requires GRPO")
            for key, actual, expected in (
                ("train_batch_size", config.data.train_batch_size, config.rlvr.prompts_per_update),
                ("gen_batch_size", config.data.gen_batch_size, config.rlvr.prompts_per_update),
                ("rollout.n", config.actor_rollout_ref.rollout.n, config.rlvr.num_generations),
                ("ppo_mini_batch_size", config.actor_rollout_ref.actor.ppo_mini_batch_size,
                 config.rlvr.prompts_per_update), ("ppo_epochs", config.actor_rollout_ref.actor.ppo_epochs, 1)):
                if actual != expected:
                    raise ValueError(f"Frozen contract violated: {key}={actual}, expected {expected}")
            if config.data.shuffle or config.actor_rollout_ref.actor.shuffle:
                raise ValueError("Manifest and PPO shuffling must both be disabled")
            if config.actor_rollout_ref.actor.use_dynamic_bsz:
                raise ValueError("Dynamic length packing would change frozen actor consumption order")
            if config.algorithm.get("filter_groups", {}).get("enable", False):
                raise ValueError("Group filtering is forbidden")
            if config.trainer.v1.sync.get("parameter_sync_step", 1) != 1:
                raise ValueError("One parameter sync per update is required")
            if config.trainer.total_training_steps > len(self.plan["updates"]):
                raise ValueError("Requested steps exceed immutable manifest")
            if config.actor_rollout_ref.actor.checkpoint.get("async_save", False):
                raise ValueError("Asynchronous checkpoint saves cannot certify smoke resume")
            for field in ("save_contents", "load_contents"):
                contents = config.actor_rollout_ref.actor.checkpoint.get(field, ["model", "optimizer", "extra"])
                if not {"model", "optimizer", "extra"} <= set(contents):
                    raise ValueError("Full model/optimizer/extra checkpoint contents must be frozen")
            if config.trainer.get("del_local_ckpt_after_load", False):
                raise ValueError("Resume must retain the checkpoint for inventory/state audit")
            if config.rlvr.strict_exposure and not config.rlvr.get('resume_smoke',False):
                ids = [s["sample_id"] for u in self.plan["updates"] for s in u["samples"]]
                if len(ids) != len(set(ids)):
                    raise ValueError("Strict experiment manifest repeats training samples")
            super().__init__(config)

        def _build_replay_buffer(self):
            sampler = self.config.trainer.v1.sampler
            return OrderedReplayBuffer(
                trainer_mode="sync", trainer_config=self.config.trainer.v1.sync,
                max_off_policy_threshold=sampler.max_off_policy_threshold,
                max_off_policy_strategy=sampler.max_off_policy_strategy,
                sampler_kwargs={"manifest_path": self.config.rlvr.manifest_path,
                                "num_generations": self.config.rlvr.num_generations,
                                "timeout_seconds": self.config.rlvr.get("rollout_timeout_seconds", 3600)},
                refill_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("Refill is forbidden")))

        def on_init_end(self):
            self._initial_checkpoint_steps = self.global_steps
            if self.global_steps:
                contract = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "rlvr_contract.json"
                if not contract.is_file():
                    raise RuntimeError("Resume checkpoint has no RLVR exposure/manifest contract")
                saved = json.loads(contract.read_text(encoding="utf-8"))
                if saved["manifest_sha256"] != self.manifest_sha:
                    raise RuntimeError("Resume manifest differs from checkpoint manifest")
                if hashlib.sha256((contract.parent / "data.pt").read_bytes()).hexdigest() != saved["data_state_sha256"]:
                    raise RuntimeError("Checkpoint data progress file changed after it was saved")
                actor_path = contract.parent / "actor"
                restored = checkpoint_events(self.run_dir / "checkpoint_events", actor_path, "load_complete")
                expected_ranks = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
                if len(restored) != expected_ranks:
                    raise RuntimeError("Missing per-rank actual optimizer/scheduler/RNG load evidence")
                comparisons = {}
                for rank, row in restored.items():
                    previous = saved["rank_states"][rank]["summary"]
                    current = row["summary"]
                    if current != previous:
                        raise RuntimeError(f"Restored optimizer/scheduler/RNG state differs on rank {rank}")
                    comparisons[rank] = {"state_matches_saved": True, "restored": current,
                                         "load_audit": row.get("load_audit")}
                    if self.config.rlvr.get('resume_smoke',False) and not all(
                        row.get('load_audit',{}).get(k) is True for k in
                        ('model_lora_exact','optimizer_all_tensors_exact','scheduler_full_state_exact')):
                        raise RuntimeError('Resume smoke lacks exact model/optimizer/scheduler evidence')
                actual_data_progress = data_progress_summary(self.train_dataloader.state_dict())
                if actual_data_progress != saved["data_progress"]:
                    raise RuntimeError("Restored dataloader progress differs from saved checkpoint")
                from rlvr.contracts import atomic_json
                atomic_json(self.run_dir / f"resume-check-step{self.global_steps}.json", {
                    "checkpoint_step": self.global_steps, "ranks": comparisons,
                    "data_progress": actual_data_progress, "data_progress_matches": True})
            ledger = self.run_dir / "exposure.jsonl"
            if ledger.exists():
                records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                if any(r["update_id"] > self.global_steps for r in records):
                    raise RuntimeError("Uncheckpointed exposures exist; automatic re-exposure is forbidden")
            if self.global_steps:
                path = self.run_dir / "updates.jsonl"
                complete = [json.loads(line)["update_id"] for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
                if sorted(complete) != list(range(1, self.global_steps + 1)):
                    raise RuntimeError("Checkpoint has no complete audited prior-update ledger")
            super().on_init_end()

        def on_train_begin(self):
            super().on_train_begin()
            if self.config.rlvr.phase in ("smoke", "reward_dev") and self._initial_checkpoint_steps == 0:
                self._run_eval_suite("pre", checkpoint_step=0)

        def _validate(self):
            return self._run_eval_suite("scheduled", checkpoint_step=self.global_steps)

        def _run_eval_suite(self, label, checkpoint_step):
            from omegaconf import OmegaConf
            from rlvr.contracts import stable_seed, stable_uid, atomic_json, ids_hash
            from rlvr.metrics import summarise_records
            from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
            model_type = json.loads((Path(self.config.actor_rollout_ref.model.path) / "config.json").read_text())["model_type"]
            template_kwargs = OmegaConf.to_container(self.config.data.apply_chat_template_kwargs, resolve=True)
            builder = create_continuous_token_builder(self.tokenizer, hf_model_type=model_type,
                                                     chat_template_kwargs=template_kwargs)
            expected_prompt_hashes = {}
            # Iterate the finite validation dataloader to freeze the exact source order.
            samples = []
            for data in self.val_dataloader:
                ids = data.get("sample_id")
                if ids is None:
                    ids = [extra["sample_id"] for extra in data["extra_info"]]
                for position, sample_id in enumerate(ids):
                    token_ids = builder.build_initial_tokens(list(data["raw_prompt"][position]))
                    if len(token_ids) > int(self.config.data.max_prompt_length):
                        raise RuntimeError("Evaluation prompt exceeds its frozen context contract")
                    expected_prompt_hashes[str(sample_id)] = hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
                    samples.append({"sample_id": str(sample_id), "sequence_position": len(samples),
                                    "seeds": [stable_seed(str(sample_id), i, int(self.config.rlvr.base_seed),
                                                         "rlvr-eval-seed-v1") for i in range(8)]})
            if len(samples) != len({row["sample_id"] for row in samples}):
                raise RuntimeError("Validation dataset repeats a sample")
            if self.config.rlvr.phase == "reward_dev" and len(samples) != 400:
                raise RuntimeError("Reward development evaluation must contain exactly 400 fixed samples")
            seed_manifest = {"namespace": "rlvr-eval-seed-v1", "base_seed": int(self.config.rlvr.base_seed),
                             "ids_sha256": ids_hash([s["sample_id"] for s in samples]), "samples": samples}
            submission_contract = {**seed_manifest, "prompt_token_hashes": expected_prompt_hashes,
                "prompt_template_kwargs": template_kwargs,
                "sampling_modes": {"greedy": {"generations": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
                                   "sampled_g8": {"generations": 8, "temperature": float(self.config.rlvr.eval.temperature),
                                                  "top_p": float(self.config.rlvr.eval.top_p), "top_k": int(self.config.rlvr.eval.top_k)}},
                "max_response_tokens": int(self.config.actor_rollout_ref.rollout.response_length)}
            atomic_json(self.run_dir / "evaluation_submission_contract.json", submission_contract, frozen=True)
            directory = self.run_dir / "evaluation" / f"{label}-step{checkpoint_step}"
            directory.mkdir(parents=True, exist_ok=True)
            if label == 'pre' and self.config.rlvr.get('reuse_pre_eval_from'):
                from rlvr.eval_reuse import reuse_pre_eval
                metrics = reuse_pre_eval(self.config.rlvr.reuse_pre_eval_from, self.run_dir, submission_contract)
                print('PRE_EVAL_REUSED: no generation requests; entering training updates', flush=True)
                return metrics
            original_manager = self.agent_loop_manager
            original_kwargs = OmegaConf.to_container(self.config.actor_rollout_ref.rollout.val_kwargs, resolve=True)
            all_metrics, suite_records = {}, {}
            trainer = self

            class EvalManager:
                def __init__(self, mode, generations):
                    self.mode, self.generations, self.offset = mode, generations, 0

                def generate_sequences(self, batch):
                    ids = tu.get(batch, "sample_id", default=None)
                    if ids is None:
                        ids = [extra["sample_id"] for extra in tu.get(batch, "extra_info")]
                    expected_samples = samples[self.offset:self.offset+len(batch)]
                    if list(map(str, ids)) != [s["sample_id"] for s in expected_samples]:
                        raise RuntimeError("Validation source order changed during evaluation")
                    uids = tu.get(batch, "uid")
                    expected_batch = [{**s, "group_id": str(uid)} for s, uid in zip(expected_samples, uids)]
                    tu.assign_non_tensor_stack(batch, "sample_id", list(map(str, ids)))
                    tu.assign_non_tensor_stack(batch, "agent_name", ["rlvr_single_turn"] * len(batch))
                    tu.assign_non_tensor_stack(batch, "sequence_position", [s["sequence_position"] for s in expected_samples])
                    tu.assign_non_tensor_stack(batch, "rlvr_eval_mode", [self.mode] * len(batch))
                    tu.assign_non_tensor_stack(batch, "__rollout_n__", [self.generations] * len(batch))
                    trainer.replay_buffer.validation_expected = (expected_batch, self.generations)
                    self.offset += len(batch)
                    original_manager.generate_sequences(batch)

            def capture(batch):
                fields = ["sample_id", "session_id", "prompts", "responses", "response_mask", "rm_scores", "extra_fields"]
                data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
                def rows(name):
                    value = data[name]
                    if hasattr(value, "tolist") and not getattr(value, "is_nested", False):
                        return value.tolist()
                    return [value[i].tolist() if hasattr(value[i], "tolist") else value[i] for i in range(len(batch))]
                columns = {name: rows(name) for name in fields}
                for i in range(len(batch)):
                    extra = columns["extra_fields"][i]
                    info = extra["reward_extra_info"]
                    sid = str(columns["sample_id"][i])
                    record = {**info, "sample_id": sid, "rollout_index": int(columns["session_id"][i]),
                              "group_id": stable_uid(sid), "mode": mode, "checkpoint_step": checkpoint_step,
                              "seed": int(extra["sampling_seed"]), "seed_namespace": extra["seed_namespace"],
                              "response_token_ids": columns["responses"][i],
                              "prompt_token_ids": columns["prompts"][i],
                              "response_mask": columns["response_mask"][i], "advantages": None,
                              "native_finish_reason": extra["native_finish_reason"],
                              "native_sampling_seed": extra["native_sampling_seed"],
                              "native_stop_reason": extra["native_stop_reason"],
                              "total_reward": sum(columns["rm_scores"][i]), "reward_extra_info": info,
                              "response": self.tokenizer.decode(columns["responses"][i], skip_special_tokens=False),
                              "extra_fields": extra}
                    current_records.append(record)

            try:
                for mode, generations in ((("greedy", 1),) if self.config.rlvr.get("benchmark_greedy_repeat", False) else (("greedy", 1), ("sampled_g8", 8))):
                    current_records = []
                    self.agent_loop_manager = EvalManager(mode, generations)
                    self.replay_buffer.validation_callback = capture
                    kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
                    kwargs.n, kwargs.do_sample = generations, mode != "greedy"
                    kwargs.temperature = 0.0 if mode == "greedy" else float(self.config.rlvr.eval.temperature)
                    kwargs.top_p = 1.0 if mode == "greedy" else float(self.config.rlvr.eval.top_p)
                    kwargs.top_k = -1 if mode == "greedy" else int(self.config.rlvr.eval.top_k)
                    parent_metrics = super()._validate()
                    contract = verify_eval_contract(current_records, samples, generations, int(self.config.rlvr.base_seed))
                    mode_metrics = summarise_records(current_records, num_generations=generations)
                    from statistics import mean
                    grouped_samples = {}
                    for record in current_records:
                        grouped_samples.setdefault(record["sample_id"], []).append(record)
                    per_sample = [{"sample_id": sid, "mode": mode, "domain": rows[0]["domain"],
                                   "difficulty_bucket": rows[0].get("difficulty_bucket"),
                                   "correct_count": sum(int(row["correctness"]) for row in rows),
                                   "generations": len(rows), "accuracy": mean(float(row["correctness"]) for row in rows)}
                                  for sid, rows in grouped_samples.items()]
                    domain_accuracy = {domain: mean(row["accuracy"] for row in per_sample if row["domain"] == domain)
                                       for domain in ("knowledge", "math")
                                       if any(row["domain"] == domain for row in per_sample)}
                    metric_name = "Sampled-Acc@8" if generations == 8 else "Greedy-Acc"
                    mode_metrics.update({f"{domain}/{metric_name}": value for domain, value in domain_accuracy.items()})
                    mode_metrics[f"equal_weight/{metric_name}"] = mean(domain_accuracy.values()) if len(domain_accuracy) == 2 else None
                    append_jsonl_batch(directory / f"{mode}-per_sample.jsonl", per_sample)
                    with (directory / f"{mode}-per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
                        writer.writeheader()
                        writer.writerows(per_sample)
                    atomic_json(directory / f"{mode}-contract.json", contract)
                    path = directory / f"{mode}.jsonl"
                    if path.exists():
                        raise RuntimeError("Evaluation output already exists; frozen results cannot be overwritten")
                    append_jsonl_batch(path, current_records)
                    suite_records[mode] = current_records
                    prompt_hashes = {}
                    param_contracts = set()
                    for record in current_records:
                        prompt_hash = hashlib.sha256(json.dumps(record["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()
                        if record["sample_id"] in prompt_hashes and prompt_hashes[record["sample_id"]] != prompt_hash:
                            raise RuntimeError("One evaluation sample used inconsistent prompt tokens")
                        prompt_hashes[record["sample_id"]] = prompt_hash
                        params = {key: value for key, value in record["extra_fields"]["actual_sampling_params"].items() if key != "seed"}
                        param_contracts.add(json.dumps(params, sort_keys=True, separators=(",", ":")))
                    if len(param_contracts) != 1:
                        raise RuntimeError("Evaluation mode used inconsistent effective sampling parameters")
                    if prompt_hashes != expected_prompt_hashes:
                        raise RuntimeError("Observed evaluation prompt tokens differ from the contract saved before rollout")
                    if "prompt_token_hashes" in seed_manifest and seed_manifest["prompt_token_hashes"] != prompt_hashes:
                        raise RuntimeError("Greedy and sampled evaluations used different prompts")
                    seed_manifest["prompt_token_hashes"] = prompt_hashes
                    seed_manifest.setdefault("sampling_modes", {})[mode] = {
                        "generations": generations, "parameters": json.loads(next(iter(param_contracts))),
                        "effective_max_tokens": {record["sample_id"]: record["extra_fields"]["effective_max_tokens"]
                                                 for record in current_records}}
                    all_metrics.update({f"evaluation/{mode}/{key}": value for key, value in mode_metrics.items()
                                        if isinstance(value, (int, float))})
                    all_metrics.update({f"evaluation/{mode}/verl/{key}": value for key, value in parent_metrics.items()})
            finally:
                self.agent_loop_manager = original_manager
                self.replay_buffer.validation_expected = None
                self.replay_buffer.validation_callback = None
                for key, value in original_kwargs.items():
                    self.config.actor_rollout_ref.rollout.val_kwargs[key] = value
            seed_manifest["seed_map_sha256"] = hashlib.sha256(json.dumps(samples, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            seed_manifest["prompt_template_kwargs"] = OmegaConf.to_container(self.config.data.apply_chat_template_kwargs, resolve=True)
            atomic_json(self.run_dir / "evaluation_seed_manifest.json", seed_manifest, frozen=True)
            atomic_json(directory / "summary.json", {"checkpoint_step": checkpoint_step,
                        "sample_count": len(samples), "seed_manifest": seed_manifest, "metrics": all_metrics})
            self._latest_eval_records = suite_records
            return all_metrics

        def _fetch_one_gen_batch(self):
            if not 1 <= self.global_steps <= len(self.plan["updates"]):
                raise RuntimeError("Dataloader exhausted immutable plan; wraparound forbidden")
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            try:
                batch = next(self.train_dataloader_it)
            except StopIteration:
                if self.config.rlvr.strict_exposure and not self.config.rlvr.get('resume_smoke',False):
                    raise RuntimeError("Dataloader exhausted before manifest; wraparound forbidden") from None
                # Only the smoke contract permits deliberate reuse of its small input dataset.
                if str(self.config.rlvr.phase) != "smoke" and not self.config.rlvr.get('resume_smoke',False):
                    raise RuntimeError("Dataset reuse is permitted only in smoke") from None
                self.train_dataloader_it = iter(self.train_dataloader)
                batch = next(self.train_dataloader_it)
            samples = self.plan["updates"][self.global_steps - 1]["samples"]
            actual = batch.get("sample_id")
            if actual is None:
                actual = [row["sample_id"] for row in batch["extra_info"]]
            if list(map(str, actual)) != [str(s["sample_id"]) for s in samples]:
                raise RuntimeError("Dataset prompt order differs from frozen manifest")
            batch["uid"] = np.array([s["group_id"] for s in samples], dtype=object)
            batch["sample_id"] = np.array([s["sample_id"] for s in samples], dtype=object)
            batch["group_id"] = batch["uid"]
            batch["sequence_position"] = np.array([s["sequence_position"] for s in samples])
            seed_array = np.empty(len(samples), dtype=object)
            seed_array[:] = [list(s["seeds"]) for s in samples]
            batch["rlvr_seeds"] = seed_array
            batch["agent_name"] = np.array(["rlvr_single_turn"] * len(samples), dtype=object)
            # Reject per-row n overrides rather than allowing difficulty-based exposure changes.
            if "__rollout_n__" in batch and any(int(n) != self.config.rlvr.num_generations for n in batch["__rollout_n__"]):
                raise RuntimeError("Per-sample generation overrides are forbidden")
            return tu.get_tensordict(batch)

        def _submit_batch_to_rollout(self, batch):
            ledger = self.run_dir / "exposure.jsonl"
            if ledger.exists() and any(json.loads(line)["update_id"] == self.global_steps
                                       for line in ledger.read_text(encoding="utf-8").splitlines()):
                raise RuntimeError("The manifest batch was already exposed")
            append_jsonl(ledger, {"update_id": self.global_steps, "manifest_sha256": self.manifest_sha,
                                 "samples": self.plan["updates"][self.global_steps - 1]["samples"],
                                 "status": "dispatched"})
            from rlvr.profiling import stage_event
            self._rollout_started = stage_event("start", "rollout", self.run_dir,
                                                self.config.rlvr.phase, self.global_steps)
            return super()._submit_batch_to_rollout(batch)

        def on_sample_end(self):
            from rlvr.profiling import stage_event
            self._rollout_elapsed = (time.monotonic_ns() - self._rollout_started["monotonic_ns"]) / 1e9
            stage_event("end", "rollout", self.run_dir, self.config.rlvr.phase, self.global_steps,
                        duration_s=self._rollout_elapsed,
                        status="complete")
            super().on_sample_end()

        def step(self, metrics, timing_raw):
            from rlvr.profiling import profile_stage
            self._update_started_ns = time.monotonic_ns()
            with profile_stage("update", self.run_dir, self.config.rlvr.phase, self.global_steps):
                return super().step(metrics, timing_raw)

        def _compute_advantage(self,batch,metrics):
            return super()._compute_advantage(batch,metrics)

        def _compute_old_log_prob(self, batch, metrics):
            from rlvr.profiling import profile_stage
            with profile_stage("old_log_prob", self.run_dir, self.config.rlvr.phase, self.global_steps):
                result = super()._compute_old_log_prob(batch, metrics)
                verify_entropy_observation(metrics)
                return result

        def _compute_ref_log_prob(self, batch, metrics):
            from rlvr.profiling import profile_stage
            with profile_stage("ref_log_prob", self.run_dir, self.config.rlvr.phase, self.global_steps):
                return super()._compute_ref_log_prob(batch, metrics)

        def _balance_batch(self, batch, metrics, **kwargs):
            expected = self.config.rlvr.prompts_per_update * self.config.rlvr.num_generations
            if len(batch) != expected or any(tag.get("is_padding", False) for tag in batch.tags):
                raise RuntimeError("Effective batch incomplete or padded")
            group = self.actor_rollout_wg
            mapping = group._dispatch_info.get("actor")
            if mapping is None:
                mapping = group._query_dispatch_info("actor")
                group._dispatch_info["actor"] = mapping
            self._actor_dp_size = max(mapping) + 1
            if expected % self._get_required_batch_multiple(self._actor_dp_size):
                raise RuntimeError("Frozen effective batch incompatible with actor DP/minibatch")
            # Ordered contiguous split is independent of output/reward length.
            metrics["rlvr/padding_count"] = 0
            metrics["rlvr/effective_responses"] = expected
            return batch

        def _update_actor(self, batch, metrics):
            batch.extra_info["rlvr_update_id"] = self.global_steps
            directory = self.run_dir / "optimizer_events"
            before = optimizer_snapshot(directory)
            result = super()._update_actor(batch, metrics)
            world = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
            events = verify_optimizer_events(directory, before, world)
            self._optimizer_events = events
            metrics["rlvr/global_optimizer_updates"] = 1
            return result

        def _save_checkpoint(self):
            super()._save_checkpoint()
            path = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "rlvr_contract.json"
            temporary = path.with_suffix(".tmp")
            actor_path = path.parent / "actor"
            ranks = checkpoint_events(self.run_dir / "checkpoint_events", actor_path, "save_complete")
            expected_ranks = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
            if len(ranks) != expected_ranks:
                raise RuntimeError("Full checkpoint lacks per-rank actual state/inventory evidence")
            for row in ranks.values():
                optimizer = row["summary"]["optimizer"]
                if optimizer["step_min"] != self.global_steps or optimizer["step_max"] != self.global_steps:
                    raise RuntimeError("Checkpoint optimizer state is not at the saved update")
            inventory = [{"path": str(item.relative_to(path.parent)), "bytes": item.stat().st_size}
                         for item in sorted(path.parent.rglob("*")) if item.is_file() and item != path]
            data_path = path.parent / "data.pt"
            if not data_path.is_file():
                raise RuntimeError("Checkpoint dataloader state was not saved")
            temporary.write_text(json.dumps({"manifest_sha256": self.manifest_sha,
                "completed_update": self.global_steps, "rank_states": ranks,
                "data_progress": data_progress_summary(self.train_dataloader.state_dict()),
                "data_state_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "inventory": inventory, "inventory_bytes": sum(item["bytes"] for item in inventory)}), encoding="utf-8")
            os.replace(temporary, path)

        def _records_for_batch(self, batch):
            if self._last_records_step == self.global_steps:
                return self._records
            fields = ["sample_id", "session_id", "prompts", "responses", "response_mask",
                      "advantages", "rm_scores", "extra_fields", "reward_model"]
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            def rows(name):
                value = data[name]
                if hasattr(value, "tolist") and not getattr(value, "is_nested", False):
                    return value.tolist()
                return [value[i].tolist() if hasattr(value[i], "tolist") else value[i]
                        for i in range(len(batch))]
            columns = {name: rows(name) for name in fields}
            records = []
            for i, key in enumerate(batch.keys):
                extra = columns["extra_fields"][i]
                reward_info = extra.get("reward_extra_info")
                if not isinstance(reward_info, dict) or "correctness" not in reward_info:
                    raise RuntimeError("RewardManager must emit per-component reward_extra_info")
                record = {"update_id": self.global_steps, "trajectory_key": key,
                          "group_id": key.rsplit("_", 2)[0], "sample_id": str(columns["sample_id"][i]),
                          "generation_index": int(columns["session_id"][i]),
                          "response": self.tokenizer.decode(columns["responses"][i], skip_special_tokens=False),
                          "prompt_token_ids": columns["prompts"][i],
                          "response_token_ids": columns["responses"][i],
                          "response_mask": columns["response_mask"][i], "advantages": columns["advantages"][i],
                          "reward": sum(columns["rm_scores"][i]), "reward_extra_info": reward_info,
                          "extra_fields": extra, "reward_model": columns["reward_model"][i], **reward_info}
                sample = self.plan["updates"][self.global_steps - 1]["samples"][i // self.config.rlvr.num_generations]
                record.update(rollout_index=record["generation_index"], total_reward=record["reward"],
                              seed=int(extra["sampling_seed"]), sequence_position=sample["sequence_position"],
                              update_committed=True)
                if record["seed"] != sample["seeds"][record["rollout_index"]]:
                    raise RuntimeError("Actual rollout sampling seed differs from frozen manifest")
                records.append(record)
            self._records, self._last_records_step = records, self.global_steps
            return records

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
            from rlvr.metrics import summarise_records
            records = self._records_for_batch(batch)
            metrics.update({f"rlvr/{key}": value for key, value in summarise_records(records).items()
                            if isinstance(value, (int, float))})
            histogram = [0] * (self.config.rlvr.num_generations + 1)
            for start in range(0, len(records), self.config.rlvr.num_generations):
                group = records[start:start + self.config.rlvr.num_generations]
                histogram[sum(int(r["correctness"]) for r in group)] += 1
            if sum(histogram) != self.config.rlvr.prompts_per_update:
                raise RuntimeError("Incorrect GRPO group histogram denominator")
            metrics.update({f"rlvr/correct_count_{i}_groups": count for i, count in enumerate(histogram)})
            metrics.update({f"rlvr/correct_count_{i}_ratio": count / sum(histogram)
                            for i, count in enumerate(histogram)})
            append_jsonl(self.run_dir / "metrics.jsonl", {"update_id": global_steps,
                         "metrics": metrics, "timing": timing_raw, "correct_count_histogram": histogram})
            # Always dump raw components, even if upstream rollout_data_dir is unset.
            append_jsonl_batch(self.run_dir / "rollouts.jsonl", records)
            measurement = self.config.rlvr.get("measurement")
            if not measurement:
                warmup = self.config.rlvr.get("warmup_updates", self.config.rlvr.get("throughput", {}).get("warmup_steps", 0))
                measurement = "warmup" if global_steps <= warmup else "steady"
            if self.config.rlvr.get("direct_r0", False):
                from rlvr.direct_r0 import measurement_label
                measurement = measurement_label(global_steps)
            stage_seconds = {name: timing_raw.get(source) for name, source in (
                ("actor_update", "update_actor"), ("reward", "reward"),
                ("old_logprob", "old_log_prob"), ("ref_logprob", "ref"),
                ("advantage", "adv"), ("weight_sync", "update_weights"))}
            stage_seconds["rollout"] = self._rollout_elapsed
            intervals = sorted((row["extra_fields"]["reward_start_monotonic_ns"],
                                row["extra_fields"]["reward_end_monotonic_ns"]) for row in records)
            union = []
            for start, end in intervals:
                if union and start <= union[-1][1]:
                    union[-1] = (union[-1][0], max(end, union[-1][1]))
                else:
                    union.append((start, end))
            stage_seconds["reward"] = sum(end-start for start, end in union) / 1e9
            kv_samples = [row["extra_fields"]["kv_cache"] for row in records
                          if row["extra_fields"].get("kv_cache") and
                          row["extra_fields"]["kv_cache"]["observed_monotonic_ns"] >= self._rollout_started["monotonic_ns"]]
            kv_cache = max(kv_samples, key=lambda item: item["metrics"].get("kv_cache_usage_fraction", -1)) if kv_samples else None
            update_record = {"event": "update_complete", "phase": self.config.rlvr.phase,
                "measurement": measurement, "update_id": global_steps,
                "start_monotonic_ns": self._update_started_ns, "end_monotonic_ns": time.monotonic_ns(),
                "status": "complete", "prompt_count": self.config.rlvr.prompts_per_update,
                "response_count": len(records), "group_count": self.config.rlvr.prompts_per_update,
                "generation_count": self.config.rlvr.num_generations, "finite_update": True,
                "optimizer_steps_by_rank": {str(row["rank"]): row["successful_steps"] for row in self._optimizer_events},
                "generated_tokens": sum(sum(row["response_mask"]) for row in records), "oom": False,
                "stage_seconds": stage_seconds, "kv_cache": kv_cache,
                "manifest_sha256": self.manifest_sha, "optimizer_events": self._optimizer_events}
            from rlvr.metrics import group_metrics
            details = group_metrics(records)["groups"]
            metadata = {row["group_id"]: row for row in records}
            group_records = []
            joint_histograms = {}
            domain_histograms = {}
            bucket_histograms = {}
            for group in details:
                example = metadata[group["group_id"]]
                domain, bucket = example.get("domain"), example.get("difficulty_bucket")
                group_records.append({"update_id": global_steps, "domain": domain,
                                      "difficulty_bucket": bucket, "n_correct": group["correct_count"],
                                      "trajectory_keys": [row["trajectory_key"] for row in records if row["group_id"] == group["group_id"]],
                                      "raw_components_and_advantages": "rollouts.jsonl", **group})
                joint_histograms.setdefault((domain, bucket), [0]*len(histogram))[group["correct_count"]] += 1
                domain_histograms.setdefault(domain, [0]*len(histogram))[group["correct_count"]] += 1
                bucket_histograms.setdefault(bucket, [0]*len(histogram))[group["correct_count"]] += 1
            append_jsonl_batch(self.run_dir / "groups.jsonl", group_records)
            def dense(counts, **metadata):
                total = sum(counts)
                ratios = [count/total for count in counts]
                if abs(sum(ratios)-1) > 1e-12:
                    raise RuntimeError("Group histogram ratios do not sum to one")
                return {**metadata, "group_count": total, "counts": counts, "ratios": ratios}
            append_jsonl(self.run_dir / "group_histograms.jsonl", {"update_id": global_steps,
                         "correct_count_histogram": histogram,
                         "correct_count_ratios": [count/sum(histogram) for count in histogram],
                         "group_count": sum(histogram), "bins": list(range(len(histogram))),
                         "by_domain_bucket": [dense(counts, domain=domain, difficulty_bucket=bucket)
                                              for (domain, bucket), counts in joint_histograms.items()],
                         "by_domain": [dense(counts, domain=domain) for domain, counts in domain_histograms.items()],
                         "by_difficulty_bucket": [dense(counts, difficulty_bucket=bucket) for bucket, counts in bucket_histograms.items()]})
            update_record["end_monotonic_ns"] = time.monotonic_ns()
            append_jsonl(self.run_dir / "updates.jsonl", update_record)
            from rlvr.profiling import stage_event
            stage_event("update_complete", "update", self.run_dir, self.config.rlvr.phase, global_steps,
                        prompts=self.config.rlvr.prompts_per_update, responses=len(records),
                        output_tokens=sum(len(row["response_token_ids"]) for row in records),
                        optimizer_steps=1, optimizer_finite=True,
                        reward_profile=self.config.rlvr.reward_profile)
            if self.config.rlvr.get("direct_r0", False):
                from rlvr.direct_r0 import qualify_committed_prefix
                qualify_committed_prefix(self.run_dir, global_steps)

        def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
            # _compute_metrics already persisted full records once in manifest order.
            self._records_for_batch(batch)

    return RLVRTrainer


def build_task_runner():
    import ray

    class Runner:
        def run(self, config):
            from omegaconf import OmegaConf
            import transfer_queue as tq
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ
            from rlvr.rollout import install_extensions
            install_extensions()
            if os.environ.get("RLVR_RUN_DIR") != str(config.rlvr.run_dir):
                raise RuntimeError("Set RLVR_RUN_DIR in Ray runtime_env.env_vars before launching")
            OmegaConf.resolve(config)
            config.transfer_queue.enable = True
            tq.init(config.transfer_queue)
            trainer = None
            succeeded = False
            deferred_checkpoint = config.rlvr.phase == "reward_dev"
            if deferred_checkpoint:
                from rlvr.contracts import atomic_json
                checkpoint_policy = {"order": ["train", "exposure", "post_eval", "checkpoint"],
                                     "configured_native_save_freq": int(config.trainer.save_freq),
                                     "effective_native_save_freq": -1,
                                     "checkpoint_update": int(config.trainer.total_training_steps),
                                     "status": "planned"}
                # The native fit loop saves before post-eval. Keep that behavior
                # for smoke and defer R0's single full save until evaluation ends.
                config.trainer.save_freq = -1
                atomic_json(Path(config.rlvr.run_dir) / "final_checkpoint_policy.json", checkpoint_policy)
            try:
                trainer = build_trainer_class()(config)
                trainer.init()
                if config.rlvr.get('init_contract_only', False):
                    from rlvr.contracts import atomic_json
                    atomic_json(Path(config.rlvr.run_dir) / 'initialization_only.json',
                                {'passed': True, 'optimizer_updates': 0, 'rollouts': 0})
                    succeeded = True
                    return
                manager = AgentLoopManagerTQ.create(
                    config=config, llm_client=trainer.get_llm_client(),
                    teacher_client=trainer.get_teacher_client(),
                    reward_loop_worker_handles=trainer.get_reward_handles())
                trainer.fit(manager)
                if config.rlvr.get('resume_smoke',False):
                    from rlvr.contracts import atomic_json
                    path=Path(config.rlvr.run_dir)
                    check=json.loads((path/'resume-check-step50.json').read_text())
                    updates=[json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()]
                    continued=[row for row in updates if row['update_id']>50]
                    if len(continued)!=1 or continued[0]['update_id']!=51 or not continued[0]['finite_update']:
                        raise RuntimeError('Resume smoke requires exactly one valid continuation update')
                    save_completed_checkpoint_after_eval(trainer,51)
                    atomic_json(path/'resume_smoke_result.json',{'passed':True,'restored_step':50,
                        'continuation_update':51,'continuation_responses':256,'excluded_from_R0':True,
                        'restored_state':check})
                else:
                    finalize_exposure(config.rlvr.run_dir, config.rlvr.manifest_path,
                                      successful_updates=config.trainer.total_training_steps,
                                      allow_benchmark_prefix=config.rlvr.phase == 'benchmark')
                if config.rlvr.phase == "benchmark" and config.rlvr.get("benchmark_greedy_repeat", False):
                    import copy
                    original_loader = trainer.val_dataloader
                    frozen_batch = copy.deepcopy(next(iter(original_loader)))
                    try:
                        trainer.val_dataloader = [copy.deepcopy(frozen_batch)]
                        trainer._run_eval_suite("greedy-check-1", checkpoint_step=config.trainer.total_training_steps)
                        first = trainer._latest_eval_records["greedy"]
                        if config.rlvr.get('adapter_sync_audit', False):
                            from rlvr.adapter_audit import audit_state
                            trained_state = audit_state(require_trained=True)
                            from rlvr.contracts import atomic_json
                            atomic_json(Path(config.rlvr.run_dir) / 'trained-adapter-before-repeat.json', trained_state)
                            if not all(row['extra_fields'].get('native_batch_invariant') is True
                                       and row['extra_fields']['native_adapter_audit']['sha256'] == trained_state['sha256']
                                       for row in first):
                                raise RuntimeError('First evaluation did not use the audited trained adapter and invariant engine')
                        trainer.val_dataloader = [copy.deepcopy(frozen_batch)]
                        trainer._run_eval_suite("greedy-check-2", checkpoint_step=config.trainer.total_training_steps)
                        comparison = compare_repeat_evaluation(first, trainer._latest_eval_records["greedy"])
                        from rlvr.contracts import atomic_json
                        atomic_json(Path(config.rlvr.run_dir) / "greedy-repeat-check.json", comparison)
                        if not comparison["exact_token_seed_termination_reward_match"]:
                            raise RuntimeError("Selected throughput candidate failed fixed-environment Greedy repeat")
                        if config.rlvr.get('adapter_sync_audit', False):
                            comparisons = {'round2_vs_round1': comparison}
                            for round_id in range(2, int(config.rlvr.get('greedy_repeat_rounds', 3)) + 1):
                                if round_id > 2:
                                    trainer.val_dataloader = [copy.deepcopy(frozen_batch)]
                                    trainer._run_eval_suite(f'greedy-check-{round_id}', checkpoint_step=config.trainer.total_training_steps)
                                records = trainer._latest_eval_records['greedy']
                                state = audit_state(require_trained=True)
                                if state['sha256'] != trained_state['sha256']:
                                    raise RuntimeError('Trained adapter bytes changed between repeat rounds')
                                if not all(row['extra_fields'].get('native_batch_invariant') is True
                                           and row['extra_fields']['native_adapter_audit']['sha256'] == state['sha256']
                                           for row in records):
                                    raise RuntimeError('Evaluation request adapter bytes or execution mode mismatch')
                                current = compare_repeat_evaluation(first, records)
                                comparisons[f'round{round_id}_vs_round1'] = current
                                if not current['exact_token_seed_termination_reward_match']:
                                    raise RuntimeError('Trained adapter integration Greedy repeat failed')
                            atomic_json(Path(config.rlvr.run_dir) / 'trained-greedy-integration-check.json',
                                        {'passed': True, 'adapter_sha256': trained_state['sha256'],
                                         'rounds': int(config.rlvr.get('greedy_repeat_rounds', 3)),
                                         'comparisons': comparisons, 'scope': 'real verl post-update LoRA evaluation'})
                            if config.rlvr.get('integration_save_checkpoint', False):
                                save_completed_checkpoint_after_eval(trainer, config.trainer.total_training_steps)
                    finally:
                        trainer.val_dataloader = original_loader
                if config.rlvr.phase in ("smoke", "reward_dev"):
                    trainer._run_eval_suite("post", checkpoint_step=config.trainer.total_training_steps)
                    if config.rlvr.phase == "smoke":
                        first = trainer._latest_eval_records
                        trainer._run_eval_suite("smoke-repeat", checkpoint_step=config.trainer.total_training_steps)
                        from rlvr.contracts import atomic_json
                        comparison = {mode: compare_repeat_evaluation(first[mode], trainer._latest_eval_records[mode])
                                      for mode in first}
                        atomic_json(Path(config.rlvr.run_dir) / f"smoke-eval-repeat-step{config.trainer.total_training_steps}.json", comparison)
                        repeat_gate = repeat_comparison_gate(comparison)
                        atomic_json(Path(config.rlvr.run_dir) / f"smoke-eval-repeat-policy-step{config.trainer.total_training_steps}.json", repeat_gate)
                        if not repeat_gate['passed']:
                            raise RuntimeError(f"Smoke repeat contract failed: {repeat_gate['failures']}")
                if deferred_checkpoint:
                    from rlvr.profiling import profile_stage
                    step = int(config.trainer.total_training_steps)
                    post_summary = Path(config.rlvr.run_dir) / "evaluation" / f"post-step{step}" / "summary.json"
                    if not post_summary.is_file():
                        raise RuntimeError("Final checkpoint requires completed post evaluation")
                    with profile_stage("checkpoint_save", config.rlvr.run_dir, config.rlvr.phase, step):
                        save_completed_checkpoint_after_eval(trainer, step)
                    checkpoint_policy.update(status="complete", post_eval_summary_sha256=
                                             hashlib.sha256(post_summary.read_bytes()).hexdigest())
                    atomic_json(Path(config.rlvr.run_dir) / "final_checkpoint_policy.json", checkpoint_policy)
                succeeded = True
            finally:
                try:
                    if trainer is not None and getattr(trainer, "logger", None) is not None:
                        trainer.logger.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()

    return ray.remote(Runner)


def __getattr__(name):
    if name == "RLVRTaskRunner":
        value = build_task_runner()
    elif name == "RLVRTrainer":
        value = build_trainer_class()
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
