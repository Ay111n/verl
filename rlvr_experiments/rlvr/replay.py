"""Manifest ordered, complete-group sampling for the pinned verl V1 API."""
from __future__ import annotations

import json
import time
from pathlib import Path


def load_plan(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    updates = data.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ValueError("Manifest must contain a nonempty updates list")
    for index, update in enumerate(updates, 1):
        if update.get("update_id") != index:
            raise ValueError("Manifest update_id must be consecutive, starting at 1")
        samples = update["samples"]
        ids = [s["group_id"] for s in samples]
        if len(ids) != len(set(ids)) or any("_" in uid for uid in ids):
            raise ValueError("Group IDs must be unique and contain no underscores")
        if any(len(s["seeds"]) != data.get("num_generations", 8) for s in samples):
            raise ValueError("Every manifest sample needs a seed for each generation")
    from rlvr.resume import extended_resume_plan
    return extended_resume_plan(data)


def ordered_trajectory_keys(samples, partition_snapshot, generations):
    expected = [f"{s['group_id']}_{generation}_0" for s in samples for generation in range(generations)]
    selected = {s["group_id"] for s in samples}
    actual = {key for key, tag in partition_snapshot.items()
              if not tag.get("is_prompt", False) and key.split("_", 1)[0] in selected}
    if actual != set(expected):
        raise RuntimeError(f"Incomplete/extra trajectories: missing={sorted(set(expected)-actual)}, "
                           f"extra={sorted(actual-set(expected))}")
    if any(partition_snapshot[key].get("is_padding", False) for key in expected):
        raise RuntimeError("Padding is forbidden by the exposure contract")
    return expected


def build_replay_class():
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
    from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

    class OrderedReplayBuffer(ReplayBuffer):
        def __init__(self, **kwargs):
            if kwargs["trainer_mode"] != "sync":
                raise ValueError("Strict RLVR supports synchronous training only")
            options = dict(kwargs.get("sampler_kwargs") or {})
            self.plan = load_plan(options["manifest_path"])
            self.generations = int(options.get("num_generations", 8))
            self.timeout_seconds = float(options.get("timeout_seconds", 3600))
            self.validation_expected = None
            self.validation_callback = None
            kwargs.update(filter_groups_metric=None, sync_refill_failed_groups=False)
            super().__init__(**kwargs)

        def sample(self, global_steps, partition_id="train", batch_size=None):
            if partition_id != "train":
                if self.validation_expected is None:
                    raise RuntimeError("Validation must install its frozen sample/seed contract")
                samples, generations = self.validation_expected
                if len(samples) != batch_size:
                    raise RuntimeError("Validation batch differs from submitted contract")
                expected = {s["group_id"] for s in samples}
                deadline = time.monotonic() + self.timeout_seconds
                while True:
                    self._sync_metadata_from_transfer_queue()
                    if expected & self.failure_keys[partition_id]:
                        raise RuntimeError("Evaluation rollout failed; retry forbidden")
                    if expected <= self.finished_keys[partition_id]:
                        snapshot = dict(self.partitions[partition_id])
                        keys = ordered_trajectory_keys(samples, snapshot, generations)
                        result = KVBatchMeta(partition_id=partition_id, keys=keys, tags=[snapshot[k] for k in keys])
                        if self.validation_callback:
                            self.validation_callback(result)
                        tq.kv_clear(partition_id=partition_id, keys=[s["group_id"] for s in samples])
                        return result, {}
                    if time.monotonic() > deadline:
                        raise TimeoutError("Evaluation batch timed out")
                    time.sleep(self.poll_interval)
            if not 1 <= global_steps <= len(self.plan["updates"]):
                raise RuntimeError("Training requested a batch outside the frozen manifest")
            samples = self.plan["updates"][global_steps - 1]["samples"]
            if batch_size != len(samples):
                raise RuntimeError("Effective prompt batch differs from manifest")
            expected = {s["group_id"] for s in samples}
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                self._sync_metadata_from_transfer_queue()
                seen = (self.pending_keys[partition_id] | self.running_keys[partition_id]
                        | self.finished_keys[partition_id] | self.failure_keys[partition_id])
                if seen - expected:
                    raise RuntimeError(f"Unscheduled prompts in transfer queue: {seen-expected}")
                failed = expected & self.failure_keys[partition_id]
                if failed:
                    raise RuntimeError(f"Rollout failure; retry/refill forbidden: {sorted(failed)}")
                if expected <= self.finished_keys[partition_id]:
                    snapshot = dict(self.partitions[partition_id])
                    keys = ordered_trajectory_keys(samples, snapshot, self.generations)
                    tags = [snapshot[key] for key in keys]
                    tq.kv_clear(partition_id=partition_id, keys=[s["group_id"] for s in samples])
                    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags), {}
                if time.monotonic() > deadline:
                    raise TimeoutError("Manifest batch timed out; no refill was dispatched")
                time.sleep(self.poll_interval)

    return OrderedReplayBuffer


def __getattr__(name):
    if name == "OrderedReplayBuffer":
        value = build_replay_class()
        globals()[name] = value
        return value
    raise AttributeError(name)
