"""Instrument successful optimizer calls without copying an engine implementation."""
from __future__ import annotations

import functools
import json
import os
import time
import math
import hashlib
from collections import Counter
from pathlib import Path

_installed = False


def optimizer_audit_metrics(successful_steps, duration_seconds):
    """Use the per-rank list format consumed by verl's DP metric gather.

    TrainingWorker flattens gathered lists once. Scalars inserted here become
    a list of scalars after the gather and cannot be flattened by that consumer.
    Actual step counts remain audited separately in optimizer_events.
    """
    return {"rlvr_optimizer_successful_steps": [int(successful_steps)],
            "rlvr_train_batch_seconds": [float(duration_seconds)]}


def summarize_optimizer_state(optimizer):
    steps = []
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            steps.append(int(value.item() if hasattr(value, "item") else value))
    return {"state_entries": len(optimizer.state), "step_entries": len(steps),
            "step_min": min(steps) if steps else None, "step_max": max(steps) if steps else None,
            "step_distribution": dict(Counter(map(str, steps)))}


def summarize_checkpoint_state(engine):
    rng = engine.checkpoint_manager.get_rng_state()
    digests = {}
    for name, value in rng.items():
        if hasattr(value, "cpu"):
            raw = value.cpu().numpy().tobytes()
        elif name == "numpy":
            raw = json.dumps([part.tolist() if hasattr(part, "tolist") else part for part in value],
                             separators=(",", ":")).encode()
        else:
            raw = repr(value).encode()
        digests[name] = hashlib.sha256(raw).hexdigest()
    scheduler = engine.lr_scheduler.state_dict() if engine.lr_scheduler is not None else {}
    return {"optimizer": summarize_optimizer_state(engine.optimizer),
            "scheduler_last_epoch": scheduler.get("last_epoch"),
            "scheduler_step_count": scheduler.get("_step_count"),
            "scheduler_last_lr": scheduler.get("_last_lr"), "rng_sha256": digests}


def record_optimizer_batch(path, *, rank, attempted, successful, grad_norm, **extra):
    finite = math.isfinite(float(grad_norm))
    record = {"rank": int(rank), "attempted_steps": int(attempted),
              "successful_steps": int(successful), "grad_norm": float(grad_norm) if finite else None,
              "grad_norm_finite": finite,
              "timestamp": time.time(), "pid": os.getpid(), **extra}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def install_worker_hooks():
    global _installed
    if _installed:
        return
    from verl.workers.engine.base import BaseEngine
    original = BaseEngine.train_batch

    @functools.wraps(original)
    def instrumented(self, data, loss_function):
        run_dir = os.environ.get("RLVR_RUN_DIR")
        if not run_dir:
            raise RuntimeError("RLVR_RUN_DIR must be propagated to all Ray training workers")
        counter = [0]
        if not hasattr(self.optimizer, "register_step_post_hook"):
            raise RuntimeError("Optimizer lacks register_step_post_hook; actual step cannot be audited")
        hook = self.optimizer.register_step_post_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
        optimizer_step = self.optimizer_step
        observed_norm = []
        def capture_norm():
            norm = optimizer_step()
            observed_norm.append(norm)
            return norm
        self.optimizer_step = capture_norm
        started = time.monotonic()
        try:
            result = original(self, data, loss_function)
        finally:
            hook.remove()
            self.optimizer_step = optimizer_step
        import torch
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if len(observed_norm) != 1:
            raise RuntimeError("Expected exactly one optimizer_step gradient-norm result")
        norm = observed_norm[0]
        if hasattr(norm, "item"):
            norm = norm.item()
        from verl.utils import tensordict_utils as tu
        update_id = int(tu.get(data, "rlvr_update_id", default=0))
        optimizer_state = summarize_optimizer_state(self.optimizer)
        peft_audit = {}
        if getattr(self, '_is_lora', False):
            from rlvr.peft_contract import verify_adapter_optimizer
            peft_audit = verify_adapter_optimizer(self.module, self.optimizer)
        record_optimizer_batch(Path(run_dir) / "optimizer_events" / f"rank{rank}-pid{os.getpid()}.jsonl",
                               rank=rank, attempted=1, successful=counter[0], grad_norm=norm,
                               update_id=update_id, optimizer=optimizer_state, **peft_audit)
        result.setdefault("metrics", {}).update(
            optimizer_audit_metrics(counter[0], time.monotonic() - started))
        if counter[0] != 1 or not math.isfinite(float(norm)):
            raise RuntimeError(f"Expected one successful optimizer.step per update, got {counter[0]}")
        if optimizer_state["step_min"] != update_id or optimizer_state["step_max"] != update_id:
            raise RuntimeError("Actual optimizer state step does not equal committed update_id")
        return result

    BaseEngine.train_batch = instrumented
    # The outer worker method includes train_mode entry/exit offloads. The engine
    # method above only audits actual optimizer calls inside that full phase.
    from verl.workers.engine_workers import TrainingWorker
    worker_original = TrainingWorker.train_mini_batch

    @functools.wraps(worker_original)
    def profiled_update(self, data):
        import torch
        from verl.utils import tensordict_utils as tu
        from rlvr.profiling import profile_stage, diagnostic_profile
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        update_id = int(tu.get(data, "rlvr_update_id", default=os.environ.get("RLVR_UPDATE_ID", "0")))
        if update_id <= 0:
            raise RuntimeError("Actor update has no explicit RLVR update_id")
        phase = os.environ.get("RLVR_PHASE", "unknown")
        run_dir = os.environ["RLVR_RUN_DIR"]
        diagnostic_updates = json.loads(os.environ.get("RLVR_DIAGNOSTIC_UPDATES", "[]"))
        with diagnostic_profile(run_dir, update_id, rank, enabled=update_id in diagnostic_updates):
            with profile_stage("actor_update", run_dir, phase, update_id, rank=rank):
                return worker_original(self, data)

    TrainingWorker.train_mini_batch = profiled_update
    import verl.workers.engine.fsdp.transformer_impl as fsdp_module
    from rlvr.peft_contract import install_lora_audit_hook
    install_lora_audit_hook(fsdp_module)
    for method_name, event in (("save_checkpoint", "save_complete"), ("load_checkpoint", "load_complete")):
        checkpoint_method = getattr(fsdp_module.FSDPEngine, method_name)
        def checkpoint_traced(self, *args, _original=checkpoint_method, _event=event, **kwargs):
            result = _original(self, *args, **kwargs)
            if self.optimizer is not None:
                import torch
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                local_path = str(kwargs.get("local_path", args[0] if args else ""))
                summary = summarize_checkpoint_state(self)
                load_audit = None
                if _event == 'load_complete' and os.environ.get('RLVR_RESUME_SMOKE') == '1':
                    from rlvr.resume import audit_loaded_tensors
                    load_audit = audit_loaded_tensors(self,local_path,rank)
                inventory = [{"path": str(path.relative_to(local_path)), "bytes": path.stat().st_size}
                             for path in sorted(Path(local_path).rglob("*")) if path.is_file()]
                record = {"event": _event, "rank": rank, "pid": os.getpid(),
                          "checkpoint_path": local_path, "timestamp_ns": time.time_ns(),
                          "summary": summary, "inventory": inventory,
                          "load_audit": load_audit,
                          "inventory_bytes": sum(item["bytes"] for item in inventory)}
                path = Path(os.environ["RLVR_RUN_DIR"]) / "checkpoint_events" / f"rank{rank}-pid{os.getpid()}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            return result
        setattr(fsdp_module.FSDPEngine, method_name, functools.wraps(checkpoint_method)(checkpoint_traced))
    # Wrap the imported aliases used by FSDP itself, not merely the defining utils
    # module: this observes model and optimizer copies in train-mode transitions.
    for name in ("load_fsdp_model_to_gpu", "load_fsdp_optimizer",
                 "offload_fsdp_model_to_cpu", "offload_fsdp_optimizer"):
        method = getattr(fsdp_module, name)
        def traced(*args, _method=method, _name=name, **kwargs):
            from rlvr.profiling import offload_range
            with offload_range(_name):
                return _method(*args, **kwargs)
        setattr(fsdp_module, name, functools.wraps(method)(traced))
    _installed = True
