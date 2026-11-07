"""Observe and enforce native LoRA training; no replacement engine implementation."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def trainability_summary(module):
    trainable, frozen, total = [], 0, 0
    for name, param in module.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            if '.lora_A.' not in name and '.lora_B.' not in name:
                raise RuntimeError(f'LoRA run has a trainable base parameter: {name}')
            trainable.append({'name': name, 'numel': count, 'dtype': str(param.dtype)})
        else:
            frozen += count
    if not trainable or not frozen:
        raise RuntimeError('LoRA requires trainable adapters and a frozen base model')
    observed = {name for name in TARGETS if any(f'.{name}.' in row['name'] for row in trainable)}
    if observed != set(TARGETS):
        raise RuntimeError(f'LoRA target coverage differs from frozen contract: {observed}')
    return {'training_mode': 'lora', 'base_frozen': True,
            'trainable_parameters': sum(row['numel'] for row in trainable),
            'frozen_parameters': frozen, 'total_parameters': total,
            'trainable_tensors': len(trainable), 'target_modules': list(TARGETS),
            'trainable_parameter_metadata': trainable}


def verify_adapter_optimizer(module, optimizer):
    """Check real gradients and Adam state cannot update frozen base weights."""
    params = list(module.named_parameters())
    frozen = [(name, p) for name, p in params if not p.requires_grad]
    adapters = [(name, p) for name, p in params if p.requires_grad]
    if not frozen or not adapters:
        raise RuntimeError('LoRA training lost frozen base or trainable adapters')
    if any('.lora_A.' not in name and '.lora_B.' not in name for name, p in adapters):
        raise RuntimeError('A non-adapter parameter became trainable')
    if any(p.grad is not None or optimizer.state.get(p) for name, p in frozen):
        raise RuntimeError('Frozen base parameter has a gradient or optimizer state')
    state_ids = {id(p) for p in optimizer.state}
    if not state_ids.issubset({id(p) for name, p in adapters}):
        raise RuntimeError('Optimizer state contains a non-adapter parameter')
    return {'base_gradient_and_optimizer_state_absent': True,
            'optimizer_state_adapter_only': True, 'adapter_optimizer_state_tensors': len(state_ids)}


def adapter_B_zero_observation(module):
    tensors = [p for name, p in module.named_parameters() if '.lora_B.' in name]
    if not tensors:
        raise RuntimeError('LoRA B parameters are missing')
    if any(p.is_meta for p in tensors):
        return None
    return all(bool((p.to_local() if hasattr(p, 'to_local') else p).detach().eq(0).all()) for p in tensors)


def record_materialized_lora(engine):
    from rlvr.adapter_init import enforce_step0
    enforce_step0(engine.module)
    summary = trainability_summary(engine.module)
    if adapter_B_zero_observation(engine.module) is not True:
        raise RuntimeError('Materialized fresh LoRA B weights are missing, meta or nonzero')
    summary.update(verify_adapter_optimizer(engine.module, engine.optimizer))
    import torch
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    summary.update(rank=rank, stage='materialized', lora_rank=32, lora_alpha=64, fresh_lora_B_zero=True)
    from rlvr.contracts import atomic_json
    path = Path(os.environ['RLVR_RUN_DIR']) / 'peft_audit' / f'rank{rank}-pid{os.getpid()}-materialized.json'
    atomic_json(path, summary, frozen=True)
    print(f"RLVR materialized LoRA rank={rank}: {summary['trainable_parameters']} trainable; "
          f"{summary['frozen_parameters']} frozen; fresh B=0; adapter-only optimizer", flush=True)


def install_lora_audit_hook(fsdp_module):
    original = fsdp_module.FSDPEngine._build_lora_module
    if getattr(original, '_rlvr_lora_audit', False):
        return

    @functools.wraps(original)
    def audited(self, module):
        from rlvr.adapter_init import isolated_adapter_rng
        with isolated_adapter_rng(int(os.environ['RLVR_ADAPTER_SEED'])):
            result = original(self, module)
        summary = trainability_summary(result)
        peft = result.peft_config['default']
        if peft.r != 32 or peft.lora_alpha != 64 or peft.bias != 'none' or peft.lora_dropout != 0:
            raise RuntimeError('Native PEFT settings differ from frozen LoRA configuration')
        b_zero = adapter_B_zero_observation(result)
        if b_zero is False:
            raise RuntimeError('Fresh LoRA B weights must initialize to zero')
        import torch
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        summary.update(rank=rank, lora_rank=peft.r, lora_alpha=peft.lora_alpha,
                       lora_dropout=peft.lora_dropout, fresh_lora_B_zero=b_zero,
                       stage='before_fsdp', numerical_check_deferred=b_zero is None)
        from rlvr.contracts import atomic_json
        path = Path(os.environ['RLVR_RUN_DIR']) / 'peft_audit' / f'rank{rank}-pid{os.getpid()}.json'
        atomic_json(path, summary, frozen=True)
        print(f"RLVR LoRA audit rank={rank}: {summary['trainable_parameters']} trainable; "
              f"{summary['frozen_parameters']} frozen; rank=32 alpha=64", flush=True)
        return result

    audited._rlvr_lora_audit = True
    fsdp_module.FSDPEngine._build_lora_module = audited
    build_original = fsdp_module.FSDPEngine._build_model_optimizer

    @functools.wraps(build_original)
    def materialized(self):
        result = build_original(self)
        if self._is_lora and self.optimizer is not None:
            record_materialized_lora(self)
        return result

    fsdp_module.FSDPEngine._build_model_optimizer = materialized
