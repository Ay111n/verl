"""Optional byte audits of the real FSDP -> native vLLM LoRA transfer."""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def tensor_digest(tensors):
    """Versioned, name-sorted digest; preserve BF16 bytes without float conversion."""
    import torch
    digest = hashlib.sha256(b'rlvr-adapter-bytes-v1\0')
    cpu = {}
    for name, tensor in tensors.items():
        value = tensor.detach().to('cpu').contiguous().clone()
        cpu[name] = value
    for name in sorted(cpu):
        value = cpu[name]
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    b = [value for name, value in cpu.items() if 'lora_B' in name]
    return {'digest_version': 'rlvr-adapter-bytes-v1', 'sha256': digest.hexdigest(),
            'tensor_count': len(cpu), 'numel': sum(value.numel() for value in cpu.values()),
            'b_nonzero': any(bool(value.ne(0).any()) for value in b)}, cpu


def record_transfer(role, tensors, peft_config=None):
    import torch
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    metadata, cpu = tensor_digest(tensors)
    if metadata['tensor_count'] != 504 or metadata['numel'] != 87293952:
        raise RuntimeError(f'Incomplete adapter payload at {role}: {metadata}')
    directory = Path(os.environ['RLVR_RUN_DIR']) / 'adapter_sync_audit'
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / f'{role}-rank{rank}-latest.json'
    index = json.loads(latest.read_text())['sync_index'] + 1 if latest.exists() else 0
    metadata.update(role=role, rank=rank, pid=os.getpid(), sync_index=index,
                    observed_unix_ns=time.time_ns())
    if role == 'sink' and rank == 0 and metadata['b_nonzero']:
        from safetensors.torch import save_file
        export = directory / ('trained-' + metadata['sha256'])
        export.mkdir(exist_ok=True)
        weight_file = export / 'native_adapter_payload.safetensors'
        if not weight_file.exists():
            temporary = export / 'native_adapter_payload.tmp'
            save_file(cpu, str(temporary)); temporary.replace(weight_file)
            (export / 'native_peft_config.json').write_text(json.dumps(peft_config, default=str, indent=2))
        metadata['export_path'] = str(weight_file)
        metadata['export_file_sha256'] = hashlib.sha256(weight_file.read_bytes()).hexdigest()
    from rlvr.contracts import atomic_json
    atomic_json(latest, metadata)
    with (directory / f'{role}-rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(metadata) + '\n'); stream.flush(); os.fsync(stream.fileno())
    return metadata


def audit_state(require_trained=False, include_source=True):
    directory = Path(os.environ['RLVR_RUN_DIR']) / 'adapter_sync_audit'
    states = []
    for role in (['source', 'sink'] if include_source else ['sink']):
        values = [json.loads(path.read_text()) for path in directory.glob(f'{role}-rank*-latest.json')]
        if len(values) != 2 or {value['rank'] for value in values} != {0, 1}:
            raise RuntimeError(f'Missing two-rank {role} adapter audit')
        states.extend(values)
    digests = {state['sha256'] for state in states}
    if len(digests) != 1 or (require_trained and not all(state['b_nonzero'] for state in states)):
        raise RuntimeError('Adapter rank/source/sink mismatch or fresh zero-B adapter after training')
    return {'sha256': next(iter(digests)), 'b_nonzero': all(state['b_nonzero'] for state in states),
            'states': states}


def install_source_audit():
    if os.environ.get('RLVR_ADAPTER_SYNC_AUDIT') != '1':
        return
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    original = FSDPEngine.get_per_tensor_param
    if getattr(original, '_rlvr_adapter_digest', False):
        return
    @functools.wraps(original)
    def audited(self, *args, **kwargs):
        weights, peft_config = original(self, *args, **kwargs)
        if peft_config is None:
            return weights, peft_config
        def iterator():
            captured = {}
            for name, tensor in weights:
                if 'lora_A' in name or 'lora_B' in name:
                    captured[name] = tensor.detach().to('cpu').contiguous().clone()
                yield name, tensor
            if captured:
                record_transfer('source', captured, peft_config)
        return iterator(), peft_config
    audited._rlvr_adapter_digest = True
    FSDPEngine.get_per_tensor_param = audited


def __getattr__(name):
    if name != 'AuditedColocateWorkerExtension':
        raise AttributeError(name)
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
    class AuditedColocateWorkerExtension(vLLMColocateWorkerExtension):
        def _update_weights(self, weights, peft_config, base_sync_done):
            # Preserve the upstream loader. Audit the actual IPC-received payload
            # only after successful native adapter loading; no parameter replacement.
            weights = list(weights)
            result = vLLMColocateWorkerExtension._update_weights(self, weights, peft_config, base_sync_done)
            if peft_config and base_sync_done:
                record_transfer('sink', dict(weights), peft_config)
            return result
    AuditedColocateWorkerExtension.__qualname__ = name
    globals()[name] = AuditedColocateWorkerExtension
    return AuditedColocateWorkerExtension
