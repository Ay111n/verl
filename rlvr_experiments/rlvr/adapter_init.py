"""Isolated adapter RNG and canonical full-parameter step-0 contract."""
import contextlib, hashlib, json, os, random
from pathlib import Path

@contextlib.contextmanager
def isolated_adapter_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

def canonical_adapter(module):
    import torch
    result = {}
    for name, p in sorted(module.named_parameters()):
        if '.lora_A.' not in name and '.lora_B.' not in name:
            continue
        if p.is_meta:
            raise RuntimeError('Canonical adapter must be materialized')
        t = p.detach()
        if hasattr(t, 'full_tensor'):
            # All ranks collectively reconstruct the same global tensor, never
            # hash unrelated local FSDP shards. NCCL collectives require CUDA.
            if t.device.type == 'cpu':
                t = t.to(torch.device('cuda', torch.cuda.current_device()))
            t = t.full_tensor()
        t = t.cpu().contiguous()
        key = name.replace('._fsdp_wrapped_module.', '.')
        result[key] = {'shape': list(t.shape), 'dtype': str(t.dtype),
                       'sha256': hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest(),
                       'zero': bool(t.eq(0).all())}
    if not result or not all(v['zero'] for n,v in result.items() if '.lora_B.' in n):
        raise RuntimeError('Missing adapter or fresh B is nonzero')
    body = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
    return {'parameters': result, 'sha256': hashlib.sha256(body).hexdigest()}

def enforce_step0(module):
    import torch
    from rlvr.contracts import atomic_json
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    contract = canonical_adapter(module)
    contract.update(version='rlvr-adapter-init-v1', adapter_seed=int(os.environ['RLVR_ADAPTER_SEED']),
                    stage='step0_materialized_global', global_step=0)
    if torch.distributed.is_initialized():
        digests = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(digests, contract['sha256'])
        if len(set(digests)) != 1:
            raise RuntimeError('Reconstructed global adapter differs across ranks')
    reference = Path(os.environ['RLVR_ADAPTER_INIT_REFERENCE'])
    if rank == 0 and not reference.exists():
        atomic_json(reference, contract, frozen=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    expected = json.loads(reference.read_text())
    if expected != contract:
        raise RuntimeError('Step-0 adapter differs from frozen initialization reference')
    atomic_json(Path(os.environ['RLVR_RUN_DIR']) / 'peft_audit' / f'step0-rank{rank}.json',
                {**contract, 'rank': rank, 'reference': str(reference), 'matches_reference': True}, frozen=True)
    print(f'RLVR STEP0 rank={rank} adapter_seed={contract["adapter_seed"]} sha256={contract["sha256"]}', flush=True)
