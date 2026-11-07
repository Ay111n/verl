"""Read-only CPU-side validation of the selected runtime and extension interfaces."""
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for offline_flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[offline_flag] = '1'
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_MODULES_CACHE'] = str(ROOT / '.cache/huggingface/modules')
os.environ['HF_DATASETS_CACHE'] = str(ROOT / '.cache/huggingface/datasets')
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from rlvr.contracts import atomic_json

expected = {'torch': '2.10.0+cu128', 'vllm': '0.19.1', 'transformers': '5.12.1',
            'TransferQueue': '0.1.10', 'ray': '2.55.1'}
actual = {key: metadata.version(key) for key in expected}
if actual != expected or sys.version_info[:2] != (3, 11):
    raise RuntimeError(f'Candidate version mismatch: {actual} / {sys.version}')
modules = ['torch', 'vllm', 'transformers', 'transfer_queue', 'ray',
           'verl.trainer.main_ppo', 'verl.trainer.ppo.v1.trainer_sync',
           'verl.workers.engine.fsdp.transformer_impl']
for name in modules:
    importlib.import_module(name)
    print(f'imported {name}', flush=True)
from rlvr.launch import compose_config, validate_algorithm
from rlvr.rollout import install_extensions, build_replica_class, build_agent_class
from rlvr.trainer import build_trainer_class
from omegaconf import OmegaConf
install_extensions()
build_replica_class()
build_agent_class()
import ray
from rlvr.rollout import ObservedVLLMServer
ray.cloudpickle.dumps(ray.remote(ObservedVLLMServer).__ray_metadata__.modified_class)
print('serialized modified vLLM actor class', flush=True)
build_trainer_class()
for name in ('smoke', 'reward_dev'):
    config = compose_config(name, [])
    validate_algorithm(config)
    OmegaConf.resolve(config)
    print(f'composed {name}', flush=True)
formal = compose_config('formal_16k', [])
try:
    validate_algorithm(formal)
except ValueError:
    pass
else:
    raise RuntimeError('Unset formal budget was accepted')
atomic_json(ROOT / 'env/runtime_check.json', {'status': 'cpu_interfaces_verified',
            'python': sys.version, 'packages': actual, 'modules': modules,
            'formal_unset_budget_rejected': True})
