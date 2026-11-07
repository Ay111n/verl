"""Audited entry point shared by smoke, benchmark and reward experiments."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

for offline_flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[offline_flag] = '1'

from rlvr.contracts import atomic_json, formal_budget_guard, sha256_file

ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = ROOT.parent
from rlvr.paths import load_paths
MODEL = load_paths()['model_path']
EXPECTED_COMMIT = '8718ca30a3f002f93b7c4fd99b9b2506718681bc'


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256_file(path)
            for folder in ('rlvr', 'configs', 'scripts', 'tests')
            for path in sorted((ROOT / folder).rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.yaml', '.sh')}


def verify_model(source_root, output):
    if MODEL is None or not MODEL.is_dir():raise FileNotFoundError("Configure a real merged model path before model verification")
    contract = json.loads((Path(source_root) / 'reward_dev/reward_ablation_contract.json').read_text())
    frozen = contract['fixed_rollout_contract']['model']
    if Path(frozen['path']) != MODEL:
        raise ValueError('Unexpected Mixed-SFT checkpoint path')
    verified = {}
    for filename, expected in frozen['files'].items():
        path = MODEL / filename
        actual = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError(f'Model fingerprint mismatch: {filename}')
        verified[filename] = actual
        print(f'verified {filename}', flush=True)
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards or not shards.issubset(verified):
        raise ValueError('Not every indexed model shard was verified')
    atomic_json(output, {'model_path': str(MODEL), 'files': verified,
                         'indexed_shards': sorted(shards), 'status': 'verified'})


def environment_manifest():
    packages = {}
    for name in ('torch', 'vllm', 'transformers', 'ray', 'TransferQueue', 'hydra-core', 'tensordict', 'flash-attn'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.check_output(['git', '-C', str(VERL_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f'Unvalidated verl commit {commit}')
    return {'python': platform.python_version(), 'executable': sys.executable,
            'verl_commit': commit, 'packages': packages, 'custom_source_hashes': source_hashes()}


def compose_config(name, overrides):
    paths=load_paths()
    os.environ['RLVR_PROJECT_DIR']=ROOT.as_posix()
    os.environ['RLVR_SOURCE_ROOT']=paths['source_root'].as_posix()
    for key in ('prepared_dir', 'prepared_eval_dir'):
        if paths.get(key) is None: raise ValueError(f'Configure {key} for this machine profile')
        os.environ['RLVR_' + key.upper()] = str(paths[key])
    os.environ['RLVR_MODEL_PATH']=str(paths['model_path'] or '__MODEL_NOT_CONFIGURED__')
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name=name, overrides=[
            f'hydra.searchpath=[file://{VERL_ROOT}/verl/trainer/config]', *overrides])


def validate_algorithm(config):
    if config.rlvr.reward_profile != 'R0':
        raise ValueError('This engineering revision supports outcome-only R0')
    from omegaconf import OmegaConf
    frozen = {
        'data.train_batch_size': 32, 'data.gen_batch_size': 32,
        'data.max_response_length': 4096, 'data.shuffle': False,
        'actor_rollout_ref.rollout.n': 8,
        'actor_rollout_ref.rollout.response_length': 4096,
        'actor_rollout_ref.rollout.temperature': 0.6,
        'actor_rollout_ref.rollout.top_p': 0.95, 'actor_rollout_ref.rollout.top_k': 20,
        'actor_rollout_ref.actor.strategy': 'fsdp2',
        'actor_rollout_ref.actor.ppo_mini_batch_size': 32,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.shuffle': False,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.clip_ratio': 0.2,
        'actor_rollout_ref.actor.entropy_coeff': 0.,
        'actor_rollout_ref.actor.use_kl_loss': True,
        'actor_rollout_ref.actor.kl_loss_coef': 0.001,
        'algorithm.adv_estimator': 'grpo',
        'algorithm.norm_adv_by_std_in_grpo': True,
        'algorithm.use_kl_in_reward': False,
        'trainer.n_gpus_per_node': 2, 'trainer.nnodes': 1,
        'rlvr.training_mode': 'lora',
        'actor_rollout_ref.model.lora_rank': 32,
        'actor_rollout_ref.model.lora_alpha': 64,
        'actor_rollout_ref.model.target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                 'gate_proj', 'up_proj', 'down_proj'],
        'actor_rollout_ref.model.lora_adapter_path': None,
        'actor_rollout_ref.model.lora.merge': False,
        'actor_rollout_ref.actor.checkpoint.save_lora_only': True,
        'actor_rollout_ref.rollout.load_format': 'safetensors',
        'actor_rollout_ref.rollout.layered_summon': True,
    }
    for key, expected in frozen.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f'Frozen algorithm violation: {key}')
    formal_budget_guard(config.rlvr.phase, config.rlvr.train_updates)
    if any(dict(config.rlvr.component_flags).values()):
        raise ValueError('Additional reward components are disabled')
    if str(config.rlvr.phase) == 'formal_16k':
        raise ValueError('Formal 16k training is outside the authorized first round')
    return frozen


def verify_data(config):
    manifest = json.loads(Path(config.rlvr.data_manifest_path).read_text())
    for filename, expected in manifest['source_files'].items():
        source_root=Path(config.rlvr.source_root)
        if sha256_file(source_root / filename) != expected:
            raise ValueError(f'Frozen source changed: {filename}')
    by_path = {str(Path(value['path']).resolve()): value for value in manifest['roles'].values()}
    for file in [*config.data.train_files, *config.data.val_files]:
        path = str(Path(file).resolve())
        if path not in by_path or sha256_file(path) != by_path[path]['sha256']:
            raise ValueError(f'Prepared data differs from its verified manifest: {path}')
    plans = {str(Path(value['path']).resolve()): value for value in manifest['plans'].values()}
    path = str(Path(config.rlvr.manifest_path).resolve())
    if path not in plans or sha256_file(path) != plans[path]['sha256']:
        raise ValueError('Training plan differs from frozen data manifest')


def run(args):
    paths=load_paths()
    expected_python=paths['python']
    if expected_python is None or Path(sys.executable).resolve() != expected_python.resolve():
        raise ValueError('Use the selected inference conda interpreter for experiment execution')
    from omegaconf import OmegaConf, open_dict
    run_dir = Path(args.run_dir).resolve()
    config = compose_config(args.config, args.overrides)
    direct_r0 = getattr(args, 'direct_r0', False)
    if direct_r0:
        from rlvr.direct_r0 import POLICY, validate_direct_start
        validate_direct_start(config, args.resume_from, args.hardware_gate)
        if args.diagnostic_updates and args.diagnostic_updates != POLICY['diagnostic_updates']:
            raise ValueError('Direct R0 profiles only updates 2, 3, 4')
        args.diagnostic_updates = list(POLICY['diagnostic_updates'])
    with open_dict(config):
        config.rlvr.run_dir = str(run_dir)
        if direct_r0:
            config.rlvr.direct_r0 = True
            config.rlvr.diagnostic_updates = args.diagnostic_updates
        config.actor_rollout_ref.model.external_lib = 'rlvr.rollout'
        config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(ROOT / 'configs/agent_loops.yaml')
        config.actor_rollout_ref.rollout.agent.default_agent_loop = 'rlvr_single_turn'
        config.trainer.use_v1 = True
        config.trainer.v1.trainer_mode = 'sync'
        config.trainer.resume_mode = 'resume_path' if args.resume_from else 'disable'
        if args.resume_from:
            config.trainer.resume_from_path = str(Path(args.resume_from).resolve())
        config.ray_kwargs.ray_init.num_cpus = min(64, os.cpu_count() or 24)
        config.ray_kwargs.ray_init.num_gpus = 2
        config.ray_kwargs.ray_init.object_store_memory = 8 * 1024**3
        # Ray's UNIX socket paths must remain short. Never attach to an existing cluster.
        config.ray_kwargs.ray_init._temp_dir = '/tmp/rlvr-ray-' + run_dir.name[-20:]
        config.ray_kwargs.ray_init.runtime_env.py_executable = sys.executable
        config.ray_kwargs.ray_init.runtime_env.env_vars = {
            'RLVR_RUN_DIR': str(run_dir), 'RLVR_PHASE': str(config.rlvr.phase),
            'RLVR_ADAPTER_SEED': str(int(config.rlvr.adapter_seed)),
            'RLVR_ADAPTER_INIT_REFERENCE': str(config.rlvr.adapter_init_reference),
            'PYTHONHASHSEED': '42',
            'RLVR_DIAGNOSTIC_UPDATES': json.dumps(args.diagnostic_updates),
            'RLVR_RESOLVED_CONFIG_PATH': str(run_dir / 'resolved_config.json'),
            'PYTHONPATH': os.pathsep.join([str(ROOT), str(VERL_ROOT)]),
            'CUDA_HOME': '/usr/local/cuda-12.8',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'HF_HOME': str(ROOT / '.cache/huggingface'),
            'HF_MODULES_CACHE': str(ROOT / '.cache/huggingface/modules'),
            'HF_DATASETS_CACHE': str(ROOT / '.cache/huggingface/datasets'),
            'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
            'VLLM_BATCH_INVARIANT': str(int(config.rlvr.get('vllm_batch_invariant', 1))),
            'RLVR_ADAPTER_SYNC_AUDIT': str(int(bool(config.rlvr.get('adapter_sync_audit', False)))),
        }
    frozen_algorithm = validate_algorithm(config)
    OmegaConf.resolve(config)
    if args.dry_run:
        print(OmegaConf.to_yaml(config), flush=True)
        return
    standalone=config.rlvr.get('standalone_contract')
    if standalone:
        from rlvr.standalone_contract import validate_execution
        standalone_validation=validate_execution(config)
    if config.rlvr.phase == 'reward_dev' and not direct_r0 and not standalone:
        if not args.hardware_gate:
            raise ValueError('R0 requires a measured, accepted hardware gate')
        gate = json.loads(Path(args.hardware_gate).read_text())
        if gate.get('decision') != 'suitable_for_first_round' or gate.get('r0_authorized') is not True:
            raise ValueError('Hardware gate does not authorize R0')
        measured = json.loads((Path(gate['run_dir']) / 'resolved_config.json').read_text())
        if measured['actor_rollout_ref'] != OmegaConf.to_container(config.actor_rollout_ref, resolve=True):
            raise ValueError('R0 execution parameters differ from the measured accepted hardware candidate')
        if json.loads((Path(gate['run_dir']) / 'frozen_algorithm.json').read_text()) != frozen_algorithm:
            raise ValueError('R0 algorithm differs from measured hardware candidate')
    if run_dir.exists() and not args.resume_from and any(run_dir.iterdir()):
        raise ValueError('Use a new run directory; failed run records cannot be overwritten')
    run_dir.mkdir(parents=True, exist_ok=True)
    verify_data(config)
    model_manifest = ROOT / 'data/model_verified.json'
    if not model_manifest.is_file() or json.loads(model_manifest.read_text()).get('status') != 'verified':
        raise ValueError('Run --verify-model before any GPU execution')
    # Recheck sizes/mtime against the recorded snapshot; full shard hashing was performed above.
    verified_model = json.loads(model_manifest.read_text())
    for filename, record in verified_model['files'].items():
        if (MODEL / filename).stat().st_size != record['bytes']:
            raise ValueError(f'Model changed after fingerprint verification: {filename}')
    environment = environment_manifest()
    if args.resume_from and json.loads((run_dir / 'environment.json').read_text()) != environment:
        raise ValueError('Resume runtime or extension source differs from its checkpoint run')
    atomic_json(run_dir / 'environment.json', environment, frozen=bool(args.resume_from))
    if standalone:
        atomic_json(run_dir / 'standalone_contract_validation.json', standalone_validation, frozen=True)
    atomic_json(run_dir / 'frozen_algorithm.json', frozen_algorithm, frozen=True)
    if direct_r0:
        atomic_json(run_dir / 'direct_r0_policy.json', POLICY, frozen=True)
    atomic_json(run_dir / ('invocation_resume.json' if args.resume_from else 'invocation.json'), {'argv': sys.argv, 'command': shlex.join(sys.argv),
                'resume_from': args.resume_from, 'model_manifest_sha256': sha256_file(model_manifest),
                'hardware_gate_sha256': sha256_file(args.hardware_gate) if args.hardware_gate else None,
                'direct_r0': direct_r0})
    (run_dir / ('resolved_resume.yaml' if args.resume_from else 'resolved.yaml')).write_text(OmegaConf.to_yaml(config))
    atomic_json(run_dir / 'resolved_config.json', OmegaConf.to_container(config, resolve=True))
    status = {'phase': str(config.rlvr.phase), 'status': 'running', 'started_unix_ns': time.time_ns()}
    atomic_json(run_dir / 'status.json', status)
    from rlvr.hardware import GpuLease, GpuMonitor, select_idle_gpu_pair
    pair = args.gpus or [gpu.index for gpu in select_idle_gpu_pair()]
    try:
        with GpuLease(pair):
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, pair))
            for key, value in config.ray_kwargs.ray_init.runtime_env.env_vars.items():
                os.environ[key] = str(value)
            with open_dict(config):
                config.ray_kwargs.ray_init.runtime_env.env_vars['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            atomic_json(run_dir / 'gpu_selection.json', {'physical_gpu_ids': pair})
            with GpuMonitor(run_dir / 'gpu_samples.jsonl', pair):
                from verl.trainer.main_ppo import run_ppo
                from rlvr.trainer import RLVRTaskRunner
                run_ppo(config, task_runner_class=RLVRTaskRunner)
        status.update(status='complete', finished_unix_ns=time.time_ns())
    except BaseException as exc:
        status.update(status='failed', finished_unix_ns=time.time_ns(),
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if 'ray' in sys.modules and sys.modules['ray'].is_initialized():
            sys.modules['ray'].shutdown()
        atomic_json(run_dir / 'status.json', status)
        from rlvr.hardware import cleanup_run_processes
        cleanup = cleanup_run_processes(run_dir, status['status'])
        cleanup['timestamp_unix_ns'] = time.time_ns()
        with (run_dir / 'process_cleanup.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(cleanup) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-data', action='store_true')
    parser.add_argument('--verify-model', action='store_true')
    parser.add_argument('--path-profile', choices=['local','server'])
    parser.add_argument('--source-root')
    parser.add_argument('--prepared-dir')
    parser.add_argument('--evaluation-source-root')
    parser.add_argument('--prepared-eval-dir')
    parser.add_argument('--config', default='smoke', choices=['smoke', 'reward_dev', 'formal_16k'])
    parser.add_argument('--run-dir', default=str(ROOT / 'runs/unset'))
    parser.add_argument('--resume-from')
    parser.add_argument('--hardware-gate')
    parser.add_argument('--direct-r0', action='store_true',
                        help='Fresh 50-update R0; qualify hardware on its first 14 committed updates')
    parser.add_argument('--gpus', type=int, nargs=2)
    parser.add_argument('--diagnostic-updates', type=int, nargs='*', default=[])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    if args.path_profile:os.environ['RLVR_PATH_PROFILE']=args.path_profile
    global MODEL
    paths=load_paths();MODEL=paths['model_path']
    args.source_root=args.source_root or str(paths['source_root'])
    args.prepared_dir=args.prepared_dir or str(paths['prepared_dir'])
    args.evaluation_source_root=args.evaluation_source_root or paths.get('evaluation_source_root')
    args.prepared_eval_dir=args.prepared_eval_dir or paths.get('prepared_eval_dir')
    if args.prepare_data:
        from rlvr.data import prepare_data
        print(json.dumps(prepare_data(args.source_root, args.prepared_dir, evaluation_source_root=args.evaluation_source_root, evaluation_output_root=args.prepared_eval_dir), ensure_ascii=False, default=str))
    elif args.verify_model:
        verify_model(args.source_root, ROOT / 'data/model_verified.json')
    else:
        run(args)


if __name__ == '__main__':
    main()
