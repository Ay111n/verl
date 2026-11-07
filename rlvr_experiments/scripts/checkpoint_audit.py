"""Inspect native complete checkpoints; cleanup is restricted to verified smoke."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlvr.contracts import atomic_json, sha256_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('--cleanup-smoke', action='store_true')
args = parser.parse_args()
directory = args.run_dir.resolve()
state = json.loads((directory / 'status.json').read_text())
exposure = json.loads((directory / 'exposure_summary.json').read_text())
if state['status'] != 'complete' or exposure.get('actual_matches_plan') is not True:
    raise ValueError('Checkpoint audit requires an actually completed audited run')
step = exposure['completed_updates']
checkpoint = directory / 'checkpoints' / f'global_step_{step}'
files = sorted(path for path in checkpoint.rglob('*') if path.is_file())
names = [str(path.relative_to(checkpoint)) for path in files]
for component in ('model_world_size_2_rank_', 'optim_world_size_2_rank_', 'extra_state_world_size_2_rank_'):
    for rank in (0, 1):
        if not any(component + str(rank) in name for name in names):
            raise ValueError(f'Missing full checkpoint component {component}{rank}')
if not any('data.pt' in name for name in names) or 'rlvr_contract.json' not in names:
    raise ValueError('Missing data progress or RLVR resume contract')
inventory = {'run_dir': str(directory), 'checkpoint_dir': str(checkpoint), 'step': step,
             'logical_bytes': sum(path.stat().st_size for path in files),
             'files': [{'path': str(path.relative_to(checkpoint)), 'bytes': path.stat().st_size} for path in files],
             'manifest_sha256': exposure['manifest_sha256'], 'cleanup': False}
if args.cleanup_smoke:
    if state['phase'] != 'smoke' or step != 3 or exposure.get('partial_smoke') is True:
        raise ValueError('Cleanup is allowed only after complete smoke save/resume')
    repeat = json.loads((directory / f'smoke-eval-repeat-step{step}.json').read_text())
    from rlvr.trainer import repeat_comparison_gate
    if not repeat_comparison_gate(repeat)['passed']:
        raise ValueError('Smoke repeat evaluation did not pass')
    # Resolve and explicitly constrain the recursive-delete target.
    target = (directory / 'checkpoints').resolve()
    if target.parent != directory or directory not in target.parents or target.is_symlink():
        raise ValueError('Checkpoint cleanup target escaped its run directory')
    inventory['cleanup'] = True
    atomic_json(directory / 'checkpoint_inventory.json', inventory)
    shutil.rmtree(target)
else:
    atomic_json(directory / 'checkpoint_inventory.json', inventory)
print(json.dumps({'step': step, 'logical_bytes': inventory['logical_bytes'], 'cleanup': inventory['cleanup']}))
