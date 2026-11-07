"""Add training dependencies while constraining every existing conda package."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = '/opt/anaconda3/envs/inference/bin/python'
if Path(sys.executable).resolve() != Path(TARGET).resolve():
    raise ValueError('Run this script with the explicitly selected inference interpreter')
before = {d.metadata['Name']: d.version for d in metadata.distributions()}
expected = {'torch': '2.10.0+cu128', 'vllm': '0.19.1', 'transformers': '5.12.1'}
if any(metadata.version(name) != version for name, version in expected.items()):
    raise ValueError('The existing inference backend differs from the frozen model contract')
snapshot = ROOT / 'env/inference.before.json'
if not snapshot.exists():
    snapshot.write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
constraints = ROOT / 'env/inference.preserve.constraints.txt'
allowed_adjustments = {'fsspec': '2026.2.0', 'packaging': '25.0'}
constraints.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(before.items())
                               if name.lower() not in allowed_adjustments))
uv = '/home/nlp/miniconda3/bin/uv'
environment = dict(os.environ, UV_CACHE_DIR=str(ROOT / '.cache/uv'))
subprocess.run([uv, '--no-config', 'pip', 'install', '--python', TARGET, '--torch-backend', 'cu128',
                '-c', str(constraints), '-r', str(ROOT / 'env/requirements-inference.in')],
               check=True, env=environment)
after = {d.metadata['Name']: d.version for d in metadata.distributions()}
changed = {name: [version, after.get(name)] for name, version in before.items() if after.get(name) != version}
unexpected = {name: values for name, values in changed.items()
              if allowed_adjustments.get(name.lower()) != values[1]}
if unexpected:
    raise RuntimeError(f'Existing package versions changed unexpectedly: {unexpected}')
(ROOT / 'env/inference.additions.json').write_text(json.dumps({
    'interpreter': TARGET, 'python': sys.version, 'model_backend_versions_preserved': True,
    'compatibility_adjustments': changed,
    'added': {name: version for name, version in after.items() if name not in before}}, indent=2) + '\n')
with (ROOT / 'env/requirements.inference.resolved.txt').open('w') as handle:
    subprocess.run([uv, '--no-config', 'pip', 'freeze', '--python', TARGET], check=True, stdout=handle)
subprocess.run([uv, '--no-config', 'pip', 'check', '--python', TARGET], check=True)
