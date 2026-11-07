import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlvr import launch
from rlvr.paths import load_paths
from rlvr.standalone_contract import readiness


def test_retirement_is_metadata(tmp_path):
    config = tmp_path / 'paths.json'
    config.write_text(json.dumps({'schema': 'rlvr-paths-v1', 'profiles': {
        'local': {'evaluation_status': 'retired', 'evaluation_source_root': str(tmp_path)}}}))
    paths = load_paths('local', config)
    assert paths['evaluation_status'] == 'retired'
    assert paths['evaluation_source_root'] == tmp_path


def test_retired_inputs_stop_before_execution(monkeypatch):
    monkeypatch.setattr(launch, 'load_paths', lambda: {'evaluation_status': 'retired'})
    with pytest.raises(ValueError, match='retired'):
        launch.run(SimpleNamespace())


def test_readiness_rejects_retired_inputs_even_when_files_exist(tmp_path):
    from rlvr.contracts import FROZEN_HASHES
    paths = {'profile': 'local', 'evaluation_status': 'retired'}
    for name in ('contracts_dir', 'prepared_dir', 'prepared_eval_dir', 'source_root', 'evaluation_source_root', 'model_path'):
        paths[name] = tmp_path / name
        paths[name].mkdir()
    for name in ('initialization/step0-adapter-reference.json', 'evaluation/reward_dev_submission.json', 'execution/reference.json'):
        p = paths['contracts_dir'] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    for name in ('manifest.json', 'train_plan.json', 'reward_dev_train.parquet'):
        (paths['prepared_dir'] / name).touch()
    (paths['prepared_eval_dir'] / 'reward_dev_eval.parquet').touch()
    for name in FROZEN_HASHES:
        owner = 'evaluation_source_root' if name in {'rl_validation.jsonl', 'reward_dev/reward_dev_eval.jsonl'} else 'source_root'
        p = paths[owner] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    status = readiness(paths)
    assert not status['ready']
    assert len(status['missing']) == 1 and 'retired' in status['missing'][0]


@pytest.mark.parametrize('frozen_root', ['/old/prepared/', 'E:\\old\\prepared\\'])
def test_relocated_data_uses_configured_roots_and_preserves_manifest(tmp_path, frozen_root):
    def write(folder, name, data):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    source = tmp_path / 'training_source'
    evaluation_source = tmp_path / 'evaluation_source'
    prepared = tmp_path / 'training_prepared'
    evaluation = tmp_path / 'evaluation_prepared'
    _, train_sha = write(source, 'rl_train_final.jsonl', b'train\n')
    eval_source, eval_sha = write(evaluation_source, 'rl_validation.jsonl', b'eval\n')
    train, train_parquet_sha = write(prepared, 'reward_dev_train.parquet', b'training parquet fixture')
    val, eval_parquet_sha = write(evaluation, 'reward_dev_eval.parquet', b'evaluation parquet fixture')
    plan, plan_sha = write(prepared, 'train_plan.json', b'{"updates": []}\n')
    doc = {'source_files': {'rl_train_final.jsonl': train_sha, 'rl_validation.jsonl': eval_sha},
           'roles': {'reward_dev_train': {'path': frozen_root + train.name, 'sha256': train_parquet_sha},
                     'reward_dev_eval': {'path': frozen_root + val.name, 'sha256': eval_parquet_sha}},
           'plans': {'reward_dev': {'path': frozen_root + plan.name, 'sha256': plan_sha}}}
    manifest, manifest_sha = write(prepared, 'manifest.json', json.dumps(doc).encode())
    config = SimpleNamespace(rlvr=SimpleNamespace(data_manifest_path=str(manifest), source_root=str(source),
                             evaluation_source_root=str(evaluation_source), prepared_dir=str(prepared),
                             prepared_eval_dir=str(evaluation), manifest_path=str(plan)),
                             data=SimpleNamespace(train_files=[str(train)], val_files=[str(val)]))
    report = launch.verify_data(config)
    assert report['source_files'] == 2 and report['prepared_files'] == 2
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_sha
    eval_source.write_bytes(b'changed evaluation input')
    with pytest.raises(ValueError, match='Frozen source changed: rl_validation'):
        launch.verify_data(config)


def test_retired_preparation_is_blocked(monkeypatch, tmp_path):
    import sys
    monkeypatch.setattr(launch, 'load_paths', lambda: {'evaluation_status': 'retired', 'model_path': None,
                        'source_root': tmp_path, 'prepared_dir': tmp_path})
    monkeypatch.setattr(sys, 'argv', ['launch', '--prepare-data'])
    with pytest.raises(ValueError, match='read-only validation'):
        launch.main()
