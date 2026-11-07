"""Input contract checks, independent of deleted experimental outcomes."""
import json,os
from pathlib import Path
from rlvr.paths import load_paths
from rlvr.contracts import sha256_file
def readiness(paths=None):
    p=paths or load_paths();c=p['contracts_dir'];d=p['prepared_dir']
    if os.name=='nt' and p['profile']=='server':
        return {'schema':'rlvr-readiness-v1','profile':'server','ready':False,'missing':['Server profile preview only on Windows; remote filesystem was not inspected'],'starts_training':False}
    if p.get('prepared_eval_dir') is None or p.get('evaluation_source_root') is None:
        return {'schema':'rlvr-readiness-v1','profile':p['profile'],'ready':False,'missing':['Configure evaluation_source_root and prepared_eval_dir'],'starts_training':False}
    e=p['prepared_eval_dir']
    required=[c/'initialization/step0-adapter-reference.json',c/'evaluation/reward_dev_submission.json',
              c/'execution/reference.json',d/'manifest.json',d/'train_plan.json',
              d/'reward_dev_train.parquet',e/'reward_dev_eval.parquet']
    missing=[str(x) for x in required if not x.is_file()]
    if p.get('evaluation_status')=='retired':
        missing.append('Evaluation splits are retired; only historical --validate-data checks are enabled')
    from rlvr.contracts import FROZEN_HASHES
    for name in FROZEN_HASHES:
        source=p['evaluation_source_root'] if name in {'rl_validation.jsonl','reward_dev/reward_dev_eval.jsonl'} else p['source_root']
        if not (source/name).is_file(): missing.append(str(source/name))
    if p['model_path'] is None or not p['model_path'].is_dir():missing.append('model_path: '+str(p['model_path']))
    return {'schema':'rlvr-readiness-v1','profile':p['profile'],'ready':not missing,'missing':missing,
            'historical_contracts_are_not_new_hardware_acceptance':True,'starts_training':False}
def validate_execution(config):
    from omegaconf import OmegaConf
    path=Path(config.rlvr.standalone_contract)
    if not path.is_file():raise FileNotFoundError(f'Missing standalone execution contract: {path}')
    ref=json.loads(path.read_text(encoding='utf-8'))
    if ref.get('schema')!='rlvr-execution-v1' or ref.get('hardware_accepted') is not True:
        raise ValueError('Execution contract must contain actual hardware acceptance')
    frozen=Path(ref['resolved_config'])
    if sha256_file(frozen)!=ref['resolved_config_sha256']:raise ValueError('Execution reference hash mismatch')
    old=json.loads(frozen.read_text(encoding='utf-8'));current=OmegaConf.to_container(config,resolve=True)
    for key in ('actor_rollout_ref','data','algorithm'):
        if current[key]!=old[key]:raise ValueError(f'Execution contract drift: {key}')
    for key in ('base_seed','adapter_seed','train_updates','prompts_per_update','num_generations','eval','training_mode','strict_exposure','manifest_path'):
        if current['rlvr'].get(key)!=old['rlvr'].get(key):raise ValueError(f'Execution contract drift: rlvr.{key}')
    if sha256_file(config.rlvr.manifest_path)!=ref['train_manifest_sha256']:raise ValueError('Training plan hash mismatch')
    init=json.loads(Path(config.rlvr.adapter_init_reference).read_text(encoding='utf-8'))
    if init['sha256']!=ref['step0_reference_sha256']:raise ValueError('Step-zero reference drift')
    if sha256_file(config.rlvr.adapter_init_reference)!=ref['step0_file_sha256']:raise ValueError('Step-zero file drift')
    if sha256_file(config.rlvr.evaluation_reference)!=ref['evaluation_reference_sha256']:raise ValueError('Evaluation contract drift')
    import importlib.metadata as metadata
    for package,expected in ref['packages'].items():
        try:actual=metadata.version(package)
        except metadata.PackageNotFoundError:actual=None
        if actual!=expected:raise ValueError(f'Runtime package drift: {package}: {actual} != {expected}')
    return {'source_run':str(path),'source_contract':str(path),'reward_profile':config.rlvr.reward_profile,
            'algorithm_matches':True,'train_manifest_matches':True,'nonreward_settings_match':True,
            'step0_reference_sha256':init['sha256'],'train_manifest_sha256':ref['train_manifest_sha256']}
