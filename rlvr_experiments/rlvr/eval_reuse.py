"""Reuse baseline generation metrics with explicit immutable provenance."""
import json,hashlib,shutil
from pathlib import Path
from rlvr.contracts import atomic_json

def reuse_pre_eval(source_run, target_run, submission_contract):
    source=Path(source_run);target=Path(target_run)
    if json.loads((source/'status.json').read_text())['status']!='complete':
        raise ValueError('Pre-eval source must be complete')
    source_contract=json.loads((source/'evaluation_submission_contract.json').read_text())
    if submission_contract != source_contract:
        raise ValueError('Cached pre-eval submission contract mismatch')
    src=source/'evaluation/pre-step0'
    summary=json.loads((src/'summary.json').read_text())
    if summary['checkpoint_step']!=0 or summary['sample_count']!=400:
        raise ValueError('Cached pre-eval must contain 400 step-zero samples')
    expected=[x['sample_id'] for x in submission_contract['samples']]
    provenance={}
    for mode,n in [('sampled_g8',8),('greedy',1)]:
        file=src/(mode+'-per_sample.jsonl')
        rows=[json.loads(line) for line in file.open()]
        if [x['sample_id'] for x in rows]!=expected or any(x['generations']!=n for x in rows):
            raise ValueError('Cached question order or generation counts differ')
        provenance[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    # Reward summaries are run-specific and are excluded from this shared generation baseline.
    metrics={k:v for k,v in summary['metrics'].items() if not any(
        x in k for x in ('/component_', '/total_reward/', '/verl/', '/grpo/'))}
    dest=target/'evaluation/pre-step0';dest.mkdir(parents=True,exist_ok=True)
    for mode in ['sampled_g8','greedy']:
        for suffix in ['-per_sample.jsonl','-per_sample.csv','-contract.json']:
            file=src/(mode+suffix)
            if file.exists():shutil.copy2(file,dest/file.name)
    reused={**summary,'metrics':metrics,'generation_results_reused':True,'source_run':str(source),
            'reward_specific_metrics_omitted':True}
    atomic_json(dest/'summary.json',reused)
    atomic_json(target/'evaluation_seed_manifest.json',summary['seed_manifest'],frozen=True)
    atomic_json(target/'pre_eval_reuse.json',{'source_run':str(source),'source_directory':str(src),
        'source_summary_sha256':hashlib.sha256((src/'summary.json').read_bytes()).hexdigest(),
        'per_sample_sha256':provenance,'submission_contract_matches':True,
        'new_generation_requests':0,'reward_specific_metrics_omitted':True,
        'raw_generation_paths':{m:str(src/(m+'.jsonl')) for m in ['sampled_g8','greedy']}})
    return metrics
