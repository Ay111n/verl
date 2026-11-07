import json
import pytest
from rlvr.contracts import build_train_plan
from rlvr.trainer import finalize_exposure

def prepare(tmp_path,phase='benchmark'):
    plan=build_train_plan([{'id':str(i)} for i in range(64)],train_updates=2)
    manifest=tmp_path/'plan.json';manifest.write_text(json.dumps(plan))
    (tmp_path/'resolved_config.json').write_text(json.dumps({'rlvr':{'phase':phase},'trainer':{'total_training_steps':1}}))
    (tmp_path/'updates.jsonl').write_text(json.dumps({'update_id':1})+'\n')
    records=[]
    for sample in plan['updates'][0]['samples']:
        for j,seed in enumerate(sample['seeds']):
            records.append({'update_id':1,'sample_id':sample['sample_id'],'group_id':sample['group_id'],
                'sequence_position':sample['sequence_position'],'rollout_index':j,'seed':seed,
                'update_committed':True,'domain':'math','difficulty_bucket':'B1'})
    (tmp_path/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return manifest

def test_explicit_benchmark_prefix_certifies_actual_records(tmp_path):
    manifest=prepare(tmp_path)
    r=finalize_exposure(tmp_path,manifest,1,allow_benchmark_prefix=True)
    assert r['partial_benchmark'] and not r['partial_smoke']
    assert r['committed_responses']==256 and r['unique_prompts']==32

def test_default_still_rejects_partial_reward_epoch(tmp_path):
    manifest=prepare(tmp_path)
    with pytest.raises(RuntimeError,match='Only smoke'):finalize_exposure(tmp_path,manifest,1)

def test_permission_cannot_shrink_reward_dev(tmp_path):
    manifest=prepare(tmp_path,'reward_dev')
    with pytest.raises(RuntimeError,match='permission'):finalize_exposure(tmp_path,manifest,1,allow_benchmark_prefix=True)
