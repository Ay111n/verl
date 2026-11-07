import importlib.util,json,pathlib
import pytest
script=pathlib.Path(__file__).resolve().parents[1]/'scripts/comparison_statistics.py'
spec=importlib.util.spec_from_file_location('fair_stats',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fake_run(root,offset):
    p=root/'evaluation/post-step50';p.mkdir(parents=True)
    (p/'summary.json').write_text(json.dumps({'seed_manifest':{'seed':42}}))
    for mode,n in [('greedy',1),('sampled_g8',8)]:
        data=[{'sample_id':str(i),'domain':'knowledge' if i<200 else 'math','accuracy':.5+offset,
               'generations':n} for i in range(400)]
        (p/f'{mode}-per_sample.jsonl').write_text('\n'.join(json.dumps(x) for x in data))

def test_bootstrap_is_paired_stratified_and_reproducible(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';fake_run(a,0);fake_run(b,.125)
    result=module.paired_statistics(a,b,replicates=500)
    assert result==module.paired_statistics(a,b,replicates=500)
    for mode in result['statistics'].values():
        for stat in mode.values():
            assert stat['delta']==.125 and stat['lower']==.125 and stat['upper']==.125

def test_changed_eval_contract_is_rejected(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';fake_run(a,0);fake_run(b,0)
    (b/'evaluation/post-step50/summary.json').write_text(json.dumps({'seed_manifest':{'seed':99}}))
    with pytest.raises(ValueError):module.paired_statistics(a,b,replicates=10)
