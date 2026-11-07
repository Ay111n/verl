"""Preview canonical commands; validation requires standalone input contracts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from rlvr.paths import load_paths
from rlvr.standalone_contract import readiness
def command(experiment,run_dir,gpus,profile=None):
    mapping={'R0':'R0'}
    if experiment not in mapping:raise ValueError('Unsupported public experiment')
    p=load_paths(profile);run=Path(run_dir).resolve()
    if run.exists() and any(run.iterdir()):raise ValueError('Use a new empty run directory')
    if len(set(gpus))!=2:raise ValueError('Two distinct GPUs required')
    c=p['contracts_dir'];q=lambda value:json.dumps(str(value).replace('\\','/'))
    args=[str(p['python'] or sys.executable),'-u','-m','rlvr.launch','--config',
          'reward_dev','--path-profile',p['profile'],
          '--run-dir',str(run),'--gpus',*map(str,gpus),
          'rlvr.reward_profile='+mapping[experiment],'+rlvr.experiment_id='+experiment,
          'rlvr.adapter_seed=20261008','rlvr.standalone_contract='+q(c/'execution/reference.json'),
          'rlvr.adapter_init_reference='+q(c/'initialization/step0-adapter-reference.json'),
          'rlvr.evaluation_reference='+q(c/'evaluation/reward_dev_submission.json'),
          '+rlvr.vllm_batch_invariant=1']
    return args
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--experiment',required=True,choices=['R0'])
    a.add_argument('--run-dir',required=True);a.add_argument('--gpus',type=int,nargs=2,default=[6,7])
    a.add_argument('--profile',choices=['local','server']);a.add_argument('--validate',action='store_true');a.add_argument('--execute',action='store_true')
    a.add_argument('--validate-data',action='store_true',help='Read-only frozen data and path checks, including retired historical inputs')
    v=a.parse_args();paths=load_paths(v.profile);os.environ['RLVR_PATH_PROFILE']=paths['profile']
    if v.validate_data and (v.validate or v.execute):a.error('--validate-data is separate from execution validation')
    args=command(v.experiment,v.run_dir,v.gpus,paths['profile']);status=readiness(paths)
    print(json.dumps({'experiment_id':v.experiment,'argv':args,'readiness':status,'starts_training':False},ensure_ascii=False,indent=2))
    if v.validate_data:
        from rlvr.launch import compose_config,validate_algorithm,verify_data
        cfg=compose_config('reward_dev',[])
        validate_algorithm(cfg)
        report=verify_data(cfg)
        report.update(evaluation_status=paths.get('evaluation_status','active'),read_only=True)
        print(json.dumps({'data_validation':report},ensure_ascii=False,indent=2))
        return 0
    if v.validate or v.execute:
        if not status['ready']:return 2
        from rlvr.launch import compose_config,validate_algorithm,verify_data
        from rlvr.standalone_contract import validate_execution
        cfg=compose_config('reward_dev',args[args.index('--gpus')+3:])
        validate_algorithm(cfg);verify_data(cfg);validate_execution(cfg)
    if v.execute:
        if os.name=='nt' or paths['python'] is None:raise ValueError('GPU execution needs the configured Linux inference environment')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT.parent)]),PYTHONHASHSEED='42',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        return subprocess.call(args,cwd=ROOT,env=env)
    return 0
if __name__=='__main__':raise SystemExit(main())
