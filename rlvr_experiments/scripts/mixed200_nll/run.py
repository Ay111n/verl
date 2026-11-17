"""Fresh S0 mixed200 NLL run; never starts full-pool sampling."""
import os,sys,time,json,subprocess,signal,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'verl/rlvr_experiments'
sys.path.insert(0,str(P/'scripts/luffy_math_pilot'))
from common import save,rpc
NAME='luffy_mixed200_nll_20261118';run=P/'runs'/NAME
def main():
 run.mkdir(parents=True,exist_ok=True)
 assert not (run/'launched.json').exists()
 cfg=json.loads((P/'configs'/f'{NAME}.json').read_text());start=time.time()
 cfg['train_deadline_unix']=start+86400
 save(run/'effective_training_config.json',cfg)
 save(run/'launched.json',{'start_unix':start,'pid':os.getpid(),'no_full_pool_sampling':True})
 owned=[]
 def launch(args,gpus,log):
  f=(run/log).open('w')
  proc=subprocess.Popen([sys.executable,'-u',*map(str,args)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn'),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  owned.append(proc.pid);save(run/'owned_processes.json',{'pids':owned,'max_concurrent_gpus':4});return proc
 def stage(value,**kw):save(run/'status.json',{'stage':value,'time_unix':time.time(),**kw})
 try:
  stage('loading_training_models')
  samplers=[]
  for rank,gpu in enumerate(cfg['rollout_gpus'],1):
   folder=run/'rpc_workers'/f'worker{rank}'
   proc=launch([P/'scripts/luffy_math_pilot/worker.py','--model',ROOT/cfg['model_path'],'--rpc',folder,'--max-num-seqs',16],str(gpu),f'worker{rank}.log')
   samplers.append((proc,folder,gpu))
  save(run/'workers.json',[{'pid':s.pid,'rpc':str(f),'gpu':g} for s,f,g in samplers])
  deadline=time.time()+1800
  while not all((f/'ready.json').exists() for _,f,_ in samplers):
   assert all(s.poll() is None for s,_,_ in samplers),'Sampler startup failed'
   assert time.time()<deadline;time.sleep(2)
  actor=launch([P/'scripts/luffy_math_pilot/train.py','--config',run/'effective_training_config.json'],str(cfg['actor_gpu']),'train.log')
  stage('training',pid=actor.pid)
  assert actor.wait()==0,'Actor failed'
  complete=json.loads((run/'complete.json').read_text());assert complete['steps']==13 and complete['question_visits']==200
  for s,f,_ in samplers:
   rpc(f,{'request_id':'shutdown-after-training','mode':'shutdown'},timeout=120)
   s.wait(timeout=120)
  stage('evaluating_full666DEV')
  evaluation=launch([ROOT/cfg['eval_entrypoint'],'--model',ROOT/cfg['model_path'],'--arm','nll','--adapter',complete['adapter'],'--run-name',NAME],','.join(map(str,cfg['eval_gpus'])),'eval.log')
  assert evaluation.wait()==0,'Evaluation failed'
  stage('evaluation_complete_analysis_pending')
  save(run/'pipeline_complete.json',{'start_unix':start,'end_unix':time.time(),'elapsed_seconds':time.time()-start,'eval_questions':666})
 except Exception:
  save(run/'failure.json',{'error':traceback.format_exc(),'time_unix':time.time()});raise
 finally:
  for pid in owned:
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass
if __name__=='__main__':main()
