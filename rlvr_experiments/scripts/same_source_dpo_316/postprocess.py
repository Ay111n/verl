"""Watch an authorized training arm, then merge, diagnose and evaluate its final model."""
import argparse,json,os,subprocess,time
from pathlib import Path
from paired_dpo import save

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--arm",required=True,choices=["base","true","shuffled"])
    p.add_argument("--train-pid",type=int,required=True);p.add_argument("--eval-gpu",required=True)
    p.add_argument("--score-gpu",required=True);a=p.parse_args()
    root=a.root;name="same_source_dpo_316_20261116";py="/opt/anaconda3/envs/inference/bin/python"
    run=root/"verl/rlvr_experiments/runs"/name;code=Path(__file__).parent
    models=root/"finetuning/model/rl"/name/a.arm;base=root/"finetuning/model/sft/en_cot_49800/merged/checkpoint-9000"
    deadline=time.time()+12*3600
    while not (run/a.arm/"completion.json").exists():
        assert time.time()<deadline,"Training timeout"
        stat=Path(f"/proc/{a.train_pid}/stat")
        if not stat.exists() or stat.read_text().split()[2]=="Z":raise RuntimeError("Training ended without a completion record")
        time.sleep(10)
    while Path(f"/proc/{a.train_pid}/cmdline").exists() and Path(f"/proc/{a.train_pid}/cmdline").read_bytes():time.sleep(2)
    info=json.loads((run/a.arm/"completion.json").read_text())
    assert info["pairs_seen"]==316 and info["optimizer_steps"]==40
    env=os.environ.copy();env["OMP_NUM_THREADS"]="4";env["TOKENIZERS_PARALLELISM"]="false"
    def launch(cmd,log,gpu):
        local=env.copy();local["CUDA_VISIBLE_DEVICES"]=gpu;local["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
        with log.open("w") as f:return subprocess.Popen(cmd,env=local,stdout=f,stderr=subprocess.STDOUT)
    score=launch([py,"-u",str(code/"cache_reference.py"),"--data",str(root/"finetuning/rl/final_data"/name/"pairs.jsonl"),
        "--model",str(base),"--adapter",str(models/"adapter"),"--output",str(run/("post_"+a.arm))],run/(a.arm+"_post.log"),a.score_gpu)
    merge=launch([py,"-u",str(root/"finetuning/scripts/model/merge_same_source_dpo_adapter.py"),
        "--base",str(base),"--adapter",str(models/"adapter"),"--output",str(models/"merged")],run/(a.arm+"_merge.log"),"")
    assert merge.wait()==0,"Merge failed"
    evalroot=root/"benchmark/eval/results"/name;evalroot.mkdir(parents=True,exist_ok=True)
    evaluation=launch([py,"-u",str(root/"benchmark/eval/scripts/evaluate_same_source_dpo_316.py"),
        "--model",str(models/"merged"),"--arm",a.arm],evalroot/(a.arm+".log"),a.eval_gpu)
    save(run/(a.arm+"_post_launch.json"),{"score_pid":score.pid,"evaluation_pid":evaluation.pid,"merged":str(models/"merged")})
    rc_score=score.wait();rc_eval=evaluation.wait()
    assert rc_score==0 and rc_eval==0,("Post-stage failure",rc_score,rc_eval)
    save(run/(a.arm+"_all_complete.json"),{"training":True,"merge":True,"post_scores":True,"dev_evaluation":True})
if __name__=="__main__":main()
