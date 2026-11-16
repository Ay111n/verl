"""Watch an authorized training arm, then merge, diagnose and evaluate its final model."""
import argparse,json,os,subprocess,time
from pathlib import Path
from paired_dpo import save

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--arm",required=True,choices=["base","true","shuffled"])
    p.add_argument("--train-pid",type=int,required=True);p.add_argument("--gpus",required=True);a=p.parse_args()
    gpus=a.gpus.split(",");assert len(gpus)==2
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
    scores=[]
    for rank,gpu in enumerate(gpus):
        completed=run/("post_"+a.arm)/f"complete.shard{rank}.json"
        if completed.exists():
            check=json.loads(completed.read_text())
            assert check["rows"]==316 and check["adapter"]==str(models/"adapter")
            continue
        scores.append(launch([py,"-u",str(code/"cache_reference.py"),"--data",str(root/"finetuning/rl/final_data"/name/"pairs.jsonl"),
            "--model",str(base),"--adapter",str(models/"adapter"),"--output",str(run/("post_"+a.arm)),
            "--shard",str(rank),"--shards","2"],run/(a.arm+f"_post{rank}.log"),gpu))
    merge=launch([py,"-u",str(root/"finetuning/scripts/model/merge_same_source_dpo_adapter.py"),
        "--base",str(base),"--adapter",str(models/"adapter"),"--output",str(models/"merged")],run/(a.arm+"_merge_streaming.log"),"")
    assert all(score.wait()==0 for score in scores),"Post scores failed"
    evalroot=root/"benchmark/eval/results"/name;evalroot.mkdir(parents=True,exist_ok=True)
    evaluation=launch([py,"-u",str(root/"benchmark/eval/scripts/evaluate_same_source_dpo_316.py"),
        "--model",str(base),"--adapter",str(models/"adapter"),"--arm",a.arm],evalroot/(a.arm+".log"),a.gpus)
    save(run/(a.arm+"_post_launch.json"),{"score_pids":[score.pid for score in scores],"evaluation_pid":evaluation.pid,"merged":str(models/"merged")})
    rc_eval=evaluation.wait()
    assert rc_eval==0,("Evaluation failure",rc_eval)
    assert merge.wait()==0,"Merge failed"
    save(run/(a.arm+"_all_complete.json"),{"training":True,"merge":True,"post_scores":True,"dev_evaluation":True})
if __name__=="__main__":main()
