"""Run matched arms and heldout evaluations on at most physical GPUs 0 and 2."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import paths, save, rpc, sha

def main():
    p=paths()
    project=Path(__file__).resolve().parents[2]
    run=p["run_root"]
    py=str(p["python"])
    cfg=project/"configs/opd_math_ab_20261116.json"
    smoke=json.loads((run/"smoke/summary.json").read_text())
    assert smoke["passed"]
    assert smoke["config"]==json.loads(cfg.read_text())
    assert not (run/"pipeline_complete.json").exists()
    processes={}
    records=[]
    def launch(name, command, gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
        log=(run/(name+".log")).open("a")
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=project)
        log.close()
        processes[name]=child
        records.append({"name":name,"pid":child.pid,"gpu":gpu,"started_unix":time.time(),"command":command})
        save(run/"pipeline_processes.json",records)
        return child

    def state(phase):
        alive={k:v.poll() for k,v in processes.items()}
        save(run/"pipeline_state.json",{"phase":phase,"updated_unix":time.time(),"processes":alive,
             "allowed_physical_gpus":[0,2],"pipeline_pid":os.getpid()})

    def wait(child, name):
        while child.poll() is None:
            state(name)
            # Preserve per-task device accounting without listing other users' commands.
            raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"],text=True)
            selected=[line for line in raw.splitlines() if int(line.split(",")[0]) in (0,2)]
            with (run/"gpu_telemetry.jsonl").open("a") as f:
                f.write(json.dumps({"time":time.time(),"phase":name,"configured_task_gpus":[0,2],"devices":selected})+"\n")
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"{name} failed with exit {child.returncode}; see {name}.log")

    evaluation=p["benchmark_root"]/"eval/scripts/opd_math_ab/evaluate.py"
    def eval_command(name, rpc_root=None):
        command=[py,"-u",str(evaluation),"--experiment",str(project),"--model",name]
        if rpc_root:command+=["--rpc",str(rpc_root)]
        return command

    save(run/"formal_protocol.json",{
        "config":json.loads(cfg.read_text()),"config_sha256":sha(cfg),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=project.parent,text=True).strip(),
        "train_sha256":sha(p["train_data"]),"eval_sha256":sha(p["eval_data"]),
        "smoke_sha256":sha(run/"smoke/summary.json"),
        "worker_gpu":2,"trainer_gpu":0,"max_concurrent_gpus":2,
        "evaluation_overlap":"S0 evaluation may use the idle rollout worker during trainer backward passes; final A/B evaluation starts after training. After both arms finish, GPU0 may host a second evaluation worker.",
        "selection":"Fixed final checkpoint after one epoch, no checkpoint selection on holdout."
    })
    baseline=launch("eval-s0",eval_command("s0"),"")
    for arm in ("answer","solution"):
        child=launch("train-"+arm,[py,"-u",str(project/"scripts/opd_math_ab/train.py"),
                     "--mode","train","--arm",arm,"--config",str(cfg)],"0")
        wait(child,"train-"+arm)
    answer_eval=launch("eval-answer",eval_command("answer"),"")
    # Trainer has exited before GPU0 is reused; no additional devices are introduced.
    rpc0=run/"rpc_eval_gpu0"
    second=launch("worker-eval-gpu0",[py,"-u",str(project/"scripts/opd_math_ab/worker.py"),
                                    "--rpc",str(rpc0),"--max-num-seqs","16"],"0")
    solution_eval=launch("eval-solution",eval_command("solution",rpc0),"")
    for label,child in [("eval-s0",baseline),("eval-answer",answer_eval),("eval-solution",solution_eval)]:
        wait(child,label)
    rpc(run/"rpc",{"request_id":"zz-final-shutdown","mode":"shutdown"})
    rpc(rpc0,{"request_id":"zz-final-shutdown","mode":"shutdown"})
    wait(second,"worker-eval-gpu0")
    state("complete")
    save(run/"pipeline_complete.json",{"completed_unix":time.time(),"max_configured_gpus":2,
          "training_examples_per_arm":342,"updates_per_arm":43,"evaluation_questions_per_model":512})

if __name__=="__main__":
    try:
        main()
    except Exception:
        p=paths()
        save(p["run_root"]/"pipeline_failure.json",{"error":traceback.format_exc(),"time":time.time()})
        raise
