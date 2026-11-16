"""Run only accepted A/B/C training and only final step-75 matched 666DEV."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import configuration, paths, save, sha, object_sha, rpc

def main():
    p=paths();project=p["project_dir"];run=p["run_root"];py=str(p["python"])
    cfg=configuration(p)
    calibration=json.loads((run/"calibration.json").read_text())
    smoke=json.loads((run/"smoke/summary.json").read_text())
    assert json.loads((run/"cpu_acceptance.json").read_text())["passed"]
    assert calibration["passed"] and smoke["passed"]
    assert smoke["config"]=={**cfg,"nll_alpha":calibration["alpha"],"calibration_sha256":sha(run/"calibration.json")}
    assert not (run/"pipeline_complete.json").exists()
    lock=run/"pipeline.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    process_records=[]
    def launch(label,command,gpus):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",
                 VLLM_WORKER_MULTIPROC_METHOD="spawn",PYTHONPATH=str(project)+":"+str(project.parent))
        with (run/(label+".log")).open("a") as f:
            child=subprocess.Popen(command,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT)
        process_records.append({"label":label,"pid":child.pid,"command":command,"gpus":gpus,"started_unix":time.time()})
        save(run/"pipeline_processes.json",process_records)
        while child.poll() is None:
            save(run/"pipeline_state.json",{"phase":label,"pid":child.pid,"controller_pid":os.getpid(),"updated_unix":time.time()})
            raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
            telemetry=[line for line in raw.splitlines() if int(line.split(",")[0]) in (3,4,5)]
            with (run/"gpu_telemetry.jsonl").open("a") as f:
                f.write(json.dumps({"unix":time.time(),"phase":label,"devices":telemetry})+"\n")
            time.sleep(15)
        process_records[-1].update(exit_code=child.returncode,finished_unix=time.time())
        save(run/"pipeline_processes.json",process_records)
        assert child.returncode==0,(label,child.returncode)
    try:
        save(run/"formal_protocol.json",{"config":smoke["config"],"train_sha256":sha(p["train_data"]),
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=project.parent,text=True).strip(),
            "code_sha256":{x.name:sha(x) for x in (project/"scripts/luffy_knowledge_388").glob("*.py")},
            "smoke_sha256":sha(run/"smoke/summary.json"),"cpu_sha256":sha(run/"cpu_acceptance.json"),
            "model":str(p["model_path"]),"evaluation_steps":[75],"test_evaluated":False,
            "sampling":"4 fresh BO answers for every question visit; no cross-arm reuse",
            "evaluation":"Existing same_source_dpo_316 native ON 666DEV wrapper, 32768/32000, seeds42+index",
            "arms":{"A":"online","B":"online + alpha demo NLL","C":"mixed advantage + 0.25 shaped demo"},
            "equal_budget":"388 questions x 3 epochs, 75 updates and 4656 online answers per arm; not equal GPU time"})
        for arm in ("A","B","C"):
            if (run/arm/"complete.json").exists():continue
            launch("train-"+arm,[py,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","--",
                str(project/"scripts/luffy_knowledge_388/train.py"),"--mode","train","--arm",arm],"3,4")
        # No more training depends on the dedicated rollout worker.
        rpc(run/"rpc",{"request_id":"zz-final-training-shutdown","mode":"shutdown"})
        evaluation=p["benchmark_root"]/"eval/scripts/evaluate_same_source_dpo_316.py"
        for arm in ("A","B","C"):
            marker=run/("evaluation-"+arm+"-complete.json")
            if marker.exists():continue
            complete=json.loads((run/arm/"complete.json").read_text())
            assert complete["steps"]==75
            launch("eval-"+arm,[py,"-u",str(evaluation),"--model",str(p["model_path"]),"--arm",arm,
                "--adapter",complete["final_adapter"],"--run-name",cfg["name"]],"3,4")
            save(marker,{"step":75,"arm":arm,"adapter_sha256":complete["final_adapter_sha256"]})
        save(run/"pipeline_complete.json",{"success":True,"arms":["A","B","C"],"updates_per_arm":75,
            "rollouts_per_arm":4656,"dev_questions_per_arm":666,"evaluation_steps":[75],"test_evaluated":False,
            "completed_unix":time.time()})
        save(run/"pipeline_state.json",{"phase":"complete","updated_unix":time.time()})
    except Exception:
        save(run/"pipeline_failure.json",{"error":traceback.format_exc(),"unix":time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)
if __name__=="__main__":main()
