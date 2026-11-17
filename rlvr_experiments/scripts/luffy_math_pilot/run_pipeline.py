"""Bounded controller: reference repair, one training arm, full matched 666DEV."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,PROJECT,save,read_rows,rpc,sha
PY=sys.executable
def alive(pid):
    p=Path(f"/proc/{pid}/stat")
    return p.exists() and p.read_text().split()[2]!="Z"
def stop_owned(pid,marker):
    path=Path(f"/proc/{pid}/cmdline")
    if path.exists():
        assert marker in path.read_bytes().decode(errors="replace")
        os.killpg(pid,signal.SIGTERM)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());cfg["model_path"]=str(ROOT/cfg["model_path"]);run=PROJECT/"runs"/cfg["name"]
    actor=int((run/"train.pid").read_text());sampler=int((run/"worker.pid").read_text())
    status=run/"pipeline_status.json";started=time.time()
    def stage(name,**kw):
        value={"stage":name,"time_unix":time.time(),**kw};save(status,value);print(json.dumps(value),flush=True)
    stage("checking_baseline")
    check=subprocess.run([PY,str(ROOT/"benchmark/eval/scripts/analyze_luffy_math_pilot.py"),"--root",str(ROOT),"--check-baseline"],capture_output=True,text=True)
    (run/"baseline_check.log").write_text(check.stdout+check.stderr);assert check.returncode==0,check.stderr
    stage("waiting_for_reference_repairs",candidates=128,repair_requests=30)
    response=run/"rpc/repair-english-01.response.json"
    while not response.exists():
        assert alive(sampler),"Reference sampler exited"
        if time.time()>cfg["train_deadline_unix"]-1800:raise TimeoutError("Reference preparation exhausted training allocation")
        time.sleep(2)
    stage("finalizing_verified_training_data")
    result=subprocess.run([PY,str(ROOT/"finetuning/rl/scripts/prepare_luffy_math_pilot.py"),
        "--finalize",str(a.config)],capture_output=True,text=True)
    (run/"data_preparation.log").write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    manifest=json.loads((ROOT/"finetuning/rl/final_data"/cfg["name"]/"manifest.json").read_text())
    stage("training",training_questions=manifest["training_questions"],repaired_references=manifest["repaired_references"])
    while not (run/"complete.json").exists():
        assert alive(actor),"Training actor exited before completion; inspect train.log"
        if time.time()>cfg["train_deadline_unix"]+900:
            stop_owned(actor,"luffy_math_pilot/train.py")
            raise TimeoutError("Training exceeded its reserved allocation")
        time.sleep(2)
    complete=json.loads((run/"complete.json").read_text())
    while alive(actor):time.sleep(1)
    stage("releasing_training_sampler",steps=complete["steps"])
    rpc(run/"rpc",{"request_id":"shutdown-training-sampler","mode":"shutdown"},timeout=120)
    deadline=time.time()+120
    while alive(sampler):
        if time.time()>deadline:stop_owned(sampler,"luffy_math_pilot/worker.py");break
        time.sleep(1)
    stage("evaluating_full_666dev",adapter=complete["adapter"],budget=32000)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0,5",TOKENIZERS_PARALLELISM="false")
    log=(run/"eval.log").open("w")
    proc=subprocess.Popen([PY,"-u",str(ROOT/"benchmark/eval/scripts/evaluate_same_source_dpo_316.py"),
        "--model",cfg["model_path"],"--arm","mixed","--adapter",complete["adapter"],"--run-name",cfg["name"]],
        env=env,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
    save(run/"eval_process.json",{"pid":proc.pid,"devices":[0,5],"start_unix":time.time()})
    while proc.poll() is None:
        if time.time()>cfg["overall_deadline_unix"]-180:
            stop_owned(proc.pid,"evaluate_same_source_dpo_316.py")
            raise TimeoutError("Evaluation stopped at end-to-end deadline; partial files retained")
        time.sleep(2)
    assert proc.returncode==0,"Evaluation failed; inspect eval.log"
    stage("cpu_analysis")
    result=subprocess.run([PY,str(ROOT/"benchmark/eval/scripts/analyze_luffy_math_pilot.py"),"--root",str(ROOT)],
        capture_output=True,text=True)
    (run/"analysis.log").write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    stage("complete",end_to_end_seconds=time.time()-cfg["overall_start_unix"])
    save(run/"pipeline_complete.json",{"status":"complete","started_unix":cfg["overall_start_unix"],
        "ended_unix":time.time(),"end_to_end_seconds":time.time()-cfg["overall_start_unix"],
        "maximum_concurrent_gpus":2,"gpus":[0,5],"eval_questions":666})
if __name__=="__main__":
    try:main()
    except Exception:
        root=PROJECT/"runs/luffy_math_pilot_20261117"
        save(root/"pipeline_failure.json",{"error":traceback.format_exc(),"time_unix":time.time()})
        raise
