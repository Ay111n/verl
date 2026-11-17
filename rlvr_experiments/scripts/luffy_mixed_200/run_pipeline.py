"""Mixed200 train/evaluation controller; a reserved worker may sample the full pool."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PROJECT=ROOT/"verl/rlvr_experiments"
sys.path.insert(0,str(PROJECT/"scripts/luffy_math_pilot"))
from common import save,rpc
PY=sys.executable

def alive(pid):
    p=Path(f"/proc/{pid}/stat")
    return p.exists() and p.read_text().split()[2]!="Z"

def stop(pid,marker):
    p=Path(f"/proc/{pid}/cmdline")
    if p.exists() and alive(pid):
        assert marker in p.read_bytes().decode(errors="replace"),(pid,marker)
        os.killpg(pid,signal.SIGTERM)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());name=cfg["name"];run=PROJECT/"runs"/name
    workers=json.loads((run/"workers.json").read_text())
    reference=json.loads((run/"reference_process.json").read_text())["pid"]
    training_workers=workers[1:1+len(cfg["rollout_gpus"])]
    owned=[workers[0]["pid"]]+[x["pid"] for x in training_workers]+[reference]
    def save_owned():save(run/"owned_processes.json",{"pids":owned,"marker":name,"max_concurrent_gpus":4})
    save_owned()
    def stage(value,**extra):
        s={"stage":value,"time_unix":time.time(),**extra};save(run/"pipeline_status.json",s);print(json.dumps(s),flush=True)
    def alarm(signum,frame):raise TimeoutError("End-to-end deadline: preserving final three minutes for delivery")
    signal.signal(signal.SIGALRM,alarm)
    if cfg.get("strict_wall_budget",True):
        signal.alarm(max(1,int(cfg["overall_deadline_unix"]-time.time()-180)))
    try:
        stage("preparing_references",fixed_questions=200)
        data=ROOT/"finetuning/rl/final_data"/name
        while not (data/"manifest.json").exists():
            if not alive(reference):
                partial=ROOT/"finetuning/rl/processed_data"/name/"accepted_references.partial.jsonl"
                audit_path=partial.parent/"reference_audit.jsonl"
                attempts=[json.loads(line)["attempt"] for line in audit_path.read_text().splitlines()] if audit_path.exists() else []
                if time.time()+180<cfg["reference_deadline_unix"] and partial.exists() and max(attempts,default=-1)<cfg["repair_rounds"]:
                    stage("resuming_failed_reference_questions",fixed_questions=200)
                    resumed=subprocess.Popen([PY,"-u",str(ROOT/"finetuning/rl/scripts/prepare_mixed200_references.py"),
                        "--config",str(a.config),"--resume"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),
                        stdout=(run/"reference_preparation.log").open("a"),stderr=subprocess.STDOUT,
                        cwd=str(ROOT),start_new_session=True)
                    reference=resumed.pid;owned.append(reference);save_owned()
                    save(run/"reference_process.json",{"pid":reference,"resumed_unix":time.time()})
                    stage("preparing_references",fixed_questions=200,resumed=True)
                else:
                    raise RuntimeError("Reference preparation ended before all200 verified; fixed IDs retained, no training launched")
            if time.time()>cfg["reference_deadline_unix"]+15:
                raise TimeoutError("Reference preparation exceeded reserved allocation")
            time.sleep(2)
        while (run/"reference_quality_hold.json").exists() and json.loads((run/"reference_quality_hold.json").read_text()).get("training_hold"):
            stage("reference_quality_hold",reason="Known inconsistent question requires approved revision before training")
            time.sleep(30)
        manifest=json.loads((data/"manifest.json").read_text());assert manifest["training_questions"]==200
        while alive(reference):time.sleep(.5)
        stage("releasing_reference_gpu_for_actor",reference_sources=manifest["source_counts"])
        rpc(Path(workers[0]["rpc"]),{"request_id":"shutdown-reference-worker0","mode":"shutdown"},timeout=60)
        end=time.time()+90
        while alive(workers[0]["pid"]):
            if time.time()>end:stop(workers[0]["pid"],name);break
            time.sleep(1)
        time.sleep(5)
        log=(run/"train.log").open("w")
        actor_config=a.config
        if not cfg.get("strict_wall_budget",True):
            cfg["train_deadline_unix"]=time.time()+14400
            actor_config=run/"effective_training_config.json"
            save(actor_config,cfg)
        actor=subprocess.Popen([PY,"-u",str(PROJECT/"scripts/luffy_math_pilot/train.py"),"--config",str(actor_config)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg["actor_gpu"]),TOKENIZERS_PARALLELISM="false"),
            stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        owned.append(actor.pid);save_owned();save(run/"actor_process.json",{"pid":actor.pid,"gpu":cfg["actor_gpu"]})
        stage("training",questions=200,planned_updates=13,actor_gpu=cfg["actor_gpu"],rollout_gpus=cfg["rollout_gpus"])
        while actor.poll() is None:
            if cfg.get("strict_wall_budget",True) and time.time()>cfg["train_deadline_unix"]+300:
                raise TimeoutError("Training exceeded its allocation; evaluation reservation protected")
            time.sleep(2)
        assert actor.returncode==0,"Training actor failed; see train.log"
        complete=json.loads((run/"complete.json").read_text())
        stage("releasing_sampling_gpus",completed_steps=complete["steps"],question_visits=complete["question_visits"])
        for worker in training_workers:
            rpc(Path(worker["rpc"]),{"request_id":"shutdown-training-sampler","mode":"shutdown"},timeout=60)
        end=time.time()+90
        while any(alive(w["pid"]) for w in training_workers):
            if time.time()>end:
                for w in training_workers:stop(w["pid"],name)
                break
            time.sleep(1)
        time.sleep(5)
        stage("evaluating_full_666dev",adapter=complete["adapter"],max_output_tokens=32000)
        log=(run/"eval.log").open("w")
        evaluation=subprocess.Popen([PY,"-u",str(ROOT/"benchmark/eval/scripts/evaluate_same_source_dpo_316.py"),
            "--model",str(ROOT/cfg["model_path"]),"--arm","mixed","--adapter",complete["adapter"],"--run-name",name],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(map(str,cfg["eval_gpus"])),TOKENIZERS_PARALLELISM="false"),
            stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),start_new_session=True)
        owned.append(evaluation.pid);save_owned()
        save(run/"eval_process.json",{"pid":evaluation.pid,"gpus":cfg["eval_gpus"],"start_unix":time.time()})
        while evaluation.poll() is None:time.sleep(2)
        assert evaluation.returncode==0,"Evaluation failed; see eval.log"
        stage("analysis_ready",evaluation_questions=666,elapsed_seconds=time.time()-cfg["overall_start_unix"])
        save(run/"pipeline_complete.json",{"status":"evaluation_complete_analysis_pending",
            "started_unix":cfg["overall_start_unix"],"ended_unix":time.time(),
            "end_to_end_seconds":time.time()-cfg["overall_start_unix"],
            "maximum_concurrent_gpus":4,"gpus":cfg["reference_gpus"],"eval_questions":666})
    except Exception:
        save(run/"pipeline_failure.json",{"error":traceback.format_exc(),"time_unix":time.time()})
        raise
    finally:
        signal.alarm(0)
        for pid in owned:stop(pid,name)
if __name__=="__main__":main()
