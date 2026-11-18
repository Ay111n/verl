"""Run the measured pilot, fixed epoch decision, training and matched native DEV."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from contract import CONFIG, MODEL_OUT, PROJECT, ROOT, RUN, read
from scripts.same_source_dpo_316.paired_dpo import save, sha

HERE = Path(__file__).resolve().parent


def idle():
    output = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"],text=True)
    return [gpu for gpu, used, util in (map(int, r.split(",")) for r in output.splitlines())
            if used < 128 and util < 5]


def forecast(cfg, started):
    run = RUN/"pilot"
    online = read(run/"steps/checkpoint-0001.json")
    nll = read(run/"steps/checkpoint-0002.json")
    t1,t2 = [read(run/"timings"/f"{i:04d}.json") for i in (1,2)]
    complete = read(run/"complete.json")
    elapsed = time.time()-started
    per_epoch = 13*(online["generation_seconds"] + online["sync_seconds"] + t1["checkpoint_seconds"])
    per_epoch += 12.5*online["update_seconds"]
    costs = {}
    for epochs in (1,2):
        train = complete["model_load_seconds"] + epochs*per_epoch
        train += ((13*epochs)//3) * (nll["update_seconds"]+t2["checkpoint_seconds"])
        train += complete["final_sync_seconds"]
        costs[str(epochs)] = dict(training_minutes=train/60,
             end_to_end_minutes=(elapsed+train)/60+cfg["eval_historical_minutes"])
    # Include the actual pilot and boot time, a full final DEV, and the maximum NLL exposure.
    epochs = 2 if costs["2"]["end_to_end_minutes"] <= cfg["budget_minutes"] else 1
    return dict(epochs=epochs, estimates=costs, elapsed_pilot_and_startup_minutes=elapsed/60,
                eval_historical_minutes=cfg["eval_historical_minutes"], pilot_questions=16, pilot_responses=48,
                online_output_tokens=online["output_tokens"], rollout_seconds=online["generation_seconds"],
                generated_tokens_per_second=online["output_tokens"]/online["generation_seconds"],
                online_update_seconds=online["update_seconds"], nll_target_tokens=nll["target_tokens"],
                nll_update_seconds=nll["update_seconds"], sync_seconds=online["sync_seconds"],
                final_sync_seconds=complete["final_sync_seconds"], checkpoint_seconds=[t1["checkpoint_seconds"],t2["checkpoint_seconds"]],
                runtime_forecast="13 rollout waves/epoch, actor cost 12.5 full batches, up to four NLL per epoch; 2-GPU historical DEV",
                pilot_policy_discarded=True)


def supervise():
    RUN.mkdir(parents=True,exist_ok=True)
    lock = (RUN/"controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (RUN/"preflight.json").is_file(), "Complete the dataset/weight preflight first"
    if (RUN/"pipeline_complete.json").exists():
        return
    base = read(CONFIG)
    procs = []
    workers = {}
    restarts = {}
    def stage(name,**info):
        save(RUN/"status.json",dict(stage=name,unix=time.time(),pid=os.getpid(),**info))
    def launch(args,gpus,log):
        with log.open("a") as f:
            proc = subprocess.Popen([sys.executable,"-u",*map(str,args)],cwd=ROOT,
                   env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(map(str,gpus)),TOKENIZERS_PARALLELISM="false",
                            VLLM_WORKER_MULTIPROC_METHOD="spawn",PYTHONUNBUFFERED="1"),
                   stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        procs.append(proc)
        save(RUN/"owned_processes.json",dict(controller=os.getpid(),pids=[p.pid for p in procs if p.poll() is None]))
        return proc
    def end(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=50)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait(timeout=30)
        except ProcessLookupError:
            pass
    def start_worker(i,gpu):
        directory = RUN/"rpc_workers"/f"worker{i}"
        proc = launch([PROJECT/"scripts/luffy_math_pilot/worker.py","--model",ROOT/base["model_path"],
                       "--rpc",directory,"--max-num-seqs","16"],[gpu],RUN/f"worker{i}.log")
        workers[i]=(proc,directory,gpu)
    def watch_workers():
        for i,(proc,directory,gpu) in list(workers.items()):
            if proc.poll() is not None:
                restarts[i]=restarts.get(i,0)+1
                if restarts[i]>3:
                    raise RuntimeError(f"Sampling worker {i} repeatedly exited ({proc.returncode})")
                start_worker(i,gpu)
    def actor(which):
        # Each new process either creates the initial state or restores the last complete transaction.
        proc=launch([HERE/"train.py","--stage",which],[cfg["actor_gpu"]],RUN/f"{which}_actor.log")
        while proc.poll() is None:
            watch_workers()
            time.sleep(2)
        if proc.returncode:
            raise RuntimeError(f"{which} actor failed ({proc.returncode}); see {which}_actor.log")
    def shutdown_workers():
        from scripts.luffy_math_pilot.common import rpc
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(len(workers),1)) as pool:
            futures=[pool.submit(rpc,d,dict(request_id=f"shutdown-{os.getpid()}-{i}",mode="shutdown"),90)
                     for i,(p,d,g) in workers.items() if p.poll() is None]
            for job in futures:
                job.result()
        for p,_,_ in workers.values():
            p.wait(timeout=120)
        workers.clear()
    try:
        if (RUN/"launched.json").exists():
            launch_meta=read(RUN/"launched.json")
        else:
            stage("checking_idle_gpus")
            while True:
                first=set(idle())
                time.sleep(3)
                available=sorted(first.intersection(idle()))[:base["maximum_concurrent_gpus"]]
                if len(available)>=2:
                    break
                time.sleep(30)
            launch_meta=dict(start_unix=time.time(),gpus=available,python=sys.executable)
            save(RUN/"launched.json",launch_meta)
        selected=launch_meta["gpus"]
        cfg=dict(base,actor_gpu=selected[0],rollout_gpus=selected[1:],eval_gpus=selected[:2],epochs=1)
        assert len(set(selected))==len(selected)<=base["maximum_concurrent_gpus"]
        if not (RUN/"complete.json").exists():
            stage("loading_sampling_replicas",gpus=selected)
            assert set(selected)<=set(idle()),"Assigned GPUs now in use; recheck ownership before retry"
            save(RUN/"pilot_config.json",cfg)
            for i,gpu in enumerate(cfg["rollout_gpus"]):
                start_worker(i,gpu)
            before=time.monotonic()
            while not all((d/"ready.json").is_file() and read(d/"ready.json")["pid"]==p.pid
                          for p,d,g in workers.values()):
                watch_workers()
                if time.monotonic()-before>1200:
                    raise TimeoutError("Sampling model load")
                time.sleep(2)
            if not (RUN/"pilot/complete.json").exists():
                stage("pilot",gpus=selected)
                actor("pilot")
            if not (RUN/"forecast.json").exists():
                estimate=forecast(cfg,launch_meta["start_unix"])
                save(RUN/"forecast.json",estimate)
            estimate=read(RUN/"forecast.json")
            stage("forecast_ready",gpus=selected,epochs=estimate["epochs"])
            # Publish the actual estimate to the operator before starting the formal run.
            while not (RUN/"forecast_reported.json").exists():
                watch_workers()
                time.sleep(2)
            assert read(RUN/"forecast_reported.json")["forecast_sha256"]==sha(RUN/"forecast.json")
            cfg["epochs"]=estimate["epochs"]
            save(RUN/"effective_training_config.json",cfg)
            if not (RUN/"training_started.json").exists():
                save(RUN/"training_started.json",dict(unix=time.time()))
            stage("training",epochs=cfg["epochs"],gpus=selected)
            actor("formal")
            stage("releasing_training_gpus")
            shutdown_workers()
        final=read(RUN/"complete.json")
        assert final["question_visits"]==final["epochs"]*200 and final["online_responses"]==final["epochs"]*600
        assert not final["due"] and final["nll"]<=8 and sha(Path(final["adapter"])/"adapter_model.safetensors")==final["adapter_sha256"]
        stage("evaluating_666DEV",gpus=cfg["eval_gpus"])
        if not (RUN/"evaluation_started.json").exists():
            save(RUN/"evaluation_started.json",dict(unix=time.time()))
        assessment=ROOT/"benchmark/eval/analysis"/base["name"]/"candidate_result.json"
        if not assessment.exists():
            while not set(cfg["eval_gpus"])<=set(idle()):
                time.sleep(5)
            proc=launch([ROOT/base["eval_entrypoint"],"--model",ROOT/base["model_path"],"--arm","candidate",
                         "--adapter",final["adapter"],"--run-name",base["name"]],cfg["eval_gpus"],RUN/"eval.log")
            if proc.wait()!=0:
                raise RuntimeError("Native evaluation failed")
        assert assessment.is_file()
        end_time=time.time()
        save(RUN/"pipeline_complete.json",dict(start_unix=launch_meta["start_unix"],end_unix=end_time,
                 elapsed_seconds=end_time-launch_meta["start_unix"],assessment=str(assessment),questions=666))
        stage("evaluated")
    except BaseException:
        save(RUN/f"failure-{os.getpid()}.json",dict(error=traceback.format_exc(),unix=time.time()))
        raise
    finally:
        for proc in reversed(procs):
            end(proc)


if __name__=="__main__":
    supervise()
