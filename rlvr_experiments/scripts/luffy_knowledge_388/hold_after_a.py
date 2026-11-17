"""Finish the active A process and hold B/C after explicit user authorization."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback


def read(path):return json.loads(path.read_text())

def save(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    temp.replace(path)

def alive(pid):
    path=Path("/proc")/str(pid)/"stat"
    return path.exists() and path.read_text().split()[2]!="Z"

def stop_owned(pid,expected,lock):
    proc=Path("/proc")/str(pid)
    assert alive(pid), ("Expected live component",pid)
    command=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode()
    assert expected in command,(pid,command)
    assert lock.exists() and lock.read_text().strip()==str(pid),str(lock)
    os.kill(pid,signal.SIGTERM)  # Only this controller PID, never its training process group.
    for _ in range(100):
        if not alive(pid):break
        time.sleep(.1)
    assert not alive(pid),("Component failed to stop",pid)
    assert lock.read_text().strip()==str(pid)
    lock.unlink()
    return {"pid":pid,"command":command,"stopped_unix":time.time()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-root",type=Path,required=True)
    args=parser.parse_args();run=args.run_root
    lock=run/"hold_monitor.lock"
    with lock.open("x") as stream:stream.write(str(os.getpid()))
    try:
        state=read(run/"pipeline_state.json")
        assert state["phase"]=="train-A" and alive(state["pid"])
        assert not (run/"B/initialization.json").exists() and not (run/"C/initialization.json").exists()
        analysis=read(run/"analysis_stage_state.json")
        assert analysis["phase"]=="waiting_for_final_evaluations"
        request={"authorization":"User selected: finish A, hold B/C, optimize throughput first.",
                 "requested_unix":time.time(),"active_A_pid":state["pid"],
                 "original_controller_pid":state["controller_pid"],"hold_monitor_pid":os.getpid(),
                 "target_A_steps":75,"hold_arms":["B","C"],"training_configuration_changed":False}
        save(run/"hold_request.json",request)
        stopped=[stop_owned(state["controller_pid"],"luffy_knowledge_388/run_pipeline.py",run/"pipeline.lock")]
        stopped.append(stop_owned(analysis["pid"],"luffy_knowledge_388/finish_analysis.py",run/"analysis_stage.lock"))
        assert alive(state["pid"]),"The active A training must remain live"
        save(run/"hold_handoff.json",{"stopped_controllers":stopped,"A_kept_running":True,"unix":time.time()})
        save(run/"analysis_stage_state.json",{"phase":"held_until_throughput_acceptance","unix":time.time()})
        while not (run/"A/complete.json").exists():
            assert alive(state["pid"]),"A exited before writing its completion record"
            assert not (run/"B/initialization.json").exists() and not (run/"C/initialization.json").exists()
            progress=read(run/"A/progress.json")
            save(run/"pipeline_state.json",{"phase":"train-A-hold-B-C","pid":state["pid"],
                "controller_pid":os.getpid(),"A_completed_steps":progress["step"],"updated_unix":time.time()})
            time.sleep(5)
        complete=read(run/"A/complete.json")
        assert complete["steps"]==75 and complete["online_rollouts"]==4656 and complete["question_visits"]==1164
        deadline=time.monotonic()+120
        while alive(state["pid"]) and time.monotonic()<deadline:time.sleep(1)
        assert not alive(state["pid"]),"A did not exit after completion"
        assert not (run/"B/initialization.json").exists() and not (run/"C/initialization.json").exists()
        name="zz-user-hold-after-A"
        save(run/"rpc"/(name+".request.json"),{"request_id":name,"mode":"shutdown"})
        response=run/"rpc"/(name+".response.json")
        deadline=time.monotonic()+120
        while not response.exists() and time.monotonic()<deadline:time.sleep(1)
        assert response.exists() and read(response)["stopped"]
        save(run/"hold_complete.json",{"held":True,"completed_arms":["A"],"A_steps":75,
            "unstarted_arms":["B","C"],"sampling_worker_shutdown_acknowledged":True,
            "all_model_and_optimizer_checkpoints_preserved":True,"unix":time.time()})
        save(run/"pipeline_state.json",{"phase":"held-after-A","A_completed_steps":75,"updated_unix":time.time()})
        print("A completed; B/C held; dedicated rollout worker shutdown acknowledged.",flush=True)
    except Exception:
        save(run/"hold_failure.json",{"error":traceback.format_exc(),"unix":time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":main()
