"""Coordinate the approved three-arm continuation and only step-120 DEV."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from paired_dpo import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    a = p.parse_args()
    plan = json.loads((a.run / "continuation_plan.json").read_text())
    assert plan["dev_evaluated_steps"] == [120] and plan["additional_updates"] == 80
    lock = a.run / "controller.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    processes, states = {}, {}
    script = Path(__file__).parent / "continue_train.py"
    benchmark = a.root / "benchmark/eval/scripts/evaluate_same_source_dpo_316.py"

    def launch(arm, phase):
        devices = plan["hardware_assignment"][arm]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, devices)),
                   OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        output = a.root / "finetuning/model/rl" / plan["run"] / arm
        if phase == "training":
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(script), "--data", plan["data_directory"], "--model", a.model,
                   "--run", str(a.run), "--output", str(output), "--arm", arm]
        else:
            assert (a.run / arm / "completion.json").exists()
            cmd = [sys.executable, "-u", str(benchmark), "--model", a.model, "--arm", arm,
                   "--adapter", str(output / "checkpoint-120"), "--run-name", plan["run"]]
        with (a.run / (arm + "." + phase + ".log")).open("x") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes[arm] = process
        states[arm] = {"phase": phase, "pid": process.pid, "started": time.time(), "command": cmd}
        save(a.run / "pipeline_status.json", states)

    try:
        for arm in ("base", "true", "shuffled"):
            launch(arm, "training")
        while processes:
            for arm, process in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                phase = states[arm]["phase"]
                states[arm]["exit_code"] = code
                states[arm]["finished"] = time.time()
                save(a.run / f"{arm}.{phase}.process.json", states[arm])
                del processes[arm]
                if code != 0:
                    states[arm]["phase"] = phase + "_failed"
                elif phase == "training":
                    launch(arm, "evaluation")
                else:
                    states[arm]["phase"] = "completed"
                save(a.run / "pipeline_status.json", states)
            if processes:
                time.sleep(10)
        save(a.run / "pipeline_completion.json", {
            "success": all(s["phase"] == "completed" for s in states.values()), "arms": states
        })
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
