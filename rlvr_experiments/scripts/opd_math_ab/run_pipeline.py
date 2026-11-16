"""Run matched arms and heldout evaluations on at most physical GPUs 0 and 2."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import paths, save, rpc, sha


class AdoptedTrainer:
    """Wait for an existing, verified trainer without starting another copy."""
    def __init__(self, pid, project, config, complete):
        self.pid = pid
        self.returncode = None
        self.complete = complete
        proc = Path("/proc") / str(pid)
        self.expected = [str(project / "scripts/opd_math_ab/train.py"),
                         "--mode", "train", "--arm", "answer", "--config", str(config)]
        self.command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        assert self.command[-len(self.expected):] == self.expected, self.command
        env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
        assert env.get(b"CUDA_VISIBLE_DEVICES") == b"0"
        self.start_ticks = (proc / "stat").read_text().rsplit(") ", 1)[1].split()[19]

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        proc = Path("/proc") / str(self.pid)
        try:
            stat = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
            if stat[19] == self.start_ticks and stat[0] != "Z":
                command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
                assert command == self.command, "Adopted trainer command changed"
                return None
        except FileNotFoundError:
            pass
        self.returncode = 0 if self.complete.exists() else 1
        return self.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adopt-answer-pid", type=int)
    ap.add_argument("--eval-batch-size", type=int, default=64, choices=[64])
    args = ap.parse_args()
    p = paths()
    project = Path(__file__).resolve().parents[2]
    run = p["run_root"]
    py = str(p["python"])
    cfg = project / "configs/opd_math_ab_20261116.json"
    config = json.loads(cfg.read_text())
    smoke = json.loads((run / "smoke/summary.json").read_text())
    assert smoke["passed"]
    assert smoke["config"] == config
    assert not (run / "pipeline_complete.json").exists()
    eval_out = p["eval_output"] / f"batch{args.eval_batch_size}"
    processes = {}
    records = []

    adopted = None
    if args.adopt_answer_pid:
        adopted = AdoptedTrainer(args.adopt_answer_pid, project, cfg, run / "answer/complete.json")
        assert adopted.poll() is None, "Trainer must be alive at adoption"
        records.append({"name": "train-answer", "pid": adopted.pid, "gpu": "0",
                        "adopted_unix": time.time(), "command": adopted.command,
                        "process_start_ticks": adopted.start_ticks})
        processes["train-answer"] = adopted

    def launch(name, command, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        log = (run / (name + ".log")).open("a")
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=project)
        log.close()
        processes[name] = child
        records.append({"name": name, "pid": child.pid, "gpu": gpu, "started_unix": time.time(), "command": command})
        save(run / "pipeline_processes.json", records)
        return child

    def state(phase):
        alive = {k: v.poll() for k, v in processes.items()}
        save(run / "pipeline_state.json", {"phase": phase, "updated_unix": time.time(), "processes": alive,
             "allowed_physical_gpus": [0, 2], "pipeline_pid": os.getpid(),
             "evaluation_output": str(eval_out), "evaluation_request_batch_size": args.eval_batch_size})

    def wait(child, name):
        while child.poll() is None:
            state(name)
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                           "--format=csv,noheader,nounits"], text=True)
            selected = [line for line in raw.splitlines() if int(line.split(",")[0]) in (0, 2)]
            with (run / "gpu_telemetry.jsonl").open("a") as f:
                f.write(json.dumps({"time": time.time(), "phase": name, "configured_task_gpus": [0, 2],
                                    "devices": selected}) + "\n")
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"{name} failed with exit {child.returncode}; see {name}.log")

    evaluation = p["benchmark_root"] / "eval/scripts/opd_math_ab/evaluate.py"
    def eval_command(name, rpc_root=None):
        command = [py, "-u", str(evaluation), "--experiment", str(project), "--model", name,
                   "--request-batch-size", str(args.eval_batch_size), "--output-root", str(eval_out)]
        if rpc_root:
            command += ["--rpc", str(rpc_root)]
        return command

    formal = run / "formal_protocol.json"
    if formal.exists():
        frozen = json.loads(formal.read_text())
        assert frozen["config"] == config and frozen["config_sha256"] == sha(cfg)
        assert frozen["train_sha256"] == sha(p["train_data"])
        assert frozen["eval_sha256"] == sha(p["eval_data"])
        assert frozen["smoke_sha256"] == sha(run / "smoke/summary.json")
    else:
        save(formal, {
            "config": config, "config_sha256": sha(cfg),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project.parent, text=True).strip(),
            "train_sha256": sha(p["train_data"]), "eval_sha256": sha(p["eval_data"]),
            "smoke_sha256": sha(run / "smoke/summary.json"),
            "worker_gpu": 2, "trainer_gpu": 0, "max_concurrent_gpus": 2,
            "evaluation_overlap": "S0 evaluation may use the idle rollout worker during trainer backward passes; final A/B evaluation starts after training. After both arms finish, GPU0 may host a second evaluation worker.",
            "selection": "Fixed final checkpoint after one epoch, no checkpoint selection on holdout."
        })
    revision = run / "execution_revision.json"
    assert not revision.exists(), "Preserve an existing execution revision; inspect before another restart"
    save(revision, {
        "created_unix": time.time(), "pipeline_pid": os.getpid(), "adopted_answer_pid": args.adopt_answer_pid,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project.parent, text=True).strip(),
        "controller_sha256": sha(Path(__file__)), "evaluation_script_sha256": sha(evaluation),
        "formal_protocol_sha256": sha(formal), "config_unchanged": True,
        "evaluation_request_batch_size": args.eval_batch_size, "worker_max_num_seqs": 16,
        "evaluation_output": str(eval_out), "max_concurrent_physical_gpus": 2,
        "reason": "Submit 64 questions per request so 16 active decode slots refill without a batch-of-16 long-tail barrier.",
        "comparison": "All final S0/A/B evaluations use batch64. Prior partial batch16 S0 outputs are retained separately and excluded.",
        "unchanged": "Training progress, training batch8, prompts, question order, per-question seeds, sampling, output budget and frozen scoring."
    })
    baseline = launch("eval-s0-batch64", eval_command("s0"), "")
    for arm in ("answer", "solution"):
        child = adopted if arm == "answer" and adopted else launch(
            "train-" + arm, [py, "-u", str(project / "scripts/opd_math_ab/train.py"),
                            "--mode", "train", "--arm", arm, "--config", str(cfg)], "0")
        wait(child, "train-" + arm)
        complete = json.loads((run / arm / "complete.json").read_text())
        assert complete["steps"] == 43 and complete["examples"] == 342
    answer_eval = launch("eval-answer-batch64", eval_command("answer"), "")
    rpc0 = run / "rpc_eval_gpu0"
    second = launch("worker-eval-gpu0", [py, "-u", str(project / "scripts/opd_math_ab/worker.py"),
                                       "--rpc", str(rpc0), "--max-num-seqs", "16"], "0")
    solution_eval = launch("eval-solution-batch64", eval_command("solution", rpc0), "")
    for label, child in [("eval-s0", baseline), ("eval-answer", answer_eval), ("eval-solution", solution_eval)]:
        wait(child, label)
    rpc(run / "rpc", {"request_id": "zz-final-shutdown", "mode": "shutdown"})
    rpc(rpc0, {"request_id": "zz-final-shutdown", "mode": "shutdown"})
    wait(second, "worker-eval-gpu0")
    state("complete")
    save(run / "pipeline_complete.json", {"completed_unix": time.time(), "max_configured_gpus": 2,
          "training_examples_per_arm": 342, "updates_per_arm": 43, "evaluation_questions_per_model": 512,
          "evaluation_request_batch_size": args.eval_batch_size, "evaluation_output": str(eval_out)})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        p = paths()
        save(p["run_root"] / "pipeline_failure.json", {"error": traceback.format_exc(), "time": time.time()})
        raise
