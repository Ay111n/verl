"""Expand reference sampling to four GPUs only after the full666DEV evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "verl/rlvr_experiments"


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def busy_gpus():
    lookup = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    indices = {uuid.strip(): int(index) for index, uuid in (line.split(",") for line in lookup.splitlines())}
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return {indices[line.strip()] for line in raw.splitlines() if line.strip() in indices}


def handoff_cpu(pid, source, marker):
    """Stop only the CPU dispatcher at a complete, readable checkpoint."""
    while alive(pid):
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        assert "sample_remaining_mixed2000.py" in cmd and marker in cmd, (pid, cmd)
        os.kill(pid, signal.SIGSTOP)
        consistent = False
        try:
            data = (source / "rollouts.jsonl").read_bytes()
            values = [json.loads(line) for line in data.splitlines() if line.strip()]
            status = read(source / "status.json")
            consistent = len(values) == len({r["id"] for r in values}) == status["completed"]
            if consistent:
                result = {"previous_sampler_pid": pid, "completed_before_handoff": len(values),
                          "rollouts_sha256": hashlib.sha256(data).hexdigest(), "time_unix": time.time()}
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
                deadline = time.time() + 30
                while alive(pid):
                    assert time.time() < deadline, "Previous CPU dispatcher did not exit"
                    time.sleep(.2)
                return result
        except (ValueError, json.JSONDecodeError):
            pass
        finally:
            if not consistent and alive(pid):
                os.kill(pid, signal.SIGCONT)
        time.sleep(.5)
    return {"previous_sampler_pid": pid, "previous_sampler_already_exited": True, "time_unix": time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = read(args.config); name = cfg["name"]
    run = PROJECT / "runs" / name
    source = ROOT / "finetuning/rl/processed_data/luffy_mixed_2000_remaining_1800_20261117"
    workers_file = run / "background_parallel_workers.json"
    original = next(w for w in read(run / "workers.json") if w["gpu"] == 5)
    assert cfg["maximum_concurrent_gpus"] == 4
    assert cfg["rollout_gpus"] == [2, 3] and cfg["actor_gpu"] == 0 and cfg["eval_gpus"] == [0, 2]
    if args.dry_run:
        print(json.dumps({"gate": "pipeline_complete AND controller exited AND GPUs0/2/3 free",
            "existing_worker": original, "additional_gpus": [0, 2, 3], "total_limit": 4})); return

    def stage(value, **extra):
        data = {"stage": value, "time_unix": time.time(), **extra}
        save(run / "background_scaleup_status.json", data)
        print(json.dumps(data), flush=True)

    stage("waiting_for_complete666dev")
    while True:
        if (run / "background_sampling_stopped.json").exists():
            stage("stopped_by_user"); return
        if (run / "background_sampling_complete.json").exists():
            stage("already_complete_no_pending_samples"); return
        if (run / "pipeline_failure.json").exists():
            raise RuntimeError("Pilot failed; do not claim full666DEV completed or add sampling GPUs")
        if (run / "pipeline_complete.json").exists():
            finished = read(run / "pipeline_complete.json")
            assert finished["eval_questions"] == 666
            if not alive(read(run / "pipeline_process.json")["pid"]) and not ({0, 2, 3} & busy_gpus()):
                break
        time.sleep(20)
    predictions = list((ROOT / "benchmark/eval/results" / name / "mixed").rglob("predictions.jsonl"))
    assert len(predictions) == 1
    records = [json.loads(line) for line in predictions[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == len({r["uid"] for r in records}) == 666
    assert alive(original["pid"])
    stage("starting_three_additional_sampling_workers", complete_eval_questions=666)
    workers = [original]
    save(workers_file, workers)
    for gpu in [0, 2, 3]:
        rpc_dir = run / "rpc_workers" / f"full_worker{gpu}"
        assert not rpc_dir.exists(), f"Existing scaled worker needs explicit recovery: {rpc_dir}"
        rpc_dir.mkdir()
        with (run / f"full_worker{gpu}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(PROJECT / "scripts/luffy_math_pilot/worker.py"),
                "--model", str(ROOT / cfg["model_path"]), "--rpc", str(rpc_dir), "--max-num-seqs", "16"],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
                         VLLM_WORKER_MULTIPROC_METHOD="spawn"),
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append({"gpu": gpu, "pid": process.pid, "rpc": str(rpc_dir), "role": "post_eval_reference_sampling"})
        save(workers_file, workers)
    deadline = time.time() + 1200
    while not all((Path(w["rpc"]) / "ready.json").exists() for w in workers[1:]):
        assert all(alive(w["pid"]) for w in workers[1:]), "A new sampling worker failed to start"
        assert time.time() < deadline, "Sampling worker startup timed out"
        time.sleep(2)
    if (run / "background_sampling_complete.json").exists():
        sys.path.insert(0, str(PROJECT / "scripts/luffy_math_pilot"))
        from common import rpc
        for w in workers[1:]:
            rpc(w["rpc"], {"request_id": "shutdown-already-complete", "mode": "shutdown"}, timeout=120)
        stage("already_complete_during_worker_startup"); return
    assert alive(original["pid"])
    previous = read(run / "background_sampling_process.json")
    handoff = handoff_cpu(previous["pid"], source, name)
    save(run / "background_sampling_handoff.json", handoff)
    if len([line for line in (source / "rollouts.jsonl").read_text().splitlines() if line.strip()]) == 1800:
        sys.path.insert(0, str(PROJECT / "scripts/luffy_math_pilot"))
        from common import rpc
        for w in workers[1:]:
            rpc(w["rpc"], {"request_id": "shutdown-already-complete", "mode": "shutdown"}, timeout=120)
        assert (run / "background_sampling_complete.json").exists(), "Original sampler completed rows but not its final manifest; inspect before restart"
        stage("already_complete_at_handoff"); return
    script = ROOT / "finetuning/rl/scripts/complete_mixed2000_parallel.py"
    with (source / "sampling_parallel.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-u", str(script),
            "--workers-file", str(workers_file), "--model", str(ROOT / cfg["model_path"])],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false"),
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True)
    save(run / "background_sampling_process.json", {"pid": process.pid, "gpus": [w["gpu"] for w in workers],
        "worker_pids": [w["pid"] for w in workers], "started_unix": time.time(), "questions": 1800,
        "previous_sampler": previous, "source_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()})
    stage("four_gpu_sampling", sampler_pid=process.pid, gpus=[w["gpu"] for w in workers], **handoff)
    assert process.wait() == 0, "Parallel reference dispatcher failed; raw requests and checkpoints retained"
    assert (run / "background_sampling_complete.json").exists()
    stage("complete", sampled_questions=1800)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--dry-run", action="store_true")
        arguments = parser.parse_args()
        config = read(arguments.config)
        save(PROJECT / "runs" / config["name"] / "background_scaleup_failure.json",
             {"error": traceback.format_exc(), "time_unix": time.time()})
        raise
