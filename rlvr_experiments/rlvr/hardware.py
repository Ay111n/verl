"""Exclusive two-GPU leases, real one-second telemetry, and fail-closed gates.

GPU utilization below is NVML's active-time percentage, never SM occupancy.
No CUDA context is opened by this module.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess


def cleanup_run_processes(run_dir, terminal_status):
    """Reap orphaned vLLM subprocesses bearing this exact finished-run tag."""
    import psutil
    if terminal_status not in ("complete", "failed"):
        raise ValueError("Process cleanup requires a finished run")
    expected = str(Path(run_dir).resolve())
    owned = []
    for process in psutil.process_iter(["pid"]):
        if process.pid == os.getpid():
            continue
        try:
            if process.environ().get("RLVR_RUN_DIR") == expected:
                owned.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    identities = [{"pid": process.pid, "created_unix": process.create_time()} for process in owned]
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(owned, timeout=10)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    return {"run_dir": expected, "terminal_status": terminal_status,
            "owned_processes": identities, "terminated": len(gone), "killed": len(alive)}
import threading
import time


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: float
    memory_used_mib: float
    utilization_gpu: float
    pci_bus_id: str = ""
    numa_node: int | None = None
    compute_processes: int = 0

    @property
    def free_fraction(self):
        return (self.memory_total_mib - self.memory_used_mib) / self.memory_total_mib


def _command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=15).stdout


def _numa_node(bus_id):
    # NVIDIA emits an eight-digit domain; Linux sysfs uses four digits.
    address = bus_id.lower()
    if len(address.split(":")[0]) == 8:
        address = address[4:]
    try:
        value = int(Path("/sys/bus/pci/devices", address, "numa_node").read_text().strip())
        return value if value >= 0 else None
    except (OSError, ValueError):
        return None


def gpu_inventory():
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id"
    rows = csv.reader(_command(["nvidia-smi", f"--query-gpu={fields}",
                               "--format=csv,noheader,nounits"]).splitlines())
    processes = {}
    # Failure to inspect running jobs cannot be treated as an idle GPU.
    for row in csv.reader(_command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"]).splitlines()):
        if len(row) == 2:
            processes[row[0].strip()] = processes.get(row[0].strip(), 0) + 1
    result = []
    for row in rows:
        if len(row) != 7:
            raise RuntimeError(f"Unexpected GPU inventory row: {row!r}")
        index, uuid, name, total, used, utilization, bus = [part.strip() for part in row]
        result.append(GPUInfo(int(index), uuid, name, float(total), float(used),
                              float(utilization), bus, _numa_node(bus), processes.get(uuid, 0)))
    return result


def select_idle_gpu_pair(min_free_fraction=0.95, max_utilization=5, inventory=None,
                         max_used_memory_mib=8):
    devices = gpu_inventory() if inventory is None else inventory
    eligible = [gpu for gpu in devices if "L20" in gpu.name and gpu.compute_processes == 0
                and gpu.free_fraction >= min_free_fraction and gpu.memory_used_mib <= max_used_memory_mib
                and gpu.utilization_gpu <= max_utilization]
    pairs = list(itertools.combinations(eligible, 2))
    if not pairs:
        raise RuntimeError("Exactly two idle L20 GPUs are required; no qualifying pair")
    # Known same-NUMA wins; unknown affinity is explicitly retained in metadata.
    pairs.sort(key=lambda pair: (not (pair[0].numa_node is not None
                                     and pair[0].numa_node == pair[1].numa_node),
                                 -min(g.free_fraction for g in pair),
                                 tuple(g.index for g in pair)))
    return list(pairs[0])


class GpuLease:
    """Cooperating launches use persistent flock files (released on process exit).

    Also rechecks actual NVIDIA jobs after acquiring locks. This cannot reserve
    GPUs against unrelated programs that ignore locks; telemetry must catch those.
    """
    def __init__(self, gpu_ids, lock_dir="/tmp/rlvr-gpu-locks",
                 min_free_fraction=0.95, max_utilization=5, max_used_memory_mib=8,
                 idle_checks=3):
        self.gpu_ids = sorted(map(int, gpu_ids))
        if len(set(self.gpu_ids)) != 2:
            raise ValueError("A GPU lease must contain exactly two different devices")
        self.lock_dir = Path(lock_dir)
        self.min_free_fraction = min_free_fraction
        self.max_utilization = max_utilization
        self.max_used_memory_mib = max_used_memory_mib
        self.idle_checks = idle_checks
        self._handles = []
        self.metadata = {}

    def __enter__(self):
        import fcntl  # Training host is Linux; no unsafe Windows lock fallback.
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index in self.gpu_ids:
                handle = (self.lock_dir / f"gpu-{index}.lock").open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise
                self._handles.append(handle)
            for check in range(self.idle_checks):
                inventory = {gpu.index: gpu for gpu in gpu_inventory()}
                selected = [inventory[index] for index in self.gpu_ids]
                for gpu in selected:
                    if gpu.compute_processes or gpu.free_fraction < self.min_free_fraction \
                            or gpu.utilization_gpu > self.max_utilization \
                            or gpu.memory_used_mib > self.max_used_memory_mib:
                        raise RuntimeError(f"GPU {gpu.index} ceased to be idle while acquiring lease")
                if check+1 < self.idle_checks:
                    time.sleep(1)
            self.metadata = {"pid": os.getpid(), "host": socket.gethostname(),
                             "acquired_timestamp_ns": time.time_ns(),
                             "gpu_ids": self.gpu_ids, "devices": [asdict(gpu) for gpu in selected]}
            for handle in self._handles:
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(self.metadata) + "\n")
                handle.flush()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in reversed(self._handles):
            handle.close()
        self._handles.clear()


class GpuMonitor:
    def __init__(self, output_path, gpu_ids, interval=1.0):
        if interval != 1.0:
            raise ValueError("The frozen hardware probe uses one-second sampling")
        self.path = Path(output_path)
        self.gpu_ids = list(map(int, gpu_ids))
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.errors = []
        self._nvml = None

    def _initialize(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids}
        except Exception as error:
            self._nvml = None
            self._fallback_reason = f"{type(error).__name__}: {error}"

    def sample(self):
        wall, mono = time.time_ns(), time.monotonic_ns()
        result = []
        if self._nvml is not None:
            nvml = self._nvml
            for index, handle in self._handles.items():
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                uuid = nvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
                record = {"gpu_index": index, "gpu_uuid": uuid,
                          "memory_used_mib": memory.used / 2**20,
                          "memory_total_mib": memory.total / 2**20,
                          "utilization_gpu": float(utilization.gpu),
                          "utilization_memory": float(utilization.memory), "backend": "nvml"}
                try:
                    record["power_watts"] = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                except nvml.NVMLError:
                    record["power_watts"] = None
                result.append(record)
        else:
            fields = "index,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw"
            for row in csv.reader(_command(["nvidia-smi", f"--query-gpu={fields}",
                                           "--format=csv,noheader,nounits"]).splitlines()):
                index = int(row[0].strip())
                if index in self.gpu_ids:
                    result.append({"gpu_index": index, "gpu_uuid": row[1].strip(),
                                   "memory_total_mib": float(row[2]), "memory_used_mib": float(row[3]),
                                   "utilization_gpu": float(row[4]), "utilization_memory": float(row[5]),
                                   "power_watts": None if "N/A" in row[6] else float(row[6]),
                                   "backend": "nvidia-smi", "fallback_reason": self._fallback_reason})
        if {r["gpu_index"] for r in result} != set(self.gpu_ids):
            raise RuntimeError("Telemetry did not return all leased GPUs")
        for record in result:
            record.update(timestamp_ns=wall, monotonic_ns=mono,
                          measurement="gpu_active_time_percent_not_sm_occupancy")
        return result

    def _run(self):
        next_sample = time.monotonic()
        with self.path.open("a", encoding="utf-8") as stream:
            while not self._stop.is_set():
                try:
                    records = self.sample()
                except Exception as error:
                    self.errors.append(f"{type(error).__name__}: {error}")
                    records = [{"event": "telemetry_error", "timestamp_ns": time.time_ns(),
                                "monotonic_ns": time.monotonic_ns(), "error": self.errors[-1]}]
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                next_sample += self.interval
                self._stop.wait(max(0, next_sample - time.monotonic()))

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._thread = threading.Thread(target=self._run, name="rlvr-gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                self.errors.append("Telemetry thread did not finish")
        if self._nvml:
            self._nvml.nvmlShutdown()


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    left = int(position)
    return values[left] + (values[min(left + 1, len(values)-1)] - values[left]) * (position-left)


def time_weighted_utilization(samples, windows, gpu_ids, max_sample_gap_seconds=2.5):
    """Zero-order hold at actual timestamps, clipped to measured update windows.

    Missing intervals are excluded from the value and explicitly reduce coverage;
    a gate requires >=95% coverage on each leased GPU.
    """
    result = {}
    duration_ns = sum(end-start for start, end in windows)
    for index in gpu_ids:
        rows = sorted((r for r in samples if r.get("gpu_index") == index
                       and "utilization_gpu" in r), key=lambda r: r["monotonic_ns"])
        weighted = covered = 0
        for a, b in zip(rows, rows[1:]):
            start, end = a["monotonic_ns"], b["monotonic_ns"]
            if end-start > max_sample_gap_seconds*1e9 or not math.isfinite(a["utilization_gpu"]) \
                    or not 0 <= a["utilization_gpu"] <= 100:
                continue
            for left, right in windows:
                width = max(0, min(right, end)-max(left, start))
                covered += width
                weighted += width*a["utilization_gpu"]
        result[str(index)] = {"active_time_percent": weighted/covered if covered else None,
                              "full_window_lower_percent": weighted/duration_ns if duration_ns else None,
                              "full_window_upper_percent": (weighted+100*(duration_ns-covered))/duration_ns
                              if duration_ns else None,
                              "coverage": covered/duration_ns if duration_ns else 0,
                              "covered_seconds": covered/1e9}
    return result


def hardware_gate(updates, samples, diagnostics, gpu_ids, stage_records=None,
                  qualification_phase="benchmark"):
    """Evaluate actual steady updates. Unknowns cannot authorize R0.

    updates: update_id/start_monotonic_ns/end_monotonic_ns/status/prompt_count/
    response_count/group_count/generation_count/finite_update/optimizer_steps_by_rank/
    generated_tokens/oom, with optional stage_seconds/KV metrics.
    diagnostics: validated trace-analysis results, never caller asserted wall sums.
    """
    reasons, missing = [], []
    if len(set(gpu_ids)) != 2:
        missing.append("Exactly two leased GPUs must be identified")
    if qualification_phase not in ("benchmark", "reward_dev"):
        raise ValueError("Unsupported hardware qualification phase")
    steady = [u for u in updates if u.get("phase") == qualification_phase
              and u.get("measurement") == "steady"]
    if len(steady) < 10:
        missing.append(f"Need 10 steady complete updates, observed {len(steady)}")
    durations, windows = [], []
    required = ("update_id", "status", "prompt_count", "response_count", "group_count",
                "generation_count", "finite_update", "optimizer_steps_by_rank",
                "start_monotonic_ns", "end_monotonic_ns", "generated_tokens", "oom")
    for update in steady:
        absent = [key for key in required if key not in update]
        if absent:
            missing.append(f"Update {update.get('update_id')} lacks {absent}")
            continue
        expected_stages = ("rollout", "reward", "old_logprob", "ref_logprob",
                           "advantage", "actor_update", "weight_sync")
        stage_seconds = update.get("stage_seconds", {})
        for stage in expected_stages:
            value = stage_seconds.get(stage)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                missing.append(f"Update {update['update_id']} has no valid {stage} elapsed timing")
        kv = update.get("kv_cache")
        capacity_usage_keys = ("kv_cache_usage_fraction", "kv_cache_capacity_tokens", "num_gpu_blocks",
                               "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
        if (not isinstance(kv, dict) or kv.get("source") not in ("vllm_engine_metrics", "vllm_prometheus")
                or not isinstance(kv.get("observed_timestamp_ns"), int)
                or not any(isinstance(kv.get("metrics", {}).get(key), (float, int))
                           and math.isfinite(kv["metrics"][key]) for key in capacity_usage_keys)):
            missing.append(f"Update {update['update_id']} lacks observed vLLM KV-cache capacity/usage")
        if (update["status"] != "complete" or update["prompt_count"] != 32
                or update["group_count"] != 32 or update["response_count"] != 256
                or update["generation_count"] != 8 or update["finite_update"] is not True
                or update["oom"] is not False):
            reasons.append(f"Update {update['update_id']} violates complete 32x8 finite-update contract")
        steps = update["optimizer_steps_by_rank"]
        if not isinstance(steps, dict) or set(map(str, steps)) != {"0", "1"} \
                or any(value != 1 for value in steps.values()):
            reasons.append(f"Update {update['update_id']} has no exactly-one optimizer step on both ranks")
        start, end = update["start_monotonic_ns"], update["end_monotonic_ns"]
        if end <= start or not isinstance(update["generated_tokens"], int) or update["generated_tokens"] < 0:
            reasons.append(f"Update {update['update_id']} has invalid duration/token count")
            continue
        durations.append((end-start)/1e9)
        windows.append((start, end))
    if len({u.get("update_id") for u in steady}) != len(steady):
        reasons.append("Duplicate steady update IDs")
    windows.sort()
    if any(b[0] < a[1] for a, b in zip(windows, windows[1:])):
        reasons.append("Overlapping complete-update windows")
    ratio = percentile(durations, .95)/statistics.median(durations) if durations else None
    if ratio is not None and ratio > 1.5:
        reasons.append("P95/median complete-update time exceeds 1.5")
    telemetry = time_weighted_utilization(samples, windows, gpu_ids)
    for index, metric in telemetry.items():
        if metric["coverage"] < .95:
            missing.append(f"GPU {index} telemetry coverage below 95%")
    values = [r["active_time_percent"] for r in telemetry.values()]
    utilization = sum(values)/len(values) if values and all(v is not None for v in values) else None
    lower_values = [r["full_window_lower_percent"] for r in telemetry.values()]
    utilization_lower = sum(lower_values)/len(lower_values) if lower_values and all(
        value is not None for value in lower_values) else None
    upper_values = [r["full_window_upper_percent"] for r in telemetry.values()]
    utilization_upper = sum(upper_values)/len(upper_values) if upper_values and all(
        value is not None for value in upper_values) else None
    if utilization_lower is not None and utilization_lower < 40:
        if utilization_upper is not None and utilization_upper < 40:
            reasons.append("Two-GPU full-window active-time upper bound below 40%")
        else:
            missing.append("Two-GPU full-window active-time bounds straddle 40%")
    valid_diagnostics = []
    for diagnostic in diagnostics:
        if diagnostic.get("schema") != "rlvr-trace-offload-v1" \
                or diagnostic.get("evidence_status") != "complete" \
                or not diagnostic.get("trace_sha256") or not diagnostic.get("trace_path"):
            continue
        capture = diagnostic.get("capture_evidence") or {}
        if capture.get("capture_status") != "complete" or capture.get("actor_completed") is not True \
                or capture.get("profiler_completed") is not True or capture.get("observed_trace_loss_messages"):
            missing.append("Diagnostic lacks successful actor/profiler capture evidence")
            continue
        path = Path(diagnostic["trace_path"])
        if not path.is_file():
            missing.append(f"Diagnostic raw trace is missing: {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != diagnostic["trace_sha256"]:
            missing.append(f"Diagnostic trace hash changed: {path}")
            continue
        valid_diagnostics.append(diagnostic)
    by_update = {}
    for diagnostic in valid_diagnostics:
        bucket = by_update.setdefault(diagnostic.get("update_id"), {})
        if diagnostic.get("rank") in bucket:
            missing.append(f"Duplicate diagnostic trace for update {diagnostic.get('update_id')} rank {diagnostic.get('rank')}")
        bucket[diagnostic.get("rank")] = diagnostic
    complete_diagnostic_updates = {u: ranks for u, ranks in by_update.items()
                                  if set(ranks) == {0, 1}}
    if len(complete_diagnostic_updates) < 3:
        missing.append("Need at least three actor CPU+CUDA diagnostic traces on each rank")
    for update, ranks in complete_diagnostic_updates.items():
        for rank, diagnostic in ranks.items():
            value = diagnostic.get("offload_critical_path_upper_fraction")
            if value is None or not 0 <= value <= 1:
                missing.append(f"Diagnostic update {update} rank {rank} has no valid offload upper bound")
            elif value > .4:
                lower = diagnostic.get("offload_critical_path_lower_fraction")
                if lower is not None and 0 <= lower <= value and lower > .4:
                    reasons.append(f"Diagnostic update {update} rank {rank} verified offload lower bound exceeds 40%")
                else:
                    missing.append(f"Diagnostic update {update} rank {rank} offload bounds straddle 40%")
    if missing:
        decision = "insufficient_evidence"
    elif reasons:
        decision = "technically_runs_only"
    else:
        decision = "suitable_for_first_round"
    total_seconds = sum(durations)
    token_total = sum(u.get("generated_tokens", 0) for u in steady
                      if isinstance(u.get("generated_tokens", 0), int))
    response_total = sum(u.get("response_count", 0) for u in steady
                         if isinstance(u.get("response_count", 0), int))
    report = {"schema": "rlvr-hardware-gate-v1", "decision": decision,
              "qualification_phase": qualification_phase,
              "r0_authorized": decision == "suitable_for_first_round",
              "gate_failures": reasons, "missing_evidence": missing,
              "steady_update_count": len(steady), "update_seconds": durations,
              "update_median_seconds": statistics.median(durations) if durations else None,
              "update_p95_seconds": percentile(durations, .95), "p95_median_ratio": ratio,
              "gpu_active_time": telemetry, "two_gpu_mean_active_time_percent": utilization,
              "two_gpu_full_window_active_time_lower_percent": utilization_lower,
              "two_gpu_full_window_active_time_upper_percent": utilization_upper,
              "utilization_definition": "NVML active time percentage, not SM occupancy",
              "diagnostic_complete_update_count": len(complete_diagnostic_updates),
              "tokens_per_second_complete_update": token_total/total_seconds
              if total_seconds else None,
              "responses_per_second_complete_update": response_total/total_seconds
              if total_seconds else None,
              "peak_memory_used_mib_by_gpu": {str(i): max((s.get("memory_used_mib", 0)
                  for s in samples if s.get("gpu_index") == i), default=None) for i in gpu_ids}}
    stages = {}
    for update in steady:
        for stage, seconds in update.get("stage_seconds", {}).items():
            if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
                stages.setdefault(stage, []).append(seconds)
    report["stage_timings"] = {name: {"observed_count": len(values),
                                      "median_seconds": statistics.median(values),
                                      "p95_seconds": percentile(values, .95)} for name, values in stages.items()}
    rollout_times = [u.get("stage_seconds", {}).get("rollout") for u in steady]
    rollout_valid = [value for value in rollout_times if isinstance(value, (int, float))
                     and math.isfinite(value) and value >= 0]
    rollout_seconds = sum(rollout_valid)
    rollout_coverage = len(rollout_valid)
    report["rollout_tokens_per_second"] = token_total/rollout_seconds \
        if rollout_seconds > 0 and rollout_coverage == len(steady) else None
    report["rollout_responses_per_second"] = response_total/rollout_seconds \
        if rollout_seconds > 0 and rollout_coverage == len(steady) else None
    report["average_response_tokens"] = token_total/response_total if response_total and all(
        "generated_tokens" in u for u in steady) else None
    report["kv_cache_records"] = [r for r in (stage_records or []) if "kv_cache" in r]
    return report
