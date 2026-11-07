"""Real CPU/CUDA traces and conservative offload critical-path evidence.

The analyzer uses unions of time intervals, not sums of asynchronous Python
durations. Unknown idle/wait/copy time is charged to the upper bound. Only
same-thread blocking waits overlapping correlated offload copies contribute to
the lower bound. Bounds straddling the gate threshold do not authorize training.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import warnings

_local = threading.local()


def stage_event(event, stage, run_dir, phase, update_id, rank=None, **extra):
    path = Path(run_dir) / "stage_events" / f"rank{rank}-pid{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "stage": stage, "phase": phase,
              "update_id": update_id, "rank": rank, "pid": os.getpid(),
              "timestamp_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(), **extra}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    return record


@contextmanager
def _range(name):
    import torch
    with ExitStack() as stack:
        stack.enter_context(torch.profiler.record_function(name))
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield


@contextmanager
def profile_stage(stage, run_dir, phase, update_id, rank=None):
    import torch
    diagnostic = getattr(_local, "diagnostic", False)
    if diagnostic and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = stage_event("start", stage, run_dir, phase, update_id, rank)
    error = None
    with _range(f"rlvr.stage.{stage}"):
        try:
            yield
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            if diagnostic and torch.cuda.is_available():
                # Diagnostic-only boundary: captures trailing asynchronous work.
                torch.cuda.synchronize()
            end = stage_event("end", stage, run_dir, phase, update_id, rank,
                              duration_s=(time.monotonic_ns()-start["monotonic_ns"])/1e9,
                              status="failed" if error else "complete", error=error,
                              diagnostic=diagnostic)
            if stage == "actor_update" and diagnostic:
                _local.actor_stage_end = end


@contextmanager
def offload_range(name):
    with _range(f"rlvr.offload.{name}"):
        yield


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _intersection(left, right):
    left, right = _union(left), _union(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if max(a, c) < min(b, d):
            result.append((max(a, c), min(b, d)))
        if b <= d:
            i += 1
        else:
            j += 1
    return result


def _subtract(left, right):
    remaining = _union(left)
    for a, b in _union(right):
        next_remaining = []
        for c, d in remaining:
            if b <= c or a >= d:
                next_remaining.append((c, d))
            else:
                if c < a:
                    next_remaining.append((c, a))
                if b < d:
                    next_remaining.append((b, d))
        remaining = next_remaining
    return remaining


def _duration(intervals):
    return sum(b-a for a, b in _union(intervals))


def _key(event, name):
    for key, value in event.get("args", {}).items():
        if key.lower().replace(" ", "").replace("_", "") == name:
            return str(value)
    return None


def read_offload_policy(config_path):
    """Policy proof comes from the run's frozen *resolved* JSON configuration."""
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    actor = config["actor_rollout_ref"]["actor"]
    fsdp = actor.get("fsdp_config", actor.get("engine", {}))
    names = ("param_offload", "optimizer_offload", "offload_policy")
    flags = {name: fsdp.get(name) for name in names}
    return {"resolved_config_path": str(path.resolve()),
            "resolved_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "actor_offload_flags": flags,
            "all_actor_offload_disabled": all(value is False for value in flags.values())}


def observed_trace_loss_messages(text):
    """Detect reported loss/unavailable tracing, without claiming loss is impossible."""
    patterns = (r"(?:cupti|kineto|profiler|activity).{0,160}(?:dropped|lost|overflow|max_limit_reached|unavailable)",
                r"(?:dropped|lost)\s+(?:\d+\s+)?(?:cuda|cupti|activity|activities|trace|profiling|events|records)")
    result = []
    for line in text.splitlines():
        if re.search(r"(?:dropped|lost)\s+0\s+(?:events|records|activities)", line, re.I):
            continue
        if any(re.search(pattern, line, re.I) for pattern in patterns):
            result.append(line)
    return result


@contextmanager
def _native_stderr_capture(path):
    """Capture fd 2 so CUPTI/Kineto native warnings are preserved as evidence.

    The synchronous training worker owns this process; concurrent messages are
    intentionally included. The original stderr also receives the captured text.
    """
    saved_fd = os.dup(2)
    with Path(path).open("wb") as stream:
        os.dup2(stream.fileno(), 2)
        try:
            yield stream
        finally:
            stream.flush()
            os.fsync(stream.fileno())
            os.dup2(saved_fd, 2)
            data = Path(path).read_bytes()
            position = 0
            try:
                while position < len(data):
                    position += os.write(saved_fd, data[position:position+65536])
            finally:
                os.close(saved_fd)


def _capture_evidence(path, capture, trace_hash, update_id, rank):
    if capture is None:
        sidecar = path.with_suffix(".capture.json")
        if not sidecar.is_file():
            return None, ["Missing profiler/native-stderr capture evidence"]
        capture = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(capture, dict):
        return None, ["Invalid trace capture evidence shape"]
    failures = []
    if capture.get("schema") != "rlvr-trace-capture-v1" or capture.get("capture_status") != "complete" \
            or capture.get("profiler_completed") is not True or capture.get("actor_completed") is not True:
        failures.append("Profiler or actor did not complete successfully")
    if capture.get("update_id") != update_id or capture.get("rank") != int(rank):
        failures.append("Trace capture rank/update identity mismatch")
    if capture.get("trace_sha256") != trace_hash:
        failures.append("Trace differs from its completed capture checksum")
    stage = capture.get("actor_stage_end") or {}
    if stage.get("stage") != "actor_update" or stage.get("event") != "end" \
            or stage.get("status") != "complete" or stage.get("error") is not None \
            or stage.get("diagnostic") is not True or stage.get("update_id") != update_id \
            or stage.get("rank") != int(rank):
        failures.append("Missing successful diagnostic actor stage-end evidence")
    stderr = Path(capture.get("native_stderr_path", "__missing__"))
    if not stderr.is_file():
        stderr = path.with_suffix(".stderr.log")
    if not stderr.is_file():
        failures.append("Native profiler stderr evidence is missing")
    elif hashlib.sha256(stderr.read_bytes()).hexdigest() != capture.get("native_stderr_sha256"):
        failures.append("Native profiler stderr checksum mismatch")
    else:
        losses = observed_trace_loss_messages(stderr.read_text(encoding="utf-8", errors="replace") + "\n" +
                                              "\n".join(capture.get("python_warnings", [])))
        if losses or capture.get("observed_trace_loss_messages"):
            failures.append("Observed CUPTI/Kineto loss or unavailable tracing: " + "; ".join(losses or capture["observed_trace_loss_messages"]))
    return capture, failures


def _bookkeeping_copies(cpu_events, copies):
    """Recognize only correlated <=16-byte DtoH scalar reads, not arbitrary copies."""
    scalar = [event for event in cpu_events if event.get("name") in ("aten::item", "aten::_local_scalar_dense")]
    from .trace_stream import ContainmentIndex
    index = ContainmentIndex(scalar)
    identifiers = set()
    for event in cpu_events:
        if index.contains(event):
            for kind in ("externalid", "correlation"):
                if _key(event, kind) is not None:
                    identifiers.add((kind, _key(event, kind)))
    recognized = []
    for event in copies:
        byte_count = _key(event, "bytes")
        try:
            numeric_bytes = float(byte_count)
            byte_count = int(numeric_bytes)
            if numeric_bytes != byte_count:
                continue
        except (ValueError, TypeError):
            continue
        direction = str(event.get("name", "")).lower() + " " + str(event.get("args", {})).lower()
        if ("dtoh" in direction or "device to host" in direction or "devicetohost" in direction) \
                and 0 < byte_count <= 16 and any((kind, _key(event, kind)) in identifiers
                                                for kind in ("externalid", "correlation")):
            recognized.append(event)
    return recognized


def analyze_trace(trace_path, update_id, rank, offload_policy=None, capture=None):
    from .trace_stream import analyze_cached
    return analyze_cached(trace_path, update_id, rank, offload_policy, capture)


def _analyze_trace_legacy(trace_path, update_id, rank, offload_policy=None, capture=None):
    from .trace_stream import ContainmentIndex, file_sha256
    path = Path(trace_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload if isinstance(payload, list) else payload.get("traceEvents", [])
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    actor = [e for e in complete if e.get("name") == "rlvr.stage.actor_update"]
    result = {"schema": "rlvr-trace-offload-v1", "update_id": update_id, "rank": int(rank),
              "trace_path": str(path.resolve()), "trace_sha256": file_sha256(path),
              "evidence_status": "insufficient", "limitations": [],
              "method": "interval-union conservative bounds; no asynchronous wall-time sums"}
    capture, capture_failures = _capture_evidence(path, capture, result["trace_sha256"], update_id, rank)
    result["capture_evidence"] = capture
    result["loss_detection_scope"] = "Observed native stderr/Python warnings and trace activities; cannot prove absolute absence of CUPTI loss"
    if capture_failures:
        result["limitations"].extend(capture_failures)
        return result
    if len(actor) != 1:
        result["limitations"].append("Trace requires exactly one instrumented actor_update interval")
        return result
    actor = actor[0]
    window = [(actor["ts"], actor["ts"]+actor["dur"])]
    actor_seconds = actor["dur"]/1e6
    def interval(event):
        return _intersection([(event["ts"], event["ts"]+event["dur"])], window)
    gpu = [e for e in complete if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and interval(e)]
    kernels = [e for e in gpu if e.get("cat") == "kernel"]
    if not kernels:
        result["limitations"].append("No CUDA kernels in actor interval: CPU-only/unusable CUDA trace")
        return result
    runtime = [e for e in complete if e.get("cat") in ("cuda_runtime", "cuda_driver") and interval(e)]
    if not runtime:
        result["limitations"].append("Missing CUDA runtime/driver API activities in actor trace")
        return result
    copies = [e for e in gpu if e.get("cat") == "gpu_memcpy"]
    cpu_events = [e for e in complete if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset")]
    bookkeeping = _bookkeeping_copies(cpu_events, copies)
    result["bookkeeping_copies"] = [{"name": e["name"], "ts_us": e["ts"], "duration_us": e["dur"],
                                       "bytes": int(_key(e, "bytes")), "correlation": _key(e, "correlation"),
                                       "external_id": _key(e, "externalid")} for e in bookkeeping]
    copies = [e for e in copies if e not in bookkeeping]
    if offload_policy and offload_policy.get("all_actor_offload_disabled"):
        try:
            actual = read_offload_policy(offload_policy["resolved_config_path"])
        except (OSError, KeyError, ValueError) as error:
            result["limitations"].append(f"Cannot verify frozen offload config: {error}")
            return result
        if actual != offload_policy:
            result["limitations"].append("Offload policy differs from the frozen resolved config hash/flags")
            return result
        if copies:
            result["limitations"].append("Disabled offload config has unclassified GPU transfers")
            return result
        result.update(evidence_status="complete", actor_update_seconds=actor_seconds,
                      offload_critical_path_lower_seconds=0., offload_critical_path_upper_seconds=0.,
                      offload_critical_path_lower_fraction=0., offload_critical_path_upper_fraction=0.,
                      offload_policy=actual, cuda_kernel_count=len(kernels), attributed_copy_count=0,
                      limitations=["No actor offload by frozen config; only proven scalar bookkeeping copies excluded",
                                   "This policy concerns actor_update; ref and rollout are separate stages"])
        return result
    ranges = [e for e in complete if str(e.get("name", "")).startswith("rlvr.offload.")
              and interval(e)]
    if not ranges:
        result["limitations"].append("No observed instrumented offload operations; cannot certify offload policy")
        return result
    # Associate descendant launches/ops with range by thread and containment,
    # then match their external IDs/correlations to GPU CUPTI activities.
    identifiers = set()
    range_index = ContainmentIndex(ranges)
    for cpu in complete:
        if cpu.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        if range_index.contains(cpu):
            for kind in ("externalid", "correlation"):
                value = _key(cpu, kind)
                if value is not None:
                    identifiers.add((kind, value))
    attributed = [e for e in copies if any((kind, _key(e, kind)) in identifiers
                                          for kind in ("externalid", "correlation"))]
    if not attributed:
        result["limitations"].append("No offload GPU transfers could be linked to instrumented CPU ranges")
        return result
    # Any memcpy may be offload until proved otherwise (e.g. NCCL staging).
    # The idle complement includes unknown CPU work; it is intentionally a
    # conservative upper bound, not a claim that those gaps are CPU waiting.
    activity = _union([item for event in gpu for item in interval(event)])
    copy_intervals = _union([item for event in copies for item in interval(event)])
    attributed_intervals = _union([item for event in attributed for item in interval(event)])
    idle = _subtract(window, activity)
    sync_names = ("synchronize",)
    blocking = [e for e in complete if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and any(word in str(e.get("name", "")).lower() for word in sync_names)
                and interval(e)]
    blocking_intervals = _union([item for event in blocking for item in interval(event)])
    runtime_intervals = _union([item for event in runtime for item in interval(event)])
    # Lower bound requires a blocking actor thread AND correlated copy activity
    # AND no overlapping kernel. A Python offload marker alone never proves wait.
    same_thread = _union([item for event in blocking if event.get("pid") == actor.get("pid")
                          and event.get("tid") == actor.get("tid") for item in interval(event)])
    kernel_intervals = _union([item for event in kernels for item in interval(event)])
    lower = _subtract(_intersection(attributed_intervals, same_thread), kernel_intervals)
    cpu_offload = _union([item for event in ranges for item in interval(event)])
    upper = _union(copy_intervals + idle + runtime_intervals + cpu_offload)
    lower_seconds, upper_seconds = _duration(lower)/1e6, _duration(upper)/1e6
    result.update(evidence_status="complete", actor_update_seconds=actor_seconds,
                  offload_critical_path_lower_seconds=lower_seconds,
                  offload_critical_path_upper_seconds=upper_seconds,
                  offload_critical_path_lower_fraction=lower_seconds/actor_seconds,
                  offload_critical_path_upper_fraction=upper_seconds/actor_seconds,
                  attributed_copy_union_seconds=_duration(attributed_intervals)/1e6,
                  all_copy_union_seconds=_duration(copy_intervals)/1e6,
                  unclassified_idle_upper_seconds=_duration(idle)/1e6,
                  all_sync_union_seconds=_duration(blocking_intervals)/1e6,
                  all_cuda_api_union_seconds=_duration(runtime_intervals)/1e6,
                  offload_cpu_enqueue_or_wait_union_seconds=_duration(cpu_offload)/1e6,
                  observed_offload_range_count=len(ranges), attributed_copy_count=len(attributed),
                  cuda_kernel_count=len(kernels),
                  limitations=["Bounds are conservative, not an exact CPU-wait measurement",
                               "Unknown idle and all CUDA API host-call time is included only in the upper bound",
                               "Diagnostic synchronization/profiler overhead is not steady throughput"])
    return result


@contextmanager
def diagnostic_profile(run_dir, update_id, rank, enabled=True, resolved_config_path=None):
    if not enabled:
        yield
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Diagnostic update requires CPU+CUDA profiling, not CPU-only evidence")
    directory = Path(run_dir) / "traces"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"update{update_id}-rank{rank}-pid{os.getpid()}.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing diagnostic trace {path}")
    previous = getattr(_local, "diagnostic", False)
    previous_actor = getattr(_local, "actor_stage_end", None)
    _local.diagnostic = True
    _local.actor_stage_end = None
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                      record_shapes=False, profile_memory=False, with_stack=False)
    stderr_path = path.with_suffix(".stderr.log")
    actor_completed = profiler_completed = False
    failure = None
    try:
        with _native_stderr_capture(stderr_path) as stderr_stream:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                try:
                    with profiler:
                        yield
                        actor_completed = True
                    profiler_completed = True
                except BaseException as error:
                    failure = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    profiler.export_chrome_trace(str(path))
                    stderr_stream.flush()
                    os.fsync(stderr_stream.fileno())
                    warning_text = [str(item.message) for item in caught_warnings]
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                    from .trace_stream import file_sha256
                    capture = {"schema": "rlvr-trace-capture-v1", "update_id": update_id,
                               "rank": int(rank), "pid": os.getpid(),
                               "capture_status": "complete" if actor_completed and profiler_completed else "failed",
                               "actor_completed": actor_completed, "profiler_completed": profiler_completed,
                               "actor_stage_end": _local.actor_stage_end, "failure": failure,
                               "trace_sha256": file_sha256(path),
                               "native_stderr_path": str(stderr_path.resolve()),
                               "native_stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
                               "python_warnings": warning_text,
                               "observed_trace_loss_messages": observed_trace_loss_messages(stderr_text+"\n"+"\n".join(warning_text)),
                               "loss_detection_scope": "Observed native stderr/Python warnings; not proof of absolute no-loss"}
                    path.with_suffix(".capture.json").write_text(json.dumps(capture, indent=2)+"\n", encoding="utf-8")
                    config_path = resolved_config_path or os.environ.get("RLVR_RESOLVED_CONFIG_PATH")
                    policy = read_offload_policy(config_path) if config_path else None
                    # Analysis runs in the separate CPU report process after committed
                    # diagnostics. A queued capture never qualifies as analyzed evidence.
                    analysis = {"schema": "rlvr-trace-offload-v1", "update_id": update_id,
                                "rank": int(rank), "trace_path": str(path.resolve()),
                                "trace_sha256": capture["trace_sha256"], "offload_policy": policy,
                                "evidence_status": "queued", "capture_evidence": capture}
                    path.with_suffix(".offload.json").write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n",
                                                               encoding="utf-8")
    finally:
        _local.diagnostic = previous
        _local.actor_stage_end = previous_actor
