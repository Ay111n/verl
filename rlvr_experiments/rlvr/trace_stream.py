"""Bounded-memory trace analysis and exact thread-local containment indexes."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

VERSION = "rlvr-stream-trace-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContainmentIndex:
    def __init__(self, regions):
        grouped = {}
        for region in regions:
            grouped.setdefault((region.get("pid"), region.get("tid")), []).append(
                (region["ts"], region["ts"] + region["dur"]))
        self.groups = {}
        for key, intervals in grouped.items():
            starts, ends, maximum = [], [], float("-inf")
            for start, end in sorted(intervals):
                maximum = max(maximum, end)
                starts.append(start)
                ends.append(maximum)
            self.groups[key] = starts, ends

    def contains(self, event):
        pair = self.groups.get((event.get("pid"), event.get("tid")))
        if pair is None:
            return False
        starts, ends = pair
        position = bisect_right(starts, event["ts"]) - 1
        return position >= 0 and ends[position] >= event["ts"] + event["dur"]


def events(path):
    import ijson
    with Path(path).open("rb") as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        stream.seek(0)
        prefix = "item" if first == b"[" else "traceEvents.item"
        for event in ijson.items(stream, prefix, use_float=True):
            if event.get("ph") == "X" and event.get("dur", 0) > 0:
                yield event


def disabled_offload_analysis(path, update_id, rank, policy, capture, trace_hash):
    from .profiling import _capture_evidence, _key, read_offload_policy
    result = {"schema": "rlvr-trace-offload-v1", "update_id": update_id, "rank": int(rank),
              "trace_path": str(path.resolve()), "trace_sha256": trace_hash,
              "evidence_status": "insufficient", "limitations": [],
              "method": "interval-union conservative bounds; no asynchronous wall-time sums"}
    capture, failures = _capture_evidence(path, capture, trace_hash, update_id, rank)
    result["capture_evidence"] = capture
    result["loss_detection_scope"] = "Observed native stderr/Python warnings and trace activities; cannot prove absolute absence of CUPTI loss"
    if failures:
        result["limitations"].extend(failures)
        return result
    actor, scalar = [], []
    gpu_categories = ("kernel", "gpu_memcpy", "gpu_memset")
    for event in events(path):
        if event.get("name") == "rlvr.stage.actor_update":
            actor.append(event)
        if event.get("cat") not in gpu_categories and event.get("name") in ("aten::item", "aten::_local_scalar_dense"):
            scalar.append({key: event.get(key) for key in ("pid", "tid", "ts", "dur")})
    if len(actor) != 1:
        result["limitations"].append("Trace requires exactly one instrumented actor_update interval")
        return result
    actor = actor[0]
    left, right = actor["ts"], actor["ts"] + actor["dur"]
    index = ContainmentIndex(scalar)
    identifiers, copies, kernels, runtime = set(), [], 0, False
    for event in events(path):
        category = event.get("cat")
        overlaps = event["ts"] < right and event["ts"] + event["dur"] > left
        if overlaps:
            if category == "kernel":
                kernels += 1
            elif category == "gpu_memcpy":
                copies.append(event)
            elif category in ("cuda_runtime", "cuda_driver"):
                runtime = True
        if category not in gpu_categories and index.contains(event):
            for kind in ("externalid", "correlation"):
                value = _key(event, kind)
                if value is not None:
                    identifiers.add((kind, value))
    if not kernels:
        result["limitations"].append("No CUDA kernels in actor interval: CPU-only/unusable CUDA trace")
        return result
    if not runtime:
        result["limitations"].append("Missing CUDA runtime/driver API activities in actor trace")
        return result
    bookkeeping = []
    for event in copies:
        try:
            numeric = float(_key(event, "bytes"))
            count = int(numeric)
            if count != numeric:
                continue
        except (ValueError, TypeError, OverflowError):
            continue
        direction = str(event.get("name", "")).lower() + " " + str(event.get("args", {})).lower()
        if any(word in direction for word in ("dtoh", "device to host", "devicetohost")) and 0 < count <= 16 \
                and any((kind, _key(event, kind)) in identifiers for kind in ("externalid", "correlation")):
            bookkeeping.append(event)
    result["bookkeeping_copies"] = [{"name": e["name"], "ts_us": e["ts"], "duration_us": e["dur"],
                                    "bytes": int(_key(e, "bytes")), "correlation": _key(e, "correlation"),
                                    "external_id": _key(e, "externalid")} for e in bookkeeping]
    try:
        actual = read_offload_policy(policy["resolved_config_path"])
    except (OSError, KeyError, ValueError) as error:
        result["limitations"].append(f"Cannot verify frozen offload config: {error}")
        return result
    if actual != policy:
        result["limitations"].append("Offload policy differs from the frozen resolved config hash/flags")
        return result
    if any(event not in bookkeeping for event in copies):
        result["limitations"].append("Disabled offload config has unclassified GPU transfers")
        return result
    result.update(evidence_status="complete", actor_update_seconds=actor["dur"] / 1e6,
                  offload_critical_path_lower_seconds=0., offload_critical_path_upper_seconds=0.,
                  offload_critical_path_lower_fraction=0., offload_critical_path_upper_fraction=0.,
                  offload_policy=actual, cuda_kernel_count=kernels, attributed_copy_count=0,
                  limitations=["No actor offload by frozen config; only proven scalar bookkeeping copies excluded",
                               "This policy concerns actor_update; ref and rollout are separate stages"])
    return result


def analyze_cached(path, update_id, rank, policy=None, capture=None):
    from .profiling import _analyze_trace_legacy
    path = Path(path)
    trace_hash = file_sha256(path)
    sidecar = path.with_suffix(".capture.json")
    bound_capture = capture if capture is not None else (json.loads(sidecar.read_text()) if sidecar.exists() else None)
    stderr = Path((bound_capture or {}).get("native_stderr_path", "__missing__"))
    if not stderr.is_file():
        stderr = path.with_suffix(".stderr.log")
    policy_path = Path((policy or {}).get("resolved_config_path", "__missing__"))
    binding = {"version": VERSION, "analyzer_sha256": file_sha256(__file__),
               "profiling_sha256": file_sha256(Path(__file__).with_name("profiling.py")),
               "trace_sha256": trace_hash, "update_id": update_id, "rank": int(rank),
               "policy": policy, "capture": bound_capture,
               "actual_policy_sha256": file_sha256(policy_path) if policy_path.is_file() else None,
               "actual_stderr_sha256": file_sha256(stderr) if stderr.is_file() else None}
    cache_path = path.with_suffix(".analysis-cache.json")
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            result_hash = hashlib.sha256(json.dumps(cache["result"], sort_keys=True).encode()).hexdigest()
            if cache.get("binding") == binding and cache.get("result_sha256") == result_hash:
                return cache["result"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if policy and policy.get("all_actor_offload_disabled"):
        result = disabled_offload_analysis(path, update_id, rank, policy, capture, trace_hash)
    else:
        # Enabled-offload attribution retains the conservative interval-union path.
        result = _analyze_trace_legacy(path, update_id, rank, policy, capture)
    cache = {"binding": binding, "result": result,
             "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    temporary.replace(cache_path)
    return result
