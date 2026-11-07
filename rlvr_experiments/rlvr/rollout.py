"""External vLLM observer and manifest-seeded single-turn agent."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import time
from pathlib import Path

_capture = contextvars.ContextVar("rlvr_backend_capture", default=None)


class EngineObserver:
    """Transparent proxy: preserve every backend yield and observe native metadata."""
    def __init__(self, engine):
        self.engine = engine
        self.last_kv_sample_ns = 0
        self.latest_kv_sample = None

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def generate(self, *args, **kwargs):
        async def observe():
            target = _capture.get()
            params = kwargs.get("sampling_params")
            if target is not None:
                target["effective_max_tokens"] = getattr(params, "max_tokens", None)
                target["native_sampling_seed"] = getattr(params, "seed", None)
                target["native_sampling_params"] = {name: getattr(params, name, None) for name in (
                    "temperature", "top_p", "top_k", "seed", "max_tokens", "min_tokens",
                    "repetition_penalty", "presence_penalty", "frequency_penalty", "ignore_eos", "detokenize")}
                target["backend_request_id"] = kwargs.get("request_id")
                lora = kwargs.get("lora_request")
                target["native_lora_request"] = {"enabled": lora is not None,
                    "lora_int_id": getattr(lora, "lora_int_id", None),
                    "lora_name": getattr(lora, "lora_name", None)}
                if os.environ.get('RLVR_ADAPTER_SYNC_AUDIT') == '1' and getattr(params, 'temperature', None) == 0:
                    from rlvr.adapter_audit import audit_state
                    target['native_adapter_audit'] = audit_state(require_trained=True, include_source=False)
                    target['native_batch_invariant'] = os.environ.get('VLLM_BATCH_INVARIANT') == '1'
            async for output in self.engine.generate(*args, **kwargs):
                if os.environ.get("RLVR_RUN_DIR"):
                    now = time.monotonic_ns()
                    if now - self.last_kv_sample_ns >= 500_000_000:
                        self.latest_kv_sample = observe_kv_cache(self.engine)
                        self.last_kv_sample_ns = now
                if target is not None:
                    if self.latest_kv_sample is not None:
                        target["kv_cache"] = self.latest_kv_sample
                    completions = output.outputs
                    target["backend_abort"] = not bool(completions)
                    if completions:
                        completion = completions[0]
                        target.update(native_finish_reason=getattr(completion, "finish_reason", None),
                                      native_stop_reason=getattr(completion, "stop_reason", None),
                                      raw_response_token_ids=list(completion.token_ids))
                        target["backend_abort"] = target["native_finish_reason"] == "abort"
                    else:
                        target.update(native_finish_reason=None, native_stop_reason=None,
                                      raw_response_token_ids=[])
                yield output
        return observe()


def observe_kv_cache(engine):
    """Use the actual vLLM19 Prometheus snapshot API, never infer cache capacity."""
    from dataclasses import asdict
    from vllm.v1.metrics.reader import get_metrics_snapshot
    selected = [asdict(metric) for metric in get_metrics_snapshot()
                if metric.name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_preemptions", "vllm:num_preemptions_total")]
    values = [float(metric["value"]) for metric in selected if metric["name"] in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")]
    metrics = {metric["name"]: float(metric["value"]) for metric in selected}
    if values:
        metrics["kv_cache_usage_fraction"] = max(values)
    config = getattr(getattr(engine, "vllm_config", None), "cache_config", None)
    blocks = getattr(config, "num_gpu_blocks", None)
    if blocks is not None:
        metrics["num_gpu_blocks"] = int(blocks)
    path = Path(os.environ["RLVR_RUN_DIR"]) / "kv_metrics" / f"server-pid{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"source": "vllm_prometheus", "observed_timestamp_ns": time.time_ns(),
              "observed_monotonic_ns": time.monotonic_ns(), "metrics": metrics,
              "raw_metrics": selected, "raw_metrics_path": str(path),
              "sampling_period_seconds": .5,
              "status": "observed" if metrics else "unavailable"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
    if not metrics:
        raise RuntimeError(f"Native KV cache statistics unavailable; raw observation saved to {path}")
    return record


class RequestClient:
    def __init__(self, client, request_id, seed, priority):
        self.client, self.request_id, self.seed, self.priority = client, request_id, seed, priority

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def generate(self, **kwargs):
        params = dict(kwargs["sampling_params"])
        params["seed"] = self.seed
        kwargs.update(request_id=self.request_id, priority=self.priority, sampling_params=params)
        output = await self.client.generate(**kwargs)
        output.extra_fields.update(rlvr_request_id=self.request_id, sampling_seed=self.seed,
                                   actual_sampling_params=params)
        return output


def request_identity(namespace, sample_id, generation, update_id):
    value = json.dumps([namespace, str(sample_id), int(generation), int(update_id)],
                       ensure_ascii=False, separators=(",", ":"))
    return "rlvr-" + hashlib.sha256(value.encode()).hexdigest()


def request_sampling_params(params, eval_mode, eval_config):
    """Apply the evaluation contract per request, independent of worker config copies."""
    result = dict(params)
    if eval_mode == "greedy":
        result.update(temperature=0.0, top_p=1.0, top_k=-1)
    elif eval_mode == "sampled_g8":
        result.update(temperature=float(eval_config.temperature), top_p=float(eval_config.top_p),
                      top_k=int(eval_config.top_k))
    elif eval_mode is not None:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    return result


def sampling_contract_matches(actual, expected):
    # vLLM 0.19 canonicalizes disabled top-k from -1 to 0, including greedy mode.
    def top_k(value):
        return 0 if value in (-1, 0) else value
    return (actual.get("temperature") == expected["temperature"]
            and actual.get("top_p") == expected["top_p"]
            and top_k(actual.get("top_k")) == top_k(expected["top_k"]))


def build_agent_class():
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from rlvr.contracts import stable_seed

    class SeededSingleTurnAgentLoop(SingleTurnAgentLoop):
        async def run(self, sampling_params, priority=0, **kwargs):
            generation = int(kwargs["session_id"])
            extra = kwargs.get("extra_info") or {}
            sample_id = kwargs.get("sample_id", extra.get("sample_id"))
            if sample_id is None:
                raise RuntimeError("Dataset must retain sample_id for deterministic sampling")
            seeds = kwargs.get("rlvr_seeds")
            namespace = "rlvr-train-seed-v1" if seeds is not None else "rlvr-eval-seed-v1"
            seed = int(seeds[generation]) if seeds is not None else stable_seed(
                sample_id, generation, base_seed=int(self.config.rlvr.base_seed), namespace=namespace)
            update_id = int(kwargs.get("global_steps", 0))
            request_namespace = namespace + ":" + str(kwargs.get("rlvr_eval_mode", "train"))
            request_id = request_identity(request_namespace, sample_id, generation, update_id)
            # Parent may choose a deterministic ID based on priority; the client wrapper always
            # sends the unique ID and the independently copied per-generation parameters.
            priority = int(kwargs.get("sequence_position", 0)) * int(self.config.rlvr.num_generations) + generation
            if seeds is None:
                priority += 1_000_000_000 if kwargs.get("rlvr_eval_mode") == "greedy" else 2_000_000_000
            parent_client = self.server_manager
            self.server_manager = RequestClient(parent_client, request_id, seed, priority)
            try:
                from rlvr.rollout import request_sampling_params
                params = request_sampling_params(sampling_params, kwargs.get("rlvr_eval_mode"), self.config.rlvr.eval)
                output = await super().run(params, priority=priority, **kwargs)
            finally:
                self.server_manager = parent_client
            output.extra_fields.update(sample_id=str(sample_id), generation_index=generation,
                                       sampling_seed=seed, rlvr_request_id=request_id,
                                       seed_namespace=namespace)
            if output.extra_fields.get("backend_abort"):
                raise RuntimeError("Backend abort is a failed exposure; automatic retry forbidden")
            if output.extra_fields.get("native_finish_reason") is None:
                raise RuntimeError("Native vLLM termination observer was not installed")
            if output.extra_fields.get("native_sampling_seed") != seed:
                raise RuntimeError("Backend SamplingParams.seed differs from the per-generation manifest seed")
            if self.config.rlvr.get("training_mode") == "lora" and not output.extra_fields.get("native_lora_request", {}).get("enabled"):
                raise RuntimeError("LoRA rollout did not submit the active adapter to vLLM")
            native_params = output.extra_fields.get("native_sampling_params") or {}
            from rlvr.rollout import sampling_contract_matches
            if not sampling_contract_matches(native_params, params):
                raise RuntimeError(f"Native sampling contract mismatch: actual={native_params}, requested={params}")
            if output.extra_fields.get("effective_max_tokens") != int(self.config.actor_rollout_ref.rollout.response_length):
                raise RuntimeError("Backend response token budget was reduced from its frozen limit")
            # Compute the same configured RewardManager here so asynchronous reward
            # intervals are observable. Upstream _compute_score sees reward_score
            # already assigned and preserves its usual transport/postprocess path.
            import numpy as np
            import torch
            from tensordict import TensorDict
            from verl.protocol import DataProto
            from rlvr.reward import RLVRRewardManager
            from rlvr.profiling import profile_stage
            if not output.response_ids:
                raise RuntimeError("Empty token response cannot form a complete RLVR trajectory")
            non_tensor = {}
            for name, value in kwargs.items():
                array = np.empty(1, dtype=object)
                array[0] = value
                non_tensor[name] = array
            array = np.empty(1, dtype=object)
            array[0] = output.extra_fields
            non_tensor["tool_extra_fields"] = array
            prompt_ids = torch.tensor([output.prompt_ids], dtype=torch.long)
            response_ids = torch.tensor([output.response_ids], dtype=torch.long)
            data = DataProto(batch=TensorDict({"prompts": prompt_ids, "responses": response_ids,
                "attention_mask": torch.ones((1, len(output.prompt_ids)+len(output.response_ids)), dtype=torch.long)},
                batch_size=1), non_tensor_batch=non_tensor)
            reward_start = time.monotonic_ns()
            with profile_stage("reward", os.environ["RLVR_RUN_DIR"], os.environ.get("RLVR_PHASE", "unknown"), update_id):
                scored = await RLVRRewardManager(self.config, self.tokenizer).run_single(data)
            output.reward_score = scored["reward_score"]
            output.extra_fields["reward_extra_info"] = scored["reward_extra_info"]
            output.extra_fields.update(reward_start_monotonic_ns=reward_start,
                                       reward_end_monotonic_ns=time.monotonic_ns())
            return output

    return SeededSingleTurnAgentLoop


def build_replica_class():
    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

    class ObservedVLLMServer(vLLMHttpServer):
        def _get_worker_extension_cls(self):
            if os.environ.get('RLVR_ADAPTER_SYNC_AUDIT') == '1':
                return 'rlvr.adapter_audit.AuditedColocateWorkerExtension'
            return super()._get_worker_extension_cls()
        async def launch_server(self, *args, **kwargs):
            from rlvr.rollout import EngineObserver
            result = await super().launch_server(*args, **kwargs)
            self.engine = EngineObserver(self.engine)
            return result

        async def generate(self, *args, **kwargs):
            # Ray wraps actor methods and may pickle their functions by value;
            # keep the per-process ContextVar behind a runtime module lookup.
            import rlvr.rollout as observer_module
            capture = {"termination_provenance": "native_vllm"}
            token = observer_module._capture.set(capture)
            try:
                output = await super().generate(*args, **kwargs)
                if output.stop_reason == "aborted":
                    capture.setdefault("backend_abort", True)
                    capture.setdefault("native_finish_reason", None)
                    capture.setdefault("native_stop_reason", None)
                    capture.setdefault("raw_response_token_ids", list(output.token_ids))
                elif "native_finish_reason" not in capture:
                    raise RuntimeError("Backend completed without observable native finish metadata")
                output.extra_fields.update(capture)
                return output
            finally:
                observer_module._capture.reset(token)

    class ObservedVLLMReplica(vLLMReplica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(ObservedVLLMServer)

    # Ray/cloudpickle must reference the module class, rather than trying to
    # serialize the module's ContextVar captured by a factory-local method.
    for cls in (ObservedVLLMServer, ObservedVLLMReplica):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    return ObservedVLLMReplica


def install_extensions():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from rlvr.worker_hooks import install_worker_hooks
    RolloutReplicaRegistry.register("vllm", build_replica_class)
    install_worker_hooks()
    from rlvr.adapter_audit import install_source_audit
    install_source_audit()


def __getattr__(name):
    if name == "SeededSingleTurnAgentLoop":
        value = build_agent_class()
        globals()[name] = value
        return value
    if name in ("ObservedVLLMServer", "ObservedVLLMReplica"):
        build_replica_class()
        return globals()[name]
    raise AttributeError(name)


# HFModelConfig imports this module in every worker through model.external_lib.
# CPU helpers remain importable without verl; launcher explicitly calls this as well.
try:
    import verl.workers.rollout.replica  # noqa: F401
except ImportError:
    pass
else:
    install_extensions()
