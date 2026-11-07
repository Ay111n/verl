"""CPU tests use explicitly synthetic traces, never claim live GPU performance."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import subprocess
import sys
import shutil
import importlib.util

from rlvr.hardware import GPUInfo, GpuMonitor, hardware_gate, select_idle_gpu_pair, time_weighted_utilization
from rlvr.profiling import analyze_trace, read_offload_policy, observed_trace_loss_messages, _native_stderr_capture, _union, _subtract, _duration
_report_spec = importlib.util.spec_from_file_location(
    "rlvr_benchmark_report", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_report.py")
_report_module = importlib.util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(_report_module)
build_report, normalize_updates = _report_module.build_report, _report_module.normalize_updates


def event(name, start, duration, category="cpu_op", args=None, pid=1, tid=1):
    return {"name": name, "ph": "X", "ts": start, "dur": duration,
            "cat": category, "pid": pid, "tid": tid, "args": args or {}}


class HardwareGateTests(unittest.TestCase):
    def test_direct_r0_phase_is_explicit_and_default_still_requires_benchmark(self):
        updates = self.updates()
        for row in updates:
            row['phase'] = 'reward_dev'
        diagnostics = self.diagnostics()
        self.assertFalse(hardware_gate(updates, self.samples(), diagnostics, [0, 1])['r0_authorized'])
        report = hardware_gate(updates, self.samples(), diagnostics, [0, 1], qualification_phase='reward_dev')
        self.assertTrue(report['r0_authorized'])
        self.assertEqual(report['qualification_phase'], 'reward_dev')
        with self.assertRaises(ValueError):
            hardware_gate(updates, self.samples(), diagnostics, [0, 1], qualification_phase='formal_16k')

    def test_cleanup_only_finished_run_tag_and_excludes_current_process(self):
        from rlvr.hardware import cleanup_run_processes
        expected = str(self.directory.resolve())
        current, owned, unrelated = (mock.Mock(pid=pid) for pid in (10, 20, 30))
        current.environ.return_value = owned.environ.return_value = {"RLVR_RUN_DIR": expected}
        unrelated.environ.return_value = {"RLVR_RUN_DIR": expected + "-other"}
        owned.create_time.return_value = 100.0
        with mock.patch("os.getpid", return_value=10), \
                mock.patch("psutil.process_iter", return_value=[current, owned, unrelated]), \
                mock.patch("psutil.wait_procs", return_value=([], [owned])):
            result = cleanup_run_processes(self.directory, "failed")
        self.assertEqual(result["owned_processes"], [{"pid": 20, "created_unix": 100.0}])
        owned.terminate.assert_called_once()
        owned.kill.assert_called_once()
        current.terminate.assert_not_called()
        unrelated.terminate.assert_not_called()
        with self.assertRaises(ValueError):
            cleanup_run_processes(self.directory, "running")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def updates(self):
        return [{"phase": "benchmark", "measurement": "steady", "update_id": i,
                 "status": "complete", "prompt_count": 32, "response_count": 256,
                 "group_count": 32, "generation_count": 8, "finite_update": True,
                 "optimizer_steps_by_rank": {"0": 1, "1": 1}, "generated_tokens": 1024,
                 "start_monotonic_ns": i*10_000_000_000, "end_monotonic_ns": (i+1)*10_000_000_000,
                 "oom": False, "stage_seconds": {"rollout": 5., "actor_update": 3.,
                      "reward": .1, "old_logprob": .2, "ref_logprob": .2,
                      "advantage": .1, "weight_sync": .2},
                 "kv_cache": {"source": "vllm_engine_metrics", "observed_timestamp_ns": 123,
                              "metrics": {"kv_cache_usage_fraction": .5}}}
                for i in range(10)]

    def samples(self, utilization=80):
        return [{"gpu_index": gpu, "monotonic_ns": second*1_000_000_000,
                 "utilization_gpu": utilization, "memory_used_mib": 10000}
                for second in range(101) for gpu in (0, 1)]

    def trace(self, update=0, rank=0, mode="passing"):
        # Synthetic Chrome/CUPTI-shaped events, microsecond clock.
        events = [event("rlvr.stage.actor_update", 0, 100),
                  event("rlvr.offload.load", 0, 5),
                  event("cudaMemcpyAsync", 1, 1, "cuda_runtime", {"correlation": 7}),
                  event("Memcpy HtoD", 1, 9, "gpu_memcpy", {"correlation": 7}, pid=100),
                  event("train_kernel", 10, 90, "kernel", pid=100)]
        if mode == "ambiguous":
            events[-1]["dur"] = 20
        elif mode == "bound_failure":
            events[3]["dur"] = 60
            events.append(event("cudaStreamSynchronize", 0, 61, "cuda_runtime"))
            events[-2]["ts"] = 61
            events[-2]["dur"] = 39
        elif mode == "missing_cuda":
            events = [e for e in events if e["cat"] not in ("kernel", "gpu_memcpy")]
        path = self.directory / f"synthetic-{mode}-{update}-{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        self.capture(path, update, rank)
        return analyze_trace(path, update, rank)

    def capture(self, path, update=0, rank=0, *, stderr="", actor_completed=True):
        stderr_path = path.with_suffix(".stderr.log")
        stderr_path.write_text(stderr, encoding="utf-8")
        evidence = {"schema": "rlvr-trace-capture-v1", "update_id": update, "rank": rank,
                    "capture_status": "complete", "actor_completed": actor_completed,
                    "profiler_completed": True, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "native_stderr_path": str(stderr_path),
                    "native_stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
                    "observed_trace_loss_messages": observed_trace_loss_messages(stderr), "python_warnings": [],
                    "actor_stage_end": {"event": "end", "stage": "actor_update", "status": "complete",
                                        "error": None, "diagnostic": True, "update_id": update, "rank": rank}}
        path.with_suffix(".capture.json").write_text(json.dumps(evidence), encoding="utf-8")
        return evidence

    def diagnostics(self, mode="passing"):
        return [self.trace(i, rank, mode) for i in range(3) for rank in (0, 1)]

    def bound_runs(self):
        """Entirely synthetic run/data artifacts exercising report binding logic."""
        candidate, diagnostic = self.directory / "candidate", self.directory / "diagnostic"
        data_dir = self.directory / "prepared"
        data_dir.mkdir()
        manifest = {"schema_version": "synthetic-test", "prompt_version": "synthetic-test",
                    "source_files": {"synthetic.jsonl": "same-frozen-source"}, "roles": {}, "plans": {}}
        common_plan = {"schema_version": "synthetic-test", "seed_version": "synthetic-test", "base_seed": 42,
                       "dataset_order_seed": 42, "prompts_per_update": 32, "num_generations": 8,
                       "shuffle": "fixed-test-order"}
        for phase, budget in (("benchmark", 10), ("diagnostic", 3)):
            data = data_dir / f"{phase}.parquet"
            data.write_bytes(b"SYNTHETIC CPU fixture only")
            manifest["roles"][f"{phase}_train"] = {"path": str(data),
                 "sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "source_sha256": "same-frozen-source"}
            plan = data_dir / f"{phase}.json"
            plan.write_text(json.dumps({**common_plan, "train_updates": budget}))
            manifest["plans"][phase] = {"path": str(plan), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
        manifest_path = data_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        for folder, phase, budget in ((candidate, "benchmark", 10), (diagnostic, "diagnostic", 3)):
            folder.mkdir()
            config = {"actor_rollout_ref": {"actor": {"fsdp_config": {"param_offload": True}}},
                      "algorithm": {"adv_estimator": "grpo"},
                      "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"VLLM_BATCH_INVARIANT": "1"}}}},
                      "data": {"train_batch_size": 32, "gen_batch_size": 32, "max_prompt_length": 4096,
                               "max_response_length": 4096, "shuffle": False, "seed": 42,
                               "apply_chat_template_kwargs": {"enable_thinking": True},
                               "train_files": [manifest["roles"][f"{phase}_train"]["path"]]},
                      "rlvr": {"prompts_per_update": 32, "num_generations": 8, "base_seed": 42,
                               "reward_profile": "R0", "component_flags": {}, "train_updates": budget,
                               "data_manifest_path": str(manifest_path),
                               "manifest_path": manifest["plans"][phase]["path"]}}
            values = {"resolved_config": config, "status": {"status": "complete"},
                      "environment": {"python": "synthetic", "executable": "synthetic", "verl_commit": "synthetic",
                                      "packages": {"torch": "synthetic"}, "custom_source_hashes": {"test": "same"}},
                      "gpu_selection": {"physical_gpu_ids": [0, 1]}, "frozen_algorithm": {"batch": 32}}
            for name, value in values.items():
                (folder / f"{name}.json").write_text(json.dumps(value))
            samples = [{**row, "gpu_uuid": f"SYNTHETIC-uuid{row['gpu_index']}"} for row in self.samples()]
            (folder / "gpu_samples.jsonl").write_text("\n".join(map(json.dumps, samples))+"\n")
            updates = [{**row, "update_id": i+1, "manifest_sha256": manifest["plans"][phase]["sha256"],
                        "measurement": "steady" if phase == "benchmark" else "diagnostic"}
                       for i, row in enumerate(self.updates()[:budget])]
            (folder / "updates.jsonl").write_text("\n".join(map(json.dumps, updates))+"\n")
        (diagnostic / "traces").mkdir()
        for update in (1, 2, 3):
            for rank in (0, 1):
                evidence = self.trace(update, rank)
                original = Path(evidence["trace_path"])
                moved = diagnostic / "traces" / original.name
                for suffix in (".json", ".stderr.log", ".capture.json"):
                    shutil.move(original.with_suffix(suffix), moved.with_suffix(suffix))
                evidence = analyze_trace(moved, update, rank)
                moved.with_suffix(".offload.json").write_text(json.dumps(evidence))
        return candidate, diagnostic

    def test_pair_prefers_same_numa_and_rejects_busy(self):
        def gpu(i, numa, used=0, processes=0):
            return GPUInfo(i, f"uuid{i}", "NVIDIA L20", 46080, used, 0, numa_node=numa,
                           compute_processes=processes)
        selected = select_idle_gpu_pair(inventory=[gpu(0, 0), gpu(1, 1), gpu(2, 1), gpu(3, 0, processes=1)])
        self.assertEqual([g.index for g in selected], [1, 2])
        with self.assertRaises(RuntimeError):
            select_idle_gpu_pair(inventory=[gpu(0, 0, used=20), gpu(1, 0)])

    def test_time_weighted_coverage_and_gaps(self):
        samples = [{"gpu_index": 0, "monotonic_ns": 0, "utilization_gpu": 10},
                   {"gpu_index": 0, "monotonic_ns": 1_000_000_000, "utilization_gpu": 90},
                   {"gpu_index": 0, "monotonic_ns": 3_000_000_000, "utilization_gpu": 0}]
        metric = time_weighted_utilization(samples, [(0, 3_000_000_000)], [0])["0"]
        self.assertAlmostEqual(metric["active_time_percent"], 190/3)
        self.assertEqual(metric["coverage"], 1)
        metric = time_weighted_utilization(samples, [(0, 4_000_000_000)], [0])["0"]
        self.assertEqual(metric["coverage"], .75)
        self.assertAlmostEqual(metric["full_window_lower_percent"], 47.5)

    def test_partial_coverage_cannot_raise_full_window_utilization_above_threshold(self):
        samples = [s for s in self.samples(41) if s["monotonic_ns"] <= 95_000_000_000]
        report = hardware_gate(self.updates(), samples, self.diagnostics(), [0, 1])
        self.assertEqual(report["two_gpu_mean_active_time_percent"], 41)
        self.assertAlmostEqual(report["two_gpu_full_window_active_time_lower_percent"], 38.95)
        self.assertEqual(report["decision"], "insufficient_evidence")
        samples = [s for s in self.samples(45) if s["monotonic_ns"] <= 95_000_000_000]
        self.assertTrue(hardware_gate(self.updates(), samples, self.diagnostics(), [0, 1])["r0_authorized"])

    def test_nvidia_smi_sample_has_real_clock_and_absolute_gpu_ids(self):
        monitor = GpuMonitor(self.directory / "sample.jsonl", [1, 2])
        monitor._fallback_reason = "synthetic unit-test input"
        with mock.patch("rlvr.hardware._command", return_value=
                        "1, GPU-uuid1, 46080, 0, 0, 0, 50\n2, GPU-uuid2, 46080, 123, 80, 30, 75\n"):
            records = monitor.sample()
        self.assertEqual([r["gpu_index"] for r in records], [1, 2])
        self.assertEqual(records[1]["utilization_gpu"], 80)
        self.assertGreater(records[0]["monotonic_ns"], 0)
        self.assertEqual(records[0]["measurement"], "gpu_active_time_percent_not_sm_occupancy")

    def test_trace_changed_after_analysis_cannot_authorize(self):
        diagnostics = self.diagnostics()
        Path(diagnostics[0]["trace_path"]).write_text("{}", encoding="utf-8")
        report = hardware_gate(self.updates(), self.samples(), diagnostics, [0, 1])
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertTrue(any("hash changed" in reason for reason in report["missing_evidence"]))

    def test_missing_stages_or_kv_cache_cannot_authorize(self):
        updates = self.updates()
        updates[0]["kv_cache"] = None
        del updates[1]["stage_seconds"]["weight_sync"]
        report = hardware_gate(updates, self.samples(), self.diagnostics(), [0, 1])
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertTrue(any("KV-cache" in reason for reason in report["missing_evidence"]))
        self.assertTrue(any("weight_sync" in reason for reason in report["missing_evidence"]))

    def test_no_offload_proven_by_resolved_config_and_cuda_trace(self):
        config = self.directory / "resolved_config.json"
        config.write_text(json.dumps({"actor_rollout_ref": {"actor": {"fsdp_config": {
            "param_offload": False, "optimizer_offload": False, "offload_policy": False}}}}))
        policy = read_offload_policy(config)
        path = self.directory / "synthetic-nooffload.json"
        path.write_text(json.dumps({"traceEvents": [event("rlvr.stage.actor_update", 0, 100),
                                                   event("cudaLaunchKernel", 1, 1, "cuda_runtime"),
                                                   event("train_kernel", 0, 100, "kernel", pid=100)]}))
        self.capture(path)
        evidence = analyze_trace(path, 0, 0, policy)
        self.assertEqual(evidence["evidence_status"], "complete")
        self.assertEqual(evidence["offload_critical_path_upper_fraction"], 0)
        config.write_text("{}")
        self.assertEqual(analyze_trace(path, 0, 0, policy)["evidence_status"], "insufficient")

    def test_scalar_bookkeeping_copy_is_proven_and_unknown_copy_is_rejected(self):
        config = self.directory / "scalar-config.json"
        config.write_text(json.dumps({"actor_rollout_ref": {"actor": {"fsdp_config": {
            "param_offload": False, "optimizer_offload": False, "offload_policy": False}}}}))
        policy = read_offload_policy(config)
        events = [event("rlvr.stage.actor_update", 0, 100), event("train_kernel", 0, 100, "kernel", pid=100),
                  event("aten::_local_scalar_dense", 20, 10),
                  event("cudaMemcpy", 21, 8, "cuda_runtime", {"correlation": 99}),
                  event("Memcpy DtoH", 23, 2, "gpu_memcpy", {"correlation": 99, "bytes": 4}, pid=100)]
        path = self.directory / "synthetic-scalar.json"
        path.write_text(json.dumps({"traceEvents": events}))
        self.capture(path)
        report = analyze_trace(path, 0, 0, policy)
        self.assertEqual(report["evidence_status"], "complete")
        self.assertEqual(len(report["bookkeeping_copies"]), 1)
        events[-1]["args"]["bytes"] = 4096
        path.write_text(json.dumps({"traceEvents": events}))
        self.capture(path)
        self.assertEqual(analyze_trace(path, 0, 0, policy)["evidence_status"], "insufficient")

    def test_all_cuda_api_intervals_are_in_conservative_upper(self):
        evidence = self.trace()
        path = Path(evidence["trace_path"])
        payload = json.loads(path.read_text())
        payload["traceEvents"].append(event("cudaMemcpy", 15, 60, "cuda_runtime"))
        path.write_text(json.dumps(payload))
        self.capture(path)
        result = analyze_trace(path, 0, 0)
        self.assertAlmostEqual(result["offload_critical_path_upper_fraction"], .7)
        self.assertEqual(result["offload_critical_path_lower_fraction"], 0)

    def test_failed_capture_missing_cuda_api_or_observed_cupti_loss_rejected(self):
        evidence = self.trace()
        path = Path(evidence["trace_path"])
        self.capture(path, actor_completed=False)
        self.assertEqual(analyze_trace(path, 0, 0)["evidence_status"], "insufficient")
        self.capture(path, stderr="SYNTHETIC unit test CUPTI dropped 7 events\n")
        self.assertEqual(analyze_trace(path, 0, 0)["evidence_status"], "insufficient")
        payload = json.loads(path.read_text())
        payload["traceEvents"] = [e for e in payload["traceEvents"] if e["cat"] != "cuda_runtime"]
        path.write_text(json.dumps(payload))
        self.capture(path)
        result = analyze_trace(path, 0, 0)
        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertTrue(any("API" in reason for reason in result["limitations"]))

    def test_native_stderr_is_captured_replayed_and_restored_on_exception(self):
        path = self.directory / "native-unit-test.stderr.log"
        saved = os.dup(2)
        with tempfile.TemporaryFile() as sink:
            os.dup2(sink.fileno(), 2)
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    with _native_stderr_capture(path):
                        os.write(2, b"SYNTHETIC native profiler log\n")
                        raise RuntimeError("synthetic actor failure")
                os.write(2, b"SYNTHETIC restored stderr\n")
                sink.seek(0)
                output = sink.read()
                self.assertIn(b"native profiler log", output)
                self.assertIn(b"restored stderr", output)
                self.assertNotIn("restored stderr", path.read_text())
            finally:
                os.dup2(saved, 2)
                os.close(saved)

    def test_stage_aliases_fill_null_values_from_actual_records(self):
        updates = [{"update_id": 1, "stage_seconds": {"old_logprob": None}}]
        records = [{"event": "end", "status": "complete", "update_id": 1,
                    "stage": "old_log_prob", "rank": 0, "pid": 1, "duration_s": 3}]
        self.assertEqual(normalize_updates(updates, records)[0]["stage_seconds"]["old_logprob"], 3)

    def test_report_binds_completed_same_environment_same_gpu_and_frozen_data(self):
        candidate, diagnostic = self.bound_runs()
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertTrue(report["r0_authorized"], report["missing_evidence"])

    def test_report_rejects_diagnostic_environment_hardware_or_data_mismatch(self):
        candidate, diagnostic = self.bound_runs()
        env_path = diagnostic / "environment.json"
        original = env_path.read_text()
        environment = json.loads(original)
        environment["packages"]["torch"] = "different-test-version"
        env_path.write_text(json.dumps(environment))
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertTrue(any("environment mismatch" in line for line in report["missing_evidence"]))
        env_path.write_text(original)
        gpu_path = diagnostic / "gpu_selection.json"
        gpu_path.write_text(json.dumps({"physical_gpu_ids": [1, 0]}))
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertFalse(report["r0_authorized"])
        self.assertTrue(any("GPU binding" in line for line in report["missing_evidence"]))
        gpu_path.write_text(json.dumps({"physical_gpu_ids": [0, 1]}))
        config_path = diagnostic / "resolved_config.json"
        config = json.loads(config_path.read_text())
        config["data"]["max_response_length"] = 1024
        config_path.write_text(json.dumps(config))
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertTrue(any("data_fields mismatch" in line for line in report["missing_evidence"]))

    def test_report_rejects_incomplete_candidate_and_changed_frozen_data(self):
        candidate, diagnostic = self.bound_runs()
        (candidate / "status.json").write_text(json.dumps({"status": "failed"}))
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertFalse(report["r0_authorized"])
        self.assertTrue(any("Run did not complete" in line for line in report["missing_evidence"]))
        (candidate / "status.json").write_text(json.dumps({"status": "complete"}))
        data = Path(json.loads((candidate / "resolved_config.json").read_text())["data"]["train_files"][0])
        data.write_bytes(b"SYNTHETIC changed fixture")
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertFalse(report["r0_authorized"])
        self.assertTrue(any("Prepared benchmark data differs" in line for line in report["missing_evidence"]))

    def test_missing_timing_and_truncated_raw_jsonl_produce_insufficient_report(self):
        candidate, diagnostic = self.bound_runs()
        updates_path = candidate / "updates.jsonl"
        rows = [json.loads(line) for line in updates_path.read_text().splitlines()]
        rows[0]["stage_seconds"]["rollout"] = None
        updates_path.write_text("\n".join(map(json.dumps, rows))+"\n{SYNTHETIC truncated\n")
        report = build_report(candidate, [0, 1], [diagnostic])
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertTrue(any("Invalid raw JSONL" in line for line in report["missing_evidence"]))
        self.assertIsNone(report["rollout_tokens_per_second"])

    def test_report_cli_missing_actual_data_refuses_and_summarizes_candidates(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_report.py"
        first, second = self.directory / "candidate1", self.directory / "candidate2"
        first.mkdir()
        second.mkdir()
        result = subprocess.run([sys.executable, str(script), str(first), str(second),
                                 "--gpu-ids", "0", "1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads((first / "hardware_candidates.json").read_text())
        self.assertEqual(report["decision"], "no_accepted_candidate")
        self.assertIsNone(report["selected_candidate_run_dir"])
        self.assertEqual(len(report["candidates"]), 2)
        self.assertTrue((second / "hardware_gate.json").exists())

    def test_interval_union_does_not_sum_overlaps(self):
        self.assertEqual(_union([(0, 20), (10, 30)]), [(0, 30)])
        self.assertEqual(_duration([(0, 20), (10, 30)]), 30)
        self.assertEqual(_subtract([(0, 30)], [(10, 20)]), [(0, 10), (20, 30)])

    def test_trace_has_conservative_bounds(self):
        trace = self.trace()
        self.assertEqual(trace["evidence_status"], "complete")
        self.assertEqual(trace["offload_critical_path_lower_fraction"], 0)
        self.assertAlmostEqual(trace["offload_critical_path_upper_fraction"], .1)

    def test_pass_requires_ten_real_update_records_and_three_both_rank_traces(self):
        report = hardware_gate(self.updates(), self.samples(), self.diagnostics(), [0, 1])
        self.assertTrue(report["r0_authorized"])
        self.assertEqual(report["steady_update_count"], 10)
        self.assertAlmostEqual(report["rollout_tokens_per_second"], 204.8)
        self.assertAlmostEqual(report["rollout_responses_per_second"], 51.2)
        self.assertEqual(report["average_response_tokens"], 4)

    def test_missing_trace_and_telemetry_are_insufficient(self):
        report = hardware_gate(self.updates(), [], [], [0, 1])
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertFalse(report["r0_authorized"])

    def test_cpu_only_trace_rejected(self):
        self.assertEqual(self.trace(mode="missing_cuda")["evidence_status"], "insufficient")

    def test_ambiguous_offload_is_insufficient_not_claimed_bound(self):
        report = hardware_gate(self.updates(), self.samples(), self.diagnostics("ambiguous"), [0, 1])
        self.assertEqual(report["decision"], "insufficient_evidence")
        self.assertTrue(any("straddle" in reason for reason in report["missing_evidence"]))
        self.assertFalse(any("offload" in reason for reason in report["gate_failures"]))

    def test_actual_lower_bound_failure_is_technically_only(self):
        report = hardware_gate(self.updates(), self.samples(), self.diagnostics("bound_failure"), [0, 1])
        self.assertEqual(report["decision"], "technically_runs_only")
        self.assertFalse(report["r0_authorized"])

    def test_low_utilization_is_technically_only(self):
        report = hardware_gate(self.updates(), self.samples(20), self.diagnostics(), [0, 1])
        self.assertEqual(report["decision"], "technically_runs_only")

    def test_failed_finite_and_optimizer_counts(self):
        updates = self.updates()
        updates[0]["finite_update"] = False
        updates[1]["optimizer_steps_by_rank"]["1"] = 0
        report = hardware_gate(updates, self.samples(), self.diagnostics(), [0, 1])
        self.assertFalse(report["r0_authorized"])
        self.assertEqual(len(report["gate_failures"]), 2)

    def test_skipped_prompt_and_duplicates_rejected(self):
        updates = self.updates()
        updates[0]["prompt_count"] = 31
        updates[1]["update_id"] = 0
        report = hardware_gate(updates, self.samples(), self.diagnostics(), [0, 1])
        self.assertFalse(report["r0_authorized"])

    def test_p95_median_instability_rejected(self):
        updates = self.updates()
        updates[-1]["end_monotonic_ns"] += 90_000_000_000
        samples = self.samples() + [{"gpu_index": gpu, "monotonic_ns": second*1_000_000_000,
                                    "utilization_gpu": 80, "memory_used_mib": 10000}
                                   for second in range(101, 191) for gpu in (0, 1)]
        report = hardware_gate(updates, samples, self.diagnostics(), [0, 1])
        self.assertEqual(report["decision"], "technically_runs_only")
        self.assertGreater(report["p95_median_ratio"], 1.5)


if __name__ == "__main__":
    unittest.main()
