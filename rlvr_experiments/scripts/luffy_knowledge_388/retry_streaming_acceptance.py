"""Retry only failed streaming acceptance, retaining all completed performance evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from common import paths,save


def main():
    p=paths();base=p["run_root"]/"performance_audit_20261117";out=base/"streaming_retry_01"
    assert not out.exists();out.mkdir();scripts=Path(__file__).resolve().parent;py=str(p["python"])
    assert json.loads((p["run_root"]/"hold_complete.json").read_text())["held"]
    assert (base/"performance_failure.json").exists()
    save(base/"initial_attempt_acceptance_state.json",json.loads((base/"acceptance_state.json").read_text()))
    lock=out/"retry.lock";lock.write_text(str(os.getpid()));children={};workers={};processes=[]
    def state(phase,**extra):
        value={"phase":phase,"controller_pid":os.getpid(),"attempt":"streaming_retry_01","unix":time.time(),**extra}
        save(out/"acceptance_state.json",value);save(base/"acceptance_state.json",value)
    def launch(label,command,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",PYTHONPATH=str(p["project_dir"])+":"+str(p["project_dir"].parent))
        with (out/(label+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=p["project_dir"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[label]=child;processes.append({"label":label,"pid":child.pid,"gpu":gpu,"command":command,"started_unix":time.time()})
        save(out/"processes.json",processes);return child
    def check_workers():
        for label,child in workers.items():assert child.poll() is None,(label,child.returncode)
    def wait_file(file,timeout=2400):
        deadline=time.monotonic()+timeout
        while not file.exists():
            check_workers();assert time.monotonic()<deadline,("Timeout",str(file));time.sleep(.2)
        data=json.loads(file.read_text());assert not data.get("error"),data;return data
    def wait_child(label,child,timeout=3600):
        deadline=time.monotonic()+timeout
        while child.poll() is None:
            check_workers();assert time.monotonic()<deadline,("Timeout",label);time.sleep(1)
        assert child.returncode==0,(label,child.returncode)
    def rpc(root,request):
        save(root/(request["request_id"]+".request.json"),request)
        return wait_file(root/(request["request_id"]+".response.json"))
    try:
        used=[int(subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()) for g in (3,4,5)]
        assert max(used)<100,("The original three GPU slots must be free",used)
        actor_report=json.loads((base/"actor_benchmark/summary.json").read_text())
        candidates=[x for x in actor_report["results"] if x["passed"] and x["name"].startswith("batched")]
        best=min(candidates,key=lambda x:x["seconds"]);micro=int(best["name"].split("mb")[1].split("_")[0]);chunk=int(best["name"].split("chunk")[1])
        roots=[out/"sampling_gpu4",out/"sampling_gpu5"]
        state("retry_loading_models_and_checking_stream_interface")
        for gpu,root in zip((4,5),roots):
            label=f"sampler-gpu{gpu}";workers[label]=launch(label,[py,"-u",str(scripts/"stream_worker.py"),"--rpc",str(root)],gpu)
        gate=out/"workers_accepted.json"
        stream=launch("streaming-benchmark",[py,"-u",str(scripts/"benchmark_streaming.py"),"--out",str(out/"streaming_benchmark"),
            "--rpc",*[str(x) for x in roots],"--micro-batch",str(micro),"--head-chunk",str(chunk),"--readiness-check",str(gate)],3)
        for root in roots:wait_file(root/"ready.json",1200)
        source=json.loads((p["run_root"]/"rpc/A-step-051.request.json").read_text())
        old=json.loads((p["run_root"]/"rpc/A-step-051.response.json").read_text())
        first=source["prompts"][0];tokens=old["records"][0]["response_token_ids"][:64];errors=[]
        for index,root in enumerate(roots):
            request={"request_id":f"retry-score-{index}","mode":"score","adapter":source["adapter"],"adapter_sha256":source["adapter_sha256"],
                "prompts":[{"id":first["id"],"prompt_token_ids":first["prompt_token_ids"]+tokens,"score_start":len(first["prompt_token_ids"]),"seed":42}]}
            scores=rpc(root,request)["records"][0]["logprobs"]
            error=sum(abs(a-b) for a,b in zip(scores,old["records"][0]["raw_logprobs"][:64]))/64;assert error<.08;errors.append(error)
            warm={**source,"request_id":f"retry-stream-check-{index}","stream_groups":True,"max_new_tokens":8,
                "prompts":[{**x,"question_index":0} for x in source["prompts"][:4]]}
            result=rpc(root,warm);assert len(result["records"])==4
            assert [x["rollout_index"] for x in result["records"]]==list(range(4))
            assert all(0<len(x["response_token_ids"])<=8 for x in result["records"])
            assert (root/(warm["request_id"]+".groups")/"group-000.json").exists()
        save(gate,{"passed":True,"score_mae_by_worker":errors,"short_streaming_responses":8,"max_tokens_per_short_probe":8,"parameter_updates":0})
        state("retry_measuring_overlap",micro_batch=micro,head_chunk=chunk)
        wait_file(out/"streaming_benchmark/stream_phase_complete.json")
        # The timed overlap phase has ended. Release samplers and prepare the
        # legacy actor models while the first actor performs its untimed replay.
        for index,root in enumerate(roots):save(root/f"zz-retry-stop-{index}.request.json",{"request_id":f"zz-retry-stop-{index}","mode":"shutdown"})
        for index,root in enumerate(roots):
            file=root/f"zz-retry-stop-{index}.response.json";deadline=time.monotonic()+120
            while not file.exists():assert time.monotonic()<deadline;time.sleep(.1)
            assert json.loads(file.read_text())["stopped"]
        for child in workers.values():child.wait(timeout=120)
        workers.clear();deadline=time.monotonic()+120
        while True:
            used=[int(subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()) for g in (4,5)]
            if max(used)<100:break
            assert time.monotonic()<deadline;time.sleep(1)
        state("retry_validating_gradient_and_preparing_legacy_actors")
        start_gate=out/"legacy_timing_gate.json"
        legacy=launch("legacy-dp-benchmark",[py,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","--",
            str(scripts/"benchmark_legacy_dp.py"),"--request",str(base/"sampling_gpu4/perf-single-64.request.json"),
            "--response",str(base/"sampling_gpu4/perf-single-64.response.json"),"--out",str(out/"legacy_dp_benchmark"),"--start-gate",str(start_gate)],"4,5")
        wait_child("streaming-benchmark",stream)
        stream_report=json.loads((out/"streaming_benchmark/summary.json").read_text());assert stream_report["passed"]
        save(start_gate,{"stream_actor_exited":True,"unix":time.time()});state("retry_measuring_original_two_actor_update")
        wait_child("legacy-dp-benchmark",legacy)
        legacy_report=json.loads((out/"legacy_dp_benchmark/summary.json").read_text());assert legacy_report["passed"]
        assert not (p["run_root"]/"B/initialization.json").exists() and not (p["run_root"]/"C/initialization.json").exists()
        single=json.loads((base/"single_sampling_summary.json").read_text());dual=json.loads((base/"dual_sampling_summary.json").read_text())
        balanced=json.loads((base/"balanced_sampling_summary.json").read_text())
        summary={"passed":True,"completed_unix":time.time(),"completed_attempt":"streaming_retry_01","initial_failure_retained":str(base/"performance_failure.json"),
            "A_completed_steps":75,"B_C_still_held":True,"parameter_updates_performed":False,
            "completed_full_answer_rollouts":256,"aborted_full_answer_request_slots":64,"short_streaming_check_responses":8,"score_warmup_requests":4,
            "single_sampling":single,"dual_sampling":dual,"balanced_sampling":balanced,"actor_benchmark":actor_report,"selected_batching":best,
            "streaming_benchmark":stream_report,"legacy_two_actor_benchmark":legacy_report,"legacy_actor_gpu_indices":[4,5],
            "legacy_nonoverlapped_measured_stage_sum_seconds":single["wall_seconds"]+legacy_report["seconds"],
            "legacy_stage_sum_is_not_a_separate_continuous_full_run":True,
            "scope":"Performance and numerical checks only. Reuse completed measurements; original failed request IDs and aborted requests remain recorded. No formal B/C training or DEV."}
        save(out/"summary.json",summary);save(base/"performance_acceptance.json",summary);state("complete")
    except Exception:
        save(out/"failure.json",{"error":traceback.format_exc(),"unix":time.time()});state("failed")
        for child in children.values():
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()
