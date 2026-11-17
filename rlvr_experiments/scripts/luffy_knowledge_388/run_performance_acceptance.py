"""Run bounded benchmarks after the authorized A-only finish; never start B/C."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import paths,read_rows,save,verify


def main():
    p=paths();run=p["run_root"];out=run/"performance_audit_20261117";scripts=Path(__file__).resolve().parent
    py=str(p["python"]);processes=[];children={};workers={};out.mkdir(exist_ok=True)
    lock=out/"acceptance.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    def state(phase,**fields):save(out/"acceptance_state.json",{"phase":phase,"controller_pid":os.getpid(),"unix":time.time(),**fields})
    def launch(label,command,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",
                 VLLM_WORKER_MULTIPROC_METHOD="spawn",PYTHONPATH=str(p["project_dir"])+":"+str(p["project_dir"].parent))
        with (out/(label+".log")).open("w") as log:child=subprocess.Popen(command,env=env,cwd=p["project_dir"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[label]=child;processes.append({"label":label,"pid":child.pid,"command":command,"gpu":gpu,"started_unix":time.time()})
        save(out/"acceptance_processes.json",processes);return child
    def ensure_worker_alive():
        for label,child in workers.items():assert child.poll() is None,(label,child.returncode)
    def wait_file(file,timeout=1800):
        deadline=time.monotonic()+timeout
        while not file.exists():
            ensure_worker_alive();assert time.monotonic()<deadline,("Timed out",str(file));time.sleep(.2)
        data=json.loads(file.read_text());assert not data.get("error"),data
        return data
    def wait_child(label,child,timeout=3600):
        deadline=time.monotonic()+timeout
        while child.poll() is None:
            ensure_worker_alive();assert time.monotonic()<deadline,("Timed out child",label);time.sleep(2)
        assert child.returncode==0,(label,child.returncode)
    def rpc(root,request):
        save(root/(request["request_id"]+".request.json"),request)
        return wait_file(root/(request["request_id"]+".response.json"))
    try:
        deadline=time.monotonic()+21600
        while not (run/"hold_complete.json").exists():
            assert not (run/"hold_failure.json").exists(),(run/"hold_failure.json").read_text() if (run/"hold_failure.json").exists() else ""
            assert time.monotonic()<deadline,"Timed out waiting for A to finish"
            state("waiting_for_A_completion",A=json.loads((run/"A/progress.json").read_text())["step"])
            time.sleep(5)
        hold=json.loads((run/"hold_complete.json").read_text());assert hold["held"] and hold["A_steps"]==75
        assert not (run/"B/initialization.json").exists() and not (run/"C/initialization.json").exists()
        deadline=time.monotonic()+900
        while True:
            used=[int(subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()) for g in (3,4,5)]
            if max(used)<100:break
            state("waiting_for_our_gpus_to_be_released",used_mib=used)
            assert time.monotonic()<deadline,"GPUs3/4/5 remain occupied; no other process was touched"
            time.sleep(3)
        state("starting_bounded_benchmarks")
        actor=launch("actor-benchmark",[py,"-u",str(scripts/"benchmark_actor.py"),"--out",str(out/"actor_benchmark")],3)
        roots=[out/"sampling_gpu4",out/"sampling_gpu5"]
        for gpu,root in zip((4,5),roots):
            assert not (root/"ready.json").exists()
            label=f"sampler-gpu{gpu}"
            workers[label]=launch(label,[py,"-u",str(scripts/"stream_worker.py"),"--rpc",str(root)],gpu)
        for root in roots:wait_file(root/"ready.json",600)
        source=json.loads((run/"rpc/A-step-051.request.json").read_text())
        source_outputs=json.loads((run/"rpc/A-step-051.response.json").read_text())
        ids=list(dict.fromkeys(x["id"] for x in source["prompts"]));lookup={r["id"]:r for r in read_rows(p["train_data"])}
        first=source["prompts"][0];tokens=source_outputs["records"][0]["response_token_ids"][:64]
        for index,root in enumerate(roots):
            warm={"request_id":f"perf-warm-{index}","mode":"score","adapter":source["adapter"],
                  "adapter_sha256":source["adapter_sha256"],"prompts":[{"id":first["id"],
                  "prompt_token_ids":first["prompt_token_ids"]+tokens,"score_start":len(first["prompt_token_ids"]),"seed":42}]}
            score=rpc(root,warm)["records"][0]["logprobs"]
            reference=source_outputs["records"][0]["raw_logprobs"][:64]
            assert sum(abs(a-b) for a,b in zip(score,reference))/64<.08
        def summarize(name,values,elapsed):
            records=[x for value in values for x in value["records"]]
            assert len(records)==64 and len({(x["id"],x["rollout_index"]) for x in records})==64
            expected={(x["id"],x["rollout_index"]):x for x in source["prompts"]}
            for x in records:
                ref=expected[(x["id"],x["rollout_index"])];assert x["seed"]==ref["seed"] and x["prompt_token_ids"]==ref["prompt_token_ids"]
            verifications=[verify(lookup[x["id"]],x,8192) for x in records];tokens=sum(len(x["response_token_ids"]) for x in records)
            return {"name":name,"questions":16,"responses":64,"wall_seconds":elapsed,
                    "worker_seconds":[v["elapsed_seconds"] for v in values],"output_tokens":tokens,
                    "tokens_per_wall_second":tokens/elapsed,"correct":sum(v["correctness"] for v in verifications),
                    "truncated":sum(v["truncated"] for v in verifications)}
        state("single_sampler_64_responses")
        req={**source,"request_id":"perf-single-64","prompts":[{**x,"question_index":ids.index(x["id"])} for x in source["prompts"]]}
        start=time.monotonic();single=rpc(roots[0],req);single_summary=summarize("single_sampler",[single],time.monotonic()-start)
        save(out/"single_sampling_summary.json",single_summary)
        state("two_samplers_same_64_responses")
        start=time.monotonic()
        for shard,root in enumerate(roots):
            req={**source,"request_id":f"perf-dual-{shard}","prompts":[{**x,"question_index":ids.index(x["id"])} for x in source["prompts"] if ids.index(x["id"])%2==shard]}
            assert len(req["prompts"])==32;save(root/(req["request_id"]+".request.json"),req)
        dual=[wait_file(root/(f"perf-dual-{i}.response.json")) for i,root in enumerate(roots)]
        dual_summary=summarize("two_samplers",dual,time.monotonic()-start)
        single_map={(x["id"],x["rollout_index"]):x for x in single["records"]}
        dual_summary["exact_token_matches_to_single"]=sum(x["response_token_ids"]==single_map[(x["id"],x["rollout_index"])]["response_token_ids"] for v in dual for x in v["records"])
        save(out/"dual_sampling_summary.json",dual_summary)
        state("waiting_for_actor_batching_benchmark")
        wait_child("actor-benchmark",actor)
        actor_report=json.loads((out/"actor_benchmark/summary.json").read_text());assert actor_report["complete"]
        candidates=[r for r in actor_report["results"] if r["passed"] and r["name"].startswith("batched")]
        assert candidates,"No batched execution passed GPU numerical acceptance"
        best=min(candidates,key=lambda r:r["seconds"]);micro=int(best["name"].split("mb")[1].split("_")[0]);chunk=int(best["name"].split("chunk")[1])
        state("overlapping_complete_groups_with_gradient_computation",micro_batch=micro,head_chunk=chunk)
        stream=launch("streaming-benchmark",[py,"-u",str(scripts/"benchmark_streaming.py"),"--out",str(out/"streaming_benchmark"),"--rpc",*[str(x) for x in roots],"--micro-batch",str(micro),"--head-chunk",str(chunk)],3)
        wait_child("streaming-benchmark",stream)
        stream_report=json.loads((out/"streaming_benchmark/summary.json").read_text());assert stream_report["complete"] and stream_report["passed"]
        for index,root in enumerate(roots):
            req={"request_id":f"zz-perf-shutdown-{index}","mode":"shutdown"}
            # A worker may exit after acknowledging shutdown, so do not use the
            # all-workers-alive guard when collecting shutdown acknowledgements.
            save(root/(req["request_id"]+".request.json"),req)
        for index,root in enumerate(roots):
            file=root/(f"zz-perf-shutdown-{index}.response.json");deadline=time.monotonic()+120
            while not file.exists():assert time.monotonic()<deadline;time.sleep(.2)
            assert json.loads(file.read_text())["stopped"]
        for child in workers.values():child.wait(timeout=120)
        workers.clear()
        state("measuring_original_two_actor_update_on_existing_outputs")
        deadline=time.monotonic()+120
        while True:
            used=[int(subprocess.check_output(["nvidia-smi","-i",str(g),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip()) for g in (3,4)]
            if max(used)<100:break
            assert time.monotonic()<deadline,"Benchmark actor GPUs not released; no other process was touched"
            time.sleep(2)
        legacy=launch("legacy-dp-benchmark",[py,"-m","torch.distributed.run","--standalone","--nproc_per_node=2","--",
            str(scripts/"benchmark_legacy_dp.py"),"--request",str(roots[0]/"perf-single-64.request.json"),
            "--response",str(roots[0]/"perf-single-64.response.json"),"--out",str(out/"legacy_dp_benchmark")],"3,4")
        wait_child("legacy-dp-benchmark",legacy)
        legacy_report=json.loads((out/"legacy_dp_benchmark/summary.json").read_text());assert legacy_report["passed"]
        assert not (run/"B/initialization.json").exists() and not (run/"C/initialization.json").exists()
        summary={"passed":True,"completed_unix":time.time(),"A_completed_steps":75,"B_C_still_held":True,
            "parameter_updates_performed":False,"new_answer_rollouts":192,"warmup_score_requests":2,
            "single_sampling":single_summary,"dual_sampling":dual_summary,
            "actor_benchmark":actor_report,"selected_batching":best,"streaming_benchmark":stream_report,
            "legacy_two_actor_benchmark":legacy_report,
            "legacy_nonoverlapped_measured_stage_sum_seconds":single_summary["wall_seconds"]+legacy_report["seconds"],
            "legacy_stage_sum_is_not_a_separate_continuous_full_run":True,
            "scope":"Bounded performance/numerical acceptance on the original three GPU slots after A; no formal B/C training or DEV evaluation."}
        save(out/"performance_acceptance.json",summary);state("complete")
    except Exception:
        save(out/"performance_failure.json",{"error":traceback.format_exc(),"unix":time.time()})
        state("failed")
        # Terminate only explicitly launched benchmark process groups. The
        # completed A experiment and every unrelated GPU process are untouched.
        import signal
        for label,child in children.items():
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()
