"""Run the frozen CPU comparison after the three final DEV evaluations finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import configuration, paths, save, sha

def validate_inputs(p):
    manifest=json.loads((p["run_root"]/"analysis_input_manifest.json").read_text())
    root=p["benchmark_root"].parent
    for item in manifest["files"]:
        path=root/item["path"]
        assert path.is_file() and sha(path)==item["sha256"],str(path)
    return manifest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    p=paths();cfg=configuration(p);run=p["run_root"];manifest=validate_inputs(p)
    if args.check:
        print(json.dumps({"analysis_inputs_verified":len(manifest["files"])}));return
    lock=run/"analysis_stage.lock"
    with lock.open("x") as stream:stream.write(str(os.getpid()))
    try:
        while not (run/"pipeline_complete.json").exists():
            if (run/"pipeline_failure.json").exists():
                raise RuntimeError("Training/evaluation pipeline failed; no analysis was run")
            save(run/"analysis_stage_state.json",{"phase":"waiting_for_final_evaluations","pid":os.getpid(),"unix":time.time()})
            time.sleep(15)
        complete=json.loads((run/"pipeline_complete.json").read_text())
        assert complete["success"] and complete["evaluation_steps"]==[75] and not complete["test_evaluated"]
        validate_inputs(p)
        save(run/"analysis_stage_state.json",{"phase":"analyzing","pid":os.getpid(),"unix":time.time()})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2")
        command=[str(p["python"]),"-u",str(p["benchmark_root"]/"eval/scripts/analyze_luffy_knowledge_388.py"),
                 "--root",str(p["benchmark_root"].parent),"--run-name",cfg["name"]]
        with (run/"analysis.log").open("w") as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,("CPU comparison failed",result.returncode)
        output=p["benchmark_root"]/"eval/analysis"/cfg["name"]
        report=json.loads((output/"comparison.json").read_text())
        assert report["matched_protocol"] and report["dev_evaluated_steps"]==[75] and not report["test_evaluated"]
        save(run/"analysis_complete.json",{"passed":True,"completed_unix":time.time(),
            "input_manifest_sha256":sha(run/"analysis_input_manifest.json"),
            "comparison_sha256":sha(output/"comparison.json"),"report_sha256":sha(output/"report.txt"),
            "primary_comparison":"C_vs_B","baseline_reused":True,"test_evaluated":False})
        save(run/"analysis_stage_state.json",{"phase":"complete","unix":time.time()})
        print(json.dumps({"analysis_complete":True,"report":str(output/"report.txt")}),flush=True)
    except Exception:
        save(run/"analysis_failure.json",{"error":traceback.format_exc(),"unix":time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=="__main__":main()
