"""Run conditional prefix GRPO plus teacher-note distillation on unhinted student histories."""
import sys
import os
import tempfile
import subprocess
from pathlib import Path
from prepare_opsd import CONFIG, OUT, PROJECT, ROOT, RUN

sys.path[:0] = [str(PROJECT), str(PROJECT/"scripts/prefix_rescue_512"),
                 str(PROJECT/"scripts/knowledge_grpo_intermittent_nll_v3")]
from scripts.prefix_rescue_512.candidate import train
from run import supervise


def main():
    if sys.argv[1:] == ["run"]:
        ipc = ROOT / ".runtime/opsd512-1121"
        cache = ROOT / ".runtime" / RUN.name
        for key, value in {"TMPDIR":ipc, "TMP":ipc, "TEMP":ipc, "VLLM_RPC_BASE_PATH":ipc,
                           "VLLM_CACHE_ROOT":cache/"vllm", "TORCHINDUCTOR_CACHE_DIR":cache/"inductor",
                           "TRITON_CACHE_DIR":cache/"triton", "CUDA_CACHE_PATH":cache/"cuda"}.items():
            value.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.environ[key] = str(value)
        tempfile.tempdir = str(ipc)
        supervise(CONFIG, RUN, online_actor=Path(__file__).resolve())
        report = ROOT/"benchmark/eval/analysis"/RUN.name/"compare_candidate.py"
        subprocess.run([sys.executable, "-B", str(report)], cwd=ROOT, check=True)
    elif sys.argv[1:] == ["train"]:
        from scripts.prefix_opsd_512.objective import online
        train(CONFIG, RUN, OUT, auxiliary_update=online)
    else:
        raise SystemExit("Use run or train")


if __name__ == "__main__":
    main()
