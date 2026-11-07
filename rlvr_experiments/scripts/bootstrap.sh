#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="$(dirname -- "$ROOT")"
UV=${UV:-/home/nlp/miniconda3/bin/uv}
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/python"
export CUDA_HOME=/usr/local/cuda-12.8
mkdir -p "$ROOT/env" "$ROOT/runs"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv "$ROOT/.venv" --python 3.12 --python-preference managed
fi
"$UV" --no-config pip install --python "$ROOT/.venv/bin/python" --torch-backend cu128 -r "$ROOT/env/requirements-cu128.in"
"$UV" --no-config pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$VERL_ROOT"
"$UV" --no-config pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/env/requirements.resolved.txt"
"$ROOT/.venv/bin/python" -m pip check 2>/dev/null || "$UV" --no-config pip check --python "$ROOT/.venv/bin/python"
