#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
experiment_config="${1:-configs/baseline.json}"
# Run on a Vast CUDA image with Python 3.10-3.12, >=40 GB VRAM and >=32 GB host RAM.
python3 - <<'PY'
import sys
assert (3,10) <= sys.version_info[:2] < (3,13), 'Use Python 3.10-3.12 for the pinned environment'
PY
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[gpu,test]'
mkdir -p runs .cache vendor data
export HF_HOME="$PWD/.cache/huggingface"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Use a Vast CUDA GPU image with compatible driver'
assert torch.cuda.is_bf16_supported(), 'Baseline requires BF16 support'
print(torch.cuda.get_device_name(0))
print('GPU GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
if [[ ! -d vendor/LLaDA/.git ]]; then
  git clone https://github.com/ML-GSAI/LLaDA.git vendor/LLaDA
fi
git -C vendor/LLaDA fetch origin 9182493720ed723ef8031210d85959364e51cbe0
git -C vendor/LLaDA checkout --detach 9182493720ed723ef8031210d85959364e51cbe0
cmp vendor/LLaDA/generate.py dllm_latent/third_party/llada_generate.py
python -m pytest -q
# Checkpoint download happens only on the GPU host; DOWNLOAD_MODEL=0 defers it.
if [[ "${DOWNLOAD_MODEL:-1}" == 1 ]]; then
  python - "$experiment_config" <<'PY'
import json
import sys
from huggingface_hub import snapshot_download
cfg=json.load(open(sys.argv[1]))
snapshot_download(cfg['model_id'],revision=cfg['model_revision'])
PY
fi
python -m pip freeze > runs/bootstrap_packages.txt
printf '%s\n' 'Ready. source .venv/bin/activate; export HF_HOME="$PWD/.cache/huggingface"' 'See README.md for smoke, collection, analysis, and full-run commands.'
