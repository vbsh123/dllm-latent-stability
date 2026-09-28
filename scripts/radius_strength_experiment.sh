#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
count="${1:-200}"
tag="${2:-gsm_radius200_strength}"
policy="${3:-configs/rollout_policy.json}"
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Question count must be positive'; exit 1; }
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run tag'; exit 1; }
[[ -f "$policy" ]] || { echo "Missing policy: $policy"; exit 1; }
for suffix in parity measured; do
  [[ ! -e "runs/${tag}_${suffix}" ]] || { echo "Output exists: runs/${tag}_${suffix}; choose a fresh tag"; exit 1; }
done
common=(--config configs/credit_instruct_block64.json --policy-config "$policy"
        --split train --early-stop --layer 16 --radius 2.0 --bonus-weight 1
        --double-bonus-start 1 --no-trace
        --methods latent no_geometry_radius no_geometry_double)
python -m dllm_latent.gsm8k_rollout "${common[@]}" --limit 3 \
  --parity-only --parity-prompts 3 --out "runs/${tag}_parity"
python -m dllm_latent.gsm8k_rollout "${common[@]}" --limit "$count" \
  --out "runs/${tag}_measured"
