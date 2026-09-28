#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
count="${1:-200}"
tag="${2:-gsm_early_stop}"
policy="${3:-configs/rollout_policy.json}"
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Question count must be positive'; exit 1; }
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Run tag must contain letters, numbers, underscores or hyphens'; exit 1; }
for suffix in parity radius050 radius100; do
  [[ ! -e "runs/${tag}_${suffix}" ]] || { echo "Output exists: runs/${tag}_${suffix}; choose a fresh tag"; exit 1; }
done
[[ ! -e "runs/${tag}_tpf.csv" ]] || { echo 'TPF output exists; choose a fresh tag'; exit 1; }
common=(--config configs/credit_instruct_block64.json --policy-config "$policy"
        --split train --early-stop --layer 16 --no-trace)
# Check baseline without stopping against upstream, then verify early-stop prefix
# and trace invariance for each selected method on one of these three prompts.
python -m dllm_latent.gsm8k_rollout "${common[@]}" --limit 3 \
  --methods credit no_geometry latent --radius 0.50 \
  --parity-only --parity-prompts 3 --out "runs/${tag}_parity"
python -m dllm_latent.gsm8k_rollout "${common[@]}" --limit "$count" \
  --methods credit no_geometry latent --radius 0.50 --out "runs/${tag}_radius050"
python -m dllm_latent.gsm8k_rollout "${common[@]}" --limit "$count" \
  --methods latent --radius 1.00 --out "runs/${tag}_radius100"
python -m dllm_latent.audit_tpf \
  --run "runs/${tag}_radius050" "runs/${tag}_radius100" --out "runs/${tag}_tpf.csv"
