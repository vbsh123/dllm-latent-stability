#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
count="${1:-100}"
tag="${2:-phase1_${count}}"
experiment_config="${3:-configs/baseline.json}"
policy_config="${4:-configs/rollout_policy.json}"
config_key=$(python - "$experiment_config" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()[:12])
PY
)
prompt_file="data/prepared_${config_key}_${count}.jsonl"
if [[ ! -f "$prompt_file" ]]; then
  python -m dllm_latent.prompts --config "$experiment_config" --count "$count" --out "$prompt_file"
fi
python -m dllm_latent.collect --config "$experiment_config" --policy-config "$policy_config" --prompts "$prompt_file" --out "runs/$tag" --verify-parity
python -m dllm_latent.policy_timelines --run "runs/$tag" --out "runs/${tag}_timelines"
python -m dllm_latent.policy_timelines --run "runs/$tag" --out "runs/${tag}_pre_eos" --pre-eos-only
