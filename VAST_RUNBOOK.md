# New early-stop comparison

After activating the environment and pulling current main, run:

```bash
bash scripts/early_stop_experiment.sh 200 gsm_early_stop200 runs/gsm_dev200_v2/policy.json
```

This performs a small parity preflight, then compares Credit, no_geometry, latent
layer16/radius .50 and latent layer16/radius1.00 on 200 questions each. Every method
uses finalized-prefix EOS/EOT stopping. It adds explicit TPF columns and preserves
old fixed-span results in their original directories. Detailed stopping semantics,
denominators and an offline historical TPF audit are in ROLLOUT_EXPERIMENT.md.
The older commands below omit --early-stop and retain the previous fixed-span mode.

# Policy update before running

Use the current configs/rollout_policy.json (persistent_regions_v2). Latent now keeps
multiple frozen regions and boosts logits with saved credit; no support-count gate
or three-observation warm-up remains. Old saved policy JSONs are incompatible. Existing
commands below still apply. Upload the updated source and config together, use fresh
run directories, and rerun parity before collecting benchmark timings. All-layer
full-sequence diagnostics may need substantial memory if many regions are discovered;
start with the small pilot. See ROLLOUT_EXPERIMENT.md for the current exact mechanism.

# Run the experiment on Vast

Recommended comparison: LLaDA-8B-Instruct,64-token blocks, all32 layers.
Base/full-sequence remains selectable through configs/baseline.json.

## Hardware and template

Recommended starting rental: one L40S (48 GB GPU memory), about8 CPU cores,
64 GB system RAM (32 GB minimum), and200 GB disk for setup and an initial campaign.
These are capacity recommendations, not measured requirements. Check disk growth
on the pilot before launching500–1000 prompts. All-layer observation uses considerable
CPU memory and time as well as GPU inference.

NVIDIA specifies48 GB for [L40S](https://www.nvidia.com/en-us/data-center/l40s/)
and24 GB for [RTX4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
An ordinary4090 is not a40 GB card. A24 GB4090 may fit some configurations, but the
current complete experiment is unprofiled; prefer the L40S for the first run. A listing's
system RAM is distinct from GPU memory. Two24 GB GPUs do not provide one48 GB allocation:
the current collector loads the whole model on one GPU and does not shard it.

Select an SSH-enabled PyTorch/CUDA Linux template with Python3.10–3.12 and a host
driver compatible with CUDA12.4. The bootstrap creates its own virtual environment
and installs pinned PyTorch2.6.0+cu124. The template needs git, python3/venv and tar;
most PyTorch templates already provide these. Do not use Python3.13 for these pins.
Use an on-demand rental for the initial setup to avoid interruption during profiling.
No instance has been rented or real checkpoint executed by this local setup.

## Clone on the remote instance

Connect using the SSH command displayed by Vast, or open the instance terminal.
The project is public and does not require GitHub credentials:

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/vbsh123/dllm-latent-stability.git
cd dllm-latent-stability
```

This is a standalone project; cloning RollingForcing does not install it.
For future updates, run `git pull --ff-only` inside this checkout before starting
new runs. Keep old result directories so their manifests identify the source used.

## Bootstrap on the remote instance

Run the following inside the Vast terminal. Use the existing tmux session if Vast
opens one automatically; otherwise start `tmux new -s dllm` before long commands.

```bash
cd /workspace/dllm-latent-stability
nvidia-smi
python3 --version
bash scripts/bootstrap_vast.sh configs/credit_instruct_block64.json
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
```

Bootstrap installs dependencies, verifies CUDA/BF16 support, fetches the pinned
upstream code, checks its sampler matches the vendored copy, runs CPU tests, and
downloads the pinned Instruct checkpoint **on the instance**. No paid experiment is
automatically launched. `DOWNLOAD_MODEL=0 bash scripts/bootstrap_vast.sh configs/credit_instruct_block64.json` defers the
checkpoint download until collection if desired.

## Main experiment: actual GSM8K generations

After bootstrap/activation above, run:

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 3 \
  --gen-length 128 --steps 128 --warmup 0 --verify-parity --out runs/gsm_smoke
python -m dllm_latent.gsm8k_rollout --split train --limit 100 \
  --verify-parity --out runs/gsm_dev100
```

Before the200-question benchmark, verify20 prompts across the lengths of40 candidates:

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 40 \
  --methods baseline credit latent --parity-only --parity-prompts 20 \
  --out runs/gsm_parity20
```

Tracing is now off by default; use --trace only for a separate explanatory run.

For the first200-question three-method comparison:

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 200 \
  --methods baseline credit latent --verify-parity --out runs/gsm_dev200
```

Read summary.csv for accuracy,output_tps,full_span_tps,seconds per answer and forwards.
The pre-EOS output TPS and suffix-inclusive TPS are separated to expose suffix bias.

Each method actually commits tokens and generates its own answer. See
[ROLLOUT_EXPERIMENT.md](ROLLOUT_EXPERIMENT.md) for accumulated support, tuning and frozen
heldout commands. Evaluate final-answer accuracy, actual forwards and decoder time.
The small smoke is for execution checks; shortened answers and cold timing are not
scientific evidence. The following older commands are OPTIONAL baseline-trajectory
diagnostics and do not run these actual competing policies.

## Smoke, full-length pilot, then100 prompts

First run one supplied prompt with two64-token blocks, all layers, and an
observed-versus-unobserved output parity check:

```bash
python -m dllm_latent.collect --config configs/credit_instruct_block64.json \
  --prompts data/smoke.jsonl --out runs/smoke128 \
  --gen-length 128 --steps 128 --limit 1 --verify-parity
python -m dllm_latent.policy_timelines --run runs/smoke128 --out runs/smoke128_timelines
```

No qualified rule is expected from such a small sample: calibration requires10
contributing prompt groups. Smoke tests execution, not scientific performance.

Then profile the three supplied prompts at the default256-token/256-step length:

```bash
python -m dllm_latent.collect --config configs/credit_instruct_block64.json \
  --prompts data/smoke.jsonl --out runs/pilot256 --verify-parity
du -sh runs/pilot256
df -h /workspace
free -h
```

Inspect `runs/pilot256/*.json` for `peak_gpu_bytes`, runtime and generated output,
and `parity.json` for identical output. The short pilot does not fully stress the
512-token maximum input length; use the real prompt campaign to verify that margin.
Collector runtime includes observation and first-prompt parity overhead and must
not be interpreted as decoder performance.

Once the pilot works, run preparation, collection, tuning/heldout analysis and a
separate pre-EOS sensitivity:

```bash
bash scripts/full_experiment.sh 100 initial100 configs/credit_instruct_block64.json
```

Equivalent individual commands, useful for inspecting each stage:

```bash
python -m dllm_latent.prompts --config configs/credit_instruct_block64.json \
  --count 100 --out data/prepared_instruct_100.jsonl
python -m dllm_latent.collect --config configs/credit_instruct_block64.json \
  --prompts data/prepared_instruct_100.jsonl \
  --out runs/initial100 --verify-parity
python -m dllm_latent.policy_timelines --run runs/initial100 --out runs/initial100_timelines
python -m dllm_latent.policy_timelines --run runs/initial100 \
  --out runs/initial100_pre_eos --pre-eos-only
```

Use either the script or individual commands; analysis refuses to overwrite a
nonempty directory. Collection can resume completed shards with an identical
manifest. After changing code or scientific configuration, use a fresh run directory.

## Results and scaling

Read `runs/initial100_timelines/FINAL_EXPERIMENT_REPORT.md`, `frozen_rules.json`,
`primary_rules.json`, `heldout_comparisons.csv`, `credit_vs_latent_examples.csv`,
and the plots. If nothing qualifies, comparison/example plots may be absent;
inspect discovery/calibration failures instead of relaxing rules on heldout data.
These are first-trigger decisions on the unchanged baseline, not actual speedups.

Once pilot disk usage and memory are understood:

```bash
bash scripts/full_experiment.sh 500 real500 configs/credit_instruct_block64.json
# Or, as an alternative larger campaign:
bash scripts/full_experiment.sh 1000 real1000 configs/credit_instruct_block64.json
```

Do not treat overlapping100/500/1000 prompt snapshots as independent replication.
If you use heldout findings to change the method, reserve genuinely new confirmation
prompts. All-layer statistics can create large datasets; measure rather than assuming
the initial200 GB allocation suffices for1000 prompts.

For optional descriptive confidence/lexical-history controls, see README.md's
supplementary audit commands. That audit loads all rows in host RAM and uses the
older cosine comparator, not the tuned anchored region.

## Retrieve results

From your **local WSL terminal**, copy the run files before destroying the instance:

```bash
scp -P PORT -r root@HOST:/workspace/dllm-latent-stability/runs ./vast-dllm-results
scp -P PORT -r root@HOST:/workspace/dllm-latent-stability/data ./vast-dllm-data
```

Keep both analysis outputs and trajectory/manifests/prompt snapshots so results can
be audited. Then manage the rented instance in Vast. Model weights need not be copied
back to your local machine.
