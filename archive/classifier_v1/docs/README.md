# Hidden trajectory versus output history: Phase 1

An observational falsification experiment for LLaDA-8B-Instruct. It runs the pinned,
unmodified upstream sampler, collects causal features, and compares output history
against output history plus hidden trajectories. No new decoder is implemented.

**Status: CPU tests and synthetic integration only. Real model results require Vast.**
Do not interpret synthetic metrics as evidence for or against the research hypothesis.

The project is isolated from RollingForcing's video dependencies. Read
[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), [CREDITDECODING_NOTES.md](CREDITDECODING_NOTES.md),
and [FINAL_EXPERIMENT_REPORT.md](FINAL_EXPERIMENT_REPORT.md).

## Vast bootstrap and smoke

Copy this repository (or this entire experiment directory) to the instance, then:

```bash
cd /workspace/RollingForcing/experiments/dllm_latent
bash scripts/bootstrap_vast.sh
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
python -m dllm_latent.collect --prompts data/smoke.jsonl --out runs/smoke \
  --gen-length 32 --steps 32 --block-length 16 --verify-parity
```

Use a CUDA 12.4-compatible image/driver, Python 3.10–3.12, about 40 GB GPU memory,
at least 32 GB host RAM, and roughly 50 GB free disk for checkpoint, environment,
and initial trajectories (more for large campaigns). Actual VRAM/time are recorded;
no GPU fit claim has been validated locally. The bootstrap downloads weights on the
GPU host; `DOWNLOAD_MODEL=0 bash scripts/bootstrap_vast.sh` defers that download.
It installs the model's pinned remote code via Transformers and checks out LLaDA
at the pinned source revision. No compilation/FlashAttention/cache optimization.

The smoke tests three prompt types and compares the first observed output with a
fresh unobserved upstream run. Check `parity.json`, per-generation JSON outputs,
row counts and peak memory before scaling. Three prompts are deliberately insufficient
for a scientific classifier comparison; analysis refuses fewer than ten prompt groups.

## Initial 100 prompts

```bash
python -m dllm_latent.prompts --count 100 --out data/prepared_100.jsonl
python -m dllm_latent.collect --prompts data/prepared_100.jsonl --out runs/initial100 --verify-parity
python -m dllm_latent.analyze --run runs/initial100 --out runs/initial100_analysis
```

The prepared snapshot balances Dolly open QA, GSM8K reasoning, and Dolly creative
writing. Dataset revisions, source row IDs and snapshot hash are recorded. Reference
answers never enter the prompt or predictors. Custom JSONL accepts `id`, `text`, `task`.
Identical normalized texts share a split even under different IDs. Near duplicates
need a separate corpus audit. Oversized prompts are rejected, never silently truncated.

Collection resumes completed generation shards when the entire manifest matches.
A changed config, prompt snapshot, package environment or source requires a new run
directory. Analysis always requires a fresh output directory. Never merge directories
from different configurations as if they were one experiment.

## 500–1,000 prompts and sensitivity runs

```bash
bash scripts/full_experiment.sh 500 real500
bash scripts/full_experiment.sh 1000 real1000
```

This runs collection, SAFE analysis, separate STABLE analysis and pre-EOS sensitivity.
Optional additional checks (new output directory each time):

```bash
python -m dllm_latent.analyze --run runs/real500 --out runs/real500_commit_rows --include-commit-rows
python -m dllm_latent.analyze --run runs/real500 --out runs/real500_seed42 --seed 42
python -m dllm_latent.analyze --run runs/real500 --out runs/real500_all_rows --max-rows-per-generation 0
python -m dllm_latent.collect --prompts data/prepared_100.jsonl --out runs/block32 --block-length 32
python -m dllm_latent.collect --prompts data/prepared_100.jsonl --out runs/steps128 --steps 128
python -m dllm_latent.collect --prompts data/prepared_100.jsonl --out runs/long512 --gen-length 512 --steps 512
```

Analyze those collection variants separately. Long-form prompts in the primary run
still have only 256 output tokens; the 512-token sensitivity is necessary to assess
longer generations. Default 256 steps yields one baseline commit per step; 128 steps
checks a genuinely parallel fixed-schedule baseline. GPU observer overhead is not
decoder latency: do not use this pipeline to claim an inference speedup.

## Outputs and interpretation

Each generation produces scalar-feature Parquet plus prompt/final-output JSON.
The collector tracks still-masked positions in the active block from its activation
through the commit step. Other blocks' predictions and post-commit logits are excluded.
Hidden hooks capture residual outputs after transformer blocks 8/16/24/32 and after
final RMSNorm. No raw hidden vectors or full vocabulary distributions are written.

`safe_to_commit`: current raw top-1 equals final baseline token.
`stable_to_end`: raw top-1 remains that same token through its eventual commitment;
thereafter baseline state is absorbing. Thus A→B→A can be SAFE at the first A but
not STABLE. Labels are created only after generation. Neither means semantic truth
or proves safety under counterfactual early insertion.

Main classifier analysis excludes commit-step rows (trivial SAFE positives), keeps
all pre-EOS and post-EOS positions, and samples up to 512 eligible rows per generation
for manageable host memory. All feature groups use exactly the same sampled rows.
The complete trajectories remain on disk; recurrence/examples use all held-out rows.
Run the pre-EOS and all-row sensitivities to examine EOS and length/survival biases.

Artifacts include:

- `FINAL_EXPERIMENT_REPORT.md`, `ANALYSIS_REPORT.md`, `metrics.json`, frozen validation thresholds and achieved
  held-out precision/recall at requested 95/97/99% levels; descriptive test PR-envelope
  recall is separately labeled and must not be used as an operational threshold.
- AUROC, AUPRC, Brier, ECE, calibration tables, PR curves, task breakdowns,
  prompt-macro Brier, paired prompt-cluster bootstrap intervals for B versus D.
- `feature_groups.json`, `split.json`, `sampling.csv`, `cohorts.csv`,
  `analysis_config.json`, `heldout_predictions.parquet` for auditing.
- Confidence × lexical-streak × latent-stability tables, changed-token subsets,
  output-risk controls, recurrence denominators, examples including failures,
  and layer/convergence/progress plots.

Split uses a seeded hash of normalized prompt text: train/validation/test probability
60/20/20%, stable across larger snapshots and cohort filters. Logistic regression
uses train-only imputation/scaling, fixed C=1, no class balancing. Validation selects
score thresholds with at least 50 selected rows and reports empirical precision;
test precision is not guaranteed. No threshold meeting that rule means abstention,
zero recall and undefined precision. Cluster intervals address within-prompt dependence,
but not model-fitting uncertainty. Small bins and 99% precision need many prompts.

The primary falsification comparison is B versus D. A gain only over A or C is
insufficient. Layer tests and onset analyses are exploratory. A proposed practical
screen is ≥.002 Brier improvement with a positive 95% paired cluster interval, plus
≥.01 recall gain at 97% target while both held-out precisions reach 97%. It is an
explicit autonomous choice requiring sensitivity analysis, not a significance theorem.
Phase 2 stays deferred until replicated, robust incremental evidence exists.

## Local CPU verification (no checkpoint)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
python -m pytest -q
python -m dllm_latent.synthetic --out runs/synthetic --prompts 60
python -m dllm_latent.analyze --run runs/synthetic --out runs/synthetic_analysis --bootstrap 30
```

Only synthetic arrays and a tiny fake denoiser are used locally. `collect` refuses
CPU execution before calling any Hugging Face loading function. The model source,
metadata and research paper may be inspected without downloading model weights.
