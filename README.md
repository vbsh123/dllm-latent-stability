# Training-free latent stability in diffusion language models

Standalone project: `/mnt/c/Users/User/code/dllm-latent-stability`.
The main experiment now runs **actual independent GSM8K generations** under baseline,
confidence, CreditDecoding, persistent latent-region credit and combined policies.
Quality is final numeric answer accuracy, not agreement with baseline wording.
No classifier or learned weights. Start with [ROLLOUT_EXPERIMENT.md](ROLLOUT_EXPERIMENT.md)
for mechanisms, caveats, development/heldout commands and outputs.

The earlier unchanged-baseline, all32-layer diagnostic remains available below for
mechanism analysis. Its shadow timelines do not replace actual accuracy evaluation.
Classifier prototypes remain inactive in `archive/classifier_v1/`.

## Clone and run on Vast

Use a CUDA instance with Python 3.10–3.12; an L40S is the recommended first GPU.
Inside its terminal:

```bash
git clone https://github.com/vbsh123/dllm-latent-stability.git
cd dllm-latent-stability
bash scripts/bootstrap_vast.sh configs/credit_instruct_block64.json
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"

# Check real-checkpoint parity before the measured experiment.
python -m dllm_latent.gsm8k_rollout \
  --split train --limit 40 --methods baseline credit latent \
  --parity-only --parity-prompts 20 --out runs/gsm_parity_v2

# After parity passes, run the development comparison.
python -m dllm_latent.gsm8k_rollout \
  --split train --limit 200 --methods baseline credit latent \
  --no-trace --out runs/gsm_dev200_v2
```

Bootstrap installs dependencies, checks the pinned upstream sampler, runs tests,
and downloads the chosen checkpoint on the GPU host. It does not start an experiment.
Use fresh output directories. Actual model accuracy, speed and GPU memory fit are
not yet validated; the CPU suite is engineering validation only. See
[VAST_RUNBOOK.md](VAST_RUNBOOK.md) for hardware, setup and result retrieval details.

## Additive credit hybrids

Opt-in `no_geometry_radius` adds an existing-region-match reward to unrestricted
position accumulation. `no_geometry_credit` adds unrestricted position credit to
CreditDecoding's full token-history vector before logit fusion. `--bonus-weight 1`
is the default extra weight; use layer16/radius1.00 for the requested radius hybrid.
`no_geometry_double` is an optional unconditional extra-credit control, useful because
stronger boosting alone may explain gains. See [the hybrid protocol](ROLLOUT_EXPERIMENT.md)
for the 200-question commands and exact formulas. Existing methods are unchanged.

## Early stopping and explicit TPF

Use `--early-stop` to terminate once an EOS/EOT is committed and every preceding
position is finalized. It does not stop at an uncommitted prediction or leave holes
in the delivered prefix. Prior commands default to no early stopping.

The requested 200-question comparison (Credit, no-geometry, layer16/radius .50,
and layer16/radius1.00) is prepared as:

```bash
bash scripts/early_stop_experiment.sh 200 gsm_early_stop200 runs/gsm_dev200_v2/policy.json
```

This includes a small parity preflight and 800 measured answers. All new summaries
report output_tpf, output_tpf_with_stop and full_span_tpf; do not assume 256 generated
tokens per answer with early stopping. `python -m dllm_latent.audit_tpf --help` explains
how to calculate TPF from old saved outputs without rerunning inference. See the
[current stopping/TPF protocol](ROLLOUT_EXPERIMENT.md) for exact conventions and caveats.

## Checkpoint and decoding configuration

The recommended CreditDecoding comparison now uses **LLaDA-8B-Instruct with64-token
blocks**. Base and full-sequence remain available as sensitivity configurations.

| Configuration | Checkpoint | Output block size | Prompt format |
|---|---|---|---|
| `configs/credit_instruct_block64.json` | LLaDA-8B-Instruct |64 | Checkpoint chat template |
| `configs/instruct_full.json` | LLaDA-8B-Instruct | Full sequence | Checkpoint chat template |
| `configs/base_block64.json` | LLaDA-8B-Base |64 | Plain Question/Answer |
| `configs/baseline.json` | LLaDA-8B-Base | Full sequence | Plain Question/Answer |

Instruct revision is pinned to `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
This is the same named8B checkpoint used by CreditDecoding; their exact immutable
revision is not established. The paper also tested other models, not implemented here.
Matching this checkpoint/block size still does not reproduce their early stopping,
benchmark setup, or actual accelerated rollout. We preserve full baseline trajectories.

Recommended Vast commands after uploading the project:

```bash
bash scripts/bootstrap_vast.sh configs/credit_instruct_block64.json
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
python -m dllm_latent.collect --config configs/credit_instruct_block64.json \
  --prompts data/smoke.jsonl --out runs/instruct_block_smoke \
  --gen-length 128 --steps 128 --limit 1 --verify-parity
bash scripts/full_experiment.sh 100 instruct_block100 configs/credit_instruct_block64.json
```

The smoke exercises two64-token blocks. Full runs use256 output tokens/steps and
four64-token blocks. `collect --block-length 64` or `--block-length full` can override
a configuration; generation length must be divisible by block length, and total
steps by the number of blocks. Histories begin when a block becomes active and reset
at each boundary. Only still-masked positions in that block contribute observations;
completed blocks provide context. Timeline steps stay global across all blocks.

Preparation and collection share the pinned tokenizer's chat template for Instruct,
with one user message and an assistant generation prefix. Full-run prompt snapshot
names include a configuration hash to avoid mixing Base/chat token-length filtering.
Direct Python commands default to the retained Base/full-sequence config unless
`--config` is supplied. The older Base walkthrough below remains usable.

## Current stability rule and diagnostics

Each masked position retains a bank of frozen latent-region anchors and credits.
An observation matches the closest normalized anchor within radius .05, or creates
a new region. Every visit, including the first, adds p(raw_top1)^.2 to that region;
all its saved regional credits decay by .7 each step. Returning to A after visiting
B reuses A's balance. Anchors do not move, and no old region is discarded.

The current raw top1 receives its current region's credit via the paper's fusion:
`q=softmax(logits + .65*log(1+mapped_region_credit))`. Commit using boosted confidence
>=.95, exactly the threshold used by Credit. There is no hit-count acceptance gate.
Combined sums token and mapped region credit before fusion. Defaults are untuned.

Actual decoding and observational collection call the SAME region_distribution()
and RegionSupport.update() in `decoding.py`. policy_timelines compares recorded
boosted-confidence crossings against Credit. Legacy sliding-window geometry and
`timelines.py` remain separate descriptive alternatives, not calibration engines
for this policy. Old gate configs/traces are incompatible and require recollection.
Tune layer/radius/decay/fusion on development data, then freeze for heldout evaluation.
No classifier is trained. See ROLLOUT_EXPERIMENT.md for bank-memory costs and caveats.

Read [CODE_FLOW.md](CODE_FLOW.md) for the entry points and call sequence, and
[ROLLOUT_EXPERIMENT.md](ROLLOUT_EXPERIMENT.md) for the full mechanism and benchmark
commands. Actual timing defaults to no tracing. Both collectors support length-
stratified parity checks; the rollout also has --parity-only for a separate preflight.

## Measure what boosting contributes

New runs report `boost_enabled_commits` (raw confidence below tau, boosted confidence
above it), `already_confident_commits`, and separate actual fallback/scheduled counts.
This works with `--no-trace`; old runs cannot recover the missing raw confidence.
Use `--methods confidence credit latent` to include raw-confidence parallel decoding.
See [the attribution procedure](ROLLOUT_EXPERIMENT.md#how-many-commitments-did-boosting-enable)
for a short run and exact denominators. A large boosted count alone does not prove
that hidden-state geometry helped.

## No-geometry control

`--methods no_geometry latent --layer 16 --radius 0.50` compares unconditional
per-position credit accumulation with the latent region method at ten times the
original radius. No-geometry uses the same probability weighting, decay, logit fusion,
threshold and fallback, but never reads hidden states. It is an optional control,
not a replacement for the main method. See
[the paired control run](ROLLOUT_EXPERIMENT.md#no-geometry-control-versus-a-much-larger-region)
for the 200-question command and region-creation/reuse metrics.

## CPU checks

Use Python 3.10–3.12. No full checkpoint is downloaded or executed locally.

```bash
cd /mnt/c/Users/User/code/dllm-latent-stability
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
python -m pytest -q
python -m dllm_latent.synthetic --out runs/synthetic --prompts 60
python -m dllm_latent.timelines --run runs/synthetic --out runs/legacy_synthetic_timelines --bootstrap 30
```

The synthetic generator uses small vectors and four representative layers plus norm
for fast plumbing checks; the collector and fake-model parity test cover all 32.
Synthetic findings provide no scientific evidence. scikit-learn supplies descriptive
metrics only; the active code fits no estimator.

## Vast smoke, collection and full experiment

For instance selection, local upload, setup, profiling and result retrieval, follow
[VAST_RUNBOOK.md](VAST_RUNBOOK.md).

Copy the project to `/workspace/dllm-latent-stability`, then:

```bash
cd /workspace/dllm-latent-stability
bash scripts/bootstrap_vast.sh
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
python -m dllm_latent.collect --prompts data/smoke.jsonl --out runs/smoke \
  --gen-length 32 --steps 32 --verify-parity
python -m dllm_latent.policy_timelines --run runs/smoke --out runs/smoke_timelines

python -m dllm_latent.prompts --count 100 --out data/prepared_base_100.jsonl
python -m dllm_latent.collect --prompts data/prepared_base_100.jsonl --out runs/initial100 --verify-parity
python -m dllm_latent.policy_timelines --run runs/initial100 --out runs/initial100_timelines
python -m dllm_latent.policy_timelines --run runs/initial100 --out runs/initial100_pre_eos --pre-eos-only

bash scripts/full_experiment.sh 500 real500
bash scripts/full_experiment.sh 1000 real1000
```

Bootstrap installs pinned dependencies, the upstream sampler and checkpoint on the
GPU host. Use a CUDA 12.4 compatible driver, BF16 GPU with about 40 GB VRAM, Python
3.10–3.12, at least 32 GB host RAM and about 50 GB initial disk plus trajectory storage.
`DOWNLOAD_MODEL=0` defers model download until collection. Real GPU fit and checkpoint
parity are unverified; profile the small smoke before scaling. Full scripts also run
a separately tuned pre-EOS sensitivity, which must not replace the primary result
based on favorable heldout outcomes.

For the retained Base/full-sequence configuration: 256 output tokens, 256 steps, batch 1, temperature 0, CFG 0, no cache/early
stop. All still-masked output positions are eligible from the first step. The upstream
API uses `block_length=gen_length`; there are **no sequential decoding blocks**.
Base prompts use plain `Question: ...\nAnswer:` continuation with no chat template;
`prompt_format: raw` is configurable. Pinned Dolly QA/creative and GSM8K prompts are
balanced, deduplicated and filtered to the 512-token prompt cap without truncation.

All 32 residual outputs are captured **after each complete transformer layer, before
the next layer**, not attention-only outputs. A separate `final_norm` capture observes
the final RMSNorm output immediately upstream of the language-model head. RMSNorm
rescales activations by their root-mean-square magnitude, with learned per-coordinate
scales; unlike LayerNorm it does not subtract the mean. Layer32 and final_norm are
distinct tensors. See DESIGN_DECISIONS.md for exact revisions and geometry.

All-layer capture adds copying/statistics, not model forwards. Shared-policy state now
also lives on the model device for every captured layer, so GPU memory increases;
profile the Vast smoke before scaling. Six previous float32
vectors per layer are retained (largest K+C=7 includes current); for 32 layers,
256 positions and width4096, that is about **768 MiB host RAM**, excluding final norm,
current captures, temporaries, output histories and scalar rows. Full-sequence decoding
produces 32,896 rows per prompt. Scalar columns across all layers can dominate storage
and RAM even without raw-vector dumps. `policy_timelines` reads one generation/layer at a
time and stores per-position summaries. The optional descriptive audit below loads
all scalar rows and needs more host memory. Collector timings include observation
overhead and are unsuitable for speedup claims.

## Outputs and supplementary diagnostics

Shared-policy timeline outputs include comparisons.csv, per-layer first-trigger and
Credit comparison Parquet, paired intervals and the exact policy.json. There is no
automatic offline radius search: the policy was evaluated during collection. Legacy
anchored-window tuning outputs belong only to explicitly labeled legacy experiments.

The separate **descriptive cosine audit** provides confidence/lexical/credit-conditioned
rates, changed-top1 examples, AUROC/AUPRC, and Brier/calibration for probability scores:

```bash
python -m dllm_latent.analyze --run runs/initial100 --out runs/initial100_descriptive
python -m dllm_latent.analyze --run runs/initial100 --out runs/initial100_stable --target stable_to_end
```

This audit retains the earlier fixed layer16/K3/cosine.99 comparator, its full fixed
grid and 80% exploratory/20% heldout split (same heldout hash boundary). It does **not**
evaluate the tuned anchored rule and must not be presented as doing so. Credit scores
there refer to the raw candidate; the timeline module uses Credit's actual winner.
Layer/window/radius decisions must use the timeline partition protocol, not heldout
plots. Supplementary audits exclude commitment rows by default; timelines retain them
to locate baseline commitment but count only earlier proposals for precision.

SAFE_TO_COMMIT means agreement with the final baseline token. STABLE_TO_END separately
means raw top1 never changes again through baseline commitment, after which the state
is fixed. Neither label establishes semantic correctness or safety after early insertion.
The pre-final-EOS filter is retrospective cohort analysis and never an online feature.
Use fresh directories after configuration/code changes; collection resumes only with
an identical manifest. Old traces without shared policy_accept decisions and their policy manifest must
be recollected. No actual model inference or scientific conclusions yet.


### Radius expansion and strength controls

Run the same 200 development questions (three methods, layer16/radius2,
bonus weight1, early stop, tracing off):

```bash
bash scripts/radius_strength_experiment.sh 200 gsm_radius200_strength runs/gsm_dev200_v2/policy.json
```

Results: `runs/gsm_radius200_strength_measured/summary.csv` and
`paired_comparisons.csv`. The script first runs a three-prompt parity check.
Double credit is unconditional from the first observation. For the additional
first-visit-matched control, run:

```bash
python -m dllm_latent.gsm8k_rollout \
  --config configs/credit_instruct_block64.json \
  --policy-config runs/gsm_dev200_v2/policy.json \
  --split train --limit 200 --methods no_geometry_double \
  --bonus-weight 1 --double-bonus-start 2 \
  --early-stop --no-trace --out runs/gsm_double_delayed200
```

This delays only the extra credit until a position's second active observation;
base credit still starts immediately. Other methods ignore this option. Keep
both controls distinct when combining results. Existing answer-extraction issues
remain; this development round does not establish equal task accuracy.


### Double base plus radius2 versus unconditional triple credit

```bash
python -m dllm_latent.gsm8k_rollout \
  --config configs/credit_instruct_block64.json \
  --policy-config runs/gsm_dev200_v2/policy.json \
  --split train --limit 200 \
  --methods no_geometry_radius no_geometry_double \
  --layer 16 --radius 2.0 --base-weight 2 --bonus-weight 1 \
  --double-bonus-start 1 --early-stop --no-trace \
  --out runs/gsm_doublebase_radius2_vs_triple200
```

This runs200 questions per method. Under these settings, no_geometry_radius earns
2x pmax^gamma on new anchors and3x on existing-region matches; no_geometry_double
earns3x unconditionally, including the first observation. The name is retained for
compatibility; inspect policy.json to identify strength. --base-weight defaults to1
and affects only these two methods. Fusion, decay and acceptance remain unchanged.


### Audit saved answers and metrics (no inference)

On Vast, after the selected runs have finished:

```bash
python -m dllm_latent.audit_gsm8k \
  --run runs/gsm_unconditional6x_200 runs/gsm_unconditional12x_200 \
  --out runs/grading_audit_6x_12x
```

Reads existing files; no model or tokenizer download. Outputs summary.csv,
integrity_issues.csv, answers.csv, review.csv and paired_legacy.csv. Original outputs
are unchanged. An integrity issue returns exit status1 after writing reports. Use
only run paths present on this instance, and a fresh audit output directory.
`automatic_correct_fraction_all_questions` is provisional: ambiguous cases count
as unresolved rather than silently becoming correct. Inspect full answers in review.csv
and sample unflagged answers before claiming corrected accuracy. See
[EVALUATION_AUDIT.md](EVALUATION_AUDIT.md) for confirmed bugs and audit limits.


### Pinned OpenCompass scoring of the existing 3x run

```bash
python -m dllm_latent.opencompass_score \
  --run runs/gsm_doublebase_radius2_vs_triple200 \
  --methods no_geometry_double \
  --out runs/opencompass_3x
```

No inference or download. Requires the original generations.jsonl, questions.jsonl
and manifest.json; recover these from the old instance/backup if necessary. This
run's no_geometry_double used base2+bonus1, so it is the3x control. Output includes
summary.csv (accuracy in PERCENT), answers.csv, disagreements.csv and a provenance
manifest. Omit --methods to score every method. Original scoring code is vendored
with revision/hash/license in dllm_latent/third_party/opencompass. This uses standard
OpenCompass GSM8K scoring, not a verified CreditDecoding evaluation configuration.

Before deleting a GPU instance, preserve the entire run directory, not just summary.csv.
It contains generated answers, token IDs, questions, policy and reproduction metadata.
Cloud destination/authentication must be configured separately; git does not back up runs.
