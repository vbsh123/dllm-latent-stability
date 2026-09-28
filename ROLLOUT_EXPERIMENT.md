# Actual GSM8K generation experiment

The main question is now **accuracy versus compute under actual commitment policies**.
Each policy generates its own answer and feeds its committed tokens into subsequent
model forwards. It may use different wording from the baseline. No baseline-token
agreement is used to grade its answer. The prior trajectory experiment remains optional
for explaining behavior; it is not the quality metric for these runs.

## Policies

| Method | Commitment rule |
|---|---|
| baseline | Original deterministic fixed transfer schedule; CPU parity checked against vendored upstream |
| confidence | Raw top1 probability >= tau |
| credit | Published CreditDecoding Eq6/7; enhanced top1 probability >= tau |
| latent | Persistent matched-region credit boosts current raw top1 via Eq7; enhanced confidence >= tau |
| combined | Sum token credit and current-candidate regional credit, apply Eq7, accept enhanced winner at confidence >= tau |

Credit parameters are alpha=.65,beta=.7,gamma=.2, updated before fusion. It is **not**
a raw-credit-count threshold: q=softmax(logits+alpha*log(1+C)), and max(q)>=tau accepts
q's winning token. Credit resets per output block. Its trajectory now evolves under
its own actual token insertions, not under baseline insertions.

Default checkpoint/config: pinned LLaDA-8B-Instruct, block64,256 output tokens/256
maximum forward passes, batch1, greedy, no CFG/KV cache/EOS early termination.
The same named checkpoint and block size match the paper; evaluation prompts, exact
revision, fallback and stopping details differ. This is not an exact reproduction.

## Persistent region credit and logit fusion (v2)

Each active masked position has its own bank of regions. The first finite, nonzero
hidden vector outside all saved regions creates a frozen anchor a_j=h_t and frozen
scale s_j=||a_j||_2. There is no three-vector warm-up. That visit earns credit, just
as the first top1 occurrence does in CreditDecoding. Stored anchors never move,
merge or get evicted during the block. Returning to an old region reuses its credit.

For each observation, choose the saved anchor with smallest normalized distance
`||h_t-a_j||_2/s_j` if that distance is <=r. Ties choose the oldest region. Otherwise
create a new region. Exactly one region earns evidence; overlapping regions do not
both receive it. All saved credits for the active position decay on every step:

```
R_t(j) = beta * R_(t-1)(j) + 1[j == matched_or_new_region] * p_t(raw_top1)^gamma
L_t(v) = 1[v == raw_top1] * R_t(current_region)
q_t = softmax(logits_t + alpha * log(1 + L_t))
commit argmax(q_t) wherever max(q_t) >= tau
```

This uses binary region membership, but confidence-weighted increments rather than
unit increments, matching the paper's Eq6. A distant state earns credit in a separate
region and does not erase old evidence or inherit an unrelated region's balance.
Token identity changes within a region preserve its credit. The current winner gets
the region's total evidence even if earlier visitors predicted different words.
Giving a uniform bonus to every token would cancel in softmax, so we explicitly
boost only the current raw top1. This mapping is our hypothesis, not a paper claim.

Fixed anchors prevent a region from following a walking trajectory. They do not
prove semantic equivalence or future correctness. Invalid vectors earn no regional
credit and create no region, but existing credit still decays. Inactive positions
neither earn nor decay regional credit. Entire state is released at block boundaries.

Defaults: layer16, r=.05, beta=.7, alpha=.65, gamma=.2, tau=.95. The last four values
match the fixed CreditDecoding comparison; layer and radius are untuned. No latent
support threshold, consecutive mode, reset radius or mean-drift guard remains.
Old policy JSONs containing these settings fail rather than silently changing meaning.
`mechanism=persistent_regions_v2` identifies the new policy. There is no fixed earliest
hit count: acceptance depends on the boosted probability. Even many hits need not
reach tau. For constant p=.6, the default discounted credit saturates and boosted
confidence stays below .95; forced fallback may still account for many commitments.

The optional combined ablation uses `C_token + L` inside the same log fusion, not
an OR of acceptance gates. Its stronger total evidence can itself improve confidence;
it requires tuning and is not evidence of incremental latent value by construction.
The primary comparison remains separate baseline, credit and latent runs.

**One authoritative implementation:** rollout and PolicyHistory both call
region_distribution(), which calls RegionSupport.update() and fuse_credit().
Collection retains hidden states on the model device until logits are available,
then saves policy_max_p/top1/accept/support/region_id/region_count/new_region/close.
Policy timelines threshold those boosted probabilities; the configured latent tau
is saved alongside each Credit cutoff. Old traces and legacy sliding-center rules
cannot reproduce this method and require recollection.

Banks grow as needed (geometric GPU allocation), with no eviction or raw vector dumps.
Memory scales with positions x discovered regions x hidden width x captured layers.
Only one layer is used in timed rollouts. All-layer diagnostics, particularly full-
sequence decoding with many distinct regions, can consume substantial memory: run a
small memory pilot first. No approximation silently limits retained region history.

## Forced progress and budgets

Default `--fallback top1`: if no position passes a method's gate, commit the most
confident active candidate for that method. Credit, latent and combined use their own enhanced confidence
for fallback; confidence uses raw confidence. This is an explicit addition
to Algorithm1's underspecified no-crossing behavior. Forced commits are labeled and
reported; a latent run dominated by fallback is not evidence that latent support helped.
The confidence policy isolates the effect of threshold decoding from historical credit.

`--fallback none` is the strict-threshold sensitivity. Each block has the same cap
steps/(number of blocks). If masks remain at the cap, the run is incomplete and counts
as incorrect; no silent final fill or removal from accuracy denominators. Standard
baseline uses its fixed per-step quota. At default256 steps, all policies with top1
fallback can normally finish within256 forwards. Mask-valued model proposals can still
prevent completion and are exposed rather than silently suppressed.

## Run on Vast

Upload the updated project, then bootstrap as described in VAST_RUNBOOK.md:

```bash
cd /workspace/dllm-latent-stability
bash scripts/bootstrap_vast.sh configs/credit_instruct_block64.json
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"

# Engineering smoke:3 training questions,2 blocks, shortened answers, no warmup.
python -m dllm_latent.gsm8k_rollout --split train --limit 3 \
  --gen-length 128 --steps 128 --warmup 0 --verify-parity --out runs/gsm_smoke

# Initial development comparison: real independent answers for all five methods.
python -m dllm_latent.gsm8k_rollout --split train --limit 100 \
  --verify-parity --out runs/gsm_dev100
```

The GPU guard runs before model/tokenizer/dataset loading. Do not run actual inference
locally. Short smoke accuracy is not a scientific result: outputs may be truncated
and timing is cold. Development and heldout runs use one untimed warmup per method,
then rotate method order across questions. Check memory, runtime and output examples
before paying for a larger run. There is no automatic large parameter sweep.

Tune only on train questions. CLI overrides make small comparisons explicit, e.g.:

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 100 \
  --layer 24 --radius .025 --decay .8 --latent-alpha .65 \
  --out runs/gsm_dev_variant
python -m dllm_latent.gsm8k_rollout --split train --limit 100 \
  --latent-alpha 0 --out runs/gsm_dev_no_latent_boost
```

For a different development cohort, use `--offset 100`; ordering is deterministic
and examples are selected before any generation. Default question text gets an explicit
instruction to reason and end with `#### number`, using the checkpoint's chat template.
No reference solution or gold answer is placed in the generation prompt.

Choose a development run based on its quality/compute tradeoff, then pass its saved
policy.json unchanged to heldout evaluation (replace the illustrative run below with
the run actually chosen):

```bash
python -m dllm_latent.gsm8k_rollout --split test --limit 500 \
  --policy-config runs/gsm_dev100/policy.json --out runs/gsm_test500
```

Do not tune parameters after viewing test accuracy. Freeze generation length/block
size too: policy.json stores gates, while manifest.json stores model/decoding settings.
If development changes these, supply the same --config/--gen-length/--steps/--block-length
on test. For longer reasoning,256 versus512 tokens is a development sensitivity, not
an improvement to select after inspecting test. More questions narrow uncertainty;
500 and1000 snapshots overlap and are not independent replications.

## Grading and outputs

Dataset: pinned openai/gsm8k main; train for development, test for heldout. The primary
parser takes the last standalone `#### number` line and compares Decimal-normalized
numeric values; commas, signs and decimal equivalents are supported. Missing/invalid
format is incorrect. This adapts the official GSM8K `####` convention but differs
from its first-match/string-comparison helper. The secondary last-number score is
reported separately and can over-credit incidental numbers. Format/completion rates
make these limitations visible. Generated text is cut at first EOS/EOT for grading;
computation still fills the configured span consistently for all methods.

- `generations.jsonl`: full individual answers, gold/predicted numbers, correctness,
  actual commit events, time, forward counts and completion/fallback information.
- `actual_commitments.parquet` and `paired_commitments.csv`: tokens, text, actual
  steps and decision reasons for each policy. Positions have DIFFERENT contexts
  across methods; timing differences are descriptive, not same-state causal labels.
- `summary.csv`: accuracy, completion/format rates, forced fraction, mean forwards,
  mean synchronized decoder seconds and peak GPU memory.
- `paired_comparisons.csv`: accuracy change and1000 paired-question bootstrap interval,
  improved/regressed answer counts and forward/time ratios versus baseline and Credit.
- `accuracy_vs_forwards.png`, `FINAL_EXPERIMENT_REPORT.md`, `manifest.json`, `policy.json`.

Timing includes commitment calculations, layer hooks and (only if enabled) trace collection, but
excludes model loading, dataset preparation, warmup and text grading. Tracing is OFF by default (`--no-trace` remains an explicit alias). Use `--trace`
only for a separate explanatory run. GPU counters accumulate without per-step Python
conversions, and trace payload transfers happen after the denoising loop. GPU-driven
control flow can still synchronize; no claim of a synchronization-free implementation
is made. Do not compare trace-on times with trace-off times. Do not infer wall-clock speedup from fewer forwards.
A small accuracy difference within uncertainty is not evidence of superiority; inspect
regressions, fallback use and run-to-run timing variation. No real GPU experiment has
yet been executed by the local implementation work. Use fresh output directories;
this actual-rollout runner currently does not resume partial runs.

Source: [official GSM8K grading helper](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py)
and [CreditDecoding final paper](https://aclanthology.org/2026.acl-long.509.pdf).

## First200-question comparison

After the Vast parity/smoke check passes, compare baseline,credit and latent on the
same200 training questions. This is a development pilot, not the frozen test result.

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 200 \
  --methods baseline credit latent --verify-parity --out runs/gsm_dev200
```

summary.csv now includes output_tps,full_span_tps and mean_completed_output_tokens.
Output TPS = total pre-EOS/EOT tokens from completed outputs / total decoder seconds.
Failed attempts contribute time but no delivered output tokens. Full-span TPS also
counts generated suffix tokens and is labeled separately because no EOS early stop
is used. Prompt tokens and intermediate predictions are excluded. These are ratios
of totals, not averages of individual question TPS. Read them with accuracy, mean
seconds per answer, mean forward passes, output length, formatting and completion.
Different verbosity can change TPS even without a useful speed/quality improvement.

The initial settings remain untuned. One changed outcome is0.5 percentage points
on200 questions; small differences require more evidence and paired uncertainty.
Use these train results to choose settings, then freeze policy and decoding config
before assessing an untouched test cohort. Do not call a train-tuned result heldout.

## Preflight before the200-question benchmark

Run20 exact baseline parity checks drawn across the lengths of40 development prompts,
plus one trace-on/off token/counter/forward-count check per selected method:

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 40 \
  --methods baseline credit latent --parity-only --parity-prompts 20 \
  --out runs/gsm_parity20
```

This generates parity outputs but does not run the measured benchmark. Read parity.json
for actual checked IDs,lengths and counts; a3-prompt smoke can only check3. Each actual
configuration intended for benchmarking needs parity, including changed block mode or
length. Collection's --verify-parity similarly covers up to20 length-stratified prompts.
No real-checkpoint parity has yet been run locally.

After parity passes, run the200 train questions with tracing off (the default):

```bash
python -m dllm_latent.gsm8k_rollout --split train --limit 200 \
  --methods baseline credit latent --no-trace --out runs/gsm_dev200
# Separate explanatory run on a few of the same deterministic questions:
python -m dllm_latent.gsm8k_rollout --split train --limit 3 \
  --methods baseline credit latent --trace --out runs/gsm_trace3
# Strict-gate diagnostic: no forced-progress fallback.
python -m dllm_latent.gsm8k_rollout --split train --limit 200 \
  --methods baseline credit latent --fallback none --no-trace --out runs/gsm_no_fallback200
```

Fallback means selecting one best candidate when a gate selects none. It remains
explicitly enabled by default, not silently removed. summary.csv separately reports
latent/credit/combined/confidence/fallback commitment counts and fallback fraction. It also
reports unchanged_context_steps: without fallback, a deterministic model can repeat
exactly the same masked input, producing trivially identical hidden states. That can
inflate apparent latent stability. Thus fallback=none is a diagnostic ablation with
its own caveat, not automatically the more scientifically valid primary policy.

## How many commitments did boosting enable?

The current runner saves acceptance_counts per answer and aggregates these in
summary.csv, including --no-trace runs. Counters accumulate on the GPU and transfer
only after decoding; no per-step CPU logging is required.

- boost_enabled_commits: actual non-mask insertions accepted by threshold where raw
  maximum probability < tau but the policy's enhanced maximum >= tau.
- already_confident_commits: threshold insertions with raw maximum >= tau.
- actual_fallback_commits: actual non-mask forced-progress insertions.
- actual_scheduled_commits: actual non-mask baseline schedule insertions.

These four buckets partition actual insertions. boost_enabled_fraction divides by
all actual insertions; boost_enabled_fraction_of_threshold divides only by boosted
plus already-confident threshold acceptances. Undefined denominators are blank.
boost_changed_candidate_commits separately counts threshold insertions whose enhanced
winner differs from raw top1; it overlaps the first two buckets. A position may have
passed raw threshold with a different token, so this distinction matters for Credit.
Old *_commits reason counts include attempted mask-valued proposals, while new actual
attribution excludes them. This only differs for pathological mask-valued predictions.

With --trace, commits also contain raw_max_p, raw_selected_p, raw_top1 and boost_enabled;
score remains the final policy confidence. Existing old results do not contain this
information and cannot be retroactively attributed. Reports show missing attribution
as unavailable, never as zero. The acceptance rule and generated tokens are unchanged.

This answers a local question on each method's own current state. It does not measure
causal forwards saved or prove geometry helped: earlier boosted decisions changed
subsequent inputs. Confidence-only and no-geometry controls are still necessary.

For an initial 40-question development check (change to 200 for the original cohort):

```bash
git pull --ff-only
python -m dllm_latent.gsm8k_rollout \
  --config configs/credit_instruct_block64.json \
  --policy-config runs/gsm_dev200_v2/policy.json \
  --split train --limit 40 --methods confidence credit latent \
  --no-trace --out runs/gsm_boost_audit40
```

This adds the missing raw-confidence comparator and records boost attribution for
Credit and latent. It does not rerun the expensive fixed-schedule baseline. Keep
hardware/configuration fixed and use the new run for within-run timing comparisons;
the added instrumentation has a small but unmeasured runtime cost.
