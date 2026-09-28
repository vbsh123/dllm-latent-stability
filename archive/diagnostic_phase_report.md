# Training-free Phase1 implementation report

**Current update:** Instruct/block64 is now supported and recommended for the closer
CreditDecoding comparison; Base/full remains available. See the update section below,
which supersedes earlier Base-only/full-sequence-only scope statements.

**No real-model results yet.** This is an implementation and CPU-validation report.
The user's correction removed classifiers; later corrections selected Base,
full-sequence decoding, all layers, closeness tuning and timeline diagnostics.

## 1. What was implemented

A standalone project with unchanged upstream baseline generation, all-layer observer,
causal output/latent history, frozen-anchor stability checks, explicit tolerance
selection, first-trigger comparisons, uncertainty summaries, plots, tests and Vast
bootstrap/run scripts. Collection never inserts tokens or boosts logits. Earlier
classifiers remain inactive under archive/classifier_v1.

## 2. CreditDecoding interpretation/reproduction

The final ACL paper was read, including Eq6/7, Algorithm1, adaptive settings and
failure analysis. We retain full-vocabulary decayed candidate credit and normalize
credit-enhanced probabilities. Fixed parameters are alpha.65/beta.7/gamma.2; adaptive
uses gamma1 and alpha=beta=1-mask_ratio. Current evidence updates before fusion.
Enhanced winner IDs are retained because they can differ from raw top1.

The timeline is the first enhanced-max-probability crossing at .95/.97/.99 on the
unchanged baseline states. It is an observational shadow timeline, **not an actual
CreditDecoding rollout or measured commitment time**. The first proposal remains
counted even if wrong. Missing crossings are censored at baseline commitment.
No forced-progress fallback or model feedback is simulated. See
CREDITDECODING_NOTES.md for equations, approximations and paper comparisons.

Main paper checkpoints were LLaDA-8B-Instruct and LLaDA-MoE-Instruct; additional
checks used SDAR-8B-Chat-b32, LLaDA2-Mini and LLaDA2-Flash. Main methods were standard
decoding, Fast-dLLM without KV cache, and CreditDecoding. Their main block64/early-stop
setup and Instruct-tuned parameters differ from our Base/full-sequence experiment.
Source: https://aclanthology.org/2026.acl-long.509.pdf, Tables1–3 and section5.

## 3. Experimental setup

LLaDA-8B-Base revision0f2787f2d87eac5eed8a087d5ecd24277e6255b2; upstream sampler
9182493720ed723ef8031210d85959364e51cbe0. Default256 tokens/256 steps, full span,
temperature0/CFG0, batch1/BF16, no cache or early stop. Plain Question/Answer
continuation; pinned balanced Dolly QA/creative and GSM8K prompts. All32 post-layer
residual outputs plus separate final-RMSNorm output; hidden width4096.

From K reference vectors, freeze their mean and RMS vector-norm scale. Require C
subsequent vectors within r of that center, reference-cloud radius <=r/2, and each
rolling K-center displacement from the same anchor <=r/4. All distances use the
same frozen scale. Search K2/3/4, C2/3, r=.01/.025/.05/.1/.2/.4. The anchor is never
moved to accommodate confirmation points. This limits measured drift, not future drift.

Prompt-hash partitions:60% discovery,20% calibration,20% heldout. Nominate one setting
per layer/precision target on discovery, maximizing correct early lead per position.
Minimum50 early events across10 groups, independently required on discovery and
calibration. Requested empirical precision targets95/97/99%; failed calibration
means abstention, without a runner-up search. Primary layer is designated from
discovery performance among survivors; heldout results do not choose it. This is
label-based hyperparameter tuning, not a trained classifier or guaranteed precision.

## 4. Main results

Scientific results are pending Vast inference.36 CPU tests passed. A60-generation
random synthetic pipeline qualified no rule; a separate60-generation engineered
stationary/correct fixture exercised the acceptance, calibration and heldout paths.
Neither fixture is evidence for latent convergence or any useful real-model radius.
The second fixture deliberately makes acceptance correct and must never be presented
as a positive research result. VALIDATION.md records checks and limitations.

## 5. Output history versus output history plus latent history

The primary comparison asks when a frozen latent rule first proposes a token versus
Credit's first threshold crossing, whether each candidate agrees with the final
baseline, and how much observed lead occurs. Report safe and unsafe earlier proposals,
achieved precision, coverage, and paired prompt uncertainty. If credit never crosses,
report censoring and only a lower bound on shadow lead. Actual baseline commit time
is recorded separately. Raw confidence and adaptive-credit timelines are also saved.

The older optional descriptive audit contains confidence/streak/credit-conditioned
rates and an output-OR-cosine union. It uses its fixed cosine comparator, **not the
tuned anchor**. Coarse conditioning cannot establish that latent history is independent
of every possible output-history signal. No claim that the hypothesis succeeded is
justified before real data, matched precision and error analysis.

## 6. Layer analysis

Collection now defaults to all32 layers plus final normalization. This adds copies,
CPU statistics and storage, not model forwards. All layers receive the same explicit
grid/protocol. Per-layer results, failures and safe/unsafe lead plots are saved.
Multiple comparisons remain a limitation; independent replication is required for
exploratory layer discoveries. No real best layer is known. The old layer16 default
belongs only to the supplementary descriptive cosine audit.

## 7. Latent-before-lexical examples

Timeline files include both first-trigger steps, proposed tokens, final token, baseline
commit step, lead and safety. Deterministic examples and unsafe first triggers are
saved separately. The supplementary audit supplies step/top1/probability/token-history/
cosine/final-token/SAFE/STABLE tables for changed-top1 cases and coarse confidence
controls. Real recurring examples and conditional rates remain unavailable.

## 8. Failure cases

Tests demonstrate that adjacent cosine1 can hide radial drift. The frozen-anchor
rule rejects a constructed drifting-center case even when each probe is nearby,
and accepts a compact stationary cloud only after sufficient observations. Missing,
zero-scale or nonfinite anchored histories cannot pass. A trajectory may stabilize
around the wrong answer, drift more slowly than the tolerance, or move after the
trigger. An unsafe first trigger cannot be replaced by a later correct one. No
qualifying setting is retained as a negative/insufficient-evidence result.

## 9. Potential leakage/bias audit

Features use observations through the current step only; references exclude their
confirmation points. Future data enter labels, baseline endpoints and retrospective
EOS cohorts only. Prompt groups, not token-step rows, determine partitions. First
proposals count once per position; commit-time positives do not inflate early-event
precision. Credit safety is computed for its own enhanced candidate. Selection and
calibration are separate from heldout measurement; tuning after seeing heldout
requires fresh confirmation data. Near-duplicates, task imbalance, prompt weighting,
mask survival, common residual directions and public-data familiarity remain concerns.

## 10. Limitations

No real checkpoint inference, memory profiling or observer parity yet. One model,
proxy agreement labels, no semantic task scoring or actual early-insertion intervention.
The fixed credit parameters were tuned on Instruct while our latent thresholds are
tuned on Base; this asymmetry needs sensitivity analysis.50 observations across10
groups do not certify99% population precision. Bounded confirmation cannot prove
absence of future drift. The grid and fixed cloud/drift ratios may miss useful rules.
Actual speed/quality tradeoffs require separate policy rollouts after Phase1 evidence.

## Decisions I Made Without Explicit User Instruction

The requested Base/full-sequence/all-layer/training-free scope is user-specified.
The following details are autonomous; DESIGN_DECISIONS.md has the complete record.

| Decision | Choice | Why | Alternatives | Likely impact |
|---|---|---|---|---|
| Exact revisions | Pinned Base and upstream sampler hashes above | Reproducibility | Latest revision | Model/sampler behavior |
| Baseline schedule |256 tokens/256 steps, no early stop | Long unchanged histories |128 steps; longer generation | High |
| Prompt framing |Plain Question/Answer | Base continuation | Raw/few-shot | High |
| Tensor |Post-layer residual and separate final norm | Internal versus logit-adjacent geometry | Attention-only/internal tensor | High |
| Closeness |Frozen mean/RMS norm with raw Euclidean distances | Avoid following center/scale | Unit vectors, EMA, ellipsoid | High |
| Drift gates |Reference r/2; center displacement r/4 | Compactness plus bounded motion | Independent tolerances | High; sensitivity needed |
| Grid |K2/3/4,C2/3,r.01–.4 | Bounded search | Longer windows/denser grid | High |
| Tuning |60/20/20 prompt split; maximize correct early lead | Independent confirmation | Coverage or confidence-bound objective | High |
| Support |50 early events,10 groups per development split | Reject tiny apparent successes | More support or lower-bound constraints | High at99% precision |
| Failure policy |Abstain after failed calibration | Prevent repeated calibration search | Try runner-up | Reduced coverage |
| Credit proxy |First enhanced-max-q crossing, own winner | Correct evidence timing on shared states | Actual rollout with fallback | High; not actual commit time |
| No crossing |Censor at baseline commitment | Avoid invented times | Substitute baseline step | High |
| Primary layer |Discovery objective among calibration survivors | Avoid heldout winner selection | Heldout best layer | Selection uncertainty |
| Storage |Six hidden vectors; float32 scalar Parquet | Bounded history | Raw dumps, sparse sketches | Recollection for new features |
| Uncertainty |200 paired prompt bootstrap draws | Correlated positions | More draws/stratification | Small-group uncertainty remains |

## Deviations from requested design

Classifiers were removed after explicit correction. Current timeline comparison is
observational threshold evidence, not an actual CreditDecoding decoding run; actual
rollouts would violate the chosen unchanged-baseline Phase1 design. No confidence
boosting or Phase2 method is implemented. All-layer collection is now the default.
The older four-layer/cosine-only/no-tuning design is superseded; its descriptive
comparison remains explicitly labeled as supplementary.

## Approximations

Equation-level credit reproduction without verified author code; interpreted Eq6
rather than ambiguous ScatterAdd wording; full-span mask ratio for adaptive credit;
no Eq13 all-vocabulary enhancement. Float32 statistics and BF16 activations, no
whitening/covariance estimation. Coarse conditional bins and finite search grid.
Shadow timelines omit policy feedback and forced-progress fallback. Paper coefficients
transfer to Base without retuning; this is recorded rather than treated as exact reproduction.

## Failed attempts

Initial implementation followed the original classifier request, then was archived
when the user clarified training-free intent. Initial placement inside RollingForcing
was corrected. Quarter-depth sampling and pairwise/rolling-center checks were
insufficient for the later all-layer and drift requirements and were extended.
Random-fixture abstention is an expected validation outcome, not a scientific failure.

## Potential sources of bias

EOS suffixes, surviving masks, baseline-agreement labels, shared-trajectory feedback
omission, unequal parameter tuning between methods, short persistence windows,
residual anisotropy, quantization, public-dataset contamination and layer selection.
Repeatedly inspecting heldout plots would invalidate independent confirmation.

## What I would change if I ran this again

Resolve training-free scope and standalone location first, then validate small Vast
runs before increasing data. Predeclare sensitivity grids for drift ratios and credit
transfer, keep a second untouched prompt cohort for layer discoveries, and eventually
compare actual policy rollouts at matched precision/quality if diagnostics succeed.

## Three design choices most likely to change the conclusion

1. Base versus Instruct, full-sequence schedule and EOS handling.
2. Raw geometry, radius, confirmation duration and the r/2,r/4 drift constraints.
3. Shared-baseline proxy timelines versus actual policies, including how credit
   parameters and operating precision are matched.

## Update: Instruct and block-mode comparison

The user now explicitly allows block decoding and the same checkpoint as the paper
for a closer comparison. Recommended configuration: configs/credit_instruct_block64.json,
GSAI-ML/LLaDA-8B-Instruct at08b83a6feb34df1a6011b80c3c00c7563e963b07,256 output
tokens/steps,64-token blocks, all32 residual layers plus final norm. The32-layer/4096-wide
architecture and chat template were checked from small remote metadata files only.
No weights were downloaded locally. The paper's exact checkpoint commit is not known;
this matches the named checkpoint, not a verified paper revision.

Decision: preserve independent checkpoint/block sensitivities. Choice: add Instruct/full,
Base/block64 and Instruct/block64 configs; retain Base/full as the CLI default for
backward compatibility, but recommend Instruct/block64 in the Vast runbook. Why: avoid
confounding architecture/checkpoint and schedule effects. Alternatives: replace Base
entirely or expose arbitrary unverified model IDs. Likely impact: high; compare the
four configurations on identical prompt text where possible, checking length eligibility.

Decision: Instruct formatting. Choice: pinned tokenizer chat template, one user message,
assistant generation prefix; no added system prompt or exemplars. Why: native Instruct
format. Alternatives: paper benchmark-specific few-shot wrappers. Likely impact: high;
this is not exact OpenCompass evaluation. Dataset preparation and collection call the
same formatter; prompt snapshot names in the full script include the config hash.

Decision: block histories. Choice: reset latent/output/credit history at block entry;
observe only active-block masks, not future masked blocks. Adaptive credit uses the
active block's mask ratio. Why: histories represent eligible commitment opportunities.
Alternatives: track future blocks before eligibility or global credit/mask ratio.
Likely impact: high; full-sequence and block histories are different experiments.
Global step numbers are retained, so first-trigger lead compares the same position
on the same baseline clock. An unsafe first trigger remains final for that rule.

Decision: paper alignment boundary. Choice: keep the pinned unchanged baseline without
EOS early termination and use shadow credit threshold crossings. Why: preserve Phase1
complete trajectories. Alternatives: reproduce full CreditDecoding policy and benchmark
suite now. Likely impact: high; matched checkpoint/block size is a closer comparison,
not reproduction of speedup, benchmark quality, exact prompts or author model revision.
Closeness radii must be tuned independently for each configuration using the existing
prompt partitions, never transplanted silently from Base to Instruct.

The bootstrap accepts a config path as its first argument and the full-experiment
script as its third argument. `collect --block-length 64` and `--block-length full`
are supported. Changed config/code needs a new run directory. CPU checks cover
unmodified-upstream parity at block boundaries, history/credit resets, active-position
eligibility, native chat-template calls and global-step timeline accounting.

## Explicit reset after an excursion (current mechanism)

Decision: episode-based reference/confirmation instead of independently accepting
overlapping windows. Choice made: collect K consecutive vectors; validate cloud
radius, freeze mean and RMS vector-norm scale, then check every confirmation prefix
against that SAME anchor. Any cloud, point-distance or anchor-displacement failure
invalidates the entire candidate episode. The failing point seeds a new reference
window, and all preceding reference/confirmation points are excluded. Why: prevent
old stability evidence surviving an excursion. Alternatives: discard the failing
point too, sliding-window-only tests, or resetting only on top1 identity changes.
Could affect scientific result: yes; fresh episodes are more conservative and can
detect later than overlapping windows. The current point as a seed allows settling
in a new region without discarding its first observation.

State is independent per position/layer/radius/window/confirmation rule. Reference
compactness is tested when K points are available; confirmation distance and drift
are checked on every subsequent point, not just after C points. No token identity
or future label enters these resets. A top1 change alone is allowed: latent-before-
lexical convergence is still the hypothesis. Distance refers to the position's
hidden vector, not word IDs or a separate token-embedding semantic distance. During
reference formation no frozen anchor exists yet; a failed K-point cloud starts over.

Collection keeps causal C0/C1 prefix statistics in addition to C2/C3. A small scalar
state machine reconstructs each episode in timeline analysis, using the exact matching
reference window at each prefix. This avoids storing separate4096-dimensional anchors
for all36 threshold/window settings at every layer. Raw hidden ring storage stays at
six previous vectors; scalar storage increases. Old traces lacking prefixes must be
recollected. The stateless AnchoredRegionRule.evaluate remains a window comparator;
the actual timeline acceptance uses episode_triggers in stability.py.

Resets apply before the first proposal. Once a method first proposed a token, its
proposal remains in the observational evaluation even if later drift reveals an error.
Baseline generation and independent CreditDecoding history are not reset by the latent
rule. No confidence boost, model training, or actual commitment is introduced.

Validation:36 CPU tests passed, including jump/restart, excursion-and-return,
drift-veto reset even inside the distance ball, prefix causality and preserving an
earlier trigger after a later jump. A small engineered episode integration fixture
checks report/selection paths; it is not scientific evidence.
