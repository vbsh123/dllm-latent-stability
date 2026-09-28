# Design decisions

Every entry records an autonomous choice before real model results. Scientific sensitivities are explicit.

## Project location
- **Decision:** Project location
- **Choice made:** Isolated experiments/dllm_latent package in current repository
- **Why:** Keep this deliverable reviewable without coupling video dependencies
- **Alternatives considered:** Separate repository
- **Could this affect the scientific result?:** Engineering only; no existing user edits touched

## Model/checkpoint
- **Decision:** Model/checkpoint
- **Choice made:** GSAI-ML/LLaDA-8B-Instruct; model and remote code revision 08b83a6feb34df1a6011b80c3c00c7563e963b07
- **Why:** Open weights, same family as CreditDecoding, plausible 40 GB BF16 budget
- **Alternatives considered:** Dream, LLaDA Base, newer LLaDA variants
- **Could this affect the scientific result?:** High: conclusions initially apply only to this model

## Sampler source
- **Decision:** Sampler source
- **Choice made:** Unmodified ML-GSAI/LLaDA generate.py at 9182493720ed723ef8031210d85959364e51cbe0; byte hash d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e
- **Why:** Direct call plus read-only hooks avoids reimplementing baseline decisions
- **Alternatives considered:** Rewritten sampler, older source revision
- **Could this affect the scientific result?:** High: this revision uses native-logit-dtype softmax; older versions use float64; source has no root LICENSE

## Decoding configuration
- **Decision:** Decoding configuration
- **Choice made:** 256 generated tokens, 256 total steps, block length 64; temperature=0, CFG=0, low_confidence, no suppression, no early stop or KV cache
- **Why:** Conservative normal upstream baseline with long within-block trajectories
- **Alternatives considered:** 128 steps; block32; length512; threshold sampler
- **Could this affect the scientific result?:** High: commands provided for steps128/block32/length512 sensitivity; one token per step at primary setting

## Smoke size
- **Decision:** Smoke size
- **Choice made:** 3 handcrafted prompts, length/steps32, block16; separate synthetic fixture of60 groups
- **Why:** Cheap inference/hook check before collecting100 prompts
- **Alternatives considered:** Classifier analysis on3 prompts
- **Could this affect the scientific result?:** Smoke cannot support statistical conclusions

## Relevant observations
- **Decision:** Relevant observations
- **Choice made:** Still-masked positions in active block, before every forward commit, ending at commit
- **Why:** Comparable block-local online history and avoid trivial post-commit states
- **Alternatives considered:** Track future blocks; raw post-commit predictions
- **Could this affect the scientific result?:** High: no claims about pre-activation convergence or post-commit raw logits

## Target semantics
- **Decision:** Target semantics
- **Choice made:** SAFE matches final baseline token; STABLE requires all remaining relevant predictions equal that token, then absorbing decoder state
- **Why:** Separate current correctness from permanent lexical stabilization
- **Alternatives considered:** Next-step stability; semantic correctness; raw logits after commitment
- **Could this affect the scientific result?:** High: neither label establishes counterfactual causal safety

## Commit-row filtering
- **Decision:** Commit-row filtering
- **Choice made:** Store all relevant rows; primary predictors exclude committed_this_step; run inclusion sensitivity
- **Why:** Avoid trivially correct observations offering no earlier commitment opportunity
- **Alternatives considered:** Include commit steps in primary
- **Could this affect the scientific result?:** High: conditions on baseline decision; changes target population, recorded as metadata, never predictor

## EOS handling
- **Decision:** EOS handling
- **Choice made:** No early stop in baseline; retain all positions in primary; pre-final-EOS-only sensitivity excludes final EOS itself and suffix
- **Why:** Preserve upstream behavior while auditing easy repeated EOS/padding labels
- **Alternatives considered:** Early stopping; online EOS-based cohort
- **Could this affect the scientific result?:** High: final-EOS filter is retrospective outcome-conditioned analysis, not deployable feature

## Layers
- **Decision:** Layers
- **Choice made:** Post-block outputs8,16,24,32 plus post-final-RMSNorm
- **Why:** Early-middle/middle/late/final plus directly logit-adjacent representation
- **Alternatives considered:** All layers; attention-only; pre-normalization values
- **Could this affect the scientific result?:** High: report each layer and B+each layer, validation-selected layer; multiple tests exploratory

## Exact hidden tensor
- **Decision:** Exact hidden tensor
- **Choice made:** Residual stream after attention and MLP residual additions from transformer.blocks[layer-1] output tuple[0]; before next block norm. final_norm is transformer.ln_f output
- **Why:** Pinned modeling code explicitly inspected; avoids hidden_states off-by-one ambiguity
- **Alternatives considered:** HF output_hidden_states tuple; attention output
- **Could this affect the scientific result?:** High: pre/post RMSNorm sensitivity collected directly; selected blocks use RMSNorm internally

## Latent geometry
- **Decision:** Latent geometry
- **Choice made:** Float32 raw-vector cosine with L2 normalization, denominator<=1e-12 gives missing; clip[-1,1]; raw and relative L2 plus vector norm
- **Why:** Scale-insensitive cosine with explicit norm controls and movement
- **Alternatives considered:** Centered/whitened cosine; angular distance; float64
- **Could this affect the scientific result?:** Moderate: anisotropy and BF16 quantization can mask movement; no whitening fit

## Window definition
- **Decision:** Window definition
- **Choice made:** Lags1,2; windows K=2,3,4 consecutive transitions (K+1 hidden states); means/minima/population variance, summed L2 and cumulative L2
- **Why:** All requested windows cheaply computed online; no future padding
- **Alternatives considered:** K hidden states instead of transitions; partial windows
- **Could this affect the scientific result?:** Moderate: complete windows only; early steps missing, imputed using training rows

## Memory/storage
- **Decision:** Memory/storage
- **Choice made:** Two hidden states and four scalar transitions per layer; current block full-vocabulary moments/EMA/credit/four probability frames on CPU; scalar Parquet per generation
- **Why:** Exact histories for newly returning candidates without hidden dumps
- **Alternatives considered:** Top-k history approximations; raw hidden/distribution dumps
- **Could this affect the scientific result?:** Engineering tradeoff: about10 MiB hidden ring, several hundred MiB CPU vocabulary state at block64, lower GPU memory but slower collection

## Output-history features
- **Decision:** Output-history features
- **Choice made:** Current p1/p2/margin/entropy; prior top1 fraction, streak/change count; exact current-candidate lag1..4 probabilities, delta, all-prefix moments/EMA; consecutive full-distribution JS
- **Why:** Strong baseline must retain support for candidate even when it was not top1
- **Alternatives considered:** Winning-confidence-only EMA; sparse top-k approximations
- **Could this affect the scientific result?:** High: no raw token identity or semantic embedding as classifier input; prefix moments include current step

## History constants
- **Decision:** History constants
- **Choice made:** Candidate EMA decay .7 with first probability initialization; population variance over current/past steps; history resets per block
- **Why:** Simple causal summary, tied to paper decay scale
- **Alternatives considered:** EMA .5/.9; sample variance; cross-block memory
- **Could this affect the scientific result?:** Moderate: requires sensitivity if conclusions marginal; all-vocab running moments are float32

## Credit reproduction
- **Decision:** Credit reproduction
- **Choice made:** ACL final Eq6/7, alpha=.65 beta=.7 gamma=.2; update then fuse; fixed and adaptive variants both enter B/D
- **Why:** Final paper gives parameters; serious history comparison
- **Alternatives considered:** Older arXiv formulation; published source if located; full-distribution Eq13
- **Could this affect the scientific result?:** High: equation-level observational reproduction, no accelerated CreditDecoding rollout or benchmark claim

## Credit ambiguities
- **Decision:** Credit ambiguities
- **Choice made:** Single increment in Algorithm1 ScatterAdd per Eq6; adaptive gamma1 and alpha=beta=1-current-block-mask-ratio
- **Why:** Consistent equation/prose interpretation; explicit block scope
- **Alternatives considered:** Literal double-C notation; global sequence mask ratio
- **Could this affect the scientific result?:** Moderate: exact matching to unavailable released implementation unverified; full-distribution enhancement omitted

## Probability precision
- **Decision:** Probability precision
- **Choice made:** Observational probabilities/statistics float32 on CPU; top1 from raw logits argmax; decoder BF16 softmax unchanged
- **Why:** Avoid large float64 probability tensors without touching sampling
- **Alternatives considered:** Float64 statistics; use native BF16 probabilities
- **Could this affect the scientific result?:** Small but possible near-boundary impact; geometry starts from BF16 model activations

## Datasets
- **Decision:** Datasets
- **Choice made:** Balanced Dolly open_qa, GSM8K test, Dolly creative_writing; original prompts with optional context; no answers; no appended task directives
- **Why:** Instruction, math, longer-form task intent without templated synthetic corpus
- **Alternatives considered:** MMLU, HumanEval, broader held-out instruction suites
- **Could this affect the scientific result?:** High: fixed256-token output limits long form; public data contamination and task confounds remain

## Dataset pinning
- **Decision:** Dataset pinning
- **Choice made:** Dolly bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a; GSM8K740312add88f781978c0658806c59bc2815b9866; seeded shuffle/round-robin and normalized-text dedup
- **Why:** Replayable100/500/1000 snapshots; source IDs/hash saved
- **Alternatives considered:** Floating dataset main; handselected examples
- **Could this affect the scientific result?:** Moderate: same first100 recur in larger snapshots; treat initial results as exploratory and freeze protocol before final test

## Prompt limits
- **Decision:** Prompt limits
- **Choice made:** Chat template from pinned tokenizer, no extra special tokens; reject >512 tokens or mask tokens; prepare filters before sampling
- **Why:** Bound full-sequence logits memory without truncating meaning
- **Alternatives considered:** Truncate inputs; larger prompt budget
- **Could this affect the scientific result?:** Moderate: excludes long/harder prompts; rejection counts recorded; near duplicates not automatically deduplicated

## Splits
- **Decision:** Splits
- **Choice made:** Hash(seed1729, normalized prompt text): train<.6, validation<.8, test otherwise; same group across repeats and cohort filters
- **Why:** No same-generation leakage; stable expanding datasets and filters
- **Alternatives considered:** Random row split prohibited; stratified group split
- **Could this affect the scientific result?:** High: task proportions can differ in small sets; report cohorts/task metrics; refuse <10 groups or one-class partitions

## Analysis row budget
- **Decision:** Analysis row budget
- **Choice made:** Uniform deterministic cap512 eligible rows per generation; same rows for every model; full trajectories retained for recurrence/examples; --max-rows-per-generation0 disables
- **Why:** Bound host RAM and logistic fit cost for1000 prompts
- **Alternatives considered:** Every row; per-position weighting
- **Could this affect the scientific result?:** Moderate: estimand is sampled token-step population, long histories oversampled within generation; macro Brier and full-row sensitivity supplied

## Classifier
- **Decision:** Classifier
- **Choice made:** Logistic regression fixed C1, lbfgs, max_iter2000, no class weights; train-only median imputation with missing indicators and standardization
- **Why:** Simple falsification model with calibrated-probability objective; no hyperparameter search on test
- **Alternatives considered:** Gradient boosting, regularization tuning, splines/interactions
- **Could this affect the scientific result?:** High: negative result only for these features/model capacity; optional trees deferred

## Common controls
- **Decision:** Common controls
- **Choice made:** Progress, block progress, mask ratio and position in all A/B/C/D groups
- **Why:** Avoid credit/latent history merely recovering denoising stage
- **Alternatives considered:** Strict features-only groups; task/token-ID covariates
- **Could this affect the scientific result?:** Moderate: A/C names include common controls; task identity excluded, task-stratified reporting provided

## Thresholds/calibration
- **Decision:** Thresholds/calibration
- **Choice made:** Validation maximum recall at empirical precision .95/.97/.99 with >=50 selected rows; freeze threshold; ten equal-width bins/ECE; test PR envelope separately descriptive
- **Why:** Operationally meaningful high-precision evaluation without optimizing threshold on test
- **Alternatives considered:** Confidence-bound thresholding; Platt/isotonic calibration
- **Could this affect the scientific result?:** High: empirical precision is not a guarantee; repeated rows dependent; small support and empty selections explicit

## Uncertainty
- **Decision:** Uncertainty
- **Choice made:** 200 paired prompt-cluster bootstrap draws, seed1729; fixed fitted predictors/thresholds; percentile95% intervals
- **Why:** Account for within-generation row dependence in B-vs-D gains
- **Alternatives considered:** Row bootstrap prohibited; refit/nested bootstrap; more draws
- **Could this affect the scientific result?:** Moderate: intervals omit training uncertainty and layer-selection multiplicity; increase repeats for final reporting

## Convergence analyses
- **Decision:** Convergence analyses
- **Choice made:** High/low = training layer-cosine quartiles; joint confidence0.1 bins and lexical streak1/2-3/4+; B-risk0.1 bins; weak evidence B<.5; primary examples layer16
- **Why:** Train-derived thresholds and output controls, no outcome-picked anecdotes
- **Alternatives considered:** Absolute .99 cosine cutoff; matching/conditional flexible models
- **Could this affect the scientific result?:** High: bins are descriptive and imperfect controls; first step shares streak1 bin but unstable-specific subsets require changed=1

## Onset/examples
- **Decision:** Onset/examples
- **Choice made:** First two consecutive high-cosine transitions; also sustained-to-commit onset; compare retrospectively with first STABLE row; first12 qualifying positions by deterministic shard/position order; separate failures
- **Why:** Measure recurrence and false apparent convergence; avoid favorable-outcome selection
- **Alternatives considered:** Earliest single high cosine; rank examples by correctness
- **Could this affect the scientific result?:** Moderate: thresholds exploratory; recurrence uses full held-out trajectories, including EOS suffix, irrespective of classifier cohort; no semantic-equivalence inference

## Practical success screen
- **Decision:** Practical success screen
- **Choice made:** Brier gain>=.002 with cluster CI above0, recall gain>=.01 at97% target with both test precisions>=.97; require robustness before Phase2
- **Why:** Explicit falsification-oriented gate beyond generic accuracy
- **Alternatives considered:** Different utility costs; higher precision; effect-size thresholds
- **Could this affect the scientific result?:** High: judgment call, sensitivity required; never automatically authorizes decoder work

## Runtime/reproducibility
- **Decision:** Runtime/reproducibility
- **Choice made:** Batch1 BF16, inference mode/eval, seed1729, no quantization; Python3.10-3.12, pinned core dependencies; full freeze/model/source hashes
- **Why:** Plausible40 GB budget with replayable code and bounded collection
- **Alternatives considered:** FP16/FP32; quantized models; batched/cached inference
- **Could this affect the scientific result?:** Moderate: GPU numerical nondeterminism possible; real fit/parity pending; no local weights or inference

## Phase scope
- **Decision:** Phase scope
- **Choice made:** Phase1 only; actual inference on Vast operated separately; final report distinguishes implementation from empirical result
- **Why:** User prohibits local full-model run and requires evidence before decoder design
- **Alternatives considered:** Immediate learned commitment decoder
- **Could this affect the scientific result?:** Scientific conclusion remains unknown until real collection; no cloud account or spending initiated
