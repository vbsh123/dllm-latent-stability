# Actual-generation experiment implementation report

**Implementation and CPU validation only. No real-model quality or speed results yet.**
The current method is persistent_regions_v2. Earlier single-region resets and direct
support-count commitment are superseded. No full model was downloaded or run locally.

## 1. What was implemented

Independent baseline, confidence, CreditDecoding, latent-region and combined GSM8K
rollouts; numeric answer grading; synchronized timing, output TPS, forward counts,
fallback counts and paired-question uncertainty. Collection and rollout share the
same region bank and logit fusion. No classifier is trained.

Every masked position retains separate frozen anchors and discounted regional credit.
A current state matches the nearest anchor within normalized L2 radius or creates a
new region. Previous regions survive excursions. Each visit, including the first,
earns p(raw_top1)^gamma. Current regional credit boosts the current raw top1 through
softmax(logits + alpha*log(1+credit)); boosted confidence determines commitment.

## 2. CreditDecoding interpretation/reproduction

The token-history comparator implements final paper Eq6/7: beta=.7 global token-credit
decay, p(top1)^.2 increment to raw top1, alpha=.65 logit fusion, then enhanced-winner
confidence >= tau. Latent uses the same defaults and acceptance rule but maps region
history to the current candidate. That mapping is our proposal, not a paper formula.

This is not 100% paper reproduction: author checkpoint revision is unknown; zero-shot
GSM8K prompt/format, finite block cap, no EOS early termination and explicit optional
forced-progress fallback differ. The optional all-vocabulary enhancement is omitted.
Credit's token-history equations were not changed by this update.

## 3. Experimental setup

Recommended: pinned GSAI-ML/LLaDA-8B-Instruct revision
08b83a6feb34df1a6011b80c3c00c7563e963b07; block64, length256/steps256, BF16, batch1,
greedy, no CFG/KV cache. Base and full-sequence configs remain available. Pinned
openai/gsm8k train is development; test is held out until settings are frozen.

Latent defaults: layer16 of32, normalized radius .05, beta .7, alpha .65, gamma .2,
common confidence tau .95. Capture post-complete-layer residual output, before final
RMSNorm for numbered layers. All32 layers plus optional final normalization are
available in observational diagnostics; actual timing captures one selected layer.
The anchor is the first observation in a region and scale is its frozen L2 norm.
No K-reference warm-up, consecutive requirement, count gate or large-excursion reset.

Fallback top1 remains the default: when nothing crosses tau, insert the candidate
with largest method-specific confidence (boosted for latent/Credit/combined). Count
these separately. Fallback none leaves unresolved masks at the shared cap and counts
incomplete answers as incorrect. Tracing is off by default for timing.

## 4. Main results

**82 CPU tests passed** after this update. Tests cover manual A/A/B/A/B credit
arithmetic, retaining multiple regions across bank growth, position independence,
frozen-anchor drift, candidate changes, exact boost equations, no count-based bypass,
boosted fallback ranking, diagnostic/rollout state parity, and existing sampler,
trace-invariance, grading and reporting checks. Fourteen dependency deprecation
warnings remain. These tests provide engineering evidence, not research results.

Real checkpoint parity, GPU memory fit, GSM8K accuracy, throughput and speedup remain
unmeasured. Run the 20-prompt Vast parity preflight before the 200-question train pilot.

## 5. OUTPUT_HISTORY vs OUTPUT_HISTORY + LATENT_HISTORY

No empirical result yet. Primary pilot: baseline, CreditDecoding and latent-region
independent generations on the same questions. Optional combined policy sums token
and mapped region credits before the common fusion/acceptance rule. It no longer
ORs two gates. Greater combined evidence strength is a confound; superiority needs
matched quality/compute comparisons and development-only parameter sensitivity.

## 6. Layer analysis

Layer16 is an untuned starting point, not a finding. All-layer collection uses the
same online regional rule after logits are available. Diagnostics save boosted
probabilities and region IDs/counts, then compare first proposal times with Credit.
No best layer or middle-before-final-layer conclusion has been established.

## 7. Latent-before-lexical examples

Synthetic tests show that a fixed hidden state can accumulate region credit while
raw top1 alternates between two vocabulary tokens. The current candidate gets the
shared regional evidence. An A/A/B/A sequence retains A and B separately; credit
balances match the discounted recurrence exactly. These are deliberately constructed
mechanism tests, not examples of a recurring phenomenon in an actual dLLM.

## 8. Failure cases

- Frequent new regions may produce little reusable evidence and consume memory.
- Nearby hidden states may correspond to incompatible words; credit sharing can
  amplify a wrong current candidate.
- Discounted evidence saturates. A constant raw confidence of .6 does not reach the
  default .95 threshold under these coefficients, even after many observations.
- Without fallback, unchanged deterministic inputs can manufacture repeated states.
- With fallback, apparent speed/quality may largely reflect forced ordinary picks.
- Fixed-length output can truncate reasoning and includes post-EOS compute.

## 9. Potential leakage/bias audit

Features, anchors and credits use only current/past active-block observations. Gold
answers enter grading only. Diagnostic baseline agreement is not semantic correctness
or a counterfactual guarantee. First proposals are retained even if wrong. Prompt-
group partitions separate observational discovery/calibration/heldout data. GSM8K
settings must be selected on train, frozen, then evaluated on untouched test.

Independent policy trajectories have different contexts; matching token positions
across them is descriptive, not a causal same-state comparison. Timings include real
policy/hook/region-search/allocation overhead. Use no-trace runs and compare accuracy,
seconds per answer, output length, TPS, forwards and fallback together.

## 10. Limitations

No real inference result; arbitrary latent layer/radius; greedy order-dependent region
creation; isotropic normalized L2 rather than validated semantic geometry; no bank
eviction. Memory scales with positions x regions x hidden width x captured layers.
All-layer full-sequence runs with many regions can exceed practical memory, even
though no raw trajectory dump is made. Profile a small GPU pilot first.

## Decisions I Made Without Explicit User Instruction

Full evolving log: DESIGN_DECISIONS.md. Explicitly requested changes were retaining
regions and using Credit-style logit boosting instead of a latent acceptance gate.

| Decision | Choice | Why | Alternatives | Likely impact |
|---|---|---|---|---|
| New region creation | First observation is frozen anchor and receives credit | Honor singleton B evidence in user's example | K-vector compact reference | High: removes warm-up |
| Matching | Nearest normalized L2 anchor within radius; oldest tie | Exactly one region receives each hit | All matches; oldest match | High: order and geometry matter |
| Vocabulary mapping | Current raw top1 gets full current-region credit | Transfer latent evidence across changing words | Per-region token histogram; distribution weighting | High |
| Coefficients | Paper alpha/beta/gamma and same tau | Controlled initial comparison | Unit-hit/.9 decay; tuned settings | High |
| Combined | Sum token and mapped region credit before fusion | Remove old OR gate consistently | Separate weights; omit combined | High: evidence-strength confound |
| Memory | Dynamic GPU bank; no cap or eviction | Preserve all requested region history | Fixed-cap bank; CPU offload | Memory/runtime cost |
| Invalid states | No new evidence, decay retained history | Avoid poisoning without erasure | Reset bank | Usually small |
| Compatibility | Reject old gates/configs/traces | Avoid silently mixing algorithms | Automatic conversion | Requires recollection |

## Deviations from requested design

The original classifier-based Phase1 and diagnostics-only restriction were superseded
by explicit user requests for a training-free method and real answer generation.
Binary region membership remains, but credit increments are p^gamma rather than
unit hits to match the latest request for the paper's formula. First-visit anchors
replace the older three-vector reference. Both changes are documented, not tuned.

## Approximations

CreditDecoding reproduction caveats are above. The regional mapping and combined sum
are new hypotheses, not claimed reproductions. Old sliding-window diagnostic rules
remain explicitly legacy and cannot calibrate persistent_regions_v2.

## Failed attempts

Earlier direct-support commitment and destructive region reset did not match the
latest requested method and were replaced. One new overlap test initially asserted
the wrong nearest region: .7/11.5 is smaller than .8/10. Its expected value was corrected;
no production algorithm was adjusted to hide that error. No GPU experiment failed
or succeeded because none has been run.

## Potential sources of bias

Train-set parameter selection, output formatting, truncation, fallback fraction,
unchanged contexts, region order, stronger combined evidence, post-EOS work, GPU
warmup/allocator behavior and wording-dependent output lengths can affect conclusions.

## What I would change if I ran this again

Run parity and memory preflight, inspect a small separate traced run, then choose a
small predefined layer/radius sensitivity on development data. Freeze settings before
heldout evaluation. Report failures and no improvement as valid outcomes.

## Three design choices most likely to change the conclusion

1. Layer and normalized region geometry/radius.
2. Assigning a region's full credit to the current candidate across token changes.
3. Credit strength and confidence threshold, together with forced-progress fallback.

See ROLLOUT_EXPERIMENT.md and VAST_RUNBOOK.md for executable commands.

## Additional boost-attribution instrumentation

The runner now separates actual boost-enabled threshold insertions (raw max < tau,
enhanced max >= tau), already-confident insertions, fallback and scheduled insertions.
Changed enhanced winners are counted separately. Traces save both raw and enhanced
confidence, and aggregate counters work without tracing. Older runs cannot recover
these values; missing fields are reported unavailable. No causal speed contribution
or geometric stability conclusion follows from the count alone. See the current
ROLLOUT_EXPERIMENT.md attribution procedure for a confidence/Credit/latent comparison.
