# Current latent comparison update (v2)

CreditDecoding's existing token-credit Eq6/7 implementation is unchanged. The latent
comparison now retains a bank of frozen regions, uses beta=.7 and p(top1)^.2 credit
increments, maps the current region's balance to the current top1, and calls the
same alpha=.65 logit fusion and confidence threshold. The bank-to-token mapping is
our proposal, not part of CreditDecoding. Regions are not discarded on excursions.
Combined sums token and mapped regional credits before fusion. See
ROLLOUT_EXPERIMENT.md for the authoritative current method. Earlier diagnostic-only
statements below describe prior phases, not current actual-rollout behavior.

# Actual CreditDecoding rollout update

The user now requests actual independently generated GSM8K answers. decoding.py
implements Eq6/7 on GPU and commits enhanced winners when max-q>=tau; subsequent
forwards use those changed inputs. Older statements below saying credit never feeds
back apply ONLY to the retained diagnostic collector, not the new rollout module.

The default shared top1 forced-progress fallback is an explicit addition to Algorithm1
when no threshold passes, logged as fallback. --fallback none provides a strict
threshold sensitivity with incomplete outputs counted as failures at the cap. No EOS
early termination,exact benchmark prompting,or verified paper commit revision is
claimed. Actual accuracy is graded from final numeric answers,not baseline wording.

# CreditDecoding reading and reproduction notes

**Current update:** Instruct/block64 is now supported and recommended for the closer
CreditDecoding comparison; Base/full remains available. See the update section below,
which supersedes earlier Base-only/full-sequence-only scope statements.

Primary source: Wang et al., ACL 2026, pp. 11105–11123,
[final paper](https://aclanthology.org/2026.acl-long.509.pdf),
[publication record](https://aclanthology.org/2026.acl-long.509/).
Read the full final paper, including Algorithm 1, tuning-free schedule,
hyperparameter studies, full-distribution variant, and failure analysis.
The older [arXiv v1](https://arxiv.org/html/2510.06133v1) is not the controlling version.

## Implemented mechanism

With chronological index t increasing (opposite the paper's reverse-diffusion index),
for a masked position i and raw top-1 candidate v*:

```
C[t,i,v] = beta * C[t-1,i,v] + 1[v=v*] * p[t,i,v]^gamma
q[t,i,v] = p[t,i,v] * (1+C[t,i,v])^alpha / sum_u p[t,i,u]*(1+C[t,i,u])^alpha
```

`TraceCredit` implements these equations over the full vocabulary, including the
normalization denominator and credit retained for candidates that leave and later
return to top-1. State resets at the start of each full-sequence generation. Current observations
update credit before fusion. This is not merely an EMA of winning confidence or a
cosine substitute. Raw candidate credit, its enhanced probability, maximum enhanced
probability, enhanced winner ID, and whether the enhanced winner agrees with raw top-1
are observational output-history scores. Timeline safety uses the enhanced winner's
own agreement with the final baseline token, not the raw candidate's label.
The actual baseline commits raw candidates; q never feeds back into generation.

Fixed settings are alpha=.65, beta=.7, gamma=.2 (final paper §5.1, p.11110).
The adaptive variant (§4.3) uses gamma=1 and alpha=beta=1-mask_ratio, computed from
the full output span (the single upstream API block) before this step's baseline commits. Both variants are collected
at negligible extra inference cost and audited without fitting a predictor. Raw confidence and
raw credit-enhanced candidate probability also have independent score evaluations.

Algorithm 1 line 13's ScatterAdd notation is ambiguous if read as adding the old
C twice. We follow Eq.6 and the accompanying global-decay/focused-enhancement prose:
one decayed C plus one increment. This interpretation is explicit and unit tested.

## Relationship to the hypothesis

Credit uses token identities across steps; the fixed latent check uses a position's
internal trajectory. Output-history summaries retain candidate probability moments,
EMA and lagged probabilities even when the candidate was not top-1. The active
analysis compares fixed rules and conditional outcome rates at matched coarse
confidence, lexical-streak and credit-evidence levels. There is no classifier or
learned combination. This cannot establish independence from every conceivable
output-history rule; it audits whether this fixed stability check adds useful evidence.

## Limits of reproduction

No public CreditDecoding repository was found through GitHub repository search,
the final ACL paper, or the arXiv article. This does not prove none exists.
The implementation reproduces the equations as observational scoring features,
not CreditDecoding's accelerated rollout, benchmark scores, speedup, OpenCompass
few-shot settings, threshold decoder, or EOS early stopping. Those would change
the Phase 1 baseline trajectory and belong to a later decoder experiment.

The timeline module compares the first max-q threshold crossing at .95/.97/.99
with the first anchored-latent trigger and the actual baseline commit step. It uses
the enhanced argmax candidate, preserves an unsafe first proposal, and censors
positions with no crossing by baseline commitment. It does not reproduce the
decoder's forced-progress/top-k fallback or claim an actual CreditDecoding commit
time. A true rollout changes the context after its first differing commitment.
Thus these are shadow evidence timelines, not measured speedups. The adaptive
variant and raw confidence also have separate first-trigger summaries.

The paper's optional all-vocabulary enhancement (Eq.13) is not implemented; the
main method updates only the top-1 candidate, but remembers credits over all tokens.
Adaptive mask ratio is interpreted as block-local; sequence-global ratio is a
possible sensitivity experiment. Scoring runs in NumPy float32; small numerical
variation from a GPU implementation is possible. Unit tests compare against an
independent logit-domain formula.

The paper's reasoning-task failures and block-size effects motivate task-specific
results and block/step sensitivity runs. Agreement with a baseline's final token
is a proxy label, not ground-truth correctness, and does not estimate the downstream
effect of actually inserting a token early.

## What the paper evaluated

Main Table1 compares standard decoding, Fast-dLLM without KV cache, and
CreditDecoding on LLaDA-8B-Instruct and LLaDA-MoE-Instruct (7B total/1B active).
Main tasks are MMLU, SQuAD2, DROP, KorBench, HumanEval, LiveCodeBench, GSM8K and
MATH. Main generation length/steps are256, block size64, with early stopping;
the fixed credit coefficients were tuned on LLaDA-8B-Instruct.

Table2 extends evaluation to SDAR-8B-Chat-b32, LLaDA2-Mini (16B/1B active) and
LLaDA2-Flash (100B/6B active), comparing Fast-dLLM, fixed CreditDecoding and its
adaptive variant on the five-task analysis subset. Table3 adds credit to fixed
top2/top4/top8, top-k-margin and threshold(.95) strategies. Orthogonality studies
also examine KV caching, early stopping, compilation and MoE FP8 quantization.

Our Base checkpoint/full-sequence/no-early-stop setting differs materially from
these experiments. It tests transfer of the evidence mechanism; it is not an exact
reproduction of their reported benchmark results. No immutable checkpoint revision
from the paper is claimed here; our own chosen Base revision is explicitly pinned.

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
