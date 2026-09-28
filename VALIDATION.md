# No-geometry control validation

88 CPU tests pass. New checks independently verify control credit arithmetic across
changing token identities; exact equality with regional fusion when every vector
matches one anchor; execution on a model exposing no hidden states; per-block credit
reset; boost attribution; trace invariance; region creation/reuse denominators; and
paired reporting against the control. No real-model inference was run locally.

Historical validation follows.

# Boost attribution validation

82 CPU tests pass after adding same-state boost-enabled commitment counters. Tests
separate raw threshold crossings from enhanced crossings, exclude fallback/schedule
and mask-valued proposals, audit changed winners, check missing historical counters,
and verify identical counters/tokens with tracing on and off. All transfers for
instrumentation remain after the decoding loop. No full-model inference was performed
locally. Existing GPU output files cannot supply these newly added counters.

The local decoder contained an appended shell inspection snippet; it was preserved
in ignored .local_backups before removal from the importable module. This repair and
the counters do not change the decoding policy. Historical validation follows.

# Current validation: persistent_regions_v2

75 CPU tests passed (14 matplotlib/Pyparsing deprecation warnings). No full model
weights were downloaded and no real inference was run locally. Current checks cover
A/A/B/A/B retained credit, no eviction during bank growth, independent positions,
frozen-anchor drift, nearest-region assignment, invalid vectors, cross-token credit,
exact paper-style logit fusion, shared confidence acceptance, no count-gate bypass,
boosted fallback ranking, per-block reset, diagnostic/rollout parity, trace invariance,
legacy sampler parity and GSM8K grading/reporting. Old config keys are rejected.

An initial new test expected the wrong nearest normalized anchor; the arithmetic
showed region B was closer, and the test expectation was corrected. All tests passed
following that correction and the remaining reporting changes.

GPU checkpoint parity, memory fit, accuracy and throughput are still unverified.
The Vast commands remain the same; use fresh output directories and the new policy
JSON. Old traces and tuned direct-support gate settings cannot be reused.

The sections below are historical validation for earlier policy versions. In
particular, old observation-7 expectations are no longer the current mechanism.

---

# Current training-free validation

- 57 CPU tests now covered and passing with Python3.12 and torch2.6.0+cpu.
- Coverage includes credit equations and retained candidate histories; causal cosine,
  centroid and anchored-region features; warmup/zero histories; separate SAFE/STABLE
  and credit-candidate labels; first unsafe trigger retention; unsorted-row alignment;
  censoring/lower bounds; rejection of commit-only precision; tuning abstention;
  stationary acceptance and cumulative center-drift rejection; prompt formatting;
  full-sequence config validation; all32-layer unchanged-upstream fake-model parity;
  hook cleanup and CPU inference refusal.
- 60 random synthetic generations completed timeline grid selection, abstention and
  reporting. No setting qualified. This is plumbing validation, not negative research
  evidence about the model.
- 60 deliberately engineered stationary/correct fixtures completed discovery,
  calibration, heldout comparisons, paired intervals, example files and plots. All15
  layer/target settings passed in this constructed five-layer fixture. The fixture
  deliberately makes the proposed labels correct; it is not a positive research result.
- The earlier30-generation descriptive cosine audit completed plots, conditional
  tables and reporting; that audit is supplementary, not the tuned anchored method.
- Python compilation and shell syntax checks passed. Matplotlib's transitive Pyparsing
  deprecation warnings do not affect outcomes.
- No active classifier estimator is fitted. Hyperparameter selection does use
  development labels, as requested; real radii remain untuned until GPU trajectories.
- RollingForcing's pre-existing tokentrim/__init__.py modification was preserved.

CPU fixture artifacts are under /tmp/dllm_research/anchored_random_audit and
/tmp/dllm_research/anchored_stationary_audit in this workspace. Their manifests and
reports explicitly mark synthetic provenance. validation/anchored_smoke.json records
engineering-only outcomes. Random fixture command:

```bash
python -m dllm_latent.synthetic --out runs/synthetic --prompts 60
python -m dllm_latent.timelines --run runs/synthetic --out runs/synthetic_timelines --bootstrap 20
```

No full checkpoint was downloaded or run locally. GPU memory/throughput, real-model
observer parity, actual prepared datasets and all scientific results remain unverified.
The acceptance fixture only tests file/selection/report paths. It cannot justify a
closeness threshold, best layer, early-commit safety, or speedup claim.

Latest Instruct/block-mode update:31 tests pass, including upstream output parity
with two blocks at both one-token and multiple-token-per-step schedules, active-block
mask eligibility, latent/output/credit reset on block entry, native chat-template
invocation and global-step timeline leads. All five configurations pass schedule
validation; both shell scripts pass syntax checks. Only tiny Instruct configuration
and tokenizer-template metadata were fetched locally, not checkpoint weights.

The latest episode-reset change passes36 CPU tests. New tests cover large jumps,
returning to the old region without resuming its confirmations, drift-triggered
resets, causal prefix invariance, and retaining an earlier proposal after later drift.
Older anchored fixture results above validate the prior overlapping-window version,
not the new episode algorithm. New engineered integration uses20 synthetic groups
with support thresholds lowered to1 solely to exercise selection/report paths.
No lowered support setting is recommended for real experiments.

## Actual rollout validation

The last full-suite run passed56 tests; the actual-decoding module then passed all21
of its tests after one additional end-to-end latent-commitment test was added, for57
unique passing CPU tests total. The remaining36 earlier tests were unchanged.

Coverage includes actual policy-dependent next model inputs; confidence acceleration;
GPU-formula CreditDecoding versus independent NumPy equations; credit-triggered
commitments and per-block resets; latent multi-token early commitment; binary support
decay and nonconsecutive reuse; large-excursion/drift/consecutive resets; nonfinite
recovery; budget exhaustion without silent completion; explicit fallback counts;
hook cleanup; original fixed-schedule parity including equal-confidence tie order;
answer-level numeric equivalence despite different wording; malformed/incomplete
answers; paired reports, actual commitment tables and plots; and refusal of real
inference on CPU before downloading model/tokenizer/dataset.

Python compilation, CLI help and shell syntax checks passed. Real GPU parity remains
a required Vast smoke check. No claim of measured GSM8K accuracy, quality improvement,
model forward savings or wall-clock speedup follows from these synthetic tests.

## Shared-rule and measurement review update

70 CPU tests pass in the complete suite. Shared-policy tests compare exact anchor,
scale,reference count,ready/reset state,support and acceptance on identical vector
streams, with manually asserted stationary/moderate-miss/large-reset trigger times.
The default first confirmation beyondr/4 now counts; observation7 is the earliest
uninterrupted default acceptance. All five methods pass trace-on/off token/counter/
forward invariance. Length-stratified parity selection is tested, and a synthetic
collector-to-policy_timelines integration verifies first-trigger step6 (zero-based).
Explicit mean-guard behavior remains tested only with that ablation enabled.

No GPU checkpoint/parity/timing run was executed locally. The expanded --parity-only
preflight is still required on Vast. Shell syntax and CLI-help checks pass.
