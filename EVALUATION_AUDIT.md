# Evaluation and decoding audit — 2026-09-30

## Outcome and scope

Confirmed answer-extraction defects. No decoding, closed-form credit, forward-count,
stop-prefix, or ratio-of-totals defect was found in the inspected paths and synthetic
tests. This is not a guarantee of no bugs. Remote generations were unavailable locally;
no checkpoint was downloaded and no real model inference ran locally.

## Confirmed grading defects (original grader retained for reproducibility)

- Lenient grading selects the last numeric substring anywhere. `#### 24 gallons in
  2 weeks.` extracts2: false negative against24, false positive against2.
- Strict grading accepts only an isolated numeric marker line. `#### 64 dollars.`
  and inline `Total = #### 64` fail despite a clear numeric answer.
- Strict grading selects the last VALID marker, not the last marker. `#### 12` followed
  by `Correction: #### 9 dollars.` can incorrectly earn credit against12.
- Lenient numbers do not parse fractions/exponents as numbers (e.g.1/2 becomes2).
- These flaws can affect either direction and need not affect all methods equally.
  The published strict/lenient tables cannot establish semantic accuracy preservation.

## Decoder/metric checks

Each method owns independent mutable inputs and credit state, and every model call
increments forwards. Answers are generated without gold input: prompts contain the
question plus format instruction; gold enters grading only. State resets per block.
Only active masked positions can be selected; previously inserted tokens stay fixed.
Independent scalar math at strengths1/3/6/12/36 matches actual synthetic model-call
counts, completed tokens and actual-insertion count partitions, including fallbacks.
Existing tests cover upstream baseline parity, actual feedback into later model
inputs, history across token changes, block reset, budget failure and trace invariance.

Early stop requires committed EOS/EOT and a filled prefix. Unfilled suffix masks are
allowed then; an unfinished prefix cannot be scored correct. completion_rate means
finished decoding under this rule (or filled length budget), NOT a complete reasoning
solution. Tokens after the first stop are excluded from output TPF/TPS; full_span
metrics intentionally include filled suffix positions. Ratio-of-totals numerators
count no prompt/intermediate tokens; incomplete attempts still consume time/forwards.
GPU timing synchronizes before and after decoding and includes policy overhead.

Method-name caveat: no_geometry_double names the implementation; base_weight plus
bonus_weight determines actual increments. Credit comparator coefficients remain
fixed while our settings have been tuned, so this is not a matched tuning study.

## Offline audit deliverable

`python -m dllm_latent.audit_gsm8k --run RUN... --out NEW_DIRECTORY`

Run on completed, non-writing directories. Reads manifest.json, questions.jsonl,
generations.jsonl and (when present) summary.csv. Checks IDs, snapshot prompt/gold,
legacy-grade reproduction, token-derived lengths, EOS prefix, counts, forward budgets,
timing validity, and summary reproduction. Returns nonzero on integrity issues.
It cannot prove actual GPU call counts from a JSON file, independently authenticate
questions against the dataset, or verify text/token consistency without tokenization.

New conservative extraction chooses the last explicit answer marker/box/answer phrase
without access to gold. It supports numeric fractions, signs, decimals, commas and
scientific notation; units/punctuation after explicit numeric markers are tolerated.
Extra numbers, expressions, missing markers and prose answers require review. Original
scores and generations are never overwritten. Review includes full outputs and score
disagreements; unflagged samples should also be checked. The automatic fraction is a
review diagnostic, not an official benchmark score or a guarantee of correct grading.
No gold-number search or best-of-multiple-candidates scoring is performed.

## Remaining work

Run the audit on the actual Vast files; inspect issues and manually review ambiguous
answers. Freeze a validated grading protocol before heldout evaluation. All existing
200-question runs are reused train development data, not independent generalization
evidence. Repeated parameter selection and unmatched hardware/tuning remain limitations.
