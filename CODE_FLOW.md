# Reading flow for the current experiment

1. **configs/credit_instruct_block64.json**: pinned model, blocks, lengths and prompt
   format. **configs/rollout_policy.json**: common confidence threshold, fallback,
   latent layer/radius and paper-style coefficients; mechanism=persistent_regions_v2.
2. **dllm_latent/gsm8k_rollout.py, main()**: GPU guard, GSM8K prompt selection,
   provenance, parity, warmups, independently timed generations and grading.
3. **dllm_latent/decoding.py, RegionSupport.update()**: decay saved region credits,
   find nearest frozen anchor, create a region if needed, add p(top1)^gamma. Previous
   regions survive excursions. This returns evidence, not a commitment decision.
4. **region_distribution()**: map current regional evidence to the current raw
   top1 token. Optional combined sums that evidence with token-history credit.
5. **update_token_credit(), credit_distribution(), fuse_credit()**: paper Eq6/7.
   Both latent and Credit call the same fusion operation.
6. **decode()**: model forward, policy distribution, shared confidence acceptance,
   optional best-confidence fallback, actual token insertion, repeat. Banks are
   per position and per active block. The baseline retains its original schedule.
7. **grade()/write_report()**: final numeric answer quality, TPS/time/forwards,
   fallback counts, separate combined counts and paired uncertainty.
8. **tests/test_decoding.py, tests/test_shared_policy.py**: A/A/B/A/B arithmetic,
   frozen-anchor drift, multiple regions/positions, candidate switching, exact
   logit fusion, no count-based commitment, and identical diagnostic/rollout state.

```text
question → prompt → decode
  → model logits + selected hidden state
  → region bank → current-region credit → current-token logit bonus
  → softmax → common confidence threshold (+ optional fallback)
  → insert tokens → next model forward
  → grade final answer and report time/accuracy
```

For optional baseline trajectory collection, read **collect.Observer** and
**policy_observer.PolicyHistory**. Layer hooks retain selected hidden states until
the model returns logits. PolicyHistory calls the same region_distribution() on the
model device, then copies scalar evidence, boosted confidence and region IDs to disk.
**policy_timelines** compares first boosted-confidence proposals with Credit on those
shared baseline states. This does not establish counterfactual answer correctness.

The old classifier is archived. features.py sliding-window descriptors, stability.py
and timelines.py are legacy alternatives. Old gate configs and scalar traces cannot
be reused to calibrate the new bank-and-boost method. Full details and run commands:
[ROLLOUT_EXPERIMENT.md](ROLLOUT_EXPERIMENT.md).
