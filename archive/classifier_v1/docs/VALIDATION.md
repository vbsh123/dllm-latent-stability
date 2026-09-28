# Local validation — no real model inference

Verified on Python3.12, torch2.6.0+cpu, pinned NumPy1.26.4/pandas2.2.3/
scikit-learn1.6.1/Transformers4.46.3. Full environment: validation/cpu_packages.txt.

- `python -m pytest -q`: **13 passed**. Only small synthetic tensors and a fake
  denoiser; no tokenizer/checkpoint download. Tests cover Eq6/7 versus independent
  logit fusion, returned candidates, adaptive mask handling, exact candidate moments,
  zero-vector/window geometry, separate labels, future-prefix invariance,
  feature allowlists, prompt-group isolation, tied/unattainable precision thresholds,
  paired cluster bootstrap, hook cleanup and CPU-inference refusal.
- Sampler parity: **exact output equality** with/without hooks, invoking the actual
  unmodified upstream function on a tiny fake model. Active-mask/commit row counts,
  block resets and committed-row labels verified. This is not actual-checkpoint parity.
- 60 synthetic generations: collection-fixture → Parquet → SAFE analysis → metrics,
  bootstrap, layer/control tables, recurrence/examples and plots completed.
- Same fixtures: separate STABLE analysis and generated FINAL_EXPERIMENT_REPORT.md
  completed. SAFE and STABLE target metrics differ, as expected from their definitions.
- Synthetic numeric regression artifacts are in validation/synthetic_safe_metrics.json
  and validation/synthetic_stable_metrics.json. They are **not scientific evidence**.
  Synthetic D performed worse than B; the reporting path retained that result.
- `bash -n scripts/bootstrap_vast.sh scripts/full_experiment.sh`: passed.
- `python -m compileall -q dllm_latent tests`: passed.
- Vendored sampler matches pinned upstream bytes; SHA256
  d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e.

Matplotlib emits Pyparsing deprecation warnings from a transitive dependency. They
are non-failing. No tests suppressed model errors. An initial dependency attempt
under default Python3.13 was abandoned; the supported environment is Python3.10–3.12.

Not verified locally: actual checkpoint forward/hook compatibility, full dataset
preparation downloads, Vast bootstrap on a fresh instance, GPU memory/time, actual
model observer parity, real statistical power, classifier results or hypothesis.
These are explicitly the first smoke checks on Vast. No Phase2 implementation.
