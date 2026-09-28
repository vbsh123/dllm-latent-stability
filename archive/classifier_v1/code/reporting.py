"""Materialize the requested full report alongside each completed analysis."""
import json
from pathlib import Path


def write_final_report(out, run, manifest, results, analysis_text, recurrence):
    source = Path(__file__).resolve().parent.parent
    decisions = source/'DECISIONS_TABLE.md'
    table = decisions.read_text() if decisions.exists() else 'Decision ledger unavailable in this installation; consult the source checkout.'
    synthetic = manifest.get('synthetic',False)
    status = '**SYNTHETIC VALIDATION ONLY: these numbers are not model evidence.**' if synthetic else '**Completed observational Phase 1 analysis. No early-commit intervention was performed.**'
    b,d = results['B_OUTPUT_HISTORY']['test'],results['D_OUTPUT_PLUS_LATENT']['test']
    delta = b['brier']-d['brier']
    ci = results['paired_cluster_bootstrap']['brier_improvement']
    hp_b,hp_d = b['high_precision']['0.97'],d['high_precision']['0.97']
    screen = (delta>=.002 and ci['low']>0 and hp_d['recall']-hp_b['recall']>=.01 and
              hp_b['precision'] is not None and hp_d['precision'] is not None and
              min(hp_b['precision'],hp_d['precision'])>=.97)
    verdict = 'Not applicable to synthetic fixtures.' if synthetic else (
        'The predeclared practical screen passes; replication and sensitivity checks are still required before Phase 2.' if screen else
        'The predeclared practical screen does not pass. This experiment is insufficient support for Phase 2; it does not rule out every possible latent feature or nonlinear predictor.')
    sections = [
        '# Final experiment report',status,
        '## 1. What was implemented\nUnchanged pinned upstream generation with observational hooks, online scalar features, separated SAFE/STABLE labels, prompt-group splits and A/B/C/D logistic models. No new decoder.',
        '## 2. CreditDecoding interpretation/reproduction\nACL final Eq6/7, update before fusion, alpha=.65 beta=.7 gamma=.2; adaptive gamma1 with block-local alpha=beta=1-mask-ratio. Exact full-vocabulary normalization and retained old-candidate credit. Observational equation reproduction, not the paper\'s accelerated rollout. Author implementation was not located. See source CREDITDECODING_NOTES.md.',
        '## 3. Experimental setup\nRun manifest and analysis configuration are the authoritative settings.\n\n```json\n'+json.dumps(manifest.get('config',manifest),indent=2)+'\n```\n\nSource revision, dependency versions and prompt hash are in the collection manifest. Splits/features/sampling/cohorts are separate analysis artifacts.',
        '## 4. Main results\n'+analysis_text,
        f'## 5. OUTPUT_HISTORY versus OUTPUT_HISTORY + LATENT_HISTORY\nBrier improvement B→D: {delta:.6f}; paired cluster interval [{ci["low"]:.6f}, {ci["high"]:.6f}].\n\n{verdict}\n\nA confidence-only gain is insufficient. Operational recall must be read with achieved held-out precision. Descriptive test-envelope recalls in metrics.json are not deployable thresholds.',
        f'## 6. Layer analysis\nValidation-selected layer model: `{results["validation_selected_layer"]}`. See layer_gains.csv/png and per-layer metrics. All individual test comparisons are exploratory and uncorrected for multiplicity. Progress/stability plots condition on positions that remain masked.',
        '## 7. Latent-before-lexical examples\nSee examples.csv, convergence_leads.csv, convergence_lead_summary.csv and full_unstable_recurrence.csv. Training quartiles define high/low. Two-consecutive-high and sustained-to-end onset are both recorded. Full held-out recurrence counts: '+json.dumps(recurrence)+'. These use complete baseline trajectories, independently of classifier row sampling/EOS cohort.',
        '## 8. Failure cases\nSee failure_examples.csv for deterministically selected high-stability yet unsafe trajectories; an empty file means no such examples met this criterion in this run, not universal safety. Read per-task metrics and any failed high-precision operating points in metrics.json. Fixed empirical validation precision does not guarantee test precision.',
        '## 9. Potential leakage/bias audit\nPrompt-hash groups isolate all token steps and duplicate texts. Features use only present/past observations; final labels/EOS/commit distance do not enter X. Train-only preprocessing, validation-selected thresholds. Task/public-data contamination and near duplicates remain possible. No row bootstrap. Retrospective filtering, survivor bias, row caps and repeated layer tests remain limitations.',
        '## 10. Limitations\nFinal-token agreement is neither semantic correctness nor causal safety. One checkpoint, baseline and linear predictor; no quality/speed result. Bootstrap conditions on fitted models and does not include training uncertainty. High precision needs many independent prompts. BF16 and anisotropic residuals may create spurious apparent stability.',
        '## Deviations from requested design\nCreditDecoding is scored on baseline traces, not run as an altered decoder. Default analysis caps each generation at512 eligible rows; complete scalar trajectories remain available. Optional trees and Phase2 are deferred. Real-model status is explicitly distinguished from synthetic fixtures.',
        '## Approximations\nBlock-local adaptive mask ratio, float32 statistics, no optional full-distribution credit-enhancement variant, coarse confidence/history controls and quartile/onset heuristics. Algorithm1 ScatterAdd is interpreted consistently with Eq6 as a single increment.',
        '## Failed attempts\nImplementation preparation encountered unavailable PDF text tooling, no located author-code repository, and an incompatible default Python3.13 environment; resolved with pypdf and Python3.12. This report does not infer unlogged failed GPU attempts; inspect the run logs separately.',
        '## Potential sources of bias\nActive-block and noncommit selection, EOS suffixes, prompt length caps, task imbalance, public benchmark exposure, correlated rows, row sampling, fixed model capacity, many layer comparisons and validation threshold optimism.',
        '## What I would change if I ran this again\nReplicate across prompt groups, schedules and a second checkpoint. Test a nonlinear output-history baseline with matched D capacity, confidence-bound precision thresholds and nested uncertainty. Validate causal early insertion only after robust incremental evidence.',
        '## Three design choices most likely to change the conclusion\n1. Checkpoint and baseline schedule/block/length.\n2. Output-history coverage and predictor capacity.\n3. Commit/EOS/survivor filtering and row weighting.',
        '## Decisions I Made Without Explicit User Instruction\n'+table,
    ]
    (out/'FINAL_EXPERIMENT_REPORT.md').write_text('\n\n'.join(sections)+'\n')
