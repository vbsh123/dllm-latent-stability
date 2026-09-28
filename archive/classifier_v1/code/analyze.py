"""Grouped held-out falsification analysis. No per-row random train/test splits."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import feature_groups


def split_groups(frame, seed):
    groups = np.array(sorted(frame.prompt_group.unique()))
    if len(groups) < 10:
        raise ValueError('Need >=10 distinct prompt groups for analysis; collect smoke trajectories or use synthetic with 60 groups')
    # Stable under adding prompts and cohort filters, independent of labels.
    mapping = {}
    for g in groups:
        u = int(hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest()[:16],16)/2**64
        mapping[str(g)] = 'train' if u < .6 else ('validation' if u < .8 else 'test')
    return mapping


def operating_point(y, score, target, min_selected=50):
    """Max validation recall subject to empirical precision + minimum support.

    Ties use >= semantics. No claim of guaranteed precision on new prompts.
    """
    p, r, thresholds = precision_recall_curve(y, score)
    counts = len(score)-np.searchsorted(np.sort(score), thresholds, side='left')
    valid = np.flatnonzero((p[:-1] >= target) & (counts >= min_selected))
    return float(thresholds[valid[np.argmax(r[valid])]]) if len(valid) else None


def decision_metrics(y, score, threshold):
    take = np.zeros(len(y), bool) if threshold is None else score >= threshold
    tp = int(np.sum(y[take])); n = int(take.sum())
    return dict(threshold=threshold, selected=n, precision=float(tp/n) if n else None,
                recall=float(tp/max(1,np.sum(y))))


def calibration(y, score):
    bins = np.minimum((score*10).astype(int), 9)
    records = []
    for b in range(10):
        m = bins == b
        if m.any():
            records.append(dict(bin=b, n=int(m.sum()), predicted=float(score[m].mean()), observed=float(y[m].mean())))
    ece = sum(x['n']*abs(x['predicted']-x['observed']) for x in records)/len(y)
    return records, float(ece)


def metrics(y, score, thresholds):
    cal, ece = calibration(y, score)
    return dict(auroc=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
                auprc=float(average_precision_score(y,score)) if np.sum(y) else None,
                brier=float(brier_score_loss(y,score)), ece=ece, calibration=cal,
                at_05=decision_metrics(y,score,.5),
                high_precision={str(k):decision_metrics(y,score,v) for k,v in thresholds.items()},
                descriptive_test_recall_at_precision={str(p):decision_metrics(y,score,operating_point(y,score,p,1))
                                                      for p in [.95,.97,.99]})


def paired_bootstrap(frame, y, b, d, tb, td, repeats, seed):
    """Resample prompts as clusters, paired across B/D; frozen fitted models."""
    groups = frame.prompt_group.to_numpy(); keys = np.unique(groups)
    indices = [np.flatnonzero(groups == key) for key in keys]
    rng = np.random.default_rng(seed); values = []
    for _ in range(repeats):
        ix = np.concatenate([indices[j] for j in rng.integers(0,len(keys),len(keys))])
        yy, bb, dd = y[ix], b[ix], d[ix]
        row = {'brier_improvement': float(np.mean((yy-bb)**2-(yy-dd)**2))}
        if len(np.unique(yy))==2:
            row['auroc_gain'] = roc_auc_score(yy,dd)-roc_auc_score(yy,bb)
            row['auprc_gain'] = average_precision_score(yy,dd)-average_precision_score(yy,bb)
        for target in tb:
            row[f'recall_gain_at_{target}'] = decision_metrics(yy,dd,td[target])['recall']-decision_metrics(yy,bb,tb[target])['recall']
            for label, score, threshold in [('B',bb,tb[target]),('D',dd,td[target])]:
                precision = decision_metrics(yy,score,threshold)['precision']
                if precision is not None:
                    row[f'{label}_precision_at_{target}'] = precision
        values.append(row)
    df = pd.DataFrame(values)
    return {c:dict(mean=float(df[c].mean()), low=float(df[c].quantile(.025)), high=float(df[c].quantile(.975))) for c in df}


def load_rows(paths, max_rows, seed, include_commits, pre_eos):
    frames = []; counts = []
    for path in paths:
        df = pd.read_parquet(path); raw = len(df)
        if not include_commits:
            df = df[~df.committed_this_step]
        if pre_eos:
            df = df[df.before_final_eos]
        eligible = len(df)
        if max_rows and len(df)>max_rows:
            file_seed = (seed+int(hashlib.sha256(path.name.encode()).hexdigest()[:8],16)) % 2**32
            df = df.sample(n=max_rows, random_state=file_seed)
        counts.append(dict(file=path.name, raw=raw, eligible=eligible, sampled=len(df)))
        frames.append(df)
    return pd.concat(frames,ignore_index=True), counts


def descriptive(train, test, out):
    """Thresholds only from training; joint confidence and lexical-history strata."""
    cuts = {}; tables = []; unstable_tables = []
    for col in [c for c in train if c.startswith('lat_') and c.endswith('__cos_lag1')]:
        vals = train[col].dropna()
        if not len(vals):
            continue
        lo, hi = vals.quantile([.25,.75]); cuts[col] = [float(lo),float(hi)]
        d = test.copy()
        d['latent_band'] = np.where(d[col] >= hi, 'high', np.where(d[col] <= lo, 'low', 'middle'))
        d.loc[d[col].isna(),'latent_band'] = 'missing'
        d['confidence_bucket'] = np.minimum((d.p1*10).astype(int),9)/10
        d['lexical_bucket'] = pd.cut(d.out_streak, [0,1,3,np.inf], labels=['changed_or_first','2-3','4+']).astype(str)
        d['layer'] = col
        for keys, dest, source in [(['layer','confidence_bucket','lexical_bucket','latent_band'],tables,d),
                                   (['layer','latent_band'],unstable_tables,d[d.out_changed==1])]:
            table = source.groupby(keys,observed=True).agg(n=('safe_to_commit','size'), safe_probability=('safe_to_commit','mean'),
                      stable_probability=('stable_to_end','mean'), prompts=('prompt_group','nunique')).reset_index()
            dest.append(table)
        plot = d[d.latent_band.isin(['low','high'])].groupby(['confidence_bucket','latent_band']).safe_to_commit.mean().unstack()
        if not plot.empty:
            plot.plot(marker='o'); plt.ylabel('P(SAFE_TO_COMMIT)'); plt.title(col); plt.ylim(0,1)
            plt.tight_layout(); plt.savefig(out/f'confidence_{col}.png'); plt.close()
        edges = np.unique(vals.quantile(np.linspace(0,1,11)).to_numpy())
        if len(edges)>2:
            edges[0],edges[-1] = -np.inf,np.inf
            d['stability_bin'] = pd.cut(d[col],edges)
            curve = d.groupby('stability_bin',observed=True).agg(cosine=(col,'mean'),safe=('safe_to_commit','mean'),n=('top1','size'))
            curve.to_csv(out/f'stability_safety_{col}.csv')
            plt.plot(curve.cosine,curve.safe,marker='o'); plt.xlabel('Latent cosine'); plt.ylabel('P(SAFE_TO_COMMIT)')
            plt.ylim(0,1); plt.tight_layout(); plt.savefig(out/f'stability_safety_{col}.png'); plt.close()
    pd.concat(tables,ignore_index=True).to_csv(out/'confidence_lexical_controls.csv',index=False)
    pd.concat(unstable_tables,ignore_index=True).to_csv(out/'unstable_lexical.csv',index=False)
    stability = [c for c in train if c.startswith('lat_') and c.endswith('__cos_lag1')]
    d = test.copy(); d['progress_bin'] = np.round(d.ctx_block_progress,1)
    d.groupby('progress_bin')[stability].mean().plot()
    plt.ylabel('Mean cosine (surviving masked positions)'); plt.tight_layout(); plt.savefig(out/'progress_stability.png'); plt.close()
    return cuts


def trajectory_analysis(paths, split, cuts, out):
    """Full, unsampled held-out trajectories; two consecutive stable transitions.

    Lead is retrospective descriptive evidence; never included in predictors.
    Examples are sorted by IDs, not by whether they support the hypothesis.
    """
    leads = []; examples = []; failures = []; recurrence = []; seen = set(); counts = {'high_unstable':0,'high_unstable_safe':0,'positions':set()}
    for path in paths:
        df = pd.read_parquet(path)
        if split.get(str(df.prompt_group.iloc[0])) != 'test':
            continue
        for (_,pos), seq in df.groupby(['generation_id','position'],sort=True):
            seq = seq.sort_values('step')
            lexical = seq.loc[seq.stable_to_end==1,'step']
            for col, (lo,hi) in cuts.items():
                high = (seq[col] >= hi)
                onset = seq.loc[high & high.shift(1,fill_value=False),'step']
                sustained = high.iloc[::-1].cummin().iloc[::-1]
                sustained_onset = seq.loc[sustained & high.shift(-1,fill_value=False),'step']
                leads.append(dict(generation_id=seq.generation_id.iloc[0], position=int(pos), layer=col,
                                  latent_onset=int(onset.iloc[0]) if len(onset) else None,
                                  lexical_onset=int(lexical.iloc[0]) if len(lexical) else None,
                                  lead_steps=int(lexical.iloc[0]-onset.iloc[0]) if len(onset) and len(lexical) else None,
                                  sustained_lead_steps=int(lexical.iloc[0]-sustained_onset.iloc[0]) if len(sustained_onset) and len(lexical) else None))
                eligible = seq[(seq.out_changed==1) & ~seq.committed_this_step]
                for band,mask in [('high',eligible[col]>=hi),('low',(eligible[col]<=lo) & (eligible[col]<hi))]:
                    subset=eligible[mask]
                    recurrence.append(dict(layer=col,band=band,n=len(subset),safe=int(subset.safe_to_commit.sum()),
                                           position_present=int(len(subset)>0),generation_id=str(seq.generation_id.iloc[0])))
            # Prespecified middle layer examples, includes safe and unsafe cases.
            col = 'lat_layer16__cos_lag1'
            if col in cuts:
                hit = seq[(seq.out_changed==1) & (seq[col]>=cuts[col][1]) & ~seq.committed_this_step]
                counts['high_unstable'] += len(hit); counts['high_unstable_safe'] += int(hit.safe_to_commit.sum())
                if len(hit):
                    key = (str(seq.generation_id.iloc[0]),int(pos)); counts['positions'].add(key)
                    if len(seen)<12:
                        seen.add(key); examples.append(seq)
                    if (hit.safe_to_commit==0).any() and len(failures)<12:
                        failures.append(seq)
    lead = pd.DataFrame(leads); lead.to_csv(out/'convergence_leads.csv',index=False)
    if len(lead):
        summary = lead.groupby('layer').agg(positions=('position','size'), both_observed=('lead_steps','count'),
                                            mean_lead=('lead_steps','mean'), median_lead=('lead_steps','median'))
        summary['latent_earlier_count'] = lead.assign(earlier=lead.lead_steps>0).groupby('layer').earlier.sum()
        summary.to_csv(out/'convergence_lead_summary.csv')
    counts['positions'] = len(counts['positions'])
    (out/'recurrence_counts.json').write_text(json.dumps(counts,indent=2))
    cols = ['generation_id','position','step','top1_text','p1','token_history','lat_layer16__cos_lag1','final_text','safe_to_commit','stable_to_end']
    ex = pd.concat(examples,ignore_index=True)[cols] if examples else pd.DataFrame(columns=cols)
    ex.to_csv(out/'examples.csv',index=False)
    fail = pd.concat(failures,ignore_index=True)[cols] if failures else pd.DataFrame(columns=cols)
    fail.to_csv(out/'failure_examples.csv',index=False)
    if recurrence:
        recurring = pd.DataFrame(recurrence).groupby(['layer','band']).agg(n=('n','sum'),safe=('safe','sum'),positions=('position_present','sum'))
        recurring['safe_probability'] = recurring.safe/recurring.n.replace(0,np.nan)
        recurring.to_csv(out/'full_unstable_recurrence.csv')
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=1729); ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--max-rows-per-generation',type=int,default=512)
    ap.add_argument('--min-selected',type=int,default=50)
    ap.add_argument('--include-commit-rows',action='store_true'); ap.add_argument('--pre-eos-only',action='store_true')
    ap.add_argument('--target',choices=['safe_to_commit','stable_to_end'],default='safe_to_commit')
    args = ap.parse_args(); run = Path(args.run); out = Path(args.out)
    if args.bootstrap < 1 or args.max_rows_per_generation < 0 or args.min_selected < 1:
        raise ValueError('Require bootstrap >=1, max rows >=0, min selected >=1')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty analysis output directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((run/'manifest.json').read_text())
    paths = sorted(run.glob('*.parquet'))
    if not paths:
        raise ValueError('No trajectory shards')
    df, counts = load_rows(paths,args.max_rows_per_generation,args.seed,args.include_commit_rows,args.pre_eos_only)
    df = df.replace([np.inf,-np.inf],np.nan)
    mapping = split_groups(df,args.seed); df['split'] = df.prompt_group.map(mapping)
    (out/'split.json').write_text(json.dumps(mapping,indent=2)); pd.DataFrame(counts).to_csv(out/'sampling.csv',index=False)
    (out/'analysis_config.json').write_text(json.dumps(vars(args),indent=2))
    df.groupby(['split','task']).agg(rows=('top1','size'), prompts=('prompt_group','nunique'), prevalence=(args.target,'mean')).to_csv(out/'cohorts.csv')
    frames = {name:df[df.split==name].copy() for name in ['train','validation','test']}
    for name, frame in frames.items():
        if frame[args.target].nunique()<2:
            raise ValueError(f'{name} has only one target class; cannot make a valid comparison. Collect more prompts.')
    groups = feature_groups(df.columns)
    (out/'feature_groups.json').write_text(json.dumps(groups,indent=2))
    results, predictions, thresholds = {}, {}, {}
    test = frames['test']; y = test[args.target].to_numpy()
    for name, columns in groups.items():
        model = make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=args.seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(frames['train'][columns],frames['train'][args.target])
        valp = model.predict_proba(frames['validation'][columns])[:,1]
        score = model.predict_proba(test[columns])[:,1]
        ts = {str(p): operating_point(frames['validation'][args.target].to_numpy(),valp,p,args.min_selected) for p in [.95,.97,.99]}
        thresholds[name] = ts; predictions[name] = score
        results[name] = dict(test=metrics(y,score,ts), validation_brier=float(brier_score_loss(frames['validation'][args.target],valp)))
        cal = results[name]['test']['calibration']
        if name in ['A_CURRENT_CONFIDENCE','B_OUTPUT_HISTORY','C_LATENT_HISTORY','D_OUTPUT_PLUS_LATENT']:
            plt.plot([v['predicted'] for v in cal],[v['observed'] for v in cal],marker='o',label=name)
        print(f'{name}: Brier={results[name]["test"]["brier"]:.6f}',flush=True)
    plt.plot([0,1],[0,1],color='black',linestyle='--'); plt.legend(fontsize=7); plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
    plt.tight_layout(); plt.savefig(out/'calibration.png'); plt.close()
    # Raw confidence and the actual credit-enhanced probability are also score baselines.
    for name,col in [('RAW_CONFIDENCE','p1'),('RAW_CREDIT','out_credit_candidate_p'),('RAW_ADAPTIVE_CREDIT','out_adaptive_credit_candidate_p')]:
        ts = {str(p):operating_point(frames['validation'][args.target].to_numpy(),frames['validation'][col].to_numpy(),p,args.min_selected) for p in [.95,.97,.99]}
        results[name] = dict(test=metrics(y,test[col].to_numpy(),ts))
    bn,dn = 'B_OUTPUT_HISTORY','D_OUTPUT_PLUS_LATENT'
    ci = paired_bootstrap(test,y,predictions[bn],predictions[dn],thresholds[bn],thresholds[dn],args.bootstrap,args.seed)
    results['paired_cluster_bootstrap'] = ci
    layer_names = [n for n in groups if n.startswith('D_lat_')]
    results['validation_selected_layer'] = min(layer_names,key=lambda n:results[n]['validation_brier'])
    gains = {n: results[bn]['test']['brier']-results[n]['test']['brier'] for n in layer_names}
    pd.Series(gains).to_csv(out/'layer_gains.csv',header=['brier_improvement'])
    pd.Series(gains).plot.bar(); plt.ylabel('Brier improvement over output history'); plt.tight_layout(); plt.savefig(out/'layer_gains.png'); plt.close()
    saved = test[['generation_id','prompt_group','position','step','task',args.target,'p1','out_changed']].copy()
    for name, pred in predictions.items(): saved[name] = pred
    saved.to_parquet(out/'heldout_predictions.parquet',index=False)
    results['per_task'] = {}
    for task in sorted(test.task.unique()):
        mask = test.task.to_numpy()==task
        results['per_task'][task] = {name:metrics(y[mask],predictions[name][mask],thresholds[name])
                                    for name in [bn,dn]}
    results['prompt_macro_brier'] = {}
    for name in [bn,dn]:
        errors = pd.DataFrame({'group':test.prompt_group.to_numpy(),'error':(y-predictions[name])**2})
        results['prompt_macro_brier'][name] = float(errors.groupby('group').error.mean().mean())
    # Weak output evidence uses a fixed score threshold, no outcome-based selection.
    weak_cut = .5  # fixed and interpretable: B predicts unsafe more likely than safe
    weak = (test.out_changed.to_numpy()==1) & (predictions[bn]<weak_cut)
    results['weak_output_unstable'] = dict(rows=int(weak.sum()), prompts=int(test.loc[weak,'prompt_group'].nunique()),
                    safe_probability=float(y[weak].mean()) if weak.any() else None,
                    B_brier=float(np.mean((y[weak]-predictions[bn][weak])**2)) if weak.any() else None,
                    D_brier=float(np.mean((y[weak]-predictions[dn][weak])**2)) if weak.any() else None)
    # Matched coarse output-risk strata provide a stronger descriptive control.
    controlled = test.copy(); controlled['output_risk_bin'] = np.minimum((predictions[bn]*10).astype(int),9)
    middle = 'lat_layer16__cos_lag1'
    high_cut = frames['train'][middle].quantile(.75)
    controlled = controlled[controlled[middle].notna()]
    controlled['high_latent'] = controlled[middle]>=high_cut
    controlled.groupby(['output_risk_bin','out_changed','high_latent']).agg(n=(args.target,'size'),
                    probability=(args.target,'mean'),prompts=('prompt_group','nunique')).to_csv(out/'output_risk_controls.csv')
    cuts = descriptive(frames['train'],test,out)
    recurrence = trajectory_analysis(paths,mapping,cuts,out)
    (out/'latent_thresholds.json').write_text(json.dumps(cuts,indent=2))
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    for name in (bn,dn):
        p,r,_ = precision_recall_curve(y,predictions[name]); plt.plot(r,p,label=name)
    plt.ylim(.9,1); plt.xlabel('Recall'); plt.ylabel('Precision (descriptive test curve)'); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out/'high_precision_pr.png'); plt.close()
    synthetic = manifest.get('synthetic',False)
    lines = ['# Phase 1 analysis', '', '**SYNTHETIC PIPELINE VALIDATION ONLY — no research evidence.**' if synthetic else '**Observational baseline-trajectory evidence, not a causal early-commit test.**','',
             f'Target: `{args.target}`. {len(df)} analyzed rows; {df.prompt_group.nunique()} prompt groups. Hash-group split probabilities: 60/20/20%.',
             'Thresholds chosen on validation; achieved held-out precision may fall below the requested target.', '',
             '| Model | AUROC | AUPRC | Brier | ECE |','|---|---:|---:|---:|---:|']
    for name in groups:
        m = results[name]['test']; lines.append(f"| {name} | {m['auroc']:.5f} | {m['auprc']:.5f} | {m['brier']:.5f} | {m['ece']:.5f} |")
    lines += ['', '| Model | Validation precision target | Test precision | Test recall | Selected |','|---|---:|---:|---:|---:|']
    for name in [bn,dn]:
        for target,m in results[name]['test']['high_precision'].items():
            lines.append(f"| {name} | {target} | {m['precision']} | {m['recall']:.5f} | {m['selected']} |")
    improvement = results[bn]['test']['brier']-results[dn]['test']['brier']
    lines += ['', f'B → D Brier improvement: {improvement:.6f}. Paired prompt bootstrap intervals: `{json.dumps(ci)}`.',
              'A small gain, an interval crossing zero, or failure at high precision is insufficient support for Phase 2.',
              'Predeclared practical screen: Brier improvement >=0.002 with cluster CI above zero, and >=0.01 recall gain at validation-target 97% while BOTH test precisions meet 97%. This is a screen, not a proof or automatic decoder authorization.',
              f'Full held-out trajectory recurrence counts: `{json.dumps(recurrence)}`.',
              'Layer comparisons are exploratory; select layers using validation only. CIs condition on trained models and do not include training-set uncertainty.',
              'See examples.csv (includes failures), convergence_lead_summary.csv, confidence_lexical_controls.csv, output_risk_controls.csv, and plots.',
              'The main analysis excludes baseline commit rows. Repeat with --include-commit-rows, --pre-eos-only, --target stable_to_end, and alternative split seeds. No ground-truth quality or speed claim follows from these labels.']
    (out/'ANALYSIS_REPORT.md').write_text('\n'.join(lines)+'\n')
    from .reporting import write_final_report
    write_final_report(out,run,manifest,results,'\n'.join(lines),recurrence)
    print('\n'.join(lines[-7:]))
    if (out/'examples.csv').exists():
        print(pd.read_csv(out/'examples.csv').head(24).to_string(index=False))


if __name__ == '__main__':
    main()
