"""Training-free stability audit of unchanged baseline trajectories; never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from .stability import StabilityRule


THRESHOLDS = (.95,.98,.99,.995)
WINDOWS = (2,3,4)


def partition(group,seed):
    u=int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64
    return 'heldout' if u >= .8 else 'exploratory'


def summarize(y, accepted):
    y=np.asarray(y,dtype=int); accepted=np.asarray(accepted,dtype=bool)
    n=int(accepted.sum()); tp=int(y[accepted].sum())
    return dict(rows=len(y),selected=n,precision=tp/n if n else None,
                recall=tp/int(y.sum()) if y.sum() else None,
                coverage=n/len(y) if len(y) else None)


def raw_score_metrics(y,score,probability=False):
    y=np.asarray(y); score=np.asarray(score); valid=np.isfinite(score)
    y,score=y[valid],score[valid]
    result=dict(rows=len(y),auroc=None,auprc=None)
    if len(np.unique(y))==2:
        result.update(auroc=float(roc_auc_score(y,score)),auprc=float(average_precision_score(y,score)))
    if probability and len(y):
        result['brier']=float(np.mean((y-score)**2))
        bins=np.minimum((score*10).astype(int),9)
        result['calibration']=[dict(bin=i,n=int((bins==i).sum()),predicted=float(score[bins==i].mean()),
                                    observed=float(y[bins==i].mean())) for i in range(10) if (bins==i).any()]
    return result


def bootstrap_gain(frame, baseline, combined, target, repeats,seed):
    """Paired prompt-cluster intervals for fixed rules, with no threshold selection."""
    groups=frame.prompt_group.to_numpy(); keys=np.unique(groups)
    if len(keys)<2: return {'status':'fewer than two prompt groups'}
    indices=[np.flatnonzero(groups==g) for g in keys]
    rng=np.random.default_rng(seed); y=frame[target].to_numpy(); draws=[]
    for _ in range(repeats):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(keys),len(keys))])
        b=summarize(y[ix],baseline[ix]); d=summarize(y[ix],combined[ix])
        if b['recall'] is not None and d['recall'] is not None:
            draws.append(d['recall']-b['recall'])
    if not draws: return {'status':'no positive labels in resamples'}
    return dict(recall_gain_low=float(np.quantile(draws,.025)),recall_gain_high=float(np.quantile(draws,.975)),
                prompts=len(keys),draws=len(draws),note='Read with achieved precision; adding checks can admit unsafe tokens.')


def analyze(frame,out,target,seed,repeats):
    layers=sorted({c.split('__')[0] for c in frame if c.startswith('lat_') and c.endswith('__w3_cos_min')})
    if not layers: raise ValueError('No latent trajectory features found')
    frame=frame.copy(); frame['partition']=frame.prompt_group.map(lambda g:partition(g,seed))
    frame.groupby(['partition','task']).agg(rows=(target,'size'),prompts=('prompt_group','nunique'),positive_rate=(target,'mean')).to_csv(out/'cohorts.csv')
    records=[]; scores={}; boot={}; regions=[]; layer_scores=[]
    for split in ('exploratory','heldout'):
        df=frame[frame.partition==split]; y=df[target].to_numpy()
        if df.empty: continue
        base=np.maximum(df.p1.to_numpy(),df.out_credit_candidate_p.to_numpy())
        scores[split]={name:raw_score_metrics(y,df[col],True) for name,col in
                       [('confidence','p1'),('credit','out_credit_candidate_p'),('adaptive_credit','out_adaptive_credit_candidate_p')]}
        for t in (.95,.97,.99):
            for name,col in [('confidence','p1'),('credit','out_credit_candidate_p'),('adaptive_credit','out_adaptive_credit_candidate_p')]:
                records.append(dict(partition=split,rule=name,threshold=t,**summarize(y,df[col].to_numpy()>=t)))
        for layer in layers:
            for k in WINDOWS:
                key=f'{layer}__w{k}_cos_min'; score=df[key].to_numpy()
                scores[split][key]=raw_score_metrics(y,score)
                layer_scores.append(dict(partition=split,layer=layer,measurement=f'w{k}_cos_min',**scores[split][key]))
                for t in THRESHOLDS:
                    accept=StabilityRule(k,t).evaluate({f'w{k}_cos_min':score})
                    records.append(dict(partition=split,rule=layer,window=k,threshold=t,**summarize(y,accept)))
            # Region distances are diagnostics, with no tuned acceptance radius.
            for geometry in ('raw','unit'):
                for metric in ('center_distance','radius','center_drift'):
                    key=f'{layer}__w3_{geometry}_{metric}'
                    if key not in df: continue
                    value=df[key].to_numpy()
                    result=raw_score_metrics(y,-value)
                    layer_scores.append(dict(partition=split,layer=layer,measurement=f'w3_{geometry}_{metric}',**result))
                    for label in (0,1):
                        subset=value[(y==label) & np.isfinite(value)]
                        if len(subset):
                            regions.append(dict(partition=split,layer=layer,geometry=geometry,measurement=metric,
                                                target_value=label,rows=len(subset),p25=float(np.quantile(subset,.25)),
                                                median=float(np.median(subset)),p75=float(np.quantile(subset,.75))))
        # Explicit observational add-on, not a deployed decoder or a probability model.
        primary='lat_layer16' if 'lat_layer16' in layers else layers[len(layers)//2]
        stable=StabilityRule().evaluate({'w3_cos_min':df[f'{primary}__w3_cos_min'].to_numpy()})
        for t in (.95,.97,.99):
            baseline=base>=t; added=stable & ~baseline; combined=baseline|stable
            for name,mask in [('output_baseline',baseline),('latent_additional',added),('output_or_latent',combined)]:
                records.append(dict(partition=split,rule=name,layer=primary,window=3,threshold=t,
                                    latent_threshold=.99,**summarize(y,mask)))
            boot[f'{split}_{t}']=bootstrap_gain(df,baseline,combined,target,repeats,seed)
        df=df.copy(); df['latent_stable']=stable
        df['confidence_bin']=np.minimum((df.p1*10).astype(int),9)/10
        df['lexical_bucket']=pd.cut(df.out_streak,[0,1,3,np.inf],labels=['1','2-3','4+']).astype(str)
        df['credit_strong']=df.out_credit_candidate_p>=.95
        df['latent_observed']=df[f'{primary}__w3_cos_min'].notna()
        observed=df[df.latent_observed]
        observed.groupby(['confidence_bin','lexical_bucket','credit_strong','latent_stable'],observed=True).agg(
            rows=(target,'size'),positive_rate=(target,'mean'),prompts=('prompt_group','nunique')).to_csv(out/f'{split}_conditional_rates.csv')
        unstable=observed[observed.out_changed==1]
        unstable.groupby(['credit_strong','latent_stable']).agg(rows=(target,'size'),positive_rate=(target,'mean'),
            prompts=('prompt_group','nunique')).to_csv(out/f'{split}_lexical_changes.csv')
        curve=observed.groupby(['confidence_bin','latent_stable'])[target].mean().unstack()
        if not curve.empty:
            curve.plot(marker='o'); plt.ylabel(f'P({target})'); plt.ylim(0,1)
            plt.tight_layout(); plt.savefig(out/f'{split}_confidence_control.png'); plt.close()
        candidate=unstable[unstable.latent_stable & ~unstable.credit_strong]
        keys=candidate[['generation_id','position']].drop_duplicates().sort_values(['generation_id','position']).head(12)
        examples=df.merge(keys,on=['generation_id','position']).sort_values(['generation_id','position','step'])
        columns=['generation_id','position','step','top1_text','p1','token_history',f'{primary}__w3_cos_min','final_text','safe_to_commit','stable_to_end']
        examples[columns].to_csv(out/f'{split}_examples.csv',index=False)
        candidate[candidate[target]==0][columns].head(100).to_csv(out/f'{split}_unsafe_stable.csv',index=False)
    pd.DataFrame(records).to_csv(out/'fixed_rule_metrics.csv',index=False)
    pd.DataFrame(regions).to_csv(out/'region_diagnostics.csv',index=False)
    pd.DataFrame(layer_scores).to_csv(out/'layer_scores.csv',index=False)
    (out/'score_metrics.json').write_text(json.dumps(scores,indent=2,allow_nan=False))
    (out/'paired_prompt_intervals.json').write_text(json.dumps(boot,indent=2))
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--target',choices=['safe_to_commit','stable_to_end'],default='safe_to_commit')
    ap.add_argument('--seed',type=int,default=1729); ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--include-commit-rows',action='store_true'); ap.add_argument('--pre-eos-only',action='store_true')
    args=ap.parse_args(); run=Path(args.run); out=Path(args.out)
    if args.bootstrap<1: raise ValueError('bootstrap must be positive')
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty analysis directory')
    frames=[]
    for p in sorted(run.glob('*.parquet')):
        df=pd.read_parquet(p)
        if not args.include_commit_rows: df=df[~df.committed_this_step]
        if args.pre_eos_only: df=df[df.before_final_eos]
        frames.append(df)
    if not frames: raise ValueError('No trajectory shards')
    frame=pd.concat(frames,ignore_index=True)
    if frame.empty: raise ValueError('No eligible rows')
    out.mkdir(parents=True,exist_ok=True)
    records=analyze(frame,out,args.target,args.seed,args.bootstrap)
    manifest=json.loads((run/'manifest.json').read_text())
    settings=dict(vars(args),training_free=True,primary_layer=16,primary_window=3,primary_cosine=.99,
                  sensitivity_windows=WINDOWS,sensitivity_cosines=THRESHOLDS,no_row_sampling=True)
    (out/'analysis_config.json').write_text(json.dumps(settings,indent=2))
    status='SYNTHETIC ENGINEERING CHECK ONLY; no model evidence.' if manifest.get('synthetic') else 'Observational real-model trajectories; no intervention or causal safety claim.'
    text=f'''# Training-free experiment report

{status}

No classifier, fitting, learned weights, imputation, or data-selected threshold was used.
Fixed checks use minimum cosine over K consecutive transitions. Missing history fails
the check. The primary check is layer16, K3, cosine>=.99; the complete predefined grid
is reported without selecting a winner. These thresholds are hypotheses, not validated
safety guarantees. Scores are not calibrated probabilities of safe commitment.

Target: {args.target}. Eligible rows: {len(frame)}. Groups: {frame.prompt_group.nunique()}.
Full configuration: analysis_config.json. Partition is deterministic by prompt hash:
80% exploratory and 20% heldout, with no training partition and no fitted estimator.

CreditDecoding Eq6/7 is computed on unchanged baseline trajectories. Fixed confidence,
credit and adaptive-credit thresholds .95/.97/.99 are reported with achieved precision
and recall. These are probability cutoffs, NOT guaranteed precision levels. The
output-or-latent comparison is an observational rule audit; its additional selections
and unsafe cases must be inspected before any decoder intervention. Bootstrap recall
gains alone cannot justify a change if precision falls.

See fixed_rule_metrics.csv, conditional_rates/lexical_changes tables, examples and
unsafe_stable files. The tables compare latent checks at matched coarse confidence,
lexical-history and credit-evidence levels. This tests a fixed stability signal beyond
output evidence; it does not prove independence from every possible output-history rule.
Cosine receives rank metrics only; Brier/calibration apply only to raw output probability
scores. No learned calibration is performed. No claim of 95/97/99% safe-commit precision
is made unless that precision is actually observed with adequate independent support.

Region diagnostics measure each new state against the PREVIOUS window's centroid,
the maximum radius of that previous cloud, and the shift of the rolling center.
Both raw-vector relative distances and unit-vector distances are recorded. No region
radius has been tuned or used to change decoding. region_diagnostics.csv and
layer_scores.csv support an exploratory layer/geometry comparison; lock any choices
before using heldout results for confirmation. Selecting a layer is parameter tuning
even without training a classifier. Old cosine presets remain an explicit comparator.

Limitations: baseline-relative labels, one checkpoint, fixed thresholds, cosine
anisotropy, BF16 quantization, EOS/survival effects, coarse bins, multiple comparisons,
and bootstrap conditional on this corpus. All rows are loaded; large runs require
adequate host RAM. Fixed constants and alternatives are in DESIGN_DECISIONS.md.
Phase2 remains deferred; a positive observational result is insufficient by itself.
'''
    (out/'FINAL_EXPERIMENT_REPORT.md').write_text(text)
    print(status)
    print(pd.DataFrame(records).query("rule in ['output_baseline','latent_additional','output_or_latent']").to_string(index=False))


if __name__=='__main__': main()
