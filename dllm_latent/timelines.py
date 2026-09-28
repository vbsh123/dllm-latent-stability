"""LEGACY anchored-window rule (not the actual rollout policy).

Use policy_timelines for current shared-policy comparisons. This module retains
the earlier, different window/drift detector for explicitly labeled ablations.
Tune explicit region tolerances; compare first-trigger timelines on baseline states.

No learned predictor. Credit uses its enhanced winner, including changed winners.
No-crossing events are censored at baseline commitment, never invented as commits.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .stability import AnchoredRegionRule, episode_triggers

RADII=(.01,.025,.05,.1,.2,.4)
WINDOWS=(2,3,4)
CONFIRMATIONS=(2,3)
CREDIT_CUTOFFS=(.95,.97,.99)
CORE=['generation_id','prompt_group','task','position','step','top1','final_token',
      'safe_to_commit','stable_to_end','committed_this_step','before_final_eos',
      'p1','credit_top1','out_credit_max_p','adaptive_credit_top1','out_adaptive_credit_max_p']


def split_group(group,seed):
    u=int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64
    return 'discovery' if u<.6 else ('calibration' if u<.8 else 'heldout')


def first_events(frame, accepted, candidate='top1', method='region', sorted_rows=False):
    """Exactly the first crossing, even if unsafe; each position contributes once."""
    accepted=np.asarray(accepted,dtype=bool)
    if len(accepted)!=len(frame): raise ValueError('Acceptance length mismatch')
    if not sorted_rows:
        frame=frame.assign(_accepted=accepted).sort_values(['position','step'])
        accepted=frame.pop('_accepted').to_numpy()
    positions=frame.position.to_numpy()
    last=np.r_[np.flatnonzero(positions[:-1]!=positions[1:]),len(frame)-1]
    final=frame.iloc[last]
    if not final.committed_this_step.all():
        raise ValueError('Need full trajectories including baseline commit rows')
    result=final[['generation_id','prompt_group','task','position','step','final_token','before_final_eos']].rename(columns={'step':'baseline_step'}).set_index('position')
    hit_indices=np.flatnonzero(accepted)
    if len(hit_indices):
        hit_positions=positions[hit_indices]
        hit_indices=hit_indices[np.r_[True,hit_positions[1:]!=hit_positions[:-1]]]
    hit=frame.iloc[hit_indices].set_index('position')
    result['trigger_step']=hit.step
    result['proposed_token']=hit[candidate]
    result['observed']=result.trigger_step.notna()
    result['early']=result.observed & (result.trigger_step<result.baseline_step)
    result['safe']=result.observed & (result.proposed_token==result.final_token)
    result['lead_steps']=result.baseline_step-result.trigger_step
    result['censored']=~result.observed
    result['method']=method
    if candidate=='top1':
        result['stable_to_end_at_trigger']=hit.stable_to_end
    return result.reset_index()


class Counts:
    def __init__(self):
        self.positions=0; self.selected=0; self.safe=0; self.safe_lead=0.; self.prompts=set()
    def add(self,events):
        self.positions+=len(events)
        early=events[events.early]
        self.selected+=len(early); self.safe+=int(early.safe.sum())
        self.safe_lead+=float(early.loc[early.safe,'lead_steps'].sum())
        self.prompts.update(early.prompt_group.unique())
    def record(self):
        return dict(positions=self.positions,early_selected=self.selected,early_safe=self.safe,
                    early_unsafe=self.selected-self.safe,precision=self.safe/self.selected if self.selected else None,
                    selection_prompts=len(self.prompts),coverage=self.selected/self.positions if self.positions else 0.,
                    safe_lead_per_position=self.safe_lead/self.positions if self.positions else 0.)


def qualifies(record,target,min_events,min_prompts):
    return (record['early_selected']>=min_events and record['selection_prompts']>=min_prompts and
            record['precision'] is not None and record['precision']>=target)


def choose_rule(records,target,min_events,min_prompts):
    eligible=[r for r in records if qualifies(r,target,min_events,min_prompts)]
    # Maximize correct early commitment lead per observed position, with fixed ties.
    return max(eligible,key=lambda r:(r['safe_lead_per_position'],r['precision'],-r['radius'],r['confirmations'],r['window'])) if eligible else None


def rule_key(rule): return f'k{rule.window}_c{rule.confirmations}_r{rule.radius:g}'


def columns_for(layer):
    return [f'{layer}__a{k}_c{c}_{metric}' for k in WINDOWS for c in (0,1,2,3) for metric in ('distance','radius','drift')]


def accept_region(df,layer,rule):
    return episode_triggers({f'a{rule.window}_c{c}_{metric}':
                          df[f'{layer}__a{rule.window}_c{c}_{metric}'].to_numpy()
                          for c in range(rule.confirmations+1) for metric in ('distance','radius','drift')},
                          df.position.to_numpy(),df.step.to_numpy(),rule)


def joined_comparison(region, credit, cutoff):
    keys=['generation_id','position']
    c=credit[keys+['trigger_step','proposed_token','observed','safe','early']].rename(columns={
        'trigger_step':'credit_step','proposed_token':'credit_token','observed':'credit_observed','safe':'credit_safe','early':'credit_early'})
    result=region.merge(c,on=keys,validate='one_to_one')
    both=result.observed & result.credit_observed
    result['credit_cutoff']=cutoff
    result['both_observed']=both
    result['lead_over_credit']=np.where(both,result.credit_step-result.trigger_step,np.nan)
    result['latent_before_credit']=both & (result.trigger_step<result.credit_step)
    result['latent_with_credit_censored']=result.early & ~result.credit_observed
    result['lead_lower_bound_if_credit_censored']=np.where(result.latent_with_credit_censored,result.baseline_step-result.trigger_step,np.nan)
    return result


def comparison_summary(joined):
    counts=Counts(); counts.add(joined)
    row=counts.record()
    both=joined[joined.both_observed]
    earlier=joined[joined.latent_before_credit]
    row.update(credit_observed=int(joined.credit_observed.sum()),latent_observed=int(joined.observed.sum()),
               latent_before_credit=len(earlier),latent_before_credit_safe=int(earlier.safe.sum()),
               latent_before_credit_unsafe=int((~earlier.safe).sum()),
               earlier_precision=float(earlier.safe.mean()) if len(earlier) else None,
               credit_censored_latent_early=int(joined.latent_with_credit_censored.sum()),
               mean_lead_over_credit_when_both=float(both.lead_over_credit.mean()) if len(both) else None)
    return row


def prompt_bootstrap(joined,repeats,seed):
    """Paired prompt intervals on early-trigger precision and safe/unsafe coverage."""
    df=joined.copy()
    df['region_early_safe']=(df.early & df.safe).astype(int)
    df['credit_early_safe']=(df.credit_early & df.credit_safe).astype(int)
    df['region_early_unsafe']=(df.early & ~df.safe).astype(int)
    df['credit_early_unsafe']=(df.credit_early & ~df.credit_safe).astype(int)
    df['one']=1
    totals=df.groupby('prompt_group')[['one','early','region_early_safe','credit_early','credit_early_safe',
                                      'region_early_unsafe','credit_early_unsafe']].sum().to_numpy(dtype=float)
    if len(totals)<2: return {'status':'insufficient prompt groups'}
    rng=np.random.default_rng(seed); rows=[]
    for _ in range(repeats):
        n,r,rs,c,cs,ru,cu=totals[rng.integers(0,len(totals),len(totals))].sum(axis=0)
        rows.append(dict(region_precision=rs/r if r else np.nan,credit_precision=cs/c if c else np.nan,
                         safe_early_coverage_gain=(rs-cs)/n,unsafe_early_coverage_gain=(ru-cu)/n))
    boot=pd.DataFrame(rows)
    return {col:dict(low=float(boot[col].dropna().quantile(.025)),high=float(boot[col].dropna().quantile(.975)),
                     valid_draws=int(boot[col].notna().sum())) for col in boot if boot[col].notna().any()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=1729); ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--precision-targets',type=float,nargs='+',default=[.95,.97,.99])
    ap.add_argument('--min-events',type=int,default=50); ap.add_argument('--min-prompts',type=int,default=10)
    ap.add_argument('--pre-eos-only',action='store_true')
    args=ap.parse_args()
    if args.min_events<1 or args.min_prompts<1 or args.bootstrap<1 or any(not 0<p<=1 for p in args.precision_targets):
        raise ValueError('Invalid precision/support/bootstrap settings')
    run=Path(args.run); out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Use an empty analysis directory')
    paths=sorted(run.glob('*.parquet'))
    if not paths: raise ValueError('No trajectories')
    schema=pq.read_schema(paths[0]).names
    layers=sorted({c.split('__')[0] for c in schema if c.startswith('lat_') and c.endswith('__a3_c2_distance')})
    if not layers or not set(CORE)<=set(schema) or any(not set(columns_for(layer))<=set(schema) for layer in layers):
        raise ValueError('Recollect with episode-prefix anchored features and credit winner IDs; old traces cannot provide exact timelines')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text())
    catalog=[]; credit_records=[]
    for path in paths:
        df=pd.read_parquet(path,columns=CORE).sort_values(['position','step']).reset_index(drop=True)
        if df.generation_id.nunique()!=1 or df.prompt_group.nunique()!=1:
            raise ValueError('Expect one complete generation per shard')
        if args.pre_eos_only: df=df[df.before_final_eos].reset_index(drop=True)
        if df.empty: continue
        split=split_group(str(df.prompt_group.iloc[0]),args.seed)
        catalog.append(dict(path=path,split=split,group=str(df.prompt_group.iloc[0]),generation=str(df.generation_id.iloc[0])))
        for method,score,candidate in [('confidence','p1','top1'),('credit','out_credit_max_p','credit_top1'),
                                       ('adaptive_credit','out_adaptive_credit_max_p','adaptive_credit_top1')]:
            for cutoff in CREDIT_CUTOFFS:
                events=first_events(df,df[score].to_numpy()>=cutoff,candidate,method,sorted_rows=True)
                events['cutoff']=cutoff; events['partition']=split
                credit_records.append(events)
    if not catalog: raise ValueError('No eligible generation')
    credit_all=pd.concat(credit_records,ignore_index=True)
    credit_all.to_parquet(out/'output_first_triggers.parquet',index=False)
    summaries=[]
    for (split,method,cutoff),group in credit_all.groupby(['partition','method','cutoff']):
        counts=Counts(); counts.add(group)
        summaries.append(dict(partition=split,method=method,cutoff=cutoff,observed=int(group.observed.sum()),
                              censored=int(group.censored.sum()),**counts.record()))
    pd.DataFrame(summaries).to_csv(out/'output_timeline_summary.csv',index=False)
    (out/'split.json').write_text(json.dumps({c['group']:c['split'] for c in catalog},indent=2))
    (out/'analysis_config.json').write_text(json.dumps(dict(vars(args),radii=RADII,windows=WINDOWS,
        confirmations=CONFIRMATIONS,semantics='shadow first crossing on baseline states; no policy feedback'),indent=2))
    rules=[AnchoredRegionRule(r,k,c) for r in RADII for k in WINDOWS for c in CONFIRMATIONS]
    discovery_rows=[]; choices=[]; comparison_rows=[]; intervals={}; selected_timelines=[]; example_comparisons=[]
    def read_layer(item,layer):
        d=pd.read_parquet(item['path'],columns=CORE+columns_for(layer)).sort_values(['position','step']).reset_index(drop=True)
        if args.pre_eos_only: d=d[d.before_final_eos].reset_index(drop=True)
        return d
    for layer in layers:
        counts={rule_key(rule):Counts() for rule in rules}
        for item in catalog:
            if item['split']!='discovery': continue
            df=read_layer(item,layer)
            for rule in rules:
                counts[rule_key(rule)].add(first_events(df,accept_region(df,layer,rule),sorted_rows=True))
        candidates=[dict(layer=layer,**asdict(rule),**counts[rule_key(rule)].record()) for rule in rules]
        discovery_rows.extend(candidates)
        nominated={target:choose_rule(candidates,target,args.min_events,args.min_prompts) for target in args.precision_targets}
        cache={}
        for target,nomination in nominated.items():
            if nomination is None:
                choices.append(dict(layer=layer,precision_target=target,status='no qualifying discovery setting'))
                continue
            rule=AnchoredRegionRule(nomination['radius'],nomination['window'],nomination['confirmations'])
            key=rule_key(rule)
            if key not in cache:
                cal=Counts()
                for item in catalog:
                    if item['split']!='calibration': continue
                    df=read_layer(item,layer); cal.add(first_events(df,accept_region(df,layer,rule),sorted_rows=True))
                cache[key]=cal.record()
            calibrated=cache[key]
            passed=qualifies(calibrated,target,args.min_events,args.min_prompts)
            choice=dict(layer=layer,precision_target=target,**asdict(rule),
                        status='frozen after calibration' if passed else 'calibration failed; abstain',
                        discovery=nomination,calibration=calibrated)
            choices.append(choice)
            if not passed: continue
            frames=[]
            for item in catalog:
                if item['split']!='heldout': continue
                df=read_layer(item,layer)
                frames.append(first_events(df,accept_region(df,layer,rule),sorted_rows=True))
            if not frames: continue
            events=pd.concat(frames,ignore_index=True)
            events['layer']=layer; events['precision_target']=target
            selected_timelines.append(events)
            for cutoff in CREDIT_CUTOFFS:
                credit=credit_all[(credit_all.partition=='heldout') & (credit_all.method=='credit') & (credit_all.cutoff==cutoff)]
                joined=joined_comparison(events,credit,cutoff)
                joined.to_parquet(out/f'comparison_{layer}_p{target:g}_credit{cutoff:g}.parquet',index=False)
                example_comparisons.append(joined.sort_values(['generation_id','position']).head(12))
                comparison_rows.append(dict(layer=layer,precision_target=target,credit_cutoff=cutoff,**comparison_summary(joined)))
                intervals[f'{layer}_p{target:g}_credit{cutoff:g}']=prompt_bootstrap(joined,args.bootstrap,args.seed)
        print(f'{layer}: discovery grid evaluated; nominees checked on calibration',flush=True)
    pd.DataFrame(discovery_rows).to_csv(out/'discovery_grid.csv',index=False)
    (out/'frozen_rules.json').write_text(json.dumps(choices,indent=2,allow_nan=False))
    pd.DataFrame(comparison_rows).to_csv(out/'heldout_comparisons.csv',index=False)
    if example_comparisons:
        pd.concat(example_comparisons,ignore_index=True).to_csv(out/'credit_vs_latent_examples.csv',index=False)
        gain=pd.DataFrame(comparison_rows)
        gain=gain[(gain.precision_target==args.precision_targets[0]) & (gain.credit_cutoff==.95)]
        if len(gain):
            gain=gain.set_index('layer')[['latent_before_credit_safe','latent_before_credit_unsafe']]
            gain.plot.bar(stacked=True,figsize=(max(7,len(gain)*.3),4))
            plt.ylabel('First latent triggers before Credit shadow trigger'); plt.tight_layout()
            plt.savefig(out/'layer_advance_and_errors.png'); plt.close()
    (out/'paired_prompt_intervals.json').write_text(json.dumps(intervals,indent=2,allow_nan=False))
    # Designate one primary layer/rule using discovery objective among calibration
    # survivors, without looking at its heldout performance. All layers remain reported.
    primary={}
    for target in args.precision_targets:
        eligible=[c for c in choices if c['precision_target']==target and c['status']=='frozen after calibration']
        primary[str(target)]=max(eligible,key=lambda c:c['discovery']['safe_lead_per_position']) if eligible else None
    (out/'primary_rules.json').write_text(json.dumps(primary,indent=2,allow_nan=False))
    if selected_timelines:
        events=pd.concat(selected_timelines,ignore_index=True)
        events.to_parquet(out/'region_first_triggers.parquet',index=False)
        examples=events.sort_values(['layer','precision_target','generation_id','position']).groupby(['layer','precision_target']).head(12)
        examples.to_csv(out/'timeline_examples.csv',index=False)
        events[events.early & ~events.safe].head(100).to_csv(out/'unsafe_first_triggers.csv',index=False)
        for layer,group in events[events.precision_target==args.precision_targets[0]].groupby('layer'):
            observed=group[group.early & group.safe].trigger_step.sort_values().to_numpy()
            if len(observed): plt.step(observed,np.arange(1,len(observed)+1)/len(group),where='post',alpha=.5,label=layer)
        cd=credit_all[(credit_all.partition=='heldout') & (credit_all.method=='credit') & (credit_all.cutoff==.95)]
        observed=cd[cd.early & cd.safe].trigger_step.sort_values().to_numpy()
        if len(observed): plt.step(observed,np.arange(1,len(observed)+1)/len(cd),color='black',linewidth=2,label='Credit shadow .95')
        plt.xlabel('Baseline denoising step (not actual method forward count)'); plt.ylabel('Fraction with safe early first trigger')
        plt.legend(fontsize=5,ncol=3); plt.tight_layout(); plt.savefig(out/'first_trigger_timelines.png'); plt.close()
    status='SYNTHETIC VALIDATION ONLY' if manifest.get('synthetic') else 'OBSERVATIONAL BASELINE TIMELINES'
    report=f'''# Timeline and closeness diagnostics

**{status}**

Collection model: {manifest.get('config',{}).get('model_id','synthetic fixture')}.
Output block length: {manifest.get('config',{}).get('block_length','synthetic fixture')}.
Steps are global baseline forward indices; in block mode histories start at block
eligibility and reset at block boundaries. Future blocks do not contribute history.

All available collected layers were searched. No classifier or learned weights. Explicit
radius/window/confirmation settings are selected on discovery prompts (60%), checked
once on calibration prompts (20%), then frozen for heldout prompts (20%). No fallback
to a second setting after calibration failure. Requested empirical precision targets:
{args.precision_targets}; minimum early events {args.min_events}, minimum contributing
prompt groups {args.min_prompts}, required separately for discovery and calibration.
These are empirical support requirements, not guarantees of population precision.

Anchored rule: freeze a center and RMS scale from K previous vectors. Require all C
confirmation vectors within radius r, reference cloud radius <=r/2, and maximum rolling
center displacement from the frozen center <=r/4. K=2/3/4, C=2/3, r={RADII}. Compare
all three quantities in raw geometry relative to the SAME frozen scale. This blocks
re-centering/expanding the region to accommodate the confirmation points. It cannot
prove absence of future drift. Adjacent pairwise cosine is no longer the tuned rule.

The checker forms one candidate episode per position/layer/rule. It validates the
reference cloud when K observations arrive, then checks every confirmation prefix.
Any distance, compactness or drift failure discards that anchor and all earlier
confirmation/reference points. The failing point seeds a new K-point reference
window. A top1 identity change alone does not reset latent history. The first trigger
is irreversible for evaluation; a later jump never erases an earlier error. Earlier
overlapping-window diagnostics are superseded by this explicit reset protocol.

Each position contributes its FIRST proposal, including wrong-token proposals. Tuning
maximizes correct lead over baseline commitment per position, constrained by early-event
precision. A commit-time-only event cannot inflate tuning precision. The same proposal
is not counted repeatedly at later denoising steps. No later safe event replaces an
unsafe first event. No-crossing results are censored at baseline commitment.

CreditDecoding uses the actual credit-enhanced top1 ID and maximum probability. That ID
may differ from raw top1. Its first threshold crossing at .95/.97/.99 is compared with
the region first trigger and the baseline actual commit step. These are SHADOW decisions
on shared baseline states. An actual CreditDecoding rollout changes the context and can
have different steps/tokens. No wall-clock speedup, actual CreditDecoding commitment
timeline or causal early-commit safety is established here. Censored comparisons report
a lower bound on shadow lead, not an invented CreditDecoding commit time.

Files: output_first_triggers.parquet and output_timeline_summary.csv; discovery_grid.csv;
frozen_rules.json (including failures); primary_rules.json; heldout_comparisons.csv;
per-layer comparison Parquet; credit_vs_latent_examples.csv with both trigger steps;
region_first_triggers.parquet and example/failure tables
when a setting qualifies; paired_prompt_intervals.json. No qualifying setting is an
honest negative/insufficient-evidence result, not permission to relax precision using
the heldout set. Primary layers are designated without inspecting heldout scores;
individual layer comparisons remain exploratory and multiple-comparison sensitive.

SAFE means agreement with the final baseline token, not ground truth or counterfactual
safety. Region STABLE_TO_END is separately recorded at its trigger. Credit's candidate
may differ, so its safety is computed from its own candidate ID. Compare gains with
unsafe counts and achieved precision; earlier errors are not benefits. Stage-relative
confidence, task differences, EOS suffixes, mask survival, quantization and public-data
contamination remain possible biases. Use the pre-EOS sensitivity.

Tuning uses label-derived hyperparameter selection, as requested. No weights or
classifier are trained. The pipeline is implemented; real closeness values remain
UNTUNED until actual GPU trajectories pass through this procedure. Paper parameters
for CreditDecoding are fixed; even the Instruct/block64 configuration is not an exact
paper reproduction: early stopping, benchmark prompting and actual policy rollout differ.
'''
    (out/'FINAL_EXPERIMENT_REPORT.md').write_text(report)
    print(f'{status}: {sum(c["status"]=="frozen after calibration" for c in choices)} layer/target settings passed calibration')


if __name__=='__main__': main()
