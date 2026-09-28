"""First-trigger diagnostics from the SAME configured RegionSupport as actual rollout.

No independent drift formula or offline radius fitting. Change the policy config
and recollect for a different rule. Legacy sliding-window statistics remain descriptive.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .timelines import CORE,CREDIT_CUTOFFS,first_events,Counts,split_group,joined_comparison,comparison_summary,prompt_bootstrap


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=1729);ap.add_argument('--bootstrap',type=int,default=200)
    ap.add_argument('--pre-eos-only',action='store_true');args=ap.parse_args()
    if args.bootstrap<1:raise ValueError('Bootstrap count must be positive')
    run=Path(args.run);out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    paths=sorted(run.glob('*.parquet'))
    if not paths:raise ValueError('No collected trajectories')
    manifest=json.loads((run/'manifest.json').read_text())
    policy=manifest.get('config',{}).get('latent_policy')
    if policy is None or policy.get('mechanism')!='persistent_regions_v2':
        raise ValueError('Recollect with persistent_regions_v2; legacy gates are not logit boosts')
    latent_cutoff=manifest['config'].get('policy_threshold',.95)
    credit_cutoffs=sorted(set((*CREDIT_CUTOFFS,latent_cutoff)))
    schema=pq.read_schema(paths[0]).names
    layers=sorted({c.split('__')[0] for c in schema if c.endswith('__policy_max_p')})
    if not layers:raise ValueError('No shared-policy decisions; old drift statistics cannot reproduce them')
    out.mkdir(parents=True,exist_ok=True)
    (out/'policy.json').write_text(json.dumps(dict(latent=policy,threshold=manifest['config'].get('policy_threshold',.95)),indent=2))
    summaries=[];intervals={}
    for layer in layers:
        events=[];credits=[]
        for path in paths:
            frame=pd.read_parquet(path,columns=CORE+[f'{layer}__policy_max_p',f'{layer}__policy_top1']).sort_values(['position','step'])
            if frame.generation_id.nunique()!=1 or frame.prompt_group.nunique()!=1:raise ValueError('One complete generation per shard required')
            if args.pre_eos_only:frame=frame[frame.before_final_eos]
            if frame.empty:continue
            partition=split_group(str(frame.prompt_group.iloc[0]),args.seed)
            event=first_events(frame,frame[f'{layer}__policy_max_p'].to_numpy()>=latent_cutoff,
                               candidate=f'{layer}__policy_top1',method='latent_region_boost',sorted_rows=True)
            event['partition']=partition;events.append(event)
            for cutoff in credit_cutoffs:
                credit=first_events(frame,frame.out_credit_max_p.to_numpy()>=cutoff,'credit_top1','credit',sorted_rows=True)
                credit['cutoff']=cutoff;credits.append(credit)
        if not events:continue
        latent=pd.concat(events,ignore_index=True);credit=pd.concat(credits,ignore_index=True)
        latent.to_parquet(out/f'{layer}_first_triggers.parquet',index=False)
        for cutoff in credit_cutoffs:
            joined=joined_comparison(latent,credit[credit.cutoff==cutoff],cutoff)
            joined['latent_cutoff']=latent_cutoff
            joined.to_parquet(out/f'{layer}_credit{cutoff:g}.parquet',index=False)
            for partition,group in joined.groupby('partition'):
                summaries.append(dict(layer=layer,partition=partition,latent_cutoff=latent_cutoff,credit_cutoff=cutoff,**comparison_summary(group)))
                intervals[f'{layer}_{partition}_{cutoff}']=prompt_bootstrap(group,args.bootstrap,args.seed)
        print(f'{layer}: shared-policy first triggers compared',flush=True)
    pd.DataFrame(summaries).to_csv(out/'comparisons.csv',index=False)
    (out/'paired_intervals.json').write_text(json.dumps(intervals,indent=2,allow_nan=False))
    (out/'analysis_config.json').write_text(json.dumps(vars(args),indent=2))
    status='SYNTHETIC VALIDATION ONLY' if manifest.get('synthetic') else 'OBSERVATIONAL BASELINE DIAGNOSTICS'
    (out/'FINAL_EXPERIMENT_REPORT.md').write_text(status+'\n\n'+'''# Shared-policy observational diagnostics

Decisions were recorded online by policy_observer.PolicyHistory, which delegates to
exactly the RegionSupport class used by actual decoding. policy.json records its
parameters. No alternate sliding-center formula or offline radius search is used.
Changing a radius/decay/fusion coefficient requires recollecting decisions with that configuration.
Scalar summaries alone do not reconstruct an arbitrary persistent-region policy.

Tables compare first proposals with Credit enhanced-winner threshold crossings on
shared baseline states. SAFE here is final-baseline-token agreement, not semantic
correctness or proof of counterfactual safety. Only actual independent task rollouts
measure answer quality. No wall-clock speedup is implied by these shadow timelines.

Partitions are prompt-group discovery/calibration/heldout. There is no automatic
selection in this report. Choose settings/layers on development data and preserve
heldout data. Do not transfer a chosen threshold from legacy anchored-window diagnostics.
Final-normalization capture is descriptive; actual policy currently selects layer1..32.
''')


if __name__=='__main__':main()
