"""Generate independent GSM8K answers under actual commitment policies (GPU only)."""
import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .decoding import LatentSettings, decode
from .prompt_format import render_prompt
from .prompts import SOURCES
from .collect import validate_config
from .parity import select_parity_indices

NUMBER=r'[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
MARKER=re.compile(r'^[ \t]*####[ \t]*\$?[ \t]*('+NUMBER+r')[ \t]*$',re.MULTILINE)
INSTRUCTION='Solve the problem and show your reasoning. End your answer with a line of the form: #### number.'
DEFAULT_METHODS=('baseline','confidence','credit','latent','combined')
METHODS=DEFAULT_METHODS+('no_geometry','no_geometry_radius','no_geometry_credit','no_geometry_double')


def numeric(text):
    try:
        value=Decimal(text.replace(',','').strip())
        return str(value.normalize()) if value.is_finite() else None
    except InvalidOperation: return None


def extract_answer(text,strict=True):
    matches=list(MARKER.finditer(text)) if strict else list(re.finditer(NUMBER,text))
    if not matches: return None
    return numeric(matches[-1].group(1 if strict else 0))


def grade(text,gold,complete=True):
    target=extract_answer(gold)
    if target is None: raise ValueError('GSM8K gold lacks a valid numeric answer')
    strict=extract_answer(text);lenient=extract_answer(text,False)
    return dict(gold=target,predicted=strict,lenient_predicted=lenient,format_valid=strict is not None,
                correct=bool(complete and strict==target),lenient_correct=bool(complete and lenient==target))


def summarize_boost_attribution(records):
    """Never misreport older runs without instrumentation as having zero boost use."""
    keys=('boost_enabled','already_confident','fallback','scheduled','changed_candidate')
    available=bool(records) and all(all(k in r.get('acceptance_counts',{}) for k in keys) for r in records)
    result=dict(boost_attribution_available=available,
        boost_enabled_commits=None,already_confident_commits=None,
        actual_fallback_commits=None,actual_scheduled_commits=None,
        boost_changed_candidate_commits=None,boost_enabled_fraction=None,
        boost_enabled_fraction_of_threshold=None)
    if not available:return result
    total={k:sum(r['acceptance_counts'][k] for r in records) for k in keys}
    threshold=total['boost_enabled']+total['already_confident']
    all_commits=threshold+total['fallback']+total['scheduled']
    result.update(boost_enabled_commits=total['boost_enabled'],
        already_confident_commits=total['already_confident'],
        actual_fallback_commits=total['fallback'],actual_scheduled_commits=total['scheduled'],
        boost_changed_candidate_commits=total['changed_candidate'],
        boost_enabled_fraction=total['boost_enabled']/all_commits if all_commits else None,
        boost_enabled_fraction_of_threshold=total['boost_enabled']/threshold if threshold else None)
    return result


def summarize_region_observations(records):
    available=bool(records) and all(r.get('region_observation_counts') is not None for r in records)
    result=dict(region_observations_available=available,new_region_observations=None,
                reused_region_observations=None,invalid_region_observations=None,new_region_fraction=None)
    if available:
        counts={k:sum(r['region_observation_counts'][k] for r in records) for k in ('new','reused','invalid')}
        valid=counts['new']+counts['reused']
        result.update(new_region_observations=counts['new'],reused_region_observations=counts['reused'],
                      invalid_region_observations=counts['invalid'],
                      new_region_fraction=counts['new']/valid if valid else None)
    return result


def output_lengths(tokens,mask_id,stop_ids):
    end=next((i for i,t in enumerate(tokens) if t in stop_ids),len(tokens))
    visible=sum(t!=mask_id for t in tokens[:end])
    return dict(visible_output_tokens=visible,
                output_tokens_with_stop=visible+int(end<len(tokens)),
                generated_span_tokens=sum(t!=mask_id for t in tokens)),end


def summarize_tpf(records):
    """Ratio of totals; unsuccessful attempts contribute forwards but no delivered tokens."""
    forwards=sum(r['forwards'] for r in records)
    def metric(key):
        if not forwards or any(key not in r for r in records):return None
        return sum(r[key]*r['complete'] for r in records)/forwards
    return dict(output_tpf=metric('visible_output_tokens'),
                output_tpf_with_stop=metric('output_tokens_with_stop'),
                full_span_tpf=metric('generated_span_tokens'),
                mean_output_tpf=sum(r['visible_output_tokens']*r['complete']/r['forwards']
                    for r in records)/len(records) if records and all(r['forwards'] for r in records) else None,
                early_stop_enabled=all(r.get('early_stop_enabled',False) for r in records) if records else False,
                eos_stop_rate=sum(r.get('eos_stopped',False) for r in records)/len(records) if records else None)


def write_report(records,out,bootstrap=1000):
    out=Path(out);df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('commits','output','token_ids','question','formatted_prompt')} for r in records])
    if df.duplicated(['id','method']).any(): raise ValueError('Duplicate question/method result')
    df.to_csv(out/'question_results.csv',index=False)
    rows=[]
    for method,g in df.groupby('method',sort=False):
        rows.append(dict(method=method,questions=len(g),accuracy=float(g.correct.mean()),lenient_accuracy=float(g.lenient_correct.mean()),
            completion_rate=float(g.complete.mean()),format_rate=float(g.format_valid.mean()),
            mean_forwards=float(g.forwards.mean()),mean_decoder_seconds=float(g.seconds.mean()),
            output_tps=float((g.visible_output_tokens*g.complete).sum()/g.seconds.sum()),
            full_span_tps=float((g.generated_span_tokens*g.complete).sum()/g.seconds.sum()),
            mean_completed_output_tokens=float((g.visible_output_tokens*g.complete).sum()/max(1,g.complete.sum())),
            latent_commits=sum(r.get('counts',{}).get('latent',0) for r in records if r['method']==method),
            combined_commits=sum(r.get('counts',{}).get('combined',0) for r in records if r['method']==method),
            no_geometry_commits=sum(r.get('counts',{}).get('no_geometry',0) for r in records if r['method']==method),
            no_geometry_radius_commits=sum(r.get('counts',{}).get('no_geometry_radius',0) for r in records if r['method']==method),
            no_geometry_credit_commits=sum(r.get('counts',{}).get('no_geometry_credit',0) for r in records if r['method']==method),
            no_geometry_double_commits=sum(r.get('counts',{}).get('no_geometry_double',0) for r in records if r['method']==method),
            credit_commits=sum(r.get('counts',{}).get('credit',0) for r in records if r['method']==method),
            confidence_commits=sum(r.get('counts',{}).get('confidence',0) for r in records if r['method']==method),
            fallback_commits=int(g.fallback_commits.sum()),
            unchanged_context_steps=sum(r.get('unchanged_context_steps',0) for r in records if r['method']==method),
            forced_fraction=float(g.fallback_commits.sum()/max(1,g.total_commits.sum())),peak_gpu_gib=float(g.peak_gpu_bytes.max()/2**30),
            **summarize_boost_attribution([r for r in records if r['method']==method]),
            **summarize_region_observations([r for r in records if r['method']==method]),
            **summarize_tpf([r for r in records if r['method']==method])))
    summary=pd.DataFrame(rows)
    base=summary.loc[summary.method=='baseline','output_tpf']
    summary['output_tpf_relative_to_baseline']=summary.output_tpf/float(base.iloc[0]) if len(base) and base.iloc[0]>0 else None
    summary.to_csv(out/'summary.csv',index=False)
    rng=np.random.default_rng(1729);paired=[]
    for reference in ('baseline','credit','no_geometry','no_geometry_double','no_geometry_credit'):
        ref=df[df.method==reference].set_index('id')
        if ref.empty: continue
        for method,g in df.groupby('method'):
            if method==reference: continue
            g=g.set_index('id')
            if set(g.index)!=set(ref.index): raise ValueError('Compare identical question sets; missing results are not dropped')
            g=g.loc[ref.index];n=len(g)
            delta=g.correct.to_numpy(dtype=float)-ref.correct.to_numpy(dtype=float)
            draws=[]
            for _ in range(bootstrap):
                ix=rng.integers(0,n,n)
                draws.append(float(delta[ix].mean()))
            paired.append(dict(reference=reference,method=method,accuracy_delta=float(delta.mean()),
                accuracy_delta_low=float(np.quantile(draws,.025)),accuracy_delta_high=float(np.quantile(draws,.975)),
                forward_speed_ratio=float(ref.forwards.sum()/max(1,g.forwards.sum())),
                output_tpf_ratio=float(((g.visible_output_tokens*g.complete).sum()/g.forwards.sum())/
                    ((ref.visible_output_tokens*ref.complete).sum()/ref.forwards.sum()))
                    if (ref.visible_output_tokens*ref.complete).sum()>0 else None,
                measured_time_speed_ratio=float(ref.seconds.sum()/g.seconds.sum()) if g.seconds.sum()>0 else None,
                improved=int((delta>0).sum()),regressed=int((delta<0).sum())))
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    events=[]
    for r in records:
        for event in r['commits']:
            events.append(dict(id=r['id'],method=r['method'],answer_correct=r['correct'],**event))
    if events:
        ev=pd.DataFrame(events);ev.to_parquet(out/'actual_commitments.parquet',index=False)
        # Separate actual rollouts: matching positions does not imply equal contexts.
        first=ev.sort_values('step').drop_duplicates(['id','method','position'])
        first.pivot(index=['id','position'],columns='method',values=['step','token','token_text','reason']).to_csv(out/'paired_commitments.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for row in rows:
        plt.scatter(row['mean_forwards'],row['accuracy']);plt.annotate(row['method'],(row['mean_forwards'],row['accuracy']))
    plt.xlabel('Mean actual model forward passes');plt.ylabel('GSM8K final-answer accuracy');plt.ylim(0,1)
    plt.tight_layout();plt.savefig(out/'accuracy_vs_forwards.png');plt.close()
    report='''# Actual GSM8K policy rollout report

Each method generated its own answer, feeding its actual commitments into subsequent
forwards. Primary quality is numeric final-answer accuracy, NOT token agreement with
the baseline. Different valid wording is allowed. Strict grading takes the last
explicit #### number; an additional last-number metric exposes formatting sensitivity.
Incomplete prefixes count as incorrect even if a numeric answer is present. All
policies share the configured stopping rule, block size, length and forward cap.
With --early-stop, only a committed EOS/EOT with every preceding position finalized
terminates decoding. Unfilled suffix masks then do not make the answer incomplete.
Without --early-stop, the entire span must be filled. Check manifest.json for the mode.
Text is truncated at the first generated EOS/EOT for grading after generation.

Read manifest.json and policy.json before interpreting results. Shipped latent settings
are unvalidated starting values. Tune on GSM8K train, freeze, then run test; inspecting
test results to adjust thresholds/layers invalidates a heldout claim. The accuracy
intervals resample paired questions and do not certify noninferiority or correct for
parameter searches. A small smoke cannot establish superiority.

summary.csv reports accuracy, completion, formatting, forced-commit fraction, forward
passes and synchronized decoder seconds including policy/trace overhead. Loading,
preparation and warmup are excluded. paired_comparisons.csv reports paired accuracy
changes and compute ratios against baseline and CreditDecoding. Outputs and actual
commit events are saved, including token text. Different policies have different
contexts: a matched position's commit-step difference is not a same-state counterfactual.

Credit uses published Eq6/7 parameters and enhanced-token thresholding. The shared
optional top1 forced-progress fallback is an explicit addition to the paper pseudocode;
its usage is reported. No claim of exact paper benchmark reproduction. Latent credit
is retained in multiple frozen regions, decays globally, and receives p(top1)^gamma
on each binary region match (including new regions). Matched-region credit boosts
only the current raw top1 through the same logit fusion and confidence threshold.
Combined sums token and mapped regional credits before fusion; it has no OR gate.
No classifier is trained. Saved old support-gate settings are incompatible.
All uncompleted and incorrectly formatted answers remain in the denominator.

Boost attribution partitions actual non-mask insertions into boosted threshold
crossings (raw max < tau but enhanced max >= tau), already-confident threshold
acceptances, fallback, and baseline scheduled insertions. These counters work with
--no-trace. Changed candidate counts separately audit enhanced winners differing
from raw argmax. This is a same-state decision comparison, not a causal estimate of
saved forwards or final accuracy: trajectories already contain past boosted decisions.
Old records lacking acceptance_counts have unavailable attribution, not zero use.

The optional no_geometry control maintains a single discounted p(top1)^gamma balance
per position regardless of token identity, with no hidden capture, anchors, or distance
checks. Its layer/radius settings are ignored; coefficients and threshold are shared
with latent. It resets per block and uses the same fallback. Speed differences include
the saved geometric overhead; compare forwards as well as time and answer quality.
Latent runs also record new/reused/invalid region observations across all active steps,
including first visits and steps where no token is committed. No-geometry and older
runs have unavailable geometry counters, not zero creation rates.

Optional hybrids add position-level accumulation to radius matching or token credit.
no_geometry_radius earns (1 + bonus_weight*existing_match)*p(top1)^gamma each step in
one position balance; new anchors do not receive a match bonus. no_geometry_credit
adds the original token-credit vector (scaled by bonus_weight) to the position credit
mapped to current top1, before shared log fusion. no_geometry_double adds the same
extra increment unconditionally, a control for stronger boosting. This changes total
credit strength, so a faster hybrid alone is not evidence that its selector is useful.
Layer/radius apply only to the radius hybrid; all keep the common stopping rule.

output_tps is the total number of pre-EOS/EOT output tokens in completed generations
divided by total decoder seconds, including time spent on incomplete attempts.
full_span_tps also counts the generated suffix after EOS/EOT. It can overstate useful
throughput by counting suffix work, including positions already committed before early stopping. Neither metric counts prompt
tokens or intermediate predictions; ratios of totals are used, not averages of per-
answer TPS. Report accuracy and seconds per answer alongside TPS because verbosity
can change output length and throughput without improving task performance.
'''
    report+='\nTPF uses total delivered tokens / total forwards, including failed-attempt forwards.\n'
    report+='output_tpf excludes EOS/EOT; output_tpf_with_stop includes the first stop token.\n'
    report+='full_span_tpf counts every filled span position, including suffix work.\n'
    report+='mean_output_tpf separately averages per-answer ratios. Paper aggregation and exact\n'
    report+='stop-token conventions are not fully reproduced merely by enabling early stop.\n'
    (out/'FINAL_EXPERIMENT_REPORT.md').write_text(report+'\n'+summary.to_string(index=False)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/credit_instruct_block64.json')
    ap.add_argument('--policy-config',default='configs/rollout_policy.json')
    ap.add_argument('--out',required=True);ap.add_argument('--split',choices=['train','test'],default='train')
    ap.add_argument('--limit',type=int,default=100);ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=list(DEFAULT_METHODS))
    ap.add_argument('--gen-length',type=int);ap.add_argument('--steps',type=int);ap.add_argument('--block-length')
    ap.add_argument('--layer',type=int);ap.add_argument('--radius',type=float);ap.add_argument('--decay',type=float)
    ap.add_argument('--latent-alpha',type=float);ap.add_argument('--latent-gamma',type=float)
    ap.add_argument('--bonus-weight',type=float,help='Hybrid extra-credit multiplier; default 1')
    ap.add_argument('--double-bonus-start',type=int,choices=[1,2],help='Only no_geometry_double: first extra credit on observation 1 (default) or 2')
    ap.add_argument('--confidence-threshold',type=float);ap.add_argument('--fallback',choices=['top1','none'])
    ap.add_argument('--warmup',type=int,default=1)
    stopping=ap.add_mutually_exclusive_group()
    stopping.add_argument('--early-stop',dest='early_stop',action='store_true',help='Stop only at committed EOS/EOT with a fully finalized prefix')
    stopping.add_argument('--no-early-stop',dest='early_stop',action='store_false')
    ap.set_defaults(early_stop=False)
    tracing=ap.add_mutually_exclusive_group()
    tracing.add_argument('--trace',dest='trace',action='store_true',help='Separate explanatory run; off by default for timing')
    tracing.add_argument('--no-trace',dest='trace',action='store_false')
    ap.set_defaults(trace=False)
    ap.add_argument('--verify-parity',action='store_true')
    ap.add_argument('--parity-prompts',type=int,default=20)
    ap.add_argument('--parity-only',action='store_true',help='Verify baseline and trace invariance, then stop before measured runs')
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available(): raise SystemExit('Actual rollouts require CUDA; use CPU synthetic unit tests locally.')
    if args.limit<1 or args.offset<0 or args.warmup<0 or args.parity_prompts<1 or len(set(args.methods))!=len(args.methods): raise ValueError('Invalid sample/method settings')
    cfg=json.loads(Path(args.config).read_text());policy=json.loads(Path(args.policy_config).read_text())
    cfg['early_stop']=args.early_stop
    for key in ('gen_length','steps'):
        if getattr(args,key) is not None: cfg[key]=getattr(args,key)
    block=args.block_length or cfg['block_length'];cfg['block_length']=cfg['gen_length'] if block=='full' else int(block)
    validate_config(cfg)
    for key,arg in [('layer','layer'),('radius','radius'),('decay','decay'),('alpha','latent_alpha'),('gamma','latent_gamma'),('bonus_weight','bonus_weight'),('double_bonus_start','double_bonus_start')]:
        if getattr(args,arg) is not None: policy['latent'][key]=getattr(args,arg)
    if args.confidence_threshold is not None: policy['threshold']=args.confidence_threshold
    if args.fallback is not None: policy['fallback']=args.fallback
    latent=LatentSettings(**policy['latent'])
    policy['latent']=asdict(latent)  # Persist defaults too, including the hybrid weight.
    if not 0<policy['threshold']<=1 or policy['fallback'] not in ('top1','none'): raise ValueError('Invalid policy')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Use an empty run directory; never mix policy outputs')
    out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModel,AutoTokenizer
    from datasets import load_dataset
    tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],trust_remote_code=True)
    stop_ids={tok.eos_token_id} if tok.eos_token_id is not None else set()
    eot=tok.convert_tokens_to_ids('<|eot_id|>')
    if eot is not None and eot!=tok.unk_token_id:stop_ids.add(eot)
    cfg['stop_token_ids']=sorted(stop_ids)
    if args.early_stop and (not stop_ids or cfg['mask_id'] in stop_ids):raise ValueError('Tokenizer has no valid stop token set')
    dataset_id,revision=SOURCES['gsm8k'];data=load_dataset(dataset_id,'main',revision=revision,split=args.split)
    order=list(range(len(data)));random.Random(cfg['seed']).shuffle(order)
    prepared=[];skipped=[]
    for index in order:
        row=data[index];formatted=render_prompt(row['question']+'\n\n'+INSTRUCTION,cfg,tok)
        ids=tok(formatted,add_special_tokens=False,return_tensors='pt').input_ids
        if ids.shape[1]>cfg['max_prompt_tokens'] or (ids==cfg['mask_id']).any():
            skipped.append(index);continue
        if extract_answer(row['answer']) is None: raise ValueError('Invalid reference answer')
        prepared.append(dict(id=f'gsm8k-{args.split}-{index}',question=row['question'],answer=row['answer'],formatted=formatted,ids=ids))
        if len(prepared)>=args.offset+args.limit: break
    prepared=prepared[args.offset:args.offset+args.limit]
    if len(prepared)!=args.limit: raise ValueError('Insufficient eligible questions')
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('dllm_latent').rglob('*.py'))}
    manifest=dict(config=cfg,policy=policy,arguments=vars(args),dataset=dataset_id,dataset_revision=revision,
        split=args.split,question_ids=[p['id'] for p in prepared],excluded_rows=skipped,source_sha256=source,
        instruction=INSTRUCTION,early_stop=args.early_stop,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        python=platform.python_version(),cuda=torch.version.cuda,
        packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'policy.json').write_text(json.dumps(policy,indent=2))
    (out/'questions.jsonl').write_text(''.join(json.dumps({k:v for k,v in p.items() if k!='ids'})+'\n' for p in prepared))
    model=AutoModel.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],trust_remote_code=True,
        torch_dtype=getattr(torch,cfg['dtype']),low_cpu_mem_usage=True).to('cuda').eval()
    def run(method,ids,trace=None,early_stop=None):
        run_cfg=cfg if early_stop is None else dict(cfg,early_stop=early_stop)
        return decode(model,ids,run_cfg,method,policy['threshold'],latent,policy['fallback'],args.trace if trace is None else trace)
    if args.verify_parity or args.parity_only:
        from .third_party.llada_generate import generate
        selected=select_parity_indices([p['ids'].shape[1] for p in prepared],args.parity_prompts)
        checks=[]; trace_checks=[];stop_checks=[]
        for index in selected:
            p=prepared[index];ids=p['ids'].to('cuda')
            with torch.inference_mode():
                torch.manual_seed(cfg['seed'])
                original=generate(model,ids,steps=cfg['steps'],gen_length=cfg['gen_length'],block_length=cfg['block_length'],mask_id=cfg['mask_id'])
                torch.manual_seed(cfg['seed']);actual=run('baseline',ids,False,early_stop=False)['tokens']
            if not torch.equal(original,actual): raise AssertionError(f'Baseline parity failed: {p["id"]}')
            checks.append(dict(id=p['id'],prompt_tokens=int(ids.shape[1]),identical=True))
            print(f'Parity {len(checks)}/{len(selected)}: {p["id"]}, prompt tokens={ids.shape[1]}',flush=True)
        ids=prepared[selected[len(selected)//2]]['ids'].to('cuda')
        for method in args.methods:
            torch.manual_seed(cfg['seed']);traced=run(method,ids,True)
            torch.manual_seed(cfg['seed']);plain=run(method,ids,False)
            if not torch.equal(traced['tokens'],plain['tokens']) or traced['counts']!=plain['counts'] or traced['acceptance_counts']!=plain['acceptance_counts'] or traced['region_observation_counts']!=plain['region_observation_counts'] or traced['forwards']!=plain['forwards']:
                raise AssertionError(f'Trace invariance failed: {method}')
            for key in ('complete','eos_stopped','stop_position','stop_reason','unresolved_prefix_masks'):
                if traced[key]!=plain[key]:raise AssertionError(f'Trace stop invariance failed: {method}, {key}')
            trace_checks.append(dict(method=method,identical=True))
            if args.early_stop:
                torch.manual_seed(cfg['seed']);full=run(method,ids,False,early_stop=False)
                if plain['eos_stopped']:
                    end=ids.shape[1]+plain['stop_position']+1
                    same=torch.equal(plain['tokens'][:,:end],full['tokens'][:,:end])
                else: same=torch.equal(plain['tokens'],full['tokens'])
                if not same or plain['forwards']>full['forwards']:
                    raise AssertionError(f'Early-stop prefix parity failed: {method}')
                stop_checks.append(dict(method=method,identical_prefix=True,eos_stopped=plain['eos_stopped'],
                                        forwards=plain['forwards'],full_forwards=full['forwards']))
        (out/'parity.json').write_text(json.dumps(dict(requested=args.parity_prompts,checked=len(checks),checks=checks,
            trace_checks=trace_checks,early_stop_checks=stop_checks,baseline_parity_early_stop=False,trace_prompt_id=prepared[selected[len(selected)//2]]['id']),indent=2))
        if args.parity_only:
            print('Parity passed; no measured experiment was run.');return
    for method in args.methods:
        for _ in range(args.warmup):
            torch.manual_seed(cfg['seed']);run(method,prepared[0]['ids'].to('cuda'))
    torch.cuda.synchronize()
    records=[]
    for i,p in enumerate(prepared):
        methods=args.methods[i%len(args.methods):]+args.methods[:i%len(args.methods)]
        ids=p['ids'].to('cuda')
        for method in methods:
            torch.manual_seed(cfg['seed']);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();result=run(method,ids);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated();tokens=result.pop('tokens')[0,ids.shape[1]:].cpu().tolist()
            lengths,end=output_lengths(tokens,cfg['mask_id'],stop_ids)
            output=tok.decode(tokens[:end],skip_special_tokens=True)
            for event in result['commits']: event['token_text']=tok.decode([event['token']])
            record=dict(id=p['id'],method=method,question=p['question'],formatted_prompt=p['formatted'],output=output,token_ids=tokens,
                **lengths,
                seconds=elapsed,peak_gpu_bytes=peak,fallback_commits=result['counts'].get('fallback',0),
                total_commits=sum(result['counts'].values()),**result,**grade(output,p['answer'],result['complete']))
            records.append(record)
            with (out/'generations.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
            print(f'{i+1}/{len(prepared)} {method}: correct={record["correct"]}, forwards={record["forwards"]}, seconds={elapsed:.2f}',flush=True)
    print(write_report(records,out).to_string(index=False))


if __name__=='__main__': main()
