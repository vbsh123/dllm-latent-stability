"""Audit/regrade saved Vast generations without model downloads or GPU inference."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from .answer_audit import exact_number, extract_review_answer


def dump_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def audit_run(run):
    from .gsm8k_rollout import extract_answer, output_lengths
    manifest=json.loads((run/'manifest.json').read_text())
    cfg=manifest['config']
    questions={}
    for line in (run/'questions.jsonl').read_text().splitlines():
        q=json.loads(line)
        if q['id'] in questions:raise ValueError('Duplicate question ID')
        questions[q['id']]=q
    expected=set(manifest['question_ids'])
    if set(questions)!=expected:raise ValueError('Question snapshot differs from manifest')
    rows=[];issues=[];seen=set();groups={}
    def check(ok,r,message):
        if not ok:issues.append(dict(run=str(run),method=r['method'],id=r['id'],issue=message))
    with (run/'generations.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=(r['id'],r['method'])
            if key in seen:raise ValueError(f'Duplicate generation: {key}')
            seen.add(key)
            if r['id'] not in questions:raise ValueError('Unexpected generated question')
            q=questions[r['id']]
            gold=exact_number(extract_answer(q['answer']) or '')
            if gold is None:raise ValueError('Invalid gold in question snapshot')
            check(exact_number(r['gold'])==gold,r,'saved_gold_mismatch')
            check(r['question']==q['question'] and r['formatted_prompt']==q['formatted'],r,'prompt_snapshot_mismatch')
            strict=bool(r['complete'] and exact_number(extract_answer(r['output']) or '')==gold)
            loose=bool(r['complete'] and exact_number(extract_answer(r['output'],False) or '')==gold)
            check(strict==r['correct'] and loose==r['lenient_correct'],r,'legacy_grade_not_reproducible')
            tokens=r['token_ids'];mask=cfg['mask_id'];stops=set(cfg['stop_token_ids'])
            lengths,end=output_lengths(tokens,mask,stops)
            check(len(tokens)==cfg['gen_length'],r,'generation_span_length_mismatch')
            for k,v in lengths.items():check(r.get(k)==v,r,'length_mismatch:'+k)
            check(0<r['forwards']<=cfg['steps'],r,'invalid_forward_count')
            check(math.isfinite(r['seconds']) and r['seconds']>0,r,'invalid_timing')
            check(r.get('early_stop_enabled')==cfg.get('early_stop',False),r,'stop_config_mismatch')
            if r.get('eos_stopped'):
                check(end<len(tokens) and mask not in tokens[:end] and r['stop_position']==end and r['complete'],r,'invalid_finalized_stop')
            elif r['complete']:
                check(mask not in tokens,r,'complete_with_masks_without_stop')
            check(r['unresolved_masks']==tokens.count(mask),r,'unresolved_mask_count_mismatch')
            a=r.get('acceptance_counts')
            if a:
                inserted=sum(a[k] for k in ('boost_enabled','already_confident','fallback','scheduled'))
                check(inserted==lengths['generated_span_tokens'],r,'actual_commit_partition_mismatch')
                check(0<=a['changed_candidate']<=a['boost_enabled']+a['already_confident'],r,'changed_candidate_count_invalid')
            check(sum(r['counts'].values())==r['total_commits'],r,'total_commit_count_mismatch')
            check(r['counts'].get('fallback',0)==r['fallback_commits'],r,'fallback_count_mismatch')
            extracted=extract_review_answer(r['output'])
            auto=bool(r['complete'] and not extracted['needs_review'] and extracted['candidate']==gold)
            row=dict(run=str(run),method=r['method'],id=r['id'],gold=gold,
                legacy_strict_correct=strict,legacy_lenient_correct=loose,complete=r['complete'],
                **extracted,automatic_correct=auto,output=r['output'],question=q['question'],
                manual_answer='',manual_note='')
            rows.append(row);groups.setdefault(r['method'],[]).append((r,row))
    for method in manifest['arguments']['methods']:
        if {r['id'] for r,_ in groups.get(method,[])}!=expected:
            issues.append(dict(run=str(run),method=method,id='ALL',issue='missing_or_extra_questions'))
    for method in groups:
        if method not in manifest['arguments']['methods']:
            issues.append(dict(run=str(run),method=method,id='ALL',issue='unexpected_method'))
    summary=[]
    for method,pairs in groups.items():
        n=len(pairs);pending=sum(row['needs_review'] for _,row in pairs)
        good=sum(row['automatic_correct'] for _,row in pairs)
        fwd=sum(r['forwards'] for r,_ in pairs)
        seconds=sum(r['seconds'] for r,_ in pairs)
        delivered=sum(r['visible_output_tokens']*r['complete'] for r,_ in pairs)
        summary.append(dict(run=str(run),method=method,questions=n,
            strict_accuracy=sum(row['legacy_strict_correct'] for _,row in pairs)/n,
            lenient_accuracy=sum(row['legacy_lenient_correct'] for _,row in pairs)/n,
            automatic_correct=good,needs_manual_review=pending,
            automatic_correct_fraction_all_questions=good/n,
            mean_forwards=fwd/n,output_tpf=delivered/fwd if fwd>0 else None,
            output_tps=delivered/seconds if math.isfinite(seconds) and seconds>0 else None))
    saved_path=run/'summary.csv'
    if saved_path.exists():
        with saved_path.open() as f:saved=list(csv.DictReader(f))
        by_method={r['method']:r for r in saved}
        if len(by_method)!=len(saved) or set(by_method)!=set(groups):
            issues.append(dict(run=str(run),method='ALL',id='ALL',issue='summary_method_set_mismatch'))
        for computed in summary:
            prior=by_method.get(computed['method'],{})
            for key,oldkey in [('strict_accuracy','accuracy'),('lenient_accuracy','lenient_accuracy'),
                               ('mean_forwards','mean_forwards'),('output_tpf','output_tpf'),('output_tps','output_tps')]:
                if oldkey not in prior:continue  # Older summaries may omit TPF.
                try:ok=math.isclose(float(prior[oldkey]),computed[key],rel_tol=1e-7,abs_tol=1e-8)
                except (ValueError,TypeError):ok=False
                if not ok:issues.append(dict(run=str(run),method=computed['method'],id='ALL',issue='summary_mismatch:'+oldkey))
    return rows,summary,issues


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',nargs='+',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('Use a fresh audit directory; originals are never overwritten')
    rows=[];summary=[];issues=[];sources={}
    for name in args.run:
        run=Path(name);a,b,c=audit_run(run);rows+=a;summary+=b;issues+=c
        for filename in ('manifest.json','questions.jsonl','generations.jsonl'):
            path=run/filename;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    out.mkdir(parents=True)
    dump_csv(out/'answers.csv',rows)
    review=[r for r in rows if r['needs_review'] or r['legacy_strict_correct']!=r['legacy_lenient_correct']
            or r['automatic_correct']!=r['legacy_lenient_correct']]
    dump_csv(out/'review.csv',review)
    dump_csv(out/'summary.csv',summary);dump_csv(out/'integrity_issues.csv',issues)
    groups={}
    for r in rows:groups.setdefault((r['run'],r['method']),{})[r['id']]=r
    paired=[]
    for (ka,a),(kb,b) in itertools.combinations(groups.items(),2):
        # Never quietly compare overlapping subsets or different question text/gold.
        same=set(a)==set(b) and all((a[i]['question'],a[i]['gold'])==(b[i]['question'],b[i]['gold']) for i in a)
        record=dict(reference=str(ka),method=str(kb),matched_questions=same)
        for metric in ('legacy_strict_correct','legacy_lenient_correct'):
            record[metric+'_improved']=sum(b[i][metric] and not a[i][metric] for i in a) if same else None
            record[metric+'_regressed']=sum(a[i][metric] and not b[i][metric] for i in a) if same else None
        paired.append(record)
    dump_csv(out/'paired_legacy.csv',paired)
    (out/'audit_manifest.json').write_text(json.dumps(dict(source_sha256=sources,
        parser_sha256=hashlib.sha256(Path(__file__).with_name('answer_audit.py').read_bytes()).hexdigest()),indent=2))
    (out/'README.md').write_text('''# Offline evaluation audit

Original files and grades are unchanged. No model/tokenizer is loaded. Integrity
checks recompute token lengths, TPF, stop-prefix validity, count partitions and
legacy grades; gold and prompts are checked against the saved question snapshot.
This does not verify dataset provenance against an external copy, regenerate text
from token IDs, or prove that saved forward counts reflect actual GPU calls.

The new parser uses the last explicit answer marker/box/answer phrase and never
looks at gold to choose a candidate. Units and punctuation can be accepted;
expressions, extra numbers, prose or missing markers are flagged for review.
Unresolved answers remain in the denominator and are not automatically correct.
automatic_correct_fraction_all_questions is a conservative diagnostic, NOT a
validated accuracy estimate or an official GSM8K score. Review unflagged cases too.
review.csv preserves full answer text, including disagreements with old graders;
manual_answer/manual_note are blank review fields (not automatically ingested).
paired_legacy.csv compares only identical question sets, using original graders.
Inspect integrity_issues.csv before using any metrics; nonempty issues exit with 1.
''')
    for s in summary:print(json.dumps(s))
    print(f'Integrity issues: {len(issues)}. Review rows: {len(review)}. Reports: {out}')
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
