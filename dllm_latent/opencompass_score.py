"""Pinned upstream GSM8K scoring on saved answers; no model or OpenCompass install."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

VENDOR=Path(__file__).parent/'third_party'/'opencompass'
PROVENANCE=json.loads((VENDOR/'provenance.json').read_text())
SOURCE=VENDOR/'gsm8k_upstream.py'


def upstream_functions():
    """Execute the exact selected upstream AST, omitting registry decorators/imports.

    Gsm8kEvaluator's methods do not depend on BaseEvaluator behavior; object stands
    in for that framework base. Extraction and equality/score bodies are unchanged.
    """
    source=SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest()!=PROVENANCE['sha256']['opencompass/datasets/gsm8k.py']:
        raise ValueError('Vendored OpenCompass source hash mismatch')
    wanted={'gsm8k_dataset_postprocess','gsm8k_postprocess','Gsm8kEvaluator'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise ValueError('Missing upstream scoring definitions')
    for node in nodes:node.decorator_list=[]
    namespace={'re':re,'BaseEvaluator':object}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace


def score_run(run,methods=None):
    api=upstream_functions();evaluator=api['Gsm8kEvaluator']()
    questions={q['id']:q for q in map(json.loads,(run/'questions.jsonl').read_text().splitlines())}
    manifest=json.loads((run/'manifest.json').read_text())
    if set(questions)!=set(manifest['question_ids']):raise ValueError('Question snapshot differs from manifest')
    groups={};seen=set()
    with (run/'generations.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if methods and r['method'] not in methods:continue
            key=(r['id'],r['method'])
            if key in seen:raise ValueError('Duplicate question/method result')
            seen.add(key)
            q=questions[r['id']]
            if r['question']!=q['question'] or r['formatted_prompt']!=q['formatted']:
                raise ValueError('Saved prompt/question mismatch')
            groups.setdefault(r['method'],[]).append(r)
    requested=set(methods or manifest['arguments']['methods'])
    if set(groups)!=requested:raise ValueError('Missing or unexpected methods')
    summaries=[];details=[]
    for method,records in groups.items():
        if {r['id'] for r in records}!=set(questions):raise ValueError('Incomplete run: missing question results')
        predictions=[api['gsm8k_postprocess'](r['output']) for r in records]
        references=[api['gsm8k_dataset_postprocess'](questions[r['id']]['answer']) for r in records]
        scores=evaluator.score(predictions,references)
        correct=[d['correct'] for d in scores['details']]
        summaries.append(dict(method=method,questions=len(records),
            opencompass_accuracy_percent=scores['accuracy'],
            opencompass_correct=sum(correct),
            opencompass_completed_accuracy_percent=100*sum(c and r['complete'] for c,r in zip(correct,records))/len(records),
            legacy_strict_accuracy_percent=100*sum(r['correct'] for r in records)/len(records),
            legacy_lenient_accuracy_percent=100*sum(r['lenient_correct'] for r in records)/len(records),
            disagreements_with_lenient=sum(c!=r['lenient_correct'] for c,r in zip(correct,records)),
            complete=sum(r['complete'] for r in records),
            base_weight=manifest['policy']['latent'].get('base_weight',1),
            bonus_weight=manifest['policy']['latent'].get('bonus_weight',1)))
        for r,d in zip(records,scores['details']):
            details.append(dict(id=r['id'],method=method,opencompass_prediction=d['pred'],
                reference=d['answer'],opencompass_correct=d['correct'],complete=r['complete'],
                legacy_strict_correct=r['correct'],legacy_lenient_correct=r['lenient_correct'],output=r['output']))
    return summaries,details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True)
    ap.add_argument('--methods',nargs='+')
    ap.add_argument('--out',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.out)
    if out.exists():raise ValueError('Choose a fresh output directory')
    summaries,details=score_run(run,args.methods)
    out.mkdir(parents=True)
    for filename,rows in [('summary.csv',summaries),('answers.csv',details),
                          ('disagreements.csv',[r for r in details if r['opencompass_correct']!=r['legacy_lenient_correct']])]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list((rows or details)[0]));writer.writeheader();writer.writerows(rows)
    hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('generations.jsonl','questions.jsonl','manifest.json')}
    (out/'scoring_manifest.json').write_text(json.dumps(dict(upstream=PROVENANCE,input_run=str(run),
        input_sha256=hashes,methods=args.methods,adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notes='Pinned standard GSM8K postprocessor + evaluator only. Not a verified CreditDecoding config or full OpenCompass pipeline. Accuracy is percent. Raw upstream score does not gate incomplete outputs; completed score does.'),indent=2))
    for row in summaries:print(json.dumps(row))
    print(f'Saved {out}/summary.csv and disagreements.csv; original results unchanged.')


if __name__=='__main__':main()
