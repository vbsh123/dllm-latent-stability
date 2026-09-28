"""Prepare a reproducible, balanced prompt snapshot; no reference answers used."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .prompt_format import render_prompt

SOURCES = {
    'dolly': ('databricks/databricks-dolly-15k','bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a'),
    'gsm8k': ('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count',type=int,default=100); ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--out',required=True); ap.add_argument('--config',default='configs/baseline.json')
    args = ap.parse_args()
    if args.count<3: raise ValueError('Use >=3 prompts for a balanced snapshot')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    cfg = json.loads(Path(args.config).read_text())
    tok = AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],trust_remote_code=True)
    dolly = load_dataset(SOURCES['dolly'][0], revision=SOURCES['dolly'][1], split='train')
    math = load_dataset(SOURCES['gsm8k'][0], 'main', revision=SOURCES['gsm8k'][1], split='test')
    pools = {'general':[], 'math':[], 'longform':[]}
    for i,r in enumerate(dolly):
        task = {'open_qa':'general','creative_writing':'longform'}.get(r['category'])
        if not task: continue
        text = r['instruction'] + ('\n\n'+r['context'] if r['context'] else '')
        pools[task].append(dict(id=f'dolly-{i}',text=text,task=task,source=SOURCES['dolly'][0],source_row=i))
    for i,r in enumerate(math):
        pools['math'].append(dict(id=f'gsm8k-test-{i}',text=r['question'],task='math',source=SOURCES['gsm8k'][0],source_row=i))
    rng = np.random.default_rng(args.seed); chosen = []; seen = set(); excluded = {}
    # Round-robin after shuffling makes 100-prompt snapshot a prefix of the 1000-prompt snapshot.
    for task,pool in pools.items(): rng.shuffle(pool)
    cursors = {task:0 for task in pools}
    while len(chosen)<args.count:
        task = list(pools)[len(chosen)%3]; pool = pools[task]
        while True:
            if cursors[task]>=len(pool): raise ValueError(f'Insufficient eligible {task} prompts')
            p = pool[cursors[task]]; cursors[task]+=1
            normalized = ' '.join(p['text'].split())
            text = render_prompt(p['text'],cfg,tok)
            ids = tok(text,add_special_tokens=False).input_ids
            if normalized in seen or len(ids)>cfg['max_prompt_tokens'] or cfg['mask_id'] in ids:
                excluded[task] = excluded.get(task,0)+1; continue
            seen.add(normalized); chosen.append(p); break
    dest = Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise ValueError('Refusing to overwrite prompt snapshot')
    dest.write_text(''.join(json.dumps(p)+'\n' for p in chosen))
    dest.with_suffix('.manifest.json').write_text(json.dumps(dict(sources=SOURCES,count=args.count,seed=args.seed,
                  tokenizer_revision=cfg['model_revision'],prompt_format=cfg['prompt_format'],excluded=excluded,
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))


if __name__ == '__main__': main()
