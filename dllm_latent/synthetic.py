"""CPU synthetic integration fixture; deliberately NOT evidence for the hypothesis."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import OutputHistory, LatentHistory, label_rows


def generate_rows(seed, generation):
    rng = np.random.default_rng(seed); positions, vocab, steps = 12, 7, 12
    output = OutputHistory(positions,vocab)
    latent = {f'layer{n:02d}':LatentHistory() for n in [8,16,24,32]}
    latent['final_norm'] = LatentHistory()
    hidden = {name:rng.normal(size=(positions,16)).astype(np.float32) for name in latent}
    final = rng.integers(0,vocab,positions); commits = rng.permutation(np.arange(steps))
    rows = []
    for t in range(steps):
        p = rng.dirichlet(np.ones(vocab),size=positions).astype(np.float32)
        for i in range(positions):
            if rng.random()<.35+.6*t/steps or t>=commits[i]:
                p[i,final[i]] += rng.uniform(.3,1.5)
        p /= p.sum(1,keepdims=True)
        active = commits>=t
        tokens,features,histories = output.update(p,active)
        # Final baseline token must equal top1 at commit.
        for i in np.flatnonzero(commits==t): final[i] = tokens[i]
        for name,tracker in latent.items():
            hidden[name] += rng.normal(size=hidden[name].shape).astype(np.float32)*(.25+1-t/steps)
            features.update({f'lat_{name}__{k}':v for k,v in tracker.update(hidden[name]).items()})
        for i in np.flatnonzero(active):
            row = dict(generation_id=f'synthetic-{generation}',prompt_group=f'group-{generation}',task=['general','math','longform'][generation%3],
                       position=int(i),step=t,block=0,top1=int(tokens[i]),top1_text=f'token{tokens[i]}',token_history=histories[i],
                       committed_this_step=bool(commits[i]==t),ctx_progress=t/steps,ctx_block_progress=t/steps,
                       ctx_mask_ratio=float(active.mean()),ctx_position=i/positions)
            row.update({k:float(v[i]) for k,v in features.items()}); rows.append(row)
    rows = label_rows(rows,final)
    for r in rows: r['final_text'] = f'token{r["final_token"]}'
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True); ap.add_argument('--prompts',type=int,default=60)
    args = ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Use empty synthetic run directory')
    (out/'manifest.json').write_text(json.dumps(dict(synthetic=True,seed=1729,prompts=args.prompts)))
    for i in range(args.prompts): pd.DataFrame(generate_rows(1729+i,i)).to_parquet(out/f'{i:04d}.parquet',index=False)
    print(f'SYNTHETIC ONLY: wrote {args.prompts} CPU fixture trajectories')


if __name__ == '__main__': main()
