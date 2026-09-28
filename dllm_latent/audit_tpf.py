"""Recompute TPF from saved generations, with no model load or inference."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .gsm8k_rollout import summarize_tpf


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',nargs='+',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('Use a new output file; do not overwrite prior reports')
    rows=[]
    for run in args.run:
        groups={}
        with (Path(run)/'generations.jsonl').open() as f:
            for line in f:
                r=json.loads(line);groups.setdefault(r['method'],[]).append(r)
        for method,records in groups.items():
            if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate question/method rows')
            rows.append(dict(run=run,method=method,questions=len(records),**summarize_tpf(records)))
    out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out,index=False)
    print(frame.to_string(index=False))
    print('Historical no-stop forwards are unchanged; this does not simulate early stopping.')


if __name__=='__main__':main()
