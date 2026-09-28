"""GPU-only collection using the unchanged, pinned upstream generate function."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from .features import OutputHistory, LatentHistory, softmax, label_rows
from .prompt_format import render_prompt
from .policy_observer import PolicyHistory
from .decoding import LatentSettings
from .parity import select_parity_indices


class Observer:
    def __init__(self, model, config, prompt_length, metadata):
        self.model, self.cfg, self.prompt_length, self.metadata = model, config, prompt_length, metadata
        self.rows, self.handles, self.previous_rows = [], [], []
        self.step, self.block = 0, -1
        self.captured = {}
        self.policy_stats = {}
        self.policy_settings=LatentSettings(**config.get('latent_policy',{}))

    def mark_commits(self, x):
        for r in self.previous_rows:
            r['committed_this_step'] = bool(x[0, self.prompt_length+r['position']].item() != self.cfg['mask_id'])

    def before(self, module, args, kwargs):
        x = args[0] if args else kwargs['input_ids']
        self.mark_commits(x)
        steps_per_block = self.cfg['steps']//(self.cfg['gen_length']//self.cfg['block_length'])
        block = self.step//steps_per_block
        if block != self.block:
            self.block = block
            self.output = None
            self.latent = {}
            self.policy = {}
        self.start = self.prompt_length+block*self.cfg['block_length']
        self.stop = self.start+self.cfg['block_length']
        self.active = (x[0, self.start:self.stop] == self.cfg['mask_id']).cpu().numpy()
        self.captured.clear()
        self.policy_stats.clear()

    def capture(self, name):
        def hook(module, args, result):
            h = result[0] if isinstance(result, tuple) else result
            h=h[0,self.start:self.stop].detach().float()
            # Keep selected states on the model device until logits are available.
            self.captured[name] = h.clone()
        return hook

    def after(self, module, args, result):
        device_logits = result.logits[0, self.start:self.stop].detach().float()
        logits = device_logits.cpu().numpy()
        p = softmax(logits)
        if self.output is None:
            self.output = OutputHistory(*p.shape)
        tokens, feats, histories = self.output.update(p, self.active, logits.argmax(axis=1))
        for name, device_h in self.captured.items():
            if name not in self.policy:
                self.policy[name]=PolicyHistory(self.policy_settings,self.cfg.get('policy_threshold',.95))
            self.policy_stats[name]=self.policy[name].update(device_h,self.active,device_logits)
            h=device_h.cpu().numpy().copy()
            stats = self.latent.setdefault(name, LatentHistory()).update(h)
            stats.update(self.policy_stats[name])
            feats.update({f'lat_{name}__{k}':v for k,v in stats.items()})
        steps_per_block = self.cfg['steps']//(self.cfg['gen_length']//self.cfg['block_length'])
        self.previous_rows = []
        for i in np.flatnonzero(self.active):
            pos = self.block*self.cfg['block_length']+int(i)
            r = dict(self.metadata, step=self.step, block=self.block, position=pos, top1=int(tokens[i]),
                     token_history=histories[i], committed_this_step=False,
                     ctx_progress=self.step/self.cfg['steps'],
                     ctx_block_progress=(self.step % steps_per_block)/steps_per_block,
                     ctx_mask_ratio=float(np.mean(self.active)), ctx_position=pos/self.cfg['gen_length'])
            r.update({k:(int(v[i]) if k.endswith('_top1') else float(v[i])) for k,v in feats.items()})
            self.rows.append(r); self.previous_rows.append(r)
        self.step += 1
        # Returning None is essential: do not replace or mutate the forward output.

    def __enter__(self):
        tr = self.model.model.transformer
        if len(tr.blocks) != 32:
            raise ValueError('Expected pinned 32-layer LLaDA architecture')
        self.handles.append(self.model.register_forward_pre_hook(self.before, with_kwargs=True))
        for layer in self.cfg['layers']:
            self.handles.append(tr.blocks[layer-1].register_forward_hook(self.capture(f'layer{layer:02d}')))
        if self.cfg['final_norm']:
            self.handles.append(tr.ln_f.register_forward_hook(self.capture('final_norm')))
        self.handles.append(self.model.register_forward_hook(self.after))
        return self

    def finish(self, x, eos_ids):
        self.mark_commits(x)
        if self.step != self.cfg['steps']:
            raise ValueError('Unexpected upstream forward count')
        final = x[0, self.prompt_length:].cpu().tolist()
        if self.cfg['mask_id'] in final:
            raise ValueError('Baseline left mask tokens: trajectory invalid; no silent repair')
        return label_rows(self.rows, final, eos_ids), final

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


def validate_config(cfg):
    length, block, steps = (cfg[k] for k in ('gen_length', 'block_length', 'steps'))
    if min(length, block, steps) <= 0 or length % block or steps % (length//block) or steps > length:
        raise ValueError('Require positive length divisible by block, steps divisible by block count, steps <= length')
    if any(l < 1 or l > 32 for l in cfg['layers']):
        raise ValueError('Layers must be 1-based within [1,32]')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/baseline.json')
    ap.add_argument('--prompts', default='data/smoke.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--gen-length', type=int); ap.add_argument('--steps', type=int)
    ap.add_argument('--block-length', help='Positive output block size, or full for the entire output span')
    ap.add_argument('--layers', type=int, nargs='+', help='Explicit 1-based diagnostic layers; default from config')
    ap.add_argument('--verify-parity', action='store_true', help='Replay length-stratified prompts without hooks; require identical output')
    ap.add_argument('--parity-prompts',type=int,default=20)
    ap.add_argument('--policy-config',default='configs/rollout_policy.json')
    args = ap.parse_args()
    import torch
    # Guard before importing/loading HF or contacting model hub.
    if not torch.cuda.is_available():
        raise SystemExit('Real collection requires a CUDA GPU. Use pytest/synthetic for local CPU checks.')
    from transformers import AutoModel, AutoTokenizer
    from .third_party.llada_generate import generate
    cfg = json.loads(Path(args.config).read_text())
    policy_config=json.loads(Path(args.policy_config).read_text())
    cfg['latent_policy']=policy_config['latent']
    cfg['policy_threshold']=policy_config['threshold']
    LatentSettings(**cfg['latent_policy'])
    if args.parity_prompts<1: raise ValueError('Parity prompt count must be positive')
    for key in ('gen_length','steps'):
        if getattr(args,key) is not None:
            cfg[key] = getattr(args,key)
    block = args.block_length if args.block_length is not None else cfg['block_length']
    cfg['block_length'] = cfg['gen_length'] if block == 'full' else int(block)
    if args.layers is not None:
        cfg['layers'] = args.layers
    validate_config(cfg)
    prompts = [json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    prompts = prompts[:args.limit]
    if not prompts:
        raise ValueError('No prompts')
    for p in prompts:
        for key in ('id','text','task'):
            if not isinstance(p.get(key), str) or not p[key]:
                raise ValueError(f'Prompt requires nonempty {key}')
    if len({p['id'] for p in prompts}) != len(prompts):
        raise ValueError('Duplicate generation IDs')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=cfg, prompts_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
                    limit=args.limit, synthetic=False, python=platform.python_version(),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    manifest['source_sha256'] = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.rglob('*.py'))}
    manifest['packages'] = subprocess.check_output([os.sys.executable,'-m','pip','freeze'], text=True).splitlines()
    existing = out/'manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError('Output manifest differs. Use a new directory to avoid mixing experiments.')
    existing.write_text(json.dumps(manifest, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], trust_remote_code=True)
    lengths=[len(tokenizer(render_prompt(p['text'],cfg,tokenizer),add_special_tokens=False).input_ids) for p in prompts]
    parity_indices=set(select_parity_indices(lengths,args.parity_prompts)) if args.verify_parity else set()
    parity_path=out/'parity.json'
    parity_records=json.loads(parity_path.read_text()).get('checks',[]) if parity_path.exists() else []
    model = AutoModel.from_pretrained(cfg['model_id'], revision=cfg['model_revision'], trust_remote_code=True,
                                     torch_dtype=getattr(torch,cfg['dtype']), low_cpu_mem_usage=True).to('cuda').eval()
    kwargs = dict(steps=cfg['steps'], gen_length=cfg['gen_length'], block_length=cfg['block_length'],
                  temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=cfg['mask_id'])
    for j, prompt in enumerate(prompts):
        file_id = hashlib.sha256(prompt['id'].encode()).hexdigest()[:20]
        dest = out/f'{file_id}.parquet'
        needs_parity = j in parity_indices and prompt['id'] not in {p['generation_id'] for p in parity_records}
        if dest.exists() and (out/f'{file_id}.json').exists() and not needs_parity:
            continue
        text = render_prompt(prompt['text'],cfg,tokenizer)
        ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
        if ids.shape[1] > cfg['max_prompt_tokens']:
            raise ValueError(f"Prompt {prompt['id']} exceeds token cap; curate it explicitly, no silent truncation")
        if (ids == cfg['mask_id']).any():
            raise ValueError('Prompt contains mask token')
        torch.manual_seed(cfg['seed']); torch.cuda.reset_peak_memory_stats()
        start = time.time()
        # Group by normalized prompt text, even if a dataset duplicates it under other IDs.
        group = hashlib.sha256(' '.join(prompt['text'].split()).encode()).hexdigest()
        with torch.inference_mode(), Observer(model, cfg, ids.shape[1],
                dict(generation_id=prompt['id'], prompt_group=group, task=prompt['task'])) as obs:
            result = generate(model, ids, **kwargs)
            rows, final = obs.finish(result, [tokenizer.eos_token_id, 126348])
        if needs_parity:
            torch.manual_seed(cfg['seed'])
            with torch.inference_mode():
                plain = generate(model, ids, **kwargs)
            if not torch.equal(plain, result):
                raise AssertionError('Observed/unobserved baseline mismatch')
            parity_records.append(dict(generation_id=prompt['id'],prompt_tokens=lengths[j],identical=True))
            parity_path.write_text(json.dumps(dict(requested=args.parity_prompts,selected=len(parity_indices),checks=parity_records),indent=2))
        df = pd.DataFrame(rows)
        for col in df.select_dtypes(include=['float64']):
            df[col]=df[col].astype('float32')
        df['top1_text'] = [tokenizer.decode([int(t)]) for t in df.top1]
        df['final_text'] = [tokenizer.decode([int(t)]) for t in df.final_token]
        tmp = dest.with_suffix('.parquet.tmp'); df.to_parquet(tmp, index=False); tmp.replace(dest)
        details = dict(prompt=prompt, final_tokens=final, output=tokenizer.decode(final),
                       formatted_prompt=text,seconds=time.time()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(), rows=len(rows))
        (out/f'{file_id}.json').write_text(json.dumps(details, indent=2))
        print(f'{j+1}/{len(prompts)} {prompt["id"]}: {len(rows)} rows, {details["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
