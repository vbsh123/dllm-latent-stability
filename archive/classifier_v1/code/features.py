"""Causal online statistics; no labels or final tokens enter this module."""
from collections import deque
import numpy as np


def softmax(logits):
    z = np.asarray(logits, dtype=np.float32)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TraceCredit:
    """ACL 2026 Eq. 6/7. Dense exact vocabulary state, reset per block.

    Update precedes fusion. Masked positions only receive boosts; all decay.
    Adaptive coefficients use CURRENT block mask ratio (explicit interpretation).
    """
    def __init__(self, shape, adaptive=False, alpha=.65, beta=.7, gamma=.2):
        self.credit = np.zeros(shape, dtype=np.float32)
        self.adaptive, self.alpha, self.beta, self.gamma = adaptive, alpha, beta, gamma

    def update(self, p, active, top1=None):
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        if self.adaptive:
            alpha = beta = 1 - np.mean(active)
            gamma = 1.
        token = p.argmax(axis=1) if top1 is None else top1
        self.credit *= beta
        idx = np.flatnonzero(active)
        self.credit[idx, token[idx]] += p[idx, token[idx]] ** gamma
        q = p * (1 + self.credit) ** alpha
        q /= q.sum(axis=1, keepdims=True)
        return q, self.credit[np.arange(len(p)), token].copy()


class OutputHistory:
    def __init__(self, positions, vocabulary):
        shape = (positions, vocabulary)
        self.n = 0
        self.mean = np.zeros(shape, np.float32)
        self.m2 = np.zeros(shape, np.float32)
        self.ema = np.zeros(shape, np.float32)
        self.counts = np.zeros(shape, np.int32)
        self.previous = None
        self.streak = np.zeros(positions, np.int32)
        self.changes = np.zeros(positions, np.int32)
        self.probs = deque(maxlen=4)
        self.tokens = deque(maxlen=4)
        self.credits = {"credit": TraceCredit(shape), "adaptive_credit": TraceCredit(shape, adaptive=True)}

    def update(self, p, active, top1=None):
        idx = np.arange(len(p)); token = p.argmax(axis=1) if top1 is None else top1
        top = np.partition(p, -2, axis=1)[:, -2:]
        p1, p2 = top.max(axis=1), top.min(axis=1)
        changed = np.zeros(len(p), bool) if self.previous is None else token != self.previous
        fraction = self.counts[idx, token] / self.n if self.n else np.full(len(p), np.nan)
        self.streak = np.where(changed, 1, self.streak + 1)
        self.changes += changed
        out = dict(p1=p1, p2=p2, margin=p1-p2,
                   entropy=-(p*np.log(np.maximum(p, 1e-30))).sum(axis=1),
                   out_changed=changed.astype(float), out_streak=self.streak.copy(),
                   out_changes=self.changes.copy(), out_candidate_top1_fraction=fraction,
                   out_history_count=np.full(len(p), self.n))
        if self.probs:
            prev = self.probs[-1]; mix = .5 * (prev + p)
            out['out_js'] = .5 * ((p*np.log(np.maximum(p, 1e-30)/np.maximum(mix, 1e-30))).sum(1) +
                                     (prev*np.log(np.maximum(prev, 1e-30)/np.maximum(mix, 1e-30))).sum(1))
        else:
            out['out_js'] = np.full(len(p), np.nan)
        for lag in range(1, 5):
            out[f'out_candidate_p_lag{lag}'] = self.probs[-lag][idx, token] if len(self.probs) >= lag else np.full(len(p), np.nan)
        out['out_candidate_delta'] = p1-out['out_candidate_p_lag1']
        self.n += 1
        delta = p-self.mean; self.mean += delta/self.n; self.m2 += delta*(p-self.mean)
        self.ema = p.copy() if self.n == 1 else .7*self.ema+.3*p
        out.update(out_candidate_mean=self.mean[idx, token].copy(),
                   out_candidate_variance=self.m2[idx, token]/self.n,
                   out_candidate_ema=self.ema[idx, token].copy())
        for name, tracker in self.credits.items():
            q, c = tracker.update(p, active, token)
            out[f'out_{name}'] = c
            out[f'out_{name}_candidate_p'] = q[idx, token]
            out[f'out_{name}_max_p'] = q.max(1)
            out[f'out_{name}_agrees'] = (q.argmax(1) == token).astype(float)
        self.counts[idx, token] += 1
        histories = [','.join(str(int(v[i])) for v in self.tokens) for i in idx]
        self.probs.append(p.copy()); self.tokens.append(token.copy()); self.previous = token.copy()
        return token, out, histories


def cosine(a, b):
    denom = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.clip(np.divide((a*b).sum(-1), denom, out=np.full_like(denom, np.nan), where=denom>1e-12), -1, 1)


class LatentHistory:
    def __init__(self):
        self.hidden = deque(maxlen=2)
        self.sim = deque(maxlen=4)
        self.movement = deque(maxlen=4)
        self.total = None

    def update(self, hidden):
        h = np.asarray(hidden, dtype=np.float32)
        n = len(h); missing = np.full(n, np.nan)
        out = {"cos_lag1": missing.copy(), "cos_lag2": missing.copy(), "l2": missing.copy(),
               "relative_l2": missing.copy(), "norm": np.linalg.norm(h, axis=-1)}
        if self.total is None:
            self.total = np.zeros(n)
        if self.hidden:
            out['cos_lag1'] = cosine(h, self.hidden[-1])
            out['l2'] = np.linalg.norm(h-self.hidden[-1], axis=-1)
            out['relative_l2'] = out['l2']/np.maximum(np.linalg.norm(self.hidden[-1], axis=-1), 1e-12)
            self.sim.append(out['cos_lag1']); self.movement.append(out['l2']); self.total += out['l2']
        if len(self.hidden) >= 2:
            out['cos_lag2'] = cosine(h, self.hidden[-2])
        out['cumulative_l2'] = self.total.copy()
        for k in (2, 3, 4):
            vals = np.array(list(self.sim)[-k:])
            ready = len(vals) == k
            for name, fn in [('mean', np.mean), ('min', np.min), ('variance', np.var)]:
                out[f'w{k}_cos_{name}'] = fn(vals, axis=0) if ready else missing.copy()
            out[f'w{k}_l2_sum'] = np.sum(list(self.movement)[-k:], axis=0) if ready else missing.copy()
        self.hidden.append(h.copy())
        return out


def label_rows(rows, final_tokens, eos_ids=()):
    """Labels only. After commit, the decoder state is absorbing, not re-predicted.

    SAFE may become true then false then true; STABLE is true only on final suffix.
    """
    eos = next((i for i,v in enumerate(final_tokens) if v in eos_ids), len(final_tokens))
    by_position = {}
    for r in rows:
        by_position.setdefault(r['position'], []).append(r)
    for pos, seq in by_position.items():
        stable = True
        for r in reversed(seq):
            safe = int(r['top1'] == final_tokens[pos])
            stable = stable and bool(safe)
            r.update(safe_to_commit=safe, stable_to_end=int(stable), final_token=int(final_tokens[pos]),
                     before_final_eos=pos < eos,
                     steps_until_commit=seq[-1]['step']-r['step'])
    return rows


CURRENT = ['p1', 'p2', 'margin', 'entropy']
CONTEXT = ['ctx_progress', 'ctx_block_progress', 'ctx_mask_ratio', 'ctx_position']


def feature_groups(columns):
    """Allowlist by causal prefixes; metadata and labels cannot enter X."""
    output = CURRENT + [c for c in columns if c.startswith('out_')]
    latent = [c for c in columns if c.startswith('lat_')]
    groups = {'A_CURRENT_CONFIDENCE': CURRENT, 'B_OUTPUT_HISTORY': output,
              'C_LATENT_HISTORY': latent, 'D_OUTPUT_PLUS_LATENT': output+latent}
    for layer in sorted({c.split('__')[0] for c in latent}):
        cols = [c for c in latent if c.startswith(layer+'__')]
        groups['C_'+layer] = cols
        groups['D_'+layer] = output+cols
    return {k: CONTEXT+v for k,v in groups.items()}
