"""Training-free causal checks. No fitted parameters, labels, or future observations."""
from dataclasses import dataclass
import numpy as np
from .features import LatentHistory


@dataclass(frozen=True)
class StabilityRule:
    window: int = 3
    cosine_threshold: float = .99

    def __post_init__(self):
        if self.window not in (2,3,4):
            raise ValueError('window must be 2, 3, or 4 transitions')
        if not -1 <= self.cosine_threshold <= 1:
            raise ValueError('cosine threshold must lie in [-1,1]')

    def evaluate(self, statistics):
        score = np.asarray(statistics[f'w{self.window}_cos_min'])
        return np.isfinite(score) & (score >= self.cosine_threshold)


class StabilityChecker:
    """Track fixed token positions; reset this instance for each generation.

    True means the minimum cosine across the last K transitions passes a fixed
    threshold. It does NOT certify semantic correctness or safe early insertion.
    """
    def __init__(self, rule=None):
        self.rule = rule or StabilityRule()
        self.history = LatentHistory()

    def update(self, hidden):
        statistics = self.history.update(hidden)
        return self.rule.evaluate(statistics), statistics


@dataclass(frozen=True)
class RegionRule:
    """Optional region check; caller must supply all three distance tolerances.

    This does not select thresholds or modify decoding. Raw geometry distances
    are relative to the RMS norm of prior vectors; unit geometry uses unit vectors.
    """
    max_distance: float
    max_radius: float
    max_drift: float
    window: int = 3
    geometry: str = 'raw'

    def __post_init__(self):
        if self.window not in (2,3,4) or self.geometry not in ('raw','unit'):
            raise ValueError('Require window 2/3/4 and geometry raw/unit')
        if any(not np.isfinite(x) or x<0 for x in (self.max_distance,self.max_radius,self.max_drift)):
            raise ValueError('Region tolerances must be finite and nonnegative')

    def evaluate(self,statistics):
        prefix=f'w{self.window}_{self.geometry}_'
        comparisons=[]
        for metric,threshold in [('center_distance',self.max_distance),('radius',self.max_radius),('center_drift',self.max_drift)]:
            score=np.asarray(statistics[prefix+metric])
            comparisons.append(np.isfinite(score) & (score<=threshold))
        return np.logical_and.reduce(comparisons)


@dataclass(frozen=True)
class AnchoredRegionRule:
    """Closeness to a frozen center over multiple confirmation steps.

    One scalar tolerance, r, controls max point distance r, reference radius r/2,
    and cumulative rolling-center displacement r/4, all relative to frozen scale.
    These fixed ratios reduce the tuning search; they are explicit assumptions.
    """
    radius: float
    window: int = 3
    confirmations: int = 2

    def __post_init__(self):
        if self.window not in (2,3,4) or self.confirmations not in (2,3):
            raise ValueError('Require window 2/3/4 and confirmations 2/3')
        if not np.isfinite(self.radius) or self.radius<=0:
            raise ValueError('radius must be finite and positive')

    def evaluate(self,statistics):
        prefix=f'a{self.window}_c{self.confirmations}_'
        checks=[]
        for metric,limit in [('distance',self.radius),('radius',self.radius/2),('drift',self.radius/4)]:
            value=np.asarray(statistics[prefix+metric])
            checks.append(np.isfinite(value) & (value<=limit))
        return np.logical_and.reduce(checks)


def episode_triggers(statistics, positions, steps, rule):
    """First trigger from consecutive candidate episodes, one sorted generation.

    Build K reference observations, freeze the anchor, then check each confirmation
    immediately. Any failed cloud/distance/drift check discards the episode. The
    failing point seeds the next reference window; none of its predecessors survive.
    Prefix statistics reconstruct the SAME anchor throughout confirmation without
    retaining separate raw vectors for every radius. No labels or token IDs enter.
    """
    passes=[]
    for c in range(rule.confirmations+1):
        prefix=f'a{rule.window}_c{c}_'
        checks=[]
        for metric,limit in [('distance',rule.radius),('radius',rule.radius/2),('drift',rule.radius/4)]:
            value=np.asarray(statistics[prefix+metric])
            checks.append(np.isfinite(value) & (value<=limit))
        passes.append(np.logical_and.reduce(checks))
    result=np.zeros(len(positions),dtype=bool)
    previous_position=None; previous_step=None; seen=0; detected=False
    for i,(position,step) in enumerate(zip(positions,steps)):
        if position!=previous_position or previous_step is None or step!=previous_step+1:
            seen=0; detected=False
        previous_position=position; previous_step=step
        if detected: continue  # Never erase an already recorded proposal.
        seen+=1
        if seen<rule.window: continue
        confirmation=seen-rule.window
        if not passes[confirmation][i]:
            seen=1  # Start a fresh reference window with the failing current point.
        elif confirmation==rule.confirmations:
            result[i]=True; detected=True
    return result
