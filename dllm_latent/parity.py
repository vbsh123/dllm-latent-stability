"""Deterministic prompt-length coverage for exact baseline parity checks."""
import numpy as np


def select_parity_indices(lengths,count):
    if count<1: raise ValueError('Parity prompt count must be positive')
    if not lengths: return []
    ordered=sorted(range(len(lengths)),key=lambda i:(lengths[i],i))
    ranks=np.linspace(0,len(ordered)-1,min(count,len(ordered)),dtype=int)
    return [ordered[int(rank)] for rank in ranks]
