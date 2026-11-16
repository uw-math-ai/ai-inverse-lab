"""Path-wise Monte Carlo on the CPU with numpy.

Uses Kevin's stream layout: for each item seed and ordered pair, an independent
PCG64 stream from SeedSequence([seed, source, target]). Statistically equivalent to the
torch backend but not bit-identical, because the random streams differ.
"""
from __future__ import annotations

import time

import numpy as np

from .mc_common import MCItem, check_item_counts


class NumpyMC:
    name = "numpy"

    def __init__(self):
        self.device = "cpu"
        self.walk_steps = 0           # sum over steps of live walks (for throughput)

    def run_items(self, items: list[MCItem], stop=None):
        """Yield (index, counts (k, k, buckets, T+2), seconds) for each item in order."""
        for idx, item in enumerate(items):
            if stop is not None and stop():
                return
            started = time.perf_counter()
            counts = self.run_item(item)
            yield idx, counts, time.perf_counter() - started

    def run_item(self, item: MCItem) -> np.ndarray:
        T, k, D, cemetery = item.T, item.k, item.D, item.n
        nb, nbins = len(item.checkpoints), T + 2
        out = np.zeros((k, k, nb, nbins), dtype=np.int64)
        buckets = np.searchsorted(item.checkpoints, item.start + np.arange(item.m), side="right")
        for i, source in enumerate(item.B):
            for j, target in enumerate(item.B):
                rng = np.random.default_rng(np.random.SeedSequence([item.seed, int(source), int(target)]))
                states = np.full(item.m, source, dtype=np.int64)
                bk = buckets
                flat = np.zeros(nb * nbins, dtype=np.int64)
                for t in range(T):
                    if states.size == 0:
                        break
                    self.walk_steps += states.size
                    # each walk samples from its current vertex's categorical law
                    u = rng.random((states.size, 1))
                    col = np.minimum((u > item.cdf[states]).sum(axis=1), D - 1)
                    nxt = item.nbr[states, col]
                    hit = nxt == target
                    fin = hit | (nxt == cemetery)
                    if fin.any():
                        b = np.where(hit[fin], t, T)
                        flat += np.bincount(bk[fin] * nbins + b, minlength=nb * nbins)
                        keep = ~fin
                        states, bk = nxt[keep], bk[keep]
                    else:
                        states = nxt
                if states.size:
                    flat += np.bincount(bk * nbins + (T + 1), minlength=nb * nbins)
                out[i, j] = flat.reshape(nb, nbins)
        check_item_counts(item, out)
        return out
