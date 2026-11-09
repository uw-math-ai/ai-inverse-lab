"""Exact first-passage distributions, batched over graphs (torch float64 on GPU, numpy fallback).

Recursion (all start vertices, targets in B, restricted to B x B at the end):
    r(1, x, y) = K[x, y]
    r(t, x, y) = sum_{z != y} K[x, z] r(t-1, z, y)
Independent check: forward propagation of each start distribution with the target made
absorbing, which also gives the killed and censored masses. Both must agree to 1e-12 and
hits + killed + censored must equal 1.

Graphs in a batch are padded to a common n_max with zero rows and columns. Padded vertices
are unreachable, so they do not change any real entry.
"""
from __future__ import annotations

import numpy as np

AGREE_ATOL = 1e-12
MASS_ATOL = 1e-11


class _NumpyOps:
    def zeros(self, shape):
        return np.zeros(shape)

    def arange(self, n):
        return np.arange(n)

    def asarray(self, a):
        return np.asarray(a)

    def numpy(self, a):
        return np.asarray(a)


class _TorchOps:
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, torch.device(device)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.torch.float64, device=self.device)

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def asarray(self, a):
        return self.torch.as_tensor(a, device=self.device)

    def numpy(self, a):
        return a.cpu().numpy()


def _core(ops, K, kill, B, src, tgt, T):
    """K (G,n,n), kill (G,n), B (G,k) padded index, src/tgt (G,k*k) per ordered pair."""
    G, n, _ = K.shape
    k = B.shape[1]
    g = ops.arange(G)[:, None]
    jj = ops.arange(k)[None, :]
    pp = ops.arange(k * k)[None, :]

    # recursion: column j of `prev` is r(t, ., B[j]) over all start vertices
    r = ops.zeros((G, T, k, k))
    prev = K.swapaxes(1, 2)[g, B].swapaxes(1, 2)         # (G, n, k): K[:, :, B]
    r[:, 0] = prev[g, B]
    for t in range(1, T):
        prev[g, B, jj] = 0.0                             # exclude an earlier arrival at y
        prev = K @ prev
        r[:, t] = prev[g, B]

    # forward check: live mass of each (source, target) walk that has not hit or died
    live = ops.zeros((G, k * k, n))
    live[g, pp, src] = 1.0
    hits = ops.zeros((G, T, k * k))
    killed = ops.zeros((G, k * k))
    for t in range(T):
        killed = killed + (live * kill[:, None, :]).sum(-1)
        live = live @ K
        hits[:, t] = live[g, pp, tgt]
        live[g, pp, tgt] = 0.0
    censored = live.sum(-1)
    return r, hits.reshape(G, T, k, k), killed.reshape(G, k, k), censored.reshape(G, k, k)


def _group(kernels, Bs, T, ops) -> list[dict]:
    G = len(kernels)
    n = max(kern.n for kern in kernels)
    k = max(len(b) for b in Bs)
    K = np.zeros((G, n, n))
    kill = np.zeros((G, n))
    B = np.zeros((G, k), dtype=np.int64)
    for i, (kern, b) in enumerate(zip(kernels, Bs)):
        K[i, :kern.n, :kern.n] = kern.K
        kill[i, :kern.n] = kern.kill
        B[i, :len(b)] = b
    src = np.repeat(B, k, axis=1)                         # pair p = i*k + j: source B[i]
    tgt = np.tile(B, (1, k))                              #                   target B[j]
    out = _core(ops, ops.asarray(K), ops.asarray(kill), ops.asarray(B),
                ops.asarray(src), ops.asarray(tgt), T)
    r, hits, killed, censored = (ops.numpy(a) for a in out)
    results = []
    for i, b in enumerate(Bs):
        kb = len(b)
        ri, hi = r[i, :, :kb, :kb], hits[i, :, :kb, :kb]
        ki, ci = killed[i, :kb, :kb], censored[i, :kb, :kb]
        disagreement = float(np.abs(ri - hi).max())
        mass_error = float(np.abs(ri.sum(axis=0) + ki + ci - 1.0).max())
        if disagreement > AGREE_ATOL:
            raise ArithmeticError(f"exact recursion and forward check disagree by {disagreement:.3g}")
        if mass_error > MASS_ATOL:
            raise ArithmeticError(f"exact probability mass misses 1 by {mass_error:.3g}")
        results.append({"r": ri.copy(), "killed": ki.copy(), "censored": ci.copy(),
                        "max_disagreement": disagreement, "mass_error": mass_error})
    return results


def exact_batch(kernels, Bs, T: int, device=None, max_elements: float = 3e7) -> list[dict]:
    """Exact r (T, k, k), killed (k, k), censored (k, k) for each (kernel, B).

    device=None uses numpy; otherwise a torch device string. Graphs are sorted by size and
    split into groups whose padded K stays below `max_elements` entries.
    """
    if not kernels:
        return []
    ops = _NumpyOps() if device is None else _TorchOps(device)
    order = sorted(range(len(kernels)), key=lambda i: kernels[i].n)
    results: list[dict | None] = [None] * len(kernels)
    group: list[int] = []
    for idx in order + [None]:
        if idx is not None:
            n = kernels[idx].n
            if not group or (len(group) + 1) * n * n <= max_elements:
                group.append(idx)
                continue
        out = _group([kernels[i] for i in group], [np.asarray(Bs[i]) for i in group], T, ops)
        for i, res in zip(group, out):
            results[i] = res
        group = [idx] if idx is not None else []
    return results


def exact_single(kernel, B, T: int, device=None) -> dict:
    return exact_batch([kernel], [B], T, device)[0]
