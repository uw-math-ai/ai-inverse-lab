"""The explicit heat/killing kernel K, the augmented stochastic matrix P, and step-size rules.

    s(x)    = sum_y W[x, y] / mu(x)
    K[x, y] = h W[x, y] / mu(x)                (x != y)
    K[x, x] = 1 - h (s(x) + q(x))
    kill(x) = h q(x)                          (probability of moving to the cemetery)

P is (n+1) x (n+1) with the cemetery as the last, absorbing state. Rows are never
renormalised: an h that would make a holding probability negative is an error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class KernelError(ValueError):
    """The step size or the inputs do not give a valid stochastic kernel."""


@dataclass(frozen=True)
class Kernel:
    K: np.ndarray        # (n, n) sub-stochastic kernel on X
    kill: np.ndarray     # (n,) killing probabilities h q
    P: np.ndarray        # (n+1, n+1) row-stochastic, cemetery = index n
    dt: float
    dt_mode: str         # "auto" | "explicit"
    rate: np.ndarray     # s + q

    @property
    def n(self) -> int:
        return int(self.K.shape[0])


def resolve_dt(W: np.ndarray, mu: np.ndarray, q: np.ndarray, dt="auto") -> tuple[float, str]:
    """auto: h = min(1, 1 / max(s + q)). explicit: require 0 < h <= 1 and h max(s + q) <= 1."""
    rate = W.sum(axis=1) / mu + q
    top = float(rate.max()) if rate.size else 0.0
    if dt == "auto":
        return (1.0 if top <= 1.0 else 1.0 / top), "auto"
    h = float(dt)
    if not (0.0 < h <= 1.0):
        raise KernelError(f"dt must satisfy 0 < dt <= 1, got {h}")
    if h * top > 1.0 + 1e-12:
        raise KernelError(f"dt={h:.12g} is too large for this graph: need dt * max(s+q) <= 1, "
                          f"i.e. dt <= {1.0 / top:.12g}. Use dt: auto.")
    return h, "explicit"


def build_kernel(W: np.ndarray, mu: np.ndarray, q: np.ndarray | None = None, dt="auto") -> Kernel:
    W = np.asarray(W, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = W.shape[0]
    q = np.zeros(n) if q is None else np.asarray(q, dtype=float)
    if W.shape != (n, n) or mu.shape != (n,) or q.shape != (n,):
        raise KernelError(f"shape mismatch: W {W.shape}, mu {mu.shape}, q {q.shape}")
    if np.any(mu <= 0) or np.any(q < 0) or np.any(W < 0) or np.any(np.diag(W) != 0):
        raise KernelError("need mu > 0, q >= 0, W >= 0 with zero diagonal")
    h, mode = resolve_dt(W, mu, q, dt)
    rate = W.sum(axis=1) / mu + q
    K = h * W / mu[:, None]
    diagonal = 1.0 - h * rate
    if diagonal.min() < -1e-12:
        raise KernelError("potential and step size produce a negative holding probability")
    np.fill_diagonal(K, np.maximum(diagonal, 0.0))   # clips only round-off below zero
    kill = h * q
    P = np.zeros((n + 1, n + 1))
    P[:n, :n], P[:n, n], P[n, n] = K, kill, 1.0
    if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise KernelError("augmented transition matrix is not row-stochastic")
    return Kernel(K=K, kill=kill, P=P, dt=h, dt_mode=mode, rate=rate)


def transition_table(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse sampling table: for each state, its reachable states and their cumulative probs.

    Returns nbr (n+1, D) int64 and cdf (n+1, D) float64, D = max out-degree including the
    holding step and the cemetery. Rows are padded with their last state and cdf = 1, and the
    last real entry is set to exactly 1 so a uniform u in [0, 1) always lands on a real state.
    """
    n1 = P.shape[0]
    support = [np.flatnonzero(P[x] > 0) for x in range(n1)]
    D = max(len(cols) for cols in support)
    nbr = np.empty((n1, D), dtype=np.int64)
    cdf = np.ones((n1, D))
    for x, cols in enumerate(support):
        c = np.minimum(np.cumsum(P[x, cols]), 1.0)
        c[-1] = 1.0
        nbr[x, :len(cols)], nbr[x, len(cols):] = cols, cols[-1]
        cdf[x, :len(cols)] = c
    return nbr, cdf


def spectral_info(K: np.ndarray, mu: np.ndarray) -> dict:
    """Eigenvalues of K via the symmetric similarity D^{1/2} K D^{-1/2}, D = diag(mu).

    mu(x) K[x, y] = h W[x, y] is symmetric, so K is self-adjoint in l^2(mu) and has a real
    spectrum. spectral_gap = lambda_1 - lambda_2 (lambda_1 = 1 when q = 0).
    """
    root = np.sqrt(mu)
    S = root[:, None] * K / root[None, :]
    S = (S + S.T) / 2
    ev = np.linalg.eigvalsh(S)
    lam1 = float(ev[-1])
    lam2 = float(ev[-2]) if ev.size > 1 else float("nan")
    second = np.abs(ev[:-1]).max() if ev.size > 1 else float("nan")
    return {"lambda_max": lam1, "lambda_2": lam2, "lambda_min": float(ev[0]),
            "spectral_gap": lam1 - lam2, "abs_spectral_gap": float(lam1 - second)}
