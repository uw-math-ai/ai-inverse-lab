"""Differentiable forward map for vertex-centrality reconstruction (Gao, Li & Yang, arXiv:2603.20710).

Known:   W (all edge weights), mu on B, potential q, step h, observed set B, horizon T.
Unknown: mu on U = X \\ B.

We work with a_x = 1 / mu_x, because the fptsim kernel is affine in a:
    K[x, y] = h W[x, y] a_x            (x != y)
    K[x, x] = 1 - h (a_x wdeg_x + q_x)
so dK/da_x is nonzero only in row x:  h W[x, :] off the diagonal and -h wdeg_x on it.

The first-passage recursion is the one in fptsim.exact (restricted to start vertices in X):
    r(1, ., y) = K[:, y];   r(t, ., y) = K @ (r(t-1, ., y) with entry y zeroed)
and its Jacobian is propagated in forward mode alongside it, exactly (no finite differences).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InverseProblem:
    """Everything a solver is allowed to see except the measurements."""
    W: np.ndarray                 # (n, n) symmetric weights, zero diagonal
    mu_known: np.ndarray          # (n,) mu on B; entries on U are NaN
    B: np.ndarray                 # (k,) observed vertices
    q: np.ndarray                 # (n,) potential (known)
    dt: float                     # step size h used to generate the data
    T: int                        # number of first-passage time bins
    U: np.ndarray = field(init=False)   # (m,) unknown vertices, sorted

    def __post_init__(self):
        self.W = np.asarray(self.W, dtype=float)
        self.B = np.asarray(self.B, dtype=np.int64)
        self.q = np.asarray(self.q, dtype=float)
        self.mu_known = np.asarray(self.mu_known, dtype=float)
        inB = np.zeros(self.n, dtype=bool)
        inB[self.B] = True
        self.U = np.flatnonzero(~inB).astype(np.int64)
        self.mu_known = np.where(inB, self.mu_known, np.nan)

    @property
    def n(self) -> int:
        return int(self.W.shape[0])

    @property
    def k(self) -> int:
        return int(self.B.size)

    @property
    def m(self) -> int:
        return int(self.U.size)

    @property
    def wdeg(self) -> np.ndarray:
        return self.W.sum(axis=1)

    def a_upper(self) -> np.ndarray:
        """Largest a_x on U that keeps the holding probability K[x, x] >= 0."""
        wd = self.wdeg[self.U]
        return (1.0 - self.dt * self.q[self.U]) / (self.dt * wd)

    def mu_floor(self) -> np.ndarray:
        """Smallest admissible mu on U (inverse of a_upper)."""
        return 1.0 / self.a_upper()

    def full_mu(self, mu_U: np.ndarray) -> np.ndarray:
        mu = self.mu_known.copy()
        mu[self.U] = mu_U
        return mu

    def kernel(self, mu: np.ndarray) -> np.ndarray:
        """K (n, n) for a full mu vector, with the same conventions as fptsim.kernel."""
        h = self.dt
        K = h * self.W / mu[:, None]
        np.fill_diagonal(K, 1.0 - h * (self.wdeg / mu + self.q))
        return K

    def dK_rows(self) -> np.ndarray:
        """D (m, n): row p is dK[U_p, :] / d a_{U_p}."""
        D = self.dt * self.W[self.U].copy()
        D[np.arange(self.m), self.U] = -self.dt * self.wdeg[self.U]
        return D


def first_passage(K: np.ndarray, B: np.ndarray, T: int) -> np.ndarray:
    """r (T, k, k): r[t-1, i, j] = P(first passage from B[i] to B[j] happens at step t)."""
    k = len(B)
    jj = np.arange(k)
    prev = K[:, B].copy()
    r = np.empty((T, k, k))
    r[0] = prev[B]
    for t in range(1, T):
        prev[B, jj] = 0.0
        prev = K @ prev
        r[t] = prev[B]
    return r


def first_passage_jac(prob: InverseProblem, mu: np.ndarray, T: int | None = None):
    """r (T, k, k) and J (T, k, k, m) = d r / d a_U at the full vector mu."""
    T = prob.T if T is None else T
    K = prob.kernel(mu)
    B, U, m, k = prob.B, prob.U, prob.m, prob.k
    jj, pp = np.arange(k), np.arange(m)
    D = prob.dK_rows()
    prev = K[:, B].copy()                         # (n, k)
    dprev = np.zeros((m, prob.n, k))
    dprev[pp, U, :] = D[:, B]
    r = np.empty((T, k, k))
    J = np.empty((T, m, k, k))
    r[0], J[0] = prev[B], dprev[:, B, :]
    for t in range(1, T):
        prev[B, jj] = 0.0
        dprev[:, B, jj] = 0.0
        G = D @ prev                              # (m, k): (dK_p @ prev) lives in row U_p
        dprev = np.matmul(K, dprev)
        dprev[pp, U, :] += G
        prev = K @ prev
        r[t], J[t] = prev[B], dprev[:, B, :]
    return r, np.moveaxis(J, 1, -1)


def fisher_information(prob: InverseProblem, mu: np.ndarray, N: int) -> np.ndarray:
    """Multinomial Fisher information for a_U from N walks per ordered pair.

    Per pair the bins are the T hit times plus one "no hit by T" bin (killed + censored),
    whose probability is 1 - sum_t r. F = N sum_bins (dp dp^T) / p.
    """
    r, J = first_passage_jac(prob, mu)
    T, k, _, m = J.shape
    p = r.reshape(T, k * k)
    dp = J.reshape(T, k * k, m)
    rest = np.clip(1.0 - p.sum(axis=0), 0.0, 1.0)
    drest = -dp.sum(axis=0)
    P = np.concatenate([p, rest[None]], axis=0)            # (T+1, k*k)
    dP = np.concatenate([dp, drest[None]], axis=0)         # (T+1, k*k, m)
    live = P > 1e-15
    w = np.where(live, 1.0 / np.where(live, P, 1.0), 0.0)
    return N * np.einsum("bp,bpi,bpj->ij", w, dP, dP)


def crb_relative_error(prob: InverseProblem, mu: np.ndarray, N: int) -> float:
    """Cramér–Rao lower bound on ||mu_hat - mu||_2 / ||mu||_2 over U, for an unbiased estimator.

    var(a) >= F^{-1}; mu = 1/a, so var(mu_x) ~ mu_x^4 var(a_x) (delta method). inf if F is singular.
    """
    if prob.m == 0:
        return 0.0
    F = fisher_information(prob, mu, N)
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return float("inf")
    var_a = np.diag(cov)
    if not np.all(np.isfinite(var_a)) or np.any(var_a < 0):
        return float("inf")
    mu_U = mu[prob.U]
    return float(np.sqrt(np.sum(mu_U ** 4 * var_a)) / np.linalg.norm(mu_U))


def jacobian_condition(prob: InverseProblem, mu: np.ndarray) -> float:
    """Condition number of the column-scaled Jacobian (scale-free local identifiability check)."""
    if prob.m == 0:
        return 1.0
    _, J = first_passage_jac(prob, mu)
    Jm = J.reshape(-1, prob.m) * (1.0 / mu[prob.U])[None, :]   # d r / d log a
    s = np.linalg.svd(Jm, compute_uv=False)
    return float("inf") if s[-1] <= 0 else float(s[0] / s[-1])
