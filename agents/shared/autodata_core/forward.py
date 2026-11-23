"""Forward map r(mu) and its exact Jacobian, in pure Python.

Same mathematics as "Toggable Simulation of Data Code/autodata/forward.py":
unknowns a_x = 1/mu_x on U = X \\ B; K is affine in a, and dK/da_x lives in row x only
(h W[x, :] off the diagonal, -h wdeg_x on it). The recursion
    r(1, ., y) = K[:, y];   r(t, ., y) = K @ (r(t-1, ., y) with entry y zeroed)
is run on one wide matrix M = [prev | dprev_1 | ... | dprev_m] (n rows, k(m+1) columns), so the
value and all m directional derivatives come out of a single sparse product per step.
"""
from __future__ import annotations

import math

from .linalg import gram, spd_inverse_diag, sym_eigenvalues


class InverseProblem:
    """What a solver may see: W (via neighbour lists), mu on B, q, h, B, T."""

    def __init__(self, W, mu_known, B, q, dt, T):
        n = len(W)
        self.n, self.B, self.q, self.dt, self.T = n, list(B), list(q), float(dt), int(T)
        self.k = len(self.B)
        inB = set(self.B)
        self.U = [x for x in range(n) if x not in inB]
        self.m = len(self.U)
        self.nbrs = [[(y, W[x][y]) for y in range(n) if W[x][y] > 0] for x in range(n)]
        self.wdeg = [sum(w for _, w in self.nbrs[x]) for x in range(n)]
        self.mu_known = [mu_known[x] if x in inB else float("nan") for x in range(n)]
        self.bpos = {b: j for j, b in enumerate(self.B)}
        h = self.dt
        # dK rows for each unknown: [(column, derivative)]
        self.drows = [[(y, h * w) for y, w in self.nbrs[x]] + [(x, -h * self.wdeg[x])] for x in self.U]

    def a_upper(self):
        return [(1.0 - self.dt * self.q[x]) / (self.dt * self.wdeg[x]) for x in self.U]

    def mu_floor(self):
        return [1.0 / a for a in self.a_upper()]

    def full_mu(self, mu_U):
        mu = self.mu_known[:]
        for x, v in zip(self.U, mu_U):
            mu[x] = v
        return mu

    def kernel_rows(self, mu):
        h = self.dt
        rows = []
        for x in range(self.n):
            inv = h / mu[x]
            row = [(y, inv * w) for y, w in self.nbrs[x]]
            row.append((x, 1.0 - h * (self.wdeg[x] / mu[x] + self.q[x])))
            rows.append(row)
        return rows


def _step(rows, M, width):
    """new[x] = sum_y K[x, y] M[y] for wide rows of length `width`."""
    out = []
    for row in rows:
        (y0, v0), rest = row[0], row[1:]
        acc = [v0 * b for b in M[y0]]
        for y, v in rest:
            My = M[y]
            acc = [a + v * b for a, b in zip(acc, My)]
        out.append(acc)
    return out


def first_passage(prob: InverseProblem, mu, T=None):
    """r as a flat list ordered (t, i, j), length T k^2."""
    T = prob.T if T is None else T
    rows, B, k, bpos = prob.kernel_rows(mu), prob.B, prob.k, prob.bpos
    prev = [[0.0] * k for _ in range(prob.n)]
    for x, row in enumerate(rows):
        for y, v in row:
            j = bpos.get(y)
            if j is not None:
                prev[x][j] = v
    out = []
    for t in range(T):
        if t:
            for j, b in enumerate(B):
                prev[b][j] = 0.0
            prev = _step(rows, prev, k)
        for b in B:
            out.extend(prev[b])
    return out


def first_passage_jac(prob: InverseProblem, mu, T=None):
    """(r flat (t,i,j), Jacobian columns: cols[p] = d r / d a_{U_p}, same ordering)."""
    T = prob.T if T is None else T
    rows, B, U, k, m, bpos = prob.kernel_rows(mu), prob.B, prob.U, prob.k, prob.m, prob.bpos
    width = k * (m + 1)
    M = [[0.0] * width for _ in range(prob.n)]
    for x, row in enumerate(rows):
        for y, v in row:
            j = bpos.get(y)
            if j is not None:
                M[x][j] = v
    for p, drow in enumerate(prob.drows):
        base = (p + 1) * k
        for y, d in drow:
            j = bpos.get(y)
            if j is not None:
                M[U[p]][base + j] = d
    r, cols = [], [[] for _ in range(m)]
    for t in range(T):
        if t:
            for j, b in enumerate(B):
                Mb = M[b]
                for blk in range(m + 1):
                    Mb[blk * k + j] = 0.0
            G = []
            for drow in prob.drows:
                g = [0.0] * k
                for y, d in drow:
                    My = M[y]
                    g = [a + d * My[j] for j, a in enumerate(g)]
                G.append(g)
            M = _step(rows, M, width)
            for p in range(m):
                row, base = M[U[p]], (p + 1) * k
                gp = G[p]
                for j in range(k):
                    row[base + j] += gp[j]
        for b in B:
            Mb = M[b]
            r.extend(Mb[:k])
            for p in range(m):
                cols[p].extend(Mb[(p + 1) * k:(p + 2) * k])
    return r, cols


def fisher_information(prob: InverseProblem, mu, N):
    """Multinomial Fisher information for a_U: bins are the T hit times plus one 'no hit by T' bin."""
    r, cols = first_passage_jac(prob, mu)
    k2, m = prob.k * prob.k, prob.m
    w = [1.0 / v if v > 1e-15 else 0.0 for v in r]
    scaled = [[c * math.sqrt(wi) for c, wi in zip(col, w)] for col in cols]
    F = gram(scaled)
    hit = [sum(r[t * k2 + pr] for t in range(prob.T)) for pr in range(k2)]
    rest = [min(max(1.0 - h, 0.0), 1.0) for h in hit]
    drest = [[-sum(col[t * k2 + pr] for t in range(prob.T)) for pr in range(k2)] for col in cols]
    for pr in range(k2):
        if rest[pr] > 1e-15:
            inv = 1.0 / rest[pr]
            for p in range(m):
                dp = drest[p][pr] * inv
                if dp:
                    for q in range(m):
                        F[p][q] += dp * drest[q][pr]
    return [[N * v for v in row] for row in F]


def crb_relative_error(prob: InverseProblem, mu, N) -> float:
    """Cramer-Rao lower bound on ||mu_hat - mu|| / ||mu|| over U (delta method var(mu) = mu^4 var(a))."""
    if prob.m == 0:
        return 0.0
    try:
        var_a = spd_inverse_diag(fisher_information(prob, mu, N))
    except (ValueError, ZeroDivisionError):
        return math.inf
    mu_U = [mu[x] for x in prob.U]
    num = math.sqrt(sum(v ** 4 * va for v, va in zip(mu_U, var_a)))
    return num / math.sqrt(sum(v * v for v in mu_U))


def jacobian_condition(prob: InverseProblem, mu) -> float:
    """Condition number of d r / d log a (scale-free local identifiability check)."""
    if prob.m == 0:
        return 1.0
    _, cols = first_passage_jac(prob, mu)
    scaled = [[c / mu[x] for c in col] for col, x in zip(cols, prob.U)]
    ev = sym_eigenvalues(gram(scaled))
    if ev[0] <= 0 or ev[-1] <= 0:
        return math.inf
    return math.sqrt(ev[-1] / ev[0])
