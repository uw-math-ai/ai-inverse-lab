"""Weak / strong inverse solvers for mu on X \\ B (pure Python).

prior   no first-passage data: mu_x = c wdeg_x, c = median(mu_B / wdeg_B)
lsq     weighted least squares on r(t), t <= T_use, projected Levenberg-Marquardt in
        z = a / a_max in (0, 1], analytic Jacobian, binomial inverse-variance weights,
        `restarts` extra random starts besides the prior. prior_weight > 0 adds
        prior_weight * s * ||z - z_prior||^2 with s = mean diag(J^T J) at the prior, so vertices the
        data barely see stay near the degree prior (a MAP fit) instead of chasing noise.
"""
from __future__ import annotations

import math
import statistics

from .forward import first_passage, first_passage_jac
from .linalg import dot, gram, solve as linsolve

SOLVERS = {}


def register(name):
    def deco(fn):
        SOLVERS[name] = fn
        return fn
    return deco


def solve(prob, obs, solver_cfg, rng):
    """Full mu estimate from solver_cfg['solver']; obs = {'r': flat r_obs, 'N': int | None}."""
    name = solver_cfg["solver"]
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(SOLVERS)}")
    params = {k: v for k, v in solver_cfg.items() if k != "solver"}
    mu_U = SOLVERS[name](prob, obs, rng, **params)
    return prob.full_mu([max(v, f) for v, f in zip(mu_U, prob.mu_floor())])


def prior_guess(prob):
    ratios = [prob.mu_known[b] / prob.wdeg[b] for b in prob.B]
    c = statistics.median(ratios) if ratios else 1.0
    return [max(c * prob.wdeg[x], f) for x, f in zip(prob.U, prob.mu_floor())]


@register("prior")
def prior_solver(prob, obs, rng, **_):
    return prior_guess(prob)


@register("lsq")
def lsq_solver(prob, obs, rng, T=None, restarts=0, max_iter=100, tol=1e-6, prior_weight=0.0, **_):
    if prob.m == 0:
        return []
    T_use = prob.T if T is None else int(min(max(int(T), 1), prob.T))
    data = obs["r"][:T_use * prob.k * prob.k]
    N = obs["N"]
    weight = [1.0] * len(data) if N is None else \
        [1.0 / math.sqrt(max(d * (1 - d), 1.0 / N) / N) for d in data]
    a_max = prob.a_upper()
    z_prior = [min(max(1.0 / (g * am), 1e-4), 1.0) for g, am in zip(prior_guess(prob), a_max)]
    starts = [z_prior] + [[rng.uniform(0.05, 1.0) for _ in range(prob.m)] for _ in range(int(restarts))]
    reg = 0.0
    if prior_weight > 0:
        _, J0 = _residual(prob, z_prior, a_max, data, weight, T_use)
        reg = float(prior_weight) * sum(dot(c, c) for c in J0) / prob.m
    best_z, best_cost = None, math.inf
    for z0 in starts:
        z, cost = _lm(prob, data, weight, T_use, a_max, z0, int(max_iter), tol, z_prior, reg)
        if cost < best_cost:
            best_z, best_cost = z, cost
    return [1.0 / (z * am) for z, am in zip(best_z, a_max)]


def _residual(prob, z, a_max, data, weight, T_use, jac=True):
    mu = prob.full_mu([1.0 / (zi * am) for zi, am in zip(z, a_max)])
    if not jac:
        r = first_passage(prob, mu, T_use)
        return [w * (a - b) for w, a, b in zip(weight, r, data)], None
    r, cols = first_passage_jac(prob, mu, T_use)
    res = [w * (a - b) for w, a, b in zip(weight, r, data)]
    Jz = [[c * w * am for c, w in zip(col, weight)] for col, am in zip(cols, a_max)]
    return res, Jz


def _lm(prob, data, weight, T_use, a_max, z, max_iter, tol, z_ref, reg, lo=1e-4, hi=1.0):
    def penalty(zz):
        return reg * sum((a - b) ** 2 for a, b in zip(zz, z_ref)) if reg else 0.0

    res, J = _residual(prob, z, a_max, data, weight, T_use)
    cost = dot(res, res) + penalty(z)
    lam = 1e-3
    m = len(z)
    for _ in range(max_iter):
        g = [dot(col, res) + reg * (zi - zr) for col, zi, zr in zip(J, z, z_ref)]
        H = gram(J)
        for i in range(m):
            H[i][i] += reg
        top = max(H[i][i] for i in range(m))
        diag = [max(H[i][i], 1e-9 * top, 1e-300) for i in range(m)]
        improved, done = False, False
        while lam < 1e12:
            A = [[H[i][j] + (lam * diag[i] if i == j else 0.0) for j in range(m)] for i in range(m)]
            try:
                step = linsolve(A, [-gi for gi in g])
            except ZeroDivisionError:
                lam *= 10
                continue
            z_new = [min(max(zi + si, lo), hi) for zi, si in zip(z, step)]
            res_new, _ = _residual(prob, z_new, a_max, data, weight, T_use, jac=False)
            cost_new = dot(res_new, res_new) + penalty(z_new)
            if cost_new < cost:
                done = (cost - cost_new) <= tol * max(cost, 1e-30) or \
                    max(abs(a - b) for a, b in zip(z_new, z)) < 1e-12
                z, cost = z_new, cost_new
                res, J = _residual(prob, z, a_max, data, weight, T_use)
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 4
        if not improved or done:
            break
    return z, cost


def relative_error(prob, mu_hat, mu_true) -> float:
    if prob.m == 0:
        return 0.0
    num = sum((mu_hat[x] - mu_true[x]) ** 2 for x in prob.U)
    den = sum(mu_true[x] ** 2 for x in prob.U)
    return math.sqrt(num / den)
