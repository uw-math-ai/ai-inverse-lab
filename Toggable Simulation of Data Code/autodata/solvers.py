"""Inverse solvers: mu on U = X \\ B from first-passage counts on B x B.

The harness needs a *weak* and a *strong* solver (the roles Qwen-4B and Qwen-397B play in
Autodata). An item is useful when it separates them. Built-in solvers:

prior     uses no first-passage data: mu_x = c * wdeg_x with c = median(mu_B / wdeg_B),
          i.e. "centrality follows weighted degree, calibrated on the observed vertices".
lsq       weighted nonlinear least squares on r(t), t <= T_use, by projected Levenberg-Marquardt
          in z = a / a_max in (0, 1], with the analytic Jacobian from forward.py and
          inverse-variance (binomial) weights. `restarts` extra random starts besides the prior.

Default roles (configs/autodata_*.yaml): weak = lsq with a short horizon T_use (it only "sees"
vertices close to B and keeps the prior elsewhere), strong = lsq on the full horizon with restarts.
Register more solvers (e.g. a trained network) with @register("name").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .forward import InverseProblem, first_passage_jac

SOLVERS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        SOLVERS[name] = fn
        return fn
    return deco


@dataclass
class Observation:
    """Measured first-passage data for one noise draw."""
    counts: np.ndarray            # (T, k, k) first hits at t = 1..T
    N: int | None                 # walks per ordered pair; None = exact (noise-free) r
    r: np.ndarray | None = None   # (T, k, k) exact data when N is None

    @property
    def r_obs(self) -> np.ndarray:
        return self.r if self.N is None else self.counts / self.N


def solve(prob: InverseProblem, obs: Observation, solver_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Full mu estimate (n,) from the solver named in solver_cfg['solver']."""
    params = {k: v for k, v in solver_cfg.items() if k != "solver"}
    name = solver_cfg["solver"]
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(SOLVERS)}")
    mu_U = np.asarray(SOLVERS[name](prob, obs, rng, **params), dtype=float)
    return prob.full_mu(np.maximum(mu_U, prob.mu_floor()))


@register("prior")
def prior_solver(prob: InverseProblem, obs: Observation, rng, **_) -> np.ndarray:
    return prior_guess(prob)


def prior_guess(prob: InverseProblem) -> np.ndarray:
    wd = prob.wdeg
    ratio = prob.mu_known[prob.B] / wd[prob.B]
    c = float(np.median(ratio)) if ratio.size else 1.0
    return np.maximum(c * wd[prob.U], prob.mu_floor())


@register("lsq")
def lsq_solver(prob: InverseProblem, obs: Observation, rng, T: int | None = None,
               restarts: int = 0, max_iter: int = 100, tol: float = 1e-6, prior_weight: float = 0.0,
               **_) -> np.ndarray:
    """prior_weight > 0 adds prior_weight * s * ||z - z_prior||^2, s = mean diag(J^T J) at the prior:
    vertices the data barely see stay near the degree prior (a MAP fit) instead of chasing noise."""
    if prob.m == 0:
        return np.zeros(0)
    T_use = prob.T if T is None else int(min(max(T, 1), prob.T))
    data = obs.r_obs[:T_use].reshape(-1)
    if obs.N is None:
        weight = np.ones_like(data)
    else:
        var = np.maximum(data * (1 - data), 1.0 / obs.N) / obs.N
        weight = 1.0 / np.sqrt(var)
    a_max = prob.a_upper()
    z_prior = np.clip(1.0 / prior_guess(prob) / a_max, 1e-4, 1.0)
    starts = [z_prior] + [rng.uniform(0.05, 1.0, prob.m) for _ in range(int(restarts))]
    reg = 0.0
    if prior_weight > 0:
        _, J0 = _residual(prob, z_prior, a_max, data, weight, T_use)
        reg = float(prior_weight) * float(np.sum(J0 * J0)) / prob.m
    best_z, best_cost = None, np.inf
    for z0 in starts:
        z, cost = _levenberg_marquardt(prob, data, weight, T_use, a_max, np.clip(z0, 1e-4, 1.0),
                                       max_iter, tol, z_prior, reg)
        if cost < best_cost:
            best_z, best_cost = z, cost
    return 1.0 / (best_z * a_max)


def _residual(prob, z, a_max, data, weight, T_use, jac=True):
    mu = prob.full_mu(1.0 / (z * a_max))
    r, J = first_passage_jac(prob, mu, T_use)
    res = weight * (r.reshape(-1) - data)
    if not jac:
        return res, None
    Jz = (J.reshape(-1, prob.m) * a_max[None, :]) * weight[:, None]   # d r / d z
    return res, Jz


def _levenberg_marquardt(prob, data, weight, T_use, a_max, z, max_iter, tol, z_ref, reg,
                         lo=1e-4, hi=1.0):
    def penalty(zz):
        return reg * float(np.sum((zz - z_ref) ** 2)) if reg else 0.0

    res, J = _residual(prob, z, a_max, data, weight, T_use)
    cost = float(res @ res) + penalty(z)
    lam = 1e-3
    for _ in range(max_iter):
        g = J.T @ res + reg * (z - z_ref)
        H = J.T @ J + reg * np.eye(len(z))
        diag = np.maximum(np.diag(H), max(1e-9 * float(np.diag(H).max()), 1e-300))
        improved = False
        while lam < 1e12:
            try:
                step = -np.linalg.solve(H + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            z_new = np.clip(z + step, lo, hi)
            res_new, _ = _residual(prob, z_new, a_max, data, weight, T_use, jac=False)
            cost_new = float(res_new @ res_new) + penalty(z_new)
            if cost_new < cost:
                done = (cost - cost_new) <= tol * max(cost, 1e-30) or np.max(np.abs(z_new - z)) < 1e-12
                z, cost = z_new, cost_new
                res, J = _residual(prob, z, a_max, data, weight, T_use)
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 4
        if not improved or done:
            break
    return z, cost


def relative_error(prob: InverseProblem, mu_hat: np.ndarray, mu_true: np.ndarray) -> float:
    """||mu_hat - mu||_2 / ||mu||_2 restricted to the unknown vertices U."""
    if prob.m == 0:
        return 0.0
    d = mu_hat[prob.U] - mu_true[prob.U]
    return float(np.linalg.norm(d) / np.linalg.norm(mu_true[prob.U]))
