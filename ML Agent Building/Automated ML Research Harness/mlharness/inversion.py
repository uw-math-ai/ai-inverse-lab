"""Forward-model-based and hybrid inversion methods (V2).

All methods share the common contract ``fit(X, y) -> self`` / ``predict(X) -> (n, p)`` used by
the runner, so physics, hybrid and data-driven methods are evaluated identically. Physics methods
reach the simulator only through a ``ProblemContext`` (authorised, budgeted forward evaluations).

* ``PhysicsInversion`` — per-sample (batched) maximum-likelihood / MAP estimation by bounded
  Levenberg–Marquardt through the forward operator, optionally in log-parameters, with multistart
  and Tikhonov regularisation toward the prior centre. ``fit`` uses the training set only to check
  the forward model against data (residual noise level, model-mismatch flag).
* ``HybridInversion`` — a learned initializer (any V1 library pipeline) followed by a few LM
  iterations through the forward model.
* ``LogTargetRegressor`` / ``NoiseAugmented`` — wrappers that let ML proposals change the target
  parametrisation or train with matched measurement noise.

All classes are importable top-level classes, so fitted objects pickle; the context drops the live
simulator when pickled and rebuilds it on load.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .problem import ProblemContext, fd_jacobian


# ---------------------------------------------------------------------------------------------
# batched, bounded Levenberg–Marquardt
# ---------------------------------------------------------------------------------------------
def _to_z(theta, transform):
    return np.log(np.maximum(theta, 1e-300)) if transform == "log" else theta


def _from_z(z, transform):
    return np.exp(z) if transform == "log" else z


def lm_solve(fwd, X, z0, zlo, zhi, transform="log", iters=40, tikhonov=0.0, z_prior=None, sigma=1.0,
             h=1e-5, rtol=1e-9, chunk=256, return_jac=False):
    """Minimise ||(F(theta(z)) - x)/sigma||^2 + tikhonov * ||z - z_prior||^2 per sample, z in [zlo, zhi].

    Returns (z, cost, n_iter_used, J at solution or None). Samples converge independently and stop
    consuming forward evaluations once their relative improvement stalls."""
    n, p = z0.shape
    Z = np.empty_like(z0)
    C = np.empty(n)
    J_out = [None] * n if return_jac else None
    used = 0
    lam_t = float(tikhonov or 0.0)
    zp = z_prior if z_prior is not None else np.zeros(p)
    s2 = float(sigma) ** 2
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        z = np.clip(z0[a:b].copy(), zlo, zhi)
        x = X[a:b]
        r = fwd(_from_z(z, transform)) - x
        cost = (r ** 2).sum(1) / s2 + lam_t * ((z - zp) ** 2).sum(1)
        mu = np.full(b - a, 1e-2)
        active = np.ones(b - a, dtype=bool)
        stall = np.zeros(b - a, dtype=int)
        Jl = None
        for it in range(int(iters)):
            idx = np.where(active)[0]
            if len(idx) == 0:
                break
            used = max(used, it + 1)
            zi = z[idx]
            Ji = fd_jacobian(lambda th: fwd(th), _from_z(zi, transform), h=h, transform=transform)  # dF/dz
            JtJ = np.einsum("nmi,nmj->nij", Ji, Ji) / s2 + lam_t * np.eye(p)[None]
            g = np.einsum("nmi,nm->ni", Ji, r[idx]) / s2 + lam_t * (zi - zp)
            diag = np.maximum(np.diagonal(JtJ, axis1=1, axis2=2), 1e-12)
            A = JtJ + mu[idx, None, None] * (np.eye(p)[None] * diag[:, None, :])
            try:
                step = -np.linalg.solve(A, g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = -g / diag
            cand = np.clip(zi + step, zlo, zhi)
            rc = fwd(_from_z(cand, transform)) - x[idx]
            cc = (rc ** 2).sum(1) / s2 + lam_t * ((cand - zp) ** 2).sum(1)
            ok = cc < cost[idx]
            rel = np.where(ok, (cost[idx] - cc) / np.maximum(cost[idx], 1e-300), 0.0)
            acc = idx[ok]
            z[acc], r[acc], cost[acc] = cand[ok], rc[ok], cc[ok]
            mu[idx] = np.where(ok, mu[idx] / 3.0, mu[idx] * 4.0).clip(1e-10, 1e10)
            stall[idx] = np.where(rel < rtol, stall[idx] + 1, 0)
            active[idx] = (stall[idx] < 3) & (mu[idx] < 1e9)
        Z[a:b], C[a:b] = z, cost
        if return_jac:
            Jl = fd_jacobian(fwd, _from_z(z, transform), h=h, transform=transform)
            for k in range(b - a):
                J_out[a + k] = Jl[k]
    return Z, C, used, (np.stack(J_out) if return_jac else None)


class PhysicsInversion(BaseEstimator, RegressorMixin):
    """Per-sample MLE/MAP inversion through the authorised forward model."""

    def __init__(self, context: ProblemContext | None = None, transform="log", multistart=3, iters=40,
                 tikhonov=0.0, fd_step=1e-5, seed=0, uncertainty=True, check_samples=200,
                 estimator="mode", posterior_samples=64, proposal_inflation=1.5):
        self.context = context
        self.transform = transform
        self.multistart = multistart
        self.iters = iters
        self.tikhonov = tikhonov
        self.fd_step = fd_step
        self.seed = seed
        self.uncertainty = uncertainty
        self.check_samples = check_samples
        self.estimator = estimator                    # "mode" (MLE/MAP) | "posterior_mean" (MMSE)
        self.posterior_samples = posterior_samples
        self.proposal_inflation = proposal_inflation

    # -- helpers ---------------------------------------------------------------------------
    def _bounds(self, p):
        ctx = self.context
        lo = ctx.lower if ctx is not None and ctx.lower is not None else np.full(p, 1e-6 if self.transform == "log" else -1e6)
        hi = ctx.upper if ctx is not None and ctx.upper is not None else np.full(p, 1e6)
        return np.asarray(lo, float), np.asarray(hi, float)

    def _sigma(self):
        if getattr(self, "sigma_", None):
            return self.sigma_
        return (self.context.noise_std if self.context is not None and self.context.noise_std else 1.0)

    def fit(self, X, y=None):
        if self.context is None or not self.context.forward_available:
            raise PermissionError("PhysicsInversion needs an authorised forward model (context.forward)")
        y = None if y is None else np.asarray(y, dtype=float)
        self._y1d = y is not None and y.ndim == 1
        self.n_params_out_ = (1 if self._y1d else y.shape[1]) if y is not None else self.context.n_params
        self.sigma_ = None
        self.fit_report_ = {}
        if y is not None and len(y):
            # Use the training set only to validate the forward model: residual noise and mismatch.
            k = min(len(y), int(self.check_samples))
            Y = y[:k].reshape(k, -1)
            res = np.asarray(X[:k], float) - self.context.forward(Y)
            est = float(np.sqrt(np.mean(res ** 2)))
            nominal = self.context.noise_std
            self.fit_report_ = {"residual_std_at_truth": est, "nominal_noise_std": nominal,
                                "model_mismatch": bool(nominal and est > 1.5 * nominal),
                                "checked_samples": int(k)}
            self.sigma_ = est if est > 0 else None
        return self

    def _starts(self, n, p, lo, hi, init=None):
        rng = np.random.default_rng(self.seed)
        zlo, zhi = _to_z(lo, self.transform), _to_z(hi, self.transform)
        starts = []
        if init is not None:
            starts.append(np.clip(_to_z(np.clip(init, lo, hi), self.transform), zlo, zhi))
        center = (np.sqrt(lo * hi) if self.transform == "log" else 0.5 * (lo + hi))
        starts.append(np.broadcast_to(_to_z(center, self.transform), (n, p)).copy())
        for _ in range(max(0, int(self.multistart) - 1)):
            starts.append(rng.uniform(zlo, zhi, size=(n, p)))
        return starts, zlo, zhi

    def _solve(self, X, init=None, iters=None):
        X = np.asarray(X, dtype=float)
        p = int(self.n_params_out_ or self.context.n_params)
        lo, hi = self._bounds(p)
        starts, zlo, zhi = self._starts(len(X), p, lo, hi, init)
        zprior = _to_z(np.sqrt(lo * hi) if self.transform == "log" else 0.5 * (lo + hi), self.transform)
        best_z = best_c = None
        t0 = time.time()
        n_iter = 0
        for s in starts:
            z, c, used, _ = lm_solve(self.context.forward, X, s, zlo, zhi, self.transform,
                                     iters or self.iters, self.tikhonov, zprior, self._sigma(), self.fd_step)
            n_iter = max(n_iter, used)
            if best_z is None:
                best_z, best_c = z, c
            else:
                m = c < best_c
                best_z[m], best_c[m] = z[m], c[m]
        theta = _from_z(best_z, self.transform)
        self.last_cost_ = best_c
        self.last_chi2_per_dof_ = float(np.mean(best_c) / X.shape[1])
        self.last_iters_ = n_iter
        self.last_std_ = None
        self.last_ess_ = None
        need_h = self.uncertainty or self.estimator == "posterior_mean"
        if need_h:
            try:
                J = fd_jacobian(self.context.forward, theta, h=self.fd_step, transform=self.transform)
                s2 = self._sigma() ** 2
                H = np.einsum("nmi,nmj->nij", J, J) / s2 + float(self.tikhonov or 0) * np.eye(p)[None]
                cov = np.linalg.inv(H)
                if self.estimator == "posterior_mean":
                    theta = self._posterior_mean(X, best_z, cov, zlo, zhi)
                else:
                    sz = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0, None))
                    self.last_std_ = theta * sz if self.transform == "log" else sz
            except np.linalg.LinAlgError:
                self.last_std_ = None
        self.last_solve_seconds_ = time.time() - t0
        return theta

    def _posterior_mean(self, X, z_hat, cov_z, zlo, zhi):
        """MMSE estimate under the box-uniform prior: self-normalised importance sampling from the
        (inflated) Laplace approximation around the mode. Also sets posterior std and ESS."""
        n, p = z_hat.shape
        S = max(8, int(self.posterior_samples))
        L = np.linalg.cholesky(cov_z * float(self.proposal_inflation) ** 2 + 1e-18 * np.eye(p)[None])
        # draws are seeded per sample from the observation itself, so a prediction never depends on which
        # other samples share its batch (exported models re-predict identically)
        import zlib
        eps = np.empty((S, n, p))
        for i in range(n):
            key = zlib.crc32(np.ascontiguousarray(X[i], dtype=np.float64).tobytes())
            eps[:, i, :] = np.random.default_rng([int(self.seed) + 7, key]).standard_normal((S, p))
        Zs = z_hat[None] + np.einsum("nij,snj->sni", L, eps)
        inside = np.all((Zs >= zlo) & (Zs <= zhi), axis=2)
        s2 = self._sigma() ** 2
        logt = np.full((S, n), -np.inf)
        for s in range(S):                                   # S batched forward calls of size n
            m = inside[s]
            if not m.any():
                continue
            th = _from_z(Zs[s, m], self.transform)
            r = self.context.forward(th) - X[m]
            lp = Zs[s, m].sum(1) if self.transform == "log" else 0.0   # uniform prior in theta -> e^z Jacobian
            logt[s, m] = -0.5 * (r ** 2).sum(1) / s2 + lp
        logw = logt + 0.5 * (eps ** 2).sum(2)                # minus log q (normalising const cancels per sample)
        logw -= logw.max(axis=0, keepdims=True)
        w = np.exp(logw)
        w[~np.isfinite(w)] = 0.0
        tot = w.sum(0)
        bad = tot <= 0
        w = w / np.where(bad, 1.0, tot)
        Th = _from_z(Zs, self.transform)
        mean = np.einsum("sn,snp->np", w, Th)
        var = np.einsum("sn,snp->np", w, (Th - mean[None]) ** 2)
        mode = _from_z(z_hat, self.transform)
        mean[bad] = mode[bad]
        self.last_std_ = np.sqrt(var)
        self.last_ess_ = float(np.mean(1.0 / np.maximum((w ** 2).sum(0), 1e-300)))
        return mean

    def predict(self, X):
        theta = self._solve(X)
        return theta.ravel() if getattr(self, "_y1d", False) else theta

    def predict_std(self):
        return self.last_std_


class HybridInversion(PhysicsInversion):
    """Learned initializer (V1 pipeline) followed by LM refinement through the forward model."""

    def __init__(self, context: ProblemContext | None = None, initializer=None, transform="log", iters=12,
                 tikhonov=0.0, extra_starts=0, fd_step=1e-5, seed=0, uncertainty=True, check_samples=200,
                 estimator="mode", posterior_samples=64, proposal_inflation=1.5):
        super().__init__(context=context, transform=transform, multistart=extra_starts, iters=iters,
                         tikhonov=tikhonov, fd_step=fd_step, seed=seed, uncertainty=uncertainty,
                         check_samples=check_samples, estimator=estimator, posterior_samples=posterior_samples,
                         proposal_inflation=proposal_inflation)
        self.initializer = initializer
        self.extra_starts = extra_starts

    def fit(self, X, y):
        super().fit(X, y)
        t0 = time.time()
        self.initializer.fit(X, y)
        self.fit_report_["initializer_fit_seconds"] = time.time() - t0
        return self

    def _starts(self, n, p, lo, hi, init=None):
        starts, zlo, zhi = super()._starts(n, p, lo, hi, init)
        # ML start first; the prior-centre start is kept only if extra_starts > 0
        keep = 1 + max(0, int(self.extra_starts))
        return starts[:keep], zlo, zhi

    def predict(self, X):
        init = np.asarray(self.initializer.predict(X), dtype=float)
        init = init.reshape(len(X), -1)
        self.last_init_ = init
        theta = self._solve(X, init=init)
        return theta.ravel() if getattr(self, "_y1d", False) else theta


# ---------------------------------------------------------------------------------------------
# ML wrappers
# ---------------------------------------------------------------------------------------------
class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Regress log(y) (multiplicative errors, positivity) and return exp of the prediction."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("log target requires strictly positive targets")
        self.estimator.fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(np.asarray(self.estimator.predict(X), dtype=float))


class NoiseAugmented(BaseEstimator, RegressorMixin):
    """Train on the data plus ``copies`` re-noised replicas (X + N(0, (scale*sigma)^2))."""

    def __init__(self, estimator=None, copies=1, sigma=1e-3, scale=1.0, seed=0):
        self.estimator = estimator
        self.copies = copies
        self.sigma = sigma
        self.scale = scale
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        Xs = [X] + [X + rng.normal(0, self.sigma * self.scale, X.shape) for _ in range(int(self.copies))]
        self.estimator.fit(np.concatenate(Xs), np.concatenate([y] * (1 + int(self.copies))))
        return self

    def predict(self, X):
        return self.estimator.predict(X)
