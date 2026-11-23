import json
import itertools
import numpy as np


def _q(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {'p10': float(np.percentile(a, 10)), 'p50': float(np.percentile(a, 50)), 'p90': float(np.percentile(a, 90)), 'max': float(np.max(a)), 'mean': float(np.mean(a)), 'std': float(np.std(a))}


class GateError(RuntimeError):
    pass


class TransparentLogMLE:
    # Box-constrained Gaussian likelihood procedure in log k: batched Levenberg-Marquardt with projected steps.
    # Likelihood uses only the supplied 80-dim observation vector and context.noise_std.
    # Jacobian: forward FD in log k (backward at upper bound). Starts: log-box midpoint plus seeded
    # uniform log-box draws, independent of targets. Selection: argmin chi2 over starts only.
    # Prediction: exp(best-objective point); no posterior sampling.
    # Hard gates in fit (GateError => arm fails before any prediction is scored):
    #   G1 forward dim == feature dim == 80
    #   G2 y_train within box (original k scale, not log)
    #   G3 forward deterministic (bit-identical repeat call)
    #   G4 forward identity on 200 train rows: mean chi2 in band, residual std in band
    #   G5 forward finite on 16 corners and 1000 uniform draws
    #   G6 noiseless recovery on 100 train rows: rel_l2 <= threshold
    #   G7 batched vs sub-batch forward agree within batch_tol

    def __init__(self, params, context):
        self.ctx = context
        self.iters = int(params.get('iters', 30))
        self.n_starts = int(params.get('multistart', 3))
        self.h = float(params.get('fd_step', 1e-4))
        self.seed = int(params.get('start_seed', 20261114))
        self.chi2_lo = float(params.get('chi2_mean_lo', 74.0))
        self.chi2_hi = float(params.get('chi2_mean_hi', 86.0))
        self.rstd_lo = float(params.get('resid_std_lo', 0.95e-3))
        self.rstd_hi = float(params.get('resid_std_hi', 1.05e-3))
        self.noiseless_max_rel = float(params.get('noiseless_max_rel_l2', 0.01))
        self.batch_tol = float(params.get('batch_tol', 1e-9))
        self.p = int(context.n_params)
        self.lower = np.asarray(context.lower, dtype=float)
        self.upper = np.asarray(context.upper, dtype=float)
        self.lo = np.log(self.lower)
        self.hi = np.log(self.upper)
        self.sigma = float(context.noise_std)
        self.forward_calls = 0
        self.solve_fallbacks = 0
        self.bad_systems = 0
        self.diagnostics = {}

    def _gate(self, d, name, ok):
        ok = bool(ok)
        d.setdefault('gates', {})[name] = ok
        if not ok:
            print('TRANSPARENT_MLE_GATE_FAILED ' + name + ' ' + json.dumps(d))
            raise GateError('gate ' + name + ' failed')

    def _fwd_k(self, k):
        k = np.atleast_2d(np.asarray(k, dtype=float))
        self.forward_calls += int(k.shape[0])
        return np.asarray(self.ctx.forward(k), dtype=float).reshape(k.shape[0], -1)

    def _fwd(self, z):
        return self._fwd_k(np.exp(np.clip(z, self.lo, self.hi)))

    def _lm(self, Y, z0):
        n, p = z0.shape
        s, h = self.sigma, self.h
        z = np.clip(z0.copy(), self.lo, self.hi)
        f = self._fwd(z)
        r = (f - Y) / s
        cost = np.sum(r * r, axis=1)
        cost = np.where(np.isfinite(cost), cost, np.inf)
        lam = np.full(n, 1e-2)
        status = np.zeros(n, dtype=int)  # 0 running or capped, 1 tolerance, 2 stuck at lambda max
        used = np.zeros(n, dtype=int)
        eye = np.eye(p)
        for _ in range(self.iters):
            idx = np.where(status == 0)[0]
            if idx.size == 0:
                break
            za, fa, ra = z[idx], f[idx], r[idx]
            sign = np.where(za + h > self.hi, -1.0, 1.0)
            pert = za[None, :, :] + (h * sign)[None, :, :] * eye[:, None, :]
            fp = self._fwd(pert.reshape(-1, p)).reshape(p, idx.size, -1)
            J = ((fp - fa[None]) / s).transpose(1, 2, 0) / (h * sign)[:, None, :]
            JtJ = np.einsum('nmp,nmq->npq', J, J)
            g = np.array(np.einsum('nmp,nm->np', J, ra))
            dg = np.array(np.einsum('npp->np', JtJ)) + 1e-12
            A = JtJ + lam[idx][:, None, None] * (dg[:, :, None] * eye[None])
            bad = ~(np.isfinite(A).all(axis=(1, 2)) & np.isfinite(g).all(axis=1))
            if bad.any():
                A[bad] = eye
                g[bad] = 0.0
                self.bad_systems += int(bad.sum())
            try:
                step = -np.linalg.solve(A, g[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                self.solve_fallbacks += 1
                step = np.stack([-np.linalg.lstsq(A[i], g[i], rcond=None)[0] for i in range(idx.size)])
            zn = np.clip(za + step, self.lo, self.hi)
            fn = self._fwd(zn)
            rn = (fn - Y[idx]) / s
            cn = np.sum(rn * rn, axis=1)
            cold = cost[idx].copy()
            ok = np.isfinite(cn) & (cn < cold)
            ai = idx[ok]
            z[ai] = zn[ok]
            f[ai] = fn[ok]
            r[ai] = rn[ok]
            cost[ai] = cn[ok]
            lam[idx] = np.where(ok, np.maximum(lam[idx] * 0.3, 1e-9), np.minimum(lam[idx] * 10.0, 1e9))
            used[idx] += 1
            dec = np.where(ok, cold - cn, 0.0)
            small = ok & ((dec <= 1e-9 * np.maximum(cold, 1.0)) | (np.max(np.abs(zn - za), axis=1) < 1e-9))
            stuck = (~ok) & (lam[idx] >= 1e9)
            status[idx[small]] = 1
            status[idx[stuck]] = 2
        return z, cost, used, status

    def _solve(self, Y):
        Y = np.asarray(Y, dtype=float)
        n, m = Y.shape
        rng = np.random.default_rng(self.seed)
        starts = [np.tile(0.5 * (self.lo + self.hi), (n, 1))]
        for _ in range(self.n_starts - 1):
            starts.append(rng.uniform(self.lo, self.hi, size=(n, self.p)))
        fb0, bs0 = self.solve_fallbacks, self.bad_systems
        res = [self._lm(Y, s0) for s0 in starts]
        Z = np.stack([q[0] for q in res])
        C = np.stack([q[1] for q in res])
        U = np.stack([q[2] for q in res])
        S = np.stack([q[3] for q in res])
        best = np.argmin(C, axis=0)
        ar = np.arange(n)
        zb, cb, sb = Z[best, ar], C[best, ar], S[best, ar]
        kb = np.exp(zb)
        spread = np.max(np.abs(Z - zb[None]), axis=2).max(axis=0)
        pinned = np.any((np.abs(kb - self.lower) < 1e-6) | (np.abs(kb - self.upper) < 1e-6), axis=1)
        fin = np.isfinite(cb)
        stats = {
            'n_rows': int(n), 'n_obs_per_row': int(m),
            'chi2_per_dof_best': _q(cb[fin] / m),
            'frac_tolerance_converged': float(np.mean(sb == 1)),
            'frac_stuck_lambda_max': float(np.mean(sb == 2)),
            'frac_iteration_capped': float(np.mean(sb == 0)),
            'mean_iters_used_best': float(np.mean(U[best, ar])),
            'frac_any_component_pinned_1e-6': float(np.mean(pinned)),
            'pinned_row_indices': [int(i) for i in np.where(pinned)[0]],
            'frac_starts_disagree_gt_1e-3_logk': float(np.mean(spread > 1e-3)),
            'n_rows_starts_disagree_gt_1e-3_logk': int(np.sum(spread > 1e-3)),
            'frac_nonfinite_objective': float(np.mean(~fin)),
            'best_start_counts': [int(np.sum(best == j)) for j in range(self.n_starts)],
            'linear_solve_fallbacks': int(self.solve_fallbacks - fb0),
            'nonfinite_linear_systems': int(self.bad_systems - bs0),
            'selection_rule': 'argmin chi2 over starts (objective only)',
            'start_rule': 'start0 = log-box midpoint; others = uniform in log box, seed ' + str(self.seed),
        }
        return kb, stats, pinned

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        c0 = self.forward_calls
        d = {'x_shape': list(X.shape), 'y_shape': list(y.shape), 'x_mean': float(np.mean(X)), 'x_std': float(np.std(X)), 'y_min': float(np.min(y)), 'y_max': float(np.max(y))}
        self._gate(d, 'G2_y_original_k_scale', y.ndim == 2 and y.shape[1] == self.p and np.min(y) >= np.min(self.lower) - 1e-9 and np.max(y) <= np.max(self.upper) + 1e-9)
        m = min(200, len(y))
        Fa = self._fwd_k(y[:m])
        d['obs_dim_forward'] = int(Fa.shape[1])
        d['obs_dim_features'] = int(X.shape[1])
        self._gate(d, 'G1_obs_dim_80', Fa.shape[1] == 80 and X.shape[1] == 80)
        F1 = self._fwd_k(y[:4])
        F2 = self._fwd_k(y[:4])
        d['determinism_max_abs_diff'] = float(np.max(np.abs(F1 - F2)))
        self._gate(d, 'G3_forward_deterministic', np.array_equal(F1, F2))
        Fs = np.vstack([self._fwd_k(y[i:i + 1]) for i in range(4)])
        bdiff = float(max(np.max(np.abs(F1 - Fa[:4])), np.max(np.abs(Fs - Fa[:4]))))
        d['batch_consistency_max_abs_diff'] = bdiff
        d['batch_tol'] = self.batch_tol
        self._gate(d, 'G7_batch_consistent', bdiff <= self.batch_tol)
        res = (X[:m] - Fa) / self.sigma
        chi2 = np.sum(res * res, axis=1)
        rstd = float(np.std(res) * self.sigma)
        d['forward_identity_rows'] = int(m)
        d['chi2_at_truth'] = _q(chi2)
        d['chi2_mean_band'] = [self.chi2_lo, self.chi2_hi]
        d['residual_std_at_truth'] = rstd
        d['residual_std_band'] = [self.rstd_lo, self.rstd_hi]
        d['max_abs_residual_over_sigma'] = float(np.max(np.abs(res)))
        cm = float(np.mean(chi2))
        self._gate(d, 'G4_forward_identity', self.chi2_lo <= cm <= self.chi2_hi and self.rstd_lo <= rstd <= self.rstd_hi)
        corners = np.array(list(itertools.product(*zip(self.lower, self.upper))))
        draws = np.random.default_rng(self.seed + 1).uniform(self.lower, self.upper, size=(1000, self.p))
        out = self._fwd_k(np.vstack([corners, draws]))
        nc = len(corners)
        d['stability_corners_finite'] = bool(np.isfinite(out[:nc]).all())
        d['stability_1000_draws_finite'] = bool(np.isfinite(out[nc:]).all())
        d['corner_output_min'] = float(np.nanmin(out[:nc]))
        d['draws_output_min'] = float(np.nanmin(out[nc:]))
        self._gate(d, 'G5_stability_finite', d['stability_corners_finite'] and d['stability_1000_draws_finite'])
        nn = min(100, len(y))
        yt = y[:nn]
        k_hat, st, pin = self._solve(self._fwd_k(yt))
        lerr = np.abs(np.log(k_hat) - np.log(yt))
        rel = float(np.linalg.norm(k_hat - yt) / np.linalg.norm(yt))
        d['noiseless_recovery_rows'] = int(nn)
        d['noiseless_max_abs_logk_err'] = float(np.max(lerr))
        d['noiseless_median_abs_logk_err'] = float(np.median(lerr))
        d['noiseless_rel_l2'] = rel
        d['noiseless_rel_l2_threshold'] = self.noiseless_max_rel
        d['noiseless_rel_l2_pinned'] = float(np.linalg.norm(k_hat[pin] - yt[pin]) / np.linalg.norm(yt[pin])) if pin.any() else None
        d['noiseless_rel_l2_unpinned'] = float(np.linalg.norm(k_hat[~pin] - yt[~pin]) / np.linalg.norm(yt[~pin])) if (~pin).any() else None
        d['noiseless_solver_stats'] = st
        d['fit_forward_calls'] = int(self.forward_calls - c0)
        self._gate(d, 'G6_noiseless_recovery', rel <= self.noiseless_max_rel)
        self.diagnostics['fit_train_only'] = d
        print('TRANSPARENT_MLE_FIT_DIAGNOSTICS ' + json.dumps(d))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        c0 = self.forward_calls
        k, st, pin = self._solve(X)
        st['x_shape'] = list(X.shape)
        st['predict_forward_calls'] = int(self.forward_calls - c0)
        st['total_forward_calls'] = int(self.forward_calls)
        p50 = st['chi2_per_dof_best'].get('p50', float('inf'))
        st['reporting_criteria'] = {
            'nonfinite_objective_eq_0': bool(st['frac_nonfinite_objective'] == 0.0),
            'iteration_capped_le_0.10': bool(st['frac_iteration_capped'] <= 0.10),
            'chi2_per_dof_p50_le_1.2': bool(p50 <= 1.2),
        }
        st['reporting_criteria_all_pass'] = bool(all(st['reporting_criteria'].values()))
        self.diagnostics['predict'] = st
        print('TRANSPARENT_MLE_PREDICT_DIAGNOSTICS ' + json.dumps(st))
        return k


def build_method(params, context):
    return TransparentLogMLE(params, context)
