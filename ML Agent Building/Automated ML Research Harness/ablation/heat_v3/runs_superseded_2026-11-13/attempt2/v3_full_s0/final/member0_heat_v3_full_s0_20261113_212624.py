import numpy as np
import time


def build_method(params, context):
    lower = np.asarray(context.lower, dtype=float).ravel()
    upper = np.asarray(context.upper, dtype=float).ravel()
    lo = np.log(lower)
    hi = np.log(upper)
    p = int(context.n_params)
    sigma = float(context.noise_std)
    iters = int(params.get('iters', 40))
    nstart = int(params.get('multistart', 3))
    h = float(params.get('fd_step', 1e-4))
    budget = int(params.get('max_calls', 2800000))
    seed = int(params.get('seed', 0))
    n_rec = int(params.get('n_recovery', 20))
    t_fit = float(params.get('fit_time_limit', 20.0))
    t_all = float(params.get('time_limit', 95.0))
    abs_tol = 1e-6
    cost_tol = 1e-8
    try:
        from scipy.stats import chi2
        q99 = float(chi2.ppf(0.99, 80 - p))
        med76 = float(chi2.ppf(0.5, 80 - p))
    except Exception:
        q99 = 107.6
        med76 = 75.3
    eye = np.eye(p)

    def pct(a, v):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, v)) if len(a) else None

    def med(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else None

    class BoxLMVec:
        def __init__(self):
            self.perm = 'as_returned'
            self.calls = 0
            self.predict_calls = 0
            self.layout_ok = False
            self.solver_pass = False
            self.t_start = time.time()

        def _perm(self, F):
            n, m = F.shape
            if self.perm == 'sensor_major_to_time_major':
                return F.reshape(n, 4, m // 4).transpose(0, 2, 1).reshape(n, m)
            if self.perm == 'time_major_to_sensor_major':
                return F.reshape(n, m // 4, 4).transpose(0, 2, 1).reshape(n, m)
            return F

        def fwd(self, th_log):
            th_log = np.atleast_2d(th_log)
            self.calls += len(th_log)
            F = np.asarray(context.forward(np.exp(th_log)), dtype=float).reshape(len(th_log), -1)
            return self._perm(F)

        def _jac(self, th, F):
            N, m = F.shape
            step = np.where(th + h <= hi, h, -h)
            pert = np.repeat(th[:, None, :], p, axis=1)
            idx = np.arange(p)
            pert[:, idx, idx] += step
            Fp = self.fwd(pert.reshape(-1, p)).reshape(N, p, m)
            J = (Fp - F[:, None, :]) / (step[:, :, None] * sigma)
            return J.transpose(0, 2, 1)

        def _solve(self, X, seed_base, deadline):
            n, m = X.shape
            center = np.asarray(context.prior_center(), dtype=float).ravel()
            out = np.tile(center, (n, 1))
            per_row = nstart * (1 + iters * (p + 1) + p)
            n_proc = int(max(0, min(n, (budget - self.calls) // per_row)))
            calls0 = self.calls
            sse = np.full(n, np.nan)
            nonfinite = np.zeros(n, bool)
            nonconv = np.zeros(n, bool)
            stalled = np.zeros(n, bool)
            bstuck = np.zeros(n, bool)
            disagree = np.zeros(n, bool)
            bfall = np.zeros(n, bool)
            bfall[n_proc:] = True
            cond = np.full(n, np.nan)
            fsd = np.full((n, p), np.nan)
            pgn = np.full(n, np.nan)
            iters_done = 0
            timed_out = False
            if n_proc > 0:
                N = n_proc * nstart
                th = np.empty((n_proc, nstart, p))
                for i in range(n_proc):
                    rng = np.random.default_rng([seed_base, i])
                    th[i] = np.log(rng.uniform(lower, upper, size=(nstart, p)))
                th = th.reshape(N, p)
                obs = np.repeat(X[:n_proc], nstart, axis=0)
                F = self.fwd(th)
                r = (obs - F) / sigma
                cost = (r ** 2).sum(1)
                alive = np.isfinite(cost)
                lam = np.full(N, 1e-2)
                last_step = np.zeros(N)
                last_dcost = np.zeros(N)
                last_acc = np.ones(N, bool)
                for it in range(iters):
                    if time.time() > deadline:
                        timed_out = True
                        break
                    J = self._jac(th, F)
                    jok = np.isfinite(J).all(axis=(1, 2))
                    cost[alive & ~jok] = np.nan
                    alive = alive & jok
                    Js = np.where(alive[:, None, None], J, 0.0)
                    rs = np.where(alive[:, None], r, 0.0)
                    g = np.einsum('nmp,nm->np', Js, rs)
                    H = np.einsum('nmp,nmq->npq', Js, Js)
                    dg = np.einsum('npp->np', H)
                    A = H + (lam[:, None] * dg)[:, :, None] * eye + 1e-12 * eye
                    A[~alive] = eye
                    try:
                        d = np.linalg.solve(A, g[..., None])[..., 0]
                    except np.linalg.LinAlgError:
                        d = np.einsum('npq,nq->np', np.linalg.pinv(A), g)
                    th_new = np.clip(th + d, lo, hi)
                    F_new = self.fwd(th_new)
                    r_new = (obs - F_new) / sigma
                    c_new = (r_new ** 2).sum(1)
                    acc = alive & np.isfinite(c_new) & (c_new <= cost)
                    last_step = np.where(acc, np.abs(th_new - th).max(1), 0.0)
                    last_dcost = np.where(acc, (cost - c_new) / np.maximum(cost, 1e-12), 0.0)
                    th[acc] = th_new[acc]
                    F[acc] = F_new[acc]
                    r[acc] = r_new[acc]
                    cost[acc] = c_new[acc]
                    lam = np.where(acc, np.maximum(lam / 3.0, 1e-9), np.minimum(lam * 4.0, 1e12))
                    last_acc = acc | ~alive
                    iters_done = it + 1
                Jf = self._jac(th, F)
                C = cost.reshape(n_proc, nstart)
                T = th.reshape(n_proc, nstart, p)
                ok = np.isfinite(C)
                anyok = ok.any(1)
                best = np.argmin(np.where(ok, C, np.inf), axis=1)
                ri = np.arange(n_proc)
                bi = ri * nstart + best
                tb = T[ri, best]
                kb = np.exp(tb)
                idx = np.where(anyok)[0]
                nonfinite[:n_proc] = ~anyok
                out[idx] = kb[idx]
                sse[idx] = C[idx, best[idx]]
                nc = (last_step[bi] > abs_tol) & (last_dcost[bi] > cost_tol)
                nonconv[idx] = nc[idx]
                stalled[idx] = (~last_acc[bi])[idx]
                at_lo = np.abs(tb - lo) < abs_tol
                at_hi = np.abs(tb - hi) < abs_tol
                bstuck[idx] = (at_lo | at_hi).any(1)[idx]
                relk = np.abs(np.exp(T) - kb[:, None, :]) / kb[:, None, :]
                disagree[idx] = (((relk > 0.01).any(2)) & ok).any(1)[idx]
                Jb = Jf[bi]
                gb = np.einsum('nmp,nm->np', Jb, r[bi])
                gp = np.where((at_lo & (gb < 0)) | (at_hi & (gb > 0)), 0.0, gb)
                pg = np.sqrt((gp ** 2).sum(1))
                pgn[idx] = pg[idx]
                Hb = np.einsum('nmp,nmq->npq', Jb, Jb)
                for i in idx:
                    if not np.all(np.isfinite(Hb[i])):
                        continue
                    try:
                        sv = np.linalg.svd(Hb[i], compute_uv=False)
                        cond[i] = float(sv.max() / max(sv.min(), 1e-300))
                        fsd[i] = np.sqrt(np.clip(np.diag(np.linalg.pinv(Hb[i])), 0, None))
                    except Exception:
                        pass
            fin = sse[np.isfinite(sse)]
            stats = {
                'n_rows': int(n),
                'n_obs': int(m),
                'n_processed': int(n_proc),
                'iters_requested': iters,
                'iters_done': int(iters_done),
                'timed_out': bool(timed_out),
                'forward_calls': int(self.calls - calls0),
                'calls_per_processed_row': float((self.calls - calls0) / max(n_proc, 1)),
                'n_nonfinite': int(nonfinite.sum()),
                'n_nonconverged_abs_step_and_cost': int(nonconv.sum()),
                'n_stalled_no_accepted_step_final_iter': int(stalled.sum()),
                'n_conv_not_stalled': int((~nonconv & ~stalled & ~bfall & ~nonfinite).sum()),
                'n_conv_stalled': int((~nonconv & stalled).sum()),
                'n_nonconv_not_stalled': int((nonconv & ~stalled).sum()),
                'n_nonconv_stalled': int((nonconv & stalled).sum()),
                'n_bound_stuck': int(bstuck.sum()),
                'n_budget_fallback': int(bfall.sum()),
                'frac_start_disagree_gt_1pct': float(disagree.sum() / max(n_proc, 1)),
                'sse_expected_chi2_df': int(80 - p),
                'chi2_76_median_ref': med76,
                'sse_mean': float(fin.mean()) if len(fin) else None,
                'sse_median': med(sse),
                'sse_p05': pct(sse, 5),
                'sse_p95': pct(sse, 95),
                'sse_max': float(fin.max()) if len(fin) else None,
                'sse_q99_threshold': q99,
                'n_sse_above_q99': int((fin > q99).sum()),
                'expected_n_above_q99': float(0.01 * len(fin)),
                'sse_median_bound_stuck': med(sse[bstuck]),
                'sse_median_not_bound_stuck': med(sse[~bstuck]),
                'proj_grad_norm_median': med(pgn),
                'proj_grad_norm_p95': pct(pgn, 95),
                'proj_grad_norm_median_conv_stalled': med(pgn[~nonconv & stalled]),
                'local_fisher_sd_logk_median_per_target': [med(fsd[:, j]) for j in range(p)],
                'local_fisher_cond_median': med(cond),
                'local_fisher_note': 'local sensitivity at the final iterate of the best start; not a global identifiability guarantee',
            }
            return out, stats

        def fit(self, X, y):
            self.t_start = time.time()
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            col_mean = float(np.abs(X.mean(0)).max())
            col_std = float(np.abs(X.std(0) - 1.0).max())
            if col_mean < 1e-3 and col_std < 1e-3:
                raise RuntimeError('PHYSICS_ABORT X appears standardised')
            n = int(min(20, len(y)))
            if (y[:n] < lower - 1e-9).any() or (y[:n] > upper + 1e-9).any():
                raise RuntimeError('PHYSICS_ABORT training y not in raw k bounds')
            m = int(X.shape[1])
            self.perm = 'as_returned'
            F0 = self.fwd(np.log(y[:n]))
            names = ['as_returned']
            if m % 4 == 0:
                names += ['sensor_major_to_time_major', 'time_major_to_sensor_major']
            sses = {}
            for name in names:
                self.perm = name
                sses[name] = float((((X[:n] - self._perm(F0)) / sigma) ** 2).sum(1).mean())
            self.perm = min(sses, key=sses.get)
            best_sse = sses[self.perm]
            self.layout_ok = bool(abs(best_sse - m) < 12.0)
            print('PHYSICS_LAYOUT_REPORT', {'layout_sse_means': sses, 'perm_used': self.perm, 'layout_sse_mean': best_sse, 'layout_ok_rule': '|min_sse_mean - n_obs| < 12 over 20 rows', 'layout_ok': self.layout_ok, 'x_col_mean_absmax': col_mean, 'x_col_std_dev_from_1_max': col_std, 'layout_calls': int(self.calls)})
            nr = int(min(n_rec, len(y)))
            rec, st = self._solve(X[:nr], seed + 1000000, self.t_start + t_fit)
            yt = y[:nr]
            rel = float(np.linalg.norm(rec - yt) / max(np.linalg.norm(yt), 1e-12))
            per_target = [float(np.linalg.norm(rec[:, j] - yt[:, j]) / max(np.linalg.norm(yt[:, j]), 1e-12)) for j in range(p)]
            logerr = np.log(rec) - np.log(yt)
            c_rel = rel < 0.10
            c_med = (st['sse_median'] is not None) and (abs(st['sse_median'] - med76) < 10.0)
            c_q99 = st['n_sse_above_q99'] <= 1
            c_dis = st['frac_start_disagree_gt_1pct'] <= 0.25
            c_run = (not st['timed_out']) and st['n_budget_fallback'] == 0 and st['n_nonfinite'] == 0
            self.solver_pass = bool(c_rel and c_med and c_q99 and c_dis and c_run)
            print('SOLVER_VERIFICATION', {'n_recovery_rows': nr, 'recovery_rel_l2_frobenius': rel, 'recovery_rel_l2_per_target': per_target, 'recovery_logk_rmse_per_target': [float(np.sqrt(np.mean(logerr[:, j] ** 2))) for j in range(p)], 'local_fisher_sd_logk_median_per_target': st['local_fisher_sd_logk_median_per_target'], 'rule': 'rel_l2<0.10 and |sse_median-chi2_76 median|<10 and n_sse_above_q99<=1 and frac_disagree<=0.25 and no timeout, fallback or non-finite rows', 'pass_rel': bool(c_rel), 'pass_sse_median': bool(c_med), 'pass_q99': bool(c_q99), 'pass_disagree': bool(c_dis), 'pass_run': bool(c_run), 'solver_pass': self.solver_pass, 'seed_namespace': seed + 1000000, 'fit_elapsed_s': float(time.time() - self.t_start), 'solver_stats': st})
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            if len(X) >= 50:
                cm = float(np.abs(X.mean(0)).max())
                cs = float(np.abs(X.std(0) - 1.0).max())
                if cm < 1e-3 and cs < 1e-3:
                    raise RuntimeError('PHYSICS_ABORT predict X appears standardised')
            self.predict_calls += 1
            calls_before = int(self.calls)
            t0 = time.time()
            out, st = self._solve(X, seed, self.t_start + t_all)
            val_sse_flag = (st['sse_median'] is None) or (st['sse_median'] > med76 + 15.0)
            gates = bool(self.layout_ok and self.solver_pass and (not st['timed_out']) and st['n_budget_fallback'] == 0 and self.predict_calls == 1 and not val_sse_flag)
            report = dict(st)
            report.update({'n_val': int(len(X)), 'predict_calls': int(self.predict_calls), 'calls_before_predict': calls_before, 'total_forward_calls_arm': int(self.calls), 'budget_cap': budget, 'perm_used': self.perm, 'layout_ok': self.layout_ok, 'solver_pass': self.solver_pass, 'predict_elapsed_s': float(time.time() - t0), 'arm_elapsed_s': float(time.time() - self.t_start), 'fd_step_log': h, 'seed_scheme': 'default_rng([seed, row index within call])', 'val_sse_median_mismatch_flag': bool(val_sse_flag), 'arm_gates_ok': gates, 'arm_gates_rule': 'layout_ok and solver_pass and not timed_out and n_budget_fallback==0 and predict_calls==1 and not val_sse flag; necessary, not sufficient', 'facts': context.facts})
            print('PHYSICS_LM_REPORT', report)
            return out

    return BoxLMVec()
