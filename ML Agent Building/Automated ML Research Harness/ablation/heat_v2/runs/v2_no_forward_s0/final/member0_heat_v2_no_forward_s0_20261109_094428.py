import numpy as np

def _get(c, k, d):
    try:
        v = c[k]
        if v is not None:
            return v
    except Exception:
        pass
    try:
        if k == 'lower':
            return c.lower
        if k == 'upper':
            return c.upper
        if k == 'noise_std':
            return c.noise_std
    except Exception:
        pass
    return d

def _rbf(a, b, g):
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.exp(-g * np.maximum(d2, 0.0))

def _lin(z):
    return np.hstack([np.ones((len(z), 1)), z])

class SurrogateLaplace:
    def __init__(self, params, context):
        self.m = int(params.get('m', 40))
        self.iters = int(params.get('iters', 30))
        self.S = int(params.get('samples', 96))
        self.infl = float(params.get('inflate', 1.5))
        self.est = params.get('estimator', 'posterior_mean')
        self.lo = np.log(np.asarray(_get(context, 'lower', [0.5] * 4), float))
        self.hi = np.log(np.asarray(_get(context, 'upper', [2.0] * 4), float))
        self.sig = float(np.mean(np.asarray(_get(context, 'noise_std', 1e-3), float)))
        self.p = len(self.lo)
        self.c = self.hi - self.lo
        self.diagnostics = {}

    def tz(self, y):
        return np.clip((np.log(np.clip(y, 1e-8, None)) - self.lo) / self.c, 0.0, 1.0)

    def fz(self, z):
        return np.exp(self.lo + np.clip(z, 0.0, 1.0) * self.c)

    def fit(self, X, y):
        X = np.asarray(X, float)
        Z = self.tz(np.asarray(y, float))
        self.mu = X.mean(0)
        Xc = X - self.mu
        Vt = np.linalg.svd(Xc, full_matrices=False)[2]
        m = min(self.m, Vt.shape[0])
        self.V = Vt[:m].T
        Y = Xc @ self.V
        n = len(Z)
        s2 = self.sig ** 2
        idx = np.random.RandomState(0).permutation(n)
        nv = max(30, n // 6)
        va, tr = idx[:nv], idx[nv:]
        Bt = np.linalg.lstsq(_lin(Z[tr]), Y[tr], rcond=None)[0]
        Rt = Y[tr] - _lin(Z[tr]) @ Bt
        Pv = _lin(Z[va]) @ Bt
        best = None
        for g in [3.0, 10.0, 30.0, 100.0]:
            Kt = _rbf(Z[tr], Z[tr], g)
            Kv = _rbf(Z[va], Z[tr], g)
            for al in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                A = np.linalg.solve(Kt + al * np.eye(len(tr)), Rt)
                E = Y[va] - Pv - Kv @ A
                sc = np.log(np.maximum((E ** 2).mean(0), s2)).sum()
                if best is None or sc < best[0]:
                    best = (sc, g, al, E)
        _, self.g, self.al, E = best
        C = E.T @ E / len(E)
        C = 0.8 * C + 0.2 * np.diag(np.diag(C)) + 1e-2 * s2 * np.eye(m)
        self.W = np.linalg.inv(np.linalg.cholesky(C))
        self.Zt = Z
        self.B = np.linalg.lstsq(_lin(Z), Y, rcond=None)[0]
        K = _rbf(Z, Z, self.g)
        self.A = np.linalg.solve(K + self.al * np.eye(n), Y - _lin(Z) @ self.B)
        self.Yw = Y @ self.W.T
        self.diagnostics['emu_resid_over_sigma2_pc1_10'] = [float(v) for v in (np.diag(C)[:10] / s2)]
        self.diagnostics['gamma'] = float(self.g)
        self.diagnostics['alpha'] = float(self.al)
        return self

    def emu(self, z, jac=True):
        K = _rbf(z, self.Zt, self.g)
        KA = K @ self.A
        f = _lin(z) @ self.B + KA
        if not jac:
            return f
        J = np.empty((len(z), f.shape[1], self.p))
        for d in range(self.p):
            J[:, :, d] = self.B[1 + d][None, :] - 2.0 * self.g * (z[:, d:d + 1] * KA - (K * self.Zt[:, d][None, :]) @ self.A)
        return f, J

    def _phi(self, yw, z):
        r = yw - self.emu(z, False) @ self.W.T
        return 0.5 * (r ** 2).sum(1) - z @ self.c

    def _mode(self, yw, z):
        lam = np.full(len(z), 1e-2)
        phi = self._phi(yw, z)
        I = np.eye(self.p)[None]
        for _ in range(self.iters):
            f, J = self.emu(z)
            r = yw - f @ self.W.T
            Jw = np.einsum('ij,bjd->bid', self.W, J)
            H = np.einsum('bid,bie->bde', Jw, Jw)
            gr = np.einsum('bid,bi->bd', Jw, r) + self.c
            D = np.diagonal(H, 0, 1, 2)
            Hd = H + lam[:, None, None] * I * (D[:, None, :] + 1e-6)
            dz = np.linalg.solve(Hd, gr[:, :, None])[:, :, 0]
            zn = np.clip(z + dz, 0.0, 1.0)
            pn = self._phi(yw, zn)
            ok = pn < phi
            z = np.where(ok[:, None], zn, z)
            phi = np.where(ok, pn, phi)
            lam = np.where(ok, lam * 0.3, lam * 10.0)
        return z, phi

    def _post(self, yw, z0, rng):
        f, J = self.emu(z0)
        Jw = np.einsum('ij,bjd->bid', self.W, J)
        H = np.einsum('bid,bie->bde', Jw, Jw)
        Cv = np.linalg.inv(H + np.eye(self.p)[None]) * self.infl ** 2
        Cv = 0.5 * (Cv + np.swapaxes(Cv, 1, 2))
        Lc = np.linalg.cholesky(Cv)
        B, S, p = len(z0), self.S, self.p
        e = rng.standard_normal((B, S, p))
        zs = z0[:, None, :] + np.einsum('bde,bse->bsd', Lc, e)
        inb = np.all((zs >= 0.0) & (zs <= 1.0), axis=2)
        ft = (self.emu(zs.reshape(-1, p), False) @ self.W.T).reshape(B, S, -1)
        lt = -0.5 * ((yw[:, None, :] - ft) ** 2).sum(2) + zs @ self.c
        lq = -0.5 * (e ** 2).sum(2)
        lw = np.where(inb, lt - lq, -np.inf)
        mx = lw.max(1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        w = np.where(inb, np.exp(lw - mx), 0.0)
        s = w.sum(1, keepdims=True)
        w = w / np.maximum(s, 1e-300)
        ess = 1.0 / np.maximum((w ** 2).sum(1), 1e-12)
        km = np.einsum('bs,bsd->bd', w, self.fz(zs))
        bad = (ess < 3.0) | (s[:, 0] <= 0.0) | (~np.isfinite(km).all(1))
        km[bad] = self.fz(z0)[bad]
        return km, ess, bad, np.diagonal(H, 0, 1, 2)

    def predict(self, X):
        X = np.asarray(X, float)
        Yw = ((X - self.mu) @ self.V) @ self.W.T
        rng = np.random.RandomState(1)
        out = np.empty((len(X), self.p))
        fb, es, ch, info = [], [], [], []
        bs = 25
        for s in range(0, len(X), bs):
            yw = Yw[s:s + bs]
            d2 = (yw ** 2).sum(1)[:, None] + (self.Yw ** 2).sum(1)[None, :] - 2.0 * yw @ self.Yw.T
            nn = np.argsort(d2, 1)[:, :2]
            bz, bp = None, None
            for j in range(nn.shape[1]):
                z, ph = self._mode(yw, self.Zt[nn[:, j]].copy())
                if bz is None:
                    bz, bp = z, ph
                else:
                    t = ph < bp
                    bz = np.where(t[:, None], z, bz)
                    bp = np.where(t, ph, bp)
            ch.append(2.0 * (bp + bz @ self.c) / Yw.shape[1])
            if self.est == 'mode':
                out[s:s + bs] = self.fz(bz)
            else:
                km, ess, bad, hd = self._post(yw, bz, rng)
                out[s:s + bs] = km
                fb.append(bad)
                es.append(ess)
                info.append(hd)
        self.diagnostics['median_chi2_over_m'] = float(np.median(np.concatenate(ch)))
        if fb:
            self.diagnostics['fallback_rate'] = float(np.mean(np.concatenate(fb)))
            self.diagnostics['mean_ess'] = float(np.mean(np.concatenate(es)))
            self.diagnostics['median_gn_info_diag'] = [float(v) for v in np.median(np.vstack(info), 0)]
        return np.clip(out, np.exp(self.lo), np.exp(self.hi))

def build_method(params, context):
    return SurrogateLaplace(params, context)
