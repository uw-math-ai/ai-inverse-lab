import numpy as np

class M:
    def F(s, z):
        return np.asarray(s.c.forward(np.exp(z)), float).reshape(len(z), -1)
    def J(s, z, f0):
        n, p = z.shape
        h = np.where(z + 1e-4 > s.hi, -1e-4, 1e-4)
        Z = (z[:, None, :] + np.eye(p)[None] * h[:, None, :]).reshape(n * p, p)
        Fp = s.F(Z).reshape(n, p, -1)
        return ((Fp - f0[:, None, :]) / h[:, :, None]).transpose(0, 2, 1)
    def lm(s, z, Y):
        p = z.shape[1]
        lam = np.full(len(z), 1e-2)
        f = s.F(z)
        r = ((f - Y) ** 2).sum(1)
        for _ in range(s.it):
            Jm = s.J(z, f)
            g = np.einsum('nmp,nm->np', Jm, Y - f)
            H = np.einsum('nmp,nmq->npq', Jm, Jm)
            D = np.einsum('npp->np', H) + 1e-12
            A = H + (lam[:, None] * D)[:, :, None] * np.eye(p)
            zn = np.clip(z + np.linalg.solve(A, g[..., None])[..., 0], s.lo, s.hi)
            fn = s.F(zn)
            rn = ((fn - Y) ** 2).sum(1)
            ok = rn < r
            z[ok] = zn[ok]
            f[ok] = fn[ok]
            r[ok] = rn[ok]
            lam = np.clip(np.where(ok, lam * 0.3, lam * 10.0), 1e-7, 1e7)
        return z, f, r
    def block(s, Y):
        n = len(Y)
        p = len(s.lo)
        z0 = np.tile(0.5 * (s.lo + s.hi), (n, 1))
        za, fa, ra = s.lm(z0, Y)
        zb, fb, rb = s.lm(np.clip(za[:, ::-1].copy(), s.lo, s.hi), Y)
        w = rb < ra - 1e-12
        z = np.where(w[:, None], zb, za)
        f = np.where(w[:, None], fb, fa)
        Jm = s.J(z, f)
        H = np.einsum('nmp,nmq->npq', Jm, Jm) / s.sig ** 2 + 1e-6 * np.eye(p)
        L = np.linalg.cholesky(np.linalg.inv(H))
        eps = s.rng.standard_normal((n, s.ns, p))
        eps[:, 0] = 0.0
        Zs = z[:, None, :] + s.infl * np.einsum('nsp,nqp->nsq', eps, L)
        inb = np.all((Zs >= s.lo) & (Zs <= s.hi), axis=2)
        Zc = np.clip(Zs, s.lo, s.hi)
        Fs = s.F(Zc.reshape(-1, p)).reshape(n, s.ns, -1)
        ll = -0.5 * ((Fs - Y[:, None, :]) ** 2).sum(2) / s.sig ** 2 + 0.5 * (eps ** 2).sum(2)
        ll = np.where(inb, ll, -np.inf)
        K = np.exp(Zc)
        out = {'mode': np.exp(z)}
        for nm, a in (('mean', ll + Zc.sum(2)), ('mean_noJ', ll)):
            a = a - a.max(1, keepdims=True)
            wt = np.exp(a)
            wt = wt / wt.sum(1, keepdims=True)
            out[nm] = np.einsum('ns,nsp->np', wt, K)
            if nm == 'mean':
                q = wt / (K ** 2).sum(2)
                out['ratio'] = np.einsum('ns,nsp->np', q, K) / q.sum(1, keepdims=True)
                s.ess.append(1.0 / (wt ** 2).sum(1))
        s.mw.append(w)
        return out
    def run(s, X):
        X = np.asarray(X, float)
        res = {}
        for i in range(0, len(X), s.ch):
            o = s.block(X[i:i + s.ch])
            for k in o:
                res.setdefault(k, []).append(o[k])
        return {k: np.concatenate(v) for k, v in res.items()}
    def fit(s, X, y):
        nf = min(s.nfit, len(X))
        o = s.run(np.asarray(X, float)[:nf])
        yy = np.asarray(y, float)[:nf]
        sc = {}
        for k in o:
            sc[k] = float(np.mean(np.linalg.norm(o[k] - yy, axis=1) / np.linalg.norm(yy, axis=1)))
        s.scores = sc
        s.choice = min(sc, key=sc.get)
        s.fit_ess = np.concatenate(s.ess)
        s.fit_mirror_win = float(np.mean(np.concatenate(s.mw)))
        s.ess = []
        s.mw = []
        return s
    def predict(s, X):
        o = s.run(X)
        s.last = o
        return np.asarray(s.c.clip(o[s.choice]), float)

def build_method(params, context):
    m = M()
    m.c = context
    m.lo = np.log(np.asarray(context.lower, float))
    m.hi = np.log(np.asarray(context.upper, float))
    m.sig = float(context.noise_std)
    m.it = int(params.get('iters', 20))
    m.ns = int(params.get('ns', 96))
    m.infl = float(params.get('infl', 1.2))
    m.ch = int(params.get('chunk', 100))
    m.nfit = int(params.get('nfit', 150))
    m.rng = np.random.default_rng(0)
    m.ess = []
    m.mw = []
    m.choice = 'mean'
    return m
