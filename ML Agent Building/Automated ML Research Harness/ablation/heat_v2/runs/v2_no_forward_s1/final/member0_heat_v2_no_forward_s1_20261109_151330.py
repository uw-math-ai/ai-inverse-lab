import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge

class SurrInv:
    def __init__(self, p):
        self.nc = int(p.get('n_components', 40))
        self.iters = int(p.get('iters', 15))
        self.lam0 = float(p.get('lam', 0.01))
        self.lo = float(p.get('lo', np.log(0.5)))
        self.hi = float(p.get('hi', np.log(2.0)))
        self.sig = float(p.get('sig', 1e-3))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        z = (np.log(np.clip(y, 1e-9, None)) - self.lo) / (self.hi - self.lo)
        nc = min(self.nc, X.shape[1], X.shape[0] - 1)
        self.pca = PCA(n_components=nc).fit(X)
        S = self.pca.transform(X) / self.sig
        n = len(z)
        idx = np.random.RandomState(0).permutation(n)
        m = int(0.8 * n)
        tr, va = idx[:m], idx[m:]
        best = None
        for g in (1.0, 3.0, 10.0):
            for a in (1e-4, 1e-3, 1e-2, 1e-1):
                f = KernelRidge(alpha=a, kernel='rbf', gamma=g).fit(z[tr], S[tr])
                ev = np.mean((f.predict(z[va]) - S[va]) ** 2, axis=0)
                if best is None or ev.sum() < best[0].sum():
                    best = (ev, g, a)
        self.surr_var_ = best[0]
        self.gate_pass_ = bool(np.all(best[0] <= 2.25))
        self.sw = 1.0 / np.sqrt(np.maximum(best[0], 1.0))
        self.fwd = KernelRidge(alpha=best[2], kernel='rbf', gamma=best[1]).fit(z, S)
        self.k = min(20, S.shape[1])
        sc = S[:, :self.k]
        self.ssd = sc.std(axis=0) + 1e-12
        self.inv = KernelRidge(alpha=1e-3, kernel='rbf', gamma=1.0 / self.k).fit(sc / self.ssd, z)
        return self

    def predict(self, X):
        S = self.pca.transform(np.asarray(X, float)) / self.sig
        z = np.clip(self.inv.predict(S[:, :self.k] / self.ssd), 0.0, 1.0)
        n, p = z.shape
        E = np.eye(p)
        h = 1e-3
        f0 = self.fwd.predict(z)
        r = (f0 - S) * self.sw
        cost = np.sum(r ** 2, axis=1)
        lam = np.full(n, self.lam0)
        for it in range(self.iters):
            J = np.stack([(self.fwd.predict(z + h * E[j]) - f0) / h for j in range(p)], axis=2) * self.sw[None, :, None]
            A = np.einsum('ndi,ndj->nij', J, J) + lam[:, None, None] * E[None]
            g = np.einsum('ndi,nd->ni', J, r)
            step = np.linalg.solve(A, g[:, :, None])[:, :, 0]
            zn = np.clip(z - step, 0.0, 1.0)
            fn = self.fwd.predict(zn)
            rn = (fn - S) * self.sw
            cn = np.sum(rn ** 2, axis=1)
            acc = cn < cost
            z[acc] = zn[acc]
            f0[acc] = fn[acc]
            r[acc] = rn[acc]
            cost[acc] = cn[acc]
            lam = np.where(acc, lam * 0.3, lam * 10.0)
            lam = np.clip(lam, 1e-6, 1e6)
        return np.exp(self.lo + z * (self.hi - self.lo))

def build_method(params, context):
    return SurrInv(params)
