import numpy as np

class LocalGN:
    def __init__(self, p):
        self.Ks = [int(k) for k in p.get('Ks', [24, 48, 96])]
        self.K = int(p.get('K', 48))
        self.it = int(p.get('iters', 2))
        self.llo = np.log(float(p.get('lo', 0.5)))
        self.lhi = np.log(float(p.get('hi', 2.0)))
        self.s2 = float(p.get('sigma', 1e-3)) ** 2
        self.lam = float(p.get('lam', 1.0))
        self.nh = int(p.get('n_hold', 250))
        self.mmse = bool(p.get('mmse', False))
        self.ns = int(p.get('n_samples', 256))

    def _store(self, X, Y):
        self.X = X
        self.Y = Y
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-12
        self.Z = (X - self.mu) / self.sd
        self.tsd = Y.std(0) + 1e-12

    def fit(self, X, y):
        X = np.asarray(X, float)
        Y = np.log(np.asarray(y, float))
        n = len(X)
        if len(self.Ks) > 1 and n >= self.nh + 4 * max(self.Ks):
            perm = np.random.RandomState(0).permutation(n)
            h, r = perm[:self.nh], perm[self.nh:]
            self._store(X[r], Y[r])
            kh = np.exp(Y[h])
            best, bk = np.inf, self.K
            for K in self.Ks:
                self.K = K
                e = np.linalg.norm(self.predict(X[h]) - kh) / np.linalg.norm(kh)
                if e < best:
                    best, bk = e, K
            self.K = bk
        self._store(X, Y)
        return self

    def _nn(self, d):
        K = min(self.K, len(d) - 1)
        return np.argpartition(d, K)[:K]

    def _step(self, x, idx):
        T = self.Y[idx]
        t0 = T.mean(0)
        A = np.hstack([np.ones((len(idx), 1)), T - t0])
        C = np.linalg.lstsq(A, self.X[idx], rcond=None)[0]
        R = self.X[idx] - A @ C
        dof = max(len(idx) - A.shape[1], 1)
        v = np.maximum((R ** 2).sum(0) / dof, self.s2)
        B = C[1:]
        W = B / v
        H = W @ B.T + self.lam * np.eye(B.shape[0])
        return t0 + np.linalg.solve(H, W @ (x - C[0])), H

    def _post(self, t, H, rng):
        tc = np.clip(t, self.llo, self.lhi)
        if not self.mmse:
            return np.exp(tc)
        S = np.linalg.inv(H)
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(len(t))
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return np.exp(tc)
        Q = t + rng.standard_normal((self.ns, len(t))) @ L.T
        ok = np.all((Q >= self.llo) & (Q <= self.lhi), axis=1)
        if ok.sum() < 4:
            return np.exp(tc)
        Q = Q[ok]
        s = Q.sum(1)
        w = np.exp(s - s.max())
        return (w[:, None] * np.exp(Q)).sum(0) / w.sum()

    def predict(self, X):
        X = np.asarray(X, float)
        Zq = (X - self.mu) / self.sd
        rng = np.random.RandomState(1)
        out = np.zeros((len(X), self.Y.shape[1]))
        for i in range(len(X)):
            t, H = self._step(X[i], self._nn(((self.Z - Zq[i]) ** 2).sum(1)))
            for _ in range(self.it):
                tc = np.clip(t, self.llo, self.lhi)
                t, H = self._step(X[i], self._nn((((self.Y - tc) / self.tsd) ** 2).sum(1)))
            out[i] = self._post(t, H, rng)
        return out

def build_method(params, context):
    return LocalGN(params)
