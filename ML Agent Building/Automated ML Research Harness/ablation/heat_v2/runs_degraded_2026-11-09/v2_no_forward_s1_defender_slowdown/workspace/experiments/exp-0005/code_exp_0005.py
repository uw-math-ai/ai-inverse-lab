import numpy as np

def _sq(A, B):
    return np.maximum((A*A).sum(1)[:, None] + (B*B).sum(1)[None, :] - 2.0*A.dot(B.T), 0.0)

class PCAKRR:
    def _w(self, m, mode):
        ev = self.ev[:m]
        if mode == 'wiener':
            return np.sqrt(np.maximum(ev - self.sig**2, 1e-12)) / ev
        return 1.0 / np.sqrt(ev)
    def _z(self, X, m, w):
        return (np.asarray(X, float) - self.mu).dot(self.Vt[:m].T) * w
    def fit(self, X, y):
        X = np.asarray(X, float); Y = np.log(np.asarray(y, float))
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.Vt = Vt; self.ev = S**2 / (X.shape[0] - 1)
        n = X.shape[0]; idx = np.random.RandomState(self.seed).permutation(n)
        va = idx[:n // 3]; tr = idx[n // 3:]; ym = Y[tr].mean(0); T = np.exp(Y[va])
        best = (np.inf, None)
        for fac in [3.0, 10.0, 30.0]:
            m = int(np.sum(self.ev > fac * self.sig**2)); m = int(min(max(m, 4), self.maxpc, len(self.ev)))
            for mode in ['whiten', 'wiener']:
                Z = self._z(X, m, self._w(m, mode))
                Dtr = _sq(Z[tr], Z[tr]); Dva = _sq(Z[va], Z[tr]); med = np.median(Dtr) + 1e-12
                for gm in [0.1, 0.3, 1.0, 3.0]:
                    g = gm / med
                    L, Q = np.linalg.eigh(np.exp(-g * Dtr)); L = np.maximum(L, 0.0)
                    QtY = Q.T.dot(Y[tr] - ym); Kq = np.exp(-g * Dva).dot(Q)
                    for a in [1e-5, 1e-4, 1e-3, 1e-2]:
                        P = np.exp(Kq.dot(QtY / (L + a)[:, None]) + ym)
                        e = np.linalg.norm(P - T) / np.linalg.norm(T)
                        if e < best[0]:
                            best = (e, (m, mode, g * med, a, fac))
        m, mode, gm, a, fac = best[1]
        self.m = m; self.mode = mode; self.a = a; self.fac = fac; self.cv_err = best[0]
        self.w = self._w(m, mode); self.Z = self._z(X, m, self.w)
        D = _sq(self.Z, self.Z); self.g = gm / (np.median(D) + 1e-12)
        self.ym = Y.mean(0)
        self.c = np.linalg.solve(np.exp(-self.g * D) + a * np.eye(n), Y - self.ym)
        return self
    def predict(self, X):
        Zq = self._z(X, self.m, self.w)
        return np.exp(np.exp(-self.g * _sq(Zq, self.Z)).dot(self.c) + self.ym)

def build_model(params, task, seed):
    m = PCAKRR(); m.sig = float(params.get('sigma', 1e-3)); m.maxpc = int(params.get('maxpc', 64)); m.seed = int(seed)
    return m
