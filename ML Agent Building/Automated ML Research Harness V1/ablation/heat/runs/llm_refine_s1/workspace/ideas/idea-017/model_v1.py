import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x) / 1e-3)

def _feats(X):
    n, d = X.shape
    L = _slog(X)
    out = [X, L]
    S = 4
    T = d // S
    if S * T == d:
        for A in (L.reshape(n, S, T), L.reshape(n, T, S).transpose(0, 2, 1)):
            dt = np.diff(A, axis=2)
            d2 = np.diff(A, n=2, axis=2)
            ds = np.diff(A, axis=1)
            out += [dt.reshape(n, -1), d2.reshape(n, -1), ds.reshape(n, -1),
                    dt.mean(axis=2), dt[:, :, -max(1, T // 3):].mean(axis=2)]
    return np.hstack(out)

class PhysKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
    def fit(self, X, y):
        F = _feats(np.asarray(X, dtype=float))
        self.sc_ = StandardScaler().fit(F)
        Fs = np.nan_to_num(self.sc_.transform(F))
        g = self.gamma_mult / Fs.shape[1]
        self.m_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g).fit(Fs, y)
        return self
    def predict(self, X):
        Fs = np.nan_to_num(self.sc_.transform(_feats(np.asarray(X, dtype=float))))
        return self.m_.predict(Fs)

def build_model(params, task, seed):
    return PhysKRR(alpha=params.get('alpha', 1e-3), gamma_mult=params.get('gamma_mult', 0.5))
