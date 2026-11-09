import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

def _phys(X):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    L = np.log(np.clip(np.abs(X), 1e-4, None))
    feats = [X, L]
    if d % 4 == 0:
        T = d // 4
        for A in (L.reshape(n, 4, T), L.reshape(n, T, 4).transpose(0, 2, 1)):
            dt = np.diff(A, axis=2)
            ds = np.diff(A, axis=1)
            feats += [dt.reshape(n, -1), ds.reshape(n, -1), dt.mean(axis=2)]
    return np.hstack(feats)

class PhysKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
    def fit(self, X, y):
        F = _phys(X)
        self.sc_ = StandardScaler().fit(F)
        Z = np.nan_to_num(self.sc_.transform(F))
        g = self.gamma_mult / Z.shape[1]
        self.m_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g).fit(Z, y)
        return self
    def predict(self, X):
        Z = np.nan_to_num(self.sc_.transform(_phys(X)))
        return self.m_.predict(Z)

def build_model(params, task, seed):
    return PhysKRR(alpha=params.get('alpha', 1e-3), gamma_mult=params.get('gamma_mult', 0.5))
