import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

def _feats(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    scale = np.median(np.abs(X)) + 1e-12
    L = np.log(np.clip(np.abs(X), 1e-3 * scale, None))
    out = [X, L]
    if X.shape[1] == 80:
        for shape in [(4, 20), (20, 4)]:
            A = L.reshape(n, *shape)
            B = A if shape == (4, 20) else A.transpose(0, 2, 1)
            out.append(np.diff(B, axis=2).reshape(n, -1))
            out.append(np.diff(B, axis=1).reshape(n, -1))
    return np.hstack(out)

class PhysKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
    def fit(self, X, y):
        F = _feats(X)
        self.sc_ = StandardScaler().fit(F)
        Fs = np.nan_to_num(self.sc_.transform(F))
        g = self.gamma_mult / Fs.shape[1]
        self.m_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g).fit(Fs, y)
        return self
    def predict(self, X):
        Fs = np.nan_to_num(self.sc_.transform(_feats(X)))
        return self.m_.predict(Fs)

def build_model(params, task, seed):
    return PhysKRR(alpha=float(params.get('alpha', 1e-3)), gamma_mult=float(params.get('gamma_mult', 0.5)))
