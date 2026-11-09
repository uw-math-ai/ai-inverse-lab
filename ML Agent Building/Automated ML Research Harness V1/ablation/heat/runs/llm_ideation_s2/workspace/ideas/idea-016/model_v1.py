import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

class PhysFeatKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5, n_sensors=4):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
        self.n_sensors = n_sensors

    def _feats(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        s = self.n_sensors
        T = d // s
        eps = max(1e-6, 1e-3 * np.median(np.abs(X)))
        L = np.log(np.clip(np.abs(X), eps, None))
        A = L.reshape(n, s, T)  # assumed sensor-major
        B = L.reshape(n, T, s).transpose(0, 2, 1)  # alt interpretation (time-major)
        out = [L]
        for M in (A, B):
            dt = np.diff(M, axis=2)            # local decay rates d(log u)/dt
            dx = np.diff(M, axis=1)            # inter-sensor log-gradients
            late = dt[:, :, dt.shape[2] // 2:].mean(axis=2)  # principal-mode decay estimate
            early = dt[:, :, :max(1, dt.shape[2] // 4)].mean(axis=2)
            ratio = late[:, 1:] - late[:, :-1]
            out += [dt.reshape(n, -1), dx.reshape(n, -1), late, early, ratio]
        F = np.concatenate(out, axis=1)
        return np.nan_to_num(F)

    def fit(self, X, y):
        F = self._feats(X)
        self.sc_ = StandardScaler().fit(F)
        Z = self.sc_.transform(F)
        gamma = self.gamma_mult / (Z.shape[1] * max(Z.var(), 1e-12))
        self.m_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=gamma).fit(Z, y)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(self._feats(X)))

def build_model(params, task, seed):
    return PhysFeatKRR(alpha=params.get('alpha', 1e-3), gamma_mult=params.get('gamma_mult', 0.5), n_sensors=int(params.get('n_sensors', 4)))
