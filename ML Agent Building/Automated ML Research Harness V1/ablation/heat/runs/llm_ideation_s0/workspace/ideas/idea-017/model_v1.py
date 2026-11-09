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
        feats = [X]
        lg = np.log(np.abs(X) + 1e-3)
        feats.append(lg)
        for A in (X[:, :s*T].reshape(n, s, T), X[:, :s*T].reshape(n, T, s).transpose(0, 2, 1)):
            dt = np.diff(A, axis=2).reshape(n, -1)
            dx = np.diff(A, axis=1).reshape(n, -1)
            dxx = (A[:, 2:, :] - 2*A[:, 1:-1, :] + A[:, :-2, :]).reshape(n, -1) if s >= 3 else np.zeros((n, 0))
            la = np.log(np.abs(A) + 1e-3)
            rate = np.diff(la, axis=2).reshape(n, -1)
            feats += [dt, dx, dxx, rate]
        F = np.hstack(feats)
        F[~np.isfinite(F)] = 0.0
        return F
    def fit(self, X, y):
        F = self._feats(X)
        self.sc_ = StandardScaler().fit(F)
        Z = self.sc_.transform(F)
        g = self.gamma_mult / Z.shape[1]
        self.m_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g).fit(Z, y)
        return self
    def predict(self, X):
        return self.m_.predict(self.sc_.transform(self._feats(X)))

def build_model(params, task, seed):
    return PhysFeatKRR(alpha=float(params.get('alpha', 1e-3)), gamma_mult=float(params.get('gamma_mult', 0.5)), n_sensors=int(params.get('n_sensors', 4)))
