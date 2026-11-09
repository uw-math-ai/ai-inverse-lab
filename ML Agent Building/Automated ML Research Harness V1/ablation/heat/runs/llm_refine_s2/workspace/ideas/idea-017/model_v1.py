import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

class PhysFeatKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5, n_sensors=4):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
        self.n_sensors = n_sensors

    def _feats(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        S = self.n_sensors
        T = d // S
        L = np.log(np.abs(X) + self.eps_)
        out = [L]
        for A in (L[:, :S*T].reshape(n, S, T), L[:, :S*T].reshape(n, T, S).transpose(0, 2, 1)):
            out.append(np.diff(A, axis=2).reshape(n, -1))
            out.append(np.diff(A, axis=1).reshape(n, -1))
        for R in (X[:, :S*T].reshape(n, S, T), X[:, :S*T].reshape(n, T, S).transpose(0, 2, 1)):
            I = R.mean(axis=2)
            out.append(I)
            out.append(np.log(np.abs(I) + self.eps_))
        return np.hstack(out)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.eps_ = 1e-3 * (np.abs(X).max() + 1e-12)
        F = self._feats(X)
        self.sc_ = StandardScaler().fit(F)
        F = self.sc_.transform(F)
        self.krr_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma_mult / F.shape[1])
        self.krr_.fit(F, y)
        return self

    def predict(self, X):
        F = self.sc_.transform(self._feats(X))
        return self.krr_.predict(F)

def build_model(params, task, seed):
    return PhysFeatKRR(alpha=params.get('alpha', 1e-3), gamma_mult=params.get('gamma_mult', 0.5), n_sensors=params.get('n_sensors', 4))
