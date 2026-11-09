import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

class PhysResidualStack(BaseEstimator, RegressorMixin):
    def __init__(self, ridge_alpha=1e-3, krr_alpha=1e-3, gamma_mult=0.5, seed=0):
        self.ridge_alpha = ridge_alpha
        self.krr_alpha = krr_alpha
        self.gamma_mult = gamma_mult
        self.seed = seed

    def _logf(self, X):
        return np.log(np.abs(X) + self.eps_)

    def _std(self, A, key):
        return (A - self.mu_[key]) / self.sd_[key]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        med = np.median(np.abs(X))
        self.eps_ = 0.01 * med if med > 0 else 1e-6
        L = self._logf(X)
        self.mu_ = {'r': X.mean(0), 'l': L.mean(0)}
        self.sd_ = {'r': X.std(0) + 1e-12, 'l': L.std(0) + 1e-12}
        Ls = self._std(L, 'l')
        Xs = self._std(X, 'r')
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit(Ls, y)
        res = y - self.ridge_.predict(Ls)
        Z = np.hstack([Xs, Ls])
        gamma = self.gamma_mult / (Z.shape[1] * Z.var())
        self.krr_ = KernelRidge(alpha=self.krr_alpha, kernel='rbf', gamma=gamma).fit(Z, res)
        self.single_ = (np.asarray(y).shape[1] == 1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        L = self._logf(X)
        Ls = self._std(L, 'l')
        Xs = self._std(X, 'r')
        Z = np.hstack([Xs, Ls])
        p = self.ridge_.predict(Ls) + self.krr_.predict(Z)
        if p.ndim == 1:
            p = p[:, None]
        return p[:, 0] if self.single_ else p

def build_model(params, task, seed):
    return PhysResidualStack(ridge_alpha=float(params.get('ridge_alpha', 1e-3)), krr_alpha=float(params.get('krr_alpha', 1e-3)), gamma_mult=float(params.get('gamma_mult', 0.5)), seed=seed)
