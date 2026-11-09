import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import cho_factor, cho_solve

class LinRBFKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=0.5, lin_weight=1.0):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
        self.lin_weight = lin_weight
    def _k(self, A, B):
        sqA = (A**2).sum(1)[:, None]
        sqB = (B**2).sum(1)[None, :]
        d2 = np.maximum(sqA + sqB - 2 * A @ B.T, 0.0)
        return np.exp(-self.gamma_ * d2) + self.lin_weight * (A @ B.T) / A.shape[1]
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.X_ = X
        v = X.var()
        self.gamma_ = self.gamma_mult / (X.shape[1] * (v if v > 0 else 1.0))
        self.ymean_ = y.mean(0)
        K = self._k(X, X)
        K[np.diag_indices_from(K)] += self.alpha
        c = cho_factor(K, lower=True)
        self.coef_ = cho_solve(c, y - self.ymean_)
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._k(X, self.X_) @ self.coef_ + self.ymean_

def build_model(params, task, seed):
    return LinRBFKRR(alpha=params.get('alpha', 1e-3), gamma_mult=params.get('gamma_mult', 0.5), lin_weight=params.get('lin_weight', 1.0))
