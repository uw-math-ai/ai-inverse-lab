import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def _phys(X):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    L = np.log(np.abs(X) + 1e-3)
    feats = [X, L]
    if d % 4 == 0:
        T = d // 4
        for A in (L.reshape(n, 4, T), L.reshape(n, T, 4).transpose(0, 2, 1)):
            dt = np.diff(A, axis=2)
            ds = np.diff(A, axis=1)
            feats.append(dt.reshape(n, -1))
            feats.append(ds.reshape(n, -1))
            feats.append(dt.mean(axis=2))
    F = np.concatenate(feats, axis=1)
    return np.nan_to_num(F)

class PhysMLP(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(256, 256), alpha=1e-4, lr=1e-3, seed=0):
        self.hidden = hidden; self.alpha = alpha; self.lr = lr; self.seed = seed
    def fit(self, X, y):
        F = _phys(X)
        self.sc_ = StandardScaler().fit(F)
        self.m_ = MLPRegressor(hidden_layer_sizes=tuple(self.hidden), alpha=self.alpha, learning_rate_init=self.lr, max_iter=400, early_stopping=True, n_iter_no_change=20, random_state=self.seed)
        self.m_.fit(self.sc_.transform(F), y)
        return self
    def predict(self, X):
        return self.m_.predict(self.sc_.transform(_phys(X)))

def build_model(params, task, seed):
    return PhysMLP(hidden=params.get('hidden', [256, 256]), alpha=params.get('alpha', 1e-4), lr=params.get('lr', 1e-3), seed=seed)
