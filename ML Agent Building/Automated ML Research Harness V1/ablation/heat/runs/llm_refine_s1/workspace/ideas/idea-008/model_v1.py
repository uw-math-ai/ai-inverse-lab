import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

class RidgePlusGBM(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, seed=0):
        self.alpha = alpha; self.learning_rate = learning_rate; self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes; self.l2_regularization = l2_regularization; self.seed = seed
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if y.ndim == 1: y = y[:, None]
        oof = np.zeros_like(y, dtype=float)
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(X):
            oof[va] = Ridge(alpha=self.alpha).fit(X[tr], y[tr]).predict(X[va]).reshape(len(va), -1)
        self.lin_ = Ridge(alpha=self.alpha).fit(X, y)
        R = y - oof
        Z = np.hstack([X, oof])
        self.gbms_ = []
        for j in range(y.shape[1]):
            g = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=self.seed)
            g.fit(Z, R[:, j]); self.gbms_.append(g)
        self.ndim_ = y.shape[1]
        return self
    def predict(self, X):
        X = np.asarray(X)
        base = self.lin_.predict(X).reshape(len(X), -1)
        Z = np.hstack([X, base])
        corr = np.column_stack([g.predict(Z) for g in self.gbms_])
        out = base + corr
        return out[:, 0] if self.ndim_ == 1 else out

def build_model(params, task, seed):
    return RidgePlusGBM(alpha=params.get('alpha', 1.0), learning_rate=params.get('learning_rate', 0.05), max_iter=params.get('max_iter', 400), max_leaf_nodes=params.get('max_leaf_nodes', 15), l2_regularization=params.get('l2_regularization', 1.0), seed=seed)
