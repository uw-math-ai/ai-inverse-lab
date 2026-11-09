import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeBoost(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, seed=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.seed = seed
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        self.lin_ = Ridge(alpha=self.alpha).fit(X, y)
        r = y - self.lin_.predict(X).reshape(y.shape)
        Z = np.hstack([X, self.lin_.predict(X).reshape(y.shape)])
        self.boosters_ = []
        for j in range(y.shape[1]):
            b = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=self.seed)
            b.fit(Z, r[:, j])
            self.boosters_.append(b)
        self.ny_ = y.shape[1]
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        base = self.lin_.predict(X).reshape(X.shape[0], self.ny_)
        Z = np.hstack([X, base])
        corr = np.column_stack([b.predict(Z) for b in self.boosters_])
        out = base + corr
        return out[:, 0] if self.ny_ == 1 else out

def build_model(params, task, seed):
    return RidgeBoost(alpha=params.get('alpha', 0.1), learning_rate=params.get('learning_rate', 0.05), max_iter=params.get('max_iter', 400), max_leaf_nodes=params.get('max_leaf_nodes', 15), l2_regularization=params.get('l2_regularization', 1.0), seed=seed)
