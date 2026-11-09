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
        self._1d = (y.ndim == 1)
        Y = y.reshape(-1, 1) if self._1d else y
        self.lin_ = Ridge(alpha=self.alpha).fit(X, Y)
        R = Y - self.lin_.predict(X)
        self.boosters_ = []
        for j in range(R.shape[1]):
            b = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=self.seed + j)
            b.fit(X, R[:, j])
            self.boosters_.append(b)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        P = self.lin_.predict(X)
        P = P.reshape(len(X), -1)
        for j, b in enumerate(self.boosters_):
            P[:, j] = P[:, j] + b.predict(X)
        return P[:, 0] if self._1d else P

def build_model(params, task, seed):
    return RidgeBoost(alpha=float(params.get('alpha', 0.1)), learning_rate=float(params.get('learning_rate', 0.05)), max_iter=int(params.get('max_iter', 400)), max_leaf_nodes=int(params.get('max_leaf_nodes', 15)), l2_regularization=float(params.get('l2_regularization', 1.0)), seed=seed)
