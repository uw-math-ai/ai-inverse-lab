import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeResidualGBM(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01, learning_rate=0.05, max_iter=300, max_leaf_nodes=15, seed=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._1d = (y.ndim == 1)
        Y = y.reshape(-1, 1) if self._1d else y
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, Y)
        R = Y - self.ridge_.predict(X).reshape(Y.shape)
        lin = self.ridge_.predict(X).reshape(Y.shape)
        Xa = np.hstack([X, lin])
        self.boosters_ = []
        for j in range(Y.shape[1]):
            b = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=self.seed + j)
            b.fit(Xa, R[:, j])
            self.boosters_.append(b)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        lin = self.ridge_.predict(X)
        lin = lin.reshape(X.shape[0], -1)
        Xa = np.hstack([X, lin])
        corr = np.column_stack([b.predict(Xa) for b in self.boosters_])
        out = lin + corr
        return out.ravel() if self._1d else out

def build_model(params: dict, task: str, seed: int):
    return RidgeResidualGBM(alpha=float(params.get('alpha', 0.01)), learning_rate=float(params.get('learning_rate', 0.05)), max_iter=int(params.get('max_iter', 300)), max_leaf_nodes=int(params.get('max_leaf_nodes', 15)), seed=seed)
