import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeResidualGBM(BaseEstimator, RegressorMixin):
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
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, y)
        R = y - self.ridge_.predict(X).reshape(y.shape)
        lin = self.ridge_.predict(X).reshape(y.shape)
        Xa = np.hstack([X, lin])
        self.boosters_ = []
        for j in range(y.shape[1]):
            b = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=self.seed + j)
            b.fit(Xa, R[:, j])
            self.boosters_.append(b)
        self.n_out_ = y.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        lin = self.ridge_.predict(X).reshape(X.shape[0], self.n_out_)
        Xa = np.hstack([X, lin])
        corr = np.column_stack([b.predict(Xa) for b in self.boosters_])
        out = lin + corr
        return out if self.n_out_ > 1 else out.ravel()

def build_model(params, task, seed):
    return RidgeResidualGBM(alpha=params.get('alpha', 0.1), learning_rate=params.get('learning_rate', 0.05), max_iter=params.get('max_iter', 400), max_leaf_nodes=params.get('max_leaf_nodes', 15), l2_regularization=params.get('l2_regularization', 1.0), seed=seed)
