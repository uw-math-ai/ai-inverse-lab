import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeResidualGBM(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=0.1, seed=0):
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
        Y = y.reshape(len(y), -1)
        self.lin_ = Ridge(alpha=self.alpha).fit(X, Y)
        R = Y - self.lin_.predict(X).reshape(Y.shape)
        L = self.lin_.predict(X).reshape(Y.shape)
        Xa = np.hstack([X, L])
        self.gbms_ = []
        for j in range(Y.shape[1]):
            m = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=self.l2_regularization, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=self.seed + j)
            m.fit(Xa, R[:, j])
            self.gbms_.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        L = self.lin_.predict(X).reshape(len(X), -1)
        Xa = np.hstack([X, L])
        P = L + np.column_stack([m.predict(Xa) for m in self.gbms_])
        return P.ravel() if self._1d else P

def build_model(params, task, seed):
    return RidgeResidualGBM(alpha=params.get('alpha', 1.0), learning_rate=params.get('learning_rate', 0.05), max_iter=params.get('max_iter', 300), max_leaf_nodes=params.get('max_leaf_nodes', 15), l2_regularization=params.get('l2_regularization', 0.1), seed=seed)
