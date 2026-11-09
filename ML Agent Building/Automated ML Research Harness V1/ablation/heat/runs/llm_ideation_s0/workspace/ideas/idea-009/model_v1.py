import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeResidualGBM(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, learning_rate=0.05, max_iter=300, max_leaf_nodes=15, seed=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        self.ridge_ = Ridge(alpha=self.alpha).fit(X, y)
        R = y - self.ridge_.predict(X).reshape(y.shape)
        Xa = np.hstack([X, self.ridge_.predict(X).reshape(y.shape)])
        self.gbms_ = []
        for j in range(y.shape[1]):
            m = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1, random_state=self.seed)
            m.fit(Xa, R[:, j])
            self.gbms_.append(m)
        self.ny_ = y.shape[1]
        return self
    def predict(self, X):
        base = self.ridge_.predict(X).reshape(X.shape[0], self.ny_)
        Xa = np.hstack([X, base])
        corr = np.column_stack([m.predict(Xa) for m in self.gbms_])
        out = base + corr
        return out[:, 0] if self.ny_ == 1 else out

def build_model(params, task, seed):
    return RidgeResidualGBM(alpha=params.get('alpha', 1.0), learning_rate=params.get('learning_rate', 0.05), max_iter=params.get('max_iter', 300), max_leaf_nodes=params.get('max_leaf_nodes', 15), seed=seed)
