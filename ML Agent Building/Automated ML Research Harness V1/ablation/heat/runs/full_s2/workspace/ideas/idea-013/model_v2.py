import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


class RidgeBoost(RegressorMixin, BaseEstimator):
    def __init__(self, params=None, seed=0):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        p = dict(self.params or {})
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.single_ = (y.ndim == 1)
        if y.ndim == 1:
            y = y[:, None]
        self.ridge_ = Ridge(alpha=float(p.get('alpha', 1.0))).fit(X, y)
        base = np.asarray(self.ridge_.predict(X)).reshape(y.shape)
        res = y - base
        self.boosters_ = []
        for j in range(y.shape[1]):
            m = HistGradientBoostingRegressor(
                learning_rate=float(p.get('learning_rate', 0.05)),
                max_iter=int(p.get('max_iter', 300)),
                max_leaf_nodes=int(p.get('max_leaf_nodes', 15)),
                l2_regularization=float(p.get('l2_regularization', 1.0)),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.seed,
            )
            m.fit(X, res[:, j])
            self.boosters_.append(m)
        self.n_outputs_ = y.shape[1]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        base = np.asarray(self.ridge_.predict(X))
        if base.ndim == 1:
            base = base[:, None]
        corr = np.column_stack([m.predict(X) for m in self.boosters_])
        out = base + corr
        return out[:, 0] if self.single_ else out


def build_model(params, task, seed):
    return RidgeBoost(params=dict(params or {}), seed=seed)
