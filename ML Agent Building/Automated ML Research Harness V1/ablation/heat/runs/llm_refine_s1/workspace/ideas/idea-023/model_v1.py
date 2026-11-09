import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

class BasePlusGBM(BaseEstimator, RegressorMixin):
    def __init__(self, base='ridge', alpha=1.0, gamma_mult=0.3, learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, n_folds=5, seed=0):
        self.base = base
        self.alpha = alpha
        self.gamma_mult = gamma_mult
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.n_folds = n_folds
        self.seed = seed

    def _make_base(self):
        if self.base == 'krr':
            return KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma_)
        return Ridge(alpha=self.alpha)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._1d = (y.ndim == 1)
        if self._1d:
            y = y[:, None]
        v = float(X.var())
        self.gamma_ = self.gamma_mult / (X.shape[1] * (v if v > 0 else 1.0))
        oof = np.zeros_like(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            m = self._make_base()
            m.fit(X[tr], y[tr])
            oof[va] = np.asarray(m.predict(X[va])).reshape(len(va), -1)
        self.base_ = self._make_base()
        self.base_.fit(X, y)
        Z = np.hstack([X, oof])
        R = y - oof
        self.gbms_ = []
        for j in range(y.shape[1]):
            g = HistGradientBoostingRegressor(learning_rate=self.learning_rate, max_iter=self.max_iter, max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization, early_stopping=self.early_stopping, validation_fraction=0.1, n_iter_no_change=30, random_state=self.seed + j)
            g.fit(Z, R[:, j])
            self.gbms_.append(g)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        b = np.asarray(self.base_.predict(X)).reshape(len(X), -1)
        Z = np.hstack([X, b])
        r = np.column_stack([g.predict(Z) for g in self.gbms_])
        out = b + r
        return out[:, 0] if self._1d else out

def build_model(params, task, seed):
    return BasePlusGBM(seed=seed, **params)
