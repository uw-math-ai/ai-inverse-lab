import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

class KRRPlusGBM(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, seed=0):
        self.params = params or {}
        self.seed = seed

    def _krr(self, X):
        p = self.params
        g = p.get('gamma_mult', 0.2) / (X.shape[1] * max(float(X.var()), 1e-12))
        return KernelRidge(kernel='rbf', alpha=p.get('alpha', 3e-4), gamma=g)

    def _gbm_in(self, X, b):
        return np.hstack([X, b]) if self.params.get('stack', True) else X

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if y.ndim == 1: y = y[:, None]
        p = self.params
        oof = np.zeros_like(y)
        kf = KFold(n_splits=int(p.get('n_folds', 4)), shuffle=True, random_state=self.seed)
        for tr, te in kf.split(X):
            m = self._krr(X[tr]).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te]).reshape(len(te), -1)
        self.base_ = self._krr(X).fit(X, y)
        R = y - oof
        Z = self._gbm_in(X, oof)
        self.gbms_ = []
        for j in range(y.shape[1]):
            g = HistGradientBoostingRegressor(learning_rate=p.get('learning_rate', 0.04), max_iter=int(p.get('max_iter', 300)), max_leaf_nodes=int(p.get('max_leaf_nodes', 7)), min_samples_leaf=int(p.get('min_samples_leaf', 40)), l2_regularization=p.get('l2_regularization', 3.0), early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=self.seed + j)
            g.fit(Z, R[:, j])
            self.gbms_.append(g)
        self.ny_ = y.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        b = self.base_.predict(X).reshape(X.shape[0], -1)
        Z = self._gbm_in(X, b)
        s = self.params.get('shrink', 0.7)
        corr = np.column_stack([g.predict(Z) for g in self.gbms_])
        out = b + s * corr
        return out[:, 0] if self.ny_ == 1 else out

def build_model(params, task, seed):
    return KRRPlusGBM(params=params, seed=seed)
