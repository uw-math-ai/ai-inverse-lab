import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

class MultiScaleLogKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=3e-3, gmults=(0.15, 0.3, 0.6), seed=0):
        self.alpha = alpha
        self.gmults = gmults
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
            self.flat_ = True
        else:
            self.flat_ = False
        self.shift_ = -y.min(axis=0) + 1.0
        z = np.log(y + self.shift_)
        self.zmean_ = z.mean(axis=0)
        zc = z - self.zmean_
        base = 1.0 / (X.shape[1] * max(X.var(), 1e-12))
        self.models_ = []
        for g in self.gmults:
            m = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g * base)
            m.fit(X, zc)
            self.models_.append(m)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        zp = np.mean([m.predict(X) for m in self.models_], axis=0)
        if zp.ndim == 1:
            zp = zp[:, None]
        yp = np.exp(zp + self.zmean_) - self.shift_
        return yp.ravel() if self.flat_ else yp

def build_model(params, task, seed):
    return MultiScaleLogKRR(alpha=params.get('alpha', 3e-3), gmults=tuple(params.get('gmults', (0.15, 0.3, 0.6))), seed=seed)
