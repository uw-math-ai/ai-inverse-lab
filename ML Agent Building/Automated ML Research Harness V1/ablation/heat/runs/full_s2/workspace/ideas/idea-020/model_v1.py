import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

class LogKRRPerOutputCV(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=(1e-3, 3e-3, 1e-2), gmults=(0.15, 0.3, 0.5), seed=0):
        self.alphas = alphas
        self.gmults = gmults
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.squeeze_ = (y.ndim == 1)
        if self.squeeze_:
            y = y[:, None]
        self.shift_ = -y.min(axis=0) + 1.0
        z = np.log(y + self.shift_)
        self.zm_ = z.mean(axis=0)
        self.zs_ = z.std(axis=0) + 1e-12
        zn = (z - self.zm_) / self.zs_
        base = 1.0 / (X.shape[1] * max(X.var(), 1e-12))
        combos = [(a, g) for a in self.alphas for g in self.gmults]
        err = np.zeros((len(combos), y.shape[1]))
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            for ci, (a, g) in enumerate(combos):
                m = KernelRidge(alpha=a, kernel='rbf', gamma=g * base).fit(X[tr], zn[tr])
                p = np.exp(m.predict(X[va]) * self.zs_ + self.zm_) - self.shift_
                err[ci] += np.sum((p - y[va]) ** 2, axis=0)
        best = err.argmin(axis=0)
        self.models_ = []
        for ci in np.unique(best):
            cols = np.where(best == ci)[0]
            a, g = combos[ci]
            m = KernelRidge(alpha=a, kernel='rbf', gamma=g * base).fit(X, zn[:, cols])
            self.models_.append((cols, m))
        self.n_out_ = y.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        zn = np.zeros((X.shape[0], self.n_out_))
        for cols, m in self.models_:
            p = m.predict(X)
            if p.ndim == 1:
                p = p[:, None]
            zn[:, cols] = p
        out = np.exp(zn * self.zs_ + self.zm_) - self.shift_
        return out[:, 0] if self.squeeze_ else out

def build_model(params, task, seed):
    return LogKRRPerOutputCV(seed=seed, **params)
