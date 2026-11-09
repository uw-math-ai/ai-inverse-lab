import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

class PhysFeatKRR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, gamma_mult=1.0, eps=1e-3):
        self.alpha = alpha
        self.gamma_mult = gamma_mult
        self.eps = eps

    def _feat(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        L = np.log(np.clip(np.abs(X), self.eps, None))
        feats = [X, L]
        if X.shape[1] == 80:
            layouts = [L.reshape(n, 4, 20), L.reshape(n, 20, 4).transpose(0, 2, 1)]
        else:
            layouts = []
        for A in layouts:
            T = A.shape[2]
            t = np.arange(T, dtype=float)
            tc = t - t.mean()
            slope = (A * tc).sum(2) / (tc ** 2).sum()
            h = T // 2
            tl = t[h:] - t[h:].mean()
            late = (A[:, :, h:] * tl).sum(2) / (tl ** 2).sum()
            dt = np.diff(A, axis=2).reshape(n, -1)
            dsp = np.diff(A, axis=1).reshape(n, -1)
            feats += [slope, late, np.diff(slope, axis=1), dt, dsp]
        return np.hstack(feats)

    def fit(self, X, y):
        F = self._feat(X)
        self.mu_ = F.mean(0)
        self.sd_ = F.std(0) + 1e-8
        Z = (F - self.mu_) / self.sd_
        g = self.gamma_mult / Z.shape[1]
        self.model_ = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=g)
        self.model_.fit(Z, y)
        return self

    def predict(self, X):
        Z = (self._feat(X) - self.mu_) / self.sd_
        return self.model_.predict(Z)

def build_model(params, task, seed):
    kw = {k: params[k] for k in ('alpha', 'gamma_mult', 'eps') if k in params}
    return PhysFeatKRR(**kw)
