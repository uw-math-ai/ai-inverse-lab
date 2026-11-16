import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

class LogKRR(BaseEstimator, RegressorMixin):
    def __init__(self, seed=0):
        self.seed = seed
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        # targets arrive standardised; shift to positive before log
        self.shift_ = -y.min(axis=0) + 1.0
        z = np.log(y + self.shift_)
        self.zm_ = z.mean(0); self.zs_ = z.std(0) + 1e-12
        zt = (z - self.zm_) / self.zs_
        d = X.shape[1]; v = X.var() + 1e-12
        base = 1.0 / (d * v)
        grid = {'alpha': [1e-4, 1e-3, 1e-2], 'gamma': [base * g for g in (0.1, 0.3, 1.0)]}
        n = X.shape[0]
        idx = np.random.RandomState(self.seed).permutation(n)[:min(n, 1500)]
        gs = GridSearchCV(KernelRidge(kernel='rbf'), grid, cv=KFold(3, shuffle=True, random_state=self.seed), scoring='neg_mean_squared_error')
        gs.fit(X[idx], zt[idx])
        self.model_ = KernelRidge(kernel='rbf', **gs.best_params_).fit(X, zt)
        return self
    def predict(self, X):
        z = self.model_.predict(X) * self.zs_ + self.zm_
        return np.exp(z) - self.shift_

def build_model(params, task, seed):
    return LogKRR(seed=seed)
