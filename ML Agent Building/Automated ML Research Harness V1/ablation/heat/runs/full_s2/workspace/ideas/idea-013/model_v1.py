import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class RidgeBoost:
    def __init__(self, params, seed):
        self.p = params
        self.seed = seed
    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        self.r = Ridge(alpha=self.p.get('alpha', 1.0)).fit(X, y)
        res = y - self.r.predict(X).reshape(y.shape)
        self.b = []
        for j in range(y.shape[1]):
            m = HistGradientBoostingRegressor(learning_rate=self.p.get('learning_rate', 0.05), max_iter=int(self.p.get('max_iter', 300)), max_leaf_nodes=int(self.p.get('max_leaf_nodes', 15)), l2_regularization=self.p.get('l2_regularization', 1.0), early_stopping=True, validation_fraction=0.1, random_state=self.seed)
            m.fit(X, res[:, j])
            self.b.append(m)
        self.single = (np.asarray(y).shape[1] == 1)
        return self
    def predict(self, X):
        base = self.r.predict(X)
        if base.ndim == 1:
            base = base[:, None]
        corr = np.column_stack([m.predict(X) for m in self.b])
        out = base + corr
        return out[:, 0] if self.single else out
    def get_params(self, deep=True):
        return {'params': self.p, 'seed': self.seed}
    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, 'p' if k == 'params' else k, v)
        return self

def build_model(params, task, seed):
    return RidgeBoost(params, seed)
