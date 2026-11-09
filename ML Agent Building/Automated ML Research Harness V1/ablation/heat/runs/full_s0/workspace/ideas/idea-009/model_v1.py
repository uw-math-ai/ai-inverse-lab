import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

class RidgeResidualTrees(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01, n_estimators=300, max_features=0.6, min_samples_leaf=3, seed=0):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        self.lin_ = Ridge(alpha=self.alpha).fit(X, y)
        r = y - self.lin_.predict(X)
        self.et_ = ExtraTreesRegressor(n_estimators=self.n_estimators, max_features=self.max_features, min_samples_leaf=self.min_samples_leaf, random_state=self.seed, n_jobs=1).fit(X, r)
        return self
    def predict(self, X):
        X = np.asarray(X)
        return self.lin_.predict(X) + self.et_.predict(X)

def build_model(params, task, seed):
    return RidgeResidualTrees(alpha=params.get('alpha', 0.01), n_estimators=int(params.get('n_estimators', 300)), max_features=params.get('max_features', 0.6), min_samples_leaf=int(params.get('min_samples_leaf', 3)), seed=seed)
