import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _to_r(y):
    return 1.0 / np.clip(y, 1e-3, None)

def _to_k(r):
    return 1.0 / np.clip(r, 0.5, 2.0)

def build_model(params, task, seed):
    inner = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(512, 256), alpha=1e-4, learning_rate_init=1e-3, max_iter=400, early_stopping=True, n_iter_no_change=20, random_state=seed))
    return TransformedTargetRegressor(regressor=inner, func=_to_r, inverse_func=_to_k, check_inverse=False)
