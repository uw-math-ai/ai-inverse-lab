"""Feature transforms selectable by ideas. Kept in an importable module so fitted models pickle."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _raw(X, shape, eps):
    return X


def _log(X, shape, eps):
    return np.log(np.abs(X) + eps)


def _sqrt(X, shape, eps):
    return np.sqrt(np.abs(X))


def _fpt_tensor(X, shape):
    if shape is None or len(shape) != 3:
        raise ValueError("fpt features need feature_shape (T, |B|, |B|)")
    return X.reshape(len(X), *shape)


def _fpt_summary(X, shape, eps):
    """Per-pair hitting mass, conditional mean / std of the first-passage time, and r(1,x,y)."""
    r = _fpt_tensor(X, shape)                       # (S, T, b, b)
    T = r.shape[1]
    t = np.arange(1, T + 1, dtype=float)[None, :, None, None]
    mass = r.sum(axis=1)
    safe = np.maximum(mass, eps)
    mean_t = (r * t).sum(axis=1) / safe
    var_t = np.maximum((r * t ** 2).sum(axis=1) / safe - mean_t ** 2, 0.0)
    first = r[:, 0]
    feats = [mass, mean_t, np.sqrt(var_t), first, np.log(mass + eps)]
    return np.concatenate([f.reshape(len(X), -1) for f in feats], axis=1)


def _fpt_early(X, shape, eps):
    """Only the first three time steps (local structure), log-scaled."""
    r = _fpt_tensor(X, shape)[:, : min(3, shape[0])]
    return np.log(r.reshape(len(X), -1) + eps)


FEATURE_FUNCS = {"raw": _raw, "log": _log, "sqrt": _sqrt, "fpt_summary": _fpt_summary,
                 "fpt_early": _fpt_early}


class FeatureTransform(BaseEstimator, TransformerMixin):
    """``kind`` is one name or several joined with '+', e.g. 'log+fpt_summary'."""

    def __init__(self, kind: str = "raw", shape=None, eps: float = 1e-4):
        self.kind = kind
        self.shape = shape
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        shape = tuple(self.shape) if self.shape is not None else None
        parts = [FEATURE_FUNCS[k](X, shape, self.eps) for k in self.kind.split("+")]
        return np.hstack(parts) if len(parts) > 1 else parts[0]


def valid_feature_set(kind: str, allowed: list[str]) -> bool:
    return kind in allowed or all(k in FEATURE_FUNCS and k in allowed for k in kind.split("+"))
