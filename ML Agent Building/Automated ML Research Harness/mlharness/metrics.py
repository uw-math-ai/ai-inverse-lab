"""Evaluation metrics with a known optimisation direction."""
from __future__ import annotations

import math

import numpy as np


def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(len(a), -1)


def rel_l2(y, p):
    """Relative Frobenius error ||P - Y||_F / ||Y||_F (the lab's FRNE, as a fraction)."""
    y, p = _as2d(y), _as2d(p)
    denom = np.linalg.norm(y)
    return float(np.linalg.norm(p - y) / (denom if denom > 0 else 1.0))


def rmse(y, p):
    y, p = _as2d(y), _as2d(p)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y, p):
    y, p = _as2d(y), _as2d(p)
    return float(np.mean(np.abs(p - y)))


def r2(y, p):
    y, p = _as2d(y), _as2d(p)
    ss_res = ((y - p) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    ok = ss_tot > 0
    if not ok.any():
        return 0.0
    return float(np.mean(1.0 - ss_res[ok] / ss_tot[ok]))


def accuracy(y, p):
    return float(np.mean(np.asarray(y).ravel() == np.asarray(p).ravel()))


def f1_macro(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(y).ravel(), np.asarray(p).ravel(), average="macro"))


def balanced_accuracy(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(np.asarray(y).ravel(), np.asarray(p).ravel()))


# name -> (direction, function, task)
METRICS = {
    "rel_l2": ("min", rel_l2, "regression"),
    "rmse": ("min", rmse, "regression"),
    "mae": ("min", mae, "regression"),
    "r2": ("max", r2, "regression"),
    "accuracy": ("max", accuracy, "classification"),
    "f1_macro": ("max", f1_macro, "classification"),
    "balanced_accuracy": ("max", balanced_accuracy, "classification"),
}

DEFAULT_PRIMARY = {"regression": "rel_l2", "classification": "accuracy"}
DEFAULT_SECONDARY = {"regression": ["r2", "rmse"], "classification": ["f1_macro"]}


def direction(name: str) -> str:
    return METRICS[name][0]


def compute(name: str, y, p) -> float:
    return METRICS[name][1](y, p)


def compute_all(names, y, p) -> dict:
    out = {}
    for n in names:
        try:
            v = compute(n, y, p)
            out[n] = v if math.isfinite(v) else None
        except Exception:
            out[n] = None
    return out


def to_score(value, name: str) -> float:
    """Map a metric value to a 'higher is better' score. Missing/NaN -> -inf."""
    if value is None or not math.isfinite(value):
        return -math.inf
    return value if direction(name) == "max" else -value


def meets(value, name: str, threshold) -> bool:
    if threshold is None or value is None:
        return False
    return value >= threshold if direction(name) == "max" else value <= threshold


def relative_gain(new_score: float, old_score: float) -> float:
    """Relative improvement of a 'higher is better' score (works for negated losses)."""
    if not math.isfinite(old_score):
        return math.inf if math.isfinite(new_score) else 0.0
    if not math.isfinite(new_score):
        return -math.inf
    return (new_score - old_score) / max(abs(old_score), 1e-12)
