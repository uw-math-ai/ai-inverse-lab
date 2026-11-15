"""The exported final model: a single estimator or a weighted ensemble of fitted pipelines."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def combine(preds: list, weights: list, task: str):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    if task == "regression":
        stacked = np.stack([np.asarray(p, dtype=float) for p in preds])
        return np.tensordot(w, stacked, axes=1)
    # weighted vote
    labels = np.unique(np.concatenate([np.asarray(p).ravel() for p in preds]))
    votes = np.zeros((len(preds[0]), len(labels)))
    for wi, p in zip(w, preds):
        votes[np.arange(len(p)), np.searchsorted(labels, np.asarray(p).ravel())] += wi
    return labels[votes.argmax(axis=1)]


class FinalModel:
    def __init__(self, members: list, weights: list, task: str, configs: list, meta: dict):
        self.members = members
        self.weights = weights
        self.task = task
        self.configs = configs
        self.meta = meta

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = [m.predict(X) for m in self.members]
        return preds[0] if len(preds) == 1 else combine(preds, self.weights, self.task)


def load_model(path) -> FinalModel:
    """Load ``<run>/final/model.pkl`` (re-imports any agent-written code the model depends on)."""
    path = Path(path)
    if path.is_dir():
        path = path / "final" / "model.pkl" if (path / "final").exists() else path / "model.pkl"
    manifest = path.with_name("manifest.json")
    if manifest.exists():
        from .methods import load_custom_module
        for item in json.loads(manifest.read_text()).get("custom_modules", []):
            code = path.parent / item["file"]
            if code.exists():
                load_custom_module(str(code), item["module_name"])
    with open(path, "rb") as f:
        return pickle.load(f)


def certify_model(path_or_model, X, *, atol=1e-10, rtol=1e-8) -> dict:
    """Check a pristine export on development inputs, without consuming any labels.

    Every independent check starts from the same serialized state. Checks never rewrite the
    candidate. Stateful, batch-sensitive, stochastic, malformed or nonfinite predictions are
    explicitly uncertified; a configured random seed alone is not a certificate.
    """
    checks, reasons, differences = {}, [], {}
    certificate = {"schema_version": 3, "certified": False, "passed": False,
                   "protocol": "pristine-repeat-reload-partition-v1", "atol": float(atol),
                   "rtol": float(rtol), "checks": checks, "reasons": reasons,
                   "max_abs_differences": differences, "input_role": "development"}
    try:
        if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
            raise ValueError("certification tolerances must be finite and nonnegative")
        X = np.asarray(X)
        certificate["n_rows"] = len(X)
        if len(X) < 2:
            raise ValueError("certification requires at least two development rows")
        if isinstance(path_or_model, (str, Path)):
            def fresh():
                return load_model(path_or_model)
        else:
            pristine = pickle.dumps(path_or_model)
            def fresh():
                return pickle.loads(pristine)
        model = fresh()
        canonical = np.asarray(model.predict(X)).copy()

        def valid(p):
            if p.ndim == 0 or p.shape[0] != len(X):
                return False
            if np.issubdtype(p.dtype, np.number):
                return bool(np.isfinite(p).all())
            return bool(all(v is not None for v in p.ravel()))

        checks["finite_shape"] = valid(canonical)
        certificate["output_shape"] = list(canonical.shape)
        predictions = {"repeat": np.asarray(model.predict(X)).copy(),
                       "reload": np.asarray(fresh().predict(X)).copy()}
        batch_model = fresh()
        partitions = np.array_split(X, min(3, len(X)))
        predictions["partition"] = np.concatenate([np.asarray(batch_model.predict(part)).copy()
                                                    for part in partitions], axis=0)
        for name, prediction in predictions.items():
            ok = checks["finite_shape"] and valid(prediction) and prediction.shape == canonical.shape
            if ok and np.issubdtype(canonical.dtype, np.number):
                differences[name] = float(np.max(np.abs(prediction - canonical)))
                ok = bool(np.allclose(prediction, canonical, atol=atol, rtol=rtol))
            elif ok:
                ok = bool(np.array_equal(prediction, canonical))
            checks[name] = ok
        for name, passed in checks.items():
            if not passed:
                reasons.append(f"{name} check failed")
        certificate["certified"] = certificate["passed"] = bool(checks and all(checks.values()))
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}: {str(exc)[:400]}")
    return certificate
