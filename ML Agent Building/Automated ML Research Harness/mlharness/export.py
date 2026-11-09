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
