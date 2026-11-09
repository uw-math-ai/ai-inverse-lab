"""Synthetic-data simulators. Each returns (X, y) for a fixed problem definition."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

GENERIC_FEATURE_SETS = ["raw", "log", "sqrt"]


class Simulator(ABC):
    name = "base"
    task = "regression"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` i.i.d. samples. Must be deterministic given ``seed``."""

    @property
    def feature_shape(self):
        """Structured shape of one feature row (used by domain feature transforms), or None."""
        return None

    @property
    def feature_sets(self) -> list[str]:
        return list(GENERIC_FEATURE_SETS)

    @property
    def suggested_feature_sets(self) -> list[str]:
        """Domain-informed feature sets a 'domain' ideator should prefer."""
        return []

    def describe(self) -> str:
        return f"{self.name} simulator with params {json.dumps(self.params, default=str)}"

    def config_hash(self) -> str:
        blob = json.dumps({"name": self.name, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def get_simulator(cfg: dict) -> Simulator:
    from .graph_fpt import GraphFPTSimulator
    from .tabular import ClassificationSimulator, FriedmanSimulator, LinearRegressionSimulator
    registry = {
        "graph_fpt": GraphFPTSimulator,
        "friedman": FriedmanSimulator,
        "regression": LinearRegressionSimulator,
        "classification": ClassificationSimulator,
    }
    name = cfg.get("name", "friedman")
    params = dict(cfg.get("params") or {})
    if name == "python":
        return PythonFileSimulator(**params)
    if name not in registry:
        raise ValueError(f"unknown simulator {name!r}; choose from {sorted(registry) + ['python']}")
    return registry[name](**params)


SIMULATOR_CATALOG = {
    "graph_fpt": "First-passage-time distributions of random walks on a weighted graph observed on a "
                 "vertex subset B; targets are edge weights and/or vertex potential (inverse problem).",
    "friedman": "Friedman #1 nonlinear regression benchmark.",
    "regression": "Linear multi-output regression (sklearn make_regression).",
    "classification": "Synthetic multi-class classification (sklearn make_classification).",
    "python": "User simulator: params {path: file.py, function: simulate, ...}; "
              "simulate(n, seed, **kwargs) -> (X, y).",
}


class PythonFileSimulator(Simulator):
    """Load ``simulate(n, seed, **kwargs) -> (X, y)`` from a user file.

    Optional module attributes: ``TASK`` ('regression'|'classification'), ``FEATURE_SHAPE``,
    ``DESCRIPTION``.
    """
    name = "python"

    def __init__(self, path: str, function: str = "simulate", **kwargs):
        super().__init__(path=path, function=function, **kwargs)
        self.kwargs = kwargs
        spec = importlib.util.spec_from_file_location("mlh_user_simulator", Path(path))
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load simulator file {path}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.fn = getattr(self.module, function)
        self.task = getattr(self.module, "TASK", "regression")

    @property
    def feature_shape(self):
        return getattr(self.module, "FEATURE_SHAPE", None)

    def describe(self):
        return getattr(self.module, "DESCRIPTION", super().describe())

    def generate(self, n, seed):
        X, y = self.fn(n, seed, **self.kwargs)
        return np.asarray(X, dtype=float), np.asarray(y)

    def config_hash(self):
        src = Path(self.params["path"]).read_bytes()
        blob = json.dumps(self.params, sort_keys=True, default=str).encode() + src
        return hashlib.sha1(blob).hexdigest()[:12]
