"""Methodology library: the space of ML approaches agents can propose, mutate and build.

An *idea config* is a JSON-able dict:
    {"method": key, "params": {...}, "features": "raw", "scaler": "standard", "pca": 0,
     "code_path": optional path to agent-written code defining build_model(params, task, seed)}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class Method:
    key: str
    family: str
    tasks: tuple
    description: str
    defaults: dict
    space: dict = field(default_factory=dict)  # name -> ("log"|"float"|"int", lo, hi) | ("choice", [...])
    scaling: float = 1.0      # assumed fit-time exponent in n (projects screen cost -> full cost)
    gpu: bool = False
    target_scale: bool = True


LIBRARY: dict[str, Method] = {}


def _reg(m: Method):
    LIBRARY[m.key] = m


_reg(Method("mean_baseline", "baseline", ("regression", "classification"),
            "Predict the training mean (regression) or majority class. Reference point only.",
            {}, {}, 0.1, target_scale=False))
_reg(Method("ridge", "linear", ("regression",),
            "L2-regularised linear least squares; strong baseline for smooth near-linear inverse maps.",
            {"alpha": 1.0}, {"alpha": ("log", 1e-6, 1e3)}, 1.0))
_reg(Method("pls", "linear", ("regression",),
            "Partial least squares; projects correlated features onto a few latent directions.",
            {"n_components": 10}, {"n_components": ("int", 2, 60)}, 1.0, target_scale=False))
_reg(Method("logistic", "linear", ("classification",),
            "Multinomial logistic regression with L2 penalty.",
            {"C": 1.0}, {"C": ("log", 1e-3, 1e3)}, 1.0))
_reg(Method("kernel_ridge", "kernel", ("regression",),
            "Kernel ridge regression with an RBF kernel; smooth nonlinear interpolant, O(n^2) memory. "
            "Bandwidth gamma = gamma_mult / (n_features * Var(X)) (sklearn 'scale' heuristic times a multiplier).",
            {"alpha": 1e-2, "gamma_mult": 1.0}, {"alpha": ("log", 1e-5, 1e1), "gamma_mult": ("log", 1e-2, 1e2)},
            2.2))
_reg(Method("svm", "kernel", ("classification", "regression"),
            "Support vector machine (RBF). For multi-output regression each target is fitted separately.",
            {"C": 1.0, "gamma": "scale"}, {"C": ("log", 1e-2, 1e3)}, 2.0))
_reg(Method("knn", "instance", ("regression", "classification"),
            "k-nearest neighbours in (scaled) feature space.",
            {"n_neighbors": 10, "weights": "distance"},
            {"n_neighbors": ("int", 1, 60), "weights": ("choice", ["uniform", "distance"])}, 1.0))
_reg(Method("random_forest", "ensemble", ("regression", "classification"),
            "Bagged decision trees; robust, little tuning, handles interactions.",
            {"n_estimators": 200, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
            {"n_estimators": ("int", 50, 500), "max_depth": ("choice", [None, 8, 16, 32]),
             "max_features": ("float", 0.1, 1.0), "min_samples_leaf": ("int", 1, 10)}, 1.2))
_reg(Method("extra_trees", "ensemble", ("regression", "classification"),
            "Extremely randomised trees; lower variance than random forests, fast.",
            {"n_estimators": 300, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
            {"n_estimators": ("int", 50, 600), "max_depth": ("choice", [None, 8, 16, 32]),
             "max_features": ("float", 0.1, 1.0), "min_samples_leaf": ("int", 1, 10)}, 1.2))
_reg(Method("hist_gbm", "boosting", ("regression", "classification"),
            "Histogram gradient boosting (one booster per target for multi-output regression).",
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "l2_regularization": 0.0},
            {"learning_rate": ("log", 0.01, 0.3), "max_iter": ("int", 50, 600),
             "max_leaf_nodes": ("int", 7, 127), "l2_regularization": ("log", 1e-6, 10.0)}, 1.1))
_reg(Method("mlp_sklearn", "neural", ("regression", "classification"),
            "scikit-learn multilayer perceptron (Adam, early stopping).",
            {"hidden": [256, 256], "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 300},
            {"hidden": ("choice", [[128], [256, 256], [512, 256], [256, 256, 256]]),
             "alpha": ("log", 1e-6, 1e-1), "learning_rate_init": ("log", 1e-4, 1e-2)}, 1.0))
_reg(Method("torch_mlp", "neural", ("regression", "classification"),
            "PyTorch MLP (LayerNorm, GELU, AdamW, cosine LR, early stopping). Uses a GPU if allocated.",
            {"hidden": [256, 256], "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 150,
             "batch_size": 128, "activation": "gelu"},
            {"hidden": ("choice", [[128, 128], [256, 256], [512, 512], [512, 256, 128], [1024, 512]]),
             "dropout": ("float", 0.0, 0.3), "lr": ("log", 1e-4, 3e-3), "weight_decay": ("log", 1e-6, 1e-2),
             "epochs": ("int", 50, 400), "batch_size": ("choice", [64, 128, 256]),
             "activation": ("choice", ["gelu", "relu", "silu", "tanh"])}, 1.0, gpu=True))

SCALERS = ["standard", "robust", "none"]


def methods_for(task: str, constraints: dict | None = None) -> dict[str, Method]:
    constraints = constraints or {}
    allowed = constraints.get("allowed_methods", "all")
    forbidden = set(constraints.get("forbidden_methods") or [])
    out = {}
    for k, m in LIBRARY.items():
        if task not in m.tasks or k in forbidden:
            continue
        if allowed != "all" and k not in allowed and k != "mean_baseline":
            continue
        if m.gpu and importlib.util.find_spec("torch") is None:
            continue
        out[k] = m
    return out


def catalog_text(methods: dict[str, Method]) -> str:
    lines = []
    for k, m in methods.items():
        if k == "mean_baseline":
            continue
        space = {p: list(v) if v[0] != "choice" else ["choice", v[1]] for p, v in m.space.items()}
        lines.append(f"- {k} [{m.family}]: {m.description} defaults={json.dumps(m.defaults)} "
                     f"search_space={json.dumps(space)}")
    return "\n".join(lines)


# -- parameter handling -------------------------------------------------------------------
def sample_value(spec, rng):
    kind = spec[0]
    if kind == "log":
        return float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
    if kind == "float":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "int":
        return int(rng.integers(spec[1], spec[2] + 1))
    if kind == "choice":
        return spec[1][int(rng.integers(len(spec[1])))]
    raise ValueError(kind)


def sample_params(m: Method, rng) -> dict:
    p = dict(m.defaults)
    for name, spec in m.space.items():
        p[name] = sample_value(spec, rng)
    return p


def mutate_params(m: Method, params: dict, rng, strength: float = 0.5) -> dict:
    """Local perturbation: multiplicative on log params, additive on others, resample choices."""
    p = dict(params)
    names = list(m.space) or []
    if not names:
        return p
    k = max(1, int(rng.integers(1, min(3, len(names)) + 1)))
    for name in rng.choice(names, size=k, replace=False):
        spec = m.space[name]
        cur = p.get(name, m.defaults.get(name))
        if spec[0] == "log" and isinstance(cur, (int, float)) and cur > 0:
            p[name] = float(cur * math.exp(rng.normal(0, strength * 1.5)))
        elif spec[0] == "float" and isinstance(cur, (int, float)):
            p[name] = float(cur + rng.normal(0, strength * (spec[2] - spec[1]) / 3))
        elif spec[0] == "int" and isinstance(cur, (int, float)):
            step = max(1, int(round(strength * (spec[2] - spec[1]) / 4)))
            p[name] = int(cur + rng.integers(-step, step + 1))
        else:
            p[name] = sample_value(spec, rng)
    return clip_params(m, p)[0]


def clip_params(m: Method, params: dict) -> tuple[dict, list[str]]:
    """Drop unknown params and clip to the search space. Returns (params, notes)."""
    notes, out = [], dict(m.defaults)
    for name, val in (params or {}).items():
        if name not in m.defaults and name not in m.space:
            notes.append(f"dropped unknown param {name!r}")
            continue
        spec = m.space.get(name)
        if spec and spec[0] in ("log", "float", "int") and isinstance(val, (int, float)) and val is not None:
            lo, hi = spec[1], spec[2]
            nv = min(max(val, lo), hi)
            if spec[0] == "int":
                nv = int(round(nv))
            if nv != val:
                notes.append(f"clipped {name} {val} -> {nv}")
            val = nv
        elif spec and spec[0] == "choice" and val not in spec[1]:
            if isinstance(val, (list, tuple)) and all(isinstance(v, (int, float)) for v in val) and val:
                val = [int(max(8, min(2048, v))) for v in val][:6]   # any reasonable layer list
            else:
                notes.append(f"{name}={val!r} not in {spec[1]}; using default")
                val = m.defaults.get(name)
        out[name] = val
    return out, notes


def config_hash(cfg: dict) -> str:
    keep = {k: cfg.get(k) for k in ("method", "params", "features", "scaler", "pca")}
    if cfg.get("code_path"):
        keep["code"] = hashlib.sha1(Path(cfg["code_path"]).read_bytes()).hexdigest()
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


# -- building -----------------------------------------------------------------------------
class ScaledRBFKernelRidge(BaseEstimator, RegressorMixin):
    """RBF kernel ridge whose bandwidth adapts to the data scale at fit time (robust to PCA/features)."""

    def __init__(self, alpha=1e-2, gamma_mult=1.0):
        self.alpha = alpha
        self.gamma_mult = gamma_mult

    def fit(self, X, y):
        from sklearn.kernel_ridge import KernelRidge
        X = np.asarray(X, dtype=float)
        var = float(X.var()) or 1.0
        self.gamma_ = self.gamma_mult / (X.shape[1] * var)
        self.model_ = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def _estimator(key: str, p: dict, task: str, device: str, seed: int, threads: int, n_features: int):
    from sklearn import dummy, ensemble, kernel_ridge, linear_model, neighbors, neural_network, svm
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.multioutput import MultiOutputRegressor
    reg = task == "regression"
    if key == "mean_baseline":
        return dummy.DummyRegressor() if reg else dummy.DummyClassifier(strategy="most_frequent")
    if key == "ridge":
        return linear_model.Ridge(alpha=p["alpha"])
    if key == "pls":
        return PLSRegression(n_components=int(min(p["n_components"], n_features)))
    if key == "logistic":
        return linear_model.LogisticRegression(C=p["C"], max_iter=2000)
    if key == "kernel_ridge":
        return ScaledRBFKernelRidge(alpha=p["alpha"], gamma_mult=p.get("gamma_mult", 1.0))
    if key == "svm":
        if reg:
            return MultiOutputRegressor(svm.SVR(C=p["C"], gamma=p.get("gamma", "scale")), n_jobs=threads)
        return svm.SVC(C=p["C"], gamma=p.get("gamma", "scale"))
    if key == "knn":
        cls = neighbors.KNeighborsRegressor if reg else neighbors.KNeighborsClassifier
        return cls(n_neighbors=int(p["n_neighbors"]), weights=p["weights"], n_jobs=threads)
    if key in ("random_forest", "extra_trees"):
        base = {"random_forest": (ensemble.RandomForestRegressor, ensemble.RandomForestClassifier),
                "extra_trees": (ensemble.ExtraTreesRegressor, ensemble.ExtraTreesClassifier)}[key]
        cls = base[0] if reg else base[1]
        return cls(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                   max_features=p["max_features"], min_samples_leaf=int(p["min_samples_leaf"]),
                   n_jobs=threads, random_state=seed)
    if key == "hist_gbm":
        kw = dict(learning_rate=p["learning_rate"], max_iter=int(p["max_iter"]),
                  max_leaf_nodes=int(p["max_leaf_nodes"]), l2_regularization=p["l2_regularization"],
                  early_stopping=True, random_state=seed)
        if reg:
            return MultiOutputRegressor(ensemble.HistGradientBoostingRegressor(**kw), n_jobs=1)
        return ensemble.HistGradientBoostingClassifier(**kw)
    if key == "mlp_sklearn":
        cls = neural_network.MLPRegressor if reg else neural_network.MLPClassifier
        return cls(hidden_layer_sizes=tuple(int(h) for h in p["hidden"]), alpha=p["alpha"],
                   learning_rate_init=p["learning_rate_init"], max_iter=int(p["max_iter"]),
                   early_stopping=True, random_state=seed)
    if key == "torch_mlp":
        from .torch_models import TorchMLP
        return TorchMLP(task=task, hidden=tuple(int(h) for h in p["hidden"]), dropout=p["dropout"],
                        lr=p["lr"], weight_decay=p["weight_decay"], epochs=int(p["epochs"]),
                        batch_size=int(p["batch_size"]), activation=p["activation"], device=device,
                        seed=seed, n_threads=threads)
    raise KeyError(key)


def load_custom_module(code_path: str, module_name: str | None = None):
    """Import agent-written code under a stable module name so fitted objects can be pickled."""
    path = Path(code_path)
    name = module_name or "mlh_custom_" + hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_pipeline(cfg: dict, task: str, feature_shape, n_features: int, device="cpu", seed=0,
                   threads=1):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    from .features import FeatureTransform

    key = cfg["method"]
    steps = [("features", FeatureTransform(cfg.get("features", "raw"), shape=feature_shape))]
    scaler = cfg.get("scaler", "standard")
    if scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scale", RobustScaler()))
    if int(cfg.get("pca") or 0) > 0:
        steps.append(("pca", PCA(n_components=int(cfg["pca"]), random_state=seed)))
    if key == "custom":
        mod = load_custom_module(cfg["code_path"], cfg.get("module_name"))
        est = mod.build_model(dict(cfg.get("params") or {}), task, seed)
        target_scale = bool(cfg.get("params", {}).get("target_scale", True))
    else:
        m = LIBRARY[key]
        est = _estimator(key, cfg.get("params") or {}, task, device, seed, threads,
                         int(cfg.get("pca") or 0) or n_features)
        target_scale = m.target_scale
    steps.append(("model", est))
    pipe = Pipeline(steps)
    if task == "regression" and target_scale:
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return pipe
