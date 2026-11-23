"""V2 method interface: one JSON-able *method spec* for data-driven, physics-based and hybrid methods.

The V1 ``X -> y`` estimator configs remain one kind among several; every kind is built into an
object with ``fit(X, y)`` / ``predict(X)`` and evaluated through the same runner/metrics contract.

Method spec kinds
-----------------
library        {"kind": "library", "method": <V1 key>, "params", "features", "scaler", "pca",
                "target_transform": "none"|"log", "noise_aug": copies, "noise_scale": float}
physics        {"kind": "physics", "params": {transform, multistart, iters, tikhonov, estimator,
                posterior_samples}}  -- MLE/MAP/posterior-mean through the authorised forward model
hybrid         {"kind": "hybrid", "init": <library spec>, "params": {transform, iters, tikhonov,
                extra_starts, estimator, posterior_samples}}  -- learned initializer + LM refinement
custom_ml      {"kind": "custom_ml", "code": "...", "params", "features", "scaler", "pca"}
                agent code defining build_model(params, task, seed) (V1 contract)
custom_solver  {"kind": "custom_solver", "code": "...", "params", "uses_forward": bool}
                agent code defining build_method(params, context) -> object with fit/predict; may
                call context.forward(theta) when authorised
random_search  {"kind": "random_search", "n": k}  -- control: expanded into k random library specs
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import methods as ML
from .features import valid_feature_set

KINDS = ("library", "physics", "hybrid", "custom_ml", "custom_solver", "random_search")

PHYSICS = ML.Method(
    "physics_lm", "physics", ("regression",),
    "Per-sample bounded Levenberg-Marquardt through the forward model (MLE; MAP with tikhonov>0; "
    "estimator=posterior_mean gives the MMSE estimate under the box prior by importance sampling).",
    {"transform": "log", "multistart": 1, "iters": 30, "tikhonov": 0.0, "estimator": "mode",
     "posterior_samples": 128},
    {"transform": ("choice", ["log", "none"]), "multistart": ("int", 1, 6), "iters": ("int", 5, 80),
     "tikhonov": ("float", 0.0, 10.0),
     "estimator": ("choice", ["mode", "posterior_mean"]), "posterior_samples": ("int", 32, 512)},
    scaling=0.0, target_scale=False)

HYBRID = ML.Method(
    "hybrid_lm", "hybrid", ("regression",),
    "Learned initializer (any library method) followed by a few LM iterations through the forward model.",
    {"transform": "log", "iters": 10, "tikhonov": 0.0, "extra_starts": 0, "estimator": "mode",
     "posterior_samples": 128},
    {"transform": ("choice", ["log", "none"]), "iters": ("int", 1, 40), "extra_starts": ("int", 0, 3),
     "tikhonov": ("float", 0.0, 10.0),
     "estimator": ("choice", ["mode", "posterior_mean"]), "posterior_samples": ("int", 32, 512)},
    scaling=1.0, target_scale=False)

LIB_EXTRA_DEFAULTS = {"target_transform": "none", "noise_aug": 0, "noise_scale": 1.0}


def family(ms: dict) -> str:
    k = ms.get("kind")
    if k in ("library", "custom_ml", "random_search"):
        return "data-driven"
    if k == "physics":
        return "physics"
    if k == "hybrid":
        return "hybrid"
    if k == "custom_solver":
        return "physics-custom" if ms.get("uses_forward") else "custom"
    return "unknown"


def uses_forward(ms: dict) -> bool:
    return ms.get("kind") in ("physics", "hybrid") or (ms.get("kind") == "custom_solver" and bool(ms.get("uses_forward")))


def trains(ms: dict) -> bool:
    return ms.get("kind") in ("library", "hybrid", "custom_ml", "custom_solver")


def wants_gpu(ms: dict) -> bool:
    lib = ms.get("init") if ms.get("kind") == "hybrid" else ms
    return bool(lib) and lib.get("method") == "torch_mlp"


def code_of(ms: dict) -> str | None:
    if ms.get("code"):
        return ms["code"]
    if ms.get("code_path") and Path(ms["code_path"]).exists():
        return Path(ms["code_path"]).read_text(encoding="utf-8")
    return None


def method_hash(ms: dict) -> str:
    keep = {k: v for k, v in ms.items() if k not in ("code", "code_path", "module_name", "title")}
    if ms.get("kind") == "hybrid" and isinstance(ms.get("init"), dict):
        keep["init"] = {k: v for k, v in ms["init"].items() if k not in ("code", "code_path", "module_name")}
    code = code_of(ms)
    if code:
        keep["code_sha"] = hashlib.sha1(code.encode()).hexdigest()
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def label(ms: dict) -> str:
    k = ms.get("kind")
    if k == "library":
        extra = ("+logy" if ms.get("target_transform") == "log" else "") + \
                (f"+aug{ms.get('noise_aug')}" if int(ms.get("noise_aug") or 0) else "")
        pca = f"+pca{ms.get('pca')}" if int(ms.get("pca") or 0) else ""
        return f"{ms.get('method')}/{ms.get('features', 'raw')}{pca}{extra}"
    if k == "physics":
        p = ms.get("params") or {}
        est = "MMSE" if p.get("estimator") == "posterior_mean" else ("MAP" if p.get("tikhonov") else "MLE")
        return f"physics-{est}[{p.get('transform', 'log')},ms={p.get('multistart')},it={p.get('iters')}]"
    if k == "hybrid":
        p = ms.get("params") or {}
        return f"hybrid[{label(ms.get('init') or {})}->LM{p.get('iters')}]"
    if k == "custom_ml":
        return "custom-ml"
    if k == "custom_solver":
        return "custom-solver" + ("(fwd)" if ms.get("uses_forward") else "")
    if k == "random_search":
        return f"random-search x{ms.get('n')}"
    return str(k)


def as_v1_config(ms: dict) -> dict:
    """V1-shaped config for display / V1 helpers (method label, params, features)."""
    lib = ms if ms.get("kind") in ("library", "custom_ml") else (ms.get("init") or {})
    return {"method": label(ms), "params": ms.get("params") or {}, "features": lib.get("features", "-"),
            "scaler": lib.get("scaler", "-"), "pca": lib.get("pca", 0)}


# ---------------------------------------------------------------------------------------------
# normalisation (used by the implementation gate)
# ---------------------------------------------------------------------------------------------
def _norm_library(raw: dict, available: dict, feature_sets: list, n_features: int, n_train: int):
    notes, errors = [], []
    ms = {"kind": "library", "method": raw.get("method"), "params": dict(raw.get("params") or {}),
          "features": raw.get("features") or "raw", "scaler": raw.get("scaler") or "standard",
          "pca": int(raw.get("pca") or 0)}
    for k, v in LIB_EXTRA_DEFAULTS.items():
        ms[k] = raw.get(k, v)
    key = ms["method"]
    if key not in ML.LIBRARY:
        errors.append(f"unknown library method {key!r}")
    elif key not in available:
        errors.append(f"library method {key!r} is excluded by the spec or unavailable on this machine")
    else:
        ms["params"], n = ML.clip_params(available[key], ms["params"])
        notes += n
    if not valid_feature_set(ms["features"], feature_sets):
        notes.append(f"feature set {ms['features']!r} invalid; using 'raw'")
        ms["features"] = "raw"
    if ms["scaler"] not in ML.SCALERS:
        notes.append(f"scaler {ms['scaler']!r} invalid; using 'standard'")
        ms["scaler"] = "standard"
    if ms["pca"] and not 0 < ms["pca"] < min(n_features, n_train):
        notes.append(f"pca={ms['pca']} invalid; disabled")
        ms["pca"] = 0
    if ms["target_transform"] not in ("none", "log"):
        notes.append(f"target_transform {ms['target_transform']!r} invalid; using 'none'")
        ms["target_transform"] = "none"
    ms["noise_aug"] = int(max(0, min(4, int(ms.get("noise_aug") or 0))))
    ms["noise_scale"] = float(max(0.1, min(5.0, float(ms.get("noise_scale") or 1.0))))
    return ms, notes, errors


def normalize(raw: dict, available: dict, feature_sets: list, n_features: int, n_train: int):
    """Return (method_spec, notes, errors). Unknown fields are dropped; params are clipped."""
    raw = copy.deepcopy(raw or {})
    kind = raw.get("kind") or ("library" if raw.get("method") in ML.LIBRARY else None)
    if kind not in KINDS:
        return {"kind": kind}, [], [f"unknown method kind {kind!r}; expected one of {list(KINDS)}"]
    if kind == "library":
        return _norm_library(raw, available, feature_sets, n_features, n_train)
    if kind in ("physics", "hybrid"):
        m = PHYSICS if kind == "physics" else HYBRID
        params, notes = ML.clip_params(m, raw.get("params") or {})
        ms = {"kind": kind, "params": params}
        errors = []
        if kind == "hybrid":
            init_raw = raw.get("init") or {"method": "ridge", "params": {"alpha": 1e-3}, "features": "log"}
            init_raw = {**init_raw, "kind": "library"}
            ms["init"], n2, e2 = _norm_library(init_raw, available, feature_sets, n_features, n_train)
            notes += [f"init: {x}" for x in n2]
            errors += [f"init: {x}" for x in e2]
        return ms, notes, errors
    if kind == "custom_ml":
        ms = {"kind": "custom_ml", "code": raw.get("code") or "", "params": dict(raw.get("params") or {}),
              "features": raw.get("features") or "raw", "scaler": raw.get("scaler") or "standard",
              "pca": int(raw.get("pca") or 0), "target_transform": raw.get("target_transform", "none")}
        notes = []
        if not valid_feature_set(ms["features"], feature_sets):
            notes.append(f"feature set {ms['features']!r} invalid; using 'raw'")
            ms["features"] = "raw"
        if ms["scaler"] not in ML.SCALERS:
            ms["scaler"] = "standard"
        if not ms["code"]:
            notes.append("no code yet: the engineer generates it after the research phase")
        return ms, notes, []
    if kind == "custom_solver":
        ms = {"kind": "custom_solver", "code": raw.get("code") or "", "params": dict(raw.get("params") or {}),
              "uses_forward": bool(raw.get("uses_forward", "forward" in (raw.get("code") or "")))}
        return ms, ([] if ms["code"] else ["no code yet: the engineer generates it after the research phase"]), []
    n = int(raw.get("n") or 3)
    return {"kind": "random_search", "n": max(1, min(8, n))}, [], []


# ---------------------------------------------------------------------------------------------
# cost model (a-priori estimates; the screen measures the real cost)
# ---------------------------------------------------------------------------------------------
def forward_evals_per_sample(ms: dict, n_params: int) -> float:
    p = ms.get("params") or {}
    iters_eff = min(int(p.get("iters", 30)), 12)
    per_start = iters_eff * (2 * n_params + 1)
    if ms.get("kind") == "physics":
        starts = int(p.get("multistart", 1))
    elif ms.get("kind") == "hybrid":
        starts = 1 + int(p.get("extra_starts", 0))
    else:
        return 0.0
    post = int(p.get("posterior_samples", 0)) if p.get("estimator") == "posterior_mean" else 0
    return starts * per_start + 2 * n_params + post


LIB_FULL_SECONDS = {"ridge": 1, "pls": 1, "knn": 2, "kernel_ridge": 15, "svm": 30, "random_forest": 25,
                    "extra_trees": 15, "hist_gbm": 40, "mlp_sklearn": 25, "torch_mlp": 40}


def estimate_seconds(ms: dict, n_eval: int, n_train: int, n_params: int, sec_per_forward: float | None) -> float:
    """Rough cost of one train+evaluate at full fidelity (used before anything has run)."""
    k = ms.get("kind")
    scale = n_train / 3000.0
    if k == "library":
        return LIB_FULL_SECONDS.get(ms.get("method"), 20) * scale * (1 + int(ms.get("noise_aug") or 0))
    if k in ("physics", "hybrid"):
        fwd = forward_evals_per_sample(ms, n_params) * n_eval * (sec_per_forward or 5e-5)
        init = estimate_seconds(ms["init"], n_eval, n_train, n_params, sec_per_forward) if k == "hybrid" else 0.0
        return fwd + init + 2.0
    return 30.0 * scale


# ---------------------------------------------------------------------------------------------
# building (runs inside the isolated runner process)
# ---------------------------------------------------------------------------------------------
class SolverContext:
    """The narrow, picklable facade agent-written solvers receive (no simulator internals)."""

    def __init__(self, ctx):
        self._ctx = ctx

    @property
    def forward_available(self):
        return self._ctx.forward_available

    @property
    def lower(self):
        return None if self._ctx.lower is None else self._ctx.lower.copy()

    @property
    def upper(self):
        return None if self._ctx.upper is None else self._ctx.upper.copy()

    @property
    def noise_std(self):
        return self._ctx.noise_std

    @property
    def param_names(self):
        return list(self._ctx.param_names)

    @property
    def n_params(self):
        return self._ctx.n_params

    @property
    def facts(self):
        return dict(self._ctx.facts)

    def forward(self, theta):
        return self._ctx.forward(theta)

    def clip(self, theta):
        return self._ctx.clip(theta)

    def prior_center(self):
        return self._ctx.prior_center()

    def jacobian(self, theta, h=1e-5, transform="none"):
        from .problem import fd_jacobian
        return fd_jacobian(self._ctx.forward, np.atleast_2d(np.asarray(theta, float)), h, transform)


def _library_estimator(ms, task, feature_shape, n_features, context, device, seed, threads):
    cfg = {k: ms.get(k) for k in ("method", "params", "features", "scaler", "pca", "code_path", "module_name")}
    if ms.get("kind") == "custom_ml":
        cfg["method"] = "custom"
    est = ML.build_pipeline(cfg, task, feature_shape, n_features, device=device, seed=seed, threads=threads)
    from .inversion import LogTargetRegressor, NoiseAugmented
    if ms.get("target_transform") == "log":
        est = LogTargetRegressor(est)
    if int(ms.get("noise_aug") or 0) > 0:
        sigma = context.noise_std if context is not None and context.noise_std else float(ms.get("noise_sigma", 1e-3))
        est = NoiseAugmented(est, copies=int(ms["noise_aug"]), sigma=sigma, scale=float(ms.get("noise_scale", 1.0)),
                             seed=seed)
    return est


def build(ms: dict, task: str, feature_shape, n_features: int, context=None, device="cpu", seed=0, threads=1):
    from .inversion import HybridInversion, PhysicsInversion
    k = ms.get("kind")
    if k in ("library", "custom_ml"):
        return _library_estimator(ms, task, feature_shape, n_features, context, device, seed, threads)
    if k == "physics":
        return PhysicsInversion(context=context, seed=seed, **(ms.get("params") or {}))
    if k == "hybrid":
        init = _library_estimator(ms["init"], task, feature_shape, n_features, context, device, seed, threads)
        return HybridInversion(context=context, initializer=init, seed=seed, **(ms.get("params") or {}))
    if k == "custom_solver":
        mod = ML.load_custom_module(ms["code_path"], ms.get("module_name"))
        obj = mod.build_method(dict(ms.get("params") or {}), SolverContext(context))
        if not (hasattr(obj, "fit") and hasattr(obj, "predict")):
            raise TypeError("build_method must return an object with fit(X, y) and predict(X)")
        return obj
    raise ValueError(f"cannot build method kind {k!r}")


def random_library_specs(n: int, available: dict, feature_sets: list, n_features: int, rng) -> list[dict]:
    """Uniform random library configs (the V1 'random' control)."""
    keys = [k for k in available if k != "mean_baseline"]
    out = []
    for _ in range(n):
        key = keys[int(rng.integers(len(keys)))]
        m = available[key]
        out.append({"kind": "library", "method": key, "params": ML.sample_params(m, rng),
                    "features": feature_sets[int(rng.integers(len(feature_sets)))],
                    "scaler": ML.SCALERS[int(rng.integers(len(ML.SCALERS)))],
                    "pca": 0 if rng.random() < 0.5 else int(rng.choice([c for c in (8, 16, 32, 64) if c < n_features] or [0])),
                    **LIB_EXTRA_DEFAULTS})
    return out


def mutate(ms: dict, available: dict, feature_sets: list, rng, strength: float = 0.4) -> dict:
    """Local perturbation of a method spec (heuristic refinement)."""
    child = copy.deepcopy(ms)
    k = ms.get("kind")
    if k == "library" and ms.get("method") in available:
        child["params"] = ML.mutate_params(available[ms["method"]], ms.get("params") or {}, rng, strength)
        if rng.random() < 0.2 and len(feature_sets) > 1:
            child["features"] = str(rng.choice([f for f in feature_sets if f != ms.get("features")]))
    elif k == "physics":
        child["params"] = ML.mutate_params(PHYSICS, ms.get("params") or {}, rng, strength)
    elif k == "hybrid":
        if rng.random() < 0.5 and ms["init"].get("method") in available:
            child["init"]["params"] = ML.mutate_params(available[ms["init"]["method"]], ms["init"].get("params") or {},
                                                       rng, strength)
        else:
            child["params"] = ML.mutate_params(HYBRID, ms.get("params") or {}, rng, strength)
    return child
