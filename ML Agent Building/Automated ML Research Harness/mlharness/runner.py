"""Isolated experiment execution: ``python -m mlharness.runner job.json``.

Runs in its own process so crashes, memory blow-ups and runaway jobs cannot take down the
scheduler; the parent enforces the timeout. Writes ``result.json`` next to the job file.

Modes
-----
train_eval : fit on a nested subsample (``fraction``) of the training split, score on validation.
final      : fit every member on train+val, score the (weighted) combination on the untouched test
             split, export ``model.pkl`` + ``manifest.json``.
V2 modes (method specs from ``methods_v2``: library / physics / hybrid / custom_ml / custom_solver):
method_eval: fit on a nested training fraction, predict a nested validation subsample (or all of it),
             report metrics, cost, latency, forward evaluations and uncertainty calibration.
final_v2   : fit members on train+val, evaluate once on test (+ bootstrap CI, distribution-shift sets,
             latency), export the model with reproducibility metadata, re-load it and check determinism.
Jobs that carry agent-written code run under the audit-hook sandbox (``mlharness.safety``).
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path


def _subsample(n: int, fraction: float, seed: int, min_n: int = 30):
    import numpy as np
    m = int(min(n, max(min_n, round(n * fraction))))
    return np.random.default_rng(seed).permutation(n)[:m]   # same seed -> nested subsets


def _n_features_after(cfg, feature_shape, X):
    from .features import FeatureTransform
    return FeatureTransform(cfg.get("features", "raw"), shape=feature_shape).transform(X[:2]).shape[1]


def _fit_member(cfg, job, X, y):
    from .methods import build_pipeline
    nf = _n_features_after(cfg, job.get("feature_shape"), X)
    model = build_pipeline(cfg, job["task"], job.get("feature_shape"), nf, device=job.get("device", "cpu"),
                           seed=int(job.get("seed", 0)), threads=int(job.get("threads", 1)))
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    msgs = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})[:5]
    return model, time.time() - t0, msgs


def _n_params(model):
    est = model
    for attr in ("regressor_", "regressor"):
        est = getattr(est, attr, est)
    est = getattr(est, "steps", [[None, est]])[-1][1]
    return getattr(est, "n_params_", None)


def run_train_eval(job, data, out_dir: Path):
    import numpy as np

    from . import metrics as M
    Xtr, ytr = data["X_train"], data["y_train"]
    idx = _subsample(len(Xtr), float(job["fraction"]), int(job.get("data_seed", 0)))
    Xs, ys = Xtr[idx], ytr[idx]
    model, fit_s, warn = _fit_member(job["idea"], job, Xs, ys)
    t0 = time.time()
    pv = model.predict(data["X_val"])
    pred_s = time.time() - t0
    pv = np.asarray(pv)
    if job["task"] == "regression":
        pv = pv.reshape(data["y_val"].shape)
        if not np.all(np.isfinite(pv)):
            raise FloatingPointError("model produced non-finite validation predictions")
    names = [job["primary"]] + [m for m in job.get("secondary", []) if m != job["primary"]]
    val = M.compute_all(names, data["y_val"], pv)
    k = min(len(Xs), 2000)
    ptr = np.asarray(model.predict(Xs[:k]))
    train = M.compute_all([job["primary"]], ys[:k], ptr.reshape(ys[:k].shape) if job["task"] == "regression" else ptr)
    np.save(out_dir / "val_pred.npy", pv)
    if job.get("save_model"):
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    return {"val": val, "train": train, "n_train_used": int(len(idx)), "fit_seconds": fit_s,
            "predict_seconds": pred_s, "warnings": warn, "n_params": _n_params(model),
            "val_pred_path": str(out_dir / "val_pred.npy")}


def run_final(job, data, out_dir: Path):
    import numpy as np

    from . import metrics as M
    from .export import FinalModel, combine
    X = np.concatenate([data["X_train"], data["X_val"]])
    y = np.concatenate([data["y_train"], data["y_val"]])
    final_dir = Path(job["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    members, fit_total, custom = [], 0.0, []
    for i, cfg in enumerate(job["members"]):
        cfg = dict(cfg)
        if cfg.get("code_path"):
            dst = final_dir / f"member{i}_model.py"
            shutil.copy(cfg["code_path"], dst)
            cfg["module_name"] = f"mlh_final_member{i}_{job['run_id']}"
            cfg["code_path"] = str(dst)
            custom.append({"file": dst.name, "module_name": cfg["module_name"]})
        model, fit_s, _ = _fit_member(cfg, job, X, y)
        members.append(model)
        fit_total += fit_s
    preds = [np.asarray(m.predict(data["X_test"])) for m in members]
    pt = preds[0] if len(preds) == 1 else combine(preds, job["weights"], job["task"])
    if job["task"] == "regression":
        pt = np.asarray(pt).reshape(data["y_test"].shape)
    names = [job["primary"]] + [m for m in job.get("secondary", []) if m != job["primary"]]
    test = M.compute_all(names, data["y_test"], pt)
    fm = FinalModel(members, job["weights"], job["task"], job["members"],
                    {"test": test, "run_id": job["run_id"], "n_train": int(len(X))})
    with open(final_dir / "model.pkl", "wb") as f:
        pickle.dump(fm, f)
    np.save(final_dir / "test_pred.npy", pt)
    (final_dir / "manifest.json").write_text(json.dumps(
        {"members": job["members"], "weights": job["weights"], "task": job["task"], "test": test,
         "custom_modules": custom, "feature_dim": int(data["X_test"].shape[1]),
         "target_shape": list(data["y_test"].shape[1:])}, indent=2))
    return {"test": test, "fit_seconds": fit_total, "model_path": str(final_dir / "model.pkl")}



# ============================================================================================
# V2
# ============================================================================================
def _val_subsample(n: int, k, seed: int):
    import numpy as np
    if not k or int(k) >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed + 11).permutation(n)[: int(k)])   # nested across k


def _make_context(job, authorized=None):
    from .problem import ProblemContext
    cfg = job.get("simulator")
    if not cfg:
        return None
    return ProblemContext(cfg, authorized=bool(job.get("forward_access")) if authorized is None else authorized,
                          max_forward_evals=job.get("max_forward_evals"))


def _materialize_code(ms: dict, dest_dir: Path, tag: str) -> dict:
    """Write agent code next to the job so it is importable (and pickles) under a stable module name."""
    ms = dict(ms)
    if ms.get("kind") in ("custom_ml", "custom_solver"):
        code = ms.get("code")
        if not code and ms.get("code_path"):
            code = Path(ms["code_path"]).read_text(encoding="utf-8")
        dst = Path(dest_dir) / f"{tag}.py"
        dst.write_text(code or "", encoding="utf-8")
        ms["code_path"], ms["module_name"] = str(dst), f"mlh_{tag}"
        ms.pop("code", None)
    return ms


def _has_code(ms_list) -> bool:
    return any(m.get("kind") in ("custom_ml", "custom_solver") for m in ms_list)


def _preload_and_guard(ms_list, allowed_dirs):
    """Import the numerical stack *before* the guard (imports read files and may probe hardware)."""
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.special  # noqa: F401
    import sklearn.decomposition  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.kernel_ridge  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    import sklearn.neighbors  # noqa: F401
    import sklearn.neural_network  # noqa: F401
    import sklearn.preprocessing  # noqa: F401
    codes = " ".join(Path(m["code_path"]).read_text(encoding="utf-8") if m.get("code_path") else (m.get("code") or "")
                     for m in ms_list)
    if "torch" in codes:
        import torch  # noqa: F401
    from .safety import install_guard
    install_guard(allowed_dirs)


def _n_features_for(ms, feature_shape, X):
    from .features import FeatureTransform
    lib = ms.get("init") if ms.get("kind") == "hybrid" else ms
    if lib and lib.get("kind") in ("library", "custom_ml"):
        return FeatureTransform(lib.get("features", "raw"), shape=feature_shape).transform(X[:2]).shape[1]
    return X.shape[1]


def _per_target(y, p):
    import numpy as np
    y2, p2 = np.asarray(y, float).reshape(len(y), -1), np.asarray(p, float).reshape(len(p), -1)
    return [float(np.linalg.norm(p2[:, j] - y2[:, j]) / max(np.linalg.norm(y2[:, j]), 1e-300))
            for j in range(y2.shape[1])]


def _uncertainty_report(model, y, p):
    import numpy as np
    fn = getattr(model, "predict_std", None)
    std = fn() if callable(fn) else None
    if std is None:
        return None, None
    std = np.asarray(std, float).reshape(np.asarray(p).shape)
    err = np.abs(np.asarray(p, float) - np.asarray(y, float))
    z = err / np.maximum(std, 1e-300)
    return std, {"coverage_1sigma": float(np.mean(z < 1)), "coverage_2sigma": float(np.mean(z < 2)),
                 "mean_std": float(np.mean(std)), "z_rms": float(np.sqrt(np.mean(z ** 2)))}


def _model_diag(model):
    out = {}
    for k in ("fit_report_", "last_chi2_per_dof_", "last_solve_seconds_", "last_iters_", "last_ess_"):
        v = getattr(model, k, None)
        if v is not None:
            out[k.rstrip("_")] = v
    return out


def run_method_eval(job, data, out_dir: Path):
    import numpy as np

    from . import methods_v2 as M2
    from . import metrics as M
    for k in [k for k in data if k not in ("X_train", "y_train", "X_val", "y_val")]:
        del data[k]                       # test / shift splits never reach development jobs
    ms = _materialize_code(job["method"], out_dir, f"code_{job['exp_id'].replace('-', '_')}")
    if _has_code([ms]):
        _preload_and_guard([ms], [out_dir])
    ctx = _make_context(job)
    Xtr, ytr = data["X_train"], data["y_train"]
    idx = _subsample(len(Xtr), float(job["fraction"]), int(job.get("data_seed", 0)))
    Xs, ys = Xtr[idx], ytr[idx]
    vidx = _val_subsample(len(data["X_val"]), job.get("val_subsample"), int(job.get("data_seed", 0)))
    Xv, yv = data["X_val"][vidx], data["y_val"][vidx]
    nf = _n_features_for(ms, job.get("feature_shape"), Xs)
    model = M2.build(ms, job["task"], job.get("feature_shape"), nf, context=ctx, device=job.get("device", "cpu"),
                     seed=int(job.get("seed", 0)), threads=int(job.get("threads", 1)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model.fit(Xs, ys)
        fit_s = time.time() - t0
        fwd_fit = ctx.forward_evals if ctx is not None else 0
        t0 = time.time()
        pv = np.asarray(model.predict(Xv))
        pred_s = time.time() - t0
    warn = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})[:5]
    if job["task"] == "regression":
        pv = pv.reshape(yv.shape)
        if not np.all(np.isfinite(pv)):
            raise FloatingPointError("method produced non-finite validation predictions")
    names = [job["primary"]] + [m for m in job.get("secondary", []) if m != job["primary"]]
    val = M.compute_all(names, yv, pv)
    out = {"val": val, "n_train_used": int(len(idx)), "n_val_used": int(len(vidx)), "fit_seconds": fit_s,
           "predict_seconds": pred_s, "latency_ms_per_sample": 1000.0 * pred_s / max(1, len(vidx)),
           "warnings": warn, "kind": ms.get("kind"), "family": M2.family(ms), "label": M2.label(ms),
           "forward_evals": (ctx.forward_evals if ctx is not None else 0), "forward_evals_fit": fwd_fit,
           "diagnostics": _model_diag(model)}
    if job["task"] == "regression":
        out["val_per_target_rel_l2"] = _per_target(yv, pv)
        _, unc = _uncertainty_report(model, yv, pv)
        if unc:
            out["uncertainty"] = unc
    if ms.get("kind") in ("library", "custom_ml"):
        k = min(len(Xs), 2000)
        ptr = np.asarray(model.predict(Xs[:k]))
        out["train"] = M.compute_all([job["primary"]], ys[:k],
                                     ptr.reshape(ys[:k].shape) if job["task"] == "regression" else ptr)
    np.save(out_dir / "val_pred.npy", pv)
    np.save(out_dir / "val_idx.npy", vidx)
    out["val_pred_path"] = str(out_dir / "val_pred.npy")
    return out


def _bootstrap(name, y, p, n_boot: int, seed: int):
    import numpy as np

    from . import metrics as M
    rng = np.random.default_rng(seed)
    vals = np.asarray([M.compute(name, y[b], p[b]) for b in (rng.integers(0, len(y), len(y)) for _ in range(int(n_boot)))])
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))], "n_boot": int(n_boot)}


def _versions():
    import platform as _pf
    out = {"python": sys.version.split()[0], "platform": _pf.platform()}
    for mod in ("numpy", "scipy", "sklearn", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def run_final_v2(job, data, out_dir: Path):
    import hashlib

    import numpy as np

    from . import methods_v2 as M2
    from . import metrics as M
    from .export import FinalModel, combine, load_model
    final_dir = Path(job["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    specs = [_materialize_code(ms, final_dir, f"member{i}_{job['run_id']}") for i, ms in enumerate(job["members_v2"])]
    if _has_code(specs):
        _preload_and_guard(specs, [out_dir, final_dir])
    sim_copy = None
    if (job.get("simulator") or {}).get("name") == "python":
        from .simulators import resolve_user_path
        src = resolve_user_path(job["simulator"]["params"]["path"])
        if src.exists():
            sim_copy = final_dir / ("simulator_" + src.name)
            sim_copy.write_bytes(src.read_bytes())
    X = np.concatenate([data["X_train"], data["X_val"]])
    y = np.concatenate([data["y_train"], data["y_val"]])
    members, fit_total, member_info = [], 0.0, []
    for i, ms in enumerate(specs):
        ctx = _make_context(job, authorized=bool(job["forward_access_members"][i]))
        if ctx is not None and sim_copy is not None:
            ctx.fallback_path = str(sim_copy)
        nf = _n_features_for(ms, job.get("feature_shape"), X)
        model = M2.build(ms, job["task"], job.get("feature_shape"), nf, context=ctx, device=job.get("device", "cpu"),
                         seed=int(job.get("seed", 0)), threads=int(job.get("threads", 1)))
        t0 = time.time()
        model.fit(X, y)
        fit_total += time.time() - t0
        members.append(model)
        member_info.append({"label": M2.label(ms), "kind": ms.get("kind"), "family": M2.family(ms),
                            "uses_forward": M2.uses_forward(ms), "method_hash": M2.method_hash(ms)})
    names = [job["primary"]] + [m for m in job.get("secondary", []) if m != job["primary"]]

    def predict_all(Xe):
        t0 = time.time()
        preds = [np.asarray(m.predict(Xe)) for m in members]
        pt = preds[0] if len(preds) == 1 else combine(preds, job["weights"], job["task"])
        return np.asarray(pt), preds, time.time() - t0

    pt, preds, pred_s = predict_all(data["X_test"])
    if job["task"] == "regression":
        pt = pt.reshape(data["y_test"].shape)
    test = M.compute_all(names, data["y_test"], pt)
    out = {"test": test, "fit_seconds": fit_total, "predict_seconds": pred_s,
           "latency_ms_per_sample": 1000.0 * pred_s / max(1, len(pt)),
           "test_bootstrap": _bootstrap(job["primary"], data["y_test"], pt, job.get("bootstrap", 200),
                                        int(job.get("seed", 0)))}
    if job["task"] == "regression":
        out["test_per_target_rel_l2"] = _per_target(data["y_test"], pt)
        out["member_test"] = [{"label": mi["label"], job["primary"]: M.compute(
            job["primary"], data["y_test"], np.asarray(p).reshape(data["y_test"].shape))}
            for mi, p in zip(member_info, preds)]
        if len(members) == 1:
            _, unc = _uncertainty_report(members[0], data["y_test"], pt)
            if unc:
                out["test_uncertainty"] = unc
        err = np.linalg.norm((pt - data["y_test"]).reshape(len(pt), -1), axis=1) / \
            np.maximum(np.linalg.norm(data["y_test"].reshape(len(pt), -1), axis=1), 1e-300)
        out["test_sample_rel_error_quantiles"] = {q: float(np.quantile(err, float(q))) for q in ("0.5", "0.9", "0.99")}
    shifts = {}
    for key in sorted(k for k in data if k.startswith("X_shift_")):
        name = key[len("X_shift_"):]
        ps, _, _ = predict_all(data[key])
        ys_ = data["y_shift_" + name]
        if job["task"] == "regression":
            ps = ps.reshape(ys_.shape)
        shifts[name] = M.compute_all(names, ys_, ps)
    out["shift"] = shifts
    clean_specs = [{k: v for k, v in s.items() if k != "code"} for s in specs]
    fm = FinalModel(members, job["weights"], job["task"], clean_specs,
                    {"test": test, "run_id": job["run_id"], "n_train": int(len(X)), "pipeline": "v2"})
    with open(final_dir / "model.pkl", "wb") as f:
        pickle.dump(fm, f)
    np.save(final_dir / "test_pred.npy", pt)
    custom = [{"file": Path(s["code_path"]).name, "module_name": s["module_name"]} for s in specs if s.get("code_path")]
    split_sha = hashlib.sha1(np.ascontiguousarray(data["X_test"]).tobytes()).hexdigest()[:16]
    manifest = {"pipeline": "v2", "members": clean_specs, "member_info": member_info, "weights": job["weights"],
                "task": job["task"], "test": test, "test_bootstrap": out["test_bootstrap"], "shift": shifts,
                "custom_modules": custom, "feature_dim": int(data["X_test"].shape[1]),
                "target_shape": list(data["y_test"].shape[1:]), "simulator": job.get("simulator"),
                "simulator_copy": sim_copy.name if sim_copy else None,
                "forward_access_members": job["forward_access_members"],
                "reproducibility": {"seed": job.get("seed"), "data_seed": job.get("data_seed"),
                                    "data_cache_key": job.get("data_cache_key"), "test_split_sha1": split_sha,
                                    "spec_hash": job.get("spec_hash"), "versions": _versions(),
                                    "harness_version": job.get("harness_version")},
                "research_lineage": job.get("lineage", {})}
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    # reproducibility evidence: re-load the exported artefact and re-predict a slice of the test set
    k = min(50, len(pt))
    re = np.asarray(load_model(final_dir / "model.pkl").predict(data["X_test"][:k]))
    if job["task"] == "regression":
        out["reload_max_abs_diff"] = float(np.max(np.abs(re.reshape(pt[:k].shape) - pt[:k])))
    else:
        out["reload_max_abs_diff"] = float(np.mean(re.ravel() != pt[:k].ravel()))
    out["model_path"] = str(final_dir / "model.pkl")
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    job_path = Path(argv[0])
    job = json.loads(job_path.read_text())
    out_dir = job_path.parent
    t0 = time.time()
    result = {"exp_id": job.get("exp_id"), "idea_id": job.get("idea_id"), "mode": job["mode"],
              "fraction": job.get("fraction"), "device": job.get("device", "cpu")}
    try:
        import numpy as np
        with np.load(job["data_path"]) as d:
            data = {k: d[k] for k in d.files}
        if job["mode"] == "train_eval":
            result.update(run_train_eval(job, data, out_dir))
        elif job["mode"] == "method_eval":
            result.update(run_method_eval(job, data, out_dir))
        elif job["mode"] == "final_v2":
            result.update(run_final_v2(job, data, out_dir))
        else:
            result.update(run_final(job, data, out_dir))
        result["status"] = "ok"
    except PermissionError as exc:      # sandbox violations and unauthorised forward-model access
        result.update(status="error", error=f"{type(exc).__name__}: {exc}"[:600],
                      traceback=traceback.format_exc()[-4000:], safety_violation=True)
    except MemoryError:
        result.update(status="error", error="MemoryError", traceback=traceback.format_exc()[-4000:])
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}"[:600],
                      traceback=traceback.format_exc()[-4000:])
    result["elapsed"] = time.time() - t0
    (out_dir / "result.json").write_text(json.dumps(result, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
