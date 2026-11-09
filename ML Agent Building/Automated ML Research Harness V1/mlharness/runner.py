"""Isolated experiment execution: ``python -m mlharness.runner job.json``.

Runs in its own process so crashes, memory blow-ups and runaway jobs cannot take down the
scheduler; the parent enforces the timeout. Writes ``result.json`` next to the job file.

Modes
-----
train_eval : fit on a nested subsample (``fraction``) of the training split, score on validation.
final      : fit every member on train+val, score the (weighted) combination on the untouched test
             split, export ``model.pkl`` + ``manifest.json``.
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
        else:
            result.update(run_final(job, data, out_dir))
        result["status"] = "ok"
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
