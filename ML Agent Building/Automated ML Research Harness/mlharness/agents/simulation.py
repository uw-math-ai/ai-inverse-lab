"""Simulation agent: generates (and caches) the synthetic train/val/test data."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .. import HARNESS_ROOT
from .. import artifacts as A
from ..simulators import get_simulator
from .base import Agent, log


class SimulationAgent(Agent):
    role = "simulation"

    def __init__(self, *a, cache_dir: Path | None = None, **kw):
        super().__init__(*a, **kw)
        self.cache_dir = Path(cache_dir or HARNESS_ROOT / ".cache" / "datasets")
        self.simulator = get_simulator(self.spec.simulator)

    def generate(self, run_dir: Path) -> tuple[Path, A.Artifact]:
        d = self.spec.data
        n_total = d["n_train"] + d["n_val"] + d["n_test"]
        key = hashlib.sha1(json.dumps({"sim": self.simulator.config_hash(), "n": n_total,
                                       "seed": d["seed"]}).encode()).hexdigest()[:16]
        cached = self.cache_dir / f"{key}.npz"
        out = Path(run_dir) / "data" / "dataset.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if cached.exists():
            shutil.copy(cached, out)
            gen_s, from_cache = 0.0, True
            log.info("[%s] dataset %s loaded from cache", self.name, key)
        else:
            X, y = self.simulator.generate(n_total, int(d["seed"]))
            # one draw, then split: simulators with hidden global parameters stay consistent
            perm = np.random.default_rng(int(d["seed"]) + 1).permutation(n_total)
            a, b = d["n_train"], d["n_train"] + d["n_val"]
            parts = {"X_train": X[perm[:a]], "y_train": y[perm[:a]], "X_val": X[perm[a:b]],
                     "y_val": y[perm[a:b]], "X_test": X[perm[b:]], "y_test": y[perm[b:]]}
            np.savez(out, **parts)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, cached)
            gen_s, from_cache = time.time() - t0, False
            log.info("[%s] generated %d samples with %s in %.1fs", self.name, n_total, self.simulator.name, gen_s)
        with np.load(out) as z:
            Xtr, ytr = z["X_train"], z["y_train"]
            stats = {"X_shape": {k: list(z[k].shape) for k in z.files},
                     "n_features": int(Xtr.shape[1]),
                     "target_dim": int(1 if ytr.ndim == 1 else ytr.shape[1]),
                     "feature_min": float(Xtr.min()), "feature_max": float(Xtr.max()),
                     "nan_count": int(sum(np.isnan(z[k]).sum() for k in z.files if z[k].dtype.kind == "f")),
                     "target_mean": np.atleast_1d(ytr.mean(axis=0)).round(4).tolist()[:20],
                     "target_std": np.atleast_1d(ytr.std(axis=0)).round(4).tolist()[:20]}
            if self.spec.task == "classification":
                vals, cnt = np.unique(ytr, return_counts=True)
                stats["class_counts"] = {str(v): int(c) for v, c in zip(vals, cnt)}
        issues = []
        if stats["nan_count"]:
            issues.append(f"{stats['nan_count']} NaNs in data")
        if self.spec.task == "regression" and max(stats["target_std"]) == 0:
            issues.append("targets are constant")
        art = self.emit(A.DATASET, {"path": str(out), "cache_key": key, "from_cache": from_cache,
                                    "generation_seconds": round(gen_s, 3), "simulator": self.spec.simulator,
                                    "simulator_hash": self.simulator.config_hash(),
                                    "feature_shape": self.simulator.feature_shape,
                                    "feature_sets": self.simulator.feature_sets, "splits": d,
                                    "stats": stats, "issues": issues},
                        summary=f"{self.simulator.name}: {n_total} samples, {stats['n_features']} features, "
                                f"target dim {stats['target_dim']}", stage="data", files=[str(out)])
        if issues:
            raise RuntimeError(f"simulated data failed validation: {issues}")
        return out, art
