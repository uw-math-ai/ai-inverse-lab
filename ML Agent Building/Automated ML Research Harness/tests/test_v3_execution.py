"""V3 software fixtures for export integrity, bounded execution and runtime provenance."""
from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mlharness.executor import Executor, cache_key
from mlharness.export import certify_model, load_model
from mlharness.resources import BudgetReservations
from mlharness.runner import run_final_v3, run_method_eval


class LinearFixture:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0] * 2


class StatefulFixture(LinearFixture):
    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return super().predict(X) + self.calls


class BatchFixture(LinearFixture):
    def predict(self, X):
        return super().predict(X) + len(X)


class NonfiniteFixture(LinearFixture):
    def predict(self, X):
        return np.full(len(X), np.nan)


class ExecutionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(24, dtype=float).reshape(12, 2)
        self.method = {"kind": "library", "method": "ridge", "params": {"alpha": 0.01},
                       "features": "raw", "scaler": "standard", "pca": 0}

    def test_certificate_pristine_repeat_reload_partition(self):
        cert = certify_model(LinearFixture(), self.X)
        self.assertTrue(cert["certified"], cert)
        self.assertEqual(set(cert["checks"]), {"finite_shape", "repeat", "reload", "partition"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            path.write_bytes(pickle.dumps(LinearFixture()))
            before = path.read_bytes()
            self.assertTrue(certify_model(path, self.X)["passed"])
            self.assertEqual(before, path.read_bytes())

    def test_certificate_rejects_stateful_batch_and_nonfinite(self):
        for model, failing in ((StatefulFixture(), "repeat"), (BatchFixture(), "partition"),
                               (NonfiniteFixture(), "finite_shape")):
            cert = certify_model(model, self.X)
            self.assertFalse(cert["certified"], cert)
            self.assertFalse(cert["checks"][failing], cert)

    def test_final_exports_pristine_and_scores_exact_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"X_train": self.X[:6], "y_train": self.X[:6, 0]*2,
                    "X_val": self.X[6:9], "y_val": self.X[6:9, 0]*2,
                    "X_test": self.X[9:], "y_test": self.X[9:, 0]*2}
            job = {"pipeline": "v3", "members_v2": [self.method], "weights": [1], "task": "regression",
                   "run_id": "fixture", "final_dir": str(root / "final"), "forward_access_members": [False],
                   "primary": "rel_l2", "secondary": ["rmse"], "seed": 2, "data_seed": 3, "bootstrap": 10}
            with patch("mlharness.methods_v2.build", return_value=LinearFixture()):
                out = run_final_v3(job, data, root)
            self.assertTrue(out["export_certificate"]["certified"], out)
            manifest = json.loads((root / "final/manifest.json").read_text())
            self.assertEqual(manifest["export_certificate"], out["export_certificate"])
            self.assertEqual(manifest["model_sha256"], hashlib.sha256((root / "final/model.pkl").read_bytes()).hexdigest())
            exported = load_model(out["model_path"])
            self.assertNotIn("test", exported.meta)
            np.testing.assert_array_equal(exported.predict(data["X_test"]), np.load(root / "final/test_pred.npy"))
            self.assertEqual(out["test"]["rel_l2"], 0)

    def test_failed_certificate_does_not_touch_final_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Deliberately omit every final array: an uncertified model must not access them.
            data = {"X_train": self.X[:6], "y_train": self.X[:6, 0]*2,
                    "X_val": self.X[6:9], "y_val": self.X[6:9, 0]*2}
            job = {"pipeline": "v3", "members_v2": [self.method], "weights": [1], "task": "regression",
                   "run_id": "bad_fixture", "final_dir": str(root / "final"),
                   "forward_access_members": [False], "primary": "rel_l2"}
            with patch("mlharness.methods_v2.build", return_value=StatefulFixture()):
                out = run_final_v3(job, data, root)
            self.assertTrue(out["final_evaluation_blocked"])
            self.assertFalse(out["certified"])
            self.assertEqual(out["test"], {})
            self.assertFalse((root / "final/test_pred.npy").exists())

    def test_atomic_reservations_protect_confirmation_and_reconcile_failure(self):
        budget = BudgetReservations(100, 25, hard=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            admitted = list(pool.map(lambda n: budget.reserve(f"group{n}", 30), range(8)))
        self.assertEqual(sum(admitted), 2)
        self.assertTrue(budget.reserve("export", 25, confirmation=True))
        self.assertFalse(budget.reserve("extra", 20))
        for n, ok in enumerate(admitted):
            if ok:
                budget.release(f"group{n}", 5)
        budget.release("export", 10)
        snap = budget.snapshot()
        self.assertEqual(snap["reserved_seconds"], 0)
        self.assertEqual(snap["used_seconds"], 20)
        self.assertEqual(snap["policy"], "timeout_bounds")
        self.assertTrue(budget.reserve("after_failure", 60))

    def test_invalid_reservations_and_explicit_timeout(self):
        with self.assertRaises(ValueError):
            BudgetReservations(10, 11)
        budget = BudgetReservations(10)
        for amount in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                budget.reserve("invalid", amount)
        executor = object.__new__(Executor)
        executor.resources = SimpleNamespace(wall_remaining=lambda: 20)
        executor.spec = SimpleNamespace(budget={"experiment_timeout": 100})
        self.assertEqual(executor._timeout({"pipeline": "v3", "mode": "final_v3", "timeout_seconds": 7}, "final"), 7)
        with self.assertRaises(ValueError):
            executor._timeout({"pipeline": "v3", "timeout_seconds": float("nan")}, "final")

    def test_v3_cache_protocol_and_diagnostic_separation(self):
        job = {"pipeline": "v3", "mode": "method_eval", "method": self.method,
               "fraction": 1, "seed": 0, "data_seed": 0}
        first = cache_key(job, "fixture_data")
        self.assertNotEqual(first, cache_key({**job, "diagnostic_requests": [{"kind": "surrogate"}]}, "fixture_data"))
        self.assertNotEqual(first, cache_key({**job, "protocol_version": "different"}, "fixture_data"))
        self.assertNotEqual(first, cache_key({**job, "pipeline": "v2"}, "fixture_data"))

    def test_method_eval_runtime_provenance_and_row_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = {"pipeline": "v3", "method": self.method, "task": "regression", "exp_id": "arm_fixture",
                   "fraction": 1, "seed": 0, "data_seed": 1, "primary": "rel_l2",
                   "study_id": "study", "arm_id": "control", "claim_ids": ["comparison"]}
            data = {"X_train": self.X[:8], "y_train": self.X[:8, 0]*2,
                    "X_val": self.X[8:], "y_val": self.X[8:, 0]*2}
            with patch("mlharness.methods_v2.build", return_value=LinearFixture()):
                out = run_method_eval(job, data, Path(tmp))
            self.assertTrue(out["verification"]["passed"])
            self.assertFalse(out["verification"]["intervention_verified"])
            self.assertEqual(out["arm_id"], "control")
            self.assertTrue(out["row_hash"])
            self.assertTrue(Path(out["val_idx_path"]).exists())


if __name__ == "__main__":
    unittest.main()
