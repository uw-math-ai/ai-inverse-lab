"""Offline fixtures for evidence routing, durable packages and evaluation protocols."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mlharness import artifacts as A
from mlharness.knowledge_v3 import (EvidenceRouter, ExposureRegistry, MethodPackageStore,
                                  REDACTED, evaluate_program, safe_projection,
                                  validate_evaluation_protocol)
from mlharness.memory import InMemoryBackend, ResearchMemory


class TestEvidenceRoutingV3(unittest.TestCase):
    def test_recursive_projection_and_free_text(self):
        raw = {"data": [{"metrics": {"test": 0.0123, "validation": 0.5}},
                        {"split": "test", "value": 0.0123},
                        {"stage": "final", "score": 0.0123},
                        {"split_role": "final_held_out", "score": 0.0123}],
               "text": "Winner test error was 0.0123", "title": "training failure",
               "test_bootstrap": {"ci": [0.01, 0.02]}, "nan": float("nan")}
        view = safe_projection(raw)
        self.assertNotIn("0.0123", json.dumps(view))
        self.assertEqual(view["data"][0]["metrics"]["validation"], 0.5)
        self.assertEqual(view["text"], REDACTED)
        self.assertEqual(view["title"], "training failure")
        self.assertIsNone(view["nan"])
        self.assertEqual(raw["data"][0]["metrics"]["test"], 0.0123)

    def test_local_source_never_queries_theorems(self):
        class Forbidden:
            def search(self, *args, **kwargs):
                raise AssertionError("local queries must never reach retrieval")
        router = EvidenceRouter({"noise_scale": {"added_std": "sigma * scale"}}, theorem_client=Forbidden())
        first = router.resolve({"type": "local", "key": "noise_scale", "id": "r1"})
        second = router.resolve({"type": "local", "key": "noise_scale", "id": "r2", "agent_id": "other"})
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["data"]["added_std"], "sigma * scale")
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["request_id"], "r2")
        self.assertEqual(second["agent_id"], "other")
        self.assertEqual(router.resolve({"type": "local", "key": "C:/private.env"})["status"], "unavailable")

    def test_artifact_projection_and_final_denial(self):
        with tempfile.TemporaryDirectory() as d:
            store = A.FileSystemStore(Path(d) / "run")
            dev = store.put(A.RESULT, {"val": 0.5, "nested": {"test": 0.0123}}, author="fixture", stage="evaluate")
            final = store.put(A.RESULT, {"val": 0.0123}, author="fixture", stage="final")
            router = EvidenceRouter({}, store=store)
            r = router.resolve({"type": "artifact", "artifact_id": dev.id})
            self.assertEqual(r["status"], "ok")
            self.assertNotIn("0.0123", json.dumps(r))
            self.assertEqual(router.resolve({"type": "artifact", "artifact_id": final.id})["status"], "denied")
            self.assertEqual(router.resolve({"type": "artifact", "artifact_id": "missing"})["status"], "unavailable")

    def test_frozen_memory_prioritizes_failures_and_hides_nested_holdout(self):
        mem = ResearchMemory(backend=InMemoryBackend())
        failure = mem.add("failure", title="ridge failed", text="ridge numerical failure",
                          data={"nested": {"test_metrics": {"value": 0.0123}}})
        frozen = EvidenceRouter({}, memory=mem, memory_mode="frozen")
        later = mem.add("finding", title="ridge finding", text="ridge promising")
        cumulative = EvidenceRouter({}, memory=mem)
        a = frozen.resolve({"type": "memory", "query": "ridge"})
        b = cumulative.resolve({"type": "memory", "query": "ridge"})
        self.assertEqual([r["id"] for r in a["data"]], [failure])
        self.assertIn(later, [r["id"] for r in b["data"]])
        self.assertNotIn("0.0123", json.dumps(a))
        self.assertEqual(a["memory_mode"], "frozen")
        state = frozen.snapshot_state()
        resumed = EvidenceRouter({}, memory=mem, memory_mode="frozen")
        resumed.restore_snapshot(state)
        self.assertEqual(resumed.snapshot_id, frozen.snapshot_id)
        self.assertEqual([r["id"] for r in resumed.resolve({"type": "memory", "query": "ridge"})["data"]], [failure])
        state["records"][0]["title"] = "altered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            resumed.restore_snapshot(state)

    def test_theorem_is_a_lead_and_deduplicated(self):
        class Fake:
            calls = 0
            def search(self, query):
                self.calls += 1
                return {"ok": True, "refs": [{"id": "thm1", "body": "Assume a smooth coefficient"}]}
        fake = Fake()
        router = EvidenceRouter({}, theorem_client=fake)
        req = {"type": "theorem", "query": "uniqueness", "assumptions_to_problem": {"smooth": "unknown"}}
        r = router.resolve(req)
        router.resolve(req)
        self.assertEqual(fake.calls, 1)
        self.assertFalse(r["data"]["applicability_verified"])
        self.assertIn("not proof", r["uncertainty"][0])
        self.assertEqual(router.resolve({"type": "empirical_literature", "query": "methods"})["status"], "unsupported")

    def test_diagnostic_executes_adapter_and_preserves_failure(self):
        seen = []
        def diagnostic(req):
            seen.append(req["id"])
            if req.get("fail"):
                raise ValueError("missing diagnostic output")
            if req.get("unsupported"):
                return {"status": "unsupported", "reason": "no adapter"}
            return {"variance": 0.5}
        router = EvidenceRouter({}, diagnostic_fn=diagnostic)
        self.assertEqual(router.resolve({"type": "diagnostic", "id": "d1"})["data"]["variance"], 0.5)
        self.assertEqual(router.resolve({"type": "diagnostic", "id": "d2", "fail": True})["status"], "error")
        self.assertEqual(router.resolve({"type": "diagnostic", "id": "d3", "unsupported": True})["status"], "unsupported")
        self.assertEqual(seen, ["d1", "d2", "d3"])


class TestMethodPackagesV3(unittest.TestCase):
    def test_package_survives_source_removal_and_is_runnable(self):
        from mlharness import methods_v2 as M2
        code = "from sklearn.linear_model import Ridge\ndef build_model(params, task, seed):\n    return Ridge(alpha=1.0)\n"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            origin = root / "run"
            origin.mkdir()
            source = origin / "solver.py"
            source.write_text(code, encoding="utf-8")
            packages = MethodPackageStore(root / "packages")
            saved = packages.save({"kind": "custom_ml", "code_path": str(source), "params": {},
                                   "features": "raw", "scaler": "none", "target_transform": "none"},
                                  failure_conditions=["unmeasured outside training noise regime"])
            shutil.rmtree(origin)
            loaded = packages.load(saved["package_id"])
            self.assertTrue(loaded["content_verified"])
            self.assertFalse(loaded["implementation_verified"])
            # Materialize only this fixed local fixture. Production execution remains isolated.
            materialized = root / "materialized.py"
            materialized.write_text(loaded["method"]["code"], encoding="utf-8")
            method = {**loaded["method"], "code_path": str(materialized), "module_name": "v3_package_fixture"}
            model = M2.build(method, "regression", None, 2, seed=0, threads=1)
            x = np.arange(20, dtype=float).reshape(10, 2)
            model.fit(x, x[:, :1])
            self.assertEqual(np.asarray(model.predict(x)).shape, (10, 1))

    def test_tamper_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            packages = MethodPackageStore(d)
            saved = packages.save({"kind": "library", "method": "ridge"})
            path = Path(saved["path"])
            data = json.loads(path.read_text())
            data["method"]["method"] = "knn"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                packages.load(saved["package_id"])
            with self.assertRaises(ValueError):
                packages.load("../../private")

    def test_package_reapplies_gate(self):
        class Gate:
            def check(self, method, **kwargs):
                return {"approved": False, "reasons": ["custom code disabled"]}
        with tempfile.TemporaryDirectory() as d:
            packages = MethodPackageStore(d)
            saved = packages.save({"kind": "library", "method": "ridge"})
            with self.assertRaisesRegex(ValueError, "rejected by implementation gate"):
                packages.load(saved["package_id"], gate=Gate())

    def test_package_evidence_route_returns_regated_method(self):
        class Gate:
            calls = 0
            def check(self, method, **kwargs):
                self.calls += 1
                return {"approved": True, "method": method}
        with tempfile.TemporaryDirectory() as d:
            packages, gate = MethodPackageStore(d), Gate()
            saved = packages.save({"kind": "library", "method": "ridge"})
            router = EvidenceRouter({}, package_store=packages, package_gate=gate)
            evidence = router.resolve({"type": "method_package", "package_id": saved["package_id"]})
            self.assertEqual(evidence["status"], "ok")
            self.assertEqual(evidence["data"]["method"]["method"], "ridge")
            self.assertEqual(gate.calls, 1)
            self.assertTrue(evidence["data"]["gate_approved"])
            self.assertFalse(evidence["data"]["implementation_verified"])


class TestExposureAndEvaluationV3(unittest.TestCase):
    def test_persistent_exposure_independent_of_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "exposures.sqlite"
            a, b = ExposureRegistry(path), ExposureRegistry(path)
            token = a.reserve("dataset", "split", "run1", "selection", strict=True)
            self.assertEqual(a.reserve("dataset", "split", "run1", "selection", strict=True), token)
            self.assertEqual(b.count("dataset", "split"), 1)
            a.begin(token)
            with self.assertRaises(PermissionError):
                b.begin(token)
            with self.assertRaises(PermissionError):
                b.reserve("other-dataset-name", "split", "run2", "selection", strict=True)
            a.complete(token)
            a.complete(token)
            self.assertEqual(b.get(token)["status"], "completed")
            with self.assertRaises(PermissionError):
                b.reserve("dataset", "split", "run2", "selection", strict=True)
            self.assertNotEqual(b.reserve("dataset", "split", "development", "selection", strict=False), token)

    def test_concurrent_strict_reservation_allows_one(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "exposures.sqlite"
            ExposureRegistry(path)
            def attempt(i):
                try:
                    return ExposureRegistry(path).reserve("d", "s", f"r{i}", "selection", strict=True)
                except PermissionError:
                    return None
            with ThreadPoolExecutor(max_workers=4) as pool:
                tokens = list(pool.map(attempt, range(8)))
            self.assertEqual(sum(x is not None for x in tokens), 1)

    def test_confounded_protocol_and_missing_snapshot_rejected(self):
        p = {"instances": [{"instance_id": "i1", "dataset_seed": 0, "search_seeds": [1, 2]}],
             "memory": {"mode": "frozen"}, "comparisons": [
                 {"arms": [{"dataset_seed": 0}, {"dataset_seed": 1}], "changed_components": ["critic", "pilots"]}]}
        r = validate_evaluation_protocol(p)
        self.assertFalse(r["valid"])
        self.assertTrue(any("confounds dataset_seed" in x for x in r["errors"]))
        self.assertTrue(any("exactly one" in x for x in r["errors"]))
        self.assertTrue(any("snapshot_id" in x for x in r["errors"]))

    def test_valid_protocol_keeps_dataset_and_model_seeds_distinct(self):
        p = {"instances": [{"instance_id": "i1", "problem_id": "heat", "dataset_seed": 7, "search_seeds": [1, 2]},
                           {"instance_id": "i2", "problem_id": "heat", "dataset_seed": 8, "search_seeds": [1, 2]}],
             "memory": {"mode": "frozen", "snapshot_id": "frozen-hash"},
             "comparisons": [{"baseline_hash": "ridge-hash", "budget_policy": "equal_worker_seconds",
                              "changed_components": ["critic"], "arms": [{"dataset_seed": 7}, {"dataset_seed": 7}]}],
             "split_roles": {"validation": "development", "test": "final"}}
        result = validate_evaluation_protocol(p)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["normalized"]["instances"][0]["dataset_seed"], 7)

    def test_good_baseline_does_not_hide_incomplete_study(self):
        result = evaluate_program({"final": {"test": {"rel_l2": 0.01}}, "studies": [
            {"status": "incomplete", "arms": [{"required": True, "status": "ok"}, {"required": True, "status": "failed"}]}],
            "verdicts": [{"outcome": "inconclusive"}, {"outcome": "contradicted"}], "resources": {"compute_used_s": 12}})
        self.assertEqual(result["study_completion_rate"], 0)
        self.assertEqual(result["orphan_studies"], 1)
        self.assertEqual(result["claim_resolution_rate"], 0.5)
        self.assertIsNone(result["transfer"]["established"])
        self.assertEqual(result["cost"]["compute_used_s"], 12)


if __name__ == "__main__":
    unittest.main()
