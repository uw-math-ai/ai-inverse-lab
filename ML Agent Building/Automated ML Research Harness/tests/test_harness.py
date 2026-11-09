"""Tests for the harness. Run from the harness folder:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlharness import artifacts as A  # noqa: E402
from mlharness import methods as ML  # noqa: E402
from mlharness import metrics as M  # noqa: E402
from mlharness.agents.critic import check_code  # noqa: E402
from mlharness.llm import LLMBackend, LLMResponse, extract_json  # noqa: E402
from mlharness.simulators import get_simulator  # noqa: E402
from mlharness.simulators.graph_fpt import GraphFPTSimulator, exact_first_passage  # noqa: E402
from mlharness.spec import compile_spec  # noqa: E402

HW = {"cpus": 4, "gpus": [], "accelerator": "cpu", "torch": False, "platform": "test", "python": "3"}


def tiny_spec(**over):
    raw = {"name": "tiny", "problem": "tiny friedman test",
           "simulator": {"name": "friedman", "params": {"n_features": 6, "noise": 0.5}},
           "data": {"n_train": 300, "n_val": 100, "n_test": 100, "seed": 0},
           "metrics": {"primary": "rmse", "secondary": ["r2"]},
           "budget": {"compute_seconds": 120, "max_ideas": 12, "experiment_timeout": 60, "llm_calls": 20},
           "hardware": {"workers": 2},
           "constraints": {"allowed_methods": ["ridge", "knn", "extra_trees"]},
           "search": {"ideators": ["classical", "ensembles"], "ideas_per_ideator": 2, "refine_rounds": 1,
                      "refine_children": 1, "min_full": 2},
           "llm": {"backend": "none"}}
    raw.update(over)
    return compile_spec(raw, hardware=HW)


class TestMetrics(unittest.TestCase):
    def test_directions_and_scores(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(M.rel_l2(y, y), 0.0)
        self.assertGreater(M.to_score(0.1, "rel_l2"), M.to_score(0.2, "rel_l2"))
        self.assertGreater(M.to_score(0.9, "r2"), M.to_score(0.5, "r2"))
        self.assertTrue(M.meets(0.04, "rel_l2", 0.05))
        self.assertFalse(M.meets(0.8, "accuracy", 0.9))
        self.assertEqual(M.to_score(None, "rmse"), -np.inf)


class TestArtifacts(unittest.TestCase):
    def test_store_lineage_and_resume(self):
        d = Path(tempfile.mkdtemp())
        try:
            s = A.FileSystemStore(d)
            a = s.put(A.SPEC, {"x": 1}, author="t")
            b = s.put(A.HYPOTHESIS, {"y": np.float64(2.0)}, author="t", idea_id="idea-1", parents=[a.id])
            c = s.put(A.RESULT, {}, author="t", idea_id="idea-1", parents=[b.id])
            self.assertEqual([x.id for x in s.lineage(c.id)], [c.id, b.id, a.id])
            with self.assertRaises(FileExistsError):
                A.FileSystemStore(d)
            s2 = A.FileSystemStore(d, resume=True)
            self.assertEqual(len(s2.all()), 3)
            self.assertEqual(len(s2.query(idea_id="idea-1")), 2)
            self.assertNotEqual(s2.put(A.CRITIQUE, {}, author="t").id, c.id)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestSimulators(unittest.TestCase):
    def test_graph_fpt_mass_and_determinism(self):
        sim = GraphFPTSimulator(walks_per_pair=0, potential={"mode": "uniform", "low": 0, "high": 0.5},
                                target="both")
        X, y = sim.generate(20, 3)
        r = X.reshape(20, *sim.feature_shape)
        self.assertTrue(np.all(r >= -1e-15))
        self.assertTrue(np.all(r.sum(axis=1) <= 1 + 1e-12))       # hits within T never exceed 1
        self.assertEqual(y.shape, (20, len(sim.edges) + sim.n))
        X2, _ = sim.generate(20, 3)
        np.testing.assert_array_equal(X, X2)

    def test_exact_recursion_matches_matrix_power_definition(self):
        sim = GraphFPTSimulator(walks_per_pair=0)
        w, q = sim.sample_parameters(2, np.random.default_rng(1))
        K = sim.kernel(w, q)
        r = exact_first_passage(K, sim.B, 4)
        # brute force: propagate the distribution, removing mass that has already reached y
        for s in range(2):
            for j, yv in enumerate(sim.B):
                for i, xv in enumerate(sim.B):
                    v = np.zeros(sim.n)
                    v[xv] = 1.0
                    v = v @ K[s]            # step 1 from x (a holding step counts as a return)
                    self.assertAlmostEqual(r[s, 0, i, j], v[yv], places=12)
                    for t in range(1, 4):
                        v[yv] = 0.0
                        v = v @ K[s]
                        self.assertAlmostEqual(r[s, t, i, j], v[yv], places=12)

    def test_noisy_counts_are_close_to_exact(self):
        exact = GraphFPTSimulator(walks_per_pair=0).generate(5, 9)[0]
        noisy = GraphFPTSimulator(walks_per_pair=100000).generate(5, 9)[0]
        self.assertLess(np.abs(exact - noisy).max(), 0.01)

    def test_cross_check_with_lab_fptsim(self):
        fpt_root = ROOT.parent.parent / "Toggable Simulation of Data Code"
        if not (fpt_root / "fptsim").exists():
            self.skipTest("fptsim not found")
        sys.path.insert(0, str(fpt_root))
        try:
            from fptsim.exact import exact_single
            from fptsim.kernel import build_kernel
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"fptsim import failed: {exc}")
        sim = GraphFPTSimulator(walks_per_pair=0, potential={"mode": "sparse", "fraction": 0.5, "high": 0.4})
        w, q = sim.sample_parameters(2, np.random.default_rng(0))
        r = exact_first_passage(sim.kernel(w, q), sim.B, sim.T)
        for s in range(2):
            W = np.zeros((sim.n, sim.n))
            for (u, v), wt in zip(sim.edges, w[s]):
                W[u, v] = W[v, u] = wt
            ref = exact_single(build_kernel(W, (W > 0).sum(1).astype(float), q[s], "auto"), sim.B, sim.T)
            np.testing.assert_allclose(ref["r"], r[s], atol=1e-13)


class TestMethods(unittest.TestCase):
    def test_every_method_builds_and_fits(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 6))
        for task in ("regression", "classification"):
            y = X[:, :2] @ np.array([[1.0, 0.5], [0.2, -1.0]]) if task == "regression" else (X[:, 0] > 0).astype(int)
            for key, m in ML.methods_for(task).items():
                cfg = {"method": key, "params": dict(m.defaults), "features": "raw", "scaler": "standard", "pca": 0}
                if key == "torch_mlp":
                    cfg["params"]["epochs"] = 3
                if key == "hist_gbm":
                    cfg["params"]["max_iter"] = 10
                model = ML.build_pipeline(cfg, task, None, 6)
                model.fit(X, y)
                p = np.asarray(model.predict(X))
                self.assertEqual(len(p), 80, key)

    def test_clip_and_mutate_stay_in_space(self):
        m = ML.LIBRARY["knn"]
        p, notes = ML.clip_params(m, {"n_neighbors": 1000, "bogus": 1})
        self.assertEqual(p["n_neighbors"], 60)
        self.assertTrue(any("bogus" in n for n in notes))
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = ML.mutate_params(m, p, rng)
            self.assertTrue(1 <= q["n_neighbors"] <= 60)

    def test_fpt_features(self):
        sim = GraphFPTSimulator(walks_per_pair=0)
        X, _ = sim.generate(4, 0)
        from mlharness.features import FeatureTransform
        for kind in sim.feature_sets:
            Z = FeatureTransform(kind, shape=sim.feature_shape).transform(X)
            self.assertEqual(len(Z), 4)
            self.assertTrue(np.all(np.isfinite(Z)), kind)


class TestCriticAndLLMUtils(unittest.TestCase):
    def test_code_checks(self):
        self.assertEqual(check_code("from sklearn.linear_model import Ridge\ndef build_model(p, t, s):\n    return Ridge()"), [])
        self.assertTrue(check_code("import subprocess\ndef build_model(p,t,s): pass"))
        self.assertTrue(check_code("def build_model(p,t,s):\n    open('x','w')"))
        self.assertTrue(check_code("import os\ndef build_model(p,t,s):\n    os.system('ls')"))
        self.assertEqual(check_code("def build_model(p,t,s):\n    net.eval()\n    return net"), [])  # torch eval ok
        self.assertTrue(check_code("def f(): pass"))
        self.assertTrue(check_code("def build_model(:"))

    def test_extract_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Here you go: {"a": [1, 2]} thanks'), {"a": [1, 2]})
        self.assertIsNone(extract_json("no json here"))

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            compile_spec({"simulator": {"name": "friedman"}, "metrics": {"primary": "accuracy"}}, hardware=HW)
        s = tiny_spec()
        self.assertEqual(s.task, "regression")
        self.assertEqual(s.success["metric"], "rmse")
        self.assertEqual(s.search["rungs"][-1], 1.0)


GOOD_CODE = ("from sklearn.linear_model import Ridge\n"
             "def build_model(params, task, seed):\n"
             "    return Ridge(alpha=params.get('alpha', 1.0))\n")
BAD_CODE = ("from sklearn.linear_model import Ridge\n"
            "def build_model(params, task, seed):\n"
            "    return Ridgee(alpha=1.0)\n")


class ScriptedLLM(LLMBackend):
    """Deterministic stand-in for a coding agent; exercises every LLM code path."""
    name = "scripted"
    model = "scripted-1"

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt, timeout=None):
        self.calls.append(system[:40])
        if "fix Python model code" in system:
            out = {"code": GOOD_CODE, "explanation": "typo Ridgee -> Ridge"}
        elif "critic agent" in system:
            out = {"reject": [], "advice": "prefer regularised models"}
        elif "reporting agent" in system:
            return LLMResponse("Ridge-type models worked best.", 0.01, self.name, self.model)
        elif "Best full-fidelity ideas" in prompt:
            parent = re.search(r"parent (idea-\d+)", prompt).group(1)
            out = {"ideas": [{"parent": parent, "title": "stronger ridge", "hypothesis": "more shrinkage",
                              "method": "ridge", "params": {"alpha": 7.0}, "features": "raw"}]}
        elif "Propose" in prompt:
            out = {"ideas": [
                {"title": "ridge", "hypothesis": "linear", "method": "ridge", "params": {"alpha": 0.3},
                 "features": "raw"},
                {"title": "custom ridge", "hypothesis": "agent code", "method": "custom", "code": BAD_CODE,
                 "params": {"alpha": 2.0}},
                {"title": "invalid", "hypothesis": "nope", "method": "does_not_exist"}]}
        else:
            out = {}
        return LLMResponse(json.dumps(out), 0.01, self.name, self.model, cost_usd=0.0)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mlh_test_"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_offline_run_resume_and_predict(self):
        from mlharness.export import load_model
        from mlharness.scheduler import ResearchScheduler
        spec = tiny_spec()
        out = ResearchScheduler(spec, self.dir / "run", verbose=False).run()
        fin = out["final"]
        self.assertTrue(Path(fin["model_path"]).exists())
        self.assertIsNotNone(fin["test"]["rmse"])
        model = load_model(self.dir / "run")
        with np.load(self.dir / "run" / "data" / "dataset.npz") as z:
            p = model.predict(z["X_test"])
            self.assertAlmostEqual(M.rmse(z["y_test"], p), fin["test"]["rmse"], places=6)
        self.assertTrue((self.dir / "run" / "report.md").exists())
        store = A.FileSystemStore(self.dir / "run", resume=True)
        self.assertTrue(store.query(A.CHECKPOINT))
        kinds = {a.kind for a in store.lineage(store.latest(A.CHECKPOINT).id)}
        self.assertTrue({"checkpoint", "result", "plan", "hypothesis", "spec", "dataset"} <= kinds)
        # the whole report can be rebuilt from the artifact ledger alone
        from mlharness.state import RunState, rebuild_report
        st = RunState(self.dir / "run")
        self.assertEqual(st.final["members"], fin["members"])
        self.assertAlmostEqual(st.final["test"]["rmse"], fin["test"]["rmse"])
        self.assertIn("Lineage of the final model", rebuild_report(self.dir / "run"))
        # resume: every train_eval experiment is served from the cache
        ResearchScheduler(spec, self.dir / "run", resume=True, verbose=False).run()
        store = A.FileSystemStore(self.dir / "run", resume=True)
        cached = [a for a in store.query(A.RESULT) if a.data.get("cached")]
        self.assertGreater(len(cached), 3)

    def test_llm_paths_with_scripted_agent(self):
        from mlharness.scheduler import ResearchScheduler
        spec = tiny_spec(constraints={"allowed_methods": ["ridge", "knn"], "allow_custom_code": True},
                         search={"ideators": ["classical", "ensembles"], "ideas_per_ideator": 3,
                                 "refine_rounds": 1, "refine_children": 1, "min_full": 2})
        llm = ScriptedLLM()
        sch = ResearchScheduler(spec, self.dir / "run", llm=llm, verbose=False)
        out = sch.run()
        origins = {i["origin"] for i in sch.ideas.values()}
        self.assertIn("llm", origins)
        custom = [i for i in sch.ideas.values() if i["config"]["method"] == "custom"]
        self.assertTrue(custom)
        self.assertEqual(custom[0].get("code_version"), 2)            # repaired after the crash
        self.assertEqual(sch.results[custom[0]["id"]][spec.search["screen_fraction"]]["status"], "ok")
        self.assertTrue(any("unknown method" in r for r in sch.pruned.values()))
        self.assertTrue(any(i["origin"] == "llm" and i["parent"] for i in sch.ideas.values()))  # refinement
        self.assertIn("Ridge-type", (self.dir / "run" / "report.md").read_text(encoding="utf-8"))
        self.assertIsNotNone(out["final"]["test"]["rmse"])


if __name__ == "__main__":
    unittest.main()
