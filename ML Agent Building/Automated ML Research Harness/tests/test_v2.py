"""V2 tests. Run from the harness folder:  python -m unittest discover -s tests -v

Covers: TheoremSearch adapter (success, fallback, outages), applicability screen, claim hygiene,
research memory (persistence, retrieval, contradictions, redaction, cross-run), research-loop
termination (max rounds, quality, LLM budget), proposal provenance and four-researcher responses,
the implementation gate, the runtime sandbox, physics inversion + forward budget, V2 cache keys,
budget enforcement, remote GPU dispatch (fake SSH), .env handling, and end-to-end V2 runs (offline
and with a scripted LLM exercising every agent JSON protocol).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlharness import artifacts as A  # noqa: E402
from mlharness import methods_v2 as M2  # noqa: E402
from mlharness.llm import LLMBackend, LLMResponse  # noqa: E402
from mlharness.memory import InMemoryBackend, ResearchMemory  # noqa: E402
from mlharness.problem import ForwardAccessDenied, ForwardBudgetExceeded, ProblemContext  # noqa: E402
from mlharness.spec import compile_spec  # noqa: E402
from mlharness.tools import LiteratureTool, TheoremSearchClient, assess_applicability  # noqa: E402
from mlharness.tools.theoremsearch import HTTPResponse, TransportError  # noqa: E402

HW = {"cpus": 4, "gpus": [], "accelerator": "cpu", "torch": False, "platform": "test", "python": "3"}
SIM = {"name": "python", "params": {"path": "examples/heat_diffusion_simulator.py", "function": "simulate",
                                    "n_segments": 2, "n_grid": 20, "n_steps": 60, "sensors": [3, 8, 12, 17],
                                    "every": 6, "noise": 0.001}}

SEARCH_PAYLOAD = {"theorems": [{
    "slogan_id": 1, "theorem_id": 101, "name": "Theorem 2.1",
    "body": "Let $u$ solve the heat equation $u_t = (k(x) u_x)_x$ on $(0,1)$ with Dirichlet boundary conditions and "
            "let the coefficient $k \\in C^1$. Then $k$ is uniquely determined by $u(x_0, t)$ at a fixed point.",
    "slogan": "Uniqueness of the diffusion coefficient from a single point measurement.", "theorem_type": "theorem",
    "label": None, "link": None, "similarity": 0.8, "score": 0.8, "has_metadata": True,
    "paper": {"paper_id": "1234.5678v1", "source": "arXiv", "title": "Inverse coefficient problems", "authors": ["A. Author"],
              "link": "http://arxiv.org/abs/1234.5678v1", "year": 2020, "citations": 3}}]}
EMB_PAYLOAD = {"results": [{"statement_id": "abc-1", "paper_id": "p-1", "name": "Lemma 3", "kind": "lemma",
                            "body": "For the parabolic problem the Levenberg-Marquardt iteration converges locally.",
                            "slogan": "LM converges", "title": "Regularization", "authors": ["B. Author"],
                            "url": "https://arxiv.org/abs/1111.2222", "year": 2019, "similarity": 0.7, "score": 0.7}]}


def fake_transport(mode="ok", calls=None):
    def t(method, url, *, params=None, json_body=None, timeout=30.0):
        if calls is not None:
            calls.append((method, url))
        if mode == "timeout":
            raise TransportError("timeout", "read timed out")
        if mode == "network":
            raise TransportError("network", "connection refused")
        if mode == "auth":
            return HTTPResponse(401, '{"detail": "unauthorised"}')
        if mode == "rate":
            return HTTPResponse(429, "slow down", {"Retry-After": "0"})
        if mode == "malformed":
            return HTTPResponse(200, "<html>not json</html>")
        if mode == "server_then_embedding":
            return HTTPResponse(500, "boom") if url.endswith("/search") else HTTPResponse(200, json.dumps(EMB_PAYLOAD))
        if url.endswith("/ping"):
            return HTTPResponse(200, '"pong"')
        return HTTPResponse(200, json.dumps(SEARCH_PAYLOAD if url.endswith("/search") else EMB_PAYLOAD))
    return t


def client(mode="ok", calls=None, **kw):
    return TheoremSearchClient(transport=fake_transport(mode, calls), cache_dir=None, retries=1, max_retry_wait=0.01, **kw)


def tiny_v2(**over):
    raw = {"name": "tiny_heat", "pipeline": "v2", "problem": "recover two conductivities of a rod",
           "simulator": SIM, "data": {"n_train": 240, "n_val": 80, "n_test": 80, "seed": 1},
           "metrics": {"primary": "rel_l2", "secondary": ["r2"]},
           "budget": {"compute_seconds": 150, "experiment_timeout": 60},
           "hardware": {"workers": 2},
           "constraints": {"allowed_methods": ["ridge", "knn", "kernel_ridge", "extra_trees"]},
           "research": {"max_rounds": 2, "min_rounds": 2, "proposals_per_researcher": 2, "pilot_tests": True,
                        "max_pilots_per_round": 2, "pilot_val_subsample": 20, "wall_seconds": 600},
           "literature": {"enabled": True, "max_total_queries": 30},
           "execution": {"portfolio_size": 5, "eval_subsample": 40, "refine_rounds": 1, "bootstrap": 50},
           "evaluation": {"shift_sets": [{"name": "noise_x3", "params": {"noise": 0.003}}], "shift_n": 40},
           "llm": {"backend": "none"}}
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) and isinstance(raw.get(k), dict) else v
    return compile_spec(raw, hardware=HW)


# ------------------------------------------------------------------------------------------------
class TestTheoremSearch(unittest.TestCase):
    def test_success_parses_attribution(self):
        r = client().search("uniqueness of conductivity", 3)
        self.assertTrue(r.ok)
        ref = r.refs[0]
        self.assertEqual(ref.ref_id, "ts:thm:101")
        self.assertEqual(ref.paper_title, "Inverse coefficient problems")
        self.assertIn("arxiv.org", ref.link)
        self.assertIn("A. Author", ref.citation())

    def test_fallback_to_embedding_endpoint(self):
        r = client("server_then_embedding").search("lm convergence")
        self.assertTrue(r.ok)
        self.assertTrue(r.fallback_used)
        self.assertEqual(r.refs[0].ref_id, "ts:stmt:abc-1")

    def test_outages_are_reported_not_raised(self):
        cases = {"timeout": "timeout", "network": "network", "auth": "auth", "rate": "rate_limited",
                 "malformed": "malformed"}
        for mode, kind in cases.items():
            calls = []
            r = client(mode, calls).search("q")
            self.assertFalse(r.ok, mode)
            self.assertEqual(r.error_kind, kind, mode)
            if mode == "auth":                       # no pointless fallback on authentication errors
                self.assertTrue(all(u.endswith("/search") for _, u in calls))
        self.assertEqual(client(enabled=False).search("q").error_kind, "disabled")

    def test_literature_tool_records_failures_and_budget(self):
        d = Path(tempfile.mkdtemp())
        try:
            store = A.FileSystemStore(d)
            mem = ResearchMemory(backend=InMemoryBackend())
            lt = LiteratureTool(client("timeout"), store, mem, facts={}, max_total=2, per_agent_round=1)
            r1 = lt.search("a", "agent", 1)
            r2 = lt.search("b", "agent", 1)              # per-agent-round budget
            self.assertEqual(r1.error_kind, "timeout")
            self.assertEqual(r2.error_kind, "budget")
            self.assertTrue(lt.incomplete())
            self.assertTrue(store.query(A.FAILURE, where=lambda a: a.data.get("source") == "literature"))
            self.assertTrue(store.query(A.LITERATURE))
            ok = LiteratureTool(client(), store, mem, facts={"pde": "parabolic"}, max_total=5)
            ok.search("c", "agent", 1)
            self.assertTrue(ok.known("ts:thm:101"))
            self.assertFalse(ok.known("ts:thm:999"))
            self.assertTrue(mem.has_citation("ts:thm:101"))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestApplicabilityAndClaims(unittest.TestCase):
    FACTS = ProblemContext(SIM, authorized=False).facts

    def test_screen_flags_assumption_conflicts(self):
        a = assess_applicability({"body": SEARCH_PAYLOAD["theorems"][0]["body"], "slogan": ""}, self.FACTS)
        self.assertEqual(a["applies"], "partial")                      # C^1 coefficient vs piecewise constant
        self.assertTrue(any("piecewise" in c for c in a["conflicts"]))
        b = assess_applicability({"body": "Let n >= 2 and u solve the heat equation in R^n. Then ...", "slogan": ""},
                                 self.FACTS)
        self.assertEqual(b["applies"], "no")
        c = assess_applicability({"body": "A policy gradient converges at rate 1/t on [0,1].", "slogan": ""}, self.FACTS)
        self.assertNotEqual(c["applies"], "plausible")                 # generic matches are not enough

    def test_claim_hygiene_never_labels_conjecture_proved(self):
        from mlharness.research.proposals import claim_hygiene, normalize_proposal
        lt = LiteratureTool(client(), None, None, facts=self.FACTS)
        lt.search("x", "a", 1)
        p = normalize_proposal({"title": "t", "hypothesis": "We proved the estimator is optimal.",
                                "status": "formally-verified", "citations": [{"ref_id": "ts:thm:999"}],
                                "method": {"kind": "physics"}}, pid="P-1", author="a", persona="mathematical",
                               round_idx=1, origin="llm")
        notes = claim_hygiene(p, lt)
        self.assertEqual(p["status"], "conjectured")
        self.assertTrue(any("formal" in n for n in notes))
        self.assertTrue(any("unverifiable" in n for n in notes))
        q = normalize_proposal({"title": "t", "status": "literature-supported",
                                "citations": [{"ref_id": "ts:thm:101", "claimed_applicability": "yes"}],
                                "method": {"kind": "physics"}}, pid="P-2", author="a", persona="m", round_idx=1, origin="llm")
        claim_hygiene(q, lt)
        self.assertEqual(q["status"], "literature-supported")         # verified citation, screened partial


class TestMemory(unittest.TestCase):
    def test_persistence_retrieval_contradiction_redaction(self):
        d = Path(tempfile.mkdtemp())
        try:
            path = d / "m.jsonl"
            ctx = {"simulator_hash": "abc", "problem_class": "heat"}
            m1 = ResearchMemory(path=path)
            h = m1.add("hypothesis", title="MLE through the forward model", text="levenberg marquardt likelihood",
                       run_id="run1", status="conjectured", context=ctx)
            m1.add("observation", title="physics MLE val 0.019", run_id="run1", status="experimentally-supported",
                   context=ctx, parents=[h], data={"method_label": "physics-MLE", "val": {"rel_l2": 0.019}})
            m1.add("observation", title="physics MLE val 0.05 under heavy noise", run_id="run2", status="contradicted",
                   context=ctx, parents=[h], data={"method_label": "physics-MLE", "val": {"rel_l2": 0.05}})
            m1.add("finding", title="best method", run_id="run1", context=ctx, data={"test": {"rel_l2": 0.018},
                                                                                       "val": {"rel_l2": 0.019}},
                   redact=("test",))
            m1.add("hypothesis", title="unrelated graph method", text="random walk", run_id="run1", context={"simulator_hash": "zzz"})
            m2 = ResearchMemory(path=path)                       # a new process / run
            self.assertEqual(m2.loaded_from_disk, 5)
            hits = m2.search("levenberg likelihood forward", context=ctx)
            self.assertEqual(hits[0]["id"], h)
            ev = m2.evidence_for("physics-MLE", ctx)
            self.assertEqual({r["status"] for r in ev}, {"experimentally-supported", "contradicted"})  # coexist
            dig = m2.digest("best method", context=ctx, exclude_run="run3")
            self.assertIn("redacted", m2.redacted(m2.search("best method", types=["finding"])[0])["data"]["test"])
            self.assertNotIn("0.018", dig)
            self.assertEqual([r["id"] for r in m2.lineage(ev[0]["id"])][-1], h)
            off = ResearchMemory(path=path, read=False)
            self.assertEqual(off.search("levenberg"), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestPhysics(unittest.TestCase):
    def test_lm_recovers_and_forward_access_is_controlled(self):
        from mlharness.inversion import PhysicsInversion
        sim = {**SIM, "params": {**SIM["params"], "noise": 0.0}}
        ctx = ProblemContext(sim, authorized=True)
        rng = np.random.default_rng(0)
        k = rng.uniform(0.6, 1.9, (12, 2))
        X = ctx.forward(k)
        m = PhysicsInversion(ctx, multistart=2, iters=40, uncertainty=False).fit(X, k)
        self.assertLess(np.abs(m.predict(X) - k).max(), 1e-3)
        with self.assertRaises(ForwardAccessDenied):
            ProblemContext(sim, authorized=False).forward(k)
        with self.assertRaises(ForwardBudgetExceeded):
            ProblemContext(sim, authorized=True, max_forward_evals=5).forward(k)
        import pickle
        m2 = pickle.loads(pickle.dumps(m))                      # exported physics models rebuild their context
        self.assertTrue(m2.context.forward_available)

    def test_posterior_mean_is_batch_independent(self):
        from mlharness.inversion import PhysicsInversion
        ctx = ProblemContext(SIM, authorized=True)
        rng = np.random.default_rng(1)
        k = rng.uniform(0.6, 1.9, (10, 2))
        X = ctx.forward(k) + rng.normal(0, 1e-3, (10, ctx.forward(k[:1]).shape[1]))
        m = PhysicsInversion(ctx, estimator="posterior_mean", posterior_samples=32).fit(X, k)
        np.testing.assert_allclose(m.predict(X)[:3], m.predict(X[:3]), rtol=0, atol=1e-12)


class TestGateAndSafety(unittest.TestCase):
    def gate(self, **cons):
        from mlharness.methods import methods_for
        from mlharness.research import ImplementationGate
        spec = tiny_v2(constraints=cons) if cons else tiny_v2()
        store = A.MemoryStore()
        prob = {"simulator_has_forward": True, "n_params": 2, "sec_per_forward": 1e-5}
        return ImplementationGate(spec, store, methods_for("regression", spec.constraints), ["raw", "log", "sqrt"], 44, prob,
                                  {"target_min": 0.5, "target_dim": 2})

    def test_gate_checks(self):
        g = self.gate()
        self.assertTrue(g.check({"kind": "physics", "params": {}}, label="ok")["approved"])
        self.assertTrue(g.check({"kind": "library", "method": "ridge", "params": {"alpha": 1e9}}, label="clip")["approved"])
        bad_code = "import subprocess\ndef build_method(params, context):\n    return None\n"
        v = g.check({"kind": "custom_solver", "code": bad_code}, label="unsafe")
        self.assertFalse(v["approved"])
        self.assertFalse(v["checks"]["code_safe"])
        dunder = "def build_method(params, context):\n    return context.__class__\n"
        self.assertFalse(g.check({"kind": "custom_solver", "code": dunder}, label="dunder")["approved"])
        leak = "def build_method(params, context):\n    path = 'dataset.npz'\n    return path\n"
        self.assertFalse(g.check({"kind": "custom_solver", "code": leak}, label="leak")["checks"]["code_safe"])
        v = g.check({"kind": "library", "method": "ridge", "params": {"alpha": 1.0, "source": "y_test"}}, label="leak2")
        self.assertFalse(v["approved"])
        self.assertFalse(v["checks"]["no_leakage"])
        self.assertFalse(g.check({"kind": "bogus"}, label="bogus")["approved"])
        g2 = self.gate(allow_forward_model=False)
        v = g2.check({"kind": "physics", "params": {}}, label="noauth")
        self.assertFalse(v["approved"])
        self.assertFalse(v["checks"]["forward_authorised"])
        g3 = self.gate(max_forward_evals=10)
        self.assertFalse(g3.check({"kind": "physics", "params": {}}, label="fwdbudget")["checks"]["resources_ok"])

    def test_runtime_sandbox_blocks_writes_processes_network(self):
        d = Path(tempfile.mkdtemp())
        outside = Path(tempfile.mkdtemp(dir=str(ROOT)))
        try:
            script = textwrap.dedent(f"""
                import sys, json
                sys.path.insert(0, {str(ROOT)!r})
                from mlharness.safety import install_guard
                install_guard([{str(d)!r}])
                res = {{}}
                open({str(d / 'ok.txt')!r}, 'w').write('fine'); res['inside'] = 'ok'
                for name, fn in [('outside_write', lambda: open({str(outside / 'bad.txt')!r}, 'w')),
                                 ('process', lambda: __import__('subprocess').run(['echo', 'x'])),
                                 ('network', lambda: __import__('socket').create_connection(('127.0.0.1', 9)))]:
                    try:
                        fn(); res[name] = 'ALLOWED'
                    except PermissionError as e:
                        res[name] = 'blocked'
                    except Exception as e:
                        res[name] = 'other:' + type(e).__name__
                print(json.dumps(res))
            """)
            out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
            res = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(res["inside"], "ok")
            self.assertEqual(res["outside_write"], "blocked")
            self.assertEqual(res["process"], "blocked")
            self.assertEqual(res["network"], "blocked")
            self.assertFalse((outside / "bad.txt").exists())
        finally:
            shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(outside, ignore_errors=True)


class TestCacheAndBudget(unittest.TestCase):
    def test_v2_cache_key(self):
        from mlharness.executor import cache_key
        base = {"mode": "method_eval", "method": {"kind": "physics", "params": {"iters": 10}}, "fraction": 1.0,
                "val_subsample": 40, "seed": 0, "data_seed": 1, "forward_access": True, "simulator_hash": "s"}
        k0 = cache_key(base, "data")
        self.assertEqual(k0, cache_key(json.loads(json.dumps(base)), "data"))
        for change in ({"method": {"kind": "physics", "params": {"iters": 11}}}, {"fraction": 0.5},
                       {"val_subsample": None}, {"forward_access": False}, {"simulator_hash": "t"}):
            self.assertNotEqual(k0, cache_key({**base, **change}, "data"), change)
        self.assertNotEqual(k0, cache_key(base, "other-data"))
        c1 = {**base, "method": {"kind": "custom_solver", "code": "def build_method(p, c):\n    return 1\n"}}
        c2 = {**base, "method": {"kind": "custom_solver", "code": "def build_method(p, c):\n    return 2\n"}}
        self.assertNotEqual(cache_key(c1, "d"), cache_key(c2, "d"))

    def test_stage_budget_skips_jobs(self):
        from mlharness.executor import Executor
        from mlharness.resources import STAGE_CAPS_V2, ResourceManager
        spec = tiny_v2()
        res = ResourceManager(10, 1000, 1, [], 5, caps=STAGE_CAPS_V2)
        res.charge(5, "baseline")                                 # pilot cap = 12% of 10s = 1.2s -> exhausted
        ex = Executor(spec, A.MemoryStore(), res, "d")
        out = ex.run([{"mode": "method_eval", "exp_id": "e1", "idea_id": "i", "plan_artifact": None, "fraction": 1.0,
                       "method": {"kind": "physics"}, "val_subsample": 5, "seed": 0, "data_seed": 0}], "pilot")
        self.assertEqual(out[0]["status"], "skipped_budget")
        self.assertEqual(res.jobs, 1)

    def test_research_llm_budget(self):
        spec = tiny_v2(research={"llm_calls": 7})
        self.assertGreaterEqual(spec.budget["llm_calls"], 7 + 8)


class TestEnvAndRemote(unittest.TestCase):
    def test_env_file_names_only_and_scrub(self):
        d = Path(tempfile.mkdtemp())
        try:
            p = d / ".env"
            p.write_text("MLH_TEST_USER=alice\nMLH_TEST_PASSWORD='s3cret'\n# c\n", encoding="utf-8")
            from mlharness.envfile import load_env_file, scrubbed_env
            names = load_env_file(p, override=True)
            self.assertEqual(sorted(names), ["MLH_TEST_PASSWORD", "MLH_TEST_USER"])
            self.assertEqual(os.environ["MLH_TEST_PASSWORD"], "s3cret")
            env = scrubbed_env()
            self.assertNotIn("MLH_TEST_PASSWORD", env)
            self.assertIn("MLH_TEST_USER", env)
        finally:
            os.environ.pop("MLH_TEST_USER", None)
            os.environ.pop("MLH_TEST_PASSWORD", None)
            shutil.rmtree(d, ignore_errors=True)

    def test_remote_runner_with_fake_ssh(self):
        from mlharness.remote import RemoteRunner
        commands = []
        local = Path(tempfile.mkdtemp())

        class Chan:
            def __init__(self, code):
                self.code = code

            def recv_exit_status(self):
                return self.code

        class Stream:
            def __init__(self, text, code=0):
                self.text, self.channel = text, Chan(code)

            def read(self):
                return self.text.encode()

        class FakeSFTP:
            files = {}

            def stat(self, p):
                if p not in self.files:
                    raise IOError(p)

            def put(self, a, b):
                self.files[b] = Path(a).read_bytes()

            def open(self, p, mode="r"):
                sftp = self

                class F:
                    def write(self, data):
                        sftp.files[p] = data.encode() if isinstance(data, str) else data

                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        return False
                return F()

            def get(self, remote, localp):
                if remote.endswith("result.json"):
                    Path(localp).write_text(json.dumps({"status": "ok", "val": {"rel_l2": 0.1}}))
                else:
                    raise IOError(remote)

        class FakeClient:
            def exec_command(self, cmd, timeout=None):
                commands.append(cmd)
                if cmd.startswith("hostname"):
                    return None, Stream("gpuhost\n0, TITAN RTX, 1, 24000, 0\npython 3.10.0\nstack 1.26 1.11 1.4\ntorch 2.1 True 1\n"), Stream("")
                if cmd.startswith("echo $HOME"):
                    return None, Stream("/home/u\n"), Stream("")
                if "test -d" in cmd:
                    return None, Stream("absent\n"), Stream("")
                return None, Stream(""), Stream("")

            def open_sftp(self):
                return FakeSFTP()

            def close(self):
                pass

        try:
            data = local / "d.npz"
            np.savez(data, X_train=np.zeros((2, 2)))
            rr = RemoteRunner({"enabled": True, "host": "americano"}, client_factory=lambda cfg: FakeClient())
            job = {"mode": "method_eval", "exp_id": "exp-1", "wants_gpu": True, "data_path": str(data),
                   "job_dir": str(local), "threads": 2, "simulator": {"name": "friedman"}}
            self.assertTrue(rr.accepts(job))
            self.assertFalse(rr.accepts({**job, "mode": "final_v2"}))
            out = rr.run(job, 30)
            self.assertEqual(out["status"], "done")
            self.assertTrue((local / "result.json").exists())
            run_cmds = [c for c in commands if "mlharness.runner" in c]
            self.assertEqual(len(run_cmds), 1)
            self.assertIn("timeout 30", run_cmds[0])
            self.assertTrue(all(("python" not in c) or ("mlharness.runner" in c) or c.startswith("hostname")
                                for c in commands))                       # no other remote python is executed
            bad = RemoteRunner({"enabled": True, "host": "x", "max_failures": 1},
                               client_factory=lambda cfg: (_ for _ in ()).throw(OSError("unreachable")))
            self.assertEqual(bad.run(job, 5)["status"], "remote_unavailable")
            self.assertTrue(bad.disabled)
        finally:
            shutil.rmtree(local, ignore_errors=True)


# ------------------------------------------------------------------------------------------------
SOLVER_CODE = ('import numpy as np\n'
               'class Best:\n'
               '    def __init__(self, context, k):\n'
               '        self.context, self.k = context, k\n'
               '    def fit(self, X, y):\n'
               '        return self\n'
               '    def predict(self, X):\n'
               '        rng = np.random.default_rng(0)\n'
               '        lo, hi = self.context.lower, self.context.upper\n'
               '        cand = rng.uniform(lo, hi, size=(self.k, len(lo)))\n'
               '        F = self.context.forward(cand)\n'
               '        d = ((X[:, None, :] - F[None, :, :]) ** 2).sum(2)\n'
               '        return cand[d.argmin(1)]\n'
               'def build_method(params, context):\n'
               '    return Best(context, int(params.get("k", 200)))\n')


def _prop(title, method, **kw):
    return {"title": title, "hypothesis": f"{title} works", "motivation": "because", "assumptions": ["gaussian noise"],
            "evidence": [{"claim": "diagnostics show full rank", "source": "diagnostics"}],
            "citations": [{"ref_id": "ts:thm:101", "claimed_applicability": "partial", "note": "C^1 vs piecewise"}],
            "method": method, "falsifiable_test": "beats data-driven", "feasibility": {"cost": "low", "notes": "cheap"},
            "prediction": {"relative_to": "best_data_driven", "improvement": 0.0}, "if_fails": "mismatch", **kw}


class ScriptedLLM(LLMBackend):
    name = "scripted"
    model = "scripted-2"

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt, timeout=None):
        self.calls.append(system[:60])
        role = re.search(r"You are the ([\w-]+) researcher", system)
        if "shared research critic" in system:
            ids = re.findall(r'"id": "(P-\d+)"', prompt)
            out = {"reviews": [{"proposal_id": i, "score": 0.8, "challenges": ["check identifiability"],
                                "changed_assumptions": ["noise is exactly Gaussian"], "questions": ["why?"],
                                "evidence_requests": [{"query": "stability of coefficient identification", "why": "x"}],
                                "suggested_tests": [{"description": "pilot", "pilot": True}]} for i in ids],
                   "portfolio": {"summary": "diverse", "disagreements": [{"between": ids[:2], "issue": "ML vs physics"}],
                                 "converged": False}}
        elif "research engineer" in system:
            out = {"code": SOLVER_CODE, "explanation": "nearest prior draw"}
        elif "reporting agent" in system:
            return LLMResponse("Physics inversion is established; ML is not.", 0.01, self.name, self.model)
        elif role and "choose up to" in prompt:
            out = {"queries": [{"query": f"{role.group(1)} heat inverse problem", "why": "background"}]}
        elif role and "meaningfully different" in prompt:
            r = role.group(1)
            if r == "mathematical":
                ps = [_prop("LM MLE", {"kind": "physics", "params": {"iters": 15}})]
            elif r == "machine-learning":
                ps = [_prop("ridge on log", {"kind": "library", "method": "ridge", "params": {"alpha": 0.01},
                                             "features": "log"}),
                      _prop("knn", {"kind": "library", "method": "knn", "params": {"n_neighbors": 5}})]
            elif r == "experimental":
                ps = [_prop("nearest prior draw", {"kind": "custom_solver", "code": SOLVER_CODE, "params": {"k": 300},
                                                   "uses_forward": True})]
            else:
                ps = [_prop("hybrid ridge->LM", {"kind": "hybrid", "init": {"method": "ridge", "params": {"alpha": 0.01},
                                                                            "features": "log"}, "params": {"iters": 5}}),
                      _prop("custom without code", {"kind": "custom_solver", "params": {}, "uses_forward": True})]
            out = {"proposals": ps}
        elif role and "Respond to EVERY critic point" in prompt:
            ids = re.findall(r'"id": "(P-\d+)"', prompt)
            r = role.group(1)
            resp = []
            for i in ids:
                if r == "machine-learning" and i == ids[-1]:
                    resp.append({"proposal_id": i, "action": "abandon", "abandon_reason": "kNN is dominated by ridge",
                                 "response_to_critic": "agreed"})
                elif r == "mathematical":
                    resp.append({"proposal_id": i, "action": "revise", "response_to_critic": "added a second start",
                                 "proposal": _prop("LM MLE 2 starts", {"kind": "physics", "params": {"iters": 15, "multistart": 2}})})
                elif r == "exploratory":
                    resp.append({"proposal_id": i, "action": "branch", "response_to_critic": "branching",
                                 "proposal": _prop("hybrid MAP branch", {"kind": "hybrid", "init": {"method": "ridge", "features": "log"},
                                                                         "params": {"iters": 5, "tikhonov": 0.5}})})
                else:
                    resp.append({"proposal_id": i, "action": "defend", "response_to_critic": "the pilot supports it"})
            out = {"responses": resp, "new_proposals": []}
        else:
            out = {}
        return LLMResponse(json.dumps(out), 0.01, self.name, self.model, cost_usd=0.0)


class TestResearchLoopAndEndToEnd(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mlh_v2_"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def sched(self, spec, llm=None, mem_path=None, sub="run", lit=None):
        from mlharness.scheduler_v2 import ResearchSchedulerV2
        mem = ResearchMemory(path=mem_path or (self.dir / "mem.jsonl"))
        return ResearchSchedulerV2(spec, self.dir / sub, llm=llm, verbose=False, memory=mem,
                                   literature_client=lit or client())

    def test_offline_end_to_end_cross_run_memory_and_export(self):
        from mlharness import metrics as M
        from mlharness.export import load_model
        spec = tiny_v2()
        s1 = self.sched(spec, sub="r1")
        out = s1.run()
        fin = out["final"]
        self.assertTrue(Path(fin["model_path"]).exists())
        rounds = s1.research["rounds"]
        self.assertEqual(len(rounds), 2)
        personas = {a["persona"] for a in rounds[1]["actions"]}
        self.assertEqual(personas, {"mathematical", "ml", "experimental", "exploratory"})   # all four responded
        self.assertTrue(any(i.get("family") == "physics" and i["id"] in s1.results for i in s1.ideas.values()))
        model = load_model(self.dir / "r1")
        with np.load(self.dir / "r1" / "data" / "dataset.npz") as z:
            self.assertAlmostEqual(M.rel_l2(z["y_test"], model.predict(z["X_test"])), fin["test"]["rel_l2"], places=6)
            self.assertIn("X_shift_noise_x3", z.files)
        manifest = json.loads((self.dir / "r1" / "final" / "manifest.json").read_text())
        self.assertIn("reproducibility", manifest)
        self.assertLess(fin["result"]["reload_max_abs_diff"], 1e-9)
        rep = (self.dir / "r1" / "report.md").read_text(encoding="utf-8")
        for section in ("Established evidence", "Literature-supported", "Conjectures", "Failed hypotheses",
                        "Remaining uncertainty", "Idea evolution", "Lineage of the final method"):
            self.assertIn(section, rep)
        self.assertTrue((self.dir / "r1" / "research_trace.json").exists())
        store = A.FileSystemStore(self.dir / "r1", resume=True)
        kinds = {a.kind for a in store.lineage(store.latest(A.CHECKPOINT).id)}
        self.assertTrue({"checkpoint", "result", "plan", "hypothesis", "gate", "proposal", "spec"} <= kinds)
        # revised proposals keep provenance; originals remain stored
        props = [a.data for a in store.query(A.PROPOSAL)]
        revised = [p for p in props if p["action"] == "revise"]
        self.assertTrue(all(p["parent_id"] in {q["id"] for q in props} for p in revised))
        # second run: prior records are retrieved and the test split reuse is detected
        s2 = self.sched(spec, sub="r2")
        s2.run()
        prior = {i for r in s2.memory.retrievals for i in r["ids"] if s2.memory.get(i)["run_id"] == s1.run_id}
        self.assertTrue(prior, "second run did not retrieve first-run memory")
        self.assertEqual(s2.test_access_prior, 1)
        # resume: experiments are served from the cache and research is not repeated
        from mlharness.scheduler_v2 import ResearchSchedulerV2
        s3 = ResearchSchedulerV2(spec, self.dir / "r1", resume=True, verbose=False,
                                 memory=ResearchMemory(path=self.dir / "mem.jsonl"), literature_client=client())
        s3.run()
        self.assertTrue(s3.research.get("resumed"))
        store = A.FileSystemStore(self.dir / "r1", resume=True)
        self.assertGreater(len([a for a in store.query(A.RESULT) if a.data.get("cached")]), 3)

    def test_iteration_cap_and_quality_stop(self):
        spec = tiny_v2(research={"quality_threshold": 1.01, "max_rounds": 2, "pilot_tests": False},
                       budget={"compute_seconds": 1})
        s = self.sched(spec)
        s.stage_spec(); s.stage_data(); s.stage_baseline(); s.stage_research()
        self.assertIn("iteration cap", s.research["stop_reason"])
        spec = tiny_v2(research={"quality_threshold": 0.0, "max_rounds": 4, "min_rounds": 2, "pilot_tests": False})
        s = self.sched(spec, sub="q")
        s.stage_spec(); s.stage_data(); s.stage_baseline(); s.stage_research()
        self.assertIn("quality criteria", s.research["stop_reason"])
        self.assertEqual(len(s.research["rounds"]), 2)

    def test_llm_budget_halts_and_preserves_partial_work(self):
        spec = tiny_v2(research={"max_rounds": 4, "min_rounds": 4, "llm_calls": 9, "llm_reserve": 0, "pilot_tests": False},
                       budget={"llm_calls": 9})
        s = self.sched(spec, llm=ScriptedLLM())
        s.stage_spec(); s.stage_data(); s.stage_baseline(); s.stage_research()
        self.assertIn("LLM budget", s.research["stop_reason"])
        self.assertTrue(s.research["active"])
        self.assertLessEqual(s.resources.llm_by_phase.get("research", 0), 9)

    def test_scripted_llm_end_to_end(self):
        llm = ScriptedLLM()
        spec = tiny_v2(research={"max_rounds": 2, "llm_calls": 40}, execution={"refine_rounds": 0})
        s = self.sched(spec, llm=llm)
        out = s.run()
        R = s.research
        acts = [a for rd in R["rounds"] for a in rd["actions"]]
        self.assertTrue({"propose", "revise", "defend", "abandon", "branch"} <= {a["action"] for a in acts})
        self.assertEqual({a["persona"] for a in R["rounds"][1]["actions"]},
                         {"mathematical", "ml", "experimental", "exploratory"})
        self.assertTrue(any(a["reason"] == "kNN is dominated by ridge" for a in R["abandoned"]))
        self.assertTrue(all(p["origin"] == "llm" for p in R["proposals"].values()))
        # the agent-written solver used the forward model inside the sandbox and was evaluated
        cs = [i for i in s.ideas.values() if i.get("mspec", {}).get("kind") == "custom_solver"]
        self.assertTrue(cs)
        self.assertTrue(any(s.v2res.get(i["id"], {}).get("screen", {}).get("status") == "ok" for i in cs))
        self.assertTrue(any(s.v2res.get(i["id"], {}).get("screen", {}).get("forward_evals", 0) > 0 for i in cs))
        # the proposal without code got code from the engineer (LLM codegen) and passed the gate
        self.assertGreaterEqual(len(cs), 2)
        self.assertTrue(s.pilot_records)                                # critic-requested pilots ran
        self.assertTrue(any(c["portfolio"].get("disagreements") for c in R["critiques"]))
        self.assertIn("Physics inversion is established", (self.dir / "run" / "report.md").read_text(encoding="utf-8"))
        self.assertIsNotNone(out["final"]["test"]["rel_l2"])
        self.assertTrue(s.prediction_outcomes)

    def test_literature_outage_is_recorded_and_run_completes(self):
        spec = tiny_v2(research={"pilot_tests": False}, execution={"refine_rounds": 0})
        s = self.sched(spec, lit=client("network"))
        out = s.run()
        self.assertTrue(s.literature.incomplete())
        self.assertIsNotNone(out["final"])
        self.assertIn("Literature check INCOMPLETE", (self.dir / "run" / "report.md").read_text(encoding="utf-8"))

    def test_no_forward_model_blocks_physics(self):
        spec = tiny_v2(constraints={"allowed_methods": ["ridge", "knn"], "allow_forward_model": False},
                       research={"pilot_tests": False}, execution={"refine_rounds": 0})
        s = self.sched(spec)
        s.run()
        self.assertFalse(any(M2.uses_forward(i["mspec"]) for i in s.ideas.values() if i.get("mspec")))


if __name__ == "__main__":
    unittest.main()
