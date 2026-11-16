"""Live-model interface hardening found by the first V3 ablation (2026-11-13)."""
import json
import unittest

from mlharness.llm import LLMResponse
from mlharness.research.program_v3 import RoleRouter, compact_for_model


class CompactionTests(unittest.TestCase):
    def test_long_numeric_lists_are_summarized_short_ones_kept(self):
        ctx = {"evidence": {"r1": {"per_sample_l2": [float(i) for i in range(750)], "per_target": [0.1, 0.2],
                                   "labels": ["a"] * 100, "flag": True}}}
        out = compact_for_model(ctx)
        s = out["evidence"]["r1"]["per_sample_l2"]
        self.assertTrue(s["elided_numeric_list"])
        self.assertEqual((s["n"], s["min"], s["max"], s["p50"]), (750, 0.0, 749.0, 374.0))
        self.assertEqual(out["evidence"]["r1"]["per_target"], [0.1, 0.2])
        self.assertEqual(out["evidence"]["r1"]["labels"], ["a"] * 100)      # non-numeric lists kept
        self.assertEqual(len(ctx["evidence"]["r1"]["per_sample_l2"]), 750)  # input not mutated

    def test_router_sends_compacted_prompt_and_records_full_usage(self):
        seen = {}

        class Backend:
            name, model, is_null = "fake", "fake-model", False

            def complete(self, system, prompt, timeout=None):
                seen["prompt"] = prompt
                return LLMResponse('{"type": "stop"}', 0.1, "fake", "fake-model", 0.01, 5, 7,
                                   cache_creation_tokens=11, cache_read_tokens=13, thinking_tokens=2)

        router = RoleRouter(Backend())
        value = router.ask("orchestrator", {"x": list(range(1000))}, actor="o", purpose="t")
        self.assertEqual(value, {"type": "stop"})
        self.assertLess(len(seen["prompt"]), 400)
        self.assertTrue(json.loads(seen["prompt"])["x"]["elided_numeric_list"])
        ev = router.events[-1]
        self.assertEqual((ev["input_tokens"], ev["output_tokens"], ev["cache_creation_tokens"],
                          ev["cache_read_tokens"], ev["thinking_tokens"]), (5, 7, 11, 13, 2))


class UsageLimitTests(unittest.TestCase):
    MSG = "claude error: You've hit your 5-hour limit · resets 2:10am (America/Los_Angeles)"

    def test_reset_time_parsing(self):
        import datetime as dt
        from mlharness.llm import usage_limit_wait
        now = dt.datetime(2026, 11, 13, 22, 0).timestamp()
        self.assertAlmostEqual(usage_limit_wait(self.MSG, now), 4 * 3600 + 600, delta=1)
        self.assertAlmostEqual(usage_limit_wait("resets 9pm", dt.datetime(2026, 11, 13, 20, 0).timestamp()), 3600,
                               delta=1)
        self.assertEqual(usage_limit_wait("no time here"), 1800.0)

    def test_wait_and_retry_only_when_enabled(self):
        import os
        from unittest import mock

        from mlharness.llm import ClaudeCodeLLM, LLMError
        llm = ClaudeCodeLLM("claude", workdir=None)
        calls = {"n": 0}

        def once(system, prompt, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError(self.MSG)
            return LLMResponse("{}", 1.0, "claude_code", "m")

        with mock.patch.object(llm, "_complete_once", side_effect=once), \
                mock.patch("mlharness.llm.time.sleep") as sleep, \
                mock.patch("mlharness.llm.usage_limit_wait", return_value=100.0):
            with mock.patch.dict(os.environ, {"MLH_WAIT_ON_USAGE_LIMIT": "1"}):
                resp = llm.complete("s", "p")
            self.assertEqual(resp.waited_seconds, 220.0)
            sleep.assert_called_once_with(220.0)
            calls["n"] = 0
            with mock.patch.dict(os.environ, {"MLH_WAIT_ON_USAGE_LIMIT": "0"}):
                with self.assertRaises(LLMError):
                    llm.complete("s", "p")

    def test_router_pauses_wall_clock_for_waits(self):
        import types

        class Backend:
            name, model, is_null = "fake", "m", False

            def complete(self, system, prompt, timeout=None):
                r = LLMResponse('{"type": "stop"}', 1000.0 + 2.0, "fake", "m")
                r.waited_seconds = 1000.0
                return r

        res = types.SimpleNamespace(start=5000.0, charged=[], try_llm_call=lambda: True)
        res.charge_llm = lambda s, c: res.charged.append(s)
        router = RoleRouter(Backend(), resources=res)
        router.ask("orchestrator", {}, actor="o", purpose="t")
        self.assertEqual(res.start, 6000.0)
        self.assertEqual(res.charged, [2.0])
        self.assertEqual(router.events[-1]["usage_limit_wait_seconds"], 1000.0)


class ExportabilityTests(unittest.TestCase):
    NESTED = ("import numpy as np\n"
              "def build_method(params, context):\n"
              "    class BoxLMVec:\n"
              "        def fit(self, X, y):\n"
              "            return self\n"
              "        def predict(self, X):\n"
              "            return np.zeros((len(X), 2))\n"
              "    return BoxLMVec()\n")
    TOP = ("import numpy as np\n"
           "class BoxLMVec:\n"
           "    def fit(self, X, y):\n"
           "        return self\n"
           "    def predict(self, X):\n"
           "        return np.zeros((len(X), 2))\n"
           "def build_method(params, context):\n"
           "    return BoxLMVec()\n")

    def test_gate_rejects_class_nested_in_contract_function(self):
        from mlharness.safety import check_code_v2
        problems = check_code_v2(self.NESTED, "build_method")
        self.assertTrue(any("module level" in p and "BoxLMVec" in p for p in problems), problems)
        self.assertEqual(check_code_v2(self.TOP, "build_method"), [])

    def test_v3_dev_eval_fails_visibly_on_unpicklable_model(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        import numpy as np

        from mlharness import runner

        class Unpicklable:
            def __init__(self):
                self.f = lambda x: x

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros((len(X), 2))

        rng = np.random.default_rng(0)
        data = {"X_train": rng.normal(size=(20, 3)), "y_train": rng.normal(size=(20, 2)),
                "X_val": rng.normal(size=(8, 3)), "y_val": rng.normal(size=(8, 2))}
        job = {"method": {"kind": "library", "method": "ridge"}, "exp_id": "e-1", "task": "regression",
               "fraction": 1.0, "primary": "rel_l2", "pipeline": "v3", "simulator": {"name": "friedman"}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(runner, "_make_context", return_value=None), \
                mock.patch("mlharness.methods_v2.build", return_value=Unpicklable()):
            with self.assertRaisesRegex(TypeError, "not exportable"):
                runner.run_method_eval(job, dict(data), Path(d))


if __name__ == "__main__":
    unittest.main()
