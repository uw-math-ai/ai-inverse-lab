"""Offline end-to-end regressions for orchestration, budgets, and audit records."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflow.backends import DemoBackend
from workflow.benchmarks import load_benchmark
from workflow.runtime import RunConfig, run_workflow
from workflow.schemas import Critique
from workflow.simulator import run_simulator_request


class RecordingBackend:
    """A deterministic fake backend with optional controlled provider failures."""

    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def invoke(self, stage, schema, messages):
        self.calls.append({"stage": stage, "schema": schema.__name__, "messages": deepcopy(messages)})
        response = DemoBackend().invoke(stage, schema, messages)
        response["usage"] = {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12}
        if self.transform:
            return self.transform(len(self.calls), stage, schema, response)
        return response


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="math-workflow-tests-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def run_fake(self, mode="multi", backend=None, **kwargs):
        kwargs.setdefault("stop_on_verified", True)
        return run_workflow(RunConfig(mode=mode, backend="demo", rounds=2, **kwargs),
                            self.output, backend or RecordingBackend())

    def test_multi_verifies_and_records_the_last_repair_within_four_calls(self):
        backend = RecordingBackend()
        path, summary = self.run_fake(backend=backend)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["model_calls"], 4)
        self.assertEqual([call["stage"] for call in backend.calls], ["CA", "SA", "Repair", "SA"])
        records = summary["records"]
        self.assertEqual([record["iteration"] for record in records], [1, 2])
        self.assertEqual([record["model_calls"] for record in records], [2, 4])
        self.assertTrue(records[-1]["proposal"]["repairs"])
        self.assertIsNotNone(records[-1]["critique"])
        self.assertEqual(records[0]["verification"]["status"], "failed")
        self.assertEqual(records[-1]["verification"]["status"], "verified")
        self.assertEqual(records[-1]["verification"]["target"], "formal_proof")
        self.assertEqual(summary["stop_reason"], "formal_target_verified")
        self.assertEqual(summary["simulator_calls"], 0)
        self.assertEqual((path / "proof.lean").read_text(), records[-1]["verification"]["source"])
        self.assertEqual(self.read_jsonl(path / "iterations.jsonl"), records)
        self.assertEqual(json.loads((path / "summary.json").read_text()), summary)
        events = self.read_jsonl(path / "events.jsonl")
        self.assertEqual([event["iteration"] for event in events if event["kind"] == "verification"], [1, 2])
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1]["kind"], "run_finished")
        self.assertIn("Iteration 2", (path / "report.md").read_text())
        self.assertEqual(summary["usage"]["total_tokens"], 48)
        evaluations = self.read_jsonl(path / "agent_evaluations.jsonl")
        self.assertEqual(evaluations, summary["agent_evaluations"])
        self.assertEqual([row["role"] for row in evaluations], ["CA", "SA", "Repair", "SA"])
        self.assertEqual([row["score"]["score"] for row in evaluations], [1, 1, 3, 1])
        self.assertTrue(evaluations[2]["progress"]["improvement_confirmed"])
        self.assertEqual(evaluations[2]["progress"]["score_delta"], 2)
        self.assertTrue(evaluations[3]["score"]["provisional"])
        self.assertEqual(sum(row["usage"]["total_tokens"] for row in evaluations), 48)

    def test_single_receives_equal_call_cap_and_stops_when_lean_verifies(self):
        backend = RecordingBackend()
        path, summary = self.run_fake(mode="single", backend=backend)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["max_model_calls"], 4)
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual([call["stage"] for call in backend.calls], ["CA", "Repair"])
        self.assertEqual(len(summary["records"]), 2)
        self.assertTrue(all(record["critique"] is None for record in summary["records"]))
        self.assertEqual(summary["records"][-1]["iteration"], 2)
        self.assertEqual(summary["stop_reason"], "formal_target_verified")
        self.assertEqual(len([event for event in self.read_jsonl(path / "events.jsonl")
                              if event["kind"] == "verification"]), 2)
        for call in backend.calls[1:]:
            payload = json.loads(call["messages"][-1]["content"])
            self.assertNotIn("skeptic", payload["feedback"])
            self.assertIn("verification", payload["feedback"])

    def test_bad_replies_preserve_requests_raw_outputs_and_previous_records(self):
        for failure in ("length", "max_tokens", "malformed", "parse_error", "refusal"):
            with self.subTest(failure=failure):
                def transform(call_id, stage, schema, response):
                    if call_id != 3:
                        return response
                    response["raw"]["content"] = "PARTIAL_RESPONSE_PRESERVED"
                    if failure in ("length", "max_tokens"):
                        response["raw"]["response_metadata"]["finish_reason"] = failure
                    elif failure == "malformed":
                        response["parsed"].pop("proof")
                    elif failure == "parse_error":
                        response["parsing_error"] = "Malformed structured payload"
                        response["parsed"] = None
                    else:
                        response["raw"]["additional_kwargs"] = {"refusal": "Provider refusal"}
                    return response

                path, summary = self.run_fake(backend=RecordingBackend(transform))
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["stop_reason"], "stage_error")
                self.assertEqual(summary["model_calls"], 3)
                self.assertEqual(len(summary["records"]), 1)
                self.assertTrue(summary["error"])
                self.assertEqual(self.read_jsonl(path / "iterations.jsonl"), summary["records"])
                events = self.read_jsonl(path / "events.jsonl")
                requests = [event for event in events if event["kind"] == "model_request"]
                responses = [event for event in events if event["kind"] == "model_response"]
                self.assertEqual(len(requests), 3)
                self.assertEqual(requests[-1]["stage"], "Repair")
                self.assertEqual(responses[-1]["response"]["raw"]["content"], "PARTIAL_RESPONSE_PRESERVED")
                self.assertEqual(events[-1]["status"], "failed")
                self.assertIn("Status: **failed**", (path / "report.md").read_text())

    def test_provider_exception_has_error_event_and_does_not_erase_prior_work(self):
        def transform(call_id, stage, schema, response):
            if call_id == 3:
                raise RuntimeError("Synthetic provider connection failure")
            return response

        path, summary = self.run_fake(backend=RecordingBackend(transform))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(len(summary["records"]), 1)
        self.assertEqual(summary["usage"]["metadata_missing_calls"], 1)
        events = self.read_jsonl(path / "events.jsonl")
        self.assertEqual(len([event for event in events if event["kind"] == "model_request"]), 3)
        self.assertEqual(len([event for event in events if event["kind"] == "model_response"]), 2)
        error = next(event for event in events if event["kind"] == "model_error")
        self.assertEqual(error["call_id"], 3)
        self.assertIn("Synthetic provider", error["error"])
        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(len(summary["agent_evaluations"]), 3)
        self.assertEqual(summary["agent_evaluations"][-1]["call_status"], "provider_error")
        self.assertIsNone(summary["agent_evaluations"][-1]["score"]["score"])

    def test_skeptic_failure_keeps_the_conjecturer_training_record(self):
        def transform(call_id, stage, schema, response):
            if stage == "SA":
                raise RuntimeError("Synthetic skeptic provider failure")
            return response

        path, summary = self.run_fake(backend=RecordingBackend(transform))
        self.assertEqual(summary["records"], [])
        rows = summary["agent_evaluations"]
        self.assertEqual([row["role"] for row in rows], ["CA", "SA"])
        self.assertEqual(rows[0]["score"]["score"], 1)
        self.assertTrue(rows[0]["score"]["provisional"])
        self.assertEqual(rows[0]["verification"]["status"], "not_recorded")
        self.assertIsNone(rows[1]["score"]["score"])
        self.assertEqual(self.read_jsonl(path / "agent_evaluations.jsonl"), rows)

    def test_simulator_budget_is_global_across_agents_and_rounds(self):
        request = {"operation": "first_passage", "graph": "path3", "q": 0.0,
                   "q_other": None, "T": 3, "h": 0.1, "observed": None}

        def transform(call_id, stage, schema, response):
            response["parsed"]["simulator_requests"] = [deepcopy(request), deepcopy(request)]
            return response

        with patch("workflow.runtime.run_simulator_request", wraps=run_simulator_request) as simulator:
            path, summary = self.run_fake(backend=RecordingBackend(transform), max_simulator_calls=1)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(simulator.call_count, 1)
        self.assertEqual(summary["simulator_calls"], 1)
        attempts = [event for event in self.read_jsonl(path / "events.jsonl")
                    if event["kind"] == "simulator_result"]
        self.assertEqual(len(attempts), 8)
        self.assertEqual(sum(event["result"]["status"] == "skipped" for event in attempts), 7)
        self.assertEqual(attempts[0]["result"]["status"], "completed")
        self.assertEqual({event["origin"] for event in attempts}, {"proposal", "skeptic"})
        self.assertNotIn("q_other", simulator.call_args.args[0])
        self.assertNotIn("observed", simulator.call_args.args[0])
        self.assertEqual([len(record["simulator_results"]) for record in summary["records"]], [4, 4])

    def test_unsupported_simulator_payload_cannot_execute_model_code(self):
        marker = self.output / "model_code_must_never_execute"
        expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"

        def transform(call_id, stage, schema, response):
            response["parsed"]["proof"] = expression
            response["parsed"]["simulator_requests"] = [{
                "operation": "python", "graph": "path3", "q": 0.0,
                "q_other": None, "T": 3, "h": 0.1, "observed": None, "code": expression,
            }]
            response["raw"]["content"] = expression
            return response

        with patch("workflow.runtime.run_simulator_request", wraps=run_simulator_request) as simulator:
            path, summary = self.run_fake(backend=RecordingBackend(transform))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["records"], [])
        simulator.assert_not_called()
        self.assertFalse(marker.exists())
        responses = [event for event in self.read_jsonl(path / "events.jsonl")
                     if event["kind"] == "model_response"]
        self.assertEqual(responses[0]["response"]["raw"]["content"], expression)

    def test_plain_proof_text_is_never_executed(self):
        marker = self.output / "proof_text_must_never_execute"
        expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"

        def transform(call_id, stage, schema, response):
            if schema is not Critique:
                response["parsed"]["proof"] = expression
                response["parsed"]["certificate"] = None
                response["parsed"]["simulator_requests"] = []
            return response

        _, summary = self.run_fake(backend=RecordingBackend(transform))
        self.assertEqual(summary["status"], "completed")
        self.assertFalse(marker.exists())
        self.assertTrue(all(record["verification"]["status"] == "inconclusive"
                            for record in summary["records"]))
        self.assertEqual(summary["max_score"], 1)

    def test_initial_prompts_are_proof_blind_and_later_feedback_is_explicit(self):
        backend = RecordingBackend()
        _, summary = self.run_fake(backend=backend)
        expected = load_benchmark()
        for call in backend.calls:
            payload = json.loads(call["messages"][-1]["content"])
            self.assertEqual(payload["benchmark"], expected)
            self.assertNotIn("proof", payload["benchmark"])
            self.assertFalse(payload["benchmark"]["visibility"]["agent_retrieval_tools"])
            self.assertFalse(payload["benchmark"]["visibility"]["agent_filesystem_tools"])
            self.assertEqual(payload["benchmark"]["visibility"]["published_proofs"], "withheld")
            specification = payload["benchmark"]["verification_contract"]["specification"]
            self.assertNotIn("theorem recoveryFormula", specification)
            self.assertNotIn("intro K inst", specification)
        initial = json.loads(backend.calls[0]["messages"][-1]["content"])
        self.assertEqual(set(initial), {"benchmark"})
        self.assertNotIn("controllability_rank", backend.calls[0]["messages"][-1]["content"])
        repair = json.loads(backend.calls[2]["messages"][-1]["content"])
        self.assertEqual(repair["feedback"]["verification"], summary["records"][0]["verification"])
        self.assertEqual(repair["feedback"]["skeptic"], summary["records"][0]["critique"])

    def test_demo_repairs_a_lean_proof_and_certifies_only_the_fixed_target(self):
        _, summary = self.run_fake()
        initial, repaired = summary["records"]
        self.assertEqual(initial["proposal"]["certificate"]["proof"], "rfl")
        self.assertIn("grind", repaired["proposal"]["certificate"]["proof"])
        self.assertEqual(summary["max_score"], 3)
        self.assertEqual(summary["final_score"], 3)
        self.assertFalse(initial["verification"]["independently_verified"])
        self.assertTrue(repaired["verification"]["independently_verified"])
        self.assertEqual(repaired["score"]["evidence_type"], "lean_kernel")

    def test_missing_lean_stops_before_paid_model_calls(self):
        backend = RecordingBackend()
        with patch("workflow.runtime.lean_status", return_value={"available": False, "reason": "Missing test compiler"}):
            path, summary = self.run_fake(backend=backend)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["model_calls"], 0)
        self.assertEqual(backend.calls, [])
        self.assertIn("Lean is required", summary["error"]["message"])


if __name__ == "__main__":
    unittest.main()
