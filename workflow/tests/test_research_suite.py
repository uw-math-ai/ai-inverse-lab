"""Evidence invariants for the benchmark suite and ten-iteration workflow."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from workflow.backends import DemoBackend
from workflow.benchmarks import load_benchmark
from workflow.dashboard import DashboardStore
from workflow.evaluation import aggregate_runs
from workflow.research_suite import SUITE_BENCHMARK_IDS, frozen_problem
from workflow.runtime import RunConfig, run_workflow
from workflow.schemas import Proposal
from workflow.suite_references import check_suite, REFERENCE_PROOFS


class NoProofBackend(DemoBackend):
    def invoke(self, stage, schema, messages):
        result = super().invoke(stage, schema, messages)
        if schema is Proposal:
            result["parsed"]["certificate"] = None
            if stage == "CA":
                result["parsed"]["milestone_attempts"] = [{"benchmark_id": "bench-affine-injective", "proof": REFERENCE_PROOFS["bench-affine-injective"]}]
        return result


class ResearchSuiteTests(unittest.TestCase):
    def test_verified_first_attempt_continues_and_survives_later_failures(self):
        class RegressingBackend(DemoBackend):
            def invoke(self, stage, schema, messages):
                result = super().invoke(stage, schema, messages)
                if schema is Proposal:
                    result["parsed"]["certificate"]["proof"] = (
                        REFERENCE_PROOFS["bench-affine-injective"] if stage == "CA" else "rfl")
                return result

        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(rounds=10, benchmark_id="bench-affine-injective", verifier_model="test-adviser")
            path, summary = run_workflow(config, Path(directory), RegressingBackend())
            self.assertEqual(summary["status"], "completed", summary["error"])
            self.assertEqual(len(summary["records"]), 10)
            self.assertEqual(summary["model_calls"], 30)
            self.assertEqual(summary["stop_reason"], "iteration_limit")
            self.assertEqual(summary["trajectory"]["first_verified_iteration"], 1)
            self.assertTrue(summary["has_verified_proof"])
            self.assertEqual(summary["verified_iteration"], 1)
            self.assertEqual(summary["records"][-1]["verification"]["status"], "failed")
            self.assertEqual((path / "verified-proof.lean").read_text(), summary["records"][0]["verification"]["source"])
            self.assertNotEqual((path / "proof.lean").read_text(), (path / "verified-proof.lean").read_text())
            checkpoint = json.loads((path / "verified-checkpoint.json").read_text())
            self.assertEqual(checkpoint["iteration"], 1)
            payload = json.loads(summary["agent_evaluations"][3]["input"][-1]["content"])
            self.assertEqual(payload["verified_checkpoint"]["iteration"], 1)
            self.assertEqual(aggregate_runs([summary])["by_mode"]["multi"]["lean_verified_runs"], 1)

    def test_iteration_ceiling_is_ten_for_every_budget_policy(self):
        for mode in ("single", "multi"):
            for policy in ("calls", "iterations"):
                config = RunConfig(mode=mode, rounds=10, budget_policy=policy)
                self.assertEqual(config.max_iterations, 10)
        for rounds in (11, 20, 45):
            with self.assertRaises(ValueError):
                RunConfig(rounds=rounds)
        with self.assertRaises(ValueError):
            RunConfig(stop_on_verified="false")

    def test_reference_proofs_check_and_stay_out_of_public_packets(self):
        audit = check_suite()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["independent_human_review"], "pending")
        for identifier in SUITE_BENCHMARK_IDS:
            packet = load_benchmark(identifier)
            self.assertNotIn(REFERENCE_PROOFS[identifier], json.dumps(packet))
        self.assertEqual(frozen_problem()["status"], "frozen_reference_not_executable")
        with self.assertRaises(ValueError):
            load_benchmark(frozen_problem()["id"])

    def test_ten_iterations_preserve_verified_milestone_and_exact_role_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(rounds=10, budget_policy="iterations", benchmark_id="bench-two-measurements",
                               ca_model="author-model", sa_model="skeptic-model", verifier_model="verifier-model")
            path, summary = run_workflow(config, Path(directory), NoProofBackend())
            self.assertEqual(summary["status"], "completed", summary["error"])
            self.assertEqual(len(summary["records"]), 10)
            self.assertEqual(summary["model_calls"], 30)
            self.assertEqual(summary["stop_reason"], "iteration_limit")
            self.assertIsNone(summary["trajectory"]["first_verified_iteration"])
            self.assertEqual(summary["trajectory"]["verified_milestones"], 1)
            self.assertIsNone(summary["proof_ledger"]["best_verified_attempt"])
            self.assertEqual(summary["records"][0]["ledger"]["attempts"][-1]["iteration"], 1)
            self.assertEqual(json.loads((path / "proof-ledger.json").read_text()), summary["proof_ledger"])
            rows = summary["agent_evaluations"]
            self.assertEqual([r["model"] for r in rows[:3]], ["author-model", "skeptic-model", "verifier-model"])
            payload = json.loads(rows[3]["input"][-1]["content"])
            self.assertEqual(payload["proof_ledger"]["milestones"]["bench-affine-injective"]["status"], "formally_verified")
            va_payload = json.loads(rows[2]["input"][-1]["content"])
            self.assertNotIn("verification", va_payload)
            self.assertIsNone(rows[2]["score"]["score"])
            self.assertTrue(rows[2]["dimensions"]["abstained"])
            self.assertTrue(all(g["status"] == ("checker_observation" if g["source"] == "verification" else "reported_unconfirmed") for g in summary["proof_ledger"]["gaps"]))

    def test_iteration_policy_and_model_controls_are_part_of_pair_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            summaries = []
            for mode in ("single", "multi"):
                _, summary = run_workflow(RunConfig(mode=mode, rounds=1, budget_policy="iterations",
                                                    pair_id="pair", benchmark_id="bench-affine-injective"), Path(directory))
                summaries.append(summary)
            self.assertEqual([r["model_calls"] for r in summaries], [1, 2])
            self.assertEqual(aggregate_runs(summaries)["paired"]["matched_pairs"], 1)
            changed = deepcopy(summaries)
            changed[1]["role_models"]["SA"] = "different-model"
            self.assertEqual(aggregate_runs(changed)["paired"]["matched_pairs"], 0)

    def test_dashboard_suite_job_keeps_model_settings_and_excludes_q(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DashboardStore(Path(directory), start_worker=False)
            try:
                job = store.submit_job({"kind": "suite", "ca_model": "author", "sa_model": "skeptic", "verifier_model": "judge"})
                config = job["config"]
                self.assertEqual(config["rounds"], 10)
                self.assertEqual(config["max_output_tokens"], 16000)
                command = store._command(store.jobs[job["id"]])
                self.assertIn("suite", command)
                self.assertIn("--verifier-model", command)
                workspace = store.workspace()
                self.assertEqual(len(workspace["suite"]["benchmark_ids"]), 5)
                self.assertNotIn(workspace["q_problem"]["id"], workspace["suite"]["benchmark_ids"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
