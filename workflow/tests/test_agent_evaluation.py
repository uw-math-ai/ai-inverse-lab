"""Role-specific evidence, review binding, progress, and missing-data checks."""

from copy import deepcopy
import unittest

from workflow.agent_evaluation import evaluate_actions, progress, summarize_agents
from workflow.evaluation import aggregate_runs, proposal_digest, score_iteration, validate_review


class AgentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.packet = {"statement": "Fixed theorem", "assumptions": ["Known hypothesis"]}
        self.proposal = {"claim": self.packet["statement"], "assumptions": self.packet["assumptions"],
                         "proof": "An attempted proof", "repairs": [], "certificate": None}
        self.critique = {"assessment": "The induction step is missing", "target_preserved": True,
                         "assumptions_preserved": True, "gaps": [{"gap_id": "G1", "description": "Missing induction",
                         "scope": "general_proof", "suggested_repair": "Prove the induction step"}]}
        self.record = {"run_id": "test-run", "iteration": 1, "proposal": self.proposal,
                       "critique": self.critique, "proof_gaps": self.critique["gaps"], "previous_proof_gaps": [],
                       "verification": {"status": "failed"}, "score": score_iteration(self.proposal, {})}
        self.calls = [{"call_id": 1, "stage": "CA", "status": "completed", "output": self.proposal,
                       "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}, "elapsed_seconds": 0.1},
                      {"call_id": 2, "stage": "SA", "status": "completed", "output": self.critique,
                       "usage": {}, "elapsed_seconds": 0.2}]

    def review(self, **judgments):
        review = {"run_id": "test-run", "iteration": 1, "proposal_sha256": proposal_digest(self.proposal),
                  "reviewer": "Synthetic test reviewer", "evidence": "Synthetic test of review binding only",
                  "target_preserved": True, "assumptions_preserved": True, "complete_proof_validated": False,
                  "repaired_gap_ids": [], "critique_review": {
                      "critique_sha256": proposal_digest(self.critique), "confirmed_gap_ids": [],
                      "dismissed_gap_ids": [], "no_gap_assessment_validated": False, **judgments}}
        return review

    def test_failed_or_successful_lean_status_does_not_confirm_the_skeptic(self):
        for status in ("failed", "verified", "unavailable"):
            self.record["verification"]["status"] = status
            rows = evaluate_actions(self.record, self.packet, self.calls)
            self.assertEqual([row["score"]["score"] for row in rows], [1, 1])
            self.assertTrue(rows[1]["score"]["provisional"])
            self.assertEqual(rows[1]["dimensions"]["unreviewed_gap_ids"], ["G1"])
            self.assertEqual(rows[0]["claim"], "Fixed theorem")
            self.assertEqual(rows[1]["proposed_proof"], "An attempted proof")

    def test_independent_critique_review_is_bound_to_both_outputs(self):
        raw = self.review(confirmed_gap_ids=["G1"])
        checked = validate_review(raw, self.record, "test-run")
        rows = evaluate_actions(self.record, self.packet, self.calls, review=checked)
        self.assertEqual(rows[1]["score"]["score"], 3)
        self.assertEqual(rows[1]["dimensions"]["reviewed_precision"], 1)
        for forged in (raw, dict(checked, _validated=True)):
            self.assertEqual(evaluate_actions(self.record, self.packet, self.calls, review=forged)[1]["score"]["score"], 1)
        checked["critique_review"]["confirmed_gap_ids"] = []
        self.assertEqual(evaluate_actions(self.record, self.packet, self.calls, review=checked)[1]["score"]["score"], 1)
        raw["critique_review"]["critique_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate_review(raw, self.record, "test-run")

    def test_invalid_judgments_are_rejected_and_dismissal_can_reduce_score(self):
        for changes in ({"confirmed_gap_ids": ["unknown"]}, {"confirmed_gap_ids": ["G1", "G1"]},
                        {"confirmed_gap_ids": ["G1"], "dismissed_gap_ids": ["G1"]},
                        {"no_gap_assessment_validated": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_review(self.review(**changes), self.record, "test-run")
        checked = validate_review(self.review(dismissed_gap_ids=["G1"]), self.record, "test-run")
        row = evaluate_actions(self.record, self.packet, self.calls, review=checked)[1]
        self.assertEqual(row["score"]["score"], 0)
        self.assertEqual(row["dimensions"]["reviewed_precision"], 0)

    def test_mechanical_mismatch_is_separate_from_mathematical_correctness(self):
        self.proposal["claim"] = "A different target"
        self.critique["target_preserved"] = False
        row = evaluate_actions(self.record, self.packet, self.calls)[1]
        self.assertEqual(row["score"]["score"], 2)
        self.assertEqual(row["score"]["evidence_type"], "packet_comparison")
        self.assertEqual(row["dimensions"]["confirmed_gap_ids"], [])

    def test_new_contract_does_not_reward_skeptic_for_literal_paraphrase_difference(self):
        self.packet["verification_contract"] = {"target_binding": "repository_owned"}
        self.proposal["claim"] = "Reworded theorem"
        self.critique["target_preserved"] = False
        rows = evaluate_actions(self.record, self.packet, self.calls)
        self.assertEqual(rows[1]["score"]["score"], 1)
        self.assertTrue(rows[1]["score"]["provisional"])
        self.assertIsNone(rows[1]["dimensions"]["target_check_correct"])
        self.assertIsNone(rows[0]["dimensions"]["target_preserved"])

    def test_repairs_reference_prior_gaps_and_do_not_self_certify(self):
        prior = deepcopy(self.record)
        prior["agent_evaluations"] = evaluate_actions(prior, self.packet, self.calls)
        self.record["iteration"] = 2
        self.record["previous_proof_gaps"] = deepcopy(prior["proof_gaps"])
        self.proposal["repairs"] = [{"gap_id": "G1", "change": "Claimed fix"}, {"gap_id": "invented", "change": "Another fix"}]
        calls = [dict(self.calls[0], stage="Repair", call_id=3)]
        row = evaluate_actions(self.record, self.packet, calls, prior)[0]
        self.assertEqual([item["known_prior_gap"] for item in row["dimensions"]["repair_links"]], [True, False])
        self.assertFalse(any(item["independently_confirmed"] for item in row["dimensions"]["repair_links"]))
        self.assertEqual(row["progress"]["direction"], "unchanged")
        self.assertEqual(row["previous_proposal"], prior["proposal"])

    def test_progress_requires_evidence_and_can_regress(self):
        previous = {"iteration": 1, "score": {"score": 1}}
        for score, provisional, direction, confirmed in ((3, False, "improved", True), (2, True, "improved", False), (0, False, "regressed", False)):
            result = progress({"score": {"score": score, "provisional": provisional}}, previous)
            self.assertEqual(result["direction"], direction)
            self.assertEqual(result["improvement_confirmed"], confirmed)

    def test_provider_failures_are_unscored_and_unknown_tokens_are_not_zero(self):
        calls = [dict(self.calls[0], status="provider_error", output=None, usage={})]
        rows = evaluate_actions(self.record, self.packet, calls)
        self.assertIsNone(rows[0]["score"]["score"])
        metrics = summarize_agents(rows)["CA"]
        self.assertEqual(metrics["failed_calls"], 1)
        self.assertIsNone(metrics["reported_tokens"])
        self.assertEqual(metrics["missing_token_records"], 1)

    def test_aggregate_keeps_demo_live_modes_and_models_separate(self):
        rows = evaluate_actions(self.record, self.packet, self.calls)
        result = aggregate_runs([{"mode": mode, "backend": backend, "model": model, "agent_evaluations": rows}
                                 for mode, backend, model in (("multi", "demo", "a"), ("multi", "openai", "a"), ("single", "openai", "b"))])
        self.assertEqual(len(result["by_agent"]), 3)


if __name__ == "__main__":
    unittest.main()
