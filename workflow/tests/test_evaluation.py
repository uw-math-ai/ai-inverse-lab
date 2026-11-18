"""Scoring regressions: computation/model certainty must not become proof credit."""

import copy
import json
import unittest

from workflow.evaluation import aggregate_runs, proposal_digest, score_iteration, validate_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.proposal = {
            "claim": "The fixed target theorem.",
            "assumptions": ["The fixed permitted assumptions."],
            "proof": "A proposed argument, whose correctness requires review.",
            "repairs": [{"gap_id": "G1", "change": "Supply the missing induction step."}],
        }
        self.record = {
            "run_id": "run-a", "iteration": 2, "proposal": self.proposal,
            "proposal_sha256": proposal_digest(self.proposal),
            "previous_proof_gaps": [{"gap_id": "G1", "description": "Missing induction."}],
            "proof_gaps": [{"gap_id": "G2", "description": "A new gap."}],
        }
        self.review = {
            "run_id": "run-a", "iteration": 2,
            "proposal_sha256": proposal_digest(self.proposal),
            "reviewer": "Independent reviewer Alice",
            "target_preserved": True, "assumptions_preserved": True,
            "complete_proof_validated": False, "repaired_gap_ids": ["G1"],
            "evidence": "Checked the base case and induction transition in proof paragraph 2.",
        }

    def test_canonical_hash_binds_entire_proposal(self):
        reordered = dict(reversed(list(self.proposal.items())))
        self.assertEqual(proposal_digest(self.proposal), proposal_digest(reordered))
        altered = dict(self.proposal, proof="Different proof.")
        self.assertNotEqual(proposal_digest(self.proposal), proposal_digest(altered))

    def test_no_proof_scores_zero(self):
        result = score_iteration(dict(self.proposal, proof="  "), {"status": "passed"})
        self.assertEqual(result["score"], 0)
        self.assertTrue(result["provisional"])

    def test_computational_checks_and_model_certainty_stay_provisional(self):
        for verification in (
            {"status": "passed", "checks": 1000},
            {"status": "verified", "complete_proof_validated": True},
            {"status": "unsupported"},
            {"status": "claim_mismatch"},
            {"status": "failed"},
        ):
            with self.subTest(verification=verification):
                result = score_iteration(self.proposal, verification)
                self.assertEqual(result["score"], 1)
                self.assertTrue(result["provisional"])
                self.assertEqual(result["confirmed_repairs"], [])

    def test_forged_validated_marker_does_not_award_review_credit(self):
        forged = dict(self.review, _validated=True, complete_proof_validated=True)
        result = score_iteration(self.proposal, {}, forged)
        self.assertEqual(result["score"], 1)
        self.assertTrue(result["provisional"])

    def test_specific_reviewed_repair_scores_two(self):
        checked = validate_review(self.review, self.record, "run-a")
        result = score_iteration(self.proposal, {}, checked)
        self.assertEqual(result["score"], 2)
        self.assertFalse(result["provisional"])
        self.assertEqual(result["confirmed_repairs"], ["G1"])

    def test_independent_complete_proof_review_scores_three(self):
        self.review.update(complete_proof_validated=True, repaired_gap_ids=[])
        checked = validate_review(self.review, self.record, "run-a")
        result = score_iteration(self.proposal, {}, checked)
        self.assertEqual(result["score"], 3)
        self.assertFalse(result["provisional"])

    def test_review_does_not_credit_target_or_assumption_changes(self):
        for field in ("target_preserved", "assumptions_preserved"):
            review = dict(self.review, complete_proof_validated=True, **{field: False})
            checked = validate_review(review, self.record, "run-a")
            result = score_iteration(self.proposal, {}, checked)
            self.assertEqual(result["score"], 1)
            self.assertEqual(result["confirmed_repairs"], [])

    def test_review_is_bound_to_run_iteration_and_exact_proposal(self):
        for field, value in (
            ("run_id", "different-run"), ("iteration", 1), ("iteration", True),
            ("proposal_sha256", "0" * 64), ("reviewer", ""), ("evidence", " "),
            ("complete_proof_validated", "true"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_review(dict(self.review, **{field: value}), self.record, "run-a")

    def test_repairs_require_prior_gap_and_proposed_specific_change(self):
        for gap_ids in (["G2"], ["invented"], ["G1", "G1"]):
            with self.subTest(gap_ids=gap_ids):
                with self.assertRaises(ValueError):
                    validate_review(dict(self.review, repaired_gap_ids=gap_ids), self.record, "run-a")
        record = copy.deepcopy(self.record)
        record["proposal"]["repairs"] = []
        record["proposal_sha256"] = proposal_digest(record["proposal"])
        review = dict(self.review, proposal_sha256=record["proposal_sha256"])
        with self.assertRaises(ValueError):
            validate_review(review, record, "run-a")

    def test_numerical_or_symbolic_checks_are_not_proof_review(self):
        for evidence_type in ("numerical", "finite_tests", "symbolic_checks", "simulation"):
            for complete, repairs in ((True, []), (False, ["G1"])):
                with self.subTest(evidence_type=evidence_type, complete=complete):
                    with self.assertRaises(ValueError):
                        validate_review(dict(
                            self.review, evidence_type=evidence_type,
                            complete_proof_validated=complete, repaired_gap_ids=repairs,
                        ), self.record, "run-a")

    def test_loaded_review_requires_revalidation(self):
        checked = validate_review(self.review, self.record, "run-a")
        loaded = json.loads(json.dumps(checked))
        self.assertEqual(score_iteration(self.proposal, {}, loaded)["score"], 1)
        rechecked = validate_review(loaded, self.record, "run-a")
        self.assertEqual(score_iteration(self.proposal, {}, rechecked)["score"], 2)

    def test_named_gap_without_a_proposed_change_is_not_a_repair(self):
        record = copy.deepcopy(self.record)
        record["proposal"]["repairs"][0]["change"] = " "
        record["proposal_sha256"] = proposal_digest(record["proposal"])
        review = dict(self.review, proposal_sha256=record["proposal_sha256"])
        with self.assertRaises(ValueError):
            validate_review(review, record, "run-a")

    def test_finite_or_numerical_gap_cannot_receive_proof_repair_credit(self):
        for scope in ("finite_certificate", "numerical_probe", "numerical_only"):
            with self.subTest(scope=scope):
                record = copy.deepcopy(self.record)
                record["previous_proof_gaps"][0]["scope"] = scope
                with self.assertRaisesRegex(ValueError, "general-proof"):
                    validate_review(self.review, record, "run-a")

    def test_duplicate_gap_id_cannot_disguise_a_finite_certificate_gap(self):
        record = copy.deepcopy(self.record)
        record["previous_proof_gaps"].append({"gap_id": "G1", "scope": "finite_certificate"})
        with self.assertRaises(ValueError):
            validate_review(self.review, record, "run-a")

    def test_mutation_after_review_invalidates_receipt(self):
        checked = validate_review(self.review, self.record, "run-a")
        checked["complete_proof_validated"] = True
        self.assertEqual(score_iteration(self.proposal, {}, checked)["score"], 1)
        checked = validate_review(self.review, self.record, "run-a")
        changed = dict(self.proposal, proof="Another proof.")
        self.assertEqual(score_iteration(changed, {}, checked)["score"], 1)


class AggregateTests(unittest.TestCase):
    def make_run(self, mode, score=1, provisional=True, **overrides):
        run = {
            "run_id": "run-" + mode, "mode": mode, "pair_id": "pair-a",
            "status": "completed", "backend": "demo", "model": "test-model", "benchmark_sha256": "abc",
            "seed": 11, "max_model_calls": 4, "max_output_tokens": 500,
            "max_simulator_calls": 2, "model_calls": 4, "simulator_calls": 0,
            "elapsed_seconds": 1.5,
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "metadata_missing_calls": 0},
            "records": [{
                "proposal": {"repairs": []}, "proof_gaps": [],
                "score": {"score": score, "provisional": provisional, "confirmed_repairs": []},
            }],
        }
        run.update(overrides)
        return run

    def test_empty_dataset_is_explicit(self):
        result = aggregate_runs([])
        self.assertEqual(result["by_mode"], {})
        self.assertEqual(result["paired"]["matched_pairs"], 0)
        self.assertIsNone(result["paired"]["mean_final_score_difference"])

    def test_workflow_completion_is_distinct_from_verified_proof(self):
        result = aggregate_runs([self.make_run("single"), self.make_run("multi", 3, False)])
        self.assertEqual(result["by_mode"]["single"]["completed_runs"], 1)
        self.assertEqual(result["by_mode"]["single"]["independently_verified_runs"], 0)
        self.assertEqual(result["by_mode"]["multi"]["independently_verified_runs"], 1)
        self.assertEqual(result["paired"]["mean_final_score_difference"], 2)
        self.assertTrue(result["paired"]["pairs"][0]["provisional"])

    def test_failed_runs_stay_in_denominator(self):
        runs = [self.make_run("multi", 3, False), self.make_run("multi", status="failed", records=[], final_score=0)]
        mode = aggregate_runs(runs)["by_mode"]["multi"]
        self.assertEqual(mode["runs"], 2)
        self.assertEqual(mode["failed_runs"], 1)
        self.assertEqual(mode["independently_verified_rate"], 0.5)
        self.assertEqual(mode["mean_final_score"], 1.5)

    def test_unknown_usage_is_not_zero_usage(self):
        result = aggregate_runs([self.make_run("single", usage={})])
        usage = result["by_mode"]["single"]["tokens"]["total_tokens"]
        self.assertIsNone(usage["total"])
        self.assertEqual(usage["missing_runs"], 1)

    def test_pair_token_difference_is_unknown_when_metadata_is_incomplete(self):
        single = self.make_run("single")
        multi = self.make_run("multi")
        multi["usage"]["metadata_missing_calls"] = 1
        result = aggregate_runs([single, multi])
        self.assertIsNone(result["paired"]["pairs"][0]["total_tokens_difference"])

    def test_pairs_require_matched_controls(self):
        for field, value in (("seed", 12), ("backend", "openai"), ("model", "other-model"), ("max_model_calls", 8), ("benchmark_sha256", "other")):
            with self.subTest(field=field):
                result = aggregate_runs([self.make_run("single"), self.make_run("multi", **{field: value})])
                self.assertEqual(result["paired"]["matched_pairs"], 0)
                self.assertIn(field, result["paired"]["excluded_pairs"][0]["reason"])

    def test_duplicate_or_unmatched_pairs_are_not_silently_matched(self):
        result = aggregate_runs([self.make_run("single"), self.make_run("single"), self.make_run("multi")])
        self.assertEqual(result["paired"]["matched_pairs"], 0)
        result = aggregate_runs([self.make_run("single", pair_id=None), self.make_run("multi", pair_id=None)])
        self.assertEqual(result["paired"]["unpaired_runs"], 2)

    def test_confirmed_repairs_count_distinct_gaps_once_per_run(self):
        run = self.make_run("multi", 2, False)
        record = run["records"][0]
        record["proof_gaps"] = [{"gap_id": "G1"}]
        record["proposal"]["repairs"] = [{"gap_id": "G1", "change": "A repair."}]
        record["score"]["confirmed_repairs"] = ["G1"]
        run["records"].append(copy.deepcopy(record))
        result = aggregate_runs([run])["by_mode"]["multi"]["repair_metrics"]
        self.assertEqual(result["distinct_confirmed_gap_repairs"], 1)
        self.assertEqual(result["distinct_attempted_gap_repairs"], 1)
        self.assertEqual(result["confirmed_fraction_of_observed_gaps"], 1)


if __name__ == "__main__":
    unittest.main()
