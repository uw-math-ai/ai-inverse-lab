"""Offline CLI regression tests; synthetic attestations exist only in temp files."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from workflow.response_call import main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="math-workflow-cli-tests-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def cli(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main([str(argument) for argument in arguments])
        return code, output.getvalue(), errors.getvalue()

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_demo(self):
        code, _, errors = self.cli(
            "run", "--backend", "demo", "--rounds", "2",
            "--max-simulator-calls", "0", "--output", self.output,
        )
        self.assertEqual(code, 0, errors)
        return next(self.output.glob("*/summary.json")).parent

    def review_template(self, run):
        code, output, errors = self.cli("review-template", run, "--reviewer", "Synthetic test reviewer")
        self.assertEqual(code, 0, errors)
        path = Path(output.strip())
        return path, self.read_json(path)

    def test_independent_review_sidecar_preserves_original_audit_files(self):
        run = self.run_demo()
        originals = {name: (run / name).read_bytes() for name in (
            "summary.json", "manifest.json", "events.jsonl", "iterations.jsonl", "report.md",
        )}
        path, review = self.review_template(run)
        self.assertEqual(review["evidence"], "")
        self.assertFalse(review["complete_proof_validated"])
        # This is a synthetic attestation to test plumbing, not an assessment of
        # the demo proof. The entire temporary run is deleted after this test.
        review.update(
            target_preserved=True, assumptions_preserved=True,
            complete_proof_validated=True,
            evidence="SYNTHETIC UNIT-TEST ATTESTATION ONLY: exercise the reviewed-result path.",
        )
        self.write_json(path, review)
        code, output, errors = self.cli("review", run, path)
        self.assertEqual(code, 0, errors)
        sidecar = self.read_json(Path(output.strip()))
        self.assertEqual(sidecar["review"]["proposal_sha256"], review["proposal_sha256"])
        self.assertEqual(sidecar["score"]["score"], 3)
        self.assertFalse(sidecar["score"]["provisional"])
        self.assertEqual(self.read_json(run / "summary_adjudicated.json")["final_score"], 3)
        for name, original in originals.items():
            self.assertEqual((run / name).read_bytes(), original, name)
        raw_result, reviewed_result = self.output / "raw.json", self.output / "reviewed.json"
        self.assertEqual(self.cli("aggregate", run, "--output", raw_result)[0], 0)
        self.assertEqual(self.cli("aggregate", run, "--reviewed", "--output", reviewed_result)[0], 0)
        self.assertEqual(self.read_json(raw_result)["by_mode"]["multi"]["independently_verified_runs"], 1)
        self.assertEqual(self.read_json(reviewed_result)["by_mode"]["multi"]["independently_verified_runs"], 1)

    def test_invalid_reviews_create_no_sidecar_and_do_not_change_summary(self):
        run = self.run_demo()
        original = (run / "summary.json").read_bytes()
        path, template = self.review_template(run)
        valid_fields = dict(
            target_preserved=True, assumptions_preserved=True,
            evidence="SYNTHETIC TEST EVIDENCE ONLY.",
        )
        bad_reviews = [
            template,
            dict(template, **valid_fields, proposal_sha256="wrong"),
            dict(template, **valid_fields, repaired_gap_ids=["nonexistent"]),
            dict(template, **valid_fields, complete_proof_validated=True, evidence_type="numerical"),
            dict(template, **valid_fields, repaired_gap_ids=["short_horizon"]),
        ]
        for review in bad_reviews:
            with self.subTest(review=review):
                self.write_json(path, review)
                code, _, errors = self.cli("review", run, path)
                self.assertEqual(code, 2)
                self.assertIn("Error:", errors)
                self.assertFalse((run / "reviews").exists())
                self.assertFalse((run / "summary_adjudicated.json").exists())
                self.assertEqual((run / "summary.json").read_bytes(), original)

    def test_reviewed_aggregation_revalidates_provenance_in_saved_sidecar(self):
        run = self.run_demo()
        path, review = self.review_template(run)
        review.update(evidence="SYNTHETIC TEST EVIDENCE ONLY.")
        self.write_json(path, review)
        code, output, errors = self.cli("review", run, path)
        self.assertEqual(code, 0, errors)
        sidecar_path = Path(output.strip())
        sidecar = self.read_json(sidecar_path)
        sidecar["review"]["proposal_sha256"] = "tampered"
        self.write_json(sidecar_path, sidecar)
        aggregate_path = self.output / "invalid_aggregate.json"
        code, _, errors = self.cli("aggregate", run, "--reviewed", "--output", aggregate_path)
        self.assertEqual(code, 2)
        self.assertIn("proposal_sha256", errors)
        self.assertFalse(aggregate_path.exists())

    def test_saved_score_is_recomputed_from_review_instead_of_trusted(self):
        run = self.run_demo()
        path, review = self.review_template(run)
        review.update(evidence="SYNTHETIC TEST EVIDENCE ONLY.")
        self.write_json(path, review)
        code, output, errors = self.cli("review", run, path)
        self.assertEqual(code, 0, errors)
        sidecar_path = Path(output.strip())
        sidecar = self.read_json(sidecar_path)
        sidecar["score"].update(score=3, provisional=False)
        self.write_json(sidecar_path, sidecar)
        aggregate_path = self.output / "recomputed.json"
        self.assertEqual(self.cli("aggregate", run, "--reviewed", "--output", aggregate_path)[0], 0)
        mode = self.read_json(aggregate_path)["by_mode"]["multi"]
        self.assertEqual(mode["mean_final_score"], 1)
        self.assertEqual(mode["independently_verified_runs"], 0)

    def test_compare_alternates_modes_and_records_matching_trial_controls(self):
        code, _, errors = self.cli(
            "compare", "--backend", "demo", "--rounds", "1", "--trials", "2",
            "--seed", "22", "--max-simulator-calls", "0", "--output", self.output,
        )
        self.assertEqual(code, 0, errors)
        comparison_path = next(self.output.glob("comparison-*/comparison.json"))
        comparison = self.read_json(comparison_path)
        self.assertEqual(comparison["paired"]["matched_pairs"], 2)
        self.assertEqual(comparison["paired"]["excluded_pairs"], [])
        summaries = [self.read_json(Path(path) / "summary.json") for path in comparison["run_directories"]]
        self.assertEqual([run["mode"] for run in summaries], ["single", "multi", "multi", "single"])
        self.assertEqual([run["seed"] for run in summaries], [22, 22, 23, 23])
        self.assertTrue(all(run["model_calls"] == run["max_model_calls"] == 2 for run in summaries))
        self.assertTrue((comparison_path.parent / "comparison.md").exists())
        for name, mode in comparison["by_mode"].items():
            self.assertEqual(mode["runs"], 2)
            self.assertEqual(mode["provisional_runs"], 2 if name == "multi" else 0)
            self.assertEqual(mode["independently_verified_runs"], 0 if name == "multi" else 2)

    def test_invalid_comparison_trial_count_fails_without_creating_runs(self):
        code, _, errors = self.cli("compare", "--trials", "0", "--output", self.output)
        self.assertEqual(code, 2)
        self.assertIn("trials", errors)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_training_export_keeps_originals_and_applies_bound_critique_review(self):
        run = self.run_demo()
        original = (run / "agent_evaluations.jsonl").read_bytes()
        path = self.output / "review.json"
        record = self.read_json(run / "summary.json")["records"][0]
        review = {"run_id": record["run_id"], "iteration": 1, "proposal_sha256": record["proposal_sha256"],
                  "reviewer": "Synthetic reviewer", "evidence": "Synthetic test of exact critique review export.",
                  "target_preserved": True, "assumptions_preserved": True,
                  "complete_proof_validated": False, "repaired_gap_ids": [],
                  "critique_review": {"critique_sha256": record["critique_sha256"],
                                      "confirmed_gap_ids": ["field_algebra"], "dismissed_gap_ids": [],
                                      "no_gap_assessment_validated": False}}
        self.write_json(path, review)
        self.assertEqual(self.cli("review", run, path)[0], 0)
        automatic, reviewed = self.output / "automatic.jsonl", self.output / "reviewed.jsonl"
        self.assertEqual(self.cli("export-training", run, "--output", automatic)[0], 0)
        self.assertEqual(self.cli("export-training", run, "--reviewed", "--output", reviewed)[0], 0)
        before = [json.loads(line) for line in automatic.read_text().splitlines()]
        after = [json.loads(line) for line in reviewed.read_text().splitlines()]
        self.assertEqual(len(after), 4)
        self.assertEqual(before[1]["score"]["score"], 1)
        self.assertEqual(after[1]["score"]["score"], 3)
        self.assertEqual(after[3]["progress"]["previous_score"], 3)
        self.assertEqual(after[2]["score"]["score"], 3)  # Preserve the recorded Lean outcome.
        self.assertEqual((run / "agent_evaluations.jsonl").read_bytes(), original)
        self.assertEqual(self.cli("export-training", run, "--output", run / "summary.json")[0], 2)

        # A skeptic-only review must not downgrade an existing Lean proof.
        second = self.read_json(run / "summary.json")["records"][1]
        review.update(iteration=2, proposal_sha256=second["proposal_sha256"], assess_proof=False,
                      target_preserved=False, assumptions_preserved=False,
                      critique_review={"critique_sha256": second["critique_sha256"], "confirmed_gap_ids": [],
                                       "dismissed_gap_ids": [], "no_gap_assessment_validated": True})
        self.write_json(path, review)
        self.assertEqual(self.cli("review", run, path)[0], 0)
        adjudicated = self.read_json(run / "summary_adjudicated.json")
        self.assertEqual(adjudicated["final_score"], 3)
        self.assertEqual(adjudicated["agent_evaluations"][3]["score"]["score"], 3)
        self.assertEqual(adjudicated["agent_evaluations"][1]["score"]["score"], 3)

    def test_historical_training_scores_are_not_invented(self):
        run = self.output / "old-run"
        run.mkdir()
        self.write_json(run / "summary.json", {"records": []})
        destination = self.output / "old.jsonl"
        code, _, error = self.cli("export-training", run, "--output", destination)
        self.assertEqual(code, 2)
        self.assertIn("predates", error)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
