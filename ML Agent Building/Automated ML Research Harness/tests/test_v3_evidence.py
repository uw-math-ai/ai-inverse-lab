"""Small scientific-accounting fixtures; no research campaign or live services."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from mlharness import metrics as M
from mlharness.research.contracts_v3 import build_problem_contract, evaluate_claim, method_contract
from mlharness.research.studies_v3 import normalize_study, paired_analysis, study_completion


def result(value=.1, **changes):
    return {"status": "ok", "val_metrics": {"rel_l2": value, "accuracy": value},
            "row_hash": "same-validation-rows", "seed": 2, "fidelity": 1.0,
            "artifact_id": "evidence-1", **changes}


def study():
    return {"id": "S1", "question": "Does alpha improve error?",
            "arms": [{"id": "t", "method": {"kind": "library", "params": {"alpha": 1}}},
                     {"id": "c", "role": "control", "method": {"kind": "library", "params": {"alpha": 2}}}],
            "claims": [{"id": "C1", "kind": "comparison", "treatment": "t", "control": "c",
                        "metric": "rel_l2", "operator": "at_least_better", "margin": .1}],
            "matching": {"allowed_varying_factors": ["params.alpha"]}}


class ContractTests(unittest.TestCase):
    def test_actual_shapes_metric_and_noise_contract(self):
        spec = {"task": "regression", "metrics": {"primary": "rel_l2", "secondary": ["rmse"]},
                "constraints": {"allow_forward_model": False}}
        contract = build_problem_contract(spec, np.zeros((7, 2, 3)), np.ones((7, 2)), {"val": "v"})
        self.assertEqual(contract["input_shape"], [2, 3])
        self.assertEqual(contract["flattened_feature_count"], 6)
        self.assertEqual(contract["target_count"], 2)
        self.assertEqual(contract["splits"]["val"], {"role": "development", "fingerprint": "v"})
        self.assertEqual(contract["metrics"]["rel_l2"]["aggregation"], "global_frobenius")
        self.assertIn("already noisy", contract["noise"]["augmentation"])
        self.assertFalse(contract["privileges"]["forward_authorized"])
        json.dumps(contract, allow_nan=False)
        with self.assertRaises(ValueError):
            build_problem_contract(spec, np.full((3, 2), np.nan), np.zeros((3, 1)))

    def test_requested_effective_normalization_and_source_identity(self):
        requested = {"kind": "library", "params": {"alpha": -2}}
        effective = {"kind": "library", "params": {"alpha": .1}, "features": "raw"}
        contract = method_contract(requested, effective, "def build_model(): pass")
        self.assertNotEqual(contract["requested_hash"], contract["effective_hash"])
        self.assertEqual({x["field"] for x in contract["normalization_changes"]}, {"params.alpha", "features"})
        self.assertEqual(len(contract["source_hash"]), 64)
        self.assertFalse(contract["diagnostics"]["verified"])
        self.assertEqual(requested["params"]["alpha"], -2)


class ClaimTests(unittest.TestCase):
    def claim(self, **changes):
        return {"id": "C", "kind": "comparison", "treatment": "t", "control": "c", "metric": "rel_l2",
                "operator": "at_least_worse", "margin": .3, **changes}

    def test_at_least_worse_has_negative_gain_and_equality(self):
        self.assertEqual(evaluate_claim(self.claim(), {"t": result(.09), "c": result(.1)})["outcome"], "contradicted")
        verdict = evaluate_claim(self.claim(), {"t": result(.13), "c": result(.1)})
        self.assertEqual(verdict["outcome"], "supported")
        self.assertAlmostEqual(verdict["effect"], -.3)
        self.assertEqual(evaluate_claim(self.claim(operator="no_more_than_worse"),
                                       {"t": result(.12), "c": result(.1)})["outcome"], "supported")

    def test_higher_metric_gain_and_equivalence(self):
        arms = {"t": result(.8), "c": result(1)}
        self.assertEqual(evaluate_claim(self.claim(metric="accuracy", margin=.1), arms)["outcome"], "supported")
        self.assertEqual(evaluate_claim(self.claim(operator="equivalent", margin=.3), arms)["outcome"], "supported")
        self.assertEqual(evaluate_claim(self.claim(operator="equivalent", margin=.1), arms)["outcome"], "contradicted")

    def test_nonfinite_zero_missing_and_unmatched_never_support(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            verdict = evaluate_claim(self.claim(), {"t": result(value), "c": result(.1)})
            self.assertEqual(verdict["outcome"], "inconclusive")
            json.dumps(verdict, allow_nan=False)
        self.assertEqual(evaluate_claim(self.claim(), {"t": result(.1), "c": result(0)})["outcome"], "inconclusive")
        self.assertEqual(evaluate_claim(self.claim(), {"t": result(.1)})["outcome"], "inconclusive")
        for field, value in (("row_hash", "other"), ("seed", 9), ("fidelity", .5)):
            verdict = evaluate_claim(self.claim(), {"t": result(.13), "c": result(.1, **{field: value})})
            self.assertIn(f"matched:{field}", verdict["unmet_requirements"])
        self.assertEqual(evaluate_claim(self.claim(margin=float("nan")), {})["outcome"], "untestable")

    def test_scalar_threshold_cannot_certify_mechanism_or_proof(self):
        claim = {"id": "threshold", "kind": "performance", "treatment": "t", "metric": "rel_l2",
                 "operator": "<=", "threshold": .2}
        arms = {"t": result(.1)}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "supported")
        self.assertEqual(evaluate_claim({**claim, "kind": "mechanism"}, arms)["outcome"], "inconclusive")
        self.assertEqual(evaluate_claim({**claim, "kind": "mathematical"}, arms)["outcome"], "untestable")
        self.assertEqual(evaluate_claim({**claim, "expect": "worse"}, arms)["outcome"], "untestable")

    def test_claim_requires_actual_implementation_verification(self):
        claim = self.claim(requires_implementation_verification=True)
        arms = {"t": result(.13), "c": result(.1)}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "inconclusive")
        for arm in arms.values():
            arm["verification"] = {"passed": True}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "supported")

    def test_missing_diagnostic_and_unenforced_gate_block(self):
        claim = self.claim(required_diagnostics=["fidelity"], requires_enforced_gate="surrogate_gate")
        arms = {"t": result(.13), "c": result(.1)}
        diagnostics = {"fidelity": {"status": "ok"}, "surrogate_gate": {"status": "ok", "gate_passed": False}}
        self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], "inconclusive")
        diagnostics["surrogate_gate"].update(enforced=True, fallback_executed=True)
        self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], "supported")
        diagnostics["fidelity"]["status"] = "unsupported"
        self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], "inconclusive")

    def test_paired_interval_must_clear_registered_boundary(self):
        claim = self.claim(operator="at_least_better", margin=.1, requires_uncertainty=True)
        arms = {"t": result(.08), "c": result(.1)}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "inconclusive")
        for interval, outcome in (([.15, .25], "supported"), ([0, .2], "inconclusive"), ([-.2, .05], "contradicted")):
            diagnostics = {"paired:t:c": {"status": "ok", "relative_effect_interval": interval,
                           "metric": "rel_l2", "treatment_arm": "t", "control_arm": "c",
                           "row_hash": arms["t"]["row_hash"], "seed": 2, "fidelity": 1.}}
            self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], outcome)

    def test_paired_uncertainty_identity_protocol_and_estimand_match(self):
        claim = self.claim(operator="at_least_better", margin=.1, requires_uncertainty=True,
                           analysis_id="registered-ci")
        arms = {"t": result(.08, protocol_version="V3.1"), "c": result(.1, protocol_version="V3.1")}
        paired = {"status": "ok", "metric": "rel_l2", "treatment_arm": "t", "control_arm": "c",
                  "row_hash": arms["t"]["row_hash"], "seed": 2, "fidelity": 1.,
                  "protocol_version": "V3.1", "estimand": "relative_improvement", "metric_direction": "min",
                  "effect_direction": "positive_is_better", "relative_effect_interval": [.15, .25],
                  "treatment": .08, "control": .1}
        self.assertEqual(evaluate_claim(claim, arms, {"registered-ci": paired})["outcome"], "supported")
        for field, value in (("metric", "accuracy"), ("treatment_arm", "other"), ("control_arm", "other"),
                             ("row_hash", "other"), ("seed", 12), ("fidelity", .5),
                             ("protocol_version", "V2"), ("estimand", "absolute_difference"),
                             ("direction", "max"), ("metric_direction", "max"),
                             ("effect_direction", "negative_is_better"), ("treatment_id", "conflicting")):
            bad = {**paired, field: value}
            self.assertEqual(evaluate_claim(claim, arms, {"registered-ci": bad})["outcome"], "inconclusive", field)
        for field in ("metric", "treatment_arm", "control_arm", "row_hash", "seed", "fidelity", "protocol_version"):
            bad = {k: v for k, v in paired.items() if k != field}
            self.assertEqual(evaluate_claim(claim, arms, {"registered-ci": bad})["outcome"], "inconclusive", field)
        aliases = {k: v for k, v in paired.items() if k not in ("treatment_arm", "control_arm")}
        aliases.update(treatment_id="t", control_id="c")
        self.assertEqual(evaluate_claim(claim, arms, {"registered-ci": aliases})["outcome"], "supported")

    def test_scalar_uncertainty_is_mandatory_and_uses_scalar_estimand(self):
        claim = {"id": "scalar", "kind": "performance", "treatment": "t", "metric": "rel_l2",
                 "operator": "<=", "threshold": .2, "requires_uncertainty": True}
        arms = {"t": result(.1)}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "inconclusive")
        base = {"status": "ok", "kind": "scalar_metric", "metric": "rel_l2", "treatment": "t",
                "row_hash": arms["t"]["row_hash"]}
        name = "scalar:t:rel_l2"
        for interval, outcome in (([.05, .15], "supported"), ([.1, .3], "inconclusive"),
                                  ([.21, .3], "contradicted"), ([.2, .2], "supported")):
            diagnostics = {name: {**base, "metric_interval": interval}}
            verdict = evaluate_claim(claim, arms, diagnostics)
            self.assertEqual(verdict["outcome"], outcome)
            self.assertEqual(verdict["interval"], interval)
        for changes in ({"kind": "paired_comparison"}, {"metric": "accuracy"},
                        {"treatment": "other-arm"}, {"row_hash": "other-rows"},
                        {"metric_interval": [.3, .1]}, {"metric_interval": [0., float("inf")]}):
            diagnostics = {name: {**base, "metric_interval": [.05, .15], **changes}}
            self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], "inconclusive")
        comparison = {"status": "ok", "relative_effect_interval": [.05, .15], "interval": [.05, .15]}
        self.assertEqual(evaluate_claim(claim, arms, {name: comparison})["outcome"], "inconclusive")

    def test_scalar_uncertainty_handles_maximization_and_strict_boundary(self):
        claim = {"id": "accuracy", "kind": "performance", "treatment": "t", "metric": "accuracy",
                 "operator": ">", "threshold": .8, "uncertainty_required": True, "analysis_id": "accuracy-ci"}
        arms = {"t": result(.9)}
        base = {"status": "ok", "kind": "scalar_metric", "metric": "accuracy", "treatment": "t",
                "row_hash": arms["t"]["row_hash"]}
        for interval, outcome in (([.85, .95], "supported"), ([.8, .9], "inconclusive"),
                                  ([.7, .8], "contradicted")):
            diagnostics = {"accuracy-ci": {**base, "metric_interval": interval}}
            self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], outcome)

    def test_malformed_claim_evidence_fails_closed(self):
        for claim, arms, diagnostics in (([], {}, {}), (self.claim(), [], {}), (self.claim(), {}, []),
                                         (self.claim(required_diagnostics="bad"), {}, {}),
                                         (self.claim(control=[]), {}, {}), (self.claim(metric=[]), {}, {}),
                                         (self.claim(operator=[]), {}, {})):
            self.assertEqual(evaluate_claim(claim, arms, diagnostics)["outcome"], "untestable")
        malformed = {"t": result(.13, val_metrics=[.13]), "c": result(.1)}
        self.assertEqual(evaluate_claim(self.claim(), malformed)["outcome"], "inconclusive")
        malformed = {"t": result(.13, evidence_ids=3), "c": result(.1)}
        self.assertEqual(evaluate_claim(self.claim(), malformed)["outcome"], "inconclusive")

    def test_effective_interventions_obey_registered_varying_paths(self):
        claim = self.claim(operator="at_least_better", margin=.1, allowed_varying_factors=["params.alpha"])
        arms = {"t": result(.08, effective_method={"features": "log", "params": {"alpha": 1}}),
                "c": result(.1, effective_method={"features": "raw", "params": {"alpha": 2}})}
        verdict = evaluate_claim(claim, arms)
        self.assertEqual(verdict["outcome"], "inconclusive")
        self.assertIn("unregistered varying factor:features", verdict["unmet_requirements"])
        arms["t"]["effective_method"]["features"] = "raw"
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "supported")
        for arm in arms.values():
            arm["method_contract"] = {"effective": arm.pop("effective_method")}
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "supported")
        arms["t"]["method_contract"]["effective"]["params"]["alpha_extra"] = 2
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "inconclusive")
        arms["c"].pop("method_contract")
        self.assertEqual(evaluate_claim(claim, arms)["outcome"], "inconclusive")


class StudyTests(unittest.TestCase):
    def test_normalization_graph_and_stable_claims(self):
        s = normalize_study(study())
        self.assertEqual(s["required_arm_ids"], ["t", "c"])
        self.assertEqual(s["claims"][0]["study_id"], "S1")
        for changes in ({"dependencies": {"t": ["c"], "c": ["t"]}},
                        {"dependencies": [{"before": "missing", "after": "t"}]}):
            with self.assertRaises(ValueError):
                normalize_study({**study(), **changes})
        s = study()
        s["claims"][0]["control"] = "absent"
        with self.assertRaises(ValueError):
            normalize_study(s)
        s = study()
        s["arms"][1]["required"] = False
        with self.assertRaises(ValueError):
            normalize_study(s)

    def test_control_failure_kept_and_negative_study_completes(self):
        s = study()
        partial = study_completion(s, {"t": result(.1), "c": result(status="crash")})
        self.assertEqual(partial["status"], "incomplete")
        self.assertEqual(partial["failed_arms"], ["c"])
        self.assertEqual(partial["orphaned_arm_ids"], ["t"])
        self.assertEqual(study_completion(s, {"t": result(.2), "c": result(.1)})["status"], "complete")

    def test_implicit_scalar_uncertainty_work_is_required_but_overlap_completes(self):
        s = {"id": "scalar-study", "question": "Is the threshold met?",
             "arms": [{"id": "t", "method": {"kind": "library", "method": "ridge"}}],
             "claims": [{"id": "threshold", "kind": "performance", "treatment": "t", "metric": "rel_l2",
                         "operator": "<=", "threshold": .2, "requires_uncertainty": True}]}
        arms = {"t": result(.1)}
        missing = study_completion(s, arms)
        self.assertEqual(missing["status"], "incomplete")
        self.assertEqual(missing["completed_work"], 1)
        self.assertEqual(missing["completed_claim_work"], 0)
        self.assertIn("matched scalar metric interval", missing["missing_claim_requirements"][0]["requirements"])
        for interval, outcome in (([.1, .3], "inconclusive"), ([.21, .3], "contradicted")):
            diagnostics = {"scalar:t:rel_l2": {"status": "ok", "kind": "scalar_metric", "metric": "rel_l2",
                           "treatment": "t", "row_hash": arms["t"]["row_hash"], "metric_interval": interval}}
            self.assertEqual(evaluate_claim(s["claims"][0], arms, diagnostics)["outcome"], outcome)
            self.assertEqual(study_completion(s, arms, diagnostics)["status"], "complete")

    def test_implicit_verification_mechanism_and_proof_work_remains_visible(self):
        for requirement in ({"requires_implementation_verification": True},
                            {"required_evidence": ["source-check"]},
                            {"kind": "mechanism"}, {"kind": "mathematical"}):
            s = study()
            s["claims"][0].update(requirement)
            arms = {"t": result(.08), "c": result(.1)}
            out = study_completion(s, arms)
            self.assertEqual(out["status"], "incomplete", requirement)
            self.assertTrue(out["missing_claim_requirements"])
        s = study()
        s["claims"][0]["requires_implementation_verification"] = True
        arms = {"t": result(.08, verification={"passed": True}),
                "c": result(.1, verification={"passed": True})}
        self.assertEqual(study_completion(s, arms)["status"], "complete")
        s["claims"].append({"id": "optional-proof", "kind": "mathematical", "required": False})
        self.assertEqual(study_completion(s, arms)["status"], "complete")

    def test_measured_zero_relative_reference_is_complete_and_inconclusive(self):
        s, arms = study(), {"t": result(.08), "c": result(0.)}
        self.assertEqual(evaluate_claim(s["claims"][0], arms)["outcome"], "inconclusive")
        self.assertEqual(study_completion(s, arms)["status"], "complete")

    def test_diagnostics_and_interventions_remain_obligations(self):
        s = study()
        s["diagnostics"] = [{"id": "trace", "required": True}]
        s["arms"][0]["method"]["features"] = "log"
        out = study_completion(s, {"t": result(.08), "c": result(.1)})
        self.assertEqual(out["missing_diagnostics"], ["trace"])
        self.assertEqual(out["matching_issues"][0]["field"], "features")
        self.assertEqual(out["status"], "incomplete")

    def test_shared_prediction_protocol_reuses_artifact_identity(self):
        s = study()
        s["shared_computation"] = {"id": "predictions-fixed", "protocol": "persisted_prediction_analysis"}
        arms = {"t": result(.08, shared_artifact_id="predictions-fixed"),
                "c": result(.1, shared_artifact_id="predictions-fixed")}
        self.assertEqual(study_completion(s, arms)["status"], "complete")
        arms["t"]["shared_artifact_id"] = "different"
        self.assertEqual(study_completion(s, arms)["status"], "incomplete")
        s["shared_computation"]["protocol"] = "arbitrary_shared_fit"
        self.assertEqual(study_completion(s, arms)["status"], "incomplete")

    def test_paired_bootstrap_recomputes_global_metric(self):
        y = np.array([[1., 2.], [20., 1.], [2., 1.], [3., 7.]])
        t, c = y + np.array([[1., 0.], [0., 1.], [1., 0.], [1., 1.]]), y + 2
        out = paired_analysis(y, t, c, n_bootstrap=80, seed=5)
        self.assertAlmostEqual(out["treatment"], M.rel_l2(y, t))
        rng, expected = np.random.default_rng(5), []
        for _ in range(80):
            rows = rng.integers(0, len(y), len(y))
            a, b = M.rel_l2(y[rows], t[rows]), M.rel_l2(y[rows], c[rows])
            expected.append((b-a)/abs(b))
        np.testing.assert_allclose(out["relative_effect_interval"], np.quantile(expected, [.025, .975]))
        self.assertEqual(len(out["per_target_rmse"]["treatment"]), 2)
        self.assertEqual(len(out["per_sample_squared_error"]["treatment"]), 4)
        json.dumps(out, allow_nan=False)
        with self.assertRaises(ValueError):
            paired_analysis(y, t[:-1], c)


if __name__ == "__main__":
    unittest.main()
