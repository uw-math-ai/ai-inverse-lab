"""Versioned, conservative scientific contracts and per-claim evidence checks.

These functions consume development evidence only. They never run a simulator, read a
prediction path, infer mechanism from accuracy, or upgrade an author's assertion to proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import operator

import numpy as np

from .. import metrics as M

SCHEMA_VERSION = 3
METRIC_DEFINITIONS = {
    "rel_l2": "||prediction - target||_F / ||target||_F; denominator is 1 when target norm is zero",
    "rmse": "sqrt(mean((prediction - target)^2)) over all rows and targets",
    "mae": "mean(abs(prediction - target)) over all rows and targets",
    "r2": "mean target-wise R^2 over targets with nonzero target variance",
    "accuracy": "mean(predicted label == target label) over rows",
    "f1_macro": "unweighted mean of per-class F1 scores",
    "balanced_accuracy": "unweighted mean of per-class recall",
}


def content_hash(value):
    """Content identity for JSON records; refuse nonfinite scientific metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _spec_dict(spec):
    return spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)


def build_problem_contract(spec, X_train, y_train, split_fingerprints=None):
    cfg = _spec_dict(spec)
    X, y = np.asarray(X_train), np.asarray(y_train)
    if X.ndim < 2 or y.ndim < 1 or len(X) != len(y) or len(X) == 0:
        raise ValueError("training arrays require equal, nonempty row counts and feature dimensions")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("training arrays must be finite")
    metrics = cfg.get("metrics") or {}
    primary = metrics.get("primary") or M.DEFAULT_PRIMARY[cfg.get("task", "regression")]
    names = list(dict.fromkeys([primary] + list(metrics.get("secondary") or [])))
    if any(name not in M.METRICS for name in names):
        raise ValueError("unknown metric in scientific contract")
    constraints = cfg.get("constraints") or {}
    # Simulator metadata is public problem structure. No arrays/forward evaluations are exposed.
    summary, metadata_error = {}, None
    if cfg.get("simulator"):
        try:
            from ..problem import ProblemContext
            summary = ProblemContext(cfg["simulator"], authorized=False).public_summary()
        except (OSError, ValueError, KeyError, ImportError) as exc:
            metadata_error = f"{type(exc).__name__}: problem metadata unavailable"
    fingerprints = copy.deepcopy(split_fingerprints or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "task": cfg.get("task", "regression"), "objective": cfg.get("problem", ""),
        "input_shape": list(X.shape[1:]), "target_shape": list(y.shape[1:]),
        "flattened_feature_count": int(np.prod(X.shape[1:])),
        "target_count": int(np.prod(y.shape[1:])) if y.ndim > 1 else 1,
        "training_rows": len(X),
        "splits": {name: {"role": role, "fingerprint": fingerprints.get(name)}
                   for name, role in (("train", "training"), ("val", "development"),
                                      ("test", "final_held_out"))},
        "primary_metric": primary,
        "metrics": {name: {"name": name, "direction": M.direction(name),
                            "definition": METRIC_DEFINITIONS[name],
                            "aggregation": "global_frobenius" if name == "rel_l2" else "metric_defined"}
                    for name in names},
        "transform_order": ["observation_noise_augmentation", "features", "scaler", "pca", "estimator"],
        "target_transform": "method-specific: optional log before estimator fit, exp after prediction",
        "noise": {"std": summary.get("noise_std"), "units": "raw observation units",
                  "model": summary.get("noise_model"),
                  "augmentation": "copies add independent N(0,(noise_scale*noise_std)^2) to already noisy training observations",
                  "replica_variance_if_independent": "noise_std^2 * (1 + noise_scale^2)",
                  "original_observations_retained": True},
        "privileges": {"forward_authorized": bool(constraints.get("allow_forward_model", False)),
                       "simulator_has_forward": summary.get("simulator_has_forward", False),
                       "max_forward_evals": constraints.get("max_forward_evals"),
                       "custom_code": bool(constraints.get("allow_custom_code", True))},
        "bounds": {"lower": summary.get("lower"), "upper": summary.get("upper")},
        "prior": summary.get("prior"), "facts": summary.get("facts", {}),
        "metadata_error": metadata_error,
        "diagnostic_limits": {"local_fisher": "local sensitivity under specified noise/prior assumptions; not a global identifiability or Bayes guarantee",
                              "held_out": "final test and stress observations cannot inform research decisions"},
    }


def _flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else str(key)
        out.update(_flatten(item, name))
    return out


def method_contract(requested, effective, source=None):
    requested, effective = copy.deepcopy(requested), copy.deepcopy(effective)
    rflat, eflat = _flatten(requested), _flatten(effective)
    changes = [{"field": name, "requested": rflat.get(name), "effective": eflat.get(name),
                "requested_present": name in rflat, "effective_present": name in eflat}
               for name in sorted(set(rflat) | set(eflat)) if rflat.get(name) != eflat.get(name)
               or (name in rflat) != (name in eflat)]
    code = source if source is not None else effective.get("code")
    return {"schema_version": SCHEMA_VERSION, "requested": requested, "effective": effective,
            "requested_hash": content_hash(requested), "effective_hash": content_hash(effective),
            "source_hash": hashlib.sha256(code.encode()).hexdigest() if isinstance(code, str) else None,
            "normalization_changes": changes,
            "interface": {"fit": "fit(X_train,y_train)", "predict": "predict(X)->one finite prediction per row",
                          "factory": "build_method(params,context)" if effective.get("kind") == "custom_solver"
                          else "method-kind-specific factory"},
            "diagnostics": {"declared": list(effective.get("diagnostics") or []), "observed": [], "verified": False},
            "inference_policy": effective.get("inference_policy", "repeatability requires exported-artifact certification")}


def _finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value)


def diagnostic_complete(value):
    return isinstance(value, dict) and value.get("status") in ("ok", "complete", "supported", "passed")


def _result_metric(result, name):
    if not isinstance(result, dict):
        return None
    values = result.get("val_metrics") or result.get("val") or {}
    return values.get(name) if isinstance(values, dict) else None


def _result_ok(result):
    return isinstance(result, dict) and result.get("status") in ("ok", "done", "complete")


def evaluate_claim(claim, arms, diagnostics=None):
    """Evaluate exactly one registered predicate after checking its evidence obligations.

    Operators are ``<=``, ``>=``, ``<``, ``>`` for scalar thresholds, or
    ``at_least_better``, ``at_least_worse``, ``no_more_than_worse``, ``equivalent``
    for signed relative improvement. Zero reference has no relative interpretation.
    """
    malformed_claim = not isinstance(claim, dict)
    claim = claim if isinstance(claim, dict) else {}
    diagnostics = {} if diagnostics is None else diagnostics
    verdict = {"schema_version": 3, "claim_id": claim.get("id"), "claim_version": claim.get("version", 1),
               "outcome": "inconclusive", "predicate": None, "values": {}, "effect": None,
               "interval": None, "evidence_ids": [], "unmet_requirements": [], "reason": ""}
    unmet = verdict["unmet_requirements"]
    if malformed_claim or not isinstance(arms, dict) or not isinstance(diagnostics, dict):
        verdict.update(outcome="untestable", reason="claim, arm evidence and diagnostic evidence must be mappings")
        return verdict
    for name in ("required_diagnostics", "required_evidence", "allowed_varying_factors"):
        references = claim.get(name) or []
        if not isinstance(references, list) or not all(isinstance(ref, str) for ref in references):
            verdict.update(outcome="untestable", reason=f"{name} must be a list of evidence identifiers")
            return verdict
    for name in ("treatment", "control", "treatment_arm", "control_arm", "analysis_id", "mechanism_check", "requires_enforced_gate"):
        if claim.get(name) is not None and not isinstance(claim[name], str):
            verdict.update(outcome="untestable", reason=f"{name} must be an evidence identifier")
            return verdict
    kind = claim.get("kind", "performance")
    if kind not in ("performance", "comparison", "mechanism", "mathematical"):
        verdict.update(outcome="untestable", reason="unknown claim kind")
        return verdict
    if claim.get("expect") == "worse" or claim.get("legacy_ambiguous"):
        verdict.update(outcome="untestable", reason="ambiguous legacy worse semantics require explicit re-registration")
        return verdict
    for name in claim.get("required_diagnostics") or []:
        diagnostic = diagnostics.get(name)
        if not diagnostic_complete(diagnostic):
            unmet.append(f"diagnostic:{name}")
        elif isinstance(diagnostic, dict):
            references = diagnostic.get("evidence_ids") or []
            if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
                verdict["evidence_ids"].extend(references)
            else:
                unmet.append(f"malformed diagnostic evidence identifiers:{name}")
    for name in claim.get("required_evidence") or []:
        if not diagnostic_complete(diagnostics.get(name)):
            unmet.append(f"evidence:{name}")
    if kind == "mathematical":
        # Machine proof checking is an explicit extension; arbitrary author flags are not proof.
        unmet.append("independently verified proof evidence")
        verdict.update(outcome="untestable", reason="mathematical guarantees require an independently verified proof adapter")
        return verdict
    if kind == "mechanism":
        if not claim.get("required_diagnostics"):
            unmet.append("mechanism-specific diagnostic registration")
        check = diagnostics.get(claim.get("mechanism_check", ""), {})
        if not (diagnostic_complete(check) and check.get("verified") is True and
                isinstance(check.get("predicate_passed"), bool)):
            unmet.append("verified executable mechanism predicate")
    if claim.get("requires_enforced_gate"):
        gate = diagnostics.get(claim["requires_enforced_gate"], {})
        if not (diagnostic_complete(gate) and gate.get("enforced") is True
                and isinstance(gate.get("gate_passed"), bool)
                and (gate["gate_passed"] or gate.get("fallback_executed") is True)):
            unmet.append("observed gate enforcement or predeclared fallback")
    treatment_id = claim.get("treatment") or claim.get("treatment_arm")
    treatment = arms.get(treatment_id)
    if not _result_ok(treatment):
        unmet.append(f"treatment:{treatment_id}")
    metric = claim.get("metric", "rel_l2")
    if not isinstance(metric, str) or metric not in M.METRICS:
        verdict.update(outcome="untestable", reason="unknown registered metric")
        return verdict
    margin = claim.get("margin", claim.get("threshold"))
    if not _finite(margin):
        verdict.update(outcome="untestable", reason="predicate requires a finite numeric margin/threshold")
        return verdict
    op = claim.get("operator", "<=" if M.direction(metric) == "min" else ">=")
    if not isinstance(op, str):
        verdict.update(outcome="untestable", reason="predicate operator must be a string")
        return verdict
    control_id = claim.get("control") or claim.get("control_arm")
    comparative = kind == "comparison" or control_id is not None or op in (
        "at_least_better", "at_least_worse", "no_more_than_worse", "equivalent")
    control = arms.get(control_id) if comparative else None
    if comparative:
        if control_id in ("best_family", "best_data_driven", "best_physics"):
            unmet.append("frozen explicit comparator identity")
        if not _result_ok(control):
            unmet.append(f"control:{control_id}")
        if _result_ok(treatment) and _result_ok(control):
            for field in ("row_hash", "seed", "fidelity"):
                if field not in treatment or field not in control or treatment[field] != control[field]:
                    unmet.append(f"matched:{field}")
            if treatment.get("row_hash") is None:
                unmet.append("known validation rows")
            for name in ("train_row_hash", "dataset_hash", "protocol_version"):
                if (name in treatment or name in control) and treatment.get(name) != control.get(name):
                    unmet.append(f"matched:{name}")
            allowed = claim.get("allowed_varying_factors")
            if allowed is not None:
                def effective_method(result):
                    method = result.get("effective_method")
                    contract = result.get("method_contract")
                    if method is None and isinstance(contract, dict):
                        method = contract.get("effective")
                    return method
                tm, cm = effective_method(treatment), effective_method(control)
                if tm is not None or cm is not None:
                    if not isinstance(tm, dict) or not isinstance(cm, dict):
                        unmet.append("matched effective method specifications")
                    else:
                        tf, cf = _flatten(tm), _flatten(cm)
                        for field in sorted(set(tf) | set(cf)):
                            changed = tf.get(field) != cf.get(field) or (field in tf) != (field in cf)
                            if changed and not any(field == path or field.startswith(path + ".") for path in allowed):
                                unmet.append(f"unregistered varying factor:{field}")
    for result in (treatment, control):
        if isinstance(result, dict):
            references = result.get("evidence_ids") or []
            if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
                verdict["evidence_ids"].extend(references)
            else:
                unmet.append("malformed result evidence identifiers")
            if isinstance(result.get("artifact_id"), str) and result["artifact_id"]:
                verdict["evidence_ids"].append(result["artifact_id"])
            verification = result.get("verification")
            if isinstance(verification, dict) and verification.get("passed", verification.get("verified")) is False:
                unmet.append("runtime implementation verification")
            if claim.get("requires_implementation_verification") and not (
                isinstance(verification, dict) and verification.get("passed", verification.get("verified")) is True
            ):
                unmet.append("verified implementation contract")
    if (claim.get("required_fidelity") is not None and isinstance(treatment, dict)
        and treatment.get("fidelity") != claim["required_fidelity"]):
        unmet.append("required fidelity")
    t = _result_metric(treatment, metric) if treatment else None
    c = _result_metric(control, metric) if control else None
    verdict["values"] = {"metric": metric, "treatment": float(t) if _finite(t) else None,
                         "control": float(c) if _finite(c) else None}
    if not _finite(t):
        unmet.append("finite treatment metric")
    if comparative and not _finite(c):
        unmet.append("finite control metric")
    verdict["predicate"] = {"metric": metric, "operator": op, "margin": float(margin),
                            "treatment": treatment_id, "control": control_id}
    if unmet:
        verdict["reason"] = "required evidence is missing or unmatched"
        return verdict
    if comparative:
        if c == 0:
            verdict["reason"] = "relative effect is undefined for a zero reference metric"
            verdict["unmet_requirements"].append("nonzero reference metric")
            return verdict
        effect = (c - t if M.direction(metric) == "min" else t - c) / abs(c)
        verdict["effect"] = float(effect)
        if margin < 0 and op in ("at_least_better", "at_least_worse", "no_more_than_worse", "equivalent"):
            verdict.update(outcome="untestable", reason="effect magnitude margins must be nonnegative")
            return verdict
        predicates = {"at_least_better": lambda x: x >= margin,
                      "at_least_worse": lambda x: x <= -margin,
                      "no_more_than_worse": lambda x: x >= -margin,
                      "equivalent": lambda x: abs(x) <= margin,
                      ">=": lambda x: x >= margin, "<=": lambda x: x <= margin,
                      ">": lambda x: x > margin, "<": lambda x: x < margin}
        if op not in predicates:
            verdict.update(outcome="untestable", reason="unknown comparison operator")
            return verdict
        passed = predicates[op](effect)
        if claim.get("requires_uncertainty") or claim.get("uncertainty_required"):
            name = claim.get("analysis_id", f"paired:{treatment_id}:{control_id}")
            analysis = diagnostics.get(name) or {}
            analysis = analysis if isinstance(analysis, dict) else {}
            interval = analysis.get("relative_effect_interval") or analysis.get("interval")
            # Bind uncertainty to the actual registered pair. ``treatment`` and
            # ``control`` in paired_analysis are metric values, not arm identity.
            identities = (("treatment_arm", "treatment_id", treatment_id),
                          ("control_arm", "control_id", control_id))
            matched = analysis.get("metric") == metric
            for primary, alias, expected in identities:
                matched = matched and analysis.get(primary, analysis.get(alias)) == expected
                if primary in analysis and alias in analysis:
                    matched = matched and analysis[primary] == analysis[alias]
            for field in ("row_hash", "seed", "fidelity"):
                matched = matched and field in analysis and analysis[field] == treatment.get(field)
            expected_protocol = claim.get("required_protocol_version", claim.get("protocol_version"))
            if expected_protocol is None:
                expected_protocol = treatment.get("protocol_version")
            if expected_protocol is not None:
                matched = matched and analysis.get("protocol_version") == expected_protocol
            expected_estimand = claim.get("estimand", "relative_improvement")
            if "estimand" in analysis:
                matched = matched and analysis["estimand"] == expected_estimand
            for field in ("direction", "metric_direction"):
                if field in analysis:
                    matched = matched and analysis[field] == M.direction(metric)
            if "effect_direction" in analysis:
                matched = matched and analysis["effect_direction"] == "positive_is_better"
            if (not diagnostic_complete(analysis) or not matched or not isinstance(interval, (list, tuple))
                or len(interval) != 2 or not all(_finite(x) for x in interval) or interval[0] > interval[1]):
                verdict["unmet_requirements"].append("matched paired effect interval")
                verdict["reason"] = "required paired uncertainty is unavailable or belongs to another metric, pair or protocol"
                return verdict
            verdict["interval"] = list(interval)
            ends = [predicates[op](x) for x in interval]
            if op == "equivalent":
                passed = all(ends)
                contradicted = interval[1] < -margin or interval[0] > margin
            else:
                passed, contradicted = all(ends), not any(ends)
            if not passed and not contradicted:
                verdict["reason"] = "paired interval overlaps the registered decision boundary"
                return verdict
    else:
        predicates = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}
        if op not in predicates:
            verdict.update(outcome="untestable", reason="unknown scalar predicate operator")
            return verdict
        passed = predicates[op](t, margin)
        if claim.get("requires_uncertainty") or claim.get("uncertainty_required"):
            # A paired-effect interval has different units/estimand and cannot certify
            # a scalar metric threshold. Require explicit identity and row matching.
            name = claim.get("analysis_id", f"scalar:{treatment_id}:{metric}")
            analysis = diagnostics.get(name)
            analysis = analysis if isinstance(analysis, dict) else {}
            interval = analysis.get("metric_interval")
            matched = (analysis.get("kind") == "scalar_metric"
                       and analysis.get("metric") == metric
                       and analysis.get("treatment", analysis.get("treatment_arm")) == treatment_id
                       and treatment.get("row_hash") is not None
                       and analysis.get("row_hash") == treatment["row_hash"])
            if (not diagnostic_complete(analysis) or not matched
                or not isinstance(interval, (list, tuple)) or len(interval) != 2
                or not all(_finite(x) for x in interval) or interval[0] > interval[1]):
                verdict["unmet_requirements"].append("matched scalar metric interval")
                verdict["reason"] = "required scalar uncertainty is unavailable or belongs to another estimand"
                return verdict
            verdict["interval"] = list(interval)
            ends = [predicates[op](x, margin) for x in interval]
            if not all(ends) and any(ends):
                verdict["reason"] = "scalar metric interval overlaps the registered decision boundary"
                return verdict
            passed = all(ends)
    if kind == "mechanism":
        passed = passed and diagnostics[claim["mechanism_check"]]["predicate_passed"]
    verdict.update(outcome="supported" if passed else "contradicted",
                   reason="registered predicate met" if passed else "registered predicate not met; this does not establish absence of an effect")
    verdict["evidence_ids"] = list(dict.fromkeys(verdict["evidence_ids"]))
    return verdict
