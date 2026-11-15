"""Whole-study validation, completion accounting and reusable prediction diagnostics."""
from __future__ import annotations

import copy

import numpy as np

from .. import metrics as M
from .contracts_v3 import _flatten, _result_ok, diagnostic_complete, evaluate_claim


def normalize_study(study):
    st = copy.deepcopy(study)
    if not isinstance(st, dict) or not isinstance(st.get("id"), str) or not st["id"]:
        raise ValueError("study requires a nonempty stable id")
    if not isinstance(st.get("question"), str) or not st["question"].strip():
        raise ValueError("study requires a scientific question")
    arms = st.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("study requires at least one arm")
    ids = []
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("id"), str) or not arm["id"]:
            raise ValueError("each arm requires a stable id")
        if arm["id"] in ids:
            raise ValueError("duplicate arm id")
        if not isinstance(arm.get("method"), dict):
            raise ValueError("each arm requires a method spec")
        ids.append(arm["id"])
        arm.setdefault("required", True)
        if not isinstance(arm["required"], bool):
            raise ValueError("arm required flag must be boolean")
        arm.setdefault("role", "treatment")
        if arm["role"] not in ("treatment", "control", "baseline", "diagnostic", "exploratory"):
            raise ValueError("unknown arm role")
        arm.setdefault("seed", 0)
        arm.setdefault("fidelity", 1.0)
        if not isinstance(arm["fidelity"], (int, float)) or not 0 < arm["fidelity"] <= 1:
            raise ValueError("fidelity must be in (0,1]")
        arm.setdefault("intervention", {})
    if not any(a["required"] for a in arms):
        raise ValueError("study requires at least one required arm")
    claims = st.setdefault("claims", [])
    claim_ids = set()
    for claim in claims:
        if not isinstance(claim, dict) or not claim.get("id") or claim["id"] in claim_ids:
            raise ValueError("claims require unique stable ids")
        claim_ids.add(claim["id"])
        claim.setdefault("study_id", st["id"])
        if claim["study_id"] != st["id"]:
            raise ValueError("claim belongs to another study")
        for field in ("treatment", "control", "treatment_arm", "control_arm"):
            if claim.get(field) is not None and claim[field] not in ids:
                raise ValueError(f"claim references unknown {field}")
            if (claim.get(field) is not None and claim.get("required", True)
                and not next(arm for arm in arms if arm["id"] == claim[field])["required"]):
                raise ValueError("a required claim cannot depend on an optional arm")
    # Local arm dependencies support {arm:[parents]} or explicit before/after edges.
    graph = {name: [] for name in ids}
    dependencies = st.setdefault("dependencies", [])
    if isinstance(dependencies, dict):
        edges = [(parent, child) for child, parents in dependencies.items() for parent in parents]
    elif isinstance(dependencies, list):
        edges = []
        for edge in dependencies:
            if isinstance(edge, dict) and "before" in edge and "after" in edge:
                edges.append((edge["before"], edge["after"]))
            elif isinstance(edge, str):
                # A named upstream study is validated by the program-level graph.
                continue
            else:
                raise ValueError("dependencies require study ids or before/after edges")
    else:
        raise ValueError("invalid dependency graph")
    for arm in arms:
        edges.extend((parent, arm["id"]) for parent in arm.get("depends_on", []))
    for parent, child in edges:
        if parent not in graph or child not in graph:
            raise ValueError("dependency references unknown arm")
        graph[child].append(parent)
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            raise ValueError("cyclic arm dependencies")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)
    for node in graph:
        visit(node)
    st.setdefault("schema_version", 3)
    st.setdefault("version", 1)
    st.setdefault("priority", 1)
    st.setdefault("status", "planned")
    st.setdefault("owner", None)
    st.setdefault("collaborators", [])
    st.setdefault("diagnostics", [])
    st.setdefault("matching", {})
    st.setdefault("evidence_ids", [])
    st.setdefault("merge_mappings", {})
    st["required_arm_ids"] = [a["id"] for a in arms if a["required"]]
    return st


def study_completion(study, results, diagnostics=None):
    st, diagnostics = normalize_study(study), diagnostics or {}
    required = [a for a in st["arms"] if a["required"]]
    missing = [a["id"] for a in required if a["id"] not in results]
    failed = [a["id"] for a in required if a["id"] in results and not _result_ok(results[a["id"]])]
    missing_diagnostics = []
    for request in st["diagnostics"]:
        name = request if isinstance(request, str) else request.get("id")
        if isinstance(request, dict) and not request.get("required", True):
            continue
        if not diagnostic_complete(diagnostics.get(name)):
            missing_diagnostics.append(name)
    # Claim-specific diagnostics are obligations even if the top-level list omitted them.
    for claim in st["claims"]:
        if not claim.get("required", True):
            continue
        for name in claim.get("required_diagnostics", []):
            if not diagnostic_complete(diagnostics.get(name)) and name not in missing_diagnostics:
                missing_diagnostics.append(name)
    matching_issues = []
    arm_by_id = {a["id"]: a for a in st["arms"]}
    for claim in st["claims"]:
        tid = claim.get("treatment", claim.get("treatment_arm"))
        cid = claim.get("control", claim.get("control_arm"))
        if not cid or not _result_ok(results.get(tid)) or not _result_ok(results.get(cid)):
            continue
        t, c = results[tid], results[cid]
        for name in ("row_hash", "seed", "fidelity"):
            if name not in t or name not in c or t[name] is None or t[name] != c[name]:
                matching_issues.append({"claim_id": claim["id"], "field": name, "reason": "missing or unmatched evidence"})
        matching = st["matching"]
        allowed = claim.get("allowed_varying_factors", matching.get("allowed_varying_factors"))
        if allowed is not None:
            tmethod = t.get("effective_method") or arm_by_id[tid]["method"]
            cmethod = c.get("effective_method") or arm_by_id[cid]["method"]
            tf, cf = _flatten(tmethod), _flatten(cmethod)
            for field in sorted(set(tf) | set(cf)):
                if tf.get(field) != cf.get(field) and not any(field == x or field.startswith(x + ".") for x in allowed):
                    matching_issues.append({"claim_id": claim["id"], "field": field, "reason": "unregistered varying factor"})
    shared = st.get("shared_computation") or {}
    if shared:
        shared_id = shared.get("id")
        supported = shared.get("protocol") == "persisted_prediction_analysis"
        if not supported:
            matching_issues.append({"field": "shared_computation", "reason": "unsupported shared-fit/trajectory protocol"})
        elif any(not results.get(a["id"], {}).get("shared_artifact_id") == shared_id for a in required):
            matching_issues.append({"field": "shared_computation", "reason": "shared artifact identity not verified"})
    # Completion tracks required evidence acquisition, independently of whether a
    # registered hypothesis won. Reuse the verdict validator so implicit work such
    # as uncertainty, factory verification and mechanism checks cannot disappear
    # merely because it was not duplicated in the top-level diagnostic list.
    missing_claim_requirements = []
    required_claims = [claim for claim in st["claims"] if claim.get("required", True)]
    for claim in required_claims:
        verdict = evaluate_claim(claim, results, diagnostics)
        # A measured zero reference makes a relative estimand undefined, but does
        # not imply that a registered computation was omitted. Likewise a measured
        # interval overlapping a decision boundary has no unmet evidence and is a
        # completed, inconclusive result.
        requirements = [item for item in verdict["unmet_requirements"]
                        if item != "nonzero reference metric"]
        if verdict["outcome"] == "untestable" and not requirements:
            requirements.append("executable registered claim predicate")
        if requirements:
            missing_claim_requirements.append({"claim_id": claim["id"], "requirements": requirements,
                                               "reason": verdict["reason"]})
    completed = len(required) - len(missing) - len(failed)
    status = "complete" if not (missing or failed or missing_diagnostics or matching_issues or missing_claim_requirements) else "incomplete"
    if st.get("status") in ("deferred", "abandoned", "superseded") and status != "complete":
        status = st["status"]
    return {"schema_version": 3, "study_id": st["id"], "status": status,
            "reason": "all registered required work completed" if status == "complete" else "required work remains unresolved",
            "missing_arms": missing, "failed_arms": failed, "missing_diagnostics": missing_diagnostics,
            "missing_claim_requirements": missing_claim_requirements,
            "required_claim_work": len(required_claims),
            "completed_claim_work": len(required_claims) - len(missing_claim_requirements),
            "matching_issues": matching_issues, "required_work": len(required), "completed_work": completed,
            "completion_fraction": completed / len(required),
            "orphaned_arm_ids": [a["id"] for a in required if _result_ok(results.get(a["id"]))]
            if (missing or failed) and completed else []}


def paired_analysis(y, treatment, control, metric="rel_l2", n_bootstrap=200, seed=0):
    """Reuse matched prediction arrays, bootstrapping rows and recomputing the metric.

    The interval describes validation-row variation conditional on these fitted models.
    It does not measure variation across independently trained models or problem instances.
    """
    if metric not in M.METRICS:
        raise ValueError("unknown metric")
    y, t, c = np.asarray(y), np.asarray(treatment), np.asarray(control)
    if y.shape != t.shape or y.shape != c.shape or y.ndim < 1 or len(y) < 2:
        raise ValueError("paired analysis requires at least two rows with identical target/prediction shapes")
    if not all(np.isfinite(v).all() for v in (y, t, c)):
        raise ValueError("paired analysis requires finite targets and predictions")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("at least two bootstrap resamples required")
    tv, cv = M.compute(metric, y, t), M.compute(metric, y, c)
    sign = 1 if M.direction(metric) == "min" else -1
    effect = sign * (cv - tv)
    rng, effects, gains = np.random.default_rng(seed), [], []
    for _ in range(n_bootstrap):
        rows = rng.integers(0, len(y), len(y))
        bt, bc = M.compute(metric, y[rows], t[rows]), M.compute(metric, y[rows], c[rows])
        difference = sign * (bc - bt)
        if np.isfinite(difference):
            effects.append(difference)
        if bc != 0 and np.isfinite(difference) and np.isfinite(bc):
            gains.append(difference / abs(bc))
    if len(effects) != n_bootstrap:
        return {"status": "unsupported", "reason": "nonfinite metric in paired bootstrap"}
    flat_y = y.reshape(len(y), -1)
    terr, cerr = (t - y).reshape(len(y), -1), (c - y).reshape(len(y), -1)
    return {"schema_version": 3, "status": "ok", "metric": metric, "n_rows": len(y),
            "estimand": "relative_improvement", "metric_direction": M.direction(metric),
            "effect_direction": "positive_is_better",
            "treatment": float(tv), "control": float(cv), "absolute_effect": float(effect),
            "relative_effect": float(effect / abs(cv)) if cv != 0 else None,
            "absolute_effect_interval": np.quantile(effects, [.025, .975]).tolist(),
            "relative_effect_interval": np.quantile(gains, [.025, .975]).tolist() if len(gains) == n_bootstrap else None,
            "n_bootstrap": n_bootstrap, "seed": seed, "matching": "paired validation rows",
            "uncertainty_scope": "row sampling conditional on fitted models; excludes training/task seed variation",
            "per_target_rmse": {"treatment": np.sqrt(np.mean(terr ** 2, axis=0)).tolist(),
                                "control": np.sqrt(np.mean(cerr ** 2, axis=0)).tolist()},
            "per_sample_squared_error": {"treatment": np.sum(terr ** 2, axis=1).tolist(),
                                         "control": np.sum(cerr ** 2, axis=1).tolist()},
            "target_squared_norm": np.sum(flat_y ** 2, axis=1).tolist()}
