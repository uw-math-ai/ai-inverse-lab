"""Evidence-based role assessments and self-contained training records.

These are ordinal rubrics, not rewards assigned by the agents themselves.
Lean checks a fixed theorem, not the accuracy of a skeptic's diagnosis.
"""

from collections import Counter, defaultdict
from copy import deepcopy

from .evaluation import is_validated_review, proposal_digest


RUBRIC = {
    "version": "agent-iteration-v3",
    "CA_and_Repair": {
        "0": "No proof attempt supplied.",
        "1": "A prose or Lean proof was attempted; correctness is not established.",
        "2": "Independent review confirms a specific prior proof gap was repaired.",
        "3": "Lean verifies the fixed formal target, or independent review validates a complete proof.",
    },
    "SA": {
        "0": "No usable critique, or all reported gaps were independently dismissed.",
        "1": "Critique recorded; mathematical findings remain unconfirmed.",
        "2": "A packet mismatch was correctly detected, or some findings were independently confirmed.",
        "3": "Independent review validates all gap findings (or an empty-gap assessment), and packet checks are correct.",
    },
    "limits": "Scores are ordinal and role-specific. A Lean failure does not validate a skeptic gap. "
              "A Lean success does not establish which repair or agent caused it. Missing usage is unknown, not zero. "
              "Provider failures receive no mathematical score. Demo results measure software behavior only.",
}


def _skeptic_score(proposal: dict, critique: dict, benchmark: dict, review: dict | None) -> tuple[dict, dict]:
    gaps = critique.get("gaps", [])
    ids = {gap["gap_id"] for gap in gaps}
    same_target = proposal.get("claim") == benchmark["statement"]
    same_assumptions = proposal.get("assumptions") == benchmark["assumptions"]
    server_target = benchmark.get("verification_contract", {}).get("target_binding") == "repository_owned"
    dimensions = {
        "target_check_correct": critique.get("target_preserved") is same_target,
        "assumptions_check_correct": critique.get("assumptions_preserved") is same_assumptions,
        "reported_gap_count": len(gaps),
        "actionable_gap_count": sum(bool(gap.get("suggested_repair", "").strip()) for gap in gaps),
        "confirmed_gap_ids": [], "dismissed_gap_ids": [], "unreviewed_gap_ids": sorted(ids),
        "reviewed_precision": None,
    }
    score = {"score": 1, "provisional": True, "evidence_type": "unreviewed_critique",
             "rationale": "Critique recorded. Lean's result does not establish whether these diagnoses are correct."}
    if server_target:
        dimensions.update(claim_text_matches=same_target, assumptions_text_match=same_assumptions,
                          target_check_correct=True if same_target and critique.get("target_preserved") else None,
                          assumptions_check_correct=True if same_assumptions and critique.get("assumptions_preserved") else None,
                          narrative_equivalence="not_assessed")
    if not server_target and ((not same_target and critique.get("target_preserved") is False)
            or (not same_assumptions and critique.get("assumptions_preserved") is False)):
        score.update(score=2, provisional=False, evidence_type="packet_comparison",
                     rationale="The skeptic correctly flagged a literal mismatch with the fixed target or assumptions. Mathematical gap claims still require review.")
    if is_validated_review(review, proposal) and review is not None:
        if server_target and review.get("assess_proof", True):
            dimensions["target_check_correct"] = critique.get("target_preserved") is review["target_preserved"]
            dimensions["assumptions_check_correct"] = critique.get("assumptions_preserved") is review["assumptions_preserved"]
        judgment = review.get("critique_review")
        if judgment and judgment["critique_sha256"] == proposal_digest(critique):
            confirmed, dismissed = set(judgment["confirmed_gap_ids"]), set(judgment["dismissed_gap_ids"])
            dimensions.update(confirmed_gap_ids=sorted(confirmed), dismissed_gap_ids=sorted(dismissed),
                              unreviewed_gap_ids=sorted(ids - confirmed - dismissed),
                              reviewed_precision=len(confirmed) / len(confirmed | dismissed) if confirmed | dismissed else None)
            if ((ids and confirmed == ids) or judgment["no_gap_assessment_validated"]) and all(
                dimensions[name] for name in ("target_check_correct", "assumptions_check_correct")
            ):
                score.update(score=3, provisional=False, evidence_type="independent_review",
                             rationale="Independent review validates every reported gap, or explicitly validates the no-gap assessment.")
            elif confirmed:
                score.update(score=2, provisional=False, evidence_type="independent_review",
                             rationale="Independent review confirms some findings; remaining or dismissed findings are listed separately.")
            elif ids and dismissed == ids and score["evidence_type"] != "packet_comparison":
                score.update(score=0, provisional=False, evidence_type="independent_review",
                             rationale="Independent review dismisses all reported gaps.")
    return score, dimensions


def progress(current: dict, previous: dict | None) -> dict:
    before, after = previous.get("score", {}).get("score") if previous else None, current.get("score", {}).get("score")
    delta = after - before if type(after) is int and type(before) is int else None
    direction = ("baseline" if previous is None else "unassessed") if delta is None else (
        "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged")
    return {"previous_iteration": previous.get("iteration") if previous else None,
            "previous_score": before, "score_delta": delta, "direction": direction,
            "improvement_confirmed": delta is not None and delta > 0 and current["score"].get("provisional") is False}


def evaluate_actions(record: dict, benchmark: dict, calls: list[dict], previous: dict | None = None,
                     review: dict | None = None) -> list[dict]:
    """Report caller-supplied proof scores; never infer proof credit from status text.

    At runtime ``record.score`` comes directly from score_iteration with the
    live Lean receipt. Review overlays use a validated independent review.
    """
    proposal = record.get("proposal") or {}
    if review is not None and (review.get("run_id") != record["run_id"] or review.get("iteration") != record["iteration"]):
        review = None
    prior = previous.get("proposal", {}) if previous else {}
    previous_rows = previous.get("agent_evaluations", []) if previous else []
    prior_gaps = {gap["gap_id"]: gap for gap in record.get("previous_proof_gaps", [])}
    rows = []
    for call in calls:
        role = call["stage"]
        output = call.get("output")
        score = deepcopy(record.get("score", {"score": 0, "provisional": True, "rationale": "No proof attempt."}))
        dimensions = {}
        if call["status"] != "completed":
            score = {"score": None if call["status"] == "provider_error" else 0,
                     "provisional": False, "evidence_type": "call_status",
                     "rationale": "Provider failure; mathematical ability was not assessed." if call["status"] == "provider_error"
                     else "The call produced no valid structured output. Its raw response remains in the event log."}
        elif role == "VA":
            prediction = output.get("predicted_status")
            result = record.get("verification", {})
            conclusive = result.get("status") in ("verified", "failed", "rejected") and not result.get("compiler", {}).get("output_truncated")
            correct = prediction == ("verified" if result.get("status") == "verified" else "not_verified") if conclusive and prediction != "uncertain" else None
            dimensions = {"predicted_status": prediction, "checker_status": result.get("status"),
                          "prediction_correct": correct, "abstained": prediction == "uncertain",
                          "diagnosis_validated": False}
            score = {"score": None, "provisional": True, "evidence_type": "checker_prediction",
                     "rationale": "Prediction accuracy is measured against conclusive checker results; the diagnostic explanation remains unreviewed. Not comparable to the 0–3 proof rubric."}
        elif role == "SA":
            score, dimensions = _skeptic_score(proposal, output, benchmark, review)
        else:
            repairs = output.get("repairs", [])
            server_target = benchmark.get("verification_contract", {}).get("target_binding") == "repository_owned"
            dimensions = {
                "target_preserved": output.get("claim") == benchmark["statement"],
                "assumptions_preserved": output.get("assumptions") == benchmark["assumptions"],
                "prose_proof_submitted": bool(output.get("proof", "").strip()),
                "lean_proof_submitted": bool((output.get("certificate") or {}).get("proof")),
                "proof_changed": (output.get("proof"), output.get("certificate")) != (prior.get("proof"), prior.get("certificate")) if previous else None,
                "repair_links": [{**repair, "known_prior_gap": repair["gap_id"] in prior_gaps,
                                  "prior_gap": prior_gaps.get(repair["gap_id"]),
                                  "independently_confirmed": repair["gap_id"] in score.get("confirmed_repairs", [])}
                                 for repair in repairs],
            }
        if role in ("CA", "Repair") and call["status"] == "completed" and server_target:
            dimensions.update(claim_text_matches=dimensions["target_preserved"],
                              assumptions_text_match=dimensions["assumptions_preserved"],
                              target_preserved=True if dimensions["target_preserved"] else None,
                              assumptions_preserved=True if dimensions["assumptions_preserved"] else None)
        row = {
            "schema_version": 1, "rubric_version": RUBRIC["version"],
            "run_id": record["run_id"], "iteration": record["iteration"], "role": role,
            "call_id": call["call_id"], "call_status": call["status"],
            "input": deepcopy(call.get("input")), "input_sha256": call.get("input_sha256"),
            "model": call.get("model"),
            "usage": deepcopy(call.get("usage", {})), "elapsed_seconds": call.get("elapsed_seconds"),
            "output_sha256": proposal_digest(output) if output is not None else None,
            "claim": proposal.get("claim", benchmark["statement"]),
            "assumptions": proposal.get("assumptions", benchmark["assumptions"]),
            "proposed_proof": proposal.get("proof"), "lean_proof": (proposal.get("certificate") or {}).get("proof"),
            "proof_gaps": deepcopy(record.get("proof_gaps", [])), "repairs": deepcopy(proposal.get("repairs", [])),
            "previous_proposal": deepcopy(prior) if role == "Repair" else None,
            "output": deepcopy(output),
            "verification": {key: record.get("verification", {}).get(key) for key in ("engine", "status", "summary", "source_sha256")},
            "score": score, "dimensions": dimensions,
        }
        comparable = next((item for item in previous_rows if (item["role"] if item["role"] in ("SA", "VA") else "author") == (role if role in ("SA", "VA") else "author")), None)
        row["progress"] = progress(row, comparable)
        rows.append(row)
    return rows


def summarize_agents(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["role"]].append(row)
    result = {}
    for role, members in grouped.items():
        tokens = [row.get("usage", {}).get("total_tokens") for row in members]
        known = [value for value in tokens if type(value) is int]
        result[role] = {
            "actions": len(members), "failed_calls": sum(row["call_status"] != "completed" for row in members),
            "score_counts": dict(Counter(str(row["score"]["score"]) for row in members if row["score"]["score"] is not None)),
            "unscored_actions": sum(row["score"]["score"] is None for row in members),
            "provisional_actions": sum(row["score"].get("provisional", True) for row in members),
            "confirmed_improvements": sum(row["progress"]["improvement_confirmed"] for row in members),
            "regressions": sum(row["progress"]["direction"] == "regressed" for row in members),
            "reported_tokens": sum(known) if known else None, "missing_token_records": len(tokens) - len(known),
            "elapsed_seconds": sum(row.get("elapsed_seconds") or 0 for row in members),
            "checker_predictions": {
                "correct": sum(row.get("dimensions", {}).get("prediction_correct") is True for row in members),
                "incorrect": sum(row.get("dimensions", {}).get("prediction_correct") is False for row in members),
                "abstained": sum(row.get("dimensions", {}).get("abstained") is True for row in members),
            },
        }
    return result
