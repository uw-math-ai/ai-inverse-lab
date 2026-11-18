"""Persistent evidence ledger: claims remain distinct from checked outcomes."""
from copy import deepcopy
from .records import digest


def describe_verification(result: dict) -> dict:
    """Explain the recorded checking stage without changing the original verdict."""
    status = result.get("status")
    compiler = result.get("compiler")
    if status == "rejected" and not compiler:
        return {"stage": "submission_check", "label": "Submission blocked before Lean",
                "explanation": "The submitted certificate did not meet the proof interface. Lean did not run. This is not a rejection of the mathematical claim by the verification agent."}
    if status == "rejected":
        return {"stage": "axiom_audit", "label": "Proof audit did not pass",
                "explanation": "The compiler ran, but the required trusted-axiom audit did not pass."}
    if status == "verified":
        return {"stage": "lean_check", "label": "Formal proof verified",
                "explanation": "Lean checked the exact target and the axiom audit passed. The accompanying prose still requires mathematical review."}
    if status == "failed" and compiler and not compiler.get("output_truncated"):
        return {"stage": "lean_check", "label": "Lean could not complete this proof",
                "explanation": "Lean found a syntax, type, or unfinished-goal error in this attempt. This does not refute the theorem."}
    if status == "inconclusive" and not compiler:
        return {"stage": "missing_certificate", "label": "No checkable proof submitted",
                "explanation": "There is no accepted formal proof body to check. Prose alone is not certified by Lean."}
    return {"stage": "inconclusive", "label": "No conclusive proof check",
            "explanation": result.get("summary") or "The checker has not established a result."}


def advance_ledger(previous: dict | None, record: dict, benchmark: dict) -> dict:
    ledger = deepcopy(previous) if previous else {
        "version": 1, "target_id": benchmark["id"], "benchmark_sha256": digest(benchmark),
        "attempts": [], "milestones": {}, "gaps": [], "best_verified_attempt": None,
    }
    iteration = record["iteration"]
    verified = record["score"].get("evidence_type") == "lean_kernel" and record["score"].get("score") == 3
    ledger["attempts"].append({"iteration": iteration, "proposal_sha256": record["proposal_sha256"],
                               "formal_status": record["verification"]["status"],
                               "verification_display": describe_verification(record["verification"]),
                               "proof_status": "formally_verified" if verified else "unverified",
                               "claimed_repairs": deepcopy(record["proposal"].get("repairs", []))})
    if verified:
        ledger["best_verified_attempt"] = iteration
    for item in record.get("milestone_checks", []):
        if item["status"] == "verified":
            ledger["milestones"].setdefault(item["benchmark_id"], {
                "status": "formally_verified", "first_iteration": iteration,
                "source_sha256": item.get("source_sha256"), "proof": item["proof"],
                "dependencies": [], "relevance": "repository-listed supporting milestone; does not prove the main target",
            })
    for gap in record.get("proof_gaps", []):
        key = (gap.get("source"), gap["gap_id"], gap.get("description"))
        existing = next((g for g in ledger["gaps"] if (g["source"], g["gap_id"], g["description"]) == key), None)
        if existing is None:
            ledger["gaps"].append({"source": gap.get("source"), "gap_id": gap["gap_id"],
                                   "description": gap.get("description"), "first_iteration": iteration,
                                   "status": "checker_observation" if gap.get("source") == "verification" else "reported_unconfirmed",
                                   "repair_claim_iterations": []})
    for repair in record["proposal"].get("repairs", []):
        for gap in ledger["gaps"]:
            if gap["gap_id"] == repair["gap_id"] and gap["first_iteration"] < iteration:
                gap["repair_claim_iterations"].append(iteration)
    ledger["note"] = "A verified target does not validate individual critique diagnoses or attribute success to an agent. Original claims remain immutable; independent reviews are separate."
    return ledger


def trajectory_metrics(records: list[dict]) -> dict:
    accepted = [r for r in records if r.get("score", {}).get("evidence_type") == "lean_kernel" and r["score"].get("score") == 3]
    return {"first_verified_iteration": accepted[0]["iteration"] if accepted else None,
            "time_to_first_verified_seconds": accepted[0]["elapsed_seconds"] if accepted else None,
            "calls_to_first_verified": accepted[0]["model_calls"] if accepted else None,
            "verified_milestones": len(records[-1].get("ledger", {}).get("milestones", {})) if records else 0,
            "repeated_proposals": sum(r["proposal_sha256"] in {p["proposal_sha256"] for p in records[:i]} for i, r in enumerate(records)),
            "confirmed_repairs": len({gap for r in records for gap in r.get("score", {}).get("confirmed_repairs", [])}),
            "attempts_reaching_lean": sum(bool(r.get("verification", {}).get("compiler")) for r in records),
            "repeated_certificates": sum(bool(r.get("progress_evidence", {}).get("same_certificate_as_previous")) for r in records),
            "history": [{"iteration": r["iteration"], "formal_status": r["verification"]["status"],
                         "verified_milestones": len(r.get("ledger", {}).get("milestones", {})),
                         "model_calls": r["model_calls"], "elapsed_seconds": r["elapsed_seconds"],
                         "evidence": r.get("progress_evidence")} for r in records]}


def iteration_evidence(record: dict, previous: dict | None) -> dict:
    """Measured transitions, not a fabricated distance to a proof."""
    result = record["verification"]
    certificate = record["proposal"].get("certificate")
    before = (previous or {}).get("proposal", {}).get("certificate")
    milestones = set(record.get("ledger", {}).get("milestones", {}))
    prior_milestones = set((previous or {}).get("ledger", {}).get("milestones", {}))
    proved = record["score"].get("score") == 3 and record["score"].get("evidence_type") == "lean_kernel"
    return {"stage": describe_verification(result)["label"], "reached_lean": bool(result.get("compiler")),
            "first_compiler_attempt_after_block": bool(previous and result.get("compiler") and not previous["verification"].get("compiler")),
            "same_certificate_as_previous": bool(certificate and previous and certificate == before),
            "new_verified_milestones": sorted(milestones - prior_milestones),
            "target_verified": proved,
            "note": "Reaching Lean is an interface improvement. Only verified milestones or the full target establish mathematical progress; changed wording and fewer diagnostics do not."}
