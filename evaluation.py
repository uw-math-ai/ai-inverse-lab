"""Conservative proof scoring and descriptive, budget-matched run comparisons.

Finite experiments and model statements are evidence to inspect, not proofs.
Score 3 can also come from a fresh Lean receipt for the exact fixed target.
The human reviewer remains responsible for the mathematical assessment; this
module validates its provenance and structure, not the truth of review prose.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

from .verification import is_verified_receipt


class _ValidatedReview(dict):
    """In-process review receipt; a JSON ``_validated`` flag is not trusted."""

    def __init__(self, review: dict) -> None:
        super().__init__(json.loads(json.dumps(review, allow_nan=False)))
        self.receipt_digest = proposal_digest(self)


def proposal_digest(proposal: dict) -> str:
    """Hash all proposal fields with canonical JSON (including proof and repairs)."""
    encoded = json.dumps(
        proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gap_ids(gaps: list) -> set[str]:
    return {
        gap["gap_id"] for gap in gaps
        if isinstance(gap, dict) and isinstance(gap.get("gap_id"), str)
        and gap["gap_id"].strip()
    }


def validate_review(review: dict, record: dict, run_id: str) -> dict:
    """Bind a manual review to a run, iteration, and exact proposal.

    ``record.previous_proof_gaps`` must be populated by the caller from earlier
    records in this run. A repaired gap must both exist there and be explicitly
    addressed in the current proposal's ``repairs``. Evidence defaults to
    ``human_mathematical_review``; numerical/symbolic experiments may be recorded
    as evidence but cannot certify repairs or a complete proof. Correcting an
    auxiliary finite rank calculation is not a repair of the general proof: the
    reviewer must explain which mathematical implication or proof gap is fixed.

    Raises ValueError for malformed, mismatched, or unsupported attestations.
    The returned dict is JSON serializable, but must be revalidated after loading.
    """
    if not isinstance(review, dict) or not isinstance(record, dict):
        raise ValueError("Review and record must be JSON objects.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A nonempty run_id is required.")
    if review.get("run_id") != run_id or record.get("run_id") != run_id:
        raise ValueError("Review run_id does not match the recorded run.")
    iteration = review.get("iteration")
    if type(iteration) is not int or iteration < 1 or iteration != record.get("iteration"):
        raise ValueError("Review iteration must match a positive recorded iteration.")
    proposal = record.get("proposal")
    if not isinstance(proposal, dict):
        raise ValueError("Record has no proposal to review.")
    digest = proposal_digest(proposal)
    if review.get("proposal_sha256") != digest:
        raise ValueError("Review proposal_sha256 does not match the proposal.")
    if record.get("proposal_sha256", digest) != digest:
        raise ValueError("Recorded proposal_sha256 does not match its proposal.")
    for name in ("reviewer", "evidence"):
        if not isinstance(review.get(name), str) or not review[name].strip():
            raise ValueError(f"Review {name} must be a nonempty string.")
    for name in ("target_preserved", "assumptions_preserved", "complete_proof_validated"):
        if type(review.get(name)) is not bool:
            raise ValueError(f"Review {name} must be a boolean.")
    repairs = review.get("repaired_gap_ids")
    if not isinstance(repairs, list) or any(
        not isinstance(gap_id, str) or not gap_id.strip() for gap_id in repairs
    ) or len(set(repairs)) != len(repairs):
        raise ValueError("repaired_gap_ids must be a list of distinct nonempty IDs.")
    previous_gaps = record.get("previous_proof_gaps", [])
    prior_gaps = _gap_ids(previous_gaps)
    experimental_gaps = _gap_ids([
        gap for gap in previous_gaps if isinstance(gap, dict)
        and gap.get("scope", "general_proof") != "general_proof"
    ])
    prior_gaps -= experimental_gaps
    proposed_repairs = _gap_ids([
        repair for repair in proposal.get("repairs", [])
        if isinstance(repair, dict) and isinstance(repair.get("change"), str)
        and repair["change"].strip()
    ])
    if set(repairs) - (prior_gaps & proposed_repairs):
        raise ValueError("Every repaired gap must occur in earlier general-proof gaps and proposed repairs; finite certificate or numerical gaps do not qualify.")
    evidence_type = review.get("evidence_type", "human_mathematical_review")
    proof_evidence = {"human_mathematical_review", "formal_proof"}
    experimental_evidence = {"numerical", "finite_tests", "symbolic_checks", "simulation"}
    if not isinstance(evidence_type, str) or evidence_type not in proof_evidence | experimental_evidence:
        raise ValueError("Unsupported evidence_type; use a documented review evidence category.")
    if evidence_type not in proof_evidence and (review["complete_proof_validated"] or repairs):
        raise ValueError("Experimental checks cannot certify a proof or a repaired proof gap.")
    normalized = dict(review)
    normalized["evidence_type"] = evidence_type
    assess_proof = review.get("assess_proof", True)
    if type(assess_proof) is not bool:
        raise ValueError("assess_proof must be a boolean.")
    if not assess_proof and (review["complete_proof_validated"] or repairs):
        raise ValueError("Select proof assessment before certifying a proof or repair.")
    normalized["assess_proof"] = assess_proof
    critique_review = review.get("critique_review")
    if critique_review is not None:
        critique = record.get("critique")
        if not isinstance(critique_review, dict) or not isinstance(critique, dict):
            raise ValueError("A critique review requires a recorded critique.")
        if critique_review.get("critique_sha256") != proposal_digest(critique):
            raise ValueError("Critique review hash does not match the exact skeptic output.")
        known = _gap_ids(critique.get("gaps", []))
        if len(known) != len(critique.get("gaps", [])):
            raise ValueError("Duplicate skeptic gap IDs are ambiguous and cannot be independently adjudicated.")
        judged = []
        for name in ("confirmed_gap_ids", "dismissed_gap_ids"):
            values = critique_review.get(name)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"Critique {name} must be a list of gap IDs.")
            if len(set(values)) != len(values) or set(values) - known:
                raise ValueError("Critique judgments must reference distinct recorded skeptic gap IDs.")
            judged.append(set(values))
        if judged[0] & judged[1]:
            raise ValueError("A skeptic gap cannot be both confirmed and dismissed.")
        no_gaps = critique_review.get("no_gap_assessment_validated")
        if type(no_gaps) is not bool or (no_gaps and known):
            raise ValueError("A validated no-gap assessment requires an empty skeptic gap list.")
        if evidence_type not in proof_evidence:
            raise ValueError("Critique judgments require independent mathematical review.")
    return _ValidatedReview(normalized)


def is_validated_review(review: dict | None, proposal: dict) -> bool:
    return (isinstance(review, _ValidatedReview)
            and review.receipt_digest == proposal_digest(review)
            and review.get("proposal_sha256") == proposal_digest(proposal))


def score_iteration(proposal: dict, verification: dict, review: dict | None = None) -> dict:
    """Assign 0/1, reviewed 2/3, or 3 from a fresh Lean verification receipt.

    Verification results remain available in the iteration record. Neither a
    passing numerical test nor a serialized status label promotes a claim.
    A Lean receipt certifies only the fixed formal statement. Reviews must first
    pass ``validate_review``; model-generated review
    dictionaries, including ones with an invented approval flag, are ignored.
    """
    proof = proposal.get("proof", "")
    has_proof = isinstance(proof, str) and bool(proof.strip())
    if isinstance(proof, list):
        has_proof = any(isinstance(step, str) and bool(step.strip()) for step in proof)
    has_proof = has_proof or bool((proposal.get("certificate") or {}).get("proof", "").strip())
    reviewed = is_validated_review(review, proposal) and review is not None and review.get("assess_proof", True)
    result = {
        "score": int(has_proof),
        "provisional": not reviewed,
        "rationale": "Proof attempt recorded; its correctness is not independently established."
        if has_proof else "No proof attempt was supplied.",
        "confirmed_repairs": [],
    }
    if not reviewed:
        if is_verified_receipt(proposal, verification):
            result.update(score=3, provisional=False, evidence_type="lean_kernel",
                          rationale="Lean checked the exact fixed formal target and audited its axioms. "
                                    "This score applies only to that formal statement, not prose or the broader inverse problem.")
            return result
        if review is not None:
            result["rationale"] += " Review was not validated for this exact proposal."
        return result
    assert review is not None
    if not has_proof:
        return result
    if not review["target_preserved"] or not review["assumptions_preserved"]:
        result["rationale"] = "Independent review found the target or permitted assumptions were not preserved."
        return result
    if review["complete_proof_validated"]:
        result.update(
            score=3,
            evidence_type="independent_review",
            rationale="An independent review attests a complete proof of the unchanged target under the permitted assumptions.",
            confirmed_repairs=list(review["repaired_gap_ids"]),
        )
    elif review["repaired_gap_ids"]:
        result.update(
            score=2,
            evidence_type="independent_review",
            rationale="An independent review confirms specific prior proof gaps were repaired while preserving the target and assumptions.",
            confirmed_repairs=list(review["repaired_gap_ids"]),
        )
    else:
        result["rationale"] = "Independent review does not establish a complete proof or confirm a specific repaired gap."
    return result


def _number(value: Any) -> float | int | None:
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value
    return None


def _scores(run: dict) -> list[dict]:
    return [record["score"] for record in run.get("records", []) if isinstance(record.get("score"), dict)]


def _final_score(run: dict) -> int:
    scores = _scores(run)
    value = scores[-1].get("score", 0) if scores else run.get("final_score", 0)
    return value if type(value) is int and 0 <= value <= 3 else 0


def _verified(run: dict) -> bool:
    scores = _scores(run)
    return bool(scores and scores[-1].get("score") == 3 and scores[-1].get("provisional") is False)


def _quantities(runs: list[dict], key: str, *, usage: bool = False) -> dict:
    values = [_number((run.get("usage") or {}).get(key) if usage else run.get(key)) for run in runs]
    observed = [value for value in values if value is not None]
    return {
        "total": sum(observed) if observed else None,
        "mean": sum(observed) / len(observed) if observed else None,
        "observed_runs": len(observed),
        "missing_runs": len(runs) - len(observed),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Describe runs by mode and matched ``multi - single`` trial differences.

    A pair must have one run per mode and equal benchmark, backend, seed and
    configured call/output/simulator caps. Failed runs stay in the denominator.
    Observed token totals are not equivalent to a fixed monetary budget. Scores
    are ordinal descriptive outcomes; no significance or causal claim is made.
    """
    from .agent_evaluation import summarize_agents

    agent_groups = defaultdict(list)
    for run in runs:
        group = (str(run.get("mode", "unknown")), str(run.get("backend", "unknown")), str(run.get("model", "unknown")))
        agent_groups[group].extend(run.get("agent_evaluations", []))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("mode", "unknown"))].append(run)
    by_mode = {}
    for mode, members in sorted(grouped.items()):
        final_scores = [_final_score(run) for run in members]
        provisional = [not _scores(run) or _scores(run)[-1].get("provisional", True) for run in members]
        attempted_repairs = confirmed_repairs = observed_gaps = 0
        runs_with_repairs = 0
        for run in members:
            records = run.get("records", [])
            attempted_repairs += len(set().union(*[
                _gap_ids(record.get("proposal", {}).get("repairs", [])) for record in records
            ]))
            gaps = set().union(*[_gap_ids(record.get("proof_gaps", [])) for record in records])
            observed_gaps += len(gaps)
            confirmed = set().union(*[
                set(score.get("confirmed_repairs", [])) for score in _scores(run)
                if score.get("provisional") is False and score.get("score", 0) >= 2
            ])
            confirmed_repairs += len(confirmed)
            runs_with_repairs += bool(confirmed)
        by_mode[mode] = {
            "runs": len(members),
            "completed_runs": sum(run.get("status") in {"completed", "complete"} for run in members),
            "failed_runs": sum(run.get("status") in {"failed", "error"} for run in members),
            "independently_verified_runs": sum(_verified(run) for run in members),
            "lean_verified_runs": sum(bool(_scores(run) and _scores(run)[-1].get("evidence_type") == "lean_kernel" and _verified(run)) for run in members),
            "independently_verified_rate": sum(_verified(run) for run in members) / len(members),
            "provisional_runs": sum(provisional),
            "mean_final_score": sum(final_scores) / len(members),
            "final_score_counts": dict(sorted(Counter(map(str, final_scores)).items())),
            "provisional_score_counts": dict(sorted(Counter(str(score) for score, flag in zip(final_scores, provisional) if flag).items())),
            "reviewed_score_counts": dict(sorted(Counter(str(score) for score, flag in zip(final_scores, provisional) if not flag).items())),
            "model_calls": _quantities(members, "model_calls"),
            "simulator_calls": _quantities(members, "simulator_calls"),
            "elapsed_seconds": _quantities(members, "elapsed_seconds"),
            "tokens": {key: _quantities(members, key, usage=True) for key in ("input_tokens", "output_tokens", "total_tokens")},
            "metadata_missing_calls": _quantities(members, "metadata_missing_calls", usage=True),
            "repair_metrics": {
                "distinct_observed_gaps": observed_gaps,
                "distinct_attempted_gap_repairs": attempted_repairs,
                "distinct_confirmed_gap_repairs": confirmed_repairs,
                "runs_with_confirmed_repairs": runs_with_repairs,
                "confirmed_fraction_of_observed_gaps": confirmed_repairs / observed_gaps if observed_gaps else None,
            },
        }
    pairs: dict[str, list[dict]] = defaultdict(list)
    unpaired_runs = 0
    for run in runs:
        if isinstance(run.get("pair_id"), str) and run["pair_id"]:
            pairs[run["pair_id"]].append(run)
        else:
            unpaired_runs += 1
    matched, excluded = [], []
    matching_keys = (
        "benchmark_sha256", "backend", "model", "seed", "max_model_calls",
        "max_output_tokens", "max_simulator_calls",
    )
    for pair_id, members in sorted(pairs.items()):
        modes = Counter(run.get("mode") for run in members)
        if modes != {"single": 1, "multi": 1}:
            excluded.append({"pair_id": pair_id, "reason": "Requires exactly one single run and one multi run."})
            continue
        single = next(run for run in members if run["mode"] == "single")
        multi = next(run for run in members if run["mode"] == "multi")
        mismatches = [key for key in matching_keys if key not in single or key not in multi or single[key] != multi[key]]
        if mismatches:
            excluded.append({"pair_id": pair_id, "reason": "Missing or mismatched controls: " + ", ".join(mismatches)})
            continue
        item = {
            "pair_id": pair_id, "single_run_id": single.get("run_id"),
            "multi_run_id": multi.get("run_id"),
            "final_score_difference": _final_score(multi) - _final_score(single),
            "verified_difference": int(_verified(multi)) - int(_verified(single)),
            "provisional": any(not _scores(run) or _scores(run)[-1].get("provisional", True) for run in members),
        }
        for key in ("model_calls", "elapsed_seconds"):
            left, right = _number(single.get(key)), _number(multi.get(key))
            item[key + "_difference"] = right - left if left is not None and right is not None else None
        usages = [run.get("usage") or {} for run in (single, multi)]
        left_tokens, right_tokens = [_number(usage.get("total_tokens")) for usage in usages]
        item["total_tokens_difference"] = (
            right_tokens - left_tokens
            if left_tokens is not None and right_tokens is not None
            and all(usage.get("metadata_missing_calls") == 0 for usage in usages)
            else None
        )
        matched.append(item)
    return {
        "schema_version": 1,
        "by_mode": by_mode,
        "by_agent": [{"mode": mode, "backend": backend, "model": model, "roles": summarize_agents(rows)}
                     for (mode, backend, model), rows in sorted(agent_groups.items()) if rows],
        "paired": {
            "direction": "multi minus single",
            "matched_pairs": len(matched), "pairs": matched,
            "excluded_pairs": excluded, "unpaired_runs": unpaired_runs,
            "mean_final_score_difference": sum(pair["final_score_difference"] for pair in matched) / len(matched) if matched else None,
            "mean_verified_difference": sum(pair["verified_difference"] for pair in matched) / len(matched) if matched else None,
        },
        "interpretation": "Scores are ordinal; automated 0/1 outcomes remain provisional. Workflow completion and independent proof verification are separate. Matched pairs share configured budgets, not actual token cost. These are descriptive results, not a significance test.",
    }
