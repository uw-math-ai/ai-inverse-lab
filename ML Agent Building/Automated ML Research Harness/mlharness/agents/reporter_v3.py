"""Factual V3 reporting: the persisted evidence bundle is the sole input."""
from __future__ import annotations

import json
from pathlib import Path

from ..knowledge_v3 import evaluate_program


def write_report_v3(run_dir, bundle=None):
    run_dir = Path(run_dir)
    if bundle is None:
        bundle = json.loads((run_dir / "evidence_bundle.json").read_text(encoding="utf-8"))
    studies = bundle.get("studies", [])
    final = bundle.get("final") or {}
    cert = final.get("certificate") or (final.get("result") or {}).get("export_certificate") or {}
    metrics = evaluate_program(bundle)
    (run_dir / "program_evaluation.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    lines = [f"# V3 research product: {bundle.get('run_id')}", "", str(bundle.get("objective", "")), "",
             f"Program stop: {bundle.get('program', {}).get('stop_reason')}", "",
             f"Model export certified: **{bool(cert.get('certified', False))}**. "
             f"Studies complete: **{sum(s.get('status') == 'complete' for s in studies)}/{len(studies)}**.", "",
             "Model performance and study completion are separate outcomes.", ""]
    if bundle.get("error"):
        lines += [f"Run error: {bundle['error']}", ""]
    lines += ["## Actual exported method", ""]
    for info in (final.get("result") or {}).get("member_info", []):
        lines += [f"- {info.get('label')} (kind `{info.get('kind')}`, method hash `{info.get('method_hash')}`, "
                  f"source hash `{info.get('source_hash')}`, forward access `{info.get('uses_forward')}`).",
                  f"  Observed fit diagnostics: `{json.dumps(info.get('fit_diagnostics', {}), sort_keys=True)}`"]
    if not final:
        lines += ["No final model was exported."]
    lines += ["", f"Validation selection value: `{final.get('val')}`. Final metrics: `{json.dumps(final.get('test', {}), sort_keys=True)}`.",
              "", f"Export certificate: `{json.dumps(cert, sort_keys=True)}`.", "",
              f"Lineage: `{json.dumps(final.get('lineage', {}), sort_keys=True)}`.", "",
              "## Registered studies and exact claim outcomes", "",
              "| Study | Required work | State | Reason |", "|---|---:|---|---|"]
    for study in studies:
        lines.append(f"| {study['id']} | {study.get('completed_work', 0)}/{study.get('required_work', len(study['arms']))} "
                     f"| {study.get('status')} | {study.get('defer_reason', study.get('reason', ''))} |")
    for study in studies:
        lines += ["", f"### {study['id']}: {study['question']}", "",
                  f"Owner: `{study.get('owner')}`; registered evidence: `{study.get('evidence_ids', [])}`.", ""]
        for arm in study["arms"]:
            result = bundle.get("arm_results", {}).get(arm["id"], {})
            contract = arm.get("method_contract", {})
            lines += [f"- Arm `{arm['id']}` ({arm.get('role')}, required={arm.get('required', True)}): "
                      f"`{result.get('status', 'missing')}`; method `{result.get('label', arm['method'].get('kind'))}`; "
                      f"metrics `{json.dumps(result.get('val_metrics', result.get('val', {})), sort_keys=True)}`; "
                      f"effective hash `{contract.get('effective_hash')}`.",
                      f"  Normalization changes: `{json.dumps(contract.get('normalization_changes', []), sort_keys=True)}`."]
        lines += ["", "Diagnostics and absences:", ""]
        for request in study.get("diagnostics", []):
            rid = request["id"] if isinstance(request, dict) else request
            observed = study.get("diagnostic_results", {}).get(rid, {})
            lines.append(f"- `{rid}`: `{observed.get('status', 'missing')}`; artifact `{observed.get('artifact_id')}`; "
                         f"reason `{observed.get('reason', 'registered computation returned')}`.")
        for verdict in study.get("verdicts", []):
            lines += ["", f"Claim `{verdict['claim_id']}`: **{verdict['outcome']}**. {verdict['reason']}.",
                      f"Predicate: `{json.dumps(verdict.get('predicate'), sort_keys=True)}`; "
                      f"values `{json.dumps(verdict.get('values'), sort_keys=True)}`; effect `{verdict.get('effect')}`; "
                      f"interval `{verdict.get('interval')}`.",
                      f"Evidence `{verdict.get('evidence_ids', [])}`; unmet requirements `{verdict.get('unmet_requirements', [])}`."]
    lines += ["", "## Independent critic and uncertainty", ""]
    for objection in bundle.get("program", {}).get("objections", []):
        lines.append(f"- `{objection['id']}` ({objection['scope']}, {objection['status']}): {objection['requirement']}. "
                     f"Resolution: `{objection.get('resolution', 'unresolved')}`.")
    result = final.get("result") or {}
    for key in ("test_bootstrap", "test_per_target_rel_l2", "test_uncertainty", "shift"):
        lines += ["", f"{key}: `{json.dumps(result.get(key), sort_keys=True)}`."]
    lines += ["", "## Scientific contract", "", "```json", json.dumps(bundle.get("contract", {}), indent=2, sort_keys=True), "```",
              "", "## Delegation, fallback attribution and costs", "",
              f"Agent instances: `{len(bundle.get('agents', {}))}`; specialist tasks: `{len(bundle.get('specialist_tasks', []))}`.", ""]
    counts = {}
    for event in bundle.get("role_events", []):
        reason = event.get("fallback_reason")
        if reason:
            counts[reason] = counts.get(reason, 0)+1
    lines += [f"Fallback reasons (role calls): `{json.dumps(counts, sort_keys=True)}`.", "",
              f"Costs: `{json.dumps(bundle.get('costs'), sort_keys=True)}`.", "",
              f"Reservation policy: `{json.dumps(bundle.get('reservations'), sort_keys=True)}`.", "",
              "## Limits", ""]
    lines.extend(f"- {limit}" for limit in bundle.get("limits", []))
    lines += ["", "Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. "
              "Evaluation readiness metrics: `program_evaluation.json`.", ""]
    report = "\n".join(lines)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    if final.get("model_path"):
        card = ["# V3 exported model", "", f"Selected methods: {', '.join(final.get('labels', []))}.", "",
                f"Export certified: **{bool(cert.get('certified', False))}**.", "",
                f"Scientific studies complete: {sum(s.get('status') == 'complete' for s in studies)}/{len(studies)}. "
                "Passing model accuracy does not certify unfinished scientific claims.", "",
                "```json", json.dumps(cert, indent=2, sort_keys=True), "```", "",
                f"Validation: `{final.get('val')}`. Final metrics: `{json.dumps(final.get('test', {}), sort_keys=True)}`.", "",
                "See ../report.md and ../evidence_bundle.json for exact claims, applicability, controls, "
                "diagnostics, unresolved objections, provenance and costs.", ""]
        (Path(final["model_path"]).parent / "model_card.md").write_text("\n".join(card), encoding="utf-8")
    return report
