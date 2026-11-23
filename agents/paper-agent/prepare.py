"""Extract manuscript evidence from a V3 run without calling a model."""
import argparse
import json
from pathlib import Path


def prepare(bundle, spec, evaluation):
    if bundle.get("pipeline") != "v3":
        raise ValueError("Expected a V3 evidence bundle")

    def fact(value, pointer, file="evidence_bundle.json"):
        return {"value": value, "source": {"file": file, "json_pointer": pointer}}

    final = bundle.get("final") or {}
    result = final.get("result") or {}
    objections = (bundle.get("program") or {}).get("objections") or []
    studies = []
    for index, study in enumerate(bundle["studies"]):
        path = f"/studies/{index}"
        arms = []
        for arm in study["arms"]:
            arm_id = arm["id"]
            measured = bundle["arm_results"].get(arm_id)
            if measured is None:
                continue
            pointer = "/arm_results/" + arm_id.replace("~", "~0").replace("/", "~1")
            fields = ("label", "status", "val", "seed", "fidelity", "forward_evals", "artifact_id")
            arms.append(fact({key: measured[key] for key in fields if key in measured}, pointer))
        verdicts = study.get("verdicts") or []
        cited_objections = {
            requirement for verdict in verdicts
            for requirement in (verdict.get("unmet_requirements") or [])
        }
        studies.append({
            "id": study["id"],
            "question": fact(study["question"], path + "/question"),
            "status": fact(study["status"], path + "/status"),
            "planned_arm_ids": [
                fact(arm["id"], f"{path}/arms/{j}/id")
                for j, arm in enumerate(study["arms"])
            ],
            "validation_arms": arms,
            "registered_claims": [
                fact(claim, f"{path}/claims/{i}")
                for i, claim in enumerate(study.get("claims") or [])
            ],
            "diagnostic_checks": [
                fact({key: diagnostic[key] for key in ("status", "metric", "seed", "matching", "uncertainty_scope")
                      if key in diagnostic}, path + "/diagnostic_results/" + name.replace("~", "~0").replace("/", "~1"))
                for name, diagnostic in (study.get("diagnostic_results") or {}).items()
                if isinstance(diagnostic, dict)
            ],
            "review_objections": [
                fact({key: obj[key] for key in ("id", "scope", "status", "blocking", "requirement")
                      if key in obj}, f"/program/objections/{i}")
                for i, obj in enumerate(objections)
                if obj.get("id") in cited_objections and obj.get("scope") == study["id"]
            ],
            "recorded_verdicts": [
                fact(verdict, f"{path}/verdicts/{i}")
                for i, verdict in enumerate(verdicts)
            ],
        })
    return {
        "schema_version": 1,
        "run_id": fact(bundle["run_id"], "/run_id"),
        "objective": fact(bundle["objective"], "/objective"),
        "primary_metric": fact(bundle["contract"]["primary_metric"], "/contract/primary_metric"),
        "data_configuration": fact(spec["data"], "/data", "spec.json"),
        "success_criterion": fact(spec["success"], "/success", "spec.json") if spec.get("success") else None,
        "selection_objectives": fact(spec["objectives"], "/objectives", "spec.json") if spec.get("objectives") else None,
        "audit_outcome": fact(final["audit"]["success"], "/final/audit/success")
        if isinstance((final.get("audit") or {}).get("success"), bool) else None,
        "evaluation_warnings": [
            fact(warning, f"/evaluation_protocol/warnings/{i}")
            for i, warning in enumerate((bundle.get("evaluation_protocol") or {}).get("warnings") or [])
        ],
        "study_summary": {
            key: fact(evaluation[key], "/" + key, "program_evaluation.json")
            for key in ("study_count", "studies_complete", "claim_outcomes")
        },
        "final_model": {
            "methods": fact(final["labels"], "/final/labels") if "labels" in final else None,
            "validation_selection_score": fact(final["val"], "/final/val") if "val" in final else None,
            "held_out_metrics": fact(final["test"], "/final/test") if "test" in final else None,
            "test_bootstrap": fact(result["test_bootstrap"], "/final/result/test_bootstrap")
            if "test_bootstrap" in result else None,
        },
        "studies": studies,
        "writing_notes": [
            "Study arms use validation metrics. Final held-out metrics are listed separately.",
            "Recorded verdicts are historical outputs. Verify unresolved objections and later qualifications before making scientific claims.",
            "Use the original run files for diagnostic detail and the ablation report for campaign-level qualifications.",
        ],
    }


def prepare_v1(run):
    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    spec = read("spec.json")
    manifest = read("final/manifest.json")
    status = read("status.json")
    records = [json.loads(line) for line in (run / "artifacts.jsonl").read_text().splitlines() if line.strip()]

    def fact(value, file, pointer):
        return {"value": value, "source": {"file": file, "json_pointer": pointer}}

    def field(record, file, key):
        return fact(record[key], file, "/" + key)

    def last(kind, stage, accepted):
        for index in reversed(range(len(records))):
            item = records[index]
            if item["kind"] == kind and item["stage"] == stage and accepted(item["data"]):
                return index, item["data"]
        raise ValueError(f"Missing {stage} {kind} record")

    checkpoint_index, checkpoint = last("checkpoint", "final", lambda d: "test" in d)
    if checkpoint["test"] != manifest["test"]:
        raise ValueError("Final checkpoint and manifest disagree on the test measurements")
    audit_index, audit = last("critique", "final", lambda d: d.get("type") == "final_audit")
    base_index, base = last("result", "baseline", lambda d: d.get("status") == "ok")
    selection_index, selection = last("decision", "final", lambda d: d.get("decision") == "select_final")
    def log_fact(i, key, row):
        return fact(row[key], "artifacts.jsonl", f"/{i}/data/{key}")

    return {
        "schema_version": 1,
        "run_id": field(status, "status.json", "run_id"),
        "objective": field(spec, "spec.json", "problem"),
        "primary_metric": fact(spec["metrics"]["primary"], "spec.json", "/metrics/primary"),
        "data_configuration": field(spec, "spec.json", "data"),
        "task_description": field(spec, "spec.json", "simulator_description"),
        "simulator": field(spec, "spec.json", "simulator"),
        "selection_objectives": field(spec, "spec.json", "objectives"),
        "success_criterion": field(spec, "spec.json", "success"),
        "audit_outcome": log_fact(audit_index, "success", audit),
        "baseline_validation": log_fact(base_index, "val", base),
        "selection": log_fact(selection_index, "reason", selection),
        "ensemble_members": field(manifest, "final/manifest.json", "members"),
        "ensemble_weights": field(manifest, "final/manifest.json", "weights"),
        "final_model": {
            "methods": fact([checkpoint["kind"]], "artifacts.jsonl", f"/{checkpoint_index}/data/kind"),
            "validation_selection_score": log_fact(audit_index, "val_estimate", audit),
            "held_out_metrics": field(manifest, "final/manifest.json", "test"),
        },
        "studies": [],
    }


def extract_run(run):
    if (run / "evidence_bundle.json").is_file():
        names = ("evidence_bundle.json", "spec.json", "program_evaluation.json")
        return prepare(*(json.loads((run / name).read_text(encoding="utf-8")) for name in names))
    if (run / "artifacts.jsonl").is_file():
        return prepare_v1(run)
    raise ValueError("Expected a harness run directory containing evidence_bundle.json or artifacts.jsonl.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        names = ("evidence_bundle.json", "spec.json", "program_evaluation.json")
        inputs = [json.loads((args.run_dir / name).read_text(encoding="utf-8")) for name in names]
        output = json.dumps(prepare(*inputs), ensure_ascii=False, indent=2, allow_nan=False)
        args.output.write_text(output + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Could not prepare paper input: {error}\n")


if __name__ == "__main__":
    main()
