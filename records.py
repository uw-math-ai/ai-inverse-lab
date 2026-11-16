"""Append-only events and immutable iteration snapshots for evaluation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace a summary atomically; events/iteration files are append-only."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class RunLog:
    def __init__(self, output_root: Path, mode: str):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + mode + "-" + uuid4().hex[:10]
        self.path = output_root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        (self.path / "agent_evaluations.jsonl").touch()

    def event(self, kind: str, **data: Any) -> None:
        self.sequence += 1
        event = {"schema_version": 1, "run_id": self.run_id, "sequence": self.sequence,
                 "timestamp": utc_now(), "kind": kind, **data}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def iteration(self, record: dict) -> None:
        with (self.path / "iterations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.event("iteration_recorded", iteration=record["iteration"], score=record["score"])

    def agent_evaluation(self, row: dict) -> None:
        with (self.path / "agent_evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.event("agent_evaluated", call_id=row["call_id"], stage=row["role"],
                   iteration=row["iteration"], score=row["score"], progress=row["progress"])


def render_report(manifest: dict, records: list[dict], summary: dict) -> str:
    lines = ["# Mathematical research run", "", f"Run: `{summary['run_id']}`",
             f"Mode: **{summary['mode']}** · Backend: **{manifest['backend']}**",
             f"Status: **{summary['status']}** · Stop reason: {summary['stop_reason']}", "",
             "> Lean certification applies only to the fixed formal target. Agent prose and simulator evidence require separate review.",
             "", "## Benchmark", "", manifest["benchmark"]["statement"], ""]
    if summary.get("agent_evaluations"):
        lines.extend(["## Agent evaluation by iteration", "",
                      "Scores use separate proof and skeptic rubrics. Provisional findings are not established progress. Lean does not validate a skeptic's diagnosis.", "",
                      "| Iteration | Role | Score | Evidence | Change | Tokens | Seconds |",
                      "| --- | --- | ---: | --- | --- | ---: | ---: |"])
        for row in summary["agent_evaluations"]:
            score = row["score"]
            lines.append(f"| {row['iteration']} | {row['role']} | {score['score'] if score['score'] is not None else 'unscored'} | "
                         f"{'provisional' if score.get('provisional') else score.get('evidence_type', 'reviewed')} | "
                         f"{row['progress']['direction']} | {row['usage'].get('total_tokens', 'unknown')} | {row.get('elapsed_seconds', 0):.3f} |")
        lines.append("")
    for record in records:
        proposal = record["proposal"]
        lines.extend([f"## Iteration {record['iteration']}", "", "### Claim", "", proposal["claim"],
                      "", "### Assumptions", "", *[f"- {a}" for a in proposal["assumptions"]],
                      "", "### Proposed proof", "", proposal["proof"] or "No proof supplied.",
                      "", "### Lean proof body", "", "```lean", (proposal.get("certificate") or {}).get("proof", "No Lean proof submitted."), "```",
                      "", "### Proof gaps and critique", "", json.dumps(record["proof_gaps"], indent=2),
                      "", "### Claimed repairs", "", json.dumps(proposal["repairs"], indent=2),
                      "", "### Verification", "", "```json", json.dumps(record["verification"], indent=2), "```",
                      "", f"**Score: {record['score']['score']} ({'provisional' if record['score']['provisional'] else 'verified or reviewed'}).** {record['score']['rationale']}", ""])
        for row in record.get("agent_evaluations", []):
            lines.extend([f"### {row['role']} evaluation", "", row["score"]["rationale"], "",
                          "```json", json.dumps(row["dimensions"], indent=2), "```", ""])
    lines.extend(["## Usage", "", "```json", json.dumps(summary["usage"], indent=2), "```", "",
                  "Prompts, raw model responses, simulator results, and errors are in `events.jsonl`. "
                  "Iteration snapshots are in `iterations.jsonl`. Self-contained agent training records are in `agent_evaluations.jsonl`. Independent reviews are stored separately.", ""])
    return "\n".join(lines)
