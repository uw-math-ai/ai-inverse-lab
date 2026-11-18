"""Command-line entry point for proof-blind research runs and paired evaluation."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "workflow"

from .benchmarks import BENCHMARK_IDS, DEFAULT_BENCHMARK_ID, load_benchmark
from .agent_evaluation import evaluate_actions, progress, summarize_agents
from .credentials import check_openai_key, configure_openai, openai_api_key
from .evaluation import aggregate_runs, proposal_digest, score_iteration, validate_review
from .records import utc_now, write_json, write_jsonl
from .runtime import RunConfig, run_workflow
from .verification import lean_status, verify
from .research_suite import SUITE_BENCHMARK_IDS, SUITE_ID
from .ledger import trajectory_metrics


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _review_record(summary: dict, iteration: int) -> dict:
    records = summary["records"]
    matches = [record for record in records if record["iteration"] == iteration]
    if len(matches) != 1:
        raise ValueError("Iteration does not identify one completed proposal in this run.")
    record = deepcopy(matches[0])
    record["previous_proof_gaps"] = [gap for prior in records if prior["iteration"] < iteration
                                      for gap in prior["proof_gaps"]]
    return record


def reviewed_summary(run_directory: Path) -> dict:
    """Revalidate reviews from disk, preserving the original automatic summary."""
    summary = _read_json(run_directory / "summary.json")
    applied = []
    latest_reviews = {}
    for path in sorted((run_directory / "reviews").glob("*.json")):
        saved = _read_json(path)
        review = saved["review"]
        record = _review_record(summary, review["iteration"])
        validated = validate_review(review, record, summary["run_id"])
        judgment = validated.get("critique_review") or {}
        if any(judgment.get(key) for key in ("confirmed_gap_ids", "dismissed_gap_ids", "no_gap_assessment_validated")):
            latest_reviews[record["iteration"]] = validated
        if validated.get("assess_proof", True):
            score = score_iteration(record["proposal"], record["verification"], validated)
            for current in summary["records"]:
                if current["iteration"] == record["iteration"]:
                    current["score"] = score
        applied.append(path.name)
    summary["applied_review_files"] = applied
    summary["final_score"] = summary["records"][-1]["score"]["score"] if summary["records"] else 0
    summary["max_score"] = max((record["score"]["score"] for record in summary["records"]), default=0)
    if "agent_evaluations" in summary:
        benchmark = _read_json(run_directory / "manifest.json")["benchmark"]
        previous = None
        for record in summary["records"]:
            calls = [{"stage": row["role"], "status": row["call_status"],
                      **{key: row[key] for key in ("call_id", "output", "usage", "elapsed_seconds", "input", "input_sha256", "model") if key in row}}
                     for row in record.get("agent_evaluations", [])]
            record["agent_evaluations"] = evaluate_actions(
                record, benchmark, calls, previous, latest_reviews.get(record["iteration"]))
            for row in record["agent_evaluations"]:
                row.update(mode=summary["mode"], backend=summary["backend"], model=row.get("model") or summary["model"])
            previous = record
        completed = {record["iteration"] for record in summary["records"]}
        pending = [row for row in summary["agent_evaluations"] if row["iteration"] not in completed]
        for row in pending:
            previous_rows = previous.get("agent_evaluations", []) if previous else []
            comparable = next((item for item in previous_rows if (item["role"] if item["role"] in ("SA", "VA") else "author") == (row["role"] if row["role"] in ("SA", "VA") else "author")), None)
            row["progress"] = progress(row, comparable)
        summary["agent_evaluations"] = [row for record in summary["records"] for row in record.get("agent_evaluations", [])] + pending
        summary["agent_metrics"] = summarize_agents(summary["agent_evaluations"])
    summary["trajectory"] = trajectory_metrics(summary["records"])
    return summary


def apply_review(run_directory: Path, review_path: Path) -> Path:
    summary = _read_json(run_directory / "summary.json")
    review = _read_json(review_path)
    record = _review_record(summary, review.get("iteration"))
    validated = validate_review(review, record, summary["run_id"])
    score = (score_iteration(record["proposal"], record["verification"], validated)
             if validated.get("assess_proof", True)
             else _review_record(reviewed_summary(run_directory), record["iteration"])["score"])
    directory = run_directory / "reviews"
    directory.mkdir(exist_ok=True)
    timestamp = utc_now()
    path = directory / (timestamp.replace(":", "-") + "-" + uuid4().hex[:8] + ".json")
    write_json(path, {"applied_at": timestamp, "review": dict(validated), "score": score})
    write_json(run_directory / "summary_adjudicated.json", reviewed_summary(run_directory))
    return path


def _comparison_report(aggregate: dict, run_paths: list[Path]) -> str:
    lines = ["# Single-agent and multi-agent comparison", "",
             "> Demo runs are scripted software checks. Lean verification applies only to the fixed formal target; human reviews assess interpretation.", "",
             "| Mode | Runs | Failed | Independently verified | Mean final score | Model calls | Total tokens |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, values in aggregate["by_mode"].items():
        lines.append(f"| {mode} | {values['runs']} | {values['failed_runs']} | {values['independently_verified_runs']} | "
                     f"{values['mean_final_score']:.2f} | {values['model_calls']['total']} | {values['tokens']['total_tokens']['total']} |")
    lines.extend(["", f"Matched pairs: {aggregate['paired']['matched_pairs']}", "", aggregate["interpretation"],
                  "", "## Run records", ""])
    lines.extend(f"- [{path.name}]({path.name}/report.md)" for path in run_paths)
    return "\n".join(lines) + "\n"


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("demo", "openai"), default="demo")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--ca-model", default="")
    parser.add_argument("--stop-on-verified", action="store_true", help="Optionally stop when Lean first verifies the target; otherwise continue refinement.")
    parser.add_argument("--reasoning-effort", choices=("", "none", "low", "medium", "high"), default="")
    parser.add_argument("--sa-model", default="")
    parser.add_argument("--verifier-model", default="", help="Optional independent verifier adviser model.")
    parser.add_argument("--budget-policy", choices=("calls", "iterations"), default="calls")
    parser.add_argument("--rounds", type=int, default=10, help="Iteration target; calls policy gives the single agent an equal call budget.")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--max-simulator-calls", type=int, default=0)
    parser.add_argument("--benchmark", choices=BENCHMARK_IDS, default=DEFAULT_BENCHMARK_ID)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(mode=getattr(args, "mode", "multi"), backend=args.backend,
                     model=args.model, rounds=args.rounds, max_output_tokens=args.max_output_tokens,
                     max_simulator_calls=args.max_simulator_calls, seed=args.seed,
                     benchmark_id=args.benchmark, ca_model=args.ca_model, sa_model=args.sa_model,
                     verifier_model=args.verifier_model, budget_policy=args.budget_policy, reasoning_effort=args.reasoning_effort, stop_on_verified=args.stop_on_verified)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure-openai", help="Privately enter, check, and save an OpenAI API key.")
    setup.add_argument("--model", default="gpt-4o-mini")
    check = commands.add_parser("check-openai", help="Check API credentials and model metadata without generating tokens.")
    check.add_argument("--model", default="gpt-4o-mini")
    export = commands.add_parser("export-training", help="Export one training record per agent call, optionally with independent reviews.")
    export.add_argument("run_directory", type=Path)
    export.add_argument("--reviewed", action="store_true")
    export.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run one budgeted workflow.")
    _add_run_options(run)
    run.add_argument("--mode", choices=("multi", "single"), default="multi")
    compare = commands.add_parser("compare", help="Run paired, budget-matched arms with alternating execution order.")
    _add_run_options(compare)
    compare.add_argument("--trials", type=int, default=3)
    suite = commands.add_parser("suite", help="Run all development benchmarks as paired comparisons.")
    _add_run_options(suite)
    suite.add_argument("--trials", type=int, default=1)
    commands.add_parser("suite-check", help="Check all private reference proofs in Lean.")
    benchmark = commands.add_parser("benchmark", help="Print exactly the public benchmark packet.")
    benchmark.add_argument("--benchmark", choices=BENCHMARK_IDS, default=DEFAULT_BENCHMARK_ID)
    commands.add_parser("lean-status", help="Check the local pinned Lean compiler.")
    recheck = commands.add_parser("verify-lean", help="Recheck a recorded proposal with the local Lean compiler.")
    recheck.add_argument("run_directory", type=Path)
    recheck.add_argument("--iteration", type=int)
    template = commands.add_parser("review-template", help="Create an unfilled independent-review form.")
    template.add_argument("run_directory", type=Path)
    template.add_argument("--iteration", type=int)
    template.add_argument("--reviewer", default="")
    review = commands.add_parser("review", help="Apply an independently authored review of one exact proposal.")
    review.add_argument("run_directory", type=Path)
    review.add_argument("review_file", type=Path)
    aggregate = commands.add_parser("aggregate", help="Recompute descriptive results, optionally with human reviews.")
    aggregate.add_argument("run_directories", nargs="+", type=Path)
    aggregate.add_argument("--reviewed", action="store_true")
    aggregate.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ["run"])
    try:
        if args.command == "suite-check":
            from .suite_references import check_suite
            result = check_suite()
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        if args.command == "suite":
            if not 1 <= args.trials <= 10:
                raise ValueError("trials must be in [1, 10].")
            base = _config(args)
            if base.seed + args.trials - 1 > 2**31 - 1:
                raise ValueError("Seed plus trials exceeds the supported range.")
            from .suite_references import check_suite
            audit = check_suite()
            if not audit["passed"]:
                raise ValueError("Suite reference validation failed. Run suite-check for diagnostics.")
            directory = args.output / ("comparison-suite-" + uuid4().hex[:12])
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "suite-review.json", audit)
            runs, paths = [], []
            for identifier in SUITE_BENCHMARK_IDS:
                for trial in range(args.trials):
                    for mode in (("single", "multi") if trial % 2 == 0 else ("multi", "single")):
                        config = replace(base, mode=mode, benchmark_id=identifier, seed=base.seed + trial,
                                         pair_id=f"{directory.name}-{identifier}-{trial}")
                        path, summary = run_workflow(config, directory)
                        paths.append(path)
                        runs.append(summary)
                        result = aggregate_runs(runs)
                        result.update(suite_id=SUITE_ID, backend=base.backend,
                                      run_directories=[str(p.resolve()) for p in paths],
                                      planned_runs=len(SUITE_BENCHMARK_IDS)*args.trials*2,
                                      campaign_complete=False,
                                      by_benchmark={key: aggregate_runs([r for r in runs if r["benchmark_id"] == key]) for key in SUITE_BENCHMARK_IDS})
                        write_json(directory / "comparison.json", result)
                        print(f"{identifier} {mode}: {summary['status']} ({path.name})", flush=True)
            result["campaign_complete"] = True
            write_json(directory / "comparison.json", result)
            (directory / "comparison.md").write_text(_comparison_report(result, paths))
            print(f"Suite comparison: {directory / 'comparison.md'}")
            return 0 if all(r["status"] == "completed" for r in runs) else 1
        if args.command == "export-training":
            if args.output.exists():
                raise ValueError("Export destination already exists. Choose a new file to preserve existing records.")
            summary = reviewed_summary(args.run_directory) if args.reviewed else _read_json(args.run_directory / "summary.json")
            if "agent_evaluations" not in summary:
                raise ValueError("This historical run predates per-agent evaluation; no scores will be fabricated.")
            write_jsonl(args.output, summary["agent_evaluations"])
            print(f"Exported {len(summary['agent_evaluations'])} agent records: {args.output}")
            return 0
        if args.command == "configure-openai":
            return configure_openai(args.model)
        if args.command == "check-openai":
            key = openai_api_key()
            if not key:
                raise ValueError("No OpenAI API key configured. Run configure-openai first.")
            success, message = check_openai_key(key, args.model)
            print(message)
            return 0 if success else 1
        if args.command == "lean-status":
            status = lean_status()
            print(json.dumps(status, indent=2))
            return 0 if status["available"] else 1
        if args.command == "verify-lean":
            summary = _read_json(args.run_directory / "summary.json")
            if not summary.get("records"):
                raise ValueError("No completed proposal to recheck.")
            iteration = args.iteration if args.iteration is not None else summary["records"][-1]["iteration"]
            record = _review_record(summary, iteration)
            manifest = _read_json(args.run_directory / "manifest.json")
            result = verify(record["proposal"], manifest["benchmark"])
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "verified" else 1
        if args.command == "benchmark":
            print(json.dumps(load_benchmark(args.benchmark), indent=2))
            return 0
        if args.command == "run":
            path, summary = run_workflow(_config(args), args.output)
            print(f"{summary['status']}: {path / 'report.md'}")
            if summary["error"]:
                print(summary["error"]["message"], file=sys.stderr)
            return 0 if summary["status"] == "completed" else 1
        if args.command == "compare":
            if not 1 <= args.trials <= 100:
                raise ValueError("trials must be in [1, 100].")
            base = _config(args)
            directory = args.output / ("comparison-" + uuid4().hex[:12])
            directory.mkdir(parents=True, exist_ok=False)
            runs, paths = [], []
            for trial in range(args.trials):
                pair_id = directory.name + f"-trial-{trial + 1}"
                for mode in (("single", "multi") if trial % 2 == 0 else ("multi", "single")):
                    config = replace(base, mode=mode, seed=base.seed + trial, pair_id=pair_id)
                    path, summary = run_workflow(config, directory)
                    paths.append(path)
                    runs.append(summary)
                    print(f"Trial {trial + 1} {mode}: {summary['status']} ({path.name})", flush=True)
            result = aggregate_runs(runs)
            result["run_directories"] = [str(path.resolve()) for path in paths]
            result["backend"] = base.backend
            write_json(directory / "comparison.json", result)
            (directory / "comparison.md").write_text(_comparison_report(result, paths), encoding="utf-8")
            print(f"Comparison: {directory / 'comparison.md'}")
            return 0 if all(run["status"] == "completed" for run in runs) else 1
        if args.command == "review-template":
            summary = _read_json(args.run_directory / "summary.json")
            if not summary["records"]:
                raise ValueError("No completed proposal is available for review.")
            iteration = args.iteration if args.iteration is not None else summary["records"][-1]["iteration"]
            record = _review_record(summary, iteration)
            value = {"run_id": summary["run_id"], "iteration": iteration,
                     "assess_proof": True,
                     "proposal_sha256": record["proposal_sha256"], "reviewer": args.reviewer,
                     "target_preserved": False, "assumptions_preserved": False,
                     "complete_proof_validated": False, "repaired_gap_ids": [],
                     "evidence_type": "human_mathematical_review", "evidence": ""}
            if record.get("critique"):
                value["critique_review"] = {"critique_sha256": proposal_digest(record["critique"]),
                                           "confirmed_gap_ids": [], "dismissed_gap_ids": [],
                                           "no_gap_assessment_validated": False}
            path = args.run_directory / f"review-template-{iteration}-{uuid4().hex[:8]}.json"
            write_json(path, value)
            print(path)
            return 0
        if args.command == "review":
            print(apply_review(args.run_directory, args.review_file))
            return 0
        summaries = [reviewed_summary(path) if args.reviewed else _read_json(path / "summary.json")
                     for path in args.run_directories]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, aggregate_runs(summaries))
        print(args.output)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
