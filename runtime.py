"""Budgeted LangGraph CA -> SA -> verification -> repair research loop."""

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .backends import DemoBackend, OpenAIBackend
from .agent_evaluation import RUBRIC, evaluate_actions, summarize_agents
from .benchmarks import DEFAULT_BENCHMARK_ID, load_benchmark
from .evaluation import proposal_digest, score_iteration
from .records import RunLog, digest, render_report, utc_now, write_json
from .schemas import Critique, Proposal
from .simulator import run_simulator_request
from .verification import is_verified_receipt, lean_status, verify


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["multi", "single"] = "multi"
    backend: Literal["demo", "openai"] = "demo"
    model: str = "gpt-4o-mini"
    rounds: int = 3
    max_output_tokens: int = 2400
    max_simulator_calls: int = 0
    seed: int = 2026
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    pair_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("multi", "single") or self.backend not in ("demo", "openai"):
            raise ValueError("Unknown mode or backend.")
        load_benchmark(self.benchmark_id)
        for name, low, high in (("rounds", 1, 20), ("max_output_tokens", 256, 16000),
                                ("max_simulator_calls", 0, 80), ("seed", 0, 2**31 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}].")

    @property
    def max_model_calls(self) -> int:
        return 2 * self.rounds


class ResearchState(TypedDict, total=False):
    iteration: int
    proposal: dict
    critique: dict | None
    verification: dict
    simulator_results: list[dict]


COMMON_PROMPT = (
    "You are working on a proof-blind mathematical benchmark. Use only the supplied "
    "statement, definitions, assumptions and allowed prior results. You have no retrieval "
    "or filesystem access. Do not cite or reproduce a remembered published proof; disclose "
    "recognition in remaining_gaps (Proposal) or assessment (Critique). Keep the original target and all its "
    "assumptions explicit; copy claim and assumptions exactly from the packet. A finite example "
    "cannot prove the universal claim. Return the required structured "
    "schema. In a Proposal, use an empty proof when no proof is available. Never certify your own proof. "
    "Simulator requests are optional bounded numerical probes on path3, paper8 or paper9; "
    "use a fixed h for comparing nonnegative killing potentials. The simulator uses first "
    "positive-time passage/return observations, which differ from occupancy probabilities. "
    "The primary path is Lean formal verification. Simulator probes are disabled by default "
    "and never certify a proof. In a Proposal, "
    "null certificate and empty repair lists are valid. Empty simulator_requests are valid "
    "for either schema. For Lean-ready benchmarks submit certificate.kind=lean4, the exact target_id, "
    "and a proof body within the supplied proof-language contract. The server fixes the theorem and "
    "audits its axioms. No sorry, admit, new axioms, imports or executable metaprogramming. "
    "Use compiler diagnostics to repair proofs. For formalization_pending targets use certificate=null."
)


def _environment() -> dict:
    dependencies = {}
    for package in ("langgraph", "langchain", "langchain-openai", "pydantic", "numpy"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    # Content hashes remain useful on machines without a usable git executable.
    project = Path(__file__).resolve().parent.parent
    source_paths = list((project / "workflow").glob("*.py"))
    source_paths += list((project / "forwardSimulation" / "forward_sim").glob("*.py"))
    source_paths += list((project / "workflow" / "lean" / "MathAILab").glob("*.lean"))
    sources = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(source_paths)}
    return {"python": platform.python_version(), "dependencies": dependencies, "lean": lean_status(), "source_sha256": sources}


class ResearchRun:
    def __init__(self, config: RunConfig, log: RunLog, backend: Any = None):
        self.config, self.log = config, log
        self.benchmark = load_benchmark(config.benchmark_id)
        self.backend = backend
        self.records: list[dict] = []
        self.calls: list[dict] = []
        self.agent_evaluations: list[dict] = []
        self.model_calls = 0
        self.simulator_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "metadata_missing_calls": 0}
        self.started = time.perf_counter()

    def call(self, stage: str, schema: type[BaseModel], system: str, payload: dict, iteration: int) -> dict:
        if self.model_calls >= self.config.max_model_calls:
            raise RuntimeError("Model call budget exhausted.")
        messages = [{"role": "system", "content": COMMON_PROMPT + "\n\n" + system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        self.model_calls += 1
        call_id = self.model_calls
        self.log.event("model_request", call_id=call_id, stage=stage, iteration=iteration,
                       messages=messages, output_schema_name=schema.__name__, output_schema=schema.model_json_schema())
        started = time.perf_counter()
        call = {"call_id": call_id, "stage": stage, "iteration": iteration,
                "status": "invalid_output", "output": None, "usage": {}}
        self.calls.append(call)
        try:
            response = self.backend.invoke(stage, schema, messages)
        except Exception as exc:
            call.update(status="provider_error", elapsed_seconds=time.perf_counter() - started)
            self.usage["metadata_missing_calls"] += 1
            self.log.event("model_error", call_id=call_id, stage=stage, iteration=iteration,
                           error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter()-started)
            raise
        call["elapsed_seconds"] = time.perf_counter() - started
        self.log.event("model_response", call_id=call_id, stage=stage, iteration=iteration,
                       response=response, elapsed_seconds=time.perf_counter()-started)
        usage = response.get("usage")
        if not isinstance(usage, dict) or not all(key in usage for key in ("input_tokens", "output_tokens", "total_tokens")):
            self.usage["metadata_missing_calls"] += 1
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    self.usage[key] += value
                    call["usage"][key] = value
        metadata = response.get("raw", {}).get("response_metadata", {})
        refusal = response.get("raw", {}).get("additional_kwargs", {}).get("refusal")
        if metadata.get("finish_reason") in ("length", "max_tokens"):
            raise ValueError(f"{stage} exceeded its output cap; partial response was preserved.")
        if refusal or response.get("parsing_error") or response.get("parsed") is None:
            raise ValueError(f"{stage} did not return a valid structured answer; see model_response event.")
        # Validate even injected/fake backends, not just provider-parsed output.
        result = schema.model_validate(response["parsed"]).model_dump()
        call.update(status="completed", output=result)
        return result

    def ca(self, state: ResearchState) -> dict:
        proposal = self.call("CA", Proposal,
            "Propose a precise claim and attempt an independent proof of the given theorem. "
            "Repairs must be empty in the initial attempt. State unproved steps candidly.",
            {"benchmark": self.benchmark}, 1)
        return {"iteration": 1, "proposal": proposal, "critique": None}

    def sa(self, state: ResearchState) -> dict:
        critique = self.call("SA", Critique,
            "Act as an independent skeptic. Check quantifiers, assumptions, repeated eigenvalues, "
            "time indexing, missing arguments and possible counterexamples. Do not take the "
            "author's confidence as evidence. Give stable gap_id values so repairs can refer to "
            "them. Mark scope=general_proof for Lean proof gaps and numerical_probe for "
            "simulator findings. target_preserved and assumptions_preserved are unverified reviewer opinions.",
            {"benchmark": self.benchmark, "proposal": state["proposal"]}, state["iteration"])
        return {"critique": critique}

    def verification(self, state: ResearchState) -> dict:
        results = []
        requests = [("proposal", item) for item in state["proposal"]["simulator_requests"]]
        if state.get("critique"):
            requests += [("skeptic", item) for item in state["critique"]["simulator_requests"]]
        for origin, request in requests:
            cleaned = {key: value for key, value in request.items() if value is not None}
            if self.simulator_calls >= self.config.max_simulator_calls:
                result = {"status": "skipped", "reason": "Per-run simulator budget exhausted.", "evidence_kind": "none"}
            else:
                self.simulator_calls += 1
                result = run_simulator_request(cleaned)
            result_record = {"origin": origin, "request": cleaned, "result": result}
            results.append(result_record)
            self.log.event("simulator_result", iteration=state["iteration"], **result_record)
        result = verify(state["proposal"], self.benchmark)
        if result.get("source"):
            (self.log.path / "proof.lean").write_text(result["source"], encoding="utf-8")
        else:
            (self.log.path / "proof.lean").unlink(missing_ok=True)
        write_json(self.log.path / "lean-verification.json", result)
        self.log.event("verification", iteration=state["iteration"], result=result)
        return {"verification": result, "simulator_results": results}

    def record(self, state: ResearchState) -> dict:
        critique = state.get("critique")
        gaps = [{**gap, "source": "skeptic", "assessment_status": "reported"} for gap in critique["gaps"]] if critique else []
        gaps += [{"gap_id": f"verification_{state['iteration']}_{i}", "description": gap,
                  "severity": "unassessed", "source": "verification", "scope": "general_proof", "assessment_status": "checker_finding"}
                 for i, gap in enumerate(state["verification"]["gaps"], 1)]
        gaps += [{"gap_id": f"author_{state['iteration']}_{i}", "description": gap,
                  "severity": "unassessed", "source": "author", "scope": "general_proof", "assessment_status": "self_reported"}
                 for i, gap in enumerate(state["proposal"]["remaining_gaps"], 1)]
        score = score_iteration(state["proposal"], state["verification"])
        gaps = [{**gap, "source_iteration": state["iteration"]} for gap in gaps]
        record = {"schema_version": 2, "run_id": self.log.run_id, "iteration": state["iteration"],
                  "proposal_sha256": proposal_digest(state["proposal"]), "proposal": state["proposal"],
                  "critique_sha256": proposal_digest(critique) if critique else None,
                  "critique": critique, "proof_gaps": gaps, "verification": state["verification"],
                  "simulator_results": state.get("simulator_results", []), "score": score,
                  "model_calls": self.model_calls, "simulator_calls": self.simulator_calls,
                  "elapsed_seconds": time.perf_counter() - self.started}
        self.evaluate_record(record)
        self.records.append(record)
        self.log.iteration(record)
        return {}

    def evaluate_record(self, record: dict) -> None:
        record["previous_proof_gaps"] = [gap for prior in self.records for gap in prior["proof_gaps"]]
        calls = [call for call in self.calls if call["iteration"] == record["iteration"]]
        record["agent_evaluations"] = evaluate_actions(
            record, self.benchmark, calls, self.records[-1] if self.records else None)
        for row in record["agent_evaluations"]:
            row.update(mode=self.config.mode, backend=self.config.backend, model=self.config.model)
            self.agent_evaluations.append(row)
            self.log.agent_evaluation(row)

    def evaluate_unfinished_calls(self) -> None:
        completed = {record["iteration"] for record in self.records}
        for iteration in sorted({call["iteration"] for call in self.calls} - completed):
            calls = [call for call in self.calls if call["iteration"] == iteration]
            proposal = next((call["output"] for call in calls if call["stage"] != "SA" and call["output"]), {})
            critique = next((call["output"] for call in calls if call["stage"] == "SA"), None)
            gaps = [{**gap, "source": "skeptic", "assessment_status": "reported", "source_iteration": iteration}
                    for gap in (critique or {}).get("gaps", [])]
            self.evaluate_record({"run_id": self.log.run_id, "iteration": iteration,
                                  "proposal": proposal, "critique": critique, "proof_gaps": gaps,
                                  "verification": {"status": "not_recorded", "summary": "The iteration stopped before a verification result was recorded."},
                                  "score": score_iteration(proposal, {})})

    def repair(self, state: ResearchState) -> dict:
        iteration = state["iteration"] + 1
        feedback = {"verification": state["verification"], "simulator_results": state.get("simulator_results", [])}
        if self.config.mode == "multi":
            feedback["skeptic"] = state.get("critique")
            instruction = "Repair the proposal using the skeptic and tool feedback."
        else:
            instruction = "As the same single researcher, critique and revise your own preceding attempt using tool feedback."
        proposal = self.call("Repair", Proposal, instruction +
            " Refer to prior gap_id values for claimed repairs. Explain each change, keep the "
            "original theorem target/assumptions, and explicitly list unresolved gaps. "
            "Use the actual Lean compiler diagnostics; a failed check does not refute the theorem.",
            {"benchmark": self.benchmark, "proposal": state["proposal"], "feedback": feedback,
             "previous_proof_gaps": self.records[-1]["proof_gaps"]}, iteration)
        return {"iteration": iteration, "proposal": proposal, "critique": None}

    def route(self, state: ResearchState) -> str:
        if is_verified_receipt(state["proposal"], state["verification"]):
            return END
        # Each multi-agent revision reserves one call for its skeptic. Every
        # repaired attempt is verified and recorded before the graph can stop.
        needed = 2 if self.config.mode == "multi" else 1
        return "Repair" if self.model_calls + needed <= self.config.max_model_calls else END

    def graph(self):
        builder = StateGraph(ResearchState)
        for name, handler in (("CA", self.ca), ("SA", self.sa), ("Verification", self.verification),
                              ("Record", self.record), ("Repair", self.repair)):
            builder.add_node(name, handler)
        builder.add_edge(START, "CA")
        target = "SA" if self.config.mode == "multi" else "Verification"
        builder.add_edge("CA", target)
        builder.add_edge("SA", "Verification")
        builder.add_edge("Verification", "Record")
        builder.add_conditional_edges("Record", self.route, {"Repair": "Repair", END: END})
        builder.add_edge("Repair", target)
        return builder.compile()


def run_workflow(config: RunConfig, output_root: Path, backend: Any = None) -> tuple[Path, dict]:
    log = RunLog(Path(output_root), config.mode)
    run = ResearchRun(config, log, backend)
    manifest = {"schema_version": 2, "run_id": log.run_id, "created_at": utc_now(),
                "agent_evaluation_rubric": RUBRIC,
                **asdict(config), "max_model_calls": config.max_model_calls,
                "benchmark": run.benchmark, "benchmark_sha256": digest(run.benchmark),
                "environment": _environment(),
                "budget_note": "Equal model-call/output caps and simulator-call caps across arms; actual tokens may differ.",
                "single_agent_policy": "Single researcher revises with Lean feedback, no separate skeptic.",
                "demo_warning": "Scripted demo is plumbing validation, not research evaluation." if config.backend == "demo" else None}
    write_json(log.path / "manifest.json", manifest)
    log.event("run_started", config=asdict(config), benchmark_sha256=manifest["benchmark_sha256"])
    status, stop_reason = "completed", "model_call_budget_exhausted"
    error = None
    try:
        if run.benchmark["verification_contract"].get("status") == "ready" and not lean_status()["available"]:
            raise RuntimeError("Lean is required before starting this formal run: " + str(lean_status()["reason"]))
        if run.backend is None:
            run.backend = DemoBackend() if config.backend == "demo" else OpenAIBackend(config.model, config.max_output_tokens, config.seed)
        run.graph().invoke({}, {"recursion_limit": config.max_model_calls * 6 + 10})
        if run.records and run.records[-1]["verification"].get("status") == "verified":
            stop_reason = "formal_target_verified"
    except Exception as exc:
        status, stop_reason = "failed", "stage_error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        log.event("run_error", **error)
    run.evaluate_unfinished_calls()
    summary = {"schema_version": 2, "run_id": log.run_id, "mode": config.mode,
               "backend": config.backend, "model": config.model, "pair_id": config.pair_id,
               "seed": config.seed, "max_model_calls": config.max_model_calls,
               "max_output_tokens": config.max_output_tokens, "max_simulator_calls": config.max_simulator_calls,
               "benchmark_id": config.benchmark_id, "benchmark_sha256": manifest["benchmark_sha256"],
               "status": status, "stop_reason": stop_reason, "error": error,
               "records": run.records, "model_calls": run.model_calls, "simulator_calls": run.simulator_calls,
               "agent_evaluations": run.agent_evaluations, "agent_metrics": summarize_agents(run.agent_evaluations),
               "usage": run.usage, "elapsed_seconds": time.perf_counter()-run.started,
               "final_score": run.records[-1]["score"]["score"] if run.records else 0,
               "max_score": max((r["score"]["score"] for r in run.records), default=0)}
    write_json(log.path / "summary.json", summary)
    (log.path / "report.md").write_text(render_report(manifest, run.records, summary), encoding="utf-8")
    log.event("run_finished", status=status, stop_reason=stop_reason, error=error)
    return log.path, summary
