"""Budgeted LangGraph CA -> SA -> verification -> repair research loop."""

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .backends import DemoBackend, OpenAIBackend
from .agent_evaluation import RUBRIC, evaluate_actions, summarize_agents
from .benchmarks import DEFAULT_BENCHMARK_ID, load_benchmark
from .evaluation import proposal_digest, score_iteration
from .records import RunLog, digest, render_report, utc_now, write_json
from .schemas import Critique, Proposal, VerificationAdvice
from .ledger import advance_ledger, trajectory_metrics, iteration_evidence

MAX_ITERATIONS = 10
from .simulator import run_simulator_request
from .verification import is_verified_receipt, lean_status, verify


@dataclass(frozen=True)
class RunConfig:
    mode: Literal["multi", "single"] = "multi"
    backend: Literal["demo", "openai"] = "demo"
    model: str = "gpt-4o-mini"
    rounds: int = 10
    max_output_tokens: int = 16000
    max_simulator_calls: int = 0
    seed: int = 2026
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    pair_id: str | None = None
    ca_model: str = ""
    sa_model: str = ""
    verifier_model: str = ""
    budget_policy: Literal["calls", "iterations"] = "calls"
    reasoning_effort: str = ""
    stop_on_verified: bool = False

    def __post_init__(self) -> None:
        if type(self.stop_on_verified) is not bool:
            raise ValueError("stop_on_verified must be a boolean.")
        if self.mode not in ("multi", "single") or self.backend not in ("demo", "openai"):
            raise ValueError("Unknown mode or backend.")
        if self.budget_policy not in ("calls", "iterations"):
            raise ValueError("budget_policy must be calls or iterations.")
        if self.reasoning_effort not in ("", "none", "low", "medium", "high"):
            raise ValueError("reasoning_effort must be empty, none, low, medium or high.")
        for name in ("model", "ca_model", "sa_model", "verifier_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}", value)) or (name == "model" and not value):
                raise ValueError(f"{name} must be a model identifier (120 characters maximum).")
        load_benchmark(self.benchmark_id)
        for name, low, high in (("rounds", 1, MAX_ITERATIONS), ("max_output_tokens", 256, 131072),
                                ("max_simulator_calls", 0, 80), ("seed", 0, 2**31 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}].")

    @property
    def max_model_calls(self) -> int:
        return (2 + bool(self.verifier_model)) * self.rounds

    @property
    def max_iterations(self) -> int:
        calls_per_iteration = (2 if self.mode == "multi" else 1) + bool(self.verifier_model)
        configured = self.rounds if self.budget_policy == "iterations" else self.max_model_calls // calls_per_iteration
        return min(configured, MAX_ITERATIONS)

    def role_model(self, stage: str) -> str:
        return (self.sa_model if stage == "SA" else self.verifier_model if stage == "VA" else self.ca_model) or self.model


class ResearchState(TypedDict, total=False):
    iteration: int
    proposal: dict
    critique: dict | None
    verification: dict
    simulator_results: list[dict]
    verification_advice: dict | None
    milestone_checks: list[dict]


COMMON_PROMPT = (
    "You are working on a proof-blind mathematical benchmark. Use only the supplied "
    "statement, definitions, assumptions and allowed prior results. You have no retrieval "
    "or filesystem access. Do not cite or reproduce a remembered published proof; disclose "
    "recognition in remaining_gaps (Proposal) or assessment (Critique). Keep the original target and all its "
    "assumptions explicit. The server owns the formal goal and hypotheses. Prose wording is "
    "reviewed separately; harmless paraphrasing does not block Lean. Never replace the target_id. A finite example "
    "cannot prove the universal claim. Return the required structured "
    "schema. Write proof as readable mathematical prose with short paragraphs and equations delimited "
    "by $...$ or $$...$$. Do not put Lean tactics, JSON, or raw code in the prose proof field. "
    "Put executable Lean ONLY in certificate.proof. Critiques and advice must also use readable prose "
    "and explicit math delimiters. In a Proposal, use an empty proof when no proof is available. Never certify your own proof. "
    "Simulator requests are optional bounded numerical probes on path3, paper8 or paper9; "
    "use a fixed h for comparing nonnegative killing potentials. The simulator uses first "
    "positive-time passage/return observations, which differ from occupancy probabilities. "
    "The primary path is Lean formal verification. Simulator probes are disabled by default "
    "and never certify a proof. In a Proposal, "
    "null certificate and empty repair lists are valid. Empty simulator_requests are valid "
    "for either schema. For Lean-ready benchmarks submit certificate.kind=lean4, the exact target_id, "
    "and a proof body within the supplied proof-language contract. The server fixes the theorem and "
    "audits its axioms. No sorry, admit, new axioms, imports or executable metaprogramming. "
    "Optional milestone_attempts may prove only listed public milestone IDs. All other subclaims remain unverified. "
    "Read verification_contract.proof_language before writing Lean. Only Lean core is installed; "
    "do not use remembered Mathlib lemma names. Local names may be introduced normally. "
    "A preflight rejection means Lean did not run; remove unsupported constructs, do not repeat them. "
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
        self.ledger = None
        self.verified_checkpoint = None
        self.role_backends = {}
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
        payload = dict(payload)
        if stage in ("CA", "Repair") and self.benchmark.get("milestone_ids"):
            payload["milestone_packets"] = [load_benchmark(key) for key in self.benchmark["milestone_ids"]]
        messages = [{"role": "system", "content": COMMON_PROMPT + "\n\n" + system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        self.model_calls += 1
        call_id = self.model_calls
        self.log.event("model_request", call_id=call_id, stage=stage, iteration=iteration,
                       model=self.config.role_model(stage), messages=messages, output_schema_name=schema.__name__, output_schema=schema.model_json_schema())
        started = time.perf_counter()
        call = {"call_id": call_id, "stage": stage, "iteration": iteration,
                "status": "invalid_output", "output": None, "usage": {},
                "input": messages, "input_sha256": digest(messages), "model": self.config.role_model(stage)}
        self.calls.append(call)
        try:
            backend = self.backend
            if backend is None:
                model = self.config.role_model(stage)
                if model not in self.role_backends:
                    self.role_backends[model] = OpenAIBackend(model, self.config.max_output_tokens, self.config.seed, self.config.reasoning_effort)
                backend = self.role_backends[model]
            response = backend.invoke(stage, schema, messages)
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

    def verifier_agent(self, state: ResearchState) -> dict:
        if not self.config.verifier_model:
            return {"verification_advice": None}
        advice = self.call("VA", VerificationAdvice,
            "Independently predict whether the exact submitted Lean proof will verify. Inspect theorem alignment and mathematical obligations. "
            "Return uncertain when you cannot tell. Your prediction is not certification. You have not yet seen this attempt's compiler result.",
            {"benchmark": self.benchmark, "proposal": state["proposal"]}, state["iteration"])
        return {"verification_advice": advice}

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
        if is_verified_receipt(state["proposal"], result) and self.verified_checkpoint is None:
            self.verified_checkpoint = {"iteration": state["iteration"], "proposal": state["proposal"],
                                        "verification": result}
            (self.log.path / "verified-proof.lean").write_text(result["source"], encoding="utf-8")
            write_json(self.log.path / "verified-checkpoint.json", self.verified_checkpoint)
        if result.get("source"):
            (self.log.path / "proof.lean").write_text(result["source"], encoding="utf-8")
        else:
            (self.log.path / "proof.lean").unlink(missing_ok=True)
        write_json(self.log.path / "lean-verification.json", result)
        self.log.event("verification", iteration=state["iteration"], result=result)
        milestone_checks = []
        for attempt in state["proposal"].get("milestone_attempts", []):
            identifier = attempt["benchmark_id"]
            if identifier not in self.benchmark.get("milestone_ids", []):
                milestone_checks.append({**attempt, "status": "rejected", "reason": "Not a listed supporting milestone."})
                continue
            packet = load_benchmark(identifier)
            candidate = {"claim": packet["statement"], "assumptions": packet["assumptions"],
                         "certificate": {"kind": "lean4", "target_id": identifier, "proof": attempt["proof"]}}
            checked = verify(candidate, packet)
            milestone_checks.append({**attempt, **checked, "benchmark_id": identifier})
        return {"verification": result, "simulator_results": results, "milestone_checks": milestone_checks}

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
                  "critique": critique, "verification_advice": state.get("verification_advice"),
                  "milestone_checks": state.get("milestone_checks", []), "proof_gaps": gaps, "verification": state["verification"],
                  "simulator_results": state.get("simulator_results", []), "score": score,
                  "model_calls": self.model_calls, "simulator_calls": self.simulator_calls,
                  "elapsed_seconds": time.perf_counter() - self.started}
        self.ledger = advance_ledger(self.ledger, record, self.benchmark)
        record["ledger"] = self.ledger
        record["progress_evidence"] = iteration_evidence(record, self.records[-1] if self.records else None)
        write_json(self.log.path / "proof-ledger.json", self.ledger)
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
            row.update(mode=self.config.mode, backend=self.config.backend, model=self.config.role_model(row["role"]))
            self.agent_evaluations.append(row)
            self.log.agent_evaluation(row)

    def evaluate_unfinished_calls(self) -> None:
        completed = {record["iteration"] for record in self.records}
        for iteration in sorted({call["iteration"] for call in self.calls} - completed):
            calls = [call for call in self.calls if call["iteration"] == iteration]
            proposal = next((call["output"] for call in calls if call["stage"] in ("CA", "Repair") and call["output"]), {})
            critique = next((call["output"] for call in calls if call["stage"] == "SA"), None)
            gaps = [{**gap, "source": "skeptic", "assessment_status": "reported", "source_iteration": iteration}
                    for gap in (critique or {}).get("gaps", [])]
            self.evaluate_record({"run_id": self.log.run_id, "iteration": iteration,
                                  "proposal": proposal, "critique": critique, "proof_gaps": gaps,
                                  "verification": {"status": "not_recorded", "summary": "The iteration stopped before a verification result was recorded."},
                                  "score": score_iteration(proposal, {})})

    def repair(self, state: ResearchState) -> dict:
        iteration = state["iteration"] + 1
        feedback = {"verification": state["verification"], "simulator_results": state.get("simulator_results", []),
                    "verification_advice": state.get("verification_advice")}
        if self.config.mode == "multi":
            feedback["skeptic"] = state.get("critique")
            instruction = "Repair the proposal using the skeptic and tool feedback."
        else:
            instruction = "As the same single researcher, critique and revise your own preceding attempt using tool feedback."
        if self.verified_checkpoint is not None:
            instruction += (
                " A proof of the fixed target has already passed Lean; the verified checkpoint is preserved. "
                "Use this round to audit the prose-to-formal correspondence, improve clarity or simplify the proof, "
                "or develop a distinct supported proof. Preserve the target and hypotheses. Do not invent gaps "
                "or claim that rewriting an already proved result is new mathematical progress. If no useful "
                "improvement is found, retain the valid certificate and say so explicitly.")
        proposal = self.call("Repair", Proposal, instruction +
            " Refer to prior gap_id values for claimed repairs. Explain each change, keep the "
            "original theorem target/assumptions, and explicitly list unresolved gaps. "
            "Use the actual Lean compiler diagnostics; a failed check does not refute the theorem. "
            "If the same certificate or blocker has recurred, change proof strategy rather than "
            "paraphrasing the explanation or repeating an unavailable lemma. You may submit separately "
            "checked supporting milestones from the packet while the main proof remains incomplete.",
            {"benchmark": self.benchmark, "proposal": state["proposal"], "feedback": feedback,
             "previous_proof_gaps": [gap for record in self.records for gap in record["proof_gaps"]],
             "proof_ledger": self.ledger, "verified_checkpoint": self.verified_checkpoint}, iteration)
        return {"iteration": iteration, "proposal": proposal, "critique": None}

    def route(self, state: ResearchState) -> str:
        if self.config.stop_on_verified and is_verified_receipt(state["proposal"], state["verification"]):
            return END
        # Each multi-agent revision reserves one call for its skeptic. Every
        # repaired attempt is verified and recorded before the graph can stop.
        if state["iteration"] >= self.config.max_iterations:
            return END
        needed = (2 if self.config.mode == "multi" else 1) + bool(self.config.verifier_model)
        return "Repair" if self.model_calls + needed <= self.config.max_model_calls else END

    def graph(self):
        builder = StateGraph(ResearchState)
        for name, handler in (("CA", self.ca), ("SA", self.sa), ("VerifierAgent", self.verifier_agent), ("Verification", self.verification),
                              ("Record", self.record), ("Repair", self.repair)):
            builder.add_node(name, handler)
        builder.add_edge(START, "CA")
        target = "SA" if self.config.mode == "multi" else "VerifierAgent"
        builder.add_edge("CA", target)
        builder.add_edge("SA", "VerifierAgent")
        builder.add_edge("VerifierAgent", "Verification")
        builder.add_edge("Verification", "Record")
        builder.add_conditional_edges("Record", self.route, {"Repair": "Repair", END: END})
        builder.add_edge("Repair", target)
        return builder.compile()


def run_workflow(config: RunConfig, output_root: Path, backend: Any = None) -> tuple[Path, dict]:
    log = RunLog(Path(output_root), config.mode)
    run = ResearchRun(config, log, backend)
    manifest = {"schema_version": 2, "run_id": log.run_id, "created_at": utc_now(),
                "agent_evaluation_rubric": RUBRIC,
                **asdict(config), "max_model_calls": config.max_model_calls, "max_iterations": config.max_iterations,
                "benchmark": run.benchmark, "benchmark_sha256": digest(run.benchmark),
                "environment": _environment(),
                "budget_note": "Equal iteration limits" if config.budget_policy == "iterations" else "Equal model-call caps; actual tokens may differ",
                "seed_policy": "Seed labels paired trials; it is not sent to models and does not guarantee deterministic sampling.",
                "single_agent_policy": "Single researcher revises with Lean feedback, no separate skeptic.",
                "demo_warning": "Scripted demo is plumbing validation, not research evaluation." if config.backend == "demo" else None}
    write_json(log.path / "manifest.json", manifest)
    log.event("run_started", config=asdict(config), benchmark_sha256=manifest["benchmark_sha256"])
    status, stop_reason = "completed", "iteration_limit" if config.budget_policy == "iterations" else "model_call_budget_exhausted"
    error = None
    try:
        if run.benchmark["verification_contract"].get("status") == "ready" and not lean_status()["available"]:
            raise RuntimeError("Lean is required before starting this formal run: " + str(lean_status()["reason"]))
        if run.backend is None:
            run.backend = DemoBackend() if config.backend == "demo" else None
        run.graph().invoke({}, {"recursion_limit": config.max_model_calls * 6 + 10})
        if config.stop_on_verified and run.records and run.records[-1]["verification"].get("status") == "verified":
            stop_reason = "formal_target_verified"
        elif len(run.records) >= config.max_iterations:
            stop_reason = "iteration_limit"
    except Exception as exc:
        status, stop_reason = "failed", "stage_error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        log.event("run_error", **error)
    run.evaluate_unfinished_calls()
    summary = {"schema_version": 2, "run_id": log.run_id, "mode": config.mode,
               "backend": config.backend, "model": config.model, "pair_id": config.pair_id,
               "role_models": {role: config.role_model(role) for role in ("CA", "SA", "VA")},
               "verifier_enabled": bool(config.verifier_model), "budget_policy": config.budget_policy, "rounds": config.rounds,
               "reasoning_effort": config.reasoning_effort,
               "stop_on_verified": config.stop_on_verified, "max_iterations": config.max_iterations,
               "has_verified_proof": run.verified_checkpoint is not None,
               "verified_iteration": run.verified_checkpoint["iteration"] if run.verified_checkpoint else None,
               "seed": config.seed, "max_model_calls": config.max_model_calls,
               "max_output_tokens": config.max_output_tokens, "max_simulator_calls": config.max_simulator_calls,
               "benchmark_id": config.benchmark_id, "benchmark_sha256": manifest["benchmark_sha256"],
               "status": status, "stop_reason": stop_reason, "error": error,
               "records": run.records, "proof_ledger": run.ledger, "trajectory": trajectory_metrics(run.records),
               "model_calls": run.model_calls, "simulator_calls": run.simulator_calls,
               "agent_evaluations": run.agent_evaluations, "agent_metrics": summarize_agents(run.agent_evaluations),
               "usage": run.usage, "elapsed_seconds": time.perf_counter()-run.started,
               "final_score": run.records[-1]["score"]["score"] if run.records else 0,
               "max_score": max((r["score"]["score"] for r in run.records), default=0)}
    write_json(log.path / "summary.json", summary)
    (log.path / "report.md").write_text(render_report(manifest, run.records, summary), encoding="utf-8")
    log.event("run_finished", status=status, stop_reason=stop_reason, error=error)
    return log.path, summary
