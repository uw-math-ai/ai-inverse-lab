"""Study-centered, action-driven V3 runtime over V2's isolated execution foundation."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from . import HARNESS_ROOT
from . import artifacts as A
from . import methods_v2 as M2
from . import metrics as M
from .knowledge_v3 import (EvidenceRouter, ExposureRegistry, MethodPackageStore, safe_projection,
                           validate_evaluation_protocol)
from .research.contracts_v3 import build_problem_contract, evaluate_claim, method_contract
from .research.program_v3 import DelegationManager, RoleRouter
from .research.studies_v3 import normalize_study, paired_analysis, study_completion
from .resources import BudgetReservations
from .scheduler_v2 import ResearchSchedulerV2


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


class ResearchSchedulerV3(ResearchSchedulerV2):
    """The injectable decision function receives only the safe development program view.

    ``decision_fn(view) -> ProgramAction`` can script software tests or host an external
    reasoning policy. Otherwise role backends are asked for structured actions, with a bounded
    evidence-driven offline policy on unavailable/malformed responses.
    """
    def __init__(self, spec, run_dir, *, decision_fn=None, role_llms=None, **kwargs):
        if spec.pipeline != "v3":
            raise ValueError("ResearchSchedulerV3 requires pipeline: v3")
        super().__init__(spec, run_dir, **kwargs)
        self.cfg = spec.program
        self.decision_fn = decision_fn
        self.router = RoleRouter(self.llm, self.cfg.get("roles"), role_llms,
                                 resources=self.resources, emit=self._emit)
        self.delegation = DelegationManager(self.cfg, emit=self._emit)
        lead = self.delegation.spawn("orchestrator", question=spec.problem)
        critic = self.delegation.spawn("critic", question="independent design and synthesis review")
        self.lead_id, self.critic_id = lead["id"], critic["id"]
        self.program = {"schema_version": 3, "objective": spec.problem, "questions": [], "studies": {},
                        "actions": [], "evidence": {}, "objections": [], "messages": [], "stop_reason": None}
        self.arm_results = {}
        self.arm_ideas = {}
        self.executions = {}
        self.packages = []
        self.confirmation = float(self.cfg.get("confirmation_seconds") or
                                  spec.budget["compute_seconds"]*self.cfg["confirmation_fraction"])
        self.reservations = BudgetReservations(spec.budget["compute_seconds"], self.confirmation,
                                               hard=self.cfg["budget_mode"] == "hard")
        self.exposures = ExposureRegistry(Path(spec.evaluation.get("exposure_registry") or
                                              HARNESS_ROOT / ".cache" / "v3_exposure.sqlite"))
        self.package_store = MethodPackageStore(HARNESS_ROOT / ".research_memory" / "v3_packages")

    def _emit(self, kind, data=None, **kw):
        return self.store.put(kind, data, stage=self.stage, **kw)

    def run(self):
        result_path = self.run_dir / "v3_result.json"
        if self.resume and (self.run_dir / "spec.json").exists():
            persisted_spec = json.loads((self.run_dir / "spec.json").read_text(encoding="utf-8"))
            if _hash(persisted_spec) != _hash(self.spec.to_dict()):
                raise ValueError("V3 resume requires the identical compiled specification")
        if self.resume and result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        started = time.time()
        error = None
        try:
            self.stage_spec()
            self.stage_data()
            with np.load(self.data_path) as z:
                self.row_hash = _array_hash(np.arange(len(z["y_val"])))
                self.split_hash = _array_hash(z["X_test"])
                self.contract = build_problem_contract(self.spec, z["X_train"], z["y_train"],
                                                       {"train": _array_hash(z["X_train"]),
                                                        "val": _array_hash(z["X_val"]), "test": self.split_hash})
            self.contract_art = self._emit(A.DATASET, {"type": "problem_contract_v3", "contract": self.contract},
                                           author="scheduler", summary="canonical scientific contract")
            self.evidence_router = EvidenceRouter(self.contract, store=self.store, memory=self.memory,
                                                   literature=self.lit_client,
                                                   diagnostic_fn=lambda r: self._diagnose(r),
                                                   memory_mode=self.spec.memory.get("mode", "cumulative"),
                                                   package_store=self.package_store, package_gate=self.gate)
            protocol = dict(self.spec.evaluation.get("protocol") or {})
            protocol.setdefault("dataset_seed", self.spec.data["seed"])
            protocol.setdefault("training_seed", self.spec.seed)
            protocol.setdefault("problem_instance_id", _hash(self.spec.simulator))
            protocol.setdefault("dataset_id", self.data_key)
            protocol.setdefault("memory_mode", "cumulative" if self.spec.memory.get("write", True) else "frozen")
            protocol.setdefault("instances", [{"instance_id": self.data_key,
                "problem_id": _hash(self.spec.simulator), "dataset_seed": int(self.spec.data["seed"]),
                "training_seeds": [int(self.spec.seed)], "search_seeds": [int(self.spec.seed)]}])
            protocol.setdefault("memory", {"mode": self.spec.memory.get("mode", "cumulative"),
                "snapshot_id": self.evidence_router.snapshot_id})
            self.protocol = validate_evaluation_protocol(protocol)
            if not self.protocol.get("valid", False):
                raise ValueError(f"invalid evaluation protocol: {self.protocol.get('errors')}")
            self.research = {"proposals": {}, "active": [], "rounds": [], "stop_reason": None}
            self._restore()
            declared_memory = (self.spec.evaluation.get("protocol") or {}).get("memory") or {}
            if declared_memory.get("mode", self.evidence_router.memory_mode) != self.evidence_router.memory_mode:
                raise ValueError("evaluation protocol memory mode disagrees with the runtime memory policy")
            if self.evidence_router.memory_mode == "frozen":
                if declared_memory.get("snapshot_id", self.evidence_router.snapshot_id) != self.evidence_router.snapshot_id:
                    raise ValueError("declared frozen-memory snapshot does not match the safe runtime snapshot")
                self.protocol["normalized"]["memory"]["snapshot_id"] = self.evidence_router.snapshot_id
            self.enter("program")
            for _ in range(int(self.cfg["max_actions"])-len(self.program["actions"])):
                if self.program["stop_reason"]:
                    break
                if self.resources.wall_remaining() <= 0:
                    self.program["stop_reason"] = "wall budget exhausted"
                    break
                action = self._next_action()
                self._perform(action)
            if not self.program["stop_reason"]:
                self.program["stop_reason"] = "action limit reached"
            self._critic_review(None, synthesis=True)
            self.research["stop_reason"] = self.program["stop_reason"]
            self._save_state()
            self.final = self._finalize()
            self._write_knowledge()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._emit(A.FAILURE, {"source": "v3_scheduler", "error": error}, author="scheduler",
                       summary=error)
        finally:
            self.enter("report")
            self._write_bundle(error, time.time()-started)
            self.stage = "failed" if error else "done"
            self.status(error=error, pipeline="v3", elapsed=time.time()-started,
                        research_stop=self.program["stop_reason"])
            if self.remote is not None:
                self.remote.close()
        if error:
            raise RuntimeError(error)
        out = {"run_dir": str(self.run_dir), "final": self.final, "resources": self.resources.snapshot(),
               "research": {"stop_reason": self.program["stop_reason"]},
               "evidence_bundle": str(self.run_dir / "evidence_bundle.json")}
        self._write_json(result_path, out)
        return out

    @staticmethod
    def _write_json(path, value):
        pending = Path(path).with_suffix(".pending")
        pending.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
        pending.replace(path)

    def _save_state(self):
        self._emit(A.DECISION, {"type": "v3_program_snapshot", "program": self.program,
                   "agents": self.delegation.agents, "tasks": self.delegation.tasks,
                   "arm_results": self.arm_results, "arm_ideas": self.arm_ideas,
                   "ideas": self.ideas, "executions": self.executions,
                   "resources": self.resources.snapshot(), "reservations": self.reservations.snapshot(),
                   "run_id": self.run_id,
                   "role_events": self.router.events,
                   "memory_snapshot": self.evidence_router.snapshot_state()
                   if self.evidence_router.memory_mode == "frozen" else None},
                   author=self.lead_id, summary="persisted V3 program state")

    def _restore(self):
        if not self.resume:
            return
        prior_exp_ids = [a.data.get("exp_id") for a in self.store.query(A.PLAN) if a.data.get("exp_id")]
        self.engineer._ids = itertools.count(1+max([int(x.rsplit("-", 1)[1]) for x in prior_exp_ids] or [0]))
        snapshots = self.store.query(A.DECISION, where=lambda a: a.data.get("type") == "v3_program_snapshot")
        if not snapshots:
            return
        d = snapshots[-1].data
        self.run_id = d.get("run_id", self.run_id)
        if d.get("memory_snapshot"):
            self.evidence_router.restore_snapshot(d["memory_snapshot"])
            self.protocol["normalized"].setdefault("memory", {})["snapshot_id"] = self.evidence_router.snapshot_id
        self.program = d["program"]
        self.delegation.agents = d["agents"]
        self.delegation.tasks = d["tasks"]
        self.router.events = d.get("role_events", [])
        self.delegation._counter = max(int(x.rsplit("-", 1)[1]) for x in d["agents"])
        self.arm_results, self.arm_ideas, self.ideas, self.executions = (
            d["arm_results"], d["arm_ideas"], d["ideas"], d["executions"])
        self._ids = itertools.count(1+max([int(x.rsplit("-", 1)[1]) for x in self.ideas] or [0]))
        for aid, iid in self.arm_ideas.items():
            r = self.arm_results.get(aid)
            if r and r.get("status") == "ok":
                self.results[iid] = {1.0: r}
                self.v2res[iid] = {"full": r}
        used = float(d["reservations"].get("used_seconds", 0))
        self.reservations.used = used
        self.resources.used = float(d["resources"].get("compute_used_s", used))
        self.resources.jobs = int(d["resources"].get("jobs", 0))
        self.resources.by_stage = dict(d["resources"].get("by_stage_s", {}))
        self.resources.start = time.time()-float(d["resources"].get("wall_elapsed_s", 0))
        for key in ("llm_calls", "llm_seconds", "llm_cost_usd", "llm_by_phase",
                    "literature_queries", "literature_failures"):
            if key in d["resources"]:
                setattr(self.resources, key, d["resources"][key])

    def _model_program(self):
        """Program state for model prompts. The persisted program, validators and scripted
        ``decision_fn`` views are unchanged; only duplicated history is shortened: action records
        reference their study instead of embedding a full copy (code included), and superseded
        studies become stubs that point at their successor."""
        def dedupe(value, code):
            # agent code is embedded in method, method_contract.requested/effective and gate.method
            if isinstance(value, dict):
                return {k: ("<same as arm.method.code>" if k == "code" and v == code else dedupe(v, code))
                        for k, v in value.items()}
            if isinstance(value, list):
                return [dedupe(v, code) for v in value]
            return value

        def live(s):
            s = dict(s)
            arms = []
            for arm in s.get("arms", []):
                code = (arm.get("method") or {}).get("code")
                if code:
                    arm = {k: (v if k == "method" else dedupe(v, code)) for k, v in arm.items()}
                arms.append(arm)
            s["arms"] = arms
            return s

        prog = dict(self.program)
        prog["studies"] = {sid: (live(s) if s.get("status") != "superseded" else
                                 {k: s.get(k) for k in ("id", "status", "superseded_by", "supersedes", "owner",
                                                        "question", "merged_from")
                                  if s.get(k) is not None} | {"note": "superseded; see successor"})
                           for sid, s in self.program["studies"].items()}
        actions = []
        for a in self.program["actions"]:
            a = dict(a)
            if isinstance(a.get("study"), dict):
                a["study"] = {"id": a["study"].get("id"), "note": "full study in program.studies"}
            actions.append(a)
        prog["actions"] = actions
        return prog

    def _view(self, model=False):
        return safe_projection({"contract": self.contract,
                                "program": self._model_program() if model else self.program,
                                "agents": self.delegation.agents, "tasks": self.delegation.tasks,
                                "resources": self.reservations.snapshot(),
                                "interface": "Return action {type,target,rationale,...}. Allowed types: decompose, spawn, "
                                             "assign, propose_study, execute, diagnose, retrieve, delegate, revise, merge, "
                                             "defer, resolve_objection, escalate, stop. Studies require explicit matched arms "
                                             "and typed claims; every claim names stable treatment/control IDs.",
                                "interface_guide": self._interface_guide()})

    def _interface_guide(self):
        """Exact action/study schema for model-driven roles.

        Mirrors the validators in ``_dispatch_action``/``normalize_study`` and the V2 method-spec
        normalizer, so live models can form valid actions instead of discovering the schema one
        rejection at a time. Static except for the method catalogue and the last rejection.
        """
        from .research.codegen import CONTRACT, RULES
        researchers = [a["id"] for a in self.delegation.agents.values() if a["role"] == "researcher"]
        rejected = [a for a in self.program["actions"] if a.get("status") == "rejected"]
        if self.program["questions"]:
            example = self._default_study(researchers[0] if researchers else "<researcher agent id>")
        else:      # _default_study reads the first registered question
            self.program["questions"].append({"id": "Q-example", "question": "<question text>"})
            try:
                example = self._default_study("<researcher agent id>")
            finally:
                self.program["questions"].pop()
        example = {"note": "valid example only; choose your own study/arm/claim ids and methods", **example}
        return {
            "order": ["decompose questions (once) -> spawn researcher(s) -> propose_study with owner = a spawned "
                      "researcher id -> execute the study -> diagnose/revise/propose follow-ups from measured "
                      "evidence -> stop. A study must be executed before it has evidence; final export uses the "
                      "best executed full-fidelity arm, so execute at least one study."],
            "researcher_ids": researchers,
            "actions": {
                "decompose": {"type": "decompose", "questions": [{"id": "Q-1", "question": "text", "priority": 1}]},
                "spawn": {"type": "spawn", "persona": "mathematical|ml|experimental|exploratory", "question": "Q-1"},
                "assign": {"type": "assign", "target": "<researcher id>", "question": "Q-2"},
                "propose_study": {"type": "propose_study", "owner": "<researcher id>", "study": "<study object>"},
                "execute": {"type": "execute", "target": "<study id>"},
                "diagnose": {"type": "diagnose", "study_id": "<study id>",
                             "kind": "paired|per_target|per_sample|prediction_reuse",
                             "treatment": "<arm id, optional>", "control": "<arm id, optional>"},
                "retrieve": {"type": "retrieve", "request": {"source_type": "local|artifact|diagnostic|memory|theorem",
                                                              "query": "text"}},
                "delegate": {"type": "delegate", "parent_id": "<researcher id>",
                             "kind": "coding_execution|analysis|retrieval|source",
                             "request": {"question": "text",
                                         "study": "coding_execution: optional complete study object to register and run "
                                                  "(otherwise a coding specialist writes one from the question)",
                                         "study_id": "analysis: REQUIRED executed study id",
                                         "kind": "analysis: paired|per_target|per_sample|prediction_reuse",
                                         "treatment": "analysis: arm id", "control": "analysis: arm id",
                                         "source_type": "retrieval/source: local|artifact|diagnostic|memory|theorem"}},
                "revise": {"type": "revise", "target": "<old study id>", "study": "<study object with a NEW id>"},
                "merge": {"type": "merge", "id": "<new study id>", "study_ids": ["<id>", "<id>"]},
                "defer": {"type": "defer", "target": "<study id>", "rationale": "text"},
                "resolve_objection": {"type": "resolve_objection", "target": "<objection id>",
                                      "resolution": "evidence|revision|acknowledged_uncertainty",
                                      "evidence_ids": ["<program.evidence key>"], "revision_id": "<study id>"},
                "escalate": {"type": "escalate", "target": "<agent id>"},
                "stop": {"type": "stop", "rationale": "text"}},
            "study_schema": {
                "id": "unique string, never reused (revisions need a new id)", "question": "nonempty text",
                "owner": "researcher agent id", "priority": "int",
                "arms": "list; each {id: unique string, role: treatment|control|baseline|diagnostic|exploratory, "
                        "required: bool, method: <method spec>, seed: int, fidelity: (0,1], intervention: {..}}",
                "matching": {"allowed_varying_factors": ["dotted method fields that differ between treatment and "
                                                         "control, e.g. method, params, params.alpha"]},
                "diagnostics": "list of {id, kind: paired|per_target|per_sample, treatment: arm id, control: arm id, "
                               "required: bool}",
                "claims": "list of {id, kind: comparison|performance, text, metric, treatment: arm id, control: arm id, "
                          "operator: at_least_better|at_least_worse|no_more_than_worse|equivalent, margin: >=0 "
                          "relative effect, allowed_varying_factors, requires_uncertainty: bool, analysis_id: "
                          "paired diagnostic id, required_diagnostics: [diagnostic ids]}",
                "dependencies": "optional list of prerequisite study ids"},
            "method_specs": {
                "library": {"kind": "library", "method": "<one of library_methods>", "params": {},
                            "features": "<one of feature_sets>", "scaler": "standard|none", "pca": "0 or n components",
                            "target_transform": "none|log", "noise_aug": "copies (int)", "noise_scale": "float"},
                "physics": {"kind": "physics", "params": dict(M2.PHYSICS.defaults),
                            "note": "forward-model MLE/MAP/posterior mean; only if contract.privileges.forward_authorized"},
                "hybrid": {"kind": "hybrid", "init": "<library spec>", "params": dict(M2.HYBRID.defaults),
                           "note": "learned initializer + Levenberg-Marquardt through the forward model; forward_authorized only"},
                "custom_ml": {"kind": "custom_ml", "code": "python source", "params": {}, "features": "raw",
                              "scaler": "standard", "pca": 0, "contract": CONTRACT["custom_ml"]},
                "custom_solver": {"kind": "custom_solver", "code": "python source", "params": {},
                                  "uses_forward": "bool (true requires forward_authorized)",
                                  "contract": CONTRACT["custom_solver"]},
                "random_search": {"kind": "random_search", "n": "1-8 random library configurations"},
                "code_rules": RULES},
            "library_methods": {k: dict(getattr(v, "defaults", {}) or {}) for k, v in self.available.items()},
            "feature_sets": list(self.feature_sets),
            "forward_authorized": bool(self.problem.get("forward_available")),
            "example_study": example,
            "last_rejection": ({"action": {k: rejected[-1].get(k) for k in ("id", "type")},
                                "error": rejected[-1].get("error")} if rejected else None)}

    def _next_action(self):
        if self.decision_fn is not None:
            result = self.decision_fn(copy.deepcopy(self._view()))
            if not isinstance(result, dict):
                raise ValueError("decision_fn must return an action dictionary")
            return result
        result = self.router.ask("orchestrator", self._view(model=True), actor=self.lead_id, purpose="choose program action")
        if result is not None and self._valid_action(result):
            return result
        if result is not None:
            self._emit(A.FAILURE, {"source": "program_policy", "fallback_reason": "invalid_schema"},
                       author=self.lead_id, summary="invalid program action; using offline policy")
        return self._offline_action()

    @staticmethod
    def _valid_action(action):
        return isinstance(action, dict) and action.get("type") in {
            "decompose", "spawn", "assign", "propose_study", "execute", "diagnose", "retrieve", "delegate",
            "revise", "merge", "defer", "resolve_objection", "escalate", "stop"}

    def _offline_action(self):
        p = self.program
        if not p["questions"]:
            return {"type": "decompose", "questions": self.cfg.get("questions") or
                    [{"id": "Q-1", "question": "Which validated method improves on frozen conventional controls?",
                      "priority": 1}], "rationale": "identify an answerable empirical question"}
        people = [a for a in self.delegation.agents.values() if a["role"] == "researcher"]
        target_count = min(2, max(1, self.cfg["max_agents"]-3), max(1, self.cfg["max_active_agents"]-3))
        if len(people) < target_count:
            roles = self.spec.research.get("researchers") or ["experimental"]
            return {"type": "spawn", "persona": roles[len(people)%len(roles)],
                    "question": p["questions"][0]["id"], "rationale": "assign an independent research perspective"}
        if not p["studies"]:
            studies = self.cfg.get("studies") or [self._default_study(people[0]["id"])]
            return {"type": "propose_study", "study": studies[0], "owner": people[0]["id"],
                    "rationale": "register required comparisons before execution"}
        for study in sorted(p["studies"].values(), key=lambda s: s.get("priority", 1)):
            if study.get("status") in ("planned", "running"):
                return {"type": "execute", "target": study["id"], "rationale": "obtain full registered evidence"}
        configured = self.cfg.get("studies") or []
        for s in configured:
            if s["id"] not in p["studies"]:
                return {"type": "propose_study", "study": s, "owner": people[len(p["studies"])%len(people)]["id"],
                        "rationale": "continue the declared research program after observed results"}
        for s in p["studies"].values():
            if s.get("status") == "incomplete" and not s.get("defer_reason"):
                return {"type": "defer", "target": s["id"], "rationale": "required evidence missing; preserve uncertainty"}
        # A contradicted result changes the next scientific action: analyze the matched error
        # distribution and stop the original line, rather than repeat the same debate.
        for s in p["studies"].values():
            if any(v["outcome"] == "contradicted" for v in s.get("verdicts", [])) and not s.get("followup"):
                if self.cfg.get("auto_followup", True):
                    return {"type": "diagnose", "study_id": s["id"], "kind": "per_sample",
                            "rationale": "contradicted registered margin warrants a targeted error analysis"}
        return {"type": "stop", "rationale": "registered work resolved or explicitly deferred; no justified next study"}

    def _default_study(self, owner):
        control_key = "ridge" if "ridge" in self.available else next(iter(self.available))
        treatment_key = "extra_trees" if "extra_trees" in self.available else control_key
        def method(key):
            return {"kind": "library", "method": key, "params": dict(self.available[key].defaults),
                    "features": "raw", "scaler": "standard", "pca": 0}
        arms = [{"id": "S-1.control", "role": "baseline", "required": True, "method": method(control_key),
                 "seed": self.spec.seed, "fidelity": 1.0, "intervention": {"estimator": control_key}},
                {"id": "S-1.treatment", "role": "treatment", "required": True, "method": method(treatment_key),
                 "seed": self.spec.seed, "fidelity": 1.0, "intervention": {"estimator": treatment_key}}]
        if self.problem.get("forward_available"):
            arms.append({"id": "S-1.physics", "role": "baseline", "required": True,
                         "method": {"kind": "physics", "params": dict(M2.PHYSICS.defaults)},
                         "seed": self.spec.seed, "fidelity": 1.0, "intervention": {"estimator": "physics"}})
        return {"id": "S-1", "question": self.program["questions"][0]["question"], "owner": owner,
                "priority": 1, "arms": arms, "matching": {"allowed_varying_factors": ["method", "params"]},
                "diagnostics": [{"id": "S-1.paired", "kind": "paired", "treatment": "S-1.treatment",
                                 "control": "S-1.control", "required": True}],
                "claims": [{"id": "C-1", "kind": "comparison", "text": "The treatment improves validation error",
                            "metric": self.spec.primary, "treatment": "S-1.treatment", "control": "S-1.control",
                            "operator": "at_least_better", "margin": 0.01,
                            "allowed_varying_factors": ["method", "params"],
                            "requires_implementation_verification": True, "required_diagnostics": ["S-1.paired"]}]}

    def _perform(self, action):
        if not self._valid_action(action):
            raise ValueError("unknown program action")
        action = copy.deepcopy(action)
        rec = {"id": f"action-{len(self.program['actions'])+1:03d}", **action, "status": "running"}
        self.program["actions"].append(rec)
        before = self._emit(A.DECISION, rec, author=self.lead_id, summary=f"before {rec['id']}: {action['type']}")
        try:
            result = self._dispatch_action(action)
            rec.update(status="complete", result=result)
        except (ValueError, KeyError, TypeError) as exc:
            rec.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            self._emit(A.FAILURE, rec, author=self.lead_id, parents=[before.id], summary=f"action rejected: {exc}")
        self._emit(A.DECISION, rec, author=self.lead_id, parents=[before.id], summary=f"after {rec['id']}: {rec['status']}")
        self._save_state()

    def _dispatch_action(self, a):
        kind = a["type"]
        target = a.get("target")
        if kind == "decompose":
            questions = a.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError("decomposition requires questions")
            ids = {q["id"] for q in self.program["questions"]}
            for q in questions:
                if not isinstance(q, dict) or not q.get("id") or not q.get("question") or q["id"] in ids:
                    raise ValueError("questions require unique IDs and text")
                ids.add(q["id"])
            self.program["questions"].extend(copy.deepcopy(questions))
            return {"questions": len(questions)}
        if kind == "spawn":
            persona = a.get("persona", "experimental")
            if persona not in ("mathematical", "ml", "experimental", "exploratory"):
                raise ValueError("unknown researcher persona")
            person = self.delegation.spawn("researcher", persona=persona, parent_id=self.lead_id,
                                            question=a.get("question"))
            self._researcher_turn(person)
            return {"agent_id": person["id"]}
        if kind == "assign":
            person = self.delegation.agents[target]
            if person["role"] != "researcher":
                raise ValueError("only researcher tasks can be reassigned")
            person["question"] = a["question"]
            self._researcher_turn(person)
            return {"agent_id": target}
        if kind == "propose_study":
            return self._register_study(a["study"], a.get("owner"))
        if kind == "execute":
            return self._execute_study(self.program["studies"][target])
        if kind == "diagnose":
            result = self._diagnose(a)
            if a.get("study_id") in self.program["studies"]:
                self.program["studies"][a["study_id"]]["followup"] = result
            return result
        if kind == "retrieve":
            result = self.evidence_router.resolve(a.get("request", a), self._view())
            self._record_evidence(result, self.lead_id)
            return result
        if kind == "delegate":
            return self._delegate(a["parent_id"], a["kind"], a.get("request") or {})
        if kind == "escalate":
            actor = self.delegation.agents[target]
            return self.router.ask(actor["role"], self._view(model=True), actor=target, purpose="escalated scientific decision",
                                   escalate=True) or {"status": "unavailable"}
        if kind == "revise":
            old = self.program["studies"][target]
            revised = copy.deepcopy(a["study"])
            if revised["id"] == old["id"]:
                raise ValueError("revision must create an immutable new study ID")
            old.update(status="superseded", superseded_by=revised["id"])
            revised["supersedes"] = old["id"]
            for claim in revised.get("claims", []):
                claim["study_id"] = revised["id"]
            return self._register_study(revised, old.get("owner"))
        if kind == "merge":
            ids = a["study_ids"]
            sources = [self.program["studies"][x] for x in ids]
            merged = {"id": a["id"], "question": a.get("question", "Combined registered studies"),
                      "owner": a.get("owner", sources[0].get("owner")), "arms": [], "claims": [], "diagnostics": [],
                      "merged_from": ids, "dependencies": []}
            for key in ("arms", "claims", "diagnostics"):
                seen = {}
                for s in sources:
                    for item in s.get(key, []):
                        if item["id"] in seen and seen[item["id"]] != item:
                            raise ValueError(f"conflicting merged {key} ID")
                        seen[item["id"]] = copy.deepcopy(item)
                merged[key] = list(seen.values())
            for claim in merged["claims"]:
                claim["study_id"] = merged["id"]
            result = self._register_study(merged, merged["owner"])
            for s in sources:
                s.update(status="superseded", superseded_by=merged["id"])
            return result
        if kind == "defer":
            s = self.program["studies"][target]
            s.update(status="deferred", defer_reason=a.get("rationale", "explicit program deferral"))
            return {"study_id": target, "status": "deferred"}
        if kind == "resolve_objection":
            objection = next(o for o in self.program["objections"] if o["id"] == target)
            resolution = a.get("resolution")
            if resolution not in ("evidence", "revision", "acknowledged_uncertainty"):
                raise ValueError("resolution needs evidence, revision or acknowledged_uncertainty")
            evidence_ids = a.get("evidence_ids", [])
            if resolution == "evidence":
                if not evidence_ids or not all(x in self.program["evidence"] for x in evidence_ids):
                    raise ValueError("objection evidence must reference measured program evidence")
                s = self.program["studies"].get(objection.get("scope"))
                if not s or s.get("status") != "complete":
                    raise ValueError("missing study work cannot be resolved by assertion")
                for eid in evidence_ids:
                    evidence = self.program["evidence"][eid]
                    if evidence.get("study_id") != s["id"]:
                        raise ValueError("objection evidence must belong to the challenged study")
                review = self.router.ask("critic", {"objection": objection,
                    "evidence": {eid: self.program["evidence"][eid] for eid in evidence_ids},
                    "interface": "Return {requirement_satisfied: boolean, reason: text}. "
                                 "Approve only if the cited evidence resolves this exact objection."},
                    actor=self.critic_id, purpose="review objection resolution")
                if not review or review.get("requirement_satisfied") is not True:
                    raise ValueError("independent critic has not accepted this requirement resolution")
            if resolution == "revision":
                revised = self.program["studies"].get(a.get("revision_id"))
                if not revised or not (revised.get("supersedes") == objection["scope"] or
                                       objection["scope"] in revised.get("merged_from", [])):
                    raise ValueError("revision must reference a successor of the challenged study")
            objection.update(status="acknowledged" if resolution == "acknowledged_uncertainty" else "resolved",
                             resolution=resolution, evidence_ids=evidence_ids, responsible_actor=self.lead_id)
            if resolution != "acknowledged_uncertainty":
                for study in self.program["studies"].values():
                    if objection["scope"] in ("program", study["id"]) and study.get("verdicts"):
                        self._reconcile_study(study)
            return objection
        if kind == "stop":
            self.program["stop_reason"] = a.get("rationale") or "orchestrator stopped"
            return {"stop_reason": self.program["stop_reason"]}
        raise ValueError("unsupported action")

    def _researcher_turn(self, person):
        if person["turns"] >= self.cfg["max_agent_turns"]:
            return
        person["turns"] += 1
        response = self.router.ask("researcher", {"task": person, "evidence": self._view(model=True),
            "interface": "Return {study?: registered study, specialist_requests?: [{kind,request}], interpretation?: text}. "
                         "Own your question and commission bounded technical work when needed."},
            actor=person["id"], purpose="researcher investigation")
        if not response:
            return
        for request in response.get("specialist_requests", []):
            if not isinstance(request, dict):
                continue
            self._delegate(person["id"], request.get("kind"), request.get("request") or {})
        if isinstance(response.get("study"), dict):
            self._register_study(response["study"], person["id"])
        if response.get("interpretation"):
            self.program["messages"].append({"author": person["id"], "question": person["question"],
                "text": str(response["interpretation"]), "status": "researcher_interpretation_unverified",
                "evidence_ids": list(person["evidence_ids"])})

    def _delegate(self, parent_id, kind, request):
        def callback(agent, payload):
            if kind == "coding_execution":
                # Code generation is reasoning only. Every returned method still goes through
                # the same deterministic gate and isolated executor as human methods.
                study = payload.get("study")
                if not study:
                    guide = self._interface_guide()
                    response = self.router.ask("specialist", {"request": payload, "contract": self.contract,
                        "interface": "Return {study: <study object>} with executable custom code or library methods. "
                                     "The study must follow interface_guide.study_schema (unique id, question, arms "
                                     "with method specs, claims naming treatment/control arm ids).",
                        "interface_guide": {k: guide[k] for k in ("study_schema", "method_specs", "library_methods",
                                                                   "feature_sets", "forward_authorized", "example_study")},
                        "existing_study_ids": sorted(self.program["studies"])},
                        actor=agent["id"], purpose="implement diagnostic/study")
                    study = (response or {}).get("study")
                if not isinstance(study, dict):
                    return {"status": "unsupported", "uncertainty": ["no executable study returned"]}
                if not study.get("id"):
                    # an id is a stable label only; derive one from the commissioning task
                    study = dict(study, id=f"S-{agent['id']}")
                    for claim in study.get("claims", []) if isinstance(study.get("claims"), list) else []:
                        if isinstance(claim, dict) and claim.get("study_id") in (None, ""):
                            claim.pop("study_id", None)
                self._register_study(study, parent_id)
                return self._execute_study(self.program["studies"][study["id"]], parent_task=agent["id"])
            if kind == "analysis":
                return self._diagnose(payload)
            result = self.evidence_router.resolve(payload, {"contract": self.contract})
            self._record_evidence(result, agent["id"])
            return result
        task = self.delegation.dispatch(parent_id, kind, request, callback)
        parent = self.delegation.agents[parent_id]
        if task.get("artifact_id"):
            parent["evidence_ids"].append(task["artifact_id"])
        interpretation = self.router.ask("researcher", {"task_result": safe_projection(task),
            "question": parent["question"], "interface": "Return {interpretation: text, evidence_ids: [...]}"},
            actor=parent_id, purpose="interpret specialist findings")
        self.program["messages"].append({"author": parent_id, "task_id": task["id"],
            "interpretation": interpretation or {"status": "uninterpreted", "reason": "no researcher model response"},
            "evidence_ids": [task.get("artifact_id")]})
        return task

    def _register_study(self, raw, owner=None):
        if len(self.program["studies"]) >= self.cfg["max_studies"]:
            raise ValueError("study limit")
        raw = copy.deepcopy(raw)
        # A random-search arm owns all concrete candidates. Keep its identity and obligations
        # instead of dropping the search policy at the implementation gate.
        children = []
        for arm in raw.get("arms", []):
            if arm.get("method", {}).get("kind") != "random_search":
                continue
            count = int(arm["method"].get("n", 2))
            if not 1 <= count <= int(self.cfg.get("max_search_candidates", 8)):
                raise ValueError("random_search candidate limit exceeded")
            specs = M2.random_library_specs(count, self.available, self.feature_sets, self.n_features,
                    np.random.default_rng(int(arm.get("seed", self.spec.seed))+len(self.program["studies"])*1009))
            arm["search_candidate_ids"] = []
            for index, ms in enumerate(specs):
                cid = f"{arm['id']}.candidate-{index+1}"
                arm["search_candidate_ids"].append(cid)
                children.append({"id": cid, "role": "exploratory", "required": arm.get("required", True),
                                 "method": ms, "seed": arm.get("seed", self.spec.seed),
                                 "fidelity": arm.get("fidelity", 1.0), "search_parent": arm["id"],
                                 "intervention": {"search_candidate": index+1}})
            for claim in raw.get("claims", []):
                if arm["id"] in (claim.get("treatment"), claim.get("control")):
                    # Selecting a winner on these validation rows is exploratory. Confirming
                    # effects requires a newly frozen arm on separately declared evidence.
                    claim.setdefault("required_evidence", []).append(f"confirmation:{arm['id']}")
        raw.setdefault("arms", []).extend(children)
        study = normalize_study(raw)
        if study["id"] in self.program["studies"]:
            raise ValueError("study IDs immutable; revise with a new ID")
        owner = owner or study.get("owner")
        if owner not in self.delegation.agents or self.delegation.agents[owner]["role"] != "researcher":
            raise ValueError("study requires a registered researcher owner")
        existing_arms = {a["id"]: a for s in self.program["studies"].values() for a in s["arms"]}
        for arm in study["arms"]:
            arm.setdefault("seed", self.spec.seed)
            arm.setdefault("fidelity", 1.0)
            identity_fields = ("method", "seed", "fidelity", "role", "required", "intervention",
                               "shared_fit_id", "shared_trajectory_id", "depends_on")
            if arm["id"] in existing_arms and any(arm.get(k) != existing_arms[arm["id"]].get(k) for k in identity_fields):
                raise ValueError("an arm ID cannot change scientific or execution identity")
        upstream = [x for x in study.get("dependencies", []) if isinstance(x, str)] \
            if isinstance(study.get("dependencies"), list) else []
        if study["id"] in upstream:
            raise ValueError("cyclic study dependency")
        def reaches(current, target, seen=None):
            seen = set() if seen is None else seen
            if current == target:
                return True
            if current in seen:
                return False
            seen.add(current)
            return any(reaches(x, target, seen) for x in self.program["studies"].get(current, {}).get("dependencies", [])
                       if isinstance(x, str))
        if any(reaches(dep, study["id"]) for dep in upstream):
            raise ValueError("cyclic study dependency")
        study.update(owner=owner, status="planned", registered_at=time.time(), verdicts=[])
        art = self._emit(A.PROPOSAL, {"type": "study_v3", "study": study}, author=owner,
                         summary=f"registered {study['id']}: {study['question']}")
        study["artifact_id"] = art.id
        self.program["studies"][study["id"]] = study
        self._critic_review(study)
        return {"study_id": study["id"], "artifact_id": art.id, "status": "planned"}

    def _critic_review(self, study, synthesis=False):
        scope = study["id"] if study else "program"
        response = self.router.ask("critic", {"scope": scope, "study": safe_projection(study),
            "contract": self.contract, "program": self._view(model=True) if synthesis else None,
            "interface": "Return {objections:[{requirement,blocking:boolean}]}. Independently challenge design, "
                         "matched controls, measurements, implementation, missing evidence and synthesis claims."},
            actor=self.critic_id, purpose="independent synthesis review" if synthesis else "independent study review")
        objections = (response or {}).get("objections", [])
        if study:
            for claim in study.get("claims", []):
                if claim.get("kind") in ("comparison", "mechanism") and not claim.get("control"):
                    objections.append({"requirement": f"claim {claim['id']} lacks a frozen control", "blocking": True})
                if claim.get("kind") in ("mechanism", "mathematical"):
                    objections.append({"requirement": f"claim {claim['id']} requires direct mechanism/proof evidence",
                                       "blocking": True})
        elif synthesis:
            for s in self.program["studies"].values():
                if s.get("status") != "complete":
                    objections.append({"requirement": f"{s['id']} remains {s.get('status')}", "blocking": True,
                                       "scope": s["id"]})
        for raw in objections:
            if not isinstance(raw, dict) or not raw.get("requirement"):
                continue
            objection_scope = raw.get("scope", scope)
            if objection_scope != "program" and objection_scope not in self.program["studies"]:
                continue
            if any(o["scope"] == objection_scope and o["requirement"] == raw["requirement"] for o in self.program["objections"]):
                continue
            objection = {"id": f"objection-{len(self.program['objections'])+1:03d}", "scope": objection_scope,
                         "requirement": str(raw["requirement"]), "blocking": bool(raw.get("blocking", True)),
                         "status": "open", "author": self.critic_id}
            art = self._emit(A.CRITIQUE, objection, author=self.critic_id, summary=objection["requirement"])
            objection["artifact_id"] = art.id
            self.program["objections"].append(objection)
            if objection["blocking"]:
                for affected in self.program["studies"].values():
                    if objection_scope not in ("program", affected["id"]):
                        continue
                    for verdict in affected.get("verdicts", []):
                        if verdict["outcome"] == "supported":
                            verdict.update(outcome="inconclusive", reason="independent blocking objection unresolved")
                            verdict["unmet_requirements"].append(objection["id"])

    def _execute_study(self, study, parent_task=None):
        sid = study["id"]
        for dep in study.get("dependencies", []) if isinstance(study.get("dependencies"), list) else []:
            if not isinstance(dep, str):
                continue
            depid = dep
            if self.program["studies"].get(depid, {}).get("status") != "complete":
                study.update(status="deferred", reason=f"dependency {depid} incomplete")
                return {"status": "deferred", "reason": study["reason"]}
        for arm in study["arms"]:
            if arm.get("shared_fit_id") or arm.get("shared_trajectory_id"):
                study.update(status="deferred", reason="shared fitted-state adapter unavailable; use prediction_reuse")
                return {"status": "deferred", "reason": study["reason"]}
        sharing = study.get("shared_computation") or {}
        if sharing and sharing.get("protocol") != "persisted_prediction_analysis":
            study.update(status="deferred", reason="unsupported shared computation protocol")
            return {"status": "deferred", "reason": study["reason"]}
        admitted = []
        required_failure = []
        for arm in study["arms"]:
            if arm.get("search_candidate_ids"):
                continue
            if arm["method"].get("kind") == "method_package":
                loaded = self.package_store.load(arm["method"]["package_id"], gate=self.gate)
                arm["package_provenance"] = {k: v for k, v in loaded.items() if k != "method"}
                arm["requested_package"] = arm["method"]
                arm["method"] = loaded["method"]
            gate = self.gate.check(arm["method"], label=f"{sid}/{arm['id']}", parents=[study["artifact_id"]])
            arm["method_contract"] = method_contract(arm["method"], gate["method"], arm["method"].get("code"))
            arm["gate"] = gate
            if not gate["approved"]:
                self.arm_results[arm["id"]] = {"status": "failed_implementation", "error": gate["reasons"]}
                if arm.get("required", True):
                    required_failure.append(arm["id"])
                continue
            admitted.append(arm)
        if required_failure:
            study.update(status="failed_implementation", reason=f"required arms rejected: {required_failure}")
            self._reconcile_study(study)
            study["status"] = "failed_implementation"
            study["reason"] = f"required arms rejected: {required_failure}"
            return {"status": "failed_implementation", "failed_arms": required_failure}
        pending, aliases, pending_keys = [], [], {}
        for arm in admitted:
            execution_key = _hash({"method": arm["method_contract"].get("effective_hash"),
                                   "effective": arm["gate"]["method"], "seed": arm["seed"],
                                   "fidelity": arm["fidelity"], "data": self.data_key,
                                   "diagnostics": study.get("diagnostics", [])})
            if execution_key in self.executions:
                prior = copy.deepcopy(self.executions[execution_key])
                prior.update(reused=True, execution_key=execution_key, arm_id=arm["id"], study_id=sid)
                self.arm_results[arm["id"]] = prior
                self.arm_ideas[arm["id"]] = prior["idea_id"]
                continue
            if execution_key in pending_keys:
                aliases.append((arm, pending_keys[execution_key]))
                continue
            pending_keys[execution_key] = arm["id"]
            pending.append((arm, execution_key))
        timeout = float(self.spec.budget["experiment_timeout"])
        estimates = [timeout if self.reservations.hard else min(timeout, max(2.0, float(a["gate"]["est_seconds"])+2))
                     for a, _ in pending]
        amount = sum(estimates)
        group = f"study:{sid}:{len(self.program['actions'])}"
        if pending and not self.reservations.reserve(group, amount):
            study.update(status="deferred", reason="whole required group exceeds available study budget")
            return {"status": "deferred", "reason": study["reason"], "reservation": amount}
        study["status"] = "running"
        jobs, meta, results = [], [], []
        try:
            for (arm, key), est in zip(pending, estimates):
                gate = arm["gate"]
                entry = {"entry_id": arm["id"], "title": f"{sid}: {arm['id']}",
                         "source": "baseline" if arm.get("role") == "baseline" else "proposal"}
                idea = self._new_idea(entry, gate["method"], gate, entry["source"])
                idea.update(study_id=sid, arm_id=arm["id"], baseline_identity=arm.get("role") == "baseline")
                self.arm_ideas[arm["id"]] = idea["id"]
                job = self.plan_v2(idea["id"], gate["method"], gate["forward_access"], arm["fidelity"], None,
                                   "study", parents=[study["artifact_id"], gate["artifact_id"]])
                job.update(pipeline="v3", schema_version=3, protocol_version=3, study_id=sid, arm_id=arm["id"],
                           claim_ids=[c["id"] for c in study.get("claims", [])],
                           diagnostic_requests=study.get("diagnostics", []), method_contract=arm["method_contract"],
                           row_hash=self.row_hash, parent_agent_id=study["owner"], parent_task_id=parent_task,
                           seed=arm["seed"], budget_group=group,
                           timeout_seconds=min(timeout, self.resources.wall_remaining()) if not self.reservations.hard else est)
                jobs.append(job)
                meta.append((arm, key, idea))
            if jobs:
                graph = {arm["id"]: set(arm.get("depends_on", [])) for arm, _, _ in meta}
                edges = study.get("dependencies", [])
                if isinstance(edges, dict):
                    for child, parents in edges.items():
                        if child in graph:
                            graph[child].update(parents)
                else:
                    for edge in edges:
                        if isinstance(edge, dict) and edge.get("after") in graph:
                            graph[edge["after"]].add(edge["before"])
                alias_map = {arm["id"]: original for arm, original in aliases}
                for aid in graph:
                    graph[aid] = {alias_map.get(parent, parent) for parent in graph[aid]}
                    graph[aid].discard(aid)
                remaining = {arm["id"]: (job, index) for index, (job, (arm, _, _)) in enumerate(zip(jobs, meta))}
                ordered = [None]*len(jobs)
                finished = {aid: r for aid, r in self.arm_results.items()}
                while remaining:
                    ready = [aid for aid in remaining if graph[aid].issubset(finished)]
                    if not ready:
                        raise ValueError("unresolved arm dependency order")
                    runnable = []
                    for aid in ready:
                        job, index = remaining[aid]
                        if any(finished[parent].get("status") != "ok" for parent in graph[aid]):
                            result = {"status": "skipped_dependency", "idea_id": job["idea_id"],
                                      "error": "prerequisite arm failed", "wall_seconds": 0}
                            art = self._emit(A.RESULT, result, author="study-executor", parents=[job["plan_artifact"]],
                                             summary=f"{aid}: prerequisite failed")
                            result["artifact_id"] = art.id
                            ordered[index], finished[aid] = result, result
                        else:
                            runnable.append(aid)
                    wave = self.executor.run([remaining[aid][0] for aid in runnable], "study")
                    for aid, result in zip(runnable, wave):
                        ordered[remaining[aid][1]] = result
                        finished[aid] = result
                    for aid in ready:
                        remaining.pop(aid)
                results = ordered
            for (arm, key, idea), result in zip(meta, results):
                measured_rows = None
                if result.get("val_idx_path"):
                    measured_rows = _array_hash(np.load(result["val_idx_path"]))
                elif result.get("val_pred_path"):
                    idx_path = Path(result["val_pred_path"]).with_name("val_idx.npy")
                    if idx_path.exists():
                        measured_rows = _array_hash(np.load(idx_path))
                result.update(val_metrics=result.get("val", {}), method_hash=M2.method_hash(arm["gate"]["method"]),
                              row_hash=measured_rows, seed=arm["seed"], fidelity=arm["fidelity"],
                              intervention=arm.get("intervention", {}), study_id=sid, arm_id=arm["id"],
                              effective_method=arm["gate"]["method"],
                              parent_agent_id=study["owner"], parent_task_id=parent_task,
                              pred_path=result.get("val_pred_path"),
                              indices_path=str(Path(result["val_pred_path"]).with_name("val_idx.npy"))
                              if result.get("val_pred_path") else None)
                self.arm_results[arm["id"]] = result
                self._record_v2(idea, result, "full", "study")
                if result.get("status") == "ok":
                    self.executions[key] = copy.deepcopy(result)
                self.program["evidence"][result["artifact_id"]] = safe_projection(result)
                owner = self.delegation.agents[study["owner"]]
                owner["evidence_ids"].append(result["artifact_id"])
                owner["seconds"] += float(result.get("wall_seconds", 0))
        finally:
            if pending:
                self.reservations.release(group, sum(float(r.get("wall_seconds", 0)) for r in results if not r.get("cached")))
        for arm, original in aliases:
            result = copy.deepcopy(self.arm_results[original])
            result.update(reused=True, arm_id=arm["id"], reused_from_arm=original)
            self.arm_results[arm["id"]] = result
            self.arm_ideas[arm["id"]] = self.arm_ideas[original]
            if arm.get("role") == "baseline":
                self.ideas[self.arm_ideas[original]]["baseline_identity"] = True
        for arm in study["arms"]:
            candidate_ids = arm.get("search_candidate_ids")
            if not candidate_ids:
                continue
            candidates = [(aid, self.arm_results.get(aid, {})) for aid in candidate_ids]
            successful = [(aid, r) for aid, r in candidates if r.get("status") == "ok"]
            if len(successful) != len(candidate_ids):
                self.arm_results[arm["id"]] = {"status": "incomplete_search", "candidate_ids": candidate_ids,
                                               "error": "not all registered random-search candidates succeeded"}
                continue
            winner, result = max(successful, key=lambda item: M.to_score(item[1]["val"].get(self.spec.primary), self.spec.primary))
            selected = copy.deepcopy(result)
            selected.update(arm_id=arm["id"], selected_candidate=winner, search_policy=arm["method"],
                            candidate_ids=candidate_ids, exploratory_selection=True)
            self.arm_results[arm["id"]] = selected
            self.arm_ideas[arm["id"]] = self.arm_ideas[winner]
            actual = self.ideas[self.arm_ideas[winner]]["mspec"]
            arm["method_contract"] = method_contract(arm["method"], actual, actual.get("code"))
        if sharing:
            required = [a for a in study["arms"] if a.get("required", True)]
            idea_ids = {self.arm_ideas.get(a["id"]) for a in required}
            if len(idea_ids) == 1 and None not in idea_ids:
                source = self.arm_results.get(required[0]["id"], {})
                if source.get("status") == "ok" and source.get("val_pred_path"):
                    shared = {"type": "shared_prediction_computation_v3", "id": sharing.get("id"),
                              "study_id": sid, "source_artifact": source.get("artifact_id"),
                              "prediction_hash": _array_hash(np.load(source["val_pred_path"])),
                              "arms": [a["id"] for a in required],
                              "protocol": "persisted_prediction_analysis"}
                    shared_art = self._emit(A.RESULT, shared, author="study-executor",
                                            parents=[source.get("artifact_id")], summary="reused one prediction computation")
                    self.program["evidence"][shared_art.id] = shared
                    for a in required:
                        self.arm_results[a["id"]].update(shared_artifact_id=sharing.get("id"),
                                                        shared_evidence_artifact=shared_art.id)
        self._reconcile_study(study)
        self.program["messages"].append({"author": "executor", "recipient": study["owner"], "study_id": sid,
             "status": study["status"], "verdicts": study["verdicts"], "evidence_ids": study.get("evidence_ids", [])})
        self._researcher_turn(self.delegation.agents[study["owner"]])
        self._critic_review(study)
        return {"study_id": sid, "status": study["status"], "verdicts": study["verdicts"],
                "evidence_ids": study.get("evidence_ids", [])}

    def _reconcile_study(self, study):
        diagnostics = {}
        for request in study.get("diagnostics", []):
            if isinstance(request, str):
                request = {"id": request, "kind": "unsupported_named_diagnostic"}
            result = self._diagnose({**request, "study_id": study["id"]})
            diagnostics[request["id"]] = result
        study["diagnostic_results"] = diagnostics
        completion = study_completion(study, self.arm_results, diagnostics)
        study.update(completion)
        study["completion"] = completion
        for arm in study["arms"]:
            arm["status"] = self.arm_results.get(arm["id"], {}).get("status", "missing")
        study["verdicts"] = []
        for claim in study.get("claims", []):
            claim = dict(claim)
            claim.setdefault("allowed_varying_factors", study.get("matching", {}).get("allowed_varying_factors", []))
            verdict = evaluate_claim(claim, self.arm_results, diagnostics)
            matching_issues = [issue for issue in completion.get("matching_issues", [])
                               if issue.get("claim_id") in (None, claim["id"])]
            if matching_issues:
                verdict.update(outcome="inconclusive", reason="registered matching protocol not satisfied")
                verdict["unmet_requirements"].extend(f"matching:{issue['field']}" for issue in matching_issues)
            blockers = [o["id"] for o in self.program["objections"] if o["scope"] == study["id"] and
                        o["blocking"] and o["status"] != "resolved"]
            blockers.extend(o["id"] for o in self.program["objections"] if o["scope"] == "program" and
                            o["blocking"] and o["status"] != "resolved")
            if blockers and verdict["outcome"] == "supported":
                verdict.update(outcome="inconclusive", reason="independent blocking objections unresolved")
                verdict["unmet_requirements"].extend(blockers)
            study["verdicts"].append(verdict)
        study["evidence_ids"] = [self.arm_results[a["id"]]["artifact_id"] for a in study["arms"]
                                 if self.arm_results.get(a["id"], {}).get("artifact_id")]
        art = self._emit(A.RESULT, {"type": "study_evidence_v3", "study": study}, author="study-executor",
                         parents=study["evidence_ids"], summary=f"{study['id']}: {study['status']}")
        self.program["evidence"][art.id] = safe_projection({"study_id": study["id"], **completion,
                                                           "verdicts": study["verdicts"]})

    def _record_evidence(self, result, author):
        art = self._emit(A.LITERATURE, result, author=author, summary=f"evidence: {result.get('source_type')} {result.get('status')}")
        self.program["evidence"][art.id] = safe_projection(result)
        return art.id

    def _diagnose(self, request):
        sid = request.get("study_id")
        study = self.program["studies"].get(sid, {})
        kind = request.get("kind", "paired")
        treatment = request.get("treatment") or next((a["id"] for a in study.get("arms", [])
                                                      if a.get("role") == "treatment"), None)
        control = request.get("control") or next((a["id"] for a in study.get("arms", [])
                                                  if a.get("role") in ("control", "baseline")), None)
        tr, cr = self.arm_results.get(treatment), self.arm_results.get(control)
        if not tr or tr.get("status") != "ok" or not tr.get("val_pred_path"):
            return {"status": "missing", "reason": "treatment predictions unavailable"}
        p = np.load(tr["val_pred_path"])
        if kind == "paired":
            if not cr or cr.get("status") != "ok" or not cr.get("val_pred_path"):
                return {"status": "missing", "reason": "control predictions unavailable"}
            if any(tr.get(k) != cr.get(k) for k in ("row_hash", "seed", "fidelity")):
                return {"status": "error", "reason": "paired rows/seed/fidelity differ"}
            result = paired_analysis(self.y_val, p, np.load(cr["val_pred_path"]), self.spec.primary,
                                     n_bootstrap=int(self.spec.execution.get("bootstrap", 200)), seed=self.spec.seed)
        elif kind in ("per_target", "per_sample", "stratified"):
            yt = np.asarray(self.y_val).reshape(len(self.y_val), -1)
            pt = np.asarray(p).reshape(yt.shape)
            err = pt-yt
            result = {"status": "complete", "metric": self.spec.primary,
                      "per_target_rel_l2": [float(np.linalg.norm(err[:, j])/max(np.linalg.norm(yt[:, j]), 1e-12))
                                            for j in range(yt.shape[1])],
                      "per_sample_l2": np.linalg.norm(err, axis=1).tolist()}
            if kind == "stratified":
                strata = request.get("strata")
                if not isinstance(strata, list) or len(strata) != len(yt):
                    return {"status": "unsupported", "reason": "strata must be predeclared validation row labels"}
                result["strata"] = {str(label): {"n": int(np.sum(np.asarray(strata) == label)),
                    "value": M.compute(self.spec.primary, self.y_val[np.asarray(strata) == label], p[np.asarray(strata) == label])}
                    for label in sorted(set(strata), key=str)}
        elif kind == "prediction_reuse":
            result = {"status": "complete", "source_artifact": tr.get("artifact_id"),
                      "prediction_hash": _array_hash(p), "metric": M.compute(self.spec.primary, self.y_val, p),
                      "shared_computation": "immutable predictions reused without fitting or inference"}
        else:
            return {"status": "unsupported", "reason": f"diagnostic adapter {kind!r} unavailable"}
        result.setdefault("status", "complete")
        result.update(treatment_arm=treatment, control_arm=control, row_hash=tr.get("row_hash"),
                      seed=tr.get("seed"), fidelity=tr.get("fidelity"), protocol_version=3,
                      metric=self.spec.primary)
        result["source_artifacts"] = [r["artifact_id"] for r in (tr, cr) if r and r.get("artifact_id")]
        result["id"] = request.get("id")
        art = self._emit(A.RESULT, {"type": "diagnostic_v3", "request": request, "result": result},
                         author="analysis-specialist", parents=result["source_artifacts"],
                         summary=f"{kind} analysis: {sid}")
        result["artifact_id"] = art.id
        self.program["evidence"][art.id] = safe_projection(result)
        return result

    def _finalize(self):
        if not self.ranked(1.0):
            raise RuntimeError("no successful full-fidelity study arm available for final export")
        saved = self.run_dir / "v3_final.json"
        if self.resume and saved.exists():
            return json.loads(saved.read_text(encoding="utf-8"))
        # Freeze selection before final access. V2 selection is reused with a wrapper that
        # decorates the actual final job and reserves exposure only once its members are known.
        original_run = self.executor.run
        def final_run(jobs, stage):
            if stage != "final":
                return original_run(jobs, stage)
            job = jobs[0]
            selection = _hash({"members": job["members_v2"], "weights": job["weights"], "spec": self.spec.hash()})
            prior = self.store.query(A.RESULT, where=lambda x: x.data.get("stage") == "final" and
                                     x.data.get("status") == "ok" and x.data.get("model_path"))
            if prior:
                historical = prior[-1].data
                self.resources.charge(float(historical.get("wall_seconds", 0)), "final", historical.get("idea_id"))
                self.reservations.used += float(historical.get("wall_seconds", 0))
                return [dict(prior[-1].data, artifact_id=prior[-1].id)]
            existing = self.exposures.entries(self.data_key, self.split_hash)
            if any(e.get("run_id") == self.run_id and e.get("status") != "completed" for e in existing):
                raise RuntimeError("interrupted final access remains uncertain; refusing to reevaluate held-out data")
            group = "final-export"
            if not self.reservations.reserve(group, self.confirmation, confirmation=True):
                raise RuntimeError("reserved export budget unavailable")
            results = []
            try:
                token = self.exposures.reserve(self.data_key, self.split_hash, self.run_id, selection,
                                               strict=bool(self.spec.evaluation.get("strict_holdout", False)))
                job.update(pipeline="v3", schema_version=3, protocol_version=3, budget_group=group,
                           timeout_seconds=min(self.confirmation, self.resources.wall_remaining()),
                           selection_hash=selection, exposure_token=token)
                self.exposures.begin(token)
                results = original_run(jobs, stage)
                if results and results[0].get("status") == "ok" and not results[0].get("final_evaluation_blocked"):
                    self.exposures.complete(token)
                return results
            finally:
                self.reservations.release(group, sum(float(r.get("wall_seconds", 0)) for r in results))
        self.executor.run = final_run
        try:
            final = self._select_export()
            final["certificate"] = final["result"].get("export_certificate", final["result"].get("certificate"))
            final["audit"]["model_certified"] = bool((final["certificate"] or {}).get("certified", False))
            final["audit"]["scientific_completion"] = all(s.get("status") == "complete" for s in self.program["studies"].values())
            self._write_json(saved, final)
            return final
        finally:
            self.executor.run = original_run

    def _select_export(self):
        """Reuse validation ranking/ensemble code; export via the V3 isolated runner."""
        from . import __version__
        self.enter("final")
        ranked = self.ranked(1.0)
        best = ranked[0]
        val = self.value(self.results[best["id"]][1.0])
        members, weights, kind = [best], [1.0], "single"
        if self.spec.execution.get("ensemble", True):
            previous_max = self.spec.search["ensemble_max"]
            try:
                self.spec.search["ensemble_max"] = int(self.spec.execution.get("ensemble_max", 5))
                ens = self._greedy_ensemble(ranked)
            finally:
                self.spec.search["ensemble_max"] = previous_max
            if ens:
                counts, ens_val = ens
                gain = M.relative_gain(M.to_score(ens_val, self.spec.primary), M.to_score(val, self.spec.primary))
                if len(counts) > 1 and gain >= float(self.spec.execution.get("min_improvement", .003)):
                    members, weights, val, kind = [self.ideas[x] for x in counts], list(counts.values()), ens_val, "ensemble"
        lineage = {m["id"]: {"label": m["label"], "origin": m["origin"], "study_id": m.get("study_id"),
                    "arm_id": m.get("arm_id"), "gate": m["gate_artifact"],
                    "baseline_identity": m.get("baseline_identity", False)} for m in members}
        expid = self.engineer.new_exp_id()
        out_dir = self.store.workspace("experiments", expid)
        job = {"mode": "final_v3", "exp_id": expid, "idea_id": best["id"], "pipeline": "v3",
               "members_v2": [m["mspec"] for m in members], "member_ids": [m["id"] for m in members],
               "forward_access_members": [bool(m["forward_access"]) for m in members], "weights": weights,
               "task": self.spec.task, "primary": self.spec.primary, "secondary": self.spec.metric_names[1:],
               "data_path": str(self.data_path), "seed": int(self.spec.seed), "data_seed": int(self.spec.data["seed"]),
               "feature_shape": self.feature_shape, "final_dir": str(self.run_dir / "final"),
               "run_id": self.run_id.replace("-", "_"), "stage": "final", "fraction": 1.0,
               "threads": int(self.spec.hardware["threads_per_worker"]),
               "wants_gpu": any(M2.wants_gpu(m["mspec"]) for m in members), "simulator": self.sim_cfg,
               "max_forward_evals": None, "bootstrap": int(self.spec.execution.get("bootstrap", 200)),
               "data_cache_key": self.data_key, "spec_hash": self.spec.hash(), "harness_version": __version__,
               "lineage": lineage, "job_dir": str(out_dir)}
        plan = self._emit(A.PLAN, {k: v for k, v in job.items() if k not in ("data_path", "members_v2")},
                          author="engineer-v3", parents=[m["hyp_artifact"] for m in members],
                          summary=f"frozen V3 export selection: {[m['label'] for m in members]}")
        job["plan_artifact"] = plan.id
        result = self.executor.run([job], "final")[0]
        if result.get("status") != "ok":
            raise RuntimeError(f"final export failed: {result.get('error')}")
        audit = self.critic.final_audit(result, val, [plan.id])
        checkpoint = self._emit(A.CHECKPOINT, {"pipeline": "v3", "model_path": result["model_path"],
            "members": job["member_ids"], "test": result.get("test"), "export_certificate": result.get("export_certificate"),
            "selection_hash": job.get("selection_hash"), "lineage": lineage}, author="scheduler",
            parents=[result["artifact_id"]], files=[result["model_path"]], summary="V3 exported model candidate")
        return {"kind": kind, "members": job["member_ids"], "labels": [m["label"] for m in members],
                "weights": weights, "val": val, "test": result.get("test", {}), "model_path": result["model_path"],
                "audit": audit, "checkpoint": checkpoint.id, "result_artifact": result["artifact_id"],
                "result": result, "lineage": lineage}

    def _write_knowledge(self):
        for aid, result in self.arm_results.items():
            if result.get("status") != "ok":
                self.memory.add("failure", title=f"V3 arm {aid}", author="scheduler", run_id=self.run_id,
                                context=self.mem_context, data=safe_projection(result), status=result.get("status"))
                continue
            idea = self.ideas[self.arm_ideas[aid]]
            package = self.package_store.save(idea["mspec"], provenance={"run_id": self.run_id,
                    "artifact_id": result.get("artifact_id"), "arm_id": aid},
                    applicability=self.mem_context, validation={"status": "executed", "metrics": result.get("val"),
                    "verification": result.get("verification"), "method_hash": result.get("method_hash"),
                    "source_hash": result.get("source_hash")}, failure_conditions=[])
            self.packages.append(package)
            self.memory.add("observation", title=f"V3 executable method {idea['label']}", author="scheduler",
                            run_id=self.run_id, context=self.mem_context, status="development_evidence",
                            data=safe_projection({"package": package, "arm_id": aid, "metrics": result.get("val")}))

    def _write_bundle(self, error, elapsed):
        from .agents.reporter_v3 import write_report_v3
        for agent in self.delegation.agents.values():
            events = [e for e in self.router.events if e.get("actor") == agent["id"]]
            calls = [e for e in events if e.get("call_attempted")]
            agent["llm_calls"] = len(calls)
            agent["llm_seconds"] = sum(float(e.get("seconds", 0)) for e in calls)
            known_cost = sum(float(e.get("cost_usd") or 0) for e in calls)
            agent["known_cost_usd"] = known_cost
            agent["cost_usd"] = None if any(e.get("cost_usd") is None for e in calls) else known_cost
            agent["cost_scope"] = "direct role calls; execution seconds include attributed child work"
        bundle = {"schema_version": 3, "pipeline": "v3", "run_id": self.run_id,
                  "objective": self.spec.problem, "contract": getattr(self, "contract", {}),
                  "program": self.program, "studies": list(self.program["studies"].values()),
                  "arm_results": self.arm_results, "agents": self.delegation.agents,
                  "specialist_tasks": self.delegation.tasks, "role_events": self.router.events,
                  "evaluation_protocol": getattr(self, "protocol", {}), "packages": self.packages,
                  "final": self.final, "costs": self.resources.snapshot(),
                  "resources": self.resources.snapshot(),
                  "reservations": self.reservations.snapshot(), "elapsed_seconds": elapsed, "error": error,
                  "limits": ["No discovery or transfer campaign was run during development.",
                             "Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.",
                             "The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.",
                             "Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. "
                             "Arbitrary in-process callbacks cannot be forcibly preempted."]}
        self._write_json(self.run_dir / "evidence_bundle.json", bundle)
        self._write_json(self.run_dir / "research_trace_v3.json", self.program)
        write_report_v3(self.run_dir, bundle)
        self._emit(A.REPORT, {"pipeline": "v3", "evidence_bundle": str(self.run_dir / "evidence_bundle.json"),
                             "facts_hash": _hash(bundle)}, author="reporter-v3",
                   files=[str(self.run_dir / "report.md")], summary="factual V3 report from persisted evidence")
