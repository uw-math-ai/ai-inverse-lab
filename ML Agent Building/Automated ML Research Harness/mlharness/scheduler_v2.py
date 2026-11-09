"""V2 research-first scheduler.

    spec -> data (+ distribution-shift sets, problem context, identifiability diagnostics)
         -> mean baseline
         -> RESEARCH: 4 parallel researchers <-> shared critic (+ TheoremSearch, research memory,
            optional pilot experiments), revise/defend/abandon/branch, bounded rounds
         -> PORTFOLIO: diverse feasible proposals + baselines (conventional ML, random search, physics)
         -> GATE: independent deterministic validity / safety / leakage / metric / budget checks
         -> SCREEN: every method at low fidelity (training fraction for trainable methods; a nested
            validation subsample for all), pruned only for crashes, infeasible projected cost or not
            beating the mean baseline — never ranked across families at unequal fidelity
         -> EVALUATE: full fidelity, budget-aware, diversity-preserving promotion
         -> REFINE: local mutation of the best methods (cheap, reliable in the V1 ablation)
         -> PREDICTIONS: each proposal's falsifiable prediction is checked against the results
         -> FINAL: greedy ensemble on validation predictions, refit on train+val, one test
            evaluation (+ bootstrap CI, shift sets, latency), export with reproducibility metadata
         -> MEMORY write-back (observations, failures, findings; test numbers redacted for agents)
         -> REPORT (established evidence / conjectures / failed hypotheses / remaining uncertainty)

Reuses the V1 artifact store, executor (process isolation, caching, budgets), resource manager and
greedy ensemble. The artifact ledger stays the canonical provenance record.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import artifacts as A
from . import methods_v2 as M2
from . import metrics as M
from .memory import ResearchMemory
from .problem import ProblemContext, diagnostics
from .remote import make_remote
from .research import (CodeEngineer, ImplementationGate, ResearchCritic, ResearcherAgent, ResearchLoop,
                       check_prediction, select_portfolio)
from .resources import STAGE_CAPS_V2, ResourceManager
from .scheduler import ResearchScheduler
from .tools import LiteratureTool, TheoremSearchClient

log = logging.getLogger("mlharness")


class ResearchSchedulerV2(ResearchScheduler):
    def __init__(self, spec, run_dir, llm=None, resume: bool = False, verbose: bool = True,
                 python_exe: str | None = None, memory: ResearchMemory | None = None, literature_client=None,
                 remote=None):
        super().__init__(spec, run_dir, llm=llm, resume=resume, verbose=verbose, python_exe=python_exe)
        hw = spec.hardware
        self.resources = ResourceManager(spec.budget["compute_seconds"], spec.budget["wall_seconds"], hw["workers"],
                                         hw["gpu_devices"], spec.budget["llm_calls"], caps=STAGE_CAPS_V2)
        mc = spec.memory or {}
        self.memory = memory if memory is not None else ResearchMemory(
            path=mc.get("path"), enabled=mc.get("enabled", True), read=mc.get("read", True), write=mc.get("write", True))
        lc = spec.literature or {}
        self.lit_client = literature_client if literature_client is not None else TheoremSearchClient(
            base_url=lc.get("base_url", "https://api.theoremsearch.com"), timeout=lc.get("timeout", 45),
            retries=lc.get("retries", 1), cache_dir="default" if lc.get("cache", True) else None,
            enabled=lc.get("enabled", True))
        self.remote = remote if remote is not None else make_remote(hw.get("remote"))
        self.resume = resume
        self.v2res: dict[str, dict] = {}
        self.research: dict | None = None
        self.portfolio: list[dict] = []
        self.excluded: list[dict] = []
        self.gate_verdicts: dict[str, dict] = {}
        self.prediction_outcomes: dict[str, dict] = {}
        self.pilot_records: dict[str, dict] = {}
        self.memory_writes = 0
        self.test_access_prior = 0
        self.rng = np.random.default_rng(spec.seed + 4242)

    # ---------------------------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        error = None
        try:
            self.stage_spec()
            self.stage_data()
            self.stage_baseline()
            self.problem["baseline_value"] = self.baseline_value
            self.stage_research()
            self.stage_portfolio()
            ideas = self.stage_gate()
            survivors = self.stage_screen_v2(ideas)
            self.stage_evaluate_v2(survivors)
            self.stage_refine_v2()
            self.stage_predictions()
            self.final = self.stage_final_v2()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.exception("research run failed")
            self.store.put(A.FAILURE, {"error": error, "stage": self.stage, "source": "scheduler"},
                           author="scheduler", stage=self.stage, summary=f"run failed in {self.stage}: {error[:120]}")
        finally:
            try:
                self.stage_memory_writeback()
            except Exception:
                log.exception("memory write-back failed")
            self.enter("report")
            try:
                from .agents.reporter_v2 import ReporterV2
                self.reporter_v2 = self.spawn(ReporterV2, "reporter-v2")
                self.reporter_v2.write(self)
            except Exception:
                log.exception("report generation failed")
            self.stage = "done" if error is None else "failed"
            self.status(error=error, elapsed=time.time() - t0, pipeline="v2",
                        research_stop=(self.research or {}).get("stop_reason"))
            if self.remote is not None:
                self.remote.close()
        if error:
            raise RuntimeError(error)
        return {"run_dir": str(self.run_dir), "final": self.final, "resources": self.resources.snapshot(),
                "research": {k: (self.research or {}).get(k) for k in ("stop_reason",)}}

    # -- data / problem context -----------------------------------------------------------------
    def stage_data(self):
        super().stage_data()
        self.executor.remote = self.remote
        from .simulators import get_simulator, resolve_user_path
        sim_cfg = copy.deepcopy(self.spec.simulator)
        if sim_cfg.get("name") == "python":
            sim_cfg.setdefault("params", {})["path"] = str(resolve_user_path(sim_cfg["params"]["path"]))
        self.sim_cfg = sim_cfg
        sim = self.sim_agent.simulator
        self.sim_hash = sim.config_hash()
        self.data_key = self.data_art.data["cache_key"]
        c = self.spec.constraints
        self.ctx = ProblemContext(sim_cfg, authorized=bool(c.get("allow_forward_model", True)))
        diag, spf = {"available": False}, None
        if self.ctx.forward_available and self.ctx.lower is not None:
            rng = np.random.default_rng(0)
            th = rng.uniform(self.ctx.lower, self.ctx.upper, size=(256, len(self.ctx.lower)))
            t = time.time()
            self.ctx.forward(th)
            spf = (time.time() - t) / 256
            diag = diagnostics(self.ctx, n_samples=32, seed=self.spec.seed)
        self.problem = {**self.ctx.public_summary(), "n_params": self.ctx.n_params, "diagnostics": diag,
                        "describe": self.ctx.describe(), "sec_per_forward": spf, "baseline_value": None}
        with np.load(self.data_path) as z:
            ytr = z["y_train"]
            self.data_stats = {"target_min": float(np.min(ytr)), "target_dim": int(1 if ytr.ndim == 1 else ytr.shape[1])}
        self.mem_context = {"simulator": sim.name, "simulator_hash": self.sim_hash,
                            "problem_class": f"{sim.name}:{Path(str(self.spec.simulator.get('params', {}).get('path', sim.name))).stem}",
                            "task": self.spec.task, "noise_regime": f"noise_std={self.ctx.noise_std}",
                            "data_regime": f"n_train={self.spec.data['n_train']}",
                            "forward_available": self.ctx.forward_available}
        self.store.put(A.DATASET, {"type": "problem_context", "problem": self.problem, "memory_context": self.mem_context},
                       author="scheduler", stage="data", parents=[self.data_art.id],
                       summary=f"problem context: forward={'yes' if self.ctx.forward_available else 'no'}, "
                               f"n_params={self.ctx.n_params}, CRB floor~{diag.get('crb_rel_l2_estimate', 'n/a')}")
        self._shift_sets(sim_cfg)
        self.gate = ImplementationGate(self.spec, self.store, self.available, self.feature_sets, self.n_features,
                                       self.problem, self.data_stats, has_remote=self.remote is not None)
        self.code_engineer = self.spawn(CodeEngineer, "engineer-v2")
        log.info("problem context: %s", self.ctx.describe()[:300])
        if diag.get("available"):
            log.info("identifiability: Fisher cond median %.3g, CRB rel_l2 floor ~%.4g, least determined %s",
                     diag["fisher_condition_median"], diag["crb_rel_l2_estimate"], diag["least_determined"])

    def _shift_sets(self, sim_cfg):
        from .simulators import get_simulator
        ev = self.spec.evaluation or {}
        cfg = ev.get("shift_sets", "auto")
        n = int(ev.get("shift_n", 300))
        params = sim_cfg.get("params") or {}
        sets = []
        if cfg == "auto":
            if isinstance(params.get("noise"), (int, float)) and params["noise"] > 0:
                sets = [{"name": "noise_x3", "params": {"noise": 3 * float(params["noise"])}}]
        elif isinstance(cfg, list):
            sets = [s for s in cfg if isinstance(s, dict) and s.get("name")]
        if not sets:
            self.shift_info = []
            return
        arrays = {}
        with np.load(self.data_path) as z:
            base = {k: z[k] for k in z.files}
        info = []
        for i, s in enumerate(sets):
            c2 = copy.deepcopy(sim_cfg)
            c2.setdefault("params", {}).update(s.get("params") or {})
            try:
                X, y = get_simulator(c2).generate(n, int(self.spec.data["seed"]) + 7919 + i)
            except Exception as exc:
                info.append({"name": s["name"], "error": f"{type(exc).__name__}: {exc}"})
                continue
            arrays[f"X_shift_{s['name']}"], arrays[f"y_shift_{s['name']}"] = X, y
            info.append({"name": s["name"], "params": s.get("params"), "n": n})
        np.savez(self.data_path, **base, **arrays)
        self.shift_info = info
        self.store.put(A.DATASET, {"type": "shift_sets", "sets": info}, author="simulator", stage="data",
                       parents=[self.data_art.id], summary=f"distribution-shift evaluation sets: {[x['name'] for x in info]}")

    # -- research -------------------------------------------------------------------------------
    def stage_research(self):
        self.enter("research")
        rc, lc = self.spec.research, self.spec.literature
        self.literature = LiteratureTool(self.lit_client, self.store, self.memory, self.resources,
                                         facts=self.problem.get("facts"), max_total=lc.get("max_total_queries", 40),
                                         per_agent_round=rc.get("queries_per_round", 2), n_results=lc.get("n_results", 4),
                                         context=self.mem_context, run_id=self.run_id)
        if self.resume and self.store.latest(A.PORTFOLIO) is not None:
            from .agents.reporter_v2 import research_state_from_store
            self.research = research_state_from_store(self.store)
            self.research["resumed"] = True
            log.info("resume: research phase already completed; reusing its portfolio")
            return
        researchers = {}
        for k, role in enumerate(rc["researchers"]):
            name = f"researcher-{role}"
            researchers[name] = self.spawn(ResearcherAgent, name, persona=role, literature=self.literature,
                                           memory=self.memory, problem=self.problem, available=self.available,
                                           feature_sets=self.feature_sets, n_features=self.n_features,
                                           seed=self.spec.seed * 1000 + k, context=self.mem_context,
                                           run_id=self.run_id, rcfg=rc)
        critic = self.spawn(ResearchCritic, "research-critic", literature=self.literature, memory=self.memory,
                            problem=self.problem, rcfg=rc, available=self.available, feature_sets=self.feature_sets,
                            n_features=self.n_features, context=self.mem_context, run_id=self.run_id,
                            exec_budget_s=float(self.spec.budget["compute_seconds"]))
        self.researchers = researchers
        loop = ResearchLoop(spec=self.spec, store=self.store, memory=self.memory, literature=self.literature,
                            researchers=researchers, critic=critic, resources=self.resources, run_id=self.run_id,
                            mem_context=self.mem_context, base_parents=[self.spec_art.id, self.data_art.id],
                            pilot_fn=self.run_pilots if rc.get("pilot_tests", True) else None,
                            uses_llm=not self.llm.is_null)
        self.research = loop.run()
        self.funnel["research_rounds"] = len(self.research["rounds"])
        self.funnel["proposal_versions"] = len(self.research["proposals"])
        self.funnel["proposals_active_at_end"] = len(self.research["active"])
        self.funnel["proposals_abandoned"] = len(self.research["abandoned"])

    # pilot experiments requested by the critic (inside the research loop, budgeted)
    def _expand(self, ms: dict) -> list[dict]:
        if ms.get("kind") == "random_search":
            return M2.random_library_specs(int(ms.get("n", 2)), self.available, self.feature_sets, self.n_features, self.rng)
        return [ms]

    def plan_v2(self, idea_id: str, ms: dict, forward_access: bool, fraction: float, val_subsample, stage: str,
                parents=()) -> dict:
        exp_id = self.engineer.new_exp_id()
        out_dir = self.store.workspace("experiments", exp_id)
        job = {"mode": "method_eval", "exp_id": exp_id, "idea_id": idea_id, "method": ms, "fraction": float(fraction),
               "val_subsample": val_subsample, "task": self.spec.task, "primary": self.spec.primary,
               "secondary": self.spec.metric_names[1:], "data_path": str(self.data_path),
               "data_seed": int(self.spec.data["seed"]), "seed": int(self.spec.seed), "feature_shape": self.feature_shape,
               "stage": stage, "threads": int(self.spec.hardware["threads_per_worker"]), "wants_gpu": M2.wants_gpu(ms),
               "simulator": self.sim_cfg, "simulator_hash": self.sim_hash, "forward_access": bool(forward_access),
               "max_forward_evals": self.spec.constraints.get("max_forward_evals")}
        art = self.store.put(A.PLAN, {k: v for k, v in job.items() if k not in ("data_path",)} |
                             {"method": {k: v for k, v in ms.items() if k != "code"}}, author="engineer-v2",
                             idea_id=idea_id, parents=list(parents), stage=stage,
                             summary=f"{exp_id}: {M2.label(ms)} train={fraction:.0%} val={'all' if not val_subsample else val_subsample}")
        job["plan_artifact"] = art.id
        job["job_dir"] = str(out_dir)
        return job

    def run_pilots(self, proposals: list[dict], round_idx: int) -> dict:
        out, jobs, meta = {}, [], []
        sub = int(self.spec.research.get("pilot_val_subsample", 60))
        frac = float(self.spec.execution.get("screen_fraction", 0.1))
        for p in proposals:
            ms_raw = p.get("method") or {}
            if ms_raw.get("kind") == "random_search":
                continue
            if self.resources.exhausted("pilot"):
                out[p["id"]] = {"status": "skipped_budget"}
                continue
            v = self.gate.check(ms_raw, label=f"pilot {p['id']}", parents=[p.get("artifact_id")], idea_id=f"pilot-{p['id']}")
            if not v["approved"]:
                out[p["id"]] = {"status": "rejected_by_gate", "error": "; ".join(v["reasons"])[:300]}
                continue
            ms = v["method"]
            jobs.append(self.plan_v2(f"pilot-{p['id']}", ms, v["forward_access"], frac if M2.trains(ms) else 1.0, sub,
                                     "pilot", parents=[p.get("artifact_id"), v["artifact_id"]]))
            meta.append((p, ms))
        if jobs:
            log.info("research round %d: running %d pilot experiment(s)", round_idx, len(jobs))
        for (p, ms), res in zip(meta, self.executor.run(jobs, "pilot")):
            val = self.value(res)
            rec = {"status": res.get("status"), "label": M2.label(ms), "value": val, "metric": self.spec.primary,
                   "n_val": res.get("n_val_used"), "train_fraction": res.get("n_train_used"),
                   "seconds": res.get("wall_seconds"), "error": (res.get("error") or "")[:300] or None,
                   "forward_evals": res.get("forward_evals"), "diagnostics": res.get("diagnostics"),
                   "uncertainty": res.get("uncertainty"), "result_artifact": res.get("artifact_id"),
                   "worse_than_baseline": bool(val is not None and self.baseline_value is not None and
                                               M.to_score(val, self.spec.primary) <= M.to_score(self.baseline_value, self.spec.primary)),
                   "note": f"pilot on {sub} validation samples (not comparable to full-fidelity numbers)"}
            out[p["id"]] = rec
            self.pilot_records[p["id"]] = rec
            self.memory.add("observation", title=f"pilot: {M2.label(ms)} ({p['title'][:80]})", author="scheduler",
                            run_id=self.run_id, status="pilot", context=self.mem_context,
                            text=f"pilot experiment for {p['title']}: {rec['status']} {self.spec.primary}={val}",
                            data={"pilot": True, "proposal_id": p["id"], "method_label": M2.label(ms),
                                  "method_hash": M2.method_hash(ms), "val": {self.spec.primary: val}, **{k: rec[k] for k in ("status", "seconds", "n_val")}},
                            artifact_refs=[{"run_id": self.run_id, "artifact_id": res.get("artifact_id")}])
        return out

    # -- portfolio + gate --------------------------------------------------------------------------
    def _norm(self, raw):
        return M2.normalize(raw, self.available, self.feature_sets, self.n_features, int(self.spec.data["n_train"]))

    def stage_portfolio(self):
        self.enter("portfolio")
        if self.research.get("resumed") and self.store.latest(A.PORTFOLIO) is not None:
            d = self.store.latest(A.PORTFOLIO).data
            self.portfolio, self.excluded = d["entries"], d["excluded"]
            return
        entries, excluded = select_portfolio(self.research["active"], self.research.get("last_critique"),
                                             self.spec.execution, self._norm, bool(self.problem.get("forward_available")),
                                             self.available)
        self.portfolio, self.excluded = entries, excluded
        props = self.research["proposals"]
        art = self.store.put(A.PORTFOLIO, {"entries": entries, "excluded": excluded,
                                           "research_stop_reason": self.research["stop_reason"]},
                             author="scheduler", stage="portfolio",
                             parents=[props[e["proposal_id"]]["artifact_id"] for e in entries if e.get("proposal_id")],
                             summary=f"portfolio: {len(entries)} methods ({sum(1 for e in entries if e['source'] == 'baseline')} "
                                     f"baselines), families {sorted({e['family'] for e in entries})}")
        self.portfolio_art = art.id
        for x in excluded:
            self.decide("not_advanced", x["reason"], parents=[props[x["proposal_id"]]["artifact_id"]]
                        if x.get("proposal_id") in props else [], proposal_id=x.get("proposal_id"))
        self.funnel["portfolio_entries"] = len(entries)
        log.info("portfolio: %s", ", ".join(f"{e['entry_id']} {M2.label(self._norm(e['method'])[0])}" for e in entries))

    def _new_idea(self, entry: dict, ms: dict, verdict: dict, origin: str, parent=None, title=None) -> dict:
        iid = self.new_idea_id()
        props = (self.research or {}).get("proposals") or {}
        prop = props.get(entry.get("proposal_id")) if entry.get("proposal_id") else None
        idea = {"id": iid, "title": title or entry["title"], "persona": entry.get("persona") or "control",
                "origin": origin, "entry_id": entry.get("entry_id"), "proposal_id": entry.get("proposal_id"),
                "root_id": entry.get("root_id"), "parent": parent, "round": 0, "mspec": ms,
                "forward_access": verdict["forward_access"], "config": M2.as_v1_config(ms), "family": M2.family(ms),
                "label": M2.label(ms), "hypothesis": (prop or {}).get("hypothesis") or entry.get("selection_reason", ""),
                "est_seconds": verdict.get("est_seconds"), "gate_artifact": verdict["artifact_id"],
                "source": entry.get("source")}
        parents = [verdict["artifact_id"]] + ([prop["artifact_id"]] if prop else []) + \
                  ([self.ideas[parent]["hyp_artifact"]] if parent in self.ideas else []) + [self.spec_art.id]
        idea["hyp_artifact"] = self.store.put(A.HYPOTHESIS, {k: v for k, v in idea.items() if k != "mspec"} |
                                              {"mspec": {k: v for k, v in ms.items() if k != "code"}},
                                              author="scheduler", idea_id=iid, parents=parents, stage="gate",
                                              summary=f"{iid} {idea['label']} ({origin})").id
        self.ideas[iid] = idea
        return idea

    def stage_gate(self) -> list[dict]:
        self.enter("gate")
        ideas = []
        props = (self.research or {}).get("proposals") or {}
        for e in self.portfolio:
            prop = props.get(e.get("proposal_id"))
            for k, raw in enumerate(self._expand(copy.deepcopy(e["method"]))):
                if raw.get("kind") in ("custom_ml", "custom_solver") and not raw.get("code"):
                    code, why = self.code_engineer.generate(prop or e, raw, self.problem["describe"])
                    if code:
                        raw["code"] = code
                    else:
                        self.decide("gate_reject", f"{e['entry_id']}: {why}", parents=[prop["artifact_id"]] if prop else [])
                        continue
                label = e["entry_id"] + (f".{k + 1}" if e["method"].get("kind") == "random_search" else "")
                v = self.gate.check(raw, label=label, parents=[prop["artifact_id"]] if prop else [])
                self.gate_verdicts[label] = v
                if not v["approved"]:
                    self.decide("gate_reject", f"{label}: {'; '.join(v['reasons'])}"[:300],
                                parents=[v["artifact_id"]], entry=e["entry_id"])
                    continue
                title = e["title"] if e["method"].get("kind") != "random_search" else f"{e['title']} #{k + 1}"
                ideas.append(self._new_idea(e, v["method"], v, "research" if e["source"] == "proposal" else "baseline",
                                            title=title))
        self.funnel["gate_approved"] = len(ideas)
        log.info("gate: %d/%d executable methods approved", len(ideas), len(self.gate_verdicts))
        return ideas

    # -- execution -----------------------------------------------------------------------------
    def _project(self, idea, res) -> float:
        ms = idea["mspec"]
        n_val = int(self.spec.data["n_val"])
        ratio = n_val / max(1, int(res.get("n_val_used") or n_val))
        fit = float(res.get("fit_seconds") or 0.0)
        pred = float(res.get("predict_seconds") or 0.0)
        overhead = max(0.0, float(res.get("wall_seconds") or 0.0) - fit - pred)
        n_tr = max(1, int(res.get("n_train_used") or 1))
        f = n_tr / max(1, int(self.spec.data["n_train"]))
        lib = ms.get("init") if ms.get("kind") == "hybrid" else ms
        alpha = 0.0
        if M2.trains(ms) and f < 0.999:
            from . import methods as ML
            m = ML.LIBRARY.get((lib or {}).get("method"))
            alpha = m.scaling if m else 1.0
        return fit * (1.0 / f) ** alpha + pred * ratio + overhead

    def _record_v2(self, idea, res, key: str, stage: str):
        self.v2res.setdefault(idea["id"], {})[key] = res
        if key == "full":
            self.results.setdefault(idea["id"], {})[1.0] = res
        if res.get("status") != "ok":
            from .agents.critic import explain_failure
            self.store.put(A.FAILURE, {"error": res.get("error"), "status": res.get("status"),
                                       "explanation": explain_failure(res.get("error") or res.get("status")),
                                       "method": idea["label"], "fidelity": key,
                                       "safety_violation": bool(res.get("safety_violation"))},
                           author="critic", idea_id=idea["id"], parents=[res.get("artifact_id")], stage=stage,
                           summary=f"{idea['label']} failed at {key}: {explain_failure(res.get('error') or res.get('status'))[:120]}")

    def _try_repair(self, idea, res) -> bool:
        ms = idea["mspec"]
        if ms.get("kind") not in ("custom_ml", "custom_solver") or res.get("safety_violation"):
            return False
        code = self.code_engineer.repair(ms, res.get("error", ""), res.get("traceback", ""), self.problem["describe"])
        if not code:
            return False
        new = dict(ms, code=code)
        v = self.gate.check(new, label=f"repair {idea['id']}", parents=[res.get("artifact_id")], idea_id=idea["id"])
        if not v["approved"]:
            return False
        idea["mspec"] = v["method"]
        idea["repaired"] = idea.get("repaired", 0) + 1
        self.store.put(A.CODE, {"version": idea["repaired"] + 1, "chars": len(code), "reason": res.get("error")},
                       author="engineer-v2", idea_id=idea["id"], parents=[res.get("artifact_id"), v["artifact_id"]],
                       stage="screen", summary=f"repaired agent code for {idea['id']}")
        return True

    def stage_screen_v2(self, ideas: list[dict]) -> list[dict]:
        self.enter("screen")
        ex = self.spec.execution
        frac, sub = float(ex.get("screen_fraction", 0.1)), int(ex.get("eval_subsample", 150))

        def job(i):
            return self.plan_v2(i["id"], i["mspec"], i["forward_access"], frac if M2.trains(i["mspec"]) else 1.0, sub,
                                "screen", parents=[i["hyp_artifact"]])
        results = self.executor.run([job(i) for i in ideas], "screen")
        retry = [i for i, r in zip(ideas, results) if r.get("status") == "error"
                 and int(ex.get("repair_attempts", 1)) > 0 and self._try_repair(i, r)]
        if retry:
            fixed = dict(zip([i["id"] for i in retry], self.executor.run([job(i) for i in retry], "screen")))
            results = [fixed.get(i["id"], r) for i, r in zip(ideas, results)]
        limit = float(self.spec.budget["experiment_timeout"])
        base = M.to_score(self.baseline_value, self.spec.primary)
        survivors = []
        for idea, res in zip(ideas, results):
            self._record_v2(idea, res, "screen", "screen")
            if res.get("status") != "ok":
                from .agents.critic import explain_failure
                self.prune(idea, f"screen failed: {explain_failure(res.get('error') or res.get('status'))}",
                           parents=[res.get("artifact_id")])
                continue
            proj = self._project(idea, res)
            idea["projected_full_seconds"] = round(proj, 2)
            unknown_scaling = idea["mspec"].get("kind") in ("custom_ml", "custom_solver")
            if proj > limit and unknown_scaling:
                # agent code may not scale with the training fraction (e.g. it simulates its own data): do not
                # prune on an extrapolation; the per-experiment timeout enforces the limit with real evidence
                idea["projected_full_seconds"] = round(min(proj, limit), 2)
                self.decide("promote_unmeasured", f"projected {proj:.0f}s exceeds the limit but this method's cost "
                                                  f"scaling is unknown; measured at full fidelity under the "
                                                  f"{limit:.0f}s timeout", idea_id=idea["id"])
                survivors.append(idea)
            elif proj > limit:
                self.prune(idea, f"projected full-fidelity cost {proj:.0f}s exceeds the {limit:.0f}s per-experiment limit",
                           parents=[res["artifact_id"]])
            elif M.relative_gain(self.score(res), base) < float(ex.get("baseline_margin", 0.02)):
                self.prune(idea, f"not meaningfully better than the mean baseline at screen fidelity "
                                 f"({self.value(res):.4g} vs {self.baseline_value:.4g})", parents=[res["artifact_id"]])
            else:
                survivors.append(idea)
        self.funnel["screen_survivors"] = len(survivors)
        log.info("screen: %d/%d methods survive (no cross-family ranking at unequal fidelity)", len(survivors), len(ideas))
        return survivors

    def _diverse_order(self, ideas, key="screen") -> list[dict]:
        scored = sorted(ideas, key=lambda i: -self.score(self.v2res[i["id"]][key]))
        first, seen = [], set()
        for i in scored:
            if i["family"] not in seen:
                first.append(i)
                seen.add(i["family"])
        return first + [i for i in scored if i not in first]

    def stage_evaluate_v2(self, survivors: list[dict]):
        self.enter("evaluate")
        order = self._diverse_order(survivors)
        avail = self.resources.stage_remaining("evaluate")
        keep, total = [], 0.0
        for i in order:
            c = float(i.get("projected_full_seconds") or 30.0)
            if keep and total + c > avail:
                self.decide("budget_cut", f"not promoted: projected {c:.0f}s does not fit the remaining {avail - total:.0f}s "
                                          f"of the evaluate budget (diversity-preserving order)", idea_id=i["id"])
                continue
            keep.append(i)
            total += c
        for i in keep:
            self.decide("promote", f"to full fidelity (projected {i.get('projected_full_seconds')}s)", idea_id=i["id"])
        results = self.executor.run([self.plan_v2(i["id"], i["mspec"], i["forward_access"], 1.0, None, "evaluate",
                                                  parents=[i["hyp_artifact"]]) for i in keep], "evaluate")
        for idea, res in zip(keep, results):
            self._record_v2(idea, res, "full", "evaluate")
            if res.get("status") not in ("ok", "skipped_budget"):
                self.prune(idea, f"full-fidelity evaluation failed: {res.get('error')}", parents=[res.get("artifact_id")])
        self.funnel["evaluated_full"] = sum(1 for i in keep if self.v2res[i["id"]].get("full", {}).get("status") == "ok")
        r = self.ranked(1.0)
        if r:
            log.info("evaluate: best %s %s val %s=%.5g", r[0]["id"], r[0]["label"], self.spec.primary,
                     self.value(self.results[r[0]["id"]][1.0]))

    def stage_refine_v2(self):
        self.enter("refine")
        ex = self.spec.execution
        rounds = int(ex.get("refine_rounds", 1))
        for rnd in range(1, rounds + 1):
            if self.resources.exhausted("refine"):
                self.stop_reason = "refine budget exhausted"
                break
            # local tuning is applied to trainable methods (where the V1 ablation showed it pays off);
            # solver knobs of physics methods are left to the research proposals themselves
            top = [i for i in self.ranked(1.0) if i.get("mspec") and i["mspec"].get("kind") in ("library", "hybrid")]
            top = top[: int(ex.get("refine_top_k", 2))]
            if not top:
                break
            best_before = self.score(self.results[self.ranked(1.0)[0]["id"]][1.0])
            hashes = {M2.method_hash(i["mspec"]) for i in self.ideas.values() if i.get("mspec")}
            children = []
            for p in top:
                for _ in range(int(ex.get("refine_children", 2))):
                    for _try in range(6):
                        cms = M2.mutate(p["mspec"], self.available, self.feature_sets, self.rng,
                                        strength=max(0.15, 0.4 / rnd))
                        h = M2.method_hash(self._norm(cms)[0])
                        if h not in hashes:
                            hashes.add(h)
                            break
                    else:
                        continue
                    v = self.gate.check(cms, label=f"refine {p['id']}", parents=[p["hyp_artifact"]])
                    if v["approved"]:
                        entry = {"title": f"refinement of {p['id']} ({p['label']})", "persona": p["persona"],
                                 "entry_id": p.get("entry_id"), "proposal_id": p.get("proposal_id"),
                                 "root_id": p.get("root_id"), "source": p.get("source")}
                        children.append(self._new_idea(entry, v["method"], v, "refine", parent=p["id"]))
            costs = {c["id"]: float(self.v2res[c["parent"]]["full"].get("wall_seconds") or 30) for c in children}
            avail, keep, tot = self.resources.stage_remaining("refine"), [], 0.0
            for c in children:
                if keep and tot + costs[c["id"]] > avail:
                    self.decide("budget_cut", f"refinement child does not fit the refine budget", idea_id=c["id"])
                    continue
                keep.append(c)
                tot += costs[c["id"]]
            results = self.executor.run([self.plan_v2(c["id"], c["mspec"], c["forward_access"], 1.0, None, "refine",
                                                      parents=[c["hyp_artifact"]]) for c in keep], "refine")
            for c, res in zip(keep, results):
                self._record_v2(c, res, "full", "refine")
                if res.get("status") not in ("ok", "skipped_budget"):
                    self.prune(c, f"refinement failed: {res.get('error')}", parents=[res.get("artifact_id")])
            best = self.ranked(1.0)
            gain = M.relative_gain(self.score(self.results[best[0]["id"]][1.0]), best_before) if best else 0.0
            self.decide("refine_round", f"round {rnd}: {len(keep)} children, gain {gain:.4f}", round=rnd, gain=gain)
            log.info("refine round %d: %d children, best gain %.2f%%", rnd, len(keep), 100 * gain)
            if gain < float(ex.get("min_improvement", 0.003)):
                self.stop_reason = f"refinement gain {gain:.4f} below min_improvement"
                break
        else:
            self.stop_reason = self.stop_reason or "completed refinement rounds"

    # -- falsifiable predictions ----------------------------------------------------------------
    def _best(self, ideas) -> float | None:
        vals = [self.value(self.results[i["id"]][1.0]) for i in ideas if i["id"] in self.results
                and 1.0 in self.results[i["id"]] and self.value(self.results[i["id"]][1.0]) is not None]
        if not vals:
            return None
        return min(vals) if M.direction(self.spec.primary) == "min" else max(vals)

    def stage_predictions(self):
        self.enter("predictions")
        props = (self.research or {}).get("proposals") or {}
        # judge each proposal as proposed: refinement children are tuning, not the hypothesis under test
        executed = [i for i in self.ideas.values() if i.get("mspec") and i.get("entry_id") and i["origin"] != "refine"]
        by_entry: dict[str, list] = {}
        for i in executed:
            by_entry.setdefault(i["entry_id"], []).append(i)
        for e in self.portfolio:
            mine = by_entry.get(e["entry_id"], [])
            own = self._best(mine)
            others = [i for i in executed if i.get("entry_id") != e["entry_id"]]
            refs = {"baseline_mean": self.baseline_value,
                    "best_data_driven": self._best([i for i in others if i["family"] == "data-driven"]),
                    "best_physics": self._best([i for i in others if i["family"] in ("physics", "physics-custom")]),
                    "best_hybrid": self._best([i for i in others if i["family"] == "hybrid"]),
                    "best_other": self._best(others)}
            for ent in self.portfolio:
                if ent.get("root_id") and ent["entry_id"] != e["entry_id"]:
                    refs[ent["root_id"]] = self._best(by_entry.get(ent["entry_id"], []))
            outcome = check_prediction(e.get("prediction"), own, self.spec.primary, refs)
            outcome.update(entry_id=e["entry_id"], proposal_id=e.get("proposal_id"), title=e["title"],
                           prediction=e.get("prediction"), family=e["family"], source=e["source"])
            self.prediction_outcomes[e["entry_id"]] = outcome
            status = {"supported": "experimentally-supported", "contradicted": "contradicted"}.get(outcome["outcome"],
                                                                                                  "inconclusive")
            outcome["new_status"] = status if e["source"] == "proposal" else "baseline"
            prop = props.get(e.get("proposal_id"))
            self.store.put(A.DECISION, {"decision": "prediction_check", **outcome}, author="scheduler",
                           stage="predictions", parents=[prop["artifact_id"]] if prop else [],
                           summary=f"{e['entry_id']} prediction {outcome['outcome']}: {outcome.get('detail', '')[:120]}")
        n = {k: sum(1 for o in self.prediction_outcomes.values() if o["outcome"] == k)
             for k in ("supported", "contradicted", "inconclusive", "untestable")}
        log.info("prediction checks: %s", n)

    # -- final ------------------------------------------------------------------------------
    def stage_final_v2(self) -> dict:
        self.enter("final")
        ranked = self.ranked(1.0)
        if not ranked:
            cands = [i for i in self.ideas.values() if i.get("mspec") and i["id"] not in self.pruned
                     and self.v2res.get(i["id"], {}).get("screen", {}).get("status") == "ok"]
            cands.sort(key=lambda i: -self.score(self.v2res[i["id"]]["screen"]))
            if not cands:
                raise RuntimeError("no successful experiment to select a final method from")
            ranked = cands[:1]
            self.decide("fallback", "no full-fidelity result; using the best screen result")
            best_val = self.value(self.v2res[ranked[0]["id"]]["screen"])
        else:
            best_val = self.value(self.results[ranked[0]["id"]][1.0])
        best = ranked[0]
        members, weights, chosen_val, kind = [best], [1.0], best_val, "single"
        ex = self.spec.execution
        if ex.get("ensemble", True) and 1.0 in self.results.get(best["id"], {}):
            self.spec.search["ensemble_max"] = int(ex.get("ensemble_max", 5))
            ens = self._greedy_ensemble(ranked)
            if ens:
                counts, ens_val = ens
                gain = M.relative_gain(M.to_score(ens_val, self.spec.primary), M.to_score(best_val, self.spec.primary))
                if len(counts) > 1 and gain >= float(ex.get("min_improvement", 0.003)):
                    members = [self.ideas[i] for i in counts]
                    weights = [float(c) for c in counts.values()]
                    chosen_val, kind = ens_val, "ensemble"
                    self.decide("select_ensemble", f"greedy ensemble of {len(members)} beats best single by {gain:.2%} on "
                                                   f"validation ({ens_val:.5g} vs {best_val:.5g})",
                                members=list(counts), weights=weights)
                else:
                    self.decide("reject_ensemble", f"ensemble gain {gain:.2%} below threshold; keeping single method")
        self.decide("select_final", f"{kind}: {[m['id'] for m in members]} ({[m['label'] for m in members]}) with val "
                                    f"{self.spec.primary}={chosen_val:.5g}", idea_id=best["id"],
                    parents=[m["hyp_artifact"] for m in members])
        self.test_access_prior = self.memory.test_access_count(self.data_key)
        if self.test_access_prior:
            log.warning("this test split has been evaluated %d time(s) before (research memory); treat test numbers "
                        "as less independent", self.test_access_prior)
        props = (self.research or {}).get("proposals") or {}
        lineage = {}
        for m in members:
            chain, pid = [], m.get("proposal_id")
            while pid and pid in props:
                chain.append(pid)
                pid = props[pid].get("parent_id") or props[pid].get("branched_from")
            lineage[m["id"]] = {"proposal_chain": chain, "entry": m.get("entry_id"), "gate": m.get("gate_artifact"),
                                "citations": sorted({c["ref_id"] for p in chain for c in props[p].get("citations", [])}),
                                "label": m["label"], "origin": m["origin"]}
        exp_id = self.engineer.new_exp_id()
        out_dir = self.store.workspace("experiments", exp_id)
        job = {"mode": "final_v2", "exp_id": exp_id, "idea_id": best["id"] if len(members) == 1 else "ensemble",
               "members_v2": [m["mspec"] for m in members], "member_ids": [m["id"] for m in members],
               "forward_access_members": [bool(m["forward_access"]) for m in members], "weights": weights,
               "task": self.spec.task, "primary": self.spec.primary, "secondary": self.spec.metric_names[1:],
               "data_path": str(self.data_path), "seed": int(self.spec.seed), "data_seed": int(self.spec.data["seed"]),
               "feature_shape": self.feature_shape, "final_dir": str(self.run_dir / "final"),
               "run_id": self.run_id.replace("-", "_"), "stage": "final", "fraction": 1.0,
               "threads": int(self.spec.hardware["threads_per_worker"]),
               "wants_gpu": any(M2.wants_gpu(m["mspec"]) for m in members), "simulator": self.sim_cfg,
               "max_forward_evals": None, "bootstrap": int(ex.get("bootstrap", 200)), "data_cache_key": self.data_key,
               "spec_hash": self.spec.hash(), "harness_version": __version__, "lineage": lineage}
        art = self.store.put(A.PLAN, {k: v for k, v in job.items() if k not in ("data_path", "members_v2")} |
                             {"members_v2": [{k: v for k, v in m["mspec"].items() if k != "code"} for m in members]},
                             author="engineer-v2", parents=[m["hyp_artifact"] for m in members], stage="final",
                             summary=f"{exp_id}: final fit of {len(members)} member(s) on train+val; test once")
        job["plan_artifact"], job["job_dir"] = art.id, str(out_dir)
        res = self.executor.run([job], "final")[0]
        if res.get("status") != "ok":
            raise RuntimeError(f"final training failed: {res.get('error')}\n{res.get('traceback', '')}")
        audit = self.critic.final_audit(res, chosen_val, [m["hyp_artifact"] for m in members])
        ck = self.store.put(A.CHECKPOINT, {"path": res["model_path"], "kind": kind, "members": [m["id"] for m in members],
                                           "weights": weights, "test": res["test"], "val_estimate": chosen_val,
                                           "labels": [m["label"] for m in members], "pipeline": "v2"},
                            author="scheduler", parents=[res["artifact_id"]] + [m["hyp_artifact"] for m in members],
                            stage="final", files=[res["model_path"]], summary=f"final method ({kind})")
        self.memory.add("test_access", title=f"test split evaluated ({self.spec.name})", run_id=self.run_id,
                        author="scheduler", context=self.mem_context, data={"data_key": self.data_key},
                        artifact_refs=[{"run_id": self.run_id, "artifact_id": ck.id}])
        log.info("FINAL %s %s | val %s=%.5g | test %s (95%% CI %s)", kind, [m["label"] for m in members],
                 self.spec.primary, chosen_val, {k: round(v, 5) for k, v in res["test"].items() if v is not None},
                 [round(x, 5) for x in res["test_bootstrap"]["ci95"]])
        return {"kind": kind, "members": [m["id"] for m in members], "labels": [m["label"] for m in members],
                "weights": weights, "val": chosen_val, "test": res["test"], "model_path": res["model_path"],
                "audit": audit, "checkpoint": ck.id, "result_artifact": res["artifact_id"], "result": res}

    # -- memory write-back ------------------------------------------------------------------------
    def stage_memory_writeback(self):
        if not getattr(self, "mem_context", None):
            return
        props = (self.research or {}).get("proposals") or {}
        mem_of = (self.research or {}).get("mem_of") or {}
        n0 = len(self.memory.records)
        for iid, idea in self.ideas.items():
            if not idea.get("mspec"):
                continue
            r = self.v2res.get(iid, {})
            full, scr = r.get("full"), r.get("screen")
            res = full or scr
            if res is None:
                continue
            ok = res.get("status") == "ok"
            oc = self.prediction_outcomes.get(idea.get("entry_id") or "", {})
            status = (oc.get("new_status") if idea["origin"] == "research" and oc else ("observed" if ok else "failed"))
            data = {"method_label": idea["label"], "method_hash": M2.method_hash(idea["mspec"]), "family": idea["family"],
                    "fidelity": "full" if full else "screen", "val": res.get("val") if ok else None,
                    "screen_val": (scr or {}).get("val"), "fit_seconds": res.get("fit_seconds"),
                    "latency_ms_per_sample": res.get("latency_ms_per_sample"), "forward_evals": res.get("forward_evals"),
                    "uncertainty": res.get("uncertainty"), "per_target": res.get("val_per_target_rel_l2"),
                    "proposal_id": idea.get("proposal_id"), "origin": idea["origin"], "pruned": self.pruned.get(iid),
                    "prediction": oc.get("prediction"), "prediction_outcome": oc.get("outcome"),
                    "prediction_detail": oc.get("detail"), "method": {k: v for k, v in idea["mspec"].items() if k != "code"}}
            if not ok:
                from .agents.critic import explain_failure
                data["explanation"] = explain_failure(res.get("error") or res.get("status"))
                data["reason"] = (res.get("error") or "")[:300]
            parents = [mem_of[idea["proposal_id"]]] if idea.get("proposal_id") in mem_of else []
            self.memory.add("observation" if ok else "failure",
                            title=f"{idea['label']}: {self.spec.primary}={self.value(res) if ok else res.get('status')}",
                            author="scheduler", run_id=self.run_id, status=status, context=self.mem_context,
                            text=f"{idea['title']} {idea.get('hypothesis', '')[:300]} "
                                 + (data.get("explanation") or ""), data=data, parents=parents,
                            artifact_refs=[{"run_id": self.run_id, "run_dir": str(self.run_dir),
                                            "artifact_id": res.get("artifact_id")}],
                            tags=[idea["family"], idea["origin"]])
        for eid, oc in self.prediction_outcomes.items():
            if oc.get("source") != "proposal" or oc["outcome"] in ("untestable",):
                continue
            pid = oc.get("proposal_id")
            p = props.get(pid) or {}
            self.memory.add("finding", title=f"{oc['outcome']}: {oc['title'][:150]}", author="scheduler",
                            run_id=self.run_id, status=oc["new_status"], context=self.mem_context,
                            text=f"{p.get('hypothesis', '')} prediction {json.dumps(oc.get('prediction'))}: {oc.get('detail', '')}",
                            data={"proposal_id": pid, "outcome": oc["outcome"], "detail": oc.get("detail"),
                                  "method_label": p.get("method_label"), "assumptions": p.get("assumptions"),
                                  "val": {self.spec.primary: oc.get("own")}},
                            parents=[mem_of.get(pid)] if pid in mem_of else [],
                            artifact_refs=[{"run_id": self.run_id, "run_dir": str(self.run_dir)}],
                            tags=["prediction", oc["outcome"]])
        if self.final:
            fr = self.final.get("result", {})
            self.memory.add("finding", title=f"best method for {self.spec.name}: {' + '.join(self.final['labels'])}",
                            author="scheduler", run_id=self.run_id, status="experimentally-supported",
                            context=self.mem_context,
                            text=f"final {self.final['kind']} {self.final['labels']} selected on validation",
                            data={"method_label": self.final["labels"][0], "val": {self.spec.primary: self.final["val"]},
                                  "test": self.final["test"], "test_bootstrap": fr.get("test_bootstrap"),
                                  "shift": fr.get("shift"), "model_path": self.final["model_path"],
                                  "members": self.final["members"]},
                            artifact_refs=[{"run_id": self.run_id, "run_dir": str(self.run_dir),
                                            "artifact_id": self.final["checkpoint"]}],
                            redact=("test", "test_bootstrap", "shift"), tags=["final"])
        self.memory_writes = len(self.memory.records) - n0
        self.store.put(A.MEMORY, {"records_written": self.memory_writes, "backend": self.memory.backend.describe(),
                                  "stats": self.memory.stats()}, author="scheduler", stage="memory",
                       summary=f"research memory: wrote {self.memory_writes} execution records "
                               f"({self.memory.stats()['records']} total)")
