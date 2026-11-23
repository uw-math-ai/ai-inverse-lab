"""Layer 3 + 4 — the overall research scheduler and the research/experiment pipeline.

Pipeline (each stage writes artifacts; the scheduler decides what advances):

  0 spec       compile + record the research specification
  1 data       simulation agent generates train/val/test (cached by config hash)
  2 baseline   trivial reference model (predict the mean / majority class)
  3 ideation   persona ideators propose methodologies in parallel
  4 prescreen  critic rejects misaligned / invalid / duplicate ideas (zero compute)
  5 screen     LeanML feasibility check on a small data fraction: crashes, cost projection,
               worse-than-baseline -> pruned; crashing agent code gets an LLM repair attempt
  6 halving    successive halving over increasing data fractions with budget-aware promotion
  7 refine     ideators build on the best full-fidelity results (mutation / LLM refinement)
               until no improvement for `patience` rounds, budget, or success
  8 final      optional greedy ensemble on validation predictions, retrain on train+val,
               single evaluation on the untouched test split, critic audit, model export
  9 report     report.md, leaderboard.csv, lineage of the final model
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import artifacts as A
from . import methods as ML
from . import metrics as M
from .agents import CriticAgent, EngineerAgent, IdeationAgent, ReporterAgent, SimulationAgent
from .executor import Executor
from .llm import make_llm
from .resources import ResourceManager

log = logging.getLogger("mlharness")


def setup_logging(run_dir: Path, verbose: bool = True):
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        if getattr(h, "_mlh", False):
            log.removeHandler(h)
            h.close()
    fh = logging.FileHandler(Path(run_dir) / "harness.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh._mlh = True
    log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        sh._mlh = True
        log.addHandler(sh)
    log.propagate = False


class ResearchScheduler:
    def __init__(self, spec, run_dir, llm=None, resume: bool = False, verbose: bool = True,
                 python_exe: str | None = None):
        self.spec = spec
        self.run_dir = Path(run_dir)
        self.store = A.FileSystemStore(self.run_dir, resume=resume)
        setup_logging(self.run_dir, verbose)
        hw = spec.hardware
        self.resources = ResourceManager(spec.budget["compute_seconds"], spec.budget["wall_seconds"],
                                         hw["workers"], hw["gpu_devices"], spec.budget["llm_calls"])
        self.llm = llm if llm is not None else make_llm(spec.llm, workdir=self.run_dir / "llm_scratch")
        self.available = ML.methods_for(spec.task, spec.constraints)
        self.python_exe = python_exe
        self.ideas: dict[str, dict] = {}
        self.results: dict[str, dict[float, dict]] = {}
        self.pruned: dict[str, str] = {}
        self.critiques: dict[str, str] = {}
        self.outcomes: dict[str, str] = {}
        self.hashes: set[str] = set()
        self.roster: list[dict] = []
        self.ideators: dict[str, IdeationAgent] = {}
        self.funnel: dict[str, int] = {}
        self.baseline_value = None
        self.final: dict | None = None
        self.stop_reason = None
        self.stage = "init"
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.run_id = f"{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    # -- helpers -------------------------------------------------------------------------
    def new_idea_id(self) -> str:
        with self._lock:
            return f"idea-{next(self._ids):03d}"

    def spawn(self, cls, name: str, **kw):
        agent = cls(name, self.spec, self.store, self.llm, self.resources, **kw)
        info = {"name": name, "role": cls.role, "persona": kw.get("persona"), "llm": self.llm.describe()}
        self.roster.append(info)
        self.store.put(A.AGENT, info, author="scheduler", summary=f"spawned {cls.role} agent {name}",
                       stage=self.stage)
        return agent

    def decide(self, decision: str, reason: str, idea_id=None, parents=(), **data):
        if idea_id and decision in ("not_promoted", "budget_cut", "prune", "promote"):
            self.outcomes[idea_id] = f"{decision}: {reason}"
        self.store.put(A.DECISION, {"decision": decision, "reason": reason, **data}, author="scheduler",
                       idea_id=idea_id, parents=parents, stage=self.stage,
                       summary=f"{decision}{' ' + idea_id if idea_id else ''}: {reason}"[:200])

    def value(self, res) -> float | None:
        return (res.get("val") or {}).get(self.spec.primary) if res and res.get("status") == "ok" else None

    def score(self, res) -> float:
        return M.to_score(self.value(res), self.spec.primary)

    def ranked(self, fraction: float, ids=None) -> list[dict]:
        out = []
        for iid in (ids if ids is not None else self.results):
            idea = self.ideas[iid]
            res = self.results.get(iid, {}).get(fraction)
            if idea["persona"] == "baseline" or iid in self.pruned or res is None:
                continue
            if math.isfinite(self.score(res)):
                out.append(idea)
        return sorted(out, key=lambda i: self.score(self.results[i["id"]][fraction]), reverse=True)

    def best_full(self):
        r = self.ranked(1.0)
        return (r[0], self.results[r[0]["id"]][1.0]) if r else (None, None)

    def status(self, **extra):
        best, res = self.best_full()
        st = {"run_id": self.run_id, "stage": self.stage, "ideas": len(self.ideas), "pruned": len(self.pruned),
              "best": {"idea": best["id"], "method": best["config"]["method"],
                       self.spec.primary: self.value(res)} if best else None,
              "resources": self.resources.snapshot(), "funnel": self.funnel, "updated": time.time(), **extra}
        (self.run_dir / "status.json").write_text(json.dumps(st, indent=1, default=str))

    def enter(self, stage):
        self.stage = stage
        log.info("=== stage: %s | compute used %.0f/%.0fs | wall %.0fs", stage, self.resources.used,
                 self.resources.total, self.resources.wall_elapsed())
        self.status()

    def context(self) -> dict:
        lb = []
        for iid, by_f in self.results.items():
            idea = self.ideas[iid]
            ok = [f for f, r in by_f.items() if r.get("status") == "ok"]
            if idea["persona"] == "baseline" or not ok:
                continue
            f = max(ok)
            lb.append({"idea_id": iid, "method": idea["config"]["method"], "features": idea["config"]["features"],
                       "params": idea["config"]["params"], "fraction": f, "value": self.value(by_f[f]),
                       "_s": (f, self.score(by_f[f]))})
        lb.sort(key=lambda r: r["_s"], reverse=True)
        for r in lb:
            r.pop("_s")
        fails = [{"idea_id": i, "method": self.ideas[i]["config"]["method"], "reason": r}
                 for i, r in self.pruned.items()]
        return {"leaderboard": lb, "failures": fails, "baseline": self.baseline_value,
                "proposed": [i["title"] for i in self.ideas.values()], "budget": self.resources.snapshot(),
                "hashes": self.hashes,
                "scores": {iid: self.value(max(v.items())[1]) for iid, v in self.results.items() if v}}

    def _record(self, idea, res, stage):
        f = float(res.get("fraction") or 1.0)
        ref = self.ranked(f)
        ref_value = self.value(self.results[ref[0]["id"]][f]) if ref else None
        self.results.setdefault(idea["id"], {})[f] = res
        if ref_value is None or self.score(res) > M.to_score(ref_value, self.spec.primary):
            ref_value = self.value(res)
        crit = self.critic.review_result(idea, res, self.baseline_value, res["artifact_id"], stage,
                                         reference_value=ref_value)
        if crit.get("explanation") and crit["explanation"] != "ok":
            self.critiques[idea["id"]] = crit["explanation"]
        return crit

    def prune(self, idea, reason, parents=()):
        self.pruned[idea["id"]] = reason
        self.decide("prune", reason, idea_id=idea["id"], parents=parents)

    def should_stop(self, stage) -> str | None:
        if self.resources.wall_remaining() <= 0:
            return "wall-clock budget exhausted"
        if self.resources.exhausted(stage):
            return f"compute budget for stage '{stage}' exhausted"
        best, res = self.best_full()
        if best and self.spec.search.get("stop_on_success") and self.spec.meets_success(
                (res.get("val") or {}).get(self.spec.success["metric"])):
            return "success criterion met on validation"
        return None

    # -- main loop -----------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        error = None
        try:
            self.stage_spec()
            self.stage_data()
            self.stage_baseline()
            ideas = self.stage_ideation()
            approved = self.stage_prescreen(ideas, "prescreen")
            survivors = self.stage_screen(approved)
            self.stage_halving(survivors)
            self.stage_refine()
            self.final = self.stage_final()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.exception("research run failed")
            self.store.put(A.FAILURE, {"error": error, "stage": self.stage, "source": "scheduler"},
                           author="scheduler", stage=self.stage, summary=f"run failed in {self.stage}: {error[:120]}")
        finally:
            self.enter("report")
            try:
                self.reporter.write(self)
            except Exception:
                log.exception("report generation failed")
            self.stage = "done" if error is None else "failed"
            self.status(error=error, elapsed=time.time() - t0)
        if error:
            raise RuntimeError(error)
        return {"run_dir": str(self.run_dir), "final": self.final, "resources": self.resources.snapshot()}

    # -- stages --------------------------------------------------------------------------
    def stage_spec(self):
        self.enter("spec")
        self.spec_art = self.store.put(A.SPEC, self.spec.to_dict(), author="scheduler", stage="spec",
                                       summary=f"research spec {self.spec.name} ({self.spec.hash()})")
        (self.run_dir / "spec.json").write_text(json.dumps(self.spec.to_dict(), indent=2, default=str))
        log.info("objective: %s", self.spec.problem.strip()[:200])
        log.info("llm backend: %s | workers: %d x %d threads | gpus: %s", self.llm.describe(),
                 self.spec.hardware["workers"], self.spec.hardware["threads_per_worker"],
                 self.spec.hardware["gpu_devices"] or "none")
        self.reporter = self.spawn(ReporterAgent, "reporter")

    def stage_data(self):
        self.enter("data")
        self.sim_agent = self.spawn(SimulationAgent, "simulator")
        self.data_path, self.data_art = self.sim_agent.generate(self.run_dir)
        d = self.data_art.data
        self.feature_shape = d["feature_shape"]
        self.feature_sets = d["feature_sets"]
        self.n_features = d["stats"]["n_features"]
        with np.load(self.data_path) as z:
            self.y_val = z["y_val"]
        self.executor = Executor(self.spec, self.store, self.resources, d["cache_key"], self.python_exe)
        self.critic = self.spawn(CriticAgent, "critic", available=self.available,
                                 feature_sets=self.feature_sets, n_features=self.n_features)
        self.engineer = self.spawn(EngineerAgent, "engineer", data_path=self.data_path,
                                   feature_shape=self.feature_shape)

    def stage_baseline(self):
        self.enter("baseline")
        idea = {"id": self.new_idea_id(), "title": "mean / majority baseline", "persona": "baseline",
                "hypothesis": "Reference: predicting the training mean. Any useful method must beat it.",
                "origin": "scheduler", "round": 0, "parent": None,
                "config": {"method": "mean_baseline", "params": {}, "features": "raw", "scaler": "none", "pca": 0}}
        idea["hyp_artifact"] = self.store.put(A.HYPOTHESIS, idea, author="scheduler", idea_id=idea["id"],
                                              stage="baseline", summary=idea["title"]).id
        self.ideas[idea["id"]] = idea
        res = self.executor.run([self.engineer.plan(idea, 1.0, "baseline")], "baseline")[0]
        self.results.setdefault(idea["id"], {})[1.0] = res
        self.baseline_value = self.value(res)
        if self.baseline_value is None:
            raise RuntimeError(f"baseline failed: {res.get('error')}\n{res.get('traceback', '')}")
        log.info("baseline val %s = %.5g", self.spec.primary, self.baseline_value)

    def stage_ideation(self) -> list[dict]:
        self.enter("ideation")
        s = self.spec.search
        personas = list(s["ideators"])
        cap = int(self.spec.budget["max_ideas"])
        per = max(1, min(int(s["ideas_per_ideator"]), cap // max(1, len(personas))))
        sim = self.sim_agent.simulator
        for k, p in enumerate(personas):
            self.ideators[p] = self.spawn(IdeationAgent, f"ideator-{p}", persona=p, available=self.available,
                                          feature_sets=self.feature_sets,
                                          suggested_features=sim.suggested_feature_sets,
                                          n_features=self.n_features, new_id=self.new_idea_id,
                                          seed=self.spec.seed * 1000 + k,
                                          base_parents=[self.spec_art.id, self.data_art.id])
        ctx = self.context()
        with ThreadPoolExecutor(max_workers=len(personas)) as pool:   # independent, parallel ideation
            batches = list(pool.map(lambda p: self.ideators[p].propose(per, ctx), personas))
        ideas = [i for b in batches for i in b]
        for i in ideas:
            self.ideas[i["id"]] = i
        self.funnel["proposed"] = len(ideas)
        log.info("%d ideas proposed by %d ideators", len(ideas), len(personas))
        return ideas

    def stage_prescreen(self, ideas: list[dict], stage: str) -> list[dict]:
        self.enter(stage) if stage == "prescreen" else None
        approved = [i for i in ideas if self.critic.review_proposal(i, self.hashes, stage)]
        for i in ideas:
            if i not in approved:
                rev = self.store.latest(A.CRITIQUE, idea_id=i["id"])
                self.prune(i, "critic: " + "; ".join(rev.data["reasons"]) if rev else "critic rejected")
        flagged = self.critic.llm_batch_review(approved, stage)
        for i in list(approved):
            if i["id"] in flagged:
                approved.remove(i)
                self.prune(i, f"critic (LLM): {flagged[i['id']]}")
        for i in approved:
            self.engineer.materialize(i)
        self.funnel[f"{stage}_approved"] = self.funnel.get(f"{stage}_approved", 0) + len(approved)
        log.info("%s: %d/%d ideas approved", stage, len(approved), len(ideas))
        return approved

    def _project_full_cost(self, idea, res, f_from, f_to=1.0) -> float:
        m = ML.LIBRARY.get(idea["config"]["method"])
        alpha = m.scaling if m else 1.0
        fit = float(res.get("fit_seconds") or 0.0)
        overhead = max(0.0, float(res.get("wall_seconds") or 0.0) - fit)
        return fit * (f_to / f_from) ** alpha + overhead

    def stage_screen(self, approved: list[dict]) -> list[dict]:
        self.enter("screen")
        f = float(self.spec.search["screen_fraction"])
        results = self.executor.run([self.engineer.plan(i, f, "screen") for i in approved], "screen")
        # repair loop for agent-written code
        retry = []
        for idea, res in zip(approved, results):
            if res.get("status") == "error" and idea["config"]["method"] == "custom":
                for _ in range(int(self.spec.search.get("repair_attempts", 1))):
                    if self.engineer.repair(idea, res.get("error", ""), res.get("traceback", "")):
                        retry.append(idea)
                    break
        if retry:
            rres = self.executor.run([self.engineer.plan(i, f, "screen") for i in retry], "screen")
            fixed = {i["id"]: r for i, r in zip(retry, rres)}
            results = [fixed.get(i["id"], r) for i, r in zip(approved, results)]
        limit = float(self.spec.budget["experiment_timeout"])
        survivors = []
        for idea, res in zip(approved, results):
            self._record(idea, res, "screen")
            if res.get("status") != "ok":
                from .agents.critic import explain_failure
                self.prune(idea, f"feasibility check failed: {explain_failure(res.get('error') or res.get('status'))}",
                           parents=[res["artifact_id"]])
                continue
            proj = self._project_full_cost(idea, res, f)
            idea["projected_full_seconds"] = round(proj, 2)
            if proj > limit:
                self.prune(idea, f"projected full-data cost {proj:.0f}s exceeds the {limit:.0f}s per-experiment limit",
                           parents=[res["artifact_id"]])
            elif M.relative_gain(self.score(res), M.to_score(self.baseline_value, self.spec.primary)) < \
                    float(self.spec.search.get("baseline_margin", 0.02)):
                self.prune(idea, f"not meaningfully better than the mean baseline at {f:.0%} data "
                                 f"({self.value(res):.4g} vs {self.baseline_value:.4g})", parents=[res["artifact_id"]])
            else:
                survivors.append(idea)
        survivors.sort(key=lambda i: self.score(self.results[i["id"]][f]), reverse=True)
        self.funnel["screen_survivors"] = len(survivors)
        log.info("screen: %d/%d ideas survive the feasibility check", len(survivors), len(approved))
        return survivors

    def _budget_trim(self, ranked: list[dict], costs: dict, stage: str) -> list[dict]:
        keep = list(ranked)
        avail = self.resources.stage_remaining(stage)
        while len(keep) > 1 and sum(costs[i["id"]] for i in keep) > avail:
            cut = keep.pop()
            self.decide("budget_cut", f"not promoted: estimated {costs[cut['id']]:.0f}s does not fit the remaining "
                                      f"{avail:.0f}s of the {stage} budget", idea_id=cut["id"])
        return keep

    def stage_halving(self, survivors: list[dict]):
        self.enter("halving")
        s = self.spec.search
        cur, prev_f = survivors, float(s["screen_fraction"])
        for f in s["rungs"]:
            if not cur:
                break
            n_keep = max(min(len(cur), int(s["min_full"])), math.ceil(len(cur) / float(s["eta"])))
            promoted, dropped = cur[:n_keep], cur[n_keep:]
            for i in dropped:
                self.decide("not_promoted", f"ranked below the top {n_keep} at fraction {prev_f:.2f}", idea_id=i["id"])
            costs = {i["id"]: self._project_full_cost(i, self.results[i["id"]][prev_f], prev_f, f) for i in promoted}
            limit = float(self.spec.budget["experiment_timeout"])
            for i in [i for i in promoted if costs[i["id"]] > limit and len(promoted) > 1]:
                promoted.remove(i)
                self.decide("not_promoted", f"projected {costs[i['id']]:.0f}s at fraction {f:.2f} exceeds the "
                                            f"{limit:.0f}s per-experiment limit", idea_id=i["id"])
            promoted = self._budget_trim(promoted, costs, "halving")
            for i in promoted:
                self.decide("promote", f"to fraction {f:.2f} (est. {costs[i['id']]:.1f}s)", idea_id=i["id"])
            results = self.executor.run([self.engineer.plan(i, f, "halving") for i in promoted], "halving")
            ok = []
            for idea, res in zip(promoted, results):
                self._record(idea, res, "halving")
                if res.get("status") == "ok":
                    ok.append(idea)
                elif res.get("status") != "skipped_budget":
                    self.prune(idea, f"failed at fraction {f:.2f}: {res.get('error')}", parents=[res["artifact_id"]])
            cur = sorted(ok, key=lambda i: self.score(self.results[i["id"]][f]), reverse=True)
            self.funnel[f"rung_{f:.2f}"] = len(cur)
            prev_f = f
            log.info("halving rung %.2f: %d evaluated, best %s", f, len(cur),
                     f"{cur[0]['id']} {self.value(self.results[cur[0]['id']][f]):.5g}" if cur else "-")
            if self.resources.exhausted("halving") and f < 1.0:
                log.info("halving budget exhausted before the full-fidelity rung")
                break

    def stage_refine(self):
        self.enter("refine")
        s = self.spec.search
        no_improve = 0
        for rnd in range(1, int(s["refine_rounds"]) + 1):
            reason = self.should_stop("refine")
            if reason is None and len(self.ideas) >= int(self.spec.budget["max_ideas"]):
                reason = "max_ideas reached"
            if reason:
                self.stop_reason = reason
                self.decide("stop_refinement", reason)
                break
            top = self.ranked(1.0)[: int(s["refine_top_k"])]
            if not top:
                break
            best_before = self.score(self.results[top[0]["id"]][1.0])
            groups: dict[str, list] = {}
            for idea in top:
                persona = idea["persona"] if idea["persona"] in self.ideators else next(iter(self.ideators))
                groups.setdefault(persona, []).append({"idea": idea, "value": self.value(self.results[idea["id"]][1.0]),
                                                       "critique": self.critiques.get(idea["id"])})
            ctx = self.context()
            n_child = int(s["refine_children"])
            with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                batches = list(pool.map(lambda kv: self.ideators[kv[0]].refine(kv[1], n_child, ctx, rnd), groups.items()))
            room = int(self.spec.budget["max_ideas"]) - len(self.ideas)
            children = [c for b in batches for c in b][: max(0, room)]
            for c in children:
                self.ideas[c["id"]] = c
            self.funnel["refine_proposed"] = self.funnel.get("refine_proposed", 0) + len(children)
            approved = self.stage_prescreen(children, "refine")
            costs = {}
            for c in approved:
                p = self.ideas.get(c["parent"])
                pr = self.results.get(c["parent"], {}).get(1.0) if p else None
                same = p is not None and p["config"]["method"] == c["config"]["method"]
                costs[c["id"]] = float(pr.get("wall_seconds", 30)) if (pr and same) else \
                    float(np.median([r.get("wall_seconds", 30) for i in top for r in [self.results[i["id"]][1.0]]]))
            approved = self._budget_trim(approved, costs, "refine")
            results = self.executor.run([self.engineer.plan(c, 1.0, "refine") for c in approved], "refine")
            for c, res in zip(approved, results):
                self._record(c, res, "refine")
                if res.get("status") not in ("ok", "skipped_budget"):
                    self.prune(c, f"refinement failed: {res.get('error')}", parents=[res["artifact_id"]])
            best_idea, best_res = self.best_full()
            gain = M.relative_gain(self.score(best_res), best_before)
            log.info("refine round %d: %d children, best %s %s=%.5g (gain %.2f%%)", rnd, len(approved),
                     best_idea["id"], self.spec.primary, self.value(best_res), 100 * gain)
            self.decide("refine_round", f"round {rnd}: best {best_idea['id']} gain {gain:.4f}",
                        round=rnd, gain=gain, best=best_idea["id"])
            if gain < float(s["min_improvement"]):
                no_improve += 1
                if no_improve >= int(s["patience"]):
                    self.stop_reason = f"no relative improvement >= {s['min_improvement']} for {no_improve} round(s)"
                    self.decide("stop_refinement", self.stop_reason)
                    break
            else:
                no_improve = 0
        else:
            self.stop_reason = self.stop_reason or "completed all refinement rounds"
            self.decide("stop_refinement", self.stop_reason)

    def _greedy_ensemble(self, ranked: list[dict]):
        """Caruana-style greedy selection with replacement on validation predictions (no training)."""
        from .export import combine
        s = self.spec.search
        pool = ranked[: int(s.get("ensemble_max", 5))]
        preds = {}
        for i in pool:
            p = self.results[i["id"]][1.0].get("val_pred_path")
            if p and Path(p).exists():
                preds[i["id"]] = np.load(p)
        if len(preds) < 2:
            return None
        chosen: list[str] = []
        best = -math.inf
        for _ in range(2 * len(preds)):
            cand = None
            for iid in preds:
                trial = chosen + [iid]
                sc = M.to_score(M.compute(self.spec.primary, self.y_val,
                                          combine([preds[t] for t in trial], [1.0] * len(trial), self.spec.task)),
                                self.spec.primary)
                if sc > best + 1e-12:
                    best, cand = sc, iid
            if cand is None:
                break
            chosen.append(cand)
        counts = {iid: chosen.count(iid) for iid in dict.fromkeys(chosen)}
        val = -best if M.direction(self.spec.primary) == "min" else best
        return counts, val

    def stage_final(self) -> dict:
        self.enter("final")
        ranked = self.ranked(1.0)
        if not ranked:
            fr = sorted({f for v in self.results.values() for f in v}, reverse=True)
            for f in fr:
                ranked = self.ranked(f)
                if ranked:
                    self.decide("fallback", f"no full-fidelity result; using best at fraction {f:.2f}")
                    break
        if not ranked:
            raise RuntimeError("no successful experiment to select a final model from")
        best = ranked[0]
        best_f = max(f for f, r in self.results[best["id"]].items() if r.get("status") == "ok")
        best_val = self.value(self.results[best["id"]][best_f])
        members, weights, chosen_val, kind = [best], [1.0], best_val, "single"
        if self.spec.search.get("ensemble") and best_f == 1.0:
            ens = self._greedy_ensemble(ranked)
            if ens:
                counts, ens_val = ens
                gain = M.relative_gain(M.to_score(ens_val, self.spec.primary), M.to_score(best_val, self.spec.primary))
                if len(counts) > 1 and gain >= float(self.spec.search["min_improvement"]):
                    members = [self.ideas[i] for i in counts]
                    weights = [float(c) for c in counts.values()]
                    chosen_val, kind = ens_val, "ensemble"
                    self.decide("select_ensemble", f"greedy ensemble of {len(members)} beats best single by "
                                f"{gain:.2%} on validation ({ens_val:.5g} vs {best_val:.5g})",
                                members=list(counts), weights=weights)
                else:
                    self.decide("reject_ensemble", f"ensemble gain {gain:.2%} below threshold; keeping single model")
                log.info("greedy ensemble %s: val %s=%.5g vs best single %.5g (gain %.2f%%) -> %s", dict(counts),
                         self.spec.primary, ens_val, best_val, 100 * gain, kind)
        self.decide("select_final", f"{kind}: {[m['id'] for m in members]} with val {self.spec.primary}={chosen_val:.5g}",
                    idea_id=best["id"], parents=[m["hyp_artifact"] for m in members])
        final_dir = self.run_dir / "final"
        job = self.engineer.plan_final(members, weights, final_dir, self.run_id.replace("-", "_"))
        res = self.executor.run([job], "final")[0]
        if res.get("status") != "ok":
            raise RuntimeError(f"final training failed: {res.get('error')}\n{res.get('traceback', '')}")
        audit = self.critic.final_audit(res, chosen_val, [m["hyp_artifact"] for m in members])
        ck = self.store.put(A.CHECKPOINT, {"path": res["model_path"], "kind": kind,
                                           "members": [m["id"] for m in members], "weights": weights,
                                           "test": res["test"], "val_estimate": chosen_val},
                            author="scheduler", parents=[res["artifact_id"]] + [m["hyp_artifact"] for m in members],
                            stage="final", files=[res["model_path"]], summary=f"final model ({kind})")
        log.info("FINAL %s %s | val %s=%.5g | test %s", kind, [m["id"] for m in members], self.spec.primary,
                 chosen_val, {k: round(v, 5) for k, v in res["test"].items() if v is not None})
        return {"kind": kind, "members": [m["id"] for m in members], "weights": weights, "val": chosen_val,
                "test": res["test"], "model_path": res["model_path"], "audit": audit, "checkpoint": ck.id,
                "result_artifact": res["artifact_id"]}
