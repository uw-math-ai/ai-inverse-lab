"""The research loop: four parallel researchers <-> shared critic, for a bounded number of rounds.

    round 1      : all researchers propose in parallel (memory + literature first)
    critique r   : the shared critic reviews every active proposal (+ portfolio-level review)
    pilots r     : optional cheap discriminating experiments the critic asked for (budgeted)
    stop check   : quality criteria (after min_rounds) | max_rounds | research wall time | LLM budget
    round r+1    : all researchers (even those with nothing active) respond to the feedback in
                   parallel: revise / defend / abandon / branch, optionally one new proposal

Halting is deterministic given the critic's output; whatever is active when the loop stops is
advanced, with the critic's unresolved points attached. Every proposal version, critique, response,
abandonment (with reason) and disagreement is written to the artifact store AND research memory.
"""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .. import artifacts as A
from ..agents.base import log
from .proposals import normalize_proposal


class ResearchLoop:
    def __init__(self, *, spec, store, memory, literature, researchers: dict, critic, resources, run_id: str,
                 mem_context: dict, base_parents=(), pilot_fn=None, uses_llm: bool = False):
        self.spec = spec
        self.rcfg = spec.research
        self.store = store
        self.memory = memory
        self.literature = literature
        self.researchers = researchers
        self.critic = critic
        self.resources = resources
        self.run_id = run_id
        self.mem_context = mem_context
        self.base_parents = [p for p in base_parents if p]
        self.pilot_fn = pilot_fn
        self.uses_llm = uses_llm
        self.proposals: dict[str, dict] = {}
        self.art_of: dict[str, str] = {}
        self.mem_of: dict[str, str] = {}
        self.active: dict[str, str] = {}        # root_id -> latest active proposal id
        self.abandoned: list[dict] = []
        self.rounds: list[dict] = []
        self.critiques: list[dict] = []
        self.pilots: dict[str, dict] = {}
        self.stop_reason: str | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.t0 = time.time()

    # -- registration -------------------------------------------------------------------------
    def _pid(self) -> str:
        with self._lock:
            return f"P-{next(self._ids):03d}"

    def _register(self, raw, researcher, round_idx, origin, action, parent=None, new_root=False) -> dict:
        p = normalize_proposal(raw, pid=self._pid(), author=researcher.name, persona=researcher.persona,
                               round_idx=round_idx, origin=origin, parent=parent, action=action)
        if new_root:
            p["root_id"] = p["id"]
            p["branched_from"] = parent["id"] if parent else None
        p["memory_refs"] = list(researcher.memory_refs)[:12]
        parents = ([self.art_of[parent["id"]]] if parent else []) + self.base_parents
        art = self.store.put(A.PROPOSAL, p, author=researcher.name, stage="research", parents=parents,
                             summary=f"{p['id']} v{p['version']} [{action}] {researcher.persona}: {p['title'][:90]}")
        p["artifact_id"] = art.id
        self.art_of[p["id"]] = art.id
        mtype ="hypothesis" if action in ("propose", "branch") else "revision"
        mid = self.memory.add(mtype, title=p["title"], author=researcher.name, run_id=self.run_id, status=p["status"],
                              text=f"{p['hypothesis']} {p['motivation']} {' '.join(p['assumptions'])}",
                              context=self.mem_context,
                              data={"proposal_id": p["id"], "root_id": p["root_id"], "version": p["version"],
                                    "action": action, "method_label": p["method_label"], "family": p["family"],
                                    "method": {k: v for k, v in (p["method"] or {}).items() if k != "code"},
                                    "assumptions": p["assumptions"], "prediction": p["prediction"],
                                    "falsifiable_test": p["falsifiable_test"], "if_fails": p["if_fails"],
                                    "response_to_critic": p["response_to_critic"]},
                              parents=[self.mem_of[parent["id"]]] if parent and parent["id"] in self.mem_of else [],
                              citations=[c["ref_id"] for c in p["citations"]],
                              artifact_refs=[{"run_id": self.run_id, "artifact_id": art.id}],
                              tags=[p["family"], researcher.persona])
        self.mem_of[p["id"]] = mid
        self.proposals[p["id"]] = p
        return p

    def active_list(self) -> list[dict]:
        return [self.proposals[pid] for pid in self.active.values()]

    def _apply(self, researcher, outputs, round_idx) -> list[dict]:
        actions = []
        cap = int(self.rcfg.get("max_active", 12))
        for o in outputs or []:
            act, target = o.get("action"), o.get("target")
            parent = self.proposals.get(target) if target else None
            rec = {"researcher": researcher.name, "persona": researcher.persona, "action": act, "target": target,
                   "origin": o.get("origin"), "response": (o.get("response_to_critic") or "")[:600]}
            if act == "propose" or (act == "branch" and parent is None):
                if len(self.active) >= cap:
                    rec["note"] = "ignored: active-proposal cap reached"
                else:
                    raw = dict(o.get("raw") or {})
                    p = self._register(raw, researcher, round_idx, o.get("origin", "?"), "propose")
                    self.active[p["root_id"]] = p["id"]
                    rec["new_id"] = p["id"]
            elif parent is None:
                rec["note"] = "ignored: unknown target proposal"
            elif act == "revise":
                raw = dict(o.get("raw") or {})
                raw["response_to_critic"] = o.get("response_to_critic", "")
                p = self._register(raw, researcher, round_idx, o.get("origin", "?"), "revise", parent=parent)
                self.active[p["root_id"]] = p["id"]
                rec["new_id"] = p["id"]
            elif act == "branch":
                raw = dict(o.get("raw") or {})
                raw["response_to_critic"] = o.get("response_to_critic", "")
                if len(self.active) < cap:
                    p = self._register(raw, researcher, round_idx, o.get("origin", "?"), "branch", parent=parent,
                                       new_root=True)
                    self.active[p["root_id"]] = p["id"]
                    rec["new_id"] = p["id"]
                else:
                    rec["note"] = "branch ignored: active-proposal cap reached"
            elif act == "abandon":
                reason = o.get("abandon_reason") or o.get("response_to_critic") or "no reason given"
                self.active.pop(parent["root_id"], None)
                self.abandoned.append({"proposal_id": parent["id"], "root_id": parent["root_id"], "title": parent["title"],
                                       "author": researcher.name, "round": round_idx, "reason": reason,
                                       "method_label": parent["method_label"]})
                art = self.store.put(A.DECISION, {"decision": "abandon_proposal", "reason": reason,
                                                  "proposal_id": parent["id"]}, author=researcher.name,
                                     stage="research", parents=[self.art_of.get(parent["id"]) or parent.get("artifact_id")],
                                     summary=f"{researcher.name} abandoned {parent['id']}: {reason[:120]}")
                self.memory.add("abandoned", title=f"abandoned: {parent['title']}", author=researcher.name,
                                run_id=self.run_id, status="abandoned", text=reason, context=self.mem_context,
                                data={"proposal_id": parent["id"], "reason": reason, "method_label": parent["method_label"],
                                      "round": round_idx},
                                parents=[self.mem_of.get(parent["id"])], artifact_refs=[{"artifact_id": art.id}])
            else:    # defend
                self.store.put(A.DECISION, {"decision": "defend_proposal", "proposal_id": parent["id"],
                                            "response": o.get("response_to_critic", "")}, author=researcher.name,
                               stage="research", parents=[parent.get("artifact_id")],
                               summary=f"{researcher.name} defended {parent['id']}: {(o.get('response_to_critic') or '')[:100]}")
                parent.setdefault("defenses", []).append({"round": round_idx, "response": o.get("response_to_critic", "")})
            actions.append(rec)
        return actions

    # -- the loop -----------------------------------------------------------------------------
    def _budget_text(self) -> str:
        s = self.resources.snapshot()
        return (f"LLM calls left {self.resources.llm_remaining('research')}; literature queries left "
                f"{self.literature.remaining() if self.literature is not None else 0}; execution compute "
                f"{s['compute_budget_s']:.0f} worker-seconds; per-experiment limit "
                f"{self.spec.budget['experiment_timeout']}s")

    def _parallel(self, fn):
        names = list(self.researchers)
        with ThreadPoolExecutor(max_workers=len(names), thread_name_prefix="researcher") as pool:
            futs = {n: pool.submit(fn, self.researchers[n]) for n in names}
            out = {}
            for n, f in futs.items():
                try:
                    out[n] = f.result()
                except Exception as exc:          # one researcher failing never stops the others
                    log.exception("researcher %s failed", n)
                    self.store.put(A.FAILURE, {"source": "researcher", "error": f"{type(exc).__name__}: {exc}"},
                                   author=n, stage="research", summary=f"{n} failed: {exc}"[:160])
                    out[n] = []
        return out

    def _others(self):
        return [{"id": p["id"], "author": p["author"], "family": p["family"], "method_label": p["method_label"],
                 "title": p["title"]} for p in self.active_list()]

    def _record_critique(self, crit, round_idx):
        art = self.store.put(A.FEEDBACK, crit, author=self.critic.name, stage="research",
                             parents=[self.proposals[pid]["artifact_id"] for pid in crit["per_proposal"]],
                             summary=f"round {round_idx} critique: {len(crit['per_proposal'])} proposals, mean quality "
                                     f"{crit['portfolio']['mean_quality']}, converged={crit['portfolio'].get('converged')}")
        crit["artifact_id"] = art.id
        for pid, fb in crit["per_proposal"].items():
            p = self.proposals[pid]
            text = " ".join(str(x) for k in ("challenges", "changed_assumptions", "questions", "counterexamples",
                                             "revision_directions", "blocking") for x in fb.get(k) or [])
            self.memory.add("critique", title=f"critique r{round_idx} of {p['title'][:120]}", author=self.critic.name,
                            run_id=self.run_id, status="recorded", text=text, context=self.mem_context,
                            data={"proposal_id": pid, "round": round_idx, "quality": fb.get("quality"),
                                  "blocking": fb.get("blocking"), "challenges": fb.get("challenges")[:6],
                                  "changed_assumptions": fb.get("changed_assumptions")[:6],
                                  "method_label": p["method_label"]},
                            parents=[self.mem_of.get(pid)], artifact_refs=[{"artifact_id": art.id}])
        for d in crit["portfolio"].get("disagreements") or []:
            self.memory.add("disagreement", title=f"unresolved: {d.get('issue', '')[:150]}", author=self.critic.name,
                            run_id=self.run_id, status="recorded", text=d.get("issue", ""), context=self.mem_context,
                            data=d, artifact_refs=[{"artifact_id": art.id}])

    def _maybe_pilots(self, crit, round_idx):
        if not (self.pilot_fn and self.rcfg.get("pilot_tests", True)):
            return {}
        want = []
        for pid, fb in crit["per_proposal"].items():
            if fb.get("blocking"):
                continue
            if any(isinstance(t, dict) and t.get("pilot") for t in fb.get("suggested_tests") or []):
                want.append(self.proposals[pid])
        want = [p for p in want if p["id"] not in self.pilots][: int(self.rcfg.get("max_pilots_per_round", 4))]
        if not want:
            return {}
        try:
            res = self.pilot_fn(want, round_idx) or {}
        except Exception as exc:
            log.exception("pilot experiments failed")
            res = {p["id"]: {"status": "error", "error": f"{type(exc).__name__}: {exc}"} for p in want}
        self.pilots.update(res)
        return res

    def _should_stop(self, crit, r) -> str | None:
        port = crit["portfolio"]
        if not self.active:
            return "all proposals were abandoned"
        if r >= int(self.rcfg.get("max_rounds", 3)):
            return f"iteration cap reached (max_rounds={self.rcfg.get('max_rounds', 3)}); unresolved issues are documented"
        if r >= int(self.rcfg.get("min_rounds", 2)) and port.get("converged") and port.get("all_meet_threshold"):
            return (f"quality criteria met after {r} rounds (every active proposal >= "
                    f"{self.rcfg.get('quality_threshold')} quality, no blocking issues, critic converged)")
        if time.time() - self.t0 > float(self.rcfg.get("wall_seconds", 1800)):
            return "research wall-clock budget exhausted"
        if self.uses_llm:
            need = len(self.researchers) + 1
            if self.resources.llm_remaining("research") < need + int(self.rcfg.get("llm_reserve", 2)):
                return f"research LLM budget exhausted (needs {need} calls for another round)"
        return None

    def run(self) -> dict:
        self.resources.phase = "research"
        self.resources.llm_phase_caps["research"] = int(self.rcfg.get("llm_calls", 30))
        try:
            r = 1
            outs = self._parallel(lambda ag: ag.propose({"round": 1, "budget_text": self._budget_text()}))
            self.rounds.append({"round": 1, "actions": [a for n in self.researchers
                                                        for a in self._apply(self.researchers[n], outs[n], 1)]})
            while True:
                crit = self.critic.review(self.active_list(), r, self.pilots)
                self._record_critique(crit, r)
                self.critiques.append(crit)
                self.rounds[-1]["critique"] = crit["artifact_id"]
                self.rounds[-1]["mean_quality"] = crit["portfolio"]["mean_quality"]
                self.rounds[-1]["active"] = [p["id"] for p in self.active_list()]
                stop = self._should_stop(crit, r)
                if stop is None:
                    pil = self._maybe_pilots(crit, r)
                    self.rounds[-1]["pilots"] = list(pil)
                if stop:
                    self.stop_reason = stop
                    break
                r += 1
                fb, port, pilots = crit["per_proposal"], crit["portfolio"], dict(self.pilots)
                others = self._others()
                rc = {"round": r, "feedback": fb, "portfolio_feedback": {k: port.get(k) for k in (
                    "summary", "diversity_warnings", "disagreements", "missing_controls", "families")},
                      "pilots": pilots, "others": others, "budget_text": self._budget_text()}
                by_author = {n: [p for p in self.active_list() if p["author"] == n] for n in self.researchers}
                outs = self._parallel(lambda ag: ag.revise(rc, by_author[ag.name]))
                self.rounds.append({"round": r, "actions": [a for n in self.researchers
                                                            for a in self._apply(self.researchers[n], outs[n], r)]})
        finally:
            self.resources.phase = "execution"
        self.store.put(A.DECISION, {"decision": "stop_research", "reason": self.stop_reason, "rounds": len(self.rounds),
                                    "active": [p["id"] for p in self.active_list()],
                                    "abandoned": [a["proposal_id"] for a in self.abandoned]},
                       author="scheduler", stage="research", summary=f"research stopped after {len(self.rounds)} "
                                                                    f"round(s): {self.stop_reason}")
        log.info("research loop: %d rounds, %d active, %d abandoned — %s", len(self.rounds), len(self.active),
                 len(self.abandoned), self.stop_reason)
        return {"active": self.active_list(), "proposals": self.proposals, "abandoned": self.abandoned,
                "rounds": self.rounds, "critiques": self.critiques, "last_critique": self.critiques[-1] if self.critiques else None,
                "stop_reason": self.stop_reason, "pilots": self.pilots, "mem_of": self.mem_of}
