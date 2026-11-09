"""Shared research critic (V2) — inside the research loop, not a pre-training veto.

Its job is to make hypotheses stronger and more informative. For every active proposal it returns
concrete, actionable feedback: challenges, changed assumptions, questions, counterexamples, evidence
requests (TheoremSearch queries the researcher's next round will run), revision directions, cheap
discriminating tests (optionally run as *pilot experiments* before the next revision), duplicates,
and a split of the proposal's claims into supported facts vs. conjectures. At portfolio level it
tracks family diversity, genuine disagreements, and whether the research has converged.

Two layers:
* a deterministic rule layer (always on; reproducible; its blocking issues are authoritative), and
* an optional LLM layer (one call per round reading all proposals, rule findings, diagnostics,
  memory, literature and pilot results), merged into the rule feedback.

The critic does NOT check code safety, budgets or leakage at execution level — that is the
independent ``ImplementationGate``.
"""
from __future__ import annotations

import json
from collections import Counter

from .. import methods_v2 as M2
from ..agents.base import Agent
from .proposals import claim_hygiene, completeness

CRITIC_SYSTEM = ("You are the shared research critic of a four-researcher autonomous research team working on an "
                 "inverse problem. Your purpose is to make hypotheses stronger and more informative, not to reject "
                 "unfamiliar ideas. Challenge mathematical/physical assumptions, identifiability and whether cited "
                 "theorems' assumptions really hold; name competing explanations, duplicates and missing controls; give "
                 "concrete changed assumptions, counterexamples, evidence requests and revision directions; prefer cheap "
                 "discriminating tests over dismissing ambitious ideas; keep the portfolio diverse; separate supported "
                 "facts from conjectures; acknowledge genuine unresolved disagreement. Empirical results outweigh "
                 "argument. Reply with a single JSON object only.")

FB_LISTS = ("challenges", "changed_assumptions", "questions", "counterexamples", "evidence_requests",
            "revision_directions", "suggested_tests", "facts", "conjectures", "blocking")


def _empty_fb():
    return {k: [] for k in FB_LISTS} | {"duplicate_of": None, "missing": [], "diversify": False}


class ResearchCritic(Agent):
    role = "research_critic"

    def __init__(self, *a, literature, memory, problem: dict, rcfg: dict, available: dict, feature_sets: list,
                 n_features: int, context: dict, run_id: str, exec_budget_s: float, **kw):
        super().__init__(*a, **kw)
        self.literature = literature
        self.memory = memory
        self.problem = problem
        self.rcfg = rcfg
        self.available = available
        self.feature_sets = feature_sets
        self.n_features = n_features
        self.mem_context = context
        self.run_id = run_id
        self.exec_budget_s = float(exec_budget_s)

    # ------------------------------------------------------------------------------------------
    def review(self, proposals: list[dict], round_idx: int, pilots: dict | None = None) -> dict:
        per = {p["id"]: self._rules(p, proposals) for p in proposals}
        port = self._portfolio_rules(proposals, per)
        llm_used = False
        if self.uses_llm and self.rcfg.get("llm_critic", True) and proposals:
            out = self._llm_review(proposals, per, port, round_idx, pilots or {})
            if out:
                llm_used = True
                self._merge(per, port, out, {p["id"] for p in proposals})
        for p in proposals:
            per[p["id"]]["quality"] = self._quality(p, per[p["id"]])
        active_q = [per[p["id"]]["quality"] for p in proposals]
        blocking = sum(1 for p in proposals if per[p["id"]]["blocking"])
        thr = float(self.rcfg.get("quality_threshold", 0.7))
        port["mean_quality"] = round(sum(active_q) / len(active_q), 3) if active_q else 0.0
        port["all_meet_threshold"] = bool(active_q) and min(active_q) >= thr and blocking == 0
        if not llm_used:
            port["converged"] = port["all_meet_threshold"]
        port["llm_used"] = llm_used
        return {"round": round_idx, "per_proposal": per, "portfolio": port}

    # -- rule layer ---------------------------------------------------------------------------
    def _rules(self, p: dict, allp: list[dict]) -> dict:
        fb = _empty_fb()
        score, missing = completeness(p)
        fb["completeness"] = round(score, 3)
        fb["missing"] = missing
        for m in missing:
            fb["revision_directions"].append(f"state the {m} explicitly")
        hyg = claim_hygiene(p, self.literature)
        for h in hyg:
            fb["challenges"].append(h)
        ms = p.get("method") or {}
        kind = ms.get("kind")
        if kind not in M2.KINDS:
            fb["blocking"].append(f"method kind {kind!r} is not executable; use one of {list(M2.KINDS)}")
        else:
            norm, notes, errors = M2.normalize(ms, self.available, self.feature_sets, self.n_features,
                                               int(self.spec.data["n_train"]))
            fb["blocking"] += errors
            if notes:
                fb["revision_directions"].append("method parameters will be adjusted: " + "; ".join(notes)[:300])
            if M2.uses_forward(norm) and not self.problem.get("forward_available"):
                fb["blocking"].append("needs the forward model, which this research spec does not authorise "
                                      "(constraints.allow_forward_model) or the simulator does not provide")
            if kind in ("custom_ml", "custom_solver") and not self.spec.constraints.get("allow_custom_code", True):
                fb["blocking"].append("agent-written code is disabled by the spec constraints")
            elif kind in ("custom_ml", "custom_solver") and not ms.get("code") and not self.uses_llm:
                fb["blocking"].append("requests agent-written code but no LLM backend is available to write it")
            elif kind in ("custom_ml", "custom_solver") and ms.get("code"):
                from ..safety import check_code_v2
                probs = check_code_v2(ms["code"], "build_model" if kind == "custom_ml" else "build_method")
                if probs:
                    fb["revision_directions"].append("the code will fail the safety gate: " + "; ".join(probs)[:300])
            est = M2.estimate_seconds(norm, int(self.spec.data["n_val"]), int(self.spec.data["n_train"]),
                                      int(self.problem.get("n_params") or 1), self.problem.get("sec_per_forward"))
            fb["est_seconds"] = round(est, 1)
            if est > 0.4 * self.exec_budget_s:
                fb["revision_directions"].append(f"estimated cost {est:.0f}s is >40% of the execution budget "
                                                 f"({self.exec_budget_s:.0f}s): make it cheaper or justify it with a pilot")
                fb["suggested_tests"].append({"description": "pilot on a small validation subset to measure real cost",
                                              "pilot": True})
        if not p.get("prediction"):
            fb["revision_directions"].append("add a machine-checkable prediction (metric threshold or relative "
                                             "improvement vs a named reference) so the experiment can falsify it")
        diag = self.problem.get("diagnostics") or {}
        noise_model = self.problem.get("noise_model")
        text = " ".join(p.get("assumptions") or []).lower()
        if kind in ("physics", "hybrid") or (kind == "custom_solver" and ms.get("uses_forward")):
            if noise_model and "noise" not in text:
                fb["changed_assumptions"].append(f"noise model (from the simulator): {noise_model}; the likelihood "
                                                 f"must match it")
            if diag.get("available"):
                fb["facts"].append(f"numerical diagnostics: full-rank Fisher at sampled parameters (median "
                                   f"condition {diag['fisher_condition_median']:.3g}); CRB rel_l2 floor ~"
                                   f"{diag['crb_rel_l2_estimate']:.4g}")
                fb["questions"].append(f"the least determined parameters are {diag['least_determined']} (CRB rel std "
                                       + ", ".join(f"{k}={diag['crb_rel_median'][k]:.3g}" for k in diag["least_determined"])
                                       + "): does your estimator exploit the prior bounds for them?")
            prm = ms.get("params") or {}
            if int(prm.get("multistart", 1)) <= 1 and kind == "physics":
                fb["counterexamples"].append("a single LM start can stop at a local minimum when the truth lies near a "
                                             "bound or the misfit is multimodal; a 2-start pilot discriminates cheaply")
                fb["suggested_tests"].append({"description": "pilot: compare 1 vs 2 starts on 60 validation samples",
                                              "pilot": True})
            fb["evidence_requests"].append({"query": "convergence of Levenberg-Marquardt for nonlinear least squares "
                                                     "parameter estimation with bounds", "why": "local convergence assumptions"})
        elif kind in ("library", "custom_ml"):
            if diag.get("available"):
                fb["questions"].append(f"the physics floor (CRB rel_l2 ~{diag['crb_rel_l2_estimate']:.4g}) is known: what "
                                       "error do you expect relative to it, and what limits a learned inverse here "
                                       "(samples, capacity, noise)?")
            if self.problem.get("forward_available") and kind == "library":
                fb["revision_directions"].append("consider whether the forward model could correct this learner "
                                                 "(e.g. as an initializer for LM) — a hybrid tests the 'approximation "
                                                 "error' explanation directly")
            prior = self.memory.evidence_for(M2.label(ms), self.mem_context) if self.memory is not None else []
            if prior:
                fb["facts"].append(f"research memory holds {len(prior)} prior observation(s) for {M2.label(ms)}")
        for c in p.get("citations") or []:
            entry = self.literature.get(c["ref_id"]) if self.literature is not None else None
            if not entry:
                continue
            a = entry.get("assessment") or {}
            if a.get("conflicts") and c["claimed_applicability"] == "yes":
                fb["challenges"].append(f"{c['ref_id']}: claimed to apply, but the statement's assumptions conflict: "
                                        + "; ".join(a["conflicts"]))
                fb["changed_assumptions"].append(f"treat {c['ref_id']} as at most partially applicable")
            if c["claimed_applicability"] in ("yes", "partial"):
                fb["conjectures"].append(f"transfer of {c['ref_id']} to this setting (screen: {a.get('applies')})")
        for e in p.get("evidence") or []:
            src = e.get("source", "reasoning")
            if src in ("diagnostics",) or (self.literature is not None and self.literature.known(src)):
                fb["facts"].append(e["claim"][:200])
            elif src.startswith("m-") and self.memory is not None and self.memory.get(src):
                fb["facts"].append(f"(memory {src}) " + e["claim"][:180])
            else:
                fb["conjectures"].append(e["claim"][:200])
        if not p.get("citations") and self.literature is not None and self.literature.enabled:
            fb["evidence_requests"].append({"query": f"{p['title'][:120]}", "why": "no literature cited"})
        # duplicates: identical method hash among active proposals; keep the earlier one
        try:
            h = M2.method_hash(M2.normalize(ms, self.available, self.feature_sets, self.n_features,
                                            int(self.spec.data["n_train"]))[0])
            for q in allp:
                if q["id"] == p["id"] or q["id"] > p["id"]:
                    continue
                qh = M2.method_hash(M2.normalize(q.get("method") or {}, self.available, self.feature_sets,
                                                 self.n_features, int(self.spec.data["n_train"]))[0])
                if qh == h:
                    fb["duplicate_of"] = q["id"]
                    fb["challenges"].append(f"executes exactly the same method as {q['id']} ({q['author']}); "
                                            "differentiate the hypothesis or merge")
                    break
        except Exception:
            pass
        fb["evidence_requests"] = fb["evidence_requests"][:2]
        return fb

    def _portfolio_rules(self, proposals, per) -> dict:
        fams = Counter(p["family"] for p in proposals)
        port = {"families": dict(fams), "n_active": len(proposals), "diversity_warnings": [], "disagreements": [],
                "summary": ""}
        if proposals and fams.most_common(1)[0][1] > max(2, 0.5 * len(proposals)):
            fam = fams.most_common(1)[0][0]
            port["diversity_warnings"].append(f"{fams[fam]}/{len(proposals)} active proposals are '{fam}'")
            infam = sorted((p for p in proposals if p["family"] == fam), key=lambda p: per[p["id"]]["completeness"])
            if infam:
                per[infam[0]["id"]]["diversify"] = True
                per[infam[0]["id"]]["revision_directions"].append(
                    f"the portfolio is dominated by '{fam}' methods: branch into an under-represented family")
        authors = Counter(p["author"] for p in proposals)
        port["authors_active"] = dict(authors)
        return port

    def _quality(self, p, fb) -> float:
        comp = fb.get("completeness", 0.0)
        falsif = 1.0 if p.get("prediction") else 0.3 if p.get("falsifiable_test") else 0.0
        verified = [c for c in p.get("citations") or [] if self.literature is not None and self.literature.known(c["ref_id"])]
        evid = min(1.0, 0.5 * len(verified) + 0.25 * sum(1 for e in p.get("evidence") or []
                                                          if e.get("source") in ("diagnostics",) or str(e.get("source", "")).startswith("m-")))
        if not (self.literature is not None and self.literature.enabled):
            evid = max(evid, 0.5)        # cannot be penalised for a tool that is switched off
        feas = 0.0 if fb.get("blocking") else 1.0
        q = 0.35 * comp + 0.25 * falsif + 0.2 * evid + 0.2 * feas
        if fb.get("llm_score") is not None:
            q = 0.7 * q + 0.3 * float(fb["llm_score"])
        if fb.get("duplicate_of"):
            q *= 0.6
        return round(min(1.0, max(0.0, q)), 3)

    # -- LLM layer ----------------------------------------------------------------------------
    def _llm_review(self, proposals, per, port, round_idx, pilots):
        from .proposals import compact
        slim = [compact(p) for p in proposals]
        rule = {pid: {k: v for k, v in fb.items() if v and k in ("blocking", "challenges", "missing", "duplicate_of",
                                                                  "est_seconds")} for pid, fb in per.items()}
        mem = self.memory.digest(self.spec.problem + " critique failure", context=self.mem_context, k=6,
                                 exclude_run=self.run_id, agent=self.name) if self.memory is not None else ""
        lit = self.literature.digest(limit=10) if self.literature is not None else ""
        d = self.problem.get("diagnostics") or {}
        prompt = (f"{self.spec.brief()}\nProblem context: {self.problem.get('describe')}\n"
                  f"Numerical diagnostics: {json.dumps({k: d.get(k) for k in ('fisher_condition_median', 'crb_rel_median', 'crb_rel_l2_estimate', 'least_determined')} if d.get('available') else 'n/a', default=str)}\n\n"
                  f"Round {round_idx} proposals (long fields abbreviated with '…'; the JSON itself is complete):\n"
                  f"{json.dumps(slim, default=str)}\n\n"
                  f"Deterministic rule findings (authoritative for blocking issues):\n{json.dumps(rule, default=str)[:3000]}\n\n"
                  f"Pilot-experiment results (empirical; weigh them above argument): {json.dumps(pilots, default=str)[:2500]}\n\n"
                  f"Research memory (prior runs):\n{mem}\n\nRetrieved literature:\n{lit}\n\n"
                  "Review EVERY proposal. Respond as JSON: "
                  '{"reviews": [{"proposal_id": str, "score": 0-1, "challenges": [str], "changed_assumptions": [str], '
                  '"questions": [str], "counterexamples": [str], "evidence_requests": [{"query": str, "why": str}], '
                  '"revision_directions": [str], "suggested_tests": [{"description": str, "pilot": bool}], '
                  '"duplicate_of": null, "facts": [str], "conjectures": [str], "blocking": [str]}], '
                  '"portfolio": {"diversity_assessment": str, "disagreements": [{"between": [str], "issue": str}], '
                  '"missing_controls": [str], "converged": bool, "summary": str}}. Set converged=true only if '
                  "every proposal is specific, falsifiable and adequately evidenced so that further debate would not "
                  "change what is worth testing.")
        return self.ask_json(CRITIC_SYSTEM, prompt, purpose=f"research critique round {round_idx}", stage="research")

    def _merge(self, per, port, out, ids):
        for r in (out.get("reviews") or []):
            if not isinstance(r, dict) or r.get("proposal_id") not in ids:
                continue
            fb = per[r["proposal_id"]]
            for k in FB_LISTS:
                for item in (r.get(k) or [])[:6]:
                    if k == "evidence_requests" and isinstance(item, str):
                        item = {"query": item, "why": ""}
                    if k == "suggested_tests" and isinstance(item, str):
                        item = {"description": item, "pilot": False}
                    if item and item not in fb[k]:
                        fb[k].append(item)
            if r.get("duplicate_of") in ids and r.get("duplicate_of") != r["proposal_id"]:
                fb["duplicate_of"] = fb["duplicate_of"] or r["duplicate_of"]
            try:
                fb["llm_score"] = max(0.0, min(1.0, float(r.get("score"))))
            except (TypeError, ValueError):
                pass
            fb["evidence_requests"] = fb["evidence_requests"][:3]
        pf = out.get("portfolio") or {}
        port["summary"] = str(pf.get("summary") or "")[:1500]
        port["diversity_assessment"] = str(pf.get("diversity_assessment") or "")[:800]
        port["missing_controls"] = [str(x)[:300] for x in (pf.get("missing_controls") or [])][:5]
        for d in (pf.get("disagreements") or [])[:6]:
            if isinstance(d, dict):
                port["disagreements"].append({"between": [str(x) for x in d.get("between") or []][:4],
                                              "issue": str(d.get("issue", ""))[:400]})
        port["converged"] = bool(pf.get("converged"))
