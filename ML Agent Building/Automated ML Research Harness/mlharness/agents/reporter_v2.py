"""V2 reporter: research-first report that separates established evidence, literature support,
conjectures, failed hypotheses and remaining uncertainty, and shows how ideas evolved.

Outputs: ``report.md``, ``research_trace.json`` (machine-readable idea evolution), ``leaderboard.csv``
and ``final/model_card.md``.
"""
from __future__ import annotations

import csv
import json
import time

from .. import artifacts as A
from .. import metrics as M
from .base import Agent


def _f(v, nd=5):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _esc(s, n=160):
    s = str(s or "").replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def research_state_from_store(store) -> dict:
    """Rebuild the research phase purely from artifacts (resume and report rebuild)."""
    props = {a.data["id"]: dict(a.data, artifact_id=a.id) for a in store.query(A.PROPOSAL)}
    crits = [dict(a.data, artifact_id=a.id) for a in store.query(A.FEEDBACK)]
    abandoned = [dict(a.data, proposal_id=a.data.get("proposal_id"), title=props.get(a.data.get("proposal_id"), {}).get("title"),
                      author=a.author, reason=a.data.get("reason"),
                      method_label=props.get(a.data.get("proposal_id"), {}).get("method_label"),
                      round=props.get(a.data.get("proposal_id"), {}).get("round"))
                 for a in store.query(A.DECISION) if a.data.get("decision") == "abandon_proposal"]
    stop = store.latest(A.DECISION, where=lambda a: a.data.get("decision") == "stop_research")
    active_ids = (stop.data.get("active") if stop else None) or []
    rounds = {}
    for p in props.values():
        rounds.setdefault(p["round"], {"round": p["round"], "actions": []})["actions"].append(
            {"researcher": p["author"], "persona": p["persona"], "action": p["action"], "target": p.get("parent_id"),
             "new_id": p["id"], "origin": p["origin"], "response": p.get("response_to_critic", "")})
    for a in store.query(A.DECISION):
        if a.data.get("decision") in ("defend_proposal", "abandon_proposal"):
            pid = a.data.get("proposal_id")
            r = props.get(pid, {}).get("round", 0) + 1     # responses happen in the round after the version
            rounds.setdefault(r, {"round": r, "actions": []})["actions"].append(
                {"researcher": a.author, "action": "defend" if a.data["decision"] == "defend_proposal" else "abandon",
                 "target": pid, "response": a.data.get("response") or a.data.get("reason", "")})
    for c in crits:
        if c.get("round") in rounds:
            rounds[c["round"]]["critique"] = c["artifact_id"]
            rounds[c["round"]]["mean_quality"] = c["portfolio"].get("mean_quality")
    return {"proposals": props, "critiques": crits, "last_critique": crits[-1] if crits else None,
            "abandoned": abandoned, "stop_reason": stop.data.get("reason") if stop else None,
            "active": [props[i] for i in active_ids if i in props], "rounds": [rounds[k] for k in sorted(rounds)],
            "pilots": {}, "mem_of": {}}


class ReporterV2(Agent):
    role = "reporter"

    def write(self, sch) -> str:
        spec, store = self.spec, self.store
        P = spec.primary
        R = sch.research or research_state_from_store(store)
        props = R.get("proposals") or {}
        res = sch.resources.snapshot()
        lit = getattr(sch, "literature", None)
        L = [f"# Research report (V2): {spec.name}\n",
             f"Run `{sch.run_id}` · {time.strftime('%Y-%m-%d %H:%M:%S')} · pipeline **v2 (research-first)** · LLM "
             f"backend **{sch.llm.describe()}** · workers {spec.hardware['workers']} × {spec.hardware['threads_per_worker']} "
             f"threads · GPUs {spec.hardware['gpu_devices'] or 'none'} · remote "
             f"{(sch.remote.host if sch.remote else 'off')}\n"]

        # 1 ---------------------------------------------------------------------------------
        L.append("## 1. Problem\n")
        L.append(f"**Problem.** {spec.problem.strip()}\n\n**Data.** {spec.simulator_description} "
                 f"(train/val/test = {spec.data['n_train']}/{spec.data['n_val']}/{spec.data['n_test']}; the test split "
                 f"is used once, at the end).\n")
        pr = getattr(sch, "problem", {}) or {}
        L.append(f"**Problem context available to methods.** {pr.get('describe', '-')}\n")
        d = pr.get("diagnostics") or {}
        if d.get("available"):
            L.append(f"**Numerical identifiability (harness-computed, {d['n_samples']} prior draws).** Fisher condition "
                     f"number median {d['fisher_condition_median']:.3g} (max {d['fisher_condition_max']:.3g}); Cramér–Rao "
                     f"relative std per parameter: " + ", ".join(f"{k} {v:.4f}" for k, v in d["crb_rel_median"].items()) +
                     f"; implied rel_l2 floor ≈ **{d['crb_rel_l2_estimate']:.4g}**; least determined: "
                     f"{', '.join(d['least_determined'])}.\n")

        # 2 ---------------------------------------------------------------------------------
        L.append("## 2. Outcome\n")
        fin = sch.final
        if fin:
            fr = fin.get("result", {})
            L.append(f"- Final method: **{fin['kind']}** of " + ", ".join(f"`{l}`" for l in fin["labels"]) +
                     (f" (weights {fin['weights']})" if fin["kind"] == "ensemble" else ""))
            L.append(f"- Validation {P} (selection estimate): **{_f(fin['val'])}** · mean-predictor baseline "
                     f"{_f(sch.baseline_value)}")
            ci = (fr.get("test_bootstrap") or {}).get("ci95")
            L.append("- **Test (held out, evaluated once):** " + ", ".join(f"{k} = **{_f(v)}**" for k, v in fin["test"].items())
                     + (f" · bootstrap 95% CI for {P}: [{_f(ci[0], 4)}, {_f(ci[1], 4)}]" if ci else ""))
            if fr.get("test_per_target_rel_l2"):
                L.append("- Per-target test rel_l2: " + ", ".join(f"{n} {v:.4f}" for n, v in
                                                                 zip(pr.get("param_names") or range(99), fr["test_per_target_rel_l2"])))
            if d.get("available"):
                L.append(f"- Relative to the Cramér–Rao floor estimate: {fin['test'][P] / d['crb_rel_l2_estimate']:.2f}× "
                         f"(the floor is an estimate at random prior draws, not at the test parameters)")
            if spec.success["threshold"] is not None:
                L.append(f"- Success criterion ({spec.success['metric']} "
                         f"{'<=' if M.direction(spec.success['metric']) == 'min' else '>='} {spec.success['threshold']}): "
                         f"**{'MET' if fin['audit'].get('success') else 'NOT MET'}**")
            for nm, mv in (fr.get("shift") or {}).items():
                L.append(f"- Robustness, shift set `{nm}`: {P} = {_f(mv.get(P))} (vs {_f(fin['test'][P])} in distribution)")
            if fr.get("test_uncertainty"):
                u = fr["test_uncertainty"]
                L.append(f"- Uncertainty calibration on test: {u['coverage_1sigma']:.1%} within 1σ, {u['coverage_2sigma']:.1%} "
                         f"within 2σ (ideal 68% / 95%)")
            L.append(f"- Inference latency: {_f(fr.get('latency_ms_per_sample'), 3)} ms/sample; fit {_f(fr.get('fit_seconds'), 3)} s; "
                     f"exported model reload check: max |Δ| = {_f(fr.get('reload_max_abs_diff'), 3)}")
            if fr.get("member_test") and len(fr["member_test"]) > 1:
                L.append("- Ensemble members on test: " + ", ".join(f"`{m['label']}` {_f(m.get(P), 4)}" for m in fr["member_test"]))
            if getattr(sch, "test_access_prior", 0):
                L.append(f"- ⚠ This test split had already been evaluated {sch.test_access_prior} time(s) in earlier runs "
                         f"(research memory): repeated development weakens its independence.")
            for iss in fin["audit"].get("issues", []):
                L.append(f"- Audit warning: {iss}")
            L.append(f"- Model: `{fin['model_path']}` — `from mlharness.export import load_model; "
                     f"m = load_model(r\"{sch.run_dir}\")` (manifest records seeds, versions, data hash, lineage)\n")
        else:
            L.append("No final method was produced (see failures).\n")

        # 3 evidence classes ----------------------------------------------------------------
        L.append("## 3. What we learned — evidence classes\n")
        outs = getattr(sch, "prediction_outcomes", {}) or {}
        sup = [o for o in outs.values() if o["outcome"] == "supported" and o.get("source") == "proposal"]
        con = [o for o in outs.values() if o["outcome"] == "contradicted" and o.get("source") == "proposal"]
        inc = [o for o in outs.values() if o["outcome"] in ("inconclusive", "untestable") and o.get("source") == "proposal"]
        L.append("### 3.1 Established evidence (experiments on the validation split)\n")
        best_fam = {}
        for iid, idea in sch.ideas.items():
            r = sch.results.get(iid, {}).get(1.0)
            if idea.get("mspec") and r and r.get("status") == "ok":
                v = sch.value(r)
                if idea["family"] not in best_fam or M.to_score(v, P) > M.to_score(best_fam[idea["family"]][1], P):
                    best_fam[idea["family"]] = (idea["label"], v)
        if best_fam:
            L.append("- Best full-fidelity validation result per method family: " +
                     "; ".join(f"**{fam}** `{lab}` {P}={_f(v)}" for fam, (lab, v) in sorted(best_fam.items(), key=lambda t: M.to_score(-1e9 if t[1][1] is None else t[1][1], P), reverse=True)))
        for o in sup:
            L.append(f"- SUPPORTED — {o['entry_id']} *{_esc(o['title'], 120)}*: {o.get('detail')}")
        if d.get("available"):
            L.append(f"- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ {d['crb_rel_l2_estimate']:.4g}")
        if not (best_fam or sup):
            L.append("- none")
        L.append("\n### 3.2 Literature-supported statements (retrieved, with applicability screen)\n")
        cited = {}
        for p in props.values():
            for c in p.get("citations") or []:
                cited.setdefault(c["ref_id"], []).append((p["id"], c))
        n_lit = 0
        for rid, uses in list(cited.items())[:14]:
            e = lit.get(rid) if lit is not None else None
            if not e:
                L.append(f"- `{rid}` — **unverifiable** (not retrieved in this run or memory); cited by {[u[0] for u in uses]}")
                continue
            a = e.get("assessment") or {}
            claims = sorted({u[1]["claimed_applicability"] for u in uses})
            L.append(f"- `{rid}` {e.get('citation', '')} — cited by {sorted({u[0] for u in uses})}; researcher claim "
                     f"{claims}; harness screen **{a.get('applies')}**"
                     + (f"; conflicts: {'; '.join(a['conflicts'])}" if a.get("conflicts") else ""))
            n_lit += 1
        if not cited:
            L.append("- no citations in proposals")
        L.append("\n_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim "
                 "and nothing here is formally verified._\n")
        L.append("### 3.3 Conjectures (not established by experiment in this run)\n")
        for o in inc:
            L.append(f"- {o['entry_id']} *{_esc(o['title'], 120)}*: {o['outcome']} — {o.get('detail')}")
        notrun = [p for p in R.get("active") or [] if p["id"] not in {e.get("proposal_id") for e in sch.portfolio}]
        for p in notrun:
            L.append(f"- {p['id']} *{_esc(p['title'], 120)}* ({p['status']}): proposed but not executed")
        if not (inc or notrun):
            L.append("- none")
        L.append("\n### 3.4 Failed hypotheses and negative results\n")
        for o in con:
            L.append(f"- CONTRADICTED — {o['entry_id']} *{_esc(o['title'], 120)}*: {o.get('detail')}")
        for a in R.get("abandoned") or []:
            L.append(f"- ABANDONED in round {a.get('round')} by {a.get('author')} — {a.get('proposal_id')} "
                     f"*{_esc(a.get('title'), 100)}*: {_esc(a.get('reason'), 220)}")
        for iid, why in sch.pruned.items():
            idea = sch.ideas.get(iid, {})
            if idea.get("mspec"):
                L.append(f"- PRUNED — {iid} `{idea.get('label')}`: {_esc(why, 200)}")
        for lab, v in (getattr(sch, "gate_verdicts", {}) or {}).items():
            if not v["approved"]:
                L.append(f"- GATE-REJECTED — {lab} `{v['method_label']}`: {_esc('; '.join(v['reasons']), 220)}")
        L.append("\n### 3.5 Remaining uncertainty\n")
        unc = []
        for e in sch.portfolio:
            for u in e.get("unresolved") or []:
                unc.append(f"- {e['entry_id']} ({_esc(e['title'], 60)}): {_esc(u, 220)}")
        for c in (R.get("last_critique") or {}).get("portfolio", {}).get("disagreements", []) or []:
            unc.append(f"- Unresolved disagreement {c.get('between')}: {_esc(c.get('issue'), 220)}")
        if lit is not None and lit.incomplete():
            st = lit.status()
            unc.append(f"- Literature check INCOMPLETE: {st['failures']} failed queries ({', '.join(st['failure_kinds']) or '-'}), "
                       f"{st['skipped_for_budget']} skipped for budget, enabled={st['enabled']}")
        if fin:
            ci = (fin.get("result", {}).get("test_bootstrap") or {}).get("ci95")
            if ci:
                unc.append(f"- Test-set sampling uncertainty: bootstrap 95% CI [{ci[0]:.4g}, {ci[1]:.4g}] for {P}; one seed, "
                           f"one dataset — between-method differences inside this interval are not established")
        L += unc[:30] or ["- none recorded"]
        L.append("")

        # 4 research phase ------------------------------------------------------------------
        L.append("## 4. Research phase: hypotheses, criticism and revision\n")
        L.append(f"Stopped after **{len(R.get('rounds') or [])} round(s)**: {R.get('stop_reason')}.\n")
        L.append("| round | researcher | action | proposal | new version | origin | response to critic |\n|---|---|---|---|---|---|---|")
        for rd in R.get("rounds") or []:
            for a in rd.get("actions") or []:
                L.append(f"| {rd['round']} | {a.get('researcher')} | {a.get('action')} | {a.get('target') or '-'} | "
                         f"{a.get('new_id') or '-'} | {a.get('origin') or '-'} | {_esc(a.get('response'), 110)} |")
        L.append("")
        L.append("| round | active proposals | mean critic quality | critique artifact |\n|---|---|---|---|")
        for rd in R.get("rounds") or []:
            L.append(f"| {rd['round']} | {len(rd.get('active') or [])} | {_f(rd.get('mean_quality'), 3)} | {rd.get('critique')} |")
        L.append("\n### Idea evolution\n")
        roots = {}
        for p in props.values():
            roots.setdefault(p["root_id"], []).append(p)
        crit_by = {}
        for c in R.get("critiques") or []:
            for pid, fb in (c.get("per_proposal") or {}).items():
                crit_by.setdefault(pid, []).append((c.get("round"), fb))
        for root, vers in roots.items():
            vers.sort(key=lambda p: p["version"])
            head = vers[0]
            L.append(f"**{root} — {_esc(head['title'], 120)}** ({head['author']}, family {head['family']}"
                     + (f", branched from {head.get('branched_from')}" if head.get("branched_from") else "") + ")")
            for p in vers:
                L.append(f"- v{p['version']} `{p['id']}` r{p['round']} [{p['action']}, {p['origin']}] `{p['method_label']}` "
                         f"status {p['status']}. *H:* {_esc(p['hypothesis'], 260)}")
                if p.get("prediction"):
                    L.append(f"  - prediction: `{json.dumps(p['prediction'])}`; if it fails: {_esc(p['if_fails'], 160)}")
                if p.get("response_to_critic"):
                    L.append(f"  - response to critic: {_esc(p['response_to_critic'], 240)}")
                for rnd, fb in crit_by.get(p["id"], [])[:1]:
                    pts = (fb.get("blocking") or []) + (fb.get("challenges") or [])[:2] + (fb.get("changed_assumptions") or [])[:1] \
                          + (fb.get("counterexamples") or [])[:1] + (fb.get("questions") or [])[:1]
                    L.append(f"  - critic r{rnd} (quality {fb.get('quality')}): " + " · ".join(_esc(x, 160) for x in pts[:5]))
            L.append("")
        for c in R.get("critiques") or []:
            pf = c.get("portfolio") or {}
            if pf.get("summary") or pf.get("diversity_warnings"):
                L.append(f"- Critic round {c.get('round')} portfolio view: {_esc(pf.get('summary'), 400)} "
                         f"{'; '.join(pf.get('diversity_warnings') or [])} families={pf.get('families')}")
        pil = getattr(sch, "pilot_records", {}) or {}
        if pil:
            L.append("\n**Pilot experiments run during research** (small validation subsets):\n")
            for pid, r in pil.items():
                L.append(f"- {pid} `{r.get('label')}`: {r.get('status')} {P}={_f(r.get('value'))} on {r.get('n_val')} "
                         f"val samples in {_f(r.get('seconds'), 3)} s" + (f" — {r['error']}" if r.get("error") else ""))
        L.append("")

        # 5 literature ------------------------------------------------------------------------
        L.append("## 5. Literature retrieval (TheoremSearch)\n")
        if lit is not None:
            st = lit.status()
            L.append(f"{st['queries']} queries, {st['failures']} failures, {st['refs']} distinct statements retrieved, "
                     f"{st['skipped_for_budget']} skipped for budget; check complete: **{not st['incomplete']}**.\n")
            L.append("| agent | round | query | result |\n|---|---|---|---|")
            for q in lit.log[:40]:
                L.append(f"| {q['agent']} | {q['round']} | {_esc(q['query'], 90)} | " +
                         (f"{len(q['results'])} refs via {q['endpoint']}{' (cache)' if q['cached'] else ''}" if q["ok"]
                          else f"FAILED: {q['error_kind']}") + " |")
        L.append("")

        # 6 memory -------------------------------------------------------------------------
        L.append("## 6. Research memory\n")
        mem = sch.memory
        ms = mem.stats()
        prior = sorted({i for r in mem.retrievals for i in r["ids"] if (mem.get(i) or {}).get("run_id") != sch.run_id})
        L.append(f"- Backend {ms['backend']}; {ms['loaded_from_disk']} records from {ms['runs']} run(s) were available at "
                 f"start; agents retrieved {len(prior)} distinct prior-run records; this run wrote "
                 f"{ms['records'] - ms['loaded_from_disk']} records (test numbers redacted for agents).")
        for i in prior[:12]:
            r = mem.get(i)
            L.append(f"  - `{i}` {r['type']} ({r.get('status')}) from `{r.get('run_id')}`: {_esc(r['title'], 140)}")
        L.append("")

        # 7 portfolio / gate / execution -------------------------------------------------------------
        L.append("## 7. Portfolio, gate and experiments\n")
        L.append("| entry | source | family | method | author | quality | why selected |\n|---|---|---|---|---|---|---|")
        from .. import methods_v2 as M2
        for e in sch.portfolio:
            try:
                lab = M2.label(M2.normalize(e["method"], sch.available, sch.feature_sets, sch.n_features, spec.data["n_train"])[0])
            except Exception:
                lab = "?"
            L.append(f"| {e['entry_id']} | {e['source']}{' (' + e['proposal_id'] + ')' if e.get('proposal_id') else ''} | "
                     f"{e['family']} | `{lab}` | {e['author']} | {_f(e.get('quality'), 3)} | {_esc(e.get('selection_reason'), 80)} |")
        for x in sch.excluded:
            L.append(f"- not advanced: {x.get('proposal_id')} *{_esc(x.get('title'), 80)}* — {_esc(x.get('reason'), 160)}")
        sec = spec.metric_names[1:]
        rows = []
        for iid, idea in sch.ideas.items():
            if not idea.get("mspec"):
                continue
            r = sch.v2res.get(iid, {})
            rows.append((idea, r.get("screen"), r.get("full")))
        rows.sort(key=lambda t: -sch.score(t[2]) if t[2] else 1e9)
        L.append(f"\n| idea | entry | origin | family | method | screen {P} | **full val {P}** | " +
                 " | ".join(f"val {m}" for m in sec) + " | fit s | ms/sample | fwd evals | 2σ cover | outcome |")
        L.append("|" + "---|" * (11 + len(sec)))
        for idea, s, f in rows:
            fv = f.get("val") if f and f.get("status") == "ok" else {}
            unc = (f or {}).get("uncertainty") or {}
            outcome = ("FINAL" if fin and idea["id"] in fin["members"] else sch.pruned.get(idea["id"]) or
                       ("evaluated" if fv else (f or s or {}).get("status", "-")))
            L.append(f"| {idea['id']} | {idea.get('entry_id') or '-'} | {idea['origin']} | {idea['family']} | `{idea['label']}` | "
                     f"{_f(sch.value(s), 4) if s else '-'} | **{_f(fv.get(P), 5)}** | " +
                     " | ".join(_f(fv.get(m), 4) for m in sec) +
                     f" | {_f((f or {}).get('fit_seconds'), 3)} | {_f((f or {}).get('latency_ms_per_sample'), 3)} | "
                     f"{_f((f or {}).get('forward_evals'))} | {_f(unc.get('coverage_2sigma'), 3)} | {_esc(outcome, 70)} |")
        L.append("")

        # 8 lineage -----------------------------------------------------------------------------
        L.append("## 8. Lineage of the final method\n")
        if fin:
            lin = (fin.get("result") or {})
            for iid in fin["members"]:
                idea = sch.ideas[iid]
                chain, pid = [], idea.get("proposal_id")
                while pid and pid in props:
                    p = props[pid]
                    chain.append(f"{pid} v{p['version']} ({p['author']}, r{p['round']}, {p['action']})")
                    pid = p.get("parent_id") or p.get("branched_from")
                par = idea.get("parent")
                L.append(f"- `{iid}` `{idea['label']}` ← " + (f"refined from {par} ← " if par else "") +
                         (" ← ".join(chain) if chain else f"baseline/control ({idea.get('title')})"))
            L.append("\nArtifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):\n")
            for a in store.lineage(fin["checkpoint"])[:45]:
                L.append(f"- `{a.id}` [{a.kind}] by {a.author}: {_esc(a.summary, 150)}")
        L.append("")

        # 9 compute -------------------------------------------------------------------------------
        L.append("## 9. Compute and cost\n")
        L.append(f"- Execution compute: {res['compute_used_s']:.1f} / {res['compute_budget_s']:.0f} worker-seconds over "
                 f"{res['jobs']} jobs; wall {res['wall_elapsed_s']:.0f} s")
        L.append("- By stage: " + ", ".join(f"{k} {v:.1f}s" for k, v in res["by_stage_s"].items()))
        L.append(f"- LLM: {res['llm_calls']} calls (by phase {res.get('llm_by_phase')}), {res['llm_seconds']:.0f} s, "
                 f"${res['llm_cost_usd']:.3f} reported · TheoremSearch: {res.get('literature_queries')} queries "
                 f"({res.get('literature_failures')} failed)\n")
        L.append("## 10. Agents\n")
        for a in sch.roster:
            L.append(f"- **{a['name']}** ({a['role']}{', ' + a['persona'] if a.get('persona') else ''}) — {a['llm']}")
        L.append("")
        L.append("## 11. Discussion\n")
        L.append(self._discussion(sch, best_fam, sup, con) + "\n")
        text = "\n".join(L)
        (sch.run_dir / "report.md").write_text(text, encoding="utf-8")
        self._trace(sch, R)
        self._csv(sch, rows)
        if fin:
            self._card(sch, fin)
        self.emit(A.REPORT, {"path": str(sch.run_dir / "report.md")}, stage="report",
                  files=[str(sch.run_dir / "report.md"), str(sch.run_dir / "research_trace.json")],
                  summary="V2 research report")
        return text

    def _discussion(self, sch, best_fam, sup, con) -> str:
        P = self.spec.primary
        fin = sch.final
        auto = (f"{len((sch.research or {}).get('proposals') or {})} proposal versions over "
                f"{len((sch.research or {}).get('rounds') or [])} research rounds; {len(sch.portfolio)} methods advanced; "
                f"{len(sup)} predictions supported, {len(con)} contradicted. ")
        if fin:
            auto += f"Final: {fin['labels']} with test {P} {fin['test'].get(P):.5g}."
        if not self.uses_llm or not fin:
            return auto
        summary = {"best_by_family": best_fam, "supported": [o.get("detail") for o in sup],
                   "contradicted": [o.get("detail") for o in con], "final": {k: fin[k] for k in ("labels", "val", "test")},
                   "abandoned": [(a.get("title"), a.get("reason")) for a in (sch.research or {}).get("abandoned", [])][:8]}
        text = self.ask("You are the reporting agent of an autonomous research team. Write concise, evidence-based "
                        "scientific discussion in Markdown (no headings). Distinguish established results from conjecture.",
                        f"{self.spec.brief()}\n\nResults:\n{json.dumps(summary, default=str)[:9000]}\n\nIn two short paragraphs "
                        "and 3-5 bullet next experiments: what the evidence establishes, which hypotheses failed and why, "
                        "and what remains uncertain.", purpose="report discussion", stage="report")
        return (text.strip() + "\n\n_Automatic summary:_ " + auto) if text else auto

    def _trace(self, sch, R):
        props = R.get("proposals") or {}
        trace = {"run_id": sch.run_id, "stop_reason": R.get("stop_reason"), "rounds": R.get("rounds"),
                 "proposals": {k: {kk: vv for kk, vv in v.items() if kk != "method" or not isinstance(vv, dict)}
                               | {"method": {a: b for a, b in (v.get("method") or {}).items() if a != "code"}}
                               for k, v in props.items()},
                 "critiques": R.get("critiques"), "abandoned": R.get("abandoned"), "portfolio": sch.portfolio,
                 "excluded": sch.excluded, "prediction_outcomes": getattr(sch, "prediction_outcomes", {}),
                 "pilots": getattr(sch, "pilot_records", {}),
                 "literature": getattr(sch, "literature", None).log if getattr(sch, "literature", None) else []}
        (sch.run_dir / "research_trace.json").write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")

    def _csv(self, sch, rows):
        names = self.spec.metric_names
        with open(sch.run_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idea", "entry", "proposal", "origin", "family", "method", "parent"] +
                       [f"val_{m}" for m in names] + ["screen_val", "fit_seconds", "latency_ms", "forward_evals"])
            for idea, s, fr in rows:
                fv = fr.get("val") if fr and fr.get("status") == "ok" else {}
                w.writerow([idea["id"], idea.get("entry_id"), idea.get("proposal_id"), idea["origin"], idea["family"],
                            idea["label"], idea.get("parent") or ""] + [fv.get(m) for m in names] +
                           [sch.value(s) if s else None, (fr or {}).get("fit_seconds"),
                            (fr or {}).get("latency_ms_per_sample"), (fr or {}).get("forward_evals")])

    def _card(self, sch, fin):
        from .. import HARNESS_ROOT
        fr = fin.get("result", {})
        lines = [f"# Model card — {self.spec.name} (V2)", "",
                 f"- Task: {self.spec.task}; data: {self.spec.simulator_description}",
                 f"- Type: {fin['kind']} of {fin['labels']} (weights {fin['weights']})",
                 f"- Fitted on train+val ({self.spec.data['n_train'] + self.spec.data['n_val']} samples); physics members "
                 f"query the simulator's forward operator at prediction time (the simulator file is copied next to the model)",
                 f"- Validation estimate ({self.spec.primary}): {_f(fin['val'])}",
                 f"- Test: {json.dumps(fin['test'])}; bootstrap: {json.dumps(fr.get('test_bootstrap'))}",
                 f"- Shift sets: {json.dumps(fr.get('shift'))}",
                 f"- Latency: {_f(fr.get('latency_ms_per_sample'), 3)} ms/sample", "", "## Usage", "```python",
                 "import sys", f"sys.path.insert(0, r\"{HARNESS_ROOT}\")", "from mlharness.export import load_model",
                 f"model = load_model(r\"{sch.run_dir}\")", "theta_hat = model.predict(X)   # raw simulator observations",
                 "```", "", "Reproducibility metadata and research lineage: `manifest.json`."]
        (sch.run_dir / "final" / "model_card.md").write_text("\n".join(lines), encoding="utf-8")
