"""Reporter agent: turns the artifact record into report.md, leaderboard.csv and a model card."""
from __future__ import annotations

import csv
import json
import time

from .. import artifacts as A
from .. import metrics as M
from .base import Agent


def _fmt(v, nd=5):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _short(d, n=70):
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


class ReporterAgent(Agent):
    role = "reporter"

    def write(self, sch) -> str:
        spec, store = self.spec, self.store
        primary = spec.primary
        res = sch.resources.snapshot()
        L = []
        L.append(f"# Research report: {spec.name}\n")
        L.append(f"Run `{sch.run_id}` · generated {time.strftime('%Y-%m-%d %H:%M:%S')} · LLM backend: "
                 f"**{sch.llm.describe()}** · workers: {spec.hardware['workers']} × "
                 f"{spec.hardware['threads_per_worker']} threads · GPUs: {spec.hardware['gpu_devices'] or 'none'}\n")

        L.append("## 1. Research specification\n")
        L.append(f"**Problem.** {spec.problem.strip()}\n")
        L.append(f"**Data.** {spec.simulator_description}\n")
        L.append("**Objectives.**\n" + "\n".join(f"- {o}" for o in spec.objectives) + "\n")

        L.append("## 2. Outcome\n")
        fin = sch.final
        if fin:
            succ = fin["audit"].get("success")
            L.append(f"- Final model: **{fin['kind']}** of {', '.join(fin['members'])}"
                     + (f" (weights {fin['weights']})" if fin["kind"] == "ensemble" else ""))
            for iid in fin["members"]:
                i = sch.ideas[iid]
                L.append(f"  - `{iid}` {i['title']} — method `{i['config']['method']}`, features "
                         f"`{i['config']['features']}`, params `{_short(i['config']['params'], 160)}`")
            L.append(f"- Validation {primary} (selection estimate): **{_fmt(fin['val'])}**; "
                     f"baseline (mean predictor): {_fmt(sch.baseline_value)}")
            L.append("- **Test metrics (held-out, evaluated once):** " +
                     ", ".join(f"{k} = **{_fmt(v)}**" for k, v in fin["test"].items()))
            if spec.success["threshold"] is not None:
                L.append(f"- Success criterion ({spec.success['metric']} "
                         f"{'<=' if M.direction(spec.success['metric']) == 'min' else '>='} "
                         f"{spec.success['threshold']}): **{'MET' if succ else 'NOT MET'}**")
            for iss in fin["audit"].get("issues", []):
                L.append(f"- Critic audit warning: {iss}")
            L.append(f"- Model file: `{fin['model_path']}` — load with "
                     f"`from mlharness.export import load_model; m = load_model(r\"{sch.run_dir}\")`")
            L.append(f"- Stopping reason for the search: {sch.stop_reason or 'pipeline completed'}\n")
        else:
            L.append("No final model was produced (see failures below).\n")

        L.append("## 3. Research funnel\n")
        L.append("| step | count |\n|---|---|")
        for k, v in sch.funnel.items():
            L.append(f"| {k} | {v} |")
        L.append(f"| ideas pruned (all reasons) | {len(sch.pruned)} |\n")

        rows = []
        for iid, by_f in sch.results.items():
            i = sch.ideas[iid]
            r = by_f.get(1.0)
            if r and r.get("status") == "ok":
                rows.append((i, r))
        rows.sort(key=lambda t: sch.score(t[1]), reverse=True)
        L.append(f"## 4. Leaderboard (full training data, validation split)\n")
        sec = spec.metric_names[1:]
        L.append("| # | idea | persona | origin | method | features | params | " + " | ".join(
            [f"val {primary}"] + [f"val {m}" for m in sec]) + " | fit s |")
        L.append("|" + "---|" * (8 + len(sec) + 1))
        for k, (i, r) in enumerate(rows, 1):
            vals = [_fmt(r["val"].get(primary))] + [_fmt(r["val"].get(m)) for m in sec]
            L.append(f"| {k} | {i['id']} | {i['persona']} | {i['origin']} | {i['config']['method']} | "
                     f"{i['config']['features']} | `{_short(i['config']['params'], 60)}` | " + " | ".join(vals) +
                     f" | {_fmt(r.get('fit_seconds'), 3)} |")
        L.append("")

        fracs = sorted({f for v in sch.results.values() for f in v})
        L.append("## 5. Multi-fidelity trace (validation " + primary + " by training-data fraction)\n")
        L.append("| idea | method | " + " | ".join(f"{f:.2f}" for f in fracs) + " | outcome |")
        L.append("|" + "---|" * (len(fracs) + 3))
        for iid, by_f in sch.results.items():
            i = sch.ideas[iid]
            cells = []
            for f in fracs:
                r = by_f.get(f)
                cells.append("-" if r is None else _fmt(sch.value(r), 4) if r.get("status") == "ok" else r.get("status"))
            if fin and iid in fin["members"]:
                outcome = "FINAL MODEL"
            elif i["persona"] == "baseline":
                outcome = "reference baseline"
            else:
                outcome = sch.pruned.get(iid) or sch.outcomes.get(iid, "evaluated at full fidelity")
                if outcome.startswith("promote") and 1.0 in by_f:
                    outcome = "evaluated at full fidelity"
            L.append(f"| {iid} | {i['config']['method']} | " + " | ".join(cells) + f" | {outcome[:90]} |")
        L.append("")

        L.append("## 6. Lineage of the final model\n")
        if fin:
            for iid in fin["members"]:
                chain, cur = [], sch.ideas.get(iid)
                while cur:
                    chain.append(f"{cur['id']} ({cur['persona']}/{cur['origin']}: {cur['title'][:60]})")
                    cur = sch.ideas.get(cur.get("parent")) if cur.get("parent") else None
                L.append(f"- Idea ancestry: " + "  ←  ".join(chain))
            L.append("\nArtifact trace (checkpoint → result → plan → hypothesis → …):\n")
            for a in store.lineage(fin["checkpoint"])[:40]:
                L.append(f"- `{a.id}` [{a.kind}] by {a.author}: {a.summary}")
        L.append("")

        L.append("## 7. Pruned ideas and failures\n")
        L.append("| idea | method | reason |\n|---|---|---|")
        for iid, why in sch.pruned.items():
            L.append(f"| {iid} | {sch.ideas[iid]['config']['method']} | {why[:160]} |")
        for f in store.query(A.FAILURE, where=lambda a: a.data.get("source") in ("llm", "scheduler")):
            L.append(f"| - | ({f.data.get('source')}) | {f.summary[:160]} |")
        L.append("")

        L.append("## 8. Critic notes\n")
        latest = {}
        for a in store.query(A.CRITIQUE):
            if a.data.get("type") in ("result_review", "batch_advice", "final_audit"):
                latest[a.idea_id or a.id] = a          # latest critique per idea
        for a in list(latest.values())[:40]:
            L.append(f"- {a.summary}")
        if not latest:
            L.append("- no issues flagged")
        L.append("")

        L.append("## 9. Compute and resource usage\n")
        L.append(f"- Compute used: {res['compute_used_s']:.1f} / {res['compute_budget_s']:.0f} worker-seconds "
                 f"over {res['jobs']} jobs; wall time {res['wall_elapsed_s']:.1f}s")
        L.append("- By stage: " + ", ".join(f"{k} {v:.1f}s" for k, v in res["by_stage_s"].items()))
        top = sorted(sch.resources.by_idea.items(), key=lambda kv: -kv[1])[:8]
        L.append("- Most expensive ideas: " + ", ".join(
            f"{k} ({sch.ideas[k]['config']['method'] if k in sch.ideas else 'final fit'}) {v:.1f}s" for k, v in top))
        L.append(f"- LLM: {res['llm_calls']} calls, {res['llm_seconds']:.1f}s, ${res['llm_cost_usd']:.4f} reported cost\n")

        L.append("## 10. Agents\n")
        for a in sch.roster:
            L.append(f"- **{a['name']}** ({a['role']}{', persona ' + a['persona'] if a.get('persona') else ''}) — {a['llm']}")
        L.append("")

        disc = self._discussion(sch, rows)
        L.append("## 11. Discussion\n")
        L.append(disc + "\n")

        text = "\n".join(L)
        (sch.run_dir / "report.md").write_text(text, encoding="utf-8")
        self._leaderboard_csv(sch, rows)
        if fin:
            self._model_card(sch, fin)
        self.emit(A.REPORT, {"path": str(sch.run_dir / "report.md")}, stage="report",
                  files=[str(sch.run_dir / "report.md")], summary="final research report")
        return text

    def _discussion(self, sch, rows):
        primary = self.spec.primary
        fams = {}
        for i, r in rows:
            fams.setdefault(i["config"]["method"], []).append(sch.value(r))
        best_by = sorted(((min(v) if M.direction(primary) == "min" else max(v), k) for k, v in fams.items()),
                         reverse=M.direction(primary) == "max")
        auto = ("Best validation " + primary + " per method: " +
                ", ".join(f"{k} {_fmt(v, 4)}" for v, k in best_by) + ". " if best_by else "")
        n_full = len(rows) - 1
        auto += (f"{len(sch.ideas) - 1} ideas were proposed; {len(sch.pruned)} were pruned by the critic or the "
                 f"feasibility screen and {n_full} were trained on the full training set.")
        if not self.uses_llm or not sch.final:
            return auto
        summary = {"leaderboard": [{"idea": i["id"], "method": i["config"]["method"], "features": i["config"]["features"],
                                    "params": i["config"]["params"], "val": sch.value(r)} for i, r in rows[:12]],
                   "pruned": sch.pruned, "final": {k: sch.final[k] for k in ("kind", "members", "val", "test")},
                   "baseline": sch.baseline_value, "critiques": dict(list(sch.critiques.items())[:12])}
        text = self.ask("You are the reporting agent of an autonomous ML research team. Write concise, "
                        "evidence-based scientific discussion in Markdown (no headings).",
                        f"{self.spec.brief()}\n\nResults:\n{json.dumps(summary, default=str)[:12000]}\n\n"
                        "In 2 short paragraphs and a bullet list of 3-5 next experiments: what worked, what "
                        "failed and why (grounded in the numbers), and what to try next.",
                        purpose="report discussion", stage="report")
        return (text.strip() + "\n\n_Automatic summary:_ " + auto) if text else auto

    def _leaderboard_csv(self, sch, rows):
        names = self.spec.metric_names
        with open(sch.run_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "idea", "persona", "origin", "parent", "method", "features", "scaler", "pca",
                        "params"] + [f"val_{m}" for m in names] + ["fit_seconds", "wall_seconds"])
            for k, (i, r) in enumerate(rows, 1):
                c = i["config"]
                w.writerow([k, i["id"], i["persona"], i["origin"], i.get("parent") or "", c["method"], c["features"],
                            c["scaler"], c["pca"], json.dumps(c["params"], default=str)] +
                           [r["val"].get(m) for m in names] + [r.get("fit_seconds"), r.get("wall_seconds")])

    def _model_card(self, sch, fin):
        lines = [f"# Model card — {self.spec.name}", "",
                 f"- Task: {self.spec.task}; data: {self.spec.simulator_description}",
                 f"- Type: {fin['kind']} of {fin['members']} (weights {fin['weights']})",
                 f"- Trained on train+val ({self.spec.data['n_train'] + self.spec.data['n_val']} samples)",
                 f"- Validation estimate ({self.spec.primary}): {_fmt(fin['val'])}",
                 f"- Test: {json.dumps(fin['test'])}", "",
                 "## Members"]
        for iid in fin["members"]:
            lines.append(f"- {iid}: `{json.dumps({k: v for k, v in sch.ideas[iid]['config'].items() if k != 'code'}, default=str)}`")
        from .. import HARNESS_ROOT
        lines += ["", "## Usage", "```python", "import sys", f"sys.path.insert(0, r\"{HARNESS_ROOT}\")",
                  "from mlharness.export import load_model",
                  f"model = load_model(r\"{sch.run_dir}\")",
                  "y_hat = model.predict(X)   # X: raw simulator features, same layout as training", "```"]
        (sch.run_dir / "final" / "model_card.md").write_text("\n".join(lines), encoding="utf-8")
