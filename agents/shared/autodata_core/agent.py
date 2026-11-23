"""Website entrypoint shared by agents/autodata-inverse and agents/autodata-inverse-llm.

Protocol (agents/instructions.md, section 3): read AGENT_INPUT_PATH, write artifacts to
AGENT_OUTPUT_DIR, log with flush, emit {"progress": x} lines, exit nonzero on failure. Reports usage
and honours the handoff signal through agents/shared/platform_usage.py; fans scenarios out to
subagents through agents/shared/platform_swarm.py when `parallel` is set.

Artifacts (a handful of files; the platform keeps at most 100):
  report.md        human-readable summary
  result.json      machine-readable summary (config, statistics, per-scenario outcome)
  dataset.jsonl    one accepted inverse-problem instance per line, with its arrays
  rounds.jsonl     every round of every scenario (spec, verifier diagnostics, scores, challenger note)
  meta_history.jsonl, best_strategy.json   (mode = meta)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import platform_usage

from .challenger import merge_strategy
from .config import ConfigError, build
from .llm import make_llm
from .loop import item_record, make_challenger, run_scenario, summarize
from .meta import optimize

STOP_MARGIN_S = 120          # stop starting new rounds this long before the time budget ends


def log(msg):
    print(msg, flush=True)


def progress(x):
    print(json.dumps({"progress": round(min(max(x, 0.0), 1.0), 4)}), flush=True)


def _atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _jsonl(rows):
    return "".join(json.dumps(r, default=str) + "\n" for r in rows)


class Run:
    def __init__(self, variant: str, default_budget_s: int | None = None):
        self.variant, self.default_budget_s = variant, default_budget_s
        self.out = Path(os.environ["AGENT_OUTPUT_DIR"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.inputs = json.loads(Path(os.environ["AGENT_INPUT_PATH"]).read_text(encoding="utf-8"))
        self.t0 = time.monotonic()
        self.items, self.results, self.stop_reason = [], [], None

    # ---------------------------------------------------------- stop conditions
    def should_stop(self):
        if platform_usage.handoff_requested():
            self.stop_reason = "handoff"
            return True
        budget = self.cfg["time_budget_seconds"]
        if budget and time.monotonic() - self.t0 > max(budget - STOP_MARGIN_S, budget * 0.5):
            self.stop_reason = "time_budget"
            return True
        return False

    # ---------------------------------------------------------- outputs
    def write_run_outputs(self, final=False, extra=None):
        summary = summarize(self.results)
        if self.cfg["mode"] == "run":
            _atomic(self.out / "rounds.jsonl", _jsonl(
                {"scenario": r["scenario"], **rec} for r in self.results for rec in r["history"]))
            _atomic(self.out / "dataset.jsonl", _jsonl(self.items))
        result = {"agent": self.variant, "mode": self.cfg["mode"], "completed": final and not self.stop_reason,
                  "stop_reason": self.stop_reason, "summary": summary,
                  "config": {k: v for k, v in self.cfg.items() if k != "scenarios"},
                  "scenario_ids": [s["id"] for s in self.cfg["scenarios"]],
                  "challenger": self.challenger_name, "elapsed_s": round(time.monotonic() - self.t0, 1)}
        if extra:
            result.update(extra)
        _atomic(self.out / "result.json", json.dumps(result, indent=1, default=str))
        return summary

    def report(self, summary, meta=None):
        lines = ["# Autodata run: inverse problem for vertex centrality", "",
                 f"- agent: `{self.variant}`, mode `{self.cfg['mode']}`, challenger `{self.challenger_name}`",
                 f"- scenarios: {summary['scenarios']} run, {summary['scenarios_accepted']} produced items, "
                 f"{summary['items']} items in {summary['rounds']} rounds",
                 f"- mean rounds to acceptance: {summary['mean_rounds_to_accept']}",
                 f"- failure share: {summary['failure_share']}  (verifier codes {summary['qv_codes']})",
                 f"- accepted weak / strong average score: {summary['accepted_weak_avg']} / "
                 f"{summary['accepted_strong_avg']}",
                 f"- elapsed: {time.monotonic() - self.t0:.0f} s"
                 + (f"; stopped early ({self.stop_reason})" if self.stop_reason else ""), "",
                 "| scenario | accepted | rounds | items |", "|---|---|---|---|"]
        for sid, ps in summary["per_scenario"].items():
            lines.append(f"| {sid} | {ps['accepted']} | {ps['rounds']} | {', '.join(ps['items'])} |")
        if meta:
            lines += ["", "## Meta-optimization", "",
                      f"- iterations run: {meta['iterations_run']} / {meta['iterations']}, accepted mutations: "
                      f"{meta['accepted']}",
                      f"- validation score: baseline {meta['baseline_score']:.4f} -> best {meta['best_score']:.4f} "
                      f"(pass rate {meta['best_pass_rate']:.3f})",
                      "- best strategy: `best_strategy.json` (pass it back as the `strategy` input)"]
        lines += ["", "Files: `dataset.jsonl` (accepted instances with arrays), `rounds.jsonl` (every round), "
                      "`result.json` (summary)."]
        _atomic(self.out / "report.md", "\n".join(lines) + "\n")

    # ---------------------------------------------------------- modes
    def on_accept(self, item_id, scenario, rec, payload):
        self.items.append(item_record(item_id, scenario, rec, payload, self.cfg["toggles"]["save_observations"]))

    def run_mode(self, llm):
        cfg = self.cfg
        strategy = merge_strategy(cfg["strategy"])
        challenger = make_challenger(cfg, strategy, llm)
        self.challenger_name = challenger.name
        seen = set()
        total = len(cfg["scenarios"])
        for i, s in enumerate(cfg["scenarios"]):
            if self.should_stop():
                break
            log(f"scenario {i + 1}/{total}: {s['id']}")
            res = run_scenario(s, cfg, challenger, seen if cfg["toggles"]["final_verifier"] else set(),
                               on_accept=self.on_accept, should_stop=self.should_stop, log=log)
            self.results.append(res)
            self.write_run_outputs()
            progress((i + 1) / total)
        summary = self.write_run_outputs(final=True)
        self.report(summary)
        return summary

    def meta_mode(self, llm):
        cfg = self.cfg
        self.challenger_name = "llm" if llm is not None and not llm.is_null else "heuristic"
        meta_llm = llm if cfg["llm_meta"] else None

        def on_iteration(it, history, population):
            _atomic(self.out / "meta_history.jsonl", _jsonl(history))
            best = max(population, key=lambda c: c["score"])
            _atomic(self.out / "best_strategy.json", json.dumps(best["strategy"], indent=1))
            progress(it / max(cfg["meta"]["iterations"], 1))

        res = optimize(cfg, llm=llm, meta_llm=meta_llm, should_stop=self.should_stop, log=log,
                       on_iteration=on_iteration)
        _atomic(self.out / "meta_history.jsonl", _jsonl(res["history"]))
        _atomic(self.out / "best_strategy.json", json.dumps(res["best_strategy"], indent=1))
        meta = {k: v for k, v in res.items() if k not in ("history", "population", "best_strategy")}
        summary = self.write_run_outputs(final=True, extra={"meta": meta, "population": [
            {k: v for k, v in c.items() if k != "strategy"} for c in res["population"]]})
        self.report(summary, meta)
        return summary

    def parallel_mode(self):
        """Fan scenarios out to subagents (one per scenario), then merge their outputs."""
        import platform_swarm as swarm
        cfg = self.cfg
        self.challenger_name = "heuristic (subagents)"
        st = swarm.status()
        servers = []
        for srv in st.get("servers") or []:
            label = srv if isinstance(srv, str) else (srv.get("label") or srv.get("name") or srv.get("id"))
            if label:
                servers.append(str(label))
        log(f"parallel: {len(cfg['scenarios'])} scenarios over servers {servers or ['(this server)']}")
        base = {k: v for k, v in self.inputs.items() if k not in ("parallel", "scenario_ids", "skip_scenarios")}
        queue = list(enumerate(cfg["scenarios"]))
        running, done, max_children = {}, 0, 10
        while queue or running:
            while queue and len(running) < max_children:
                i, s = queue.pop(0)
                cid = f"s{i:02d}-" + re.sub(r"[^a-z0-9_-]", "-", s["id"].lower())[:56]
                child_inputs = {**base, "scenario_ids": [s["id"]]}
                if self.inputs.get("custom_scenarios"):
                    child_inputs["custom_scenarios"] = [s]
                kwargs = {"inputs": child_inputs, "connection": "none"}
                if servers:
                    kwargs["server"] = servers[i % len(servers)]
                swarm.spawn(cid, **kwargs)
                running[cid] = s["id"]
                log(f"spawned {cid}" + (f" on {kwargs['server']}" if servers else ""))
            child = next(swarm.wait(list(running), timeout=None))
            sid = running.pop(child["id"])
            done += 1
            outdir = swarm.output_dir(child["id"]) if child.get("status") != "rejected" else None
            rounds = outdir / "rounds.jsonl" if outdir else None
            if child.get("status") == "succeeded" and rounds and rounds.exists():
                hist = [json.loads(l) for l in rounds.read_text(encoding="utf-8").splitlines() if l.strip()]
                for rec in hist:
                    rec.pop("scenario", None)
                accepted = [r["item_id"] for r in hist if r["status"] == "accepted"]
                self.results.append({"scenario": sid, "brief": "", "accepted": bool(accepted), "items": accepted,
                                     "rounds": len(hist), "complete": True, "seconds": 0.0, "history": hist})
                ds = outdir / "dataset.jsonl"
                if ds.exists():
                    self.items += [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()]
                log(f"{child['id']}: {len(accepted)} item(s) in {len(hist)} rounds")
            else:
                log(f"{child['id']}: {child.get('status')} ({child.get('error') or 'no outputs'})")
                self.results.append({"scenario": sid, "brief": "", "accepted": False, "items": [], "rounds": 0,
                                     "complete": False, "seconds": 0.0, "history": []})
            self.write_run_outputs()
            progress(done / len(cfg["scenarios"]))
        summary = self.write_run_outputs(final=True)
        self.report(summary)
        return summary

    # ---------------------------------------------------------- main
    def main(self):
        self.challenger_name = "-"
        try:
            self.cfg = build(self.inputs)
        except ConfigError as exc:
            print(f"invalid input: {exc}", file=sys.stderr, flush=True)
            return 2
        cfg = self.cfg
        if not cfg["time_budget_seconds"]:
            cfg["time_budget_seconds"] = self.default_budget_s
        if not cfg["scenarios"]:
            print("invalid input: no scenarios left to run", file=sys.stderr, flush=True)
            return 2
        log(f"autodata ({self.variant}): mode {cfg['mode']}, {len(cfg['scenarios'])} scenarios, "
            f"max_rounds {cfg['loop']['max_rounds']}, attempts {cfg['evaluation']['attempts']}, "
            f"noise {cfg['evaluation']['noise']}")
        progress(0.0)
        llm = None
        if self.variant == "llm":
            llm = make_llm()
            if llm.is_null:
                print("this agent needs a model connection (AGENT_PROVIDER is not set); "
                      "use autodata-inverse for the free heuristic challenger", file=sys.stderr, flush=True)
                return 2
            log(f"challenger model provider: {llm.provider}" + (f" ({llm.model})" if llm.model else ""))
        try:
            if cfg["mode"] == "meta":
                summary = self.meta_mode(llm)
            elif cfg["parallel"] and self.variant == "free":
                import platform_swarm as swarm
                if swarm.available() and not swarm.child_id():
                    summary = self.parallel_mode()
                else:
                    log("parallel requested but subagents are not available here; running sequentially")
                    summary = self.run_mode(llm)
            else:
                summary = self.run_mode(llm)
        except Exception:
            traceback.print_exc()
            try:
                self.write_run_outputs()
            except Exception:
                pass
            return 1
        if llm is not None:
            log(f"model calls: {llm.calls}, tokens reported: {llm.tokens}")
        if self.stop_reason == "handoff":
            done = [r["scenario"] for r in self.results if r.get("complete", True)]
            remaining = [s["id"] for s in cfg["scenarios"] if s["id"] not in done]
            platform_usage.write_handoff(
                f"Stopped because the model plan reached its handoff threshold. {summary['items']} items from "
                f"{len(done)} scenarios are in dataset.jsonl.",
                completed=done, next_steps=[f"run the remaining scenarios: {remaining}"],
                state={"skip_scenarios": done})
        log(f"done: {summary['items']} items from {summary['scenarios_accepted']}/{summary['scenarios']} "
            f"scenarios in {summary['rounds']} rounds" + (f" (stopped: {self.stop_reason})" if self.stop_reason else ""))
        progress(1.0)
        return 0


def main(variant: str, default_budget_s: int | None = None) -> int:
    """variant: "free" (heuristic challenger, optional subagents) or "llm" (model challenger)."""
    return Run(variant, default_budget_s).main()
