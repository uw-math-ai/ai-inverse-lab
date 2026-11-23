"""Reconstruct a run's research state purely from its artifacts.

Artifacts are the persistent research memory, so everything the reporter needs (ideas, results at
each fidelity, pruning decisions, the final model and its audit) can be rebuilt from
``artifacts.jsonl`` + ``spec.json`` without the scheduler process. Used by ``mlharness report --rebuild``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import artifacts as A
from . import metrics as M
from .spec import ResearchSpec


class _Resources:
    def __init__(self, snap: dict, by_idea: dict):
        self._snap, self.by_idea = snap, by_idea

    def snapshot(self):
        return self._snap


class _LLMLabel:
    def __init__(self, label):
        self.label, self.is_null = label, True

    def describe(self):
        return self.label


class RunState:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.store = A.FileSystemStore(self.run_dir, resume=True)
        d = json.loads((self.run_dir / "spec.json").read_text())
        self.spec = ResearchSpec(raw={}, **d)
        status = json.loads((self.run_dir / "status.json").read_text()) if (self.run_dir / "status.json").exists() else {}
        self.run_id = status.get("run_id", self.run_dir.name)
        self.funnel = status.get("funnel", {})
        self.ideas, self.results, self.pruned, self.outcomes, self.critiques = {}, {}, {}, {}, {}
        for a in self.store.query(A.HYPOTHESIS):
            idea = dict(a.data)
            idea["hyp_artifact"] = a.id
            self.ideas[idea["id"]] = idea
        for a in self.store.query(A.RESULT):
            if a.data.get("mode") == "final" or a.idea_id not in self.ideas:
                continue
            self.results.setdefault(a.idea_id, {})[float(a.data.get("fraction") or 1.0)] = a.data
        for a in self.store.query(A.DECISION):
            dec, iid = a.data.get("decision"), a.idea_id
            if iid and dec == "prune":
                self.pruned[iid] = a.data["reason"]
            if iid and dec in ("not_promoted", "budget_cut", "prune", "promote"):
                self.outcomes[iid] = f"{dec}: {a.data['reason']}"
        stops = [a.data["reason"] for a in self.store.query(A.DECISION) if a.data.get("decision") == "stop_refinement"]
        self.stop_reason = stops[-1] if stops else None
        for a in self.store.query(A.CRITIQUE):
            if a.data.get("type") == "result_review" and a.data.get("explanation") not in (None, "ok"):
                self.critiques[a.idea_id] = a.data["explanation"]
        self.roster = [a.data for a in self.store.query(A.AGENT)]
        self.llm = _LLMLabel(self.roster[0]["llm"] if self.roster else "unknown")
        by_idea = {}
        for a in self.store.query(A.RESOURCE, where=lambda x: x.data.get("type") == "experiment"):
            by_idea[a.idea_id] = by_idea.get(a.idea_id, 0.0) + float(a.data.get("seconds", 0))
        self.resources = _Resources(status.get("resources", {}), by_idea)
        base = [i for i in self.ideas.values() if i.get("persona") == "baseline"]
        self.baseline_value = self.value(self.results.get(base[0]["id"], {}).get(1.0)) if base else None
        self.final = self._final()

    def value(self, res):
        return (res.get("val") or {}).get(self.spec.primary) if res and res.get("status") == "ok" else None

    def score(self, res):
        return M.to_score(self.value(res), self.spec.primary)

    def _final(self):
        ck = self.store.latest(A.CHECKPOINT)
        if ck is None:
            return None
        audits = [a for a in self.store.query(A.CRITIQUE) if a.data.get("type") == "final_audit"]
        return {"kind": ck.data["kind"], "members": ck.data["members"], "weights": ck.data["weights"],
                "val": ck.data.get("val_estimate"), "test": ck.data["test"], "model_path": ck.data["path"],
                "audit": audits[-1].data if audits else {}, "checkpoint": ck.id}


def rebuild_report(run_dir, llm=None) -> str:
    from .agents import ReporterAgent
    st = RunState(run_dir)
    if llm is not None and not llm.is_null:
        st.llm = llm
    from .resources import ResourceManager
    res = ResourceManager(1, 1, 1, [], 1 if llm is not None and not llm.is_null else 0)
    rep = ReporterAgent("reporter", st.spec, st.store, llm, res)
    return rep.write(st)
