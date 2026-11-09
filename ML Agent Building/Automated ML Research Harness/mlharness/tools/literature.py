"""Literature tool used by the research agents and the critic (wraps TheoremSearch).

Responsibilities
* enforce query budgets (per agent per round, and per run);
* record every query as a LITERATURE artifact and a research-memory record (useful results and
  failures alike), so the report can state whether the literature check was complete;
* keep a registry of retrieved references so a proposal's citations can be *verified* (a citation
  whose ``ref_id`` was never retrieved in this run, or found in memory, is flagged as unverifiable);
* screen theorem assumptions against the problem's structural facts with a transparent keyword
  heuristic. The screen returns ``plausible | partial | no | unknown`` — it never certifies that a
  theorem applies, and retrieval is never treated as proof of a proposal's claim.
"""
from __future__ import annotations

import re
import threading

from .. import artifacts as A
from .theoremsearch import RetrievalResult, TheoremRef, TheoremSearchClient

APPLIES = ("plausible", "partial", "no", "unknown")

_LATEX = re.compile(r"\\(?:mathbb|mathcal|mathrm|operatorname|left|right|text)\b|[{}$]")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _LATEX.sub(" ", text or "")).lower()


def split_statement(body: str) -> tuple[str, str]:
    """Rough split of a statement into (hypotheses, conclusion) at the first 'then'."""
    t = _clean(body)
    m = re.search(r"\bthen\b", t)
    return (t[: m.start()], t[m.end():]) if m else (t, "")


def assess_applicability(ref: TheoremRef | dict, facts: dict) -> dict:
    """Keyword screen of a retrieved statement against problem facts. Transparent and conservative."""
    body = ref.body if isinstance(ref, TheoremRef) else (ref.get("body") or "")
    slogan = ref.slogan if isinstance(ref, TheoremRef) else (ref.get("slogan") or "")
    hyp, _ = split_statement(body)
    text = _clean(body + " " + slogan)
    f = {k: str(v).lower() for k, v in (facts or {}).items()}
    matched, conflicts, notes = [], [], []
    severe = False
    eq = f.get("pde", "")
    if eq:
        if any(w in text for w in ("parabolic", "heat equation", "heat conduction", "diffusion", "u_t", "\\partial_t u")):
            matched.append("same equation class (parabolic / diffusion)")
        elif any(w in text for w in ("elliptic", "wave equation", "hyperbolic", "schr", "navier", "maxwell")) and \
                not any(w in text for w in ("parabolic", "heat", "diffusion")):
            conflicts.append("different equation class (theorem is not about parabolic/diffusion equations)")
            severe = True
    if any(w in text for w in ("stochastic", "brownian", "white noise", "spde", "martingale")):
        conflicts.append("stochastic PDE setting; the problem is a deterministic PDE with measurement noise")
        severe = True
    dim = f.get("dimension", "")
    if dim == "1":
        if re.search(r"(n|d)\s*\\?(geq|>=|ge)\s*2|n\s*>\s*1|multi-?dimensional|\\?r\^\{?[nd3]", hyp):
            conflicts.append("theorem is stated for dimension >= 2; the problem is one-dimensional")
            severe = True
        elif re.search(r"one-dimensional|1-d|1d\b|d\s*=\s*1|n\s*=\s*1|interval|\(0,\s*1\)|\[0,\s*1\]|\(0,\s*l\)|\[0,\s*l\]|0<x<", text):
            matched.append("one-dimensional setting")
    reg = f.get("coefficient_regularity", "")
    about_coeff = re.search(r"coefficient|conductiv|diffusiv|a\(x\)|k\(x\)|sigma\(x\)|\\sigma\(x\)|p\(x\)|q\(x\)", text)
    if ("piecewise" in reg or "discontinu" in reg) and about_coeff:
        if re.search(r"c\^\{?[12]|c\^\{?\\?infty|smooth|lipschitz|h\^\{?[12]|w\^\{?1|continuously differentiable|analytic|c\^1|c\^2", hyp):
            conflicts.append("theorem assumes a smooth/regular coefficient; the problem's conductivity is "
                             "piecewise-constant (discontinuous) — the result may not transfer directly")
        if re.search(r"piecewise|discontinu|bv\b|bounded variation|l\^\{?\\?infty", text):
            matched.append("allows non-smooth / piecewise coefficients")
    bc = f.get("boundary_conditions", "")
    if "dirichlet" in bc:
        if "dirichlet" in text:
            matched.append("Dirichlet boundary conditions")
        elif any(w in text for w in ("neumann", "robin")):
            conflicts.append("theorem uses Neumann/Robin boundary conditions; problem has Dirichlet")
    obs = f.get("observations", "")
    if obs:
        if any(w in text for w in ("final time", "final temperature", "u(x,t)", "terminal", "at t=t", "u(\\cdot,t)")):
            conflicts.append("theorem uses final-time (full spatial) data; the problem observes point sensors over time")
        elif any(w in text for w in ("dirichlet-to-neumann", "boundary measurement", "cauchy data", "flux measurement")):
            conflicts.append("theorem uses boundary measurements; the problem uses interior point sensors")
        elif re.search(r"x_\{?0\}?|fixed point|single point|interior point|one point|point measurement|finitely many points", text):
            matched.append("point measurements over time")
    if f.get("noise"):
        if any(w in text for w in ("stability", "stable", "lipschitz stability", "holder", "hölder", "error estimate",
                                    "convergence rate", "noisy data", "noise level")):
            matched.append("quantitative stability / noise statement")
        elif any(w in text for w in ("uniqueness", "unique", "uniquely", "identical", "determined")):
            notes.append("uniqueness from exact data only: says nothing about stability under the sensor noise")
    if any(w in text for w in ("levenberg", "gauss-newton", "least squares", "tikhonov", "regulariz", "regularis",
                               "maximum likelihood", "cramer", "cramér", "fisher information", "posterior")):
        notes.append("method-level result: applies if its algorithmic assumptions (e.g. full-rank Jacobian, "
                     "well-specified noise model) hold — check numerically")
    if severe:
        verdict = "no"
    elif conflicts:
        verdict = "partial"
    elif len(matched) >= 2 and (any(m.startswith("same equation class") for m in matched) or
                                any(n.startswith("method-level") for n in notes)):
        verdict = "plausible"
    else:
        verdict = "unknown"
    return {"applies": verdict, "matched": matched, "conflicts": conflicts, "notes": notes,
            "method": "keyword screen of the statement against problem facts (heuristic, not a proof check)"}


class LiteratureTool:
    def __init__(self, client: TheoremSearchClient, store=None, memory=None, resources=None, facts: dict | None = None,
                 max_total: int = 40, per_agent_round: int = 2, n_results: int = 4, context: dict | None = None,
                 run_id: str | None = None):
        self.client = client
        self.store = store
        self.memory = memory
        self.resources = resources
        self.facts = dict(facts or {})
        self.max_total = int(max_total)
        self.per_agent_round = int(per_agent_round)
        self.n_results = int(n_results)
        self.context = dict(context or {})
        self.run_id = run_id
        self.refs: dict[str, dict] = {}         # ref_id -> {ref, assessment}
        self.log: list[dict] = []
        self.failures: list[dict] = []
        self.skipped_budget = 0
        self._counts: dict[tuple, int] = {}
        self._total = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.client.enabled

    def remaining(self) -> int:
        return max(0, self.max_total - self._total)

    def incomplete(self) -> bool:
        return bool(self.failures) or self.skipped_budget > 0 or not self.enabled

    def status(self) -> dict:
        return {"enabled": self.enabled, "queries": self._total, "failures": len(self.failures),
                "skipped_for_budget": self.skipped_budget, "refs": len(self.refs), "incomplete": self.incomplete(),
                "failure_kinds": sorted({f["error_kind"] for f in self.failures})}

    def known(self, ref_id: str) -> bool:
        if ref_id in self.refs:
            return True
        if self.memory is not None:
            return self.memory.has_citation(ref_id)
        return False

    def get(self, ref_id: str) -> dict | None:
        if ref_id in self.refs:
            return self.refs[ref_id]
        if self.memory is not None:
            rec = self.memory.citation(ref_id)
            if rec:
                return {"ref": rec, "assessment": rec.get("assessment") or {}}
        return None

    def search(self, query: str, agent: str, round_idx: int, purpose: str = "") -> RetrievalResult:
        query = (query or "").strip()[:300]
        with self._lock:
            k = (agent, round_idx)
            if not self.enabled:
                res = RetrievalResult(ok=False, query=query, error="literature search disabled", error_kind="disabled")
            elif self._total >= self.max_total or self._counts.get(k, 0) >= self.per_agent_round:
                self.skipped_budget += 1
                res = RetrievalResult(ok=False, query=query, error="literature query budget reached", error_kind="budget")
            else:
                self._counts[k] = self._counts.get(k, 0) + 1
                self._total += 1
                res = None
        if res is None:
            res = self.client.search(query, self.n_results)
            if self.resources is not None:
                self.resources.charge_literature(res.ok)
        assessed = []
        for r in res.refs:
            a = assess_applicability(r, self.facts)
            entry = {"ref": r.to_dict(1500), "assessment": a, "citation": r.citation()}
            assessed.append(entry)
            with self._lock:
                self.refs.setdefault(r.ref_id, entry)
        rec = {"query": query, "agent": agent, "round": round_idx, "purpose": purpose, "ok": res.ok,
               "error": res.error, "error_kind": res.error_kind, "endpoint": res.endpoint, "cached": res.cached,
               "fallback_used": res.fallback_used, "latency": round(res.latency, 2), "attempts": res.attempts,
               "results": [{"ref_id": e["ref"]["ref_id"], "name": e["ref"]["name"], "kind": e["ref"]["kind"],
                            "paper_title": e["ref"]["paper_title"], "authors": e["ref"]["authors"][:6],
                            "year": e["ref"]["year"], "link": e["ref"]["link"], "similarity": e["ref"]["similarity"],
                            "slogan": (e["ref"]["slogan"] or "")[:400], "body": (e["ref"]["body"] or "")[:900],
                            "applies": e["assessment"]["applies"], "conflicts": e["assessment"]["conflicts"],
                            "matched": e["assessment"]["matched"], "notes": e["assessment"]["notes"]}
                           for e in assessed]}
        with self._lock:
            self.log.append(rec)
            if not res.ok and res.error_kind not in ("budget",):
                self.failures.append(rec)
        art = None
        if self.store is not None:
            summ = (f"{agent} r{round_idx}: '{query[:70]}' -> {len(res.refs)} refs" if res.ok
                    else f"{agent} r{round_idx}: '{query[:70]}' FAILED ({res.error_kind})")
            art = self.store.put(A.LITERATURE, rec, author=agent, stage="research", summary=summ)
            if not res.ok and res.error_kind not in ("budget", "disabled"):
                self.store.put(A.FAILURE, {"source": "literature", "error_kind": res.error_kind, "error": res.error,
                                           "query": query}, author=agent, stage="research", parents=[art.id],
                               summary=f"literature check incomplete: {res.error_kind} for '{query[:60]}'")
        if self.memory is not None and res.error_kind != "budget":
            self.memory.add("literature", title=f"TheoremSearch: {query[:100]}", author=agent, run_id=self.run_id,
                            text=query + " " + " ".join((e["ref"]["slogan"] or "") for e in assessed),
                            status="retrieved" if res.ok else "retrieval-failed", context=self.context,
                            data={"query": query, "ok": res.ok, "error_kind": res.error_kind,
                                  "results": rec["results"][:6]},
                            citations=[e["ref"]["ref_id"] for e in assessed],
                            artifact_refs=[{"artifact_id": art.id}] if art else [])
            for e in assessed:
                self.memory.add_citation(e["ref"], e["assessment"], query=query, run_id=self.run_id)
        return res

    def digest(self, results: list[RetrievalResult] | None = None, limit: int = 8) -> str:
        """Compact text of retrieved statements for an LLM prompt (ref ids + assumptions + screen)."""
        items = []
        source = [e for r in (results or []) for e in (self.refs.get(x.ref_id) for x in r.refs) if e] \
            if results is not None else list(self.refs.values())
        seen = set()
        for e in source:
            rid = e["ref"]["ref_id"]
            if rid in seen:
                continue
            seen.add(rid)
            a = e["assessment"]
            items.append(f"[{rid}] {e['citation']}\n   statement: {(e['ref']['body'] or e['ref']['slogan'] or '')[:500]}\n"
                         f"   harness applicability screen: {a['applies']}"
                         + (f"; conflicts: {'; '.join(a['conflicts'])}" if a["conflicts"] else "")
                         + (f"; matched: {', '.join(a['matched'])}" if a["matched"] else ""))
            if len(items) >= limit:
                break
        return "\n".join(items) or "(no statements retrieved)"
