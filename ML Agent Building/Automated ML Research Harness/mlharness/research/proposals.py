"""Research proposal schema, normalisation, completeness and falsifiable-prediction checks.

A proposal is an immutable *version*; revising it creates a new version with ``parent_id`` pointing
at the old one and the same ``root_id``. Every version is stored as a PROPOSAL artifact and a
research-memory record, so the evolution of an idea is fully reconstructable.

Required content (spec §4): hypothesis, motivating reasoning, assumptions, supporting or
contradictory evidence, proposed method, falsifiable test, computational feasibility, and what is
learned if it fails. ``prediction`` is the machine-checkable form of the falsifiable test:

    {"metric": "rel_l2", "op": "<", "value": 0.02}
    {"relative_to": "best_data_driven" | "best_physics" | "best_other" | "baseline_mean" | <root id>,
     "metric": "rel_l2", "improvement": 0.10, "expect": "better" | "worse"}

``improvement`` is the relative gain of a higher-is-better score; e.g. 0.10 = "at least 10% lower
rel_l2 than the reference"; a negative value means "no more than |x| worse than the reference".
"""
from __future__ import annotations

import copy
import re

from .. import methods_v2 as M2

ACTIONS = ("propose", "revise", "defend", "abandon", "branch")
CLAIM = ("conjectured", "literature-supported", "experimentally-supported", "contradicted", "inconclusive",
         "formally-verified", "abandoned")
APPLICABILITY_CLAIMS = ("yes", "partial", "no", "unknown")
REFERENCES = ("best_data_driven", "best_physics", "best_hybrid", "best_other", "baseline_mean")
REQUIRED = ("hypothesis", "motivation", "assumptions", "evidence", "method", "falsifiable_test", "feasibility",
            "if_fails")
PROOF_WORDS = re.compile(r"\b(proved|proven|proof that|formally verified|we prove|q\.e\.d)\b", re.I)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [i for i in x if i not in (None, "")]
    return [x]


def _norm_prediction(p):
    if not isinstance(p, dict):
        return None
    out = {}
    if "metric" in p:
        out["metric"] = str(p["metric"])
    if p.get("op") in ("<", "<=", ">", ">=") and isinstance(p.get("value"), (int, float)):
        out["op"], out["value"] = p["op"], float(p["value"])
    if p.get("relative_to"):
        out["relative_to"] = str(p["relative_to"])
        try:
            out["improvement"] = float(p.get("improvement", 0.0))
        except (TypeError, ValueError):
            out["improvement"] = 0.0
        out["expect"] = "worse" if str(p.get("expect", "better")).lower().startswith("w") else "better"
    return out if ("op" in out or "relative_to" in out) else None


def normalize_proposal(raw: dict, *, pid: str, author: str, persona: str, round_idx: int, origin: str,
                       parent: dict | None = None, action: str = "propose") -> dict:
    raw = copy.deepcopy(raw or {})
    base = copy.deepcopy(parent) if parent else {}
    def pick(k, default=None):
        v = raw.get(k)
        return v if v not in (None, "", [], {}) else base.get(k, default)
    ev = []
    for e in _as_list(pick("evidence", [])):
        if isinstance(e, str):
            e = {"claim": e, "stance": "supports", "source": "reasoning"}
        if isinstance(e, dict) and e.get("claim"):
            ev.append({"claim": str(e["claim"])[:600],
                       "stance": "contradicts" if str(e.get("stance", "")).lower().startswith("contra") else "supports",
                       "source": str(e.get("source") or "reasoning")[:120]})
    cites = []
    for c in _as_list(pick("citations", [])):
        if isinstance(c, str):
            c = {"ref_id": c}
        if isinstance(c, dict) and c.get("ref_id"):
            claim = str(c.get("claimed_applicability") or c.get("applies") or "unknown").lower()
            cites.append({"ref_id": str(c["ref_id"]), "claimed_applicability": claim if claim in APPLICABILITY_CLAIMS else "unknown",
                          "note": str(c.get("note") or "")[:500]})
    feas = pick("feasibility", {}) or {}
    if isinstance(feas, str):
        feas = {"notes": feas}
    method = pick("method", {}) or {}
    status = str(pick("status", "conjectured")).lower()
    p = {
        "id": pid, "root_id": (parent or {}).get("root_id", pid), "parent_id": (parent or {}).get("id"),
        "version": int((parent or {}).get("version", 0)) + 1, "author": author, "persona": persona,
        "round": int(round_idx), "origin": origin, "action": action,
        "title": str(pick("title", "untitled proposal"))[:160],
        "hypothesis": str(pick("hypothesis", ""))[:1500], "motivation": str(pick("motivation", ""))[:1500],
        "assumptions": [str(a)[:400] for a in _as_list(pick("assumptions", []))][:12],
        "evidence": ev[:12], "citations": cites[:8], "method": method,
        "falsifiable_test": str(pick("falsifiable_test", ""))[:800],
        "prediction": _norm_prediction(pick("prediction")),
        "feasibility": {"cost": str(feas.get("cost", "unknown")), "notes": str(feas.get("notes", ""))[:400],
                        "est_seconds": feas.get("est_seconds")},
        "if_fails": str(pick("if_fails", ""))[:800], "status": status if status in CLAIM else "conjectured",
        "response_to_critic": str(raw.get("response_to_critic") or "")[:1200],
        "abandon_reason": str(raw.get("abandon_reason") or "")[:600],
        "next_queries": [str(q)[:200] for q in _as_list(raw.get("next_queries"))][:3],
        "tags": [str(t) for t in _as_list(pick("tags", []))][:8],
    }
    p["family"] = M2.family(method) if isinstance(method, dict) else "unknown"
    p["method_label"] = M2.label(method) if isinstance(method, dict) and method.get("kind") else "?"
    return p


def compact(p: dict, long: bool = False) -> dict:
    """Prompt-sized view of a proposal: every field kept, long fields trimmed individually (never cut JSON)."""
    n = 900 if long else 450

    def t(x, k=n):
        x = str(x or "")
        return x if len(x) <= k else x[:k] + "…"
    ms = dict(p.get("method") or {})
    if ms.get("code"):
        ms["code"] = t(ms["code"], 1800 if long else 600)
    return {"id": p.get("id"), "author": p.get("author"), "title": t(p.get("title"), 160),
            "hypothesis": t(p.get("hypothesis")), "motivation": t(p.get("motivation")),
            "assumptions": [t(a, 220) for a in (p.get("assumptions") or [])[:8]],
            "evidence": [{"claim": t(e.get("claim"), 220), "stance": e.get("stance"), "source": e.get("source")}
                         for e in (p.get("evidence") or [])[:5]],
            "citations": [{"ref_id": c.get("ref_id"), "claimed_applicability": c.get("claimed_applicability"),
                           "note": t(c.get("note"), 200)} for c in (p.get("citations") or [])[:4]],
            "method": ms, "falsifiable_test": t(p.get("falsifiable_test"), 350), "prediction": p.get("prediction"),
            "feasibility": p.get("feasibility"), "if_fails": t(p.get("if_fails"), 300), "status": p.get("status"),
            "response_to_critic": t(p.get("response_to_critic"), 350)}


def completeness(p: dict) -> tuple[float, list[str]]:
    missing = []
    for k in REQUIRED:
        v = p.get(k)
        if k == "method":
            if not isinstance(v, dict) or not v.get("kind"):
                missing.append("method (an executable method spec with 'kind')")
        elif k == "feasibility":
            if not v or (v.get("cost") in (None, "unknown") and not v.get("notes")):
                missing.append("computational feasibility")
        elif not v:
            missing.append(k.replace("_", " "))
    return 1.0 - len(missing) / len(REQUIRED), missing


def claim_hygiene(p: dict, literature) -> list[str]:
    """Downgrade unsupported claim statuses. Returns notes describing each change (mutates p)."""
    notes = []
    if p["status"] == "formally-verified":
        p["status"] = "conjectured"
        notes.append("status 'formally-verified' removed: no formal verification was performed by the harness")
    text = " ".join([p.get("hypothesis", ""), p.get("motivation", "")] + [e["claim"] for e in p.get("evidence", [])])
    if PROOF_WORDS.search(text):
        notes.append("the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem "
                     "is not a proof of this claim — treated as conjecture until tested")
    verified = [c for c in p.get("citations", []) if literature is not None and literature.known(c["ref_id"])]
    unknown = [c["ref_id"] for c in p.get("citations", []) if literature is None or not literature.known(c["ref_id"])]
    if unknown:
        notes.append(f"unverifiable citation(s) {unknown}: not retrieved in this run or found in research memory")
    supportive = []
    for c in verified:
        entry = literature.get(c["ref_id"]) or {}
        screen = (entry.get("assessment") or {}).get("applies", "unknown")
        c["screen"] = screen
        if c["claimed_applicability"] in ("yes", "partial") and screen in ("plausible", "partial", "unknown"):
            supportive.append(c["ref_id"])
    if p["status"] == "literature-supported" and not supportive:
        p["status"] = "conjectured"
        notes.append("status 'literature-supported' needs at least one verified citation whose assumptions were "
                     "checked; downgraded to conjectured")
    if p["status"] in ("experimentally-supported", "contradicted"):
        p["status"] = "conjectured"
        notes.append("experimental status can only be assigned by the harness after experiments")
    return notes


def check_prediction(pred: dict | None, own_value, primary: str, refs: dict) -> dict:
    """Evaluate a structured prediction against validation results.

    refs: {"best_data_driven": v, "best_physics": v, "best_other": v, "baseline_mean": v, <root_id>: v}"""
    from .. import metrics as M
    if not pred:
        return {"outcome": "untestable", "detail": "no structured prediction"}
    if own_value is None:
        return {"outcome": "inconclusive", "detail": "method did not produce a full-fidelity validation result"}
    metric = pred.get("metric") or primary
    if metric != primary:
        return {"outcome": "inconclusive", "detail": f"prediction is about {metric}; only {primary} is compared"}
    out = {"own": own_value}
    if "op" in pred:
        v = pred["value"]
        ok = {"<": own_value < v, "<=": own_value <= v, ">": own_value > v, ">=": own_value >= v}[pred["op"]]
        out.update(outcome="supported" if ok else "contradicted",
                   detail=f"val {metric} = {own_value:.5g} vs predicted {pred['op']} {v:.5g}")
        return out
    ref_name = pred["relative_to"]
    ref = refs.get(ref_name)
    if ref is None:
        return {"outcome": "inconclusive", "detail": f"reference '{ref_name}' has no result"}
    gain = M.relative_gain(M.to_score(own_value, metric), M.to_score(ref, metric))
    want = float(pred.get("improvement", 0.0))
    if pred.get("expect") == "worse":
        ok = gain < want
        txt = f"expected to be worse than {ref_name} (gain < {want:+.1%})"
    else:
        ok = gain >= want
        txt = f"expected gain >= {want:+.1%} vs {ref_name}"
    out.update(outcome="supported" if ok else "contradicted", gain=gain, reference=ref,
               detail=f"val {metric} {own_value:.5g} vs {ref_name} {ref:.5g}: gain {gain:+.2%}; {txt}")
    return out
