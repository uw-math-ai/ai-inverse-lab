"""Select a DIVERSE, feasible research portfolio (plus baselines) at the end of the research phase.

Consensus is not forced: the best proposal of every method family is taken first, then remaining
slots are filled by critic quality with a penalty for families already represented; exact method
duplicates are skipped. Proposals with blocking critic issues are excluded (with reasons). Every
selected entry carries the critic's unresolved points so the report can document them.

Baselines (when ``execution.include_baselines``): a conventional ML default, a random-search control,
and — when an authorised forward model exists — a default physics MLE. The mean predictor is the
scheduler's reference baseline and is always run.
"""
from __future__ import annotations

from .. import methods_v2 as M2


def _unresolved(fb: dict, k: int = 4) -> list[str]:
    items = []
    for key in ("blocking", "challenges", "counterexamples", "questions", "revision_directions"):
        items += [f"{key[:-1] if key.endswith('s') else key}: {x}" for x in (fb.get(key) or [])]
    return items[:k]


def baseline_methods(available: dict, forward_available: bool) -> list[dict]:
    key = "torch_mlp" if "torch_mlp" in available else ("mlp_sklearn" if "mlp_sklearn" in available else "ridge")
    m = available.get(key)
    out = [{"title": f"conventional ML baseline ({key}, library defaults)", "family": "data-driven",
            "method": {"kind": "library", "method": key, "params": dict(m.defaults) if m else {}, "features": "raw",
                       "scaler": "standard", "pca": 0},
            "why": "strong conventional learned-inverse default"},
           {"title": "random-search control (2 uniform random library configs)", "family": "data-driven",
            "method": {"kind": "random_search", "n": 2}, "why": "basic random search control"}]
    if forward_available:
        out.append({"title": "physics baseline: Levenberg-Marquardt MLE (defaults)", "family": "physics",
                    "method": {"kind": "physics", "params": dict(M2.PHYSICS.defaults)},
                    "why": "conventional optimisation through the forward model"})
    return out


def select_portfolio(active: list[dict], critique: dict | None, cfg: dict, norm_fn, forward_available: bool,
                     available: dict) -> tuple[list[dict], list[dict]]:
    """norm_fn(raw_method) -> (normalized_spec, notes, errors). Returns (entries, excluded)."""
    per = (critique or {}).get("per_proposal", {})
    size = int(cfg.get("portfolio_size", 6))
    cands, excluded = [], []
    for p in active:
        fb = per.get(p["id"], {})
        if fb.get("blocking"):
            excluded.append({"proposal_id": p["id"], "title": p["title"], "reason": "blocking: " + "; ".join(fb["blocking"])[:300]})
            continue
        ms, _, errors = norm_fn(p.get("method") or {})
        if errors:
            excluded.append({"proposal_id": p["id"], "title": p["title"], "reason": "; ".join(errors)[:300]})
            continue
        cands.append((float(fb.get("quality", 0.0)), p, ms))
    cands.sort(key=lambda t: (-t[0], t[1]["id"]))
    chosen, hashes, fam_count = [], set(), {}

    def take(q, p, ms, why):
        h = M2.method_hash(ms) if ms.get("kind") != "random_search" else f"rs-{p['id']}"
        if h in hashes:
            excluded.append({"proposal_id": p["id"], "title": p["title"], "reason": "same executable method as a "
                                                                                   "selected proposal"})
            return False
        hashes.add(h)
        fam = M2.family(ms)
        fam_count[fam] = fam_count.get(fam, 0) + 1
        chosen.append({"source": "proposal", "proposal_id": p["id"], "root_id": p["root_id"], "title": p["title"],
                       "author": p["author"], "persona": p["persona"], "family": fam, "method": p["method"],
                       "quality": q, "prediction": p.get("prediction"), "selection_reason": why,
                       "unresolved": _unresolved(per.get(p["id"], {})), "status": p.get("status")})
        return True

    seen_fam = set()
    for q, p, ms in cands:                        # pass 1: best proposal per family
        fam = M2.family(ms)
        if fam not in seen_fam and len(chosen) < size:
            if take(q, p, ms, f"best-rated proposal in family '{fam}' (quality {q:.2f})"):
                seen_fam.add(fam)
    rest = [(q, p, ms) for q, p, ms in cands if p["id"] not in {c["proposal_id"] for c in chosen}]
    while rest and len(chosen) < size:            # pass 2: quality with a same-family penalty
        rest.sort(key=lambda t: -(t[0] - 0.1 * fam_count.get(M2.family(t[2]), 0)))
        q, p, ms = rest.pop(0)
        take(q, p, ms, f"quality {q:.2f} with diversity penalty")
    selected = {c["proposal_id"] for c in chosen}
    for q, p, ms in cands:
        if p["id"] not in selected and not any(e["proposal_id"] == p["id"] for e in excluded):
            excluded.append({"proposal_id": p["id"], "title": p["title"],
                             "reason": f"portfolio size limit ({size}); quality {q:.2f}"})
    if cfg.get("include_baselines", True):
        for b in baseline_methods(available, forward_available):
            ms, _, errors = norm_fn(b["method"])
            if errors:
                continue
            h = M2.method_hash(ms) if ms.get("kind") != "random_search" else "rs-baseline"
            if h in hashes:
                for c in chosen:     # a researcher proposed exactly the baseline: mark it, do not duplicate
                    if c["method"] and M2.method_hash(norm_fn(c["method"])[0]) == h:
                        c["also_baseline"] = b["title"]
                continue
            if b["method"]["kind"] == "random_search" and any(c["method"].get("kind") == "random_search" for c in chosen):
                continue
            hashes.add(h)
            chosen.append({"source": "baseline", "proposal_id": None, "root_id": None, "title": b["title"],
                           "author": "scheduler", "persona": "control", "family": b["family"], "method": b["method"],
                           "quality": None, "prediction": None, "selection_reason": b["why"], "unresolved": [],
                           "status": "baseline"})
    for i, c in enumerate(chosen, 1):
        c["entry_id"] = f"E-{i:02d}"
    return chosen, excluded
