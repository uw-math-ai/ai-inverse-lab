"""Collect V2 ablation runs, aggregate by arm and compare with the V1 ablation and the physics reference.

python ablation/heat_v2/analyze_v2.py        -> ablation/heat_v2/summary_v2.json (+ a printed table)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "ablation" / "heat" / "summary.json"
PHYS = ROOT / "ablation" / "heat" / "physics_reference.json"
ARMS = ["v2_offline", "v2_full", "v2_no_forward", "v2_no_critic", "v2_no_literature", "v2_memory"]
FAMILIES = ("physics", "hybrid", "data-driven", "physics-custom", "custom")


def arts(run):
    with open(run / "artifacts.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(run: Path) -> dict | None:
    st = json.loads((run / "status.json").read_text())
    if st.get("stage") != "done":
        return {"run": run.name, "failed": True, "error": st.get("error")}
    if (run / "SUSPENDED.json").exists():
        return {"run": run.name, "failed": True, "suspended": True,
                "error": "machine slept during the run (wall-time compute accounting invalid); excluded"}
    A = arts(run)
    llm_fail = [a for a in A if a["kind"] == "failure" and (a.get("data") or {}).get("source") == "llm"]
    if (run / "LLM_DEGRADED.json").exists() or (llm_fail and not run.name.startswith("v2_offline")):
        return {"run": run.name, "failed": True, "degraded": True, "llm_failures": len(llm_fail),
                "error": f"LLM degraded: {len(llm_fail)} failed LLM call(s) -> heuristic fallbacks; excluded"}
    res = st["resources"]
    man = json.loads((run / "final" / "manifest.json").read_text())
    trace = json.loads((run / "research_trace.json").read_text())
    hyps = {a["data"]["id"]: a["data"] for a in A if a["kind"] == "hypothesis"}
    full = [a["data"] for a in A if a["kind"] == "result" and a["data"].get("mode") == "method_eval"
            and a["data"].get("stage") in ("evaluate", "refine") and a["data"].get("status") == "ok"]
    best_by_family = {}
    for r in full:
        fam = r.get("family") or "?"
        v = r["val"]["rel_l2"]
        if fam not in best_by_family or v < best_by_family[fam]["val"]:
            best_by_family[fam] = {"val": v, "label": r.get("label")}
    best_single = min(full, key=lambda r: r["val"]["rel_l2"]) if full else None
    research_full = [r for r in full if hyps.get(r["idea_id"], {}).get("origin") == "research"]
    baseline_full = [r for r in full if hyps.get(r["idea_id"], {}).get("origin") == "baseline"]
    ck = next(a["data"] for a in A if a["kind"] == "checkpoint")
    res_final = next(a["data"] for a in A if a["kind"] == "result" and a["data"].get("mode") == "final_v2")
    props = trace["proposals"]
    acts = Counter(a["action"] for rd in trace["rounds"] for a in rd["actions"])
    personas_responding = [sorted({a.get("persona") or a.get("researcher") for a in rd["actions"]}) for rd in trace["rounds"]]
    outcomes = Counter(o["outcome"] for o in trace["prediction_outcomes"].values() if o.get("source") == "proposal")
    fams_proposed = Counter(p["family"] for p in props.values() if p["action"] in ("propose", "branch"))
    lit = trace.get("literature") or []
    cited = {c["ref_id"] for p in props.values() for c in p.get("citations") or []}
    mem_refs = {m for p in props.values() for m in p.get("memory_refs") or []}
    custom = [p for p in props.values() if (p.get("method") or {}).get("kind") in ("custom_ml", "custom_solver")]
    return {
        "run": run.name, "test_rel_l2": man["test"]["rel_l2"], "test_r2": man["test"].get("r2"),
        "test_ci95": man["test_bootstrap"]["ci95"], "test_per_segment": res_final.get("test_per_target_rel_l2"),
        "shift": man.get("shift"), "final_val": ck["val_estimate"], "final_kind": ck["kind"],
        "final_labels": ck.get("labels"), "final_families": [m["family"] for m in man["member_info"]],
        "final_uses_forward": any(m["uses_forward"] for m in man["member_info"]),
        "test_uncertainty": res_final.get("test_uncertainty"), "latency_ms": res_final.get("latency_ms_per_sample"),
        "reload_max_abs_diff": res_final.get("reload_max_abs_diff"),
        "best_single_val": best_single["val"]["rel_l2"] if best_single else None,
        "best_single_label": best_single.get("label") if best_single else None,
        "best_research_val": min((r["val"]["rel_l2"] for r in research_full), default=None),
        "best_baseline_val": min((r["val"]["rel_l2"] for r in baseline_full), default=None),
        "best_by_family": best_by_family, "n_full": len(full),
        "research_rounds": len(trace["rounds"]), "proposal_versions": len(props),
        "proposals_initial": sum(1 for p in props.values() if p["action"] == "propose" and p["round"] == 1),
        "actions": dict(acts), "personas_responding_by_round": personas_responding,
        "abandoned": len(trace["abandoned"]), "stop_reason": trace["stop_reason"],
        "families_proposed": dict(fams_proposed), "custom_code_proposals": len(custom),
        "predictions": dict(outcomes), "pilots": len(trace.get("pilots") or {}),
        "lit_queries": len(lit), "lit_failures": sum(1 for q in lit if not q["ok"] and q["error_kind"] != "budget"),
        "lit_refs_cited": len(cited), "memory_refs_used": len(mem_refs),
        "portfolio": [(e["entry_id"], e["source"], e["family"]) for e in trace["portfolio"]],
        "llm_calls": res["llm_calls"], "llm_by_phase": res.get("llm_by_phase"), "llm_cost_usd": res["llm_cost_usd"],
        "compute_s": res["compute_used_s"], "wall_s": res["wall_elapsed_s"], "jobs": res["jobs"],
        "stage_s": res["by_stage_s"],
    }


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "min": float(a.min()),
            "max": float(a.max()), "n": len(a)}


def welch(a, b):
    try:
        from scipy.stats import ttest_ind
        a, b = [x for x in a if x is not None], [x for x in b if x is not None]
        if len(a) < 2 or len(b) < 2:
            return None
        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def main():
    runs = {}
    for d in sorted((HERE / "runs").glob("*_s*")):
        if (d / "status.json").exists():
            try:
                runs[d.name] = summarize(d)
            except Exception as exc:          # keep going; report what broke
                runs[d.name] = {"run": d.name, "failed": True, "error": f"analysis: {type(exc).__name__}: {exc}"}
    by_arm = {}
    for arm in ARMS:
        rs = [r for k, r in runs.items() if k.rsplit("_s", 1)[0] == arm and not r.get("failed")]
        if not rs:
            continue
        keys = ["test_rel_l2", "final_val", "best_single_val", "best_research_val", "best_baseline_val", "llm_calls",
                "llm_cost_usd", "compute_s", "wall_s", "research_rounds", "proposal_versions", "abandoned", "lit_queries",
                "lit_failures", "lit_refs_cited", "memory_refs_used", "pilots", "custom_code_proposals", "n_full"]
        by_arm[arm] = {k: agg([r.get(k) for r in rs]) for k in keys}
        by_arm[arm]["test_per_segment"] = np.mean([r["test_per_segment"] for r in rs], axis=0).tolist()
        by_arm[arm]["final_uses_forward"] = sum(bool(r["final_uses_forward"]) for r in rs)
        by_arm[arm]["final_labels"] = [r["final_labels"] for r in rs]
        by_arm[arm]["predictions"] = dict(sum((Counter(r["predictions"]) for r in rs), Counter()))
        by_arm[arm]["actions"] = dict(sum((Counter(r["actions"]) for r in rs), Counter()))
        by_arm[arm]["shift"] = {nm: agg([r["shift"][nm]["rel_l2"] for r in rs if r.get("shift") and nm in r["shift"]])
                                for nm in (rs[0].get("shift") or {})}
        by_arm[arm]["best_by_family"] = {f: agg([r["best_by_family"].get(f, {}).get("val") for r in rs]) for f in FAMILIES}
        by_arm[arm]["seeds"] = [r["run"] for r in rs]
    v1 = json.loads(V1.read_text())["by_arm"] if V1.exists() else {}
    phys = json.loads(PHYS.read_text()) if PHYS.exists() else {}
    v1_tests = {a: [json.loads(V1.read_text())["runs"][s]["test_rel_l2"] for s in v1[a]["seeds"]] for a in v1}
    comps = {}
    for arm in by_arm:
        mine = [r["test_rel_l2"] for k, r in runs.items() if k.rsplit("_s", 1)[0] == arm and not r.get("failed")]
        comps[arm] = {f"vs_v1_{a}": {"p_welch": welch(mine, v1_tests[a]),
                                     "rel_change": (np.mean(mine) - np.mean(v1_tests[a])) / np.mean(v1_tests[a])}
                      for a in v1_tests}
        for other in by_arm:
            if other != arm:
                theirs = [r["test_rel_l2"] for k, r in runs.items() if k.rsplit("_s", 1)[0] == other and not r.get("failed")]
                comps[arm][f"vs_{other}"] = {"p_welch": welch(mine, theirs),
                                             "rel_change": (np.mean(mine) - np.mean(theirs)) / np.mean(theirs)}
    out = {"runs": runs, "by_arm": by_arm, "comparisons": comps,
           "v1_by_arm": {a: {"test_rel_l2": v1[a]["test_rel_l2"], "best_single_test_rel_l2": v1[a].get("best_single_test_rel_l2"),
                             "llm_cost_usd": v1[a].get("llm_cost_usd"), "test_per_segment": v1[a].get("test_per_segment")}
                         for a in v1},
           "physics_reference": phys}
    (HERE / "summary_v2.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"{'arm':18s} {'n':>2s} {'test rel_l2':>18s} {'best val':>9s} {'fwd-final':>9s} {'rounds':>6s} {'versions':>8s} "
          f"{'aband':>5s} {'lit q':>5s} {'LLM':>5s} {'$':>6s} {'compute':>8s}")
    for arm, a in by_arm.items():
        t = a["test_rel_l2"]
        print(f"{arm:18s} {t['n']:2d} {t['mean']:.5f}±{t['sd']:.5f} {a['best_single_val']['mean']:9.5f} "
              f"{a['final_uses_forward']:>9d} {a['research_rounds']['mean']:6.1f} {a['proposal_versions']['mean']:8.1f} "
              f"{a['abandoned']['mean']:5.1f} {a['lit_queries']['mean']:5.1f} {a['llm_calls']['mean']:5.1f} "
              f"{a['llm_cost_usd']['mean']:6.2f} {a['compute_s']['mean']:8.1f}")
    for a, d in out["v1_by_arm"].items():
        print(f"V1 {a:15s}  {d['test_rel_l2']['mean']:.5f}±{d['test_rel_l2']['sd']:.5f}")
    if phys:
        print(f"physics reference MLE {phys['mle_test_rel_l2']:.5f}  CRB floor {phys['crb_rel_l2']:.5f}")
    for r in runs.values():
        if r.get("failed"):
            print("FAILED:", r["run"], r.get("error"))


if __name__ == "__main__":
    sys.exit(main())
