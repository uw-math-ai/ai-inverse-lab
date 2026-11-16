"""Collect V3 ablation runs, aggregate by arm, compare with V1, V2 and the physics reference.

python ablation/heat_v3/analyze_v3.py        -> ablation/heat_v3/summary_v3.json (+ a printed table)

Excluded runs (same rules as V2): not done, SUSPENDED.json (machine slept), LLM_DEGRADED.json
(model calls failed for outage/quota reasons). Per-call timeouts are a harness-handled fallback and
are reported, not excluded.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / "ablation" / "heat" / "summary.json"
V2 = ROOT / "ablation" / "heat_v2" / "summary_v2.json"
PHYS = ROOT / "ablation" / "heat" / "physics_reference.json"
ARMS = ["v3_offline", "v3_full", "v3_opus_top", "v3_no_forward", "v3_no_critic", "v3_no_literature"]
TOKEN_FIELDS = ("calls", "input_tokens", "cache_creation_tokens", "cache_read_tokens", "total_input_tokens",
                "output_tokens", "thinking_tokens", "cost_usd", "seconds")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(run: Path) -> dict:
    st = load(run / "status.json")
    if st.get("stage") != "done":
        return {"run": run.name, "failed": True, "error": st.get("error")}
    if (run / "SUSPENDED.json").exists():
        return {"run": run.name, "failed": True, "suspended": True,
                "error": "machine slept during the run (wall-time compute accounting invalid); excluded"}
    if (run / "LLM_DEGRADED.json").exists():
        return {"run": run.name, "failed": True, "degraded": True,
                "error": f"LLM degraded: {load(run / 'LLM_DEGRADED.json')['n']} failed call(s); excluded"}
    b = load(run / "evidence_bundle.json")
    pe = load(run / "program_evaluation.json")
    tok = load(run / "token_usage.json") if (run / "token_usage.json").exists() else {}
    fin, prog = b["final"], b["program"]
    res = fin["result"]
    studies = b["studies"]
    verdicts = [v for s in studies for v in s.get("verdicts", [])]
    objections = prog.get("objections", [])
    agents = list(b["agents"].values())
    researchers = [a for a in agents if a["role"] == "researcher"]
    arms = [a for s in studies for a in s.get("arms", [])]
    kinds = Counter((a.get("method") or {}).get("kind") for a in arms)
    ok_arms = {k: v for k, v in b["arm_results"].items() if v.get("status") == "ok"}
    best_arm = min(ok_arms.items(), key=lambda kv: kv[1]["val"]["rel_l2"]) if ok_arms else None
    events = b.get("role_events", [])
    devices = Counter(str(v.get("device")) for v in b["arm_results"].values() if v.get("device"))
    lit = [e for e in prog.get("evidence", {}).values() if isinstance(e, dict) and e.get("source_type") == "theorem"]
    return {
        "run": run.name,
        # model product
        # an uncertified export receives no final metrics by V3 design: keep the run, record None
        "test_rel_l2": (fin.get("test") or {}).get("rel_l2"), "test_r2": (fin.get("test") or {}).get("r2"),
        "final_evaluation_blocked": bool(res.get("final_evaluation_blocked")),
        "certificate_reasons": (fin.get("certificate") or {}).get("reasons"),
        "certificate_max_abs": (fin.get("certificate") or {}).get("max_abs_differences"),
        "test_ci95": (res.get("test_bootstrap") or {}).get("ci95"),
        "test_per_segment": res.get("test_per_target_rel_l2"), "shift": res.get("shift"),
        "final_val": fin["val"], "final_kind": fin["kind"], "final_labels": fin.get("labels"),
        "final_families": [m.get("family") for m in res.get("member_info", [])],
        "final_member_kinds": [m.get("kind") for m in res.get("member_info", [])],
        "final_uses_forward": any(m.get("uses_forward") for m in res.get("member_info", [])),
        "certified": bool((fin.get("certificate") or {}).get("certified")),
        "reload_max_abs_diff": res.get("reload_max_abs_diff"), "latency_ms": res.get("latency_ms_per_sample"),
        "test_uncertainty": res.get("test_uncertainty"),
        "best_arm_val": best_arm[1]["val"]["rel_l2"] if best_arm else None,
        "best_arm_label": best_arm[1].get("label") if best_arm else None,
        # research program
        "stop_reason": prog.get("stop_reason"),
        "actions": dict(Counter(a["type"] for a in prog["actions"])),
        "actions_total": len(prog["actions"]),
        "actions_rejected": sum(a.get("status") == "rejected" for a in prog["actions"]),
        "rejection_errors": [a.get("error") for a in prog["actions"] if a.get("status") == "rejected"],
        "questions": len(prog.get("questions", [])),
        "studies": len(studies), "studies_complete": sum(s.get("status") == "complete" for s in studies),
        "study_status": dict(Counter(s.get("status") for s in studies)),
        "study_completion_rate": pe.get("study_completion_rate"),
        "claims": len(verdicts), "claim_outcomes": dict(Counter(v["outcome"] for v in verdicts)),
        "arms_registered": len(arms), "arms_ok": len(ok_arms), "arm_method_kinds": dict(kinds),
        "custom_code_arms": kinds.get("custom_ml", 0) + kinds.get("custom_solver", 0),
        "researchers": len(researchers), "researcher_personas": dict(Counter(a.get("persona") for a in researchers)),
        "specialist_tasks": len(b.get("specialist_tasks", [])),
        "specialist_status": dict(Counter(t.get("status") for t in b.get("specialist_tasks", []))),
        "objections": len(objections), "objections_blocking": sum(bool(o.get("blocking")) for o in objections),
        "objections_resolved": sum(o.get("status") in ("resolved", "acknowledged") for o in objections),
        "theorem_requests": len(lit), "scientific_completion": fin["audit"].get("scientific_completion"),
        "role_fallbacks": dict(Counter(e.get("fallback_reason") for e in events if e.get("fallback_reason"))),
        "llm_timeouts": sum(1 for e in events if "timed out" in str(e.get("error", ""))),
        # cost / tokens / execution route
        "tokens": tok.get("total"), "tokens_by_tier": tok.get("by_tier"), "tokens_by_model": tok.get("by_model"),
        "llm_calls": b["costs"]["llm_calls"], "llm_cost_usd": (tok.get("total") or {}).get("cost_usd", 0.0),
        "compute_s": b["costs"]["compute_used_s"], "wall_s": b["costs"]["wall_elapsed_s"],
        "jobs": b["costs"]["jobs"], "stage_s": b["costs"].get("by_stage_s"), "devices": dict(devices),
    }


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": len(a)}


def welch(a, b):
    try:
        from scipy.stats import ttest_ind
        a, b = [x for x in a if x is not None], [x for x in b if x is not None]
        if len(a) < 2 or len(b) < 2:
            return None
        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def arm_of(name: str) -> str:
    return name.rsplit("_s", 1)[0]


def main():
    runs = {}
    for d in sorted((HERE / "runs").glob("*_s*")):
        if (d / "status.json").exists():
            try:
                runs[d.name] = summarize(d)
            except Exception as exc:
                runs[d.name] = {"run": d.name, "failed": True, "error": f"analysis: {type(exc).__name__}: {exc}"}
    tests = {arm: [r["test_rel_l2"] for k, r in runs.items() if arm_of(k) == arm and not r.get("failed")
                   and r.get("test_rel_l2") is not None] for arm in ARMS}
    by_arm = {}
    for arm in ARMS:
        rs = [r for k, r in runs.items() if arm_of(k) == arm and not r.get("failed")]
        if not rs:
            continue
        keys = ["test_rel_l2", "final_val", "best_arm_val", "llm_calls", "llm_cost_usd", "compute_s", "wall_s",
                "actions_total", "actions_rejected", "studies", "studies_complete", "study_completion_rate", "claims",
                "arms_registered", "custom_code_arms", "researchers", "specialist_tasks", "objections",
                "objections_blocking", "objections_resolved", "theorem_requests", "llm_timeouts"]
        A = {k: agg([r.get(k) for r in rs]) for k in keys}
        A["tokens"] = {f: agg([(r.get("tokens") or {}).get(f) for r in rs]) for f in TOKEN_FIELDS}
        tiers = sorted({t for r in rs for t in (r.get("tokens_by_tier") or {})})
        A["tokens_by_tier"] = {t: {f: agg([(r["tokens_by_tier"] or {}).get(t, {}).get(f) for r in rs])
                                   for f in ("calls", "total_input_tokens", "output_tokens", "cost_usd")} for t in tiers}
        segs = [r["test_per_segment"] for r in rs if r.get("test_per_segment")]
        A["test_per_segment"] = np.mean(segs, axis=0).tolist() if segs else None
        A["uncertified_runs"] = [r["run"] for r in rs if not r["certified"]]
        A["final_uses_forward"] = sum(bool(r["final_uses_forward"]) for r in rs)
        A["certified"] = sum(bool(r["certified"]) for r in rs)
        A["final_labels"] = [r["final_labels"] for r in rs]
        A["claim_outcomes"] = dict(sum((Counter(r["claim_outcomes"]) for r in rs), Counter()))
        A["actions"] = dict(sum((Counter(r["actions"]) for r in rs), Counter()))
        A["arm_method_kinds"] = dict(sum((Counter(r["arm_method_kinds"]) for r in rs), Counter()))
        A["shift"] = {nm: agg([r["shift"][nm]["rel_l2"] for r in rs if r.get("shift") and nm in r["shift"]])
                      for nm in (rs[0].get("shift") or {})}
        A["seeds"] = [r["run"] for r in rs]
        by_arm[arm] = A
    v1 = load(V1) if V1.exists() else {}
    v2 = load(V2) if V2.exists() else {}
    v1_tests = {a: [v1["runs"][s]["test_rel_l2"] for s in v1["by_arm"][a]["seeds"]] for a in v1.get("by_arm", {})}
    v2_tests = {a: [r["test_rel_l2"] for k, r in v2.get("runs", {}).items() if arm_of(k) == a and not r.get("failed")]
                for a in v2.get("by_arm", {})}
    comps = {}
    for arm in by_arm:
        mine = tests[arm]
        c = {}
        if not mine:
            comps[arm] = {"note": "no certified test metric"}
            continue
        for label, ref in (("v1_full", v1_tests.get("full")), ("v2_full", v2_tests.get("v2_full")),
                           ("v2_" + arm[3:], v2_tests.get("v2_" + arm[3:]))):
            if ref:
                c[f"vs_{label}"] = {"p_welch": welch(mine, ref),
                                    "rel_change": float((np.mean(mine) - np.mean(ref)) / np.mean(ref))}
        for other in by_arm:
            if other != arm and tests[other]:
                c[f"vs_{other}"] = {"p_welch": welch(mine, tests[other]),
                                    "rel_change": float((np.mean(mine) - np.mean(tests[other])) / np.mean(tests[other]))}
        comps[arm] = c
    out = {"runs": runs, "by_arm": by_arm, "comparisons": comps,
           "physics_reference": load(PHYS) if PHYS.exists() else None,
           "v1_full": v1.get("by_arm", {}).get("full", {}).get("test_rel_l2"),
           "v2_by_arm_test": {a: v2["by_arm"][a].get("test_rel_l2") for a in v2.get("by_arm", {})}}
    (HERE / "summary_v3.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(f"{'arm':18s} {'n':>2s} {'test rel_l2':>18s} {'studies ok':>10s} {'obj':>5s} {'calls':>6s} "
          f"{'in tok':>10s} {'out tok':>9s} {'$/run':>7s} {'wall min':>8s}")
    for arm, A in by_arm.items():
        t, tk = A["test_rel_l2"], A["tokens"]
        ts = f"{t['mean']:.5f} ± {t['sd']:.5f}" if t else "   uncertified   "
        print(f"{arm:18s} {len(A['seeds']):2d} {ts} "
              f"{A['studies_complete']['mean']:4.1f}/{A['studies']['mean']:<4.1f} {A['objections']['mean']:5.0f} "
              f"{A['llm_calls']['mean']:6.1f} {tk['total_input_tokens']['mean']:10,.0f} "
              f"{tk['output_tokens']['mean']:9,.0f} {A['llm_cost_usd']['mean']:7.2f} {A['wall_s']['mean'] / 60:8.1f}")
    bad = {k: r["error"] for k, r in runs.items() if r.get("failed")}
    if bad:
        print("\nexcluded:", json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
