"""Collect per-run metrics for the heat ablation and aggregate them by arm.

python ablation/heat/analyze.py [--posthoc]   -> ablation/heat/summary.json

--posthoc  re-fits each run's best *single* full-fidelity model on train+val and scores it on test
           (analysis only; the harness itself never sees these numbers). This removes the ensemble
           confound so arms can be compared on the quality of the single best idea they found.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ["random", "offline", "llm_ideation", "llm_refine", "full"]


def load(run):
    arts = [json.loads(l) for l in (run / "artifacts.jsonl").open(encoding="utf-8")]
    return sorted(arts, key=lambda a: a["created"])


def rel_l2(Y, Yh):
    return float(np.linalg.norm(Yh - Y) / np.linalg.norm(Y))


def posthoc_single(run, idea_id, ideas):
    from mlharness.runner import run_final
    job = None
    for jp in (run / "workspace" / "experiments").glob("*/job.json"):
        j = json.loads(jp.read_text())
        if j.get("idea_id") == idea_id and float(j.get("fraction") or 0) == 1.0 and j["mode"] == "train_eval":
            job = j
            break
    if job is None:
        return None
    out = HERE / "posthoc" / run.name
    out.mkdir(parents=True, exist_ok=True)
    fj = {"mode": "final", "members": [job["idea"]], "weights": [1.0], "task": job["task"],
          "primary": job["primary"], "secondary": job["secondary"], "seed": job["seed"],
          "feature_shape": job.get("feature_shape"), "threads": 6, "final_dir": str(out),
          "run_id": f"posthoc_{run.name}"}
    with np.load(job["data_path"]) as d:
        data = {k: d[k] for k in d.files}
    return run_final(fj, data, out)["test"]


def summarize(run, posthoc=False):
    arts = load(run)
    st = json.loads((run / "status.json").read_text())
    res = st["resources"]
    hyps = {a["data"]["id"]: a["data"] for a in arts if a["kind"] == "hypothesis"}
    results = [a["data"] for a in arts if a["kind"] == "result" and a["data"].get("mode") == "train_eval"]
    full = [r for r in results if float(r.get("fraction") or 0) == 1.0 and r.get("status") == "ok"
            and hyps.get(r["idea_id"], {}).get("persona") != "baseline"]
    ck = next(a["data"] for a in arts if a["kind"] == "checkpoint")
    y_test = np.load(run / "data" / "dataset.npz")["y_test"]
    pred = np.load(run / "final" / "test_pred.npy")
    best = min(full, key=lambda r: r["val"]["rel_l2"])
    pre_ref = [r for r in full if r.get("stage") != "refine"]
    best_pre = min(r["val"]["rel_l2"] for r in pre_ref) if pre_ref else None
    # anytime: best full-fidelity val vs cumulative compute (all experiments, in order)
    cum, curve, b = 0.0, [], np.inf
    for r in results:
        cum += float(r.get("wall_seconds") or 0)
        if r in full:
            b = min(b, r["val"]["rel_l2"])
        curve.append((round(cum, 1), None if not np.isfinite(b) else round(b, 5)))
    origins = Counter(h["origin"] for h in hyps.values() if h["persona"] != "baseline")
    methods_full = Counter(hyps[r["idea_id"]]["config"]["method"] for r in full)
    pruned = [a for a in arts if a["kind"] == "decision" and a["data"].get("decision") == "prune"]
    customs = [h for h in hyps.values() if h["config"]["method"] == "custom"]
    custom_ok = {r["idea_id"] for r in full if hyps[r["idea_id"]]["config"]["method"] == "custom"}
    out = {
        "run": run.name, "stop_reason": None,
        "final_kind": ck["kind"], "final_members": ck["members"],
        "final_methods": [hyps[m]["config"]["method"] + "/" + hyps[m]["config"]["features"] for m in ck["members"]],
        "final_val": ck["val_estimate"], "test_rel_l2": ck["test"]["rel_l2"], "test_r2": ck["test"]["r2"],
        "test_per_segment": [rel_l2(y_test[:, j], pred[:, j]) for j in range(y_test.shape[1])],
        "best_single_val": best["val"]["rel_l2"], "best_single_idea": best["idea_id"],
        "best_single_cfg": {k: v for k, v in hyps[best["idea_id"]]["config"].items() if k != "code"},
        "best_single_origin": hyps[best["idea_id"]]["origin"],
        "best_val_before_refine": best_pre,
        "refine_gain": (best_pre - best["val"]["rel_l2"]) / best_pre if best_pre else None,
        "n_ideas": len(hyps) - 1, "n_full": len(full), "n_experiments": len(results), "n_pruned": len(pruned),
        "origins": dict(origins), "methods_at_full": dict(methods_full),
        "n_custom": len(customs), "n_custom_reaching_full": len(custom_ok),
        "compute_s": res["compute_used_s"], "wall_s": res["wall_elapsed_s"],
        "llm_calls": res["llm_calls"], "llm_cost_usd": res["llm_cost_usd"], "llm_seconds": res["llm_seconds"],
        "funnel": st["funnel"], "anytime": curve,
    }
    for a in arts:
        if a["kind"] == "decision" and a["data"].get("decision") == "stop_refinement":
            out["stop_reason"] = a["data"]["reason"]
    out["llm_failures"] = sum(1 for a in arts if a["kind"] == "failure" and a["data"].get("source") == "llm")
    if posthoc:
        if ck["kind"] == "single" and ck["members"] == [best["idea_id"]]:
            out["best_single_test_rel_l2"] = ck["test"]["rel_l2"]
        else:
            t = posthoc_single(run, best["idea_id"], hyps)
            out["best_single_test_rel_l2"] = t["rel_l2"] if t else None
    return out


def agg(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))} if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posthoc", action="store_true")
    args = ap.parse_args()
    runs = {}
    prev = json.loads((HERE / "summary.json").read_text()) if (HERE / "summary.json").exists() else {}
    for rd in sorted((HERE / "runs").iterdir()):
        stp = rd / "status.json"
        if not stp.exists() or json.loads(stp.read_text()).get("stage") != "done":
            continue
        cached = prev.get("runs", {}).get(rd.name)
        if args.posthoc and cached and cached.get("best_single_test_rel_l2") is not None:
            runs[rd.name] = cached
            continue
        runs[rd.name] = summarize(rd, args.posthoc)
        print(rd.name, runs[rd.name]["test_rel_l2"], runs[rd.name].get("best_single_test_rel_l2"), flush=True)
    by_arm = {}
    for arm in ARMS:
        rs = [r for n, r in runs.items() if n.rsplit("_s", 1)[0] == arm]
        if not rs:
            continue
        keys = ["test_rel_l2", "final_val", "best_single_val", "best_single_test_rel_l2", "best_val_before_refine",
                "refine_gain", "n_ideas", "n_full", "n_experiments", "compute_s", "wall_s", "llm_calls", "llm_cost_usd"]
        by_arm[arm] = {k: agg([r.get(k) for r in rs]) for k in keys}
        by_arm[arm]["test_per_segment"] = np.mean([r["test_per_segment"] for r in rs], axis=0).tolist()
        by_arm[arm]["seeds"] = [r["run"] for r in rs]
    (HERE / "summary.json").write_text(json.dumps({"runs": runs, "by_arm": by_arm}, indent=1, default=str))
    for arm, a in by_arm.items():
        t = a["test_rel_l2"]
        print(f"{arm:13s} test {t['mean']:.4f} ± {t['sd']:.4f} (n={t['n']})")


if __name__ == "__main__":
    main()
