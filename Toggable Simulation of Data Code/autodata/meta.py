"""Outer loop: meta-optimize the challenger's strategy (Autodata Sec. 4, "harness optimization").

What is optimized is the *strategy* dict (challenger.DEFAULT_STRATEGY): the heuristic challenger's
knobs and the LLM challenger's guidance text. Solvers, gates, limits and seeds stay outside it, so a
candidate cannot "win" by weakening the weak solver (the cheating the paper reports).

Each iteration, as in the paper:
  1. pick a parent from the population by Boltzmann sampling, P(c) ~ exp(score_c / T), T = 0.1
  2. run the parent on a minibatch of TRAIN scenarios and collect its trajectories
  3. analyze them (rule-based statistics; optionally an LLM root-cause analysis)
  4. implement a mutation of the strategy (rule-based; optionally an LLM rewrites it)
  5. score parent and child on the VALIDATION scenarios
  6. accept the child into the population only if its score strictly exceeds the parent's
  7. append a summary to the history that later analyzers read

score = validation pass rate (fraction of scenarios that yield an accepted item within
meta.eval_rounds rounds) + 0.01 * speed, where speed rewards accepting in fewer rounds; the small
speed term only breaks ties between equal pass rates.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .challenger import DEFAULT_STRATEGY, merge_strategy
from .llm import LLMError, extract_json
from .loop import _atomic_write, _jsonable, make_challenger, run_scenario, setup_logging

log = logging.getLogger("autodata")

PARAM_RANGES = {"spread_step": (0.05, 5.0), "spread_cap": (0.5, 20.0), "size_factor": (1.05, 3.0),
                "depth_k_factor": (0.2, 0.95), "noise_factor": (0.05, 0.95), "hard_N_factor": (1.2, 20.0),
                "hard_T_factor": (1.05, 4.0), "hard_k_step": (1, 5), "explore_prob": (0.0, 0.8)}
KNOBS = ("spread", "depth", "size", "noise")


def clip_strategy(strategy: dict) -> dict:
    s = merge_strategy(strategy)
    h = s["heuristic"]
    for key, (lo, hi) in PARAM_RANGES.items():
        val = float(np.clip(float(h[key]), lo, hi))
        h[key] = int(round(val)) if key == "hard_k_step" else val
    knobs = [k for k in h.get("easy_knobs", []) if k in KNOBS]
    h["easy_knobs"] = list(dict.fromkeys(knobs)) or list(KNOBS)
    return s


def evaluate_harness(cfg: dict, strategy: dict, scenarios: list[dict], rounds: int, llm=None) -> dict:
    challenger = make_challenger(cfg, strategy, cfg["seed"], llm)
    one = {**cfg, "loop": {**cfg["loop"], "items_per_scenario": 1}}
    results = [run_scenario(s, one, challenger, set(), max_rounds=rounds) for s in scenarios]
    passed = [r["accepted"] for r in results]
    speed = [(rounds - r["rounds"] + 1) / rounds if r["accepted"] else 0.0 for r in results]
    pass_rate = float(np.mean(passed)) if results else 0.0
    return {"score": pass_rate + 0.01 * float(np.mean(speed) if results else 0.0),
            "pass_rate": pass_rate, "results": results}


def analyze(results: list[dict], llm=None) -> dict:
    status, codes = {}, {}
    for res in results:
        for rec in res["history"]:
            status[rec["status"]] = status.get(rec["status"], 0) + 1
            if rec["status"] in ("qv_failed", "invalid"):
                codes[rec["code"]] = codes.get(rec["code"], 0) + 1
    fails = {k: v for k, v in status.items() if k != "accepted"}
    buckets = {"too_easy": fails.get("too_easy", 0),
               "too_hard": fails.get("failed_strong", 0) + codes.get("noise_dominated", 0),
               "ill_posed": codes.get("unidentifiable", 0) + codes.get("low_hit_mass", 0)
               + codes.get("no_unknowns", 0),
               "invalid": fails.get("invalid", 0) + codes.get("duplicate", 0)}
    dominant = max(buckets, key=buckets.get) if any(buckets.values()) else "none"
    out = {"status_counts": status, "qv_codes": codes, "buckets": buckets, "dominant": dominant,
           "text": f"dominant failure: {dominant}; buckets {buckets}"}
    if llm is not None and not llm.is_null:
        traj = [{"scenario": r["scenario"], "accepted": r["accepted"],
                 "rounds": [{k: rec.get(k) for k in ("round", "status", "reason", "challenger_note")}
                            | {"weak": (rec.get("weak") or {}).get("avg"),
                               "strong": (rec.get("strong") or {}).get("avg"),
                               "instance": rec.get("instance")} for rec in r["history"]]}
                for r in results]
        try:
            out["text"] = llm.complete(
                "You analyze trajectories of a data-generation harness for an inverse problem. "
                "Find the systematic reasons rounds fail (too easy for the weak solver, too hard for the "
                "strong solver, ill-posed, invalid). Be concrete and brief (<= 12 bullet points).",
                json.dumps(traj, default=_jsonable)[:150_000])
        except LLMError as exc:
            log.warning("LLM analyzer failed (%s); using rule-based analysis", exc)
    return out


def mutate(strategy: dict, analysis: dict, rng: np.random.Generator) -> tuple[dict, list[str]]:
    """Rule-based implementer: push the knobs that address the dominant failure, plus one random jitter."""
    s = copy.deepcopy(merge_strategy(strategy))
    h, notes = s["heuristic"], []

    def scale(key, f):
        h[key] = h[key] * f
        notes.append(f"{key} x{f:.2f}")

    dom = analysis["dominant"]
    if dom == "too_easy":
        scale("spread_step", float(rng.uniform(1.1, 1.6)))
        scale("noise_factor", float(rng.uniform(0.6, 0.9)))
        knobs = list(h["easy_knobs"])
        front = knobs.pop(int(rng.integers(len(knobs))))
        h["easy_knobs"] = [front] + knobs
        notes.append(f"easy_knobs -> {h['easy_knobs']}")
    elif dom == "too_hard":
        scale("hard_N_factor", float(rng.uniform(1.2, 1.8)))
        scale("noise_factor", float(rng.uniform(1.05, 1.3)))
        scale("spread_cap", float(rng.uniform(0.7, 0.95)))
    elif dom == "ill_posed":
        scale("hard_T_factor", float(rng.uniform(1.1, 1.4)))
        h["hard_k_step"] += 1
        notes.append("hard_k_step +1")
    key = list(PARAM_RANGES)[int(rng.integers(len(PARAM_RANGES)))]
    if key == "hard_k_step":
        h[key] += int(rng.choice([-1, 1]))
        notes.append(f"jitter {key} -> {h[key]}")
    else:
        scale(key, float(np.exp(rng.normal(0, 0.25))))
    return clip_strategy(s), notes


def implement(strategy: dict, analysis: dict, history: list[dict], rng, llm=None) -> tuple[dict, list[str]]:
    if llm is not None and not llm.is_null:
        prompt = json.dumps({"current_strategy": strategy, "analysis": analysis["text"],
                             "buckets": analysis["buckets"], "param_ranges": PARAM_RANGES,
                             "easy_knobs_allowed": KNOBS, "history": history[-10:]}, default=_jsonable)
        try:
            text = llm.complete(
                "You improve the strategy of a challenger that proposes fptsim simulation specs for an "
                "inverse problem (recover vertex centrality mu from first-passage times). Return ONE JSON "
                "object {\"heuristic\": {...same keys, values inside param_ranges...}, \"guidance\": "
                "\"<improved instructions for an LLM challenger>\", \"why\": \"<one sentence>\"}. Make a "
                "targeted change that addresses the analysis; do not touch solvers or thresholds.", prompt)
            obj = extract_json(text)
            if isinstance(obj, dict) and isinstance(obj.get("heuristic"), dict):
                return clip_strategy(obj), [f"llm: {obj.get('why', '')}"]
            log.warning("LLM implementer returned no usable JSON; using rule-based mutation")
        except LLMError as exc:
            log.warning("LLM implementer failed (%s); using rule-based mutation", exc)
    return mutate(strategy, analysis, rng)


def optimize(cfg: dict, llm=None) -> dict:
    out = Path(cfg["output_dir"]) / "meta"
    setup_logging(out)
    m = cfg["meta"]
    rng = np.random.default_rng(cfg["seed"])
    scen = list(cfg["scenarios"])
    n_train = int(round(m["train_fraction"] * len(scen)))
    n_train = min(max(n_train, 1), len(scen) - 1) if len(scen) > 1 else len(scen)
    train, val = scen[:n_train], (scen[n_train:] or scen)
    need_llm = cfg["toggles"]["llm_challenger"] or m["llm_implementer"]
    if llm is None and need_llm:
        from .llm import make_llm
        llm = make_llm(cfg["llm"])
    meta_llm = llm if m["llm_implementer"] else None
    log.info("meta-optimization: %d iterations, train %s, validation %s",
             m["iterations"], [s["id"] for s in train], [s["id"] for s in val])
    base = clip_strategy(cfg["strategy"] or DEFAULT_STRATEGY)
    ev = evaluate_harness(cfg, base, val, m["eval_rounds"], llm)
    population = [{"id": 0, "parent": None, "strategy": base, "score": ev["score"],
                   "pass_rate": ev["pass_rate"]}]
    log.info("baseline validation score %.4f (pass rate %.3f)", ev["score"], ev["pass_rate"])
    history = []
    deterministic = not cfg["toggles"]["llm_challenger"]
    for it in range(1, m["iterations"] + 1):
        scores = np.array([c["score"] for c in population])
        w = np.exp((scores - scores.max()) / m["temperature"])
        parent = population[int(rng.choice(len(population), p=w / w.sum()))]
        mb_idx = rng.choice(len(train), size=min(m["minibatch"], len(train)), replace=False)
        mb = [train[i] for i in sorted(mb_idx)]
        tr = evaluate_harness(cfg, parent["strategy"], mb, m["eval_rounds"], llm)
        analysis = analyze(tr["results"], meta_llm)
        child_strategy, notes = implement(parent["strategy"], analysis, history, rng, meta_llm)
        parent_val = parent["score"] if deterministic else evaluate_harness(
            cfg, parent["strategy"], val, m["eval_rounds"], llm)["score"]
        cv = evaluate_harness(cfg, child_strategy, val, m["eval_rounds"], llm)
        accepted = cv["score"] > parent_val
        entry = {"iteration": it, "parent": parent["id"], "train_pass_rate": tr["pass_rate"],
                 "dominant_failure": analysis["dominant"], "buckets": analysis["buckets"],
                 "changes": notes, "parent_score": parent_val, "child_score": cv["score"],
                 "child_pass_rate": cv["pass_rate"], "accepted": accepted}
        if accepted:
            entry["child"] = len(population)
            population.append({"id": len(population), "parent": parent["id"], "strategy": child_strategy,
                               "score": cv["score"], "pass_rate": cv["pass_rate"]})
        history.append(entry)
        log.info("meta %d: parent %d (%.4f) -> child %.4f %s  [%s; %s]", it, parent["id"], parent_val,
                 cv["score"], "ACCEPTED" if accepted else "rejected", analysis["dominant"], ", ".join(notes))
        _atomic_write(out / "history.jsonl", "".join(json.dumps(h, default=_jsonable) + "\n" for h in history))
        _atomic_write(out / "population.json", json.dumps(population, indent=1, default=_jsonable))
    best = max(population, key=lambda c: c["score"])
    _atomic_write(out / "best_strategy.yaml", yaml.safe_dump(best["strategy"], sort_keys=False))
    summary = {"iterations": m["iterations"], "accepted": sum(h["accepted"] for h in history),
               "baseline_score": population[0]["score"], "best_score": best["score"], "best_id": best["id"],
               "best_pass_rate": best["pass_rate"]}
    _atomic_write(out / "summary.json", json.dumps(summary, indent=1))
    log.info("meta done: %d/%d mutations accepted, validation score %.4f -> %.4f; best strategy in %s",
             summary["accepted"], m["iterations"], summary["baseline_score"], best["score"],
             out / "best_strategy.yaml")
    return summary
