"""Outer loop (stdlib port of "Toggable Simulation of Data Code/autodata/meta.py").

Boltzmann parent selection (T = meta.temperature), run on a train minibatch, analyze, mutate
(rule-based or LLM), score parent and child on validation, accept only on a strict improvement.
score = validation pass rate + 0.01 * speed (speed only breaks ties).
"""
from __future__ import annotations

import copy
import json
import math
import random

from .challenger import DEFAULT_STRATEGY, merge_strategy
from .llm import LLMError, extract_json
from .loop import make_challenger, run_scenario

PARAM_RANGES = {"spread_step": (0.05, 5.0), "spread_cap": (0.5, 20.0), "size_factor": (1.05, 3.0),
                "depth_k_factor": (0.2, 0.95), "noise_factor": (0.05, 0.95), "hard_N_factor": (1.2, 20.0),
                "hard_T_factor": (1.05, 4.0), "hard_k_step": (1, 5), "explore_prob": (0.0, 0.8)}
KNOBS = ("spread", "depth", "size", "noise")


def clip_strategy(strategy):
    s = merge_strategy(strategy)
    h = s["heuristic"]
    for key, (lo, hi) in PARAM_RANGES.items():
        val = min(max(float(h[key]), lo), hi)
        h[key] = int(round(val)) if key == "hard_k_step" else val
    knobs = [k for k in h.get("easy_knobs", []) if k in KNOBS]
    h["easy_knobs"] = list(dict.fromkeys(knobs)) or list(KNOBS)
    return s


def evaluate_harness(cfg, strategy, scenarios, rounds, llm=None, should_stop=None, log=None):
    challenger = make_challenger(cfg, strategy, llm)
    one = {**cfg, "loop": {**cfg["loop"], "items_per_scenario": 1}}
    quiet = log or (lambda *_: None)
    results = [run_scenario(s, one, challenger, set(), max_rounds=rounds, should_stop=should_stop, log=quiet)
               for s in scenarios]
    passed = [r["accepted"] for r in results]
    speed = [(rounds - r["rounds"] + 1) / rounds if r["accepted"] else 0.0 for r in results]
    pass_rate = sum(passed) / len(passed) if passed else 0.0
    return {"score": pass_rate + 0.01 * (sum(speed) / len(speed) if speed else 0.0), "pass_rate": pass_rate,
            "results": results, "complete": all(r["complete"] for r in results)}


def analyze(results, llm=None):
    status, codes = {}, {}
    for res in results:
        for rec in res["history"]:
            status[rec["status"]] = status.get(rec["status"], 0) + 1
            if rec["status"] in ("qv_failed", "invalid"):
                codes[rec["code"]] = codes.get(rec["code"], 0) + 1
    buckets = {"too_easy": status.get("too_easy", 0),
               "too_hard": status.get("failed_strong", 0) + codes.get("noise_dominated", 0),
               "ill_posed": codes.get("unidentifiable", 0) + codes.get("low_hit_mass", 0) + codes.get("no_unknowns", 0),
               "invalid": status.get("invalid", 0) + codes.get("duplicate", 0)}
    dominant = max(buckets, key=buckets.get) if any(buckets.values()) else "none"
    out = {"status_counts": status, "qv_codes": codes, "buckets": buckets, "dominant": dominant,
           "text": f"dominant failure: {dominant}; buckets {buckets}"}
    if llm is not None and not llm.is_null:
        traj = [{"scenario": r["scenario"], "accepted": r["accepted"],
                 "rounds": [{"round": rec["round"], "status": rec["status"], "reason": rec.get("reason"),
                             "note": rec.get("challenger_note"),
                             "weak": (rec.get("weak") or {}).get("avg"), "strong": (rec.get("strong") or {}).get("avg"),
                             "instance": rec.get("instance")} for rec in r["history"]]} for r in results]
        try:
            out["text"] = llm.complete(
                "You analyze trajectories of a data-generation harness for an inverse problem. Find the "
                "systematic reasons rounds fail (too easy for the weak solver, too hard for the strong solver, "
                "ill-posed, invalid). Be concrete and brief (<= 12 bullet points).",
                json.dumps(traj, default=str)[:100_000])
        except LLMError:
            pass
    return out


def mutate(strategy, analysis, rng: random.Random):
    s = copy.deepcopy(merge_strategy(strategy))
    h, notes = s["heuristic"], []

    def scale(key, f):
        h[key] = h[key] * f
        notes.append(f"{key} x{f:.2f}")

    dom = analysis["dominant"]
    if dom == "too_easy":
        scale("spread_step", rng.uniform(1.1, 1.6))
        scale("noise_factor", rng.uniform(0.6, 0.9))
        knobs = list(h["easy_knobs"])
        front = knobs.pop(rng.randrange(len(knobs)))
        h["easy_knobs"] = [front] + knobs
        notes.append(f"easy_knobs -> {h['easy_knobs']}")
    elif dom == "too_hard":
        scale("hard_N_factor", rng.uniform(1.2, 1.8))
        scale("noise_factor", rng.uniform(1.05, 1.3))
        scale("spread_cap", rng.uniform(0.7, 0.95))
    elif dom == "ill_posed":
        scale("hard_T_factor", rng.uniform(1.1, 1.4))
        h["hard_k_step"] += 1
        notes.append("hard_k_step +1")
    key = rng.choice(list(PARAM_RANGES))
    if key == "hard_k_step":
        h[key] += rng.choice([-1, 1])
        notes.append(f"jitter {key} -> {h[key]}")
    else:
        scale(key, math.exp(rng.gauss(0, 0.25)))
    return clip_strategy(s), notes


def implement(strategy, analysis, history, rng, llm=None):
    if llm is not None and not llm.is_null:
        prompt = json.dumps({"current_strategy": strategy, "analysis": analysis["text"],
                             "buckets": analysis["buckets"], "param_ranges": PARAM_RANGES,
                             "easy_knobs_allowed": KNOBS, "history": history[-10:]}, default=str)
        try:
            obj = extract_json(llm.complete(
                "You improve the strategy of a challenger that proposes fptsim simulation specs for an inverse "
                "problem (recover vertex centrality mu from first-passage times). Return ONE JSON object "
                "{\"heuristic\": {...same keys, values inside param_ranges...}, \"guidance\": \"<improved "
                "instructions for an LLM challenger>\", \"why\": \"<one sentence>\"}. Make a targeted change "
                "that addresses the analysis; do not touch solvers or thresholds.", prompt))
            if isinstance(obj, dict) and isinstance(obj.get("heuristic"), dict):
                return clip_strategy(obj), [f"llm: {obj.get('why', '')}"]
        except (LLMError, TypeError, ValueError):
            pass
    return mutate(strategy, analysis, rng)


def optimize(cfg, llm=None, meta_llm=None, should_stop=None, log=print, on_iteration=None):
    m = cfg["meta"]
    rng = random.Random(cfg["seed"])
    scen = list(cfg["scenarios"])
    n_train = min(max(int(round(m["train_fraction"] * len(scen))), 1), len(scen) - 1) if len(scen) > 1 else len(scen)
    train, val = scen[:n_train], (scen[n_train:] or scen)
    log(f"meta-optimization: {m['iterations']} iterations, train {[s['id'] for s in train]}, "
        f"validation {[s['id'] for s in val]}")
    base = clip_strategy(cfg["strategy"] or DEFAULT_STRATEGY)
    ev = evaluate_harness(cfg, base, val, m["eval_rounds"], llm, should_stop)
    population = [{"id": 0, "parent": None, "strategy": base, "score": ev["score"], "pass_rate": ev["pass_rate"]}]
    log(f"baseline validation score {ev['score']:.4f} (pass rate {ev['pass_rate']:.3f})")
    history, stopped = [], not ev["complete"]
    deterministic = llm is None or llm.is_null
    for it in range(1, m["iterations"] + 1):
        if stopped or (should_stop is not None and should_stop()):
            stopped = True
            break
        top = max(c["score"] for c in population)
        w = [math.exp((c["score"] - top) / m["temperature"]) for c in population]
        parent = rng.choices(population, weights=w)[0]
        mb = sorted(rng.sample(range(len(train)), min(m["minibatch"], len(train))))
        tr = evaluate_harness(cfg, parent["strategy"], [train[i] for i in mb], m["eval_rounds"], llm, should_stop)
        analysis = analyze(tr["results"], meta_llm)
        child, notes = implement(parent["strategy"], analysis, history, rng, meta_llm)
        parent_val = parent["score"] if deterministic else evaluate_harness(
            cfg, parent["strategy"], val, m["eval_rounds"], llm, should_stop)["score"]
        cv = evaluate_harness(cfg, child, val, m["eval_rounds"], llm, should_stop)
        if not (tr["complete"] and cv["complete"]):
            stopped = True
            log(f"meta {it}: interrupted by the time budget / handoff; iteration discarded")
            break
        accepted = cv["score"] > parent_val
        entry = {"iteration": it, "parent": parent["id"], "train_pass_rate": tr["pass_rate"],
                 "dominant_failure": analysis["dominant"], "buckets": analysis["buckets"], "changes": notes,
                 "parent_score": parent_val, "child_score": cv["score"], "child_pass_rate": cv["pass_rate"],
                 "accepted": accepted}
        if accepted:
            entry["child"] = len(population)
            population.append({"id": len(population), "parent": parent["id"], "strategy": child,
                               "score": cv["score"], "pass_rate": cv["pass_rate"]})
        history.append(entry)
        log(f"meta {it}: parent {parent['id']} ({parent_val:.4f}) -> child {cv['score']:.4f} "
            f"{'ACCEPTED' if accepted else 'rejected'}  [{analysis['dominant']}; {', '.join(notes)}]")
        if on_iteration is not None:
            on_iteration(it, history, population)
    best = max(population, key=lambda c: c["score"])
    return {"iterations_run": len(history), "iterations": m["iterations"], "stopped_early": stopped,
            "accepted": sum(h["accepted"] for h in history), "baseline_score": population[0]["score"],
            "best_score": best["score"], "best_id": best["id"], "best_pass_rate": best["pass_rate"],
            "best_strategy": best["strategy"], "history": history, "population": population}
