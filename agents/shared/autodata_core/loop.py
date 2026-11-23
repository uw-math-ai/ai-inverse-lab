"""Inner loop (stdlib): challenger -> verifier -> weak -> strong -> accept / refine.

Mirrors "Toggable Simulation of Data Code/autodata/loop.py", but keeps results in memory and
leaves file writing to agent.py (the platform collects at most 100 artifact files).
"""
from __future__ import annotations

import copy
import hashlib
import random
import statistics
import time

from .challenger import HeuristicChallenger, LLMChallenger, categorized_feedback
from .graphs import SpecError
from .instance import build_instance, observe, score_from_error, strong_gate, verify, weak_gate
from .solvers import relative_error, solve

PURPOSE = {"instance": 0, "noise": 1, "weak": 2, "strong": 3}


def derive_seed(root, scenario_id, round_, purpose, attempt=0) -> int:
    key = f"{root}|{scenario_id}|{round_}|{PURPOSE[purpose]}|{attempt}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def make_challenger(cfg, strategy, llm=None):
    if llm is not None and not llm.is_null:
        return LLMChallenger(llm, strategy, cfg["limits"], cfg["gates"], cfg["solvers"], cfg["seed"])
    return HeuristicChallenger(strategy, cfg["limits"], cfg["seed"])


def _solver_runs(inst, observations, solver_cfg, cfg, sid, round_, role):
    out = {"rel_err": [], "score": [], "mu_hat": []}
    t0 = time.perf_counter()
    for a, obs in enumerate(observations):
        rng = random.Random(derive_seed(cfg["seed"], sid, round_, role, a))
        mu_hat = solve(inst.problem, obs, solver_cfg, rng)
        err = relative_error(inst.problem, mu_hat, inst.mu_true)
        out["rel_err"].append(err)
        out["score"].append(score_from_error(err, cfg["evaluation"]["err_ref"]))
        out["mu_hat"].append(mu_hat)
    s = out["score"]
    out.update(avg=sum(s) / len(s), max=max(s), min=min(s), seconds=time.perf_counter() - t0)
    return out


def _public(res):
    if res is None:
        return None
    return {"avg": res["avg"], "max": res["max"], "min": res["min"],
            "scores": [round(v, 6) for v in res["score"]], "rel_err": [round(v, 6) for v in res["rel_err"]],
            "seconds": round(res["seconds"], 3)}


def evaluate_spec(spec, cfg, sid, round_, seen):
    rec = {"round": round_, "spec": spec, "status": None, "code": None, "reason": "",
           "weak": None, "strong": None, "instance": None, "diagnostics": None,
           "seed": derive_seed(cfg["seed"], sid, round_, "instance")}
    try:
        inst = build_instance(spec, rec["seed"], cfg["limits"])
    except SpecError as exc:
        rec.update(status="invalid", code=exc.code, reason=str(exc))
        return rec, None
    except (TypeError, ValueError, KeyError, AttributeError) as exc:     # malformed LLM specs
        rec.update(status="invalid", code="invalid_spec", reason=f"spec could not be used: {exc}")
        return rec, None
    rec["instance"] = inst.summary()
    if cfg["toggles"]["quality_verifier"]:
        v = verify(inst, cfg["verifier"], cfg["toggles"], seen)
        rec["diagnostics"] = v["diagnostics"]
        if not v["passed"]:
            rec.update(status="qv_failed", code=v["code"], reason=v["reason"])
            return rec, None
    elif inst.fingerprint() in seen:
        rec.update(status="qv_failed", code="duplicate", reason="identical instance already accepted")
        return rec, None
    ev = cfg["evaluation"]
    observations = [observe(inst, ev["noise"], derive_seed(cfg["seed"], sid, round_, "noise", a))
                    for a in range(ev["attempts"])]
    weak = _solver_runs(inst, observations, cfg["solvers"]["weak"], cfg, sid, round_, "weak")
    rec["weak"] = _public(weak)
    w_ok, w_msg = weak_gate(weak["score"], cfg["gates"])
    strong = None
    if w_ok or not cfg["toggles"]["lazy_strong"]:
        strong = _solver_runs(inst, observations, cfg["solvers"]["strong"], cfg, sid, round_, "strong")
        rec["strong"] = _public(strong)
    if not w_ok:
        rec.update(status="too_easy", code="weak_passed", reason=f"weak solver too good: {w_msg}")
        return rec, None
    s_ok, s_msg = strong_gate(strong["score"], weak["score"], cfg["gates"])
    if not s_ok:
        rec.update(status="failed_strong", code="strong_failed", reason=f"strong gate: {s_msg}")
        return rec, None
    rec.update(status="accepted", code="ok", reason="accepted")
    return rec, {"instance": inst, "observations": observations, "weak": weak, "strong": strong}


def item_record(item_id, scenario, rec, payload, save_observations=True):
    """One dataset line: metadata plus the arrays (plain lists) of an accepted instance."""
    inst = payload["instance"]
    p, g = inst.problem, inst.graph
    arrays = {"W": g["W"], "mu_true": inst.mu_true, "B": p.B, "U": p.U, "q": g["q"], "dt": p.dt,
              "T": p.T, "N": inst.N, "r_exact": inst.r_exact, "r_layout": "flat (t, i, j), t = 1..T"}
    if save_observations:
        arrays["counts"] = [o["counts"] for o in payload["observations"]]
        arrays["mu_weak"] = payload["weak"]["mu_hat"]
        arrays["mu_strong"] = payload["strong"]["mu_hat"]
    return {"item_id": item_id, "scenario": scenario["id"], "round": rec["round"], "seed": rec["seed"],
            "fingerprint": rec["fingerprint"], "spec": rec["spec"], "instance": rec["instance"],
            "diagnostics": rec["diagnostics"], "weak": rec["weak"], "strong": rec["strong"], "arrays": arrays}


def run_scenario(scenario, cfg, challenger, seen, max_rounds=None, on_accept=None, should_stop=None,
                 log=print):
    max_rounds = max_rounds or cfg["loop"]["max_rounds"]
    want = cfg["loop"]["items_per_scenario"]
    history, accepted, t0, stopped = [], [], time.perf_counter(), False
    for round_ in range(1, max_rounds + 1):
        if should_stop is not None and should_stop():
            stopped = True
            break
        spec, note = challenger.propose(scenario, history)
        rec, payload = evaluate_spec(copy.deepcopy(spec), cfg, scenario["id"], round_, seen)
        rec["challenger_note"] = note
        history.append(rec)
        w = rec["weak"]["avg"] if rec["weak"] else None
        s = rec["strong"]["avg"] if rec["strong"] else None
        log(f"[{scenario['id']}] round {round_}: {rec['status']}"
            + (f" weak={w:.3f}" if w is not None else "") + (f" strong={s:.3f}" if s is not None else "")
            + f"  ({note})")
        if payload is not None:
            fp = payload["instance"].fingerprint()
            seen.add(fp)
            item_id = f"{scenario['id']}__r{round_:03d}"
            rec["item_id"], rec["fingerprint"] = item_id, fp
            accepted.append(item_id)
            if on_accept is not None:
                on_accept(item_id, scenario, rec, payload)
            if len(accepted) >= want:
                break
    return {"scenario": scenario["id"], "brief": scenario["brief"], "accepted": bool(accepted),
            "items": accepted, "rounds": len(history), "complete": not stopped,
            "seconds": round(time.perf_counter() - t0, 3),
            "feedback": categorized_feedback(history), "history": history}


def summarize(results):
    status, codes, first = {}, {}, []
    items = []
    for res in results:
        for rec in res["history"]:
            status[rec["status"]] = status.get(rec["status"], 0) + 1
            if rec["status"] in ("qv_failed", "invalid"):
                codes[rec["code"]] = codes.get(rec["code"], 0) + 1
            if rec["status"] == "accepted":
                items.append(rec)
        if res["accepted"]:
            first.append(next(r["round"] for r in res["history"] if r["status"] == "accepted"))
    failed = sum(v for k, v in status.items() if k != "accepted")
    return {
        "scenarios": len(results), "scenarios_accepted": sum(r["accepted"] for r in results),
        "scenarios_incomplete": [r["scenario"] for r in results if not r.get("complete", True)],
        "items": len(items), "rounds": sum(status.values()), "status_counts": status,
        "failure_share": {k: round(v / failed, 4) for k, v in status.items() if k != "accepted" and failed},
        "qv_codes": codes,
        "mean_rounds_to_accept": statistics.mean(first) if first else None,
        "accepted_weak_avg": statistics.mean(i["weak"]["avg"] for i in items) if items else None,
        "accepted_strong_avg": statistics.mean(i["strong"]["avg"] for i in items) if items else None,
        "per_scenario": {r["scenario"]: {"accepted": r["accepted"], "rounds": r["rounds"], "items": r["items"],
                                         "seconds": r["seconds"]} for r in results},
    }
