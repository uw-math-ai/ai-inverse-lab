"""The Autodata inner loop: challenger -> verifier -> weak solver -> strong solver -> accept / refine.

For every scenario (the analogue of a grounding paper):

    for round in 1..max_rounds:
        spec      = challenger.propose(scenario, history)        # fptsim toggles
        instance  = build_instance(spec, harness_seed)            # invalid -> FAILED_QV feedback
        verifier  : no_unknowns | low_hit_mass | unidentifiable | noise_dominated | duplicate
        weak      : `attempts` noise draws, score_a = exp(-rel_err_a / err_ref)
                    weak gate fails -> TOO_EASY feedback
        strong    : same draws (only if the weak gate passed, toggles.lazy_strong)
                    strong gate fails -> FAILED_STRONG feedback
        accept    : write the item (arrays + metadata) to the dataset

Every seed is derived from (config seed, scenario id, round, purpose, attempt). Two harnesses run
on the same scenarios therefore see the same graphs and noise for the same spec (common random
numbers), which keeps the meta-optimizer's parent/child comparison fair.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from .challenger import HeuristicChallenger, LLMChallenger, categorized_feedback, merge_strategy
from .instance import SpecError, build_instance, observe
from .solvers import relative_error, solve
from .verify import score_from_error, strong_gate, verify, weak_gate

log = logging.getLogger("autodata")

PURPOSE = {"instance": 0, "noise": 1, "weak": 2, "strong": 3, "challenger": 4}


def scenario_key(scenario_id: str) -> int:
    return int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16)


def derive_seed(root: int, scenario_id: str, round_: int, purpose: str, attempt: int = 0) -> int:
    ss = np.random.SeedSequence(root, spawn_key=(scenario_key(scenario_id), round_, PURPOSE[purpose], attempt))
    a, b = ss.generate_state(2, np.uint32)
    return (int(a) << 31) ^ int(b)


def make_challenger(cfg: dict, strategy: dict | None, seed: int, llm=None):
    if cfg["toggles"]["llm_challenger"]:
        if llm is None:
            from .llm import make_llm
            llm = make_llm(cfg["llm"])
        if not llm.is_null:
            return LLMChallenger(llm, strategy, cfg["limits"], cfg["gates"], cfg["solvers"], seed)
        log.warning("toggles.llm_challenger is on but no LLM backend is available; using the heuristic")
    return HeuristicChallenger(strategy, cfg["limits"], seed)


def _solver_runs(inst, observations, solver_cfg, cfg, scenario_id, round_, role):
    out = {"rel_err": [], "score": [], "mu_hat": [], "seconds": 0.0}
    t0 = time.perf_counter()
    for a, obs in enumerate(observations):
        rng = np.random.default_rng(derive_seed(cfg["seed"], scenario_id, round_, role, a))
        mu_hat = solve(inst.problem, obs, solver_cfg, rng)
        err = relative_error(inst.problem, mu_hat, inst.mu_true)
        out["rel_err"].append(err)
        out["score"].append(score_from_error(err, cfg["evaluation"]["err_ref"]))
        out["mu_hat"].append(mu_hat)
    out["seconds"] = time.perf_counter() - t0
    out["avg"] = float(np.mean(out["score"]))
    out["max"] = float(np.max(out["score"]))
    out["min"] = float(np.min(out["score"]))
    return out


def _public(res: dict | None) -> dict | None:
    if res is None:
        return None
    return {"avg": res["avg"], "max": res["max"], "min": res["min"],
            "scores": [round(s, 6) for s in res["score"]], "rel_err": [round(e, 6) for e in res["rel_err"]],
            "seconds": round(res["seconds"], 3)}


def evaluate_spec(spec: dict, cfg: dict, scenario_id: str, round_: int, seen: set[str]):
    """One round's evaluation. Returns (record, payload); payload holds arrays when accepted."""
    rec = {"round": round_, "spec": spec, "status": None, "code": None, "reason": "",
           "weak": None, "strong": None, "instance": None, "diagnostics": None}
    seed = derive_seed(cfg["seed"], scenario_id, round_, "instance")
    rec["seed"] = seed
    try:
        inst = build_instance(spec, seed, cfg["limits"])
    except SpecError as exc:
        rec.update(status="invalid", code=exc.code, reason=str(exc))
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
    observations = [observe(inst, ev["noise"], derive_seed(cfg["seed"], scenario_id, round_, "noise", a))
                    for a in range(ev["attempts"])]
    weak = _solver_runs(inst, observations, cfg["solvers"]["weak"], cfg, scenario_id, round_, "weak")
    rec["weak"] = _public(weak)
    w_ok, w_msg = weak_gate(weak["score"], cfg["gates"])
    strong = None
    if w_ok or not cfg["toggles"]["lazy_strong"]:
        strong = _solver_runs(inst, observations, cfg["solvers"]["strong"], cfg, scenario_id, round_, "strong")
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


def run_scenario(scenario: dict, cfg: dict, challenger, seen: set[str], max_rounds: int | None = None,
                 on_accept=None) -> dict:
    max_rounds = max_rounds or cfg["loop"]["max_rounds"]
    want = cfg["loop"]["items_per_scenario"]
    history, accepted, t0 = [], [], time.perf_counter()
    for round_ in range(1, max_rounds + 1):
        spec, note = challenger.propose(scenario, history)
        rec, payload = evaluate_spec(spec, cfg, scenario["id"], round_, seen)
        rec["challenger_note"] = note
        history.append(rec)
        w = rec["weak"]["avg"] if rec["weak"] else None
        s = rec["strong"]["avg"] if rec["strong"] else None
        log.info("[%s] round %d: %s%s%s  (%s)", scenario["id"], round_, rec["status"],
                 f" weak={w:.3f}" if w is not None else "", f" strong={s:.3f}" if s is not None else "",
                 note)
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
            "items": accepted, "rounds": len(history), "seconds": round(time.perf_counter() - t0, 3),
            "feedback": categorized_feedback(history), "history": history}


# ---------------------------------------------------------------- run directory
def _atomic_write(path: Path, data: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data.encode("utf-8") if isinstance(data, str) else data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write_item(dataset_dir: Path, item_id: str, payload: dict, save_observations: bool):
    import io
    inst = payload["instance"]
    g, p = inst.graph, inst.problem
    arrays = {"W": g.W, "mu_true": inst.mu_true, "B": g.B, "q": g.q, "U": p.U,
              "dt": np.array(p.dt), "T": np.array(p.T), "N": np.array(inst.N),
              "r_exact": inst.r_exact, "killed_exact": inst.killed_exact,
              "censored_exact": inst.censored_exact}
    if g.coords is not None:
        arrays["coords"] = g.coords
    if save_observations:
        arrays["counts"] = np.stack([o.counts for o in payload["observations"]])
        arrays["mu_weak"] = np.stack(payload["weak"]["mu_hat"])
        arrays["mu_strong"] = np.stack(payload["strong"]["mu_hat"])
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    _atomic_write(dataset_dir / f"{item_id}.npz", buf.getvalue())


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in list(log.handlers):
        if getattr(h, "_autodata", False):
            log.removeHandler(h)
    for h in (logging.StreamHandler(), logging.FileHandler(out_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        h._autodata = True
        log.addHandler(h)
    log.propagate = False


def run(cfg: dict, strategy: dict | None = None, llm=None) -> dict:
    """Run the inner loop over every scenario; write results and the accepted dataset."""
    out = Path(cfg["output_dir"])
    setup_logging(out)
    strategy = merge_strategy(strategy if strategy is not None else cfg["strategy"])
    _atomic_write(out / "config.resolved.yaml", yaml.safe_dump(
        {**{k: v for k, v in cfg.items() if k != "scenarios"}, "strategy": strategy,
         "scenarios": cfg["scenarios"]}, sort_keys=False))
    challenger = make_challenger(cfg, strategy, cfg["seed"], llm)
    log.info("autodata run %s: %d scenarios, challenger=%s, noise=%s, attempts=%d",
             cfg["run_name"], len(cfg["scenarios"]), challenger.name,
             cfg["evaluation"]["noise"], cfg["evaluation"]["attempts"])
    seen: set[str] = set()
    done = {}
    if cfg["toggles"]["resume"]:
        for s in cfg["scenarios"]:
            path = out / "scenarios" / s["id"] / "result.json"
            if path.exists():
                res = json.loads(path.read_text(encoding="utf-8"))
                done[s["id"]] = res
                seen.update(r["fingerprint"] for r in res["history"] if r.get("fingerprint"))
    dataset_dir = out / "dataset"

    def on_accept(item_id, scenario, rec, payload):
        write_item(dataset_dir, item_id, payload, cfg["toggles"]["save_observations"])

    for s in cfg["scenarios"]:
        if s["id"] in done:
            log.info("[%s] already done (resume) - skipped", s["id"])
            continue
        res = run_scenario(s, cfg, challenger, seen if cfg["toggles"]["final_verifier"] else set(),
                           on_accept=on_accept)
        _atomic_write(out / "scenarios" / s["id"] / "result.json",
                      json.dumps(res, indent=1, default=_jsonable))
        done[s["id"]] = res
    summary = summarize(out)
    log.info("done: %d/%d scenarios produced items, %d items, %d rounds",
             summary["scenarios_accepted"], summary["scenarios"], summary["items"], summary["rounds"])
    return summary


def summarize(out_dir: str | Path) -> dict:
    """Rebuild dataset/items.jsonl and summary.json from every scenarios/*/result.json."""
    out = Path(out_dir)
    results = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((out / "scenarios").glob("*/result.json"))]
    status_counts: dict[str, int] = {}
    qv_codes: dict[str, int] = {}
    items, rounds_to_accept = [], []
    for res in results:
        for rec in res["history"]:
            status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1
            if rec["status"] in ("qv_failed", "invalid"):
                qv_codes[rec["code"]] = qv_codes.get(rec["code"], 0) + 1
            if rec["status"] == "accepted":
                items.append({"item_id": rec["item_id"], "file": f"{rec['item_id']}.npz",
                              "scenario": res["scenario"], "round": rec["round"], "seed": rec["seed"],
                              "fingerprint": rec["fingerprint"], "spec": rec["spec"],
                              "instance": rec["instance"], "diagnostics": rec["diagnostics"],
                              "weak": rec["weak"], "strong": rec["strong"]})
        if res["accepted"]:
            first = next(r["round"] for r in res["history"] if r["status"] == "accepted")
            rounds_to_accept.append(first)
    failed = sum(v for k, v in status_counts.items() if k != "accepted")
    summary = {
        "scenarios": len(results), "scenarios_accepted": sum(r["accepted"] for r in results),
        "items": len(items), "rounds": sum(status_counts.values()), "status_counts": status_counts,
        "failure_share": {k: round(v / failed, 4) for k, v in status_counts.items() if k != "accepted" and failed},
        "qv_codes": qv_codes,
        "mean_rounds_to_accept": float(np.mean(rounds_to_accept)) if rounds_to_accept else None,
        "accepted_weak_avg": float(np.mean([i["weak"]["avg"] for i in items])) if items else None,
        "accepted_strong_avg": float(np.mean([i["strong"]["avg"] for i in items])) if items else None,
        "per_scenario": {r["scenario"]: {"accepted": r["accepted"], "rounds": r["rounds"], "items": r["items"]}
                         for r in results},
    }
    if (out / "scenarios").exists():
        _atomic_write(out / "dataset" / "items.jsonl",
                      "".join(json.dumps(i, default=_jsonable) + "\n" for i in items))
        _atomic_write(out / "summary.json", json.dumps(summary, indent=1, default=_jsonable))
    return summary


def load_dataset(run_dir: str | Path) -> list[dict]:
    """Accepted items: metadata from items.jsonl plus the arrays from each item's .npz.

    Arrays: W, mu_true, B, q, U, dt, T, N, r_exact, killed_exact, censored_exact, coords (lattices),
    and with toggles.save_observations: counts (A, T, k, k), mu_weak (A, n), mu_strong (A, n).
    The ML target is mu_true[U]; the inputs are W, B, q, mu_true[B], dt and counts / N.
    """
    run_dir = Path(run_dir)
    path = run_dir / "dataset" / "items.jsonl"
    if not path.exists():
        summarize(run_dir)
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        with np.load(run_dir / "dataset" / meta["file"]) as z:
            meta["arrays"] = {k: z[k] for k in z.files}
        items.append(meta)
    return items
