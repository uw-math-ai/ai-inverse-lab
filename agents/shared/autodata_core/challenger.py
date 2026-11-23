"""Challengers (stdlib port of "Toggable Simulation of Data Code/autodata/challenger.py").

HeuristicChallenger   rule-based, offline, deterministic given its seed; tuned by the meta loop.
LLMChallenger         asks a model for the next spec (llm.py); falls back to the heuristic when the
                      model call fails or the reply has no JSON spec. An invalid spec is passed through
                      and comes back to the model as FAILED_QV feedback.
"""
from __future__ import annotations

import copy
import json
import random

from .graphs import FAMILY_PARAMS, SpecError, validate_graphs, vertex_bound
from .llm import LLMError, extract_json

DEFAULT_STRATEGY = {
    "heuristic": {
        "easy_knobs": ["spread", "depth", "size", "noise"],
        "spread_step": 0.75, "spread_cap": 6.0, "size_factor": 1.3, "depth_k_factor": 0.7,
        "noise_factor": 0.5, "hard_N_factor": 4.0, "hard_T_factor": 1.5, "hard_k_step": 1,
        "explore_prob": 0.1,
    },
    "guidance": (
        "Good items have centrality that is NOT predictable from weighted degree (heterogeneous mu), "
        "unobserved vertices that are only visible through longer walks (depth), and enough walks N "
        "and horizon T that a full-horizon least-squares fit still recovers mu. Change one or two "
        "things at a time, guided by the feedback numbers."),
}
SIZE_KEYS = {"rows", "cols", "length", "n"}


def merge_strategy(override):
    s = copy.deepcopy(DEFAULT_STRATEGY)
    if override:
        s["heuristic"].update(override.get("heuristic") or {})
        if override.get("guidance"):
            s["guidance"] = str(override["guidance"])
    return s


def categorized_feedback(history):
    fb = {"TOO_EASY": [], "FAILED_STRONG": [], "FAILED_QV": []}
    for rec in history:
        if rec["status"] == "too_easy":
            fb["TOO_EASY"].append({"round": rec["round"], "weak_avg": rec["weak"]["avg"]})
        elif rec["status"] == "failed_strong":
            fb["FAILED_STRONG"].append({"round": rec["round"], "weak_avg": rec["weak"]["avg"],
                                        "strong_avg": rec["strong"]["avg"],
                                        "gap": rec["strong"]["avg"] - rec["weak"]["avg"]})
        elif rec["status"] in ("qv_failed", "invalid"):
            fb["FAILED_QV"].append({"round": rec["round"], "code": rec["code"], "reason": rec["reason"]})
    return {k: v for k, v in fb.items() if v}


# ---------------------------------------------------------------- spec editing
def _scale(value, factor, as_int, lo, hi):
    def one(v):
        x = min(max(v * factor, lo), hi)
        return int(round(x)) if as_int else float(x)
    if isinstance(value, dict) and "choice" in value:
        return {"choice": sorted({one(v) for v in value["choice"]})}
    if isinstance(value, list):
        a, b = one(value[0]), one(value[1])
        if as_int and b == value[1] and factor > 1:
            b = min(b + 1, int(hi))
        return [min(a, b), max(a, b)]
    out = one(value)
    if as_int and out == value and factor != 1:
        out = int(min(max(value + (1 if factor > 1 else -1), lo), hi))
    return out


def _bound(spec):
    try:
        return vertex_bound(validate_graphs(spec["graphs"]))
    except (SpecError, KeyError, TypeError):
        return 0


def _resize(spec, factor, limits):
    fams = (spec.get("graphs") or {}).get("families") or {}
    before = copy.deepcopy(fams)

    def apply(f):
        changed = False
        for name, entry in fams.items():
            for key, (kind, lo, hi) in FAMILY_PARAMS.get(name, {}).items():
                if key in SIZE_KEYS and key in entry:
                    new = _scale(entry[key], f, kind == "int", lo, hi)
                    changed |= new != entry[key]
                    entry[key] = new
        return changed

    apply(factor)
    while fams and _bound(spec) > limits["max_n"]:
        if not apply(0.85):
            break
    return fams != before


def _spread(spec, h):
    g = spec.setdefault("graphs", {})
    if g.get("presets"):
        return False
    mu = g.get("mu") or {"mode": "deg"}
    mode = mu.get("mode")
    if mode in ("deg", "weighted_deg", "constant"):
        g["mu"] = {"mode": "slack", "a": 0.0, "b": float(h["spread_step"]), "on_violation": "fix"}
        return True
    if mode == "slack":
        b = min(float(mu.get("b", 0)) + h["spread_step"], h["spread_cap"])
        changed = b != mu.get("b")
        mu["b"] = b
        return changed
    if mode == "uniform":
        mu["high"] = float(mu["high"]) * (1 + h["spread_step"])
        return True
    return False


def _B_size(spec, factor=None, step=None, limits=None):
    B = (spec.get("graphs") or {}).get("B")
    if not B:
        return False
    cap = limits["max_k"] if limits else 10 ** 9

    def change(v):
        if factor is not None:
            return _scale(v, factor, True, 1, cap)
        if isinstance(v, list):
            return [min(v[0] + step, cap), min(v[1] + step, cap)]
        if isinstance(v, dict):
            return {"choice": sorted({min(c + step, cap) for c in v["choice"]})}
        return max(1, min(int(v) + step, cap))

    mode = B.get("mode")
    if mode in ("random_k", "low_degree", "high_degree") and B.get("k") is not None:
        new = change(B["k"])
        changed, B["k"] = new != B["k"], new
        return changed
    if mode == "random_k" and B.get("frac") is not None:
        old = float(B["frac"])
        B["frac"] = float(min(max(old * (factor if factor is not None else 1.25), 0.02), 1.0))
        return B["frac"] != old
    if mode in ("boundary", "leaves"):
        old = B.get("max_k")
        if old is None and factor is None:
            return False
        B["max_k"] = int(change(int(old) if old is not None else 8))
        return B["max_k"] != old
    return False


def _set_N(spec, factor, limits):
    sim = spec.setdefault("simulation", {})
    old = int(float(sim.get("N", 10_000)))
    sim["N"] = int(min(max(old * factor, limits["min_N"]), limits["max_N"]))
    return sim["N"] != old


def _set_T(spec, factor, limits):
    sim = spec.setdefault("simulation", {})
    old = int(sim.get("T", 17))
    cap = (limits["max_T"] + 1) // 2 if sim.get("T_mode") == "paper" else limits["max_T"]
    new = int(min(max(round(old * factor), 1), cap))
    if new == old and factor > 1:
        new = min(old + 1, cap)
    sim["T"] = new
    return new != old


def _scale_q(spec, factor):
    q = (spec.get("graphs") or {}).get("q")
    if not q or q.get("mode", "zero") == "zero":
        return False
    for key in ("c", "high", "low", "amplitude"):
        if key in q:
            q[key] = float(q[key]) * factor
    if "values" in q:
        q["values"] = [float(v) * factor for v in q["values"]]
    return True


class HeuristicChallenger:
    name = "heuristic"

    def __init__(self, strategy, limits, seed=0):
        self.h = merge_strategy(strategy)["heuristic"]
        self.limits = limits
        self.rng = random.Random(seed)

    def propose(self, scenario, history):
        if not history:
            return copy.deepcopy(scenario["base_spec"]), "scenario base spec"
        last = history[-1]
        valid = [r for r in history if r["status"] != "invalid"]
        spec = copy.deepcopy(valid[-1]["spec"] if valid else scenario["base_spec"])
        h, lim, notes = self.h, self.limits, []
        if scenario.get("families") and self.rng.random() < h["explore_prob"]:
            fam = copy.deepcopy(self.rng.choice(scenario["families"]))
            g = spec.setdefault("graphs", {})
            g.pop("presets", None)
            g["families"] = fam
            g.setdefault("B", {"mode": "random_k", "k": [2, 4]})
            if g["B"].get("mode") == "boundary" and any(f not in ("grid", "grid_holes", "pattern", "ladder") for f in fam):
                g["B"] = {"mode": "random_k", "k": [2, 4]}
            notes.append(f"explore: switch families to {sorted(fam)}")
        status, code = last["status"], last.get("code")
        if status == "too_easy":
            streak = 0
            for rec in reversed(history):
                if rec["status"] != "too_easy":
                    break
                streak += 1
            knobs = list(h["easy_knobs"])
            for offset in range(len(knobs)):
                knob = knobs[(streak - 1 + offset) % len(knobs)]
                if self._easy(spec, knob):
                    notes.append(f"too easy -> {knob}")
                    break
        elif status == "failed_strong" or code == "noise_dominated":
            if _set_N(spec, h["hard_N_factor"], lim):
                notes.append("strong failed -> more walks N")
            elif _set_T(spec, h["hard_T_factor"], lim):
                notes.append("N at cap -> longer horizon T")
            elif _B_size(spec, step=int(h["hard_k_step"]), limits=lim):
                notes.append("N, T at cap -> observe more vertices")
            else:
                _resize(spec, 1 / h["size_factor"], lim)
                notes.append("all at cap -> smaller graph")
        elif code == "no_unknowns":
            _B_size(spec, factor=h["depth_k_factor"], limits=lim)
            _resize(spec, h["size_factor"], lim)
            notes.append("no unknowns -> fewer observed vertices, bigger graph")
        elif code == "low_hit_mass":
            _set_T(spec, h["hard_T_factor"], lim)
            _scale_q(spec, 0.5)
            notes.append("low hit mass -> longer T, weaker killing")
        elif code == "unidentifiable":
            if not _set_T(spec, h["hard_T_factor"], lim):
                _B_size(spec, step=int(h["hard_k_step"]), limits=lim)
            notes.append("unidentifiable -> longer T / more observers")
        elif code in ("too_large", "bad_N"):
            _resize(spec, 1 / h["size_factor"], lim)
            sim = spec.setdefault("simulation", {})
            sim["N"] = int(min(max(int(float(sim.get("N", 10_000))), lim["min_N"]), lim["max_N"]))
            _set_T(spec, 1.0, lim)
            notes.append("outside limits -> clamp to limits")
        elif code == "duplicate":
            _resize(spec, h["size_factor"], lim)
            notes.append("duplicate -> perturb size")
        elif status == "accepted":
            _resize(spec, h["size_factor"] if self.rng.random() < 0.5 else 1 / h["size_factor"], lim)
            _spread(spec, h)
            notes.append("accepted -> diversify")
        elif status == "invalid":
            notes.append(f"invalid ({code}) -> reverted to last valid spec")
        return spec, "; ".join(notes) or "no change"

    def _easy(self, spec, knob):
        h, lim = self.h, self.limits
        if knob == "spread":
            return _spread(spec, h)
        if knob == "depth":
            return _B_size(spec, factor=h["depth_k_factor"], limits=lim)
        if knob == "size":
            return _resize(spec, h["size_factor"], lim)
        if knob == "noise":
            return _set_N(spec, h["noise_factor"], lim)
        raise ValueError(f"unknown easy knob {knob!r}")


SPEC_SCHEMA = """\
Return ONE JSON object: {"rationale": "<one or two sentences>", "spec": {"graphs": {...}, "simulation": {...}}}

spec.graphs is an fptsim `graphs:` block (strictly validated; unknown keys are errors):
  families: {name: {weight?, params}}  name in grid{rows,cols} grid_holes{rows,cols,hole_frac}
            ladder{length} path{n} cycle{n} star{n} complete{n} random_tree{n} erdos_renyi{n,p}
            random_geometric{n,radius} watts_strogatz{n,k,beta} barabasi_albert{n,m} random_regular{n,d}
            pattern{patterns: [["XooX","XXXX"], ...]}
            a param is a value 5, an inclusive range [2, 6], or {"choice": [8, 12]}
  OR presets: ["paper8"] / ["paper9"]   (then omit weights, mu, B)
  weights: {dist: constant{value} | uniform{low>0,high} | lognormal{mean,sigma} | gamma{shape,scale}
            | discrete{values,probs}, rescale?: {mode: max|mean, value}}
  mu:      {mode: constant{value} | deg | weighted_deg | uniform{low,high} | slack{a,b}, on_violation?: fix}
           (slack: mu = weighted_deg * (1 + U(a, b)); mu below the weighted degree is raised to it)
  B:       {mode: random_k{k | frac} | boundary (lattices) | leaves | low_degree{k} | high_degree{k}
            | explicit{vertices}, max_k?}
  q:       {mode: zero | constant{c} | uniform{low,high} | sparse{fraction,low,high}
            | localized{center: "random"|int, decay, amplitude} | explicit{values}, on_B_only?, off_B_only?}
  dt:      "auto" (recommended)
spec.simulation: {T: int, T_mode?: "steps"|"paper", N: walks per ordered pair}
Do not set seeds, solvers or thresholds; the harness owns them."""


class LLMChallenger:
    name = "llm"

    def __init__(self, llm, strategy, limits, gates, solvers, seed=0):
        self.llm = llm
        self.strategy = merge_strategy(strategy)
        self.limits, self.gates, self.solvers = limits, gates, solvers
        self.fallback = HeuristicChallenger(strategy, limits, seed)

    def system_prompt(self):
        return (
            "You are the challenger in an Autodata-style data-generation loop for an inverse problem.\n"
            "Inverse problem (Gao, Li & Yang, arXiv:2603.20710): random walks on a weighted graph with "
            "K[x,y] = h W[x,y]/mu(x), K[x,x] = 1 - h(sum_y W[x,y]/mu(x) + q(x)). Given W, q, B and mu on "
            "B, plus first-passage-time counts r(t,x,y) for x,y in B, t <= T (N walks per pair), recover "
            "the vertex centrality mu on the unobserved vertices X \\ B.\n"
            f"A WEAK solver ({json.dumps(self.solvers['weak'])}) and a STRONG solver "
            f"({json.dumps(self.solvers['strong'])}) each reconstruct mu from the same noisy draws; "
            "score = exp(-relative_error / err_ref).\n"
            f"An item is ACCEPTED when the verifier passes and the gates hold: {json.dumps(self.gates)} "
            "(weak must FAIL, strong must SUCCEED, with a gap).\n"
            f"Harness limits: {json.dumps(self.limits)}.\n\n"
            f"Strategy guidance:\n{self.strategy['guidance']}\n\n{SPEC_SCHEMA}")

    def user_prompt(self, scenario, history):
        lines = [f"SCENARIO {scenario['id']}: {scenario['brief']}",
                 f"Base spec: {json.dumps(scenario['base_spec'])}"]
        if scenario.get("families"):
            lines.append(f"Alternative families you may use: {json.dumps(scenario['families'])}")
        if history:
            lines.append("\nPrevious rounds (most recent last):")
            for rec in history[-8:]:
                inst = rec.get("instance") or {}
                lines.append(json.dumps({
                    "round": rec["round"], "status": rec["status"], "reason": rec.get("reason"),
                    "weak_avg": (rec.get("weak") or {}).get("avg"),
                    "strong_avg": (rec.get("strong") or {}).get("avg"),
                    "instance": {k: inst.get(k) for k in ("n", "k", "unknowns", "T", "N", "mu_over_wdeg_cv") if k in inst},
                    "diagnostics": rec.get("diagnostics"), "spec": rec["spec"]}, default=str))
            lines.append("\nFeedback:")
            for cat, items in categorized_feedback(history).items():
                lines.append(f"{cat}: {json.dumps(items, default=str)}")
            lines.append("\nPropose the next spec. Address the most recent failure directly; if the same "
                         "failure repeats, try a different angle (another family, B placement, mu law).")
        else:
            lines.append("\nPropose the first spec for this scenario.")
        return "\n".join(lines)

    def propose(self, scenario, history):
        try:
            text = self.llm.complete(self.system_prompt(), self.user_prompt(scenario, history))
            obj = extract_json(text)
            if not isinstance(obj, dict) or not isinstance(obj.get("spec"), dict):
                raise LLMError("response had no JSON object with a `spec`")
            return obj["spec"], f"[llm] {obj.get('rationale', '')}".strip()
        except LLMError as exc:
            spec, note = self.fallback.propose(scenario, history)
            return spec, f"[llm failed: {str(exc)[:200]}; heuristic] {note}"
