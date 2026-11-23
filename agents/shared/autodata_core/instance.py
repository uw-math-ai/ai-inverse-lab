"""Specs -> instances -> noisy observations, and the quality verifier + acceptance gates."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

from .forward import InverseProblem, crb_relative_error, first_passage, jacobian_condition
from .graphs import SpecError, generate, validate_graphs, validate_simulation, vertex_bound

SPEC_KEYS = {"graphs", "simulation"}


def check_spec(spec, limits):
    if not isinstance(spec, dict):
        raise SpecError("invalid_spec", f"spec must be a mapping, got {type(spec).__name__}")
    extra = set(spec) - SPEC_KEYS
    if extra:
        raise SpecError("invalid_spec", f"spec: unknown keys {sorted(extra)} (allowed: graphs, simulation)")
    if not isinstance(spec.get("graphs"), dict):
        raise SpecError("invalid_spec", "spec.graphs: required mapping")
    if not isinstance(spec.get("simulation"), dict):
        raise SpecError("invalid_spec", "spec.simulation: required mapping {T, N}")
    g = validate_graphs(spec["graphs"])
    s = validate_simulation(spec["simulation"])
    bound = vertex_bound(g)
    if bound > limits["max_n"]:
        raise SpecError("too_large", f"spec can produce {bound} vertices > limits.max_n = {limits['max_n']}")
    if s["T_steps"] > limits["max_T"]:
        raise SpecError("too_large", f"T = {s['T_steps']} steps > limits.max_T = {limits['max_T']}")
    if not limits["min_N"] <= s["N"] <= limits["max_N"]:
        raise SpecError("bad_N", f"N = {s['N']} outside [{limits['min_N']}, {limits['max_N']}]")
    return g, s


class Instance:
    def __init__(self, spec, seed, graph, problem, mu_true, N, r_exact):
        self.spec, self.seed, self.graph, self.problem = spec, seed, graph, problem
        self.mu_true, self.N, self.r_exact = mu_true, N, r_exact

    def fingerprint(self) -> str:
        g = self.graph
        payload = json.dumps([[[round(v, 12) for v in row] for row in g["W"]],
                              [round(v, 12) for v in self.mu_true], g["B"],
                              [round(v, 12) for v in g["q"]], self.problem.T])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def hit_mass(self) -> float:
        p, k2 = self.problem, self.problem.k ** 2
        return sum(self.r_exact) / k2

    def summary(self) -> dict:
        g, p = self.graph, self.problem
        ratio = [self.mu_true[x] / p.wdeg[x] for x in p.U] or [0.0]
        mean = sum(ratio) / len(ratio)
        cv = math.sqrt(sum((v - mean) ** 2 for v in ratio) / len(ratio)) / mean if mean else 0.0
        return {"family": g["family"], "n": p.n, "num_edges": sum(len(r) for r in p.nbrs) // 2, "k": p.k,
                "unknowns": p.m, "T": p.T, "N": self.N, "dt": p.dt, "q_max": max(g["q"]),
                "mu_U_mean": sum(self.mu_true[x] for x in p.U) / max(p.m, 1),
                "mu_over_wdeg_cv": cv, "hit_mass_mean": self.hit_mass()}


def build_instance(spec, seed, limits) -> Instance:
    gres, sres = check_spec(spec, limits)
    g = generate(gres, seed)
    n = len(g["W"])
    if n > limits["max_n"]:
        raise SpecError("too_large", f"graph has {n} vertices > limits.max_n")
    if len(g["B"]) > limits["max_k"]:
        raise SpecError("too_large", f"|B| = {len(g['B'])} > limits.max_k = {limits['max_k']}")
    prob = InverseProblem(g["W"], g["mu"], g["B"], g["q"], g["dt"], sres["T_steps"])
    r = first_passage(prob, g["mu"])
    return Instance(copy.deepcopy(spec), seed, g, prob, list(g["mu"]), sres["N"], r)


def observe(inst: Instance, noise: str, seed: int) -> dict:
    """{'r': flat r_obs (t,i,j), 'N': N | None, 'counts': flat ints | None}.

    multinomial: per ordered pair, N outcomes over (hit at t = 1..T, no hit by T), drawn as a chain
    of binomials (random.binomialvariate, Python >= 3.12). exact: noise-free r.
    """
    if noise == "exact":
        return {"r": list(inst.r_exact), "N": None, "counts": None}
    if noise != "multinomial":
        raise ValueError(f"unknown noise model {noise!r} (multinomial | exact)")
    rng = random.Random(seed)
    T, k2, N = inst.problem.T, inst.problem.k ** 2, inst.N
    counts = [0] * (T * k2)
    for pr in range(k2):
        left, mass = N, 1.0
        for t in range(T):
            if left == 0:
                break
            p = inst.r_exact[t * k2 + pr]
            c = 0 if p <= 0 else (left if p >= mass else rng.binomialvariate(left, min(p / mass, 1.0)))
            counts[t * k2 + pr] = c
            left -= c
            mass = max(mass - p, 0.0)
    return {"r": [c / N for c in counts], "N": N, "counts": counts}


# ---------------------------------------------------------------- verifier
def verify(inst: Instance, vcfg, toggles, seen=None) -> dict:
    p = inst.problem
    diag = {"unknowns": p.m, "k": p.k, "n": p.n, "hit_mass_mean": inst.hit_mass()}

    def fail(code, reason):
        return {"passed": False, "code": code, "reason": reason, "diagnostics": diag}

    if p.m < vcfg["min_unknowns"]:
        return fail("no_unknowns", f"only {p.m} unobserved vertices (need >= {vcfg['min_unknowns']}); "
                                   "B covers (almost) the whole graph, so there is nothing to reconstruct")
    if diag["hit_mass_mean"] < vcfg["min_hit_mass"]:
        return fail("low_hit_mass", f"mean probability of reaching the target within T is "
                                    f"{diag['hit_mass_mean']:.3g} < {vcfg['min_hit_mass']}: increase T, "
                                    "reduce killing q, or move B closer together")
    if toggles.get("identifiability_check", True):
        cond = jacobian_condition(p, inst.mu_true)
        diag["jac_cond"] = cond if math.isfinite(cond) else None
        if not math.isfinite(cond) or cond > vcfg["max_jac_cond"]:
            return fail("unidentifiable", f"Jacobian of r w.r.t. mu on the unobserved vertices has condition "
                                          f"number {cond:.3g} > {vcfg['max_jac_cond']:.3g}: some vertices are "
                                          "invisible from B within T (increase T or add/move observed vertices)")
    if toggles.get("crb_check", True):
        crb = crb_relative_error(p, inst.mu_true, inst.N)
        diag["crb_rel"] = crb if math.isfinite(crb) else None
        if not math.isfinite(crb) or crb > vcfg["max_crb_rel"]:
            return fail("noise_dominated", f"Cramer-Rao bound on the relative error is {crb:.3g} > "
                                           f"{vcfg['max_crb_rel']}: at N = {inst.N} walks per pair no unbiased "
                                           "solver can reconstruct mu (increase N or T, or shrink the unobserved region)")
    if seen is not None and inst.fingerprint() in seen:
        return fail("duplicate", "an identical instance is already in the dataset; change the spec")
    return {"passed": True, "code": "ok", "reason": "", "diagnostics": diag}


def score_from_error(rel_err, err_ref):
    return math.exp(-rel_err / err_ref)


def weak_gate(scores, gates):
    avg = sum(scores) / len(scores)
    if avg > gates["weak_avg_max"]:
        return False, f"weak_avg {avg:.3f} > {gates['weak_avg_max']}"
    if max(scores) > gates["weak_max_max"]:
        return False, f"max_weak {max(scores):.3f} > {gates['weak_max_max']}"
    if min(scores) < gates["weak_min_min"]:
        return False, f"min_weak {min(scores):.3f} < {gates['weak_min_min']} (item may be broken)"
    return True, "WEAK_PASSED"


def strong_gate(strong, weak, gates):
    s, w = sum(strong) / len(strong), sum(weak) / len(weak)
    if s < gates["strong_avg_min"]:
        return False, f"strong_avg {s:.3f} < {gates['strong_avg_min']}"
    if gates["strong_avg_max"] < 1.0 and s >= gates["strong_avg_max"]:
        return False, f"strong_avg {s:.3f} >= {gates['strong_avg_max']} (saturated)"
    if s - w < gates["gap_min"]:
        return False, f"gap {s - w:.3f} < {gates['gap_min']}"
    return True, "STRONG_PASSED"
