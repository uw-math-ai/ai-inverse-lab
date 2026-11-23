"""Quality verifier (QV) and the weak/strong acceptance gates.

Autodata's QV checks a generated QA item for leakage, recall-vs-reasoning, rubric coverage. The
inverse-problem analogue checks that the instance is a *well-posed, informative* problem before
any solver compute is spent:

  no_unknowns       X \\ B is empty (nothing to reconstruct)          (leakage analogue)
  low_hit_mass      walks almost never reach B within T: the data carry no signal
  unidentifiable    the Jacobian of r w.r.t. mu|U is rank-deficient / badly conditioned at the truth
  noise_dominated   the Cramer-Rao bound on the relative error exceeds verifier.max_crb_rel, so no
                    unbiased solver can succeed at this N (the strong gate would fail anyway)
  duplicate         an identical instance is already in the dataset

Each failure carries a code (used by the heuristic challenger) and a human-readable reason (sent
to the LLM challenger as feedback).
"""
from __future__ import annotations

import numpy as np

from .forward import crb_relative_error, jacobian_condition
from .instance import Instance


def verify(inst: Instance, vcfg: dict, toggles: dict, seen: set[str] | None = None) -> dict:
    p = inst.problem
    diag = {"unknowns": p.m, "k": p.k, "n": p.n,
            "hit_mass_mean": float(inst.r_exact.sum(axis=0).mean())}

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
        diag["jac_cond"] = cond
        if not np.isfinite(cond) or cond > vcfg["max_jac_cond"]:
            return fail("unidentifiable", f"Jacobian of r w.r.t. mu on the unobserved vertices has "
                                          f"condition number {cond:.3g} > {vcfg['max_jac_cond']:.3g}: some "
                                          "vertices are invisible from B within T (increase T or add/move "
                                          "observed vertices)")
    if toggles.get("crb_check", True) and inst.N is not None:
        crb = crb_relative_error(p, inst.mu_true, inst.N)
        diag["crb_rel"] = crb
        if not np.isfinite(crb) or crb > vcfg["max_crb_rel"]:
            return fail("noise_dominated", f"Cramer-Rao bound on the relative error is {crb:.3g} > "
                                           f"{vcfg['max_crb_rel']}: at N = {inst.N} walks per pair no "
                                           "unbiased solver can reconstruct mu (increase N or T, or shrink "
                                           "the unobserved region)")
    if seen is not None and inst.fingerprint() in seen:
        return fail("duplicate", "an identical instance is already in the dataset; change the spec")
    return {"passed": True, "code": "ok", "reason": "", "diagnostics": diag}


def score_from_error(rel_err: float, err_ref: float) -> float:
    """Rubric-score analogue in [0, 1]: 1 for exact reconstruction, 1/e at rel_err = err_ref."""
    return float(np.exp(-rel_err / err_ref))


def weak_gate(scores: list[float], gates: dict) -> tuple[bool, str]:
    s = np.asarray(scores)
    if s.mean() > gates["weak_avg_max"]:
        return False, f"weak_avg {s.mean():.3f} > {gates['weak_avg_max']}"
    if s.max() > gates["weak_max_max"]:
        return False, f"max_weak {s.max():.3f} > {gates['weak_max_max']}"
    if s.min() < gates["weak_min_min"]:
        return False, f"min_weak {s.min():.3f} < {gates['weak_min_min']} (item may be broken)"
    return True, "WEAK_PASSED"


def strong_gate(strong: list[float], weak: list[float], gates: dict) -> tuple[bool, str]:
    s, w = float(np.mean(strong)), float(np.mean(weak))
    if s < gates["strong_avg_min"]:
        return False, f"strong_avg {s:.3f} < {gates['strong_avg_min']}"
    if gates["strong_avg_max"] < 1.0 and s >= gates["strong_avg_max"]:
        return False, f"strong_avg {s:.3f} >= {gates['strong_avg_max']} (saturated)"
    if s - w < gates["gap_min"]:
        return False, f"gap {s - w:.3f} < {gates['gap_min']}"
    return True, "STRONG_PASSED"
