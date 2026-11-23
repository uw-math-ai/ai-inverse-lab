"""autodata harness: forward model, solvers, specs, verifier, gates, challengers, loop, meta loop."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from autodata.challenger import HeuristicChallenger, LLMChallenger, categorized_feedback
from autodata.config import AutodataConfigError, validate
from autodata.forward import (crb_relative_error, first_passage, first_passage_jac,
                              jacobian_condition)
from autodata.instance import SpecError, build_instance, check_spec, observe
from autodata.loop import evaluate_spec, load_dataset, run, summarize
from autodata.meta import clip_strategy, mutate, optimize
from autodata.solvers import Observation, relative_error, solve
from autodata.verify import strong_gate, verify, weak_gate
from fptsim.exact import exact_single
from fptsim.kernel import build_kernel

LIMITS = {"max_n": 30, "max_k": 12, "max_T": 80, "min_N": 100, "max_N": 10 ** 7}
VCFG = {"min_unknowns": 1, "min_hit_mass": 0.02, "max_jac_cond": 1e8, "max_crb_rel": 0.10}
TOGGLES = {"identifiability_check": True, "crb_check": True}


def grid_spec(**over):
    spec = {"graphs": {"families": {"grid": {"rows": 3, "cols": 4}},
                       "weights": {"dist": "uniform", "low": 0.3, "high": 1.0},
                       "mu": {"mode": "slack", "a": 0.0, "b": 1.5},
                       "B": {"mode": "boundary", "max_k": 5}},
            "simulation": {"T": 30, "N": 50000}}
    for k, v in over.items():
        spec[k] = v
    return spec


@pytest.fixture(scope="module")
def inst():
    return build_instance(grid_spec(), 11, LIMITS)


# ---------------------------------------------------------------- forward model
@pytest.mark.parametrize("q_mode", [{"mode": "zero"}, {"mode": "sparse", "fraction": 0.4, "low": 0.1, "high": 0.6}])
def test_forward_matches_fptsim_exact(q_mode):
    spec = grid_spec()
    spec["graphs"]["q"] = q_mode
    spec["graphs"]["families"] = {"erdos_renyi": {"n": 10, "p": 0.4}}
    spec["graphs"]["B"] = {"mode": "random_k", "k": 3}
    for seed in range(4):
        i = build_instance(spec, seed, LIMITS)
        p = i.problem
        r = first_passage(p.kernel(i.mu_true), p.B, p.T)
        ref = exact_single(build_kernel(i.graph.W, i.graph.mu, i.graph.q), i.graph.B, p.T)["r"]
        assert np.abs(r - ref).max() < 1e-13
        assert np.abs(p.kernel(i.mu_true) - build_kernel(i.graph.W, i.graph.mu, i.graph.q).K).max() < 1e-14


def test_jacobian_matches_finite_differences(inst):
    p = inst.problem
    r0, J = first_passage_jac(p, inst.mu_true)
    assert np.abs(r0 - inst.r_exact).max() < 1e-13
    a = 1 / inst.mu_true
    for col, x in enumerate(p.U):
        eps = 1e-6 * a[x]
        ap, am = a.copy(), a.copy()
        ap[x] += eps
        am[x] -= eps
        fd = (first_passage(p.kernel(1 / ap), p.B, p.T) - first_passage(p.kernel(1 / am), p.B, p.T)) / (2 * eps)
        assert np.abs(fd - J[..., col]).max() < 1e-6 * max(1.0, np.abs(J[..., col]).max())


def test_crb_shrinks_like_one_over_sqrt_N(inst):
    p = inst.problem
    c1 = crb_relative_error(p, inst.mu_true, 10_000)
    c4 = crb_relative_error(p, inst.mu_true, 40_000)
    assert np.isfinite(c1) and c1 / c4 == pytest.approx(2.0, rel=1e-6)
    assert jacobian_condition(p, inst.mu_true) < 1e6


# ---------------------------------------------------------------- solvers
def test_strong_lsq_recovers_mu_from_exact_data(inst):
    obs = observe(inst, "exact", 0)
    mu = solve(inst.problem, obs, {"solver": "lsq", "restarts": 1}, np.random.default_rng(0))
    assert relative_error(inst.problem, mu, inst.mu_true) < 1e-8
    assert np.array_equal(mu[inst.problem.B], inst.mu_true[inst.problem.B])   # mu on B is never touched


def test_prior_is_exact_for_a_degree_law():
    spec = grid_spec()
    spec["graphs"]["mu"] = {"mode": "weighted_deg"}
    i = build_instance(spec, 3, LIMITS)
    mu = solve(i.problem, observe(i, "exact", 0), {"solver": "prior"}, np.random.default_rng(0))
    assert relative_error(i.problem, mu, i.mu_true) < 1e-12


def test_strong_beats_short_horizon_weak_on_noisy_data(inst):
    obs = observe(inst, "multinomial", 5)
    rng = np.random.default_rng(0)
    weak = relative_error(inst.problem, solve(inst.problem, obs, {"solver": "lsq", "T": 2}, rng), inst.mu_true)
    strong = relative_error(inst.problem, solve(inst.problem, obs, {"solver": "lsq", "restarts": 1}, rng),
                            inst.mu_true)
    assert strong < 0.05 and strong < weak


def test_unknown_solver_raises(inst):
    with pytest.raises(KeyError):
        solve(inst.problem, observe(inst, "exact", 0), {"solver": "nope"}, np.random.default_rng(0))


# ---------------------------------------------------------------- specs and observations
def test_spec_validation_and_limits():
    with pytest.raises(SpecError) as e:
        check_spec({**grid_spec(), "seed": 3}, LIMITS)
    assert e.value.code == "invalid_spec"                    # seeds belong to the harness
    bad = grid_spec()
    bad["graphs"]["mu"] = {"mode": "magic"}
    with pytest.raises(SpecError) as e:
        check_spec(bad, LIMITS)
    assert "fptsim rejected" in str(e.value)
    big = grid_spec()
    big["graphs"]["families"] = {"grid": {"rows": [3, 9], "cols": 9}}
    with pytest.raises(SpecError) as e:
        check_spec(big, LIMITS)
    assert e.value.code == "too_large"
    with pytest.raises(SpecError) as e:
        check_spec(grid_spec(simulation={"T": 10, "N": 10}), LIMITS)
    assert e.value.code == "bad_N"


def test_instance_is_deterministic_in_spec_and_seed():
    a, b = build_instance(grid_spec(), 4, LIMITS), build_instance(grid_spec(), 4, LIMITS)
    c = build_instance(grid_spec(), 5, LIMITS)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


@pytest.mark.parametrize("noise", ["multinomial", "monte_carlo"])
def test_observations_are_consistent_with_the_exact_law(inst, noise):
    small = copy.copy(inst)
    small.N = 4000
    obs = observe(small, noise, 1)
    assert obs.counts.shape == inst.r_exact.shape and obs.counts.min() >= 0
    assert np.all(obs.counts.sum(axis=0) <= small.N)
    z = (obs.r_obs - inst.r_exact) / np.sqrt(np.maximum(inst.r_exact * (1 - inst.r_exact), 1e-12) / small.N)
    assert np.abs(z[inst.r_exact > 0.01]).mean() < 1.5         # ~0.8 for an unbiased sampler


# ---------------------------------------------------------------- verifier and gates
def test_verifier_codes(inst):
    assert verify(inst, VCFG, TOGGLES)["passed"]
    full = grid_spec()
    full["graphs"]["B"] = {"mode": "explicit", "vertices": list(range(12))}
    assert verify(build_instance(full, 1, LIMITS), VCFG, TOGGLES)["code"] == "no_unknowns"
    noisy = build_instance(grid_spec(simulation={"T": 30, "N": 100}), 11, LIMITS)
    assert verify(noisy, VCFG, TOGGLES)["code"] == "noise_dominated"
    short = build_instance(grid_spec(simulation={"T": 1, "N": 50000}), 11, LIMITS)
    assert verify(short, VCFG, TOGGLES)["code"] in ("unidentifiable", "low_hit_mass")
    assert verify(inst, VCFG, TOGGLES, seen={inst.fingerprint()})["code"] == "duplicate"


def test_gates_follow_autodata_thresholds():
    g = {"weak_avg_max": 0.65, "weak_max_max": 0.75, "weak_min_min": 0.0,
         "strong_avg_min": 0.60, "strong_avg_max": 0.95, "gap_min": 0.20}
    assert weak_gate([0.4, 0.5, 0.6], g)[0]
    assert not weak_gate([0.4, 0.5, 0.8], g)[0]                 # max_weak > 0.75
    assert not weak_gate([0.7, 0.7, 0.7], g)[0]                 # weak_avg > 0.65
    assert strong_gate([0.8, 0.8], [0.5, 0.5], g)[0]
    assert not strong_gate([0.65, 0.65], [0.5, 0.5], g)[0]      # gap < 0.20
    assert not strong_gate([0.55, 0.55], [0.1, 0.1], g)[0]      # strong_avg < 0.60
    assert not strong_gate([0.97, 0.97], [0.1, 0.1], g)[0]      # saturated (only when strong_avg_max < 1)


# ---------------------------------------------------------------- challengers
def _rec(round_, status, spec, code=None, weak=0.9, strong=None):
    return {"round": round_, "status": status, "spec": spec, "code": code, "reason": code or "",
            "weak": {"avg": weak}, "strong": None if strong is None else {"avg": strong}}


def test_heuristic_challenger_moves(inst):
    scen = {"id": "s", "brief": "", "base_spec": grid_spec(), "families": []}
    ch = HeuristicChallenger(None, LIMITS, 0)
    spec0, _ = ch.propose(scen, [])
    assert spec0 == scen["base_spec"]
    deg = grid_spec()
    deg["graphs"]["mu"] = {"mode": "deg"}
    spec1, note = ch.propose(scen, [_rec(1, "too_easy", deg)])
    assert spec1["graphs"]["mu"]["mode"] == "slack" and "spread" in note
    spec2, note = ch.propose(scen, [_rec(1, "failed_strong", grid_spec(), strong=0.4)])
    assert spec2["simulation"]["N"] == 200_000 and "walks" in note
    capped = grid_spec(simulation={"T": 30, "N": LIMITS["max_N"]})
    spec3, note = ch.propose(scen, [_rec(1, "failed_strong", capped, strong=0.4)])
    assert spec3["simulation"]["T"] == 45 and spec3["simulation"]["N"] == LIMITS["max_N"]
    spec4, _ = ch.propose(scen, [_rec(1, "qv_failed", grid_spec(), code="no_unknowns")])
    check_spec(spec4, LIMITS)                                    # every move stays inside the limits
    spec5, note = ch.propose(scen, [_rec(1, "invalid", {"graphs": "garbage"}, code="invalid_spec")])
    assert spec5 == scen["base_spec"]


class FakeLLM:
    is_null, name = False, "fake"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, prompt):
        self.prompts.append((system, prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_llm_challenger_parses_falls_back_and_passes_feedback():
    from autodata.llm import LLMError
    scen = {"id": "s", "brief": "grid", "base_spec": grid_spec(), "families": []}
    good = json.dumps({"rationale": "deeper", "spec": grid_spec(simulation={"T": 40, "N": 30000})})
    llm = FakeLLM([f"```json\n{good}\n```", "no json here", LLMError("down")])
    gates = {"weak_avg_max": 0.65}
    ch = LLMChallenger(llm, None, LIMITS, gates, {"weak": {"solver": "lsq"}, "strong": {"solver": "lsq"}})
    spec, note = ch.propose(scen, [])
    assert spec["simulation"]["T"] == 40 and note.startswith("[llm]")
    hist = [_rec(1, "too_easy", grid_spec(), weak=0.9)]
    spec, note = ch.propose(scen, hist)                          # unparseable -> heuristic fallback
    assert "heuristic" in note and spec["graphs"]["mu"]["b"] > 1.5
    assert "TOO_EASY" in llm.prompts[1][1] and "Cram" not in llm.prompts[1][1]
    spec, note = ch.propose(scen, hist)                          # backend error -> heuristic fallback
    assert "llm failed" in note
    assert categorized_feedback(hist) == {"TOO_EASY": [{"round": 1, "weak_avg": 0.9}]}


# ---------------------------------------------------------------- loop, dataset, resume, meta
def small_cfg(tmp_path, **over):
    raw = {"run_name": "t", "output_dir": str(tmp_path / "run"), "seed": 3,
           "scenarios": [
               {"id": "path", "brief": "path ends", "base_spec": {
                   "graphs": {"families": {"path": {"n": 6}}, "weights": {"dist": "uniform", "low": 0.5, "high": 1.0},
                              "mu": {"mode": "weighted_deg"}, "B": {"mode": "leaves"}},
                   "simulation": {"T": 25, "N": 50000}}},
               {"id": "grid", "brief": "grid", "base_spec": grid_spec()}],
           "loop": {"max_rounds": 5},
           "solvers": {"weak": {"solver": "lsq", "T": 3, "max_iter": 30},
                       "strong": {"solver": "lsq", "restarts": 0, "max_iter": 80}},
           "verifier": {"max_jac_cond": "1e8"},                 # YAML-style numeric string
           "meta": {"iterations": 1, "minibatch": 1, "eval_rounds": 3}}
    raw.update(over)
    return validate(raw)


def test_config_rejects_unknown_keys_and_coerces_numbers(tmp_path):
    cfg = small_cfg(tmp_path)
    assert cfg["verifier"]["max_jac_cond"] == 1e8 and cfg["toggles"]["quality_verifier"] is True
    with pytest.raises(AutodataConfigError, match="gates.weak_avg"):
        small_cfg(tmp_path, gates={"weak_avg": 0.5})
    with pytest.raises(AutodataConfigError, match="solvers.strong.solver"):
        small_cfg(tmp_path, solvers={"weak": {"solver": "lsq"}, "strong": {"solver": "nope"}})


def test_evaluate_spec_records_every_outcome(tmp_path):
    cfg = small_cfg(tmp_path)
    rec, payload = evaluate_spec({"graphs": {}}, cfg, "x", 1, set())
    assert rec["status"] == "invalid" and payload is None
    rec, payload = evaluate_spec(grid_spec(simulation={"T": 30, "N": 100}), cfg, "x", 1, set())
    assert rec["status"] == "qv_failed" and rec["code"] == "noise_dominated"
    easy = grid_spec()
    easy["graphs"]["mu"] = {"mode": "weighted_deg"}
    rec, _ = evaluate_spec(easy, cfg, "x", 1, set())
    assert rec["status"] == "too_easy" and rec["strong"] is None     # lazy strong evaluation


def test_run_writes_dataset_and_resumes(tmp_path):
    cfg = small_cfg(tmp_path)
    summary = run(cfg)
    assert summary["scenarios"] == 2 and summary["items"] >= 1
    assert summary["rounds"] == sum(summary["status_counts"].values())
    items = load_dataset(cfg["output_dir"])
    assert len(items) == summary["items"]
    it = items[0]
    a = it["arrays"]
    assert a["counts"].shape == (3, int(a["T"]), len(a["B"]), len(a["B"]))
    assert it["weak"]["avg"] <= 0.65 and it["strong"]["avg"] >= 0.60
    assert it["strong"]["avg"] - it["weak"]["avg"] >= 0.20
    # the stored instance is reproducible from (spec, seed) with plain fptsim
    again = build_instance(it["spec"], it["seed"], cfg["limits"])
    assert again.fingerprint() == it["fingerprint"]
    assert np.array_equal(again.mu_true, a["mu_true"])
    before = (tmp_path / "run" / "scenarios" / "path" / "result.json").read_text()
    summary2 = run(cfg)                                          # resume: nothing is recomputed
    assert summary2["items"] == summary["items"]
    assert (tmp_path / "run" / "scenarios" / "path" / "result.json").read_text() == before
    assert summarize(cfg["output_dir"])["items"] == summary["items"]


def test_meta_optimizer_mutation_and_loop(tmp_path):
    s = clip_strategy({"heuristic": {"noise_factor": 5.0, "easy_knobs": ["bogus", "size"]}})
    assert s["heuristic"]["noise_factor"] == 0.95 and s["heuristic"]["easy_knobs"] == ["size"]
    child, notes = mutate(s, {"dominant": "too_hard"}, np.random.default_rng(0))
    assert child["heuristic"]["hard_N_factor"] > s["heuristic"]["hard_N_factor"] and notes
    cfg = small_cfg(tmp_path)
    summary = optimize(cfg)
    meta = tmp_path / "run" / "meta"
    assert summary["iterations"] == 1 and summary["best_score"] >= summary["baseline_score"]
    assert (meta / "best_strategy.yaml").exists() and (meta / "history.jsonl").exists()
    hist = [json.loads(line) for line in (meta / "history.jsonl").read_text().splitlines()]
    assert hist[0]["accepted"] == (hist[0]["child_score"] > hist[0]["parent_score"])
