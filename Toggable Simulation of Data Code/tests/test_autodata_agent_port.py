"""The website agents' stdlib port (agents/shared/autodata_core) against the NumPy harness and fptsim,
plus the manifest rules of agents/instructions.md."""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autodata.forward import crb_relative_error as np_crb
from autodata.forward import first_passage_jac as np_jac
from autodata.forward import jacobian_condition as np_cond
from autodata.instance import build_instance as np_build
from fptsim.config import ConfigError, validate_config
from fptsim.graphs import Graph

AGENTS = Path(__file__).resolve().parents[2] / "agents"
sys.path.insert(0, str(AGENTS / "shared"))

from autodata_core import forward as pf  # noqa: E402
from autodata_core import graphs as pg  # noqa: E402
from autodata_core.config import ConfigError as PortConfigError  # noqa: E402
from autodata_core.config import build, load_scenarios  # noqa: E402
from autodata_core.instance import build_instance, check_spec, observe  # noqa: E402
from autodata_core.solvers import relative_error, solve  # noqa: E402

LIMITS = {"max_n": 30, "max_k": 12, "max_T": 80, "min_N": 100, "max_N": 10 ** 7}


def spec(family, B, q=None, T=25):
    g = {"families": family, "weights": {"dist": "uniform", "low": 0.3, "high": 1.0},
         "mu": {"mode": "slack", "a": 0.0, "b": 1.5}, "B": B}
    if q:
        g["q"] = q
    return {"graphs": g, "simulation": {"T": T, "N": 50000}}


CASES = [spec({"grid": {"rows": 3, "cols": 4}}, {"mode": "boundary", "max_k": 5}),
         spec({"erdos_renyi": {"n": 10, "p": 0.4}}, {"mode": "random_k", "k": 3},
              {"mode": "sparse", "fraction": 0.4, "low": 0.2, "high": 0.9}),
         spec({"random_tree": {"n": 9}}, {"mode": "leaves"})]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_forward_jacobian_crb_match_numpy(case):
    inst = np_build(CASES[case], 5, LIMITS)
    p = inst.problem
    P = pf.InverseProblem(p.W.tolist(), inst.mu_true.tolist(), p.B.tolist(), p.q.tolist(), p.dt, p.T)
    mu = inst.mu_true.tolist()
    r, cols = pf.first_passage_jac(P, mu)
    r0, J = np_jac(p, inst.mu_true)
    assert np.abs(np.array(r) - r0.reshape(-1)).max() < 1e-14
    assert np.abs(np.array(cols).T - J.reshape(-1, p.m)).max() < 1e-13
    assert np.abs(np.array(pf.first_passage(P, mu)) - inst.r_exact.reshape(-1)).max() < 1e-14
    assert pf.crb_relative_error(P, mu, 50000) == pytest.approx(np_crb(p, inst.mu_true, 50000), rel=1e-8)
    assert pf.jacobian_condition(P, mu) == pytest.approx(np_cond(p, inst.mu_true), rel=1e-5)


def test_bundled_scenarios_validate_in_fptsim_and_the_port():
    for s in load_scenarios():
        sp = s["base_spec"]
        validate_config({"num_graphs": 1, "graphs": sp["graphs"], "simulation": sp["simulation"]})
        check_spec(sp, LIMITS)
        for fam in s.get("families") or []:
            validate_config({"num_graphs": 1, "graphs": {**sp["graphs"], "families": fam},
                             "simulation": sp["simulation"]})


@pytest.mark.parametrize("bad", [{"mu": {"mode": "magic"}}, {"weights": {"dist": "uniform", "low": 0.5}},
                                 {"B": {"mode": "random_k"}}, {"families": {"grid": {"rows": 3}}},
                                 {"extra_key": 1}, {"q": {"mode": "sparse", "fraction": 2, "low": 0, "high": 1}}])
def test_invalid_specs_are_rejected_by_both(bad):
    sp = spec({"grid": {"rows": 3, "cols": 3}}, {"mode": "random_k", "k": 2})
    sp["graphs"].update(bad)
    with pytest.raises(ConfigError):
        validate_config({"num_graphs": 1, "graphs": sp["graphs"], "simulation": sp["simulation"]})
    with pytest.raises(pg.SpecError):
        check_spec(sp, LIMITS)


FAMILIES = [{"grid": {"rows": [2, 4], "cols": [2, 4]}}, {"grid_holes": {"rows": 4, "cols": 4, "hole_frac": 0.25}},
            {"pattern": {"patterns": [["XXXX", "XooX", "XXXX"]]}}, {"ladder": {"length": 5}},
            {"path": {"n": 6}}, {"cycle": {"n": 7}}, {"star": {"n": 6}}, {"complete": {"n": 5}},
            {"random_tree": {"n": [5, 12]}}, {"erdos_renyi": {"n": 10, "p": 0.4}},
            {"random_geometric": {"n": 12, "radius": 0.5}}, {"watts_strogatz": {"n": 12, "k": 4, "beta": 0.3}},
            {"barabasi_albert": {"n": 12, "m": 2}}, {"random_regular": {"n": 9, "d": 3}}]


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: next(iter(f)))
def test_port_generates_valid_fptsim_graphs(fam):
    B = {"mode": "random_k", "k": [1, 3]}
    for seed in range(5):
        g = pg.generate(pg.validate_graphs({**spec(fam, B)["graphs"],
                                            "q": {"mode": "uniform", "low": 0.0, "high": 0.4}}), seed)
        G = Graph(W=np.array(g["W"]), mu=np.array(g["mu"]), B=np.array(g["B"]), q=np.array(g["q"]))
        G.validate(check_assumption=True)            # connected, symmetric, mu >= sum_y W, valid B
        rate = (G.W.sum(1) / G.mu + G.q).max()
        assert g["dt"] == pytest.approx(min(1.0, 1.0 / rate))
    assert pg.generate(pg.validate_graphs(spec(fam, B)["graphs"]), 3) == \
        pg.generate(pg.validate_graphs(spec(fam, B)["graphs"]), 3)


def test_port_solvers_and_noise():
    inst = build_instance(CASES[0], 11, LIMITS)
    P, rng = inst.problem, random.Random(0)
    mu = solve(P, observe(inst, "exact", 0), {"solver": "lsq"}, rng)
    assert relative_error(P, mu, inst.mu_true) < 1e-8
    obs = observe(inst, "multinomial", 4)
    k2 = P.k ** 2
    for pr in range(k2):
        assert sum(obs["counts"][t * k2 + pr] for t in range(P.T)) <= inst.N
    z = [(a - b) / max((b * (1 - b) / inst.N) ** 0.5, 1e-12) for a, b in zip(obs["r"], inst.r_exact) if b > 0.01]
    assert sum(abs(v) for v in z) / len(z) < 1.5     # ~0.8 for an unbiased sampler


def test_input_config_is_strict():
    assert build({})["solvers"]["weak"]["prior_weight"] == 0.1
    with pytest.raises(PortConfigError, match="gates"):
        build({"gates": {"weak_avg": 0.5}})
    with pytest.raises(PortConfigError, match="scenario"):
        build({"scenario_ids": ["nope"]})
    assert [s["id"] for s in build({"skip_scenarios": ["paper8_ladder"]})["scenarios"]][0] == "paper9_grid"


# ---------------------------------------------------------------- manifests (agents/instructions.md, section 2)
FIELDS = {"version", "name", "description", "image", "command", "inputs_schema", "connection", "network",
          "providers", "tier", "timeout_seconds", "cpu", "memory_mb", "datasets", "swarm"}


@pytest.mark.parametrize("slug", ["autodata-inverse", "autodata-inverse-llm"])
def test_manifest_follows_the_platform_contract(slug):
    path = AGENTS / slug / "agent.json"
    raw = path.read_text(encoding="utf-8")
    assert len(raw.encode()) <= 16 * 1024
    m = json.loads(raw)
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", slug)
    assert set(m) <= FIELDS and "id" not in m
    assert m["version"] == 1 and 1 <= len(m["name"]) <= 100 and len(m["description"]) <= 2000
    assert m["image"] == "research-agent-runtime:v1"
    assert m["command"] == ["python", "run.py"] and (AGENTS / slug / m["command"][1]).exists()
    assert m["inputs_schema"]["type"] == "object" and "$ref" not in json.dumps(m["inputs_schema"])
    assert len(json.dumps(m["inputs_schema"]).encode()) <= 16 * 1024
    assert isinstance(m["timeout_seconds"], int) and 1 <= m["timeout_seconds"] <= 86400
    assert 0.1 <= m["cpu"] <= 4 and 128 <= m["memory_mb"] <= 4096
    if m.get("network") == "provider":
        assert m["connection"] == "required"
    assert (AGENTS / slug / "README.md").exists()
    run_py = (AGENTS / slug / "run.py").read_text(encoding="utf-8")
    assert f"TIMEOUT_S = {m['timeout_seconds']}" in run_py


def test_agent_runs_with_the_standard_library_only(tmp_path):
    (tmp_path / "in.json").write_text(json.dumps({"scenario_ids": ["tree_leaves"], "max_rounds": 4}))
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.run([sys.executable, "-S", "-B", "run.py"], cwd=AGENTS / "autodata-inverse",
                          env={"AGENT_INPUT_PATH": str(tmp_path / "in.json"), "AGENT_OUTPUT_DIR": str(out),
                               "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert '{"progress": 1.0}' in proc.stdout
    result = json.loads((out / "result.json").read_text())
    assert result["summary"]["scenarios"] == 1
    lines = (out / "dataset.jsonl").read_text().splitlines()
    assert len(lines) == result["summary"]["items"]
    if lines:
        item = json.loads(lines[0])
        a = item["arrays"]
        assert len(a["counts"]) == 3 and len(a["r_exact"]) == a["T"] * len(a["B"]) ** 2
    bad = subprocess.run([sys.executable, "-S", "-B", "run.py"], cwd=AGENTS / "autodata-inverse-llm",
                         env={"AGENT_INPUT_PATH": str(tmp_path / "in.json"), "AGENT_OUTPUT_DIR": str(out),
                              "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "model connection" in bad.stderr      # no provider -> no call, clear failure
