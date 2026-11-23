"""Test 8: generators give connected graphs, mu >= sum_y W, valid dt, nonempty B, and are reproducible."""
import numpy as np
import pytest

from fptsim.config import ConfigError
from fptsim.generators import GenerationError, generate_graph, graph_record
from fptsim.graphs import is_connected
from fptsim.kernel import build_kernel

FAMILIES = {
    "grid": {"rows": [2, 5], "cols": [2, 5]},
    "grid_holes": {"rows": [3, 6], "cols": [3, 6], "hole_frac": [0.0, 0.3]},
    "pattern": {"patterns": [["X o o o X", "X X X X X", "X X X X X", "o X o X o"], ["XX", "XX"]]},
    "ladder": {"length": [2, 6]},
    "path": {"n": [2, 10]},
    "cycle": {"n": [3, 10]},
    "star": {"n": [2, 10]},
    "complete": {"n": [2, 8]},
    "random_tree": {"n": [2, 15]},
    "erdos_renyi": {"n": [5, 20], "p": [0.1, 0.5]},
    "random_geometric": {"n": [5, 20], "radius": [0.3, 0.6]},
    "watts_strogatz": {"n": [6, 20], "k": [2, 4], "beta": [0.0, 0.5]},
    "barabasi_albert": {"n": [5, 20], "m": [1, 3]},
    "random_regular": {"n": [5, 15], "d": [2, 4]},
}


def _check(g, gcfg):
    g.validate()
    assert is_connected(g.W)
    np.testing.assert_array_equal(g.W, g.W.T)
    assert np.all(np.diag(g.W) == 0)
    assert np.all(g.mu >= g.W.sum(axis=1) * (1 - 1e-12))       # on_violation: fix
    assert g.B.size >= 1 and len(set(g.B.tolist())) == g.B.size
    kern = build_kernel(g.W, g.mu, g.q, gcfg["dt"])
    assert 0 < kern.dt <= 1 and kern.dt * kern.rate.max() <= 1 + 1e-12
    np.testing.assert_allclose(kern.P.sum(axis=1), 1, rtol=0, atol=1e-12)
    assert np.all(kern.P >= 0)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family(make_config, family):
    cfg = make_config(graphs={"families": {family: {"weight": 1, **FAMILIES[family]}},
                              "weights": {"dist": "lognormal", "mean": 0.0, "sigma": 0.6},
                              "mu": {"mode": "deg", "on_violation": "fix"},
                              "B": {"mode": "random_k", "k": [1, 4]},
                              "q": {"mode": "uniform", "low": 0.0, "high": 0.5}})
    gcfg = cfg["graphs"]
    for gid in range(8):
        g = generate_graph(gcfg, cfg["master_seed"], gid)
        assert g.family == family
        _check(g, gcfg)
        again = generate_graph(gcfg, cfg["master_seed"], gid)          # same seed, same graph
        for name in ("W", "mu", "B", "q"):
            np.testing.assert_array_equal(getattr(g, name), getattr(again, name))
        assert g.params == again.params
        row = graph_record(g, gcfg["dt"])
        assert row["n"] == g.n and row["k"] == g.B.size and np.isfinite(row["spectral_gap"])


def test_different_seeds_differ(make_config):
    cfg = make_config(graphs={"families": {"erdos_renyi": {"weight": 1, "n": 12, "p": 0.4}},
                              "B": {"mode": "random_k", "k": 3}})
    a = generate_graph(cfg["graphs"], 1, 0)
    b = generate_graph(cfg["graphs"], 2, 0)
    c = generate_graph(cfg["graphs"], 1, 1)
    assert not np.array_equal(a.W, b.W) and not np.array_equal(a.W, c.W)


@pytest.mark.parametrize("B,check", [
    ({"mode": "boundary"}, lambda g: np.all(g.boundary[g.B])),
    ({"mode": "boundary", "max_k": 3}, lambda g: g.B.size <= 3),
    ({"mode": "low_degree", "k": 2}, lambda g: g.degrees[g.B].max() <= np.sort(g.degrees)[1]),
    ({"mode": "high_degree", "k": 1}, lambda g: g.degrees[g.B[0]] == g.degrees.max()),
    ({"mode": "leaves"}, lambda g: g.B.size >= 1),
    ({"mode": "random_k", "frac": 0.5}, lambda g: g.B.size == max(1, round(0.5 * g.n))),
    ({"mode": "explicit", "vertices": [0, 3]}, lambda g: g.B.tolist() == [0, 3]),
])
def test_B_modes(make_config, B, check):
    cfg = make_config(graphs={"families": {"grid": {"weight": 1, "rows": [2, 4], "cols": [2, 4]}}, "B": B})
    for gid in range(6):
        g = generate_graph(cfg["graphs"], 0, gid)
        _check(g, cfg["graphs"])
        assert check(g)


@pytest.mark.parametrize("q,check", [
    ({"mode": "zero"}, lambda g: np.all(g.q == 0)),
    ({"mode": "constant", "c": 0.3}, lambda g: np.allclose(g.q, 0.3)),
    ({"mode": "sparse", "fraction": 0.25, "low": 0.1, "high": 0.2},
     lambda g: np.count_nonzero(g.q) == round(0.25 * g.n)),
    ({"mode": "localized", "center": 0, "decay": 1.0, "amplitude": 0.5},
     lambda g: g.q[0] == 0.5 and g.q.max() == 0.5),
    ({"mode": "uniform", "low": 0.1, "high": 0.2, "on_B_only": True},
     lambda g: np.all(np.delete(g.q, g.B) == 0) and np.all(g.q[g.B] >= 0.1)),
    ({"mode": "uniform", "low": 0.1, "high": 0.2, "off_B_only": True}, lambda g: np.all(g.q[g.B] == 0)),
    ({"mode": "explicit", "values": [0, 0.1, 0, 0.2, 0, 0]}, lambda g: g.q[3] == 0.2),
])
def test_q_modes(make_config, q, check):
    cfg = make_config(graphs={"families": {"grid": {"weight": 1, "rows": 2, "cols": 3}}, "q": q})
    for gid in range(4):
        g = generate_graph(cfg["graphs"], 0, gid)
        _check(g, cfg["graphs"])
        assert check(g)


def test_mu_violation_modes(make_config):
    base = {"families": {"complete": {"weight": 1, "n": 5}}, "weights": {"dist": "constant", "value": 1.0}}
    # constant mu = 1 < weighted degree 4: fix raises mu to the row sum
    cfg = make_config(graphs={**base, "mu": {"mode": "constant", "value": 1.0, "on_violation": "fix"}})
    g = generate_graph(cfg["graphs"], 0, 0)
    np.testing.assert_allclose(g.mu, 4.0)
    # skip with a deterministic violation never succeeds
    cfg = make_config(graphs={**base, "mu": {"mode": "constant", "value": 1.0, "on_violation": "skip"},
                              "max_retries": 3})
    with pytest.raises(GenerationError):
        generate_graph(cfg["graphs"], 0, 0)
    # resample a random mu until it satisfies the assumption
    cfg = make_config(graphs={**base, "mu": {"mode": "uniform", "low": 1.0, "high": 30.0,
                                              "on_violation": "resample"}})
    g = generate_graph(cfg["graphs"], 0, 0)
    assert np.all(g.mu >= 4.0)


def test_slack_mu_and_explicit_dt(make_config):
    cfg = make_config(graphs={"families": {"cycle": {"weight": 1, "n": [4, 8]}},
                              "mu": {"mode": "slack", "a": 0.1, "b": 0.3}, "dt": 0.5,
                              "q": {"mode": "uniform", "low": 0.0, "high": 0.4}})
    for gid in range(5):
        g = generate_graph(cfg["graphs"], 0, gid)
        ratio = g.mu / g.W.sum(axis=1)
        assert np.all((ratio >= 1.1 - 1e-12) & (ratio <= 1.3 + 1e-12))
        assert build_kernel(g.W, g.mu, g.q, 0.5).dt == 0.5


def test_presets(make_config):
    cfg = make_config(graphs={"families": None, "weights": None, "mu": None, "B": None,
                              "presets": ["paper8", "paper9"]})
    assert generate_graph(cfg["graphs"], 0, 0).family == "paper8"
    assert generate_graph(cfg["graphs"], 0, 3).family == "paper9"


def test_boundary_requires_lattice(make_config):
    with pytest.raises(ConfigError, match="boundary"):
        make_config(graphs={"families": {"erdos_renyi": {"weight": 1, "n": 8, "p": 0.5}},
                            "B": {"mode": "boundary"}})
