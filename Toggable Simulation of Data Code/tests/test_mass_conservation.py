"""Test 4: hits + killed + censored = 1 (exact) and counts sum to N (Monte Carlo), with q > 0."""
import numpy as np
import pytest

from conftest import torch_available
from fptsim.exact import exact_batch
from fptsim.generators import generate_graph
from fptsim.graphs import PRESETS
from fptsim.kernel import build_kernel
from fptsim.validation import simulate_graph

BACKENDS = ["numpy"] + (["torch"] if torch_available() else [])


def _random_graphs(make_config, count=12):
    cfg = make_config(graphs={
        "families": {"erdos_renyi": {"weight": 1, "n": [5, 14], "p": [0.3, 0.6]},
                     "random_tree": {"weight": 1, "n": [4, 12]},
                     "grid_holes": {"weight": 1, "rows": [3, 4], "cols": [3, 4], "hole_frac": [0, 0.3]}},
        "q": {"mode": "uniform", "low": 0.05, "high": 0.6}})
    return [generate_graph(cfg["graphs"], cfg["master_seed"], i) for i in range(count)]


def test_exact_mass_random_graphs_batched(make_config):
    graphs = _random_graphs(make_config)
    kernels = [build_kernel(g.W, g.mu, g.q) for g in graphs]
    for device in [None] + (["cpu"] if torch_available() else []):
        res = exact_batch(kernels, [g.B for g in graphs], 25, device=device)
        for g, e in zip(graphs, res):
            mass = e["r"].sum(axis=0) + e["killed"] + e["censored"]
            np.testing.assert_allclose(mass, 1.0, rtol=0, atol=1e-12)
            assert e["killed"].min() > 0                  # q > 0 everywhere
            assert e["r"].shape == (25, len(g.B), len(g.B))


def test_batched_equals_single(make_config):
    graphs = _random_graphs(make_config, 6)
    kernels = [build_kernel(g.W, g.mu, g.q) for g in graphs]
    together = exact_batch(kernels, [g.B for g in graphs], 15)
    for kern, g, e in zip(kernels, graphs, together):
        alone = exact_batch([kern], [g.B], 15)[0]
        np.testing.assert_allclose(e["r"], alone["r"], rtol=0, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_counts_sum_to_N(make_config, backend):
    graphs = _random_graphs(make_config, 4) + [PRESETS["paper9"]().with_q([0, 0, 0, 0, 0.4, 0, 0, 0, 0])]
    for i, g in enumerate(graphs):
        N = 2_345                                          # not a multiple of the chunk size
        res = simulate_graph(g, T=12, N=N, replicates=2, backend=backend, device="cpu",
                             graph_id=i, chunk_walks=1_000, checkpoints=[100, 1000])
        for rep in res["replicates"]:
            total = rep["hits"].sum(axis=0) + rep["killed"] + rep["censored"]
            assert np.all(total == N)
            assert rep["killed"].sum() > 0
            # prefix counts at each checkpoint hold exactly that many walks per pair
            for b, m in enumerate(res["checkpoints"]):
                assert np.all(rep["prefix"][b].sum(axis=-1) == m)
