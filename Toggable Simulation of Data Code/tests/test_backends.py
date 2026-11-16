"""Test 7: torch and numpy backends agree statistically; torch packing does not change results; GPU."""
import numpy as np
import pytest

from conftest import cuda_available, torch_available
from fptsim.graphs import PRESETS
from fptsim.kernel import build_kernel, transition_table
from fptsim.mc_common import MCItem, chunk_plan, make_engine, mc_seed
from fptsim.validation import simulate_graph

needs_torch = pytest.mark.skipif(not torch_available(), reason="torch not installed")
R, N, T = 10, 5_000, 17


def _graph():
    return PRESETS["paper9"]().with_q([0, 0, 0, 0, 0.4, 0, 0, 0, 0])


@needs_torch
def test_backends_statistically_equivalent():
    g = _graph()
    a = simulate_graph(g, T, N, R, backend="numpy", device="cpu", seed=3)
    b = simulate_graph(g, T, N, R, backend="torch", device="cpu", seed=3)
    np.testing.assert_array_equal(a["exact"]["r"], b["exact"]["r"])     # backend-independent exact r
    fa = np.array([rep["metrics"]["frne_percent"] for rep in a["replicates"]])
    fb = np.array([rep["metrics"]["frne_percent"] for rep in b["replicates"]])
    expected = a["replicates"][0]["metrics"]["expected_rms_frne_percent"]
    spread = np.sqrt(fa.var(ddof=1) / R + fb.var(ddof=1) / R)
    assert abs(fa.mean() - fb.mean()) < 4 * spread + 0.05 * expected
    for f in (fa, fb):
        assert 0.75 < f.mean() / expected < 1.25
    # the two backends use different streams: same distribution, different samples
    assert not np.array_equal(a["replicates"][0]["hits"], b["replicates"][0]["hits"])


def _items(g, kern, reps=3, n=3_000, chunk=1_000):
    nbr, cdf = transition_table(kern.P)
    cps = np.array([100, 1_000, n], dtype=np.int64)
    return [MCItem(key=(0, 0, r, c), nbr=nbr, cdf=cdf, B=g.B, T=T, m=m, start=start, checkpoints=cps,
                   seed=mc_seed(5, 0, 0, r, c))
            for r in range(reps) for c, (start, m) in enumerate(chunk_plan(n, chunk))]


@needs_torch
def test_torch_packing_does_not_change_results():
    from fptsim.mc_torch import TorchMC
    g = _graph()
    kern = build_kernel(g.W, g.mu, g.q)
    items = _items(g, kern)
    one_launch = {i: c for i, c, _ in TorchMC("cpu", cpu_batch_walks=10 ** 9).run_items(items)}
    tiny = {i: c for i, c, _ in TorchMC("cpu", cpu_batch_walks=1).run_items(items)}   # one item per launch
    for i in one_launch:
        np.testing.assert_array_equal(one_launch[i], tiny[i])


@needs_torch
def test_mixed_graph_sizes_in_one_launch():
    from fptsim.mc_torch import TorchMC
    small, big = PRESETS["paper8"](), _graph()
    items = []
    for gid, g in enumerate((small, big)):
        kern = build_kernel(g.W, g.mu, g.q)
        items += [MCItem(key=(gid, *it.key[1:]), nbr=it.nbr, cdf=it.cdf, B=it.B, T=it.T, m=it.m,
                         start=it.start, checkpoints=it.checkpoints, seed=it.seed + gid)
                  for it in _items(g, kern, reps=1)]
    together = {i: c for i, c, _ in TorchMC("cpu", cpu_batch_walks=10 ** 9).run_items(items)}
    alone = {i: c for i, c, _ in TorchMC("cpu", cpu_batch_walks=1).run_items(items)}
    for i in together:
        np.testing.assert_array_equal(together[i], alone[i])


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
def test_cuda_backend():
    g = _graph()
    res = simulate_graph(g, T, 20_000, 5, backend="torch", device="cuda", seed=9)
    frnes = np.array([rep["metrics"]["frne_percent"] for rep in res["replicates"]])
    expected = res["replicates"][0]["metrics"]["expected_rms_frne_percent"]
    assert 0.7 < frnes.mean() / expected < 1.3
    again = simulate_graph(g, T, 20_000, 1, backend="torch", device="cuda", seed=9)
    np.testing.assert_array_equal(res["replicates"][0]["hits"], again["replicates"][0]["hits"])
    cpu = simulate_graph(g, T, 1_000, 1, backend="numpy", device="cpu", seed=9)
    np.testing.assert_allclose(res["exact"]["r"], cpu["exact"]["r"], rtol=0, atol=1e-15)
