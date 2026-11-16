"""Test 5: the Monte Carlo is unbiased: mean FRNE ~ expected RMS FRNE, Wilson coverage ~ 95%."""
import numpy as np

from fptsim.generators import generate_graph
from fptsim.validation import simulate_graph

R, N, T = 20, 4_000, 12


def test_frne_matches_expected_and_coverage(make_config):
    cfg = make_config(graphs={
        "families": {"erdos_renyi": {"weight": 1, "n": [6, 10], "p": [0.3, 0.6]},
                     "cycle": {"weight": 1, "n": [5, 8]}},
        "B": {"mode": "random_k", "k": 3},
        "q": {"mode": "sparse", "fraction": 0.3, "low": 0.0, "high": 0.4}})
    ratios, coverage, coverage_all = [], [], []
    for gid in range(3):
        g = generate_graph(cfg["graphs"], cfg["master_seed"], gid)
        res = simulate_graph(g, T=T, N=N, replicates=R, device="cpu", seed=11, graph_id=gid)
        frnes = [rep["metrics"]["frne_percent"] for rep in res["replicates"]]
        expected = res["replicates"][0]["metrics"]["expected_rms_frne_percent"]
        ratios.append(np.mean(frnes) / expected)
        coverage += [rep["metrics"]["ci_coverage_np5"] for rep in res["replicates"]]
        coverage_all += [rep["metrics"]["ci_coverage"] for rep in res["replicates"]]
        # independent replicates: no two identical count tensors
        assert len({rep["hits"].tobytes() for rep in res["replicates"]}) == R
    for ratio in ratios:
        assert 0.8 <= ratio <= 1.2, ratios
    assert 0.92 <= np.nanmean(coverage) <= 0.98, np.nanmean(coverage)
    assert np.mean(coverage_all) >= 0.92
