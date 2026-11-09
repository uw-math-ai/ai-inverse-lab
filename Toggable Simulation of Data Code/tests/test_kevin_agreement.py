"""Test 3: agreement with Kevin's forward_sim exact_first_passage / exact_outcomes (read-only import)."""
import numpy as np
import pytest

from fptsim.exact import exact_single
from fptsim.graphs import PRESETS
from fptsim.kernel import build_kernel

T = 17
CASES = [("paper8", 0), ("paper8", 0.2), ("paper8", [0, 0.3, 0, 0, 0, 0, 0.1, 0]),
         ("paper9", 0), ("paper9", 0.1), ("paper9", [0, 0, 0, 0, 0.4, 0, 0, 0, 0])]


@pytest.mark.parametrize("name,q", CASES)
def test_matches_kevin(kevin, name, q):
    model, engine = kevin["model"], kevin["engine"]
    cfg = model.parse_config({"graph": name, "q": q, "T": T, "N": 1000})
    K_k, kill_k, P_k = model.build_kernel(cfg)
    g = PRESETS[name]()
    q_vec = np.asarray(cfg.q, dtype=float)
    kern = build_kernel(g.W, g.mu, q_vec)
    assert kern.dt == pytest.approx(cfg.dt, abs=1e-15)
    np.testing.assert_allclose(kern.K, K_k, rtol=0, atol=1e-15)
    np.testing.assert_allclose(kern.kill, kill_k, rtol=0, atol=1e-15)
    np.testing.assert_allclose(kern.P, P_k, rtol=0, atol=1e-15)
    assert tuple(g.B.tolist()) == cfg.graph.observed

    r_k = engine.exact_first_passage(K_k, T, cfg.graph.observed)
    hits_k, killed_k, censored_k = engine.exact_outcomes(K_k, kill_k, T, cfg.graph.observed)
    ours = exact_single(kern, g.B, T)
    np.testing.assert_allclose(ours["r"], r_k, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ours["r"], hits_k, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ours["killed"], killed_k, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ours["censored"], censored_k, rtol=0, atol=1e-14)
