"""Test 1: hand-computable 2- and 3-state chains, including first returns and holding steps."""
import numpy as np
import pytest

from conftest import torch_available
from fptsim.exact import exact_single
from fptsim.kernel import build_kernel
from fptsim.validation import simulate_graph
from fptsim.graphs import Graph

DEVICES = [None] + (["cpu"] if torch_available() else [])
T = 12


def two_state(q=(0.0, 0.0)):
    # w = 0.3, mu = (1, 0.6): s = (0.3, 0.5); with q = 0 and h = 1, K = [[0.7, 0.3], [0.5, 0.5]]
    W = np.array([[0.0, 0.3], [0.3, 0.0]])
    return Graph(W=W, mu=np.array([1.0, 0.6]), B=np.array([0, 1]), q=np.asarray(q, float))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [(0.0, 0.0), (0.2, 0.1)])
def test_two_state_closed_form(device, q):
    g = two_state(q)
    kern = build_kernel(g.W, g.mu, g.q)
    assert kern.dt == 1.0
    K, kill = kern.K, kern.kill
    a, b = K[0, 1], K[1, 0]
    k00, k11 = K[0, 0], K[1, 1]
    np.testing.assert_allclose([k00, a, b, k11], [1 - 0.3 - q[0], 0.3, 0.5, 1 - 0.5 - q[1]], atol=1e-15)
    res = exact_single(kern, g.B, T, device=device)
    r = res["r"]
    t = np.arange(1, T + 1)
    np.testing.assert_allclose(r[:, 0, 1], k00 ** (t - 1) * a, atol=1e-15)
    np.testing.assert_allclose(r[:, 1, 0], k11 ** (t - 1) * b, atol=1e-15)
    # first return: a holding step at t = 1 counts as a return
    ret0 = np.where(t == 1, k00, a * k11 ** np.maximum(t - 2, 0) * b)
    ret1 = np.where(t == 1, k11, b * k00 ** np.maximum(t - 2, 0) * a)
    np.testing.assert_allclose(r[:, 0, 0], ret0, atol=1e-15)
    np.testing.assert_allclose(r[:, 1, 1], ret1, atol=1e-15)
    # killed and censored masses for 0 -> 1 and the first return 0 -> 0
    np.testing.assert_allclose(res["killed"][0, 1], kill[0] * (1 - k00 ** T) / (1 - k00), atol=1e-15)
    np.testing.assert_allclose(res["censored"][0, 1], k00 ** T, atol=1e-15)
    geo = (1 - k11 ** (T - 1)) / (1 - k11)
    np.testing.assert_allclose(res["killed"][0, 0], kill[0] + a * kill[1] * geo, atol=1e-15)
    np.testing.assert_allclose(res["censored"][0, 0], a * k11 ** (T - 1), atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
def test_three_state_complete_graph(device):
    # K3 with w = 1, mu = 4: p = 1/4 to each neighbour, hold 1/2
    W = np.ones((3, 3)) - np.eye(3)
    kern = build_kernel(W, np.full(3, 4.0), np.zeros(3))
    r = exact_single(kern, np.array([0, 1, 2]), T, device=device)["r"]
    t = np.arange(1, T + 1)
    p = 0.25
    for x in range(3):
        for y in range(3):
            if x == y:
                expected = np.where(t == 1, 0.5, 2 * p * (1 - p) ** np.maximum(t - 2, 0) * p)
            else:
                expected = (1 - p) ** (t - 1) * p
            np.testing.assert_allclose(r[:, x, y], expected, atol=1e-15)


def test_three_state_path_subset():
    # path 0 - 1 - 2, B = {0, 2}: 0 -> 2 must pass through 1, so r(1, 0, 2) = 0
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    kern = build_kernel(W, np.array([2.0, 2.0, 2.0]), np.zeros(3))
    r = exact_single(kern, np.array([0, 2]), T)["r"]
    K = kern.K
    assert r[0, 0, 1] == 0.0
    assert r[1, 0, 1] == pytest.approx(K[0, 1] * K[1, 2], abs=1e-16)
    assert r[0, 0, 0] == pytest.approx(K[0, 0], abs=1e-16)          # holding step
    assert r[1, 0, 0] == pytest.approx(K[0, 1] * K[1, 0], abs=1e-16)


def test_two_state_monte_carlo_matches():
    res = simulate_graph(two_state((0.2, 0.1)), T, N=20_000, replicates=1, chunk_walks=7_000)
    rep = res["replicates"][0]
    assert np.all(rep["hits"].sum(axis=0) + rep["killed"] + rep["censored"] == 20_000)
    assert rep["metrics"]["frne_percent"] < 4 * rep["metrics"]["expected_rms_frne_percent"]
