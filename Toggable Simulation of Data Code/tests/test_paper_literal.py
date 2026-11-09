"""Test 2: paper8 / paper9 exact r equals the notebook's literal recursion to 1e-14 (T = 17)."""
from functools import lru_cache

import numpy as np
import pytest

from fptsim.exact import exact_single
from fptsim.graphs import PRESETS
from fptsim.kernel import build_kernel
from fptsim.validation import literal_first_passage

T = 17


# ---- copied from "First Passage Time Simulation (executed on americano).ipynb" (cells 4-5) ----
def transition_matrix(W, mu):
    '''
    W size: n x n
    mu size: n

    Function returns P (n x n)
    '''
    import torch
    P = W / mu[:, None]
    stay = 1.0 - P.sum(dim=1)
    if (stay < -1e-12).any():
        raise ValueError("Assumption violated: mu_x < sum_y w_xy for some x")
    return P + torch.diag(stay.clamp_min(0.0))


def exact_first_passage_literal(P, B, t_max):
    '''
    r(1, x, y) = p_xy
    r(t, x, y) = sum_{z != y} p_xz * r(t-1, z, y),   t >= 2

    Returns r with r[t-1, i, j] = r(t, B[i], B[j]), shape (t_max, |B|, |B|).
    '''
    import torch
    P = P.cpu().double()
    n = P.shape[0]

    @lru_cache(maxsize=None)
    def r(t, x, y):
        if t == 1:
            return P[x, y].item()
        return sum(P[x, z].item() * r(t - 1, z, y)
                   for z in range(n) if z != y)

    return torch.tensor([[[r(t, int(x), int(y)) for y in B] for x in B]
                         for t in range(1, t_max + 1)], dtype=torch.float64)
# ---- end of copied code ----


def _notebook_inputs(name):
    """W, mu and B exactly as the notebook builds them (experiments 1 and 2)."""
    rows, cols, w, B = {"paper8": (2, 4, 0.25, [1, 2, 3]), "paper9": (3, 3, 1.0, [0, 1, 2])}[name]
    n = rows * cols
    W = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                W[i, i + 1] = W[i + 1, i] = w
            if r + 1 < rows:
                W[i, i + cols] = W[i + cols, i] = w
    deg = (W > 0).sum(axis=1).astype(float)
    mu = np.ones(n) if name == "paper8" else deg          # "constant" 1.0 / "deg"
    return W, mu, np.array(B)


@pytest.mark.parametrize("name", ["paper8", "paper9"])
def test_presets_match_notebook(name):
    W, mu, B = _notebook_inputs(name)
    g = PRESETS[name]()
    np.testing.assert_array_equal(g.W, W)
    np.testing.assert_array_equal(g.mu, mu)
    np.testing.assert_array_equal(g.B, B)


@pytest.mark.parametrize("name", ["paper8", "paper9"])
def test_exact_equals_notebook_literal(name):
    torch = pytest.importorskip("torch")
    W, mu, B = _notebook_inputs(name)
    P = transition_matrix(torch.as_tensor(W), torch.as_tensor(mu))
    literal = exact_first_passage_literal(P, torch.as_tensor(B), T).numpy()
    g = PRESETS[name]()
    kern = build_kernel(g.W, g.mu, g.q)
    assert kern.dt == 1.0                                  # auto h = 1 reproduces the paper's P
    np.testing.assert_allclose(kern.K, P.numpy(), rtol=0, atol=1e-15)
    for device in (None, "cpu"):
        r = exact_single(kern, g.B, T, device=device)["r"]
        assert np.abs(r - literal).max() < 1e-14


@pytest.mark.parametrize("name", ["paper8", "paper9"])
def test_exact_equals_numpy_literal(name):
    g = PRESETS[name]()
    kern = build_kernel(g.W, g.mu, g.q)
    r = exact_single(kern, g.B, T)["r"]
    assert np.abs(r - literal_first_passage(kern.K, g.B, T)).max() < 1e-14
