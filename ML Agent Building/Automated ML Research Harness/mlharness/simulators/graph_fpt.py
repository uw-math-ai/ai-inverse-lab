"""First-passage-time inverse problem on weighted graphs.

Conventions follow the lab's ``fptsim`` package ("Toggable Simulation of Data Code"):
  s(x) = sum_y W[x,y] / mu(x);  K[x,y] = h W[x,y] / mu(x);  K[x,x] = 1 - h (s(x) + q(x));
  killing probability h q(x);  h = min(1, 1 / max_x (s(x) + q(x)))   (dt: auto)
  r(1,x,y) = K[x,y];  r(t,x,y) = sum_{z != y} K[x,z] r(t-1,z,y)        (exact recursion)
The empirical r' = counts / N comes from N walks per ordered pair (x, y) in B x B. Sampling the
first-passage outcome multinomially from the exact distribution is statistically identical to
running the N walks and is orders of magnitude faster.

Each sample draws a random weight vector (and optional potential) on a fixed topology, so the ML
task is the inverse map  r'(t, x, y)|_{B x B, t <= T}  ->  (edge weights, potential).
"""
from __future__ import annotations

import numpy as np

from . import GENERIC_FEATURE_SETS, Simulator


def _grid(rows, cols):
    idx = lambda r, c: r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    boundary = [idx(r, c) for r in range(rows) for c in range(cols)
                if r in (0, rows - 1) or c in (0, cols - 1)]
    return rows * cols, edges, boundary


def build_topology(graph: dict):
    fam = graph.get("family", "grid")
    if fam == "grid":
        return _grid(int(graph.get("rows", 3)), int(graph.get("cols", 3)))
    if fam == "path":
        n = int(graph.get("n", 6))
        return n, [(i, i + 1) for i in range(n - 1)], [0, n - 1]
    if fam == "cycle":
        n = int(graph.get("n", 8))
        return n, [(i, (i + 1) % n) for i in range(n)], list(range(0, n, 2))
    if fam == "complete":
        n = int(graph.get("n", 5))
        return n, [(i, j) for i in range(n) for j in range(i + 1, n)], list(range(n))
    if fam == "custom":
        n = int(graph["n"])
        edges = [tuple(map(int, e)) for e in graph["edges"]]
        return n, edges, list(graph.get("boundary", range(n)))
    raise ValueError(f"unknown graph family {fam!r}")


def exact_first_passage(K: np.ndarray, B: np.ndarray, T: int) -> np.ndarray:
    """Batched exact r(t, x, y) for x, y in B. K: (S, n, n). Returns (S, T, |B|, |B|)."""
    b = len(B)
    KB = K[:, :, B]                         # (S, n, b): K[x, y_j]
    R = KB.copy()                           # r(1, x, y_j)
    out = [R[:, B, :]]
    cols = np.arange(b)
    for _ in range(1, T):
        diag = R[:, B, cols]                # r(t-1, y_j, y_j): mass that would pass through y_j
        R = K @ R - KB * diag[:, None, :]   # exclude z = y_j from the sum
        out.append(R[:, B, :])
    return np.stack(out, axis=1)


class GraphFPTSimulator(Simulator):
    name = "graph_fpt"
    task = "regression"

    def __init__(self, graph: dict | None = None, observed="boundary", T: int = 9,
                 walks_per_pair: int = 10000, weights: dict | None = None,
                 potential: dict | None = None, target: str = "weights"):
        graph = graph or {"family": "grid", "rows": 3, "cols": 3}
        weights = weights or {"low": 0.2, "high": 1.0}
        potential = potential or {"mode": "none"}
        super().__init__(graph=graph, observed=observed, T=T, walks_per_pair=walks_per_pair,
                         weights=weights, potential=potential, target=target)
        self.n, self.edges, boundary = build_topology(graph)
        if observed == "boundary":
            B = boundary
        elif observed == "all":
            B = list(range(self.n))
        else:
            B = [int(v) for v in observed]
        self.B = np.array(sorted(set(B)), dtype=int)
        self.T = int(T)
        if target not in ("weights", "potential", "both"):
            raise ValueError("target must be weights | potential | both")
        if target in ("potential", "both") and potential.get("mode", "none") == "none":
            raise ValueError("target includes the potential but potential.mode is 'none'")

    @property
    def feature_shape(self):
        b = len(self.B)
        return (self.T, b, b)

    @property
    def feature_sets(self):
        return GENERIC_FEATURE_SETS + ["fpt_summary", "fpt_early", "log+fpt_summary", "raw+fpt_summary"]

    @property
    def suggested_feature_sets(self):
        return ["log", "fpt_summary", "log+fpt_summary", "fpt_early"]

    def describe(self):
        p = self.params
        return (f"Random walk on a fixed {p['graph']} graph with {self.n} vertices and {len(self.edges)} "
                f"edges; edge weights ~ U[{p['weights']['low']}, {p['weights']['high']}], mu = degree, "
                f"potential {p['potential']}. Observed set B = {self.B.tolist()} (|B|={len(self.B)}). "
                f"Features: empirical first-passage probabilities r'(t, x, y) for t = 1..{self.T} and "
                f"(x, y) in B x B from N = {p['walks_per_pair']} walks per pair, flattened in (t, x, y) "
                f"order -> {self.T * len(self.B) ** 2} features. Target: {p['target']} "
                f"({self.target_dim} values). Inverse problem: recover graph parameters from "
                f"boundary first-passage data.")

    @property
    def target_dim(self):
        t = self.params["target"]
        return (len(self.edges) if t in ("weights", "both") else 0) + \
               (self.n if t in ("potential", "both") else 0)

    def sample_parameters(self, S: int, rng):
        lo, hi = self.params["weights"]["low"], self.params["weights"]["high"]
        w = rng.uniform(lo, hi, size=(S, len(self.edges)))
        pot = self.params["potential"]
        mode = pot.get("mode", "none")
        q = np.zeros((S, self.n))
        if mode == "uniform":
            q = rng.uniform(pot.get("low", 0.0), pot.get("high", 0.5), size=(S, self.n))
        elif mode == "sparse":
            mask = rng.random((S, self.n)) < pot.get("fraction", 0.2)
            q = np.where(mask, rng.uniform(pot.get("low", 0.0), pot.get("high", 0.5), (S, self.n)), 0.0)
        return w, q

    def kernel(self, w, q):
        S, n = w.shape[0], self.n
        W = np.zeros((S, n, n))
        u = np.array([e[0] for e in self.edges])
        v = np.array([e[1] for e in self.edges])
        W[:, u, v] = w
        W[:, v, u] = w
        deg = np.zeros(n)
        np.add.at(deg, u, 1)
        np.add.at(deg, v, 1)
        mu = deg * max(1.0, float(self.params["weights"]["high"]))   # guarantees mu >= weighted degree
        rate = W.sum(axis=2) / mu + q                                  # (S, n)
        h = np.minimum(1.0, 1.0 / rate.max(axis=1))                    # dt auto, per sample
        K = h[:, None, None] * W / mu[None, :, None]
        K[:, np.arange(n), np.arange(n)] = 1.0 - h[:, None] * rate
        return K

    def generate(self, n_samples, seed):
        rng = np.random.default_rng(seed)
        w, q = self.sample_parameters(n_samples, rng)
        K = self.kernel(w, q)
        r = exact_first_passage(K, self.B, self.T)                    # (S, T, b, b)
        N = int(self.params["walks_per_pair"])
        if N > 0:
            p = np.moveaxis(r, 1, -1)                                  # (S, b, b, T)
            p = np.clip(p, 0.0, 1.0)
            rest = np.clip(1.0 - p.sum(axis=-1, keepdims=True), 0.0, 1.0)   # killed + censored
            pv = np.concatenate([p, rest], axis=-1)
            pv = pv / pv.sum(axis=-1, keepdims=True)
            counts = rng.multinomial(N, pv)
            r = np.moveaxis(counts[..., : self.T] / N, -1, 1)
        X = r.reshape(n_samples, -1)
        t = self.params["target"]
        y = w if t == "weights" else q if t == "potential" else np.hstack([w, q])
        return X.astype(np.float64), y.astype(np.float64)
