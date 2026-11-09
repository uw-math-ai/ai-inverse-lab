"""Weighted graph container, lattice helpers, and the two paper graphs (arXiv:2603.20710, Sec. 5)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Graph:
    """G = (X, E, W, mu) with observed set B and potential q.

    W is a dense symmetric (n, n) weight matrix with zero diagonal; x~y iff W[x, y] > 0.
    coords are 2-D drawing positions; lattice graphs use (col, -row) as in the notebook,
    so row = -coords[:, 1] and col = coords[:, 0].
    boundary marks lattice vertices with fewer than four lattice neighbours.
    """
    W: np.ndarray
    mu: np.ndarray
    B: np.ndarray
    q: np.ndarray
    family: str = "custom"
    params: dict = field(default_factory=dict)
    coords: np.ndarray | None = None
    boundary: np.ndarray | None = None
    graph_id: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.W.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        return (self.W > 0).sum(axis=1)

    @property
    def weighted_degrees(self) -> np.ndarray:
        return self.W.sum(axis=1)

    @property
    def num_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.W, 1)))

    def edges(self) -> list[tuple[int, int]]:
        rows, cols = np.nonzero(np.triu(self.W, 1))
        return [(int(a), int(b)) for a, b in zip(rows, cols)]

    def with_q(self, q) -> "Graph":
        return replace(self, q=np.asarray(q, dtype=float))

    def validate(self, check_assumption: bool = False) -> None:
        """Raise ValueError unless the arrays describe a valid connected weighted graph."""
        W, n = self.W, self.W.shape[0] if self.W.ndim == 2 else -1
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got shape {W.shape}")
        if not np.all(np.isfinite(W)) or np.any(W < 0):
            raise ValueError("W must be finite and nonnegative")
        if not np.allclose(W, W.T, rtol=0, atol=1e-12):
            raise ValueError("W must be symmetric")
        if np.any(np.diag(W) != 0):
            raise ValueError("W must have a zero diagonal")
        for name, arr in (("mu", self.mu), ("q", self.q)):
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a finite vector of length {n}")
        if np.any(self.mu <= 0):
            raise ValueError("mu must be positive")
        if np.any(self.q < 0):
            raise ValueError("q must be nonnegative")
        B = np.asarray(self.B)
        if B.ndim != 1 or B.size == 0:
            raise ValueError("B must be a nonempty list of vertices")
        if len(set(B.tolist())) != B.size or B.min() < 0 or B.max() >= n:
            raise ValueError(f"B must contain distinct vertices in 0..{n - 1}")
        if not is_connected(W):
            raise ValueError("graph is not connected")
        if check_assumption and np.any(self.mu < W.sum(axis=1) * (1 - 1e-12)):
            raise ValueError("Assumption 1(ii) violated: mu_x < sum_y W[x, y]")


def is_connected(W: np.ndarray) -> bool:
    """Depth-first search from vertex 0 over edges with W > 0."""
    n = W.shape[0]
    if n == 0:
        return False
    adj = W > 0
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    stack = [0]
    while stack:
        x = stack.pop()
        new = np.flatnonzero(adj[x] & ~seen)
        seen[new] = True
        stack.extend(int(y) for y in new)
    return bool(seen.all())


def hop_distances(W: np.ndarray, source: int) -> np.ndarray:
    """Breadth-first hop distance from `source`; unreachable vertices get inf."""
    n = W.shape[0]
    dist = np.full(n, np.inf)
    dist[source] = 0
    queue = deque([source])
    adj = W > 0
    while queue:
        x = queue.popleft()
        for y in np.flatnonzero(adj[x]):
            if dist[y] == np.inf:
                dist[y] = dist[x] + 1
                queue.append(int(y))
    return dist


# ---------------------------------------------------------------- lattices
def grid_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    """(row, col) of a rows x cols grid, row by row; vertex index = row * cols + col."""
    return [(r, c) for r in range(rows) for c in range(cols)]


def pattern_coords(pattern: list[str], node: str = "X") -> list[tuple[int, int]]:
    """Coordinates from a picture such as ["XooX", "XXXX"]; spaces are ignored, row 0 on top."""
    rows = [line.replace(" ", "") for line in pattern]
    return [(r, c) for r, line in enumerate(rows) for c, ch in enumerate(line)
            if ch.upper() == node.upper()]


def lattice_from_coords(rc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacency of horizontally/vertically adjacent lattice points, drawing coords, boundary mask."""
    rc = [(int(r), int(c)) for r, c in rc]
    index = {p: i for i, p in enumerate(rc)}
    if len(index) != len(rc):
        raise ValueError("lattice coordinates contain duplicates")
    n = len(rc)
    A = np.zeros((n, n))
    for (r, c), i in index.items():
        for dr, dc in ((0, 1), (1, 0)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                A[i, j] = A[j, i] = 1.0
    coords = np.array([(c, -r) for r, c in rc], dtype=float).reshape(n, 2)
    return A, coords, A.sum(axis=1) < 4


# ---------------------------------------------------------------- paper graphs
def paper8() -> Graph:
    """Fig. 1: 2 x 4 ladder, w = 1/4 on every edge, mu = 1, B = {1, 2, 3}."""
    A, coords, boundary = lattice_from_coords(grid_coords(2, 4))
    return Graph(W=0.25 * A, mu=np.ones(8), B=np.array([1, 2, 3]), q=np.zeros(8),
                 family="paper8", params={"rows": 2, "cols": 4, "w": 0.25, "mu": "constant 1"},
                 coords=coords, boundary=boundary)


def paper9() -> Graph:
    """Fig. 4: 3 x 3 grid, w = 1 on every edge, mu = deg, B = {0, 1, 2}."""
    A, coords, boundary = lattice_from_coords(grid_coords(3, 3))
    return Graph(W=A, mu=A.sum(axis=1), B=np.array([0, 1, 2]), q=np.zeros(9),
                 family="paper9", params={"rows": 3, "cols": 3, "w": 1.0, "mu": "deg"},
                 coords=coords, boundary=boundary)


PRESETS = {"paper8": paper8, "paper9": paper9}
