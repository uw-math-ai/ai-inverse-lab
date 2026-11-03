"""Fixed graph presets for first-passage experiments."""
from dataclasses import dataclass
import itertools


@dataclass(frozen=True)
class Graph:
    id: str
    name: str
    rows: int
    columns: int
    edges: tuple[tuple[int, int], ...]
    observed: tuple[int, ...]
    weight: float
    mu: tuple[float, ...]
    source: str
    coordinates: tuple[tuple[float, float], ...] | None = None

    @property
    def size(self) -> int:
        return len(self.mu)

    def as_dict(self) -> dict:
        degrees = [0] * self.size
        for a, b in self.edges:
            degrees[a] += 1
            degrees[b] += 1
        if self.coordinates is None:
            coordinates = tuple(
                (i % self.columns / max(1, self.columns - 1),
                 i // self.columns / max(1, self.rows - 1))
                for i in range(self.size)
            )
        else:
            coordinates = self.coordinates
        return {
            "id": self.id, "name": self.name, "rows": self.rows,
            "columns": self.columns, "edges": [list(e) for e in self.edges],
            "observed": list(self.observed), "edge_weight": self.weight,
            "mu": list(self.mu), "degrees": degrees, "source": self.source,
            "vertices": [{"id": i, "x": coordinates[i][0], "y": coordinates[i][1],
                          "observed": i in self.observed,
                          "mu": self.mu[i], "degree": degrees[i]}
                         for i in range(self.size)],
        }


def _grid(key: str, name: str, rows: int, cols: int,
          observed: tuple[int, ...], weight: float, normalized: bool, source: str) -> Graph:
    edges = []
    for row in range(rows):
        for col in range(cols):
            vertex = row * cols + col
            if col + 1 < cols:
                edges.append((vertex, vertex + 1))
            if row + 1 < rows:
                edges.append((vertex, vertex + cols))
    degrees = [0] * (rows * cols)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    mu = tuple(float(d) if normalized else 1.0 for d in degrees)
    return Graph(key, name, rows, cols, tuple(edges), observed, weight, mu, source)


def _custom(key: str, name: str, size: int, edges: tuple[tuple[int, int], ...],
            observed: tuple[int, ...], coordinates: tuple[tuple[float, float], ...]) -> Graph:
    degrees = [0] * size
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return Graph(key, name, 1, size, edges, observed, 1.0,
                 tuple(float(d) for d in degrees), "Exploratory graph preset",
                 coordinates)


PATH7_EDGES = tuple((i, i + 1) for i in range(6))
CYCLE8_EDGES = tuple((i, (i + 1) % 8) for i in range(8))
STAR7_EDGES = tuple((0, i) for i in range(1, 7))
LOLLIPOP8_EDGES = tuple(itertools.combinations(range(4), 2)) + tuple((i, i + 1) for i in range(3, 7))


GRAPHS = {
    "paper8": _grid("paper8", "Eight-vertex ladder", 2, 4, (1, 2, 3), 0.25,
                    False, "Section 5.1 · Figure 1 · arXiv:2603.20710v1"),
    "paper9": _grid("paper9", "Nine-vertex grid", 3, 3, (0, 1, 2), 1.0,
                    True, "Section 5.2 · Figure 4 · arXiv:2603.20710v1"),
    "path7": _custom("path7", "Seven-vertex path", 7, PATH7_EDGES, (0, 3, 6),
                     tuple((i / 6, 0.5) for i in range(7))),
    "cycle8": _custom("cycle8", "Eight-vertex cycle", 8, CYCLE8_EDGES, (0, 3, 6),
                      ((0.5, 0.02), (0.84, 0.16), (0.98, 0.5), (0.84, 0.84),
                       (0.5, 0.98), (0.16, 0.84), (0.02, 0.5), (0.16, 0.16))),
    "star7": _custom("star7", "Seven-vertex star", 7, STAR7_EDGES, (0, 1, 2),
                     ((0.5, 0.5), (0.5, 0.02), (0.92, 0.26), (0.92, 0.74),
                      (0.5, 0.98), (0.08, 0.74), (0.08, 0.26))),
    "lollipop8": _custom("lollipop8", "Eight-vertex lollipop", 8, LOLLIPOP8_EDGES, (0, 3, 7),
                         ((0.02, 0.18), (0.02, 0.82), (0.28, 0.18), (0.28, 0.82),
                          (0.46, 0.72), (0.64, 0.62), (0.82, 0.52), (0.98, 0.42))),
}
