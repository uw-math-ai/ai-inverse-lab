"""The two fixed graphs transcribed from arXiv:2603.20710v1, Figs 1 and 4."""
from dataclasses import dataclass


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

    @property
    def size(self) -> int:
        return self.rows * self.columns

    def as_dict(self) -> dict:
        degrees = [0] * self.size
        for a, b in self.edges:
            degrees[a] += 1
            degrees[b] += 1
        return {
            "id": self.id, "name": self.name, "rows": self.rows,
            "columns": self.columns, "edges": [list(e) for e in self.edges],
            "observed": list(self.observed), "edge_weight": self.weight,
            "mu": list(self.mu), "degrees": degrees, "source": self.source,
            "vertices": [{"id": i, "row": i // self.columns,
                          "column": i % self.columns, "observed": i in self.observed,
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


GRAPHS = {
    "paper8": _grid("paper8", "Eight-vertex ladder", 2, 4, (1, 2, 3), 0.25,
                    False, "Section 5.1 · Figure 1 · arXiv:2603.20710v1"),
    "paper9": _grid("paper9", "Nine-vertex grid", 3, 3, (0, 1, 2), 1.0,
                    True, "Section 5.2 · Figure 4 · arXiv:2603.20710v1"),
}
