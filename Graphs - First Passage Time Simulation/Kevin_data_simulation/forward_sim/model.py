"""Validated experiment configuration and the explicit heat/killing kernel."""
from dataclasses import dataclass
import math
from typing import Any
import numpy as np
from .graphs import GRAPHS, Graph

MAX_STEPS = 1000
MAX_TRIALS = 10_000_000
MAX_WORK = 500_000_000  # worst-case transitions across all observed pairs
ANALYSIS_TYPES = {"baseline", "B_placement", "B_size", "T_sweep", "custom"}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low:,} and {high:,}.")
    return value


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


@dataclass(frozen=True)
class Experiment:
    graph: Graph
    T: int
    N: int
    q: tuple[float, ...]
    seed: int
    dt: float
    dt_mode: str
    name: str
    observed: tuple[int, ...] | None = None
    analysis_type: str = "baseline"
    case_label: str = "Baseline"

    @property
    def observation_set(self) -> tuple[int, ...]:
        return self.graph.observed if self.observed is None else self.observed

    def as_dict(self) -> dict:
        return {"graph": self.graph.id, "T": self.T, "N": self.N,
                "q": list(self.q), "seed": self.seed, "dt": self.dt,
                "dt_mode": self.dt_mode, "name": self.name,
                "physical_horizon": self.T * self.dt,
                "observed": list(self.observation_set),
                "B_size": len(self.observation_set),
                "analysis_type": self.analysis_type,
                "case_label": self.case_label}


def parse_config(raw: dict) -> Experiment:
    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration must be a JSON object.")
    extra = set(raw) - {"graph", "T", "N", "q", "seed", "dt", "name",
                        "observed", "analysis_type", "case_label"}
    if extra:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(extra)))
    graph_id = raw.get("graph", "paper8")
    if not isinstance(graph_id, str) or graph_id not in GRAPHS:
        raise ValueError("graph must be one of: " + ", ".join(GRAPHS) + ".")
    graph = GRAPHS[graph_id]
    observed_raw = raw.get("observed", graph.observed)
    if (not isinstance(observed_raw, (list, tuple)) or not observed_raw or
            any(isinstance(value, bool) or not isinstance(value, int) for value in observed_raw)):
        raise ValueError("observed must be a nonempty list of integer vertex IDs.")
    observed = tuple(observed_raw)
    if len(set(observed)) != len(observed):
        raise ValueError("observed may not contain duplicate vertices.")
    if any(value < 0 or value >= graph.size for value in observed):
        raise ValueError(f"observed vertices must be between 0 and {graph.size - 1}.")
    T = _integer(raw.get("T", 17), "T", 1, MAX_STEPS)
    N = _integer(raw.get("N", 10000), "N", 1, MAX_TRIALS)
    if T * N * len(observed) ** 2 > MAX_WORK:
        raise ValueError(f"This run exceeds {MAX_WORK:,} possible transitions. Reduce T or N.")
    seed = _integer(raw.get("seed", 2026), "seed", 0, 2**53 - 1)
    q_raw = raw.get("q", 0)
    if isinstance(q_raw, list):
        if len(q_raw) != graph.size:
            raise ValueError(f"q must have exactly {graph.size} entries in vertex order.")
        q = tuple(_real(value, f"q[{i}]") for i, value in enumerate(q_raw))
    else:
        q = (_real(q_raw, "q"),) * graph.size
    if any(value < 0 or value > 1e6 for value in q):
        raise ValueError("q must be between 0 and 1,000,000. Negative potentials are not killing probabilities.")
    rates = np.zeros(graph.size)
    for a, b in graph.edges:
        rates[a] += graph.weight / graph.mu[a]
        rates[b] += graph.weight / graph.mu[b]
    maximum = float(np.max(rates + np.asarray(q)))
    requested_dt = raw.get("dt", "auto")
    mode = "auto" if requested_dt == "auto" else "explicit"
    dt = min(1.0, 1.0 / maximum) if mode == "auto" else _real(requested_dt, "dt")
    if dt <= 0 or dt > 1 or dt * maximum > 1 + 1e-12:
        raise ValueError(f"dt must be positive, at most 1, and at most {1/maximum:.12g} for this q. Use auto.")
    name = raw.get("name", "Untitled experiment")
    if not isinstance(name, str) or len(name) > 120:
        raise ValueError("name must be a string of at most 120 characters.")
    analysis_type = raw.get("analysis_type", "baseline")
    if analysis_type not in ANALYSIS_TYPES:
        raise ValueError("analysis_type must be one of: " + ", ".join(sorted(ANALYSIS_TYPES)) + ".")
    case_label = raw.get("case_label", "Baseline")
    if not isinstance(case_label, str) or not case_label.strip() or len(case_label) > 80:
        raise ValueError("case_label must be a nonempty string of at most 80 characters.")
    return Experiment(graph, T, N, q, seed, dt, mode, name.strip() or "Untitled experiment",
                      observed, analysis_type, case_label.strip())


def build_kernel(config: Experiment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return K on X, killing probabilities, and augmented stochastic P."""
    graph = config.graph
    K = np.zeros((graph.size, graph.size), dtype=float)
    for a, b in graph.edges:
        K[a, b] = config.dt * graph.weight / graph.mu[a]
        K[b, a] = config.dt * graph.weight / graph.mu[b]
    kill = config.dt * np.asarray(config.q)
    diagonal = 1 - K.sum(axis=1) - kill
    if np.min(diagonal) < -1e-12:
        raise ValueError("Potential and step size produce a negative holding probability.")
    np.fill_diagonal(K, np.maximum(diagonal, 0))
    P = np.zeros((graph.size + 1, graph.size + 1))
    P[:-1, :-1], P[:-1, -1], P[-1, -1] = K, kill, 1.0
    if not np.allclose(P.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Transition rows must sum to one.")
    return K, kill, P
