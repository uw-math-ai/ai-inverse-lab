"""Bounded, data-only access to the repository's first-passage simulator.

``run_simulator_request`` accepts a JSON object with ``operation`` equal to
``first_passage`` or ``compare_potentials``, a preset ``graph`` (``path3``,
``paper8``, or ``paper9``), scalar/vector ``q``, positive integer ``T <= 64``,
and a fixed step size ``h``. ``observed`` may select distinct preset vertices.
``q_other`` is required only for comparisons and is forbidden otherwise.

These are floating-point evaluations of exact recurrences, not formal proofs.
Numerical equality over finitely many times never certifies all-time uniqueness.
No generated code, expressions, graph definitions, or filesystem paths execute.
"""

from dataclasses import replace
import math
from typing import Any

import numpy as np

from forwardSimulation.forward_sim.engine import exact_first_passage, exact_outcomes
from forwardSimulation.forward_sim.graphs import GRAPHS, Graph
from forwardSimulation.forward_sim.model import Experiment, build_kernel


MAX_STEPS = 64
COMPARISON_ATOL = 1e-10
LIMITATIONS = [
    "Numerical evidence is not a proof or a certified counterexample.",
    "A finite-time match does not establish all-time equality or uniqueness of q.",
    "The killed-walk simulator supports only nonnegative potentials and admissible h.",
]
PATH3 = Graph(
    "path3", "Three-vertex path A-X-D", 1, 3, ((0, 1), (1, 2)),
    (0, 2), 1.0, (1.0, 1.0, 1.0), "Workflow pilot; vertices A=0, X=1, D=2",
)
PRESETS = {**GRAPHS, "path3": PATH3}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _potential(value: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, list):
        if len(value) != size:
            raise ValueError(f"{name} must have {size} entries in vertex order.")
        result = tuple(_number(entry, f"{name}[{i}]") for i, entry in enumerate(value))
    else:
        result = (_number(value, name),) * size
    if any(entry < 0 or entry > 1_000_000 for entry in result):
        raise ValueError(f"{name} must be between 0 and 1,000,000.")
    return result


def _parse_request(request: dict) -> tuple[str, Graph, int, float, tuple, tuple | None]:
    if not isinstance(request, dict):
        raise ValueError("Simulator request must be a JSON object.")
    if any(not isinstance(key, str) for key in request):
        raise ValueError("Simulator request keys must be strings.")
    extra = set(request) - {"operation", "graph", "q", "q_other", "T", "h", "observed"}
    if extra:
        raise ValueError("Unknown simulator fields: " + ", ".join(sorted(extra)))
    operation = request.get("operation", "first_passage")
    if operation not in ("first_passage", "compare_potentials"):
        raise ValueError("operation must be first_passage or compare_potentials.")
    graph_id = request.get("graph", "path3")
    if not isinstance(graph_id, str) or graph_id not in PRESETS:
        raise ValueError("graph must be path3, paper8, or paper9.")
    graph = PRESETS[graph_id]
    observed = request.get("observed", list(graph.observed))
    if (
        not isinstance(observed, list)
        or not 1 <= len(observed) <= graph.size
        or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < graph.size
               for v in observed)
        or len(set(observed)) != len(observed)
    ):
        raise ValueError("observed must list distinct valid vertex indices.")
    graph = replace(graph, observed=tuple(observed))
    T = request.get("T", 8)
    if isinstance(T, bool) or not isinstance(T, int) or not 1 <= T <= MAX_STEPS:
        raise ValueError(f"T must be an integer between 1 and {MAX_STEPS}.")
    h = _number(request.get("h", 0.1), "h")
    if not 0 < h <= 1:
        raise ValueError("h must be positive and at most 1.")
    q = _potential(request.get("q", 0), graph.size, "q")
    if operation == "compare_potentials":
        if "q_other" not in request:
            raise ValueError("q_other is required for compare_potentials.")
        other = _potential(request["q_other"], graph.size, "q_other")
    else:
        if "q_other" in request:
            raise ValueError("q_other is only allowed for compare_potentials.")
        other = None
    return operation, graph, T, h, q, other


def _evaluate(graph: Graph, T: int, h: float, q: tuple[float, ...]) -> dict:
    # Always keep h explicit: adapting h independently to q would compare
    # different physical times and could create spurious identifiability claims.
    config = Experiment(graph, T, 1, q, 0, h, "explicit", "Agent numerical probe")
    kernel, kill, _ = build_kernel(config)
    exact = exact_first_passage(kernel, T, graph.observed)
    forward, killed, censored = exact_outcomes(kernel, kill, T, graph.observed)
    recurrence_error = float(np.max(np.abs(exact - forward)))
    mass_error = float(np.max(np.abs(exact.sum(axis=0) + killed + censored - 1)))
    if not np.allclose(exact, forward, atol=1e-12, rtol=1e-10) or mass_error > 1e-10:
        raise ArithmeticError("Independent simulator recurrences or mass accounting disagree.")
    return {"r_exact": exact, "killed": killed, "censored": censored,
            "recurrence_max_error": recurrence_error, "mass_max_error": mass_error}


def run_simulator_request(request: dict) -> dict:
    """Execute one bounded numerical request; validation failures are structured data.

    Comparison results contain a largest-difference observation, if the absolute
    residual exceeds 1e-10. A match means only indistinguishability at this
    tolerance and horizon. The same explicitly reported h is used for both q's.
    """
    base = {"evidence_kind": "numerical_only", "limitations": list(LIMITATIONS)}
    try:
        operation, graph, T, h, q, other = _parse_request(request)
        left = _evaluate(graph, T, h, q)
        result = {
            **base, "status": "completed", "operation": operation,
            "graph": graph.as_dict(), "q": list(q), "T": T, "h": h,
            "observed": list(graph.observed), "times": list(range(1, T + 1)),
            "physical_times": (np.arange(1, T + 1) * h).tolist(),
            "time_convention": "First positive-time hits/returns, t=1,...,T; physical time=t*h.",
            "tensor_axes": ["time", "source_index_in_observed", "target_index_in_observed"],
            "method": "Existing exact recurrence, evaluated in floating point and cross-checked by forward propagation.",
            "recurrence_max_error": left["recurrence_max_error"],
            "mass_max_error": left["mass_max_error"],
        }
        if other is None:
            result.update({"r_exact": left["r_exact"].tolist(),
                           "killed": left["killed"].tolist(),
                           "censored": left["censored"].tolist()})
            return result

        right = _evaluate(graph, T, h, other)
        difference = np.abs(left["r_exact"] - right["r_exact"])
        maximum = float(np.max(difference))
        match = maximum <= COMPARISON_ATOL
        witness = None
        if not match:
            time_index, source, target = np.unravel_index(np.argmax(difference), difference.shape)
            witness = {
                "t": int(time_index + 1), "physical_time": float((time_index + 1) * h),
                "source": graph.observed[source], "target": graph.observed[target],
                "r_q": float(left["r_exact"][time_index, source, target]),
                "r_q_other": float(right["r_exact"][time_index, source, target]),
            }
        result.update({
            "q_other": list(other), "potentials_distinct": q != other,
            "comparison_atol": COMPARISON_ATOL, "max_absolute_difference": maximum,
            "matches_within_tolerance": match, "distinguishing_observation": witness,
            "recurrence_max_error": max(left["recurrence_max_error"], right["recurrence_max_error"]),
            "mass_max_error": max(left["mass_max_error"], right["mass_max_error"]),
        })
        return result
    except (ValueError, TypeError, ArithmeticError) as exc:
        return {**base, "status": "error", "error": str(exc)}
