"""Independent exact recurrence and batched path-wise Monte Carlo estimator."""
from collections.abc import Callable
from datetime import datetime, timezone
import platform
import time
import numpy as np
from . import __version__
from .model import Experiment, build_kernel


class Cancelled(Exception):
    """A cooperative cancellation at a simulation checkpoint."""


def exact_first_passage(K: np.ndarray, T: int, observed: tuple[int, ...]) -> np.ndarray:
    """r[t-1,x,y]; t >= 1, with first returns when x == y."""
    indices = list(observed)
    answer = np.zeros((T, len(indices), len(indices)))
    for j, target in enumerate(indices):
        previous = K[:, target].copy()
        answer[0, :, j] = previous[indices]
        for t in range(1, T):
            previous[target] = 0  # exclude prior arrival at the target
            previous = K @ previous
            answer[t, :, j] = previous[indices]
    return answer


def exact_outcomes(K: np.ndarray, kill: np.ndarray, T: int,
                   observed: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent forward propagation, stopped on the first positive-time hit."""
    b = len(observed)
    hits = np.zeros((T, b, b))
    killed = np.zeros((b, b))
    censored = np.zeros((b, b))
    for j, target in enumerate(observed):
        live = np.zeros((b, K.shape[0]))
        live[np.arange(b), list(observed)] = 1
        for t in range(T):
            killed[:, j] += live @ kill
            live = live @ K
            hits[t, :, j] = live[:, target]
            live[:, target] = 0
        censored[:, j] = live.sum(axis=1)
    return hits, killed, censored


def wilson(counts: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% Wilson binomial intervals, including zero-count bins."""
    z = 1.959963984540054
    p = counts / N
    denominator = 1 + z*z / N
    center = (p + z*z / (2*N)) / denominator
    half = z * np.sqrt(p*(1-p)/N + z*z/(4*N*N)) / denominator
    return np.maximum(0, center-half), np.minimum(1, center+half)


def simulate(config: Experiment, progress: Callable[[dict], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> dict:
    started = time.perf_counter()
    K, kill, P = build_kernel(config)
    observed = config.observation_set
    b, n, T, N = len(observed), config.graph.size, config.T, config.N
    exact = exact_first_passage(K, T, observed)
    forward, exact_killed, exact_censored = exact_outcomes(K, kill, T, observed)
    if not np.allclose(exact, forward, atol=1e-12, rtol=1e-10):
        raise ArithmeticError("Independent exact calculations disagree.")
    if not np.allclose(exact.sum(axis=0)+exact_killed+exact_censored, 1, atol=1e-11):
        raise ArithmeticError("Exact probability mass does not sum to one.")
    counts = np.zeros((T, b, b), dtype=np.int64)
    killed = np.zeros((b, b), dtype=np.int64)
    censored = np.zeros((b, b), dtype=np.int64)
    checkpoints = sorted({N, *[m for m in (100, 1000, 10000, 100000, 1000000, 10000000) if m <= N]})
    prefix_counts = {m: np.zeros_like(counts) for m in checkpoints}
    cdf = np.cumsum(P, axis=1)
    cdf[:, -1] = 1.0
    completed = 0
    for i, source in enumerate(observed):
        for j, target in enumerate(observed):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, source, target]))
            done = 0
            for checkpoint in checkpoints:
                while done < checkpoint:
                    if cancelled and cancelled():
                        raise Cancelled("Experiment cancelled.")
                    batch = min(20000, checkpoint - done)
                    states = np.full(batch, source, dtype=np.int64)
                    for t in range(T):
                        if not len(states):
                            break
                        if cancelled and cancelled():
                            raise Cancelled("Experiment cancelled.")
                        # Each row uses its current vertex's categorical transition law.
                        next_states = (rng.random((len(states), 1)) >= cdf[states]).sum(axis=1)
                        hit = next_states == target
                        dead = next_states == n
                        counts[t, i, j] += np.count_nonzero(hit)
                        killed[i, j] += np.count_nonzero(dead)
                        states = next_states[~(hit | dead)]
                    censored[i, j] += len(states)
                    done += batch
                    completed += batch
                    if progress:
                        progress({"fraction": completed/(N*b*b), "completed_trials": completed,
                                  "total_trials": N*b*b, "pair": [source, target]})
                prefix_counts[checkpoint][:, i, j] = counts[:, i, j]
    if not np.all(counts.sum(axis=0)+killed+censored == N):
        raise ArithmeticError("A simulated trajectory was lost or counted twice.")
    empirical = counts/N
    tail_exact = exact_killed + exact_censored
    tail_empirical = (killed+censored)/N
    lower, upper = wilson(counts, N)
    tail_low, tail_high = wilson(killed+censored, N)
    norm = float(np.linalg.norm(exact))
    difference = empirical - exact
    frne = 100*float(np.linalg.norm(difference))/norm if norm else None
    tv = (np.abs(difference).sum(axis=0) + np.abs(tail_exact-tail_empirical))/2
    convergence = []
    for m in checkpoints:
        err = float(np.linalg.norm(prefix_counts[m]/m-exact))
        expected = float(np.sqrt(np.sum(exact*(1-exact))/m))
        convergence.append({"N": m, "frne_percent": 100*err/norm if norm else None,
                            "expected_rms_frne_percent": 100*expected/norm if norm else None})
    pairs = []
    for i, source in enumerate(observed):
        for j, target in enumerate(observed):
            pair_norm = float(np.linalg.norm(exact[:, i, j]))
            pairs.append({"source": source, "target": target,
                          "frne_percent": 100*float(np.linalg.norm(difference[:, i, j]))/pair_norm if pair_norm else None,
                          "total_variation": float(tv[i,j]),
                          "hit_exact": float(exact[:,i,j].sum()),
                          "hit_empirical": float(empirical[:,i,j].sum()),
                          "killed_exact": float(exact_killed[i,j]), "killed_count": int(killed[i,j]),
                          "censored_exact": float(exact_censored[i,j]), "censored_count": int(censored[i,j]),
                          "no_hit_exact": float(tail_exact[i,j]), "no_hit_empirical": float(tail_empirical[i,j]),
                          "no_hit_ci95": [float(tail_low[i,j]), float(tail_high[i,j])]})
    graph_data = config.graph.as_dict()
    graph_data["observed"] = list(observed)
    for vertex in graph_data["vertices"]:
        vertex["observed"] = vertex["id"] in observed
    return {
        "schema_version": 2, "engine_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(), "graph": graph_data,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "rng": "PCG64 / SeedSequence([seed, source, target])", "batch_size": 20000},
        "time_convention": "T is maximum step; first arrivals at t=1,...,T; physical time=t*dt",
        "tensor_axes": ["time", "source_index_in_observed", "target_index_in_observed"],
        "observed": list(observed), "times": list(range(1,T+1)),
        "physical_times": (np.arange(1,T+1)*config.dt).tolist(),
        "kernel": K.tolist(), "killing_probabilities": kill.tolist(),
        "augmented_transition": P.tolist(), "counts": counts.tolist(),
        "r_exact": exact.tolist(), "r_empirical": empirical.tolist(),
        "ci95_lower": lower.tolist(), "ci95_upper": upper.tolist(), "pairs": pairs,
        "metrics": {"frne_percent": frne, "max_absolute_error": float(np.max(np.abs(difference))),
                    "mean_total_variation": float(tv.mean()), "max_total_variation": float(tv.max()),
                    "mean_no_hit_exact": float(tail_exact.mean()),
                    "mean_no_hit_empirical": float(tail_empirical.mean()),
                    "exact_mass_max_error": float(np.max(np.abs(exact.sum(axis=0)+tail_exact-1))),
                    "elapsed_seconds": time.perf_counter()-started},
        "convergence": convergence,
    }
