"""Accuracy metrics comparing the empirical r' = counts / N with the exact r (as in Kevin's engine).

    FRNE %          = 100 ||r' - r||_F / ||r||_F over the finite-time tensor (T, |B|, |B|)
    expected RMS %  = 100 sqrt(sum r (1 - r) / N) / ||r||_F
    TV per pair     = (sum_t |r' - r| + |no-arrival' - no-arrival|) / 2, no-arrival = killed + censored
    Wilson 95% CIs per bin (T hit bins + killed + censored); coverage = fraction covering the exact value
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson(counts: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% Wilson binomial intervals, including zero-count bins."""
    p = counts / N
    denominator = 1 + Z95 * Z95 / N
    center = (p + Z95 * Z95 / (2 * N)) / denominator
    half = Z95 * np.sqrt(p * (1 - p) / N + Z95 * Z95 / (4 * N * N)) / denominator
    return np.maximum(0, center - half), np.minimum(1, center + half)


def frne(r: np.ndarray, r_emp: np.ndarray) -> float:
    norm = float(np.linalg.norm(r))
    return 100 * float(np.linalg.norm(r_emp - r)) / norm if norm else float("nan")


def expected_rms_frne(r: np.ndarray, N: int) -> float:
    norm = float(np.linalg.norm(r))
    return 100 * float(np.sqrt(np.sum(r * (1 - r)) / N)) / norm if norm else float("nan")


def mean_fpt(r: np.ndarray) -> float:
    """Mean over pairs of E[tau | tau <= T]; pairs with no mass are skipped."""
    t = np.arange(1, r.shape[0] + 1)[:, None, None]
    mass = r.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (t * r).sum(axis=0) / mass
    m = m[mass > 0]
    return float(m.mean()) if m.size else float("nan")


def _masses(prefix: str, hits, killed, censored) -> dict:
    return {f"hit_mass_{prefix}": float(hits.sum(axis=0).mean()),
            f"killed_mass_{prefix}": float(killed.mean()),
            f"censored_mass_{prefix}": float(censored.mean()),
            f"mean_fpt_{prefix}": mean_fpt(hits)}


def compute_metrics(N: int, exact: dict | None, hits=None, killed=None, censored=None,
                    prefix=None, checkpoints=None) -> tuple[dict, dict]:
    """-> (row of scalar/list metrics, extra arrays such as CIs).

    exact: {"r", "killed", "censored"} or None. hits/killed/censored are integer counts
    (T,k,k)/(k,k)/(k,k); prefix (buckets,k,k,T+2) cumulative counts at `checkpoints`.
    """
    row: dict = {}
    extras: dict = {}
    if exact is not None:
        r = exact["r"]
        row.update(_masses("exact", r, exact["killed"], exact["censored"]))
        row["exact_norm"] = float(np.linalg.norm(r))
    if hits is None:
        return row, extras
    emp, emp_k, emp_c = hits / N, killed / N, censored / N
    row.update(_masses("emp", emp, emp_k, emp_c))
    lo, hi = wilson(hits, N)
    extras["ci_lower"], extras["ci_upper"] = lo, hi
    if exact is None:
        return row, extras

    r, ex_k, ex_c = exact["r"], exact["killed"], exact["censored"]
    diff = emp - r
    norm = float(np.linalg.norm(r))
    row["frne_percent"] = frne(r, emp)
    row["expected_rms_frne_percent"] = expected_rms_frne(r, N)
    pair_norm = np.sqrt((r ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        pair = np.where(pair_norm > 0, 100 * np.sqrt((diff ** 2).sum(axis=0)) / pair_norm, np.nan)
    row["pair_frne_percent"] = [float(v) for v in pair.ravel()]   # row-major over (source, target)
    row["max_abs_error"] = float(np.abs(diff).max())
    tv = (np.abs(diff).sum(axis=0) + np.abs((ex_k + ex_c) - (emp_k + emp_c))) / 2
    row["tv_mean"], row["tv_max"] = float(tv.mean()), float(tv.max())

    # Wilson coverage over all T + 2 bins of every pair
    bins_count = np.concatenate([hits, killed[None], censored[None]], axis=0)
    bins_exact = np.concatenate([r, ex_k[None], ex_c[None]], axis=0)
    blo, bhi = wilson(bins_count, N)
    covered = (blo <= bins_exact + 1e-15) & (bins_exact <= bhi + 1e-15)
    row["ci_coverage"] = float(covered.mean())
    big = N * bins_exact >= 5
    row["ci_coverage_np5"] = float(covered[big].mean()) if big.any() else float("nan")
    row["ci_bins"] = int(covered.size)

    if prefix is not None and checkpoints is not None and len(checkpoints) > 1:
        T = r.shape[0]
        for b, m in enumerate(checkpoints):
            emp_m = np.moveaxis(prefix[b, ..., :T], -1, 0) / m
            row[f"frne_N{int(m)}"] = 100 * float(np.linalg.norm(emp_m - r)) / norm if norm else float("nan")
            row[f"exp_frne_N{int(m)}"] = expected_rms_frne(r, int(m))
    return row, extras
