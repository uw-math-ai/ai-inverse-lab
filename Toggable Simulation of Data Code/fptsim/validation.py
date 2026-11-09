"""Paper regression (Section 5), single-graph simulation helper, benchmark, and test runner."""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .exact import exact_single
from .graphs import PRESETS, Graph
from .kernel import build_kernel, transition_table
from .mc_common import (MCItem, chunk_plan, counts_to_outcomes, device_banner, make_engine,
                        mc_seed, resolve_backend)
from .metrics import compute_metrics

log = logging.getLogger("fptsim")

# FRNE reported in arXiv:2603.20710 Sec. 5 at T_paper = 9 (17 steps), N = 1e6
PAPER_FRNE = {"paper8": 0.248499, "paper9": 0.188348}


def literal_first_passage(K: np.ndarray, B, T: int) -> np.ndarray:
    """The notebook's literal recursion (numpy port), used as an independent reference."""
    n = K.shape[0]

    @lru_cache(maxsize=None)
    def r(t, x, y):
        if t == 1:
            return float(K[x, y])
        return sum(float(K[x, z]) * r(t - 1, z, y) for z in range(n) if z != y)

    return np.array([[[r(t, int(x), int(y)) for y in B] for x in B] for t in range(1, T + 1)])


def simulate_graph(graph: Graph, T: int, N: int, replicates: int = 1, backend: str = "auto",
                   device: str = "auto", seed: int = 0, graph_id: int = 0, q_scale: float = 1.0,
                   dt="auto", chunk_walks: int = 100_000, checkpoints=None) -> dict:
    """Exact + R Monte Carlo replicates for one graph, outside the run machinery."""
    backend, devices = resolve_backend(backend, device)
    kernel = build_kernel(graph.W, graph.mu, graph.q * q_scale, dt)
    exact = exact_single(kernel, graph.B, T, device=devices[0] if backend == "torch" else None)
    nbr, cdf = transition_table(kernel.P)
    cps = np.asarray(sorted({c for c in (checkpoints or []) if c < N} | {N}), dtype=np.int64)
    engine = make_engine(backend, devices[0])
    reps = []
    for r in range(replicates):
        items = [MCItem(key=(graph_id, 0, r, c), nbr=nbr, cdf=cdf, B=graph.B, T=T, m=m, start=start,
                        checkpoints=cps, seed=mc_seed(seed, graph_id, 0, r, c))
                 for c, (start, m) in enumerate(chunk_plan(N, chunk_walks))]
        t0 = time.perf_counter()
        raw = None
        for _, counts, _ in engine.run_items(items):
            raw = counts if raw is None else raw + counts
        secs = time.perf_counter() - t0
        hits, killed, censored, prefix = counts_to_outcomes(raw, T)
        assert np.all(hits.sum(axis=0) + killed + censored == N)
        metrics, extras = compute_metrics(N, exact, hits, killed, censored,
                                          prefix if len(cps) > 1 else None, cps if len(cps) > 1 else None)
        metrics["elapsed_s"] = secs
        reps.append({"metrics": metrics, "hits": hits, "killed": killed, "censored": censored,
                     "prefix": prefix, "ci_lower": extras["ci_lower"], "ci_upper": extras["ci_upper"]})
    return {"kernel": kernel, "exact": exact, "replicates": reps, "backend": backend,
            "device": devices[0], "checkpoints": cps}


def paper_regression(N: int = 1_000_000, replicates: int = 3, backend: str = "auto",
                     device: str = "auto", seed: int = 2026, T: int = 17) -> list[dict]:
    """paper8 / paper9 at T = 17 steps, q = 0: exact vs literal recursion, FRNE vs the paper."""
    rows = []
    for gid, name in enumerate(("paper8", "paper9")):
        g = PRESETS[name]()
        res = simulate_graph(g, T, N, replicates, backend, device, seed=seed, graph_id=gid)
        literal = literal_first_passage(res["kernel"].K, g.B, T)
        frnes = np.array([rep["metrics"]["frne_percent"] for rep in res["replicates"]])
        expected = res["replicates"][0]["metrics"]["expected_rms_frne_percent"]
        secs = sum(rep["metrics"]["elapsed_s"] for rep in res["replicates"])
        rows.append({"graph": name, "T": T, "N": N, "replicates": replicates,
                     "backend": res["backend"], "device": res["device"],
                     "exact_vs_literal_max_abs": float(np.abs(res["exact"]["r"] - literal).max()),
                     "exact_mass_error": res["exact"]["mass_error"],
                     "frne_mean": float(frnes.mean()), "frne_std": float(frnes.std(ddof=1)) if replicates > 1 else float("nan"),
                     "frne_min": float(frnes.min()), "frne_max": float(frnes.max()),
                     "expected_rms_frne": expected, "paper_frne": PAPER_FRNE[name],
                     "ci_coverage": float(np.mean([rep["metrics"]["ci_coverage"] for rep in res["replicates"]])),
                     "walks_per_s": N * len(g.B) ** 2 * replicates / secs if secs else float("nan")})
    return rows


def regression_passes(row: dict) -> bool:
    """Exact matches the literal recursion; FRNE agrees with the expected RMS and the paper.

    FRNE scales as N^{-1/2}, so the paper's N = 1e6 value is rescaled to the N used here.
    The seed-to-seed spread of one FRNE is roughly 10-15% of its mean, so 30% bands are
    wide enough not to fail by chance yet still catch a biased sampler.
    """
    paper = row["paper_frne"] * (1e6 / row["N"]) ** 0.5
    expected = row["expected_rms_frne"]
    return (row["exact_vs_literal_max_abs"] < 1e-14
            and abs(expected - paper) < 0.2 * paper
            and abs(row["frne_mean"] - expected) < 0.3 * expected
            and abs(row["frne_mean"] - paper) < 0.3 * paper)


def benchmark(backend: str = "auto", device: str = "auto", seconds: float = 5.0, n: int = 30,
              k: int = 4, T: int = 17, m: int = 50_000) -> dict:
    """Walks/s and walk-steps/s on a fixed random geometric graph with |B| = k."""
    from .generators import random_geometric
    from .graphs import is_connected
    backend, devices = resolve_backend(backend, device)
    rng = np.random.default_rng(0)
    while True:
        topo = random_geometric(rng, n=n, radius=0.35)
        if is_connected(topo.W):
            break
    W = topo.W
    graph = Graph(W=W, mu=W.sum(axis=1), B=np.arange(k), q=np.full(n, 0.05))
    kernel = build_kernel(graph.W, graph.mu, graph.q)
    nbr, cdf = transition_table(kernel.P)
    cps = np.asarray([m], dtype=np.int64)
    engine = make_engine(backend, devices[0])

    def batch(offset):
        return [MCItem(key=(0, 0, 0, c), nbr=nbr, cdf=cdf, B=graph.B, T=T, m=m, start=0,
                       checkpoints=cps, seed=offset + c) for c in range(4)]

    list(engine.run_items(batch(10 ** 6)[:1]))       # warm-up
    engine.walk_steps = 0
    walks, rounds, t0 = 0, 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds or rounds == 0:
        items = batch(rounds * 4)
        list(engine.run_items(items))
        walks += sum(it.walks for it in items)
        rounds += 1
    if str(devices[0]).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"backend": backend, "device": devices[0], "banner": device_banner(backend, devices[0]),
            "n": n, "pairs": k * k, "T": T, "walks": walks, "seconds": elapsed,
            "walks_per_s": walks / elapsed, "walk_steps_per_s": engine.walk_steps / elapsed}


def run_tests(extra_args=None) -> int:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "pytest", str(root / "tests"), "-q", *(extra_args or [])]
    log.info("running %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=root)
