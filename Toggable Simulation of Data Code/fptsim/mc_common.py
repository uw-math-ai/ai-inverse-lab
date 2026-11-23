"""Pieces shared by the Monte Carlo backends: work items, seeding, outcome bins, device choice.

A work item is one chunk of walks for one (graph, sweep point, replicate): m walks for each
ordered pair in B x B, with walk indices start..start+m-1 inside each pair. Each item has its
own seed, so results do not depend on how items are packed into GPU launches. That keeps
runs reproducible across GPUs, memory limits, and resumes.

Outcome bins per walk: 0..T-1 = first hit at t = 1..T, T = killed, T+1 = censored.
Counts are stored per checkpoint bucket: bucket b holds walks with index in
[cp[b-1], cp[b]), so cumulative sums over buckets give the prefix counts at each checkpoint.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("fptsim")


@dataclass
class MCItem:
    key: tuple                 # (graph_id, sweep_index, replicate, chunk)
    nbr: np.ndarray            # (n+1, D) reachable states; the cemetery is state n
    cdf: np.ndarray            # (n+1, D) cumulative probabilities
    B: np.ndarray              # (k,) observed vertices
    T: int
    m: int                     # walks per ordered pair in this chunk
    start: int                 # index of the first walk in this chunk, within each pair
    checkpoints: np.ndarray    # sorted prefix sizes, last == N
    seed: int

    @property
    def n(self) -> int:
        return int(self.nbr.shape[0]) - 1

    @property
    def D(self) -> int:
        return int(self.nbr.shape[1])

    @property
    def k(self) -> int:
        return int(len(self.B))

    @property
    def walks(self) -> int:
        return self.k * self.k * self.m


def mc_seed(master_seed: int, graph_id: int, sweep: int, replicate: int, chunk: int) -> int:
    """63-bit seed from SeedSequence(master_seed, spawn_key=(graph_id, 1, sweep, replicate, chunk))."""
    ss = np.random.SeedSequence(master_seed, spawn_key=(graph_id, 1, sweep, replicate, chunk))
    a, b = ss.generate_state(2, np.uint32)
    return (int(a) << 31) ^ int(b)


def chunk_plan(N: int, chunk: int) -> list[tuple[int, int]]:
    """[(start, m), ...] covering walk indices 0..N-1 in pieces of at most `chunk`."""
    return [(s, min(chunk, N - s)) for s in range(0, N, chunk)]


def counts_to_outcomes(c: np.ndarray, T: int):
    """(k, k, buckets, T+2) raw counts -> hits (T,k,k), killed (k,k), censored (k,k), prefix.

    prefix has shape (buckets, k, k, T+2): cumulative counts at each checkpoint.
    """
    total = c.sum(axis=2)
    hits = np.moveaxis(total[..., :T], -1, 0)
    return hits, total[..., T], total[..., T + 1], np.moveaxis(np.cumsum(c, axis=2), 2, 0)


def check_item_counts(item: MCItem, c: np.ndarray) -> None:
    """Every walk lands in exactly one bin of exactly one checkpoint bucket."""
    idx = item.start + np.arange(item.m)
    sizes = np.bincount(np.searchsorted(item.checkpoints, idx, side="right"),
                        minlength=len(item.checkpoints))
    if not np.array_equal(c.sum(axis=-1), np.broadcast_to(sizes, c.shape[:-1])):
        raise ArithmeticError(f"Monte Carlo item {item.key}: a walk was lost or counted twice")


def bytes_per_walk(D: int) -> int:
    """Rough peak bytes per live walk in one torch step (gathered cdf row dominates)."""
    return 9 * D + 96


# ---------------------------------------------------------------- execution choice
def resolve_backend(backend: str = "auto", device: str = "auto", devices=None) -> tuple[str, list[str]]:
    """-> (backend, [device, ...]). Raises ValueError for impossible combinations."""
    try:
        import torch
    except ImportError:
        torch = None
    if backend == "auto":
        backend = "torch" if torch is not None else "numpy"
    if backend == "torch" and torch is None:
        raise ValueError("backend: torch requested but torch is not installed")
    if backend == "numpy":
        if device.startswith("cuda"):
            raise ValueError("backend: numpy runs on the CPU only; use backend torch for CUDA")
        return "numpy", ["cpu"]
    cuda_ok = torch.cuda.is_available()
    if device == "auto":
        device = "cuda" if cuda_ok else "cpu"
    if device.startswith("cuda"):
        if not cuda_ok:
            raise ValueError(f"device: {device} requested but CUDA is not available")
        if devices:
            return "torch", [f"cuda:{d}" for d in devices]
        return "torch", [device if ":" in device else "cuda:0"]
    return "torch", ["cpu"]


def resolve_execution(cfg: dict) -> tuple[str, list[str]]:
    from .config import ConfigError
    try:
        return resolve_backend(cfg["backend"], cfg["device"], cfg["devices"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def device_banner(backend: str, device: str) -> str:
    if backend == "torch":
        import torch
        if device.startswith("cuda"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(idx)
            return (f"backend torch {torch.__version__} on {device}: {props.name} "
                    f"({props.total_memory / 2 ** 30:.1f} GiB, CUDA {torch.version.cuda})")
        return f"backend torch {torch.__version__} on cpu ({torch.get_num_threads()} threads)"
    return f"backend numpy {np.__version__} on cpu"


def make_engine(backend: str, device: str, cfg: dict | None = None):
    cfg = cfg or {}
    if backend == "torch":
        from .mc_torch import TorchMC
        return TorchMC(device, max_batch_walks=cfg.get("max_batch_walks", 50_000_000),
                       cpu_batch_walks=cfg.get("cpu_batch_walks", 4_000_000))
    from .mc_numpy import NumpyMC
    return NumpyMC()
