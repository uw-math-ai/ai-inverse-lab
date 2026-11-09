"""Shared test helpers. Lives at the project root so `fptsim` imports without installation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Kevin's engine, imported read-only by path (no bytecode is written into his folder)
KEVIN_DIR = ROOT.parent / "Graphs - First Passage Time Simulation" / "Kevin_data_simulation"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch
    return torch.cuda.is_available()


# blocks that an override replaces as a whole (their keys depend on the chosen mode)
REPLACED_BLOCKS = {"families", "weights", "mu", "B", "q", "sweep"}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in REPLACED_BLOCKS and isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def base_raw_config(output_dir) -> dict:
    """A small random-mix run that finishes in seconds on a CPU."""
    return {
        "run_name": "test", "output_dir": str(output_dir), "master_seed": 7, "num_graphs": 6,
        "device": "cpu", "backend": "auto", "shard_size": 2, "graphs_per_batch": 2,
        "log_interval_s": 1000,
        "toggles": {"convergence_checkpoints": True},
        "graphs": {
            "families": {"path": {"weight": 1, "n": [3, 6]},
                         "cycle": {"weight": 1, "n": [3, 6]},
                         "grid": {"weight": 1, "rows": [2, 3], "cols": [2, 3]}},
            "weights": {"dist": "uniform", "low": 0.2, "high": 1.0},
            "mu": {"mode": "slack", "a": 0.0, "b": 0.5},
            "B": {"mode": "random_k", "k": [1, 3]},
            "q": {"mode": "sparse", "fraction": 0.3, "low": 0.0, "high": 0.3},
        },
        "simulation": {"T": 8, "N": 1500, "replicates": 2, "N_checkpoints": [100, 1000],
                       "chunk_walks": 500},
    }


@pytest.fixture
def make_config(tmp_path):
    """make_config(subdir="run", **overrides) -> validated config dict."""
    from fptsim.config import validate_config

    def make(subdir="run", **overrides):
        return validate_config(deep_merge(base_raw_config(tmp_path / subdir), overrides))
    return make


@pytest.fixture(scope="session")
def kevin():
    """Kevin's forward_sim package (model, engine), or skip when the folder is absent."""
    if not (KEVIN_DIR / "forward_sim" / "engine.py").exists():
        pytest.skip(f"Kevin's forward_sim not found at {KEVIN_DIR}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(KEVIN_DIR))
    try:
        import forward_sim.engine as engine
        import forward_sim.model as model
    finally:
        sys.path.remove(str(KEVIN_DIR))
    return {"engine": engine, "model": model}
