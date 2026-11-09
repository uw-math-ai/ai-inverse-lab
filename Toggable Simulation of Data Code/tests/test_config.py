"""Test 10: config validation rejects bad input with clear messages."""
from pathlib import Path

import pytest

from conftest import base_raw_config, deep_merge
from fptsim.config import ConfigError, load_config, sweep_points, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _bad(tmp_path, override, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(deep_merge(base_raw_config(tmp_path), override))


@pytest.mark.parametrize("override,match", [
    ({"oops": 1}, r"config: unknown key\(s\) \['oops'\]"),
    ({"toggles": {"compute_exakt": True}}, r"toggles: unknown key"),
    ({"simulation": {"N": 0}}, r"simulation.N: must be >= 1"),
    ({"simulation": {"N": 1.5}}, r"simulation.N: expected an integer"),
    ({"simulation": {"T": "many"}}, r"simulation.T: expected a number"),
    ({"simulation": {"T_mode": "seconds"}}, r"simulation.T_mode: must be one of"),
    ({"simulation": {"replicates": True}}, r"simulation.replicates: expected a number, got a boolean"),
    ({"device": "gpu"}, r"device: must be auto, cpu, cuda"),
    ({"backend": "jax"}, r"backend: must be one of"),
    ({"toggles": {"resume": "yes"}}, r"toggles.resume: expected true or false"),
    ({"toggles": {"compute_exact": False, "run_monte_carlo": False}}, r"at least one of"),
    ({"graphs": {"families": {"hypercube": {"n": 8}}}}, r"graphs.families.hypercube: unknown family"),
    ({"graphs": {"families": {"grid": {"rows": [2, 6]}}}}, r"graphs.families.grid.cols: required"),
    ({"graphs": {"families": {"grid": {"rows": [6, 2], "cols": 3}}}}, r"range low 6 exceeds high 2"),
    ({"graphs": {"families": {"grid": {"rows": [1, 2, 3], "cols": 3}}}}, r"a range must be \[low, high\]"),
    ({"graphs": {"families": {"erdos_renyi": {"n": 10, "p": 1.5}}}}, r"erdos_renyi.p: must be <= 1"),
    ({"graphs": {"families": {"grid": {"weight": 0, "rows": 2, "cols": 2}}}}, r"must not all be zero"),
    ({"graphs": {"families": {"grid": {"rows": 2, "cols": 2, "size": 3}}}}, r"grid: unknown key"),
    ({"graphs": {"weights": {"dist": "uniform", "low": 0.0, "high": 1.0}}}, r"graphs.weights.low: must be > 0"),
    ({"graphs": {"weights": {"dist": "uniform", "low": 0.5}}}, r"graphs.weights.high: required"),
    ({"graphs": {"weights": {"dist": "discrete", "values": [1, 2], "probs": [0.5]}}}, r"needs 2 entries"),
    ({"graphs": {"mu": {"mode": "deg", "on_violation": "ignore"}}}, r"on_violation: must be one of"),
    ({"graphs": {"mu": {"mode": "deg", "value": 2}}}, r"graphs.mu: unknown key"),
    ({"graphs": {"B": {"mode": "random_k"}}}, r"exactly one of k or frac"),
    ({"graphs": {"B": {"mode": "random_k", "k": 2, "frac": 0.5}}}, r"exactly one of k or frac"),
    ({"graphs": {"B": {"mode": "explicit", "vertices": [1, 1]}}}, r"vertices: contains duplicates"),
    ({"graphs": {"q": {"mode": "constant", "c": -1}}}, r"graphs.q.c: must be >= 0"),
    ({"graphs": {"q": {"mode": "zero", "on_B_only": True, "off_B_only": True}}}, r"cannot both be true"),
    ({"graphs": {"dt": 1.5}}, r"graphs.dt: must be <= 1"),
    ({"graphs": {"dt": 0}}, r"graphs.dt: must be > 0"),
    ({"graphs": {"presets": ["paper8"]}}, r"not allowed together with graphs.presets"),
    ({"sweep": {"q_scale": []}}, r"sweep.q_scale: must not be empty"),
    ({"sweep": {"temperature": [1]}}, r"sweep: unknown key"),
    ({"simulation": {"N_checkpoints": [100, "x"]}}, r"N_checkpoints\[1\]"),
])
def test_rejects(tmp_path, override, match):
    _bad(tmp_path, override, match)


def test_missing_required(tmp_path):
    raw = base_raw_config(tmp_path)
    del raw["num_graphs"]
    with pytest.raises(ConfigError, match="num_graphs: required"):
        validate_config(raw)
    raw = base_raw_config(tmp_path)
    del raw["graphs"]["B"]
    with pytest.raises(ConfigError, match="graphs.B: required"):
        validate_config(raw)


def test_numeric_strings_and_defaults(tmp_path):
    cfg = validate_config(deep_merge(base_raw_config(tmp_path),
                                     {"simulation": {"N": "1e6", "N_checkpoints": ["1e2", "1_000"]},
                                      "max_batch_walks": "50_000_000"}))
    assert cfg["simulation"]["N"] == 1_000_000
    assert cfg["simulation"]["N_checkpoints"] == [100, 1000]
    assert cfg["max_batch_walks"] == 50_000_000
    assert cfg["toggles"]["save_ci"] is False
    assert validate_config(cfg) == cfg                        # resolved configs re-validate unchanged


def test_sweep_product_and_paper_T(tmp_path):
    cfg = validate_config(deep_merge(base_raw_config(tmp_path), {
        "simulation": {"T": 9, "T_mode": "paper"},
        "sweep": {"q_scale": [0.0, 1.0], "N": [100, 1000], "T": [5, 9]}}))
    points = sweep_points(cfg)
    assert len(points) == 8
    assert {p["T"] for p in points} == {9, 17}                # 2T - 1
    assert points[0] == {"index": 0, "q_scale": 0.0, "N": 100, "T_input": 5, "T": 9, "dt": "auto"}


@pytest.mark.parametrize("name", ["example_small.yaml", "paper_validation.yaml", "server_large.yaml"])
def test_shipped_configs_validate(name):
    cfg = load_config(CONFIGS / name)
    assert validate_config(cfg) == cfg
