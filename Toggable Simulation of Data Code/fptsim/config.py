"""Strict YAML configuration: schema, defaults, validation, sweeps, and parameter sampling.

Every key is checked. Unknown keys, wrong types and out-of-range values raise ConfigError
with the dotted path of the offending key. The resolved config (all defaults filled in)
is a plain nested dict that round-trips through YAML unchanged.

Parameter specs used by graph families and a few blocks:
    5            fixed value
    [2, 6]       inclusive range; ints are drawn uniformly from {2,...,6}, floats from U(2, 6)
    {choice: [8, 12, 20]}   uniform choice from a list
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


class ConfigError(ValueError):
    """Invalid configuration; the message names the offending key."""


# name -> {param: (kind, low, high)}; `pattern` takes a `patterns` list instead.
FAMILY_PARAMS: dict[str, dict[str, tuple[str, float, float]]] = {
    "grid": {"rows": ("int", 1, 10_000), "cols": ("int", 1, 10_000)},
    "grid_holes": {"rows": ("int", 1, 10_000), "cols": ("int", 1, 10_000),
                   "hole_frac": ("float", 0.0, 0.9)},
    "pattern": {},
    "ladder": {"length": ("int", 1, 100_000)},
    "path": {"n": ("int", 2, 100_000)},
    "cycle": {"n": ("int", 3, 100_000)},
    "star": {"n": ("int", 2, 100_000)},
    "complete": {"n": ("int", 2, 100_000)},
    "random_tree": {"n": ("int", 2, 100_000)},
    "erdos_renyi": {"n": ("int", 2, 100_000), "p": ("float", 0.0, 1.0)},
    "random_geometric": {"n": ("int", 2, 100_000), "radius": ("float", 0.0, 2.0)},
    "watts_strogatz": {"n": ("int", 3, 100_000), "k": ("int", 2, 100_000),
                       "beta": ("float", 0.0, 1.0)},
    "barabasi_albert": {"n": ("int", 2, 100_000), "m": ("int", 1, 100_000)},
    "random_regular": {"n": ("int", 2, 100_000), "d": ("int", 1, 100_000)},
}
LATTICE_FAMILIES = {"grid", "grid_holes", "pattern", "ladder"}
PRESET_NAMES = ("paper8", "paper9")

TOP_KEYS = {"run_name", "output_dir", "master_seed", "num_graphs", "device", "backend", "devices",
            "max_batch_walks", "cpu_batch_walks", "shard_size", "graphs_per_batch",
            "log_interval_s", "plot_sample", "toggles", "graphs", "simulation", "sweep"}
DEFAULT_TOGGLES = {"compute_exact": True, "run_monte_carlo": True, "save_counts": True,
                   "save_r_tensors": True, "save_ci": False, "convergence_checkpoints": False,
                   "make_plots": False, "resume": True}
GRAPH_KEYS = {"families", "presets", "weights", "mu", "B", "q", "dt", "max_retries"}
W_DISTS = {"constant": {"value"}, "uniform": {"low", "high"}, "lognormal": {"mean", "sigma"},
           "discrete": {"values", "probs"}, "gamma": {"shape", "scale"}}
MU_MODES = {"constant": {"value"}, "deg": set(), "weighted_deg": set(),
            "uniform": {"low", "high"}, "slack": {"a", "b"}}
B_MODES = {"random_k": {"k", "frac"}, "boundary": set(), "leaves": set(),
           "low_degree": {"k"}, "high_degree": {"k"}, "explicit": {"vertices"}}
Q_MODES = {"zero": set(), "constant": {"c"}, "uniform": {"low", "high"},
           "sparse": {"fraction", "low", "high"},
           "localized": {"center", "decay", "amplitude"}, "explicit": {"values"}}
SIM_KEYS = {"T", "T_mode", "N", "replicates", "N_checkpoints", "chunk_walks"}
SWEEP_KEYS = ("q_scale", "N", "T", "dt")
DEFAULT_CHECKPOINTS = [10 ** e for e in range(2, 13)]

_NUMERIC = re.compile(r"^[+-]?(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?$")


# ---------------------------------------------------------------- scalar checks
def _number(value: Any, where: str) -> int | float:
    # YAML 1.1 reads "1e6" as a string, so numeric strings are accepted too.
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a number, got a boolean")
    if isinstance(value, (int, float)):
        x = value
    elif isinstance(value, str) and _NUMERIC.match(value.strip()):
        x = float(value.strip().replace("_", ""))
    else:
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if isinstance(x, float) and not math.isfinite(x):
        raise ConfigError(f"{where}: must be finite, got {value!r}")
    return x


def _bounds(x, where, lo, hi, lo_open=False, hi_open=False):
    if lo is not None and (x < lo or (lo_open and x == lo)):
        raise ConfigError(f"{where}: must be {'>' if lo_open else '>='} {lo}, got {x}")
    if hi is not None and (x > hi or (hi_open and x == hi)):
        raise ConfigError(f"{where}: must be {'<' if hi_open else '<='} {hi}, got {x}")
    return x


def _int(value, where, lo=None, hi=None) -> int:
    x = _number(value, where)
    if isinstance(x, float):
        if not x.is_integer():
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        x = int(x)
    return int(_bounds(int(x), where, lo, hi))


def _float(value, where, lo=None, hi=None, lo_open=False, hi_open=False) -> float:
    return float(_bounds(float(_number(value, where)), where, lo, hi, lo_open, hi_open))


def _bool(value, where) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false, got {value!r}")
    return value


def _str(value, where) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: expected a non-empty string, got {value!r}")
    return value.strip()


def _choice(value, where, options) -> str:
    if value not in options:
        raise ConfigError(f"{where}: must be one of {sorted(options)}, got {value!r}")
    return value


def _mapping(value, where) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(d: dict, allowed, where: str) -> None:
    extra = sorted(str(k) for k in set(d) - set(allowed))
    if extra:
        raise ConfigError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def _list(value, where, nonempty=True) -> list:
    if not isinstance(value, list):
        raise ConfigError(f"{where}: expected a list, got {value!r}")
    if nonempty and not value:
        raise ConfigError(f"{where}: must not be empty")
    return value


def _spec(value, where, kind, lo, hi):
    """Parse a fixed value, an inclusive [low, high] range, or {choice: [...]}."""
    conv = _int if kind == "int" else _float
    if isinstance(value, dict):
        _check_keys(value, {"choice"}, where)
        options = _list(value.get("choice"), f"{where}.choice")
        return {"choice": [conv(v, f"{where}.choice[{i}]", lo, hi) for i, v in enumerate(options)]}
    if isinstance(value, list):
        if len(value) != 2:
            raise ConfigError(f"{where}: a range must be [low, high]; use {{choice: [...]}} for a list")
        a, b = conv(value[0], f"{where}[0]", lo, hi), conv(value[1], f"{where}[1]", lo, hi)
        if a > b:
            raise ConfigError(f"{where}: range low {a} exceeds high {b}")
        return [a, b]
    return conv(value, where, lo, hi)


def sample_spec(rng: np.random.Generator, spec, kind: str):
    """Draw one value from a spec produced by _spec."""
    if isinstance(spec, dict):
        options = spec["choice"]
        return options[int(rng.integers(len(options)))]
    if isinstance(spec, list):
        a, b = spec
        if kind == "int":
            return int(rng.integers(a, b + 1))
        return float(rng.uniform(a, b)) if b > a else float(a)
    return spec


def spec_max(spec):
    if isinstance(spec, dict):
        return max(spec["choice"])
    if isinstance(spec, list):
        return spec[1]
    return spec


def _dt(value, where):
    if value == "auto":
        return "auto"
    return _float(value, where, 0.0, 1.0, lo_open=True)


# ---------------------------------------------------------------- blocks
def _families(value) -> dict:
    fams = _mapping(value, "graphs.families")
    if not fams:
        raise ConfigError("graphs.families: at least one family is required")
    out = {}
    for name, spec in fams.items():
        where = f"graphs.families.{name}"
        if name not in FAMILY_PARAMS:
            raise ConfigError(f"{where}: unknown family; choose from {sorted(FAMILY_PARAMS)}")
        spec = _mapping(spec, where)
        params = FAMILY_PARAMS[name]
        allowed = {"weight", *params} | ({"patterns"} if name == "pattern" else set())
        _check_keys(spec, allowed, where)
        entry = {"weight": _float(spec.get("weight", 1.0), f"{where}.weight", 0.0)}
        for key, (kind, lo, hi) in params.items():
            if key not in spec:
                raise ConfigError(f"{where}.{key}: required")
            entry[key] = _spec(spec[key], f"{where}.{key}", kind, lo, hi)
        if name == "pattern":
            entry["patterns"] = _patterns(spec.get("patterns"), f"{where}.patterns")
        out[name] = entry
    if sum(e["weight"] for e in out.values()) <= 0:
        raise ConfigError("graphs.families: the sampling weights must not all be zero")
    return out


def _patterns(value, where) -> list[list[str]]:
    items = _list(value, where)
    if all(isinstance(s, str) for s in items):
        items = [items]                       # a single pattern was given
    out = []
    for i, pat in enumerate(items):
        pat = _list(pat, f"{where}[{i}]")
        if not all(isinstance(s, str) for s in pat):
            raise ConfigError(f"{where}[{i}]: a pattern is a list of strings such as ['XooX', 'XXXX']")
        if sum(s.upper().count("X") for s in pat) < 2:
            raise ConfigError(f"{where}[{i}]: a pattern needs at least two 'X' vertices")
        out.append([str(s) for s in pat])
    return out


def _weights(value) -> dict:
    w = _mapping(value, "graphs.weights")
    where = "graphs.weights"
    dist = _choice(w.get("dist", "constant"), f"{where}.dist", W_DISTS)
    _check_keys(w, {"dist", "rescale"} | W_DISTS[dist], where)
    out: dict[str, Any] = {"dist": dist}
    if dist == "constant":
        out["value"] = _float(w.get("value", 1.0), f"{where}.value", 0.0, lo_open=True)
    elif dist == "uniform":
        for key in ("low", "high"):
            if key not in w:
                raise ConfigError(f"{where}.{key}: required for dist=uniform")
        out["low"] = _float(w["low"], f"{where}.low", 0.0, lo_open=True)
        out["high"] = _float(w["high"], f"{where}.high", out["low"])
    elif dist == "lognormal":
        out["mean"] = _float(w.get("mean", 0.0), f"{where}.mean")
        out["sigma"] = _float(w.get("sigma", 0.5), f"{where}.sigma", 0.0)
    elif dist == "discrete":
        values = _list(w.get("values"), f"{where}.values")
        out["values"] = [_float(v, f"{where}.values[{i}]", 0.0, lo_open=True) for i, v in enumerate(values)]
        probs = w.get("probs")
        if probs is None:
            out["probs"] = None
        else:
            probs = [_float(p, f"{where}.probs[{i}]", 0.0) for i, p in enumerate(_list(probs, f"{where}.probs"))]
            if len(probs) != len(values):
                raise ConfigError(f"{where}.probs: needs {len(values)} entries, one per value")
            if abs(sum(probs) - 1) > 1e-9:
                raise ConfigError(f"{where}.probs: must sum to 1, got {sum(probs)}")
            out["probs"] = probs
    else:  # gamma
        out["shape"] = _float(w.get("shape", 2.0), f"{where}.shape", 0.0, lo_open=True)
        out["scale"] = _float(w.get("scale", 0.5), f"{where}.scale", 0.0, lo_open=True)
    rescale = w.get("rescale")
    if rescale is None:
        out["rescale"] = None
    else:
        rescale = _mapping(rescale, f"{where}.rescale")
        _check_keys(rescale, {"mode", "value"}, f"{where}.rescale")
        out["rescale"] = {"mode": _choice(rescale.get("mode", "max"), f"{where}.rescale.mode", {"max", "mean"}),
                          "value": _float(rescale.get("value", 1.0), f"{where}.rescale.value", 0.0, lo_open=True)}
    return out


def _mu(value) -> dict:
    m = _mapping(value, "graphs.mu")
    where = "graphs.mu"
    mode = _choice(m.get("mode", "deg"), f"{where}.mode", MU_MODES)
    _check_keys(m, {"mode", "on_violation"} | MU_MODES[mode], where)
    out: dict[str, Any] = {"mode": mode,
                           "on_violation": _choice(m.get("on_violation", "fix"), f"{where}.on_violation",
                                                   {"fix", "resample", "skip"})}
    if mode == "constant":
        out["value"] = _float(m.get("value", 1.0), f"{where}.value", 0.0, lo_open=True)
    elif mode == "uniform":
        for key in ("low", "high"):
            if key not in m:
                raise ConfigError(f"{where}.{key}: required for mode=uniform")
        out["low"] = _float(m["low"], f"{where}.low", 0.0, lo_open=True)
        out["high"] = _float(m["high"], f"{where}.high", out["low"])
    elif mode == "slack":
        out["a"] = _float(m.get("a", 0.0), f"{where}.a", 0.0)
        out["b"] = _float(m.get("b", 0.5), f"{where}.b", out["a"])
    return out


def _B(value) -> dict:
    if value is None:
        raise ConfigError("graphs.B: required (for example {mode: random_k, k: [2, 5]})")
    b = _mapping(value, "graphs.B")
    where = "graphs.B"
    mode = _choice(b.get("mode"), f"{where}.mode", B_MODES)
    extra = {"max_k"} if mode != "explicit" else set()
    _check_keys(b, {"mode"} | B_MODES[mode] | extra, where)
    out: dict[str, Any] = {"mode": mode,
                           "max_k": _int(b["max_k"], f"{where}.max_k", 1) if b.get("max_k") is not None else None}
    if mode == "random_k":
        has_k, has_frac = b.get("k") is not None, b.get("frac") is not None
        if has_k == has_frac:
            raise ConfigError(f"{where}: random_k needs exactly one of k or frac")
        out["k"] = _spec(b["k"], f"{where}.k", "int", 1, 10 ** 6) if has_k else None
        out["frac"] = _float(b["frac"], f"{where}.frac", 0.0, 1.0, lo_open=True) if has_frac else None
    elif mode in ("low_degree", "high_degree"):
        if "k" not in b:
            raise ConfigError(f"{where}.k: required for mode={mode}")
        out["k"] = _spec(b["k"], f"{where}.k", "int", 1, 10 ** 6)
    elif mode == "explicit":
        verts = _list(b.get("vertices"), f"{where}.vertices")
        verts = [_int(v, f"{where}.vertices[{i}]", 0) for i, v in enumerate(verts)]
        if len(set(verts)) != len(verts):
            raise ConfigError(f"{where}.vertices: contains duplicates")
        out["vertices"] = verts
    return out


def _q(value) -> dict:
    q = _mapping(value, "graphs.q")
    where = "graphs.q"
    mode = _choice(q.get("mode", "zero"), f"{where}.mode", Q_MODES)
    _check_keys(q, {"mode", "on_B_only", "off_B_only"} | Q_MODES[mode], where)
    out: dict[str, Any] = {"mode": mode,
                           "on_B_only": _bool(q.get("on_B_only", False), f"{where}.on_B_only"),
                           "off_B_only": _bool(q.get("off_B_only", False), f"{where}.off_B_only")}
    if out["on_B_only"] and out["off_B_only"]:
        raise ConfigError(f"{where}: on_B_only and off_B_only cannot both be true")
    if mode == "constant":
        out["c"] = _float(q.get("c", 0.0), f"{where}.c", 0.0)
    elif mode in ("uniform", "sparse"):
        out["low"] = _float(q.get("low", 0.0), f"{where}.low", 0.0)
        out["high"] = _float(q.get("high", 0.5), f"{where}.high", out["low"])
        if mode == "sparse":
            out["fraction"] = _float(q.get("fraction", 0.2), f"{where}.fraction", 0.0, 1.0)
    elif mode == "localized":
        center = q.get("center", "random")
        out["center"] = "random" if center == "random" else _int(center, f"{where}.center", 0)
        out["decay"] = _float(q.get("decay", 1.0), f"{where}.decay", 0.0, lo_open=True)
        out["amplitude"] = _float(q.get("amplitude", 0.5), f"{where}.amplitude", 0.0)
    elif mode == "explicit":
        values = _list(q.get("values"), f"{where}.values")
        out["values"] = [_float(v, f"{where}.values[{i}]", 0.0) for i, v in enumerate(values)]
    return out


def _graphs(value) -> dict:
    if value is None:
        raise ConfigError("graphs: required")
    g = _mapping(value, "graphs")
    _check_keys(g, GRAPH_KEYS, "graphs")
    out: dict[str, Any] = {}
    presets = g.get("presets")
    if presets is not None:
        presets = _list(presets, "graphs.presets")
        for i, name in enumerate(presets):
            _choice(name, f"graphs.presets[{i}]", PRESET_NAMES)
        for key in ("families", "weights", "mu", "B"):
            if g.get(key) is not None:
                raise ConfigError(f"graphs.{key}: not allowed together with graphs.presets "
                                  "(a preset fixes W, mu and B)")
        out.update(presets=list(presets), families=None, weights=None, mu=None, B=None)
    else:
        if g.get("families") is None:
            raise ConfigError("graphs.families: required (or give graphs.presets)")
        out["presets"] = None
        out["families"] = _families(g["families"])
        out["weights"] = _weights(g.get("weights") or {"dist": "constant", "value": 1.0})
        out["mu"] = _mu(g.get("mu") or {"mode": "deg"})
        out["B"] = _B(g.get("B"))
    out["q"] = _q(g.get("q", {"mode": "zero"}))
    out["dt"] = _dt(g.get("dt", "auto"), "graphs.dt")
    out["max_retries"] = _int(g.get("max_retries", 100), "graphs.max_retries", 1, 100_000)
    return out


def _simulation(value) -> dict:
    if value is None:
        raise ConfigError("simulation: required")
    s = _mapping(value, "simulation")
    _check_keys(s, SIM_KEYS, "simulation")
    for key in ("T", "N"):
        if key not in s:
            raise ConfigError(f"simulation.{key}: required")
    out = {"T": _int(s["T"], "simulation.T", 1, 1_000_000),
           "T_mode": _choice(s.get("T_mode", "steps"), "simulation.T_mode", {"steps", "paper"}),
           "N": _int(s["N"], "simulation.N", 1, 10 ** 12),
           "replicates": _int(s.get("replicates", 1), "simulation.replicates", 1, 1_000_000),
           "chunk_walks": _int(s.get("chunk_walks", 100_000), "simulation.chunk_walks", 1, 10 ** 9)}
    cps = s.get("N_checkpoints")
    if cps is None:
        out["N_checkpoints"] = None
    else:
        cps = [_int(c, f"simulation.N_checkpoints[{i}]", 1, 10 ** 12)
               for i, c in enumerate(_list(cps, "simulation.N_checkpoints"))]
        out["N_checkpoints"] = sorted(set(cps))
    return out


def _toggles(value) -> dict:
    t = _mapping(value, "toggles")
    _check_keys(t, DEFAULT_TOGGLES, "toggles")
    return {k: _bool(t.get(k, d), f"toggles.{k}") for k, d in DEFAULT_TOGGLES.items()}


def _sweep(value) -> dict | None:
    if value is None:
        return None
    s = _mapping(value, "sweep")
    _check_keys(s, SWEEP_KEYS, "sweep")
    if not s:
        return None
    out = {}
    for key, items in s.items():
        items = _list(items, f"sweep.{key}")
        where = f"sweep.{key}"
        if key == "q_scale":
            out[key] = [_float(v, f"{where}[{i}]", 0.0) for i, v in enumerate(items)]
        elif key == "N":
            out[key] = [_int(v, f"{where}[{i}]", 1, 10 ** 12) for i, v in enumerate(items)]
        elif key == "T":
            out[key] = [_int(v, f"{where}[{i}]", 1, 1_000_000) for i, v in enumerate(items)]
        else:
            out[key] = [_dt(v, f"{where}[{i}]") for i, v in enumerate(items)]
    return out


# ---------------------------------------------------------------- top level
def validate_config(raw: Any) -> dict:
    """Validate a raw config mapping and return the resolved config with defaults."""
    raw = _mapping(raw, "config")
    _check_keys(raw, TOP_KEYS, "config")
    cfg: dict[str, Any] = {}
    cfg["run_name"] = _str(raw.get("run_name", "fptsim_run"), "run_name")
    cfg["output_dir"] = _str(raw.get("output_dir", f"results/{cfg['run_name']}"), "output_dir")
    cfg["master_seed"] = _int(raw.get("master_seed", 0), "master_seed", 0, 2 ** 63 - 1)
    if "num_graphs" not in raw:
        raise ConfigError("num_graphs: required")
    cfg["num_graphs"] = _int(raw["num_graphs"], "num_graphs", 1, 10 ** 9)
    device = raw.get("device", "auto")
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(:\d+)?", device):
        raise ConfigError(f"device: must be auto, cpu, cuda or cuda:<index>, got {device!r}")
    cfg["device"] = device
    cfg["backend"] = _choice(raw.get("backend", "auto"), "backend", {"auto", "torch", "numpy"})
    devices = raw.get("devices")
    if devices is not None:
        devices = [_int(d, f"devices[{i}]", 0, 1023) for i, d in enumerate(_list(devices, "devices"))]
        if len(set(devices)) != len(devices):
            raise ConfigError("devices: contains duplicates")
    cfg["devices"] = devices
    cfg["max_batch_walks"] = _int(raw.get("max_batch_walks", 50_000_000), "max_batch_walks", 1)
    cfg["cpu_batch_walks"] = _int(raw.get("cpu_batch_walks", 4_000_000), "cpu_batch_walks", 1)
    cfg["shard_size"] = _int(raw.get("shard_size", 500), "shard_size", 1, 10 ** 7)
    cfg["graphs_per_batch"] = _int(raw.get("graphs_per_batch", 64), "graphs_per_batch", 1, 10 ** 6)
    cfg["log_interval_s"] = _float(raw.get("log_interval_s", 30.0), "log_interval_s", 0.0, lo_open=True)
    cfg["plot_sample"] = _int(raw.get("plot_sample", 5), "plot_sample", 0)
    cfg["toggles"] = _toggles(raw.get("toggles"))
    cfg["graphs"] = _graphs(raw.get("graphs"))
    cfg["simulation"] = _simulation(raw.get("simulation"))
    cfg["sweep"] = _sweep(raw.get("sweep"))
    _cross_checks(cfg)
    return cfg


def _cross_checks(cfg: dict) -> None:
    t, g = cfg["toggles"], cfg["graphs"]
    if not (t["compute_exact"] or t["run_monte_carlo"]):
        raise ConfigError("toggles: at least one of compute_exact or run_monte_carlo must be true")
    if g["B"] is not None and g["B"]["mode"] == "boundary":
        bad = [n for n, e in g["families"].items() if e["weight"] > 0 and n not in LATTICE_FAMILIES]
        if bad:
            raise ConfigError(f"graphs.B.mode: 'boundary' is only defined for lattice families "
                              f"{sorted(LATTICE_FAMILIES)}; these families have no boundary: {bad}")
    if g["families"] is not None:
        for name, entry in g["families"].items():
            if name == "watts_strogatz" and spec_max(entry["k"]) >= spec_max(entry["n"]):
                raise ConfigError("graphs.families.watts_strogatz.k: must be smaller than n")
    if g["presets"] is not None and g["q"]["mode"] == "explicit":
        sizes = {"paper8": 8, "paper9": 9}
        for name in g["presets"]:
            if len(g["q"]["values"]) != sizes[name]:
                raise ConfigError(f"graphs.q.values: preset {name} has {sizes[name]} vertices, "
                                  f"got {len(g['q']['values'])} values")


def load_config(path: str | Path) -> dict:
    import yaml
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return validate_config(raw)


def dump_config(cfg: dict) -> str:
    import yaml
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)


def science_view(cfg: dict) -> dict:
    """The settings that change results. A resumed run must match these exactly.

    num_graphs is left out on purpose: graph i depends only on (master_seed, i), so a run
    can be extended with more graphs.
    """
    t = cfg.get("toggles", {})
    return {"master_seed": cfg.get("master_seed"), "shard_size": cfg.get("shard_size"),
            "graphs": cfg.get("graphs"), "simulation": cfg.get("simulation"),
            "sweep": cfg.get("sweep"),
            "toggles": {k: t.get(k) for k in ("compute_exact", "run_monte_carlo",
                                              "convergence_checkpoints")}}


# ---------------------------------------------------------------- derived settings
def effective_T(T: int, mode: str) -> int:
    """steps: T is the maximum step. paper: the paper's T maps to 2T-1 steps."""
    return 2 * T - 1 if mode == "paper" else T


def sweep_points(cfg: dict) -> list[dict]:
    """Cartesian product of the sweep lists, in the order q_scale x N x T x dt."""
    sim, sw = cfg["simulation"], cfg["sweep"] or {}
    axes = [sw.get("q_scale", [1.0]), sw.get("N", [sim["N"]]),
            sw.get("T", [sim["T"]]), sw.get("dt", [cfg["graphs"]["dt"]])]
    points = []
    for i, (qs, N, T, dt) in enumerate(itertools.product(*axes)):
        points.append({"index": i, "q_scale": float(qs), "N": int(N), "T_input": int(T),
                       "T": effective_T(int(T), sim["T_mode"]), "dt": dt})
    return points


def checkpoints_for(cfg: dict, N: int) -> list[int]:
    """Prefix sizes at which convergence is recorded; always ends with N."""
    if not cfg["toggles"]["convergence_checkpoints"]:
        return [N]
    cps = cfg["simulation"]["N_checkpoints"] or DEFAULT_CHECKPOINTS
    return sorted({c for c in cps if c < N} | {N})
