"""Autodata harness configuration: defaults, strict key checking, scenario corpus loading.

Every block has defaults (DEFAULTS below). A user config overrides any subset; unknown keys are
errors that name the dotted key, in the same spirit as fptsim's validator.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


class AutodataConfigError(ValueError):
    pass


DEFAULTS: dict = {
    "run_name": "autodata_run",
    "output_dir": None,                    # default results/<run_name>
    "seed": 0,                             # root of every instance / noise / solver seed
    "scenarios": "configs/autodata_scenarios.yaml",   # path (relative to the config file) or inline list
    "scenario_ids": None,                  # optional subset of scenario ids to run
    "toggles": {
        "llm_challenger": False,           # LLM proposes specs; False = built-in heuristic challenger
        "quality_verifier": True,          # run the verifier before spending solver compute
        "identifiability_check": True,     # verifier: Jacobian rank / condition number at the truth
        "crb_check": True,                 # verifier: Cramér-Rao bound says the noise is not overwhelming
        "lazy_strong": True,               # evaluate the strong solver only if the weak gate passed
        "save_observations": True,         # store per-attempt counts and estimates in the item .npz
        "resume": True,                    # skip scenarios whose result.json already exists
        "final_verifier": True,            # re-verify accepted items against the dataset (dedupe)
    },
    "loop": {"max_rounds": 12, "items_per_scenario": 1},
    "evaluation": {
        "attempts": 3,                     # independent noise draws; both solvers see the same draws
        "noise": "multinomial",            # multinomial | monte_carlo | exact
        "err_ref": 0.10,                   # score = exp(-rel_err / err_ref)
    },
    # Autodata's CS gates (weak_avg <= 65 %, max_weak <= 75 %, strong_avg >= 60 %, gap >= 20 points).
    # strong_avg_max = 0.95 in the paper (keeps rubric headroom); off by default here because an
    # inverse instance that the strong solver nails is still a good instance.
    "gates": {"weak_avg_max": 0.65, "weak_max_max": 0.75, "weak_min_min": 0.0,
              "strong_avg_min": 0.60, "strong_avg_max": 1.0, "gap_min": 0.20},
    "solvers": {
        "weak": {"solver": "lsq", "T": 4, "restarts": 0, "max_iter": 60, "prior_weight": 0.1},
        "strong": {"solver": "lsq", "T": None, "restarts": 2, "max_iter": 200},
    },
    "verifier": {"min_unknowns": 1, "min_hit_mass": 0.02, "max_jac_cond": 1.0e8, "max_crb_rel": 0.10},
    # Guardrails the challenger cannot change (Autodata's agents learned to cheat, e.g. by weakening
    # the weak solver; here solvers, gates, seeds and limits live outside the challenger's reach).
    "limits": {"max_n": 30, "max_k": 12, "max_T": 80, "min_N": 100, "max_N": 1_000_000},
    "llm": {"backend": "auto", "model": "claude-opus-5-5", "effort": "medium", "timeout": 300},
    "strategy": None,                      # path or inline mapping overriding challenger.DEFAULT_STRATEGY
    "meta": {
        "iterations": 10,
        "temperature": 0.1,                # Boltzmann parent selection P(c) ~ exp(score_c / T)
        "train_fraction": 0.5,             # scenarios split into train / validation by order
        "minibatch": 3,                    # train scenarios per analysis step
        "eval_rounds": 4,                  # round budget per scenario when scoring a harness
        "llm_implementer": False,          # LLM analyzer + implementer; False = rule-based mutation
    },
}


def _merge(base: dict, override: dict, where: str) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        path = f"{where}.{key}" if where else key
        if key not in base:
            raise AutodataConfigError(f"{path}: unknown key (allowed: {sorted(base)})")
        if isinstance(base[key], dict) and key not in ("solvers",):
            if not isinstance(value, dict):
                raise AutodataConfigError(f"{path}: must be a mapping")
            out[key] = _merge(base[key], value, path)
        elif key == "solvers":
            if not isinstance(value, dict) or set(value) - {"weak", "strong"}:
                raise AutodataConfigError(f"{path}: must map weak/strong to solver settings")
            for role, sc in value.items():
                if not isinstance(sc, dict) or "solver" not in sc:
                    raise AutodataConfigError(f"{path}.{role}: needs a `solver` name")
                out[key][role] = copy.deepcopy(sc)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _num(cfg, path, lo=None, hi=None, kind=float):
    node = cfg
    for part in path.split("."):
        node = node[part]
    try:
        val = kind(float(node)) if kind is int else float(node)
    except (TypeError, ValueError):
        raise AutodataConfigError(f"{path}: must be a number, got {node!r}") from None
    if (lo is not None and val < lo) or (hi is not None and val > hi):
        raise AutodataConfigError(f"{path}: {val} outside [{lo}, {hi}]")
    parent = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = val


def validate(raw: dict | None, base_dir: Path | None = None) -> dict:
    cfg = _merge(DEFAULTS, raw or {}, "")
    if cfg["output_dir"] is None:
        cfg["output_dir"] = f"results/{cfg['run_name']}"
    for path, lo, hi, kind in [("seed", 0, 2 ** 63 - 1, int), ("loop.max_rounds", 1, 10_000, int),
                               ("loop.items_per_scenario", 1, 10_000, int),
                               ("evaluation.attempts", 1, 1000, int), ("evaluation.err_ref", 1e-9, None, float),
                               ("limits.max_n", 2, 100_000, int), ("limits.max_k", 1, 100_000, int),
                               ("limits.max_T", 1, 100_000, int), ("limits.min_N", 1, None, int),
                               ("limits.max_N", 1, None, int), ("meta.iterations", 0, 100_000, int),
                               ("meta.temperature", 1e-9, None, float), ("meta.train_fraction", 0.0, 1.0, float),
                               ("meta.minibatch", 1, 10_000, int), ("meta.eval_rounds", 1, 10_000, int)]:
        _num(cfg, path, lo, hi, kind)
    for key in cfg["gates"]:
        _num(cfg, f"gates.{key}", 0.0, 1.0)
    # YAML 1.1 reads 1e8 / 1.0e8 as strings; coerce every verifier threshold to a number
    _num(cfg, "verifier.min_unknowns", 0, None, int)
    for key in ("min_hit_mass", "max_jac_cond", "max_crb_rel"):
        _num(cfg, f"verifier.{key}", 0.0, None)
    if cfg["evaluation"]["noise"] not in ("multinomial", "monte_carlo", "exact"):
        raise AutodataConfigError("evaluation.noise: multinomial | monte_carlo | exact")
    from .solvers import SOLVERS
    for role, sc in cfg["solvers"].items():
        if sc["solver"] not in SOLVERS:
            raise AutodataConfigError(f"solvers.{role}.solver: unknown {sc['solver']!r}; have {sorted(SOLVERS)}")
    cfg["scenarios"] = load_scenarios(cfg["scenarios"], base_dir)
    if cfg["scenario_ids"]:
        wanted = [str(s) for s in cfg["scenario_ids"]]
        known = {s["id"] for s in cfg["scenarios"]}
        missing = [s for s in wanted if s not in known]
        if missing:
            raise AutodataConfigError(f"scenario_ids: unknown {missing}")
        cfg["scenarios"] = [s for s in cfg["scenarios"] if s["id"] in wanted]
    if isinstance(cfg["strategy"], str):
        path = Path(cfg["strategy"])
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        cfg["strategy"] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def load_scenarios(value, base_dir: Path | None = None) -> list[dict]:
    """A scenario is a grounding brief for the challenger: {id, brief, base_spec, families?}."""
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            candidates = [base_dir / path] if base_dir is not None else []
            candidates += [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
            path = next((c for c in candidates if c.exists()), candidates[0] if candidates else path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        value = data.get("scenarios") if isinstance(data, dict) else data
    if not isinstance(value, list) or not value:
        raise AutodataConfigError("scenarios: need a nonempty list (or a YAML file with `scenarios:`)")
    out, seen = [], set()
    for i, s in enumerate(value):
        if not isinstance(s, dict) or "id" not in s or "base_spec" not in s:
            raise AutodataConfigError(f"scenarios[{i}]: needs `id` and `base_spec`")
        extra = set(s) - {"id", "brief", "base_spec", "families"}
        if extra:
            raise AutodataConfigError(f"scenarios[{i}]: unknown keys {sorted(extra)}")
        sid = str(s["id"])
        if sid in seen:
            raise AutodataConfigError(f"scenarios[{i}]: duplicate id {sid!r}")
        seen.add(sid)
        out.append({"id": sid, "brief": str(s.get("brief", "")), "base_spec": s["base_spec"],
                    "families": list(s.get("families") or [])})
    return out


def load(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return validate(raw, path.resolve().parent)
