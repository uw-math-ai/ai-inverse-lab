"""Defaults and strict merging of the website's input JSON (the platform does not insert schema defaults)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

SCENARIOS_FILE = Path(__file__).resolve().parent / "scenarios.json"


class ConfigError(ValueError):
    pass


DEFAULTS = {
    "seed": 0,
    "toggles": {"quality_verifier": True, "identifiability_check": True, "crb_check": True,
                "lazy_strong": True, "save_observations": True, "final_verifier": True},
    "loop": {"max_rounds": 12, "items_per_scenario": 1},
    "evaluation": {"attempts": 3, "noise": "multinomial", "err_ref": 0.10},
    "gates": {"weak_avg_max": 0.65, "weak_max_max": 0.75, "weak_min_min": 0.0,
              "strong_avg_min": 0.60, "strong_avg_max": 1.0, "gap_min": 0.20},
    "solvers": {"weak": {"solver": "lsq", "T": 4, "restarts": 0, "max_iter": 60, "prior_weight": 0.1},
                "strong": {"solver": "lsq", "T": None, "restarts": 1, "max_iter": 100}},
    # 1e6 (not 1e8 as in the NumPy harness): the pure-Python check squares the condition number
    "verifier": {"min_unknowns": 1, "min_hit_mass": 0.02, "max_jac_cond": 1.0e6, "max_crb_rel": 0.10},
    "limits": {"max_n": 30, "max_k": 12, "max_T": 80, "min_N": 100, "max_N": 1_000_000},
    "strategy": None,
    "meta": {"iterations": 6, "temperature": 0.1, "train_fraction": 0.5, "minibatch": 3, "eval_rounds": 4},
}
BLOCKS = ("toggles", "loop", "evaluation", "gates", "verifier", "limits", "meta")
RUNTIME_KEYS = {"mode", "scenario_ids", "skip_scenarios", "custom_scenarios", "time_budget_seconds",
                "parallel", "llm_meta", "max_rounds", "items_per_scenario", "attempts", "noise", "err_ref"}


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_FILE.read_text(encoding="utf-8"))["scenarios"]


def _check_scenarios(value, where):
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: a nonempty list")
    out, seen = [], set()
    for i, s in enumerate(value):
        if not isinstance(s, dict) or "id" not in s or "base_spec" not in s:
            raise ConfigError(f"{where}[{i}]: needs `id` and `base_spec`")
        extra = set(s) - {"id", "brief", "base_spec", "families"}
        if extra:
            raise ConfigError(f"{where}[{i}]: unknown keys {sorted(extra)}")
        sid = str(s["id"])
        if sid in seen:
            raise ConfigError(f"{where}[{i}]: duplicate id {sid!r}")
        seen.add(sid)
        out.append({"id": sid, "brief": str(s.get("brief", "")), "base_spec": s["base_spec"],
                    "families": list(s.get("families") or [])})
    return out


def build(inputs: dict) -> dict:
    """Resolved config from the run's input JSON. Raises ConfigError naming the bad key."""
    if not isinstance(inputs, dict):
        raise ConfigError("inputs: must be a JSON object")
    cfg = copy.deepcopy(DEFAULTS)
    allowed = set(DEFAULTS) | RUNTIME_KEYS
    extra = set(inputs) - allowed
    if extra:
        raise ConfigError(f"inputs: unknown keys {sorted(extra)}")
    for key in BLOCKS:
        if key in inputs:
            if not isinstance(inputs[key], dict):
                raise ConfigError(f"{key}: must be an object")
            bad = set(inputs[key]) - set(DEFAULTS[key])
            if bad:
                raise ConfigError(f"{key}: unknown keys {sorted(bad)} (allowed: {sorted(DEFAULTS[key])})")
            cfg[key].update(inputs[key])
    if "solvers" in inputs:
        sv = inputs["solvers"]
        if not isinstance(sv, dict) or set(sv) - {"weak", "strong"}:
            raise ConfigError("solvers: an object with weak and/or strong")
        for role, sc in sv.items():
            if not isinstance(sc, dict):
                raise ConfigError(f"solvers.{role}: must be an object")
            bad = set(sc) - {"solver", "T", "restarts", "max_iter", "tol", "prior_weight"}
            if bad:
                raise ConfigError(f"solvers.{role}: unknown keys {sorted(bad)}")
            cfg["solvers"][role] = {**cfg["solvers"][role], **sc}
    for key in ("seed", "strategy"):
        if key in inputs:
            cfg[key] = inputs[key]
    for key, (block, name) in {"max_rounds": ("loop", "max_rounds"),
                               "items_per_scenario": ("loop", "items_per_scenario"),
                               "attempts": ("evaluation", "attempts"), "noise": ("evaluation", "noise"),
                               "err_ref": ("evaluation", "err_ref")}.items():
        if key in inputs:
            cfg[block][name] = inputs[key]
    from .solvers import SOLVERS
    for role, sc in cfg["solvers"].items():
        if sc.get("solver") not in SOLVERS:
            raise ConfigError(f"solvers.{role}.solver: unknown {sc.get('solver')!r}; have {sorted(SOLVERS)}")
    if cfg["evaluation"]["noise"] not in ("multinomial", "exact"):
        raise ConfigError("noise: multinomial | exact")
    for k, v in cfg["gates"].items():
        if not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ConfigError(f"gates.{k}: a number in [0, 1]")
    for k in ("max_jac_cond", "max_crb_rel", "min_hit_mass"):
        cfg["verifier"][k] = float(cfg["verifier"][k])
    scen = _check_scenarios(inputs["custom_scenarios"], "custom_scenarios") if inputs.get("custom_scenarios") \
        else load_scenarios()
    known = {s["id"] for s in scen}
    for key in ("scenario_ids", "skip_scenarios"):
        missing = [s for s in inputs.get(key) or [] if s not in known]
        if missing:
            raise ConfigError(f"{key}: unknown scenario ids {missing}; known: {sorted(known)}")
    if inputs.get("scenario_ids"):
        scen = [s for s in scen if s["id"] in set(inputs["scenario_ids"])]
    if inputs.get("skip_scenarios"):
        scen = [s for s in scen if s["id"] not in set(inputs["skip_scenarios"])]
    cfg["scenarios"] = scen
    cfg["mode"] = inputs.get("mode", "run")
    if cfg["mode"] not in ("run", "meta"):
        raise ConfigError("mode: run | meta")
    if cfg["mode"] == "meta" and len(scen) < 2:
        raise ConfigError("mode meta needs at least 2 scenarios (train + validation)")
    cfg["parallel"] = bool(inputs.get("parallel", False))
    cfg["llm_meta"] = bool(inputs.get("llm_meta", False))
    cfg["time_budget_seconds"] = inputs.get("time_budget_seconds")
    return cfg
