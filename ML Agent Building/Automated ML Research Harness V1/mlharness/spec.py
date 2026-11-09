"""Layer 1 — Research specification.

The user supplies a YAML/JSON file (or just a problem statement). ``compile_spec`` turns it into a
validated ``ResearchSpec``: defaults filled, task inferred from the simulator, metrics checked
against the task, hardware resolved, and explicit objectives + success criteria derived. The spec
is stored as the first artifact and every agent/critic checks its work against it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import metrics as M

DEFAULTS = {
    "name": "research",
    "problem": "",
    "task": None,
    "simulator": {"name": "friedman", "params": {}},
    "data": {"n_train": 2000, "n_val": 500, "n_test": 500, "seed": 0},
    "metrics": {"primary": None, "secondary": None},
    "success": {"metric": None, "threshold": None},
    "budget": {"compute_seconds": 600, "wall_seconds": None, "max_ideas": 40,
               "experiment_timeout": 180, "llm_calls": 30},
    "hardware": {"workers": "auto", "gpus": "auto", "threads_per_worker": "auto"},
    "constraints": {"allowed_methods": "all", "forbidden_methods": [], "allow_custom_code": True,
                    "notes": []},
    "search": {
        "ideators": ["classical", "ensembles", "neural", "domain"],
        "ideas_per_ideator": 4,
        "screen_fraction": 0.1,       # LeanML feasibility check on 10% of the training data
        "baseline_margin": 0.02,      # screen survivors must beat the mean baseline by >= 2% (relative)
        "rungs": [0.33, 1.0],         # successive-halving fidelities after the screen
        "eta": 3,                     # keep top 1/eta at each rung
        "min_full": 3,                # at least this many reach full fidelity
        "refine_rounds": 2,
        "refine_top_k": 3,
        "refine_children": 2,         # children per parent per round
        "patience": 1,
        "min_improvement": 0.005,     # relative score gain that counts as progress
        "ensemble": True,
        "ensemble_max": 5,
        "stop_on_success": False,
        "repair_attempts": 1,         # LLM repair attempts for crashing agent-written code
        "llm_critic": True,
    },
    "llm": {"backend": "auto", "model": None, "timeout": 300, "effort": "medium"},
    "seed": 0,
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class ResearchSpec:
    raw: dict
    name: str
    problem: str
    task: str
    simulator: dict
    data: dict
    metrics: dict
    success: dict
    budget: dict
    hardware: dict
    constraints: dict
    search: dict
    llm: dict
    seed: int
    objectives: list = field(default_factory=list)
    simulator_description: str = ""

    @property
    def primary(self) -> str:
        return self.metrics["primary"]

    @property
    def metric_names(self) -> list[str]:
        return [self.primary] + [m for m in self.metrics["secondary"] if m != self.primary]

    def meets_success(self, value) -> bool:
        return M.meets(value, self.success["metric"], self.success["threshold"])

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "problem", "task", "simulator", "data", "metrics",
                                               "success", "budget", "hardware", "constraints", "search",
                                               "llm", "seed", "objectives", "simulator_description")}

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:10]

    def brief(self) -> str:
        """Compact description given to every LLM-backed agent."""
        s = self.success
        crit = (f"{s['metric']} {'<=' if M.direction(s['metric']) == 'min' else '>='} {s['threshold']}"
                if s["threshold"] is not None else f"optimise {s['metric']} (no fixed threshold)")
        notes = "; ".join(self.constraints.get("notes") or []) or "none"
        return (f"RESEARCH OBJECTIVE: {self.problem.strip()}\n"
                f"Task: {self.task}. Data: {self.simulator_description}\n"
                f"Splits: train={self.data['n_train']} val={self.data['n_val']} test={self.data['n_test']} "
                f"(test is held out until the final evaluation).\n"
                f"Primary metric: {self.primary} ({M.direction(self.primary)}imise). Success: {crit}.\n"
                f"Compute budget: {self.budget['compute_seconds']} worker-seconds on "
                f"{self.hardware['workers']} workers, GPUs: {self.hardware['gpu_devices'] or 'none'}.\n"
                f"Constraints: allowed_methods={self.constraints['allowed_methods']}, "
                f"forbidden={self.constraints['forbidden_methods']}, notes: {notes}")


def load_raw(path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)


SPEC_SYSTEM = ("You translate a machine-learning research request into a structured research "
               "specification for an automated ML research harness. Reply with one JSON object only.")


def llm_fill_spec(raw: dict, llm) -> dict:
    """Ask an LLM to fill missing fields from the free-text problem statement."""
    from .llm import extract_json
    from .simulators import SIMULATOR_CATALOG
    prompt = (f"User request / partial spec:\n{json.dumps(raw, indent=1, default=str)}\n\n"
              f"Available simulators: {json.dumps(SIMULATOR_CATALOG, indent=1)}\n"
              f"Metrics: {list(M.METRICS)}\n"
              "Return JSON with keys: name, task (regression|classification), simulator {name, params}, "
              "data {n_train, n_val, n_test}, metrics {primary, secondary}, success {threshold}, "
              "constraints {notes: [..]}. Keep values the user already gave. For graph_fpt the params are "
              "graph{family: grid|path|cycle|complete, rows, cols, n}, observed (boundary|all|[vertices]), "
              "T, walks_per_pair, weights{low, high}, potential{mode: none|sparse|uniform, low, high, "
              "fraction}, target (weights|potential|both).")
    resp = llm.complete(SPEC_SYSTEM, prompt)
    filled = extract_json(resp.text) or {}
    return deep_merge(filled, raw)   # user-provided values win


def compile_spec(raw: dict, llm=None, hardware: dict | None = None) -> ResearchSpec:
    from .simulators import get_simulator
    raw = dict(raw or {})
    if llm is not None and not getattr(llm, "is_null", True) and "simulator" not in raw and raw.get("problem"):
        raw = llm_fill_spec(raw, llm)
    cfg = deep_merge(DEFAULTS, raw)

    sim = get_simulator(cfg["simulator"])
    task = cfg["task"] or sim.task
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be regression or classification, got {task!r}")
    if task != sim.task:
        raise ValueError(f"spec task {task!r} disagrees with simulator task {sim.task!r}")

    met = cfg["metrics"]
    met["primary"] = met.get("primary") or M.DEFAULT_PRIMARY[task]
    if met["secondary"] is None:
        met["secondary"] = M.DEFAULT_SECONDARY[task]
    for name in [met["primary"]] + list(met["secondary"]):
        if name not in M.METRICS:
            raise ValueError(f"unknown metric {name!r}; choose from {list(M.METRICS)}")
        if M.METRICS[name][2] != task:
            raise ValueError(f"metric {name!r} is for {M.METRICS[name][2]}, task is {task}")
    cfg["success"]["metric"] = cfg["success"].get("metric") or met["primary"]
    if cfg["success"]["metric"] not in [met["primary"]] + list(met["secondary"]):
        met["secondary"] = list(met["secondary"]) + [cfg["success"]["metric"]]

    # hardware
    from .resources import detect_hardware
    hw_info = hardware or detect_hardware()
    hw = cfg["hardware"]
    cpus = hw_info["cpus"]
    if hw["workers"] in (None, "auto"):
        hw["workers"] = max(1, min(8, cpus // 2 if cpus >= 4 else cpus))
    hw["workers"] = int(hw["workers"])
    if hw["threads_per_worker"] in (None, "auto"):
        hw["threads_per_worker"] = max(1, cpus // hw["workers"])
    if hw["gpus"] in (None, "auto"):
        gpus = hw_info["gpus"]
    else:
        gpus = hw_info["gpus"][: int(hw["gpus"])]
    accel = hw_info.get("accelerator", "cpu")
    hw["gpu_devices"] = ["mps"] if accel == "mps" else [f"cuda:{i}" for i in range(len(gpus))]
    hw["detected"] = {k: hw_info[k] for k in ("cpus", "gpus", "accelerator", "torch", "platform", "python")}

    b = cfg["budget"]
    if b.get("wall_seconds") is None:
        # parallel compute time x3 for stragglers, plus headroom for LLM latency (~30 s per call)
        b["wall_seconds"] = int(b["compute_seconds"] / max(1, hw["workers"]) * 3 + 300 + 30 * int(b["llm_calls"]))
    srch = cfg["search"]
    srch["rungs"] = sorted(float(r) for r in srch["rungs"])
    if not srch["rungs"] or srch["rungs"][-1] != 1.0:
        srch["rungs"].append(1.0)
    if not 0 < srch["screen_fraction"] < srch["rungs"][0]:
        srch["screen_fraction"] = min(0.1, srch["rungs"][0] / 2)

    for k in ("n_train", "n_val", "n_test"):
        cfg["data"][k] = int(cfg["data"][k])
    spec = ResearchSpec(raw=raw, name=cfg["name"], problem=cfg["problem"] or sim.describe(), task=task,
                        simulator=cfg["simulator"], data=cfg["data"], metrics=met, success=cfg["success"],
                        budget=b, hardware=hw, constraints=cfg["constraints"], search=srch, llm=cfg["llm"],
                        seed=int(cfg["seed"]), simulator_description=sim.describe())
    s = spec.success
    op = "<=" if M.direction(s["metric"]) == "min" else ">="
    spec.objectives = [
        f"Learn a {task} model from simulated data ({sim.name}) that solves: {spec.problem.strip()[:200]}",
        f"Select methods by validation {spec.primary} only; the test split is used exactly once, at the end",
        (f"Success criterion: test {s['metric']} {op} {s['threshold']}" if s["threshold"] is not None
         else f"Success criterion: best achievable {s['metric']} within budget"),
        f"Stay within {b['compute_seconds']} worker-seconds of compute and {b['wall_seconds']} s wall time",
        "Eliminate weak approaches with cheap feasibility checks before spending substantial compute",
    ] + [f"Constraint: {n}" for n in (spec.constraints.get("notes") or [])]
    if not math.isfinite(float(b["compute_seconds"])) or b["compute_seconds"] <= 0:
        raise ValueError("budget.compute_seconds must be positive")
    return spec


TEMPLATE = """# Research specification for the Automated ML Research Harness (see end.md)
name: {name}
problem: >
  {problem}
# task: regression            # inferred from the simulator when omitted
simulator:
  name: friedman              # graph_fpt | friedman | regression | classification | python
  params: {{}}
data: {{n_train: 2000, n_val: 500, n_test: 500, seed: 0}}
metrics: {{primary: null, secondary: null}}   # defaults: rel_l2 / accuracy
success: {{threshold: null}}
budget: {{compute_seconds: 600, max_ideas: 40, experiment_timeout: 180, llm_calls: 30}}
hardware: {{workers: auto, gpus: auto}}
constraints: {{allowed_methods: all, forbidden_methods: [], allow_custom_code: true, notes: []}}
llm: {{backend: auto}}         # auto | none | claude_code | anthropic | codex
"""
