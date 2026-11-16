"""Ablation driver: heat-equation conductivity inverse problem on the V1 harness.

Usage (from the harness root):
    python ablation/heat/run_ablation.py --arms random offline --seeds 0 1 2
    python ablation/heat/run_ablation.py --arms llm_ideation llm_refine full --seeds 0 1 2

Each (arm, seed) is a separate harness run in ablation/heat/runs/<arm>_s<seed>/. Runs are executed
sequentially so that compute accounting (wall-time x jobs) is not distorted by CPU contention.
Everything except the arm-specific switches below is identical across arms.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BASE = {
    "name": "heat_ablation",
    "problem": ("Given noisy temperature measurements at four sensor locations over time, recover the four "
                "unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation."),
    "simulator": {"name": "python", "params": {"path": "examples/heat_diffusion_simulator.py",
                                               "function": "simulate", "n_segments": 4, "noise": 0.001}},
    "data": {"n_train": 3000, "n_val": 750, "n_test": 750, "seed": 5},     # identical data for every run
    "metrics": {"primary": "rel_l2", "secondary": ["r2", "rmse"]},
    "success": {"threshold": 0.05},
    "budget": {"compute_seconds": 400, "max_ideas": 34, "experiment_timeout": 120, "llm_calls": 40},
    "hardware": {"workers": 6, "gpus": "auto"},
    "constraints": {"allowed_methods": "all", "forbidden_methods": [], "allow_custom_code": True,
                    "notes": ["Models may only be trained on simulated data; the test split is never used for selection"]},
    "search": {"ideators": ["classical", "ensembles", "neural", "domain"], "ideas_per_ideator": 4,
               "screen_fraction": 0.1, "rungs": [0.33, 1.0], "eta": 3, "min_full": 4,
               "refine_rounds": 3, "refine_top_k": 3, "refine_children": 2, "patience": 2,
               "min_improvement": 0.005, "ensemble": True, "repair_attempts": 1, "llm_critic": True},
    "llm": {"backend": "claude_code", "effort": "medium"},
}

# arm -> overrides (deep-merged into BASE)
ARMS = {
    # 1. same pipeline, but every proposal AND every refinement child is a uniform random draw
    "random": {"llm": {"backend": "none"},
               "search": {"ideators": ["random"], "ideas_per_ideator": 16}},
    # 2. full pipeline with the built-in heuristic agents (persona sampling + local mutation)
    "offline": {"llm": {"backend": "none"}},
    # 3. LLM proposals only: no refinement, no LLM critic, no code repair, no ensemble
    "llm_ideation": {"search": {"refine_rounds": 0, "llm_critic": False, "repair_attempts": 0,
                                "ensemble": False}},
    # 4. + LLM feedback-driven refinement (still no LLM critic / repair / ensemble)
    "llm_refine": {"search": {"llm_critic": False, "repair_attempts": 0, "ensemble": False}},
    # 5. complete harness: + LLM critic batch review, crash repair, greedy ensemble
    "full": {},
}


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    (HERE / "specs").mkdir(exist_ok=True)
    (HERE / "runs").mkdir(exist_ok=True)
    log = HERE / "driver_log.jsonl"
    for arm in args.arms:
        for seed in args.seeds:
            spec = merge(BASE, ARMS[arm])
            spec["name"] = f"heat_{arm}_s{seed}"
            spec["seed"] = seed
            spath = HERE / "specs" / f"{arm}_s{seed}.yaml"
            spath.write_text(yaml.safe_dump(spec, sort_keys=False))
            rdir = HERE / "runs" / f"{arm}_s{seed}"
            if (rdir / "status.json").exists() and json.loads((rdir / "status.json").read_text()).get("stage") == "done":
                print(f"skip {arm} s{seed} (done)")
                continue
            print(f"\n##### {arm} seed {seed} #####", flush=True)
            t0 = time.time()
            proc = subprocess.run([args.python, "-m", "mlharness", "run", "--spec", str(spath),
                                   "--run-dir", str(rdir), "--quiet"], cwd=ROOT, capture_output=True, text=True)
            rec = {"arm": arm, "seed": seed, "rc": proc.returncode, "wall": round(time.time() - t0, 1),
                   "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(proc.stdout[-800:], proc.stderr[-800:] if proc.returncode else "", flush=True)


if __name__ == "__main__":
    main()
