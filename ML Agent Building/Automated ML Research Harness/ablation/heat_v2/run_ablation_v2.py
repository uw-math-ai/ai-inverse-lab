"""V2 ablation on the V1 heat-equation benchmark (same data, budget and protocol as ablation/heat).

Usage (from the harness root):
    python ablation/heat_v2/run_ablation_v2.py --arms v2_offline --seeds 0 1 2
    python ablation/heat_v2/run_ablation_v2.py --arms v2_full v2_no_forward v2_no_critic v2_no_literature --seeds 0 1 2
    python ablation/heat_v2/run_ablation_v2.py --arms v2_memory --seeds 1 2      # after v2_full s0

Held constant (identical to the V1 ablation): data seed 5, train/val/test 3000/750/750, primary metric
rel_l2, 400 worker-seconds of execution compute, 120 s per experiment, 6 workers, search seed = run
seed. Runs are sequential so wall-time-based compute accounting is not distorted by contention.

Arms
    v2_offline        heuristic researchers + rule-based critic (no LLM), literature + forward model on
    v2_full           LLM researchers + LLM critic, literature, pilots, forward model; fresh memory
    v2_no_forward     as v2_full but the forward model is NOT authorised (research loop vs physics access)
    v2_no_critic      as v2_full but one proposal round: no critique -> revision cycle, no pilots
    v2_no_literature  as v2_full but TheoremSearch disabled
    v2_memory         as v2_full but reading/writing a memory seeded with v2_full_s0's records (cross-run)
Every run except v2_memory gets its own empty research-memory file, so arms cannot leak into each other.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BASE = {
    "name": "heat_v2_ablation", "pipeline": "v2",
    "problem": ("Given noisy temperature measurements at four sensor locations over time, recover the four "
                "unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation."),
    "simulator": {"name": "python", "params": {"path": "examples/heat_diffusion_simulator.py",
                                               "function": "simulate", "n_segments": 4, "noise": 0.001}},
    "data": {"n_train": 3000, "n_val": 750, "n_test": 750, "seed": 5},
    "metrics": {"primary": "rel_l2", "secondary": ["r2", "rmse"]},
    "success": {"threshold": 0.05},
    "budget": {"compute_seconds": 400, "experiment_timeout": 120},
    "hardware": {"workers": 6, "gpus": "auto"},
    "constraints": {"allowed_methods": "all", "forbidden_methods": [], "allow_custom_code": True,
                    "allow_forward_model": True, "max_forward_evals": 3000000,
                    "notes": ["Models may only be trained on simulated data; the test split is never used for selection"]},
    "research": {"researchers": ["mathematical", "ml", "experimental", "exploratory"], "max_rounds": 3,
                 "min_rounds": 2, "proposals_per_researcher": 2, "quality_threshold": 0.75, "llm_calls": 30,
                 "pilot_tests": True, "llm_critic": True},
    "literature": {"enabled": True, "max_total_queries": 32, "n_results": 4},
    "memory": {"enabled": True},
    "execution": {"portfolio_size": 6, "include_baselines": True, "refine_rounds": 1, "ensemble": True},
    "evaluation": {"shift_n": 300, "shift_sets": [{"name": "noise_x3", "params": {"noise": 0.003}},
                                                  {"name": "k_low_0.3", "params": {"k_low": 0.3}}]},
    "llm": {"backend": "claude_code", "effort": "medium"},
}

ARMS = {
    "v2_offline": {"llm": {"backend": "none"}},
    "v2_full": {},
    "v2_no_forward": {"constraints": {"allow_forward_model": False}},
    "v2_no_critic": {"research": {"max_rounds": 1, "min_rounds": 1, "pilot_tests": False, "llm_critic": False}},
    "v2_no_literature": {"literature": {"enabled": False}},
    "v2_memory": {},
}


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


DEGRADED = "LLM_DEGRADED.json"     # marker: an LLM arm whose LLM calls failed (quota/outage) -> not a valid sample


def llm_failures(rdir: Path) -> list[str]:
    """Errors of every failed LLM call recorded in a run's artifact ledger."""
    out = []
    p = rdir / "artifacts.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if '"source": "llm"' in line:
                d = json.loads(line).get("data") or {}
                if d.get("source") == "llm":
                    out.append(str(d.get("error", ""))[:200])
    return out


SUSPENDED = "SUSPENDED.json"       # marker: the machine slept during the run -> wall-time compute accounting is invalid


def suspend_events(t0: float, t1: float) -> list[str]:
    """Windows sleep / Modern Standby entries (Kernel-Power 42, 506) between t0 and t1 (epoch seconds)."""
    if sys.platform != "win32":
        return []
    ps = ("$s=[DateTimeOffset]::FromUnixTimeSeconds(%d).LocalDateTime;$e=[DateTimeOffset]::FromUnixTimeSeconds(%d)"
          ".LocalDateTime;Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';"
          "Id=42,506;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue | ForEach-Object "
          "{$_.TimeCreated.ToString('s')}" % (int(t0) - 1, int(t1) + 1))
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def preflight_llm(spec: dict) -> str | None:
    """One tiny call through the harness's own backend; returns an error string or None."""
    sys.path.insert(0, str(ROOT))
    from mlharness.llm import LLMError, make_llm
    try:
        make_llm({**spec.get("llm", {}), "effort": "low", "timeout": 120}).complete(
            "You are a health check.", "Reply with the single word OK.")
        return None
    except LLMError as exc:
        return str(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["v2_offline", "v2_full", "v2_no_forward", "v2_no_critic",
                                                   "v2_no_literature"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if sys.platform == "win32":        # block idle sleep while this driver lives (a closed lid still suspends)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    for d in ("specs", "runs", "memory"):
        (HERE / d).mkdir(exist_ok=True)
    log = HERE / "driver_log.jsonl"
    for arm in args.arms:
        for seed in args.seeds:
            spec = merge(BASE, ARMS[arm])
            spec["name"], spec["seed"] = f"heat_{arm}_s{seed}", seed
            spath = HERE / "specs" / f"{arm}_s{seed}.yaml"
            spath.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            rdir = HERE / "runs" / f"{arm}_s{seed}"
            st = rdir / "status.json"
            uses_llm = (spec.get("llm") or {}).get("backend") != "none"
            if (st.exists() and json.loads(st.read_text()).get("stage") == "done"
                    and not (uses_llm and (rdir / DEGRADED).exists()) and not (rdir / SUSPENDED).exists()):
                print(f"skip {arm} s{seed} (done)")
                continue
            if uses_llm:
                err = preflight_llm(spec)
                if err:
                    raise SystemExit(f"LLM backend unavailable before {arm} s{seed}: {err}\n"
                                     "Stopping so that no LLM arm silently runs on heuristic fallbacks.")
            if rdir.exists():
                shutil.rmtree(rdir)
            if arm == "v2_memory":
                mem = HERE / "memory" / "shared_memory.jsonl"
                seed_src = HERE / "memory" / "v2_full_s0.jsonl"
                if not mem.exists():
                    if not seed_src.exists():
                        raise SystemExit("v2_memory needs v2_full s0 first (its memory seeds the shared memory)")
                    shutil.copy(seed_src, mem)
            else:
                mem = HERE / "memory" / f"{arm}_s{seed}.jsonl"
                if mem.exists():
                    mem.unlink()
            print(f"\n##### {arm} seed {seed} #####", flush=True)
            t0 = time.time()
            proc = subprocess.run([args.python, "-m", "mlharness", "research", "--spec", str(spath), "--run-dir",
                                   str(rdir), "--memory", str(mem), "--quiet"], cwd=ROOT, capture_output=True, text=True)
            rec = {"arm": arm, "seed": seed, "rc": proc.returncode, "wall": round(time.time() - t0, 1),
                   "memory": str(mem.relative_to(HERE)), "stdout_tail": proc.stdout[-1500:],
                   "stderr_tail": proc.stderr[-1500:]}
            fails = llm_failures(rdir) if uses_llm else []
            rec["llm_failures"] = len(fails)
            sleeps = suspend_events(t0, time.time())
            rec["suspend_events"] = sleeps
            if sleeps:
                (rdir / SUSPENDED).write_text(json.dumps({"events": sleeps}, indent=1), encoding="utf-8")
                print(f"WARNING: machine slept during {arm} s{seed} ({sleeps}); marked {SUSPENDED}, will be redone",
                      flush=True)
            if fails:
                (rdir / DEGRADED).write_text(json.dumps({"n": len(fails), "errors": fails[:20]}, indent=1),
                                             encoding="utf-8")
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(proc.stdout[-700:], proc.stderr[-700:] if proc.returncode else "", flush=True)
            if fails:
                quota = any(w in e.lower() for e in fails for w in ("limit", "rate", "quota", "overloaded"))
                print(f"WARNING: {arm} s{seed} had {len(fails)} failed LLM call(s); marked {DEGRADED}", flush=True)
                if quota:
                    raise SystemExit("LLM usage/rate limit hit; stopping. Re-run the same command later: "
                                     "degraded runs are redone, valid ones are skipped.")


if __name__ == "__main__":
    main()
