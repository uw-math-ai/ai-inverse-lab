"""V3 ablation on the V1/V2 heat-equation benchmark (same data, budget and protocol as ablation/heat{,_v2}).

Usage (from the harness root):
    python ablation/heat_v3/run_ablation_v3.py --smoke                         # tiny live integration check
    python ablation/heat_v3/run_ablation_v3.py --arms v3_offline --seeds 0 1 2
    python ablation/heat_v3/run_ablation_v3.py --arms v3_full v3_no_forward v3_no_critic v3_no_literature --seeds 0 1 2
    python ablation/heat_v3/run_ablation_v3.py --arms v3_opus_top --seeds 0
    add  --remote americano  to dispatch every development experiment to an AMATH GPU server over SSH
    (credentials come from the repository .env by variable name; the driver probes the host first and
    refuses to start if the probe fails, so no arm silently runs on a different execution route).

Held constant (identical to V1/V2): data seed 5, train/val/test 3000/750/750, primary metric rel_l2,
400 worker-seconds of execution compute, 120 s per experiment, 6 workers, search seed = run seed,
the same two shift sets. Runs are sequential.

Model tiers (V3 goal 4: orchestrator > researchers/critic > specialists)
    tiered      orchestrator Fable 5.1 | researcher, critic Opus 5.5 | specialist Sonnet 5.5;
                every *_escalated tier is Fable 5.1
    opus_top    as tiered, but Opus 5.5 is the highest tier: orchestrator and all escalations use Opus

Arms
    v3_offline        no LLM: the offline program policy (default study ridge vs extra-trees + physics MLE)
    v3_full           tiered LLM team, independent LLM critic, TheoremSearch, forward model; fresh memory
    v3_no_forward     as v3_full, forward-model access NOT authorised
    v3_no_critic      as v3_full, critic role has no model (structural/rule objections only)
    v3_no_literature  as v3_full, TheoremSearch disabled
    v3_opus_top       as v3_full with the opus_top tiering (one run requested)

Every run gets its own empty research-memory file. Held-out exposure is recorded in a shared ablation
registry with strict_holdout off: one final evaluation per run, as in V1/V2.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FABLE, OPUS, SONNET = "claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"


def role(model, effort="medium", timeout=900):
    return {"backend": "claude_code", "model": model, "effort": effort, "timeout": timeout}


TIERS = {
    "tiered": {"orchestrator": role(FABLE), "researcher": role(OPUS), "critic": role(OPUS),
               "specialist": role(SONNET, timeout=600), "orchestrator_escalated": role(FABLE, "high"),
               "researcher_escalated": role(FABLE, "high"), "critic_escalated": role(FABLE, "high")},
    "opus_top": {"orchestrator": role(OPUS), "researcher": role(OPUS), "critic": role(OPUS),
                 "specialist": role(SONNET, timeout=600), "orchestrator_escalated": role(OPUS, "high"),
                 "researcher_escalated": role(OPUS, "high"), "critic_escalated": role(OPUS, "high")},
}

BASE = {
    "name": "heat_v3_ablation", "pipeline": "v3",
    "problem": ("Given noisy temperature measurements at four sensor locations over time, recover the four "
                "unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation."),
    "simulator": {"name": "python", "params": {"path": "examples/heat_diffusion_simulator.py",
                                               "function": "simulate", "n_segments": 4, "noise": 0.001}},
    "data": {"n_train": 3000, "n_val": 750, "n_test": 750, "seed": 5},
    "metrics": {"primary": "rel_l2", "secondary": ["r2", "rmse"]},
    "success": {"threshold": 0.05},
    # Development compute identical to V1/V2 (400 worker-s). V3 carves its protected export reserve out
    # of compute_seconds, and the default 25 % (100 s) is too small to export + certify a physics
    # method; V2 allowed the final fit 3 x 120 s outside the 400 s. So total = 400 dev + 360 reserved.
    # Wall budget raised so slow top-tier calls cannot starve the export.
    "budget": {"compute_seconds": 760, "experiment_timeout": 120, "wall_seconds": 7200},
    "hardware": {"workers": 6, "gpus": "auto"},
    "constraints": {"allowed_methods": "all", "forbidden_methods": [], "allow_custom_code": True,
                    "allow_forward_model": True, "max_forward_evals": 3000000,
                    "notes": ["Models may only be trained on simulated data; the test split is never used for selection"]},
    "research": {"researchers": ["mathematical", "ml", "experimental", "exploratory"], "llm_calls": 72},
    "literature": {"enabled": True, "max_total_queries": 32, "n_results": 4},
    "memory": {"enabled": True, "mode": "cumulative"},
    "execution": {"ensemble": True, "bootstrap": 200},
    "evaluation": {"shift_n": 300, "shift_sets": [{"name": "noise_x3", "params": {"noise": 0.003}},
                                                  {"name": "k_low_0.3", "params": {"k_low": 0.3}}],
                   "strict_holdout": False,
                   "exposure_registry": str((HERE / "exposure_registry.sqlite").relative_to(ROOT)).replace("\\", "/")},
    "program": {"max_actions": 24, "max_agents": 10, "max_active_agents": 6, "max_depth": 2, "max_children": 4,
                "max_specialist_tasks": 8, "max_agent_turns": 8, "max_studies": 6, "budget_mode": "soft",
                "confirmation_fraction": 0.25, "confirmation_seconds": 360, "auto_followup": True, "roles": TIERS["tiered"]},
    "llm": role(OPUS),          # default for any role not in program.roles
}

ARMS = {
    "v3_offline": {"llm": {"backend": "none"}, "program": {"roles": {}}},
    "v3_full": {},
    "v3_no_forward": {"constraints": {"allow_forward_model": False}},
    "v3_no_critic": {"program": {"roles": {"critic": {"backend": "none"}, "critic_escalated": {"backend": "none"}}}},
    "v3_no_literature": {"literature": {"enabled": False}},
    "v3_opus_top": {"program": {"roles": TIERS["opus_top"]}},
}

# Small live integration check: the V3 example problem, few actions, same tiering.
SMOKE = {
    "problem": ("Given noisy temperature measurements at four sensor locations over time, recover the two "
                "unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation."),
    "simulator": {"name": "python", "params": {"path": "examples/heat_diffusion_simulator.py", "function": "simulate",
                                               "n_segments": 2, "n_grid": 20, "n_steps": 60, "sensors": [3, 8, 12, 17],
                                               "every": 6, "noise": 0.001}},
    "data": {"n_train": 240, "n_val": 80, "n_test": 80, "seed": 1302},
    "budget": {"compute_seconds": 240, "experiment_timeout": 30, "wall_seconds": 2400},
    "hardware": {"workers": 2},
    "research": {"researchers": ["ml", "mathematical"], "llm_calls": 16},
    "literature": {"enabled": False},
    "evaluation": {"shift_sets": [], "exposure_registry": "ablation/heat_v3/smoke_exposure.sqlite"},
    "program": {"max_actions": 6, "max_studies": 3, "max_specialist_tasks": 2, "confirmation_seconds": 120},
}


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k == "roles" and isinstance(v, dict) and not v:
            out[k] = {}
        else:
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


DEGRADED = "LLM_DEGRADED.json"     # an LLM arm whose model calls failed (quota/outage) -> not a valid sample
SUSPENDED = "SUSPENDED.json"       # the machine slept -> wall-time compute accounting invalid
QUOTA_WORDS = ("limit", "rate", "quota", "overloaded", "credit", "usage")


def read_artifacts(rdir: Path) -> list[dict]:
    p = rdir / "artifacts.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def role_events(arts: list[dict]) -> list[dict]:
    """V3 RoleRouter records (one per role ask, attempted or not)."""
    return [a.get("data") or {} for a in arts if a.get("kind") == "resource"
            and isinstance(a.get("data"), dict) and "purpose" in a["data"] and "tier" in a["data"]]


def token_usage(rdir: Path) -> dict:
    """Per-run token/cost accounting from the artifact ledger, by tier, by served model and total."""
    ev = role_events(read_artifacts(rdir))
    fields = ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens", "thinking_tokens")

    def blank():
        return {"calls": 0, **{f: 0 for f in fields}, "total_input_tokens": 0, "cost_usd": 0.0, "seconds": 0.0}

    total, by_tier, by_model, by_purpose = blank(), defaultdict(blank), defaultdict(blank), defaultdict(blank)
    fallbacks = defaultdict(int)
    for e in ev:
        if e.get("fallback_reason"):
            fallbacks[e["fallback_reason"]] += 1
        if not e.get("call_attempted") or e.get("fallback_reason") == "service_error":
            continue
        rows = [total, by_tier[e.get("tier", "?")], by_purpose[e.get("purpose", "?")]]
        for row in rows:
            row["calls"] += 1
            for f in fields:
                row[f] += int(e.get(f) or 0)
            row["total_input_tokens"] += sum(int(e.get(f) or 0) for f in ("input_tokens", "cache_creation_tokens",
                                                                           "cache_read_tokens"))
            row["cost_usd"] += float(e.get("cost_usd") or 0)
            row["seconds"] += float(e.get("seconds") or 0)
        for m, u in (e.get("model_usage") or {"?": {}}).items():   # includes Claude Code's own side calls
            r = by_model[m]
            r["calls"] += 1
            r["input_tokens"] += int(u.get("inputTokens") or 0)
            r["output_tokens"] += int(u.get("outputTokens") or 0)
            r["cache_creation_tokens"] += int(u.get("cacheCreationInputTokens") or 0)
            r["cache_read_tokens"] += int(u.get("cacheReadInputTokens") or 0)
            r["thinking_tokens"] += int(u.get("thinkingTokens") or 0)
            r["total_input_tokens"] += sum(int(u.get(k) or 0) for k in ("inputTokens", "cacheCreationInputTokens",
                                                                          "cacheReadInputTokens"))
            r["cost_usd"] += float(u.get("costUSD") or 0)
    rnd = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    return {"total": rnd(total), "by_tier": {k: rnd(v) for k, v in by_tier.items()},
            "by_model": {k: rnd(v) for k, v in by_model.items()},
            "by_purpose": {k: rnd(v) for k, v in by_purpose.items()},
            "role_asks": len(ev), "fallbacks": dict(fallbacks)}


def remote_job_counts(rdir: Path) -> dict:
    """Where development experiments actually ran (device field of each result) + remote fallbacks."""
    arts = read_artifacts(rdir)
    devices = defaultdict(int)
    for a in arts:
        d = a.get("data") or {}
        if a.get("kind") == "result" and d.get("exp_id") and d.get("device"):
            devices[str(d["device"])] += 1
    fallbacks = sum(1 for a in arts if a.get("kind") == "failure" and (a.get("data") or {}).get("source") == "remote")
    return {"by_device": dict(devices), "remote_fallbacks_to_local": fallbacks}


def llm_timeouts(rdir: Path) -> int:
    """Calls that hit the per-role timeout: a harness-handled fallback, not a backend outage."""
    return sum(1 for e in role_events(read_artifacts(rdir))
               if e.get("fallback_reason") == "service_error" and "timed out" in str(e.get("error", "")))


def llm_failures(rdir: Path) -> list[str]:
    """Errors of failed model calls that indicate an outage/quota/auth problem (V3 role events + V2-style
    agent failures). Per-call timeouts are counted separately by llm_timeouts and do not degrade a run."""
    arts = read_artifacts(rdir)
    out = [str(e.get("error", ""))[:200] for e in role_events(arts) if e.get("fallback_reason") == "service_error"
           and "timed out" not in str(e.get("error", ""))]
    out += [str((a.get("data") or {}).get("error", ""))[:200] for a in arts
            if a.get("kind") == "failure" and (a.get("data") or {}).get("source") == "llm"]
    return out


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
    """One tiny call per distinct configured model through the harness's own backend."""
    sys.path.insert(0, str(ROOT))
    from mlharness.llm import LLMError, make_llm
    cfgs = [spec.get("llm", {})] + list((spec.get("program") or {}).get("roles", {}).values())
    seen = set()
    for cfg in cfgs:
        if (cfg or {}).get("backend", "auto") == "none":
            continue
        key = (cfg.get("backend"), cfg.get("model"))
        if key in seen:
            continue
        seen.add(key)
        try:
            make_llm({**cfg, "effort": "low", "timeout": 180}).complete("You are a health check.",
                                                                        "Reply with the single word OK.")
        except LLMError as exc:
            return f"{key}: {exc}"
    return None


REMOTE_VENV = ".mlharness_venv/bin/python"   # private venv in the remote home (system numpy on doppio is broken)


def preflight_remote(host: str) -> dict:
    """Probe the host with the harness's own runner; prefer the private venv interpreter if present."""
    sys.path.insert(0, str(ROOT))
    from mlharness.remote import RemoteRunner
    rr = RemoteRunner({"enabled": True, "host": host})
    try:
        home = rr._home()
        venv = f"{home}/{REMOTE_VENV}"
        code, out, _ = rr._exec(f"test -x {venv} && echo yes || echo no", 20)
        python = venv if "yes" in out else "python3"
    except Exception as exc:
        rr.close()
        return {"host": host, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    rr.close()
    rr = RemoteRunner({"enabled": True, "host": host, "python": python})
    try:
        info = rr.probe()
        info["python_path"] = python
        info["torch_cuda"] = bool(info.get("torch") and " True " in f" {info['torch']} ")
        return info
    finally:
        rr.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["v3_offline", "v3_full", "v3_no_forward", "v3_no_critic",
                                                   "v3_no_literature"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--remote", help="AMATH host (americano | doppio | ...) for all development experiments")
    ap.add_argument("--smoke", action="store_true", help="tiny live integration check (v3_full config, 1 run)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    # A subscription usage limit pauses the affected call until the stated reset and retries it (the
    # paused time is excluded from the run's wall clock and logged), instead of degrading the run.
    os.environ["MLH_WAIT_ON_USAGE_LIMIT"] = "1"
    if sys.platform == "win32":        # block idle sleep while this driver lives (a closed lid still suspends)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    for d in ("specs", "runs", "memory"):
        (HERE / d).mkdir(exist_ok=True)
    remote_info = None
    if args.remote:
        remote_info = preflight_remote(args.remote)
        if not remote_info.get("ok"):
            raise SystemExit(f"remote GPU host {args.remote} failed its probe: {remote_info.get('error')}\n"
                             "Fix the SERVICE_USERNAME / SERVICE_PASSWORD entries in .env (or choose another host).")
        print("remote probe:", json.dumps(remote_info))
    log = HERE / ("smoke_log.jsonl" if args.smoke else "driver_log.jsonl")
    # seed-major order: if a usage limit stops the driver, every arm has the same number of seeds so far
    plan = [("v3_smoke", 0)] if args.smoke else [(a, s) for s in args.seeds for a in args.arms]
    for arm, seed in plan:
        spec = merge(BASE, SMOKE) if arm == "v3_smoke" else merge(BASE, ARMS[arm])
        spec["name"], spec["seed"] = f"heat_{arm}_s{seed}", seed
        if args.remote:
            spec["hardware"]["remote"] = {"enabled": True, "host": args.remote, "use_for": "all",
                                          "python": remote_info["python_path"], "max_failures": 3}
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
                                 "Stopping so that no LLM arm silently runs on offline fallbacks.")
        if rdir.exists():
            shutil.rmtree(rdir)
        mem = HERE / "memory" / f"{arm}_s{seed}.jsonl"
        if mem.exists():
            mem.unlink()
        print(f"\n##### {arm} seed {seed} #####", flush=True)
        t0 = time.time()
        # Claude Code otherwise writes every (unique, never re-read) program view to the 1-hour prompt
        # cache at 2x the input price; disabling it halves cost without changing what the model sees.
        env = dict(os.environ, DISABLE_PROMPT_CACHING="1")
        proc = subprocess.run([args.python, "-m", "mlharness", "research", "--pipeline", "v3", "--spec", str(spath),
                               "--run-dir", str(rdir), "--memory", str(mem), "--quiet"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        rec = {"arm": arm, "seed": seed, "rc": proc.returncode, "wall": round(time.time() - t0, 1),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
               "remote": args.remote, "remote_gpus": (remote_info or {}).get("gpus"),
               "remote_jobs": remote_job_counts(rdir), "memory": str(mem.relative_to(HERE)),
               "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}
        fails = llm_failures(rdir) if uses_llm else []
        rec["llm_failures"] = len(fails)
        rec["llm_timeouts"] = llm_timeouts(rdir) if uses_llm else 0
        tokens = token_usage(rdir)
        (rdir / "token_usage.json").write_text(json.dumps(tokens, indent=1), encoding="utf-8") if rdir.exists() else None
        rec["tokens"] = tokens["total"]
        rec["tokens_by_tier"] = {k: {"calls": v["calls"], "total_input_tokens": v["total_input_tokens"],
                                     "output_tokens": v["output_tokens"], "cost_usd": v["cost_usd"]}
                                 for k, v in tokens["by_tier"].items()}
        rec["fallbacks"] = tokens["fallbacks"]
        rec["usage_limit_wait_s"] = round(sum(float(e.get("usage_limit_wait_seconds") or 0)
                                              for e in role_events(read_artifacts(rdir))), 1)
        sleeps = suspend_events(t0, time.time())
        rec["suspend_events"] = sleeps
        if sleeps and rdir.exists():
            (rdir / SUSPENDED).write_text(json.dumps({"events": sleeps}, indent=1), encoding="utf-8")
            print(f"WARNING: machine slept during {arm} s{seed} ({sleeps}); marked {SUSPENDED}, will be redone",
                  flush=True)
        if fails and rdir.exists():
            (rdir / DEGRADED).write_text(json.dumps({"n": len(fails), "errors": fails[:20]}, indent=1),
                                         encoding="utf-8")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        t = tokens["total"]
        print(proc.stdout[-700:], proc.stderr[-1500:] if proc.returncode else "", flush=True)
        print(f"tokens: {t['calls']} calls, in {t['total_input_tokens']:,} (cache-write {t['cache_creation_tokens']:,},"
              f" cache-read {t['cache_read_tokens']:,}), out {t['output_tokens']:,}, ${t['cost_usd']:.2f}; "
              f"fallbacks {tokens['fallbacks']}", flush=True)
        if fails:
            quota = any(w in e.lower() for e in fails for w in QUOTA_WORDS)
            print(f"WARNING: {arm} s{seed} had {len(fails)} failed LLM call(s); marked {DEGRADED}", flush=True)
            if quota:
                raise SystemExit("LLM usage/rate limit hit; stopping. Re-run the same command later: "
                                 "degraded runs are redone, valid ones are skipped.")


if __name__ == "__main__":
    main()
