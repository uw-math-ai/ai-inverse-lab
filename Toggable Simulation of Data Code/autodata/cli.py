"""python -m autodata {run, meta, summarize, show} ..."""
from __future__ import annotations

import argparse
import json
import sys


def _load(args):
    from .config import load
    cfg = load(args.config)
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = args.output_dir
    if getattr(args, "scenarios", None):
        wanted = args.scenarios.split(",")
        cfg["scenarios"] = [s for s in cfg["scenarios"] if s["id"] in wanted]
        if not cfg["scenarios"]:
            sys.exit(f"--scenarios: none of {wanted} are in the corpus")
    if getattr(args, "max_rounds", None):
        cfg["loop"]["max_rounds"] = args.max_rounds
    if getattr(args, "llm", None):
        cfg["toggles"]["llm_challenger"] = args.llm != "none"
        cfg["llm"]["backend"] = args.llm
    if getattr(args, "no_resume", False):
        cfg["toggles"]["resume"] = False
    if getattr(args, "strategy", None):
        import yaml
        with open(args.strategy, encoding="utf-8") as fh:
            cfg["strategy"] = yaml.safe_load(fh)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autodata", description="Autodata harness for fptsim inverse problems")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "inner loop over the scenario corpus; writes the accepted dataset"),
                        ("meta", "outer loop: meta-optimize the challenger strategy")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True)
        p.add_argument("--output-dir")
        p.add_argument("--scenarios", help="comma-separated scenario ids")
        p.add_argument("--max-rounds", type=int)
        p.add_argument("--llm", choices=["none", "auto", "anthropic", "claude_code"],
                       help="turn the LLM challenger on (backend) or off (none)")
        p.add_argument("--strategy", help="YAML strategy file (e.g. meta/best_strategy.yaml)")
        p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("summarize", help="rebuild items.jsonl + summary.json of a run")
    p.add_argument("--run", required=True)
    p = sub.add_parser("show", help="print the rounds of one scenario of a run")
    p.add_argument("--run", required=True)
    p.add_argument("--scenario", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from .loop import run
        print(json.dumps(run(_load(args)), indent=1))
    elif args.cmd == "meta":
        from .meta import optimize
        print(json.dumps(optimize(_load(args)), indent=1))
    elif args.cmd == "summarize":
        from .loop import summarize
        print(json.dumps(summarize(args.run), indent=1))
    elif args.cmd == "show":
        from pathlib import Path
        res = json.loads((Path(args.run) / "scenarios" / args.scenario / "result.json").read_text(encoding="utf-8"))
        print(f"{res['scenario']}: {res['brief']}\naccepted={res['accepted']} rounds={res['rounds']}")
        for rec in res["history"]:
            w = (rec.get("weak") or {}).get("avg")
            s = (rec.get("strong") or {}).get("avg")
            inst = rec.get("instance") or {}
            print(f"  r{rec['round']:>2} {rec['status']:<13} weak={'-' if w is None else f'{w:.3f}'} "
                  f"strong={'-' if s is None else f'{s:.3f}'} n={inst.get('n')} k={inst.get('k')} "
                  f"T={inst.get('T')} N={inst.get('N')} | {rec.get('challenger_note')}")
            if rec["status"] in ("qv_failed", "invalid"):
                print(f"       {rec['reason']}")


if __name__ == "__main__":
    main()
