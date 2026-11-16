"""Command line: generate | run | summarize | plot | validate | bench.

    python -m fptsim generate  --config configs/example_small.yaml
    python -m fptsim run       --config configs/example_small.yaml [--dry-run] [--resume] [--limit 10] [--shard 0/4]
    python -m fptsim summarize --run results/example_small
    python -m fptsim plot      --run results/example_small --graph-id 17
    python -m fptsim validate  [--N 1000000] [--replicates 3] [--skip-tests]
    python -m fptsim bench     [--seconds 5]
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, load_config

log = logging.getLogger("fptsim")


def _shard(text: str) -> tuple[int, int]:
    try:
        i, k = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("--shard must look like i/k, e.g. 0/4") from None
    if not (k >= 1 and 0 <= i < k):
        raise argparse.ArgumentTypeError("--shard i/k needs k >= 1 and 0 <= i < k")
    return i, k


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fptsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate and save graphs only")
    g.add_argument("--config", required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--shard", type=_shard, default=(0, 1))

    r = sub.add_parser("run", help="generate if needed, then compute exact + Monte Carlo")
    r.add_argument("--config", required=True)
    r.add_argument("--dry-run", action="store_true", help="estimate work, memory and runtime, then exit")
    r.add_argument("--resume", dest="resume", action="store_true", default=None)
    r.add_argument("--no-resume", dest="resume", action="store_false")
    r.add_argument("--limit", type=int, help="only the first N graphs of this job")
    r.add_argument("--shard", type=_shard, default=(0, 1), help="job i of k, e.g. 3/8")
    r.add_argument("--device", help="override config device (auto | cpu | cuda | cuda:N)")
    r.add_argument("--backend", choices=["auto", "torch", "numpy"], help="override config backend")
    r.add_argument("--output-dir", help="override config output_dir")

    s = sub.add_parser("summarize", help="aggregate tables for a finished run")
    s.add_argument("--run", required=True)

    pl = sub.add_parser("plot", help="plot one graph and its r vs r' panels")
    pl.add_argument("--run", required=True)
    pl.add_argument("--graph-id", type=int, required=True)
    pl.add_argument("--sweep", type=int, default=0)
    pl.add_argument("--replicate", type=int, default=0)
    pl.add_argument("--out", help="output directory (default <run>/plots)")

    v = sub.add_parser("validate", help="paper8/paper9 regression and the test suite")
    v.add_argument("--N", type=float, default=1e6)
    v.add_argument("--replicates", type=int, default=3)
    v.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    v.add_argument("--device", default="auto")
    v.add_argument("--skip-tests", action="store_true")

    b = sub.add_parser("bench", help="Monte Carlo walks/s on every available backend/device")
    b.add_argument("--seconds", type=float, default=5.0)
    b.add_argument("--backend", choices=["torch", "numpy"])
    b.add_argument("--device")
    return p


def _apply_overrides(cfg: dict, args) -> dict:
    from .config import validate_config
    changed = False
    for key in ("device", "backend", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
            changed = True
    return validate_config(cfg) if changed else cfg


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from .runner import setup_logging
    try:
        if args.command == "generate":
            from .runner import generate_only
            generate_only(load_config(args.config), limit=args.limit, shard=args.shard)
        elif args.command == "run":
            from .runner import dry_run, run
            cfg = _apply_overrides(load_config(args.config), args)
            if args.dry_run:
                dry_run(cfg, limit=args.limit, shard=args.shard)
            else:
                run(cfg, resume=args.resume, limit=args.limit, shard=args.shard)
        elif args.command == "summarize":
            from .summary import summarize
            summarize(args.run)
        elif args.command == "plot":
            import matplotlib
            matplotlib.use("Agg")
            from .plots import plot_graph_report
            for path in plot_graph_report(args.run, args.graph_id, args.sweep, args.replicate, args.out):
                print(path)
        elif args.command == "validate":
            return _validate(args)
        elif args.command == "bench":
            return _bench(args)
    except ConfigError as exc:
        setup_logging(None)
        log.error("config error: %s", exc)
        return 2
    return 0


def _validate(args) -> int:
    from .runner import setup_logging
    from .validation import paper_regression, regression_passes, run_tests
    setup_logging(None)
    rows = paper_regression(N=int(args.N), replicates=args.replicates, backend=args.backend,
                            device=args.device)
    ok = True
    print(f"\npaper regression, T = 17 steps, q = 0, N = {int(args.N):,}, R = {args.replicates}, "
          f"{rows[0]['backend']} on {rows[0]['device']}")
    print(f"{'graph':7s} {'exact-literal':>13s} {'FRNE mean':>10s} {'std':>7s} {'min':>7s} {'max':>7s} "
          f"{'expected':>9s} {'paper':>8s} {'coverage':>8s} {'walks/s':>9s}  result")
    for row in rows:
        passed = regression_passes(row)
        ok &= passed
        print(f"{row['graph']:7s} {row['exact_vs_literal_max_abs']:13.2e} {row['frne_mean']:10.4f} "
              f"{row['frne_std']:7.4f} {row['frne_min']:7.4f} {row['frne_max']:7.4f} "
              f"{row['expected_rms_frne']:9.4f} {row['paper_frne']:8.4f} {row['ci_coverage']:8.3f} "
              f"{row['walks_per_s']:9.3g}  {'PASS' if passed else 'FAIL'}")
    code = 0 if ok else 1
    if not args.skip_tests:
        code = max(code, run_tests())
    return code


def _bench(args) -> int:
    from .runner import setup_logging
    from .validation import benchmark
    setup_logging(None)
    combos = []
    if args.backend or args.device:
        combos.append((args.backend or "auto", args.device or "auto"))
    else:
        combos.append(("numpy", "cpu"))
        try:
            import torch
            combos.append(("torch", "cpu"))
            if torch.cuda.is_available():
                combos += [("torch", f"cuda:{i}") for i in range(torch.cuda.device_count())]
        except ImportError:
            pass
    print(f"{'backend':8s} {'device':8s} {'walks/s':>12s} {'walk-steps/s':>14s}  details")
    for backend, device in combos:
        res = benchmark(backend, device, seconds=args.seconds)
        print(f"{res['backend']:8s} {res['device']:8s} {res['walks_per_s']:12.4g} "
              f"{res['walk_steps_per_s']:14.4g}  n={res['n']}, pairs={res['pairs']}, T={res['T']}; "
              f"{res['banner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
