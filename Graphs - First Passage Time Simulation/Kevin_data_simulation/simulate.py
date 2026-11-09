"""Headless experiment runner; writes the same result schema as the dashboard."""
import argparse
import json
from pathlib import Path
import sys
from forward_sim.engine import simulate
from forward_sim.export import csv_text
from forward_sim.model import parse_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Result JSON; a matching CSV is also written")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    if args.output.resolve() == args.config.resolve():
        parser.error("--output must differ from the input configuration")
    try:
        config = parse_config(json.loads(args.config.read_text()))
        result = simulate(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        args.output.with_suffix(".csv").write_text(csv_text(result))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
