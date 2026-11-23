"""Autodata harness for the vertex-centrality inverse problem: rule-based challenger, no model calls.

All logic lives in agents/shared/autodata_core (standard library only). See README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

from autodata_core.agent import main  # noqa: E402

TIMEOUT_S = 14400                       # keep in step with agent.json timeout_seconds

if __name__ == "__main__":
    sys.exit(main("free", default_budget_s=TIMEOUT_S - 300))
