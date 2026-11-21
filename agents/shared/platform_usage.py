"""Usage reporting and handoff notes for agents on the research platform.

Copy this folder to ``agents/shared/`` in the registry and import it with::

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
    import platform_usage

The platform polls ``AGENT_USAGE_FILE`` while the run is active. When the plan
behind the run's connection reaches the handoff threshold (95% by default) it
creates ``AGENT_HANDOFF_SIGNAL``. Check ``handoff_requested()`` between steps and,
when it is true, call ``write_handoff()`` and finish cleanly. If an agent never
writes notes, the platform writes a basic HANDOFF.md from inputs and logs.

Outside the platform (no environment variables) every function is a no-op.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_totals = {"tokens": 0, "cost_usd": 0.0}
_plan_percent: float | None = None


def _write_report() -> None:
    target = os.environ.get("AGENT_USAGE_FILE")
    if not target:
        return
    report = dict(_totals)
    if _plan_percent is not None:
        report["plan_used_percent"] = _plan_percent
    path = Path(target)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report), encoding="utf-8")
    os.replace(temporary, path)


def add_usage(tokens: int = 0, cost_usd: float = 0.0) -> None:
    """Add one model call's usage to this run's cumulative totals."""
    _totals["tokens"] += max(int(tokens or 0), 0)
    _totals["cost_usd"] += max(float(cost_usd or 0), 0.0)
    _write_report()


def set_plan_percent(percent: float) -> None:
    """Report the provider's own plan meter (0-100), e.g. from CLI rate-limit output."""
    global _plan_percent
    _plan_percent = min(max(float(percent), 0.0), 100.0)
    _write_report()


def handoff_requested() -> bool:
    signal = os.environ.get("AGENT_HANDOFF_SIGNAL")
    return bool(signal) and Path(signal).exists()


def write_handoff(summary: str, completed=(), next_steps=(), state: dict | None = None) -> Path | None:
    """Write HANDOFF.md so a new run (on a plan with remaining usage) can continue."""
    target = os.environ.get("AGENT_HANDOFF_FILE")
    if not target:
        return None
    lines = ["# Handoff", "", summary.strip(), ""]
    if completed:
        lines += ["## Completed", "", *[f"- {item}" for item in completed], ""]
    if next_steps:
        lines += ["## Next steps", "", *[f"- {item}" for item in next_steps], ""]
    if state:
        lines += ["## State to resume from", "", "```json", json.dumps(state, indent=2, sort_keys=True), "```", ""]
    path = Path(target)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
