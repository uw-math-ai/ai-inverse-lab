"""Compute budget, hardware detection and device allocation."""
from __future__ import annotations

import importlib.util
import math
import os
import platform
import queue
import sys
import threading
import time
from contextlib import contextmanager

# Cumulative share of the compute budget each pipeline stage may consume. Unused budget rolls
# forward automatically because caps are cumulative. The final stage is never blocked.
STAGE_CAPS = {"baseline": 0.05, "screen": 0.20, "halving": 0.60, "refine": 0.85, "final": 1.0}
# V2 pipeline: research pilots, multi-profile screening, full evaluation, local refinement, final
STAGE_CAPS_V2 = {"baseline": 0.04, "pilot": 0.12, "screen": 0.30, "evaluate": 0.78, "refine": 0.90, "final": 1.0}


class BudgetReservations:
    """Atomic worker-second admission for whole required groups.

    In hard mode callers must reserve timeout bounds, including final/export work, and pass
    those bounds to the isolated executor. Process startup/termination overhead is measured
    separately and can exceed the bound; this class never describes estimates as hard caps.
    Confirmation reserve is protected from exploratory admission until consumed explicitly.
    """
    def __init__(self, total_seconds, confirmation_seconds=0, hard=False):
        self.total = float(total_seconds)
        self.confirmation = float(confirmation_seconds)
        if not math.isfinite(self.total) or not math.isfinite(self.confirmation) or not 0 <= self.confirmation <= self.total:
            raise ValueError("invalid total/confirmation budget")
        self.hard = bool(hard)
        self.used = 0.0
        self.confirmation_used = 0.0
        self.reservations = {}
        self.history = []
        self._lock = threading.Lock()

    def reserve(self, group_id, seconds, confirmation=False):
        seconds = float(seconds)
        if not group_id or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("reservation needs an id and finite nonnegative seconds")
        with self._lock:
            if group_id in self.reservations:
                raise ValueError(f"group {group_id!r} is already reserved")
            reserved = sum(r["seconds"] for r in self.reservations.values())
            confirmed = sum(r["seconds"] for r in self.reservations.values() if r["confirmation"])
            protected = max(0.0, self.confirmation - self.confirmation_used - confirmed)
            available = self.total - self.used - reserved - (0.0 if confirmation else protected)
            admitted = seconds <= available + 1e-12
            self.history.append({"action": "reserve", "group_id": group_id, "seconds": seconds,
                                 "confirmation": bool(confirmation), "admitted": admitted})
            if admitted:
                self.reservations[group_id] = {"seconds": seconds, "confirmation": bool(confirmation)}
            return admitted

    def release(self, group_id, actual_seconds=0):
        actual = float(actual_seconds)
        if not math.isfinite(actual) or actual < 0:
            raise ValueError("actual seconds must be finite and nonnegative")
        with self._lock:
            reservation = self.reservations.pop(group_id, None)
            if reservation is None:
                raise ValueError(f"unknown reservation {group_id!r}")
            self.used += actual
            if reservation["confirmation"]:
                self.confirmation_used += actual
            self.history.append({"action": "release", "group_id": group_id, "actual_seconds": actual,
                                 "reserved_seconds": reservation["seconds"],
                                 "overrun_seconds": max(0.0, actual - reservation["seconds"])})

    def snapshot(self):
        with self._lock:
            reserved = sum(r["seconds"] for r in self.reservations.values())
            return {"total_seconds": self.total, "used_seconds": self.used,
                    "reserved_seconds": reserved, "remaining_seconds": max(0.0, self.total-self.used-reserved),
                    "confirmation_seconds": self.confirmation, "confirmation_used_seconds": self.confirmation_used,
                    "hard": self.hard, "policy": "timeout_bounds" if self.hard else "soft_estimates",
                    "overrun_seconds": max(0.0, self.used-self.total),
                    "active_groups": {k: dict(v) for k, v in self.reservations.items()},
                    "history": list(self.history)}


def detect_hardware() -> dict:
    info = {"cpus": os.cpu_count() or 1, "gpus": [], "accelerator": "cpu", "torch": False,
            "platform": platform.platform(), "python": sys.version.split()[0],
            "python_exe": sys.executable}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                info["accelerator"] = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                info["gpus"] = ["mps"]
                info["accelerator"] = "mps"
        except Exception as exc:  # pragma: no cover - depends on local install
            info["torch_error"] = str(exc)
    return info


class ResourceManager:
    def __init__(self, compute_seconds: float, wall_seconds: float, workers: int,
                 gpu_devices: list[str], llm_calls: int, caps: dict | None = None):
        self.caps = dict(caps or STAGE_CAPS)
        self.total = float(compute_seconds)
        self.wall_seconds = float(wall_seconds)
        self.workers = int(workers)
        self.start = time.time()
        self.used = 0.0
        self.by_stage: dict[str, float] = {}
        self.by_idea: dict[str, float] = {}
        self.jobs = 0
        self.llm_calls_max = int(llm_calls)
        self.llm_calls = 0
        self.llm_seconds = 0.0
        self.llm_cost_usd = 0.0
        self.llm_by_phase: dict[str, int] = {}
        self.llm_phase_caps: dict[str, int] = {}   # e.g. {"research": 30}: hard cap per phase
        self.phase = "execution"            # V2: "research" while the research loop runs
        self.literature_queries = 0
        self.literature_failures = 0
        self._lock = threading.Lock()
        self._gpus: queue.Queue = queue.Queue()
        for d in gpu_devices:
            self._gpus.put(d)
        self.gpu_devices = list(gpu_devices)

    # -- compute budget ------------------------------------------------------------------
    def stage_cap(self, stage: str) -> float:
        return self.total * self.caps.get(stage, 1.0)

    def stage_remaining(self, stage: str) -> float:
        return max(0.0, self.stage_cap(stage) - self.used)

    def remaining(self) -> float:
        return max(0.0, self.total - self.used)

    def wall_elapsed(self) -> float:
        return time.time() - self.start

    def wall_remaining(self) -> float:
        return max(0.0, self.wall_seconds - self.wall_elapsed())

    def can_afford(self, stage: str, est_seconds: float) -> bool:
        return stage == "final" or (est_seconds <= self.stage_remaining(stage) and self.wall_remaining() > 0)

    def exhausted(self, stage: str) -> bool:
        return stage != "final" and (self.stage_remaining(stage) <= 0 or self.wall_remaining() <= 0)

    def charge(self, seconds: float, stage: str, idea_id: str | None = None):
        with self._lock:
            self.used += seconds
            self.jobs += 1
            self.by_stage[stage] = self.by_stage.get(stage, 0.0) + seconds
            if idea_id:
                self.by_idea[idea_id] = self.by_idea.get(idea_id, 0.0) + seconds

    # -- LLM budget ----------------------------------------------------------------------
    def try_llm_call(self) -> bool:
        with self._lock:
            if self.llm_calls >= self.llm_calls_max:
                return False
            cap = self.llm_phase_caps.get(self.phase)
            if cap is not None and self.llm_by_phase.get(self.phase, 0) >= cap:
                return False
            self.llm_calls += 1
            self.llm_by_phase[self.phase] = self.llm_by_phase.get(self.phase, 0) + 1
            return True

    def llm_remaining(self, phase: str | None = None) -> int:
        with self._lock:
            left = max(0, self.llm_calls_max - self.llm_calls)
            if phase is not None and phase in self.llm_phase_caps:
                left = min(left, max(0, self.llm_phase_caps[phase] - self.llm_by_phase.get(phase, 0)))
            return left

    def charge_literature(self, ok: bool):
        with self._lock:
            self.literature_queries += 1
            self.literature_failures += 0 if ok else 1

    def charge_llm(self, seconds: float, cost_usd: float | None):
        with self._lock:
            self.llm_seconds += seconds
            self.llm_cost_usd += cost_usd or 0.0

    # -- devices -------------------------------------------------------------------------
    @contextmanager
    def device(self, wants_gpu: bool):
        if not wants_gpu or not self.gpu_devices:
            yield "cpu"
            return
        try:
            dev = self._gpus.get(timeout=1.0)
        except queue.Empty:  # all GPUs busy -> fall back to CPU instead of stalling the pool
            yield "cpu"
            return
        try:
            yield dev
        finally:
            self._gpus.put(dev)

    def snapshot(self) -> dict:
        with self._lock:
            return {"compute_budget_s": self.total, "compute_used_s": round(self.used, 2),
                    "compute_remaining_s": round(self.remaining(), 2),
                    "wall_elapsed_s": round(self.wall_elapsed(), 2), "wall_budget_s": self.wall_seconds,
                    "jobs": self.jobs, "by_stage_s": {k: round(v, 2) for k, v in self.by_stage.items()},
                    "llm_calls": self.llm_calls, "llm_calls_max": self.llm_calls_max,
                    "llm_seconds": round(self.llm_seconds, 2), "llm_cost_usd": round(self.llm_cost_usd, 4),
                    "llm_by_phase": dict(self.llm_by_phase), "literature_queries": self.literature_queries,
                    "literature_failures": self.literature_failures,
                    "workers": self.workers, "gpus": self.gpu_devices}
