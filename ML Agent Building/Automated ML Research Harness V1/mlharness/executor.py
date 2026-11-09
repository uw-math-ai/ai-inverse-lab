"""Parallel, budgeted, isolated experiment execution."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import HARNESS_ROOT
from . import artifacts as A
from . import methods as ML

log = logging.getLogger("mlharness")


def cache_key(job: dict, data_key: str) -> str | None:
    if job["mode"] != "train_eval":
        return None
    cfg = dict(job["idea"])
    blob = {"cfg": ML.config_hash(cfg), "fraction": round(job["fraction"], 6), "data": data_key,
            "seed": job["seed"], "data_seed": job["data_seed"]}
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


class Executor:
    def __init__(self, spec, store, resources, data_key: str, python_exe: str | None = None):
        self.spec = spec
        self.store = store
        self.resources = resources
        self.data_key = data_key
        self.python = python_exe or sys.executable
        self.cache: dict[str, dict] = {}
        for art in store.query(A.RESULT):     # resume: completed experiments are never re-run
            if art.data.get("cache_key") and art.data.get("status") == "ok":
                self.cache[art.data["cache_key"]] = art.data

    def run(self, jobs: list[dict], stage: str) -> list[dict]:
        if not jobs:
            return []
        workers = max(1, min(self.resources.workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp") as pool:
            return list(pool.map(lambda j: self._run_one(j, stage), jobs))

    def _timeout(self, job, stage) -> float:
        t = float(self.spec.budget["experiment_timeout"])
        if job["mode"] == "final":
            return t * max(1, len(job["members"])) * 3
        return max(10.0, min(t, self.resources.wall_remaining()))

    def _run_one(self, job: dict, stage: str) -> dict:
        key = cache_key(job, self.data_key)
        if key and key in self.cache:
            res = dict(self.cache[key])
            res.update(exp_id=job["exp_id"], cached=True)
            art = self.store.put(A.RESULT, res, author="executor", idea_id=job["idea_id"],
                                 parents=[job["plan_artifact"]], stage=stage,
                                 summary=f"{job['exp_id']} (cached) {self._summ(res)}")
            res["artifact_id"] = art.id
            return res
        if self.resources.exhausted(stage):
            res = {"status": "skipped_budget", "exp_id": job["exp_id"], "idea_id": job["idea_id"],
                   "fraction": job.get("fraction"), "error": "compute budget for this stage exhausted"}
            art = self.store.put(A.RESULT, res, author="executor", idea_id=job["idea_id"],
                                 parents=[job["plan_artifact"]], stage=stage, summary=f"{job['exp_id']} skipped (budget)")
            res["artifact_id"] = art.id
            return res
        job_dir = Path(job["job_dir"])
        threads = str(job.get("threads", 1))
        env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                   PYTHONPATH=str(HARNESS_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   PYTHONWARNINGS="ignore")
        with self.resources.device(job.get("wants_gpu", False)) as dev:
            job["device"] = dev
            jp = job_dir / "job.json"
            jp.write_text(json.dumps(job, indent=1, default=str))
            t0 = time.time()
            timeout = self._timeout(job, stage)
            try:
                proc = subprocess.run([self.python, "-m", "mlharness.runner", str(jp)], cwd=str(HARNESS_ROOT),
                                      env=env, capture_output=True, text=True, timeout=timeout)
                stderr = proc.stderr[-3000:]
                timed_out = False
            except subprocess.TimeoutExpired:
                stderr, timed_out = "", True
            elapsed = time.time() - t0
        self.resources.charge(elapsed, stage, job["idea_id"])
        rp = job_dir / "result.json"
        if timed_out:
            res = {"status": "timeout", "error": f"timeout after {timeout:.0f}s"}
        elif rp.exists():
            res = json.loads(rp.read_text())
        else:
            res = {"status": "error", "error": "runner produced no result", "traceback": stderr}
        res.update(exp_id=job["exp_id"], idea_id=job["idea_id"], fraction=job.get("fraction"),
                   wall_seconds=round(elapsed, 3), device=dev, cache_key=key, stage=stage)
        art = self.store.put(A.RESULT, res, author="executor", idea_id=job["idea_id"],
                             parents=[job["plan_artifact"]], stage=stage, files=[str(job_dir)],
                             summary=f"{job['exp_id']} {self._summ(res)}")
        self.store.put(A.RESOURCE, {"type": "experiment", "exp_id": job["exp_id"], "seconds": round(elapsed, 3),
                                    "device": dev, "threads": job.get("threads"), "stage": stage},
                       author="executor", idea_id=job["idea_id"], parents=[art.id], stage=stage,
                       summary=f"{job['exp_id']}: {elapsed:.1f}s on {dev}")
        res["artifact_id"] = art.id
        if key and res.get("status") == "ok":
            self.cache[key] = res
        log.info("  %s %-9s %-14s f=%.2f %s (%.1fs)", job["exp_id"], job["idea_id"],
                 job["idea"]["method"] if job["mode"] == "train_eval" else "FINAL",
                 job.get("fraction") or 1.0, self._summ(res), elapsed)
        return res

    def _summ(self, res):
        if res.get("status") != "ok":
            return f"{res.get('status')}: {str(res.get('error'))[:90]}"
        part = res.get("val") or res.get("test") or {}
        return " ".join(f"{k}={v:.4g}" for k, v in part.items() if v is not None)
