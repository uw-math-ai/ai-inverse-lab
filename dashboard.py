"""Local dashboard server for inspecting and managing mathematical research runs.

Run ``python -m workflow.dashboard`` and open http://127.0.0.1:8766.
Browser requests can only access managed run artifacts and bounded CLI jobs.
No model API credentials are returned to the browser.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from .benchmarks import BENCHMARK_IDS, DEFAULT_BENCHMARK_ID, load_benchmark
from .credentials import openai_key_configured
from .evaluation import aggregate_runs
from .records import utc_now, write_json
from .response_call import apply_review, reviewed_summary
from .runtime import RunConfig
from .verification import lean_status


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "runs"
WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_DIR = Path(__file__).resolve().parent.parent
MAX_REQUEST_BYTES = 256 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TAIL_BYTES = 8 * 1024 * 1024
MAX_PENDING_JOBS = 4
JOB_TIMEOUT_SECONDS = 3600
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,119}\Z")
RUN_FILES = ("report.md", "summary.json", "events.jsonl", "iterations.jsonl", "manifest.json",
             "summary_adjudicated.json", "proof.lean", "lean-verification.json", "agent_evaluations.jsonl")
KATEX_FONT_STEMS = (
    "KaTeX_AMS-Regular", "KaTeX_Caligraphic-Bold", "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold", "KaTeX_Fraktur-Regular", "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic", "KaTeX_Main-Italic", "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic", "KaTeX_Math-Italic", "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic", "KaTeX_SansSerif-Regular", "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular", "KaTeX_Size2-Regular", "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular", "KaTeX_Typewriter-Regular",
)
STATIC_FILES = frozenset({
    "index.html", "app.js", "styles.css", "favicon.svg", "math-renderer.js", "math-copy.js",
    "vendor/katex/katex.min.js", "vendor/katex/katex.min.css",
    "vendor/katex/contrib/auto-render.min.js", "vendor/katex/LICENSE", "vendor/katex/README.md",
    *(f"vendor/katex/fonts/{stem}.{extension}"
      for stem in KATEX_FONT_STEMS for extension in ("woff2", "woff", "ttf")),
})
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml; charset=utf-8",
    ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
    ".md": "text/plain; charset=utf-8", "": "text/plain; charset=utf-8",
}
DEFAULTS = {"backend": "demo", "model": "gpt-4o-mini", "rounds": 3,
            "max_output_tokens": 2400, "max_simulator_calls": 0, "seed": 2026,
            "benchmark_id": DEFAULT_BENCHMARK_ID}
LIMITS = {"rounds": {"min": 1, "max": 20}, "max_output_tokens": {"min": 256, "max": 16000},
          "max_simulator_calls": {"min": 0, "max": 80}, "seed": {"min": 0, "max": 2**31 - 1},
          "trials": {"min": 1, "max": 10}, "max_queued_jobs": MAX_PENDING_JOBS,
          "max_active_jobs": 1, "job_timeout_seconds": JOB_TIMEOUT_SECONDS}


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _nonnegative_int(value: object, default: int = 0) -> int:
    return value if type(value) is int and value >= 0 else default


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value: {value}")


class DashboardStore:
    """Confined artifact catalog and one persisted background job queue."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, *, start_worker: bool = True):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.confined(self.root / ".dashboard" / "jobs")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.confined(self.root / "managed").mkdir(exist_ok=True)
        self.instance_lock = None
        self._acquire_lock()
        self.condition = threading.Condition(threading.RLock())
        self.review_lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.process: subprocess.Popen | None = None
        self.process_job_id: str | None = None
        self.closed = False
        for path in sorted(self.metadata_dir.glob("*.json")):
            saved = self.read_object(path)
            if not saved or not _valid_id(saved.get("id")) or saved["id"] != path.stem:
                continue
            if saved.get("kind") not in ("run", "compare") or not isinstance(saved.get("config"), dict):
                continue
            if saved.get("status") in ("queued", "running"):
                recovery = self._recover_process(saved)
                saved.update(status="interrupted", finished_at=utc_now(),
                             error="The previous dashboard stopped before this job completed. " + recovery + " No automatic retry was started.")
                self._persist(saved)
            self.jobs[saved["id"]] = saved
        self.worker = threading.Thread(target=self._work, name="research-dashboard-jobs", daemon=True)
        if start_worker:
            self.worker.start()

    def _acquire_lock(self) -> None:
        """Exclude another dashboard before inspecting or recovering its jobs."""
        path = self.confined(self.root / ".dashboard" / "instance.lock")
        handle = path.open("a+b")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b" ")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except (OSError, ImportError) as exc:
            handle.close()
            raise APIError("Another dashboard already manages this run directory, or its instance lock is unavailable.", 409) from exc
        self.instance_lock = handle

    def _release_lock(self) -> None:
        if self.instance_lock is not None:
            self.instance_lock.close()
            self.instance_lock = None

    def confined(self, path: Path) -> Path:
        """Reject traversal and every symlink below the operator-selected root."""
        path = Path(path).absolute()
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise APIError("Artifact path is outside the managed run directory.", 403) from exc
        current = self.root
        for part in relative.parts:
            if part in ("..", "."):
                raise APIError("Invalid artifact path.", 403)
            current /= part
            if current.is_symlink():
                raise APIError("Symlinked artifacts are not accessible.", 403)
        try:
            current.resolve().relative_to(self.root)
        except ValueError as exc:
            raise APIError("Artifact path escapes the managed run directory.", 403) from exc
        return current

    def read_object(self, path: Path) -> dict | None:
        try:
            path = self.confined(path)
            if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
                return None
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, UnicodeError, APIError):
            return None

    def read_lines(self, path: Path, limit: int | None = None) -> list[dict]:
        """Ignore a truncated final event without hiding previous valid records."""
        values: deque = deque(maxlen=limit)
        try:
            path = self.confined(path)
            with path.open("rb") as handle:
                size = path.stat().st_size
                if limit is not None and size > MAX_TAIL_BYTES:
                    handle.seek(size - MAX_TAIL_BYTES)
                    handle.readline()  # Drop a partial line at the seek boundary.
                elif limit is None and size > MAX_JSON_BYTES:
                    return []
                for line in handle:
                    if len(line) > MAX_JSON_BYTES:
                        continue
                    try:
                        value = json.loads(line, parse_constant=_reject_constant)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(value, dict):
                        values.append(value)
        except (OSError, APIError):
            pass
        return list(values)

    def _walk(self):
        for directory, folders, files in os.walk(self.root, followlinks=False):
            current = Path(directory)
            depth = len(current.relative_to(self.root).parts)
            folders[:] = sorted(name for name in folders
                                if name != ".dashboard" and not (current / name).is_symlink() and depth < 8)
            try:
                self.confined(current)
            except APIError:
                folders[:] = []
                continue
            yield current, set(files)

    def _safe_review_directory(self, directory: Path) -> None:
        self.confined(directory / "reviews")
        self.confined(directory / "summary.json")
        self.confined(directory / "summary_adjudicated.json")
        self.confined(directory / "summary_adjudicated.json.tmp")
        for path in (directory / "reviews").glob("*.json"):
            self.confined(path)
            if self.read_object(path) is None:
                raise APIError("A saved review is unreadable; inspect the local review artifact before adding another.", 409)

    def _run_data(self, directory: Path, files: set[str]) -> dict:
        manifest = self.read_object(directory / "manifest.json") or {}
        original = self.read_object(directory / "summary.json")
        summary = original
        reviews = []
        warnings = []
        if "manifest.json" in files and not manifest:
            warnings.append("Manifest is missing, corrupt, or inaccessible.")
        if "summary.json" in files and original is None:
            warnings.append("Summary is incomplete, corrupt, or inaccessible; showing preserved iteration records.")
        if original is not None:
            try:
                self._safe_review_directory(directory)
                review_paths = sorted((directory / "reviews").glob("*.json"))
                if review_paths:
                    summary = reviewed_summary(directory)
                    reviews = [self.read_object(path) for path in review_paths]
                    reviews = [review for review in reviews if review is not None]
            except (APIError, ValueError, KeyError, OSError, TypeError, AttributeError):
                warnings.append("Saved reviews could not be revalidated; showing original provisional results.")
        records = summary.get("records") if isinstance(summary, dict) else None
        if not isinstance(records, list):
            records = self.read_lines(directory / "iterations.jsonl")
        records = [record for record in records if isinstance(record, dict)]
        source = {**manifest, **_object(summary)}
        run_id = source.get("run_id", directory.name)
        if not _valid_id(run_id):
            run_id = directory.name
        last = records[-1] if records else {}
        score = _object(last.get("score"))
        job_id = next((part for part in directory.relative_to(self.root).parts if part in self.jobs), None)
        job = self.jobs.get(job_id, {})
        status = source.get("status") if summary else None
        if status not in ("completed", "failed", "cancelled", "interrupted", "running", "incomplete"):
            status = job.get("status") if job else "incomplete"
        if status == "queued":
            status = "incomplete"
        comparison_id = next((part for part in reversed(directory.relative_to(self.root).parts[:-1])
                              if part.startswith("comparison-") and _valid_id(part)), None)
        created = source.get("created_at")
        if not isinstance(created, str):
            created = datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc).isoformat()
        compact = {
            "id": run_id, "run_id": run_id, "mode": source.get("mode", "unknown"),
            "backend": source.get("backend", "unknown"), "model": source.get("model", "unknown"),
            "status": status or "incomplete", "created_at": created, "iteration_count": len(records),
            "model_calls": _nonnegative_int(source.get("model_calls", last.get("model_calls"))),
            "max_model_calls": _nonnegative_int(source.get("max_model_calls")),
            "usage": _object(source.get("usage")), "final_score": _nonnegative_int(score.get("score", source.get("final_score"))),
            "provisional": score.get("provisional") is not False, "reviewed": bool(reviews),
            "score_evidence": score.get("evidence_type"),
            "verification_status": _object(last.get("verification")).get("status"),
            "benchmark_id": source.get("benchmark_id", _object(source.get("benchmark")).get("id")),
            "comparison_id": comparison_id, "job_id": job_id, "error": source.get("error") or job.get("error"),
        }
        if warnings:
            compact["warnings"] = warnings
        available = []
        for filename in RUN_FILES:
            try:
                if self.confined(directory / filename).is_file():
                    available.append(filename)
            except APIError:
                continue
        return {"run": compact, "manifest": manifest, "summary": summary, "records": records,
                "reviews": reviews, "available_files": available, "directory": directory}

    def catalog(self) -> tuple[dict[str, dict], dict[str, dict]]:
        runs, comparisons = {}, {}
        comparison_paths = {}
        for directory, files in self._walk():
            if directory.name.startswith("comparison-") and _valid_id(directory.name):
                comparison_paths[directory.name] = directory
            if not files.intersection({"manifest.json", "summary.json", "iterations.jsonl", "events.jsonl"}):
                continue
            data = self._run_data(directory, files)
            identifier = data["run"]["id"]
            if _valid_id(identifier) and identifier not in runs:
                runs[identifier] = data
        for identifier, directory in comparison_paths.items():
            run_ids = [key for key, value in runs.items() if value["run"]["comparison_id"] == identifier]
            saved = self.read_object(directory / "comparison.json")
            if not run_ids and saved is None:
                continue
            summaries = [runs[key]["summary"] for key in run_ids if runs[key]["summary"] is not None]
            aggregate = saved or {}
            if summaries:
                try:
                    aggregate = aggregate_runs(summaries)
                except (ValueError, TypeError, KeyError, AttributeError):
                    pass
            # CLI comparison files contain machine-specific absolute paths.
            aggregate = {key: value for key, value in aggregate.items() if key != "run_directories"}
            job = next((self.jobs[runs[key]["run"]["job_id"]] for key in run_ids
                        if runs[key]["run"]["job_id"] in self.jobs), None)
            if job is not None:
                status = job["status"]
            elif saved is not None and isinstance(saved.get("by_mode"), dict) and isinstance(saved.get("paired"), dict):
                # This is orchestration completion, not mathematical verification.
                failed = any(_nonnegative_int(_object(mode).get("failed_runs"))
                             for mode in _object(aggregate.get("by_mode")).values())
                status = "failed" if failed else "completed"
            else:
                status = "incomplete"
            comparisons[identifier] = {"id": identifier, "aggregate": aggregate, "run_ids": run_ids,
                                       "status": status,
                                       "backend": runs[run_ids[0]]["run"]["backend"] if run_ids else saved.get("backend"),
                                       "created_at": min((runs[key]["run"]["created_at"] for key in run_ids), default="")}
        return runs, comparisons

    def _job_public(self, job: dict, runs: dict | None = None, comparisons: dict | None = None) -> dict:
        public = {key: deepcopy(value) for key, value in job.items()
                  if key in {"id", "kind", "status", "created_at", "started_at", "finished_at", "config", "error"}}
        public["run_ids"] = [key for key, value in (runs or {}).items() if value["run"]["job_id"] == job["id"]]
        public["comparison_id"] = next((key for key, value in (comparisons or {}).items()
                                        if set(value["run_ids"]) & set(public["run_ids"])), None)
        return public

    def workspace(self) -> dict:
        with self.condition:
            runs, comparisons = self.catalog()
            jobs = [self._job_public(job, runs, comparisons) for job in self.jobs.values()]
        return {"benchmark": load_benchmark(),
                "benchmarks": [{"id": key, "title": load_benchmark(key)["title"],
                                "short_title": load_benchmark(key)["short_title"],
                                "formalization": load_benchmark(key)["verification_contract"]["status"]}
                               for key in BENCHMARK_IDS],
                "settings": {"api_key_configured": openai_key_configured(),
                             "defaults": deepcopy(DEFAULTS), "limits": deepcopy(LIMITS), "lean": lean_status()},
                "runs": sorted((data["run"] for data in runs.values()), key=lambda run: run["created_at"], reverse=True),
                "jobs": sorted(jobs, key=lambda job: job["created_at"], reverse=True),
                "comparisons": sorted(comparisons.values(), key=lambda comparison: comparison["created_at"], reverse=True)}

    def run_detail(self, identifier: str) -> dict:
        if not _valid_id(identifier):
            raise APIError("Unknown run.", 404)
        with self.condition:
            runs, _ = self.catalog()
        if identifier not in runs:
            raise APIError("Unknown run.", 404)
        detail = runs[identifier]
        directory = detail.pop("directory")
        detail["events"] = self.read_lines(directory / "events.jsonl", limit=100)
        return detail

    def comparison_detail(self, identifier: str) -> dict:
        if not _valid_id(identifier):
            raise APIError("Unknown comparison.", 404)
        _, comparisons = self.catalog()
        if identifier not in comparisons:
            raise APIError("Unknown comparison.", 404)
        return comparisons[identifier]

    def run_file(self, identifier: str, filename: str) -> Path:
        if filename not in RUN_FILES or not _valid_id(identifier):
            raise APIError("Unknown artifact.", 404)
        runs, _ = self.catalog()
        if identifier not in runs:
            raise APIError("Unknown run.", 404)
        path = self.confined(runs[identifier]["directory"] / filename)
        if not path.is_file():
            raise APIError("Artifact is not yet available.", 404)
        return path

    def submit_review(self, identifier: str, review: dict) -> dict:
        if not _valid_id(identifier):
            raise APIError("Unknown run.", 404)
        with self.review_lock:
            runs, _ = self.catalog()
            if identifier not in runs:
                raise APIError("Unknown run.", 404)
            data = runs[identifier]
            if data["summary"] is None:
                raise APIError("Wait for a completed run summary before submitting a review.", 409)
            directory = data["directory"]
            self._safe_review_directory(directory)
            if list((directory / "reviews").glob("*.json")):
                try:
                    reviewed_summary(directory)
                except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                    raise APIError("An existing review failed validation; inspect it before adding another review.", 409) from exc
            self.confined(self.metadata_dir)
            path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="review-", dir=self.metadata_dir,
                                                 encoding="utf-8", delete=False) as temporary:
                    path = Path(temporary.name)
                    json.dump(review, temporary, allow_nan=False)
                saved_path = apply_review(directory, path)
                saved = self.read_object(saved_path)
                if saved is None:
                    raise APIError("The saved review could not be read.", 500)
                return {**saved, "run_id": identifier, "filename": saved_path.name}
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise APIError(str(exc)) from exc
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)

    def _persist(self, job: dict) -> None:
        destination = self.confined(self.metadata_dir / (job["id"] + ".json"))
        self.confined(destination.with_name(destination.name + ".tmp"))
        write_json(destination, job)

    def _validate_job(self, payload: dict) -> dict:
        allowed = {"kind", "mode", "backend", "model", "rounds", "max_output_tokens", "max_simulator_calls", "seed", "trials", "benchmark_id"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise APIError("Only documented job configuration fields are allowed.")
        kind = payload.get("kind", "run")
        if kind not in ("run", "compare"):
            raise APIError("kind must be run or compare.")
        config = {**DEFAULTS, "mode": "multi", "trials": 1, **payload}
        config.pop("kind", None)
        if not isinstance(config["model"], str) or MODEL_NAME.fullmatch(config["model"]) is None:
            raise APIError("model must be a model identifier of at most 120 characters.")
        if type(config["trials"]) is not int or not 1 <= config["trials"] <= 10:
            raise APIError("trials must be an integer from 1 to 10.")
        try:
            RunConfig(**{key: value for key, value in config.items() if key != "trials"})
        except (TypeError, ValueError) as exc:
            raise APIError(str(exc)) from exc
        if config["seed"] + (config["trials"] - 1 if kind == "compare" else 0) > LIMITS["seed"]["max"]:
            raise APIError("The seed plus trial count exceeds the supported seed range.")
        if config["backend"] == "openai" and not openai_key_configured():
            raise APIError("No OpenAI API key configured. Run python -m workflow.response_call configure-openai in a terminal.", 409)
        return {"kind": kind, "config": config}

    def submit_job(self, payload: dict) -> dict:
        validated = self._validate_job(payload)
        with self.condition:
            if self.closed:
                raise APIError("The dashboard is shutting down.", 503)
            pending = sum(job["status"] in ("queued", "running") for job in self.jobs.values())
            if pending >= MAX_PENDING_JOBS:
                raise APIError("The job queue is full; wait for a job to finish or cancel one.", 409)
            identifier = "job-" + uuid4().hex[:16]
            self.confined(self.root / "managed" / identifier).mkdir(exist_ok=False)
            job = {"id": identifier, **validated, "status": "queued", "created_at": utc_now(), "error": None}
            self._persist(job)
            self.jobs[identifier] = job
            self.condition.notify_all()
            return self._job_public(job)

    def _command(self, job: dict) -> list[str]:
        config = job["config"]
        directory = self.confined(self.root / "managed" / job["id"])
        command = [sys.executable, "-m", "workflow.response_call", job["kind"],
                   "--output", str(directory)]
        for key in ("backend", "model", "rounds", "max_output_tokens", "max_simulator_calls", "seed"):
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
        if "benchmark_id" in config:
            command.extend(["--benchmark", config["benchmark_id"]])
        command.extend(["--mode", config["mode"]] if job["kind"] == "run" else ["--trials", str(config["trials"])])
        return command

    def _recover_process(self, job: dict) -> str:
        """Reap an orphan only after checking its PID, group, and unique argv.

        Persisting a PID alone is insufficient: an unrelated process may reuse
        it after a restart. The managed output path in the full command is a
        per-job identity. A platform/identity mismatch is surfaced, never killed.
        """
        pid = job.get("process_id")
        if type(pid) is not int or pid <= 0:
            return "No child process identity was recorded; a prior child cannot be ruled out."
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "Its previously recorded child process is no longer running."
        except PermissionError:
            return "The prior child could not be inspected and may still be running."
        if os.name != "posix":
            return "The prior child may still be running; process identity recovery is unavailable on this platform."
        try:
            expected = self._command(job)
            if job.get("process_command") != expected or os.getpgid(pid) != pid:
                return "The stored process identity no longer matches; no process was stopped. Inspect any possible orphan locally."
            found = subprocess.run(["/bin/ps", "-p", str(pid), "-ww", "-o", "command="],
                                   capture_output=True, text=True, timeout=2, check=False)
            if found.returncode != 0:
                return "Its previously recorded child process is no longer running."
            if found.stdout.strip() != " ".join(expected):
                return "The process command no longer matches; no process was stopped. Inspect any possible orphan locally."
            os.killpg(pid, signal.SIGTERM)
            return "The matching orphaned child process was signalled to stop; preserved artifacts remain available."
        except (OSError, ValueError, KeyError, APIError, subprocess.SubprocessError):
            return "The prior child could not be safely identified and may still be running."

    @staticmethod
    def _terminate(process: subprocess.Popen, *, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    def cancel_job(self, identifier: str) -> dict:
        if not _valid_id(identifier):
            raise APIError("Unknown job.", 404)
        with self.condition:
            job = self.jobs.get(identifier)
            if job is None:
                raise APIError("Unknown job.", 404)
            if job["status"] in ("queued", "running"):
                job.update(status="cancelled", finished_at=utc_now(), error="Cancelled locally; existing artifacts were preserved.")
                self._persist(job)
                if self.process_job_id == identifier and self.process is not None:
                    self._terminate(self.process)
                self.condition.notify_all()
            runs, comparisons = self.catalog()
            return self._job_public(job, runs, comparisons)

    def _work(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or any(job["status"] == "queued" for job in self.jobs.values()))
                if self.closed:
                    return
                job = next(job for job in self.jobs.values() if job["status"] == "queued")
                job.update(status="running", started_at=utc_now())
                self._persist(job)
            process = None
            try:
                output = self.confined(self.root / "managed" / job["id"] / "job.log")
                with output.open("ab") as log:
                    command = self._command(job)
                    process = subprocess.Popen(command, cwd=PROJECT_DIR, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, shell=False,
                                               start_new_session=os.name == "posix")
                    with self.condition:
                        self.process, self.process_job_id = process, job["id"]
                        job.update(process_id=process.pid, process_command=command)
                        self._persist(job)
                        if job["status"] != "running" or self.closed:
                            self._terminate(process)
                    start = time.monotonic()
                    interrupted_at = None
                    while process.poll() is None:
                        with self.condition:
                            stopped = self.closed or job["status"] != "running"
                        if stopped:
                            interrupted_at = interrupted_at or time.monotonic()
                            self._terminate(process, force=time.monotonic() - interrupted_at > 2)
                        if time.monotonic() - start > JOB_TIMEOUT_SECONDS:
                            self._terminate(process, force=True)
                            raise TimeoutError("Job exceeded the one-hour dashboard time limit; partial artifacts were preserved.")
                        time.sleep(0.05)
                    with self.condition:
                        if job["status"] == "running":
                            job.update(status="completed" if process.returncode == 0 else "failed",
                                       finished_at=utc_now(), error=None if process.returncode == 0 else
                                       f"Workflow exited with code {process.returncode}. Inspect the preserved run records.")
            except Exception as exc:
                if process is not None:
                    self._terminate(process, force=True)
                    process.wait(timeout=5)
                with self.condition:
                    if job["status"] == "running":
                        job.update(status="failed", finished_at=utc_now(), error=str(exc))
            finally:
                with self.condition:
                    self.process, self.process_job_id = None, None
                    self._persist(job)
                    self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            for job in self.jobs.values():
                if job["status"] in ("queued", "running"):
                    job.update(status="interrupted", finished_at=utc_now(),
                               error="Dashboard stopped; partial artifacts were preserved. No automatic retry was started.")
                    self._persist(job)
            if self.process is not None:
                self._terminate(self.process)
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=5)
        if not self.worker.is_alive():
            self._release_lock()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: DashboardStore, web_dir: Path = WEB_DIR):
        if address[0] != "127.0.0.1":
            raise ValueError("The dashboard binds only to 127.0.0.1.")
        self.store, self.web_dir = store, Path(web_dir).resolve()
        super().__init__(address, DashboardHandler)

    def server_close(self) -> None:
        super().server_close()
        self.store.close()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format: str, *args) -> None:
        # Request paths may contain untrusted text; do not print raw payloads.
        pass

    def _check_host(self, *, write: bool = False) -> None:
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        host = self.headers.get("Host", "")
        if host not in allowed or len(self.headers.get_all("Host", [])) != 1:
            raise APIError("Only this dashboard's localhost origin is allowed.", 403)
        if write and self.headers.get("Origin") != "http://" + host:
            raise APIError("Writes require the dashboard's same-origin browser request.", 403)
        if write and self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise APIError("Cross-origin writes are not allowed.", 403)

    def _headers(self, status: int, content_type: str, length: int, *, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _body(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Chunked request bodies are not supported.", 400)
        if self.headers.get_content_type() != "application/json":
            raise APIError("Expected application/json.", 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise APIError("Invalid request length.") from exc
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise APIError("JSON request is empty or exceeds 256 KiB.", 413)
        body = self.rfile.read(length)
        if len(body) != length:
            raise APIError("Incomplete request body.")
        try:
            payload = json.loads(body, parse_constant=_reject_constant)
        except (ValueError, UnicodeError) as exc:
            raise APIError("Invalid JSON request.") from exc
        if not isinstance(payload, dict):
            raise APIError("Expected a JSON object.")
        return payload

    def _parts(self) -> list[str]:
        path = unquote(urlsplit(self.path).path)
        if "\\" in path or "\x00" in path or any(part in (".", "..") for part in path.split("/")):
            raise APIError("Invalid resource path.", 404)
        return path.strip("/").split("/") if path != "/" else []

    def do_GET(self) -> None:
        try:
            self._check_host()
            parts = self._parts()
            if parts == ["api", "workspace"]:
                self._json(self.server.store.workspace())
            elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self._json(self.server.store.run_detail(parts[2]))
            elif len(parts) == 3 and parts[:2] == ["api", "comparisons"]:
                self._json(self.server.store.comparison_detail(parts[2]))
            elif len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "files":
                path = self.server.store.run_file(parts[2], parts[4])
                with path.open("rb") as handle:
                    length = os.fstat(handle.fileno()).st_size
                    self._headers(200, "application/octet-stream", length, filename=path.name)
                    while chunk := handle.read(64 * 1024):
                        self.wfile.write(chunk)
            elif ("/".join(parts) if parts else "index.html") in STATIC_FILES:
                relative = "/".join(parts) if parts else "index.html"
                path = self.server.web_dir
                for part in relative.split("/"):
                    path /= part
                    if path.is_symlink():
                        raise APIError("Dashboard asset is unavailable.", 404)
                if not path.is_file() or not path.resolve().is_relative_to(self.server.web_dir):
                    raise APIError("Dashboard asset is unavailable.", 404)
                body = path.read_bytes()
                self._headers(200, STATIC_TYPES[path.suffix], len(body))
                self.wfile.write(body)
            else:
                raise APIError("Unknown resource.", 404)
        except APIError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._json({"error": "Unable to read this local artifact; inspect its files for malformed or incomplete data."}, 500)

    def do_POST(self) -> None:
        try:
            self._check_host(write=True)
            parts, payload = self._parts(), self._body()
            if parts == ["api", "jobs"]:
                self._json(self.server.store.submit_job(payload), 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                if payload:
                    raise APIError("Cancellation expects an empty JSON object.")
                self._json(self.server.store.cancel_job(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "reviews":
                self._json(self.server.store.submit_review(parts[2], payload), 201)
            else:
                raise APIError("Unknown resource.", 404)
        except APIError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._json({"error": "The operation could not be completed; existing run artifacts were preserved."}, 500)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Local run root; browser requests cannot override this directory.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    store = DashboardStore(args.data_dir)
    try:
        server = DashboardServer(("127.0.0.1", args.port), store)
    except OSError:
        store.close()
        raise
    print(f"Research dashboard: http://127.0.0.1:{args.port}", flush=True)
    def terminate_dashboard(signum, frame):
        # SIGTERM (including a task runner's Stop action) must clean up children
        # started in their own session, just as Ctrl+C does.
        raise KeyboardInterrupt

    previous_term_handler = signal.signal(signal.SIGTERM, terminate_dashboard)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, previous_term_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
