"""Local experiment API and dashboard. Run: python app.py --port 8765."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Event, Lock
from urllib.parse import urlsplit
from uuid import uuid4

from forward_sim.engine import Cancelled, simulate
from forward_sim.export import csv_text
from forward_sim.graphs import GRAPHS
from forward_sim.model import MAX_STEPS, MAX_TRIALS, MAX_WORK, build_kernel, parse_config

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.jobs: dict[str, dict] = {}
        self.events: dict[str, Event] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
        for path in directory.glob("*.json"):
            if not RUN_ID.fullmatch(path.stem):
                continue
            try:
                result = json.loads(path.read_text())
                self.jobs[path.stem] = self._summary(path.stem, result)
            except (OSError, ValueError, KeyError, TypeError):
                continue

    @staticmethod
    def _summary(run_id: str, result: dict) -> dict:
        return {"id": run_id, "status": "completed", "config": result["config"],
                "created_at": result["created_at"], "metrics": result["metrics"],
                "progress": {"fraction": 1.0}}

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((dict(job) for job in self.jobs.values()),
                          key=lambda j: j["created_at"], reverse=True)

    def get(self, run_id: str) -> dict:
        with self.lock:
            if run_id not in self.jobs:
                raise KeyError("Experiment not found.")
            return dict(self.jobs[run_id])

    def result(self, run_id: str) -> dict:
        job = self.get(run_id)
        if job["status"] != "completed":
            raise ValueError("Results are available when the experiment completes.")
        return json.loads((self.directory / f"{run_id}.json").read_text())

    def submit(self, raw: dict) -> dict:
        config = parse_config(raw)
        run_id, event = uuid4().hex, Event()
        with self.lock:
            if sum(job["status"] in ("queued", "running", "cancelling") for job in self.jobs.values()) >= 4:
                raise ValueError("The experiment queue is full (4 runs). Wait or cancel a run.")
            job = {"id": run_id, "status": "queued", "config": config.as_dict(),
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "progress": {"fraction": 0}}
            self.jobs[run_id], self.events[run_id] = job, event
        self.pool.submit(self._run, run_id, config, event)
        return self.get(run_id)

    def cancel(self, run_id: str) -> dict:
        with self.lock:
            if run_id not in self.jobs:
                raise KeyError("Experiment not found.")
            if self.jobs[run_id]["status"] in ("queued", "running", "cancelling"):
                self.events[run_id].set()
                self.jobs[run_id]["status"] = "cancelling"
        return self.get(run_id)

    def _run(self, run_id, config, event):
        def progress(value):
            with self.lock:
                self.jobs[run_id]["progress"] = value
        try:
            if event.is_set():
                raise Cancelled()
            with self.lock:
                self.jobs[run_id]["status"] = "running"
            result = simulate(config, progress=progress, cancelled=event.is_set)
            if event.is_set():
                raise Cancelled()
            result["id"] = run_id
            destination = self.directory / f"{run_id}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, allow_nan=False))
            temporary.replace(destination)
            with self.lock:
                self.jobs[run_id] = self._summary(run_id, result)
        except Cancelled:
            with self.lock:
                self.jobs[run_id]["status"] = "cancelled"
        except Exception as exc:
            with self.lock:
                self.jobs[run_id].update(status="failed", error=str(exc))
        finally:
            with self.lock:
                self.events.pop(run_id, None)

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True)


def make_handler(store: RunStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and isinstance(args[0], str) and args[0].startswith("GET /api/runs"):
                return
            super().log_message(format, *args)

        def respond(self, data, status=200, content_type="application/json; charset=utf-8", filename=None):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/api/graphs":
                    return self.respond({"graphs": [g.as_dict() for g in GRAPHS.values()],
                                         "limits": {"T": MAX_STEPS, "N": MAX_TRIALS, "work": MAX_WORK}})
                if path == "/api/runs":
                    return self.respond({"runs": store.list()})
                match = re.fullmatch(r"/api/runs/([0-9a-f]{32})(?:/(json|csv))?", path)
                if match:
                    run_id, export = match.groups()
                    if export:
                        result = store.result(run_id)
                        if export == "csv":
                            return self.respond(csv_text(result), content_type="text/csv; charset=utf-8", filename=f"{run_id}.csv")
                        return self.respond(result, filename=f"{run_id}.json")
                    job = store.get(run_id)
                    if job["status"] == "completed":
                        job["result"] = store.result(run_id)
                    return self.respond(job)
                static = {"/": (ROOT/"web/index.html", "text/html; charset=utf-8"),
                          "/app.js": (ROOT/"web/app.js", "text/javascript; charset=utf-8"),
                          "/styles.css": (ROOT/"web/styles.css", "text/css; charset=utf-8"),
                          "/readme.md": (ROOT/"readme.md", "text/plain; charset=utf-8")}
                if path in static:
                    file, content_type = static[path]
                    return self.respond(file.read_bytes(), content_type=content_type)
                return self.respond({"error": "Not found."}, 404)
            except KeyError as exc:
                return self.respond({"error": str(exc)}, 404)
            except ValueError as exc:
                return self.respond({"error": str(exc)}, 409)
            except OSError:
                return self.respond({"error": "Could not read the requested file."}, 500)

        def do_POST(self):
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.respond({"error": "Cross-origin writes are not allowed."}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > 65536:
                    raise ValueError("Request is too large (maximum 64 KB).")
                raw = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/validate":
                    config = parse_config(raw)
                    K, kill, P = build_kernel(config)
                    return self.respond({"config": config.as_dict(), "kernel": K.tolist(),
                                         "killing_probabilities": kill.tolist(), "augmented_transition": P.tolist()})
                if path == "/api/runs":
                    return self.respond(store.submit(raw), 202)
                match = re.fullmatch(r"/api/runs/([0-9a-f]{32})/cancel", path)
                if match:
                    return self.respond(store.cancel(match.group(1)))
                return self.respond({"error": "Not found."}, 404)
            except KeyError as exc:
                return self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc:
                return self.respond({"error": str(exc)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/runs")
    args = parser.parse_args()
    store = RunStore(args.data_dir)
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    except OSError:
        store.close()
        raise
    print(f"Forward Simulation Lab: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
