"""Spawn, watch, cancel and collect subagents from a swarm agent's container.

Copy this folder to ``agents/shared/`` in the registry and import it with::

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
    import platform_swarm as swarm

Only agents whose manifest has a ``"swarm"`` section can spawn subagents. Each
subagent runs in its own Docker container, on a worker server and provider
account the researcher approved at launch. It runs code from this run's
source snapshot. A typical orchestrator::

    swarm.spawn("shard-1", command=["python", "worker.py"], inputs={"shard": 1},
                server="doppio", connection="codex-lab-2")
    swarm.spawn("shard-2", inputs={"shard": 2}, connection="auto", tier="sol")
    for child in swarm.wait(["shard-1", "shard-2"], timeout=3600):
        print(child["id"], child["status"], child["result_dir"])
        result = swarm.result(child["id"])          # parsed result.json
        files = swarm.output_dir(child["id"])       # the child's /output files

Everything here is plain files: requests go to ``AGENT_SWARM_REQUESTS``,
per-child status appears in ``AGENT_SWARM_CHILDREN`` and finished results in
``AGENT_SWARM_RESULTS/<id>``. See docs/SWARM.md for the full contract.

Inside a subagent, ``parent_files()`` returns the directory of files the
parent handed over (or None) and ``child_id()`` its swarm ID.

GPU work: an agent running on a CPU server can run code it wrote on an
approved GPU server and get the outputs back::

    job = swarm.gpu_run("train-1", "/work/train", "train.py", timeout_seconds=3600)
    if job["status"] == "succeeded":
        metrics = (job["output_dir"] / "metrics.json").read_text()

The folder is copied to the GPU job's read-only ``/run/parent-files``; the job
starts in ``/work`` (writable disk, for checkpoints), writes results to
``/output`` and has no network access or provider credential.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_sequence = 0


class SwarmUnavailable(RuntimeError):
    """This run cannot spawn subagents (no swarm section, or a subagent itself)."""


def available() -> bool:
    return bool(os.environ.get("AGENT_SWARM_REQUESTS"))


def _require() -> None:
    if not available():
        raise SwarmUnavailable("Subagents need a \"swarm\" section in agent.json and cannot be nested")


def _read_json(path: Path) -> dict | None:
    # Files are replaced atomically; a retry only covers a concurrent replace.
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError:
            time.sleep(0.05)
    return None


def _submit(request: dict) -> None:
    _require()
    global _sequence
    _sequence += 1
    directory = Path(os.environ["AGENT_SWARM_REQUESTS"])
    # Sortable names keep requests in submission order.
    name = f"{time.time_ns():020d}-{_sequence:06d}-{uuid.uuid4().hex[:8]}"
    temporary = directory / f".{name}.tmp"
    temporary.write_text(json.dumps(request), encoding="utf-8")
    os.replace(temporary, directory / f"{name}.json")


def spawn(child_id: str, *, command: list[str] | None = None, inputs: dict | None = None,
          agent: str | None = None, connection: str | None = None, tier: str | None = None,
          server: str | None = None, cpu: float | None = None, memory_mb: int | None = None,
          timeout_seconds: int | None = None, datasets: list[str] | None = None,
          files: str | os.PathLike | None = None, gpus: int | None = None) -> str:
    """Ask the platform to start one subagent; returns child_id.

    child_id   lowercase letters, digits, '-' and '_' (max 64), unique in this run.
    command    argv run in the child's agent directory (default: the manifest command).
    agent      another agents/<dir> from this snapshot to run in (default: this agent).
    connection an approved account alias, "auto" (router, approved accounts), "none",
               or None to reuse this run's account.
    server     an approved server label such as "doppio", "local" for the VPS,
               or None to use this run's server.
    files      a directory under /work or /tmp copied to the child's
               /run/parent-files (64 MiB, 2000 files at most).
    datasets   a subset of this run's datasets (default: all of them).
    gpus       GPUs for the child (1-8). It runs on an approved server with enough
               GPUs (the first one unless server is given), from the GPU image,
               with no network access and no provider account.
    cpu, memory_mb, timeout_seconds may only lower the manifest's values.

    Spawning is asynchronous; rejected requests show up in status()["errors"].
    """
    request: dict = {"action": "spawn", "id": child_id, "inputs": inputs or {}}
    for key, value in (("command", command), ("agent", agent), ("connection", connection), ("tier", tier),
                       ("server", server), ("cpu", cpu), ("memory_mb", memory_mb),
                       ("timeout_seconds", timeout_seconds), ("datasets", datasets), ("gpus", gpus)):
        if value is not None:
            request[key] = value
    if files is not None:
        request["files"] = str(Path(files).resolve())
    _submit(request)
    return child_id


def gpu_run(child_id: str, folder: str | os.PathLike, command: str | list[str], *, gpus: int = 1,
            inputs: dict | None = None, server: str | None = None, timeout_seconds: int | None = None,
            memory_mb: int | None = None, datasets: list[str] | None = None,
            wait_timeout: float | None = None) -> dict:
    """Run code this agent wrote on an approved GPU server, wait, and return the outcome.

    folder   a directory under /work or /tmp holding the script and its small
             inputs (64 MiB at most); the GPU job sees it at /run/parent-files.
    command  a script name in folder (run as ``python /run/parent-files/<name>``)
             or a full argv list.
    Returns the final child status plus "output_dir" (a Path to the job's
    /output files, read-only here), "result" (result.json with artifact hashes)
    and "log_tail". A rejected request has status "rejected" and an "error".
    The job starts in /work, has no network access and no provider credential.
    """
    argv = ["python", f"/run/parent-files/{command}"] if isinstance(command, str) else list(command)
    spawn(child_id, command=argv, inputs=inputs, server=server, gpus=gpus, files=folder,
          timeout_seconds=timeout_seconds, memory_mb=memory_mb, datasets=datasets)
    final: dict = {"id": child_id, "status": "unknown"}
    for item in wait([child_id], timeout=wait_timeout):
        final = dict(item)
    final["output_dir"] = output_dir(child_id)
    final["result"] = result(child_id) if final.get("status") != "rejected" else None
    return final


def gpu_servers() -> list[dict]:
    """Approved servers with GPUs, as {"id", "label", "gpus", "mode"} (empty if none were approved)."""
    return [item for item in status().get("servers", []) if item.get("gpus")]


def cancel(child_id: str = "*") -> None:
    """Kill one running or queued subagent, or every unfinished one with "*"."""
    _submit({"action": "cancel", "id": child_id})


def status() -> dict:
    """Swarm summary: counts, child IDs, approved servers/accounts, limits and errors."""
    _require()
    path = Path(os.environ["AGENT_SWARM_STATUS"])
    # The platform writes the first status within a second or two of start.
    deadline = time.monotonic() + 15
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.25)
    return _read_json(path) or {"children": [], "counts": {}, "errors": [], "servers": [], "connections": []}


def child(child_id: str) -> dict | None:
    """One subagent's live status: status, progress, server, account, usage, log_tail, delivered."""
    _require()
    return _read_json(Path(os.environ["AGENT_SWARM_CHILDREN"]) / f"{child_id}.json")


def children() -> list[dict]:
    _require()
    found = []
    for path in sorted(Path(os.environ["AGENT_SWARM_CHILDREN"]).glob("*.json")):
        item = _read_json(path)
        if item:
            found.append(item)
    return found


def errors() -> list[dict]:
    return status().get("errors", [])


def rejected(child_id: str) -> str | None:
    """The platform's reason for rejecting a spawn request, if it was rejected."""
    for item in reversed(errors()):
        if item.get("id") == child_id:
            return item.get("error")
    return None


def wait(child_ids: Iterable[str] | None = None, timeout: float | None = None,
         poll: float = 2.0) -> Iterator[dict]:
    """Yield each subagent's final status once its results are in /swarm/results.

    Yields in completion order. A rejected request yields
    {"id": ..., "status": "rejected", "error": ...}. Raises TimeoutError if
    timeout seconds pass first (unfinished subagents keep running).
    """
    _require()
    pending = set(child_ids) if child_ids is not None else None
    seen: set[str] = set()
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if pending is None:
            pending = set(status().get("children", []))
            if not pending:
                return
        for child_id in sorted(pending - seen):
            item = child(child_id)
            if item and item.get("delivered"):
                seen.add(child_id)
                yield item
                continue
            if item is None:
                reason = rejected(child_id)
                if reason:
                    seen.add(child_id)
                    yield {"id": child_id, "status": "rejected", "error": reason, "delivered": False}
        if seen >= pending:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{len(pending - seen)} subagents still running")
        time.sleep(poll)


def result_dir(child_id: str) -> Path:
    _require()
    return Path(os.environ["AGENT_SWARM_RESULTS"]) / child_id


def result(child_id: str) -> dict | None:
    """The child's result.json: status, error, server, account, usage and an
    artifact list with sizes and SHA-256 hashes (useful for loading into a database)."""
    return _read_json(result_dir(child_id) / "result.json")


def output_dir(child_id: str) -> Path:
    """The child's /output files (read-only; copy what you want to keep into AGENT_OUTPUT_DIR)."""
    return result_dir(child_id) / "output"


# --- Inside a subagent ---------------------------------------------------------------------------------

def child_id() -> str | None:
    return os.environ.get("AGENT_SWARM_CHILD_ID") or None


def parent_run_id() -> str | None:
    return os.environ.get("AGENT_PARENT_RUN_ID") or None


def parent_files() -> Path | None:
    value = os.environ.get("AGENT_PARENT_FILES_DIR")
    return Path(value) if value else None
