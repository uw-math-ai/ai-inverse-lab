"""Crash-safe output: atomic files, per-unit parts, shard consolidation, done-log, and the loader.

Layout of output_dir/:
    config.resolved.yaml          resolved config (all defaults)
    env{.tag}.json                versions, GPU, git commit, start/end time
    run{.tag}.log                 log file
    done{.tag}.jsonl              one line per finished (graph, sweep, replicate) with its metrics
    graphs{.tag}.jsonl            one line per generated graph
    graphs{.tag}.parquet / metrics{.tag}.parquet   tables (rebuilt from the jsonl files)
    tensors/shard_XXXX.npz        arrays for graphs [XXXX*shard_size, (XXXX+1)*shard_size)
    tensors/parts/*.npz           not-yet-consolidated per-unit arrays

{.tag} is empty for a single process and e.g. ".shard2of8" or ".cuda1" otherwise, so concurrent
jobs never write to the same file. Every file is written to a temp name and then renamed.
done lines are appended only after their arrays are on disk.

Array keys inside npz files:
    g{id}__W, __mu, __B, __q, __coords, __boundary                 graph level
    g{id}__s{s}__K, __kill, __dt, __r_exact, __exact_killed, __exact_censored
    g{id}__s{s}__r{r}__counts, __killed, __censored, __prefix_counts, __checkpoints,
                     __ci_lower, __ci_upper
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- naming
def suffix(tag: str) -> str:
    return f".{tag}" if tag else ""


def shard_tag(shard: tuple[int, int]) -> str:
    i, k = shard
    return "" if k == 1 else f"shard{i}of{k}"


def parts_dir(out: Path) -> Path:
    return Path(out) / "tensors" / "parts"


def part_path(out: Path, gid: int, s: int | None = None, r: int | None = None) -> Path:
    if s is None:
        name = f"g{gid}__graph.npz"
    elif r is None:
        name = f"g{gid}__s{s}__exact.npz"
    else:
        name = f"g{gid}__s{s}__r{r}.npz"
    return parts_dir(out) / name


def shard_path(out: Path, shard: int) -> Path:
    return Path(out) / "tensors" / f"shard_{shard:04d}.npz"


def _gid_of(name: str) -> int:
    return int(name[1:name.index("__")])


# ---------------------------------------------------------------- atomic writes
def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")


def _atomic(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    _atomic(path, lambda f: f.write(text.encode("utf-8")))


def atomic_save_npz(path: Path, arrays: dict) -> None:
    _atomic(path, lambda f: np.savez_compressed(f, **arrays))


def atomic_write_parquet(df, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, default=json_default) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def repair_jsonl_tail(path: Path) -> None:
    """If a crash left a final line without a newline, end it so new lines start cleanly."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
    if last != b"\n":
        with open(path, "ab") as f:
            f.write(b"\n")


def read_jsonl(path: Path) -> list[dict]:
    """All parseable lines; a truncated line from a crash is skipped."""
    out = []
    path = Path(path)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def clean_stale_temps(out: Path, older_than_s: float = 6 * 3600) -> int:
    """Remove temp files left by killed processes (only old ones, so live writers are safe)."""
    removed = 0
    now = time.time()
    for p in list(Path(out).rglob(".*.tmp-*")):
        try:
            if now - p.stat().st_mtime > older_than_s:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------- done-log and graph rows
def done_files(out: Path) -> list[Path]:
    return sorted(Path(out).glob("done*.jsonl"))


def read_done_records(out: Path) -> list[dict]:
    return [rec for p in done_files(out) for rec in read_jsonl(p)]


def read_done_keys(out: Path) -> set[tuple[int, int, int]]:
    return {tuple(rec["key"]) for rec in read_done_records(out) if "key" in rec}


def graph_files(out: Path) -> list[Path]:
    return sorted(Path(out).glob("graphs*.jsonl"))


def read_graph_rows(out: Path) -> list[dict]:
    return [row for p in graph_files(out) for row in read_jsonl(p)]


# ---------------------------------------------------------------- shards
def consolidate_shard(out: Path, shard: int, shard_size: int) -> int:
    """Merge the parts of one shard into tensors/shard_XXXX.npz atomically, then drop the parts.

    Idempotent: a crash between the rename and the deletes leaves parts that hold the same
    data as the shard, and they are merged again next time.
    """
    pdir = parts_dir(out)
    if not pdir.exists():
        return 0
    lo, hi = shard * shard_size, (shard + 1) * shard_size
    parts = sorted(p for p in pdir.glob("g*.npz") if lo <= _gid_of(p.name) < hi)
    if not parts:
        return 0
    arrays = {}
    target = shard_path(out, shard)
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            arrays.update({k: z[k] for k in z.files})
    for p in parts:
        with np.load(p, allow_pickle=False) as z:
            arrays.update({k: z[k] for k in z.files})
    atomic_save_npz(target, arrays)
    for p in parts:
        p.unlink(missing_ok=True)
    return len(parts)


# ---------------------------------------------------------------- environment
def environment_info(backend: str, devices: list[str]) -> dict:
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "hostname": socket.gethostname(),
            "pid": os.getpid(), "argv": sys.argv, "backend": backend, "devices": devices,
            "start_time": datetime.now(timezone.utc).isoformat(), "end_time": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch"] = None
    for mod in ("networkx", "pandas", "pyarrow", "yaml"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def write_env(out: Path, tag: str, info: dict) -> None:
    atomic_write_text(Path(out) / f"env{suffix(tag)}.json", json.dumps(info, indent=2, default=json_default))


# ---------------------------------------------------------------- tables
def graphs_frame(out: Path):
    import pandas as pd
    rows = read_graph_rows(out)
    if not rows:
        files = sorted(Path(out).glob("graphs*.parquet"))
        return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("graph_id", keep="last")
    return df.sort_values("graph_id").reset_index(drop=True)


def metrics_frame(out: Path, records: list[dict] | None = None):
    import pandas as pd
    records = read_done_records(out) if records is None else records
    rows = []
    for rec in records:
        g, s, r = rec["key"]
        rows.append({"graph_id": g, "sweep_index": s, "replicate": r,
                     "status": rec.get("status", "ok"), **rec.get("metrics", {})})
    if not rows:
        return pd.DataFrame(columns=["graph_id", "sweep_index", "replicate", "status"])
    df = pd.DataFrame(rows).drop_duplicates(["graph_id", "sweep_index", "replicate"], keep="last")
    return df.sort_values(["graph_id", "sweep_index", "replicate"]).reset_index(drop=True)


# ---------------------------------------------------------------- loader
class RunData:
    """A run directory: graphs and metrics as DataFrames, tensors loaded lazily per graph."""

    def __init__(self, path):
        import yaml
        self.path = Path(path)
        cfg_path = self.path / "config.resolved.yaml"
        self.config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.shard_size = int(self.config.get("shard_size", 500))
        self.graphs = graphs_frame(self.path)
        self.metrics = metrics_frame(self.path)

    def arrays(self, gid: int) -> dict:
        """Flat {key without the g{id}__ prefix: array} from the shard file and any parts."""
        prefix = f"g{gid}__"
        out = {}
        sp = shard_path(self.path, gid // self.shard_size)
        if sp.exists():
            with np.load(sp, allow_pickle=False) as z:
                for key in z.files:
                    if key.startswith(prefix):
                        out[key[len(prefix):]] = z[key]
        pdir = parts_dir(self.path)
        if pdir.exists():
            for p in pdir.glob(f"g{gid}__*.npz"):
                with np.load(p, allow_pickle=False) as z:
                    for key in z.files:
                        if key.startswith(prefix):
                            out[key[len(prefix):]] = z[key]
        return out

    def tensors(self, gid: int) -> dict:
        """Nested: {W, mu, B, q, ..., "sweeps": {s: {K, r_exact, ..., "replicates": {r: {...}}}}}."""
        nested: dict = {"sweeps": {}}
        for key, arr in self.arrays(gid).items():
            parts = key.split("__")
            if len(parts) == 1:
                nested[parts[0]] = arr
                continue
            s = int(parts[0][1:])
            sweep = nested["sweeps"].setdefault(s, {"replicates": {}})
            if len(parts) == 2:
                sweep[parts[1]] = arr
            else:
                sweep["replicates"].setdefault(int(parts[1][1:]), {})[parts[2]] = arr
        return nested

    def graph(self, gid: int):
        """Rebuild the Graph object for plotting or further analysis."""
        from .graphs import Graph
        a = self.tensors(gid)
        row = self.graphs[self.graphs.graph_id == gid]
        family = row.family.iloc[0] if len(row) else "unknown"
        params = json.loads(row.params_json.iloc[0]) if len(row) else {}
        return Graph(W=a["W"], mu=a["mu"], B=a["B"], q=a["q"], family=family, params=params,
                     coords=a.get("coords"), boundary=a.get("boundary"), graph_id=gid)

    def r_empirical(self, gid: int, s: int = 0, r: int = 0) -> np.ndarray:
        rep = self.tensors(gid)["sweeps"][s]["replicates"][r]
        N = int(self.metrics.query("graph_id == @gid and sweep_index == @s and replicate == @r").N.iloc[0])
        return rep["counts"] / N

    def __repr__(self) -> str:
        return f"RunData({self.path}, graphs={len(self.graphs)}, metric rows={len(self.metrics)})"


def load_run(path) -> RunData:
    return RunData(path)
