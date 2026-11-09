"""Layer 2 — Shared research artifacts.

Artifacts (not agent conversation histories) are the persistent research memory. Every artifact
records the idea that produced it and its parent artifacts, so the path to the final model can be
traced. Storage is behind the abstract ``ArtifactStore`` interface; V1 ships an append-only
filesystem store (JSONL ledger + workspace files) and an in-memory store.
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Artifact kinds
SPEC = "spec"
HYPOTHESIS = "hypothesis"
PLAN = "plan"
CODE = "code"
DATASET = "dataset"
RESULT = "result"
CRITIQUE = "critique"
FAILURE = "failure"
RESOURCE = "resource"
DECISION = "decision"
CHECKPOINT = "checkpoint"
AGENT = "agent"
REPORT = "report"

KINDS = [SPEC, HYPOTHESIS, PLAN, CODE, DATASET, RESULT, CRITIQUE, FAILURE, RESOURCE, DECISION,
         CHECKPOINT, AGENT, REPORT]


@dataclass
class Artifact:
    id: str
    kind: str
    author: str
    summary: str = ""
    data: dict = field(default_factory=dict)
    idea_id: str | None = None
    parents: list = field(default_factory=list)
    stage: str | None = None
    files: list = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ArtifactStore(ABC):
    """Storage-agnostic interface used by every agent."""

    def __init__(self):
        self._lock = threading.RLock()
        self._items: dict[str, Artifact] = {}
        self._order: list[str] = []
        self._counter = 0

    # -- writing --------------------------------------------------------------------------
    def put(self, kind: str, data: dict | None = None, *, author: str, summary: str = "",
            idea_id: str | None = None, parents: Iterable[str] = (), stage: str | None = None,
            files: Iterable[str] = ()) -> Artifact:
        if kind not in KINDS:
            raise ValueError(f"unknown artifact kind {kind!r}")
        with self._lock:
            self._counter += 1
            art = Artifact(id=f"{kind[:4]}-{self._counter:05d}", kind=kind, author=author,
                           summary=summary, data=_jsonable(data or {}), idea_id=idea_id,
                           parents=[p for p in parents if p], stage=stage, files=list(files))
            self._items[art.id] = art
            self._order.append(art.id)
            self._persist(art)
            return art

    @abstractmethod
    def _persist(self, art: Artifact) -> None: ...

    @abstractmethod
    def workspace(self, *parts: str) -> Path:
        """A directory where large files (code, predictions, checkpoints) can be written."""

    # -- reading --------------------------------------------------------------------------
    def get(self, art_id: str) -> Artifact:
        return self._items[art_id]

    def all(self) -> list[Artifact]:
        with self._lock:
            return [self._items[i] for i in self._order]

    def query(self, kind: str | None = None, idea_id: str | None = None, stage: str | None = None,
              where: Callable[[Artifact], bool] | None = None) -> list[Artifact]:
        out = []
        for a in self.all():
            if kind and a.kind != kind:
                continue
            if idea_id and a.idea_id != idea_id:
                continue
            if stage and a.stage != stage:
                continue
            if where and not where(a):
                continue
            out.append(a)
        return out

    def latest(self, kind: str, **kw) -> Artifact | None:
        items = self.query(kind, **kw)
        return items[-1] if items else None

    def lineage(self, art_id: str) -> list[Artifact]:
        """All ancestors of an artifact (breadth first, the artifact itself first)."""
        seen, out, queue = set(), [], [art_id]
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in self._items:
                continue
            seen.add(cur)
            a = self._items[cur]
            out.append(a)
            queue.extend(a.parents)
        return out

    def idea_history(self, idea_id: str) -> list[Artifact]:
        return self.query(idea_id=idea_id)


class MemoryStore(ArtifactStore):
    def __init__(self, scratch: Path | None = None):
        super().__init__()
        import tempfile
        self._root = Path(scratch or tempfile.mkdtemp(prefix="mlh_"))

    def _persist(self, art):
        pass

    def workspace(self, *parts):
        p = self._root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


class FileSystemStore(ArtifactStore):
    """Append-only JSONL ledger at ``<run>/artifacts.jsonl`` plus ``<run>/workspace/...``."""

    def __init__(self, run_dir: Path, resume: bool = False):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.run_dir / "artifacts.jsonl"
        if self.ledger.exists():
            if resume:
                self._load()
            else:
                raise FileExistsError(f"{self.run_dir} already contains a run; pass resume=True "
                                      f"(--resume) or choose another --run-dir")

    def _load(self):
        with open(self.ledger, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                art = Artifact(**json.loads(line))
                self._items[art.id] = art
                self._order.append(art.id)
                self._counter = max(self._counter, int(art.id.split("-")[-1]))

    def _persist(self, art):
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(art.to_dict()) + "\n")

    def workspace(self, *parts):
        p = self.run_dir.joinpath("workspace", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def _jsonable(obj):
    """Round-trip through JSON so stored data never aliases live mutable objects."""
    return json.loads(json.dumps(obj, default=_default))


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)
