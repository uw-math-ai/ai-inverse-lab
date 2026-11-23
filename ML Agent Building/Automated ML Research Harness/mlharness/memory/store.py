"""Persistent, queryable research memory shared by all researchers and the critic, across runs.

Distinct from the artifact ledger:
* artifact store  = canonical, immutable execution history of ONE run (code, data, model lineage);
* research memory = searchable scientific knowledge accumulated over MANY runs, every record
  indexed back to artifacts (run dir + artifact id) and external citations.

Records are append-only. A revision, a new observation or a contradicting result is a *new* record
that points at earlier ones (``parents`` / ``supersedes``); nothing is overwritten, so conflicting
evidence coexists with its provenance. Status vocabulary for scientific claims:
``conjectured | literature-supported | experimentally-supported | contradicted | inconclusive |
formally-verified | abandoned | rejected`` (``formally-verified`` is reserved for claims checked by
a formal verifier; the harness never assigns it on its own).

Retrieval is a BM25 keyword index with structured filters and a context boost (same simulator /
problem class / noise regime). The backend is swappable (``MemoryBackend``); the default is one JSONL
file, so no database is required. Test-set numbers are stored but redacted from agent-facing digests,
so repeated development runs cannot turn the test split into validation data.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from .. import HARNESS_ROOT

TYPES = ("hypothesis", "revision", "critique", "disagreement", "literature", "citation", "experiment_plan",
         "observation", "failure", "abandoned", "finding", "decision", "test_access")
CLAIM_STATUSES = ("conjectured", "literature-supported", "experimentally-supported", "contradicted",
                  "inconclusive", "formally-verified", "abandoned", "rejected")
REDACT_KEYS = ("test", "test_metrics", "test_bootstrap", "test_per_target_rel_l2", "member_test", "shift")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to and or in on for with by is are be as at from that this it its into via using use "
            "we our can may not no than then".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall((text or "").lower()):
        if len(t) < 2 or t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class MemoryBackend(ABC):
    @abstractmethod
    def load(self) -> list[dict]: ...

    @abstractmethod
    def append(self, rec: dict) -> None: ...

    def describe(self) -> str:
        return type(self).__name__


class InMemoryBackend(MemoryBackend):
    def __init__(self):
        self.items: list[dict] = []

    def load(self):
        return list(self.items)

    def append(self, rec):
        self.items.append(rec)


class JsonlMemoryBackend(MemoryBackend):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self):
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue          # a torn final line from an interrupted run is skipped
        return out

    def append(self, rec):
        line = json.dumps(rec, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def describe(self):
        return f"jsonl:{self.path}"


def default_memory_path() -> Path:
    return HARNESS_ROOT / ".research_memory" / "memory.jsonl"


class ResearchMemory:
    def __init__(self, path: str | Path | None = None, backend: MemoryBackend | None = None, enabled: bool = True,
                 read: bool = True, write: bool = True):
        self.enabled = bool(enabled)
        self.read_enabled = self.enabled and bool(read)
        self.write_enabled = self.enabled and bool(write)
        self.backend = backend or (JsonlMemoryBackend(path or default_memory_path()) if self.enabled else InMemoryBackend())
        self._lock = threading.RLock()
        self.records: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self._tf: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._len: dict[str, int] = {}
        self.citations: dict[str, dict] = {}
        self.loaded_from_disk = 0
        self.retrievals: list[dict] = []          # what agents retrieved in this process (for the report)
        for rec in self.backend.load():
            self._index(rec)
        self.loaded_from_disk = len(self.records)

    # -- indexing ---------------------------------------------------------------------------
    def _doc(self, rec) -> str:
        return " ".join([rec.get("title") or "", rec.get("text") or "", " ".join(rec.get("tags") or []),
                         rec.get("type") or "", rec.get("status") or ""])

    def _index(self, rec):
        self.records.append(rec)
        self.by_id[rec["id"]] = rec
        toks = tokenize(self._doc(rec))
        tf = Counter(toks)
        self._tf[rec["id"]] = tf
        self._len[rec["id"]] = len(toks)
        for t in tf:
            self._df[t] += 1
        if rec.get("type") == "citation":
            self.citations[rec["data"]["ref"]["ref_id"]] = rec

    # -- writing ----------------------------------------------------------------------------
    def add(self, type: str, *, title: str, text: str = "", author: str = "", run_id: str | None = None,
            status: str = "recorded", context: dict | None = None, data: dict | None = None, parents=(),
            citations=(), artifact_refs=(), tags=(), supersedes: str | None = None, redact: tuple = ()) -> str | None:
        if type not in TYPES:
            raise ValueError(f"unknown memory record type {type!r}")
        rec = {"id": f"m-{time.strftime('%y%m%d')}-{uuid.uuid4().hex[:8]}", "type": type, "created": time.time(),
               "run_id": run_id, "author": author, "title": title[:300], "text": (text or "")[:4000],
               "status": status, "context": dict(context or {}), "data": json.loads(json.dumps(data or {}, default=str)),
               "parents": [p for p in parents if p], "citations": list(citations), "artifact_refs": list(artifact_refs),
               "tags": list(tags), "supersedes": supersedes, "redact": list(redact)}
        with self._lock:
            self._index(rec)
            if self.write_enabled:
                self.backend.append(rec)
        return rec["id"]

    def add_citation(self, ref: dict, assessment: dict, query: str = "", run_id: str | None = None):
        rid = ref.get("ref_id")
        if not rid:
            return None
        with self._lock:
            if rid in self.citations:
                return self.citations[rid]["id"]
        return self.add("citation", title=f"{ref.get('name') or ''} — {ref.get('paper_title') or ''}"[:300],
                        text=(ref.get("slogan") or "") + " " + (ref.get("body") or "")[:1500], author="literature",
                        run_id=run_id, status="retrieved", data={"ref": ref, "assessment": assessment, "query": query},
                        citations=[rid], tags=["theoremsearch", ref.get("kind") or ""])

    # -- reading ----------------------------------------------------------------------------
    def has_citation(self, ref_id: str) -> bool:
        return ref_id in self.citations

    def citation(self, ref_id: str) -> dict | None:
        rec = self.citations.get(ref_id)
        return {**rec["data"]["ref"], "assessment": rec["data"].get("assessment")} if rec else None

    def get(self, rec_id: str) -> dict | None:
        return self.by_id.get(rec_id)

    def _bm25(self, q_toks, rid, n_docs, avgdl, k1=1.4, b=0.75):
        tf = self._tf.get(rid) or {}
        dl = self._len.get(rid, 0) or 1
        s = 0.0
        for t in q_toks:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n_docs - self._df[t] + 0.5) / (self._df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        return s

    def search(self, query: str, k: int = 8, types=None, statuses=None, context: dict | None = None,
               exclude_run: str | None = None, only_run: str | None = None, min_score: float = 0.0) -> list[dict]:
        if not self.read_enabled:
            return []
        q = tokenize(query)
        with self._lock:
            recs = list(self.records)
            n = max(1, len(recs))
            avgdl = (sum(self._len.values()) / n) if self._len else 1.0
            scored = []
            for r in recs:
                if types and r["type"] not in types:
                    continue
                if statuses and r.get("status") not in statuses:
                    continue
                if exclude_run and r.get("run_id") == exclude_run:
                    continue
                if only_run and r.get("run_id") != only_run:
                    continue
                s = self._bm25(q, r["id"], n, avgdl) if q else 0.0
                if context:
                    rc = r.get("context") or {}
                    if context.get("simulator_hash") and rc.get("simulator_hash") == context["simulator_hash"]:
                        s += 2.0
                    elif context.get("problem_class") and rc.get("problem_class") == context["problem_class"]:
                        s += 1.0
                    if context.get("noise_regime") and rc.get("noise_regime") == context["noise_regime"]:
                        s += 0.3
                if s > min_score:
                    scored.append((s, r["created"], r))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [dict(r, _score=round(s, 3)) for s, _, r in scored[:k]]

    def lineage(self, rec_id: str) -> list[dict]:
        out, seen, queue = [], set(), [rec_id]
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in self.by_id:
                continue
            seen.add(cur)
            r = self.by_id[cur]
            out.append(r)
            queue += list(r.get("parents") or []) + ([r["supersedes"]] if r.get("supersedes") else [])
        return out

    def evidence_for(self, method_key: str, context: dict | None = None) -> list[dict]:
        """All observations about a method (by label/hash) across runs, contradictions included."""
        out = []
        for r in self.records:
            if r["type"] in ("observation", "failure", "finding") and method_key in (
                    (r.get("data") or {}).get("method_label"), (r.get("data") or {}).get("method_hash")):
                if context and context.get("simulator_hash") and (r.get("context") or {}).get("simulator_hash") not in (
                        None, context["simulator_hash"]):
                    continue
                out.append(r)
        return out

    def test_access_count(self, data_key: str) -> int:
        return sum(1 for r in self.records if r["type"] == "test_access" and (r.get("data") or {}).get("data_key") == data_key)

    @staticmethod
    def redacted(rec: dict) -> dict:
        d = dict(rec.get("data") or {})
        for k in list(d):
            if k in REDACT_KEYS or k in (rec.get("redact") or []):
                d[k] = "[redacted: held-out test result]"
        return {**rec, "data": d}

    def digest(self, query: str, context: dict | None = None, k: int = 8, exclude_run: str | None = None,
               agent: str = "", types=("finding", "observation", "failure", "abandoned", "hypothesis", "critique",
                                       "disagreement", "citation")) -> str:
        """Agent-facing text of the most relevant prior knowledge (test numbers redacted)."""
        hits = self.search(query, k=k, types=types, context=context, exclude_run=exclude_run)
        self.retrievals.append({"agent": agent, "query": query[:200], "ids": [h["id"] for h in hits],
                                "time": time.time()})
        if not hits:
            return "(no relevant prior research records)"
        lines = []
        for h in hits:
            r = self.redacted(h)
            d = r.get("data") or {}
            val = d.get("val") or d.get("val_metrics")
            bits = [f"[{r['id']}] {r['type']} ({r.get('status')}) from run {r.get('run_id') or '?'}: {r['title']}"]
            if val:
                bits.append(f"val={json.dumps(val, default=str)[:160]}")
            if d.get("reason"):
                bits.append(f"reason: {str(d['reason'])[:200]}")
            if d.get("explanation"):
                bits.append(f"why: {str(d['explanation'])[:200]}")
            if r["type"] == "citation":
                a = d.get("assessment") or {}
                bits.append(f"ref {d.get('ref', {}).get('ref_id')} screen={a.get('applies')}")
            lines.append(" | ".join(bits))
        return "\n".join(lines)

    def stats(self) -> dict:
        c = Counter(r["type"] for r in self.records)
        runs = {r.get("run_id") for r in self.records if r.get("run_id")}
        return {"records": len(self.records), "loaded_from_disk": self.loaded_from_disk, "by_type": dict(c),
                "runs": len(runs), "citations": len(self.citations), "backend": self.backend.describe(),
                "read": self.read_enabled, "write": self.write_enabled}
