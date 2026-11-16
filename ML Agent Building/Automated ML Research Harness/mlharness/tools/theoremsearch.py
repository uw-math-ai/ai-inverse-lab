"""TheoremSearch adapter (https://api.theoremsearch.com).

Implemented against the service's published OpenAPI schema (``/openapi.json``, FastAPI, checked
2026-11-09; no authentication required at that time):

* ``POST /search``  body ``SearchRequest {query, n_results, sources, types, year_range, ...}`` ->
  ``SearchResponse {theorems: [TheoremResult{theorem_id, name, body, slogan, theorem_type, link,
  paper: PaperResult{paper_id, source, title, authors, link, year, journal_ref, citations}, similarity}]}``
* ``GET /graph/embedding?query=&n_results=&mode=full`` -> ``{results: [{statement_id, paper_id, name, kind,
  body, slogan, title, authors, url, external_id, year, journal_ref, citation_count, similarity}]}``
  (used as a fallback when ``/search`` fails for a non-auth reason)
* ``GET /ping`` health check.

The client never invents fields: everything it returns comes from those response models, and every
result carries a stable ``ref_id`` (``ts:thm:<theorem_id>`` / ``ts:stmt:<statement_id>``), title,
authors, year and link so citations are verifiable. Failures are returned (never raised) as a
``RetrievalResult`` with ``error_kind`` in {timeout, network, rate_limited, auth, server, malformed,
disabled, budget}. Successful results are cached on disk (``.cache/theoremsearch``).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import HARNESS_ROOT

DEFAULT_BASE = "https://api.theoremsearch.com"


@dataclass
class TheoremRef:
    ref_id: str
    name: str | None = None
    kind: str | None = None
    body: str | None = None
    slogan: str | None = None
    paper_title: str | None = None
    authors: list = field(default_factory=list)
    year: int | None = None
    link: str | None = None
    paper_id: str | None = None
    external_id: str | None = None
    journal_ref: str | None = None
    source: str | None = None
    citations: int | None = None
    similarity: float | None = None
    endpoint: str | None = None

    def citation(self) -> str:
        au = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "") if self.authors else "unknown authors"
        bits = [f"{au} ({self.year or 'n.d.'})", f"\"{(self.paper_title or 'untitled').strip()}\""]
        if self.name:
            bits.append(self.name)
        if self.link:
            bits.append(self.link)
        return ". ".join(bits)

    def to_dict(self, body_chars: int = 1200) -> dict:
        d = asdict(self)
        for k in ("body", "slogan"):
            if d.get(k) and len(d[k]) > body_chars:
                d[k] = d[k][:body_chars] + " ..."
        return d


@dataclass
class RetrievalResult:
    ok: bool
    query: str
    refs: list = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None
    endpoint: str | None = None
    latency: float = 0.0
    cached: bool = False
    attempts: int = 0
    fallback_used: bool = False

    def to_dict(self, body_chars: int = 600) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "refs"}
        d["refs"] = [r.to_dict(body_chars) if isinstance(r, TheoremRef) else r for r in self.refs]
        return d


class HTTPResponse:
    def __init__(self, status: int, text: str, headers: dict | None = None):
        self.status, self.text, self.headers = status, text, headers or {}


class TransportError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


def default_transport(method: str, url: str, *, params=None, json_body=None, timeout: float = 30.0) -> HTTPResponse:
    try:
        import requests
    except ImportError:          # stdlib fallback
        import urllib.error
        import urllib.parse
        import urllib.request
        if params:
            url = url + "?" + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(json_body).encode() if json_body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": "mlharness/2"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return HTTPResponse(r.status, r.read().decode("utf-8", "replace"), dict(r.headers))
        except urllib.error.HTTPError as e:
            return HTTPResponse(e.code, e.read().decode("utf-8", "replace"), dict(e.headers or {}))
        except TimeoutError as e:
            raise TransportError("timeout", str(e)) from e
        except Exception as e:  # URLError, connection reset
            raise TransportError("network", str(e)) from e
    try:
        r = requests.request(method, url, params=params, json=json_body, timeout=timeout,
                             headers={"User-Agent": "mlharness/2"})
        return HTTPResponse(r.status_code, r.text, dict(r.headers))
    except requests.exceptions.Timeout as e:
        raise TransportError("timeout", str(e)) from e
    except requests.exceptions.RequestException as e:
        raise TransportError("network", str(e)) from e


def _classify(status: int) -> str | None:
    if 200 <= status < 300:
        return None
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server"
    return "client"


class TheoremSearchClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: float = 45.0, retries: int = 1,
                 max_retry_wait: float = 20.0, transport=None, cache_dir: str | Path | None = "default",
                 cache_ttl_days: float = 14.0, min_interval: float = 0.0, enabled: bool = True):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.max_retry_wait = float(max_retry_wait)
        self.transport = transport or default_transport
        self.cache_dir = (HARNESS_ROOT / ".cache" / "theoremsearch") if cache_dir == "default" else \
            (Path(cache_dir) if cache_dir else None)
        self.cache_ttl = float(cache_ttl_days) * 86400
        self.min_interval = float(min_interval)
        self.enabled = bool(enabled)
        self._last = 0.0
        self._lock = threading.Lock()

    # -- cache -----------------------------------------------------------------------------
    def _ckey(self, query: str, n: int) -> str:
        return hashlib.sha1(json.dumps([self.base, query.strip().lower(), int(n)]).encode()).hexdigest()[:20]

    def _cache_get(self, key):
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        try:
            if p.exists() and time.time() - p.stat().st_mtime < self.cache_ttl:
                d = json.loads(p.read_text(encoding="utf-8"))
                return d
        except Exception:
            return None
        return None

    def _cache_put(self, key, payload):
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    # -- HTTP --------------------------------------------------------------------------------
    def _request(self, method, path, **kw) -> tuple[HTTPResponse | None, str | None, str | None, int]:
        attempts = 0
        last_kind, last_msg = None, None
        for attempt in range(self.retries + 1):
            attempts += 1
            with self._lock:
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = self.transport(method, self.base + path, timeout=self.timeout, **kw)
            except TransportError as e:
                last_kind, last_msg = e.kind, str(e)[:300]
                if e.kind == "timeout" or attempt >= self.retries:
                    break
                time.sleep(min(2.0 * (attempt + 1), self.max_retry_wait))
                continue
            kind = _classify(r.status)
            if kind is None:
                return r, None, None, attempts
            last_kind, last_msg = kind, f"HTTP {r.status}: {r.text[:200]}"
            if kind == "rate_limited" and attempt < self.retries:
                try:
                    ra = float(r.headers.get("Retry-After") or r.headers.get("retry-after") or 2.0)
                except ValueError:
                    ra = 2.0
                if ra > self.max_retry_wait:
                    break
                time.sleep(ra)
                continue
            if kind == "server" and attempt < self.retries:
                time.sleep(min(2.0 * (attempt + 1), self.max_retry_wait))
                continue
            break
        return None, last_kind, last_msg, attempts

    # -- parsing -----------------------------------------------------------------------------
    @staticmethod
    def _parse_search(payload) -> list[TheoremRef]:
        out = []
        for t in payload.get("theorems", []):
            paper = t.get("paper") or {}
            out.append(TheoremRef(
                ref_id=f"ts:thm:{t['theorem_id']}", name=t.get("name"), kind=t.get("theorem_type"),
                body=t.get("body"), slogan=t.get("slogan"), paper_title=paper.get("title"),
                authors=list(paper.get("authors") or []), year=paper.get("year"),
                link=t.get("link") or paper.get("link"), paper_id=paper.get("paper_id"),
                external_id=paper.get("paper_id"), journal_ref=paper.get("journal_ref"), source=paper.get("source"),
                citations=paper.get("citations"), similarity=t.get("similarity"), endpoint="/search"))
        return out

    @staticmethod
    def _parse_embedding(payload) -> list[TheoremRef]:
        out = []
        for t in payload.get("results", []):
            out.append(TheoremRef(
                ref_id=f"ts:stmt:{t['statement_id']}", name=t.get("name"), kind=t.get("kind"), body=t.get("body"),
                slogan=t.get("slogan"), paper_title=t.get("title"), authors=list(t.get("authors") or []),
                year=t.get("year"), link=t.get("url"), paper_id=t.get("paper_id"), external_id=t.get("external_id"),
                journal_ref=t.get("journal_ref"), source=t.get("source"), citations=t.get("citation_count"),
                similarity=t.get("similarity"), endpoint="/graph/embedding"))
        return out

    # -- public ------------------------------------------------------------------------------
    def ping(self) -> RetrievalResult:
        t0 = time.time()
        r, kind, msg, att = self._request("GET", "/ping")
        return RetrievalResult(ok=r is not None, query="/ping", error=msg, error_kind=kind, endpoint="/ping",
                               latency=time.time() - t0, attempts=att)

    def search(self, query: str, n_results: int = 5, **filters) -> RetrievalResult:
        query = (query or "").strip()
        if not self.enabled:
            return RetrievalResult(ok=False, query=query, error="literature search disabled by configuration",
                                   error_kind="disabled")
        if not query:
            return RetrievalResult(ok=False, query=query, error="empty query", error_kind="client")
        key = self._ckey(query, n_results)
        hit = self._cache_get(key)
        if hit is not None:
            refs = [TheoremRef(**r) for r in hit["refs"]]
            return RetrievalResult(ok=True, query=query, refs=refs, endpoint=hit.get("endpoint"), cached=True)
        t0 = time.time()
        body = {"query": query, "n_results": int(n_results)}
        body.update({k: v for k, v in filters.items() if k in ("sources", "types", "year_range", "authors", "tags")})
        r, kind, msg, att = self._request("POST", "/search", json_body=body)
        endpoint, fallback = "/search", False
        refs = None
        if r is not None:
            try:
                refs = self._parse_search(json.loads(r.text))
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                kind, msg, r = "malformed", f"unexpected /search response: {type(e).__name__}: {e}"[:300], None
        if refs is None and kind != "auth":
            params = {"query": query, "n_results": int(n_results), "mode": "full"}
            r2, kind2, msg2, att2 = self._request("GET", "/graph/embedding", params=params)
            att += att2
            if r2 is not None:
                try:
                    refs = self._parse_embedding(json.loads(r2.text))
                    endpoint, fallback = "/graph/embedding", True
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    kind, msg = "malformed", f"unexpected /graph/embedding response: {type(e).__name__}: {e}"[:300]
            else:
                msg = f"/search: {msg}; /graph/embedding: {msg2}"
                kind = kind if kind in ("rate_limited", "timeout") else (kind2 or kind)
        if refs is None:
            return RetrievalResult(ok=False, query=query, error=msg, error_kind=kind or "unknown",
                                   endpoint=endpoint, latency=time.time() - t0, attempts=att)
        self._cache_put(key, {"refs": [asdict(x) for x in refs], "endpoint": endpoint})
        return RetrievalResult(ok=True, query=query, refs=refs, endpoint=endpoint, latency=time.time() - t0,
                               attempts=att, fallback_used=fallback)
