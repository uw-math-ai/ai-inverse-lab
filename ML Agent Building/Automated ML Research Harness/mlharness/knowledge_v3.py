"""Versioned evidence access, durable method packages and evaluation bookkeeping.

This module never launches a research campaign or executes downloaded code. Package
content verification and the implementation gate are separate checks; execution still
belongs to the isolated runner. Exposure reservations are independent of research memory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "v3.1"
REDACTED = "[redacted: held-out evidence]"


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode("utf-8")).hexdigest()


def _heldout_key(key) -> bool:
    key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    return (key in {"test", "shift", "shifts", "final", "heldout", "held_out", "x_test", "y_test"}
            or key.startswith(("test_", "shift_", "x_test_", "y_test_", "x_shift", "y_shift", "heldout_", "held_out_"))
            or key.endswith(("_test", "_heldout", "_held_out")))


def safe_projection(value):
    """Conservative agent view, including nested records and narrative leakage.

Structured held-out payloads are replaced wholesale. Unstructured strings mentioning
held-out/test/shift evidence are withheld because arbitrary prose cannot be reliably
parsed into safe scientific facts. This projection is for agents, never final reports.
"""
    if isinstance(value, dict):
        split = str(value.get("split", value.get("split_role", value.get("evidence_role", "")))).lower()
        if (split in {"test", "heldout", "held_out", "final", "final_stress", "shift", "final_held_out"}
                or value.get("stage") in ("final", "report")):
            return REDACTED
        extra = {str(k) for k in value.get("redact", [])} if isinstance(value.get("redact"), list) else set()
        return {str(k): REDACTED if _heldout_key(k) or str(k) in extra else safe_projection(v)
                for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_projection(v) for v in value]
    if isinstance(value, str):
        if re.search(r"\b(?:test|shift(?:ed)?)(?=\b|_)|\b(?:held[ _-]?out|final[ _-]?stress)\b", value, re.I):
            return REDACTED
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return safe_projection(str(value))


class EvidenceRouter:
    """Resolve typed requests against explicit sources, without arbitrary file access."""

    def __init__(self, contract, *, store=None, artifacts=None, memory=None, literature=None,
                 theorem_client=None, diagnostic_fn=None, source_metadata=None,
                 empirical_provider=None, memory_mode="cumulative", package_store=None, package_gate=None):
        self.contract = copy.deepcopy(contract)
        self.store = store if store is not None else artifacts
        self.memory = memory
        self.literature = literature if literature is not None else theorem_client
        self.diagnostic_fn = diagnostic_fn
        self.source_metadata = copy.deepcopy(source_metadata or {})
        self.empirical_provider = empirical_provider
        self.package_store = package_store
        self.package_gate = package_gate
        if memory_mode not in {"frozen", "cumulative"}:
            raise ValueError("memory_mode must be frozen or cumulative")
        self.memory_mode = memory_mode
        records = copy.deepcopy(getattr(memory, "records", []))
        self._snapshot = safe_projection(records)
        self.snapshot_id = content_hash(records)
        self.requests = []
        self._cache = {}

    def snapshot_state(self):
        """Persist a safe frozen view so resume cannot silently advance its knowledge."""
        return {"schema": SCHEMA, "memory_mode": self.memory_mode, "snapshot_id": self.snapshot_id,
                "records": copy.deepcopy(self._snapshot), "records_hash": content_hash(self._snapshot)}

    def restore_snapshot(self, state):
        if not isinstance(state, dict) or state.get("memory_mode") != self.memory_mode:
            raise ValueError("memory snapshot mode mismatch")
        records = state.get("records")
        if not isinstance(records, list) or content_hash(records) != state.get("records_hash"):
            raise ValueError("memory snapshot content hash mismatch")
        if any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in records):
            raise ValueError("malformed memory snapshot records")
        self._snapshot = copy.deepcopy(records)
        self.snapshot_id = str(state["snapshot_id"])

    def resolve(self, request: dict, safe_context=None) -> dict:
        req = dict(request or {})
        kind = str(req.get("source_type", req.get("type", req.get("kind", "local"))))
        aliases = {"source": "local", "contract": "local", "local_source": "local", "run_artifact": "artifact",
                   "literature": "theorem", "theorem_retrieval": "theorem", "computation": "diagnostic"}
        kind = aliases.get(kind, kind)
        rid = req.get("id", req.get("request_id", f"request-{len(self.requests) + 1}"))
        result = {"schema": SCHEMA, "request_id": rid, "source_type": kind,
                  "status": "unsupported", "data": None, "uncertainty": [],
                  "agent_id": req.get("agent_id"), "study_id": req.get("study_id"),
                  "decision_id": req.get("decision_id"), "cache_hit": False}
        # IDs/owners remain separate even when background work is shared.
        cache_req = {k: v for k, v in req.items() if k not in
                     {"id", "request_id", "agent_id", "study_id", "decision_id", "parent_id"}}
        cache_key = content_hash([kind, cache_req, safe_projection(safe_context or {})])
        try:
            if kind in {"local", "theorem"} and cache_key in self._cache:
                cached = self._cache[cache_key]
                result.update(copy.deepcopy(cached), cache_hit=True)
            elif kind == "local":
                key = req.get("key")
                data = {"contract": self.contract, "source_metadata": self.source_metadata}
                if key:
                    # Only named metadata and contract keys are addressable; no filesystem reads.
                    if key in self.contract:
                        data = self.contract[key]
                    elif key in self.source_metadata:
                        data = self.source_metadata[key]
                    else:
                        result.update(status="unavailable", uncertainty=[f"unknown local metadata key {key}"])
                        data = None
                if data is not None:
                    result.update(status="ok", data=data)
            elif kind == "artifact":
                aid = req.get("artifact_id", req.get("key"))
                try:
                    art = self.store.get(aid) if self.store is not None and aid else None
                except KeyError:
                    art = None
                if art is None:
                    result.update(status="unavailable", uncertainty=["artifact not found"])
                else:
                    data = art.to_dict() if hasattr(art, "to_dict") else (vars(art) if hasattr(art, "data") else art)
                    if isinstance(data, dict) and data.get("stage") in {"final", "report"}:
                        result.update(status="denied", uncertainty=["final/report artifacts are not research inputs"])
                    else:
                        result.update(status="ok", data=data)
            elif kind == "memory":
                query = str(req.get("query", ""))
                k = min(50, max(1, int(req.get("k", 8))))
                if self.memory is None:
                    result.update(status="unavailable", uncertainty=["research memory unavailable"])
                else:
                    if self.memory_mode == "frozen":
                        from .memory import InMemoryBackend, ResearchMemory
                        backend = InMemoryBackend()
                        backend.items = copy.deepcopy(self._snapshot)
                        frozen = ResearchMemory(backend=backend, read=getattr(self.memory, "read_enabled", True), write=False)
                        found = frozen.search(query, k=max(k, len(self._snapshot)), context=safe_context or {})
                    else:
                        found = self.memory.search(query, k=max(k, 50), context=safe_context or {})
                    order = {"failure": 0, "finding": 1, "observation": 2, "citation": 3}
                    found.sort(key=lambda r: (order.get(r.get("type"), 4), -float(r.get("_score", 0))))
                    result.update(status="ok", data=found[:k], memory_mode=self.memory_mode,
                                  snapshot_id=self.snapshot_id if self.memory_mode == "frozen" else content_hash(getattr(self.memory, "records", [])),
                                  uncertainty=["retrieved records require applicability and implementation verification"])
            elif kind == "diagnostic":
                if self.diagnostic_fn is None:
                    result["uncertainty"] = ["diagnostic adapter unavailable"]
                else:
                    payload = self.diagnostic_fn(req)
                    status = payload.get("status", "ok") if isinstance(payload, dict) else "ok"
                    result.update(status="ok" if status in ("ok", "complete", "completed") else status, data=payload)
            elif kind == "method_package":
                if self.package_store is None or self.package_gate is None:
                    result.update(status="unavailable", uncertainty=["package store and implementation gate required"])
                else:
                    payload = self.package_store.load(req.get("package_id"), gate=self.package_gate)
                    result.update(status="ok", data=payload,
                                  uncertainty=["content and static gate verified; isolated runtime validation still required"])
            elif kind == "theorem":
                if self.literature is None:
                    result.update(status="unavailable", uncertainty=["theorem retrieval unavailable"])
                else:
                    query = str(req.get("query", ""))
                    # LiteratureTool accounts per-agent queries; raw clients have a simpler API.
                    if hasattr(self.literature, "remaining") and hasattr(self.literature, "known"):
                        answer = self.literature.search(query, str(req.get("agent_id", "v3-evidence")),
                                                       int(req.get("round", 0)), purpose=str(req.get("purpose", "")))
                    else:
                        answer = self.literature.search(query)
                    data = answer.to_dict() if hasattr(answer, "to_dict") else (vars(answer) if not isinstance(answer, dict) else answer)
                    if isinstance(data, dict):
                        data = copy.deepcopy(data)
                        data["assumptions_to_problem"] = req.get("assumptions_to_problem", {})
                        data["applicability_verified"] = False
                    result.update(status="ok" if data.get("ok", True) else "unavailable", data=data,
                                  uncertainty=["retrieval and keyword applicability are leads, not proof or verified transfer"])
            elif kind == "empirical_literature":
                if self.empirical_provider is None:
                    result["uncertainty"] = ["no empirical-literature provider configured"]
                else:
                    result.update(status="ok", data=self.empirical_provider(req))
            else:
                result["uncertainty"] = [f"unknown evidence source {kind}"]
        except Exception as exc:
            result.update(status="error", data=None, uncertainty=[f"{type(exc).__name__}: {exc}"])
        result["data"] = safe_projection(result["data"])
        result["source_hash"] = content_hash(result["data"])
        if result["status"] == "ok" and kind in {"local", "theorem"}:
            self._cache[cache_key] = {k: copy.deepcopy(result[k]) for k in
                                      ("status", "data", "uncertainty", "source_hash")}
        self.requests.append(copy.deepcopy(result))
        return result


class MethodPackageStore:
    """Content-addressed packages survive removal of their originating run directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, method, *, provenance=None, applicability=None, failure_conditions=None,
             validation=None, requirements=None) -> dict:
        method = copy.deepcopy(method)
        code = method.get("code")
        if code is None and method.get("code_path"):
            code = Path(method["code_path"]).read_text(encoding="utf-8")
        method.pop("code_path", None)
        method.pop("module_name", None)
        if code is not None:
            method["code"] = str(code)
        payload = {"schema": SCHEMA, "method": method, "provenance": provenance or {},
                   "applicability": applicability or {}, "failure_conditions": failure_conditions or [],
                   "validation": validation or {}, "requirements": requirements or [],
                   "interface": "fit_predict_v2"}
        pid = content_hash(payload)
        folder = self.root / pid
        folder.mkdir(exist_ok=True)
        dest = folder / "package.json"
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        if dest.exists() and dest.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing package content has been altered")
        dest.write_text(encoded, encoding="utf-8")
        return {"package_id": pid, "path": str(dest), "content_hash": pid,
                "implementation_verified": False}

    def load(self, package_id, *, gate=None) -> dict:
        if not re.fullmatch(r"[a-f0-9]{64}", str(package_id)):
            raise ValueError("invalid content-addressed package ID")
        payload = json.loads((self.root / package_id / "package.json").read_text(encoding="utf-8"))
        if content_hash(payload) != package_id:
            raise ValueError("method package hash mismatch")
        method = copy.deepcopy(payload["method"])
        verdict = None
        if gate is not None:
            verdict = gate.check(method, label=f"memory package {package_id[:12]}")
            if not verdict.get("approved"):
                raise ValueError(f"method package rejected by implementation gate: {verdict.get('reasons')}")
            method = verdict.get("method", method)
        return {**payload, "method": method, "package_id": package_id, "content_verified": True,
                "gate_approved": verdict is not None, "implementation_verified": False,
                "gate": verdict, "uncertainty": ["runtime fit/predict verification required before scientific reuse"]}


class ExposureRegistry:
    """SQLite transactions prevent concurrent strict access, including interrupted runs.

    Same-run/same-selection reservations are idempotent, but a completed token must
    never authorize re-evaluation: the scheduler must resume its persisted final result.
    A reservation without completion remains an uncertain exposure and blocks strict use.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS exposures (token TEXT PRIMARY KEY, dataset_hash TEXT NOT NULL, "
                       "split_hash TEXT NOT NULL, run_id TEXT NOT NULL, selection_hash TEXT NOT NULL, "
                       "status TEXT NOT NULL, created REAL NOT NULL, completed REAL, "
                       "UNIQUE(dataset_hash, split_hash, run_id, selection_hash))")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, dataset_hash, split_hash, run_id, selection_hash, strict=False) -> str:
        if any(v is None or not str(v).strip() for v in (dataset_hash, split_hash, run_id, selection_hash)):
            raise ValueError("exposure identity fields must be nonempty")
        values = tuple(str(x) for x in (dataset_hash, split_hash, run_id, selection_hash))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            same = db.execute("SELECT token FROM exposures WHERE dataset_hash=? AND split_hash=? AND run_id=? "
                              "AND selection_hash=?", values).fetchone()
            if same:
                return same["token"]
            # A split is held out globally even if another dataset/configuration calls it by a different name.
            prior = db.execute("SELECT token FROM exposures WHERE split_hash=? LIMIT 1", (values[1],)).fetchone()
            if strict and prior:
                raise PermissionError("held-out split already has a reserved or completed exposure")
            token = f"exposure-{uuid.uuid4().hex}"
            db.execute("INSERT INTO exposures VALUES (?,?,?,?,?,'reserved',?,NULL)", (token, *values, time.time()))
            return token

    def complete(self, token):
        with self._connect() as db:
            updated = db.execute("UPDATE exposures SET status='completed', completed=COALESCE(completed,?) WHERE token=?",
                                 (time.time(), str(token)))
            if updated.rowcount != 1:
                raise KeyError("unknown exposure token")

    def begin(self, token):
        """Commit attempted access before dispatch; interrupted access cannot be retried."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM exposures WHERE token=?", (str(token),)).fetchone()
            if row is None:
                raise KeyError("unknown exposure token")
            if row["status"] != "reserved":
                raise PermissionError("exposure was already attempted; resume persisted final evidence")
            db.execute("UPDATE exposures SET status='attempted' WHERE token=?", (str(token),))

    def get(self, token):
        with self._connect() as db:
            row = db.execute("SELECT * FROM exposures WHERE token=?", (str(token),)).fetchone()
        return dict(row) if row else None

    def entries(self, dataset_hash=None, split_hash=None):
        terms, args = [], []
        for key, value in (("dataset_hash", dataset_hash), ("split_hash", split_hash)):
            if value is not None:
                terms.append(f"{key}=?")
                args.append(str(value))
        with self._connect() as db:
            rows = db.execute("SELECT * FROM exposures" + (" WHERE " + " AND ".join(terms) if terms else "")
                              + " ORDER BY created,token", args).fetchall()
        return [dict(r) for r in rows]

    def count(self, dataset_hash=None, split_hash=None):
        return len(self.entries(dataset_hash, split_hash))


def validate_evaluation_protocol(protocol) -> dict:
    """Validate readiness records without launching experiments or inferring transfer."""
    p = copy.deepcopy(protocol or {})
    errors, warnings = [], []
    if not isinstance(p, dict):
        return {"valid": False, "status": "invalid", "errors": ["protocol must be a mapping"], "warnings": [], "normalized": {}}
    instances = p.get("instances", [])
    if not isinstance(instances, list):
        errors.append("instances must be a list")
        instances = []
    seen = set()
    for i, instance in enumerate(instances):
        if not isinstance(instance, dict):
            errors.append(f"instance {i} must be a mapping")
            continue
        key = instance.get("instance_id", instance.get("id"))
        if not isinstance(key, str) or not key.strip():
            errors.append(f"instance {i} requires a nonempty string instance_id")
        elif key in seen:
            errors.append(f"instance {i} requires a unique instance_id")
        else:
            seen.add(key)
        if "dataset_seed" not in instance:
            errors.append(f"instance {i} needs dataset_seed distinct in meaning from search/training seeds")
        elif not isinstance(instance["dataset_seed"], int) or isinstance(instance["dataset_seed"], bool):
            errors.append(f"instance {i} dataset_seed must be an integer")
        if "search_seeds" not in instance and "training_seeds" not in instance:
            errors.append(f"instance {i} needs search_seeds or training_seeds")
        for field in ("search_seeds", "training_seeds"):
            if field in instance and (not isinstance(instance[field], list) or not instance[field]
                                      or any(not isinstance(v, int) or isinstance(v, bool) for v in instance[field])):
                errors.append(f"instance {i} {field} must be a nonempty integer list")
    if len({json.dumps([x.get("problem_id"), x.get("dataset_seed")], sort_keys=True, default=str)
            for x in instances if isinstance(x, dict)}) < 2:
        warnings.append("independent-instance transfer is unmeasured; repeated search seeds are not independent datasets")
    memory = p.get("memory", {})
    if not isinstance(memory, dict):
        errors.append("memory must be a mapping")
        memory = {}
    if isinstance(memory, dict) and memory.get("mode") == "frozen" and not memory.get("snapshot_id"):
        errors.append("frozen memory requires snapshot_id")
    comparisons = p.get("comparisons", [])
    if not isinstance(comparisons, list):
        errors.append("comparisons must be a list")
        comparisons = []
    for i, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            errors.append(f"comparison {i} must be a mapping")
            continue
        arms = comparison.get("arms", [])
        if not isinstance(arms, list) or not all(isinstance(a, dict) for a in arms):
            errors.append(f"comparison {i} arms must be mappings in a list")
            continue
        if len(arms) < 2:
            errors.append(f"comparison {i} requires at least two arms")
            continue
        if not comparison.get("baseline_hash"):
            errors.append(f"comparison {i} requires a frozen baseline_hash")
        if not comparison.get("budget_policy"):
            errors.append(f"comparison {i} requires a declared budget_policy")
        changed = comparison.get("changed_components", [])
        if not isinstance(changed, list) or any(not isinstance(v, str) for v in changed):
            errors.append(f"comparison {i} changed_components must be a string list")
            changed = []
        if len(changed) != 1 and comparison.get("kind", "component_ablation") == "component_ablation":
            errors.append(f"comparison {i} component ablation must change exactly one declared component")
        for field in ("dataset_seed", "dataset_hash", "memory_snapshot_id"):
            vals = {json.dumps(a.get(field), sort_keys=True) for a in arms if isinstance(a, dict)}
            if len(vals) > 1:
                errors.append(f"comparison {i} confounds {field}")
    roles = p.get("split_roles", {})
    if not isinstance(roles, dict):
        errors.append("split_roles must be a mapping")
        roles = {}
    if roles and any(roles.get(k) in ("development", "validation") for k in ("test", "final_stress")):
        errors.append("final evidence cannot be assigned a development role")
    return {"schema": SCHEMA, "valid": not errors, "status": "valid" if not errors else "invalid",
            "errors": errors, "warnings": warnings, "normalized": p}


def evaluate_program(bundle) -> dict:
    """Descriptive program metrics; absent discovery/transfer evidence stays unknown."""
    b = bundle or {}
    studies = b.get("studies", [])
    studies = list(studies.values()) if isinstance(studies, dict) else studies
    verdicts = b.get("claim_verdicts", b.get("verdicts"))
    if verdicts is None:
        verdicts = [v for s in studies for v in s.get("verdicts", [])]
    verdicts = list(verdicts.values()) if isinstance(verdicts, dict) else verdicts
    complete = sum(s.get("status", (s.get("completion") or {}).get("status")) == "complete" for s in studies)
    outcomes = {k: sum(v.get("outcome") == k for v in verdicts) for k in
                ("supported", "contradicted", "inconclusive", "untestable")}
    final = b.get("final") or {}
    cert = b.get("export_certificate", final.get("certificate", final.get("export_certificate", (final.get("result") or {}).get("export_certificate"))))
    orphans = 0
    for s in studies:
        completion = s.get("completion") or {}
        if s.get("orphaned_arm_ids") or completion.get("orphaned_arm_ids"):
            orphans += 1
            continue
        arms = s.get("arms", [])
        arms = list(arms.values()) if isinstance(arms, dict) else arms
        required = [a for a in arms if isinstance(a, dict) and a.get("required", True)]
        finished = sum(a.get("status") in {"ok", "complete", "completed"} for a in required)
        if finished and finished < len(required):
            orphans += 1
    unsupported = [v for v in verdicts if v.get("outcome") == "supported" and
                   (v.get("unmet_requirements") or v.get("missing") or
                    not (v.get("evidence_ids") or v.get("evidence_artifact_ids")))]
    covered = sum(bool(v.get("evidence_ids") or v.get("evidence_artifact_ids")) for v in verdicts)
    return {"schema": SCHEMA, "model_quality": final.get("test", b.get("model_quality")),
            "study_count": len(studies), "studies_complete": complete,
            "study_completion_rate": complete / len(studies) if studies else None,
            "orphan_studies": orphans, "claim_outcomes": outcomes,
            "claim_evidence_coverage": covered / len(verdicts) if verdicts else None,
            "claim_reliability": {"unsupported_support_count": len(unsupported),
                                  "supported_claims_have_evidence": not unsupported,
                                  "scope": "record integrity; not an independent truth assessment"},
            "claim_resolution_rate": (outcomes["supported"] + outcomes["contradicted"]) / len(verdicts) if verdicts else None,
            "reproducibility": cert, "cost": b.get("resources", b.get("costs", b.get("cost"))),
            "discovery": {"established": None, "reason": "method novelty requires explicit independent attribution review"},
            "transfer": {"established": None, "reason": "no independent-instance transfer evaluation supplied"},
            "time_to_resolved_questions": {"value": None, "reason": "requires registered question-start and resolution timestamps"}}
