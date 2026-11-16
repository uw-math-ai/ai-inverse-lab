"""Bounded V3 agent instances and provider-neutral role routing.

The registry controls actual callback dispatch. Persona is deliberately independent of ID.
All decisions, refusals and specialist results are emitted to the canonical artifact ledger.
"""
from __future__ import annotations

import copy
import json
import math
import time

from .. import artifacts as A
from ..llm import extract_json, make_llm


MODEL_LIST_LIMIT = 32


def compact_for_model(value, limit=MODEL_LIST_LIMIT):
    """Copy of a role context for a model prompt: long numeric lists (per-row diagnostics) become
    summary statistics. The persisted ledger/bundle keep the full arrays; only the prompt shrinks,
    which otherwise grows by ~100 kB per executed study and dominates token cost."""
    if isinstance(value, dict):
        return {k: compact_for_model(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > limit and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            a = [float(x) for x in value]
            finite = sorted(x for x in a if math.isfinite(x))
            q = (lambda f: finite[min(len(finite) - 1, int(f * (len(finite) - 1)))]) if finite else (lambda f: None)
            return {"elided_numeric_list": True, "n": len(a), "n_finite": len(finite),
                    "mean": sum(finite) / len(finite) if finite else None,
                    "min": q(0.0), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": q(1.0),
                    "first": a[:4]}
        return [compact_for_model(v, limit) for v in value]
    return value


class RoleRouter:
    def __init__(self, default, configs=None, injected=None, resources=None, emit=None):
        self.default = default
        self.configs = configs or {}
        self.backends = dict(injected or {})
        self.resources = resources
        self.emit = emit or (lambda *a, **kw: None)
        self.events = []

    def backend(self, role, escalate=False):
        tier = f"{role}_escalated" if escalate else role
        if tier not in self.backends:
            self.backends[tier] = make_llm(self.configs[tier]) if tier in self.configs else self.default
        return tier, self.backends[tier]

    def ask(self, role, context, *, actor, purpose, escalate=False):
        tier, backend = self.backend(role, escalate)
        rec = {"actor": actor, "role": role, "tier": tier, "purpose": purpose,
               "backend": backend.name, "model": backend.model, "cost_usd": None}
        value = None
        if backend.is_null:
            rec["fallback_reason"] = "offline_choice"
        elif self.resources is not None and not self.resources.try_llm_call():
            rec["fallback_reason"] = "budget_refusal"
        else:
            rec["call_attempted"] = True
            t0 = time.monotonic()
            try:
                response = backend.complete(
                    "You are a scientific research agent. Use only the supplied development evidence. "
                    "Return one JSON object matching the requested interface; never certify your own claims.",
                    json.dumps(compact_for_model(context), sort_keys=True, default=str))
                waited = float(getattr(response, "waited_seconds", 0) or 0)
                if self.resources is not None:
                    if waited:       # a usage-limit pause is not research time: stop the wall clock
                        self.resources.start += waited
                    self.resources.charge_llm(response.seconds - waited, response.cost_usd)
                rec.update(seconds=response.seconds - waited, cost_usd=response.cost_usd,
                           input_tokens=response.input_tokens, output_tokens=response.output_tokens)
                if hasattr(response, "usage_record"):     # injected test doubles may return plain responses
                    rec.update({k: v for k, v in response.usage_record().items() if v is not None})
                rec["response_model"] = response.model
                value = extract_json(response.text)
                if value is None:
                    rec["fallback_reason"] = "invalid_json"
                elif not isinstance(value, dict):
                    value = None
                    rec["fallback_reason"] = "invalid_schema"
            except Exception as exc:
                rec.update(fallback_reason="service_error", error=f"{type(exc).__name__}: {exc}",
                           seconds=time.monotonic()-t0)
                if self.resources is not None:
                    self.resources.charge_llm(rec["seconds"], None)
        self.events.append(rec)
        self.emit(A.RESOURCE, rec, author=actor, summary=f"{purpose}: {rec.get('fallback_reason', 'response')}")
        return value


class DelegationManager:
    def __init__(self, limits, *, emit=None):
        self.limits = limits
        self.emit = emit or (lambda *a, **kw: None)
        self.agents = {}
        self.tasks = []
        self._counter = 0

    def spawn(self, role, *, persona=None, parent_id=None, question=None, tier=None):
        if role not in ("orchestrator", "critic", "researcher", "specialist"):
            return self._refuse("unknown role")
        if role in ("orchestrator", "critic") and any(a["role"] == role for a in self.agents.values()):
            return self._refuse(f"independent {role} already exists")
        parent = self.agents.get(parent_id) if parent_id else None
        if parent_id and parent is None:
            return self._refuse("unknown parent")
        if role == "specialist" and (parent is None or parent["role"] not in ("researcher", "specialist")):
            return self._refuse("specialists require a researcher/specialist parent")
        if role == "researcher" and (parent is None or parent["role"] != "orchestrator"):
            return self._refuse("researchers require orchestrator parent")
        if parent and parent["role"] == "critic":
            return self._refuse("critic cannot own execution agents")
        depth = parent["depth"]+1 if parent else 0
        active = sum(a["status"] == "active" for a in self.agents.values())
        children = sum(a["parent_id"] == parent_id for a in self.agents.values()) if parent else 0
        if len(self.agents) >= self.limits.get("max_agents", 12):
            return self._refuse("total agent limit")
        if active >= self.limits.get("max_active_agents", 6):
            return self._refuse("active agent limit")
        if depth > self.limits.get("max_depth", 2) or children >= self.limits.get("max_children", 4):
            return self._refuse("depth/children limit")
        self._counter += 1
        aid = f"{role}-{self._counter:03d}"
        rec = {"id": aid, "role": role, "persona": persona, "parent_id": parent_id,
               "question": question, "depth": depth, "tier": tier or role,
               "status": "active", "turns": 0, "tasks": 0, "seconds": 0.0, "cost_usd": 0.0,
               "evidence_ids": [], "messages": []}
        self.agents[aid] = rec
        self.emit(A.AGENT, rec, author=parent_id or "scheduler", summary=f"spawned {aid}: {question}")
        return rec

    def _refuse(self, reason):
        self.emit(A.FAILURE, {"source": "delegation", "reason": reason}, author="delegation",
                  summary=f"delegation refused: {reason}")
        raise ValueError(reason)

    def dispatch(self, parent_id, kind, request, callback):
        parent = self.agents.get(parent_id)
        if parent is None or parent["role"] not in ("researcher", "specialist") or parent["status"] != "active":
            return self._refuse("task parent must be an active researcher/specialist")
        if len(self.tasks) >= self.limits.get("max_specialist_tasks", 12):
            return self._refuse("specialist task limit")
        if parent["turns"] >= self.limits.get("max_agent_turns", 8):
            return self._refuse("parent turn limit")
        if kind not in ("coding_execution", "analysis", "retrieval", "source"):
            return self._refuse("unsupported specialist capability")
        seconds_limit = float(request.get("timeout_seconds", self.limits.get("max_specialist_seconds", 60)))
        if not math.isfinite(seconds_limit) or not 0 < seconds_limit <= float(self.limits.get("max_specialist_seconds", 60)):
            return self._refuse("invalid specialist time allocation")
        agent = self.spawn("specialist", persona=kind, parent_id=parent_id, question=request.get("question"))
        tid = f"task-{len(self.tasks)+1:03d}"
        task = {"id": tid, "parent_id": parent_id, "agent_id": agent["id"], "kind": kind,
                "request": copy.deepcopy(request), "status": "running", "time_limit_seconds": seconds_limit,
                "time_policy": "adapter timeouts plus observed soft callback deadline"}
        self.tasks.append(task)
        start = time.monotonic()
        try:
            task["result"] = callback(agent, request)
            task["status"] = "complete" if task["result"].get("status") not in ("error", "unsupported") else "incomplete"
        except Exception as exc:
            task.update(status="failed", result={"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                                  "uncertainty": ["specialist work did not complete"]})
        finally:
            task["seconds"] = time.monotonic()-start
            if task["seconds"] > seconds_limit:
                task["status"] = "deadline_exceeded"
                task["deadline_overrun_seconds"] = task["seconds"]-seconds_limit
            agent.update(status="complete", seconds=task["seconds"])
            parent["seconds"] += task["seconds"]
            parent["tasks"] += 1
            parent["turns"] += 1
        art = self.emit(A.RESULT, task, author=agent["id"], summary=f"{tid} {kind}: {task['status']}")
        task["artifact_id"] = getattr(art, "id", None)
        parent["messages"].append({"task_id": tid, "artifact_id": task["artifact_id"],
                                   "result": copy.deepcopy(task["result"])})
        return task
