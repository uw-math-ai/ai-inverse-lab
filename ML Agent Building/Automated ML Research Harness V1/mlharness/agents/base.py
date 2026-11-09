"""Common agent machinery: artifact I/O and budgeted, logged, fail-soft LLM calls."""
from __future__ import annotations

import logging

from .. import artifacts as A
from ..llm import LLMError, extract_json

log = logging.getLogger("mlharness")


class Agent:
    role = "agent"

    def __init__(self, name: str, spec, store, llm, resources):
        self.name = name
        self.spec = spec
        self.store = store
        self.llm = llm
        self.resources = resources

    @property
    def uses_llm(self) -> bool:
        return self.llm is not None and not self.llm.is_null

    def emit(self, kind, data=None, **kw) -> A.Artifact:
        return self.store.put(kind, data, author=self.name, **kw)

    def ask(self, system: str, prompt: str, *, purpose: str, idea_id=None, stage=None) -> str | None:
        """One budgeted LLM call. Returns text, or None (caller falls back to heuristics)."""
        if not self.uses_llm:
            return None
        if not self.resources.try_llm_call():
            log.info("[%s] LLM call budget exhausted; using heuristics for %s", self.name, purpose)
            return None
        try:
            resp = self.llm.complete(system, prompt)
        except LLMError as exc:
            log.warning("[%s] LLM call for %s failed: %s", self.name, purpose, exc)
            self.emit(A.FAILURE, {"purpose": purpose, "error": str(exc), "source": "llm"},
                      summary=f"LLM call failed ({purpose})", idea_id=idea_id, stage=stage)
            return None
        self.resources.charge_llm(resp.seconds, resp.cost_usd)
        self.emit(A.RESOURCE, {"type": "llm_call", "purpose": purpose, "backend": resp.backend,
                               "model": resp.model, "seconds": round(resp.seconds, 2),
                               "cost_usd": resp.cost_usd, "input_tokens": resp.input_tokens,
                               "output_tokens": resp.output_tokens,
                               "prompt_chars": len(prompt), "response_chars": len(resp.text)},
                  summary=f"LLM call: {purpose}", idea_id=idea_id, stage=stage)
        return resp.text

    def ask_json(self, system, prompt, **kw):
        text = self.ask(system, prompt, **kw)
        return extract_json(text) if text is not None else None
