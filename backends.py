"""Live structured-output backend and a clearly labelled offline plumbing demo."""

import json
from typing import Any

from pydantic import BaseModel

from .credentials import openai_api_key
from .schemas import Critique, Proposal


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str, max_output_tokens: int, seed: int):
        from langchain.chat_models import init_chat_model

        key = openai_api_key()
        if not key:
            raise ValueError("No OpenAI API key configured. Run python -m workflow.response_call configure-openai.")
        self.model = init_chat_model(
            model, model_provider="openai", max_tokens=max_output_tokens,
            max_retries=0, timeout=90, temperature=0, seed=seed, api_key=key,
        )

    def invoke(self, stage: str, schema: type[BaseModel], messages: list[dict]) -> dict:
        runnable = self.model.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
        response = runnable.invoke(messages)
        raw = response["raw"]
        parsed = response.get("parsed")
        return {
            "parsed": parsed.model_dump() if isinstance(parsed, BaseModel) else parsed,
            "parsing_error": str(response["parsing_error"]) if response.get("parsing_error") else None,
            "raw": {"content": raw.content, "tool_calls": raw.tool_calls,
                    "additional_kwargs": raw.additional_kwargs,
                    "response_metadata": raw.response_metadata},
            "usage": raw.usage_metadata,
        }


class DemoBackend:
    """Scripted Lean failure/repair, not evidence of autonomous discovery."""
    name = "demo"

    def invoke(self, stage: str, schema: type[BaseModel], messages: list[dict]) -> dict[str, Any]:
        payload = json.loads(messages[-1]["content"])
        packet = payload["benchmark"]
        repaired = stage != "CA"
        if schema is Critique:
            repaired = bool(payload["proposal"]["repairs"])
            parsed = {
                "assessment": "Scripted critique. Lean must check the exact fixed goal; this demo does not measure AI research ability.",
                "target_preserved": True, "assumptions_preserved": True,
                "gaps": [] if repaired else [{
                    "gap_id": "field_algebra", "description": "Reflexivity does not use the observation equation or justify division by the nonzero time step.",
                    "severity": "major", "scope": "general_proof",
                    "suggested_repair": "Introduce the hypotheses, unfold the definitions, and prove the field identity using h ≠ 0.",
                }], "simulator_requests": [],
            }
        else:
            recovery = packet["id"] == "q-recovery-one-step"
            ready = packet["verification_contract"].get("status") == "ready"
            body = ("intro K inst V B h rate q observed hh data x hx\n"
                    "have hxdata := data x hx\n"
                    "simp only [recoverPotential, oneStepReturn] at *\n"
                    "grind") if recovery else (
                    "intro K inst V h rate q₁ q₂ hh data\n"
                    "funext x\nhave hxdata := data x\n"
                    "simp only [oneStepReturn] at hxdata\ngrind")
            parsed = {
                "claim": packet["statement"], "assumptions": packet["assumptions"],
                "proof": (
                    r"Fix $x\in B$. The observation equation gives $1-d(x)=h(r(x)+q(x))$. "
                    r"Since $h\ne0$, division by $h$ and subtraction of $r(x)$ gives the recovery formula."
                    if recovery else
                    r"At each $x$, equal observations give $h(r(x)+q_1(x))=h(r(x)+q_2(x))$. "
                    r"Cancel the nonzero $h$ and the known $r(x)$, then use function extensionality."
                ) if repaired and ready else (
                    "The recovery claim follows by reflexivity. This step still needs justification." if ready
                    else "The controllability theorem requires a reviewed formal statement before Lean proof construction."
                ),
                "repairs": [{"gap_id": "field_algebra", "change": "Use the observation equality and nonzero step in a field proof; submit the repaired Lean body."}] if repaired and ready else [],
                "remaining_gaps": [] if repaired and ready else ["The formal target has not yet been proved."],
                "certificate": {"kind": "lean4", "target_id": packet["id"], "proof": body if repaired else "rfl"} if ready else None,
                "simulator_requests": [],
            }
        validated = schema.model_validate(parsed).model_dump()
        return {"parsed": validated, "parsing_error": None,
                "raw": {"content": json.dumps(validated), "response_metadata": {"finish_reason": "stop", "scripted": True}},
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
