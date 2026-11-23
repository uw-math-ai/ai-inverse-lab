"""Model backends for the LLM challenger / analyzer / implementer.

Same conventions as the lab's ML research harness (mlharness/llm.py): one method,
``complete(system, prompt) -> text``. Every caller falls back to its heuristic when the backend is
missing or fails, so a run never depends on a model.

none         no model (heuristic challenger only)
anthropic    Anthropic Messages API via the official `anthropic` SDK
             (adaptive thinking, output_config.effort, server-side refusal fallback)
claude_code  Claude Code headless (`claude -p --output-format json`), tools disabled
auto         anthropic if the SDK and credentials are present, else claude_code, else none
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile


class LLMError(RuntimeError):
    pass


class NullLLM:
    name, is_null = "none", True

    def complete(self, system: str, prompt: str) -> str:
        raise LLMError("no LLM backend configured")


class AnthropicLLM:
    name, is_null = "anthropic", False

    def __init__(self, model: str, effort: str | None, timeout: float):
        import anthropic
        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic(timeout=timeout)

    def complete(self, system: str, prompt: str) -> str:
        import anthropic
        kw = dict(model=self.model, max_tokens=64000, system=system,
                  messages=[{"role": "user", "content": prompt}], thinking={"type": "adaptive"})
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        try:
            try:   # server-side refusal fallback; retried without it on SDKs that predate the parameter
                with self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                                      fallbacks="default", **kw) as stream:
                    msg = stream.get_final_message()
            except TypeError:
                with self.client.messages.stream(**kw) as stream:
                    msg = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc
        if msg.stop_reason == "refusal":
            raise LLMError("model declined the request")
        return "".join(b.text for b in msg.content if b.type == "text")


class ClaudeCodeLLM:
    name, is_null = "claude_code", False

    def __init__(self, exe: str, model: str | None, effort: str | None, timeout: float):
        self.exe, self.model, self.effort, self.timeout = exe, model, effort, timeout
        self.workdir = tempfile.mkdtemp(prefix="autodata_claude_")

    def complete(self, system: str, prompt: str) -> str:
        cmd = [self.exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  timeout=self.timeout, cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude timed out after {exc.timeout}s") from exc
        if proc.returncode != 0 and not proc.stdout.strip():
            raise LLMError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout
        if out.get("is_error"):
            raise LLMError(f"claude error: {str(out.get('result'))[:500]}")
        return out.get("result", "")


def _anthropic_ready() -> bool:
    import importlib.util
    return importlib.util.find_spec("anthropic") is not None and bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_PROFILE"))


def make_llm(cfg: dict):
    backend = (cfg.get("backend") or "auto").lower()
    model, effort, timeout = cfg.get("model"), cfg.get("effort"), float(cfg.get("timeout", 300))
    claude = os.environ.get("AUTODATA_CLAUDE_CLI") or shutil.which("claude")
    if backend == "auto":
        backend = "anthropic" if _anthropic_ready() else ("claude_code" if claude else "none")
    if backend == "none":
        return NullLLM()
    if backend == "anthropic":
        return AnthropicLLM(model or "claude-opus-5-5", effort, timeout)
    if backend == "claude_code":
        if not claude:
            raise LLMError("Claude Code CLI not found (set AUTODATA_CLAUDE_CLI)")
        return ClaudeCodeLLM(claude, model, effort, timeout)
    raise LLMError(f"unknown llm backend {backend!r}")


def extract_json(text: str | None):
    """First JSON object in a model response (bare or fenced); None if there is none."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None
