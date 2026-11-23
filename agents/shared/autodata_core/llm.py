"""Model calls through the Research Agent Workplace providers (agents/instructions.md, section 5).

The platform injects AGENT_PROVIDER (codex | claude | openrouter | openai), AGENT_MODEL and the
credential variables; HTTPS goes through a proxy that reaches only that provider. The invocations
below follow the platform's tested provider example. Without AGENT_PROVIDER (local runs, the free
agent) make_llm() returns NullLLM and every caller falls back to its heuristic.

Never print credentials, authorization headers or raw error bodies.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import platform_usage            # agents/shared/platform_usage.py
except ImportError:                  # pragma: no cover - only when run outside agents/
    platform_usage = None


class LLMError(RuntimeError):
    pass


class NullLLM:
    name, is_null = "none", True
    calls = tokens = 0

    def complete(self, system, prompt):
        raise LLMError("no model provider configured")


def _workdir() -> Path:
    for base in ("/work", None):
        try:
            if base and Path(base).is_dir() and os.access(base, os.W_OK):
                d = Path(base) / "llm"
                d.mkdir(exist_ok=True)
                return d
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="autodata_llm_"))


class ProviderLLM:
    is_null = False

    def __init__(self, provider: str, model: str | None, timeout: float = 600):
        self.provider, self.model, self.timeout = provider, model, timeout
        self.name = provider
        self.calls, self.tokens = 0, 0
        self.dir = _workdir()

    def _report(self, tokens=0, cost=0.0):
        self.calls += 1
        self.tokens += int(tokens or 0)
        if platform_usage is not None:
            platform_usage.add_usage(tokens=int(tokens or 0), cost_usd=float(cost or 0.0))

    def complete(self, system: str, prompt: str) -> str:
        text = f"{system}\n\n---\n\n{prompt}"
        try:
            return getattr(self, f"_{self.provider}")(text)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{self.provider} timed out after {exc.timeout}s") from None
        except subprocess.CalledProcessError as exc:
            raise LLMError(f"{self.provider} CLI exited with code {exc.returncode}") from None
        except urllib.error.HTTPError as exc:
            raise LLMError(f"{self.provider} HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f"{self.provider} connection error ({type(exc).__name__})") from None

    def _claude(self, prompt):
        env = {**os.environ, "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1"}
        args = ["claude", "-p", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
                "--no-session-persistence", "--max-turns", "1", "--output-format", "json"]
        if self.model:
            args += ["--model", self.model]
        out = subprocess.run(args, input=prompt, text=True, encoding="utf-8", stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, check=True, cwd=self.dir, env=env,
                             timeout=self.timeout).stdout
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            self._report()
            return out
        if obj.get("is_error"):
            self._report()
            raise LLMError("claude reported an error result")
        u = obj.get("usage") or {}
        tokens = sum(int(u.get(k) or 0) for k in ("input_tokens", "output_tokens",
                                                  "cache_creation_input_tokens", "cache_read_input_tokens"))
        self._report(tokens, obj.get("total_cost_usd") or 0.0)
        return obj.get("result", "")

    def _codex(self, prompt):
        codex_dir = self.dir / "codex"
        codex_dir.mkdir(exist_ok=True)
        last = codex_dir / f"last_{time.time_ns()}.md"
        args = ["codex", "-a", "never", "-c", 'web_search="disabled"', "--disable", "shell_tool",
                "--disable", "view_image", "exec", "--ephemeral", "--sandbox", "read-only",
                "--skip-git-repo-check", "--output-last-message", str(last)]
        if self.model:
            args += ["--model", self.model]
        subprocess.run([*args, "-"], input=prompt, text=True, encoding="utf-8", cwd=codex_dir, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=self.timeout)
        self._report()
        return last.read_text(encoding="utf-8") if last.exists() else ""

    def _post(self, url, body, key):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def _openrouter(self, prompt):
        obj = self._post(os.environ["OPENROUTER_BASE_URL"].rstrip("/") + "/chat/completions",
                         {"model": self.model or os.environ.get("OPENROUTER_MODEL") or "openrouter/free",
                          "messages": [{"role": "user", "content": prompt}]},
                         os.environ["OPENROUTER_API_KEY"])
        self._report((obj.get("usage") or {}).get("total_tokens", 0))
        choices = obj.get("choices") or [{}]
        return ((choices[0].get("message") or {}).get("content")) or ""

    def _openai(self, prompt):
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        obj = self._post(base + "/responses", {"model": self.model or os.environ["OPENAI_MODEL"], "input": prompt},
                         os.environ["OPENAI_API_KEY"])
        self._report((obj.get("usage") or {}).get("total_tokens", 0))
        parts = [c.get("text", "") for item in obj.get("output") or [] if item.get("type") == "message"
                 for c in item.get("content") or [] if c.get("type") == "output_text"]
        return "".join(parts) or obj.get("output_text", "")


def make_llm():
    provider = os.environ.get("AGENT_PROVIDER")
    if not provider:
        return NullLLM()
    if provider not in ("claude", "codex", "openrouter", "openai"):
        raise LLMError(f"unsupported AGENT_PROVIDER {provider!r}")
    own = {"claude": "CLAUDE_MODEL", "codex": "CODEX_MODEL", "openrouter": "OPENROUTER_MODEL",
           "openai": "OPENAI_MODEL"}[provider]
    model = os.environ.get("AGENT_MODEL") or os.environ.get(own)
    return ProviderLLM(provider, model or None)


def extract_json(text):
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
