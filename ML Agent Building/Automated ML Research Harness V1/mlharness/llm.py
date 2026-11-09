"""LLM / coding-agent backends.

Every agent talks to a model through ``LLMBackend.complete(system, prompt)`` and must cope with
``None``/failure by falling back to its built-in heuristics, so a run never depends on a model.

Backends
--------
none        : no model; agents use heuristic policies (deterministic, offline, free).
claude_code : Claude Code headless (``claude -p --output-format json``), tools disabled.
anthropic   : Anthropic Messages API via the official ``anthropic`` SDK.
codex       : OpenAI Codex CLI (``codex exec``), read-only sandbox.
auto        : anthropic (if SDK + credentials) -> claude_code -> codex -> none.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    seconds: float
    backend: str
    model: str | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMBackend:
    name = "base"
    is_null = False
    model = None

    def complete(self, system: str, prompt: str, timeout: float | None = None) -> LLMResponse:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}" + (f" ({self.model})" if self.model else "")


class NullLLM(LLMBackend):
    name = "none"
    is_null = True

    def complete(self, system, prompt, timeout=None):
        raise LLMError("no LLM backend configured")


def _newest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths, key=os.path.getmtime)[-1] if paths else None


def find_claude_cli() -> str | None:
    if os.environ.get("MLH_CLAUDE_CLI"):
        return os.environ["MLH_CLAUDE_CLI"]
    exe = shutil.which("claude")
    if exe:
        return exe
    home = Path.home()
    pats = [str(home / ".vscode" / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / "claude*"),
            str(home / ".cursor" / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / "claude*"),
            str(home / ".local" / "bin" / "claude*"), str(home / ".claude" / "local" / "claude*")]
    return _newest([p for pat in pats for p in glob.glob(pat)])


def find_codex_cli() -> str | None:
    if os.environ.get("MLH_CODEX_CLI"):
        return os.environ["MLH_CODEX_CLI"]
    exe = shutil.which("codex")
    if exe:
        return exe
    import platform
    home = Path.home()
    osname = {"nt": "windows", "posix": "darwin" if platform.system() == "Darwin" else "linux"}[os.name]
    exe = "codex.exe" if os.name == "nt" else "codex"
    pat = str(home / ".vscode" / "extensions" / "openai.chatgpt-*" / "bin" / f"{osname}-*" / exe)
    return _newest(glob.glob(pat))


class ClaudeCodeLLM(LLMBackend):
    """Claude Code in headless print mode. Runs in an empty scratch dir with all tools disabled,
    so it acts as a pure reasoning/code-writing agent; the harness executes everything itself."""
    name = "claude_code"

    def __init__(self, exe: str, model: str | None = None, effort: str | None = None,
                 timeout: float = 300, workdir: Path | None = None):
        self.exe, self.model, self.effort, self.timeout = exe, model, effort, timeout
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="mlh_claude_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def complete(self, system, prompt, timeout=None):
        cmd = [self.exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--append-system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  timeout=timeout or self.timeout, cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude timed out after {exc.timeout}s") from exc
        if proc.returncode != 0 and not proc.stdout.strip():
            raise LLMError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return LLMResponse(proc.stdout, time.time() - t0, self.name, self.model)
        if out.get("is_error"):
            raise LLMError(f"claude error: {str(out.get('result'))[:500]}")
        usage = out.get("usage") or {}
        return LLMResponse(out.get("result", ""), time.time() - t0, self.name,
                           self.model or "claude-code-default", out.get("total_cost_usd"),
                           usage.get("input_tokens"), usage.get("output_tokens"))


class AnthropicLLM(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = "medium", timeout: float = 300):
        import anthropic  # noqa: F401  (fail fast if the SDK is missing)
        self.model = model or "claude-opus-5-5"
        self.effort, self.timeout = effort, timeout
        self.client = anthropic.Anthropic(timeout=timeout)

    def complete(self, system, prompt, timeout=None):
        import anthropic
        t0 = time.time()
        kw = dict(model=self.model, max_tokens=32000, system=system,
                  messages=[{"role": "user", "content": prompt}],
                  thinking={"type": "adaptive"})
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        try:
            try:   # server-side refusal fallback (opt-in); retried without it on older SDKs
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
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        return LLMResponse(text, time.time() - t0, self.name, self.model, None,
                           getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))


class CodexLLM(LLMBackend):
    name = "codex"

    def __init__(self, exe: str, model: str | None = None, timeout: float = 300,
                 workdir: Path | None = None):
        self.exe, self.model, self.timeout = exe, model, timeout
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="mlh_codex_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def complete(self, system, prompt, timeout=None):
        out_file = self.workdir / f"last_{time.time_ns()}.txt"
        cmd = [self.exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only",
               "--output-last-message", str(out_file)]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append("-")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, input=f"{system}\n\n{prompt}", capture_output=True, text=True,
                                  encoding="utf-8", timeout=timeout or self.timeout, cwd=self.workdir)
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"codex timed out after {exc.timeout}s") from exc
        if out_file.exists():
            text = out_file.read_text(encoding="utf-8")
        elif proc.returncode == 0:
            text = proc.stdout
        else:
            raise LLMError(f"codex exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        return LLMResponse(text, time.time() - t0, self.name, self.model)


def _anthropic_ready() -> bool:
    import importlib.util
    return importlib.util.find_spec("anthropic") is not None and bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_PROFILE"))


def make_llm(cfg: dict | None, workdir: Path | None = None) -> LLMBackend:
    cfg = cfg or {}
    backend = (cfg.get("backend") or "auto").lower()
    model, timeout, effort = cfg.get("model"), float(cfg.get("timeout", 300)), cfg.get("effort")
    if backend == "auto":
        if _anthropic_ready():
            backend = "anthropic"
        elif find_claude_cli():
            backend = "claude_code"
        elif find_codex_cli():
            backend = "codex"
        else:
            backend = "none"
    if backend == "none":
        return NullLLM()
    if backend == "anthropic":
        return AnthropicLLM(model, effort, timeout)
    if backend == "claude_code":
        exe = find_claude_cli()
        if not exe:
            raise LLMError("Claude Code CLI not found (set MLH_CLAUDE_CLI)")
        return ClaudeCodeLLM(exe, model, effort, timeout, workdir)
    if backend == "codex":
        exe = find_codex_cli()
        if not exe:
            raise LLMError("Codex CLI not found (set MLH_CODEX_CLI)")
        return CodexLLM(exe, model, timeout, workdir)
    raise LLMError(f"unknown llm backend {backend!r}")


def extract_json(text: str):
    """Parse the first JSON object/array in a model response (fenced or bare)."""
    if text is None:
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
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None
