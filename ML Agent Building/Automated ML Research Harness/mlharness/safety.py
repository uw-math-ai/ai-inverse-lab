"""Static and dynamic safety controls for agent-written code (V2).

Static (before execution, in the implementation gate):
    ``check_code_v2(code, contract)``  AST scan: required entry point, banned imports/calls,
    no dunder / private attribute access, no reflective builtins, no references to test data.

Dynamic (inside the isolated runner subprocess, only for jobs carrying agent code):
    ``install_guard(allowed_dirs)``  a PEP 578 audit hook that, once installed, cannot be removed for
    the life of the process. It denies process creation, network connections, and any
    file write/delete/rename outside the job's own directories. The runner also starts the process
    with credentials scrubbed from the environment (``envfile.scrubbed_env``) and a hard timeout.

Limits (documented in the V2 implementation report): an audit hook is defence in depth, not a
container; a determined adversary with native-code access could escape it. Native-code entry points
(ctypes, cffi) are blocked statically; they are not blocked dynamically because numerical libraries
(threadpoolctl inside scikit-learn) legitimately use ctypes. For hostile code use a
container/VM runner.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

BANNED_MODULES = {"subprocess", "socket", "requests", "urllib", "urllib3", "http", "shutil", "ctypes", "cffi",
                  "ftplib", "smtplib", "webbrowser", "pickle", "marshal", "multiprocessing", "asyncio",
                  "importlib", "pathlib", "glob", "io", "builtins", "signal", "threading", "_thread",
                  "paramiko", "telnetlib", "pty", "mmap", "inspect", "gc", "sys", "os"}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "getattr", "setattr",
                "delattr", "vars", "globals", "locals", "memoryview", "help", "exit", "quit"}
ALLOWED_DUNDERS = {"__init__", "__call__", "__name__", "__len__", "__repr__"}
BANNED_ATTRS = {"_ctx", "ctypes", "ctypeslib", "f_globals", "f_locals", "gi_frame", "tb_frame"}
TEST_DATA_TOKENS = ("X_test", "y_test", "dataset.npz", "data_path", "shift_")
CONTRACTS = {"build_model": "build_model(params, task, seed)", "build_method": "build_method(params, context)"}


def check_code_v2(code: str, contract: str = "build_model") -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == contract for n in tree.body):
        problems.append(f"code must define a top-level {CONTRACTS.get(contract, contract)}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for nm in names:
                if nm.split(".")[0] in BANNED_MODULES:
                    problems.append(f"forbidden import {nm}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                problems.append(f"forbidden call {f.id}()")
            elif isinstance(f, ast.Attribute) and f.attr in {"system", "popen", "rmtree", "unlink", "fork",
                                                              "execv", "execve", "spawn", "write_text",
                                                              "write_bytes", "savetxt", "save", "savez",
                                                              "tofile", "to_csv", "dump"}:
                problems.append(f"forbidden call .{f.attr}()")
        elif isinstance(node, ast.Attribute) and (
                (node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS) or node.attr in BANNED_ATTRS):
            problems.append(f"access to attribute .{node.attr} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in ("__name__",):
            problems.append(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for tok in TEST_DATA_TOKENS:
                if tok in node.value:
                    problems.append(f"reference to held-out data ({tok!r}) is not allowed")
    return sorted(set(problems))


def references_test_data(obj) -> list[str]:
    """Leakage scan over any JSON-able method spec / proposal (strings anywhere)."""
    hits = []

    def walk(x):
        if isinstance(x, str):
            hits.extend(t for t in TEST_DATA_TOKENS if t in x)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k not in ("code",):           # code is scanned by check_code_v2
                    walk(k)
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(obj)
    return sorted(set(hits))


# ---------------------------------------------------------------------------------------------
# dynamic guard (PEP 578 audit hook)
# ---------------------------------------------------------------------------------------------
class SandboxViolation(PermissionError):
    pass


_DENY_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                "os.forkpty", "os.startfile", "_winapi.CreateProcess", "pty.spawn", "socket.connect",
                "socket.bind", "socket.sendto", "socket.getaddrinfo", "os.kill", "signal.pthread_kill",
                "winreg.CreateKey", "winreg.SetValue", "webbrowser.open"}
_PATH_WRITE_EVENTS = {"os.remove", "os.rmdir", "os.rename", "os.mkdir", "os.chmod", "os.chown", "os.link",
                      "os.symlink", "os.truncate", "shutil.rmtree", "shutil.move", "shutil.copyfile"}
_installed = False


def install_guard(allowed_dirs) -> None:
    """Install the process-wide audit hook. Writes are allowed only below ``allowed_dirs`` (+ tempdir)."""
    global _installed
    if _installed:
        return
    roots = [os.path.normcase(str(Path(d).resolve())) for d in allowed_dirs if d]
    roots.append(os.path.normcase(str(Path(tempfile.gettempdir()).resolve())))

    def allowed(path) -> bool:
        try:
            if isinstance(path, int):
                return True              # already-open descriptor
            p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        except Exception:
            return False
        return any(p == r or p.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event in _DENY_EVENTS or event.startswith(("os.exec", "os.spawn", "os.posix_spawn")):
            raise SandboxViolation(f"sandbox: '{event}' is not permitted for agent-written code")
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writing and not allowed(path):
                raise SandboxViolation(f"sandbox: writing {path!r} is outside the job directory")
        elif event in _PATH_WRITE_EVENTS:
            for a in args[:2]:
                if isinstance(a, (str, bytes, os.PathLike)) and not allowed(a):
                    raise SandboxViolation(f"sandbox: '{event}' on {a!r} is outside the job directory")

    sys.addaudithook(hook)
    _installed = True
