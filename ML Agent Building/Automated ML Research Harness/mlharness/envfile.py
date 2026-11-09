"""Minimal ``.env`` loader that never exposes secret values.

Values are read into ``os.environ`` (without overriding variables that are already set) so other
modules can look them up *by name*, e.g. ``SERVICE_USERNAME`` / ``SERVICE_PASSWORD`` for the AMATH
GPU servers. Nothing in this module logs, prints or returns a value; ``load_env_file`` returns only
the variable *names* it set.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import HARNESS_ROOT

SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "KEY", "CREDENTIAL", "AUTH")


def find_env_file(start: Path | None = None) -> Path | None:
    """Nearest ``.env`` walking up from ``start`` (default: the harness folder) to the filesystem root."""
    cur = Path(start or HARNESS_ROOT).resolve()
    for d in [cur, *cur.parents]:
        p = d / ".env"
        if p.is_file():
            return p
    return None


def load_env_file(path: str | Path | None = "auto", override: bool = False) -> list[str]:
    """Load KEY=VALUE lines into ``os.environ``. Returns the names that were set (never values)."""
    p = find_env_file() if path in (None, "auto") else Path(path)
    if p is None or not p.is_file():
        return []
    names = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if not key.replace("_", "").isalnum():
            continue
        if override or key not in os.environ:
            os.environ[key] = val
            names.append(key)
    return names


def is_secret_name(name: str) -> bool:
    up = name.upper()
    return any(m in up for m in SECRET_MARKERS) or up.startswith(("SERVICE_", "ANTHROPIC_", "OPENAI_", "AWS_"))


def scrubbed_env(base: dict | None = None, keep: tuple = ()) -> dict:
    """Copy of the environment without credentials, for running untrusted experiment code."""
    env = dict(os.environ if base is None else base)
    for k in list(env):
        if k not in keep and is_secret_name(k):
            env.pop(k)
    return env
