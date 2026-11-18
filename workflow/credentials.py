"""Private local OpenAI configuration shared by the CLI and dashboard."""

import getpass
import os
from pathlib import Path
import sys
import tempfile
import warnings


KEY_FILE = Path(__file__).resolve().parent / ".secrets" / "openai.key"


def _validate_key(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 4096 or not value.isascii() or any(
        character.isspace() or not character.isprintable() for character in value
    ):
        raise ValueError("Enter a nonempty API key without spaces or control characters.")
    return value


def openai_api_key() -> str | None:
    """An explicitly saved project key takes precedence over the shell key."""
    if KEY_FILE.is_symlink() or KEY_FILE.parent.is_symlink():
        raise ValueError("The local credential path must not be a symbolic link.")
    try:
        with KEY_FILE.open(encoding="utf-8") as handle:
            return _validate_key(handle.read(4097))
    except FileNotFoundError:
        value = os.environ.get("OPENAI_API_KEY", "").strip()
        return _validate_key(value) if value else None


def openai_key_configured() -> bool:
    try:
        return bool(openai_api_key())
    except (OSError, ValueError):
        return False


def save_openai_key(value: str) -> None:
    """Atomically replace the local key using owner-only filesystem permissions."""
    value = _validate_key(value)
    if KEY_FILE.is_symlink() or KEY_FILE.parent.is_symlink():
        raise ValueError("The local credential path must not be a symbolic link.")
    KEY_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    KEY_FILE.parent.chmod(0o700)
    descriptor, filename = tempfile.mkstemp(prefix=".openai-", dir=KEY_FILE.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
        temporary.replace(KEY_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def check_openai_key(key: str, model: str) -> tuple[bool, str]:
    """Check model metadata only; never generate tokens or print provider errors."""
    from openai import OpenAI

    try:
        with OpenAI(api_key=key, timeout=20, max_retries=0) as client:
            client.models.retrieve(model)
    except Exception as exc:
        # Authentication errors can include credential fragments. Do not expose
        # raw exceptions, response bodies, headers, or SDK object representations.
        status = getattr(exc, "status_code", None)
        messages = {
            401: "OpenAI rejected this key (HTTP 401). Check that it is active and copied correctly.",
            403: "OpenAI denied the model-metadata check (HTTP 403). Check the key's project permissions.",
            404: "The selected model is unavailable to this key (HTTP 404). Check the model name and access.",
            429: "OpenAI rate-limited the check (HTTP 429). Try again later.",
        }
        return False, messages.get(status, "Unable to check OpenAI access. Check connectivity and API configuration, then retry.")
    return True, "OpenAI accepted the key and model-metadata request. No model generation was run."


def configure_openai(model: str) -> int:
    """Read a key without terminal echo, validate access, then save it privately."""
    if not sys.stdin.isatty():
        raise ValueError("Run configure-openai in an interactive terminal to enter the key privately.")
    print("Paste your OpenAI API key below. Input is hidden and is not added to shell history.", flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = _validate_key(getpass.getpass("OpenAI API key (hidden): "))
    except getpass.GetPassWarning:
        raise ValueError("This terminal cannot hide input. Use the VS Code terminal.") from None
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled; the saved key was not changed.")
        return 1
    print("Checking access to the selected model...", flush=True)
    success, message = check_openai_key(key, model)
    print(message)
    if not success:
        print("The saved key was not changed.")
        return 1
    save_openai_key(key)
    print("Saved privately for this project. Both the CLI and dashboard now use this key.")
    return 0
