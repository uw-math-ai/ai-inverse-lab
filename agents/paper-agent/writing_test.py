"""Generate a matched pair of Results drafts using two independent model calls."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6.2-sol"
TASK = (
    "Using only the supplied research material, write the English Results section "
    "of a paper about this entire run. Choose the findings yourself. Use Markdown "
    "and return only the section. Work directly from this prompt without using tools."
)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_api(prompt, *, model=MODEL):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the environment before running the test.")
    body = {"model": model, "input": prompt, "reasoning": {"effort": "medium"},
            "store": False}
    request = Request("https://api.openai.com/v1/responses",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}.") from error
    if result.get("status") != "completed":
        raise RuntimeError(f"Response did not complete: {result.get('status')}")
    parts = [part["text"] for item in result.get("output", []) if item.get("type") == "message"
             for part in item["content"] if part.get("type") == "output_text"]
    if not any(p.strip() for p in parts):
        raise RuntimeError("Response contained no draft.")
    return "\n".join(parts), {"usage": result.get("usage"), "model_returned": result.get("model"),
                              "response_id": result.get("id")}


def openrouter_model(env):
    return env.get("AGENT_MODEL") or env.get("OPENROUTER_MODEL") or "openrouter/free"


def generate_openrouter(prompt, *, env):
    key = env.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY locally or select a validated OpenRouter connection on the website.")
    base = env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    model = openrouter_model(env)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    request = Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"OpenRouter returned HTTP {error.code} (requested model {model}).") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"OpenRouter returned invalid JSON (requested model {model}).") from error

    choices = result.get("choices") or []
    details = f"model={result.get('model') or model!r}, response_id={result.get('id')!r}"
    if not choices:
        error = result.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        raise RuntimeError(f"OpenRouter returned no choices ({details}, error_code={code!r}).")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if reason != "stop":
        native_reason = choice.get("native_finish_reason")
        raise RuntimeError(
            f"OpenRouter did not return a complete draft ({details}, finish_reason={reason!r}, "
            f"native_finish_reason={native_reason!r})."
        )
    text = (choice.get("message") or {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"OpenRouter returned an empty draft ({details}, finish_reason={reason!r}).")
    return text, {"usage": result.get("usage"), "model_returned": result.get("model"),
                  "response_id": result.get("id"), "finish_reason": reason}


def generate_codex(prompt):
    # An empty working directory keeps repository files out of the writing context.
    with tempfile.TemporaryDirectory() as empty:
        command = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
            "--sandbox", "read-only", "--cd", empty, "--model", MODEL,
            "-c", 'model_reasoning_effort="medium"', "--json", "-",
        ]
        process = subprocess.run(command, input=prompt, capture_output=True,
                                 text=True, timeout=300,
                                 env={**os.environ, "CODEX_SQLITE_HOME": empty + "/state"})
    messages = []
    usage = None
    for line in process.stdout.splitlines():
        event = json.loads(line)
        if event["type"] in ("error", "turn.failed"):
            detail = event.get("message") or event.get("error")
            raise RuntimeError(str(detail))
        if event["type"] == "item.completed":
            item = event["item"]
            if item["type"] == "agent_message":
                messages.append(item["text"])
            elif item["type"] != "reasoning":
                raise RuntimeError(f"Unexpected item in text-only trial: {item['type']}")
        if event["type"] == "turn.completed":
            usage = event.get("usage", {})
    if process.returncode or not messages or usage is None:
        raise RuntimeError(f"Model call did not complete. {process.stderr[-1500:]}")
    return messages[-1], {"usage": usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "paper_input.json")
    parser.add_argument("--guidelines", type=Path, default=ROOT / "writing_guidelines.md")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--backend", choices=("api", "codex"), default="api")
    parser.add_argument("--draft", type=Path, help="Edit the same existing draft in both conditions")
    args = parser.parse_args()
    if args.backend == "api" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY or select --backend codex for an existing Codex login.")
    evidence = args.input.read_text(encoding="utf-8")
    rules, boundary, _ = args.guidelines.read_text(encoding="utf-8").partition("## Worked example")
    if not boundary or re.findall(r"^(\d+)\. ", rules, re.M) != [str(i) for i in range(1, 28)]:
        parser.error("Expected the 27-rule guide followed by its held-out example")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    task = TASK
    source_draft = None
    if args.draft is not None:
        source_draft = args.draft.read_text(encoding="utf-8")
        task = ("Edit the supplied English Results draft for clarity and concision, "
                "preserving its scientific meaning. Use only the supplied research material. "
                "Return the revised Results section in Markdown. Do not use tools.")
    base = f"{task}\n\nResearch material (JSON):\n{evidence}"
    if source_draft is not None:
        base += f"\n\nDraft to edit:\n{source_draft}"
    prompts = {"baseline": base, "guided": f"{base}\n\nWriting guidelines:\n{rules.strip()}"}
    record = {
        "run": json.loads(evidence)["run_id"]["value"], "model_requested": MODEL,
        "backend": args.backend, "reasoning_effort": "medium", "task": task,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(evidence), "rules_sha256": sha256(rules.strip()),
        "conditions": {},
    }
    if source_draft is not None:
        record["source_draft_sha256"] = sha256(source_draft)
    if args.backend == "codex":
        record["cli_version"] = subprocess.check_output(["codex", "--version"], text=True).strip()
    generate = generate_api if args.backend == "api" else generate_codex
    for name, prompt in prompts.items():
        print(f"Generating {name}...", flush=True)
        draft, metadata = generate(prompt)
        path = out / f"results_{name}.md"
        path.write_text(draft.rstrip() + "\n", encoding="utf-8")
        record["conditions"][name] = {"prompt_sha256": sha256(prompt), **metadata,
                                      "draft_sha256": sha256(path.read_text(encoding="utf-8"))}
        (out / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
