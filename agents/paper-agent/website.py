"""Website entrypoint; the local run.py CLI remains available."""
import json
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from run import run_pipeline, write_json
from writing_test import MODEL, generate_api


def generate_openrouter(prompt, *, env):
    key = env.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Select a validated OpenRouter connection.")
    base = env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    model = env.get("AGENT_MODEL") or env.get("OPENROUTER_MODEL") or "openrouter/free"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    request = Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"OpenRouter returned HTTP {error.code}.") from error
    choices = result.get("choices")
    if not choices or choices[0].get("finish_reason") != "stop":
        raise RuntimeError("OpenRouter did not return a complete draft.")
    text = choices[0]["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenRouter returned an empty draft.")
    return text, {"usage": result.get("usage"), "model_returned": result.get("model"),
                  "response_id": result.get("id")}


def run_job(env, generate=None):
    if not env.get("AGENT_DATASETS_DIR"):
        raise ValueError("Select a dataset containing a complete harness run.")
    root = Path(env["AGENT_DATASETS_DIR"]).resolve(strict=True)
    inputs = json.loads(Path(env["AGENT_INPUT_PATH"]).read_text())
    relative = Path(inputs["run_folder"])
    folder = (root / relative).resolve()
    if relative.is_absolute() or root not in folder.parents or not folder.is_dir():
        raise ValueError("Choose a run directory inside the mounted datasets.")
    if generate is None:
        if env.get("AGENT_PROVIDER") == "openrouter":
            generate = partial(generate_openrouter, env=env)
        else:
            model = env.get("AGENT_MODEL") or env.get("OPENAI_MODEL") or MODEL
            generate = partial(generate_api, model=model)
    total_tokens = 0
    def generate_with_usage(prompt):
        nonlocal total_tokens
        response, metadata = generate(prompt)
        count = (metadata.get("usage") or {}).get("total_tokens")
        if env.get("AGENT_USAGE_FILE") and isinstance(count, int):
            total_tokens += count
            write_json(Path(env["AGENT_USAGE_FILE"]), {"tokens": total_tokens})
        return response, metadata

    output = Path(env["AGENT_OUTPUT_DIR"])
    print('{"progress": 0.05}', flush=True)
    with TemporaryDirectory() as scratch:
        result = run_pipeline(folder, Path(scratch), generate_with_usage)
        for filename in ("paper.md", "outline.md", "run.json"):
            copyfile(result / filename, output / filename)
    print('{"progress": 1.0}', flush=True)


if __name__ == "__main__":
    run_job(os.environ)
