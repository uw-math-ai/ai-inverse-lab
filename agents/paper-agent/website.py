"""Website entrypoint; the local run.py CLI remains available."""
import json
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from run import run_pipeline, write_json
from writing_test import MODEL, generate_api, generate_openrouter


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
