"""Generate an evidence-linked paper from one local harness run."""
import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

from manuscript import make_manuscript_prompt, manuscript
from outline import literature_by_id, make_prompt, render, structure
from prepare import extract_run
from writing_test import MODEL, generate_api, generate_openrouter, openrouter_model, sha256

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_pipeline(run_dir, output_parent, generate=generate_api, *, literature_path=None,
                 outline_only=False, outline_response=None):
    output_parent.mkdir(parents=True, exist_ok=True)
    out = Path(mkdtemp(prefix=datetime.now().strftime("paper_%Y%m%d_%H%M%S_"), dir=output_parent))
    record = {"status": "running", "source": str(run_dir.resolve()), "calls": []}
    print(f"Output: {out}", flush=True)

    def stage(name):
        record["stage"] = name
        write_json(out / "run.json", record)
        print(name, flush=True)

    def generate_checked(name, prompt, validate):
        # One repair uses the same evidence and validation rules as the first attempt.
        request = prompt
        for attempt in (1, 2):
            call = {"stage": name, "attempt": attempt, "prompt_sha256": sha256(request),
                    "prompt_characters": len(request)}
            record["calls"].append(call)
            stage("generate_" + name)
            raw, metadata = generate(request)
            call.update(metadata)
            (out / f"{name}_response_{attempt}.txt").write_text(raw)
            stage("validate_" + name)
            try:
                return validate(raw)
            except (ValueError, TypeError, KeyError) as exc:
                call["validation_error"] = str(exc)
                print(f"{name} attempt {attempt} failed validation: {exc}", file=sys.stderr, flush=True)
                if attempt == 2:
                    raise
                request = (prompt + "\n\nPrevious JSON response:\n" + raw
                           + "\n\nValidation feedback:\n" + str(exc)
                           + "\nCorrect the response using only the supplied evidence. Return the full JSON object.")

    try:
        stage("extract")
        data = extract_run(run_dir)
        path = literature_path if literature_path is not None else ROOT / "literature.json"
        data["literature"] = list(literature_by_id(json.loads(path.read_text())).values())
        record["run_id"] = data["run_id"]["value"]
        write_json(out / "paper_input.json", data)
        guide = (ROOT / "writing_guidelines.md").read_text().split("## Worked example", 1)[0]
        def validate_outline(raw):
            return render(raw, data), structure(json.loads(raw)["sections"])

        if outline_response is None:
            outline, headings = generate_checked("outline", make_prompt(data, guide), validate_outline)
        else:
            record["outline_response_source"] = str(outline_response.resolve())
            stage("validate_outline")
            raw = outline_response.read_text(encoding="utf-8")
            (out / "outline_response_1.txt").write_text(raw, encoding="utf-8")
            outline, headings = validate_outline(raw)
        (out / "outline.md").write_text(outline)
        if outline_only:
            record.update(stage="complete", status="completed")
            return out
        catalog, prompt = make_manuscript_prompt(data, outline, guide)

        prompt += "\nReviewed structure (level, heading, roles)\n" + json.dumps(headings)

        def validate_paper(raw):
            if structure(json.loads(raw)["sections"]) != headings:
                raise ValueError("Restore the reviewed outline's headings, levels, roles and order.")
            paper, checks = manuscript(raw, data, catalog)
            record["checks"] = checks
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError("Manuscript checks to correct: " + ", ".join(failed))
            return paper

        paper = generate_checked("paper", prompt, validate_paper)
        (out / "paper.md").write_text(paper)
        record.update(stage="complete", status="completed")
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(out / "run.json", record)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Raw V3 bundle or V1 artifact-log directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "writing_test" / "end_to_end",
                        help="Parent directory for a new, separate output per invocation")
    parser.add_argument("--literature", type=Path, help="JSON array of checked sources (id, title, url, note); replaces the supplied starter set")
    parser.add_argument("--provider", choices=("openai", "openrouter"), default="openai")
    parser.add_argument("--model", help="Override the provider's model for this run")
    parser.add_argument("--outline-only", action="store_true", help="Stop after generating and validating the outline")
    parser.add_argument("--outline-response", type=Path, help="Validate a saved outline response and skip outline generation")
    args = parser.parse_args()
    needs_model = not (args.outline_only and args.outline_response is not None)
    if args.provider == "openrouter":
        env = dict(os.environ)
        if needs_model and not env.get("OPENROUTER_API_KEY") and sys.stdin.isatty():
            env["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter API key (input hidden): ").strip()
        if args.model:
            env["AGENT_MODEL"] = args.model
        model = openrouter_model(env)
        generate = partial(generate_openrouter, env=env)
    else:
        if needs_model and not os.environ.get("OPENAI_API_KEY") and sys.stdin.isatty():
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key (input hidden): ").strip()
        model = args.model or MODEL
        generate = partial(generate_api, model=model)
    if needs_model:
        print(f"Generating with {model} via {args.provider}", flush=True)
    try:
        out = run_pipeline(args.run_dir, args.output_dir, generate,
                           literature_path=args.literature, outline_only=args.outline_only,
                           outline_response=args.outline_response)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Agent stopped: {exc}\n")
    artifact = "outline.md" if args.outline_only else "paper.md"
    print(f"Result: {out / artifact}", flush=True)


if __name__ == "__main__":
    main()
