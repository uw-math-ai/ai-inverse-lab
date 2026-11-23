"""Generate an evidence-linked paper from one local harness run."""
import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from manuscript import make_manuscript_prompt, manuscript
from outline import literature_by_id, make_prompt, render, structure
from prepare import extract_run
from writing_test import MODEL, generate_api, sha256

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_pipeline(run_dir, output_parent, generate=generate_api, *, literature_path=None):
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
            stage("generate_" + name)
            raw, metadata = generate(request)
            record["calls"].append({"stage": name, "attempt": attempt, "prompt_sha256": sha256(request), **metadata})
            (out / f"{name}_response_{attempt}.txt").write_text(raw)
            stage("validate_" + name)
            try:
                return validate(raw)
            except (ValueError, TypeError, KeyError) as exc:
                record["calls"][-1]["validation_error"] = str(exc)
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

        outline, headings = generate_checked("outline", make_prompt(data, guide), validate_outline)
        (out / "outline.md").write_text(outline)
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
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY") and sys.stdin.isatty():
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key (input hidden): ").strip()
    print(f"Generating with {MODEL}", flush=True)
    try:
        out = run_pipeline(args.run_dir, args.output_dir, literature_path=args.literature)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Agent stopped: {exc}\n")
    print(f"Paper: {out / 'paper.md'}", flush=True)


if __name__ == "__main__":
    main()
