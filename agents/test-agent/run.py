"""Deterministic integration fixture; runs without packages or API charges."""
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_SECONDS = 3


def main():
    input_path = Path(os.environ["AGENT_INPUT_PATH"])
    output_dir = Path(os.environ["AGENT_OUTPUT_DIR"])
    inputs = json.loads(input_path.read_text(encoding="utf-8"))

    topic = inputs["topic"]
    # Schema defaults are not inserted by the platform, so apply them here.
    seconds = int(inputs.get("seconds", DEFAULT_SECONDS))
    seconds = max(1, min(seconds, 60))
    fail = bool(inputs.get("fail", False))

    print(f"Research demo started: {topic}", flush=True)
    print(json.dumps({"progress": 0.0}), flush=True)
    for step in range(seconds):
        time.sleep(1)
        print(json.dumps({"progress": (step + 1) / seconds}), flush=True)

    if fail:
        print("Requested demonstration failure", file=sys.stderr, flush=True)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(
        f"# Demo report\n\nTopic: {topic}\n\n"
        f"Simulated work: {seconds} second(s).\n\n"
        "Deterministic execution completed.\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps({"topic": topic, "seconds": seconds, "completed": True}, indent=2),
        encoding="utf-8",
    )
    print("Report and result saved.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
