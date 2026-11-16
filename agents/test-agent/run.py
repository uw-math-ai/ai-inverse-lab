"""Deterministic integration fixture; runs without packages or API charges."""
import json
import os
from pathlib import Path
import time

inputs = json.loads(Path(os.environ["AGENT_INPUT_PATH"]).read_text())
seconds = inputs.get("seconds", 3)
print(f"Research demo started: {inputs['topic']}", flush=True)
for step in range(seconds):
    time.sleep(1)
    print(json.dumps({"progress": (step + 1) / seconds}), flush=True)
if inputs.get("fail", False):
    raise RuntimeError("Requested demonstration failure")
output = Path(os.environ["AGENT_OUTPUT_DIR"])
(output / "report.md").write_text(f"# Demo report\n\nTopic: {inputs['topic']}\n\nDeterministic execution completed.\n")
(output / "result.json").write_text(json.dumps({"topic": inputs["topic"], "completed": True}))
print("Report and result saved.", flush=True)
