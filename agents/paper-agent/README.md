# Research paper from a harness run

Generate an English manuscript from one completed ML Research Harness run. The agent extracts its evidence, generates an outline and manuscript, and validates the reported measurements.

## Launch on the research website

Select a shared dataset containing the raw run files and an **OpenRouter** or **OpenAI API** connection with the **sol** tier or higher. For the lab's OpenRouter subscription, select **UW Math AI Lab Open Router Subscription**. The agent uses the connection's configured model, falling back to `openrouter/free` on OpenRouter if no model is set. Model charges may apply. The default VPS worker is sufficient for the tested runs. The agent uses Python's standard library and runs on one worker.

Enter the dataset name and, if needed, the path within it to the run. For a dataset named graph-demo with a subfolder named graph_fpt_weights_claude, the website input is:

```json
{"run_folder": "graph-demo/graph_fpt_weights_claude"}
```

If the run files are at the dataset root, enter just the dataset name. Keep the original file names and relative paths.

| Harness format | Required files in the selected folder |
| --- | --- |
| V1 | `spec.json`, `status.json`, `artifacts.jsonl`, `final/manifest.json` |
| V2 (shared V1 log format) | `spec.json`, `status.json`, `artifacts.jsonl`, `final/manifest.json` |
| V3 | `spec.json`, `evidence_bundle.json`, `program_evaluation.json` |

For V2, evidence extraction and prompt preparation were checked on 17 archived heat runs. The reader uses the baseline, selection, final checkpoint, and audit records shared with V1. It does not read the separate V2 research trace. Full manuscript generation on V2 has yet to be verified.

Download `paper.md` for the reader-facing manuscript, `outline.md` for the evidence-linked outline, and `run.json` for the generation and validation record.
