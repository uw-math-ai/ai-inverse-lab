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

## Debug locally

From the repository root, start with an outline-only run on the same raw harness folder.
Use a local OpenRouter API key. If `OPENROUTER_API_KEY` is unset, the terminal
prompts for it with input hidden. Use `--model` to match a model configured on
the website. The default `openrouter/free` may route different requests to
different models.

```sh
python3 agents/paper-agent/run.py \
  --run-dir /path/to/graph_fpt_weights_claude \
  --output-dir /tmp/paper-agent-debug \
  --provider openrouter --outline-only
```

The command prints a new output directory. Its `run.json` records each attempt,
validation errors, and the stage where the run stopped. When OpenRouter rejects
or stops a draft, the error records the available HTTP status or response
details, including the returned model and finish reason where present.
Completed model responses are saved as `outline_response_1.txt`,
`outline_response_2.txt`, and their `paper_response_*.txt` counterparts.

Reuse the saved evidence and response to test changes to the outline validator
without calling a model. Set `RUN_OUTPUT` to the directory printed by the command.

```sh
python3 agents/paper-agent/outline.py "$RUN_OUTPUT/paper_input.json" \
  --response "$RUN_OUTPUT/outline_response_1.txt"
```

Remove `--outline-only` for a local run that also generates and validates the manuscript.
Pass `--outline-response "$RUN_OUTPUT/outline_response_1.txt"` to `run.py` to
revalidate a saved outline and continue to manuscript generation without
calling the model for the outline again. Combine it with `--outline-only`
to validate the saved response without making any model calls.
