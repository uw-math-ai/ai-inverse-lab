# Instructions for coding agents: website-compatible research agents

Use this guide when adding or adapting agents in **`uw-math-ai/ai-inverse-lab`** for the Research Agent Workplace website:

<https://research-104-248-218-113.sslip.io>

The website discovers definitions from the root **`agents/`** folder on the configured GitHub branch, currently **`main`**. It downloads only that folder, snapshots it at an exact Git commit, and runs the selected entrypoint in an isolated Docker container. Adding agent code does not require redeploying the website.

This file can be copied to the root of `ai-inverse-lab` and given to a coding agent as its implementation instructions.

## 1. Required repository layout

```text
ai-inverse-lab/
├── instructions.md
├── agents/
│   ├── test-agent/
│   │   ├── agent.json
│   │   ├── run.py
│   │   └── README.md
│   ├── another-agent/
│   │   ├── agent.json
│   │   ├── run.py
│   │   └── data/
│   │       └── small-input.json
│   └── shared/
│       ├── __init__.py
│       └── helpers.py
└── ... existing research files remain here ...
```

- Each discoverable agent must have `agents/<slug>/agent.json`. Definitions nested deeper, such as `agents/group/my-agent/agent.json`, are not discovered.
- Use a lowercase slug containing letters, digits and hyphens, beginning with a letter or digit, with at most 63 characters. Example: `inverse-solver`.
- The folder name is the agent ID. The manifest's `name` is its display name. Do not add an `id` field to the manifest.
- Every agent needs a real, nonempty, valid `agent.json` and the files its command uses. Do not leave placeholder manifests in `agents/`: one invalid definition prevents the entire catalog refresh.
- Keep shared utilities in `agents/shared/`, without an `agent.json`, so it is not treated as a separate agent.
- Keep all required source files and small data files inside `agents/`. Files elsewhere in the repository are not downloaded or available at runtime. Do not import the existing research tree outside `agents/`.
- Use regular files and directories. Symlinks and Git submodules inside `agents/` are unsupported, including links to code elsewhere in the repository.
- Leave unrelated repository files and existing research results in place. Do not move the entire research repository into `agents/`.

## 2. Create a strict `agent.json`

Here is a complete starting manifest for `agents/test-agent/agent.json`:

```json
{
  "version": 1,
  "name": "Test research agent",
  "description": "Creates a report and JSON result for a supplied research topic.",
  "image": "research-agent-runtime:v1",
  "command": ["python", "run.py"],
  "inputs_schema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "minLength": 1,
        "maxLength": 1000
      }
    },
    "required": ["topic"],
    "additionalProperties": false
  },
  "connection": "none",
  "network": "none",
  "timeout_seconds": 180,
  "cpu": 0.5,
  "gpu": true,
  "memory_mb": 128
}
```

Use strict JSON: no comments, trailing commas, duplicate keys or unknown fields. A manifest must be at most 16 KiB.

| Field | Contract |
| --- | --- |
| `version` | Exactly `1`. |
| `name` | Required display name, 1–100 characters. |
| `description` | Optional description, up to 2,000 characters. |
| `image` | Required exact image name approved and already built on the website's worker. Use `research-agent-runtime:v1` for standard-library Python agents. |
| `command` | Required argv array, such as `["python", "run.py"]`. There is no implicit shell, so shell operators and variable expansion do not work inside an argument. |
| `inputs_schema` | Inline JSON Schema with root `"type": "object"`. Describe every supported user input. `$ref` and `$dynamicRef` are unsupported. |
| `connection` | `"none"`, `"optional"` or `"required"`. Use `"required"` when the agent needs an OpenAI API connection selected in the website. |
| `network` | `"none"` or `"openai"`. The latter permits the provided OpenAI API gateway, not arbitrary internet access. |
| `timeout_seconds` | Integer from 1 to 86,400. Default: 3,600. |
| `cpu` | Number from 0.1 to 4. Default: 1. The worker may enforce a lower ceiling. |
| `memory_mb` | Integer from 128 to 4,096. Default: 512. The worker may enforce a lower ceiling. |

Use conservative resources for the small VPS. Do not assume that declaring a larger CPU or memory value increases the server's available resources. Input-schema defaults are not automatically inserted; supply optional defaults in the program itself.

## 3. Implement the entrypoint protocol

The command runs with working directory **`/agent/agents/<slug>`**. For the manifest above, create `agents/test-agent/run.py`:

```python
"""Minimal website-compatible agent; uses only Python's standard library."""
import json
import os
from pathlib import Path


def main():
    input_path = Path(os.environ["AGENT_INPUT_PATH"])
    output_dir = Path(os.environ["AGENT_OUTPUT_DIR"])
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    topic = inputs["topic"]

    print("Research task started", flush=True)
    print(json.dumps({"progress": 0.5}), flush=True)

    # Replace this section with the agent's actual research logic.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(
        f"# Research report\n\nTopic: {topic}\n\nExecution completed.\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps({"topic": topic, "completed": True}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"progress": 1.0}), flush=True)
    print("Report and result saved", flush=True)


if __name__ == "__main__":
    main()
```

Follow these rules when implementing real logic:

- Read the JSON input from `AGENT_INPUT_PATH`. Do not prompt with `input()` or require an interactive terminal.
- Use the manifest's input schema to declare required inputs and allowed values. Input JSON is limited to 64 KiB.
- Write downloadable reports, plots, structured results and other artifacts under `AGENT_OUTPUT_DIR`, normally `/output`. Files written to other directories are not collected as artifacts.
- Write regular files with ordinary filenames. Do not create artifact symlinks or include credentials in output files or filenames.
- Use stdout/stderr for logs, with flushing where useful. A JSON line such as `{"progress": 0.5}` reports progress between 0 and 1.
- Exit with code zero on success and a nonzero code on failure. Do not hide an unsuccessful task behind a successful exit code.
- Treat the source tree as read-only. Use `/tmp` or `/work` for temporary work, then save final outputs under `AGENT_OUTPUT_DIR`. Temporary work is discarded after the job.
- Runs have no access to the host filesystem, Docker socket, website database or GitHub credentials. They continue after the user's browser closes.

To import a shared helper, add the parent `agents/` directory to Python's import path explicitly:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.helpers import your_helper
```

Include `agents/shared/__init__.py` and implement `your_helper` in `agents/shared/helpers.py`. Do not add a path to a repository directory outside `agents/`.

## 4. Dependencies and network access

The default `research-agent-runtime:v1` image contains **Python 3.12 and its standard library**. Do not assume that NumPy, SciPy, PyTorch, the OpenAI SDK, Node.js or a Codex CLI is installed.

A `requirements.txt`, `pyproject.toml` or Dockerfile committed with an agent is not automatically installed or built. Do not run `pip install` during a job: ordinary internet access is unavailable. For additional packages or tools, the website operator must build a runtime image with those dependencies, install it on the worker, and add its exact image reference to the server's `APPROVED_IMAGES` configuration before the manifest uses it. Document the dependencies needed by the operator.

Use `"network": "none"` for agents that do not make OpenAI requests. External websites, GitHub downloads, package registries and other APIs are unavailable from jobs. Bundle required small inputs inside `agents/` or accept them through declared JSON inputs.

## 5. Agents that call OpenAI

For an agent that requires OpenAI, change these manifest values:

```json
"connection": "required",
"network": "openai"
```

These are fields in the full manifest, not a standalone manifest. The user selects an existing connection when starting the job. The worker supplies:

| Environment variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Credential from the selected connection. |
| `OPENAI_MODEL` | Model configured for that connection. |
| `OPENAI_BASE_URL` | Gateway base URL, including `/v1`, provided when `network` is `openai`. |

Use the injected base URL. For example, the Responses API URL is:

```python
url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/responses"
```

Use `OPENAI_MODEL` in the request and authenticate with `OPENAI_API_KEY`. Python's standard-library `urllib.request` is available; an SDK requires a custom runtime image containing it. Do not hard-code `https://api.openai.com` as the request destination: direct internet requests are blocked. The gateway forwards permitted `/v1/` API requests to OpenAI.

Never commit, print or save API keys. Do not print authorization headers or raw exception bodies that might contain sensitive request data. Do not put API keys in website input fields, command arguments or `agent.json`. ChatGPT/Codex subscription sign-in and browser session tokens are not supported by this platform. API-backed jobs can incur charges; use a no-API smoke test before making a paid request.

## 6. Source and output limits

Keep `agents/` small and focused on executable agent code:

- At most 64 MiB of source contents across the entire `agents/` subtree.
- At most 16 MiB per source file and 10,000 source entries.
- At most 100 discovered agent definitions.
- At most 16 KiB per manifest.
- Large files outside `agents/` do not count toward agent source limits and are not downloaded.

Keep generated research history, large datasets, model weights, virtual environments, dependency caches, Git metadata and output directories out of `agents/`. Source limits apply to all agents together, not just the selected one.

The deployment also bounds logs, artifacts, CPU, memory and execution time. Current deployment targets for outputs are 1 MiB of logs, 50 MiB of artifacts and 100 artifact files per run; operators can change these settings. Do not rely on temporary files surviving a failed, cancelled or timed-out job.

## 7. Verify before handing off

1. Confirm every `agents/*/agent.json` is nonempty, valid strict JSON and uses only supported fields. Do not leave an unfinished agent folder containing an invalid manifest.
2. Confirm every declared command and required source file exists inside `agents/`, and that imports work without the rest of the repository.
3. Test the entrypoint with sample JSON input and `AGENT_INPUT_PATH`/`AGENT_OUTPUT_DIR` set to local test paths. Run from the agent's own directory. A local test checks the input/output protocol; a website run checks compatibility with the actual container and network limits.
4. Confirm success produces readable logs and regular artifact files, and that invalid input or failed work produces a nonzero exit code.
5. Confirm any dependencies are available in the approved runtime image. A successful run in a developer's richer local environment is insufficient.
6. Keep tests free of paid model requests unless the user explicitly requests and authorizes such a run.
7. Include a short README in each agent folder documenting its purpose, expected inputs, sample input JSON, outputs, dependencies and whether it incurs API charges.

## 8. Make the agent appear on the website

1. Commit the agent files to `uw-math-ai/ai-inverse-lab` and push or merge them into the configured branch, currently `main`, using the repository's normal contribution permissions.
2. Log in to the website, open **Agent catalog**, and click **Refresh catalog**. GitHub branch protection and pull-request approvals are not required by the deployed app. The app can read this public repository without a GitHub token; private repositories need read access configured on the server.
3. Confirm the agent appears with the expected name, input form and Git commit.
4. Start a no-API smoke test. Inspect its status, logs, progress and downloadable artifacts. For OpenAI agents, select an authorized connection and run a paid test only when requested.

No website redeployment is needed for agent-code or manifest changes. A configured push webhook can also refresh the catalog, but do not assume one exists; the manual refresh is the reliable handoff step. Existing runs retain their original source snapshot, and a retry starts a new run using the current catalog revision.

When reporting completion, state which agent folders changed, give sample input JSON, list produced artifacts, and explain which checks ran. Distinguish local validation from a successful live website run. Never claim an agent is published solely because its files exist locally.
