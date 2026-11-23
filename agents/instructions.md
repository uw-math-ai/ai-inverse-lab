# Instructions for coding agents: website-compatible research agents

Use this guide when adding or changing agents in **`uw-math-ai/ai-inverse-lab`** for the Research Agent Workplace website:

<https://research-104-248-218-113.sslip.io>

The website discovers agents in the root **`agents/`** folder of the configured GitHub branch, currently **`main`**. It downloads only that folder, snapshots it at an exact Git commit, and runs the selected entrypoint in an isolated Docker container. Adding or changing agent code does not require redeploying the website.

What an agent can use, and the manifest setting that enables each feature:

| Feature | Enable with | Section |
| --- | --- | --- |
| Input form generated from a JSON Schema | `inputs_schema` | 2 |
| Live logs, progress bar, downloadable artifacts | always on | 3 |
| Model providers: Codex, Claude Code, OpenRouter, OpenAI API | `connection`, `network`, `providers` | 5 |
| Internet access (web APIs, papers, packages, datasets) | `"network": "internet"` | 5 |
| Automatic choice of the cheapest suitable plan | `tier` | 6 |
| Usage reporting and handoff notes when a plan runs low | always on, via `agents/shared/platform_usage.py` | 6 |
| Shared datasets from the lab OneDrive folder, mounted read-only | `datasets` | 7 |
| Subagents: up to 20 child containers on other servers and accounts | `swarm`, via `agents/shared/platform_swarm.py` | 8 |
| Running on an AMATH worker server instead of the VPS | chosen by the researcher at launch | 9 |
| NVIDIA GPUs (americano) | `gpus` and the `research-agent-runtime-gpu:v1` image | 4, 9 |

## 1. Repository layout

```text
ai-inverse-lab/
├── agents/
│   ├── instructions.md          ← this file (loose files here are ignored by discovery)
│   ├── test-agent/
│   │   ├── agent.json
│   │   ├── run.py
│   │   └── README.md
│   ├── inverse-solver/
│   │   ├── agent.json
│   │   ├── run.py
│   │   ├── worker.py            ← extra entrypoint, e.g. for subagents
│   │   ├── README.md
│   │   └── data/
│   │       └── small-input.json
│   └── shared/                  ← no agent.json, so it is not listed as an agent
│       ├── platform_usage.py    ← usage reporting and handoff helper
│       ├── platform_swarm.py    ← subagent helper
│       └── helpers.py           ← your own shared code
└── ... existing research files stay where they are ...
```

- Each agent must have `agents/<slug>/agent.json`. Deeper definitions, such as `agents/group/my-agent/agent.json`, are not discovered.
- The slug (folder name) is lowercase letters, digits and hyphens, starts with a letter or digit, and is at most 63 characters (`^[a-z0-9][a-z0-9-]{0,62}$`). Example: `inverse-solver`. Underscores and capitals are rejected.
- The folder name is the agent ID. The manifest's `name` is its display name. Do not add an `id` field.
- **One bad definition fails the whole catalog refresh**, and the previous catalog is kept. This includes an invalid or empty `agent.json`, a slug with illegal characters, or an `image` that is not approved. Do not leave placeholder or half-written manifests in `agents/`.
- Folders without `agent.json` (such as `agents/shared/`) are not listed in the catalog. They are still part of the snapshot, so agents can import from them, and a subagent can run code from them (section 8).
- Keep all source files and small data files inside `agents/`. Nothing else in the repository is downloaded or available at runtime. Do not import the research tree outside `agents/`.
- Use regular files and directories. Symlinks and Git submodules inside `agents/` are rejected.
- Leave unrelated repository files and existing research results in place. Do not move the research repository into `agents/`.

## 2. `agent.json`

A complete starting manifest for an agent with no model calls (`agents/test-agent/agent.json`):

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
        "maxLength": 1000,
        "title": "Topic",
        "description": "Research topic to include in the report."
      }
    },
    "required": ["topic"],
    "additionalProperties": false
  },
  "connection": "none",
  "network": "none",
  "timeout_seconds": 180,
  "cpu": 0.5,
  "memory_mb": 128
}
```

Use strict JSON: no comments, trailing commas, duplicate keys or unknown fields (an unknown field such as `"gpu"` instead of `"gpus"` invalidates the manifest). Types are strict too: write `"timeout_seconds": 180`, not `"180"`. A manifest is at most 16 KiB.

| Field | Required | Contract |
| --- | --- | --- |
| `version` | yes | Exactly `1`. |
| `name` | yes | Display name, 1–100 characters. |
| `description` | no | Up to 2,000 characters. Say what the agent does and whether it incurs model charges. |
| `image` | yes | An operator-approved image already present on the worker: `research-agent-runtime:v1`, or `research-agent-runtime-gpu:v1` for GPU agents (section 4). |
| `command` | yes | Argv array, 1–32 items, such as `["python", "run.py"]`. No shell: pipes, `&&` and `$VAR` do not work inside an argument. |
| `inputs_schema` | no | Inline JSON Schema (2020-12) with root `"type": "object"`, at most 16 KiB and 20 levels deep. `$ref` and `$dynamicRef` are rejected. Default: an object with no inputs. |
| `connection` | no | `"none"` (default), `"optional"` or `"required"`: whether the run takes a provider credential (section 5). |
| `network` | no | `"none"` (default), `"internet"`, `"provider"` or the legacy `"openai"` (section 5). `"internet"` reaches any public website and also the model providers. `"provider"` requires `"connection": "required"`. |
| `providers` | no | 1–4 of `"openai"`, `"codex"`, `"claude"`, `"openrouter"`. Restricts which saved credentials can be chosen. Omit to accept any compatible one. |
| `tier` | no | Plan tier the model router must satisfy, currently `"sol"` or `"astra"` (section 6). |
| `timeout_seconds` | no | Integer, 1–86,400. Default 3,600. |
| `cpu` | no | Number, 0.1–256. Omit it (recommended) to let the job use every CPU core on its server. Set it only to cap the job deliberately. |
| `memory_mb` | no | Integer, 128–4,096. Default 512. |
| `gpus` | no | Integer, 0–8. Default 0. NVIDIA GPUs the job needs. A nonzero value runs only on a worker server with GPUs (section 9). |
| `datasets` | no | Up to 16 unique dataset folder names to mount read-only (section 7). Default: none. |
| `swarm` | no | `{"max_children": 1–20, "max_total": 1–500}` lets the agent start subagents (section 8). Absent means no subagents. |

Notes:

- Jobs have **no CPU cap** unless the manifest sets `cpu`: they can use every core of the server they run on (24 on americano). They run at a slightly lower scheduling priority than the website itself, which only matters when the CPU is busy. Memory is still capped at **2,048 MiB** per job by default, whatever the manifest says. The VPS has only 1 CPU and 2 GB and runs at most 2 jobs at once, so run heavy work on an AMATH worker server (section 9).
- Inputs must be finite JSON, at most 64 KiB, and pass the schema before the run is queued. Use `"additionalProperties": false` to reject unexpected inputs.
- Schema `default` values are **not** inserted into the input file. Apply defaults in your code (`inputs.get("seconds", 3)`).
- `title` and `description` on each property label the website's form.

## 3. Entrypoint protocol

The command runs as an unprivileged user with working directory **`/agent/agents/<slug>`**, on a read-only root filesystem. Minimal `agents/test-agent/run.py`:

```python
"""Minimal website-compatible agent; uses only Python's standard library."""
import json
import os
from pathlib import Path


def main():
    inputs = json.loads(Path(os.environ["AGENT_INPUT_PATH"]).read_text(encoding="utf-8"))
    output_dir = Path(os.environ["AGENT_OUTPUT_DIR"])
    topic = inputs["topic"]

    print("Research task started", flush=True)
    print(json.dumps({"progress": 0.5}), flush=True)

    # Replace this section with the agent's actual research logic.
    (output_dir / "report.md").write_text(
        f"# Research report\n\nTopic: {topic}\n\nExecution completed.\n", encoding="utf-8")
    (output_dir / "result.json").write_text(
        json.dumps({"topic": topic, "completed": True}, indent=2), encoding="utf-8")

    print(json.dumps({"progress": 1.0}), flush=True)
    print("Report and result saved", flush=True)


if __name__ == "__main__":
    main()
```

### Filesystem inside the job

| Path | Access | Contents and limits |
| --- | --- | --- |
| `/agent/agents/` | read-only | The `agents/` snapshot at the run's commit. Your code lives here. |
| `/run/inputs.json` (`AGENT_INPUT_PATH`) | read-only | The researcher's input JSON. |
| `/run/datasets/<name>/` (`AGENT_DATASETS_DIR`) | read-only | Selected shared datasets (section 7). |
| `/run/parent-files/` (`AGENT_PARENT_FILES_DIR`) | read-only | Files a parent handed to this subagent (section 8). |
| `/output` (`AGENT_OUTPUT_DIR`) | writable | Collected as downloadable artifacts. In memory, 50 MiB. |
| `/work` | writable | Scratch space, 64 MiB, in memory. Discarded after the job. |
| `/tmp` | writable | Scratch space, 64 MiB, in memory, no executables. `HOME` is `/tmp/home`. |
| `/swarm/results/` (`AGENT_SWARM_RESULTS`) | read-only | Finished subagents' results (section 8, swarm agents only). |

Scratch space is small and in memory. Process large data in streams or chunks instead of writing big intermediate files.

### Environment variables every job gets

| Variable | Meaning |
| --- | --- |
| `AGENT_INPUT_PATH` | `/run/inputs.json` |
| `AGENT_OUTPUT_DIR` | `/output` |
| `AGENT_USAGE_FILE`, `AGENT_HANDOFF_SIGNAL`, `AGENT_HANDOFF_FILE`, `AGENT_HANDOFF_THRESHOLD` | Usage reporting and handoff (section 6) |

Further variables appear for provider credentials (section 5), plan tiers (6), datasets (7) and swarms (8).

### Rules

- Read inputs from `AGENT_INPUT_PATH`. Never use `input()` or expect a terminal.
- Write reports, plots and structured results under `AGENT_OUTPUT_DIR`. Files elsewhere are not collected. Write regular files with ordinary names: no symlinks, and no credentials in contents or filenames.
- Log to stdout/stderr with `flush=True`. A line that is exactly a JSON object like `{"progress": 0.5}` (a number from 0 to 1) sets the progress bar.
- Exit 0 on success and nonzero on failure. Do not hide a failed task behind exit code 0.
- Jobs have no host filesystem, Docker socket, website database or GitHub credential. They reach the internet only with `"network": "internet"` (section 5). They keep running after the researcher closes the browser. A researcher can cancel a run at any time, which kills the container immediately.

### Shared code

Put shared modules in `agents/shared/` and add that folder to the import path:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import platform_usage            # agents/shared/platform_usage.py
from helpers import your_helper  # agents/shared/helpers.py
```

Do not add any path outside `agents/`.

**Platform helpers.** `agents/shared/platform_usage.py` (section 6) and `agents/shared/platform_swarm.py` (section 8) are copies of the website's own helpers (`examples/agents/shared/` in the platform repository, `Kothapalli-Kushal/research_agent_workplace_web`). Do not edit or rename them; replace them only with newer copies from the platform. Both use only the standard library. Outside the platform, `platform_usage` does nothing and `platform_swarm.available()` returns `False`, so local tests still work.

## 4. Runtime image and dependencies

There are two approved images.

**`research-agent-runtime:v1`** (the default, on the VPS and every worker server) contains:

- Python 3.12 and its standard library;
- NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1 (with joblib and threadpoolctl) and PyYAML 6.0.3;
- the `codex` CLI (OpenAI Codex, pinned 0.169.0) and the `claude` CLI (Claude Code, pinned 2.1.326);
- `rg` (ripgrep).

**`research-agent-runtime-gpu:v1`** (GPU worker servers only, currently americano) contains everything above plus PyTorch 2.14.1 built for CUDA 12.6. Use it with `"gpus": 1` (or more) in the manifest:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # keep a CPU fallback for local tests
```

Inside a GPU job, the assigned GPUs appear as CUDA devices `0..N-1`, and `AGENT_GPUS` holds N. Each GPU job gets whole GPUs to itself. Without a `cpu` cap, NumPy, scikit-learn and PyTorch use every core. If you do set `cpu`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are set to it.

Neither image includes pandas, matplotlib, the OpenAI or Anthropic Python SDKs, or Node.js for your own scripts. A `requirements.txt`, `pyproject.toml` or Dockerfile next to an agent is not installed or built. With `"network": "internet"`, a job can download files, model weights and packages, but the root filesystem is read-only and the only writable scratch is `/work` and `/tmp`, 64 MiB each and in memory. So `pip install --target /work/pkgs <small-package>` works for small pure-Python packages only, and large downloads must be streamed or processed in chunks. Ask the operator to add any package you need regularly to the image instead.

For more packages, the website operator must rebuild the runtime image, load it on the VPS and every worker server, and list it in `APPROVED_IMAGES`. Only then can a manifest name it. Until then, use what the images contain, and document any other dependencies in the agent's README for the operator. A manifest naming an unapproved image breaks the whole catalog refresh.

Bundle small inputs inside `agents/`, take them as JSON inputs, or use shared datasets (section 7). External websites, GitHub, package registries and other APIs are reachable only with `"network": "internet"` (section 5).

## 5. Model providers

A researcher saves provider credentials ("connections") on the website and picks one, or **Automatic**, when launching a run. Agent code never stores a key.

| Provider | Credential kinds the website accepts | Allowed hosts on the `provider` network |
| --- | --- | --- |
| `codex` | OpenAI API key, or a ChatGPT/Codex CLI sign-in file (`auth.json`) | `api.openai.com`, `chatgpt.com`, `auth.openai.com` |
| `claude` | Anthropic API key, or a Claude Code token from `claude setup-token` | `api.anthropic.com`, `claude.ai`, `platform.claude.com` |
| `openrouter` | OpenRouter API key | `openrouter.ai` |
| `openai` | OpenAI API key | `api.openai.com` |

### Manifest combinations

| Goal | `connection` | `network` | Notes |
| --- | --- | --- | --- |
| No model calls | `"none"` | `"none"` | Default. |
| Codex CLI, Claude Code CLI, OpenRouter, or the OpenAI API over HTTPS | `"required"` | `"provider"` | Recommended when the agent needs nothing but the model. Add `providers` to restrict the choice. |
| Internet access, with or without a model provider (web search APIs, arXiv, GitHub, package and dataset downloads) | `"none"`, `"optional"` or `"required"` | `"internet"` | Reaches any public host, including the providers above, so model calls work too. |
| OpenAI API key through the plain-HTTP gateway | `"required"` | `"openai"` | Legacy. Only `openai` API-key connections qualify. |
| Swarm parent with no model calls of its own, whose subagents reuse the chosen account | `"optional"` | `"none"` | The credential variables are set, but the parent cannot reach the provider. |

Example for an agent that works with any of the three CLI-style providers:

```json
"connection": "required",
"network": "provider",
"providers": ["codex", "claude", "openrouter"],
"tier": "sol"
```

On the `provider` network, the job gets `HTTPS_PROXY`/`HTTP_PROXY` set to a gateway that reaches **only the selected provider's hosts**. Python's `urllib.request`, `codex` and `claude` use the proxy automatically. A Codex sign-in file (`auth.json`) works only on the `provider` and `internet` networks.

### Internet access (`"network": "internet"`)

The job gets `HTTPS_PROXY`/`HTTP_PROXY` (and lowercase variants) set to `http://egress:8080`, an internet gateway that connects to **any public host and port** except SMTP (25), and forwards plain `http://` requests too. Python's `urllib.request` and `requests`, `pip`, `git` over HTTPS, `curl`, `codex` and `claude` all use the proxy automatically:

```python
import json, urllib.request
with urllib.request.urlopen("https://export.arxiv.org/api/query?search_query=all:inverse+problems&max_results=5", timeout=60) as response:
    feed = response.read().decode()
```

Rules and limits:

- Always go through the proxy. The job has no direct route out and cannot resolve external hostnames itself. The gateway resolves them, so clients that pass the hostname to the proxy (all of those above) work. Code that calls `socket.getaddrinfo` or opens raw sockets does not. Neither do SSH, FTP or other non-HTTP protocols, so use HTTPS URLs (for example `https://github.com/...` rather than `git@github.com:...`).
- Private, university-LAN, loopback, Tailscale and cloud-metadata addresses are always refused. A job cannot reach the VPS, the worker server it runs on, or other lab machines, even by IP address.
- Each tunnel is closed after 6 hours.
- Downloads land in 64 MiB in-memory scratch (section 3). Stream large files, or use shared datasets (section 7) for big inputs.
- An internet job can send anything it reads, including its provider credential, to any website. Never print, log or upload credentials, and only fetch from sources you trust. Treat downloaded content as untrusted input, and never execute it without review.

### Variables injected with a credential

| Variable | When |
| --- | --- |
| `AGENT_PROVIDER` | Always with a credential: `codex`, `claude`, `openrouter` or `openai`. Branch on this. |
| `AGENT_MODEL` | When the connection names a model. Otherwise use the provider's default. |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | `openai` connections |
| `OPENAI_BASE_URL` | Legacy `openai` network only: `http://egress:8080/v1` |
| `CODEX_API_KEY`, `OPENAI_API_KEY`, `CODEX_MODEL` | `codex` with an API key |
| `CODEX_HOME`, `CODEX_AUTH_FILE` | `codex` with a sign-in file (`/provider-auth/auth.json`) |
| `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_MODEL` | `claude` |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL` | `openrouter` |
| `AGENT_PLAN_TIER` | The connection's plan tier, when set |

### Calling each provider

These invocations come from the platform's tested provider example (`examples/agents/provider-research/run.py`), which runs on all three CLI-style providers:

```python
import json, os, subprocess, urllib.request
provider, model = os.environ["AGENT_PROVIDER"], os.environ.get("AGENT_MODEL")

if provider == "codex":                      # headless, non-interactive Codex
    os.makedirs("/work/codex", exist_ok=True)
    args = ["codex", "-a", "never", "-c", 'web_search="disabled"', "--disable", "shell_tool",
            "--disable", "view_image", "exec", "--ephemeral", "--sandbox", "read-only",
            "--skip-git-repo-check", "--output-last-message", "/output/answer.md"]
    if model:
        args += ["--model", model]
    subprocess.run([*args, "-"], input=prompt, text=True, cwd="/work/codex", check=True)

elif provider == "claude":                   # headless Claude Code, one turn, no tools
    os.environ["CLAUDE_CODE_DISABLE_ATTACHMENTS"] = "1"
    args = ["claude", "-p", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--max-turns", "1"]
    if model:
        args += ["--model", model]
    text = subprocess.run(args, input=prompt, text=True, stdout=subprocess.PIPE, check=True).stdout

elif provider == "openrouter":               # OpenAI-compatible chat API
    request = urllib.request.Request(
        os.environ["OPENROUTER_BASE_URL"] + "/chat/completions",
        data=json.dumps({"model": model or "openrouter/free",
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)

elif provider == "openai":                   # Responses API
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        base + "/responses",
        data=json.dumps({"model": model or os.environ["OPENAI_MODEL"], "input": prompt}).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
```

These flags give Codex and Claude Code no shell or file tools. Enable only the tools a task needs, and run the CLIs from a folder under `/work`, since the source tree is read-only.

### Provider rules

- Never commit, print or save credentials. Do not print authorization headers or raw error bodies that may echo request data. Do not put keys in inputs, command arguments or `agent.json`. Logs and artifacts are redacted as a safeguard, but do not rely on that.
- With a Codex sign-in file, logs are held until the run ends, and only one job at a time can use a given sign-in file. Other jobs using it wait in the queue.
- Model calls can cost money or use up subscription limits. Build a no-model code path for smoke tests, and make paid test runs only when asked.

## 6. Plan tiers, automatic routing, usage and handoff

**Tiers.** Each saved connection has a plan tier. The operator orders tiers from cheapest to most capable, currently `sol` then `astra`. Set `"tier"` to the cheapest tier your agent works with. When a researcher picks **Automatic**, the router uses the cheapest sufficient plan that still has budget left: a `sol` agent moves up to `astra` only if no `sol` plan is available, and an `astra` agent never runs on `sol`. Without `tier`, the agent needs only the cheapest tier. A tier name the operator has not configured makes automatic routing fail.

**Usage reporting.** Every run records its wall-clock time. Report model usage so plan budgets stay accurate:

```python
import platform_usage                                 # agents/shared/platform_usage.py

platform_usage.add_usage(tokens=1234, cost_usd=0.01)  # after each model call (adds to the totals)
platform_usage.set_plan_percent(72)                   # if a CLI shows the plan's own meter, 0-100
```

Without the helper, write cumulative JSON to `AGENT_USAGE_FILE`, such as `{"tokens": 1234, "cost_usd": 0.01, "plan_used_percent": 72}`, or `input_tokens` plus `output_tokens` in place of `tokens`. Write it atomically: write a temporary file, then rename it.

**Handoff.** When the connection's plan reaches the threshold (`AGENT_HANDOFF_THRESHOLD`, default 0.95), the worker creates the `AGENT_HANDOFF_SIGNAL` file. Long-running agents should check for it between steps, write notes and exit cleanly:

```python
if platform_usage.handoff_requested():
    platform_usage.write_handoff(
        "Solved 3 of 5 parameter sweeps before the plan ran low.",
        completed=["sweep 1", "sweep 2", "sweep 3"],
        next_steps=["run sweeps 4 and 5"],
        state={"next_sweep": 4})            # written to /output/HANDOFF.md
    sys.exit(0)
```

A researcher continues the work in a new run, and automatic routing skips plans that are used up. If an agent is signalled but writes no notes, the worker writes a basic `HANDOFF.md` from the inputs and recent logs. Design long agents so that a new run can resume from `state` passed back in as input.

## 7. Shared datasets (lab OneDrive)

Datasets are folders under `datasets/` in the lab's **UW Math AI Lab INVERSE PROBLEMS DATABASE** OneDrive folder. They contain simulated data, observations, meshes and similar files. Researchers add them directly in OneDrive, or save a finished run's outputs with **Save these outputs as a shared dataset**. The website's **Data & storage** page lists them.

Declare the datasets an agent needs:

```json
"datasets": ["heat-2d-sims"]
```

They are preselected on the launch form, and researchers can add or remove datasets for each run. A declared dataset that does not exist blocks the launch. Inside the job:

```python
import os
from pathlib import Path

root = Path(os.environ["AGENT_DATASETS_DIR"]) / "heat-2d-sims"   # /run/datasets/heat-2d-sims
for path in sorted(root.rglob("*.npz")):
    ...
```

- `AGENT_DATASETS_DIR` is set only when the run has datasets. Check for it before using it, and fail clearly if a required dataset is missing.
- Datasets are read-only, and symlinks are never staged. Treat their contents as untrusted input.
- Dataset names use letters, digits, spaces, `.`, `-` and `_`, up to 100 characters.
- A run's datasets may total at most 4 GiB. Large datasets should run on an AMATH worker, not the VPS.
- Agents cannot write to OneDrive. Outputs reach it through the run archive (below) or **Save as dataset**.

**Archive.** About 30 seconds after a run finishes, its `run.json` (manifest, inputs, status, events, artifact list, usage), `run.log` and `artifacts/` are moved to `runs/<agent>/<date>_<run id>/` in the same OneDrive folder. Give artifacts descriptive names, and write a machine-readable `result.json` alongside human-readable reports, so the archive remains useful without the website.

## 8. Subagents (swarm)

A swarm agent is an orchestrator that starts **subagents** while it runs. Each subagent is a separate run in its own container. It can run on a different worker server and with a different provider account, both from the set the researcher approves at launch. A parent can have up to 20 subagents queued or running at once, watch them, cancel them and read their results.

Enable it in the manifest:

```json
"swarm": { "max_children": 20, "max_total": 100 }
```

`max_children` (1–20, default 20) caps subagents queued or running at once. `max_total` (1–500, default 100) caps subagents over the parent's lifetime. At launch, the researcher ticks the servers and accounts the agent may use. The parent's own server and account are always included.

```python
import platform_swarm as swarm                   # agents/shared/platform_swarm.py

approved = swarm.status()                         # approved servers/accounts, limits, errors
swarm.spawn("shard-1", command=["python", "worker.py"], inputs={"shard": 1},
            server="doppio", connection="lab-codex-2")
swarm.spawn("shard-2", inputs={"shard": 2}, connection="auto", tier="sol")  # router picks an approved plan
swarm.spawn("shard-3", agent="simulator", files="/work/shard-3")           # another agents/<dir>, plus files

for child in swarm.wait(["shard-1", "shard-2", "shard-3"], timeout=3600):
    print(child["id"], child["status"], child["error"], flush=True)
    result = swarm.result(child["id"])            # parsed result.json
    outputs = swarm.output_dir(child["id"])        # the child's /output (read-only)

swarm.cancel("shard-2")                           # kill one subagent
swarm.cancel("*")                                 # kill all unfinished subagents
```

`spawn` options:

| Option | Meaning |
| --- | --- |
| `child_id` | Your name for the subagent: lowercase letters, digits, `-`, `_`, up to 64 characters, unique within the run. |
| `command` | Argv to run (default: the manifest `command`). |
| `agent` | Run code from another `agents/<dir>` in the same snapshot. If it has an `agent.json`, that manifest's image, network, limits and default command apply. |
| `inputs` | JSON for the child's `/run/inputs.json`, at most 64 KiB. |
| `server` | An approved server label (such as `"doppio"`) or ID, `"local"` for the VPS, or omitted for the parent's server. |
| `connection` | An approved account alias or ID, `"auto"`, `"none"`, or omitted to reuse the parent's account. |
| `tier` | Plan tier for `connection="auto"`. |
| `cpu`, `memory_mb`, `timeout_seconds` | Lower the manifest limits for this child. They cannot raise them. |
| `datasets` | A subset of the parent's datasets (default: all of them). |
| `files` | A folder under `/work` or `/tmp`, copied to the child's read-only `/run/parent-files`. At most 64 MiB and 2,000 files. |

Rules and behavior:

- Spawning is asynchronous. A request for an unapproved server or account is rejected, not run. Check `swarm.status()["errors"]` or `swarm.rejected(child_id)`.
- A subagent cannot reuse the parent's Codex sign-in file, because only one job at a time can use it. Use another approved account or `"auto"`.
- Subagents cannot start their own subagents.
- **Wait for subagents before the parent exits.** When the parent ends, its queued subagents are cancelled and running ones are stopped.
- Each finished child's folder `/swarm/results/<id>/` holds `output/`, `run.log` and `result.json`. `result.json` is written last. It contains status, error, server, account, provider, model, commit, image, usage, inputs, command, and each artifact's path, size and SHA-256. Copy what you want to keep into the parent's `/output`.
- Inside a subagent, `AGENT_SWARM_CHILD_ID` and `AGENT_PARENT_RUN_ID` are set, plus `AGENT_PARENT_FILES_DIR` when files were handed over. `swarm.child_id()` and `swarm.parent_files()` read them. One script can act as both parent and child by branching on `swarm.child_id()`.
- Plan resources: children have no CPU cap by default, so 20 children on one server share all of its cores, and at 512 MiB each they need 10 GiB of memory. Pass `cpu=` to `spawn` to cap a child. Send children to AMATH workers, not the VPS.

The platform's swarm demo (`examples/agents/swarm-demo/` in the platform repository) is a complete orchestrator that spreads shards over servers and accounts and gathers their results.

## 9. Where runs execute

The researcher chooses the execution server at launch: the VPS (default) or a registered AMATH Docker worker such as **doppio**. Agent code does not choose, except for subagents (section 8). The same container, paths and environment apply on every server, so do not hard-code hostnames or assume host software. The VPS is small (1 CPU, 2 GB, 2 jobs at once), so put CPU-heavy, memory-heavy or large-dataset work on a worker server, and say so in the README.

**GPU agents** (`"gpus"` above 0) can only be launched on a worker server whose test found NVIDIA GPUs, such as **americano** (2× TITAN RTX, 24 GiB each). The launch form preselects such a server, and the website refuses the VPS or a server without enough GPUs. The same applies to GPU subagents (section 8). A GPU run waits in the queue while all of its server's GPUs are in use. Host memory is still capped at 2,048 MiB per job by default, and CUDA PyTorch needs about 1–2 GiB just to start, so declare `"memory_mb": 4096` for GPU agents and ask the operator to raise the cap for heavy work.

## 10. Limits

| Limit | Value |
| --- | --- |
| Source in `agents/` (all agents together) | 64 MiB total, 16 MiB per file, 10,000 entries |
| Agent definitions | 100 |
| Manifest | 16 KiB; `inputs_schema` 16 KiB |
| Input JSON | 64 KiB |
| Logs per run | 1 MiB (redacted) |
| Artifacts per run | 50 MiB and 100 files |
| Scratch (`/work`, `/tmp`) | 64 MiB each |
| CPU and memory | No CPU cap (every core of the server) unless the manifest sets `cpu`; 2,048 MiB of memory per job by default, whatever the manifest says |
| Datasets per run | 4 GiB |
| Subagents | 20 at once per parent; 1 GiB of child results delivered per parent |

Operators can change these values. Keep generated results, large datasets (use section 7 instead), model weights, virtual environments, caches and output folders out of `agents/`. The source limits apply to all agents together. Temporary files do not survive a failed, cancelled or timed-out job.

## 11. Verify before handing off

1. Every `agents/*/agent.json` is valid strict JSON with only the fields in section 2, correct types and an approved `image`. Every slug matches the slug rule.
2. Every command and source file exists inside `agents/`, and imports work without the rest of the repository. The platform helpers exist in `agents/shared/` if they are used.
3. Run the entrypoint locally from the agent's folder, with `AGENT_INPUT_PATH` pointing at a sample input file and `AGENT_OUTPUT_DIR` at an empty folder:

   ```sh
   cd agents/test-agent
   mkdir -p /tmp/out && echo '{"topic": "heat equation"}' > /tmp/in.json
   AGENT_INPUT_PATH=/tmp/in.json AGENT_OUTPUT_DIR=/tmp/out python run.py; echo "exit $?"
   ```

   On Windows PowerShell, set the variables with `$env:AGENT_INPUT_PATH = "..."`.
4. Success produces readable logs, progress lines and regular files in the output folder. Invalid input or failed work exits nonzero.
5. The code uses only Python 3.12's standard library, NumPy, SciPy, scikit-learn, PyYAML and the `codex`/`claude` CLIs (plus PyTorch for GPU agents on `research-agent-runtime-gpu:v1`). A pass in a richer local environment proves nothing.
6. Tests make no paid model requests unless the user explicitly asks for one.
7. Each agent folder has a short README covering purpose, inputs, sample input JSON, outputs, required connection/provider/tier, datasets, swarm use, recommended server, dependencies and whether it incurs charges.

A local test checks the input/output protocol only. Only a website run checks the real container, network, dataset and provider limits.

## 12. Publish to the website

1. Commit the agent files and push or merge them into `main` of `uw-math-ai/ai-inverse-lab`. Pull-request approvals are not required by the website.
2. Log in to the website, open **Agent catalog** and click **Refresh catalog**. If the refresh fails, fix the reported manifest error; the old catalog stays active meanwhile. A push webhook may also refresh the catalog, but do not assume one is configured.
3. Confirm the agent appears with the expected name, input form and Git commit.
4. Start a no-cost smoke test. Check status, logs, progress and artifacts. For provider agents, select an authorized connection and make a paid test only when asked. For swarm agents, approve servers and accounts in the **Subagents** section of the launch form.

Runs keep the source snapshot they started with. A retry or a new run uses the current catalog revision. At the next refresh, a changed agent folder updates its catalog entry, and a deleted one deactivates it.

When reporting completion, state which agent folders changed, give sample input JSON, list the artifacts produced and say which checks ran. Distinguish local validation from a successful website run, and never claim an agent is published just because its files exist locally.
