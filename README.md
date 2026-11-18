# Mathematical agent workflow demo

The demo is a Python package in [workflow/](workflow/). It includes live CA, skeptic and verification-adviser roles; Lean checking of fixed targets; ten-iteration refinement; persistent proof checkpoints; benchmark comparisons; and a local dashboard with readable mathematics and revision diffs.

## Setup

From this repository root, create a Python environment and install the tested dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r workflow/requirements.txt
```

Install the pinned Lean toolchain specified in [workflow/lean/lean-toolchain](workflow/lean/lean-toolchain), then check it:

```bash
python -m workflow.response_call lean-status
```

Configure your OpenAI API key using the hidden terminal prompt and start the dashboard:

```bash
python -m workflow.response_call configure-openai
python -m workflow.dashboard --port 8766
```

Open http://127.0.0.1:8766 and select the OpenAI backend for live runs. Launching the dashboard does not start model calls. The VS Code tasks provide the same dashboard and private credential-setup commands.

See [workflow/README.md](workflow/README.md) for benchmark definitions, per-role model settings, scoring, iteration policy, saved artifacts and verification limits. The graph simulation support imported by the workflow is included in [forwardSimulation/forward_sim/](forwardSimulation/forward_sim/); numerical probes default to disabled and are not proof certificates.

## Validation

```bash
python -m unittest discover -s workflow/tests -v
```

The tests need the pinned Lean toolchain and do not make live model API calls. Optional browser checks use [workflow/requirements-dev.txt](workflow/requirements-dev.txt) and Playwright Chromium.

Credentials, generated runs, virtual environments and caches are excluded from Git. The hidden-potential research specification remains separate from executable benchmark campaigns.
