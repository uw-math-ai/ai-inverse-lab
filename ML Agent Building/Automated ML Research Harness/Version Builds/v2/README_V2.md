# ML Research Harness V2: user guide

V2 turns the harness from an agent-assisted model search into a **research-first** system. Four
researcher agents propose falsifiable hypotheses. A shared critic strengthens those hypotheses over
several rounds. TheoremSearch supplies existing theorems, and a persistent research memory carries
findings and failures between runs. Only after that does the existing scheduler implement, run and
compare a diverse set of methods.

The V1 pipeline (`mlharness run`) still works unchanged. V2 is `mlharness research`, or any spec
with `pipeline: v2`.

```
spec ─► data (+shift sets, forward model, identifiability diagnostics) ─► mean baseline
     ─► RESEARCH  4 researchers (parallel) ⇄ shared critic, + TheoremSearch + memory + pilots
     ─► PORTFOLIO diverse proposals + baselines (conventional ML, random search, physics MLE)
     ─► GATE      deterministic validity / safety / leakage / metric / budget checks
     ─► SCREEN ─► EVALUATE ─► REFINE ─► PREDICTION CHECKS ─► FINAL (test once) ─► MEMORY ─► REPORT
```

## Quick start

Run all commands from the harness folder (`ML Agent Building/Automated ML Research Harness`).

```bash
PY=C:/Users/kusha/AppData/Local/Programs/Python/Python311/python.exe
$PY -m pip install -r requirements.txt            # paramiko is only needed for remote GPUs
$PY -m mlharness doctor                           # checks backends, TheoremSearch, memory, hardware
$PY -m mlharness research --spec specs/heat_v2.yaml                  # LLM researchers (auto backend)
$PY -m mlharness research --spec specs/heat_v2.yaml --backend none   # offline heuristic researchers
$PY -m unittest discover -s tests -v              # 36 tests (14 V1 + 22 V2)
```

Each run writes to `runs/<name>_v2_<timestamp>/`:

| file | contents |
|---|---|
| `report.md` | the research report, organised by evidence class (§3), with idea evolution, literature, memory, portfolio, lineage and cost |
| `research_trace.json` | machine-readable record of every proposal version, critique, action, pilot, prediction outcome and literature query |
| `artifacts.jsonl` | the canonical append-only artifact ledger (provenance). V2 adds the kinds `proposal`, `literature`, `feedback`, `portfolio`, `gate` and `memory`. |
| `final/model.pkl`, `final/manifest.json`, `final/model_card.md` | the exported method, with reproducibility metadata and research lineage |
| `leaderboard.csv`, `status.json`, `harness.log` | the leaderboard, live status and log |

To load and use an exported method:

```python
from mlharness.export import load_model
m = load_model(r"runs/<run_dir>")
theta = m.predict(X)          # physics members call the forward model; the simulator file is copied into final/
```

## CLI

| command | purpose |
|---|---|
| `research --spec S [--backend B] [--rounds N] [--research-llm-calls N] [--memory PATH] [--no-memory] [--no-literature] [--no-forward-model] [--remote HOST] [--budget s] [--workers n] [--resume]` | run a V2 research-first study |
| `memory search "query" [--type finding,failure] [-k 10] [-v] [--show-test]` | search the cross-run research memory (test numbers are redacted unless `--show-test`) |
| `memory show <id>` / `memory stats` | show one record and its lineage, or show store statistics |
| `literature "query" [-n 5] [--spec S]` | query TheoremSearch directly; with `--spec`, also screen applicability against that simulator's facts |
| `remote-check [--host americano]` | probe an SSH GPU server: GPUs, Python, numpy/sklearn, torch |
| `run`, `report`, `trace`, `predict`, `status`, `init`, `doctor` | V1 commands. `run` dispatches to V2 when the spec says `pipeline: v2`. |

## Configuration (V2 sections; all optional, defaults shown)

```yaml
pipeline: v2
constraints:
  allow_forward_model: true     # methods may call the simulator's forward operator through ProblemContext
  max_forward_evals: 3000000    # per job; counted inside the context
  allow_custom_code: true       # agent-written custom_ml / custom_solver code (static + runtime sandbox)
research:
  researchers: [mathematical, ml, experimental, exploratory]
  max_rounds: 3                 # propose/revise passes; each is followed by a critique
  min_rounds: 2                 # quality-based early stop is allowed only after this many critiques
  proposals_per_researcher: 2
  quality_threshold: 0.75       # every active proposal must reach this, with no blocking issue
  llm_calls: 30                 # hard cap on research-phase LLM calls (researchers + critic)
  wall_seconds: 1800
  queries_per_round: 2          # TheoremSearch queries per researcher per round
  llm_critic: true              # false = rule-based critic only
  pilot_tests: true             # run the cheap discriminating pilots the critic asks for
  max_pilots_per_round: 3
  pilot_val_subsample: 60
literature: {enabled: true, base_url: https://api.theoremsearch.com, timeout: 45, retries: 1,
             n_results: 4, max_total_queries: 40, cache: true}
memory: {enabled: true, path: null, read: true, write: true}   # null = .research_memory/memory.jsonl
execution:
  portfolio_size: 6             # research proposals advanced; baselines are added on top
  include_baselines: true       # conventional ML default, random search, physics MLE (if a forward model exists)
  screen_fraction: 0.1          # training fraction at the screen (trainable methods only)
  eval_subsample: 150           # validation samples at the screen (all methods)
  refine_rounds: 1              # local mutation of the best trainable methods
  ensemble: true
  bootstrap: 200                # bootstrap resamples for the test-set CI
evaluation:
  shift_sets: auto              # auto = 3x sensor noise if the simulator has a `noise` parameter
  shift_n: 300                  # or a list: [{name: k_low_0.3, params: {k_low: 0.3}}, ...]
hardware:
  remote: {enabled: false, host: americano, username_env: SERVICE_USERNAME,
           password_env: SERVICE_PASSWORD, env_file: auto, python: python3, use_for: gpu}
```

### Making a simulator "physics-ready"

A `python` simulator plug-in can expose two optional module-level functions. With them, researchers
can propose physics and hybrid methods, and the critic gets numerical identifiability diagnostics.
`examples/heat_diffusion_simulator.py` shows both.

```python
def forward(theta, **params):        # (n, p) parameters -> (n, m) noise-free observations
def problem_info(**params):          # {"param_names", "lower", "upper", "noise_std", "noise_model", "prior", "facts": {...}}
```

## Executable method kinds (the `method` field of a proposal)

| kind | what runs | forward model |
|---|---|---|
| `library` | any V1 estimator plus feature set, scaler and PCA, with `target_transform: log` and `noise_aug` (matched-noise augmentation) | no |
| `physics` | per-sample bounded Levenberg–Marquardt through the forward model: MLE, MAP (`tikhonov`) or posterior-mean/MMSE (`estimator: posterior_mean`), with Gauss–Newton uncertainty | yes |
| `hybrid` | a learned initializer (a library spec) followed by LM refinement | yes |
| `custom_ml` | agent code `build_model(params, task, seed)` (the V1 contract) | no |
| `custom_solver` | agent code `build_method(params, context)` returning an object with `fit`/`predict`. `context.forward`, `jacobian`, bounds and noise are available. | optional |
| `random_search` | control arm: `n` uniform random library configurations | no |

## Safety model (independent of the research critic)

1. **Gate (static, deterministic).** Checks method validity, task compatibility and forward-model
   authorisation. Agent code gets an AST check: no banned imports/calls, no dunder or reflective
   access, no held-out-data references. The gate also scans for leakage, checks metric correctness,
   and estimates cost and forward-evaluation budget.
2. **Runner (dynamic).** Every experiment runs in its own process with a timeout and with
   credentials scrubbed from the environment. Jobs carrying agent code also get a PEP 578 audit hook
   that blocks process creation, sockets, and writes, deletes or renames outside the job directory.
   Development jobs never receive the test or shift arrays.
3. **Remote.** Only the harness runner command is executed on SSH hosts. Credentials are read by
   variable name from `.env` and are never logged.

The audit hook is defence in depth, not a container. Treat genuinely hostile code as out of scope
(see the implementation report).
