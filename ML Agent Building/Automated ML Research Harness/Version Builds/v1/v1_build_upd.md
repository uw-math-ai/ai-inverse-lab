# Autonomous ML Research Harness — V1: what was built

This folder implements the system described in [description.md](description.md): a multi-agent ML
research harness. You give it a research problem; it simulates data, generates and screens many ML
methodologies cheaply, promotes only the promising ones, refines the best, and delivers a trained,
evaluated model with a full research record.

The description stops partway through Layer 3, after the scheduler's responsibilities. Layer 4 (the
research and experimental pipeline) and every unspecified detail were designed by me for V1. All of
those choices are listed in **§6, Design decisions to review**, so you can change them.

---

## 1. What I did

1. **Built the `mlharness` package** (~3,800 lines of Python). It contains the four layers, six agent
   types, a methodology library, simulators, an isolated experiment runner, and a CLI.
2. **Connected it to the lab's research.** The flagship simulator, `graph_fpt`, is the
   first-passage-time inverse problem from the lab's work. It follows the conventions of the lab's
   `fptsim` package ("Toggable Simulation of Data Code") and agrees with `fptsim`'s exact recursion to
   **~1e-16** (tested).
3. **Added four LLM / coding-agent backends.** `claude_code` (headless Claude Code CLI), `anthropic`
   (Messages API, `claude-opus-5-5`), `codex` (Codex CLI) and `none` (offline heuristic agents). The
   harness auto-detects which are available; the bundled Claude Code and Codex binaries inside VS Code
   are found automatically.
4. **Ran it end-to-end four times.** The runs used real Claude Code agents and offline heuristics (§4).
   The Claude-driven run produced the best model: **test relative L2 error 1.19 %** recovering 12 edge
   weights from boundary first-passage data. That beats the 5 % success target and the offline agents
   (1.38 %).
5. **Wrote 14 tests**, all passing. They cover the simulator (including the cross-check against
   `fptsim`), all methods, critic code-safety checks, the artifact store, and full pipeline runs. A
   scripted fake LLM exercises every LLM code path: ideation, custom code, repair, critic, refinement
   and the report discussion.
6. **Fixed problems found while running it.** These are described in §5 and §7.

### Folder layout

```
Automated ML Research Harness V1/
├── description.md               original brief
├── end.md                       this file
├── requirements.txt
├── mlharness/
│   ├── spec.py                  Layer 1: research specification compiler
│   ├── artifacts.py             Layer 2: storage-agnostic artifact store (JSONL ledger impl.)
│   ├── scheduler.py             Layers 3+4: overall scheduler + pipeline stages
│   ├── executor.py              parallel, budgeted, isolated experiment execution + result cache
│   ├── runner.py                subprocess entry point that trains/evaluates one experiment
│   ├── resources.py             compute/wall/LLM budgets, hardware detection, GPU slots
│   ├── llm.py                   none | claude_code | anthropic | codex backends
│   ├── agents/                  simulation, ideation (4 personas), critic, engineer, reporter
│   ├── methods.py               methodology library (12 methods) + search spaces + pipeline builder
│   ├── features.py              feature transforms incl. domain features for FPT data
│   ├── torch_models.py          sklearn-compatible PyTorch MLP (uses GPU when allocated)
│   ├── simulators/              graph_fpt, friedman, regression, classification, python plug-in
│   ├── export.py                final model format (single or weighted ensemble) + loader
│   ├── state.py                 rebuild a run's full state from artifacts alone
│   └── cli.py                   `python -m mlharness ...`
├── specs/                       example research specifications
├── examples/                    example user simulator (1-D heat equation inverse problem)
├── tests/test_harness.py
└── runs/                        output of the demo runs (data/workspace/models are git-ignored)
```

---

## 2. How each objective in the description is met

### §1 Project goal

| Requirement | How it is met |
|---|---|
| Take a research problem, generate synthetic data via simulation | `SimulationAgent` + `simulators/`; the problem comes from a spec file or free text (`mlharness init`) |
| Explore many ML methodologies | 4 persona ideators propose ideas in parallel from a 12-method library, or write their own model code (`method: custom`) |
| Identify and train the best model | Multi-fidelity selection on validation data, then refinement, an optional ensemble, a final retrain on train+val, and one test evaluation |
| Automate idea → experiment → evaluation → selection | The `ResearchScheduler.run()` pipeline (§3) runs with no human in the loop |
| Minimise unnecessary compute | LeanML-style screening, cost projection, successive halving, budget caps, caching (§5) |
| AI-Scientist inspiration | Independent agents propose, implement (including writing code), evaluate and critique. Agent-written code that crashes gets an LLM debug/repair loop. The reporter writes the scientific discussion. |
| LeanML inspiration | Cheap feasibility checks on 10 % of the data. Ideas that crash, are projected to be too slow, or don't beat the baseline are eliminated before real compute is spent. |
| Primary output: trained, evaluated model + artifacts | `runs/<run>/final/model.pkl`, `model_card.md`, `manifest.json`, `report.md`, `leaderboard.csv`, `artifacts.jsonl` |
| Functional and extensible over fully autonomous | Plug-in simulators, a method registry, pluggable LLM backends, an abstract artifact store |

### §2 Architecture: the four layers and the central scheduler

- **Layer 1, research specification:** `spec.py`.
- **Layer 2, shared artifacts:** `artifacts.py`.
- **Layer 3, agent orchestration:** `scheduler.py` and `agents/`.
- **Layer 4, pipeline:** the stages in `scheduler.py`, run through `executor.py` and `runner.py`.

`ResearchScheduler` is the central coordinator. Agents never call each other: they read and write
artifacts, and the scheduler passes them artifact-derived context. The system runs either with coding
agents (Claude Code or Codex) or with the Anthropic API, and has Python execution and simulation
built in. GPUs are detected and allocated to GPU-capable jobs; this machine has none, so the GPU path
is untested (§7).

### §3 Layer 1, research specification

Supported user inputs, each mapped to a spec section:

- research question → `problem`
- simulation requirements → `simulator`, `data`
- evaluation metrics → `metrics`
- compute budget → `budget`
- hardware → `hardware`, auto-detected
- methodological constraints → `constraints`: allowed or forbidden methods, custom code on/off, free-text notes

`compile_spec` turns these inputs into a validated specification. It:

- fills defaults and infers the task from the simulator;
- checks that each metric fits the task;
- resolves workers, threads per worker and GPUs;
- derives explicit **objectives** and a **success criterion**, e.g. "test rel_l2 ≤ 0.05".

If only free text is given, an LLM fills in the spec (`mlharness init --problem "..."`).

The spec is the first artifact. Every LLM agent receives `spec.brief()` with every request.

**Alignment checks** are made continuously:

- The critic rejects ideas whose method doesn't support the task, that violate constraints, are
  invalid or duplicated, or (in LLM mode) are judged misaligned by an LLM review of each batch.
- The scheduler only ever selects on the spec's primary metric on validation data.
- The final audit checks the success criterion and test-versus-validation consistency.

### §4 Layer 2, shared research artifacts

Every artifact type in the description exists as an artifact kind:

| From the description | Artifact kind |
|---|---|
| hypotheses / methodologies | `hypothesis` |
| experimental plans and implementation details | `plan`, `code` |
| datasets and simulation configs | `dataset` |
| training scripts and checkpoints | `code`, `checkpoint` (+ `workspace/experiments/<exp>/`) |
| metrics and results | `result` |
| critiques and failure explanations | `critique`, `failure` |
| compute usage | `resource` (per experiment and per LLM call, with seconds and $) |
| decisions about what advances | `decision` (prune, promote, not_promoted, budget_cut, select_final, …) |

Each artifact also records which agent created it (`agent`) and when the report was written (`report`).

- **Traceability.** Each artifact records `idea_id`, `parents`, `author` and `stage`.
  `mlharness trace <run>` walks from the final checkpoint back through result → plan → hypothesis →
  parent hypotheses → spec and dataset.
- **Artifacts are the memory.** Agents get no conversation history. Ideators are given the
  leaderboard, failures, critiques and budget derived from artifacts. `mlharness report --rebuild`
  regenerates the entire report from `artifacts.jsonl` alone, which shows the artifacts are sufficient.
- **Storage-agnostic.** The `ArtifactStore` abstract class has `FileSystemStore` (append-only JSONL
  plus a workspace directory) and `MemoryStore`. A database store only has to implement `_persist`
  and `workspace`.

### §5 Layer 3, overall scheduler responsibilities

| Responsibility | Implementation |
|---|---|
| Spawning research agents | `scheduler.spawn()` creates the simulation, critic, engineer and reporter agents plus one ideator per persona, and records an `agent` artifact for each |
| Allocating tasks | Ideation, proposal review, planning, execution, result review, repair, reporting |
| Coordinating parallel research | Ideators run concurrently (threads, since LLM calls are I/O-bound). Experiments run in a pool of N worker subprocesses. |
| Managing compute | `ResourceManager`: compute, wall-clock and LLM-call budgets, cumulative per-stage caps, per-job thread caps, GPU slot queue, per-job timeout |
| Deciding what advances | Screen pruning, successive-halving promotion, budget trimming, refinement parent selection, ensemble selection; every choice is a `decision` artifact |
| Tracking artifacts and outcomes | `status.json` (live), `artifacts.jsonl`, the report's funnel and lineage sections |
| Deciding when to stop | Wall or compute budget exhausted, `max_ideas` reached, no relative improvement ≥ `min_improvement` for `patience` rounds, optional stop-on-success, or all refinement rounds done |

---

## 3. Layer 4: the research pipeline I designed

```
spec ─► data ─► baseline ─► ideation ─► prescreen ─► screen ─► halving ─► refine ⟲ ─► final ─► report
          │         │           │            │           │          │          │          │
   simulate+cache  mean      4 personas   critic    10% data   33%→100%   children   ensemble?
   train/val/test  predictor in parallel  (0 cost)  feasibility promote    of best    retrain on
                                                    + cost      top 1/η    ideas      train+val,
                                                    projection  budget-    (LLM or    test ONCE,
                                                    + repair    aware      mutation)  audit, export
```

1. **Data.** The simulator draws all samples at once and splits them 4000/1000/1000 into
   train/val/test. The dataset is cached by a hash of its configuration.
2. **Baseline.** A mean (regression) or majority-class (classification) predictor is the reference
   every idea must beat.
3. **Ideation.** Four personas each propose `ideas_per_ideator` ideas: classical, ensembles, neural,
   and domain (physics-informed feature engineering). An idea is a method plus parameters, a feature
   set, a scaler and PCA, or custom code.
4. **Prescreen.** The critic rejects or repairs ideas at zero compute cost. Parameters are clipped to
   the search space, duplicates are removed by config hash, custom code gets a static safety check,
   and in LLM mode an LLM reviews the batch for alignment.
5. **Screen (LeanML feasibility check).** Every approved idea trains on 10 % of the training data.
   - Ideas that crash or time out are pruned, and the critic writes a failure explanation.
   - Agent-written code that crashes gets one LLM repair attempt and a re-run.
   - Fit time is extrapolated to full data as fit·(1/f)^α, where α is each method's scaling exponent,
     e.g. 2.2 for kernel ridge. Ideas projected to exceed the per-experiment limit are pruned.
   - Ideas that don't beat the baseline by ≥ 2 % are pruned.
6. **Successive halving.** Survivors are promoted at rungs of 33 % and then 100 % of the data. Each
   rung keeps the top ⌈n/η⌉ (η = 3), with at least `min_full` reaching full data.
   - Before each rung, the projected cost of each candidate is checked against the per-experiment
     limit and the stage's remaining budget. The lowest-ranked candidates are cut first.
7. **Refinement.** The top-k full-data ideas go back to the ideator of their persona, which proposes
   children. The LLM is shown the results and critiques; offline, the ideator mutates
   hyper-parameters and feature sets. Children are critic-checked and evaluated on full data. Rounds
   repeat until improvement stalls, the budget runs out, or `max_ideas` is reached.
8. **Final.** A Caruana greedy ensemble is built on the saved validation predictions (no training). It
   is used only if it beats the best single model by ≥ `min_improvement`. The chosen model is
   retrained on train+val and **evaluated once** on the untouched test split. The critic audits the
   result: test-versus-validation degradation and the success criterion. The model is exported with
   a manifest and model card.
9. **Report.** `report.md` contains the spec, outcome, funnel, leaderboard, multi-fidelity trace,
   lineage, pruned ideas, critic notes, resource usage, agent roster, and an LLM-written discussion.

---

## 4. Results from the demo runs (this laptop: 12 CPUs, no GPU)

All runs used the specs in `specs/`, and their reports are in `runs/*/report.md`.

| Run | Agents | Ideas (proposed / pruned / full-data) | Final model | Val | **Test** | Success? | Compute / wall | LLM |
|---|---|---|---|---|---|---|---|---|
| `graph_fpt_weights_claude` | Claude Code | 34 / 3 / 22 | ensemble of 2 kernel-ridge (log+fpt_summary, fpt_early) | rel_l2 0.0123 | **rel_l2 0.0119**, R² 0.9989 | ✅ (≤ 0.05) | 719 s / 582 s | 14 calls, $1.11 |
| `graph_fpt_weights_offline` | heuristic | 28 / 0 / 16 | ridge (raw features, α≈0.73) | rel_l2 0.0138 | **rel_l2 0.0138**, R² 0.9985 | ✅ | 402 s / 145 s | none |
| `quick_friedman` | heuristic | 11 / 1 / 6 | extra trees | RMSE 1.77 | **RMSE 1.73**, R² 0.88 | ❌ (≤ 1.5; noise floor 1.0) | 44 s / 20 s | none |
| `classification_demo` | heuristic | 18 / 1 / 9 | extra trees (refined child) | acc 0.880 | **acc 0.872**, F1 0.871 | ❌ (≥ 0.9) | 75 s / 24 s | none |

The task in the two graph runs: recover the 12 edge weights of a weighted 3×3 grid from noisy
first-passage-time distributions r′(t,x,y), t ≤ 9, observed on the 8 boundary vertices, with 10⁴
walks per pair (576 features). The mean-predictor baseline is rel_l2 0.359.

What the runs showed:

- **The Claude agents found a better model family.** The domain-persona agent proposed RBF kernel
  ridge on early-time and summary FPT features, and three refinement rounds improved it by 21 %, 8 %
  and 8 %. The offline heuristics converged on linear ridge.
- **Screening and pruning worked as intended.** The cost projection pruned a custom
  "ridge + boosted residuals" model (projected 731 s against a 240 s limit). The halving cut stopped a
  slow hist_gbm from reaching full data, which saved about 135 s of compute in the offline rerun. The
  critic rejected duplicates and unsafe code.
- **The selection estimate held up.** Test error was no worse than validation error in either graph
  run, so the critic audit reported no selection overfitting.
- **The agents' own analysis was useful.** The reporter agent's discussion (in the Claude run's
  report) proposes next steps such as estimating the noise floor with a Cramér–Rao bound, and
  Gauss–Newton refinement through the known forward model. These are good leads for this research.

---

## 5. Optimisations I made

**Compute efficiency (LeanML side)**
1. **10 % feasibility screen.** Most ideas are judged on 10 % of the data at roughly a tenth of the
   cost.
2. **Fit-time projection** using a scaling exponent per method. Ideas that would blow the
   per-experiment limit are pruned before any full-data run, both at the screen and before each
   halving rung.
3. **Successive halving** (η = 3) with nested data subsets: each rung's sample contains the previous
   rung's.
4. **Budget-aware promotion.** Cumulative stage caps (baseline 5 %, screen 20 %, halving 60 %,
   refine 85 %, final 100 %) mean unused budget rolls forward and the final stage is never starved.
5. **Baseline margin.** Ideas must beat the mean predictor by ≥ 2 %, which catches degenerate models
   early.
6. **Zero-cost prescreen.** Invalid, duplicate, misaligned or unsafe ideas never reach compute.
7. **Ensembles built from saved validation predictions,** with no retraining. Only the chosen final
   model is retrained.
8. **Caching.**
   - Datasets are cached across runs, keyed by a hash of the simulator config, size and seed. The
     second graph run skipped simulation.
   - Experiment results are keyed by config, fraction, data and seed, so `--resume` never re-runs a
     finished experiment.

**Throughput**

9. **Parallel worker pool.** Workers = CPUs/2, and BLAS/OMP threads are capped per job
   (CPUs/workers) to avoid oversubscription.
10. **Parallel ideators and refiners.** LLM latency overlaps instead of adding up.
11. **Fast simulator.** The exact first-passage recursion is fully vectorised over samples, and the
    Monte Carlo noise is drawn as one multinomial sample per (x,y) pair. This is statistically
    identical to simulating N walks: 6,000 samples × 64 pairs × 10⁴ walks takes 0.5 s.
12. **GPU slot queue.** Torch jobs take a free GPU and fall back to CPU instead of stalling.

**Reliability and cost**

13. **Fail-soft LLM use.** Every LLM call is budgeted (`llm_calls`), logged with seconds and $, and
    falls back to heuristics on failure. A run never depends on the model being available.
14. **Experiment isolation.** Each experiment runs in a subprocess with a hard timeout, so a crash, an
    out-of-memory error or a runaway job can't take down the scheduler.

**Fixes found while running it**

15. **Kernel ridge bandwidth.** It broke silently under PCA, predicting only the mean, because
    gamma = 1/d ignores feature variance. I replaced it with a data-scaled bandwidth
    (`gamma_mult`·1/(d·Var X)). Kernel ridge then became the best method.
16. **Code-safety false positives.** The critic rejected `model.eval()` as if it were the builtin
    `eval`. The check is now module-aware.
17. **Over-eager overfitting flag.** It is now calibrated against the best result at the same
    fidelity. The LLM reporter independently pointed out the same issue.
18. **Wall-clock default.** It now includes headroom for LLM latency. The first Claude run used
    582 of 600 s.

---

## 6. Design decisions to review

These are the choices I made for V1 where the description was silent. They are grouped so you can
decide what to change.

**Scope and problem**
- **D1. Demo problem.** I used the lab's FPT inverse problem: edge weights of a 3×3 grid, observed on
  the boundary, T = 9, N = 10⁴ walks. `specs/graph_fpt_potential.yaml` adds the potential q. Other
  graphs (path, cycle, complete, custom) and targets (weights, potential, both) are configurable.
- **D2. Fixed topology.** Each sample has the same graph and random weights, so the targets have a
  fixed size. Variable-size graphs would need padding or a graph neural network (GNN).
- **D3. Splits.** One simulator draw, split 4000/1000/1000. The test split is touched exactly once.

**Agents and LLM**
- **D4. Personas.** Four ideator personas (classical, ensembles, neural, domain), one agent each. They
  are configurable via `search.ideators`.
- **D5. LLM as a reasoner, not an operator.** Coding agents run headless with **tools disabled** and
  return JSON or code. The harness itself executes all code in sandboxed subprocesses. This makes
  runs reproducible, budgetable and safe, but the agents can't explore files or run code on their
  own. An agentic mode, where Claude Code edits and runs code in an experiment workspace, is the
  natural next step.
- **D6. Default models.** The `anthropic` backend uses `claude-opus-5-5` with adaptive thinking,
  effort `medium`, streaming, and the server-side refusal fallback. `claude_code` uses your CLI's
  default model. Override either with `--model`.
- **D7. Custom code.** Agents may write models as `build_model(params, task, seed)` returning a
  sklearn-style estimator. A static AST check blocks process, network and file calls. This is **not a
  security sandbox**; see §7.
- **D8. Critic veto.** The rule-based critic has a hard veto. The LLM critic can only reject ideas that
  are clearly misaligned, so it can't silently empty the idea pool.
- **D9. Refinement.** Parents go back to the ideator of their own persona. Children are evaluated
  directly on full data, without re-screening, because parents are already validated. As a result,
  refinement can converge on one family: in the Claude run, all refinement went to the domain persona
  (see N3).

**Pipeline numbers** (all in `search:` and `budget:`)
- **D10. Fidelity schedule.** Screen at 10 % of the data, rungs at [33 %, 100 %], η = 3, `min_full` = 3
  to 4.
- **D11. Fidelity axis.** Training-data fraction is the only fidelity dimension. Epochs or iterations
  are not used as a fidelity axis.
- **D12. Stage budget shares.** Cumulative caps of 5 / 20 / 60 / 85 / 100 %.
- **D13. Stopping.** `min_improvement` 0.5 % relative and `patience` 1 to 2 rounds. `stop_on_success`
  is off by default, so the search keeps optimising after the target is met.
- **D14. Ensemble.** Greedy, with replacement, at most 5 candidates, adopted only if it beats the best
  single model by ≥ 0.5 % on validation.
- **D15. Final model.** Retrained on train+val. Its validation score is therefore a slightly
  optimistic estimate; the test score is the honest one.

**Engineering**
- **D16. Experiment isolation.** One subprocess per experiment. This adds about 2 s of Python/sklearn
  import overhead per job, which is significant for tiny models.
- **D17. Compute accounting.** Compute is measured in worker-seconds (wall time × jobs), not CPU or
  GPU time.
- **D18. Storage.** A JSONL ledger plus files. There is no database, following the storage-agnostic
  requirement.
- **D19. Metric for the FPT problem.** `rel_l2` = ‖Ŷ−Y‖_F/‖Y‖_F, the lab's FRNE as a fraction. R² and
  RMSE are reported as secondary metrics.

---

## 7. Limitations and suggested next steps

- **Untested here:** the GPU path (no GPU on this machine), the `anthropic` backend (no API key
  configured), and the `codex` backend (the binary is detected, but I didn't spend Codex credits). All
  three follow documented interfaces and fall back to heuristics on error. The `claude_code` backend
  was tested end to end.
- **Safety:** agent-written code runs as your user, with only AST screening. Run untrusted
  configurations inside a container or VM, or set `allow_custom_code: false`.
- **N1. Subprocess overhead.** A persistent worker pool with warm imports would remove the roughly
  2 s per-job overhead (D16).
- **N2. Agentic coding mode.** Let Claude Code or Codex edit and run code in the experiment workspace
  (D5).
- **N3. Diversity quota in refinement** so that one family can't take over (D9).
- **N4. Physics-aware methods for the FPT problem,** suggested by the agents themselves:
  Gauss–Newton refinement through the exact forward model, a noise-floor or Cramér–Rao estimate, and
  more training data with Nyström kernel ridge.
- **N5. Repeated seeds or cross-validation for the final comparison,** to get confidence intervals.

---

## 8. How to use the agent

### Install

```bash
cd "ML Agent Building/Automated ML Research Harness V1"
pip install -r requirements.txt     # numpy, scipy, scikit-learn, pyyaml (+ optional torch, anthropic)
python -m mlharness doctor          # shows hardware, detected agent backends, available methods
```

On this machine, use the Python that has torch:
`C:\Users\kusha\AppData\Local\Programs\Python\Python311\python.exe`. The default `python` (3.14) has
no numpy.

### Run a research problem

```bash
# offline (free, deterministic, ~3 min)
python -m mlharness run --spec specs/graph_fpt_weights.yaml --backend none

# with Claude Code agents (~10 min, ~$1 at default settings)
python -m mlharness run --spec specs/graph_fpt_weights.yaml --backend claude_code

# other options
python -m mlharness run --spec specs/quick_friedman.yaml           # 30-second smoke test
python -m mlharness run --spec S --budget 1800 --workers 8 --run-dir runs/my_run
python -m mlharness run --spec S --run-dir runs/my_run --resume   # continue; finished experiments are reused
```

Backends: `auto` (the default in the specs) picks `anthropic` if an API key is set, then Claude Code,
then Codex, then `none`. You can point to the CLI binaries with the `MLH_CLAUDE_CLI` and
`MLH_CODEX_CLI` environment variables. For `anthropic`, run `pip install anthropic` and set
`ANTHROPIC_API_KEY`.

### Write a research spec

Copy a file from `specs/`, or generate one from text:

```bash
python -m mlharness init --problem "Recover a sparse vertex potential on a 4x4 grid from boundary FPT data" --out specs/mine.yaml
```

The key fields are below; everything else has defaults (see `DEFAULTS` in `mlharness/spec.py`).

```yaml
name: my_problem
problem: "free-text research question"            # quote it if it contains '#'
simulator: {name: graph_fpt, params: {...}}        # or friedman | regression | classification | python
data: {n_train: 4000, n_val: 1000, n_test: 1000, seed: 7}
metrics: {primary: rel_l2, secondary: [r2, rmse]}  # rel_l2 rmse mae r2 | accuracy f1_macro balanced_accuracy
success: {threshold: 0.05}
budget: {compute_seconds: 900, max_ideas: 40, experiment_timeout: 240, llm_calls: 30}
hardware: {workers: auto, gpus: auto}
constraints: {allowed_methods: all, forbidden_methods: [svm], allow_custom_code: true, notes: ["..."]}
search: {ideators: [classical, ensembles, neural, domain], ideas_per_ideator: 4, screen_fraction: 0.1,
         rungs: [0.33, 1.0], eta: 3, refine_rounds: 3, patience: 2, ensemble: true}
llm: {backend: auto, model: null, effort: medium}
```

### Use your own simulator

Write a file with `simulate(n, seed, **params) -> (X, y)`. Optionally set `TASK`, `FEATURE_SHAPE`
and `DESCRIPTION` in the same file. Then reference it from the spec:

```yaml
simulator: {name: python, params: {path: examples/heat_diffusion_simulator.py, n_segments: 4}}
```

A full example is `specs/custom_heat_equation.yaml`. To add a method, register a `Method` in
`methods.py` and add a builder branch in `_estimator`.

### Read the results

Each run writes to `runs/<name>_<timestamp>/`:

| File | Contents |
|---|---|
| `report.md` | Start here: outcome, leaderboard, multi-fidelity trace, lineage, failures, compute, discussion |
| `final/model.pkl`, `model_card.md`, `manifest.json` | The trained model |
| `leaderboard.csv` | All full-data results |
| `artifacts.jsonl` | The complete research record |
| `status.json` | Live progress (stage, best so far, budget used) |
| `harness.log` | Run log |

```bash
python -m mlharness status runs/<run>                      # live progress
python -m mlharness trace  runs/<run> [-v]                 # lineage of the final model
python -m mlharness trace  runs/<run> --idea idea-031      # everything about one idea
python -m mlharness report runs/<run> --rebuild [--backend claude_code]   # regenerate the report from artifacts
python -m mlharness predict runs/<run> --input runs/<run>/data/dataset.npz --key X_test --output preds.npy
```

To use the model in Python:

```python
import sys; sys.path.insert(0, r"...\Automated ML Research Harness V1")
from mlharness.export import load_model
model = load_model(r"...\runs\graph_fpt_weights_claude")
y_hat = model.predict(X)    # X: raw simulator features, same layout as training
```

### Run the tests

```bash
python -m unittest discover -s tests -v     # 14 tests, ~45 s
```
