# Autodata: inverse-problem data generator

This agent builds a curated dataset of instances of the vertex-centrality inverse problem from Gao, Li & Yang
([arXiv:2603.20710](https://arxiv.org/abs/2603.20710)). Each instance asks for μ on the unobserved vertices
X∖B, given first-passage times observed on B. The agent follows Meta FAIR's **Autodata** loop
([arXiv:2606.25996](https://arxiv.org/abs/2606.25996)):

1. A **challenger** proposes simulation settings: graph family and size, weights, the μ law, the observed set
   B, the potential q, the horizon T and the number of walks N.
2. A **verifier** rejects an instance before solving if there is nothing to recover, the walks rarely reach B,
   μ is not locally identifiable, or the Cramér–Rao bound says the noise is too large.
3. A **weak solver** (least squares on the first 4 steps, pulled toward a degree prior) and a **strong solver**
   (least squares on the full horizon, with a restart) reconstruct μ from the same 3 noise draws.
   Score = exp(−relative error / 0.1).
4. An instance is kept only if the weak solver fails and the strong one succeeds: weak average ≤ 0.65,
   weak best ≤ 0.75, strong average ≥ 0.60 and gap ≥ 0.20. Otherwise the challenger gets the paper's three
   feedback categories (too easy, strong failed, failed verifier) and tries again.

In this agent the challenger is **rule-based**. `mode: "meta"` runs Autodata's outer loop, which tunes the
challenger's strategy on a train/validation split of the scenarios. The model-driven challenger is
[`autodata-inverse-llm`](../autodata-inverse-llm/README.md).

The code lives in [`agents/shared/autodata_core/`](../shared/autodata_core/). It is a standard-library port of
the NumPy harness in `Toggable Simulation of Data Code/autodata/` (see `AUTODATA.md` there). Specs use
fptsim's keys and meanings, but graphs come from Python's `random`. That makes them statistically equivalent
to fptsim's, not bit-identical.

## Inputs (all optional)

| field | default | meaning |
|---|---|---|
| `mode` | `run` | `run` builds the dataset; `meta` meta-optimizes the challenger strategy |
| `scenario_ids` | all 12 | subset of the bundled scenarios (`shared/autodata_core/scenarios.json`): `paper8_ladder`, `paper9_grid`, `grid_boundary`, `grid_holes`, `tree_leaves`, `path_ends`, `cycle_sparse`, `erdos_renyi_low_degree`, `geometric_killing`, `small_world_hubs`, `scale_free_periphery`, `grid_sparse_potential` |
| `skip_scenarios` | none | scenario ids to skip, for example ones finished before a stop |
| `custom_scenarios` | none | your own corpus: `[{id, brief, base_spec: {graphs: <fptsim graphs block>, simulation: {T, T_mode?, N}}, families?}]` |
| `seed` | `0` | root seed for every graph, noise draw and solver restart |
| `max_rounds` / `items_per_scenario` | `12` / `1` | rounds per scenario / items to collect per scenario |
| `attempts`, `noise`, `err_ref` | `3`, `multinomial`, `0.10` | noise draws per round, noise model (`multinomial` \| `exact`), score scale |
| `gates`, `solvers`, `verifier`, `limits`, `toggles` | see the input form | thresholds, solver settings, verifier thresholds, guardrails, switches |
| `strategy` | built-in | challenger strategy, for example `best_strategy.json` from a meta run |
| `meta` | 6 iterations, 4 rounds | `iterations`, `temperature`, `train_fraction`, `minibatch`, `eval_rounds` |
| `parallel` | `false` | mode `run`: one subagent per scenario, spread round-robin over the servers approved at launch |
| `time_budget_seconds` | about 3 h 55 min | stop starting new work after this long, so the outputs are written before the 4 h timeout |

Sample inputs:

```json
{"scenario_ids": ["paper9_grid", "tree_leaves", "path_ends"], "max_rounds": 8}
```
```json
{"mode": "meta", "meta": {"iterations": 10}}
```
```json
{"parallel": true, "items_per_scenario": 3}
```

## Outputs (in `AGENT_OUTPUT_DIR`)

| file | content |
|---|---|
| `report.md` | summary: items, rounds, failure shares, a per-scenario table |
| `result.json` | machine-readable summary and the resolved configuration |
| `dataset.jsonl` | one accepted instance per line: `spec`, `seed`, `instance` summary, verifier `diagnostics`, `weak`/`strong` scores and `arrays` (`W`, `mu_true`, `B`, `U`, `q`, `dt`, `T`, `N`, `r_exact` and, per attempt, `counts`, `mu_weak`, `mu_strong`). `r_exact` and `counts` are flat in (t, i, j) order. The ML target is `mu_true[U]` |
| `rounds.jsonl` | every round of every scenario: spec, status, reason, scores, challenger note |
| `meta_history.jsonl`, `best_strategy.json` | mode `meta` only |

Logs show one line per round and `{"progress": x}` after each scenario. Invalid input exits with code 2, and
an internal error exits with code 1.

## Platform settings

- Connection, provider and tier: none. This agent makes no model calls and costs nothing.
- Datasets: none. Everything comes from the bundled corpus or from `custom_scenarios`.
- Swarm: `max_children` 10, `max_total` 60, used only with `parallel: true`. Children run this same agent
  with `connection: "none"`. Tick the servers it may use under **Subagents** at launch.
- Recommended server: an AMATH worker. A default run of 12 scenarios takes under a minute on a laptop CPU in
  pure Python. Larger `max_rounds`, `items_per_scenario` or `limits` scale that up, and the VPS is shared.
- Resources: 1 CPU, 512 MiB, 4 h timeout.
- Dependencies: Python 3.12 standard library (`random.binomialvariate` needs 3.12 or later), plus
  `agents/shared/platform_usage.py` and `platform_swarm.py`.

## Local test

From this folder (`-S` proves no site-packages are used):

```sh
mkdir -p /tmp/ad-out && echo '{"scenario_ids": ["paper9_grid"]}' > /tmp/ad-in.json
AGENT_INPUT_PATH=/tmp/ad-in.json AGENT_OUTPUT_DIR=/tmp/ad-out python -S run.py; echo "exit $?"
```

The port is checked against the NumPy harness and fptsim by
`Toggable Simulation of Data Code/tests/test_autodata_agent_port.py`.
