# autodata: an Autodata harness for fptsim inverse problems

`autodata/` is a re-implementation of Meta FAIR's **Autodata**, an agentic data scientist that builds training
data iteratively ([arXiv:2606.25996](https://arxiv.org/abs/2606.25996),
[blog](https://facebookresearch.github.io/RAM/blogs/autodata)). Here it is applied to the inverse problem of
Gao, Li & Yang ([arXiv:2603.20710](https://arxiv.org/abs/2603.20710)): recover the vertex centrality μ on
X∖B from first-passage times observed on B. It builds on `fptsim` and does not change it. Every item is an
fptsim spec plus a seed, so plain fptsim can regenerate it.

```bash
cd "Toggable Simulation of Data Code"
python -m autodata run  --config configs/autodata_small.yaml           # inner loop → dataset (~1–2 min CPU)
python -m autodata show --run results/autodata_small --scenario paper9_grid
python -m autodata meta --config configs/autodata_small.yaml           # outer loop → best_strategy.yaml
python -m autodata run  --config configs/autodata_small.yaml --strategy results/autodata_small/meta/best_strategy.yaml --no-resume
python -m autodata run  --config configs/autodata_small.yaml --llm auto   # LLM challenger (see §5)
python -m pytest tests/test_autodata.py
```

```python
import autodata
items = autodata.load_dataset("results/autodata_small")
a = items[0]["arrays"]   # W, mu_true, B, q, U, dt, T, N, r_exact, counts (A,T,k,k), mu_weak, mu_strong, ...
# ML target: a["mu_true"][a["U"]];  inputs: W, B, q, mu_true[B], dt, counts / N
```

## 1. How Autodata maps onto the inverse problem

| Autodata (CS-paper QA) | here |
|---|---|
| grounding paper | **scenario**: a brief plus a base fptsim spec (`configs/autodata_scenarios.yaml`) |
| challenger writes context + question + rubric | **challenger** writes fptsim toggles: family, size, weights, μ law, B mode, q, T, N |
| QA item | **instance**: a graph, known W, q, μ\|B and h, and noisy first-passage counts r′ on B×B. μ\|X∖B is hidden |
| quality verifier (leakage, recall vs reasoning, rubric) | **verifier**: unknowns exist, enough hit mass, local identifiability (Jacobian condition), Cramér–Rao bound under the chosen N, no duplicate |
| weak solver (Qwen3.5-4B), 3 samples | **weak solver**: least squares on a short horizon (T=4), pulled toward the degree prior (`prior_weight: 0.1`), so vertices the data barely see stay near the prior instead of chasing noise |
| strong solver (Qwen3.5-397B), 3 samples | **strong solver**: least squares on the full horizon with random restarts |
| judge scores answers against the rubric | score = exp(−‖μ̂−μ‖/‖μ‖ / err_ref) on X∖B, per noise draw |
| accept: weak_avg ≤ .65, max_weak ≤ .75, strong_avg ≥ .60, gap ≥ .20 | same gates, same numbers (`gates:`). The paper's `strong < .95` cap is available but off by default |
| strong evaluated only if weak passes | `toggles.lazy_strong` |
| feedback lists TOO EASY / FAILED ON STRONG / FAILED QV | same three categories, with numbers |
| meta-optimization of the harness | `meta`: Boltzmann parent (T=0.1), analyze, mutate, accept only on strictly better validation |

The weak and strong solvers see **the same** `attempts` noise draws. Every seed (graph, noise, solver restarts)
comes from `(seed, scenario id, round, purpose, attempt)` and is owned by the harness. The challenger
cannot set seeds, solvers, gates or limits. Autodata's agents learned to cheat, for example by prompting the
weak solver to be weaker, so those controls sit outside what can be optimized.

## 2. Inner loop (`autodata/loop.py`)

```
for scenario in corpus:
    for round in 1..loop.max_rounds:
        spec   ← challenger(scenario, history)                  # fptsim graphs + simulation blocks
        inst   ← fptsim validate_config + generate_graph + exact_single   (invalid → FAILED_QV)
        verify(inst)                                            (fail   → FAILED_QV with reason)
        draws  ← `attempts` multinomial (or Monte Carlo) samples of N walks per pair
        weak   ← scores on the draws;  weak gate fails          → TOO_EASY
        strong ← scores on the same draws;  strong gate fails   → FAILED_STRONG
        accept → dataset/<scenario>__rNNN.npz;  stop after loop.items_per_scenario items
```

On the shipped corpus (12 scenarios, heuristic challenger, seed 2026), every scenario produced an accepted
item in a mean of 3.4 rounds. The paper reports 6.59 for CS. Of the failed rounds, 34 % were too easy for the
weak solver, 38 % failed on the strong solver, and 28 % were stopped by the verifier, mostly as
noise-dominated. The paper reports 80 % / 13 % for CS. The challenger tends to oscillate: heterogeneous μ
makes the strong solver fail, more walks make the item too easy again. The outer loop is there to tune
that behaviour.

## 3. Outer loop (`autodata/meta.py`)

The optimized object is the **strategy**: the heuristic challenger's knobs (`challenger.DEFAULT_STRATEGY`)
and the LLM challenger's `guidance` text. Scenarios are split in order into train and validation
(`meta.train_fraction`).

1. Pick a parent from the population with probability ∝ exp(score / `meta.temperature`).
2. Run it on a minibatch of train scenarios with a budget of `meta.eval_rounds` rounds each.
3. Analyze the failures into four buckets: too_easy, too_hard, ill_posed, invalid. Optionally an LLM writes
   a root-cause analysis instead.
4. Mutate the strategy. The rule-based mutation targets the dominant bucket and adds one random jitter.
   Optionally an LLM rewrites the strategy JSON, and its values are clipped to `meta.PARAM_RANGES`.
5. Score the child on validation and keep it only if its score **strictly** exceeds the parent's.
   Score = pass rate + 0.01 × speed, where the speed term only breaks ties.

The outputs in `meta/` are `history.jsonl`, `population.json`, `best_strategy.yaml` and `summary.json`.
Pass the best strategy back to `run` with `--strategy`.

## 4. Configuration (`configs/autodata_small.yaml`; defaults in `autodata/config.py`)

Unknown keys are errors. Numeric strings such as `1e8` are accepted.

| block | keys |
|---|---|
| top | `run_name`, `output_dir`, `seed`, `scenarios` (file relative to the config, or inline list), `scenario_ids` |
| `toggles` | `llm_challenger`, `quality_verifier`, `identifiability_check`, `crb_check`, `lazy_strong`, `save_observations`, `resume`, `final_verifier` (dedupe against the dataset) |
| `loop` | `max_rounds`, `items_per_scenario` |
| `evaluation` | `attempts`, `noise` (`multinomial` \| `monte_carlo` \| `exact`), `err_ref` |
| `gates` | `weak_avg_max`, `weak_max_max`, `weak_min_min`, `strong_avg_min`, `strong_avg_max`, `gap_min` |
| `solvers` | `weak` / `strong`: `{solver: lsq \| prior, T, restarts, max_iter}`. Add more with `@autodata.solvers.register` |
| `verifier` | `min_unknowns`, `min_hit_mass`, `max_jac_cond`, `max_crb_rel` |
| `limits` | `max_n`, `max_k`, `max_T` (steps), `min_N`, `max_N`. These are guardrails the challenger cannot move |
| `llm` | `backend` (`auto` \| `none` \| `anthropic` \| `claude_code`), `model` (default `claude-opus-5-5`), `effort`, `timeout` |
| `strategy` | path or mapping that overrides the default strategy |
| `meta` | `iterations`, `temperature`, `train_fraction`, `minibatch`, `eval_rounds`, `llm_implementer` |

A scenario is `{id, brief, base_spec, families?}`. `base_spec` is `{graphs: <fptsim graphs block>,
simulation: {T, T_mode?, N}}`, and `families` lists alternative fptsim `families:` blocks the challenger may
switch to.

## 5. LLM challenger

Set `toggles.llm_challenger: true` or pass `--llm auto`. The model gets the problem statement, the spec schema,
the gates, the limits, the strategy guidance, the last eight rounds with their numbers, and the categorized
feedback. It returns `{"rationale", "spec"}`.

- An **invalid** spec is recorded and fed back as FAILED_QV, as in the paper.
- An unparseable reply or a backend error falls back to the heuristic for that round.

The `anthropic` backend uses the official SDK with adaptive thinking, `output_config.effort` and the server-side
refusal fallback (`fallbacks: "default"`). Install it with `pip install anthropic` and set
`ANTHROPIC_API_KEY`. The `claude_code` backend uses the `claude` CLI in headless mode with tools disabled.
`meta.llm_implementer: true` also lets the model analyze trajectories and rewrite the strategy.

## 6. Files

| file | role |
|---|---|
| `autodata/forward.py` | r(μ) and its exact Jacobian in a = 1/μ (forward mode), Fisher information, Cramér–Rao bound, Jacobian condition |
| `autodata/solvers.py` | `prior`, `lsq` (projected Levenberg–Marquardt, binomial weights), solver registry |
| `autodata/instance.py` | spec checks (fptsim validator + limits), `build_instance`, noise models |
| `autodata/verify.py` | verifier codes, score, weak/strong gates |
| `autodata/challenger.py` | heuristic and LLM challengers, default strategy, feedback lists |
| `autodata/loop.py` | inner loop, run directory (atomic writes, resume), `summarize`, `load_dataset` |
| `autodata/meta.py` | outer meta-optimization loop |
| `autodata/llm.py` | `none` / `anthropic` / `claude_code` backends |

The run directory holds `config.resolved.yaml`, `run.log`, `scenarios/<id>/result.json` (every round: spec,
instance summary, verifier diagnostics, weak and strong scores, challenger note), `dataset/items.jsonl` with
one `.npz` per item, `summary.json`, and `meta/` after a meta run. `items.jsonl` is rebuilt from the
`result.json` files, so an interrupted run never leaves duplicate or half-written items.

## 7. Website agents

The Research Agent Workplace runs agents from the repository's `agents/` folder, on a runtime image that has
only Python 3.12's standard library (see `agents/instructions.md`). This package cannot run there because it
needs NumPy and fptsim, so the harness is also ported to pure Python in `agents/shared/autodata_core/` and
published as two agents:

| agent | challenger | connection | notes |
|---|---|---|---|
| `agents/autodata-inverse` | rule-based | none, free | `mode: meta` for the outer loop; `parallel: true` runs one subagent per scenario |
| `agents/autodata-inverse-llm` | language model through the run's provider (Claude Code, Codex, OpenRouter, OpenAI) | required, charges | rule-based fallback when a call fails; `llm_meta` lets the model drive the meta loop |

The port uses the same spec keys, verifier, solvers, gates and loops. Graphs come from Python's `random`, so
they are statistically equivalent to fptsim's, not bit-identical. Forward values, the Jacobian, the
Cramér–Rao bound and the condition number agree with this package to round-off.
`tests/test_autodata_agent_port.py` checks that agreement, the manifests, and a standard-library-only run.

**Scope.** The paper's boundary-control reconstruction algorithm is not implemented. The strong solver is a
model-based least-squares fit, which is exact on noise-free data (tested to 1e-8) and is a natural slot for
that algorithm or for a trained network (`@register`). The orchestrator is deterministic code. Autodata uses
an LLM there, but here the LLM's judgement is limited to proposing specs and to the meta loop.
