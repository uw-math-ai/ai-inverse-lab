# Research report: heat_offline_s0

Run `heat_offline_s0_20261108_150822` · generated 2026-11-08 15:11:36 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

## 1. Research specification

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time.

**Objectives.**
- Learn a regression model from simulated data (python) that solves: Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.
- Select methods by validation rel_l2 only; the test split is used exactly once, at the end
- Success criterion: test rel_l2 <= 0.05
- Stay within 400 worker-seconds of compute and 1700 s wall time
- Eliminate weak approaches with cheap feasibility checks before spending substantial compute
- Constraint: Models may only be trained on simulated data; the test split is never used for selection

## 2. Outcome

- Final model: **ensemble** of idea-030, idea-018, idea-021 (weights [2.0, 1.0, 1.0])
  - `idea-030` torch_mlp refinement of idea-018: {"weight_decay": 2.370688972464005e-06, "activation": "silu"} — method `torch_mlp`, features `sqrt`, params `{"hidden": [256, 256], "dropout": 0.08954734302423699, "lr": 0.0015948143970774696, "weight_decay": 2.370688972464005e-06, "epochs": 157, "batch_size": 128, ...`
  - `idea-018` torch_mlp refinement of idea-012: {"epochs": 157} — method `torch_mlp`, features `sqrt`, params `{"hidden": [256, 256], "dropout": 0.08954734302423699, "lr": 0.0015948143970774696, "weight_decay": 2.331652194583013e-06, "epochs": 157, "batch_size": 128, ...`
  - `idea-021` torch_mlp refinement of idea-007: {"weight_decay": 0.00012028897186395869, "epochs": 175} — method `torch_mlp`, features `raw`, params `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.00012028897186395869, "epochs": 175, "batch_size": 128, "activation": "gelu"}`
- Validation rel_l2 (selection estimate): **0.025099**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.023446**, r2 = **0.99505**, rmse = **0.03085**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s0\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s0")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 12 |
| screen_survivors | 11 |
| rung_0.33 | 4 |
| rung_1.00 | 4 |
| refine_proposed | 17 |
| refine_approved | 17 |
| ideas pruned (all reasons) | 5 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-030 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.025676 | 0.99393 | 0.033859 | 16.5 |
| 2 | idea-018 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.026638 | 0.99347 | 0.035129 | 40.5 |
| 3 | idea-024 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.026638 | 0.99347 | 0.035129 | 22.4 |
| 4 | idea-019 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.02745 | 0.99306 | 0.036199 | 30.8 |
| 5 | idea-021 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.028827 | 0.99235 | 0.038016 | 24.2 |
| 6 | idea-012 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.029085 | 0.99221 | 0.038356 | 24.3 |
| 7 | idea-023 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.029454 | 0.99202 | 0.038841 | 19.8 |
| 8 | idea-007 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.029522 | 0.99198 | 0.038931 | 15.4 |
| 9 | idea-020 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.010483369513838214, "gamma_mult": 1.0}` | 0.031959 | 0.99059 | 0.042146 | 0.516 |
| 10 | idea-004 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.01, "gamma_mult": 1.0}` | 0.031962 | 0.99059 | 0.042149 | 0.45 |
| 11 | idea-011 | neural | heuristic | mlp_sklearn | log | `{"hidden": [256, 256], "alpha": 0.0001, "learning_rate_in...` | 0.033321 | 0.98978 | 0.043942 | 8.82 |
| 12 | idea-022 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.01, "gamma_mult": 1.5314062538082127}` | 0.036129 | 0.98798 | 0.047645 | 0.516 |
| 13 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.00452 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | ridge | 0.07546 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-003 | pls | 0.06909 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-004 | kernel_ridge | 0.06288 | 0.04196 | 0.03196 | evaluated at full fidelity |
| idea-006 | svm | 0.08846 | - | - | projected full-data cost 198s exceeds the 120s per-experiment limit |
| idea-005 | extra_trees | 0.1363 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-008 | random_forest | 0.1466 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-010 | hist_gbm | 0.1001 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-017 | extra_trees | 0.184 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-007 | torch_mlp | 0.06144 | 0.03952 | 0.02952 | evaluated at full fidelity |
| idea-011 | mlp_sklearn | 0.05789 | 0.0432 | 0.03332 | evaluated at full fidelity |
| idea-012 | torch_mlp | 0.05085 | 0.03463 | 0.02909 | evaluated at full fidelity |
| idea-013 | mlp_sklearn | 0.06807 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.02664 | FINAL MODEL |
| idea-019 | torch_mlp | - | - | 0.02745 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02883 | FINAL MODEL |
| idea-023 | torch_mlp | - | - | 0.02945 | evaluated at full fidelity |
| idea-020 | kernel_ridge | - | - | 0.03196 | evaluated at full fidelity |
| idea-022 | kernel_ridge | - | - | 0.03613 | evaluated at full fidelity |
| idea-024 | torch_mlp | - | - | 0.02664 | evaluated at full fidelity |
| idea-030 | torch_mlp | - | - | 0.02568 | FINAL MODEL |

## 6. Lineage of the final model

- Idea ancestry: idea-030 (neural/heuristic: torch_mlp refinement of idea-018: {"weight_decay": 2.3706889)  ←  idea-018 (neural/heuristic: torch_mlp refinement of idea-012: {"epochs": 157})  ←  idea-012 (neural/heuristic: torch_mlp on 'log' features)
- Idea ancestry: idea-018 (neural/heuristic: torch_mlp refinement of idea-012: {"epochs": 157})  ←  idea-012 (neural/heuristic: torch_mlp on 'log' features)
- Idea ancestry: idea-021 (neural/heuristic: torch_mlp refinement of idea-007: {"weight_decay": 0.0001202)  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00208` [checkpoint] by scheduler: final model (ensemble)
- `resu-00205` [result] by executor: exp-0030 rel_l2=0.02345 r2=0.995 rmse=0.03085
- `hypo-00182` [hypothesis] by ideator-neural: torch_mlp refinement of idea-018: {"weight_decay": 2.370688972464005e-06, "activation": "silu"}
- `hypo-00129` [hypothesis] by ideator-neural: torch_mlp refinement of idea-012: {"epochs": 157}
- `hypo-00132` [hypothesis] by ideator-neural: torch_mlp refinement of idea-007: {"weight_decay": 0.00012028897186395869, "epochs": 175}
- `plan-00204` [plan] by engineer: exp-0030: final fit of 3 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_offline_s0 (a82f1715d1)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00022` [hypothesis] by ideator-neural: torch_mlp on 'log' features
- `hypo-00020` [hypothesis] by ideator-neural: torch_mlp on 'raw' features

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-009 | ridge | critic: duplicate of an existing configuration |
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-015 | torch_mlp | critic: duplicate of an existing configuration |
| idea-016 | extra_trees | critic: duplicate of an existing configuration |
| idea-006 | svm | projected full-data cost 198s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-010: train rel_l2=0.04649 vs val 0.1001: strong overfitting, more regularisation or data needed
- idea-013: train rel_l2=0.02585 vs val 0.06807: strong overfitting, more regularisation or data needed
- idea-022: train rel_l2=0.007645 vs val 0.03613: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.023445571386356633, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 369.3 / 400 worker-seconds over 30 jobs; wall time 194.9s
- By stage: baseline 3.7s, screen 49.2s, halving 95.5s, refine 176.5s, final 44.4s
- Most expensive ideas: idea-012 (torch_mlp) 54.7s, ensemble (final fit) 44.4s, idea-018 (torch_mlp) 43.5s, idea-007 (torch_mlp) 41.0s, idea-019 (torch_mlp) 33.6s, idea-021 (torch_mlp) 27.1s, idea-024 (torch_mlp) 25.0s, idea-023 (torch_mlp) 22.8s
- LLM: 0 calls, 0.0s, $0.0000 reported cost

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **ideator-classical** (ideation, persona classical) — none
- **ideator-ensembles** (ideation, persona ensembles) — none
- **ideator-neural** (ideation, persona neural) — none
- **ideator-domain** (ideation, persona domain) — none

## 11. Discussion

Best validation rel_l2 per method: torch_mlp 0.02568, kernel_ridge 0.03196, mlp_sklearn 0.03332, mean_baseline 0.3307. 33 ideas were proposed; 5 were pruned by the critic or the feasibility screen and 12 were trained on the full training set.
