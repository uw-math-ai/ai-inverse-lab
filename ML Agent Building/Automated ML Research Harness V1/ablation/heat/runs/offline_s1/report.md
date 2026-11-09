# Research report: heat_offline_s1

Run `heat_offline_s1_20261108_152931` · generated 2026-11-08 15:33:30 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-024, idea-026, idea-025 (weights [2.0, 2.0, 1.0])
  - `idea-024` torch_mlp refinement of idea-018: {"dropout": 0.09927449747328324, "lr": 0.0006836807757312639, "activation": "sil — method `torch_mlp`, features `log`, params `{"hidden": [256, 256], "dropout": 0.09927449747328324, "lr": 0.0006836807757312639, "weight_decay": 3.613829647016976e-05, "epochs": 318, "batch_size": 256, ...`
  - `idea-026` torch_mlp refinement of idea-021: {"epochs": 147, "activation": "silu"} — method `torch_mlp`, features `sqrt`, params `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 147, "batch_size": 128, "activation": "silu"}`
  - `idea-025` torch_mlp refinement of idea-018: {"lr": 0.0005437287693891764} — method `torch_mlp`, features `log`, params `{"hidden": [256, 256], "dropout": 0.10715680130128409, "lr": 0.0005437287693891764, "weight_decay": 3.613829647016976e-05, "epochs": 318, "batch_size": 256, ...`
- Validation rel_l2 (selection estimate): **0.025946**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.024376**, r2 = **0.99464**, rmse = **0.032073**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s1\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s1")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 14 |
| screen_survivors | 13 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 17 |
| refine_approved | 17 |
| ideas pruned (all reasons) | 3 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-024 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.09927449747328324, "l...` | 0.026844 | 0.99336 | 0.035401 | 27.7 |
| 2 | idea-026 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.027167 | 0.9932 | 0.035827 | 20.2 |
| 3 | idea-025 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.10715680130128409, "l...` | 0.02777 | 0.9929 | 0.036621 | 36.9 |
| 4 | idea-018 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.10715680130128409, "l...` | 0.028235 | 0.99266 | 0.037235 | 24.5 |
| 5 | idea-021 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.028457 | 0.99254 | 0.037527 | 20 |
| 6 | idea-030 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.09927449747328324, "l...` | 0.02851 | 0.99252 | 0.037597 | 32.7 |
| 7 | idea-023 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0010562166...` | 0.029311 | 0.99209 | 0.038653 | 23.2 |
| 8 | idea-011 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.10715680130128409, "l...` | 0.02948 | 0.992 | 0.038877 | 21.3 |
| 9 | idea-019 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.10715680130128409, "l...` | 0.029774 | 0.99185 | 0.039264 | 21.6 |
| 10 | idea-016 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.01, "gamma_mult": 1.0}` | 0.031962 | 0.99059 | 0.042149 | 0.454 |
| 11 | idea-022 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.019901968507694746, "gamma_mult": 1.0}` | 0.032132 | 0.99049 | 0.042373 | 0.466 |
| 12 | idea-007 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.032246 | 0.99043 | 0.042525 | 12.4 |
| 13 | idea-010 | neural | heuristic | mlp_sklearn | log | `{"hidden": [256, 256], "alpha": 0.0001, "learning_rate_in...` | 0.033007 | 0.98997 | 0.043527 | 6.85 |
| 14 | idea-020 | classical | heuristic | kernel_ridge | raw | `{"alpha": 0.01072328709527472, "gamma_mult": 2.4271970080...` | 0.04115 | 0.98441 | 0.054267 | 0.454 |
| 15 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.001 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | ridge | 0.07546 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | pls | 0.09008 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-016 | kernel_ridge | 0.06288 | 0.04196 | 0.03196 | evaluated at full fidelity |
| idea-017 | svm | 0.08846 | - | - | projected full-data cost 192s exceeds the 120s per-experiment limit |
| idea-003 | extra_trees | 0.1356 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | random_forest | 0.149 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | hist_gbm | 0.1021 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | extra_trees | 0.1784 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | torch_mlp | 0.06526 | 0.04078 | 0.03225 | evaluated at full fidelity |
| idea-010 | mlp_sklearn | 0.05902 | 0.04205 | 0.03301 | evaluated at full fidelity |
| idea-011 | torch_mlp | 0.04915 | 0.03586 | 0.02948 | evaluated at full fidelity |
| idea-012 | mlp_sklearn | 0.06066 | 0.04687 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-013 | kernel_ridge | 0.06695 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-015 | extra_trees | 0.1362 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.02824 | evaluated at full fidelity |
| idea-019 | torch_mlp | - | - | 0.02977 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02846 | evaluated at full fidelity |
| idea-023 | torch_mlp | - | - | 0.02931 | evaluated at full fidelity |
| idea-020 | kernel_ridge | - | - | 0.04115 | evaluated at full fidelity |
| idea-022 | kernel_ridge | - | - | 0.03213 | evaluated at full fidelity |
| idea-024 | torch_mlp | - | - | 0.02684 | FINAL MODEL |
| idea-025 | torch_mlp | - | - | 0.02777 | FINAL MODEL |
| idea-026 | torch_mlp | - | - | 0.02717 | FINAL MODEL |
| idea-030 | torch_mlp | - | - | 0.02851 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-024 (neural/heuristic: torch_mlp refinement of idea-018: {"dropout": 0.099274497473)  ←  idea-018 (neural/heuristic: torch_mlp refinement of idea-011: {"lr": 0.00065550667241781)  ←  idea-011 (neural/heuristic: torch_mlp on 'log' features)
- Idea ancestry: idea-026 (neural/heuristic: torch_mlp refinement of idea-021: {"epochs": 147, "activatio)  ←  idea-021 (neural/heuristic: torch_mlp refinement of idea-007: {})  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)
- Idea ancestry: idea-025 (neural/heuristic: torch_mlp refinement of idea-018: {"lr": 0.00054372876938917)  ←  idea-018 (neural/heuristic: torch_mlp refinement of idea-011: {"lr": 0.00065550667241781)  ←  idea-011 (neural/heuristic: torch_mlp on 'log' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00222` [checkpoint] by scheduler: final model (ensemble)
- `resu-00219` [result] by executor: exp-0035 rel_l2=0.02438 r2=0.9946 rmse=0.03207
- `hypo-00171` [hypothesis] by ideator-neural: torch_mlp refinement of idea-018: {"dropout": 0.09927449747328324, "lr": 0.0006836807757312639, "activation": "sil
- `hypo-00173` [hypothesis] by ideator-neural: torch_mlp refinement of idea-021: {"epochs": 147, "activation": "silu"}
- `hypo-00172` [hypothesis] by ideator-neural: torch_mlp refinement of idea-018: {"lr": 0.0005437287693891764}
- `plan-00218` [plan] by engineer: exp-0035: final fit of 3 member(s) on train+val
- `hypo-00139` [hypothesis] by ideator-neural: torch_mlp refinement of idea-011: {"lr": 0.0006555066724178137, "batch_size": 256, "activation": "relu"}
- `spec-00001` [spec] by scheduler: research spec heat_offline_s1 (17f4af1d19)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00142` [hypothesis] by ideator-neural: torch_mlp refinement of idea-007: {}
- `hypo-00022` [hypothesis] by ideator-neural: torch_mlp on 'log' features
- `hypo-00020` [hypothesis] by ideator-neural: torch_mlp on 'raw' features

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-009 | ridge | critic: duplicate of an existing configuration |
| idea-014 | torch_mlp | critic: duplicate of an existing configuration |
| idea-017 | svm | projected full-data cost 192s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-006: train rel_l2=0.03641 vs val 0.1021: strong overfitting, more regularisation or data needed
- idea-013: train rel_l2=0.00797 vs val 0.06695: strong overfitting, more regularisation or data needed
- idea-020: train rel_l2=0.005001 vs val 0.04115: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.02437556163434319, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 455.7 / 400 worker-seconds over 35 jobs; wall time 239.5s
- By stage: baseline 2.1s, screen 51.4s, halving 85.0s, refine 233.6s, final 83.7s
- Most expensive ideas: ensemble (final fit) 83.7s, idea-011 (torch_mlp) 48.2s, idea-025 (torch_mlp) 39.7s, idea-030 (torch_mlp) 35.8s, idea-007 (torch_mlp) 34.6s, idea-024 (torch_mlp) 30.5s, idea-018 (torch_mlp) 26.9s, idea-023 (torch_mlp) 25.7s
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

Best validation rel_l2 per method: torch_mlp 0.02684, kernel_ridge 0.03196, mlp_sklearn 0.03301, mean_baseline 0.3307. 33 ideas were proposed; 3 were pruned by the critic or the feasibility screen and 14 were trained on the full training set.
