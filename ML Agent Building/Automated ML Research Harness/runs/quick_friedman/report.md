# Research report: quick_friedman

Run `quick_friedman_20261107_154445` · generated 2026-11-07 15:45:05 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

## 1. Research specification

**Problem.** Learn the Friedman #1 nonlinear regression function from noisy samples.

**Data.** friedman simulator with params {"n_features": 10, "noise": 1.0}

**Objectives.**
- Learn a regression model from simulated data (friedman) that solves: Learn the Friedman #1 nonlinear regression function from noisy samples.
- Select methods by validation rmse only; the test split is used exactly once, at the end
- Success criterion: test rmse <= 1.5
- Stay within 150 worker-seconds of compute and 350 s wall time
- Eliminate weak approaches with cheap feasibility checks before spending substantial compute

## 2. Outcome

- Final model: **single** of idea-004
  - `idea-004` extra_trees on 'raw' features — method `extra_trees`, features `raw`, params `{"n_estimators": 300, "max_depth": null, "max_features": 0.5, "min_samples_leaf": 1}`
- Validation rmse (selection estimate): **1.7704**; baseline (mean predictor): 4.7136
- **Test metrics (held-out, evaluated once):** rmse = **1.7339**, r2 = **0.87923**
- Success criterion (rmse <= 1.5): **NOT MET**
- Model file: `runs\quick_friedman\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"runs\quick_friedman")`
- Stopping reason for the search: no relative improvement >= 0.005 for 1 round(s)

## 3. Research funnel

| step | count |
|---|---|
| proposed | 8 |
| prescreen_approved | 7 |
| screen_survivors | 7 |
| rung_0.33 | 3 |
| rung_1.00 | 3 |
| refine_proposed | 3 |
| refine_approved | 3 |
| ideas pruned (all reasons) | 1 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rmse | val r2 | fit s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-004 | ensembles | heuristic | extra_trees | raw | `{"n_estimators": 300, "max_depth": null, "max_features": ...` | 1.7704 | 0.85786 | 0.525 |
| 2 | idea-006 | ensembles | heuristic | random_forest | raw | `{"n_estimators": 200, "max_depth": null, "max_features": ...` | 1.8321 | 0.84778 | 0.473 |
| 3 | idea-012 | ensembles | heuristic | random_forest | raw | `{"n_estimators": 200, "max_depth": null, "max_features": ...` | 1.8637 | 0.84249 | 0.441 |
| 4 | idea-010 | ensembles | heuristic | extra_trees | raw | `{"n_estimators": 300, "max_depth": null, "max_features": ...` | 1.9712 | 0.82378 | 0.433 |
| 5 | idea-011 | classical | heuristic | ridge | sqrt | `{"alpha": 0.9422852957949311}` | 2.5303 | 0.70965 | 0.004 |
| 6 | idea-002 | classical | heuristic | ridge | raw | `{"alpha": 1.0}` | 2.6694 | 0.67686 | 0.00498 |
| 7 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 4.7136 | -0.0075891 | 0 |

## 5. Multi-fidelity trace (validation rmse by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 4.714 | reference baseline |
| idea-002 | ridge | 2.736 | 2.7 | 2.669 | evaluated at full fidelity |
| idea-003 | pls | 2.998 | - | - | not_promoted: ranked below the top 3 at fraction 0.10 |
| idea-004 | extra_trees | 2.458 | 2.155 | 1.77 | FINAL MODEL |
| idea-006 | random_forest | 2.506 | 2.15 | 1.832 | evaluated at full fidelity |
| idea-005 | torch_mlp | 2.843 | - | - | not_promoted: ranked below the top 3 at fraction 0.10 |
| idea-008 | mlp_sklearn | 2.924 | - | - | not_promoted: ranked below the top 3 at fraction 0.10 |
| idea-009 | kernel_ridge | 2.922 | - | - | not_promoted: ranked below the top 3 at fraction 0.10 |
| idea-010 | extra_trees | - | - | 1.971 | evaluated at full fidelity |
| idea-012 | random_forest | - | - | 1.864 | evaluated at full fidelity |
| idea-011 | ridge | - | - | 2.53 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-004 (ensembles/heuristic: extra_trees on 'raw' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00104` [checkpoint] by scheduler: final model (single)
- `resu-00101` [result] by executor: exp-0018 rmse=1.734 r2=0.8792
- `hypo-00017` [hypothesis] by ideator-ensembles: extra_trees on 'raw' features
- `plan-00100` [plan] by engineer: exp-0018: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec quick_friedman (291a383e78)
- `data-00004` [dataset] by simulator: friedman: 2500 samples, 10 features, target dim 1

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-007 | ridge | critic: duplicate of an existing configuration |

## 8. Critic notes

- final audit: test rmse=1.7338516639722603, success=False, issues=0

## 9. Compute and resource usage

- Compute used: 44.2 / 150 worker-seconds over 18 jobs; wall time 19.5s
- By stage: baseline 2.0s, screen 19.9s, halving 12.8s, refine 6.6s, final 2.9s
- Most expensive ideas: idea-004 (extra_trees) 9.9s, idea-005 (torch_mlp) 7.4s, idea-006 (random_forest) 6.8s, idea-002 (ridge) 5.6s, idea-010 (extra_trees) 2.4s, idea-012 (random_forest) 2.4s, idea-008 (mlp_sklearn) 2.1s, idea-003 (pls) 2.0s
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

Best validation rmse per method: extra_trees 1.77, random_forest 1.832, ridge 2.53, mean_baseline 4.714. 11 ideas were proposed; 1 were pruned by the critic or the feasibility screen and 6 were trained on the full training set.
