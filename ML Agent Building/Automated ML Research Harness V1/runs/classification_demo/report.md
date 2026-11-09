# Research report: classification_demo

Run `classification_demo_20261107_154748` · generated 2026-11-07 15:48:12 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

## 1. Research specification

**Problem.** Classify samples from a synthetic 3-class problem with 20 features (8 informative).

**Data.** classification simulator with params {"n_features": 20, "n_informative": 8, "n_classes": 3, "class_sep": 1.0, "flip_y": 0.02}

**Objectives.**
- Learn a classification model from simulated data (classification) that solves: Classify samples from a synthetic 3-class problem with 20 features (8 informative).
- Select methods by validation accuracy only; the test split is used exactly once, at the end
- Success criterion: test accuracy >= 0.9
- Stay within 240 worker-seconds of compute and 1320 s wall time
- Eliminate weak approaches with cheap feasibility checks before spending substantial compute

## 2. Outcome

- Final model: **single** of idea-014
  - `idea-014` extra_trees refinement of idea-004: {"n_estimators": 284, "min_samples_leaf": 2} — method `extra_trees`, features `raw`, params `{"n_estimators": 284, "max_depth": null, "max_features": 0.5, "min_samples_leaf": 2}`
- Validation accuracy (selection estimate): **0.88**; baseline (mean predictor): 0.334
- **Test metrics (held-out, evaluated once):** accuracy = **0.872**, f1_macro = **0.87125**
- Success criterion (accuracy >= 0.9): **NOT MET**
- Model file: `runs\classification_demo\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"runs\classification_demo")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 12 |
| prescreen_approved | 11 |
| screen_survivors | 11 |
| rung_0.33 | 4 |
| rung_1.00 | 3 |
| refine_proposed | 6 |
| refine_approved | 6 |
| ideas pruned (all reasons) | 1 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val accuracy | val f1_macro | fit s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-014 | ensembles | heuristic | extra_trees | raw | `{"n_estimators": 284, "max_depth": null, "max_features": ...` | 0.88 | 0.87991 | 0.699 |
| 2 | idea-004 | ensembles | heuristic | extra_trees | raw | `{"n_estimators": 300, "max_depth": null, "max_features": ...` | 0.87 | 0.86945 | 0.79 |
| 3 | idea-015 | ensembles | heuristic | extra_trees | raw | `{"n_estimators": 300, "max_depth": null, "max_features": ...` | 0.87 | 0.86945 | 0.766 |
| 4 | idea-007 | ensembles | heuristic | random_forest | raw | `{"n_estimators": 200, "max_depth": null, "max_features": ...` | 0.868 | 0.86783 | 1.4 |
| 5 | idea-017 | ensembles | heuristic | random_forest | raw | `{"n_estimators": 174, "max_depth": null, "max_features": ...` | 0.868 | 0.86789 | 1.32 |
| 6 | idea-019 | ensembles | heuristic | random_forest | raw | `{"n_estimators": 200, "max_depth": null, "max_features": ...` | 0.862 | 0.86188 | 1.6 |
| 7 | idea-013 | domain | heuristic | knn | raw | `{"n_neighbors": 10, "weights": "distance"}` | 0.802 | 0.80197 | 0.00309 |
| 8 | idea-016 | domain | heuristic | knn | raw | `{"n_neighbors": 7, "weights": "distance"}` | 0.79 | 0.7901 | 0.00402 |
| 9 | idea-018 | domain | heuristic | knn | sqrt | `{"n_neighbors": 10, "weights": "distance"}` | 0.49 | 0.49047 | 0.00401 |
| 10 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.334 | 0.16692 | 0.00953 |

## 5. Multi-fidelity trace (validation accuracy by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.334 | reference baseline |
| idea-002 | logistic | 0.686 | 0.724 | - | not_promoted: ranked below the top 3 at fraction 0.33 |
| idea-003 | svm | 0.392 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-005 | knn | 0.384 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-004 | extra_trees | 0.798 | 0.838 | 0.87 | evaluated at full fidelity |
| idea-007 | random_forest | 0.734 | 0.814 | 0.868 | evaluated at full fidelity |
| idea-009 | hist_gbm | 0.428 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-006 | torch_mlp | 0.68 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-010 | mlp_sklearn | 0.368 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-012 | torch_mlp | 0.354 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-011 | extra_trees | 0.424 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-013 | knn | 0.726 | 0.762 | 0.802 | evaluated at full fidelity |
| idea-014 | extra_trees | - | - | 0.88 | FINAL MODEL |
| idea-015 | extra_trees | - | - | 0.87 | evaluated at full fidelity |
| idea-017 | random_forest | - | - | 0.868 | evaluated at full fidelity |
| idea-019 | random_forest | - | - | 0.862 | evaluated at full fidelity |
| idea-016 | knn | - | - | 0.79 | evaluated at full fidelity |
| idea-018 | knn | - | - | 0.49 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-014 (ensembles/heuristic: extra_trees refinement of idea-004: {"n_estimators": 284, "m)  ←  idea-004 (ensembles/heuristic: extra_trees on 'raw' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00150` [checkpoint] by scheduler: final model (single)
- `resu-00147` [result] by executor: exp-0026 accuracy=0.872 f1_macro=0.8712
- `hypo-00112` [hypothesis] by ideator-ensembles: extra_trees refinement of idea-004: {"n_estimators": 284, "min_samples_leaf": 2}
- `plan-00146` [plan] by engineer: exp-0026: final fit of 1 member(s) on train+val
- `hypo-00017` [hypothesis] by ideator-ensembles: extra_trees on 'raw' features
- `spec-00001` [spec] by scheduler: research spec classification_demo (d3bc6b6e9a)
- `data-00004` [dataset] by simulator: classification: 3000 samples, 20 features, target dim 1

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-008 | torch_mlp | critic: duplicate of an existing configuration |

## 8. Critic notes

- idea-003: train accuracy=0.845 vs val 0.392: strong overfitting, more regularisation or data needed
- idea-009: train accuracy=0.86 vs val 0.428: strong overfitting, more regularisation or data needed
- idea-010: train accuracy=0.64 vs val 0.368: strong overfitting, more regularisation or data needed
- final audit: test accuracy=0.872, success=False, issues=0

## 9. Compute and resource usage

- Compute used: 74.8 / 240 worker-seconds over 26 jobs; wall time 24.1s
- By stage: baseline 2.0s, screen 32.5s, halving 18.7s, refine 18.5s, final 3.2s
- Most expensive ideas: idea-007 (random_forest) 9.0s, idea-004 (extra_trees) 8.4s, idea-013 (knn) 6.4s, idea-014 (extra_trees) 6.2s, idea-006 (torch_mlp) 5.9s, idea-012 (torch_mlp) 5.9s, idea-002 (logistic) 4.2s, idea-019 (random_forest) 4.0s
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

Best validation accuracy per method: extra_trees 0.88, random_forest 0.868, knn 0.802, mean_baseline 0.334. 18 ideas were proposed; 1 were pruned by the critic or the feasibility screen and 9 were trained on the full training set.
