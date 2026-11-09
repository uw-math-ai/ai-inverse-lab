# Research report: heat_random_s1

Run `heat_random_s1_20261108_152006` · generated 2026-11-08 15:26:12 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-016, idea-013, idea-023, idea-011 (weights [3.0, 2.0, 2.0, 1.0])
  - `idea-016` random kernel_ridge on 'log' (standard, pca=0) — method `kernel_ridge`, features `log`, params `{"alpha": 0.0004422906167031346, "gamma_mult": 0.19546364342321004}`
  - `idea-013` random torch_mlp on 'raw' (standard, pca=0) — method `torch_mlp`, features `raw`, params `{"hidden": [512, 256, 128], "dropout": 0.29652243702248987, "lr": 0.002079447741866517, "weight_decay": 0.00011983043170155368, "epochs": 177, "batch_size": ...`
  - `idea-023` random torch_mlp on 'raw' (standard, pca=0) — method `torch_mlp`, features `raw`, params `{"hidden": [512, 512], "dropout": 0.04579938791735105, "lr": 0.0009345938431869064, "weight_decay": 1.0111965972213143e-05, "epochs": 167, "batch_size": 256,...`
  - `idea-011` random kernel_ridge on 'raw' (none, pca=64) — method `kernel_ridge`, features `raw`, params `{"alpha": 2.102783947671146e-05, "gamma_mult": 1.8649835924329932}`
- Validation rel_l2 (selection estimate): **0.024059**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.022792**, r2 = **0.99534**, rmse = **0.02999**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s1\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s1")`
- Stopping reason for the search: compute budget for stage 'refine' exhausted

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 16 |
| screen_survivors | 16 |
| rung_0.33 | 6 |
| rung_1.00 | 4 |
| refine_proposed | 6 |
| refine_approved | 6 |
| ideas pruned (all reasons) | 0 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-016 | random | random | kernel_ridge | log | `{"alpha": 0.0004422906167031346, "gamma_mult": 0.19546364...` | 0.027347 | 0.99311 | 0.036063 | 0.453 |
| 2 | idea-023 | random | random | torch_mlp | raw | `{"hidden": [512, 512], "dropout": 0.04579938791735105, "l...` | 0.027803 | 0.99289 | 0.036665 | 53.6 |
| 3 | idea-013 | random | random | torch_mlp | raw | `{"hidden": [512, 256, 128], "dropout": 0.2965224370224898...` | 0.028893 | 0.99232 | 0.038102 | 81.2 |
| 4 | idea-014 | random | random | mlp_sklearn | raw | `{"hidden": [256, 256], "alpha": 1.4967204889675827e-05, "...` | 0.03163 | 0.99079 | 0.041711 | 4.41 |
| 5 | idea-011 | random | random | kernel_ridge | raw | `{"alpha": 2.102783947671146e-05, "gamma_mult": 1.86498359...` | 0.033358 | 0.98975 | 0.04399 | 0.452 |
| 6 | idea-021 | random | random | hist_gbm | raw | `{"learning_rate": 0.03261371784680624, "max_iter": 469, "...` | 0.047145 | 0.97952 | 0.062172 | 35 |
| 7 | idea-019 | random | random | torch_mlp | sqrt | `{"hidden": [1024, 512], "dropout": 0.28062696302227963, "...` | 0.052415 | 0.97477 | 0.069121 | 66.7 |
| 8 | idea-020 | random | random | extra_trees | sqrt | `{"n_estimators": 341, "max_depth": null, "max_features": ...` | 0.10247 | 0.90332 | 0.13513 | 3.15 |
| 9 | idea-022 | random | random | extra_trees | log | `{"n_estimators": 109, "max_depth": null, "max_features": ...` | 0.12681 | 0.85233 | 0.16723 | 0.241 |
| 10 | idea-018 | random | random | knn | log | `{"n_neighbors": 25, "weights": "distance"}` | 0.16124 | 0.76023 | 0.21264 | 0.04 |
| 11 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.1413 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-003 | kernel_ridge | 0.224 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-004 | mlp_sklearn | 0.2353 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-005 | kernel_ridge | 0.1124 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-006 | torch_mlp | 0.07259 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-007 | pls | 0.07975 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-008 | kernel_ridge | 0.2793 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-009 | ridge | 0.0716 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-010 | pls | 0.0527 | 0.0491 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-011 | kernel_ridge | 0.05481 | 0.03482 | 0.03336 | FINAL MODEL |
| idea-012 | knn | 0.2189 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-013 | torch_mlp | 0.06959 | 0.03684 | 0.02889 | FINAL MODEL |
| idea-014 | mlp_sklearn | 0.05916 | 0.03899 | 0.03163 | evaluated at full fidelity |
| idea-015 | pls | 0.06269 | 0.05909 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-016 | kernel_ridge | 0.04366 | 0.03377 | 0.02735 | FINAL MODEL |
| idea-017 | extra_trees | 0.2405 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-018 | knn | - | - | 0.1612 | evaluated at full fidelity |
| idea-019 | torch_mlp | - | - | 0.05241 | evaluated at full fidelity |
| idea-020 | extra_trees | - | - | 0.1025 | evaluated at full fidelity |
| idea-021 | hist_gbm | - | - | 0.04715 | evaluated at full fidelity |
| idea-022 | extra_trees | - | - | 0.1268 | evaluated at full fidelity |
| idea-023 | torch_mlp | - | - | 0.0278 | FINAL MODEL |

## 6. Lineage of the final model

- Idea ancestry: idea-016 (random/random: random kernel_ridge on 'log' (standard, pca=0))
- Idea ancestry: idea-013 (random/random: random torch_mlp on 'raw' (standard, pca=0))
- Idea ancestry: idea-023 (random/random: random torch_mlp on 'raw' (standard, pca=0))
- Idea ancestry: idea-011 (random/random: random kernel_ridge on 'raw' (none, pca=64))

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00184` [checkpoint] by scheduler: final model (ensemble)
- `resu-00181` [result] by executor: exp-0034 rel_l2=0.02279 r2=0.9953 rmse=0.02999
- `hypo-00026` [hypothesis] by ideator-random: random kernel_ridge on 'log' (standard, pca=0)
- `hypo-00023` [hypothesis] by ideator-random: random torch_mlp on 'raw' (standard, pca=0)
- `hypo-00150` [hypothesis] by ideator-random: random torch_mlp on 'raw' (standard, pca=0)
- `hypo-00021` [hypothesis] by ideator-random: random kernel_ridge on 'raw' (none, pca=64)
- `plan-00180` [plan] by engineer: exp-0034: final fit of 4 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_random_s1 (84678db66e)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|

## 8. Critic notes

- idea-003: train rel_l2=0.0001153 vs val 0.224: strong overfitting, more regularisation or data needed
- idea-021: train rel_l2=0.0178 vs val 0.04715: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.0227921198126852, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 532.8 / 400 worker-seconds over 34 jobs; wall time 365.9s
- By stage: baseline 1.9s, screen 53.9s, halving 141.4s, refine 174.2s, final 161.3s
- Most expensive ideas: ensemble (final fit) 161.3s, idea-013 (torch_mlp) 128.3s, idea-019 (torch_mlp) 69.8s, idea-023 (torch_mlp) 56.5s, idea-021 (hist_gbm) 37.9s, idea-014 (mlp_sklearn) 13.3s, idea-006 (torch_mlp) 11.9s, idea-016 (kernel_ridge) 6.8s
- LLM: 0 calls, 0.0s, $0.0000 reported cost

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **ideator-random** (ideation, persona random) — none

## 11. Discussion

Best validation rel_l2 per method: kernel_ridge 0.02735, torch_mlp 0.0278, mlp_sklearn 0.03163, hist_gbm 0.04715, extra_trees 0.1025, knn 0.1612, mean_baseline 0.3307. 22 ideas were proposed; 0 were pruned by the critic or the feasibility screen and 10 were trained on the full training set.
