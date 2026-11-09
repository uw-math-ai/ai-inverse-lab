# Research report: heat_random_s2

Run `heat_random_s2_20261108_152617` · generated 2026-11-08 15:29:25 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-010, idea-025 (weights [1.0, 1.0])
  - `idea-010` random kernel_ridge on 'raw' (none, pca=64) — method `kernel_ridge`, features `raw`, params `{"alpha": 8.97887840661626e-05, "gamma_mult": 0.022183576308829955}`
  - `idea-025` random torch_mlp on 'log' (robust, pca=32) — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.10915612691301553, "lr": 0.0020211962125925662, "weight_decay": 9.12620611000693e-06, "epochs": 265, "batch_size": 6...`
- Validation rel_l2 (selection estimate): **0.024604**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.024143**, r2 = **0.99476**, rmse = **0.031767**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s2\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s2")`
- Stopping reason for the search: no relative improvement >= 0.005 for 2 round(s)

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 16 |
| screen_survivors | 14 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 12 |
| refine_approved | 12 |
| ideas pruned (all reasons) | 2 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-010 | random | random | kernel_ridge | raw | `{"alpha": 8.97887840661626e-05, "gamma_mult": 0.022183576...` | 0.026724 | 0.99342 | 0.035242 | 0.447 |
| 2 | idea-025 | random | random | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.1091561269130155...` | 0.028617 | 0.99247 | 0.037738 | 33.8 |
| 3 | idea-007 | random | random | mlp_sklearn | log | `{"hidden": [256, 256], "alpha": 0.021882750369179433, "le...` | 0.030869 | 0.99123 | 0.040708 | 4.78 |
| 4 | idea-024 | random | random | kernel_ridge | raw | `{"alpha": 0.02926883988697539, "gamma_mult": 10.882152945...` | 0.032264 | 0.99042 | 0.042548 | 0.429 |
| 5 | idea-019 | random | random | mlp_sklearn | sqrt | `{"hidden": [512, 256], "alpha": 0.0020502617389304523, "l...` | 0.032534 | 0.99025 | 0.042904 | 25 |
| 6 | idea-011 | random | random | hist_gbm | raw | `{"learning_rate": 0.23170783823303182, "max_iter": 578, "...` | 0.04788 | 0.97888 | 0.063141 | 1.44 |
| 7 | idea-028 | random | random | ridge | log | `{"alpha": 1.4628739932328214}` | 0.048114 | 0.97874 | 0.06345 | 0.0253 |
| 8 | idea-023 | random | random | hist_gbm | log | `{"learning_rate": 0.11436622565517462, "max_iter": 598, "...` | 0.049789 | 0.97716 | 0.065658 | 3.68 |
| 9 | idea-026 | random | random | hist_gbm | log | `{"learning_rate": 0.12944410732391298, "max_iter": 322, "...` | 0.050255 | 0.97673 | 0.066273 | 8.45 |
| 10 | idea-009 | random | random | kernel_ridge | log | `{"alpha": 0.0009795235823196612, "gamma_mult": 0.39329157...` | 0.05053 | 0.97646 | 0.066636 | 0.469 |
| 11 | idea-021 | random | random | pls | log | `{"n_components": 25}` | 0.082998 | 0.93668 | 0.10945 | 0.423 |
| 12 | idea-022 | random | random | random_forest | sqrt | `{"n_estimators": 484, "max_depth": 32, "max_features": 0....` | 0.085422 | 0.93281 | 0.11265 | 13.4 |
| 13 | idea-029 | random | random | pls | raw | `{"n_components": 21}` | 0.095903 | 0.91528 | 0.12647 | 0.294 |
| 14 | idea-018 | random | random | hist_gbm | sqrt | `{"learning_rate": 0.11547469281985548, "max_iter": 523, "...` | 0.10981 | 0.8888 | 0.14482 | 1.6 |
| 15 | idea-020 | random | random | svm | raw | `{"C": 539.7201028101159, "gamma": "scale"}` | 0.11249 | 0.88328 | 0.14834 | 85.8 |
| 16 | idea-027 | random | random | mlp_sklearn | sqrt | `{"hidden": [128], "alpha": 1.739377256977852e-06, "learni...` | 0.13325 | 0.8367 | 0.17572 | 6.22 |
| 17 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.00201 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.1219 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-003 | random_forest | 0.1872 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | ridge | 0.1464 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | svm | 0.1066 | - | - | projected full-data cost 198s exceeds the 120s per-experiment limit |
| idea-006 | svm | 0.3031 | - | - | projected full-data cost 198s exceeds the 120s per-experiment limit |
| idea-007 | mlp_sklearn | 0.05307 | 0.04319 | 0.03087 | evaluated at full fidelity |
| idea-008 | knn | 0.1975 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-009 | kernel_ridge | 0.08238 | 0.0553 | 0.05053 | evaluated at full fidelity |
| idea-010 | kernel_ridge | 0.03822 | 0.02987 | 0.02672 | FINAL MODEL |
| idea-011 | hist_gbm | 0.09578 | 0.06468 | 0.04788 | evaluated at full fidelity |
| idea-012 | torch_mlp | 0.1123 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | knn | 0.1698 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-014 | kernel_ridge | 0.1086 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-015 | hist_gbm | 0.2444 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-016 | ridge | 0.07967 | 0.07015 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-017 | hist_gbm | 0.1009 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | hist_gbm | - | - | 0.1098 | evaluated at full fidelity |
| idea-019 | mlp_sklearn | - | - | 0.03253 | evaluated at full fidelity |
| idea-020 | svm | - | - | 0.1125 | evaluated at full fidelity |
| idea-021 | pls | - | - | 0.083 | evaluated at full fidelity |
| idea-022 | random_forest | - | - | 0.08542 | evaluated at full fidelity |
| idea-023 | hist_gbm | - | - | 0.04979 | evaluated at full fidelity |
| idea-024 | kernel_ridge | - | - | 0.03226 | evaluated at full fidelity |
| idea-025 | torch_mlp | - | - | 0.02862 | FINAL MODEL |
| idea-026 | hist_gbm | - | - | 0.05025 | evaluated at full fidelity |
| idea-027 | mlp_sklearn | - | - | 0.1332 | evaluated at full fidelity |
| idea-028 | ridge | - | - | 0.04811 | evaluated at full fidelity |
| idea-029 | pls | - | - | 0.0959 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-010 (random/random: random kernel_ridge on 'raw' (none, pca=64))
- Idea ancestry: idea-025 (random/random: random torch_mlp on 'log' (robust, pca=32))

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00217` [checkpoint] by scheduler: final model (ensemble)
- `resu-00214` [result] by executor: exp-0039 rel_l2=0.02414 r2=0.9948 rmse=0.03177
- `hypo-00020` [hypothesis] by ideator-random: random kernel_ridge on 'raw' (none, pca=64)
- `hypo-00179` [hypothesis] by ideator-random: random torch_mlp on 'log' (robust, pca=32)
- `plan-00213` [plan] by engineer: exp-0039: final fit of 2 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_random_s2 (bebaaf399d)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-005 | svm | projected full-data cost 198s exceeds the 120s per-experiment limit |
| idea-006 | svm | projected full-data cost 198s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-009: train rel_l2=0.02116 vs val 0.0553: strong overfitting, more regularisation or data needed
- idea-011: train rel_l2=0.01695 vs val 0.04788: strong overfitting, more regularisation or data needed
- idea-017: train rel_l2=0.03773 vs val 0.1009: strong overfitting, more regularisation or data needed
- idea-023: train rel_l2=0.01848 vs val 0.04979: strong overfitting, more regularisation or data needed
- idea-026: train rel_l2=0.01601 vs val 0.05025: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.024142615404035676, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 322.5 / 400 worker-seconds over 39 jobs; wall time 187.8s
- By stage: baseline 2.4s, screen 47.3s, halving 27.6s, refine 211.1s, final 34.0s
- Most expensive ideas: idea-020 (svm) 92.7s, idea-025 (torch_mlp) 36.7s, ensemble (final fit) 34.0s, idea-019 (mlp_sklearn) 27.2s, idea-022 (random_forest) 16.1s, idea-007 (mlp_sklearn) 12.4s, idea-026 (hist_gbm) 10.8s, idea-011 (hist_gbm) 8.8s
- LLM: 0 calls, 0.0s, $0.0000 reported cost

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **ideator-random** (ideation, persona random) — none

## 11. Discussion

Best validation rel_l2 per method: kernel_ridge 0.02672, torch_mlp 0.02862, mlp_sklearn 0.03087, hist_gbm 0.04788, ridge 0.04811, pls 0.083, random_forest 0.08542, svm 0.1125, mean_baseline 0.3307. 28 ideas were proposed; 2 were pruned by the critic or the feasibility screen and 16 were trained on the full training set.
