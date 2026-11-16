# Research report: heat_random_s0

Run `heat_random_s0_20261108_151141` · generated 2026-11-08 15:15:49 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-003, idea-024, idea-014, idea-023, idea-008 (weights [2.0, 2.0, 1.0, 1.0, 1.0])
  - `idea-003` random torch_mlp on 'sqrt' (standard, pca=32) — method `torch_mlp`, features `sqrt`, params `{"hidden": [512, 512], "dropout": 0.2188489682951995, "lr": 0.0006353309749933134, "weight_decay": 0.005499075661406088, "epochs": 262, "batch_size": 64, "ac...`
  - `idea-024` random torch_mlp on 'raw' (robust, pca=0) — method `torch_mlp`, features `raw`, params `{"hidden": [512, 256, 128], "dropout": 0.004797518857071592, "lr": 0.0013169992741853, "weight_decay": 0.00011246953330982432, "epochs": 365, "batch_size": 2...`
  - `idea-014` random kernel_ridge on 'sqrt' (standard, pca=64) — method `kernel_ridge`, features `sqrt`, params `{"alpha": 0.002674921484451008, "gamma_mult": 0.062237505553654315}`
  - `idea-023` random torch_mlp on 'raw' (robust, pca=0) — method `torch_mlp`, features `raw`, params `{"hidden": [512, 512], "dropout": 0.13282584869235944, "lr": 0.002372605134687588, "weight_decay": 1.452254880825563e-06, "epochs": 239, "batch_size": 256, "...`
  - `idea-008` random mlp_sklearn on 'sqrt' (none, pca=32) — method `mlp_sklearn`, features `sqrt`, params `{"hidden": [512, 256], "alpha": 3.5580281338759494e-05, "learning_rate_init": 0.0009368514162695851, "max_iter": 300}`
- Validation rel_l2 (selection estimate): **0.025742**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.024132**, r2 = **0.99476**, rmse = **0.031753**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s0\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\random_s0")`
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
| 1 | idea-003 | random | random | torch_mlp | sqrt | `{"hidden": [512, 512], "dropout": 0.2188489682951995, "lr...` | 0.028213 | 0.99267 | 0.037206 | 30.2 |
| 2 | idea-024 | random | random | torch_mlp | raw | `{"hidden": [512, 256, 128], "dropout": 0.0047975188570715...` | 0.029554 | 0.99196 | 0.038974 | 12.2 |
| 3 | idea-023 | random | random | torch_mlp | raw | `{"hidden": [512, 512], "dropout": 0.13282584869235944, "l...` | 0.029984 | 0.99172 | 0.039541 | 11.6 |
| 4 | idea-014 | random | random | kernel_ridge | sqrt | `{"alpha": 0.002674921484451008, "gamma_mult": 0.062237505...` | 0.030184 | 0.99161 | 0.039805 | 0.457 |
| 5 | idea-008 | random | random | mlp_sklearn | sqrt | `{"hidden": [512, 256], "alpha": 3.5580281338759494e-05, "...` | 0.031168 | 0.99105 | 0.041102 | 14 |
| 6 | idea-018 | random | random | torch_mlp | raw | `{"hidden": [128, 128], "dropout": 0.22731865359248743, "l...` | 0.031826 | 0.99068 | 0.04197 | 18.6 |
| 7 | idea-025 | random | random | mlp_sklearn | log | `{"hidden": [512, 256], "alpha": 0.00014174040186245538, "...` | 0.033333 | 0.98977 | 0.043958 | 7.36 |
| 8 | idea-002 | random | random | mlp_sklearn | raw | `{"hidden": [512, 256], "alpha": 2.233230594054856e-05, "l...` | 0.033882 | 0.98943 | 0.044682 | 20 |
| 9 | idea-019 | random | random | hist_gbm | sqrt | `{"learning_rate": 0.23810316798050457, "max_iter": 151, "...` | 0.060122 | 0.96676 | 0.079285 | 2.09 |
| 10 | idea-027 | random | random | pls | log | `{"n_components": 49}` | 0.060358 | 0.96656 | 0.079596 | 0.816 |
| 11 | idea-021 | random | random | pls | sqrt | `{"n_components": 32}` | 0.061321 | 0.9654 | 0.080867 | 1.46 |
| 12 | idea-028 | random | random | ridge | log | `{"alpha": 0.0003942674553437844}` | 0.082998 | 0.93668 | 0.10945 | 0.0267 |
| 13 | idea-026 | random | random | random_forest | raw | `{"n_estimators": 172, "max_depth": null, "max_features": ...` | 0.089148 | 0.92685 | 0.11756 | 6.38 |
| 14 | idea-029 | random | random | mlp_sklearn | raw | `{"hidden": [512, 256], "alpha": 0.0624292637947709, "lear...` | 0.12173 | 0.86334 | 0.16053 | 5.96 |
| 15 | idea-022 | random | random | kernel_ridge | log | `{"alpha": 0.0002478777342785648, "gamma_mult": 16.1258388...` | 0.19901 | 0.63701 | 0.26244 | 0.471 |
| 16 | idea-020 | random | random | ridge | raw | `{"alpha": 59.25988607238994}` | 0.28865 | 0.2357 | 0.38066 | 0.00896 |
| 17 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | mlp_sklearn | 0.06997 | 0.04255 | 0.03388 | evaluated at full fidelity |
| idea-003 | torch_mlp | 0.0521 | 0.03352 | 0.02821 | FINAL MODEL |
| idea-004 | ridge | 0.1429 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | random_forest | 0.1754 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | random_forest | 0.2106 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | extra_trees | 0.2468 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | mlp_sklearn | 0.06055 | 0.03624 | 0.03117 | FINAL MODEL |
| idea-009 | svm | 0.07502 | - | - | projected full-data cost 213s exceeds the 120s per-experiment limit |
| idea-010 | svm | 0.203 | - | - | projected full-data cost 213s exceeds the 120s per-experiment limit |
| idea-011 | hist_gbm | 0.1478 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-012 | ridge | 0.0797 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | pls | 0.06871 | 0.06205 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-014 | kernel_ridge | 0.04308 | 0.03416 | 0.03018 | FINAL MODEL |
| idea-015 | kernel_ridge | 0.2121 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-016 | extra_trees | 0.1811 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-017 | knn | 0.1729 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.03183 | evaluated at full fidelity |
| idea-019 | hist_gbm | - | - | 0.06012 | evaluated at full fidelity |
| idea-020 | ridge | - | - | 0.2887 | evaluated at full fidelity |
| idea-021 | pls | - | - | 0.06132 | evaluated at full fidelity |
| idea-022 | kernel_ridge | - | - | 0.199 | evaluated at full fidelity |
| idea-023 | torch_mlp | - | - | 0.02998 | FINAL MODEL |
| idea-024 | torch_mlp | - | - | 0.02955 | FINAL MODEL |
| idea-025 | mlp_sklearn | - | - | 0.03333 | evaluated at full fidelity |
| idea-026 | random_forest | - | - | 0.08915 | evaluated at full fidelity |
| idea-027 | pls | - | - | 0.06036 | evaluated at full fidelity |
| idea-028 | ridge | - | - | 0.083 | evaluated at full fidelity |
| idea-029 | mlp_sklearn | - | - | 0.1217 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-003 (random/random: random torch_mlp on 'sqrt' (standard, pca=32))
- Idea ancestry: idea-024 (random/random: random torch_mlp on 'raw' (robust, pca=0))
- Idea ancestry: idea-014 (random/random: random kernel_ridge on 'sqrt' (standard, pca=64))
- Idea ancestry: idea-023 (random/random: random torch_mlp on 'raw' (robust, pca=0))
- Idea ancestry: idea-008 (random/random: random mlp_sklearn on 'sqrt' (none, pca=32))

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00212` [checkpoint] by scheduler: final model (ensemble)
- `resu-00209` [result] by executor: exp-0039 rel_l2=0.02413 r2=0.9948 rmse=0.03175
- `hypo-00013` [hypothesis] by ideator-random: random torch_mlp on 'sqrt' (standard, pca=32)
- `hypo-00174` [hypothesis] by ideator-random: random torch_mlp on 'raw' (robust, pca=0)
- `hypo-00024` [hypothesis] by ideator-random: random kernel_ridge on 'sqrt' (standard, pca=64)
- `hypo-00146` [hypothesis] by ideator-random: random torch_mlp on 'raw' (robust, pca=0)
- `hypo-00018` [hypothesis] by ideator-random: random mlp_sklearn on 'sqrt' (none, pca=32)
- `plan-00208` [plan] by engineer: exp-0039: final fit of 5 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_random_s0 (c272aea7b8)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-009 | svm | projected full-data cost 213s exceeds the 120s per-experiment limit |
| idea-010 | svm | projected full-data cost 213s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-009: train rel_l2=0.02876 vs val 0.07502: strong overfitting, more regularisation or data needed
- idea-019: train rel_l2=0.02296 vs val 0.06012: strong overfitting, more regularisation or data needed
- idea-022: train rel_l2=5.08e-05 vs val 0.199: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.024132286692880763, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 422.1 / 400 worker-seconds over 39 jobs; wall time 248.6s
- By stage: baseline 2.2s, screen 57.4s, halving 123.2s, refine 93.5s, final 145.7s
- Most expensive ideas: ensemble (final fit) 145.7s, idea-003 (torch_mlp) 64.6s, idea-002 (mlp_sklearn) 39.9s, idea-008 (mlp_sklearn) 33.0s, idea-018 (torch_mlp) 21.1s, idea-024 (torch_mlp) 14.7s, idea-023 (torch_mlp) 14.1s, idea-025 (mlp_sklearn) 9.4s
- LLM: 0 calls, 0.0s, $0.0000 reported cost

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **ideator-random** (ideation, persona random) — none

## 11. Discussion

Best validation rel_l2 per method: torch_mlp 0.02821, kernel_ridge 0.03018, mlp_sklearn 0.03117, hist_gbm 0.06012, pls 0.06036, ridge 0.083, random_forest 0.08915, mean_baseline 0.3307. 28 ideas were proposed; 2 were pruned by the critic or the feasibility screen and 16 were trained on the full training set.
