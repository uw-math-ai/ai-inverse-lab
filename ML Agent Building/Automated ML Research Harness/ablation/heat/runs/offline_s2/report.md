# Research report: heat_offline_s2

Run `heat_offline_s2_20261108_153336` · generated 2026-11-08 15:38:07 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-027, idea-024, idea-029, idea-025 (weights [3.0, 1.0, 1.0, 1.0])
  - `idea-027` torch_mlp refinement of idea-007: {"lr": 0.0013598589398165094, "batch_size": 64, "activation": "silu"} — method `torch_mlp`, features `log`, params `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0013598589398165094, "weight_decay": 0.0001, "epochs": 150, "batch_size": 64, "activation": "silu"}`
  - `idea-024` torch_mlp refinement of idea-018: {"dropout": 0.006462707030864809} — method `torch_mlp`, features `raw`, params `{"hidden": [256, 256], "dropout": 0.006462707030864809, "lr": 0.0008725358113262284, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": ...`
  - `idea-029` torch_mlp refinement of idea-007: {"dropout": 0.019379905337065118, "epochs": 156} — method `torch_mlp`, features `raw`, params `{"hidden": [256, 256], "dropout": 0.019379905337065118, "lr": 0.001, "weight_decay": 0.0001, "epochs": 156, "batch_size": 128, "activation": "gelu"}`
  - `idea-025` torch_mlp refinement of idea-018: {} — method `torch_mlp`, features `sqrt`, params `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0008725358113262284, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}`
- Validation rel_l2 (selection estimate): **0.025795**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.023377**, r2 = **0.99508**, rmse = **0.03076**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s2\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\offline_s2")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 13 |
| screen_survivors | 12 |
| rung_0.33 | 4 |
| rung_1.00 | 4 |
| refine_proposed | 17 |
| refine_approved | 17 |
| ideas pruned (all reasons) | 4 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-027 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0013598589...` | 0.027134 | 0.99322 | 0.035782 | 38.8 |
| 2 | idea-025 | neural | heuristic | torch_mlp | sqrt | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0008725358...` | 0.02744 | 0.99307 | 0.036187 | 31.3 |
| 3 | idea-024 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.006462707030864809, "...` | 0.028389 | 0.99258 | 0.037438 | 35.3 |
| 4 | idea-018 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0008725358...` | 0.028521 | 0.99251 | 0.037611 | 23.3 |
| 5 | idea-029 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.019379905337065118, "...` | 0.028526 | 0.99251 | 0.037618 | 36.2 |
| 6 | idea-030 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.0013497256...` | 0.028925 | 0.99229 | 0.038144 | 31 |
| 7 | idea-021 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.016504059684405884, "gamma_mult": 0.698984788...` | 0.030112 | 0.99164 | 0.03971 | 0.484 |
| 8 | idea-028 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.027032475437597764, "gamma_mult": 0.698984788...` | 0.03061 | 0.99137 | 0.040367 | 0.49 |
| 9 | idea-007 | neural | heuristic | torch_mlp | raw | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.030973 | 0.99118 | 0.040845 | 17.5 |
| 10 | idea-005 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.01, "gamma_mult": 1.0}` | 0.031962 | 0.99059 | 0.042149 | 0.441 |
| 11 | idea-019 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.008803454172753173, "gamma_mult": 1.025110473...` | 0.032165 | 0.99047 | 0.042417 | 0.484 |
| 12 | idea-020 | neural | heuristic | torch_mlp | raw | `{"hidden": [128, 128], "dropout": 0.0, "lr": 0.001, "weig...` | 0.032765 | 0.99012 | 0.043208 | 13.8 |
| 13 | idea-014 | neural | heuristic | mlp_sklearn | log | `{"hidden": [256, 256], "alpha": 0.0001, "learning_rate_in...` | 0.033413 | 0.98972 | 0.044063 | 7.15 |
| 14 | idea-026 | classical | heuristic | kernel_ridge | log | `{"alpha": 0.016450486903370594, "gamma_mult": 1.220275468...` | 0.033918 | 0.9894 | 0.044729 | 0.481 |
| 15 | idea-023 | neural | heuristic | mlp_sklearn | log | `{"hidden": [256, 256, 256], "alpha": 0.000160731559820609...` | 0.03462 | 0.98897 | 0.045654 | 8.46 |
| 16 | idea-022 | neural | heuristic | mlp_sklearn | log | `{"hidden": [128], "alpha": 9.46042294757767e-05, "learnin...` | 0.035004 | 0.98873 | 0.046162 | 1.6 |
| 17 | idea-016 | neural | heuristic | mlp_sklearn | raw | `{"hidden": [128], "alpha": 7.2133551041733885e-06, "learn...` | 0.040038 | 0.98524 | 0.0528 | 0.517 |
| 18 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | ridge | 0.07546 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-003 | pls | 0.09008 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-005 | kernel_ridge | 0.06288 | 0.04196 | 0.03196 | evaluated at full fidelity |
| idea-006 | svm | 0.09739 | - | - | projected full-data cost 189s exceeds the 120s per-experiment limit |
| idea-004 | extra_trees | 0.1361 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-009 | random_forest | 0.1478 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-010 | hist_gbm | 0.1014 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-011 | extra_trees | 0.1369 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-007 | torch_mlp | 0.06061 | 0.04011 | 0.03097 | evaluated at full fidelity |
| idea-014 | mlp_sklearn | 0.06361 | 0.04192 | 0.03341 | evaluated at full fidelity |
| idea-015 | torch_mlp | 0.07563 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-016 | mlp_sklearn | 0.06966 | 0.04738 | 0.04004 | evaluated at full fidelity |
| idea-017 | extra_trees | 0.1366 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.02852 | evaluated at full fidelity |
| idea-020 | torch_mlp | - | - | 0.03276 | evaluated at full fidelity |
| idea-022 | mlp_sklearn | - | - | 0.035 | evaluated at full fidelity |
| idea-023 | mlp_sklearn | - | - | 0.03462 | evaluated at full fidelity |
| idea-019 | kernel_ridge | - | - | 0.03216 | evaluated at full fidelity |
| idea-021 | kernel_ridge | - | - | 0.03011 | evaluated at full fidelity |
| idea-024 | torch_mlp | - | - | 0.02839 | FINAL MODEL |
| idea-025 | torch_mlp | - | - | 0.02744 | FINAL MODEL |
| idea-027 | torch_mlp | - | - | 0.02713 | FINAL MODEL |
| idea-029 | torch_mlp | - | - | 0.02853 | FINAL MODEL |
| idea-026 | kernel_ridge | - | - | 0.03392 | evaluated at full fidelity |
| idea-028 | kernel_ridge | - | - | 0.03061 | evaluated at full fidelity |
| idea-030 | torch_mlp | - | - | 0.02892 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-027 (neural/heuristic: torch_mlp refinement of idea-007: {"lr": 0.00135985893981650)  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)
- Idea ancestry: idea-024 (neural/heuristic: torch_mlp refinement of idea-018: {"dropout": 0.006462707030)  ←  idea-018 (neural/heuristic: torch_mlp refinement of idea-007: {"lr": 0.00087253581132622)  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)
- Idea ancestry: idea-029 (neural/heuristic: torch_mlp refinement of idea-007: {"dropout": 0.019379905337)  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)
- Idea ancestry: idea-025 (neural/heuristic: torch_mlp refinement of idea-018: {})  ←  idea-018 (neural/heuristic: torch_mlp refinement of idea-007: {"lr": 0.00087253581132622)  ←  idea-007 (neural/heuristic: torch_mlp on 'raw' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00220` [checkpoint] by scheduler: final model (ensemble)
- `resu-00217` [result] by executor: exp-0036 rel_l2=0.02338 r2=0.9951 rmse=0.03076
- `hypo-00165` [hypothesis] by ideator-neural: torch_mlp refinement of idea-007: {"lr": 0.0013598589398165094, "batch_size": 64, "activation": "silu"}
- `hypo-00162` [hypothesis] by ideator-neural: torch_mlp refinement of idea-018: {"dropout": 0.006462707030864809}
- `hypo-00167` [hypothesis] by ideator-neural: torch_mlp refinement of idea-007: {"dropout": 0.019379905337065118, "epochs": 156}
- `hypo-00163` [hypothesis] by ideator-neural: torch_mlp refinement of idea-018: {}
- `plan-00216` [plan] by engineer: exp-0036: final fit of 4 member(s) on train+val
- `hypo-00025` [hypothesis] by ideator-neural: torch_mlp on 'raw' features
- `spec-00001` [spec] by scheduler: research spec heat_offline_s2 (d2563933b2)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00131` [hypothesis] by ideator-neural: torch_mlp refinement of idea-007: {"lr": 0.0008725358113262284, "activation": "silu"}

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-008 | ridge | critic: duplicate of an existing configuration |
| idea-012 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-013 | torch_mlp | critic: duplicate of an existing configuration |
| idea-006 | svm | projected full-data cost 189s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-010: train rel_l2=0.03647 vs val 0.1014: strong overfitting, more regularisation or data needed
- idea-026: train rel_l2=0.01373 vs val 0.03392: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.023377062447665832, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 482.4 / 400 worker-seconds over 36 jobs; wall time 271.7s
- By stage: baseline 2.2s, screen 48.9s, halving 55.7s, refine 255.1s, final 120.5s
- Most expensive ideas: ensemble (final fit) 120.6s, idea-007 (torch_mlp) 42.0s, idea-027 (torch_mlp) 41.6s, idea-029 (torch_mlp) 39.1s, idea-024 (torch_mlp) 38.3s, idea-025 (torch_mlp) 34.2s, idea-030 (torch_mlp) 34.0s, idea-018 (torch_mlp) 26.0s
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

Best validation rel_l2 per method: torch_mlp 0.02713, kernel_ridge 0.03011, mlp_sklearn 0.03341, mean_baseline 0.3307. 33 ideas were proposed; 4 were pruned by the critic or the feasibility screen and 17 were trained on the full training set.
