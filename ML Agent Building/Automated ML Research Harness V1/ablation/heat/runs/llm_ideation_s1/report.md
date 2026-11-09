# Research report: heat_llm_ideation_s1

Run `heat_llm_ideation_s1_20261108_153813` · generated 2026-11-08 15:39:50 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-016
  - `idea-016` Deep MLP on log-temperatures for nonlinear spectral inversion — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "gelu"}`
- Validation rel_l2 (selection estimate): **0.02839**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.028627**, r2 = **0.99259**, rmse = **0.037668**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s1\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s1")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 15 |
| screen_survivors | 15 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| ideas pruned (all reasons) | 1 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-016 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.02839 | 0.99258 | 0.037439 | 19.8 |
| 2 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.029797 | 0.99182 | 0.039295 | 0.494 |
| 3 | idea-017 | domain | llm | custom | raw | `{"ridge_alpha": 0.001, "krr_alpha": 0.001, "gamma_mult": ...` | 0.029919 | 0.99175 | 0.039456 | 0.512 |
| 4 | idea-009 | ensembles | llm | custom | log | `{"alpha": 0.01, "learning_rate": 0.05, "max_iter": 300, "...` | 0.037133 | 0.98731 | 0.048969 | 3.84 |
| 5 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.00118 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04939 | 0.03804 | 0.0298 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | 0.04916 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-004 | kernel_ridge | 0.06841 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | knn | 0.174 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | hist_gbm | 0.1023 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | extra_trees | 0.1359 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | random_forest | 0.1474 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-009 | custom | 0.0515 | 0.04044 | 0.03713 | evaluated at full fidelity |
| idea-010 | torch_mlp | 0.06526 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05791 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-012 | mlp_sklearn | 0.0588 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | 0.05726 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-015 | ridge | 0.05589 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-016 | torch_mlp | 0.05378 | 0.03447 | 0.02839 | FINAL MODEL |
| idea-017 | custom | 0.05364 | 0.03952 | 0.02992 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-016 (domain/llm: Deep MLP on log-temperatures for nonlinear spectral inversio)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00156` [checkpoint] by scheduler: final model (single)
- `resu-00153` [result] by executor: exp-0026 rel_l2=0.02863 r2=0.9926 rmse=0.03767
- `hypo-00033` [hypothesis] by ideator-domain: Deep MLP on log-temperatures for nonlinear spectral inversion
- `plan-00152` [plan] by engineer: exp-0026: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_llm_ideation_s1 (864902ac2d)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |

## 8. Critic notes

- idea-004: train rel_l2=0.0005188 vs val 0.06841: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.03354 vs val 0.1023: strong overfitting, more regularisation or data needed
- idea-010: train rel_l2=0.03059 vs val 0.06526: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.02862720192074091, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 135.9 / 400 worker-seconds over 26 jobs; wall time 96.4s
- By stage: baseline 2.1s, screen 62.3s, halving 56.7s, final 14.8s
- Most expensive ideas: idea-016 (torch_mlp) 56.6s, idea-009 (custom) 13.4s, idea-013 (custom) 10.8s, idea-011 (torch_mlp) 7.8s, idea-002 (kernel_ridge) 7.3s, idea-017 (custom) 6.9s, idea-010 (torch_mlp) 6.8s, idea-003 (ridge) 4.4s
- LLM: 4 calls, 80.2s, $0.3038 reported cost

## 10. Agents

- **reporter** (reporter) — claude_code
- **simulator** (simulation) — claude_code
- **critic** (critic) — claude_code
- **engineer** (engineer) — claude_code
- **ideator-classical** (ideation, persona classical) — claude_code
- **ideator-ensembles** (ideation, persona ensembles) — claude_code
- **ideator-neural** (ideation, persona neural) — claude_code
- **ideator-domain** (ideation, persona domain) — claude_code

## 11. Discussion

The best model, a log-feature GELU MLP ([512, 256, 128], dropout 0.05, weight decay 1e-4, 200 epochs; idea-016), reached val rel_l2 = 0.0284 and held-out test rel_l2 = 0.0286 (R² = 0.993, RMSE = 0.038). That beats the 0.05 success threshold by about 43% and improves on the mean baseline (0.331) by about 11.6×. Val and test scores agree closely (Δ ≈ 0.0002), so selection on val did not overfit to it. The log transform of the features appears to be the key ingredient. Kernel ridge on log features (idea-002, α = 1e-3, γ multiplier 0.5) came within 5% of the MLP at 0.0298, and a ridge+KRR hybrid on raw features (idea-017) was nearly identical at 0.0299. This suggests the map from log-temperatures to conductivity is smooth and close to kernel-representable, and that most of the remaining gap is the noise floor rather than missing model capacity. Gradient-boosted trees on log features (idea-009) were noticeably worse at 0.0371. Their piecewise-constant fits are probably a poor match for a smooth inverse map with only 3000 training samples.

The failures were dominated by overfitting rather than underfitting. Idea-004 reached a train rel_l2 of 5.2e-4 but a val of 0.068, a roughly 130× gap that points to a near-interpolating, under-regularised model. Ideas 006 and 010 showed 3× and 2× train/val gaps (0.034→0.102 and 0.031→0.065). In every case the val error was more than double the winning models'. Together with the strong showing of regularised KRR, this indicates that with only 3000 training samples, controlling variance matters more than model expressiveness. One idea (idea-014) was pruned as a duplicate configuration, so no compute was spent on it. Note that the test numbers come from a single model and a single split. Per-conductivity errors were not reported, so we cannot tell whether one segment, for example a region far from any sensor, dominates the residual.

- **Ensemble the top three** (MLP + log-KRR + ridge/KRR hybrid), with weights fitted on val or by simple averaging. Their errors are similar but come from different model families, so a cheap ensemble may give a 5–15% reduction.
- **Per-segment error breakdown and sensor-sensitivity analysis.** This would show whether one k_i is poorly identifiable and would point to targeted fixes such as weighting the loss toward that segment.
- **Augment training with freshly simulated data and noise.** The constraints allow simulated data, so we could regenerate 10–30k rods or resample the noise each epoch. This directly attacks the variance-limited regime seen in ideas 004/006/010.
- **Predict log k and add physics-informed features**, such as finite-difference estimates of u_t and u_xx between sensors, or early-time decay rates. Log-space targets match the multiplicative structure of conductivity and may sharpen the MLP and KRR fits.
- **Tune KRR more finely** over γ ∈ [0.1, 2] and α ∈ [1e-4, 1e-2], or try a Matérn/ARD kernel. KRR is cheap within the 400 worker-second budget and is already within 5% of the best model.

_Automatic summary:_ Best validation rel_l2 per method: torch_mlp 0.02839, kernel_ridge 0.0298, custom 0.02992, mean_baseline 0.3307. 16 ideas were proposed; 1 were pruned by the critic or the feasibility screen and 4 were trained on the full training set.
