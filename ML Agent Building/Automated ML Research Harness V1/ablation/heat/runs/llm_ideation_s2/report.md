# Research report: heat_llm_ideation_s2

Run `heat_llm_ideation_s2_20261108_154010` · generated 2026-11-08 15:41:24 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-014
  - `idea-014` Log-temperature features + RBF kernel ridge — method `kernel_ridge`, features `log`, params `{"alpha": 0.001, "gamma_mult": 0.3}`
- Validation rel_l2 (selection estimate): **0.027906**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.026648**, r2 = **0.99365**, rmse = **0.035064**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s2\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s2")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 16 |
| screen_survivors | 16 |
| rung_0.33 | 6 |
| rung_1.00 | 4 |
| ideas pruned (all reasons) | 0 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-014 | domain | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.027906 | 0.99282 | 0.0368 | 0.487 |
| 2 | idea-002 | classical | llm | kernel_ridge | raw | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.028895 | 0.99231 | 0.038105 | 0.487 |
| 3 | idea-015 | domain | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.02895 | 0.99228 | 0.038177 | 7.91 |
| 4 | idea-009 | ensembles | llm | custom | log | `{"alpha": 1.0, "learning_rate": 0.05, "max_iter": 300, "m...` | 0.037051 | 0.98737 | 0.048861 | 3.69 |
| 5 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.05335 | 0.03614 | 0.0289 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | 0.04916 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-004 | kernel_ridge | 0.06557 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-005 | knn | 0.1715 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-006 | hist_gbm | 0.1022 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-007 | extra_trees | 0.1363 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-008 | random_forest | 0.151 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-009 | custom | 0.04403 | 0.03968 | 0.03705 | evaluated at full fidelity |
| idea-010 | torch_mlp | 0.06347 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-011 | torch_mlp | 0.06495 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-012 | torch_mlp | 0.05808 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-013 | custom | 0.05976 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-014 | kernel_ridge | 0.04497 | 0.03463 | 0.02791 | FINAL MODEL |
| idea-015 | torch_mlp | 0.04873 | 0.03483 | 0.02895 | evaluated at full fidelity |
| idea-016 | custom | 0.05858 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-017 | ridge | 0.05589 | 0.04925 | - | not_promoted: ranked below the top 4 at fraction 0.33 |

## 6. Lineage of the final model

- Idea ancestry: idea-014 (domain/llm: Log-temperature features + RBF kernel ridge)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00163` [checkpoint] by scheduler: final model (single)
- `resu-00160` [result] by executor: exp-0028 rel_l2=0.02665 r2=0.9936 rmse=0.03506
- `hypo-00031` [hypothesis] by ideator-domain: Log-temperature features + RBF kernel ridge
- `plan-00159` [plan] by engineer: exp-0028: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_llm_ideation_s2 (abe6cd125e)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|

## 8. Critic notes

- idea-006: train rel_l2=0.03761 vs val 0.1022: strong overfitting, more regularisation or data needed
- idea-013: train rel_l2=0.02123 vs val 0.05976: strong overfitting, more regularisation or data needed
- idea-016: train rel_l2=0.000655 vs val 0.05858: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.02664827387438972, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 116.5 / 400 worker-seconds over 28 jobs; wall time 74.2s
- By stage: baseline 2.2s, screen 69.8s, halving 41.5s, final 3.1s
- Most expensive ideas: idea-015 (torch_mlp) 24.6s, idea-013 (custom) 14.0s, idea-009 (custom) 13.0s, idea-014 (kernel_ridge) 9.9s, idea-012 (torch_mlp) 7.7s, idea-011 (torch_mlp) 7.0s, idea-002 (kernel_ridge) 7.0s, idea-010 (torch_mlp) 6.9s
- LLM: 4 calls, 84.9s, $0.3058 reported cost

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

The final model, kernel ridge regression on log-transformed features (α=1e-3, γ multiplier 0.3), reached val rel_l2 = 0.0279 and held-out test rel_l2 = 0.0266 (R² = 0.994, RMSE = 0.035). That is roughly 12× better than the mean baseline (0.331) and clears the 0.05 success threshold with about 45% margin. Test is slightly better than val, so selecting on val does not appear to have overfit it. The top three ideas sit within 4% of each other: log-KRR at 0.0279, raw-KRR at 0.0289 and a 256×256 SiLU MLP at 0.0289. This suggests the log transform adds only a small gain (about 3.5%). It also suggests the smooth-regressor family has plateaued near 0.028, which may reflect the measurement-noise floor or poorly identifiable conductivity segments rather than a lack of model capacity. Histogram gradient boosting (idea-009, 0.0371) trailed the smooth methods. This is consistent with the forward map from temperatures to k being smooth, which favours kernel and MLP function classes over axis-aligned tree splits.

Three ideas failed through clear overfitting rather than underfitting.

- **idea-016** essentially memorised the training set: train 6.6e-4 vs val 0.0586, a gap of about 90×.
- **idea-013** reached train 0.021 vs val 0.060.
- **idea-006** reached train 0.038 vs val 0.102.

With only 3,000 training samples and a high-dimensional sensor-by-time feature vector, flexible models without strong regularisation fit noise. Because the forward simulator is available and simulated-only training is allowed, data scarcity is a self-imposed limit rather than a hard one.

Next experiments, kept cheap given the 400 worker-second budget:
- **Estimate the noise floor.** Run a PDE-constrained least-squares or Bayesian fit of k on about 50 val samples. If it lands near 0.028, further model work has little headroom and effort should shift to identifiability.
- **Break down error per conductivity segment.** Segments far from the sensors, or near the Dirichlet ends, are likely to dominate rel_l2. Confirming this would justify segment-specific models or reweighting.
- **Ensemble log-KRR with the MLP.** They score the same on val but have different inductive biases. A simple average costs almost no compute and may capture any decorrelated error.
- **Simulate more training data.** Try 2–5× more samples, with noise re-sampled per epoch. Test whether this closes the train/val gap for the overfit ideas (013, 016) and whether KRR or the MLP keep improving with n, using Nyström features for KRR to keep it tractable.
- **Add physics-informed features.** Candidates are temporal differences, sensor-to-sensor gradients and early-time decay rates, possibly after PCA or denoising of the time series. The aim is to reduce effective dimensionality and noise sensitivity.

_Automatic summary:_ Best validation rel_l2 per method: kernel_ridge 0.02791, torch_mlp 0.02895, custom 0.03705, mean_baseline 0.3307. 16 ideas were proposed; 0 were pruned by the critic or the feasibility screen and 4 were trained on the full training set.
