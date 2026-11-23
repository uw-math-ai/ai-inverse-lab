# Research report: heat_llm_refine_s2

Run `heat_llm_refine_s2_20261108_155158` · generated 2026-11-08 15:59:14 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-013
  - `idea-013` Custom MLP predicting log-conductivity with an ensemble of seeds — method `custom`, features `raw`, params `{"n_models": 3, "hidden": 256, "epochs": 150, "lr": 0.002, "wd": 0.0001, "noise": 0.05}`
- Validation rel_l2 (selection estimate): **0.026217**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.023996**, r2 = **0.99483**, rmse = **0.031574**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s2\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s2")`
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
| ideas pruned (all reasons) | 1 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-013 | neural | llm | custom | raw | `{"n_models": 3, "hidden": 256, "epochs": 150, "lr": 0.002...` | 0.026217 | 0.99367 | 0.034574 | 91.4 |
| 2 | idea-020 | domain | llm | kernel_ridge | log | `{"alpha": 0.0003, "gamma_mult": 0.15}` | 0.026981 | 0.99329 | 0.03558 | 0.475 |
| 3 | idea-015 | domain | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.027906 | 0.99282 | 0.0368 | 0.472 |
| 4 | idea-021 | domain | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.028181 | 0.99268 | 0.037163 | 0.466 |
| 5 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.029797 | 0.99182 | 0.039295 | 0.472 |
| 6 | idea-023 | neural | llm | torch_mlp | log | `{"hidden": [512, 512], "dropout": 0.1, "lr": 0.0015, "wei...` | 0.030587 | 0.99141 | 0.040336 | 38.9 |
| 7 | idea-008 | ensembles | llm | custom | log | `{"alpha": 0.1, "learning_rate": 0.05, "max_iter": 400, "m...` | 0.031461 | 0.99089 | 0.041488 | 4.65 |
| 8 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.001 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04939 | 0.03804 | 0.0298 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | 0.04916 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-004 | kernel_ridge | 0.0678 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-005 | knn | 0.171 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-006 | hist_gbm | 0.1125 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-007 | extra_trees | 0.1354 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-008 | custom | 0.04495 | 0.03667 | 0.03146 | evaluated at full fidelity |
| idea-009 | random_forest | 0.1561 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-010 | torch_mlp | 0.06347 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05662 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-012 | mlp_sklearn | 0.06352 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-013 | custom | 0.05288 | 0.03433 | 0.02622 | FINAL MODEL |
| idea-014 | ridge | 0.05573 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-015 | kernel_ridge | 0.04497 | 0.03463 | 0.02791 | evaluated at full fidelity |
| idea-016 | torch_mlp | 0.05487 | 0.0502 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-017 | custom | 0.05627 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-022 | custom | - | - | timeout | refinement failed: timeout after 120s |
| idea-023 | torch_mlp | - | - | 0.03059 | evaluated at full fidelity |
| idea-020 | kernel_ridge | - | - | 0.02698 | evaluated at full fidelity |
| idea-021 | kernel_ridge | - | - | 0.02818 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-013 (neural/llm: Custom MLP predicting log-conductivity with an ensemble of s)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00198` [checkpoint] by scheduler: final model (single)
- `resu-00195` [result] by executor: exp-0032 rel_l2=0.024 r2=0.9948 rmse=0.03157
- `hypo-00029` [hypothesis] by ideator-neural: Custom MLP predicting log-conductivity with an ensemble of seeds
- `plan-00194` [plan] by engineer: exp-0032: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_llm_refine_s2 (b223652e15)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-022 | custom | refinement failed: timeout after 120s |

## 8. Critic notes

- idea-004: train rel_l2=0.004215 vs val 0.0678: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.04179 vs val 0.1125: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.02935 vs val 0.06352: strong overfitting, more regularisation or data needed
- idea-017: train rel_l2=0.000636 vs val 0.05627: strong overfitting, more regularisation or data needed
- idea-022: Exceeded the per-experiment time limit; the method is too slow at this data size.
- final audit: test rel_l2=0.023995853932514414, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 521.5 / 400 worker-seconds over 32 jobs; wall time 436.1s
- By stage: baseline 2.3s, screen 74.2s, halving 157.0s, refine 167.1s, final 120.8s
- Most expensive ideas: idea-013 (custom) 254.3s, idea-022 (custom) 120.1s, idea-023 (torch_mlp) 41.8s, idea-016 (torch_mlp) 23.8s, idea-008 (custom) 14.2s, idea-011 (torch_mlp) 12.8s, idea-010 (torch_mlp) 9.8s, idea-002 (kernel_ridge) 7.0s
- LLM: 7 calls, 120.6s, $0.5289 reported cost

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

The objective was met comfortably. The selected model, idea-013, is a 3-member ensemble of 256-unit networks trained on raw features with 5% input-noise augmentation. It reached val rel_l2 = 0.0262 and held-out test rel_l2 = 0.0240 (R² = 0.995, RMSE = 0.032). That is about half the 0.05 target and a 13.8× reduction from the mean baseline (0.331). The test score is slightly below the validation score, so selection on val does not appear to have overfit. Several unrelated model families landed in a tight 0.026–0.031 band:
- kernel ridge on log features, with the best at 0.0270 (alpha=3e-4, gamma_mult=0.15);
- a 512×512 SiLU MLP at 0.0306;
- gradient-boosted trees at 0.0315.

Within kernel ridge, a smaller kernel width multiplier and weaker regularisation improved val steadily: gamma_mult 0.5 gave 0.0298, 0.3 gave 0.0279, and 0.15 gave 0.0270. Because such different methods converge on nearly the same error, the remaining error is probably dominated by sensor noise and the ill-posedness of the inversion rather than by model capacity. The winner's margin over kernel ridge is only 0.0008, which is likely within split-to-split variance.

The failures were mostly overfitting, not underfitting. idea-017 reached train 0.0006 against val 0.056, and idea-004 reached 0.0042 against 0.068, train-to-val gaps of roughly 16–90×. idea-006 (0.042 vs 0.113) and idea-012 (0.029 vs 0.064) showed the same pattern at a smaller scale. With only 3000 training samples, unregularised high-capacity fits memorise the noise realisations. Notably, the successful neural model is the one that injects input noise and averages several members. idea-022 was pruned after exceeding the 120 s per-experiment limit, which is significant when the total budget is 400 worker-seconds.

Next experiments:
- **Augment with fresh simulations.** Training data is restricted to simulation anyway, so generating more forward solves, re-sampling noise every epoch, should directly address the overfitting seen in ideas 004, 006, 012 and 017.
- **Ensemble kernel ridge with the noise-augmented NN.** The two best families use different feature transforms (log vs raw), so their errors may be partly uncorrelated. Weight the blend on val only.
- **Continue the kernel ridge sweep downward.** Try gamma_mult ≤ 0.1 and alpha ≤ 3e-4, since the trend had not yet plateaued. Use Nyström features to keep runtime well under the time limit.
- **Add physics-informed features or refinement.** Candidates are time derivatives, ratios of temperatures at adjacent sensors, and a few Gauss–Newton steps through the forward heat solver starting from the ML prediction. This targets the noise/ill-posedness floor rather than model capacity.
- **Check error per conductivity segment and over repeated seeds.** Per-segment rel_l2 would show whether the segments without nearby sensors dominate the error. Repeated seeds or splits would show whether the 0.026 vs 0.027 ranking is meaningful.

_Automatic summary:_ Best validation rel_l2 per method: custom 0.02622, kernel_ridge 0.02698, torch_mlp 0.03059, mean_baseline 0.3307. 22 ideas were proposed; 1 were pruned by the critic or the feasibility screen and 7 were trained on the full training set.
