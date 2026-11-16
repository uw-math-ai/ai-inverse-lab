# Research report: heat_llm_ideation_s0

Run `heat_llm_ideation_s0_20261108_151616` · generated 2026-11-08 15:19:29 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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
  - `idea-016` Torch MLP on log features with PCA denoising — method `torch_mlp`, features `log`, params `{"hidden": [256, 256], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}`
- Validation rel_l2 (selection estimate): **0.026495**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.02465**, r2 = **0.99452**, rmse = **0.032435**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s0\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_ideation_s0")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 14 |
| screen_survivors | 12 |
| rung_0.33 | 4 |
| rung_1.00 | 4 |
| ideas pruned (all reasons) | 4 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-016 | domain | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.05, "lr": 0.001, "wei...` | 0.026495 | 0.99353 | 0.03494 | 31.9 |
| 2 | idea-005 | classical | llm | custom | log | `{"alpha": 0.001, "gamma_mult": 0.5, "lin_weight": 1.0}` | 0.029542 | 0.99196 | 0.038958 | 0.615 |
| 3 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.029797 | 0.99182 | 0.039295 | 0.462 |
| 4 | idea-009 | ensembles | llm | custom | log | `{"alpha": 1.0, "learning_rate": 0.05, "max_iter": 300, "m...` | 0.033832 | 0.98946 | 0.044616 | 2.73 |
| 5 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.000942 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04939 | 0.03804 | 0.0298 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-004 | kernel_ridge | 0.0678 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-005 | custom | 0.04461 | 0.03704 | 0.02954 | evaluated at full fidelity |
| idea-006 | hist_gbm | 0.1128 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-007 | extra_trees | 0.1356 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-009 | custom | 0.04555 | 0.03799 | 0.03383 | evaluated at full fidelity |
| idea-010 | torch_mlp | 0.0495 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05609 | - | - | projected full-data cost 202s exceeds the 120s per-experiment limit |
| idea-012 | mlp_sklearn | 0.06075 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-013 | custom | error | - | - | feasibility check failed: Unclassified failure: AttributeError: The following error was ra |
| idea-015 | pls | 0.06533 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |
| idea-016 | torch_mlp | 0.04887 | 0.03334 | 0.02649 | FINAL MODEL |
| idea-017 | custom | 0.05625 | - | - | not_promoted: ranked below the top 4 at fraction 0.10 |

## 6. Lineage of the final model

- Idea ancestry: idea-016 (domain/llm: Torch MLP on log features with PCA denoising)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00154` [checkpoint] by scheduler: final model (single)
- `resu-00151` [result] by executor: exp-0024 rel_l2=0.02465 r2=0.9945 rmse=0.03243
- `hypo-00033` [hypothesis] by ideator-domain: Torch MLP on log features with PCA denoising
- `plan-00150` [plan] by engineer: exp-0024: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_llm_ideation_s0 (f4a24cbdc1)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-008 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-011 | torch_mlp | projected full-data cost 202s exceeds the 120s per-experiment limit |
| idea-013 | custom | feasibility check failed: Unclassified failure: AttributeError: The following error was raised: 'NoisyResEnsemble' object has no attribute '__sklearn_tags__'. I |

## 8. Critic notes

- idea-004: train rel_l2=0.004215 vs val 0.0678: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.04262 vs val 0.1128: strong overfitting, more regularisation or data needed
- idea-011: train rel_l2=0.02557 vs val 0.05609: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.02488 vs val 0.06075: strong overfitting, more regularisation or data needed
- idea-013: Unclassified failure: AttributeError: The following error was raised: 'NoisyResEnsemble' object has no attribute '__sklearn_tags__'. It seem
- idea-017: train rel_l2=0.0005401 vs val 0.05625: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.024650187514671684, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 255.0 / 400 worker-seconds over 24 jobs; wall time 193.5s
- By stage: baseline 2.2s, screen 141.4s, halving 65.9s, final 45.5s
- Most expensive ideas: idea-016 (torch_mlp) 106.5s, idea-013 (custom) 68.3s, idea-011 (torch_mlp) 22.8s, idea-010 (torch_mlp) 13.2s, idea-009 (custom) 11.0s, idea-005 (custom) 7.0s, idea-002 (kernel_ridge) 6.8s, idea-006 (hist_gbm) 3.2s
- LLM: 4 calls, 81.0s, $0.3701 reported cost

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

The best model was a two-layer SiLU MLP on log-transformed features (idea-016, 256×256, dropout 0.05, 150 epochs). It reached val rel_l2 = 0.0265 and held-out test rel_l2 = 0.0247 (R² = 0.9945, RMSE = 0.032). That is about 13× better than the mean baseline (0.331) and comfortably under the 0.05 success threshold. Val and test agree closely (test is slightly better), so selection on val does not appear to have overfit, though this rests on a single test evaluation with no seed variance. The log transform is a common thread: all four non-baseline leaders use it. Kernel ridge on log features (idea-002, 0.0298) came within about 12% of the MLP. Adding a linear term to it (idea-005, 0.0295) changed almost nothing, which suggests the KRR was already capturing the linear component. Gradient boosting (idea-009, 0.0338) was the weakest of the competitive models, consistent with tree ensembles approximating a smooth inverse map poorly at n=3000.

The failures came from three causes: overfitting, budget limits and engineering bugs. Five ideas showed large train/val gaps. The most extreme was idea-017, with train 0.00054 against val 0.056, a roughly 100× gap that points to memorisation. idea-004 had 0.0042 vs 0.068, and idea-006 had 0.043 vs 0.113, which is poor even on train. idea-011 and idea-012 both sat near 0.025 train vs 0.056–0.061 val. None of these beat the regularised KRR or MLP, so with 3000 samples, capacity control matters more than flexibility. idea-011 was also pruned because its projected cost of 202 s exceeded the 120 s per-experiment cap. idea-013 never ran: its custom `NoisyResEnsemble` failed the sklearn ≥1.6 estimator check (`__sklearn_tags__`), most likely because it does not subclass `BaseEstimator`/`RegressorMixin` correctly. That is an implementation fault, not evidence against the idea. Two further ideas (008, 014) were dropped as duplicates.

Next experiments:
- **Blend the MLP with KRR.** Average or stack idea-016 and idea-002/005 predictions on val. Their val scores are close but they come from different model families, so their errors may be partly uncorrelated. This is cheap because both models already exist.
- **Seed-ensemble the MLP.** Train 3–5 seeds of idea-016 to measure seed variance, which the single 0.0265 does not show, and to reduce error by averaging.
- **Generate more simulated training data with fresh noise draws.** The constraints allow simulated data, and the train/val gaps above point to data scarcity. More solves of the forward heat equation, plus re-sampled sensor noise as augmentation, should help both the MLP and KRR.
- **Break down error per conductivity segment.** Compute rel_l2 for each of the four k values to check whether segments far from sensors dominate the error. If they do, try targeted features such as time-derivative or late-time decay-rate summaries.
- **Fix and rerun idea-013.** Subclass `BaseEstimator` and `RegressorMixin` so the noise-robust residual ensemble can be evaluated, and keep its cost inside the 120 s cap.

_Automatic summary:_ Best validation rel_l2 per method: torch_mlp 0.02649, custom 0.02954, kernel_ridge 0.0298, mean_baseline 0.3307. 16 ideas were proposed; 4 were pruned by the critic or the feasibility screen and 4 were trained on the full training set.
