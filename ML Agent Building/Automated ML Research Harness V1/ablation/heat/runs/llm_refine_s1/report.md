# Research report: heat_llm_refine_s1

Run `heat_llm_refine_s1_20261108_154649` · generated 2026-11-08 15:51:33 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-024
  - `idea-024` Even sharper, lighter-regularised log-temperature kernel ridge — method `kernel_ridge`, features `log`, params `{"alpha": 0.0001, "gamma_mult": 0.08}`
- Validation rel_l2 (selection estimate): **0.026501**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.025785**, r2 = **0.99405**, rmse = **0.033928**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s1\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s1")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 16 |
| screen_survivors | 14 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 17 |
| refine_approved | 17 |
| ideas pruned (all reasons) | 2 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-024 | domain | llm | kernel_ridge | log | `{"alpha": 0.0001, "gamma_mult": 0.08}` | 0.026501 | 0.99353 | 0.034948 | 0.546 |
| 2 | idea-030 | domain | llm | kernel_ridge | log | `{"alpha": 3e-05, "gamma_mult": 0.05}` | 0.026525 | 0.99352 | 0.034979 | 0.526 |
| 3 | idea-018 | domain | llm | kernel_ridge | log | `{"alpha": 0.0003, "gamma_mult": 0.15}` | 0.026981 | 0.99329 | 0.03558 | 0.493 |
| 4 | idea-025 | domain | llm | kernel_ridge | log | `{"alpha": 0.0003, "gamma_mult": 0.15}` | 0.027062 | 0.99325 | 0.035687 | 0.556 |
| 5 | idea-027 | domain | llm | kernel_ridge | log | `{"alpha": 0.0005, "gamma_mult": 0.2}` | 0.027301 | 0.99313 | 0.036003 | 0.54 |
| 6 | idea-026 | domain | llm | kernel_ridge | log | `{"alpha": 0.0002, "gamma_mult": 0.15}` | 0.027482 | 0.99304 | 0.036241 | 0.53 |
| 7 | idea-022 | ensembles | llm | custom | log | `{"base": "krr", "alpha": 0.001, "gamma_mult": 0.3, "learn...` | 0.027835 | 0.99286 | 0.036707 | 4.75 |
| 8 | idea-014 | domain | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.027906 | 0.99282 | 0.0368 | 0.465 |
| 9 | idea-020 | domain | llm | torch_mlp | log | `{"hidden": [1024, 512], "dropout": 0.05, "lr": 0.0008, "w...` | 0.028115 | 0.99272 | 0.037076 | 61.4 |
| 10 | idea-015 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.02839 | 0.99258 | 0.037439 | 41.4 |
| 11 | idea-021 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, ...` | 0.028456 | 0.99254 | 0.037526 | 33.8 |
| 12 | idea-019 | domain | llm | kernel_ridge | log | `{"alpha": 0.0005, "gamma_mult": 0.3}` | 0.02926 | 0.99211 | 0.038586 | 0.489 |
| 13 | idea-008 | ensembles | llm | custom | log | `{"alpha": 1.0, "learning_rate": 0.05, "max_iter": 400, "m...` | 0.031528 | 0.99085 | 0.041577 | 4.85 |
| 14 | idea-023 | ensembles | llm | custom | log | `{"base": "ridge", "alpha": 0.01, "learning_rate": 0.03, "...` | 0.032397 | 0.99034 | 0.042723 | 21.6 |
| 15 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.0382 | 0.98655 | 0.050376 | 0.457 |
| 16 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.05649 | 0.04592 | 0.0382 | evaluated at full fidelity |
| idea-003 | ridge | 0.07814 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | knn | 0.1845 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | svm | 0.07036 | - | - | projected full-data cost 218s exceeds the 120s per-experiment limit |
| idea-006 | hist_gbm | 0.112 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | extra_trees | 0.1383 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | custom | 0.04456 | 0.03567 | 0.03153 | evaluated at full fidelity |
| idea-009 | random_forest | 0.1563 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-010 | torch_mlp | 0.06312 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05709 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-012 | mlp_sklearn | 0.0588 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | 0.05189 | - | - | projected full-data cost 226s exceeds the 120s per-experiment limit |
| idea-014 | kernel_ridge | 0.04497 | 0.03463 | 0.02791 | evaluated at full fidelity |
| idea-015 | torch_mlp | 0.05378 | 0.03447 | 0.02839 | evaluated at full fidelity |
| idea-016 | pls | 0.05472 | 0.05209 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-017 | custom | 0.06362 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | kernel_ridge | - | - | 0.02698 | evaluated at full fidelity |
| idea-019 | kernel_ridge | - | - | 0.02926 | evaluated at full fidelity |
| idea-020 | torch_mlp | - | - | 0.02811 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02846 | evaluated at full fidelity |
| idea-022 | custom | - | - | 0.02783 | evaluated at full fidelity |
| idea-023 | custom | - | - | 0.0324 | evaluated at full fidelity |
| idea-024 | kernel_ridge | - | - | 0.0265 | FINAL MODEL |
| idea-025 | kernel_ridge | - | - | 0.02706 | evaluated at full fidelity |
| idea-026 | kernel_ridge | - | - | 0.02748 | evaluated at full fidelity |
| idea-027 | kernel_ridge | - | - | 0.0273 | evaluated at full fidelity |
| idea-030 | kernel_ridge | - | - | 0.02652 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-024 (domain/llm: Even sharper, lighter-regularised log-temperature kernel rid)  ←  idea-018 (domain/llm: Sharper-bandwidth log-temperature kernel ridge with lighter )  ←  idea-014 (domain/llm: Log-temperature RBF kernel ridge (decay-rate linearisation))

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00248` [checkpoint] by scheduler: final model (single)
- `resu-00245` [result] by executor: exp-0038 rel_l2=0.02578 r2=0.9941 rmse=0.03393
- `hypo-00192` [hypothesis] by ideator-domain: Even sharper, lighter-regularised log-temperature kernel ridge
- `plan-00244` [plan] by engineer: exp-0038: final fit of 1 member(s) on train+val
- `hypo-00157` [hypothesis] by ideator-domain: Sharper-bandwidth log-temperature kernel ridge with lighter regularisation
- `spec-00001` [spec] by scheduler: research spec heat_llm_refine_s1 (b451305a91)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00031` [hypothesis] by ideator-domain: Log-temperature RBF kernel ridge (decay-rate linearisation)

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-005 | svm | projected full-data cost 218s exceeds the 120s per-experiment limit |
| idea-013 | custom | projected full-data cost 226s exceeds the 120s per-experiment limit |

## 8. Critic notes

- idea-006: train rel_l2=0.04033 vs val 0.112: strong overfitting, more regularisation or data needed
- idea-010: train rel_l2=0.02745 vs val 0.06312: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.01996 vs val 0.0588: strong overfitting, more regularisation or data needed
- idea-017: train rel_l2=0.0005383 vs val 0.06362: strong overfitting, more regularisation or data needed
- idea-002: train rel_l2=0.0166 vs val 0.0382: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.025784743849971415, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 345.3 / 400 worker-seconds over 38 jobs; wall time 283.8s
- By stage: baseline 2.2s, screen 97.4s, halving 89.0s, refine 153.3s, final 3.3s
- Most expensive ideas: idea-015 (torch_mlp) 76.8s, idea-020 (torch_mlp) 64.3s, idea-021 (torch_mlp) 36.7s, idea-013 (custom) 27.7s, idea-023 (custom) 25.0s, idea-008 (custom) 14.9s, idea-011 (torch_mlp) 13.8s, idea-010 (torch_mlp) 11.9s
- LLM: 9 calls, 185.1s, $0.7861 reported cost

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

Kernel ridge regression on log-transformed sensor features was the clear winner. The selected model (idea-024, α=1e-4, gamma_mult=0.08) reached val rel_l2 = 0.0265 and held-out test rel_l2 = 0.0258 (R² = 0.994, RMSE = 0.034). That is about 12.5× better than the 0.331 baseline and roughly half the 0.05 success threshold. Test came in slightly below val, so selecting on val did not overfit.

The KRR sweep has saturated. The top six configurations span only 0.0265–0.0275, which is probably within the sampling noise of a 750-sample val set. The one consistent trend is that smoother kernels did better: gamma_mult 0.05–0.08 beat 0.3, which scored 0.0279–0.0293. Wider MLPs (0.0281–0.0285) and a KRR + gradient-boosting residual stack (idea-022, 0.0278) did not beat plain KRR. That suggests the remaining error is not structure a second learner can easily capture. It is more likely irreducible noise, or ill-conditioning in the inverse map from sensor readings to conductivity.

The failures were mostly overfitting. idea-017 had a train/val gap of more than 100× (train 0.00054 vs val 0.064), and idea-006 had about 3× (0.040 vs 0.112). idea-002, -010 and -012 showed similar 2–3× gaps. These are high-capacity fits memorizing the 3000-sample training set. Separately, idea-005 and idea-013 were pruned before running because their projected costs (218 s and 226 s) exceeded the 120 s per-experiment cap.

Next experiments:
- **Per-segment error breakdown.** Compute rel_l2 for each of the four conductivities to see whether one segment dominates the error, for example a segment with weak sensor coverage. That would show where to focus effort.
- **Physics-based refinement.** Use the KRR prediction as the starting point for a few Gauss–Newton or L-BFGS steps that fit the simulated forward model to the observed sensor data. Because KRR starts the solver close to the answer, this should converge quickly. It targets the ill-conditioning directly, which more learner tuning cannot.
- **More simulated training data.** Run a learning curve at 1.5k, 3k and 6k samples, which the constraints permit. This will show whether KRR and MLP are data-limited, and whether the overfitting configurations recover with more data. Nyström KRR can keep the cost manageable.
- **Anisotropic kernel (ARD) or PCA-whitened features.** Give sensors and time steps their own length-scales instead of one global gamma. This would help if the informative time window differs between sensors.
- **KRR + MLP average.** Combine idea-024 and idea-020 with weights tuned on val. It is cheap, and it pays off only if their errors are weakly correlated, so check that correlation first.

_Automatic summary:_ Best validation rel_l2 per method: kernel_ridge 0.0265, custom 0.02783, torch_mlp 0.02811, mean_baseline 0.3307. 33 ideas were proposed; 2 were pruned by the critic or the feasibility screen and 15 were trained on the full training set.
