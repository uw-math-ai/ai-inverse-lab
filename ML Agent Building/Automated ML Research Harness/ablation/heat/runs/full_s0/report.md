# Research report: heat_full_s0

Run `heat_full_s0_20261108_155934` · generated 2026-11-08 16:03:43 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-021, idea-018, idea-017 (weights [1.0, 1.0, 1.0])
  - `idea-021` SiLU deep MLP on PCA-32 sqrt features (combine best activation/feature with PCA denoising) — method `torch_mlp`, features `sqrt`, params `{"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "silu"}`
  - `idea-018` Kernel ridge on sqrt features with wider bandwidth and lower alpha — method `kernel_ridge`, features `sqrt`, params `{"alpha": 0.0003, "gamma_mult": 0.3}`
  - `idea-017` Torch MLP on sqrt-compressed temperatures — method `torch_mlp`, features `sqrt`, params `{"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, "weight_decay": 0.0001, "epochs": 150, "batch_size": 128, "activation": "silu"}`
- Validation rel_l2 (selection estimate): **0.024889**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.023161**, r2 = **0.99518**, rmse = **0.030475**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s0\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s0")`
- Stopping reason for the search: compute budget for stage 'refine' exhausted

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 14 |
| screen_survivors | 13 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 6 |
| refine_approved | 6 |
| ideas pruned (all reasons) | 3 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-021 | neural | llm | torch_mlp | sqrt | `{"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, ...` | 0.026519 | 0.99353 | 0.034972 | 36.4 |
| 2 | idea-022 | domain | llm | torch_mlp | sqrt | `{"hidden": [1024, 512], "dropout": 0.0, "lr": 0.0007, "we...` | 0.028182 | 0.9927 | 0.037164 | 64.8 |
| 3 | idea-017 | domain | llm | torch_mlp | sqrt | `{"hidden": [512, 256, 128], "dropout": 0.0, "lr": 0.001, ...` | 0.028266 | 0.99266 | 0.037275 | 18.6 |
| 4 | idea-018 | classical | llm | kernel_ridge | sqrt | `{"alpha": 0.0003, "gamma_mult": 0.3}` | 0.028393 | 0.99257 | 0.037444 | 0.461 |
| 5 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.029797 | 0.99182 | 0.039295 | 0.438 |
| 6 | idea-023 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.030484 | 0.99146 | 0.0402 | 27.4 |
| 7 | idea-020 | neural | llm | torch_mlp | log | `{"hidden": [512, 512], "dropout": 0.0, "lr": 0.0015, "wei...` | 0.03114 | 0.99107 | 0.041065 | 28.9 |
| 8 | idea-011 | neural | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.002, "weig...` | 0.031203 | 0.99103 | 0.041148 | 16.3 |
| 9 | idea-019 | classical | llm | kernel_ridge | log | `{"alpha": 0.0005, "gamma_mult": 0.7}` | 0.033448 | 0.98969 | 0.044109 | 0.462 |
| 10 | idea-009 | ensembles | llm | custom | log | `{"alpha": 0.01, "n_estimators": 300, "max_features": 0.6,...` | 0.043169 | 0.98288 | 0.056929 | 3.48 |
| 11 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.0015 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04939 | 0.03804 | 0.0298 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | 0.04916 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-004 | kernel_ridge | 0.06575 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | knn | 0.174 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | extra_trees | 0.1478 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | hist_gbm | 0.1156 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | random_forest | 0.1512 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-009 | custom | 0.05067 | 0.04677 | 0.04317 | evaluated at full fidelity |
| idea-010 | torch_mlp | 0.05494 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05078 | 0.0377 | 0.0312 | evaluated at full fidelity |
| idea-012 | mlp_sklearn | 0.05766 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | 0.04081 | - | - | projected full-data cost 259s exceeds the 120s per-experiment limit |
| idea-016 | custom | 0.07147 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-017 | torch_mlp | 0.0445 | 0.03295 | 0.02827 | FINAL MODEL |
| idea-022 | torch_mlp | - | - | 0.02818 | evaluated at full fidelity |
| idea-023 | torch_mlp | - | - | 0.03048 | evaluated at full fidelity |
| idea-018 | kernel_ridge | - | - | 0.02839 | FINAL MODEL |
| idea-019 | kernel_ridge | - | - | 0.03345 | evaluated at full fidelity |
| idea-020 | torch_mlp | - | - | 0.03114 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02652 | FINAL MODEL |

## 6. Lineage of the final model

- Idea ancestry: idea-021 (neural/llm: SiLU deep MLP on PCA-32 sqrt features (combine best activati)  ←  idea-011 (neural/llm: Torch MLP on PCA-denoised log features with tanh activation)
- Idea ancestry: idea-018 (classical/llm: Kernel ridge on sqrt features with wider bandwidth and lower)  ←  idea-002 (classical/llm: Kernel ridge (RBF) on log-temperatures with standard scaling)
- Idea ancestry: idea-017 (domain/llm: Torch MLP on sqrt-compressed temperatures)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00193` [checkpoint] by scheduler: final model (ensemble)
- `resu-00190` [result] by executor: exp-0031 rel_l2=0.02316 r2=0.9952 rmse=0.03048
- `hypo-00155` [hypothesis] by ideator-neural: SiLU deep MLP on PCA-32 sqrt features (combine best activation/feature with PCA denoising)
- `hypo-00151` [hypothesis] by ideator-classical: Kernel ridge on sqrt features with wider bandwidth and lower alpha
- `hypo-00034` [hypothesis] by ideator-domain: Torch MLP on sqrt-compressed temperatures
- `plan-00189` [plan] by engineer: exp-0031: final fit of 3 member(s) on train+val
- `hypo-00027` [hypothesis] by ideator-neural: Torch MLP on PCA-denoised log features with tanh activation
- `spec-00001` [spec] by scheduler: research spec heat_full_s0 (35db56c921)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00016` [hypothesis] by ideator-classical: Kernel ridge (RBF) on log-temperatures with standard scaling

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-015 | ridge | critic: duplicate of an existing configuration |
| idea-013 | custom | projected full-data cost 259s exceeds the 120s per-experiment limit |

## 8. Critic notes

- critic advice: No idea violates a constraint, leaks test data, or is certain to fail, so none are rejected. Guidance:

1. **Selection discipline.** Select 
- idea-004: train rel_l2=0.0001962 vs val 0.06575: strong overfitting, more regularisation or data needed
- idea-007: train rel_l2=0.04756 vs val 0.1156: strong overfitting, more regularisation or data needed
- idea-016: train rel_l2=0.0002608 vs val 0.07147: strong overfitting, more regularisation or data needed
- idea-009: train rel_l2=0.02152 vs val 0.04317: strong overfitting, more regularisation or data needed
- critic advice: All six ideas are aligned with the objective and use only simulated train/val data, so none are rejected. The current best (0.0283) already 
- final audit: test rel_l2=0.023161062842123627, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 398.8 / 400 worker-seconds over 31 jobs; wall time 248.9s
- By stage: baseline 2.3s, screen 93.2s, halving 82.4s, refine 174.6s, final 46.3s
- Most expensive ideas: idea-022 (torch_mlp) 67.7s, idea-017 (torch_mlp) 49.1s, ensemble (final fit) 46.3s, idea-011 (torch_mlp) 43.0s, idea-021 (torch_mlp) 39.2s, idea-020 (torch_mlp) 31.9s, idea-013 (custom) 30.9s, idea-023 (torch_mlp) 30.4s
- LLM: 9 calls, 124.4s, $0.6097 reported cost

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

The final ensemble met the target with room to spare. It averages idea-021 (MLP), idea-018 (kernel ridge) and idea-017 (MLP), all on sqrt features. It scored rel_l2 0.0249 on validation and 0.0232 on the held-out test set (R² 0.995, RMSE 0.030), which is about 2× better than the 0.05 target and 14× better than the mean baseline (0.331). Test was slightly better than validation, so choosing models on the validation split doesn't appear to have overfit it. The sqrt transform beat log consistently. For kernel ridge it scored 0.0284 against 0.0298 and 0.0334 with log. For the [512,256,128] SiLU MLP it scored 0.0265 (idea-021), while the same architecture on log features scored 0.0305 (idea-023, which also used dropout 0.05). Training longer gave a small gain: 150 to 200 epochs took the MLP from 0.0283 to 0.0265. A wider, shallower MLP (idea-022, [1024,512]) did not help (0.0282). The ensemble improved only slightly on the best single model (0.0265 to 0.0249), which suggests the members make similar errors.

The failed runs were mostly overfitting. Ideas 004 and 016 fit the training set almost perfectly (train rel_l2 about 2e-4) but scored 0.066 and 0.071 on validation, which looks like memorising 3000 samples. Idea-007 had a smaller gap but generalised badly (train 0.048, val 0.116). The tree-based custom model (idea-009) halved its error from train to validation (train 0.0215, val 0.0432) and finished last among the real models. Smooth function classes (RBF kernel ridge, SiLU or tanh MLPs with weight decay) worked best, which fits a map from sensor readings to k that is smooth and low-dimensional. Three ideas never ran: idea-013 was pruned because its projected cost (259 s) exceeded the 120 s per-experiment limit, and ideas 014 and 015 were duplicates of existing configurations.

- **Simulate more training data.** The overfitting above, and the fact that only simulated data is allowed, suggest the models are limited by data. Generating 10–30k extra simulations and retraining the idea-021 MLP should show whether validation error keeps falling with more data.
- **Make the ensemble more diverse.** Try MLPs trained with different seeds and on different feature sets (sqrt, log, raw), or stack the models with a learned weighting instead of a plain average, since the current members' errors seem correlated.
- **Check error per segment.** Split rel_l2 by which of the four conductivities it comes from. If one segment, likely one far from the sensors, dominates the error, try predicting log k or weighting the loss by segment.
- **Refine predictions with the physics model.** Use the ensemble's output as the starting point for a few Gauss–Newton steps that fit the simulated sensor readings to the measured ones. A 1-D solver is cheap, so this should fit the 400 s budget, and it can correct errors the regression model makes consistently.
- **Run idea-013 cheaply.** Train it on part of the data or with early stopping to bring it under the 120 s limit, so the idea is actually tested instead of dropped.

_Automatic summary:_ Best validation rel_l2 per method: torch_mlp 0.02652, kernel_ridge 0.02839, custom 0.04317, mean_baseline 0.3307. 22 ideas were proposed; 3 were pruned by the critic or the feasibility screen and 10 were trained on the full training set.
