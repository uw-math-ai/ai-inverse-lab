# Research report: heat_full_s2

Run `heat_full_s2_20261108_161319` · generated 2026-11-08 16:18:07 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-009, idea-015, idea-027 (weights [2.0, 1.0, 1.0])
  - `idea-009` Custom: log-target kernel ridge with per-output bandwidth via CV — method `custom`, features `log`, params `{}`
  - `idea-015` Log-feature MLP for nonlinear inverse map — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "silu"}`
  - `idea-027` Kernel ridge on sqrt-temperatures (alternative variance-stabilising transform) — method `kernel_ridge`, features `sqrt`, params `{"alpha": 0.003, "gamma_mult": 0.3}`
- Validation rel_l2 (selection estimate): **0.024626**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.023178**, r2 = **0.99519**, rmse = **0.030498**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s2\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s2")`
- Stopping reason for the search: no relative improvement >= 0.005 for 2 round(s)

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 15 |
| screen_survivors | 14 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 12 |
| refine_approved | 12 |
| ideas pruned (all reasons) | 2 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-009 | classical | llm | custom | log | `{}` | 0.025455 | 0.99403 | 0.033568 | 2.41 |
| 2 | idea-020 | classical | llm | custom | log | `{}` | 0.025847 | 0.99384 | 0.034085 | 6.47 |
| 3 | idea-026 | classical | llm | custom | log | `{}` | 0.026574 | 0.99349 | 0.035044 | 1.46 |
| 4 | idea-027 | classical | llm | kernel_ridge | sqrt | `{"alpha": 0.003, "gamma_mult": 0.3}` | 0.026855 | 0.99335 | 0.035414 | 0.519 |
| 5 | idea-015 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.027062 | 0.99326 | 0.035688 | 45.3 |
| 6 | idea-028 | classical | llm | kernel_ridge | log | `{"alpha": 0.003, "gamma_mult": 0.25}` | 0.027309 | 0.99313 | 0.036013 | 0.54 |
| 7 | idea-021 | classical | llm | kernel_ridge | log | `{"alpha": 0.003, "gamma_mult": 0.3}` | 0.027637 | 0.99296 | 0.036446 | 0.501 |
| 8 | idea-019 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.027777 | 0.99289 | 0.03663 | 14.7 |
| 9 | idea-006 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.027906 | 0.99282 | 0.0368 | 0.447 |
| 10 | idea-024 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.0005...` | 0.02807 | 0.99274 | 0.037017 | 19.6 |
| 11 | idea-018 | domain | llm | torch_mlp | log | `{"hidden": [1024, 512], "dropout": 0.1, "lr": 0.0007, "we...` | 0.028719 | 0.9924 | 0.037873 | 27.2 |
| 12 | idea-022 | classical | llm | kernel_ridge | log | `{"alpha": 0.01, "gamma_mult": 0.15}` | 0.029372 | 0.99205 | 0.038735 | 0.514 |
| 13 | idea-029 | classical | llm | kernel_ridge | log | `{"alpha": 0.003, "gamma_mult": 0.3}` | 0.03056 | 0.99139 | 0.0403 | 0.509 |
| 14 | idea-023 | classical | llm | svm | log | `{"C": 30.0, "gamma": "scale"}` | 0.035944 | 0.98811 | 0.047401 | 2.8 |
| 15 | idea-007 | classical | llm | ridge | log | `{"alpha": 0.1}` | 0.048125 | 0.97873 | 0.063464 | 0.0138 |
| 16 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.000599 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-006 | kernel_ridge | 0.04497 | 0.03463 | 0.02791 | evaluated at full fidelity |
| idea-007 | ridge | 0.05448 | 0.04916 | 0.04812 | evaluated at full fidelity |
| idea-008 | knn | 0.1702 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-009 | custom | 0.03331 | 0.02713 | 0.02545 | FINAL MODEL |
| idea-010 | extra_trees | 0.1492 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | hist_gbm | 0.109 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-012 | kernel_ridge | 0.05649 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | skipped_budget | - | - | feasibility check failed: Unclassified failure: compute budget for this stage exhausted |
| idea-002 | torch_mlp | 0.06145 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-003 | torch_mlp | 0.05991 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | torch_mlp | 0.06355 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | mlp_sklearn | 0.06375 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-015 | torch_mlp | 0.04806 | 0.03284 | 0.02706 | FINAL MODEL |
| idea-016 | pls | 0.05179 | 0.05013 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-017 | custom | 0.05702 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-020 | custom | - | - | 0.02585 | evaluated at full fidelity |
| idea-021 | kernel_ridge | - | - | 0.02764 | evaluated at full fidelity |
| idea-022 | kernel_ridge | - | - | 0.02937 | evaluated at full fidelity |
| idea-023 | svm | - | - | 0.03594 | evaluated at full fidelity |
| idea-018 | torch_mlp | - | - | 0.02872 | evaluated at full fidelity |
| idea-019 | torch_mlp | - | - | 0.02778 | evaluated at full fidelity |
| idea-026 | custom | - | - | 0.02657 | evaluated at full fidelity |
| idea-027 | kernel_ridge | - | - | 0.02685 | FINAL MODEL |
| idea-028 | kernel_ridge | - | - | 0.02731 | evaluated at full fidelity |
| idea-029 | kernel_ridge | - | - | 0.03056 | evaluated at full fidelity |
| idea-024 | torch_mlp | - | - | 0.02807 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-009 (classical/llm: Custom: log-target kernel ridge with per-output bandwidth vi)
- Idea ancestry: idea-015 (domain/llm: Log-feature MLP for nonlinear inverse map)
- Idea ancestry: idea-027 (classical/llm: Kernel ridge on sqrt-temperatures (alternative variance-stab)  ←  idea-009 (classical/llm: Custom: log-target kernel ridge with per-output bandwidth vi)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00238` [checkpoint] by scheduler: final model (ensemble)
- `resu-00235` [result] by executor: exp-0038 rel_l2=0.02318 r2=0.9952 rmse=0.0305
- `hypo-00024` [hypothesis] by ideator-classical: Custom: log-target kernel ridge with per-output bandwidth via CV
- `hypo-00032` [hypothesis] by ideator-domain: Log-feature MLP for nonlinear inverse map
- `hypo-00202` [hypothesis] by ideator-classical: Kernel ridge on sqrt-temperatures (alternative variance-stabilising transform)
- `plan-00234` [plan] by engineer: exp-0038: final fit of 3 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_full_s2 (19cd071d5f)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-014 | kernel_ridge | critic (LLM): Redundant. It uses the same method, features, scaler and PCA setting (kernel_ridge, log, standard, pca=0) as idea-006 and differs only in gamma_mu |
| idea-013 | custom | feasibility check failed: Unclassified failure: compute budget for this stage exhausted |

## 8. Critic notes

- critic advice: All remaining ideas fit the objective and constraints. 1) Log features: noisy late-time temperatures can be near zero or negative, so use a 
- idea-011: train rel_l2=0.03988 vs val 0.109: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.005559 vs val 0.05649: strong overfitting, more regularisation or data needed
- idea-013: Unclassified failure: compute budget for this stage exhausted
- idea-004: train rel_l2=0.02212 vs val 0.06355: strong overfitting, more regularisation or data needed
- idea-005: train rel_l2=0.0179 vs val 0.06375: strong overfitting, more regularisation or data needed
- idea-017: train rel_l2=0.000648 vs val 0.05702: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.00975 vs val 0.03463: strong overfitting, more regularisation or data needed
- critic advice: None of the ideas violate constraints or leak test data. They all target the train/val overfitting gap seen in idea-006 and idea-015. Do mod
- critic advice: All six ideas are aligned, respect the constraints, and none touch the test split. Prioritise idea-026, the multi-bandwidth ensemble, and id
- final audit: test rel_l2=0.023177903035818324, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 289.7 / 400 worker-seconds over 37 jobs; wall time 287.8s
- By stage: baseline 2.2s, screen 79.0s, halving 88.7s, refine 103.0s, final 16.8s
- Most expensive ideas: idea-015 (torch_mlp) 84.4s, idea-018 (torch_mlp) 29.9s, idea-024 (torch_mlp) 22.1s, idea-019 (torch_mlp) 17.4s, ensemble (final fit) 16.8s, idea-004 (torch_mlp) 13.4s, idea-002 (torch_mlp) 12.7s, idea-003 (torch_mlp) 11.4s
- LLM: 12 calls, 188.8s, $0.8588 reported cost

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

The pipeline cleared the success threshold comfortably. The final ensemble (idea-009 + idea-015 + idea-027) reached val rel_l2 = 0.0246 and held-out test rel_l2 = 0.0232 (R² = 0.995, RMSE = 0.0305). That is about 14× better than the 0.331 baseline and roughly half the 0.05 target. Test landing slightly below val is consistent with selection never touching the test split. The decisive choice was the input transform rather than model family. Every leaderboard entry uses a compressive transform (log or sqrt) of the sensor temperatures. Very different models then landed in a tight band of 0.0255–0.0294: the custom model idea-009 (0.0255), an MLP (idea-015, 0.0271) and kernel ridge (idea-027, 0.0269). That spread suggests the transformed features, not capacity, are doing most of the work. One controlled comparison exists: kernel ridge with identical α=0.003 and γ_mult=0.3 scored 0.0269 with sqrt features versus 0.0276 with log. That is a small but consistent edge for sqrt. Ensembling gave only a ~3% gain over the best single model (0.0255 → 0.0246). This implies the members make highly correlated errors, likely because they share an observation-noise or identifiability floor.

The failures were overfitting. Idea-017 drove train error to 0.00065 but scored 0.057 on val, a gap of about 90×. Idea-012 scored 0.0056 versus 0.056, and idea-004/005 scored about 0.018–0.022 versus 0.064. Idea-011 was worst at 0.040 versus 0.109. With only 3000 training samples and four targets, flexible models fitting the raw sensor time series memorise noise. Regularised kernels and dropout/weight-decay MLPs on transformed features avoided this. One inconsistency needs checking: idea-006 is critiqued at val 0.0346 but appears on the leaderboard at 0.0279. This probably reflects two different configurations logged under one ID, and it should be reconciled before the numbers are cited. Two ideas produced no signal. Idea-013 exhausted the stage's compute budget, and idea-014 was rightly pruned as a single γ point that duplicated idea-006. The 400 worker-second budget therefore favoured grids folded into one run over separate ideas.

- **Physics-based refinement:** use the ensemble prediction to warm-start a few Gauss–Newton or L-BFGS steps that minimise the sensor misfit under the forward heat solver. This tests whether the ~0.024 floor comes from the learned map or from the data itself.
- **Per-segment error breakdown:** measure rel_l2 for each of the four conductivities, alongside a Fisher-information or sensitivity analysis of each segment with respect to the sensors. If one poorly observed segment dominates the error, that points to an identifiability limit rather than a modelling one.
- **More simulated training data:** since all models train only on simulated data, generate 2–4× more samples and redraw the learning curve. The large train/val gaps suggest data, not architecture, is the bottleneck for the flexible models.
- **Sqrt features beyond kernel ridge:** run the MLP and the custom model on sqrt features, and try a learned power/Box-Cox exponent, to see whether sqrt's edge carries over to other model families.
- **Stacked ensemble:** replace the fixed-member average with a stacker fitted on out-of-fold predictions, and add a lower-correlation member (for example a model fit in log-k target space) so that ensembling gives more than 3%.

_Automatic summary:_ Best validation rel_l2 per method: custom 0.02545, kernel_ridge 0.02685, torch_mlp 0.02706, svm 0.03594, ridge 0.04812, mean_baseline 0.3307. 28 ideas were proposed; 2 were pruned by the critic or the feasibility screen and 15 were trained on the full training set.
