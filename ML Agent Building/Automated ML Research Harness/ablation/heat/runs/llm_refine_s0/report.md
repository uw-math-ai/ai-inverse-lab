# Research report: heat_llm_refine_s0

Run `heat_llm_refine_s0_20261108_154149` · generated 2026-11-08 15:46:22 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-011
  - `idea-011` Log-feature MLP with PCA denoising and tanh smooth activation — method `torch_mlp`, features `log`, params `{"hidden": [256, 256], "dropout": 0.1, "lr": 0.001, "weight_decay": 0.001, "epochs": 250, "batch_size": 64, "activation": "tanh"}`
- Validation rel_l2 (selection estimate): **0.027731**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.02699**, r2 = **0.99341**, rmse = **0.035513**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s0\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\llm_refine_s0")`
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
| 1 | idea-011 | neural | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.1, "lr": 0.001, "weig...` | 0.027731 | 0.99292 | 0.036569 | 57.3 |
| 2 | idea-002 | classical | llm | kernel_ridge | log | `{"alpha": 0.001, "gamma_mult": 0.3}` | 0.027906 | 0.99282 | 0.0368 | 0.43 |
| 3 | idea-015 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.02844 | 0.99256 | 0.037505 | 28.6 |
| 4 | idea-028 | neural | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.05, "lr": 0.0013, "we...` | 0.028579 | 0.99248 | 0.037688 | 27.9 |
| 5 | idea-018 | neural | llm | torch_mlp | log | `{"hidden": [512, 512], "dropout": 0.1, "lr": 0.0007, "wei...` | 0.02879 | 0.99237 | 0.037966 | 35.1 |
| 6 | idea-009 | ensembles | llm | custom | log | `{"alpha": 0.1, "learning_rate": 0.05, "max_iter": 400, "m...` | 0.031642 | 0.99079 | 0.041728 | 5.18 |
| 7 | idea-019 | neural | llm | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.15, "lr": 0.001, "wei...` | 0.037519 | 0.98704 | 0.049478 | 38.9 |
| 8 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.001 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04497 | 0.03463 | 0.02791 | evaluated at full fidelity |
| idea-003 | ridge | 0.05448 | 0.04916 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-004 | kernel_ridge | 0.06566 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | knn | 0.174 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | extra_trees | 0.1498 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | hist_gbm | 0.1014 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | random_forest | 0.1566 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-009 | custom | 0.04445 | 0.03577 | 0.03164 | evaluated at full fidelity |
| idea-010 | torch_mlp | 0.05494 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | torch_mlp | 0.04898 | 0.03458 | 0.02773 | FINAL MODEL |
| idea-012 | mlp_sklearn | 0.05769 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | error | - | - | feasibility check failed: Unclassified failure: TypeError: NoisyEnsembleMLP.__init__() mis |
| idea-015 | torch_mlp | 0.04999 | 0.03585 | 0.02844 | evaluated at full fidelity |
| idea-016 | pls | 0.05472 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-017 | custom | 0.0881 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.02879 | evaluated at full fidelity |
| idea-019 | torch_mlp | - | - | 0.03752 | evaluated at full fidelity |
| idea-028 | torch_mlp | - | - | 0.02858 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-011 (neural/llm: Log-feature MLP with PCA denoising and tanh smooth activatio)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00209` [checkpoint] by scheduler: final model (single)
- `resu-00206` [result] by executor: exp-0029 rel_l2=0.02699 r2=0.9934 rmse=0.03551
- `hypo-00027` [hypothesis] by ideator-neural: Log-feature MLP with PCA denoising and tanh smooth activation
- `plan-00205` [plan] by engineer: exp-0029: final fit of 1 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_llm_refine_s0 (a3d47e7cc4)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |
| idea-013 | custom | feasibility check failed: Unclassified failure: TypeError: NoisyEnsembleMLP.__init__() missing 2 required positional arguments: 'params' and 'seed' |

## 8. Critic notes

- idea-004: train rel_l2=0.0005702 vs val 0.06566: strong overfitting, more regularisation or data needed
- idea-007: train rel_l2=0.04763 vs val 0.1014: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.01653 vs val 0.05769: strong overfitting, more regularisation or data needed
- idea-013: Unclassified failure: TypeError: NoisyEnsembleMLP.__init__() missing 2 required positional arguments: 'params' and 'seed'
- idea-017: train rel_l2=0.02819 vs val 0.0881: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.026989630280724562, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 380.7 / 400 worker-seconds over 29 jobs; wall time 273.6s
- By stage: baseline 2.1s, screen 70.4s, halving 146.4s, refine 110.0s, final 51.8s
- Most expensive ideas: idea-011 (torch_mlp) 143.4s, idea-015 (torch_mlp) 59.6s, idea-019 (torch_mlp) 41.6s, idea-018 (torch_mlp) 37.6s, idea-028 (torch_mlp) 30.8s, idea-009 (custom) 16.5s, idea-010 (torch_mlp) 12.4s, idea-002 (kernel_ridge) 6.8s
- LLM: 10 calls, 140.1s, $0.6699 reported cost

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

The pipeline met the objective. The selected model (idea-011, a 256×256 tanh MLP on log features with dropout 0.1 and weight decay 1e-3) reached val rel_l2 = 0.0277 and held-out test rel_l2 = 0.0270, with R² = 0.993 and RMSE = 0.036. That is well under the 0.05 target and about 12× better than the mean baseline (0.331). Test came in slightly below val, so selecting on a single val split does not appear to have overfit. Two other observations point to the remaining error being a floor set by the data rather than by model capacity:

- **Different model families converge.** Kernel ridge on the same log features was essentially tied with the MLP (0.0279).
- **Extra capacity gave no gain.** The top five configurations all fall between 0.0277 and 0.0288, a spread of about 4%. That includes wider and deeper MLPs (512×512 at 0.0288, 512-256-128 GELU at 0.0284).

The likely floor is sensor noise combined with the ill-posedness of recovering k(x) from four point sensors. Gradient boosting on log features was competitive but clearly behind, at 0.0316. Every competitive entry used the log transform, which suggests it matters, though the leaderboard has no raw-feature comparison except the mean baseline.

The failures fall into three groups:

- **Overfitting.** idea-004 had train rel_l2 of 0.00057 against val 0.0657, a gap of more than 100×. idea-012 went from 0.0165 to 0.0577, idea-017 from 0.028 to 0.088, and idea-007 from 0.048 to 0.101.
- **Over-regularisation.** Tuning in the other direction also hurt: idea-019, with dropout 0.15 and weight decay 3e-3, fell to 0.0375. This suggests the regularisation sweet spot for the MLP is fairly narrow, around weight decay 1e-3 and dropout 0.05–0.1.
- **Lost ideas.** idea-013, a noisy ensemble MLP, never ran because of a constructor-signature bug (missing `params` and `seed` arguments), not because the method was weak. idea-014 was pruned as a duplicate.

One caveat on the results: each configuration was evaluated with a single seed. The differences among the top five are probably within seed-to-seed variance, so the ranking of idea-011 over the kernel ridge model (idea-002) is not meaningful.

- **Ensemble MLP + kernel ridge.** The two models are near-tied but structurally different, so averaging their predictions (or stacking them on val) is the cheapest likely gain over the single-model final.
- **Fix and rerun idea-013.** Training on noise-reinjected copies of the inputs directly targets the noise-limited regime.
- **Simulate more training data.** Simulated data is allowed by the constraints. A learning curve at 3k, 6k and 12k samples would show whether the about-0.027 floor comes from the data or from the noise.
- **Physics-based refinement.** Use the MLP prediction as the starting point for a few Gauss-Newton or adjoint iterations of the forward PDE solver on each sample. This can reduce error beyond what pure regression can reach.
- **Per-segment error and seed variance.** Break down error by conductivity segment, since segments far from sensors are likely the dominant source. Run 3–5 seeds of the top configurations to tell real improvements from noise.

_Automatic summary:_ Best validation rel_l2 per method: torch_mlp 0.02773, kernel_ridge 0.02791, custom 0.03164, mean_baseline 0.3307. 28 ideas were proposed; 2 were pruned by the critic or the feasibility screen and 7 were trained on the full training set.
