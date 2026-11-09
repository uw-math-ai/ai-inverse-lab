# Research report: heat_full_s1

Run `heat_full_s1_20261108_160410` · generated 2026-11-08 16:12:58 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **ensemble** of idea-026, idea-018, idea-020, idea-019 (weights [2.0, 1.0, 1.0, 1.0])
  - `idea-026` PCA-30 log features into tapered SiLU MLP with stronger dropout and weight decay — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.15, "lr": 0.001, "weight_decay": 0.001, "epochs": 300, "batch_size": 128, "activation": "silu"}`
  - `idea-018` Wider tapered SiLU MLP on log-temperatures with longer cosine schedule — method `torch_mlp`, features `log`, params `{"hidden": [1024, 512], "dropout": 0.05, "lr": 0.0008, "weight_decay": 0.0003, "epochs": 300, "batch_size": 128, "activation": "silu"}`
  - `idea-020` PCA-30 log-temperatures into tapered SiLU MLP with slightly stronger dropout — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.1, "lr": 0.001, "weight_decay": 0.0003, "epochs": 250, "batch_size": 128, "activation": "silu"}`
  - `idea-019` Same tapered SiLU MLP on PCA-denoised log-temperatures — method `torch_mlp`, features `log`, params `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001, "weight_decay": 0.0001, "epochs": 200, "batch_size": 128, "activation": "silu"}`
- Validation rel_l2 (selection estimate): **0.025653**; baseline (mean predictor): 0.3307
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.0246**, r2 = **0.99454**, rmse = **0.032369**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s1\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness V1\ablation\heat\runs\full_s1")`
- Stopping reason for the search: completed all refinement rounds

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 15 |
| screen_survivors | 15 |
| rung_0.33 | 5 |
| rung_1.00 | 1 |
| refine_proposed | 14 |
| refine_approved | 14 |
| ideas pruned (all reasons) | 1 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-026 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.15, "lr": 0.001,...` | 0.026408 | 0.99357 | 0.034826 | 63 |
| 2 | idea-020 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.1, "lr": 0.001, ...` | 0.026668 | 0.99345 | 0.035168 | 37.5 |
| 3 | idea-019 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.027547 | 0.99302 | 0.036327 | 26.2 |
| 4 | idea-015 | domain | llm | torch_mlp | log | `{"hidden": [512, 256, 128], "dropout": 0.05, "lr": 0.001,...` | 0.028056 | 0.99275 | 0.036998 | 42.8 |
| 5 | idea-018 | domain | llm | torch_mlp | log | `{"hidden": [1024, 512], "dropout": 0.05, "lr": 0.0008, "w...` | 0.028113 | 0.99272 | 0.037074 | 52.8 |
| 6 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.3307 | -0.0011334 | 0.4361 | 0.001 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3307 | reference baseline |
| idea-002 | kernel_ridge | 0.04939 | 0.03804 | - | budget_cut: not promoted: estimated 3s does not fit the remaining 80s of the halving budge |
| idea-003 | ridge | 0.05448 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | pls | 0.07693 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-005 | knn | 0.1708 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | extra_trees | 0.1399 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | hist_gbm | 0.112 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | custom | 0.05194 | 0.04603 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-009 | random_forest | 0.1559 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-010 | torch_mlp | 0.05245 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | torch_mlp | 0.05127 | 0.03401 | - | budget_cut: not promoted: estimated 60s does not fit the remaining 80s of the halving budg |
| idea-012 | mlp_sklearn | 0.05992 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-013 | custom | 0.04665 | 0.03486 | - | budget_cut: not promoted: estimated 42s does not fit the remaining 80s of the halving budg |
| idea-015 | torch_mlp | 0.04975 | 0.03245 | 0.02806 | evaluated at full fidelity |
| idea-016 | pls | 0.05472 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-017 | custom | 0.05704 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-018 | torch_mlp | - | - | 0.02811 | FINAL MODEL |
| idea-019 | torch_mlp | - | - | 0.02755 | FINAL MODEL |
| idea-020 | torch_mlp | - | - | 0.02667 | FINAL MODEL |
| idea-026 | torch_mlp | - | - | 0.02641 | FINAL MODEL |

## 6. Lineage of the final model

- Idea ancestry: idea-026 (domain/llm: PCA-30 log features into tapered SiLU MLP with stronger drop)  ←  idea-020 (domain/llm: PCA-30 log-temperatures into tapered SiLU MLP with slightly )  ←  idea-019 (domain/llm: Same tapered SiLU MLP on PCA-denoised log-temperatures)  ←  idea-015 (domain/llm: Log features + PyTorch MLP (smooth activations))
- Idea ancestry: idea-018 (domain/llm: Wider tapered SiLU MLP on log-temperatures with longer cosin)  ←  idea-015 (domain/llm: Log features + PyTorch MLP (smooth activations))
- Idea ancestry: idea-020 (domain/llm: PCA-30 log-temperatures into tapered SiLU MLP with slightly )  ←  idea-019 (domain/llm: Same tapered SiLU MLP on PCA-denoised log-temperatures)  ←  idea-015 (domain/llm: Log features + PyTorch MLP (smooth activations))
- Idea ancestry: idea-019 (domain/llm: Same tapered SiLU MLP on PCA-denoised log-temperatures)  ←  idea-015 (domain/llm: Log features + PyTorch MLP (smooth activations))

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00215` [checkpoint] by scheduler: final model (ensemble)
- `resu-00212` [result] by executor: exp-0028 rel_l2=0.0246 r2=0.9945 rmse=0.03237
- `hypo-00185` [hypothesis] by ideator-domain: PCA-30 log features into tapered SiLU MLP with stronger dropout and weight decay
- `hypo-00147` [hypothesis] by ideator-domain: Wider tapered SiLU MLP on log-temperatures with longer cosine schedule
- `hypo-00161` [hypothesis] by ideator-domain: PCA-30 log-temperatures into tapered SiLU MLP with slightly stronger dropout
- `hypo-00148` [hypothesis] by ideator-domain: Same tapered SiLU MLP on PCA-denoised log-temperatures
- `plan-00211` [plan] by engineer: exp-0028: final fit of 4 member(s) on train+val
- `spec-00001` [spec] by scheduler: research spec heat_full_s1 (0d79e53729)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `hypo-00032` [hypothesis] by ideator-domain: Log features + PyTorch MLP (smooth activations)

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-014 | kernel_ridge | critic: duplicate of an existing configuration |

## 8. Critic notes

- critic advice: No idea is misaligned, breaks a constraint, leaks test data, or is certain to fail, so nothing is rejected. 1) Log-feature safety: with Diri
- idea-007: train rel_l2=0.03998 vs val 0.112: strong overfitting, more regularisation or data needed
- critic advice: Both ideas are aligned and constraint-compliant. Watch the budget: with 3000 samples and no GPU, a 300-epoch [1024,512] MLP should finish qu
- critic advice: No idea violates a constraint, leaks test data, or is certain to fail, and all fit the CPU budget (about 3000 samples, small MLPs). The batc
- critic advice: All six ideas fit the objective and constraints. None leaks test data, and all are cheap enough for the CPU budget (3000 samples, about 250-
- final audit: test rel_l2=0.024599963654579036, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 542.1 / 400 worker-seconds over 28 jobs; wall time 528.0s
- By stage: baseline 2.1s, screen 91.7s, halving 111.7s, refine 190.7s, final 145.9s
- Most expensive ideas: ensemble (final fit) 145.9s, idea-015 (torch_mlp) 79.4s, idea-026 (torch_mlp) 66.0s, idea-018 (torch_mlp) 55.2s, idea-013 (custom) 46.7s, idea-020 (torch_mlp) 40.6s, idea-011 (torch_mlp) 33.8s, idea-019 (torch_mlp) 28.9s
- LLM: 12 calls, 170.4s, $0.8048 reported cost

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

The log-feature SiLU MLP family met the target. The four-member ensemble reached val rel_l2 = 0.0257 and held-out test rel_l2 = 0.0246 (R² = 0.9945, RMSE = 0.032). That is about 13× better than the mean baseline (0.331) and roughly half the 0.05 threshold. Test error is slightly below val, so selection on val does not appear to have overfit. Within the [512, 256, 128] architecture, stronger regularisation helped steadily but modestly:

| Idea | Dropout | Weight decay | Epochs | Val rel_l2 |
|---|---|---|---|---|
| 019 | 0.05 | 1e-4 | 200 | 0.0275 |
| 020 | 0.10 | 3e-4 | 250 | 0.0267 |
| 026 | 0.15 | 1e-3 | 300 | 0.0264 |

The wider two-layer [1024, 512] net (0.0281) offered nothing over the deeper one. Ensembling added a further ~3% relative gain over the best single model (0.0264 → 0.0257).

Most of these differences are not resolved by the data. idea-015 and idea-019 have identical configurations, yet their val scores differ by about 0.0005 (0.0281 vs 0.0275). That run-to-run noise is comparable to the whole spread among the top five (0.0264–0.0281). The regularisation trend is suggestive but not established, and the ranking of the top configurations is essentially a tie. The one clear failure was idea-007, which overfit badly (train 0.040 vs val 0.112, nearly a 3× gap). With only 3000 training simulations, weakly regularised high-capacity models memorise the noise realisations. The critic also correctly pruned idea-014 as a duplicate. The idea-015/019 pair, however, shows that such near-duplicates still slipped through.

Next experiments:
- **Repeat seeds:** run 3–5 seeds per top configuration to get error bars before attributing gains to dropout or weight decay. Stop tuning hyperparameters at the ~0.001 level, since that is below the noise floor.
- **Generate more data:** since only simulated training data is allowed and the simulator is cheap, scale the training set (e.g. 3k → 15–30k), possibly with fresh noise draws each epoch. This directly targets the overfitting seen in idea-007 and is likely worth more than further regularisation sweeps.
- **Break down error by segment:** compute rel_l2 for each of the four conductivity segments and as a function of the k contrast. Segments far from the sensors, or high-contrast cases, are probably the weakest; this would guide changes to features or the loss, such as predicting log k or weighting segments.
- **Physics-based refinement:** use the MLP ensemble's prediction to initialise a few Gauss–Newton or adjoint iterations of the PDE misfit against the observed sensor traces. This could push error toward the noise floor at modest compute cost per sample.
- **Structured features or architecture:** try temporal-derivative or early/late-time features, or a small 1-D CNN over the time axis per sensor, against the flat MLP.

_Automatic summary:_ Best validation rel_l2 per method: torch_mlp 0.02641, mean_baseline 0.3307. 30 ideas were proposed; 1 were pruned by the critic or the feasibility screen and 5 were trained on the full training set.
