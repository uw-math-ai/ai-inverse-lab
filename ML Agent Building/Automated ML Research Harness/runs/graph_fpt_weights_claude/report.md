# Research report: graph_fpt_weights

Run `graph_fpt_weights_20261107_153153` · generated 2026-11-07 15:42:30 · LLM backend: **claude_code** · workers: 6 × 2 threads · GPUs: none

## 1. Research specification

**Problem.** Recover the edge weights of a weighted 3x3 grid graph from empirical first-passage-time distributions r'(t, x, y) of random walks observed only on the boundary vertices (inverse problem for discrete heat equations / random walks on graphs).

**Data.** Random walk on a fixed {'family': 'grid', 'rows': 3, 'cols': 3} graph with 9 vertices and 12 edges; edge weights ~ U[0.2, 1.0], mu = degree, potential {'mode': 'none'}. Observed set B = [0, 1, 2, 3, 5, 6, 7, 8] (|B|=8). Features: empirical first-passage probabilities r'(t, x, y) for t = 1..9 and (x, y) in B x B from N = 10000 walks per pair, flattened in (t, x, y) order -> 576 features. Target: weights (12 values). Inverse problem: recover graph parameters from boundary first-passage data.

**Objectives.**
- Learn a regression model from simulated data (graph_fpt) that solves: Recover the edge weights of a weighted 3x3 grid graph from empirical first-passage-time distributions r'(t, x, y) of random walks observed only on the boundary vertices (inverse problem for discrete h
- Select methods by validation rel_l2 only; the test split is used exactly once, at the end
- Success criterion: test rel_l2 <= 0.05
- Stay within 900 worker-seconds of compute and 600 s wall time
- Eliminate weak approaches with cheap feasibility checks before spending substantial compute
- Constraint: Models may only be trained on simulated data; the test split is never used for selection

## 2. Outcome

- Final model: **ensemble** of idea-031, idea-032 (weights [2.0, 1.0])
  - `idea-031` Wide-bandwidth, low-alpha kernel ridge on log + FPT summary features — method `kernel_ridge`, features `log+fpt_summary`, params `{"alpha": 0.004, "gamma_mult": 0.09}`
  - `idea-032` Wider-bandwidth, lower-alpha kernel ridge on early-time FPT window — method `kernel_ridge`, features `fpt_early`, params `{"alpha": 0.003, "gamma_mult": 0.1}`
- Validation rel_l2 (selection estimate): **0.012299**; baseline (mean predictor): 0.35913
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.011944**, r2 = **0.99888**, rmse = **0.0076401**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `runs\graph_fpt_weights_claude\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"runs\graph_fpt_weights_claude")`
- Stopping reason for the search: pipeline completed

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 14 |
| screen_survivors | 13 |
| rung_0.33 | 5 |
| rung_1.00 | 4 |
| refine_proposed | 18 |
| refine_approved | 18 |
| ideas pruned (all reasons) | 3 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-031 | domain | llm | kernel_ridge | log+fpt_summary | `{"alpha": 0.004, "gamma_mult": 0.09}` | 0.013179 | 0.99865 | 0.0084585 | 2.42 |
| 2 | idea-033 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.0015, "gamma_mult": 0.06}` | 0.014282 | 0.99841 | 0.0091661 | 1.07 |
| 3 | idea-032 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.003, "gamma_mult": 0.1}` | 0.014288 | 0.99841 | 0.0091699 | 1.09 |
| 4 | idea-028 | domain | llm | kernel_ridge | log+fpt_summary | `{"alpha": 0.01, "gamma_mult": 0.2}` | 0.014315 | 0.9984 | 0.0091873 | 2.26 |
| 5 | idea-035 | domain | llm | kernel_ridge | sqrt | `{"alpha": 0.008, "gamma_mult": 0.1}` | 0.014512 | 0.99836 | 0.0093135 | 2.18 |
| 6 | idea-026 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.005, "gamma_mult": 0.15}` | 0.014559 | 0.99835 | 0.0093439 | 1.28 |
| 7 | idea-024 | domain | llm | kernel_ridge | sqrt | `{"alpha": 0.01, "gamma_mult": 0.2}` | 0.014805 | 0.99829 | 0.0095016 | 2.19 |
| 8 | idea-030 | domain | llm | kernel_ridge | log+fpt_summary | `{"alpha": 0.015, "gamma_mult": 0.13}` | 0.01482 | 0.99829 | 0.0095117 | 2.52 |
| 9 | idea-034 | domain | llm | kernel_ridge | sqrt | `{"alpha": 0.015, "gamma_mult": 0.12}` | 0.015102 | 0.99822 | 0.0096925 | 1.29 |
| 10 | idea-023 | domain | llm | kernel_ridge | sqrt | `{"alpha": 0.003, "gamma_mult": 0.3}` | 0.015555 | 0.99812 | 0.0099828 | 1.18 |
| 11 | idea-022 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.01, "gamma_mult": 0.25}` | 0.015649 | 0.99809 | 0.010043 | 1.04 |
| 12 | idea-027 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.01, "gamma_mult": 0.25}` | 0.015787 | 0.99806 | 0.010132 | 1.28 |
| 13 | idea-025 | domain | llm | kernel_ridge | sqrt | `{"alpha": 0.003, "gamma_mult": 0.3}` | 0.019389 | 0.99707 | 0.012444 | 2.21 |
| 14 | idea-029 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.003, "gamma_mult": 0.35}` | 0.019713 | 0.99697 | 0.012651 | 1.1 |
| 15 | idea-016 | domain | llm | kernel_ridge | fpt_early | `{"alpha": 0.001, "gamma_mult": 0.5}` | 0.019783 | 0.99695 | 0.012697 | 1.04 |
| 16 | idea-020 | neural | llm | torch_mlp | sqrt | `{"hidden": [1024, 512], "dropout": 0.25, "lr": 0.0007, "w...` | 0.01989 | 0.99692 | 0.012765 | 210 |
| 17 | idea-007 | neural | llm | torch_mlp | sqrt | `{"hidden": [1024, 512], "dropout": 0.1, "lr": 0.0007, "we...` | 0.021773 | 0.99631 | 0.013974 | 98.3 |
| 18 | idea-021 | neural | llm | torch_mlp | sqrt | `{"hidden": [512, 512], "dropout": 0.15, "lr": 0.001, "wei...` | 0.023588 | 0.99567 | 0.015138 | 91.2 |
| 19 | idea-004 | classical | llm | pls | sqrt | `{"n_components": 40}` | 0.025475 | 0.99495 | 0.016349 | 3.48 |
| 20 | idea-002 | classical | llm | ridge | log | `{"alpha": 1.0}` | 0.026577 | 0.99449 | 0.017057 | 0.135 |
| 21 | idea-018 | classical | llm | pls | fpt_early | `{"n_components": 50}` | 0.034753 | 0.99058 | 0.022305 | 1.51 |
| 22 | idea-019 | classical | llm | pls | log | `{"n_components": 25}` | 0.041018 | 0.98689 | 0.026325 | 2.64 |
| 23 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.35913 | -0.0017183 | 0.23049 | 0.00139 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3591 | reference baseline |
| idea-002 | ridge | 0.04535 | 0.03155 | 0.02658 | evaluated at full fidelity |
| idea-003 | kernel_ridge | 0.05521 | 0.03257 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-005 | custom | 0.2027 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-004 | pls | 0.03147 | 0.02721 | 0.02547 | evaluated at full fidelity |
| idea-010 | extra_trees | 0.2288 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-012 | random_forest | 0.2596 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-011 | hist_gbm | 0.1539 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-008 | mlp_sklearn | 0.09001 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-016 | kernel_ridge | 0.04656 | 0.02785 | 0.01978 | evaluated at full fidelity |
| idea-017 | custom | 0.3354 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-015 | torch_mlp | 0.05912 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-006 | torch_mlp | 0.05668 | - | - | not_promoted: ranked below the top 5 at fraction 0.10 |
| idea-007 | torch_mlp | 0.04178 | 0.02769 | 0.02177 | evaluated at full fidelity |
| idea-013 | custom | 0.04497 | - | - | projected full-data cost 731s exceeds the 240s per-experiment limit |
| idea-022 | kernel_ridge | - | - | 0.01565 | evaluated at full fidelity |
| idea-018 | pls | - | - | 0.03475 | evaluated at full fidelity |
| idea-023 | kernel_ridge | - | - | 0.01555 | evaluated at full fidelity |
| idea-019 | pls | - | - | 0.04102 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02359 | evaluated at full fidelity |
| idea-020 | torch_mlp | - | - | 0.01989 | evaluated at full fidelity |
| idea-029 | kernel_ridge | - | - | 0.01971 | evaluated at full fidelity |
| idea-027 | kernel_ridge | - | - | 0.01579 | evaluated at full fidelity |
| idea-026 | kernel_ridge | - | - | 0.01456 | evaluated at full fidelity |
| idea-025 | kernel_ridge | - | - | 0.01939 | evaluated at full fidelity |
| idea-024 | kernel_ridge | - | - | 0.0148 | evaluated at full fidelity |
| idea-028 | kernel_ridge | - | - | 0.01432 | evaluated at full fidelity |
| idea-033 | kernel_ridge | - | - | 0.01428 | evaluated at full fidelity |
| idea-032 | kernel_ridge | - | - | 0.01429 | FINAL MODEL |
| idea-034 | kernel_ridge | - | - | 0.0151 | evaluated at full fidelity |
| idea-035 | kernel_ridge | - | - | 0.01451 | evaluated at full fidelity |
| idea-031 | kernel_ridge | - | - | 0.01318 | FINAL MODEL |
| idea-030 | kernel_ridge | - | - | 0.01482 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-031 (domain/llm: Wide-bandwidth, low-alpha kernel ridge on log + FPT summary )  ←  idea-028 (domain/llm: Kernel ridge on log + FPT summary features with strong regul)  ←  idea-016 (domain/llm: Early-time FPT window + RBF kernel ridge)
- Idea ancestry: idea-032 (domain/llm: Wider-bandwidth, lower-alpha kernel ridge on early-time FPT )  ←  idea-026 (domain/llm: Wider-bandwidth kernel ridge on early-time FPT window)  ←  idea-022 (domain/llm: Stronger-regularised, smoother RBF kernel ridge on early-tim)  ←  idea-016 (domain/llm: Early-time FPT window + RBF kernel ridge)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00275` [checkpoint] by scheduler: final model (ensemble)
- `resu-00272` [result] by executor: exp-0043 rel_l2=0.01194 r2=0.9989 rmse=0.00764
- `hypo-00237` [hypothesis] by ideator-domain: Wide-bandwidth, low-alpha kernel ridge on log + FPT summary features
- `hypo-00238` [hypothesis] by ideator-domain: Wider-bandwidth, lower-alpha kernel ridge on early-time FPT window
- `plan-00271` [plan] by engineer: exp-0043: final fit of 2 member(s) on train+val
- `hypo-00202` [hypothesis] by ideator-domain: Kernel ridge on log + FPT summary features with strong regularisation
- `spec-00001` [spec] by scheduler: research spec graph_fpt_weights (3ed5b8db09)
- `data-00004` [dataset] by simulator: graph_fpt: 6000 samples, 576 features, target dim 12
- `hypo-00200` [hypothesis] by ideator-domain: Wider-bandwidth kernel ridge on early-time FPT window
- `hypo-00033` [hypothesis] by ideator-domain: Early-time FPT window + RBF kernel ridge
- `hypo-00167` [hypothesis] by ideator-domain: Stronger-regularised, smoother RBF kernel ridge on early-time FPT window

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|
| idea-009 | custom | critic: forbidden call eval(); forbidden call eval() |
| idea-014 | ridge | critic: duplicate of an existing configuration |
| idea-013 | custom | projected full-data cost 731s exceeds the 240s per-experiment limit |

## 8. Critic notes

- critic advice: No idea violates a constraint or is certain to fail, so I'm approving all of them. Points to fix during implementation:

(1) Scale is not id
- idea-002: train rel_l2=0.00515 vs val 0.04535: strong overfitting, more regularisation or data needed
- idea-003: train rel_l2=0.0003756 vs val 0.03257: strong overfitting, more regularisation or data needed
- idea-005: train rel_l2=2.518e-07 vs val 0.2027: strong overfitting, more regularisation or data needed
- idea-011: train rel_l2=0.04886 vs val 0.1539: strong overfitting, more regularisation or data needed
- idea-013: train rel_l2=0.0002245 vs val 0.04497: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.01637 vs val 0.05668: strong overfitting, more regularisation or data needed
- idea-007: train rel_l2=0.0157 vs val 0.04178: strong overfitting, more regularisation or data needed
- idea-008: train rel_l2=0.02924 vs val 0.09001: strong overfitting, more regularisation or data needed
- idea-015: train rel_l2=0.01927 vs val 0.05912: strong overfitting, more regularisation or data needed
- idea-016: train rel_l2=0.001567 vs val 0.01978: strong overfitting, more regularisation or data needed
- idea-017: train rel_l2=0.009326 vs val 0.3354: strong overfitting, more regularisation or data needed
- critic advice: All six ideas fit the objective and constraints: they train only on simulated data, select on val, and are affordable on CPU. Kernel ridge o
- idea-023: train rel_l2=0.002209 vs val 0.01555: strong overfitting, more regularisation or data needed
- critic advice: All six ideas fit the objective and constraints. None touch the test split, all train only on simulated data, and kernel ridge with n=4000 f
- idea-024: train rel_l2=0.00683 vs val 0.0148: strong overfitting, more regularisation or data needed
- idea-025: train rel_l2=0.008213 vs val 0.01939: strong overfitting, more regularisation or data needed
- idea-028: train rel_l2=0.006928 vs val 0.01432: strong overfitting, more regularisation or data needed
- idea-029: train rel_l2=0.007547 vs val 0.01971: strong overfitting, more regularisation or data needed
- critic advice: All six ideas are aligned: cheap kernel-ridge hyperparameter probes, no leakage, within budget. Several are near-duplicates (030/034 and 031
- final audit: test rel_l2=0.011943656404762963, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 719.5 / 900 worker-seconds over 43 jobs; wall time 581.9s
- By stage: baseline 1.9s, screen 174.0s, halving 154.0s, refine 381.5s, final 8.1s
- Most expensive ideas: idea-020 (torch_mlp) 212.6s, idea-007 (torch_mlp) 157.0s, idea-021 (torch_mlp) 94.1s, idea-013 (custom) 75.4s, idea-006 (torch_mlp) 23.1s, idea-015 (torch_mlp) 18.9s, idea-004 (pls) 11.2s, ensemble (final fit) 8.1s
- LLM: 13 calls, 231.9s, $1.0204 reported cost

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

The kernel ridge family accounts for the whole leaderboard. The final two-member ensemble (idea-031 on log+FPT-summary features plus idea-032 on early-time FPT features) reached val rel_l2 = 0.0123 and test rel_l2 = 0.0119 (R² = 0.9989, RMSE = 0.0076). That is about 30× better than the 0.359 baseline and well under the 0.05 success threshold. Test is no worse than val (0.0119 vs 0.0123), so selecting on validation does not appear to have overfit. The ensemble improved on the best single model (0.01318) by about 7%. Feature engineering mattered less than kernel tuning: the log+summary, fpt_early and sqrt transforms all land in a narrow band (0.0132–0.0158). The best configurations share small regularisation (α ≈ 0.0015–0.015) and a narrow RBF bandwidth (gamma_mult ≈ 0.06–0.13). Raising gamma_mult to 0.25–0.3 consistently cost about 0.001–0.002. These results suggest the map from first-passage features to weights is smooth and close to deterministic, so a well-regularised interpolator in a compressed feature space is enough.

The failures are almost all variance-dominated, given the 4000 noisy 576-dimensional training samples:
- **Memorisation:** idea-005 reached train 2.5e-7 against val 0.203.
- **Near-baseline collapse:** idea-017 reached train 0.0093 against val 0.335.
- **Generalisation gaps:** idea-011 (0.049 → 0.154) and idea-008 (0.029 → 0.090) both overfit badly.
- **Pruned runs:** idea-013 was cut for projected cost (731 s against a 240 s per-experiment limit), and its train/val gap of 0.0002 vs 0.045 suggests it would not have won anyway. idea-009 was rejected for a forbidden `eval()` call, and idea-014 was a duplicate configuration.

The critic's overfitting flag looks too strict. It also fired on idea-016 (0.0198) and idea-023 (0.0156), whose val scores are competitive. The ratio-based heuristic should be complemented by an absolute val threshold. We do not know how close 0.012 is to the noise floor: with N = 10⁴ walks, per-feature sampling error is around √(p/N) ≈ 10⁻³–5·10⁻³, and we have not yet measured how that propagates into the weights.

- **Estimate the irreducible error.** Re-simulate several fixed weight vectors under independent walk seeds, or compute a Cramér–Rao bound from the multinomial FPT likelihood. This tells us whether further regression gains are achievable.
- **Physics-based refinement.** Compute exact r(t, x, y) from the killed transition matrix as a differentiable function of the 12 weights. Starting from the KRR prediction, fit the weights to each sample's empirical r′ by variance-weighted least squares or MLE (a few Gauss–Newton steps). This exploits the known forward model rather than learning it.
- **Scale up simulated training data.** Use 20–50k samples with Nyström or random-feature KRR, which the constraints permit and which directly targets the variance-limited regime.
- **Per-edge error breakdown.** Compare the four interior edges incident to the unobserved centre vertex 4 against the boundary edges, then consider edge-specific bandwidths or targeted features if the interior edges dominate the residual.
- **Cross-validated stacking.** Stack the top KRR models across all three feature families instead of the fixed two-member average, since even the naive ensemble gained about 7%.

_Automatic summary:_ Best validation rel_l2 per method: kernel_ridge 0.01318, torch_mlp 0.01989, pls 0.02547, ridge 0.02658, mean_baseline 0.3591. 34 ideas were proposed; 3 were pruned by the critic or the feasibility screen and 22 were trained on the full training set.
