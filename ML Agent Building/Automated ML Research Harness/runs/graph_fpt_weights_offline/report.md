# Research report: graph_fpt_weights

Run `graph_fpt_weights_20261107_154519` · generated 2026-11-07 15:47:44 · LLM backend: **none** · workers: 6 × 2 threads · GPUs: none

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

- Final model: **single** of idea-019
  - `idea-019` ridge refinement of idea-002: {"alpha": 0.7285632345290244} — method `ridge`, features `raw`, params `{"alpha": 0.7285632345290244}`
- Validation rel_l2 (selection estimate): **0.013792**; baseline (mean predictor): 0.35913
- **Test metrics (held-out, evaluated once):** rel_l2 = **0.013828**, r2 = **0.99849**, rmse = **0.0088452**
- Success criterion (rel_l2 <= 0.05): **MET**
- Model file: `runs\graph_fpt_weights_offline\final\model.pkl` — load with `from mlharness.export import load_model; m = load_model(r"runs\graph_fpt_weights_offline")`
- Stopping reason for the search: no relative improvement >= 0.005 for 2 round(s)

## 3. Research funnel

| step | count |
|---|---|
| proposed | 16 |
| prescreen_approved | 16 |
| screen_survivors | 16 |
| rung_0.33 | 6 |
| rung_1.00 | 4 |
| refine_proposed | 12 |
| refine_approved | 12 |
| ideas pruned (all reasons) | 0 |

## 4. Leaderboard (full training data, validation split)

| # | idea | persona | origin | method | features | params | val rel_l2 | val r2 | val rmse | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | idea-019 | classical | heuristic | ridge | raw | `{"alpha": 0.7285632345290244}` | 0.013792 | 0.99851 | 0.0088515 | 0.092 |
| 2 | idea-029 | classical | heuristic | ridge | raw | `{"alpha": 0.892436569744487}` | 0.013798 | 0.99851 | 0.0088556 | 0.127 |
| 3 | idea-002 | classical | heuristic | ridge | raw | `{"alpha": 1.0}` | 0.013803 | 0.99851 | 0.0088588 | 0.0818 |
| 4 | idea-018 | classical | heuristic | ridge | raw | `{"alpha": 1.1767045470382502}` | 0.013812 | 0.99851 | 0.0088645 | 0.089 |
| 5 | idea-027 | classical | heuristic | ridge | raw | `{"alpha": 1.506725934055569}` | 0.013831 | 0.99851 | 0.0088767 | 0.12 |
| 6 | idea-028 | classical | heuristic | ridge | raw | `{"alpha": 1.5430466854423528}` | 0.013833 | 0.99851 | 0.0088781 | 0.127 |
| 7 | idea-022 | domain | heuristic | torch_mlp | log+fpt_summary | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.003, "weig...` | 0.016216 | 0.99795 | 0.010407 | 26.2 |
| 8 | idea-016 | domain | heuristic | torch_mlp | log+fpt_summary | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.018548 | 0.99732 | 0.011904 | 23.6 |
| 9 | idea-023 | neural | heuristic | torch_mlp | log+fpt_summary | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.019085 | 0.99717 | 0.012248 | 26 |
| 10 | idea-020 | domain | heuristic | torch_mlp | fpt_early | `{"hidden": [256, 256], "dropout": 0.0, "lr": 0.001, "weig...` | 0.019575 | 0.99701 | 0.012563 | 19.7 |
| 11 | idea-014 | neural | heuristic | torch_mlp | log | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.020665 | 0.99668 | 0.013263 | 27.1 |
| 12 | idea-021 | neural | heuristic | torch_mlp | fpt_early | `{"hidden": [256, 256], "dropout": 0.08954734302423699, "l...` | 0.022562 | 0.99603 | 0.01448 | 24.7 |
| 13 | idea-008 | domain | heuristic | ridge | log | `{"alpha": 1.0}` | 0.026577 | 0.99449 | 0.017057 | 0.12 |
| 14 | idea-025 | classical | heuristic | ridge | log | `{"alpha": 0.5013483348806097}` | 0.026595 | 0.99448 | 0.017069 | 0.163 |
| 15 | idea-024 | classical | heuristic | ridge | raw | `{"alpha": 0.7376520354075256}` | 0.031754 | 0.99214 | 0.02038 | 0.245 |
| 16 | idea-026 | classical | heuristic | ridge | raw | `{"alpha": 1.1314377407226477}` | 0.031754 | 0.99214 | 0.02038 | 0.24 |
| 17 | idea-001 | baseline | scheduler | mean_baseline | raw | `{}` | 0.35913 | -0.0017183 | 0.23049 | 0 |

## 5. Multi-fidelity trace (validation rel_l2 by training-data fraction)

| idea | method | 0.10 | 0.33 | 1.00 | outcome |
|---|---|---|---|---|---|
| idea-001 | mean_baseline | - | - | 0.3591 | reference baseline |
| idea-002 | ridge | 0.03231 | 0.01642 | 0.0138 | evaluated at full fidelity |
| idea-003 | pls | 0.1687 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-005 | kernel_ridge | 0.08603 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-007 | svm | 0.08968 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-004 | extra_trees | 0.224 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-009 | random_forest | 0.2365 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-012 | hist_gbm | 0.05685 | 0.04103 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-013 | extra_trees | 0.2616 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-006 | torch_mlp | 0.06969 | 0.03648 | - | not_promoted: ranked below the top 4 at fraction 0.33 |
| idea-010 | mlp_sklearn | 0.098 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-014 | torch_mlp | 0.06398 | 0.03186 | 0.02067 | evaluated at full fidelity |
| idea-015 | mlp_sklearn | 0.09582 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-008 | ridge | 0.04535 | 0.03155 | 0.02658 | evaluated at full fidelity |
| idea-011 | kernel_ridge | 0.07733 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-016 | torch_mlp | 0.06399 | 0.03242 | 0.01855 | evaluated at full fidelity |
| idea-017 | extra_trees | 0.2231 | - | - | not_promoted: ranked below the top 6 at fraction 0.10 |
| idea-018 | ridge | - | - | 0.01381 | evaluated at full fidelity |
| idea-019 | ridge | - | - | 0.01379 | FINAL MODEL |
| idea-020 | torch_mlp | - | - | 0.01958 | evaluated at full fidelity |
| idea-022 | torch_mlp | - | - | 0.01622 | evaluated at full fidelity |
| idea-021 | torch_mlp | - | - | 0.02256 | evaluated at full fidelity |
| idea-023 | torch_mlp | - | - | 0.01908 | evaluated at full fidelity |
| idea-024 | ridge | - | - | 0.03175 | evaluated at full fidelity |
| idea-025 | ridge | - | - | 0.0266 | evaluated at full fidelity |
| idea-026 | ridge | - | - | 0.03175 | evaluated at full fidelity |
| idea-027 | ridge | - | - | 0.01383 | evaluated at full fidelity |
| idea-028 | ridge | - | - | 0.01383 | evaluated at full fidelity |
| idea-029 | ridge | - | - | 0.0138 | evaluated at full fidelity |

## 6. Lineage of the final model

- Idea ancestry: idea-019 (classical/heuristic: ridge refinement of idea-002: {"alpha": 0.7285632345290244})  ←  idea-002 (classical/heuristic: ridge on 'raw' features)

Artifact trace (checkpoint → result → plan → hypothesis → …):

- `chec-00231` [checkpoint] by scheduler: final model (single)
- `resu-00228` [result] by executor: exp-0040 rel_l2=0.01383 r2=0.9985 rmse=0.008845
- `hypo-00163` [hypothesis] by ideator-classical: ridge refinement of idea-002: {"alpha": 0.7285632345290244}
- `plan-00227` [plan] by engineer: exp-0040: final fit of 1 member(s) on train+val
- `hypo-00015` [hypothesis] by ideator-classical: ridge on 'raw' features
- `spec-00001` [spec] by scheduler: research spec graph_fpt_weights (7b137b6a2f)
- `data-00004` [dataset] by simulator: graph_fpt: 6000 samples, 576 features, target dim 12

## 7. Pruned ideas and failures

| idea | method | reason |
|---|---|---|

## 8. Critic notes

- idea-005: train rel_l2=0.004691 vs val 0.08603: strong overfitting, more regularisation or data needed
- idea-007: train rel_l2=0.03794 vs val 0.08968: strong overfitting, more regularisation or data needed
- idea-012: train rel_l2=0.0135 vs val 0.04103: strong overfitting, more regularisation or data needed
- idea-006: train rel_l2=0.01266 vs val 0.03648: strong overfitting, more regularisation or data needed
- idea-010: train rel_l2=0.03143 vs val 0.098: strong overfitting, more regularisation or data needed
- idea-014: train rel_l2=0.01592 vs val 0.03186: strong overfitting, more regularisation or data needed
- idea-015: train rel_l2=0.03195 vs val 0.09582: strong overfitting, more regularisation or data needed
- idea-008: train rel_l2=0.00515 vs val 0.04535: strong overfitting, more regularisation or data needed
- idea-011: train rel_l2=0.004228 vs val 0.07733: strong overfitting, more regularisation or data needed
- idea-016: train rel_l2=0.00739 vs val 0.01855: strong overfitting, more regularisation or data needed
- final audit: test rel_l2=0.013827545828826, success=True, issues=0

## 9. Compute and resource usage

- Compute used: 402.3 / 900 worker-seconds over 40 jobs; wall time 144.8s
- By stage: baseline 1.9s, screen 96.2s, halving 175.2s, refine 126.6s, final 2.4s
- Most expensive ideas: idea-012 (hist_gbm) 76.7s, idea-014 (torch_mlp) 61.9s, idea-016 (torch_mlp) 54.6s, idea-022 (torch_mlp) 28.9s, idea-023 (torch_mlp) 28.7s, idea-021 (torch_mlp) 27.4s, idea-006 (torch_mlp) 24.6s, idea-020 (torch_mlp) 22.4s
- LLM: 0 calls, 0.0s, $0.0000 reported cost

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **ideator-classical** (ideation, persona classical) — none
- **ideator-ensembles** (ideation, persona ensembles) — none
- **ideator-neural** (ideation, persona neural) — none
- **ideator-domain** (ideation, persona domain) — none

## 11. Discussion

Best validation rel_l2 per method: ridge 0.01379, torch_mlp 0.01622, mean_baseline 0.3591. 28 ideas were proposed; 0 were pruned by the critic or the feasibility screen and 16 were trained on the full training set.
