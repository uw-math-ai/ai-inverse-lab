# Research report (V2): heat_v2_no_forward_s1

Run `heat_v2_no_forward_s1_20261109_005800` · 2026-11-09 01:00:21 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **ensemble** of `torch_mlp/sqrt+logy+aug1`, `torch_mlp/raw+logy+aug1` (weights [2.0, 1.0])
- Validation rel_l2 (selection estimate): **0.023096** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.02181**, r2 = **0.99572**, rmse = **0.028698** · bootstrap 95% CI for rel_l2: [0.02067, 0.02295]
- Per-target test rel_l2: k1 0.0311, k2 0.0108, k3 0.0098, k4 0.0274
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.077283 (vs 0.02181 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.057697 (vs 0.02181 in distribution)
- Inference latency: 0.00908 ms/sample; fit 62.8 s; exported model reload check: max |Δ| = 0
- Ensemble members on test: `torch_mlp/sqrt+logy+aug1` 0.02278, `torch_mlp/raw+logy+aug1` 0.02358
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **data-driven** `torch_mlp/sqrt+logy+aug1` rel_l2=0.023874
- SUPPORTED — E-01 *Log-linearisation of diffusive decay*: val rel_l2 0.03332 vs best_other 0.027918: gain -19.35%; expected gain >= -20.0% vs best_other
- SUPPORTED — E-02 *Kernel ridge regression on log features*: val rel_l2 0.030624 vs baseline_mean 0.3307: gain +90.74%; expected gain >= +90.0% vs baseline_mean
- SUPPORTED — E-03 *torch_mlp on log features with log-conductivity targets*: val rel_l2 0.029221 vs baseline_mean 0.3307: gain +91.16%; expected gain >= +90.0% vs baseline_mean
- SUPPORTED — E-04 *Kernel ridge on sqrt features with PCA denoising*: val rel_l2 0.029459 vs best_data_driven 0.027918: gain -5.52%; expected gain >= -10.0% vs best_data_driven
- SUPPORTED — E-05 *Matched-noise training augmentation*: val rel_l2 0.027918 vs best_data_driven 0.029221: gain +4.46%; expected gain >= -5.0% vs best_data_driven
- SUPPORTED — E-06 *Random-search control arm*: val rel_l2 0.082482 vs best_other 0.027918: gain -195.44%; expected to be worse than best_other (gain < +0.0%)

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:22026323` Herbert Egger, Jan-Frederik Pietschmann, Matthias Schlottbom (2014). "Identification of nonlinear heat conduction laws". Theorem 5. http://arxiv.org/abs/1404.2535v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- P-013 *Partial least squares on log features* (conjectured): proposed but not executed
- P-014 *Gradient boosting on sqrt features* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- PRUNED — idea-007 `svm/log+pca8`: projected full-fidelity cost 191s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: state the evidence explicitly
- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: the portfolio is dominated by 'data-driven' methods: branch into an under-represented family
- E-04 (Kernel ridge on sqrt features with PCA denoising): revision_direction: state the evidence explicitly
- E-05 (Matched-noise training augmentation): revision_direction: state the evidence explicitly
- E-06 (Random-search control arm): revision_direction: state the evidence explicitly
- Test-set sampling uncertainty: bootstrap 95% CI [0.02067, 0.02295] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **2 round(s)**: quality criteria met after 2 rounds (every active proposal >= 0.75 quality, no blocking issues, critic converged).

| round | researcher | action | proposal | new version | origin | response to critic |
|---|---|---|---|---|---|---|
| 1 | researcher-mathematical | propose | - | P-001 | heuristic-fallback |  |
| 1 | researcher-mathematical | propose | - | P-002 | heuristic-fallback |  |
| 1 | researcher-ml | propose | - | P-003 | heuristic-fallback |  |
| 1 | researcher-ml | propose | - | P-004 | heuristic-fallback |  |
| 1 | researcher-experimental | propose | - | P-005 | heuristic-fallback |  |
| 1 | researcher-experimental | propose | - | P-006 | heuristic-fallback |  |
| 1 | researcher-exploratory | propose | - | P-007 | heuristic-fallback |  |
| 1 | researcher-exploratory | propose | - | P-008 | heuristic-fallback |  |
| 2 | researcher-mathematical | defend | P-001 | - | heuristic-fallback | Defended: assumptions and test are explicit; the experiment decides. |
| 2 | researcher-mathematical | defend | P-002 | - | heuristic-fallback | Defended: assumptions and test are explicit; the experiment decides. |
| 2 | researcher-ml | revise | P-003 | P-009 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-ml | revise | P-004 | P-010 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-experimental | revise | P-005 | P-011 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-experimental | revise | P-006 | P-012 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-exploratory | revise | P-007 | P-013 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-exploratory | revise | P-008 | P-014 | heuristic-fallback | Revised: filled missing fields |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.817 | fdbk-00040 |
| 2 | 8 | 0.817 | fdbk-00060 |

### Idea evolution

**P-001 — Log-linearisation of diffusive decay** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, heuristic-fallback] `ridge/log+logy` status conjectured. *H:* Diffusive decay is approximately exponential in time, so log temperatures are near-linear in the conductivities; a regularised linear model in log space is a strong inverse.
  - prediction: `{"relative_to": "best_other", "improvement": -0.2, "expect": "better"}`; if it fails: the inverse map is strongly nonlinear; need a nonlinear learner
  - critic r1 (quality 1.0): 

**P-002 — Kernel ridge regression on log features** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, heuristic-fallback] `kernel_ridge/log+pca32` status conjectured. *H:* The inverse map is smooth, so RBF kernel ridge on log features is data-efficient.
  - prediction: `{"relative_to": "baseline_mean", "improvement": 0.9, "expect": "better"}`; if it fails: bandwidth mis-specified or PCA discards information
  - critic r1 (quality 1.0): 

**P-003 — torch_mlp on log features with log-conductivity targets** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, heuristic-fallback] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"relative_to": "baseline_mean", "improvement": 0.9, "expect": "better"}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - critic r1 (quality 0.756): 
- v2 `P-009` r2 [revise, heuristic-fallback] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"relative_to": "baseline_mean", "improvement": 0.9, "expect": "better"}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-004 — Kernel ridge on sqrt features with PCA denoising** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, heuristic-fallback] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - critic r1 (quality 0.756): 
- v2 `P-010` r2 [revise, heuristic-fallback] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-005 — Matched-noise training augmentation** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - critic r1 (quality 0.756): 
- v2 `P-011` r2 [revise, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-006 — Random-search control arm** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - critic r1 (quality 0.756): 
- v2 `P-012` r2 [revise, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-007 — Partial least squares on log features** (researcher-exploratory, family data-driven)
- v1 `P-007` r1 [propose, heuristic-fallback] `pls/log` status conjectured. *H:* A few latent directions explain the observations (smoothing operator), so PLS in log space is a competitive, cheap inverse.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.3, "expect": "better"}`; if it fails: nonlinearity dominates
  - critic r1 (quality 0.756): 
- v2 `P-013` r2 [revise, heuristic-fallback] `pls/log` status conjectured. *H:* A few latent directions explain the observations (smoothing operator), so PLS in log space is a competitive, cheap inverse.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.3, "expect": "better"}`; if it fails: nonlinearity dominates
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-008 — Gradient boosting on sqrt features** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, heuristic-fallback] `hist_gbm/sqrt` status conjectured. *H:* Tree boosting captures interactions between segments without feature engineering.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.4, "expect": "better"}`; if it fails: piecewise-constant trees approximate a smooth map poorly
  - critic r1 (quality 0.756): 
- v2 `P-014` r2 [revise, heuristic-fallback] `hist_gbm/sqrt` status conjectured. *H:* Tree boosting captures interactions between segments without feature engineering.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.4, "expect": "better"}`; if it fails: piecewise-constant trees approximate a smooth map poorly
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

- Critic round 1 portfolio view:  8/8 active proposals are 'data-driven' families={'data-driven': 8}
- Critic round 2 portfolio view:  8/8 active proposals are 'data-driven' families={'data-driven': 8}

## 5. Literature retrieval (TheoremSearch)

14 queries, 0 failures, 56 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | neural network approximation of the solution operator of a parameter identification inver… | 4 refs via /search (cache) |
| researcher-ml | 1 | learning rates of kernel ridge regression for smooth regression functions | 4 refs via /search (cache) |
| researcher-mathematical | 1 | identifiability of piecewise-constant (discontinuous) conductivity in the parabolic (heat… | 4 refs via /search (cache) |
| researcher-mathematical | 1 | Lipschitz stability for an inverse coefficient problem of a parabolic equation with point… | 4 refs via /search (cache) |
| researcher-exploratory | 1 | posterior mean estimator for Bayesian inverse problems with a uniform prior | 4 refs via /search (cache) |
| researcher-exploratory | 1 | learned initialization for Gauss-Newton iterations in nonlinear inverse problems | 4 refs via /search (cache) |
| researcher-experimental | 1 | Cramer-Rao lower bound for parameter estimation in partial differential equations | 4 refs via /search (cache) |
| researcher-experimental | 1 | optimal sensor placement for parameter identification in diffusion equations | 4 refs via /search (cache) |
| researcher-ml | 2 | torch_mlp on log features with log-conductivity targets | 4 refs via /search (cache) |
| researcher-experimental | 2 | Matched-noise training augmentation | 4 refs via /search (cache) |
| researcher-ml | 2 | Kernel ridge on sqrt features with PCA denoising | 4 refs via /search (cache) |
| researcher-experimental | 2 | Random-search control arm | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Partial least squares on log features | 4 refs via /search |
| researcher-exploratory | 2 | Gradient boosting on sqrt features | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 120 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | data-driven | `ridge/log+logy` | researcher-mathematical | 1 | best-rated proposal in family 'data-driven' (quality 1.00) |
| E-02 | proposal (P-002) | data-driven | `kernel_ridge/log+pca32` | researcher-mathematical | 1 | quality 1.00 with diversity penalty |
| E-03 | proposal (P-009) | data-driven | `torch_mlp/log+logy` | researcher-ml | 0.756 | quality 0.76 with diversity penalty |
| E-04 | proposal (P-010) | data-driven | `kernel_ridge/sqrt+pca32` | researcher-ml | 0.756 | quality 0.76 with diversity penalty |
| E-05 | proposal (P-011) | data-driven | `torch_mlp/log+logy+aug1` | researcher-experimental | 0.756 | quality 0.76 with diversity penalty |
| E-06 | proposal (P-012) | data-driven | `random-search x2` | researcher-experimental | 0.756 | quality 0.76 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
- not advanced: P-013 *Partial least squares on log features* — portfolio size limit (6); quality 0.76
- not advanced: P-014 *Gradient boosting on sqrt features* — portfolio size limit (6); quality 0.76

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-011 | E-05 | refine | data-driven | `torch_mlp/sqrt+logy+aug1` | - | **0.023874** | 0.9948 | 0.03148 | 27.1 | 0.0171 | 0 | - | FINAL |
| idea-010 | E-05 | refine | data-driven | `torch_mlp/raw+logy+aug1` | - | **0.025237** | 0.9941 | 0.03328 | 30.2 | 0.00536 | 0 | - | FINAL |
| idea-013 | E-03 | refine | data-driven | `torch_mlp/log+logy` | - | **0.027648** | 0.993 | 0.03646 | 11.5 | 0 | 0 | - | evaluated |
| idea-006 | E-05 | research | data-driven | `torch_mlp/log+logy+aug1` | 0.05022 | **0.027918** | 0.9928 | 0.03682 | 13.7 | 0 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/log+logy` | 0.05481 | **0.029221** | 0.9921 | 0.03853 | 10.9 | 0 | 0 | - | evaluated |
| idea-005 | E-04 | research | data-driven | `kernel_ridge/sqrt+pca32` | 0.05237 | **0.029459** | 0.992 | 0.03885 | 0.457 | 0.0473 | 0 | - | evaluated |
| idea-012 | E-03 | refine | data-driven | `torch_mlp/log+logy` | - | **0.029648** | 0.9919 | 0.0391 | 11.3 | 0.00534 | 0 | - | evaluated |
| idea-003 | E-02 | research | data-driven | `kernel_ridge/log+pca32` | 0.0553 | **0.030624** | 0.9914 | 0.04038 | 0.467 | 0.0519 | 0 | - | evaluated |
| idea-009 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **0.032246** | 0.9904 | 0.04252 | 6.18 | 0.00499 | 0 | - | evaluated |
| idea-002 | E-01 | research | data-driven | `ridge/log+logy` | 0.03931 | **0.03332** | 0.9898 | 0.04394 | 0.0131 | 0 | 0 | - | evaluated |
| idea-008 | E-06 | research | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 2.32 | 0.107 | 0 | - | evaluated |
| idea-007 | E-06 | research | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 191s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-011` `torch_mlp/sqrt+logy+aug1` ← refined from idea-006 ← P-011 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)
- `idea-010` `torch_mlp/raw+logy+aug1` ← refined from idea-006 ← P-011 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00168` [checkpoint] by scheduler: final method (ensemble)
- `resu-00165` [result] by executor: exp-0021 rel_l2=0.02181 r2=0.9957 rmse=0.0287
- `hypo-00137` [hypothesis] by scheduler: idea-011 torch_mlp/sqrt+logy+aug1 (refine)
- `hypo-00135` [hypothesis] by scheduler: idea-010 torch_mlp/raw+logy+aug1 (refine)
- `plan-00164` [plan] by engineer-v2: exp-0021: final fit of 2 member(s) on train+val; test once
- `gate-00136` [gate] by gate: refine idea-006: approved (torch_mlp/sqrt+logy+aug1)
- `prop-00055` [proposal] by researcher-experimental: P-011 v2 [revise] experimental: Matched-noise training augmentation
- `hypo-00074` [hypothesis] by scheduler: idea-006 torch_mlp/log+logy+aug1 (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s1 (8e97e94bdd)
- `gate-00134` [gate] by gate: refine idea-006: approved (torch_mlp/raw+logy+aug1)
- `prop-00035` [proposal] by researcher-experimental: P-005 v1 [propose] experimental: Matched-noise training augmentation
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00073` [gate] by gate: E-05: approved (torch_mlp/log+logy+aug1)

## 9. Compute and cost

- Execution compute: 238.4 / 400 worker-seconds over 21 jobs; wall 141 s
- By stage: baseline 2.0s, screen 31.9s, evaluate 49.9s, refine 89.5s, final 65.2s
- LLM: 10 calls (by phase {'research': 10}), 0 s, $0.000 reported · TheoremSearch: 14 queries (0 failed)

## 10. Agents

- **reporter** (reporter) — claude_code
- **simulator** (simulation) — claude_code
- **critic** (critic) — claude_code
- **engineer** (engineer) — claude_code
- **engineer-v2** (engineer) — claude_code
- **researcher-mathematical** (researcher, mathematical) — claude_code
- **researcher-ml** (researcher, ml) — claude_code
- **researcher-experimental** (researcher, experimental) — claude_code
- **researcher-exploratory** (researcher, exploratory) — claude_code
- **research-critic** (research_critic) — claude_code
- **reporter-v2** (reporter) — claude_code

## 11. Discussion

14 proposal versions over 2 research rounds; 7 methods advanced; 6 predictions supported, 0 contradicted. Final: ['torch_mlp/sqrt+logy+aug1', 'torch_mlp/raw+logy+aug1'] with test rel_l2 0.02181.
