# Research report (V2): heat_v2_no_literature_s1

Run `heat_v2_no_literature_s1_20261109_010514` · 2026-11-09 01:06:20 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 62.3 (max 1.08e+03); Cramér–Rao relative std per parameter: k1 0.0215, k2 0.0058, k3 0.0052, k4 0.0174; implied rel_l2 floor ≈ **0.01851**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=1,it=30]`
- Validation rel_l2 (selection estimate): **0.019056** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018392**, r2 = **0.99697**, rmse = **0.0242** · bootstrap 95% CI for rel_l2: [0.01741, 0.01959]
- Per-target test rel_l2: k1 0.0270, k2 0.0060, k3 0.0052, k4 0.0240
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.11551 (vs 0.018392 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051189 (vs 0.018392 in distribution)
- Uncertainty calibration on test: 69.2% within 1σ, 95.3% within 2σ (ideal 68% / 95%)
- Inference latency: 8.54 ms/sample; fit 0.00425 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=1,it=30]` rel_l2=0.019056; **hybrid** `hybrid[ridge/log+logy->LM10]` rel_l2=0.019321; **data-driven** `torch_mlp/raw+logy` rel_l2=0.027237
- SUPPORTED — E-01 *Maximum-likelihood inversion through the forward operator*: val rel_l2 0.019335 vs best_data_driven 0.029221: gain +33.83%; expected gain >= +10.0% vs best_data_driven
- SUPPORTED — E-02 *Learned initializer + Gauss–Newton/LM polishing (hybrid)*: val rel_l2 0.01934 vs best_physics 0.019056: gain -1.49%; expected gain >= -2.0% vs best_physics
- SUPPORTED — E-05 *Kernel ridge on sqrt features with PCA denoising*: val rel_l2 0.029459 vs best_data_driven 0.029221: gain -0.82%; expected gain >= -10.0% vs best_data_driven
- SUPPORTED — E-06 *Posterior-mean (MMSE) estimator under the known prior*: val rel_l2 0.019056 vs best_physics 0.019313: gain +1.33%; expected gain >= +0.0% vs best_physics
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01851

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- no citations in proposals

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- P-011 *Matched-noise training augmentation* (conjectured): proposed but not executed
- P-012 *Random-search control arm* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-03 *torch_mlp on log features with log-conductivity targets*: val rel_l2 = 0.029221 vs predicted < 0.02406
- CONTRADICTED — E-04 *MAP estimate regularised toward the prior centre*: val rel_l2 0.019313 vs best_physics 0.019056: gain -1.35%; expected gain >= +0.0% vs best_physics
- PRUNED — idea-009 `svm/log+pca8`: projected full-fidelity cost 185s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Maximum-likelihood inversion through the forward operator): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-01 (Maximum-likelihood inversion through the forward operator): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-02 (Learned initializer + Gauss–Newton/LM polishing (hybrid)): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-03 (torch_mlp on log features with log-conductivity targets): question: the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: state the evidence explicitly
- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-04 (MAP estimate regularised toward the prior centre): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-04 (MAP estimate regularised toward the prior centre): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-05 (Kernel ridge on sqrt features with PCA denoising): question: the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-05 (Kernel ridge on sqrt features with PCA denoising): revision_direction: state the evidence explicitly
- E-05 (Kernel ridge on sqrt features with PCA denoising): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-06 (Posterior-mean (MMSE) estimator under the known prior): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-06 (Posterior-mean (MMSE) estimator under the known prior): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- Literature check INCOMPLETE: 10 failed queries (disabled), 0 skipped for budget, enabled=False
- Test-set sampling uncertainty: bootstrap 95% CI [0.01741, 0.01959] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-exploratory | defend | P-007 | - | heuristic-fallback | Defended: assumptions and test are explicit; the experiment decides. |
| 2 | researcher-exploratory | revise | P-008 | P-013 | heuristic-fallback | Revised: incorporated critic assumptions |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.878 | fdbk-00040 |
| 2 | 8 | 0.878 | fdbk-00068 |

### Idea evolution

**P-001 — Maximum-likelihood inversion through the forward operator** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, heuristic-fallback] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* Per-sample MLE through the exact forward model is (near) efficient: it reaches the Cramér–Rao floor and beats every learned inverse at this training-set size.
  - prediction: `{"relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: model mismatch (chi2/dof >> 1), multimodality (single start insufficient) or non-identifiability; check chi2 and multistart
  - critic r1 (quality 0.9): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-002 — MAP estimate regularised toward the prior centre** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, heuristic-fallback] `physics-MAP[log,ms=1,it=30]` status conjectured. *H:* Weakly determined parameters ['k1', 'k4'] dominate the error; a Gaussian (Tikhonov) prior toward the box centre trades a little bias for less variance and lowers rel_l2.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: the prior box already constrains the estimate; regularisation only adds bias
  - critic r1 (quality 0.9): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-003 — torch_mlp on log features with log-conductivity targets** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, heuristic-fallback] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02406}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - critic r1 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-009` r2 [revise, heuristic-fallback] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02406}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-004 — Kernel ridge on sqrt features with PCA denoising** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, heuristic-fallback] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - critic r1 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-010` r2 [revise, heuristic-fallback] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Matched-noise training augmentation** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - critic r1 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-011` r2 [revise, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.856): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-006 — Random-search control arm** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - critic r1 (quality 0.856): 
- v2 `P-012` r2 [revise, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.856): 

**P-007 — Learned initializer + Gauss–Newton/LM polishing (hybrid)** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, heuristic-fallback] `hybrid[ridge/log+logy->LM10]` status conjectured. *H:* A cheap learned inverse places LM in the right basin, so a few iterations reach MLE accuracy without multistart; the hybrid combines ML speed with physics accuracy.
  - prediction: `{"relative_to": "best_physics", "improvement": -0.02, "expect": "better"}`; if it fails: the initializer lands outside the basin; multistart needed
  - critic r1 (quality 0.9): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-008 — Posterior-mean (MMSE) estimator under the known prior** (researcher-exploratory, family physics)
- v1 `P-008` r1 [propose, heuristic-fallback] `physics-MMSE[log,ms=1,it=30]` status conjectured. *H:* The metric is squared-error based and the prior is known (uniform box), so the posterior mean — not the likelihood mode — is the optimal estimator.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: posterior is nearly Gaussian and symmetric; mode = mean
  - critic r1 (quality 0.9): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, heuristic-fallback] `physics-MMSE[log,ms=1,it=30]` status conjectured. *H:* The metric is squared-error based and the prior is known (uniform box), so the posterior mean — not the likelihood mode — is the optimal estimator.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: posterior is nearly Gaussian and symmetric; mode = mean
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.9): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?


**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MLE[log,ms=1,it=30]`: ok rel_l2=0.019024 on 60 val samples in 1.96 s
- P-002 `physics-MAP[log,ms=1,it=30]`: ok rel_l2=0.019064 on 60 val samples in 1.96 s
- P-008 `physics-MMSE[log,ms=1,it=30]`: ok rel_l2=0.018396 on 60 val samples in 2.52 s

## 5. Literature retrieval (TheoremSearch)

0 queries, 10 failures, 0 distinct statements retrieved, 0 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | identifiability of piecewise-constant (discontinuous) conductivity in the parabolic (heat… | FAILED: disabled |
| researcher-mathematical | 1 | Lipschitz stability for an inverse coefficient problem of a parabolic equation with point… | FAILED: disabled |
| researcher-ml | 1 | neural network approximation of the solution operator of a parameter identification inver… | FAILED: disabled |
| researcher-ml | 1 | learning rates of kernel ridge regression for smooth regression functions | FAILED: disabled |
| researcher-exploratory | 1 | posterior mean estimator for Bayesian inverse problems with a uniform prior | FAILED: disabled |
| researcher-exploratory | 1 | learned initialization for Gauss-Newton iterations in nonlinear inverse problems | FAILED: disabled |
| researcher-experimental | 1 | Cramer-Rao lower bound for parameter estimation in partial differential equations | FAILED: disabled |
| researcher-experimental | 1 | optimal sensor placement for parameter identification in diffusion equations | FAILED: disabled |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_literature_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 63 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | physics | `physics-MLE[log,ms=1,it=30]` | researcher-mathematical | 0.9 | best-rated proposal in family 'physics' (quality 0.90) |
| E-02 | proposal (P-007) | hybrid | `hybrid[ridge/log+logy->LM10]` | researcher-exploratory | 0.9 | best-rated proposal in family 'hybrid' (quality 0.90) |
| E-03 | proposal (P-009) | data-driven | `torch_mlp/log+logy` | researcher-ml | 0.856 | best-rated proposal in family 'data-driven' (quality 0.86) |
| E-04 | proposal (P-002) | physics | `physics-MAP[log,ms=1,it=30]` | researcher-mathematical | 0.9 | quality 0.90 with diversity penalty |
| E-05 | proposal (P-010) | data-driven | `kernel_ridge/sqrt+pca32` | researcher-ml | 0.856 | quality 0.86 with diversity penalty |
| E-06 | proposal (P-013) | physics | `physics-MMSE[log,ms=1,it=30]` | researcher-exploratory | 0.9 | quality 0.90 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-011 *Matched-noise training augmentation* — portfolio size limit (6); quality 0.86
- not advanced: P-012 *Random-search control arm* — portfolio size limit (6); quality 0.86

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-007 | E-06 | research | physics | `physics-MMSE[log,ms=1,it=30]` | 0.02027 | **0.019056** | 0.9967 | 0.02513 | 0.0121 | 9.1 | 151321 | 0.952 | FINAL |
| idea-005 | E-04 | research | physics | `physics-MAP[log,ms=1,it=30]` | 0.02138 | **0.019313** | 0.9966 | 0.02547 | 0.00947 | 4.59 | 61292 | 0.957 | evaluated |
| idea-011 | E-02 | refine | hybrid | `hybrid[ridge/log+logy->LM10]` | - | **0.019321** | 0.9966 | 0.02548 | 0.0236 | 3.35 | 48278 | 0.957 | evaluated |
| idea-002 | E-01 | research | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0106 | 4.66 | 61517 | 0.957 | evaluated |
| idea-003 | E-02 | research | hybrid | `hybrid[ridge/log+logy->LM10]` | 0.02139 | **0.01934** | 0.9966 | 0.0255 | 0.0223 | 3.45 | 48377 | 0.957 | evaluated |
| idea-012 | E-02 | refine | hybrid | `hybrid[ridge/log+logy->LM9]` | - | **0.01934** | 0.9966 | 0.0255 | 0.026 | 3.32 | 48116 | 0.957 | evaluated |
| idea-013 | E-03 | refine | data-driven | `torch_mlp/raw+logy` | - | **0.027237** | 0.9932 | 0.03592 | 11.9 | 0.00321 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/log+logy` | 0.05481 | **0.029221** | 0.9921 | 0.03853 | 11.2 | 0.0198 | 0 | - | evaluated |
| idea-006 | E-05 | research | data-driven | `kernel_ridge/sqrt+pca32` | 0.05237 | **0.029459** | 0.992 | 0.03885 | 0.473 | 0.0465 | 0 | - | evaluated |
| idea-014 | E-03 | refine | data-driven | `torch_mlp/log+logy` | - | **0.030219** | 0.9916 | 0.03985 | 8.62 | 0.0118 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **0.032246** | 0.9904 | 0.04252 | 5.99 | 0 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 2.42 | 0.0906 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 185s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-007` `physics-MMSE[log,ms=1,it=30]` ← P-013 v2 (researcher-exploratory, r2, revise) ← P-008 v1 (researcher-exploratory, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00186` [checkpoint] by scheduler: final method (single)
- `resu-00183` [result] by executor: exp-0026 rel_l2=0.01839 r2=0.997 rmse=0.0242
- `hypo-00084` [hypothesis] by scheduler: idea-007 physics-MMSE[log,ms=1,it=30] (research)
- `plan-00182` [plan] by engineer-v2: exp-0026: final fit of 1 member(s) on train+val; test once
- `gate-00083` [gate] by gate: E-06: approved (physics-MMSE[log,ms=1,it=30])
- `prop-00066` [proposal] by researcher-exploratory: P-013 v2 [revise] exploratory: Posterior-mean (MMSE) estimator under the known prior
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_literature_s1 (9f0f28a432)
- `prop-00038` [proposal] by researcher-exploratory: P-008 v1 [propose] exploratory: Posterior-mean (MMSE) estimator under the known prior
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 144.4 / 400 worker-seconds over 26 jobs; wall 65 s
- By stage: baseline 1.9s, pilot 6.4s, screen 31.6s, evaluate 53.3s, refine 34.1s, final 16.9s
- LLM: 10 calls (by phase {'research': 10}), 0 s, $0.000 reported · TheoremSearch: 0 queries (0 failed)

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

13 proposal versions over 2 research rounds; 8 methods advanced; 4 predictions supported, 2 contradicted. Final: ['physics-MMSE[log,ms=1,it=30]'] with test rel_l2 0.018392.
