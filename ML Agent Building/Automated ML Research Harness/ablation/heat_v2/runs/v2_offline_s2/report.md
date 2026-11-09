# Research report (V2): heat_v2_offline_s2

Run `heat_v2_offline_s2_20261109_001246` · 2026-11-09 00:13:52 · pipeline **v2 (research-first)** · LLM backend **none** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=1,it=30]`
- Validation rel_l2 (selection estimate): **0.018997** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018402**, r2 = **0.99697**, rmse = **0.024214** · bootstrap 95% CI for rel_l2: [0.01732, 0.01953]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.10833 (vs 0.018402 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051296 (vs 0.018402 in distribution)
- Uncertainty calibration on test: 68.6% within 1σ, 95.2% within 2σ (ideal 68% / 95%)
- Inference latency: 8.79 ms/sample; fit 0.00901 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_offline_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_offline_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=1,it=30]` rel_l2=0.018997; **hybrid** `hybrid[ridge/log+logy->LM10]` rel_l2=0.01934; **data-driven** `torch_mlp/sqrt+logy` rel_l2=0.028831
- SUPPORTED — E-01 *Maximum-likelihood inversion through the forward operator*: val rel_l2 0.019335 vs best_data_driven 0.029154: gain +33.68%; expected gain >= +10.0% vs best_data_driven
- SUPPORTED — E-02 *Learned initializer + Gauss–Newton/LM polishing (hybrid)*: val rel_l2 0.01934 vs best_physics 0.018997: gain -1.81%; expected gain >= -2.0% vs best_physics
- SUPPORTED — E-05 *Posterior-mean (MMSE) estimator under the known prior*: val rel_l2 0.018997 vs best_physics 0.019313: gain +1.64%; expected gain >= +0.0% vs best_physics
- SUPPORTED — E-06 *Kernel ridge on sqrt features with PCA denoising*: val rel_l2 0.029459 vs best_data_driven 0.029154: gain -1.05%; expected gain >= -10.0% vs best_data_driven
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:22026323` Herbert Egger, Jan-Frederik Pietschmann, Matthias Schlottbom (2014). "Identification of nonlinear heat conduction laws". Theorem 5. http://arxiv.org/abs/1404.2535v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:26263216` Bin Hu, Kaiqing Zhang, Na Li et al. (2022). "Towards a Theoretical Foundation of Policy Optimization for Learning Control Policies". Theorem 4 (Global Convergence \& Local Faster Rates). http://arxiv.org/abs/2210.04810v1 — cited by ['P-013', 'P-014']; researcher claim ['partial']; harness screen **plausible**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- P-011 *Matched-noise training augmentation* (conjectured): proposed but not executed
- P-012 *Random-search control arm* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-03 *torch_mlp on log features with log-conductivity targets*: val rel_l2 = 0.029154 vs predicted < 0.02412
- CONTRADICTED — E-04 *MAP estimate regularised toward the prior centre*: val rel_l2 0.019313 vs best_physics 0.018997: gain -1.67%; expected gain >= +0.0% vs best_physics

### 3.5 Remaining uncertainty

- E-01 (Maximum-likelihood inversion through the forward operator): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-01 (Maximum-likelihood inversion through the forward operator): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-02 (Learned initializer + Gauss–Newton/LM polishing (hybrid)): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-03 (torch_mlp on log features with log-conductivity targets): question: the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: state the evidence explicitly
- E-03 (torch_mlp on log features with log-conductivity targets): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-04 (MAP estimate regularised toward the prior centre): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-04 (MAP estimate regularised toward the prior centre): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-05 (Posterior-mean (MMSE) estimator under the known prior): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-05 (Posterior-mean (MMSE) estimator under the known prior): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-06 (Kernel ridge on sqrt features with PCA denoising): question: the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-06 (Kernel ridge on sqrt features with PCA denoising): revision_direction: state the evidence explicitly
- E-06 (Kernel ridge on sqrt features with PCA denoising): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- Test-set sampling uncertainty: bootstrap 95% CI [0.01732, 0.01953] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **2 round(s)**: quality criteria met after 2 rounds (every active proposal >= 0.75 quality, no blocking issues, critic converged).

| round | researcher | action | proposal | new version | origin | response to critic |
|---|---|---|---|---|---|---|
| 1 | researcher-mathematical | propose | - | P-001 | heuristic |  |
| 1 | researcher-mathematical | propose | - | P-002 | heuristic |  |
| 1 | researcher-ml | propose | - | P-003 | heuristic |  |
| 1 | researcher-ml | propose | - | P-004 | heuristic |  |
| 1 | researcher-experimental | propose | - | P-005 | heuristic |  |
| 1 | researcher-experimental | propose | - | P-006 | heuristic |  |
| 1 | researcher-exploratory | propose | - | P-007 | heuristic |  |
| 1 | researcher-exploratory | propose | - | P-008 | heuristic |  |
| 2 | researcher-mathematical | defend | P-001 | - | heuristic | Defended: assumptions and test are explicit; the experiment decides. |
| 2 | researcher-mathematical | defend | P-002 | - | heuristic | Defended: assumptions and test are explicit; the experiment decides. |
| 2 | researcher-ml | revise | P-003 | P-009 | heuristic | Revised: filled missing fields |
| 2 | researcher-ml | revise | P-004 | P-010 | heuristic | Revised: filled missing fields |
| 2 | researcher-experimental | revise | P-005 | P-011 | heuristic | Revised: filled missing fields |
| 2 | researcher-experimental | revise | P-006 | P-012 | heuristic | Revised: filled missing fields |
| 2 | researcher-exploratory | revise | P-007 | P-013 | heuristic | Revised: added requested literature with applicability notes |
| 2 | researcher-exploratory | revise | P-008 | P-014 | heuristic | Revised: added requested literature with applicability notes |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.841 | fdbk-00035 |
| 2 | 8 | 0.866 | fdbk-00063 |

### Idea evolution

**P-001 — Maximum-likelihood inversion through the forward operator** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, heuristic] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* Per-sample MLE through the exact forward model is (near) efficient: it reaches the Cramér–Rao floor and beats every learned inverse at this training-set size.
  - prediction: `{"relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: model mismatch (chi2/dof >> 1), multimodality (single start insufficient) or non-identifiability; check chi2 and multistart
  - critic r1 (quality 1.0): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-002 — MAP estimate regularised toward the prior centre** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, heuristic] `physics-MAP[log,ms=1,it=30]` status conjectured. *H:* Weakly determined parameters ['k1', 'k4'] dominate the error; a Gaussian (Tikhonov) prior toward the box centre trades a little bias for less variance and lowers rel_l2.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: the prior box already constrains the estimate; regularisation only adds bias
  - critic r1 (quality 1.0): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-003 — torch_mlp on log features with log-conductivity targets** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, heuristic] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02412}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - critic r1 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-009` r2 [revise, heuristic] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02412}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-004 — Kernel ridge on sqrt features with PCA denoising** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, heuristic] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - critic r1 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-010` r2 [revise, heuristic] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Matched-noise training augmentation** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, heuristic] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - critic r1 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-011` r2 [revise, heuristic] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-006 — Random-search control arm** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, heuristic] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - critic r1 (quality 0.756): 
- v2 `P-012` r2 [revise, heuristic] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-007 — Learned initializer + Gauss–Newton/LM polishing (hybrid)** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, heuristic] `hybrid[ridge/log+logy->LM10]` status conjectured. *H:* A cheap learned inverse places LM in the right basin, so a few iterations reach MLE accuracy without multistart; the hybrid combines ML speed with physics accuracy.
  - prediction: `{"relative_to": "best_physics", "improvement": -0.02, "expect": "better"}`; if it fails: the initializer lands outside the basin; multistart needed
  - critic r1 (quality 0.85): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, heuristic] `hybrid[ridge/log+logy->LM10]` status literature-supported. *H:* A cheap learned inverse places LM in the right basin, so a few iterations reach MLE accuracy without multistart; the hybrid combines ML speed with physics accuracy.
  - prediction: `{"relative_to": "best_physics", "improvement": -0.02, "expect": "better"}`; if it fails: the initializer lands outside the basin; multistart needed
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 0.95): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-008 — Posterior-mean (MMSE) estimator under the known prior** (researcher-exploratory, family physics)
- v1 `P-008` r1 [propose, heuristic] `physics-MMSE[log,ms=1,it=30]` status conjectured. *H:* The metric is squared-error based and the prior is known (uniform box), so the posterior mean — not the likelihood mode — is the optimal estimator.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: posterior is nearly Gaussian and symmetric; mode = mean
  - critic r1 (quality 0.85): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-014` r2 [revise, heuristic] `physics-MMSE[log,ms=1,it=30]` status literature-supported. *H:* The metric is squared-error based and the prior is known (uniform box), so the posterior mean — not the likelihood mode — is the optimal estimator.
  - prediction: `{"relative_to": "best_physics", "improvement": 0.0, "expect": "better"}`; if it fails: posterior is nearly Gaussian and symmetric; mode = mean
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 0.95): a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?


**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MLE[log,ms=1,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.06 s
- P-002 `physics-MAP[log,ms=1,it=30]`: ok rel_l2=0.019064 on 60 val samples in 2.06 s
- P-008 `physics-MMSE[log,ms=1,it=30]`: ok rel_l2=0.018596 on 60 val samples in 2.66 s

## 5. Literature retrieval (TheoremSearch)

15 queries, 0 failures, 55 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | identifiability of piecewise-constant (discontinuous) conductivity in the parabolic (heat… | 4 refs via /search (cache) |
| researcher-ml | 1 | neural network approximation of the solution operator of a parameter identification inver… | 4 refs via /search (cache) |
| researcher-experimental | 1 | Cramer-Rao lower bound for parameter estimation in partial differential equations | 4 refs via /search (cache) |
| researcher-exploratory | 1 | posterior mean estimator for Bayesian inverse problems with a uniform prior | 4 refs via /search (cache) |
| researcher-mathematical | 1 | Lipschitz stability for an inverse coefficient problem of a parabolic equation with point… | 4 refs via /search (cache) |
| researcher-experimental | 1 | optimal sensor placement for parameter identification in diffusion equations | 4 refs via /search (cache) |
| researcher-ml | 1 | learning rates of kernel ridge regression for smooth regression functions | 4 refs via /search (cache) |
| researcher-exploratory | 1 | learned initialization for Gauss-Newton iterations in nonlinear inverse problems | 4 refs via /search (cache) |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 2 | torch_mlp on log features with log-conductivity targets | 4 refs via /search (cache) |
| researcher-experimental | 2 | Matched-noise training augmentation | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Learned initializer + Gauss–Newton/LM polishing (hybrid) | 4 refs via /search (cache) |
| researcher-ml | 2 | Kernel ridge on sqrt features with PCA denoising | 4 refs via /search (cache) |
| researcher-experimental | 2 | Random-search control arm | 4 refs via /search (cache) |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_offline_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 126 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | physics | `physics-MLE[log,ms=1,it=30]` | researcher-mathematical | 1 | best-rated proposal in family 'physics' (quality 1.00) |
| E-02 | proposal (P-013) | hybrid | `hybrid[ridge/log+logy->LM10]` | researcher-exploratory | 0.95 | best-rated proposal in family 'hybrid' (quality 0.95) |
| E-03 | proposal (P-009) | data-driven | `torch_mlp/log+logy` | researcher-ml | 0.756 | best-rated proposal in family 'data-driven' (quality 0.76) |
| E-04 | proposal (P-002) | physics | `physics-MAP[log,ms=1,it=30]` | researcher-mathematical | 1 | quality 1.00 with diversity penalty |
| E-05 | proposal (P-014) | physics | `physics-MMSE[log,ms=1,it=30]` | researcher-exploratory | 0.95 | quality 0.95 with diversity penalty |
| E-06 | proposal (P-010) | data-driven | `kernel_ridge/sqrt+pca32` | researcher-ml | 0.756 | quality 0.76 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-011 *Matched-noise training augmentation* — portfolio size limit (6); quality 0.76
- not advanced: P-012 *Random-search control arm* — portfolio size limit (6); quality 0.76

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-006 | E-05 | research | physics | `physics-MMSE[log,ms=1,it=30]` | 0.02042 | **0.018997** | 0.9967 | 0.02505 | 0.0107 | 8.84 | 151083 | 0.952 | FINAL |
| idea-005 | E-04 | research | physics | `physics-MAP[log,ms=1,it=30]` | 0.02138 | **0.019313** | 0.9966 | 0.02547 | 0.0105 | 4.47 | 61292 | 0.957 | evaluated |
| idea-002 | E-01 | research | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0105 | 4.64 | 61517 | 0.957 | evaluated |
| idea-011 | E-02 | refine | hybrid | `hybrid[ridge/log+logy->LM10]` | - | **0.01934** | 0.9966 | 0.0255 | 0.0225 | 3.31 | 48377 | 0.957 | evaluated |
| idea-012 | E-02 | refine | hybrid | `hybrid[ridge/log+logy->LM10]` | - | **0.01934** | 0.9966 | 0.0255 | 0.0215 | 3.28 | 48377 | 0.957 | evaluated |
| idea-003 | E-02 | research | hybrid | `hybrid[ridge/log+logy->LM10]` | 0.02139 | **0.01934** | 0.9966 | 0.0255 | 0.0193 | 3.33 | 48377 | 0.957 | evaluated |
| idea-013 | E-03 | refine | data-driven | `torch_mlp/sqrt+logy` | - | **0.028831** | 0.9923 | 0.03802 | 9.91 | 0.00115 | 0 | - | evaluated |
| idea-014 | E-03 | refine | data-driven | `torch_mlp/log+logy` | - | **0.028868** | 0.9923 | 0.03807 | 20.3 | 0.0169 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/log+logy` | 0.05089 | **0.029154** | 0.9922 | 0.03845 | 9.62 | 0.0133 | 0 | - | evaluated |
| idea-007 | E-06 | research | data-driven | `kernel_ridge/sqrt+pca32` | 0.05237 | **0.029459** | 0.992 | 0.03885 | 0.441 | 0.0517 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **0.030973** | 0.9912 | 0.04085 | 8.13 | 0.00618 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 0.993 | 0.133 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.488 | 0.051 | 0 | - | evaluated |

## 8. Lineage of the final method

- `idea-006` `physics-MMSE[log,ms=1,it=30]` ← P-014 v2 (researcher-exploratory, r2, revise) ← P-008 v1 (researcher-exploratory, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00184` [checkpoint] by scheduler: final method (single)
- `resu-00181` [result] by executor: exp-0027 rel_l2=0.0184 r2=0.997 rmse=0.02421
- `hypo-00077` [hypothesis] by scheduler: idea-006 physics-MMSE[log,ms=1,it=30] (research)
- `plan-00180` [plan] by engineer-v2: exp-0027: final fit of 1 member(s) on train+val; test once
- `gate-00076` [gate] by gate: E-05: approved (physics-MMSE[log,ms=1,it=30])
- `prop-00062` [proposal] by researcher-exploratory: P-014 v2 [revise] exploratory: Posterior-mean (MMSE) estimator under the known prior
- `spec-00001` [spec] by scheduler: research spec heat_v2_offline_s2 (b37c20e37f)
- `prop-00034` [proposal] by researcher-exploratory: P-008 v1 [propose] exploratory: Posterior-mean (MMSE) estimator under the known prior
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 154.0 / 400 worker-seconds over 27 jobs; wall 66 s
- By stage: baseline 2.1s, pilot 6.8s, screen 29.8s, evaluate 54.5s, refine 43.9s, final 16.9s
- LLM: 0 calls (by phase {}), 0 s, $0.000 reported · TheoremSearch: 15 queries (0 failed)

## 10. Agents

- **reporter** (reporter) — none
- **simulator** (simulation) — none
- **critic** (critic) — none
- **engineer** (engineer) — none
- **engineer-v2** (engineer) — none
- **researcher-mathematical** (researcher, mathematical) — none
- **researcher-ml** (researcher, ml) — none
- **researcher-experimental** (researcher, experimental) — none
- **researcher-exploratory** (researcher, exploratory) — none
- **research-critic** (research_critic) — none
- **reporter-v2** (reporter) — none

## 11. Discussion

14 proposal versions over 2 research rounds; 8 methods advanced; 4 predictions supported, 2 contradicted. Final: ['physics-MMSE[log,ms=1,it=30]'] with test rel_l2 0.018402.
