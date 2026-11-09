# Research report (V2): heat_v2_no_critic_s1

Run `heat_v2_no_critic_s1_20261109_110121` · 2026-11-09 11:06:33 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 62.3 (max 1.08e+03); Cramér–Rao relative std per parameter: k1 0.0215, k2 0.0058, k3 0.0052, k4 0.0174; implied rel_l2 floor ≈ **0.01851**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]`
- Validation rel_l2 (selection estimate): **0.019049** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018305**, r2 = **0.997**, rmse = **0.024086** · bootstrap 95% CI for rel_l2: [0.01733, 0.01946]
- Per-target test rel_l2: k1 0.0268, k2 0.0060, k3 0.0052, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.095669 (vs 0.018305 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051109 (vs 0.018305 in distribution)
- Uncertainty calibration on test: 69.3% within 1σ, 95.1% within 2σ (ideal 68% / 95%)
- Inference latency: 20.5 ms/sample; fit 73.6 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` rel_l2=0.019049; **physics** `physics-MMSE[log,ms=3,it=30]` rel_l2=0.019055; **data-driven** `kernel_ridge/log+pca40+logy+aug2` rel_l2=0.032066
- SUPPORTED — E-01 *Per-sample bounded LM in log-conductivity with box-prior posterior mean reaches near the Cramér–Rao floor*: val rel_l2 = 0.019055 vs predicted < 0.035
- SUPPORTED — E-03 *Forward-model-free amortized inversion: kernel ridge in log-temperature / log-k coordinates is already sufficient (chal…*: val rel_l2 = 0.032066 vs predicted < 0.05
- SUPPORTED — E-04 *Learned amortized initializer + few-step Levenberg-Marquardt refinement (hybrid) approaches the Cramér–Rao floor*: val rel_l2 = 0.019343 vs predicted < 0.03
- SUPPORTED — E-05 *Physics MLE via bounded LM in log-space reaches near the Cramér–Rao floor*: val rel_l2 = 0.01911 vs predicted < 0.03
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01851

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:23749594` Manuel Marschall, Gerd Wübbeler, Franko Schmähling et al. (2022). "Generative models and Bayesian inversion using Laplace approximation". Theorem 8. http://arxiv.org/abs/2203.07755v2 — cited by ['P-003', 'P-004']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22602562` Guillaume Bal, Gunther Uhlmann (2011). "Reconstruction of coefficients in scalar second-order elliptic equations from knowledge of their solutions". Theorem 2.1. http://arxiv.org/abs/1111.5051v1 — cited by ['P-003', 'P-007']; researcher claim ['no', 'partial']; harness screen **plausible**
- `ts:thm:19966089` Matteo Giordano, Hanne Kekkonen (2020). "Bernstein-von Mises theorems and uncertainty quantification for linear inverse problems". Theorem 11. http://arxiv.org/abs/1811.04058v4 — cited by ['P-004']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-005']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:24476298` Siyu Cen, Bangti Jin, Yikan Liu et al. (2023). "Recovery of Multiple Parameters in Subdiffusion from One Lateral Boundary Measurement". Theorem 3.2. http://arxiv.org/abs/2307.03263v1 — cited by ['P-005']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:19667864` Andrea Davini, Elena Kosygina (2020). "Stochastic homogenization of a class of nonconvex viscous HJ equations in one space dimension". Proposition A.3. http://arxiv.org/abs/2002.02263v3 — cited by ['P-006']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-007']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:21876495` Eva Sincich (2014). "Smoothness dependent stability in corrosion detection". Theorem 2.1. http://arxiv.org/abs/1406.5907v1 — cited by ['P-008']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-02 *An amortised learned inverse plus a few LM steps matches per-sample physics accuracy and removes local-minimum failures*: inconclusive — method did not produce a full-fidelity validation result
- P-006 *Noise-augmented learned initializer + short LM refinement (hybrid) beats pure ML and matches physics at lower cost* (conjectured): proposed but not executed
- P-007 *MMSE-not-MLE: hybrid amortized init + log-space LM with posterior-mean readout to approach the Cramér–Rao floor under a…* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-06 *Posterior-mean (MMSE) refinement beats the MLE mode under the box prior for the weakly identified k1 and k4*: val rel_l2 0.01905 vs best_hybrid 0.019343: gain +1.51%; expected gain >= +5.0% vs best_hybrid
- PRUNED — idea-003 `hybrid[torch_mlp/log+logy+aug2->LM10]`: projected full-fidelity cost 121s exceeds the 120s per-experiment limit
- PRUNED — idea-009 `svm/log+pca8`: projected full-fidelity cost 212s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Per-sample bounded LM in log-conductivity with box-prior po…): challenge: the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until tested
- E-01 (Per-sample bounded LM in log-conductivity with box-prior po…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-02 (An amortised learned inverse plus a few LM steps matches pe…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-02 (An amortised learned inverse plus a few LM steps matches pe…): revision_direction: the portfolio is dominated by 'hybrid' methods: branch into an under-represented family
- E-03 (Forward-model-free amortized inversion: kernel ridge in log…): question: the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-03 (Forward-model-free amortized inversion: kernel ridge in log…): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-04 (Learned amortized initializer + few-step Levenberg-Marquard…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-05 (Physics MLE via bounded LM in log-space reaches near the Cr…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-06 (Posterior-mean (MMSE) refinement beats the MLE mode under t…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- Test-set sampling uncertainty: bootstrap 95% CI [0.01733, 0.01946] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **1 round(s)**: iteration cap reached (max_rounds=1); unresolved issues are documented.

| round | researcher | action | proposal | new version | origin | response to critic |
|---|---|---|---|---|---|---|
| 1 | researcher-mathematical | propose | - | P-001 | llm |  |
| 1 | researcher-mathematical | propose | - | P-002 | llm |  |
| 1 | researcher-ml | propose | - | P-003 | llm |  |
| 1 | researcher-ml | propose | - | P-004 | llm |  |
| 1 | researcher-experimental | propose | - | P-005 | llm |  |
| 1 | researcher-experimental | propose | - | P-006 | llm |  |
| 1 | researcher-exploratory | propose | - | P-007 | llm |  |
| 1 | researcher-exploratory | propose | - | P-008 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.994 | fdbk-00043 |

### Idea evolution

**P-001 — Per-sample bounded LM in log-conductivity with box-prior posterior mean reaches near the Cramér–Rao floor** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MMSE[log,ms=3,it=30]` status literature-supported. *H:* Inverting the forward map F(k) one sample at a time gets close to the information limit. The method is bounded Levenberg–Marquardt in log k with multistart, followed by importance-sampled posterior mean under the uniform box prior. We hypothesise val rel_l2 <…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If rel_l2 is well above the CRB but the residuals are at noise level (chi² ≈ number of observations), the problem is practical non-identifiability or nonlinear…
  - critic r1 (quality 1.0): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-002 — An amortised learned inverse plus a few LM steps matches per-sample physics accuracy and removes local-minimum failures** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[torch_mlp/log+logy+aug2->LM10]` status conjectured. *H:* A torch MLP on log-temperature features learns a smooth approximation of the inverse map F^{-1}. It will not reach the CRB on its own; we expect val rel_l2 around 0.04–0.08, limited by finite training data and the weak k1/k4 signal. Starting from it, 10 LM it…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.4, "expect": "better"}`; if it fails: If the initializer alone already reaches the CRB-level error, refinement is unnecessary and the amortised model is the efficient choice. If refinement does not…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-003 — Learned amortized initializer + few-step Levenberg-Marquardt refinement (hybrid) approaches the Cramér–Rao floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` status conjectured. *H:* An MLP inverse map trained on simulated data lands close enough to the truth that LM through the exact forward model converges to the global MLE in a few iterations. If so, test rel_l2 should be within about 1.5x of the CRB floor (0.0185), i.e. below 0.03, an…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If LM does not improve on the initializer, either the forward model or the noise model is mismatched (model error) or the iterations are too few. Inspecting pe…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-004 — Posterior-mean (MMSE) refinement beats the MLE mode under the box prior for the weakly identified k1 and k4** (researcher-ml, family hybrid)
- v1 `P-004` r1 [propose, llm] `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` status conjectured. *H:* rel_l2 is a squared-error metric, so the Bayes-optimal estimator is the posterior mean, not the mode. Near the bounds 0.5 and 2.0, the box prior truncates the likelihood, and the MLE or mode is biased or has excess variance for k1 and k4. A hybrid that report…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.05, "expect": "better"}`; if it fails: If the two estimators tie, the posterior is effectively Gaussian and untruncated, and the remaining error is pure noise at the CRB, so further effort should no…
  - critic r1 (quality 1.0): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-005 — Physics MLE via bounded LM in log-space reaches near the Cramér–Rao floor** (researcher-experimental, family physics)
- v1 `P-005` r1 [propose, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* Because the forward model is exact (same explicit FD simulator, 40 grid, 200 steps) and noise is i.i.d. Gaussian sigma=0.001 with Fisher condition ~62, per-sample Levenberg–Marquardt with a few multistarts will converge to the global MLE for nearly all sample…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If chi^2 tail: multimodality/poor starts -> motivates hybrid learned init. If chi^2 fine but error high: CRB diagnostics underestimate error (nonlinearity at b…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-006 — Noise-augmented learned initializer + short LM refinement (hybrid) beats pure ML and matches physics at lower cost** (researcher-experimental, family hybrid)
- v1 `P-006` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM8]` status conjectured. *H:* A data-driven inverse map (torch MLP on standardized raw sensor traces trained with noise augmentation at sigma=0.001) alone will plateau around rel_l2 0.03–0.06 because finite training data (3000) limits accuracy near the bounds; refining its output with 8 L…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.3, "expect": "better"}`; if it fails: Small gain => MLP already near CRB (data-driven sufficient, prefer cheaper arm). Hybrid worse than physics => initializer biased near bounds into wrong basin; …
  - critic r1 (quality 0.95): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-007 — MMSE-not-MLE: hybrid amortized init + log-space LM with posterior-mean readout to approach the Cramér–Rao floor under a…** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM20]` status conjectured. *H:* The usual framing treats this as a point-estimation (MLE/least-squares) problem, but the metric is a squared-error-type loss and the prior is a known bounded box. k1 and k4 are weakly determined (CRB rel std 0.022 and 0.017). For these, the box-prior MMSE (po…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If mode ≈ posterior_mean, the posterior is effectively Gaussian and interior, so estimator choice is irrelevant and the remaining error is optimizer or basin f…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-008 — Forward-model-free amortized inversion: kernel ridge in log-temperature / log-k coordinates is already sufficient (chal…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `kernel_ridge/log+pca40+logy+aug2` status conjectured. *H:* The opposite framing is that physics-based refinement is unnecessary. The inverse map from sensor time series to log k is smooth and only 4-dimensional, and noise is tiny (sigma = 0.001). A smooth kernel interpolant trained on 3000 simulated samples should th…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: It quantifies the amortization gap, i.e. how far a purely data-driven smooth interpolant sits above the CRB floor. That justifies spending the forward-model bu…
  - critic r1 (quality 1.0): the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

- Critic round 1 portfolio view:  5/8 active proposals are 'hybrid' families={'physics': 2, 'hybrid': 5, 'data-driven': 1}

## 5. Literature retrieval (TheoremSearch)

8 queries, 0 failures, 21 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search (cache) |
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant coefficient in parabolic equ… | 4 refs via /search |
| researcher-ml | 1 | asymptotic efficiency of posterior mean or neural amortized estimator approaching Cramer-… | 4 refs via /search |
| researcher-mathematical | 1 | Lipschitz stability for the inverse problem of identifying a piecewise constant conductiv… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness and conditional stability of recovering discontinuous diffusion coefficient in… | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness of piecewise constant diffusion coefficient in one-dimensional heat equation f… | 4 refs via /search |
| researcher-experimental | 1 | uniqueness of conductivity coefficient heat equation from boundary or point measurements … | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_critic_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 64 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | physics | `physics-MMSE[log,ms=3,it=30]` | researcher-mathematical | 1 | best-rated proposal in family 'physics' (quality 1.00) |
| E-02 | proposal (P-002) | hybrid | `hybrid[torch_mlp/log+logy+aug2->LM10]` | researcher-mathematical | 1 | best-rated proposal in family 'hybrid' (quality 1.00) |
| E-03 | proposal (P-008) | data-driven | `kernel_ridge/log+pca40+logy+aug2` | researcher-exploratory | 1 | best-rated proposal in family 'data-driven' (quality 1.00) |
| E-04 | proposal (P-003) | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-05 | proposal (P-005) | physics | `physics-MMSE[log,ms=3,it=30]` | researcher-experimental | 1 | quality 1.00 with diversity penalty |
| E-06 | proposal (P-004) | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-007 *MMSE-not-MLE: hybrid amortized init + log-space LM with posterior-mean readout …* — portfolio size limit (6); quality 1.00
- not advanced: P-006 *Noise-augmented learned initializer + short LM refinement (hybrid) beats pure M…* — portfolio size limit (6); quality 0.95

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-012 | E-06 | refine | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | - | **0.019049** | 0.9967 | 0.02512 | 38.3 | 17.5 | 227330 | 0.952 | FINAL |
| idea-007 | E-06 | research | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | 0.02027 | **0.01905** | 0.9967 | 0.02512 | 58.3 | 15 | 227335 | 0.952 | evaluated |
| idea-002 | E-01 | research | physics | `physics-MMSE[log,ms=3,it=30]` | 0.02027 | **0.019055** | 0.9967 | 0.02513 | 0.0107 | 23 | 265215 | 0.952 | evaluated |
| idea-006 | E-05 | research | physics | `physics-MMSE[log,ms=3,it=30]` | 0.02048 | **0.01911** | 0.9966 | 0.0252 | 0.011 | 19 | 220303 | 0.947 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.011 | 5.42 | 61517 | 0.957 | evaluated |
| idea-005 | E-04 | research | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | 0.02139 | **0.019343** | 0.9966 | 0.02551 | 58.4 | 8.05 | 99356 | 0.957 | evaluated |
| idea-004 | E-03 | research | data-driven | `kernel_ridge/log+pca40+logy+aug2` | 0.05162 | **0.032066** | 0.9905 | 0.04229 | 9.56 | 0.204 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 3.67 | 0.179 | 0 | - | evaluated |
| idea-003 | E-02 | research | hybrid | `hybrid[torch_mlp/log+logy+aug2->LM10]` | 0.02027 | **-** | - | - | - | - | - | - | projected full-fidelity cost 121s exceeds the 120s per-experiment lim… |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **-** | - | - | - | - | - | - | ok |
| idea-009 | E-08 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 212s exceeds the 120s per-experiment lim… |
| idea-013 | E-06 | refine | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM9]` | - | **-** | - | - | - | - | - | - | - |
| idea-014 | E-04 | refine | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | - | **-** | - | - | - | - | - | - | - |
| idea-015 | E-04 | refine | hybrid | `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` | - | **-** | - | - | - | - | - | - | - |

## 8. Lineage of the final method

- `idea-012` `hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]` ← refined from idea-007 ← P-004 v1 (researcher-ml, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00159` [checkpoint] by scheduler: final method (single)
- `resu-00156` [result] by executor: exp-0020 rel_l2=0.01831 r2=0.997 rmse=0.02409
- `hypo-00130` [hypothesis] by scheduler: idea-012 hybrid[torch_mlp/raw+pca64+logy+aug2->LM10] (refine)
- `plan-00155` [plan] by engineer-v2: exp-0020: final fit of 1 member(s) on train+val; test once
- `gate-00129` [gate] by gate: refine idea-007: approved (hybrid[torch_mlp/raw+pca64+logy+aug2->LM10])
- `prop-00038` [proposal] by researcher-ml: P-004 v1 [propose] ml: Posterior-mean (MMSE) refinement beats the MLE mode under the box prior for the weakly ide
- `hypo-00059` [hypothesis] by scheduler: idea-007 hybrid[torch_mlp/raw+pca64+logy+aug2->LM10] (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_critic_s1 (337d36cb19)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00058` [gate] by gate: E-06: approved (hybrid[torch_mlp/raw+pca64+logy+aug2->LM10])

## 9. Compute and cost

- Execution compute: 432.1 / 400 worker-seconds over 20 jobs; wall 312 s
- By stage: baseline 2.2s, screen 68.0s, evaluate 201.0s, refine 54.0s, final 106.9s
- LLM: 8 calls (by phase {'research': 8}), 186 s, $0.744 reported · TheoremSearch: 8 queries (0 failed)

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

The evidence establishes that the objective is met with margin. The selected hybrid (a torch MLP on raw features with PCA-64, log-target, 2× augmentation, followed by 10 Levenberg–Marquardt refinement steps) reached val rel_l2 = 0.01905 and held-out test rel_l2 = 0.01831 (R² = 0.997, RMSE = 0.024). That is well under the 0.05 target. Test error is slightly below val error, so selection on val did not visibly overfit. Four pre-registered predictions were supported:

- the physics-based MMSE estimator (< 0.035)
- the kernel-ridge baseline (< 0.05)
- two hybrid variants (< 0.03)

There is a clear ordering by family. Purely data-driven models plateaued near 0.032. Physics-based and hybrid methods both landed at about 0.019. This suggests that explicitly using the forward PDE, either as a full estimator or as an LM refinement of a learned initial guess, is what removes the remaining ~40% of error.

One hypothesis failed: that a further method would beat the best hybrid (0.01934) by at least 5%. The observed gain was only 1.51% (0.01905). The physics-MMSE (0.019055) and the final hybrid (0.019049) differ by about 0.03%, which is almost certainly within sampling noise on 750 validation cases. No confidence intervals or seed replicates are reported, so these two cannot be ranked.

A plausible explanation, which is conjecture and not yet tested, is that independent methods are converging on an estimation floor near 0.019. That floor would be set by sensor noise and the weak identifiability of some conductivity segments from four point sensors. If so, further modelling effort would yield diminishing returns. Several things remain uncertain:

- whether this floor is truly Bayes-limited
- how error is distributed across the four segments
- how robust the result is to noise levels or k-ranges outside the simulated prior

Next experiments:

- **Paired bootstrap on val:** compare physics-MMSE and the hybrid per sample, with ≥3 seeds per method, to determine whether any difference among the ~0.019 methods is real.
- **Estimate the error floor directly:** use posterior sampling (MCMC) or a Cramér–Rao / Fisher-information bound on a subset of val cases. This tests whether 0.019 is near-optimal.
- **Per-segment error breakdown:** relate each segment's error to its sensor coverage, to see whether one poorly observed segment dominates rel_l2.
- **Noise-level sweep:** scale the noise by 0.5× and 2×. If the error tracks the noise proportionally, that supports the noise-limited hypothesis; if it stays flat, that points to model or optimizer limits.
- **Out-of-prior stress test:** evaluate on k values or contrasts beyond the training range, to check whether the physics-refined hybrid generalizes better than the data-driven models.

_Automatic summary:_ 8 proposal versions over 1 research rounds; 9 methods advanced; 4 predictions supported, 1 contradicted. Final: ['hybrid[torch_mlp/raw+pca64+logy+aug2->LM10]'] with test rel_l2 0.018305.
