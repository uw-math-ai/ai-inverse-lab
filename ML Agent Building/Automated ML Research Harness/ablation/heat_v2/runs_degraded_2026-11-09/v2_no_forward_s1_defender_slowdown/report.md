# Research report (V2): heat_v2_no_forward_s1

Run `heat_v2_no_forward_s1_20261109_102435` · 2026-11-09 10:39:54 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **single** of `ridge/raw+logy`
- Validation rel_l2 (selection estimate): **0.079154** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.063495**, r2 = **0.96396**, rmse = **0.083547** · bootstrap 95% CI for rel_l2: [0.06074, 0.06681]
- Per-target test rel_l2: k1 0.0938, k2 0.0134, k3 0.0123, k4 0.0852
- Success criterion (rel_l2 <= 0.05): **NOT MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.094259 (vs 0.063495 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.13481 (vs 0.063495 in distribution)
- Inference latency: 0.00319 ms/sample; fit 0.0325 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- none

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-002', 'P-007', 'P-008', 'P-009', 'P-010', 'P-019', 'P-025']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-001', 'P-005', 'P-006', 'P-012', 'P-015', 'P-016', 'P-017', 'P-022', 'P-023', 'P-024']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-003', 'P-012']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:22398569` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 9. http://arxiv.org/abs/1210.7346v1 — cited by ['P-003', 'P-015', 'P-022']; researcher claim ['partial', 'unknown']; harness screen **unknown**
- `ts:thm:22602562` Guillaume Bal, Gunther Uhlmann (2011). "Reconstruction of coefficients in scalar second-order elliptic equations from knowledge of their solutions". Theorem 2.1. http://arxiv.org/abs/1111.5051v1 — cited by ['P-004']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:23256523` Sergio Vessella (2007). "Quantitative estimates of unique continuation for parabolic equations, determination of unknown time-varying boundaries and optimal stability estimates". Proposition 4.1.3. http://arxiv.org/abs/0710.2192v1 — cited by ['P-005']; researcher claim ['unknown']; harness screen **unknown**
- `ts:thm:19142517` Marina Kleptsyna, Andrey Piatnitski, Alexandre Popier (2023). "Higher order homogenization for random non-autonomous parabolic operators". Lemma 3. http://arxiv.org/abs/1612.07478v3 — cited by ['P-006', 'P-016', 'P-023']; researcher claim ['no']; harness screen **no**; conflicts: theorem is stated for dimension >= 2; the problem is one-dimensional
- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-007']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-008']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:17797024` Xingrui Liu, Jieming Ke, Yanlong Zhao (2025). "Asymptotic Efficiency Analysis of the Recursive Least-Squares Algorithm for ARX Systems Without Projection". Lemma A.5. http://arxiv.org/abs/2505.19124v1 — cited by ['P-010']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26629947` Yichao Lu, Dean P. Foster (2014). "Fast Ridge Regression with Randomized Principal Component Analysis and Gradient Descent". Lemma 2. http://arxiv.org/abs/1405.3952v1 — cited by ['P-013', 'P-020']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22039577` Daniel Hsu, Sham M. Kakade, Tong Zhang (2014). "Random design analysis of ridge regression". Proposition 2. http://arxiv.org/abs/1106.2363v2 — cited by ['P-013', 'P-020']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26719585` Yinan Li, Fang Liu (2022). "Adaptive Noisy Data Augmentation for Regularized Estimation and Inference in Generalized Linear Models". Proposition 1 (\textbf {regularization effects of PANDA for GLM}). http://arxiv.org/abs/2204.08574v1 — cited by ['P-018']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-01 *Noise-isotropic raw-unit PCA (CV-chosen cutoff, whiten vs Wiener weighting, m<=64) + median-bandwidth RBF kernel ridge,…*: inconclusive — method did not produce a full-fidelity validation result
- E-02 *Measurement preprocessing test: noise-isotropic (unscaled) PCA front end vs per-feature standardization for the log-k M…*: inconclusive — method did not produce a full-fidelity validation result
- E-03 *Estimator-target alignment control: MSE on raw k (posterior-mean estimator) vs MSE on log k for the same MLP*: inconclusive — method did not produce a full-fidelity validation result
- E-04 *Surviving surrogate spec: learned forward net + per-feature-variance chi² refinement, encoder-calibrated anchor, paired…*: inconclusive — method did not produce a full-fidelity validation result
- E-05 *Linear control: ridge on raw traces with log-k targets measures how nonlinear the inverse map is*: inconclusive — method did not produce a full-fidelity validation result
- E-06 *Gated deep ensemble (4x) of P-003-recipe noise-injected MLPs, judged against its own single-member control*: inconclusive — method did not produce a full-fidelity validation result
- P-023 *Low-dimensional manifold, controlled: kernel ridge on noise-isotropic (unscaled) PCA scores vs P-004's standardized PCA* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 3 by researcher-mathematical — P-009 *Noise-isotropic input geometry: MLP on unscaled-PCA sensor traces (log k target) matches or beats t…*: pilot experiment failed: 
- ABANDONED in round 3 by researcher-mathematical — P-010 *Merged surrogate-MAP inversion: learned forward net + raw-unit chi^2 with self-calibrated variance …*: pilot experiment failed: 
- ABANDONED in round 3 by researcher-ml — P-012 *Canonical noise-augmented MLP on standardized raw traces, log-k target (absorbs P-005)*: Not a negative result. P-012 duplicates P-003's full-fidelity configuration exactly and is merged into P-003 at the critic's request. Recommendations carried into P-003: at least 2 seeds, per-component RMSE and signed b…
- ABANDONED in round 3 by researcher-exploratory — P-018 *Estimand ablation of P-003: direct-k targets (posterior mean under uniform-k prior) vs log-k targets*: It executes exactly the same method as P-017. Its theoretical justification also depends on an unknown metric aggregation. Under the pilot's error level (~5%), the predicted Jensen effect (~0.1%) is below what 750 val s…
- PRUNED — idea-002 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted
- PRUNED — idea-003 `torch_mlp/raw+pca32+logy+aug2`: projected full-fidelity cost 467s exceeds the 120s per-experiment limit
- PRUNED — idea-004 `torch_mlp/raw+aug2`: projected full-fidelity cost 431s exceeds the 120s per-experiment limit
- PRUNED — idea-005 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted
- PRUNED — idea-007 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted
- PRUNED — idea-008 `torch_mlp/raw`: projected full-fidelity cost 238s exceeds the 120s per-experiment limit
- PRUNED — idea-009 `svm/log+pca8`: screen failed: Unclassified failure: compute budget for this stage exhausted
- PRUNED — idea-010 `extra_trees/sqrt`: screen failed: Unclassified failure: compute budget for this stage exhausted

### 3.5 Remaining uncertainty

- E-01 (Noise-isotropic raw-unit PCA (CV-chosen cutoff, whiten vs W…): challenge: The noise floor in sample-PCA eigenvalues is not sigma^2. With d≈800 features and n=3000 rows, the Marchenko–Pastur bulk edge sits near sigma^2(1+sqrt(d/n))^2 ≈ 2.3e-6, and slightly higher once you hold out 2…
- E-01 (Noise-isotropic raw-unit PCA (CV-chosen cutoff, whiten vs W…): challenge: The 'Wiener' weight w=sqrt(ev-s2)/ev applies the square root of the Wiener gain to whitened scores, not the Wiener gain itself. This is a reasonable middle ground between whitening and full Wiener, but it sho…
- E-01 (Noise-isotropic raw-unit PCA (CV-chosen cutoff, whiten vs W…): challenge: The 1.15x-of-MLP criterion depends on P-003's full-fidelity number, which does not exist yet. The pilot ratio of 1.26x was measured at 300 training rows. KRR and MLP have different learning curves, so the rat…
- E-01 (Noise-isotropic raw-unit PCA (CV-chosen cutoff, whiten vs W…): challenge: Inner-holdout selection over about 24 configs plus alpha on a single 2:1 split carries selection noise. That noise is comparable to the 15% margin being tested.
- E-02 (Measurement preprocessing test: noise-isotropic (unscaled) …): challenge: Standardization is an invertible linear map. An MLP whose first layer is linear can in principle undo it, so standardization changes optimization and implicit regularization, not the information available. Th…
- E-02 (Measurement preprocessing test: noise-isotropic (unscaled) …): challenge: The assumption that LayerNorm makes the MLP insensitive to the scale of unscaled PCA scores is unverified for the harness torch_mlp. If the first Linear layer sees PC1 with variance orders of magnitude larger…
- E-02 (Measurement preprocessing test: noise-isotropic (unscaled) …): challenge: The k1/k4 sub-claim cuts against the proposal's own contradicting evidence. Unscaled PCA32 is more likely to truncate the small-amplitude outer-sensor directions than standardization is, so the predicted sign…
- E-02 (Measurement preprocessing test: noise-isotropic (unscaled) …): challenge: The rule estimate is 120 s, much above the self-estimate of 20–40 s, which strains a 400 s budget.
- E-03 (Estimator-target alignment control: MSE on raw k (posterior…): challenge: The posterior mean is optimal only for an aggregate Frobenius rel_l2, i.e. sum of squared errors over the sum of //k//². If rel_l2 is a mean of per-sample relative errors, the optimal estimator is E[k///k//²]…
- E-03 (Estimator-target alignment control: MSE on raw k (posterior…): challenge: With σ=1e-3 and an MLP at about 0.05 relative error, the k2/k3 posterior widths are probably far below 0.05. Any effect would sit in k1/k4 only, and an aggregate-level 'refuted' verdict could mask it.
- E-03 (Estimator-target alignment control: MSE on raw k (posterior…): challenge: The 'seed-to-seed spread' in the falsification criterion is not measured by any run. Without it the test is underspecified.
- E-03 (Estimator-target alignment control: MSE on raw k (posterior…): challenge: Both P-011 and this proposal keep the standard scaler, while P-022 argues against it. If P-022 wins, this contrast must be re-run.
- E-04 (Surviving surrogate spec: learned forward net + per-feature…): challenge: Even at rho=2·sigma, surrogate error is about 3σ² of excess variance per feature. That error is a smooth function of k, so it is strongly correlated across the ~800 features. A diagonal chi² treats it as inde…
- E-04 (Surviving surrogate spec: learned forward net + per-feature…): challenge: rho is an average and the gate keys on aggregate hold-out residual. Surrogate error concentrated near box corners or at extreme k values is where refinement matters most, and the average will not reveal it.
- E-04 (Surviving surrogate spec: learned forward net + per-feature…): challenge: The encoder is trained on 90% of the data, and the paired comparison is internal. That design is clean for the refinement question, but it gives no information about whether the final pipeline beats the full-…
- E-04 (Surviving surrogate spec: learned forward net + per-feature…): challenge: The rule-based 30 s understates two MLP trainings plus batched refinement. 60 s is more realistic.
- E-05 (Linear control: ridge on raw traces with log-k targets meas…): challenge: The prediction field (op '<', 0.15) does not encode the two-sided band 0.06–0.15 stated in the hypothesis. A result of 0.03 would 'pass' the prediction while falsifying the hypothesis.
- E-05 (Linear control: ridge on raw traces with log-k targets meas…): challenge: A single alpha=0.01 confounds linearity with regularization choice. Closed-form ridge makes an alpha path essentially free.
- E-05 (Linear control: ridge on raw traces with log-k targets meas…): counterexample: Linear ridge may win on k2/k3 (near the central pulse, close to linear) and fail only on k1/k4. The aggregate then lands in the band for a reason other than global nonlinearity.
- E-05 (Linear control: ridge on raw traces with log-k targets meas…): question: Will per-component R² be reported in log k or in k?
- E-06 (Gated deep ensemble (4x) of P-003-recipe noise-injected MLP…): challenge: The custom code standardizes X internally (mu, sd). Verify whether noise injection happens in raw units before standardization. If noise is added after standardization at σ=1e-3, the effective noise is wrong …
- E-06 (Gated deep ensemble (4x) of P-003-recipe noise-injected MLP…): challenge: The gate requires P-003 at full fidelity, but no seed-variance estimate exists. A cheaper first test of the variance hypothesis is two seeds of P-003 at full data: if the seed-to-seed val difference is tiny, …
- E-06 (Gated deep ensemble (4x) of P-003-recipe noise-injected MLP…): challenge: The 200-epoch members without early stopping are not the P-003 recipe, so the n=1 control is essential. It is correctly included.
- E-06 (Gated deep ensemble (4x) of P-003-recipe noise-injected MLP…): counterexample: Averaging exp(log k̂) across members raises the estimate toward E[k]. Some of the gain could then be the bias correction P-024 targets rather than variance reduction.
- Unresolved disagreement ['P-022', 'P-011', 'P-024']: P-022 claims standardization hurts, yet P-011 and P-024 both use the standard scaler. If P-022 is right, their contrasts are measured on a handicapped baseline.
- Unresolved disagreement ['P-020', 'P-021']: P-020 attributes the KRR–MLP gap to the metric, while P-021 attributes the MLP's residual to variance. Neither attribution can be tested before the full-data P-003 baseline and its seed spread exist.
- Unresolved disagreement ['P-025', 'P-024']: Both target regression-to-mean on k1/k4, through refinement versus estimator choice. Their gains are not additive and should be read against the same signed-bias diagnostic.
- Test-set sampling uncertainty: bootstrap 95% CI [0.06074, 0.06681] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **3 round(s)**: iteration cap reached (max_rounds=3); unresolved issues are documented.

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | (1) Duplication and budget: accepted. P-003 is now the canonical standardized-input MLP. P-001 is narrowed to… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | (1) Merge with P-008: accepted. This is the merged surrogate-inversion run. It reports the encoder-only and r… |
| 2 | researcher-mathematical | propose | - | P-011 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-012 | llm | Pilot evidence: on 300 train samples and 60 val samples the MLP reached rel_l2 = 0.0539. That is 6.1x better … |
| 2 | researcher-ml | revise | P-004 | P-013 | llm | Pilot evidence: KRR with standardized features, PCA(24) and log targets gave 0.0679 at n=300, about 1.26x wor… |
| 2 | researcher-ml | propose | - | P-014 | llm |  |
| 2 | researcher-experimental | revise | P-005 | P-015 | llm | (1) Duplicate of P-003: accepted. The executed method is now different and tests a different hypothesis: the … |
| 2 | researcher-experimental | revise | P-006 | P-016 | llm | (1) Confounding versus P-004: accepted. The revision changes only the scaler relative to P-004's raw+pca24+lo… |
| 2 | researcher-experimental | propose | - | P-017 | llm |  |
| 2 | researcher-exploratory | revise | P-007 | P-018 | llm | (1) noise_scale: I accept the critic's point. I cannot confirm that the harness treats noise_scale as a multi… |
| 2 | researcher-exploratory | revise | P-008 | P-019 | llm | (1) Duplicate of P-002: accepted. I merged the distinctive pieces into one run: the anchor prior, the interna… |
| 3 | researcher-mathematical | abandon | P-009 | - | heuristic-fallback | Accepted: the pilot contradicts feasibility. |
| 3 | researcher-mathematical | abandon | P-010 | - | heuristic-fallback | Accepted: the pilot contradicts feasibility. |
| 3 | researcher-mathematical | defend | P-011 | - | heuristic-fallback | Defended: assumptions and test are explicit; the experiment decides. |
| 3 | researcher-ml | abandon | P-012 | - | llm | (1) Duplicate of P-003: accepted. The executed spec is byte-identical to P-003's canonical run, so it should … |
| 3 | researcher-ml | revise | P-013 | P-020 | llm | (1) Whitening conflicts with the noise-isotropy rationale: accepted. Full whitening rescales a component sitt… |
| 3 | researcher-ml | revise | P-014 | P-021 | llm | (1) Recipe confound: accepted. Members now follow the P-003 recipe as closely as custom code allows: [512,256… |
| 3 | researcher-experimental | revise | P-015 | P-022 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-experimental | revise | P-016 | P-023 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-experimental | revise | P-017 | P-024 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-exploratory | abandon | P-018 | - | llm | (1) Duplicate of P-017: accepted. Both specs execute the identical method, so P-018 adds nothing executable. … |
| 3 | researcher-exploratory | revise | P-019 | P-025 | llm | (1) Duplicate of P-010: accepted. P-019 is now the single surviving surrogate spec, per the critic's revision… |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.781 | fdbk-00044 |
| 2 | 11 | 0.806 | fdbk-00081 |
| 3 | 7 | 0.794 | fdbk-00106 |

### Idea evolution

**P-001 — Smooth, injective inverse map on the compact prior box: a direct learned inverse (log-target MLP) reaches rel_l2 < 0.05** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The discrete forward map F: k in [0.5,2]^4 -> R^{4xT} (explicit FD, 40 nodes, 200 steps) is C-infinity in k. Its Jacobian has full column rank 4 everywhere on the box, so F is injective with a Lipschitz inverse on F(box). A direct regressor X -> log k with en…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If the error saturates above 0.05 and concentrates in k1 and k4, the outer conductivities are practically non-identifiable at sigma = 1e-3. In that case the po…
  - critic r1 (quality 0.85): This is a near-duplicate of P-003, P-005 and P-007. It uses the same torch_mlp, [512,256,128], log target, noise_aug=2 and noise_scale=1e-3. It differs only in… · 'Jacobian has full column rank everywhere, so F is injective' does not follow. Full rank gives only local injectivity. Global injectivity on the box needs a se… · Treat global injectivity as an empirical question, not a consequence of the local rank condition. · If the outer-sensor traces stay below ~5 sigma for most of the window, k1 and k4 get posterior widths of a large fraction of the [0.5,2] range. Under a uniform… · What is the per-sensor peak /u/ in the training features, and how does it compare with sigma=1e-3?
- v2 `P-009` r2 [revise, llm] `torch_mlp/raw+pca64+logy+aug2` status conjectured. *H:* Under i.i.d. additive noise of sigma = 1e-3 in raw units, PCA on unscaled features separates a low-dimensional signal subspace from an isotropic noise floor. A 64-component projection retains essentially all Fisher information about k and removes most of the …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.045}`; if it fails: If only k1/k4 degrade, outer-sensor signal is low-variance and must be kept by standardization or by per-sensor block PCA. If all components saturate at the sa…
  - response to critic: (1) Duplication and budget: accepted. P-003 is now the canonical standardized-input MLP. P-001 is narrowed to the one variable the critic raised: standardization rescales the i.i.d. raw-unit noise non-uniformly. This revision keeps the sam…
  - critic r2 (quality 0.845): P-009 is very close to P-015: both train an MLP on unscaled PCA of raw traces with log-k targets and aug2. It differs in pca (64 vs 32), weight_decay (1e-5 vs … · The proposal assumes that 64 unscaled components keep 'essentially all Fisher information' and gives no evidence. Fisher information about k1/k4 can sit in dir… · Replace 'leading PCs hold all Fisher info' with this: the leading PCs hold the directions of large signal variance, and how much k1/k4 sensitivity they keep mu… · Suppose the trace at sensor 5 is a near-zero-amplitude signal of order 1e-2 whose shape depends on k1. Its variance across the prior could fall below PC 64 whi… · What is the eigenvalue spectrum of the raw-feature covariance relative to σ²=1e-6? At what index does it reach the noise floor? If that index is below 32, the …

**P-002 — Learned forward surrogate + per-sample weighted least-squares inversion beats the direct inverse regressor** (researcher-mathematical, family custom)
- v1 `P-002` r1 [propose, llm] `custom-solver` status conjectured. *H:* The forward map k -> u(sensors, t) is a smooth map from a low-dimensional input (4-D) to a high-dimensional output. It is easier to learn than the inverse map, which must also average out the noise in 800 input dimensions. Training an MLP surrogate g of the f…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.15, "expect": "better"}`; if it fails: If refinement does not help, the learned forward map is not accurate to within the noise level at this sample size, so surrogate-MLE offers no advantage over a…
  - critic r1 (quality 0.825): This shares its core idea with P-008. The two should be merged into one surrogate-inversion run with an ablation on the encoder warm start, the tanh box parame… · The success criterion 'surrogate error well below sigma' is miscalibrated. Training and val features are noisy, so the surrogate's residual RMSE on held-out da… · Replace 'surrogate error <= 1e-3' with 'held-out residual RMSE within ~10% of sigma, and the k-bias induced by refining on noise-free surrogate targets is << t… · Suppose the surrogate has a systematic 2e-3 error concentrated in late-time outer-sensor features, which carry the k1 information. Then refinement shifts k1 co… · Is the gate decided on a split carved from train, so that val is not used for selection inside the method?
- v2 `P-010` r2 [revise, llm] `custom-solver` status conjectured. *H:* A 4 -> D MLP forward surrogate can reach a held-out raw-unit residual RMSE within about 1.3 sigma. Given that, per-sample MAP refinement, warm-started from a learned encoder, will lower hold-out rel_l2 relative to the encoder alone. The objective is chi^2 wit…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.15, "expect": "better"}`; if it fails: If the residual ratio is much greater than 1.3, a learned forward model at n = 2700 is not noise-limited, and amortised inversion should be preferred. If the r…
  - response to critic: (1) Merge with P-008: accepted. This is the merged surrogate-inversion run. It reports the encoder-only and refined hold-out errors from the same run, so the 15% claim does not depend on seed variance between separate runs. (2) Miscalibrat…
  - critic r2 (quality 0.865): P-010 and P-019 are now two separate 'merged' surrogate-MAP specs with the same core: a learned forward MLP, warm-started refinement from an encoder, and a gat… · The surrogate-accuracy target is demanding. A held-out RMSE of 1.3σ means an absolute error of about 1.3e-3 per feature across roughly 800 outputs, using 2700 … · Model surrogate error as extra noise that is heteroscedastic and correlated. The residual will be structured, concentrated where the map is hardest. A single s… · Suppose ρ≈1.3 overall but the residual is concentrated in outer-sensor late-time features. The MAP would then bias k1/k4 while the global χ² looks fine. · In what units is s defined, and what is the derivation for prior_weight=3?

**P-003 — Noise-augmented MLP on standardized raw sensor traces, predicting log-conductivity** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from sensor traces to k is smooth and has only 4 degrees of freedom, and the noise is small (sigma = 0.001). So an MLP should learn it well from 3000 samples. Train it on standardized raw traces, predict log k, and augment the inputs with Gaussian noi…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val error is near train error, the limit is irreducible ambiguity, meaning k1 and k4 are poorly identified, and the posterior is wide. If train error is far…
  - critic r1 (quality 0.88): This should be the canonical MLP run, absorbing P-001 and P-005 (exact duplicate) and the shared parts of P-007. · 'About 7 points per axis' (3000^(1/4)) is coverage of a smooth function, not density. It is only adequate if the inverse map is smooth at that scale. Check by … · Treat 'temperatures at outer sensors are large compared with sigma' as unverified. It is the decisive assumption and costs nothing to check. · The train-error-close-to-val-error diagnostic does not uniquely indicate irreducible ambiguity. An underfit MLP also shows a small gap. Use the CRB estimate to… · Is the 2x-ratio threshold for 'end-dominated error' measured on absolute RMSE or relative error? The two can differ because k is not normalised per component.
- v2 `P-012` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The inverse map from 4-sensor traces to k is smooth enough that a 3-layer MLP trained on all 3000 samples reaches val rel_l2 < 0.05. The pilot with 300 training samples gave 0.0539, so a 10x larger training set should cut the error by more than 10% if the mod…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A plateau near 0.054 with k1/k4 dominating means an information floor: move effort to posterior-mean shrinkage on k1/k4. Continued improvement with n but still…
  - response to critic: Pilot evidence: on 300 train samples and 60 val samples the MLP reached rel_l2 = 0.0539. That is 6.1x better than the mean predictor and just above the 0.05 target. Full fidelity uses 10x more data, so if the model is sample-limited the ta…
  - critic r2 (quality 0.91): The executed config is identical to the canonical P-003: raw input, standard scaler, pca 0, log target, aug2, noise_scale 1e-3, [512,256,128], wd 1e-4, 250 epo… · The P-001 and P-003 pilots (0.053992 and 0.053947) are nearly identical and share a label, which suggests a fixed seed. No seed-to-seed spread has been measure… · Treat the full run as the shared reference for all paired comparisons. Run it with at least 2 seeds so the paired-bootstrap CIs elsewhere have a seed-variance … · Do P-001 and P-003 share the seed? If they do, is the 4e-5 difference just nondeterminism?

**P-004 — Kernel ridge on PCA-compressed traces with log targets: a low-dimensional smooth-manifold estimator** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `kernel_ridge/raw+pca24+logy+aug1` status conjectured. *H:* The noise-free traces lie on a smooth 4-D manifold. The leading 20-30 principal components should capture nearly all of the variation that depends on k, and the rest should be close to noise (sigma = 0.001). If so, RBF kernel ridge on 24 PCA components predic…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If KRR loses badly to the MLP, the inverse map has a scale that varies across the prior, for example sharper behaviour at low k, so a single bandwidth cannot f…
  - critic r1 (quality 0.795): PCA after a standard scaler destroys the isotropy of the noise. In raw units the noise is i.i.d. with sigma=1e-3, so PCA on raw (unscaled) features is the prin… · An isotropic RBF on PCA scores with very unequal variances is dominated by the top 2-3 PCs, which mainly encode k2/k3. The k1/k4 information in low-variance PC… · Use PCA on unscaled raw features (noise-isotropic), then whiten the scores before the RBF, or use per-dimension (ARD) bandwidths. · If k1 enters only through PCs ranked 10-20 with variance a few times sigma^2, then gamma set by the 'scale' heuristic on the top PCs makes the kernel nearly co… · What fraction of raw-unit variance is explained by the top 4, 8 and 24 PCs? How does the eigenvalue spectrum compare with sigma^2 (the noise floor)?
- v2 `P-013` r2 [revise, llm] `custom-ml` status conjectured. *H:* Inputs are compressed by PCA in raw units (where the noise is isotropic), keeping components above 10*sigma^2, then whitened so weak k1/k4 directions are not swamped. On that space an RBF kernel ridge tuned by inner CV on train should match the MLP: val rel_l…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If it is still clearly worse than the MLP after fixing the metric, the inverse map has a scale that varies across the prior (for example, sharper at low k), an…
  - response to critic: Pilot evidence: KRR with standardized features, PCA(24) and log targets gave 0.0679 at n=300, about 1.26x worse than the MLP pilot. That is just inside my 1.3x falsification margin, so the pilot neither confirms nor falsifies the hypothesi…
  - critic r2 (quality 0.88): Whitening PC scores conflicts with the noise-isotropy rationale. A component with eigenvalue just above 10σ² has signal SNR of about 9:1. Whitening scales it t… · A single isotropic RBF bandwidth on whitened scores treats every PC as equally relevant to k. Many retained PCs may encode nonlinear functions of the same few … · Use a softer reweighting instead of a hard cutoff plus full whitening: scale each score by sqrt(λ_i−σ²)/λ_i (Wiener-like), or tune the cutoff factor over {3,10… · If m≈200, whitened RBF distances are dominated by about 200 nearly-noise dimensions, and KRR could do worse than P-016's unwhitened version. · How many components m does fac=10 select on this data?
- v3 `P-020` r3 [revise, llm] `custom-ml` status conjectured. *H:* When the PCA front end is chosen by inner holdout on train only (cutoff factor, whitening vs Wiener-like spectral weighting, m capped at 64), an RBF kernel ridge on the scores reaches val rel_l2 within 1.15x of P-003's full-fidelity MLP and below 0.05. Failur…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If KRR stays clearly worse than the MLP after the metric is data-selected, the inverse map has prior-dependent length scales and needs adaptive models, so KRR …
  - response to critic: (1) Whitening conflicts with the noise-isotropy rationale: accepted. Full whitening rescales a component sitting just above the cutoff to unit variance, which amplifies its noise share. The revision treats this as an empirical question. Th…
  - critic r3 (quality 0.916): The noise floor in sample-PCA eigenvalues is not sigma^2. With d≈800 features and n=3000 rows, the Marchenko–Pastur bulk edge sits near sigma^2(1+sqrt(d/n))^2 … · The 'Wiener' weight w=sqrt(ev-s2)/ev applies the square root of the Wiener gain to whitened scores, not the Wiener gain itself. This is a reasonable middle gro… · Noise-floor eigenvalue ≈ sigma^2·(1+sqrt(d/n_inner))^2 (MP edge), not sigma^2 · Suppose k1/k4 information lives in components with eigenvalues just above the MP edge. Whitening then amplifies them correctly, while Wiener shrinks them towar… · What is the actual feature dimension d (4 sensors × 200 steps, or subsampled)? It sets the MP edge.

**P-005 — Noise-matched, log-target torch MLP: is the inverse map smooth enough at sigma=0.001 that a learned regressor reaches r…** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* With one sensor centred in each conductivity segment (grid indices 5,15,25,35 of 40), each k_i has a distinct signature in the sensor time series. The signal-to-noise ratio at sigma=0.001 is high, so the data-to-k map is a smooth, well-conditioned function on…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If error is concentrated in k1 and k4, the outer segments are weakly observed in the finite time window. That is a measurement-design limit, and the next steps…
  - critic r1 (quality 0.474): executes exactly the same method as P-003 (researcher-ml); differentiate the hypothesis or merge · Its executed method is identical to P-003 (deterministic rule finding). Its distinctive content is the secondary per-parameter claim and a ridge control, and b… · The outer-sensor traces depend jointly on (k1, k2) and (k4, k3), not mainly on k1 or k4 alone. · If it is retained, what would differentiate it from P-003: a different feature set such as late-time emphasis or a per-sensor log transform?
- v2 `P-015` r2 [revise, llm] `torch_mlp/raw+pca32+logy+aug2` status conjectured. *H:* Per-feature standardization amplifies near-noise-floor features, mainly early-time readings at the outer sensors (indices 5 and 35). This raises the effective noise the MLP has to learn to ignore. If the raw features are left unscaled, the additive noise stay…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If it is worse, the low-variance PCs truncated at 32 carry the k1/k4 information. That would show outer-segment identifiability lives in small-amplitude direct…
  - response to critic: (1) Duplicate of P-003: accepted. The executed method is now different and tests a different hypothesis: the measurement-preprocessing question of standardization versus noise-isotropic PCA. The per-parameter breakdown and the ridge contro…
  - critic r2 (quality 0.886): This is the same family as P-009 (see there). It is the better-controlled of the two, since it is single-factor against P-003 apart from pca=32. Note that 'sca… · It is unverified that standardization amplifies near-floor features. Those features are early-time outer readings with near-zero variance, and standardizing th… · To isolate the scaler effect, add a 'standard scaler + pca 32' arm. Otherwise any difference cannot be assigned to noise isotropy as opposed to PCA itself. · If PCA after standardization at m=32 performs equally well, the gain comes from dimension reduction, not noise geometry. · Does the harness torch_mlp have input normalization? If not, are PC scores rescaled before the first layer?
- v3 `P-022` r3 [revise, heuristic-fallback] `torch_mlp/raw+pca32+logy+aug2` status conjectured. *H:* Per-feature standardization amplifies near-noise-floor features, mainly early-time readings at the outer sensors (indices 5 and 35). This raises the effective noise the MLP has to learn to ignore. If the raw features are left unscaled, the additive noise stay…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If it is worse, the low-variance PCs truncated at 32 carry the k1/k4 information. That would show outer-segment identifiability lives in small-amplitude direct…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.88): Standardization is an invertible linear map. An MLP whose first layer is linear can in principle undo it, so standardization changes optimization and implicit … · The assumption that LayerNorm makes the MLP insensitive to the scale of unscaled PCA scores is unverified for the harness torch_mlp. If the first Linear layer … · Standardizing the PCA scores (not the raw features) is permitted and separates subspace choice from input conditioning · A tie between unscaled-PCA32 and P-003 would be consistent both with 'preprocessing irrelevant' and with 'gain from denoising offset by k1/k4 truncation'. Per-… · Does harness torch_mlp include input normalization or LayerNorm? Check the source.

**P-006 — Competing explanation: low-dimensional smooth manifold. PCA-denoised kernel ridge on sqrt features matches the MLP.** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `kernel_ridge/sqrt+pca20+logy+aug1` status conjectured. *H:* Parabolic smoothing forces the 4-sensor time series onto a low-dimensional manifold, with roughly 4 intrinsic dimensions plus noise. Projecting onto 20 leading PCA components removes most of the i.i.d. noise energy, which is spread across all features, while …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If KRR clearly underperforms the MLP, the inverse map has anisotropic or non-stationary structure that an isotropic RBF on PCA coordinates cannot capture. That…
  - critic r1 (quality 0.865): Relative to P-004 it varies features (sqrt), alpha (1e-3), gamma_mult (0.3) and n_pca (20) all at once. A difference between the two KRR runs therefore cannot … · sqrt of noisy near-zero readings: sign(x)sqrt(/x/) has an infinite derivative at 0. It amplifies noise exactly where the outer-sensor signals are small, so it … · Under the sqrt transform, the noise std near x≈0 becomes about sqrt(sigma) ≈ 0.03 in transformed units, so it is not 'small' there. · For a reading of true value 0 plus noise N(0,1e-6), signed sqrt gives a spread of about ±0.03. That is comparable to sqrt of a genuine 1e-3 signal, so sub-nois… · How does the harness 'sqrt' feature set handle negative readings: clip, abs, or signed?
- v2 `P-016` r2 [revise, llm] `kernel_ridge/raw+pca24+logy+aug1` status conjectured. *H:* The noise-free sensor histories lie near a smooth 4-D manifold. With scaler=none, the leading 24 PCs of the raw features are signal-dominated and the discarded directions are mostly noise. An RBF kernel ridge on these scores, with log target, should reach val…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If KRR underperforms specifically on k1/k4, the isotropic metric underweights the low-variance PCs. The next step would be whitened PCA scores or per-target ba…
  - response to critic: (1) Confounding versus P-004: accepted. The revision changes only the scaler relative to P-004's raw+pca24+logy+aug1 design (standard to none). Alpha and gamma_mult are fixed at stated values, and I ask that P-004 run with the same alpha=1…
  - critic r2 (quality 0.85): The proposal itself expects underfitting: gamma 'scale' on unscaled PC scores is set mostly by PC1's variance, so the RBF is nearly flat along PCs 5–24. A loss… · The paired contrast needs P-004 rerun at alpha=1e-3 and gamma_mult=1.0. That is a second run nobody has budgeted, so the ablation is incomplete as specified. · Hold the bandwidth fixed in a normalized sense (for example a median-distance heuristic per configuration), or tune both arms by the same inner CV, so the scal… · Unscaled PCA can lose to standardized PCA at a fixed gamma heuristic purely because of the bandwidth, even if the noise-isotropy argument is correct. · Who runs the matched P-004 arm, and within which budget?
- v3 `P-023` r3 [revise, heuristic-fallback] `kernel_ridge/raw+pca24+logy+aug1` status conjectured. *H:* The noise-free sensor histories lie near a smooth 4-D manifold. With scaler=none, the leading 24 PCs of the raw features are signal-dominated and the discarded directions are mostly noise. An RBF kernel ridge on these scores, with log target, should reach val…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If KRR underperforms specifically on k1/k4, the isotropic metric underweights the low-variance PCs. The next step would be whitened PCA scores or per-target ba…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.501): This is substantially subsumed by P-020: same model family, same unscaled-PCA front end, same scaler hypothesis. P-020 already selects the metric by CV. · The proposal's own assumption list concedes that gamma 'scale' on unscaled scores is dominated by PC1 variance. A fixed gamma_mult=1.0 therefore confounds the … · Bandwidth is set by the median heuristic in each arm, so that the contrast is truly single-factor · Suppose unscaled PCA loses to P-004 at gamma_mult=1.0. That could be pure oversmoothing along low-variance PCs, which would wrongly refute the noise-floor argu… · Is there anything P-023 tests that P-020's fixed-config arm would not?

**P-007 — Log-conductivity reparametrisation with noise-augmented MLP: the inverse map is close to smooth in log k** (researcher-exploratory, family data-driven)
- v1 `P-007` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* Predicting log k instead of k from standardised raw sensor traces, with noise augmentation at the true sigma, gives val rel_l2 <= 0.05. Diffusive time scales go as 1/k, so sensor responses depend roughly multiplicatively on k and the inverse map should be smo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the log and linear targets perform equally, parametrisation is not the bottleneck and the limit is information content or sample size. If k1/k4 errors domin…
  - critic r1 (quality 0.695): noise_scale=1.0 is inconsistent with sigma=0.001 under raw-unit semantics. Confirm the harness semantics or change it to 1e-3 before execution. · noise_scale=1.0 conflicts with every other proposal, all of which use 1e-3 in raw units. If the harness applies noise_scale in raw feature units, this adds noi… · Apart from noise_scale and the SiLU activation, this is the P-003 configuration. Its distinctive and valuable content is the log-vs-none target ablation. · Use noise_scale = 1e-3 in raw units, unless the harness documents that it is a multiplier of the stated sigma. · With raw-unit semantics, noise of std 1.0 is far larger than the temperature signals, so the network would learn close to the mean predictor (~0.33). That fail…
- v2 `P-018` r2 [revise, llm] `torch_mlp/raw+aug2` status conjectured. *H:* With the P-003 MLP configuration held fixed, training on k (target_transform='none') gives val rel_l2 less than or equal to the log-target version. The gap concentrates on the weakly observed segments k1 and k4, where the Jensen bias exp(E[log k]) <= E[k] is …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If 'log' wins clearly, optimisation conditioning matters more than the estimand for this network size, and log targets should stay the team default. If neither…
  - response to critic: (1) noise_scale: I accept the critic's point. I cannot confirm that the harness treats noise_scale as a multiple of sigma, so I now use 1e-3. Under raw-unit semantics this equals sigma; under standardised-unit semantics it is below sigma, …
  - critic r2 (quality 0.45): executes exactly the same method as P-017 (researcher-experimental); differentiate the hypothesis or merge · It executes exactly the same method as P-017 (deterministic rule finding), so it should be merged. · State the exact rel_l2 formula, global ratio or per-sample mean, before claiming which estimand is Bayes-optimal. · If the metric is a per-sample relative error, training in log space (relative errors) can be closer to the metric's optimum than raw MSE. · Is rel_l2 = //K̂−K//_F///K//_F over the whole val set, or a mean of per-sample ratios?

**P-008 — Learn the well-posed direction: neural forward surrogate trained on simulations, then per-sample amortised-plus-iterati…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `custom-ml` status conjectured. *H:* The real forward model is unavailable, but the training set is a sample of the forward map. Learning that forward map (k to traces), which is smooth and well posed, is easier than learning the ill-posed inverse. Inverting the surrogate per sample by gradient-…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.15, "expect": "better"}`; if it fails: If refinement hurts, the learned surrogate is not accurate enough at the noise scale. Data-consistency inversion then needs either more simulations or a struct…
  - critic r1 (quality 0.865): This is near-duplicate in idea with P-002. Its distinctive elements are the anchor regulariser and the steps=0 vs steps=200 ablation. Merge them. · With scaler 'none', the inverse encoder trains on raw features whose scales vary by orders of magnitude. The proposal says the model standardises internally, b… · Express the misfit in units of sigma^2, so that the anchor weight has an interpretable prior-strength meaning. · If the encoder is already near the CRB, refinement through an imperfect surrogate can only add bias. The steps=0 ablation would then show no gain, which is con… · Is the misfit normalised by sigma^2?
- v2 `P-019` r2 [revise, llm] `custom-ml` status conjectured. *H:* A 4-to-D MLP forward surrogate trained on the simulations reaches held-out residual RMSE rho <= 2 sigma (noise included, so rho ~ 1 is the ideal). When it does, refining the encoder's estimate by minimising //(f(z)-x)/s_eff//^2 + anchor*//z-z0//^2 lowers rel_…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If rho >> 2, a generic MLP cannot learn the forward map to noise accuracy from 3000 samples. The next step would be a structured surrogate, e.g. a PCA basis of…
  - response to critic: (1) Duplicate of P-002: accepted. I merged the distinctive pieces into one run: the anchor prior, the internal steps=0-vs-refined comparison, and a gate based on surrogate quality. P-002's author can treat this as the merged spec. (2) Enco…
  - critic r2 (quality 0.795): It duplicates P-010 in substance (see P-010). The two merged specs should become one. · The criterion has two parts: at least 10% over the best direct regressor, and at least 10% over its own steps=0 encoder. Its encoder is trained on 90% of the d… · Use per-feature (or per-sensor-block) residual variance estimated on the hold-out instead of the scalar ρ. · Is the anchor in standardized log-k units? Is anchor=1.0 comparable to P-010's prior weight of 3?
- v3 `P-025` r3 [revise, llm] `custom-ml` status conjectured. *H:* An MLP forward surrogate (log k → standardised traces) trained on 2700 simulations reaches a hold-out per-feature residual RMS of rho ≤ 2·sigma, noise included. When it does, minimising a per-feature-weighted chi² plus an encoder-calibrated Gaussian anchor lo…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If rho >> 2, a generic MLP cannot learn the forward map to noise accuracy from 2700 pairs. The next step would be a structured surrogate: a PCA basis of the tr…
  - response to critic: (1) Duplicate of P-010: accepted. P-019 is now the single surviving surrogate spec, per the critic's revision direction. From P-010 it adopts the raw-unit chi² with self-calibrated, measured per-feature variance. (2) The 90%-data encoder h…
  - critic r3 (quality 0.851): Even at rho=2·sigma, surrogate error is about 3σ² of excess variance per feature. That error is a smooth function of k, so it is strongly correlated across the… · rho is an average and the gate keys on aggregate hold-out residual. Surrogate error concentrated near box corners or at extreme k values is where refinement ma… · Surrogate residuals should be modelled with a low-rank covariance: project residuals onto the top-r PCs of the hold-out residual matrix and use a diagonal in t… · The surrogate is consistently biased by +0.5σ along the direction where k4 enters. The chi² then confidently shifts k4 off-truth, and the refined error rises o… · What is the effective rank of the hold-out surrogate residual matrix? A rank much smaller than d confirms the correlation concern.

**P-011 — Linear control: ridge on raw traces with log-k targets measures how nonlinear the inverse map is** (researcher-mathematical, family data-driven)
- v1 `P-011` r2 [propose, llm] `ridge/raw+logy` status conjectured. *H:* Over the box [0.5,2]^4 (a 4x range in k), the inverse map is clearly nonlinear, because the diffusion response depends on k through exp(-k lambda t)-type terms. Ridge regression from raw traces to log k will therefore reach val rel_l2 between 0.06 and 0.15: f…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.15}`; if it fails: A result below 0.06 means simple models nearly suffice; effort should then go to variance reduction, not capacity. A result above 0.15 means a strongly nonline…
  - critic r2 (quality 0.8): This is a good, cheap control. The prediction op is '<0.15', but the falsification is two-sided (0.06–0.15), so the result should be recorded against both boun… · 'k1/k4 lowest R²' conflates linearity with identifiability. Low R² for a linear model can come from nonlinearity in k1/k4 rather than from weak information. · Is alpha selected, or fixed at 1e-2? A small inner-CV alpha grid costs almost nothing.

**P-014 — Deep ensemble (4x) of noise-injected MLPs, log-k heads averaged in k-space** (researcher-ml, family data-driven)
- v1 `P-014` r2 [propose, llm] `custom-ml` status conjectured. *H:* If P-003's residual error near 0.05 is partly from estimator variance (initialization and finite data), then an ensemble of 4 independently seeded MLPs will help. Each member sees fresh N(0, sigma^2) input noise every batch, and the ensemble averages exp(pred…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If the ensemble does not beat the single MLP, the residual error is not estimator variance. It is then either an information floor (check k1/k4 dominance) or b…
  - critic r2 (quality 0.74): Confound: the members use a different recipe from P-003 (150 epochs, bs 256, fresh noise each batch, internal standardization). An ensemble-vs-P-003 gap mixes … · The noise is injected in raw units before internal standardization (please confirm). Adding σ on top of features that already contain σ gives about √2σ. That i… · Judge the ensemble against the mean of its own members, not against P-003. · If members are 15% worse than P-003 because of the shorter training, a 10% ensemble gain would still lose to P-003 while showing real variance reduction. · Is the injected noise added before or after the internal standardization?
- v2 `P-021` r3 [revise, llm] `custom-ml` status conjectured. *H:* If the canonical MLP's residual error near 0.05 is partly estimator variance (initialization, minibatch order, finite data), then averaging 4 independently seeded P-003-recipe members in k-space lowers val rel_l2 by at least 8% relative to a single member of …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.05, "expect": "better"}`; if it fails: If the ensemble does not beat its own member control, the residual error is not estimator variance. It is either an information floor (check k1/k4 dominance) o…
  - response to critic: (1) Recipe confound: accepted. Members now follow the P-003 recipe as closely as custom code allows: [512,256,128] with LayerNorm and GELU, AdamW at lr 1e-3 and wd 1e-4, cosine schedule, batch 128. Epochs are 200 rather than 250 to fit the…
  - critic r3 (quality 0.758): The custom code standardizes X internally (mu, sd). Verify whether noise injection happens in raw units before standardization. If noise is added after standar… · The gate requires P-003 at full fidelity, but no seed-variance estimate exists. A cheaper first test of the variance hypothesis is two seeds of P-003 at full d… · Members' prediction disagreement is reported. Its correlation with per-sample error indicates whether the residual is variance. · Averaging exp(log k̂) across members raises the estimate toward E[k]. Some of the gain could then be the bias correction P-024 targets rather than variance red… · Is averaging done on exp(log-pred) in k-space, as stated? If so, it partially enacts P-024's shrinkage, which is a confound worth noting.

**P-017 — Estimator-target alignment control: MSE on raw k (posterior-mean estimator) vs MSE on log k for the same MLP** (researcher-experimental, family data-driven)
- v1 `P-017` r2 [propose, llm] `torch_mlp/raw+aug2` status conjectured. *H:* rel_l2 is a squared-error metric in k-space. An MLP trained with MSE on untransformed k approximates the posterior mean E[k/data], which is the MMSE estimator under the uniform box prior. A log-target MLP approximates exp(E[log k/data]), which is biased low b…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.0, "expect": "better"}`; if it fails: If log clearly wins, either the log reparametrisation improves learnability of the inverse map (P-007's claim), or the metric is not pure squared error in k. E…
  - critic r2 (quality 0.845): It is the surviving copy of the P-017/P-018 duplicate pair, so fold in P-018's signed-bias diagnostic. · The prediction is 'improvement 0.0, expect better', which is not a falsifiable prediction. Commit to a direction and magnitude, or declare it an equivalence te… · Recast it as an equivalence test: /Δ rel_l2/ < 3% relative, with a 95% paired CI. Separately, the k1/k4 signed bias of the log model should be negative and lar… · What is the exact rel_l2 formula?
- v2 `P-024` r3 [revise, heuristic-fallback] `torch_mlp/raw+aug2` status conjectured. *H:* rel_l2 is a squared-error metric in k-space. An MLP trained with MSE on untransformed k approximates the posterior mean E[k/data], which is the MMSE estimator under the uniform box prior. A log-target MLP approximates exp(E[log k/data]), which is biased low b…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.0, "expect": "better"}`; if it fails: If log clearly wins, either the log reparametrisation improves learnability of the inverse map (P-007's claim), or the metric is not pure squared error in k. E…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.869): The posterior mean is optimal only for an aggregate Frobenius rel_l2, i.e. sum of squared errors over the sum of //k//². If rel_l2 is a mean of per-sample rela… · With σ=1e-3 and an MLP at about 0.05 relative error, the k2/k3 posterior widths are probably far below 0.05. Any effect would sit in k1/k4 only, and an aggrega… · The rel_l2 aggregation is verified before interpreting the result · Raw-k MSE weights errors at large k more heavily, and k near 2 dominates //k//. Raw-k could then win aggregate rel_l2 while losing at small k, without posterio… · What signed per-component bias does the existing P-003 log model show on val? It can be computed at no cost from saved predictions.

- Critic round 1 portfolio view: No empirical results exist yet. The decisive unknown is whether the outer sensors (indices 5 and 35) receive enough heat above sigma=1e-3 within the 200-step window. Under a CFL-limited dt the total time may be short, giving diffusion length around 0.26 vs a distance of about 0.37-0.40 from the pulse centre. If the signal is weak, all methods plateau on k1/k4 and posterior-mean shrinkage becomes … 7/8 active proposals are 'data-driven' families={'data-driven': 7, 'custom': 1}
- Critic round 2 portfolio view: Empirically, the only established fact is that a standardized-input log-k MLP reaches about 0.054 with 300 training samples on 60 validation samples. That is statistically indistinguishable from the 0.05 target, and KRR is 1.26× worse at the same n. Much of the round's debate (noise-isotropic PCA, log-k estimand bias, k1/k4 information floor) rests on unmeasured quantities: the noise-floor spectr… 10/11 active proposals are 'data-driven' families={'data-driven': 10, 'custom': 1}
- Critic round 3 portfolio view: Recommended order: (1) run P-003 at full fidelity with 2 seeds, plus the free diagnostics (eigen-spectrum with MP edge, P-003 signed bias, metric definition). (2) Run P-011, about 1 s, with the band prediction fixed. (3) Run P-020 with P-023 merged in as its fixed-config arm. (4) Run P-024 only if the signed-bias diagnostic shows k1/k4 shrinkage. (5) Run P-021 and P-025 Stage 0 conditionally, if … 7/7 active proposals are 'data-driven' families={'data-driven': 7}

**Pilot experiments run during research** (small validation subsets):

- P-001 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.053992 on 60 val samples in 29.8 s
- P-003 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.053947 on 60 val samples in 43.2 s
- P-004 `kernel_ridge/raw+pca24+logy+aug1`: ok rel_l2=0.067893 on 60 val samples in 14.2 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 88 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant conductivity in parabolic eq… | 4 refs via /search (cache) |
| researcher-mathematical | 1 | Uniqueness and Lipschitz stability for identifying a piecewise constant diffusion coeffic… | 4 refs via /search (cache) |
| researcher-experimental | 1 | Lipschitz stability for recovering piecewise constant diffusion coefficient in parabolic … | 4 refs via /search |
| researcher-ml | 1 | Lipschitz stability for identifying piecewise constant coefficients in parabolic equation… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient from boundary or interior spectral data, eigenvalues … | 4 refs via /search |
| researcher-mathematical | 1 | Stability of coefficient inverse problem for heat equation with finitely many unknown par… | 4 refs via /search |
| researcher-ml | 1 | uniqueness and stability of conductivity recovery for one-dimensional heat equation from … | 4 refs via /search |
| researcher-experimental | 1 | uniqueness identifiability of conductivity in one-dimensional heat equation from boundary… | 4 refs via /search |
| researcher-exploratory | 2 | Harness documentation or code for how noise_scale is applied in noise_aug | 4 refs via /search |
| researcher-experimental | 2 | Correlation matrix of local Jacobian columns from kNN local-linear fits | 4 refs via /search |
| researcher-mathematical | 2 | Per-sensor max and time-integrated /x/ over the training set, plus the fraction of entrie… | 4 refs via /search |
| researcher-ml | 2 | Linear ridge baseline on standardized features with per-component val R^2 | 4 refs via /search |
| researcher-exploratory | 2 | sensitivity of interior heat sensors to piecewise conductivity near Dirichlet boundary | 4 refs via /search |
| researcher-mathematical | 2 | Local-linear forward Jacobian estimate: regress X on k within kNN neighbourhoods of train… | 4 refs via /search |
| researcher-experimental | 2 | sensitivity of point temperature measurements to piecewise constant conductivity heat equ… | 4 refs via /search |
| researcher-ml | 2 | identifiability piecewise constant conductivity heat equation point measurements | 4 refs via /search |
| researcher-experimental | 3 | torch_mlp architecture (LayerNorm presence) | 4 refs via /search |
| researcher-ml | 3 | Learning curve at n=300/1000/3000 with the same config | 4 refs via /search |
| researcher-exploratory | 3 | Harness rel_l2 definition | 4 refs via /search |
| researcher-mathematical | 3 | Raw-feature covariance eigenvalues vs σ² and per-sensor max /x/ across train (zero-cost d… | 4 refs via /search |
| researcher-ml | 3 | Practical identifiability of piecewise-constant diffusion coefficients from sparse interi… | 4 refs via /search |
| researcher-exploratory | 3 | posterior mean vs posterior median bias under log reparametrisation, uniform prior | 4 refs via /search |
| researcher-experimental | 3 | Per-sensor fraction of feature entries with /x/ < 3 sigma | 4 refs via /search |
| researcher-mathematical | 3 | Harness source: order of scaler/PCA and units of noise_scale | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 197 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-020) | data-driven | `custom-ml` | researcher-ml | 0.916 | best-rated proposal in family 'data-driven' (quality 0.92) |
| E-02 | proposal (P-022) | data-driven | `torch_mlp/raw+pca32+logy+aug2` | researcher-experimental | 0.88 | quality 0.88 with diversity penalty |
| E-03 | proposal (P-024) | data-driven | `torch_mlp/raw+aug2` | researcher-experimental | 0.869 | quality 0.87 with diversity penalty |
| E-04 | proposal (P-025) | data-driven | `custom-ml` | researcher-exploratory | 0.851 | quality 0.85 with diversity penalty |
| E-05 | proposal (P-011) | data-driven | `ridge/raw+logy` | researcher-mathematical | 0.785 | quality 0.79 with diversity penalty |
| E-06 | proposal (P-021) | data-driven | `custom-ml` | researcher-ml | 0.758 | quality 0.76 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-023 *Low-dimensional manifold, controlled: kernel ridge on noise-isotropic (unscaled…* — portfolio size limit (6); quality 0.50

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-002 | E-01 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-003 | E-02 | research | data-driven | `torch_mlp/raw+pca32+logy+aug2` | 0.03106 | **-** | - | - | - | - | - | - | projected full-fidelity cost 467s exceeds the 120s per-experiment lim… |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw+aug2` | 0.05127 | **-** | - | - | - | - | - | - | projected full-fidelity cost 431s exceeds the 120s per-experiment lim… |
| idea-005 | E-04 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-007 | E-06 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **-** | - | - | - | - | - | - | projected full-fidelity cost 238s exceeds the 120s per-experiment lim… |
| idea-009 | E-08 | baseline | data-driven | `svm/log+pca8` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-006 | E-05 | research | data-driven | `ridge/raw+logy` | 0.07915 | **-** | - | - | - | - | - | - | FINAL |

## 8. Lineage of the final method

- `idea-006` `ridge/raw+logy` ← P-011 v1 (researcher-mathematical, r2, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00199` [checkpoint] by scheduler: final method (single)
- `resu-00196` [result] by executor: exp-0018 rel_l2=0.0635 r2=0.964 rmse=0.08355
- `hypo-00119` [hypothesis] by scheduler: idea-006 ridge/raw+logy (research)
- `plan-00195` [plan] by engineer-v2: exp-0018: final fit of 1 member(s) on train+val; test once
- `gate-00118` [gate] by gate: E-05: approved (ridge/raw+logy)
- `prop-00071` [proposal] by researcher-mathematical: P-011 v1 [propose] mathematical: Linear control: ridge on raw traces with log-k targets measures how nonlinear the inverse 
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s1 (8e97e94bdd)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 548.0 / 400 worker-seconds over 12 jobs; wall 919 s
- By stage: baseline 2.2s, pilot 87.2s, screen 446.1s, final 12.4s
- LLM: 22 calls (by phase {'research': 19, 'execution': 3}), 1473 s, $5.273 reported · TheoremSearch: 24 queries (0 failed)

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

The evidence establishes one result. A ridge regression on raw sensor traces with a log-k target ("ridge/raw+logy") reached val rel_l2 = 0.079 and held-out test rel_l2 = 0.063 (R² = 0.964, RMSE = 0.084). That misses the success criterion of rel_l2 ≤ 0.05. Two features of this result matter:

- **Test beat val by about 20% relative.** With 750 samples per split, that gap is probably split-to-split sampling variation, not a real generalisation gain. The true error is therefore best treated as somewhere in the 0.06–0.08 range.
- **Nothing else was adjudicated.** `best_by_family` is empty and no hypothesis is listed as supported or contradicted. The run produced a single linear baseline, not a comparison among model families, so there is no evidence about whether nonlinear models help.

That a linear map on log k gets within roughly 1.3–1.6× of the target suggests the inverse map is close to log-linear over the prior range. That reading is a conjecture, not a tested claim.

No hypothesis was refuted on scientific grounds. The two substantive pilots failed with empty error messages: the noise-isotropic PCA-input MLP and the surrogate-MAP inversion with a learned forward model and self-calibrated χ². An empty "pilot experiment failed:" points to an execution or infrastructure fault, not a negative finding, so both ideas remain untested. The other two were withdrawn on design grounds:

- **P-012** was an exact duplicate of P-003 and was merged into it.
- **The direct-k vs log-k estimand ablation** was dropped because its predicted Jensen-gap effect (about 0.1%) is below what 750 validation samples can resolve at about 5% error. It is only worth running if per-component errors on k1 or k4 exceed about 15–20%.

The main open question is that the canonical noise-augmented MLP (P-003) and the P-017 diagnostics do not appear in the results. We therefore do not know:

- whether any nonlinear model beats ridge;
- whether the problem is limited by sample size or by noise;
- how error splits across k1–k4. The outer segments are plausibly worst identified, depending on where the sensors sit, but this is unverified.

Next experiments:

- **Rerun the two failed pilots with error capture.** These are the PCA-input MLP and the surrogate-MAP inversion. Logging the traceback is the first step, so that the failures become either working results or genuine negatives.
- **Execute P-003 as specified.** That means the noise-augmented MLP on standardized traces with a log-k target, at least 2 seeds, per-component RMSE and signed bias, and learning curves at n ∈ {1000, 2000, 3000}. Use the ≥25% / <10% bands to classify it as sample-limited or plateaued.
- **Tune the ridge baseline properly.** Sweep the regularisation strength and add cheap nonlinear features, such as quadratic terms on PCA scores or kernel ridge. This would show whether the ~0.06–0.08 error comes from bias or from the noise floor.
- **Estimate the irreducible error floor.** Run a classical least-squares inversion with the true forward solver on a val subset, or compute a Fisher/CRLB estimate per component. Either would tell us whether 0.05 is achievable at this noise level at all.
- **Report a split-variance estimate.** Use bootstrap confidence intervals or k-fold cross-validation on train plus val, so the val–test discrepancy can be quantified rather than assumed to be noise.

_Automatic summary:_ 25 proposal versions over 3 research rounds; 8 methods advanced; 0 predictions supported, 0 contradicted. Final: ['ridge/raw+logy'] with test rel_l2 0.063495.
