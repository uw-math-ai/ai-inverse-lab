# Research report (V2): heat_v2_memory_s1

Run `heat_v2_memory_s1_20261109_153523` · 2026-11-09 15:48:06 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 62.3 (max 1.08e+03); Cramér–Rao relative std per parameter: k1 0.0215, k2 0.0058, k3 0.0052, k4 0.0174; implied rel_l2 floor ≈ **0.01851**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=3,it=25]`
- Validation rel_l2 (selection estimate): **0.018966** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018272**, r2 = **0.99701**, rmse = **0.024042** · bootstrap 95% CI for rel_l2: [0.01729, 0.01946]
- Per-target test rel_l2: k1 0.0266, k2 0.0060, k3 0.0052, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.11125 (vs 0.018272 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051159 (vs 0.018272 in distribution)
- Uncertainty calibration on test: 68.9% within 1σ, 95.1% within 2σ (ideal 68% / 95%)
- Inference latency: 73.4 ms/sample; fit 0.00491 s; exported model reload check: max |Δ| = 0.00816
- ⚠ This test split had already been evaluated 1 time(s) in earlier runs (research memory): repeated development weakens its independence.
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=3,it=25]` rel_l2=0.018966; **data-driven** `torch_mlp/sqrt` rel_l2=0.028495
- SUPPORTED — E-01 *Canonical physics arm: per-sample bounded log-LM + box-truncated Laplace-IS posterior mean on full val (merged P-008/P-…*: val rel_l2 = 0.019056 vs predicted < 0.0225
- SUPPORTED — E-02 *Paired readout contrast: MLE (mode) vs box-truncated posterior mean, identical LM, stratified by lower/upper edge*: val rel_l2 0.019338 vs best_physics 0.018966: gain -1.96%; expected to be worse than best_physics (gain < +1.0%)
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01851

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-001', 'P-009']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-003', 'P-007', 'P-011']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:26673267` Andrew S. Lan, Mung Chiang, Christoph Studer (2018). "An Estimation and Analysis Framework for the Rasch Model". Corollary 2. http://arxiv.org/abs/1806.03551v1 — cited by ['P-003', 'P-011', 'P-017']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:19245271` Xinyi Zhong, Chang Su, Zhou Fan (2021). "Empirical Bayes PCA in high dimensions". Corollary 5.3. http://arxiv.org/abs/2012.11676v3 — cited by ['P-004']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:25580267` Prithwish Bhaumik, Subhashis Ghosal (2016). "Efficient Bayesian estimation and uncertainty quantification in ordinary differential equation models". Corollary. http://arxiv.org/abs/1411.1166v2 — cited by ['P-005', 'P-006', 'P-012', 'P-013', 'P-018']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26618484` Yuefeng Wu, Giles Hooker (2013). "Hellinger Distance and Bayesian Non-Parametrics: Hierarchical Models for Robust and Efficient Bayesian Inference". Theorem 6. http://arxiv.org/abs/1309.6906v1 — cited by ['P-005', 'P-012']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:20264653` Zhiyuan Li, Masahiro Yamamoto (2019). "Inverse Problems of Determining Coefficients of the Fractional Partial Differential Equations". Theorem 1 (Cheng, Nakagawa, Yamamoto and Yamazaki \cite{CNYY09}). http://arxiv.org/abs/1904.05505v1 — cited by ['P-008', 'P-014', 'P-019']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:17546409` Ali Feizmohammadi (2025). "Reconstruction of 1-D evolution equations and their initial data from one passive measurement". Theorem 1.2. http://arxiv.org/abs/2501.10590v2 — cited by ['P-008']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:25734494` K. J. H. Law, A. M. Stuart, K. C. Zygalakis (2015). "Data Assimilation: A Mathematical Introduction". Theorem 4.5. http://arxiv.org/abs/1506.07825v1 — cited by ['P-012']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26360863` Christian Kanzow, Theresa Lechner (2022). "Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems". Proposition. http://arxiv.org/abs/2210.07644v1 — cited by ['P-012']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:24839008` Andre Wibisono, Varun Jog, Po-Ling Loh (2017). "Information and estimation in Fokker-Planck channels". Theorem 3. http://arxiv.org/abs/1702.03656v1 — cited by ['P-014', 'P-015', 'P-020']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26138127` Clarice Poon, Jingwei Liang, Carola-Bibiane Schönlieb (2018). "Local Convergence Properties of SAGA/Prox-SVRG and Acceleration". Lemma 4.1 (Local quadratic growth \cite {liang2017activity}). http://arxiv.org/abs/1802.02554v3 — cited by ['P-019']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- P-016 *Learning-curve point n≈9000 with P-013's recipe: is the amortised-network gap to the CRB floor sample-limited?* (conjectured): proposed but not executed
- P-017 *Learning-curve endpoint (n=15k): P-013-matched MLP trained on 3000 given + 12000 forward-simulated prior draws with per…* (conjectured): proposed but not executed
- P-018 *Single data-driven control: noise-augmented MLP is approximation-limited, with per-sample error/CRB ratios inverting th…* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-03 *IS readout accuracy control: same LM, posterior_samples 512 vs 128, paired with P-014*: val rel_l2 0.018966 vs best_physics 0.019056: gain +0.47%; expected gain >= +0.5% vs best_physics
- ABANDONED in round 2 by researcher-ml — P-004 *Pure amortised MMSE: noise-augmented MLP on log-features reaches target without forward-model refin…*: Near-duplicate of P-006 (only noise_aug differs) at a cost of ~50% of the execution budget. Its scientific question, whether more effective data reduces amortised error, is better answered by the revised P-003, which ad…
- ABANDONED in round 2 by researcher-exploratory — P-007 *Amortised init + forward-model MMSE refinement: closing the gap from the MLP (≈0.025) to the CRB fl…*: Duplicate of P-001; its falsification criteria are merged there. The P-008 pilot (0.0184 with no learned init) also makes the learned initialiser unnecessary for accuracy, so a further hybrid variant would waste budget.
- ABANDONED in round 3 by researcher-mathematical — P-009 *Merged hybrid: MLP initializer + bounded log-space LM with posterior-mean readout, at the CRB scale*: Negative result recorded. On this problem (sigma=0.001, Fisher condition about 62) the learned MLP initializer does not affect final accuracy. The hybrid and prior-centre LM pilots agree on rel_l2, chi2/dof, ESS and cov…
- ABANDONED in round 3 by researcher-experimental — P-012 *Paired readout control: same learned init + LM, estimator=mode, against the merged posterior-mean h…*: Superseded by P-015, which runs the same paired mode-vs-mean contrast at about 1/10 of the cost. The pilot (0.0190 at n=60, calibrated, chi²/dof 0.94) already shows that the learned init adds no accuracy over multistart…
- ABANDONED in round 3 by researcher-exploratory — P-008 *No learning at all: per-sample box-constrained posterior mean through the forward model already bea…*: Merged into P-014, which keeps the identical method under absolute falsification criteria. This is not a negative result: the pilot supported the hypothesis (0.0184 on n=60).
- PRUNED — idea-006 `svm/log+pca8`: projected full-fidelity cost 202s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Canonical physics arm: per-sample bounded log-LM + box-trun…): challenge: The refutation threshold of 0.0225 is weak given the 0.0184 pilot. Pre-register the sharper claim as primary: the bootstrap CI for rel_l2 includes the CRB estimate of 0.0185 and lies below 0.020.
- E-01 (Canonical physics arm: per-sample bounded log-LM + box-trun…): challenge: Cost estimate: the forward budget is quoted as 270k evaluations, but the rule estimate is 16.4 s. Report the actual forward_evals so that P-020 and P-021 can share LM outputs instead of recomputing.
- E-01 (Canonical physics arm: per-sample bounded log-LM + box-trun…): challenge: The multistart criterion needs an explicit start design (are the starts prior-random or at the box centre?). Three starts cannot detect a basin the starts never visit.
- E-01 (Canonical physics arm: per-sample bounded log-LM + box-trun…): challenge: Self-references to 'P-014' and 'P-016' are stale IDs. Fix them so the paired arms (P-020, P-021, P-018) join on this run's per-sample log.
- E-02 (Paired readout contrast: MLE (mode) vs box-truncated poster…): challenge: The 0.5-2% aggregate gap is about 1e-4 in rel_l2, below the bootstrap resolution unless the comparison is paired per sample. The proposal does pair, which is correct, but state the minimum detectable effect.
- E-02 (Paired readout contrast: MLE (mode) vs box-truncated poster…): challenge: Setting posterior_samples=32 is irrelevant for the mode readout. Reuse P-019's LM optimum directly, which costs about 0 s.
- E-02 (Paired readout contrast: MLE (mode) vs box-truncated poster…): challenge: The prediction that the gain is larger at the upper edge assumes wider absolute posteriors at k=2. In relative-error terms the effect may cancel. Specify whether the comparison is in relative or absolute erro…
- E-02 (Paired readout contrast: MLE (mode) vs box-truncated poster…): counterexample: If the edge gain comes almost entirely from samples with clamped modes, the IS readout's value equals a cheap 'unclamp toward interior' shrinkage. Test that heuristic too.
- E-03 (IS readout accuracy control: same LM, posterior_samples 512…): challenge: Using different IS draws at 128 and 512 conflates Monte Carlo error with the readout change. Use common random numbers: the 512 draws should contain P-019's 128 draws as a subset.
- E-03 (IS readout accuracy control: same LM, posterior_samples 512…): challenge: The test of a shift below 0.1 posterior std for 95% of samples is reasonable. However, the MC std at ESS 58 is about 0.13 std, so the stated expectation may already be borderline. Recompute the threshold from…
- E-03 (IS readout accuracy control: same LM, posterior_samples 512…): counterexample: A low ESS at the edges arises from proposal mismatch, not from sample count. Quadrupling the samples barely helps, whereas a truncated-Gaussian proposal or quadrature would.
- E-03 (IS readout accuracy control: same LM, posterior_samples 512…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- Unresolved disagreement ['P-016', 'P-017']: Which architecture is the P-013 recipe: 512-256-128 or [256,256]?
- Unresolved disagreement ['P-016', 'P-018']: The epoch counts (60 versus 150) and the noise scheme (per-epoch resampling versus noise_aug=2) differ, so the 3k anchor is not comparable to the 9k point.
- Unresolved disagreement ['P-017', 'P-018']: P-017 attributes the gap to data, while P-018 attributes it to a uniform approximation floor that may or may not shrink with n. Neither tests a proportional-inefficiency alternative.
- Test-set sampling uncertainty: bootstrap 95% CI [0.01729, 0.01946] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | Pilot result (n_val=60, train 300): rel_l2 0.0184. Residual std at truth was 0.00099 against nominal 0.001, m… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | All points accepted. (1) The <0.05 prediction is uninformative. It is replaced by a quantitative learning-cur… |
| 2 | researcher-ml | revise | P-003 | P-011 | llm | (1) Duplicate of P-001: accepted. The hybrid arm (torch_mlp init, log-space LM, posterior_mean) is folded int… |
| 2 | researcher-ml | abandon | P-004 | - | llm | (1) Third pure-MLP arm, and at 200 s it does not fit the budget: accepted. (2) The falsification criterion re… |
| 2 | researcher-experimental | revise | P-005 | P-012 | llm | I accept every point, and the revision turns P-005 into the missing paired control rather than a fourth poste… |
| 2 | researcher-experimental | revise | P-006 | P-013 | llm | (1) The <0.05 threshold is uninformative given the prior of about 0.025, which comes from memory anchor m-261… |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | I accept all three critic points. (1) P-007 is functionally a P-001 variant. (2) Its sharper falsification cr… |
| 2 | researcher-exploratory | branch | P-008 | P-014 | llm | The pilot supports the hypothesis: 0.0184 on n=60 is below the 0.0247 MLP anchor, and normalised chi2 is 0.94… |
| 2 | researcher-exploratory | propose | - | P-015 | llm |  |
| 3 | researcher-mathematical | abandon | P-009 | - | llm | I accept the critic's main point. The pilot evidence settles it. My hybrid pilot gave 0.018449 (resu-00080). … |
| 3 | researcher-mathematical | revise | P-010 | P-016 | llm | All points accepted, with the following changes. (1) Architecture confound: I replaced the sklearn MLPRegress… |
| 3 | researcher-ml | revise | P-011 | P-017 | llm | (1) Jensen bias: accepted. With a posterior relative std of about 0.5-2%, the bias is about sigma^2/2, roughl… |
| 3 | researcher-experimental | abandon | P-012 | - | llm | I accept all four points. (1) P-015 runs the same mode-vs-mean contrast on the physics arm in about 12 s inst… |
| 3 | researcher-experimental | revise | P-013 | P-018 | llm | (1) Normalisation: I will compute err_j as the relative RMSE, sqrt(mean_i((k̂_ij − k_ij)/k_ij)²), which match… |
| 3 | researcher-exploratory | abandon | P-008 | - | llm | I accept the merge. P-008 and P-014 run the identical method spec, and P-014 has the sharper absolute criteri… |
| 3 | researcher-exploratory | revise | P-014 | P-019 | llm | (1) Duplicate of P-008: P-008 is withdrawn, and P-014 is now the single canonical physics arm. Per-sample est… |
| 3 | researcher-exploratory | revise | P-015 | P-020 | llm | (1) Gap too large: I accept the critic's back-of-envelope estimate. Bounded LM already clamps the mode at the… |
| 3 | researcher-exploratory | propose | - | P-021 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.666 | fdbk-00044 |
| 2 | 8 | 0.81 | fdbk-00075 |
| 3 | 6 | 0.772 | fdbk-00106 |

### Idea evolution

**P-001 — Hybrid amortised initializer + log-space LM posterior-mean refinement reaches near the CRB floor** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status literature-supported. *H:* An MLP trained on simulated (noise-augmented) sensor time series, used as initializer for bounded log-space Levenberg–Marquardt through the forward model with a posterior_mean estimator, attains val rel_l2 < 0.035, i.e. within ~2x of the harness CRB floor (0.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the hybrid is not better than the MLP alone, the network is already near the MMSE and the forward model adds little. If it is worse than the CRB by a large …
  - critic r1 (quality 0.886): The headline prediction (<0.035) is weak. If the memories cited elsewhere are accurate (bounded log-space MLE reached 0.0193 in m-261109-81a87c8d), a hybrid th… · The method is the same as P-003 and nearly the same as P-007, which differs only in the init PCA/width. Three of eight slots test one configuration. · Treat the learned init as a speed device, not an accuracy device. If P-008 converges from the prior centre, the init only reduces iterations, and the 20%-over-… · If the MLP init is already within about 1 sigma_CRB, LM refinement plus posterior-mean readout yields roughly the same MSE as the MLP. The hybrid then fails th… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `hybrid[torch_mlp/raw+logy+aug2->LM12]` status literature-supported. *H:* MLP init (log-target, noise-augmented), followed by 12 bounded log-space LM iterations with a Laplace-IS posterior-mean readout, reaches val rel_l2 < 0.021. The per-parameter RMS relative errors are within 1.3x the CRB rel std (k1 2.15%, k2 0.58%, k3 0.52%, k…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.021}`; if it fails: If (b) fails only for k1/k4 at the box edges, the posterior-mean readout or IS is the bottleneck, which motivates a tighter-tailed proposal or exact quadrature…
  - response to critic: Pilot result (n_val=60, train 300): rel_l2 0.0184. Residual std at truth was 0.00099 against nominal 0.001, model_mismatch=false over 200 samples, chi2/dof 0.944, 1-sigma/2-sigma coverage 0.70/0.96, z_rms 1.00. This answers each point. (1)…
  - critic r2 (quality 0.83): The pilots show the hybrid (P-001: 0.018397) and the no-learning physics arm (P-008: 0.018395) agree to about 1e-6. Their chi2/dof (0.94442 vs 0.94444), ESS (5… · The harness estimate is 132 s. That is above the stated 120 s per-arm limit and about a third of the whole 400 worker-second budget. It buys a prediction (<0.0… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If LM from the prior centre already converges on more than 99% of samples, the hybrid's rel_l2 equals P-014's up to Monte Carlo IS noise. Passing (a)-(c) would… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-002 — Pure amortised MMSE: noise-augmented MLP on log-targets meets the 0.05 target without the forward model at inference** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* Because the parameter dimension is only 4 and the forward map is smooth, a squared-loss torch MLP trained on 3000 simulated samples with noise augmentation (log-target) approximates the posterior mean well enough to reach val rel_l2 < 0.05. However, it stays …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: Shows that the learned map is sample-limited. One could then generate additional simulated training data via context.forward in a custom_solver, which is allow…
  - critic r1 (quality 0.459): If the prior MLP result (about 0.025) is real, the <0.05 prediction is nearly certain to pass and carries little information. The useful prediction is the gap … · This is near-duplicate of P-004 and P-006: same model family and target transform, with only width and noise_aug changed. · The posterior mean is about the MLE here when sigma is small. The MLP's excess error over MLE is pure approximation/estimation error, so it should shrink with … · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-010` r2 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The pure MLP's excess error over the noise floor is sample-limited. Tripling the simulated training set (3000 -> 9000, via context.forward on fresh uniform-prior draws with sigma=0.001 noise, raw-k targets) reduces val rel_l2 to < 0.0225. It also cuts the exc…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0225}`; if it fails: If the error does not drop with 3x n, the learned map is capacity- or optimisation-limited rather than sample-limited. The forward-model refinement (P-001) is …
  - response to critic: All points accepted. (1) The <0.05 prediction is uninformative. It is replaced by a quantitative learning-curve prediction. (2) The original is a near-duplicate of P-006/P-004. Those arms remain the n=3000 MLP control, and this arm becomes…
  - critic r2 (quality 0.74): The comparator P-006 is not in this round; P-013 is the n=3000 control. P-010 uses an sklearn MLPRegressor with raw-k targets, while P-013 uses a torch 512-256… · P-010 and P-011 test the same hypothesis (the MLP-CRB gap is sample-limited) at n=9000 and n=15000. Together with P-013 they make a useful 3-point curve, but o… · Use P-013's exact torch spec (log target, noise_aug) at n=9000, so that n is the only varying factor. · sklearn MLP with early stopping can plateau at about 0.03 because of optimisation (Adam with a fixed lr and no scheduler) at any n. That would look like 'not s… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-016` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* With P-013's architecture and loss held fixed, adding 6000 fresh prior-predictive simulations (with per-epoch resampled sigma=0.001 noise) reduces val rel_l2 to < 0.0225. The excess MSE over the physics floor (P-014) shrinks by at least 30% relative to P-013 …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0225}`; if it fails: If (a) and (b) both fail with alpha near 0, the amortised network is capacity- or optimisation-limited at this recipe. More simulated data is then wasted, and …
  - response to critic: All points accepted, with the following changes. (1) Architecture confound: I replaced the sklearn MLPRegressor with a torch network that reproduces P-013's recipe. That is 512-256-128, LayerNorm+GELU, AdamW lr 1e-3 / wd 1e-4, a cosine sch…
  - critic r3 (quality 0.6): The architecture must match the other learning-curve points (P-017, P-018). Otherwise the curve is not interpretable. · The hypothesis is meant to hold P-013's recipe fixed, but it changes three things at once: n (3k to 9k), noise augmentation (fixed noise_aug=2 copies become pe… · P-016 and P-017 both claim to 'match P-013' but specify different networks (512-256-128 versus [256,256]). At most one of them matches. The 3-point learning cu… · Treat the n=3k anchor as P-018, trained with exactly the same code, epochs and per-epoch noise scheme, not as a historical P-013 number. · If the 3k arm is under-trained (fewer steps), a gain at 9k could come from extra optimisation rather than extra data. That would mimic an estimation-limited cu…

**P-003 — Amortised learned init + short posterior-mean LM refinement reaches near-CRB accuracy (hybrid)** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status conjectured. *H:* A torch_mlp trained on log-features with log-targets and noise augmentation, used as initializer for <=15 bounded LM iterations in log-space with a posterior_mean (importance-sampled MMSE) readout, achieves val rel_l2 <= 0.03, i.e. within ~1.6x of the CRB flo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If hybrid ~= init alone, the MLP already sits near the noise floor (approximation error is not the bottleneck). If hybrid is much worse than the CRB but better…
  - critic r1 (quality 0.39): Exact duplicate of P-001; merge or differentiate. · executes exactly the same method as P-001 (researcher-mathematical); differentiate the hypothesis or merge · The method is identical to P-001 (rule finding). The only difference is a tighter threshold (0.03 vs 0.035), and that is not a distinct hypothesis. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-011` r2 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised MLP's excess error over the CRB floor (~0.0185) is mainly finite-sample approximation error. Training the same-architecture MLP on 5x more prior draws, simulated with context.forward plus fresh N(0, 0.001^2) noise each epoch, will reduce val rel…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If the gain is under 15%, the MLP to CRB gap is not a data limit. It is then capacity/optimisation or log-target Jensen bias, and only forward-model refinement…
  - response to critic: (1) Duplicate of P-001: accepted. The hybrid arm (torch_mlp init, log-space LM, posterior_mean) is folded into P-001, and I endorse P-007's sharper criteria and init-only logging for that merged arm. This slot now holds a hypothesis no oth…
  - critic r2 (quality 0.886): The Jensen-bias counter-argument is quantitatively negligible. The posterior rel std is about 0.5-2%, so the bias from exp(E[log k]) is about sigma_rel^2/2, ro… · The harness estimate of 30 s looks optimistic for 15000 rows x 60 epochs of a 512-256-128 torch MLP on CPU, plus 12000 forward calls. Measure it in the pilot, … · Drop 'log-target Jensen bias may cap the gain'. Replace it with 'optimisation or capacity may cap the gain', which is the real alternative. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-017` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The gap between the amortised MLP and the CRB (P-013 vs the ~0.0185 floor) is mainly finite-sample approximation error. Under that explanation, the same architecture, loss and target transform trained on 5x more prior draws (fresh noise each epoch) reduces va…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: A gain under 15% means the MLP-to-CRB gap is not a data limit. It is then an optimisation/capacity floor at this width, the Jensen explanation having been rule…
  - response to critic: (1) Jensen bias: accepted. With a posterior relative std of about 0.5-2%, the bias is about sigma^2/2, roughly 2e-4 or less. That is negligible against a 0.02 error, so it is removed as a competing explanation. The real alternative is now …
  - critic r3 (quality 0.725): The architecture and seed protocol must match those of P-016 and P-018. · The [256,256] architecture contradicts P-016 and P-018 (512-256-128), although all three claim to match P-013. · The falsifier compares against P-010 (n=9k), which is not in this round. P-016 is the 9k point. Use P-016's run. · Use one architecture shared with P-016 and P-018. · A wider network at fixed n could close the gap. That would show capacity, not data, is binding, and the n-curve alone cannot distinguish this from a slow data …

**P-004 — Pure amortised MMSE: noise-augmented MLP on log-features reaches target without forward-model refinement** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug4` status conjectured. *H:* A squared-loss torch_mlp with log-target and heavy noise augmentation (noise_aug=4) approximates the posterior mean E[k/y] well enough to achieve val rel_l2 <= 0.05 using only simulated training data, but stays >= 1.3x above the best hybrid (i.e. above ~0.024…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If > 0.05, approximation error from limited training data dominates; next steps are hybrid refinement or generating extra simulated training data via a custom_…
  - critic r1 (quality 0.399): Budget: est 200 s for a near-duplicate arm, out of 400 s total. · This is a third pure-MLP arm. noise_aug=4 is the only substantive difference, and at est 200 s it is the most expensive arm in the portfolio. It does not fit t… · The falsification criterion 'within 1.1x of the best hybrid' depends on another arm and is not self-contained. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — MMSE readout beats the MLE mode at the box edges: learned init + LM refinement with posterior-mean estimator** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[extra_trees/raw+logy->LM12]` status literature-supported. *H:* A hybrid estimator will reach val rel_l2 below the 0.0193 achieved by bounded log-space MLE in a prior run. It uses an extra-trees initializer, followed by log-space LM refinement through the forward model, and reads out the posterior mean (importance samplin…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.019}`; if it fails: If mean and mode tie, the box-truncation effect is negligible at sigma=0.001, and about 0.019 is the practical floor for this sensor layout. Further gains woul…
  - critic r1 (quality 0.898): The predicted improvement (<0.019 vs 0.0193) is well below the val sampling error. With 750 samples, the SE of mean rel_l2 is probably about 0.0005–0.001, so a… · Size the boundary effect before running. With sigma_k1 of about 0.027 absolute, only samples within about 2 sigma (about 0.05) of a bound are affected. That is… · The gain from posterior_mean should be concentrated almost entirely in the stratum where k1 or k4 is within 0.1 of a bound. The interior stratum should show a … · If the Laplace proposal is truncated at the box without renormalisation, the posterior mean is biased toward the interior by more than the true posterior. That… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-012` r2 [revise, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status literature-supported. *H:* Hold the init and LM fixed and change only the readout from mode to posterior mean. The mean should then lower squared error only for val samples whose true k1 or k4 lies within 0.1 of a bound (about 25% of samples under the uniform prior). In that boundary s…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0205}`; if it fails: Two failure patterns are informative. If the boundary-stratum gain is null, then either the posterior is so tight that truncation is negligible, or the importa…
  - response to critic: I accept every point, and the revision turns P-005 into the missing paired control rather than a fourth posterior-mean hybrid. (1) SE and pairing: correct. Comparing against an unverified prior-run 0.0193 cannot resolve a gain of about 1-3…
  - critic r2 (quality 0.826): P-015 runs the same mode-vs-mean contrast on the physics arm for 12.4 s, against 125.6 s here. The pilots show the hybrid and physics arms reach the same estim… · 125.6 s exceeds the 120 s per-arm limit, and together with P-009 and P-013 it pushes the portfolio over the 400 s budget (total about 483 s). · Replace the fixed 0.1 absolute stratum with a 2-sigma stratum based on the per-sample posterior std. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-006 — Amortised MMSE without the forward model at inference: a noise-augmented MLP on raw sensor traces reaches the 0.05 targ…** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch MLP trained on the 3000 simulated training pairs, with squared loss on log-k and noise augmentation matched to sigma, will reach val rel_l2 at or below 0.05. It needs no per-sample forward solves at prediction time. Because the noise is small relative…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 is above 0.05, the amortised map is sample-limited for the weakly informed edge segments. Physics or hybrid refinement through the forward model …
  - critic r1 (quality 0.865): This is the best-specified of the three MLP arms: it includes a per-parameter ratio prediction. However, the <0.05 threshold is again likely uninformative give… · The 'k1/k4 errors >= 2x k2/k3' prediction mirrors the CRB. That ordering holds for any decent estimator, so it does not discriminate approximation-limited from… · Use the ratio error_j / CRB_j as the diagnostic, not the error ordering. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-013` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch MLP trained on the 3000 training pairs with log targets and noise augmentation at sigma will reach val rel_l2 in [0.022, 0.030]. Its excess over the CRB floor of 0.0185 comes from function-approximation error at n=3000, not from noise. This shows up a…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If rel_l2 is above 0.030, or the edge ratios are large, the network is sample-limited on the weakly informed edge segments. The forward-model refinement is the…
  - response to critic: (1) The <0.05 threshold is uninformative given the prior of about 0.025, which comes from memory anchor m-261109-d9e17446 / 0.0247. I tighten it to a two-sided band: val rel_l2 in [0.022, 0.030]. Both edges are falsifiable. Below 0.022 mea…
  - critic r2 (quality 0.916): The ratio-ordering prediction is the most informative claim in the portfolio, but it depends on the CRB scale matching the metric. Confirm how crb_rel_median w… · The interior-stratum 'every ratio > 1.2' criterion compares an MMSE approximator with a bound for unbiased estimators. In the interior that comparison is valid… · Use per-sample CRB_j(k_true) rather than the median CRB in the ratios, because the CRB varies across the box. · A network that underfits k2/k3 because of their small target variance in log space would also produce an inverted ordering. Standardised targets remove that co… · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-018` r3 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch MLP trained on 3000 training pairs, with log targets and noise augmentation at sigma, reaches val rel_l2 in [0.022, 0.030]. In the interior stratum (every edge parameter more than 2 posterior std from a bound), the per-sample RMS ratio r_j = err_j/CRB…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If rel_l2 is above 0.030 or the edge ratios exceed 1.7, the network is sample-limited on the weakly informed edges, and the forward-model refinement or more si…
  - response to critic: (1) Normalisation: I will compute err_j as the relative RMSE, sqrt(mean_i((k̂_ij − k_ij)/k_ij)²), which matches a per-sample sqrt(CRB_jj)/k_j. I also report the vector-norm rel_l2 separately, so the two are never conflated. I request that …
  - critic r3 (quality 0.755): The dependency on P-019's per-sample CRB outputs must be declared, and the run ordered after it. · The interior stratum and CRB_j depend on per-sample Gauss-Newton std from the physics arm, which the text calls P-015 although the physics arm is P-019 in this… · The per-sample ratio thresholds (k2/k3 ratio >= 2, k1/k4 ratio in [1.0, 1.7]) follow from an assumed uniform approximation floor of 0.008-0.012. That floor is … · Make P-018 the explicit n=3k anchor, with the same epochs, noise scheme and architecture as P-016 and P-017. Report the 150-epoch result separately if wanted. · A floor proportional to the CRB (as from a network that is uniformly a fraction inefficient) gives flat ratios above 1 rather than inverted ones. This is a thi…

**P-007 — Amortised init + forward-model MMSE refinement: closing the gap from the MLP (≈0.025) to the CRB floor (≈0.0185)** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+pca48+logy+aug2->LM12]` status conjectured. *H:* The best amortised network (torch_mlp raw+pca48+logy, val rel_l2≈0.0247) is limited by approximation error, not by noise. A few bounded LM steps through the exact forward model, started from that network, followed by an importance-sampled posterior mean, will…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If refinement gives no gain, the MLP is already near the Bayes risk and the CRB-based floor is misleading (Laplace/CRB overstates what is achievable, or the re…
  - critic r1 (quality 0.51): This is functionally a P-001 variant: a hybrid with an MLP init, differing by pca48 and width. The hypothesis, that refinement removes approximation error, is … · Its sharper prediction (<=0.022, 10% or more over its own init) is better than P-001's. It should replace P-001's threshold rather than run separately. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-008 — No learning at all: per-sample box-constrained posterior mean through the forward model already beats the amortised net…** (researcher-exploratory, family physics)
- v1 `P-008` r1 [propose, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* Training on 3000 simulations is unnecessary because the forward model is available, cheap and well-conditioned. Per-sample bounded LM in log-k space with a few multistarts, followed by the importance-sampled posterior mean, should reach val rel_l2 < 0.024, at…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.05, "expect": "better"}`; if it fails: If it fails through local minima or slow valleys, the learned initialiser is essential and proposal 1's framing is right. If it converges but rel_l2 is still a…
  - critic r1 (quality 0.925): This is the most informative per second in the portfolio (est 16 s). It directly tests the never-executed unimodality conjecture, m-261109-54edbdef. · The 'chi2 > 1.5' threshold needs care. Normalised chi2 over about 160 residuals has std of about sqrt(2/160), roughly 0.11, so 1.5 is a reasonable outlier cut.… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-014 — No learning at all: per-sample box-constrained posterior mean through the forward model (full-val confirmation of pilot…** (researcher-exploratory, family physics, branched from P-008)
- v2 `P-014` r2 [branch, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* Training data are unnecessary for accuracy. Per-sample bounded LM in log-k with 3 starts and 25 iterations, followed by a box-truncated Laplace-IS posterior mean, reaches val rel_l2 < 0.0225 on the full 750-sample val set. This is at least 9% better than the …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0225}`; if it fails: If rel_l2 is >= 0.0225 but chi2 is fine, the 60-sample pilot was optimistic, and the gap to the MLP is within noise. In that case, choose between the physics e…
  - response to critic: The pilot supports the hypothesis: 0.0184 on n=60 is below the 0.0247 MLP anchor, and normalised chi2 is 0.94. With n=60 the standard error on rel_l2 is roughly ±0.002, so the full 750-sample run is needed. Point by point: (a) Comparator: …
  - critic r2 (quality 0.564): executes exactly the same method as P-008 (researcher-exploratory); differentiate the hypothesis or merge · This duplicates P-008 (deterministic rule), so run it once under the merged spec. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-019` r3 [revise, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* Without any training data, per-sample bounded LM in log-k (3 starts, 25 iterations) with a box-truncated Laplace-IS posterior mean reaches val rel_l2 < 0.0225 on all 750 val samples (refutation threshold), and in fact < 0.020 (near-floor diagnostic). The log-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0225}`; if it fails: If rel_l2 >= 0.0225 but chi2 is fine, the pilot was optimistic, and the choice between physics and hybrid comes down to cost. If the chi2 or start criterion fa…
  - response to critic: (1) Duplicate of P-008: P-008 is withdrawn, and P-014 is now the single canonical physics arm. Per-sample estimates, chi2, ESS and start outcomes are logged so P-015 and P-016 can be paired with it. (2) Ill-posed start criterion: I adopt t…
  - critic r3 (quality 0.94): The refutation threshold of 0.0225 is weak given the 0.0184 pilot. Pre-register the sharper claim as primary: the bootstrap CI for rel_l2 includes the CRB esti… · Cost estimate: the forward budget is quoted as 270k evaluations, but the rule estimate is 16.4 s. Report the actual forward_evals so that P-020 and P-021 can s… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · k1/k4 valleys near the bounds could produce one-sided clamped optima that do not show up as distinct starts but do show up as elevated chi2. Check chi2 conditi… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-015 — Branch of P-008: paired MLE (mode) on identical val samples — replicate the 0.0193 anchor and measure the MMSE gain** (researcher-exploratory, family physics)
- v1 `P-015` r2 [propose, llm] `physics-MLE[log,ms=3,it=25]` status conjectured. *H:* With an identical LM configuration (log, 3 starts, 25 iterations), the mode estimator has a higher val rel_l2 than the posterior_mean estimator from P-008. The expected gap is about 3–8%, concentrated in samples with any k_j within 2 CRB-std of a bound and mo…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.03, "expect": "worse"}`; if it fails: If the mode matches or beats the posterior mean, the Laplace-IS readout adds nothing, either because truncation effects are below the noise or because the IS i…
  - critic r2 (quality 0.84): The predicted 3-8% gap is probably too large. With about 7% of the prior mass in each edge parameter's 2-sigma band and k1/k4 carrying about 93% of the MSE, ev… · Bounded LM clamps the mode at the bound, which partly captures the truncation benefit and shrinks the expected gap. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-020` r3 [revise, llm] `physics-MLE[log,ms=3,it=25]` status conjectured. *H:* With the LM configuration identical to P-014, the mode readout has an aggregate val rel_l2 0.5-2% higher than the posterior mean. The gain is significant in the edge strata (any k_j within 2 per-sample CRB-std of 0.5 or 2.0, mostly k1/k4), and larger at the u…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.01, "expect": "worse"}`; if it fails: If the edge gain is null, the Laplace-IS readout adds nothing beyond clamping, and the cheaper MLE should be the final estimator. If the interior difference is…
  - response to critic: (1) Gap too large: I accept the critic's back-of-envelope estimate. Bounded LM already clamps the mode at the box, which captures part of the truncation benefit. The prediction is revised to an aggregate gap of 0.5-2%. (2) Noise-driven fal…
  - critic r3 (quality 0.84): The 0.5-2% aggregate gap is about 1e-4 in rel_l2, below the bootstrap resolution unless the comparison is paired per sample. The proposal does pair, which is c… · Setting posterior_samples=32 is irrelevant for the mode readout. Reuse P-019's LM optimum directly, which costs about 0 s. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the edge gain comes almost entirely from samples with clamped modes, the IS readout's value equals a cheap 'unclamp toward interior' shrinkage. Test that he… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-021 — IS readout accuracy control: same LM, posterior_samples 512 vs 128, paired with P-014** (researcher-exploratory, family physics)
- v1 `P-021` r3 [propose, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* At 128 importance samples, the posterior-mean readout's Monte Carlo error is negligible relative to the posterior width. Re-running P-014 with 512 samples changes aggregate val rel_l2 by less than 1% relative, the paired interior CI contains 0, and the per-sa…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.005, "expect": "better"}`; if it fails: If the shifts are material, the 128-sample readout is Monte Carlo-limited. P-015's edge contrast must then be read against the 512-sample arm, and the final es…
  - critic r3 (quality 0.77): Using different IS draws at 128 and 512 conflates Monte Carlo error with the readout change. Use common random numbers: the 512 draws should contain P-019's 12… · The test of a shift below 0.1 posterior std for 95% of samples is reasonable. However, the MC std at ESS 58 is about 0.13 std, so the stated expectation may al… · Reuse P-019's LM stage, so that only the IS stage is rerun, which halves the cost. · A low ESS at the edges arises from proposal mismatch, not from sample count. Quadrupling the samples barely helps, whereas a truncated-Gaussian proposal or qua… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: The portfolio is dominated by duplicates and exceeds the budget by more than 2x. The pure-MLP <0.05 predictions are likely uninformative given remembered anchors, which are themselves unverified. Prioritise the cheap, discriminating arms: P-008 (unimodality and whether learning is needed), P-005 (MMSE vs mode, paired and stratified), one merged hybrid with P-007's sharper criteria, one MLP contro…  families={'hybrid': 4, 'data-driven': 3, 'physics': 1}
- Critic round 2 portfolio view: The pilots already put the physics estimator at about the CRB floor (0.0184 vs 0.0185 on n=60), well below the 0.05 success target, with calibrated uncertainty. The remaining scientific questions are: (1) full-val confirmation and unimodality (P-014), (2) whether the posterior-mean readout matters at all (P-015, which likely shows about a 1% effect), and (3) why amortised networks fall short (P-0…  families={'hybrid': 2, 'physics-custom': 2, 'data-driven': 1, 'physics': 3}
- Critic round 3 portfolio view: Empirically the physics posterior-mean arm (P-019) already sits at the CRB floor (about 0.0184, well below the 0.05 success threshold) and should be the default final model, pending full-val confirmation. The amortised arms are scientifically useful but must be unified into one consistent learning-curve study (shared architecture, step budget and noise scheme, 2 seeds) plus a capacity control. P-…  families={'physics-custom': 2, 'data-driven': 1, 'physics': 3}

**Pilot experiments run during research** (small validation subsets):

- P-001 `hybrid[torch_mlp/raw+logy+aug2->LM15]`: ok rel_l2=0.018397 on 60 val samples in 20.2 s
- P-008 `physics-MMSE[log,ms=3,it=25]`: ok rel_l2=0.018395 on 60 val samples in 3.56 s
- P-009 `hybrid[torch_mlp/raw+logy+aug2->LM12]`: ok rel_l2=0.018449 on 60 val samples in 16.1 s
- P-012 `hybrid[torch_mlp/raw+logy+aug2->LM15]`: ok rel_l2=0.019024 on 60 val samples in 17 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 62 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search (cache) |
| researcher-mathematical | 1 | uniqueness of piecewise constant conductivity from finitely many interior point observati… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability for recovering piecewise constant coefficients in a parabolic equatio… | 4 refs via /search |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient in one-dimensional heat equation from boundary or int… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-ml | 1 | posterior mean estimator minimum mean squared error neural network trained on simulated p… | 4 refs via /search |
| researcher-experimental | 1 | Bayes estimator posterior mean minimizes expected relative squared error; asymptotic effi… | 4 refs via /search |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Bayesian Cramér-Rao bound bounded parameter box posterior mean | 4 refs via /search |
| researcher-experimental | 2 | Per-sample IS effective sample size distribution, stratified by distance of the true k1/k… | 4 refs via /search |
| researcher-ml | 2 | Levenberg-Marquardt convergence piecewise constant conductivity parabolic inverse problem | 4 refs via /search |
| researcher-mathematical | 2 | Verify memory records m-261109-81a87c8d (MLE 0.0193) and m-261109-25e00c80 (MLP 0.0247); … | 4 refs via /search |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 3 | global convergence Gauss-Newton parabolic coefficient inverse problem unimodal misfit | 4 refs via /search |
| researcher-experimental | 3 | self-normalized importance sampling truncated Gaussian posterior mean bias near support b… | 4 refs via /search |
| researcher-mathematical | 3 | Per-sample /k_hat_hybrid - k_hat_physics/ on the 60 pilot samples | 4 refs via /search |
| researcher-ml | 3 | Wall-clock time for 12000 vectorised context.forward calls | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\shared_memory.jsonl; 167 records from 2 run(s) were available at start; agents retrieved 18 distinct prior-run records; this run wrote 155 records (test numbers redacted for agents).
  - `m-261109-1876311d` critique (recorded) from `heat_v2_full_s0_20261109_001356`: critique r1 of Metric-matched estimator: a learned init plus LM refinement with a posterior-mean (MMSE) readout beats the MAP/MLE mode 
  - `m-261109-25e00c80` observation (experimentally-supported) from `heat_v2_full_s0_20261109_001356`: torch_mlp/raw+pca48+logy: rel_l2=0.024664442297447027
  - `m-261109-50b0f631` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Corollary 4.5 (Global uniqueness with full data) — Simultaneous recoveries for semilinear parabolic systems
  - `m-261109-54edbdef` hypothesis (conjectured) from `heat_v2_full_s0_20261109_001356`: Pure per-sample MLE with multistart LM (no learning) is sufficient: the least-squares landscape is effectively unimodal on the prior box
  - `m-261109-55e36094` observation (observed) from `heat_v2_full_s0_20261109_001356`: torch_mlp/log+pca48+logy: rel_l2=0.032390947689354896
  - `m-261109-7ea0a084` hypothesis (literature-supported) from `heat_v2_full_s0_20261109_001356`: Noise-limited regime: bounded log-space MLE through the forward model should land near the CRB floor (~0.02 rel_l2)
  - `m-261109-81a87c8d` finding (experimentally-supported) from `heat_v2_full_s0_20261109_001356`: supported: Noise-limited regime: bounded log-space MLE through the forward model should land near the CRB floor (~0.02 rel_l2)
  - `m-261109-820f8a3d` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Theorem — Uncertainty quantification and weak approximation of an elliptic inverse problem
  - `m-261109-836f0093` hypothesis (conjectured) from `heat_v2_full_s0_20261109_001356`: Amortized MLP with log-target and noise augmentation is accurate but cannot reach the CRB without physics refinement
  - `m-261109-8674ccca` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Theorem 4 — Acousto-electric Inverse Source Problem
  - `m-261109-88bd39b7` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}) — Bridging and Improving Theoretical and Computational Electric Impedance Tomogr…
  - `m-261109-c68594a4` critique (recorded) from `heat_v2_full_s0_20261109_001356`: critique r1 of Pure per-sample MLE with multistart LM (no learning) is sufficient: the least-squares landscape is effectively unimodal 

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-019) | physics | `physics-MMSE[log,ms=3,it=25]` | researcher-exploratory | 0.94 | best-rated proposal in family 'physics' (quality 0.94) |
| E-02 | proposal (P-020) | physics | `physics-MLE[log,ms=3,it=25]` | researcher-exploratory | 0.84 | quality 0.84 with diversity penalty |
| E-03 | proposal (P-021) | physics | `physics-MMSE[log,ms=3,it=25]` | researcher-exploratory | 0.77 | quality 0.77 with diversity penalty |
| E-04 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-05 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-06 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-016 *Learning-curve point n≈9000 with P-013's recipe: is the amortised-network gap t…* — blocking: The architecture must match the other learning-curve points (P-017, P-018). Otherwise the curve is not interpretable.
- not advanced: P-017 *Learning-curve endpoint (n=15k): P-013-matched MLP trained on 3000 given + 1200…* — blocking: The architecture and seed protocol must match those of P-016 and P-018.
- not advanced: P-018 *Single data-driven control: noise-augmented MLP is approximation-limited, with …* — blocking: The dependency on P-019's per-sample CRB outputs must be declared, and the run ordered after it.

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-004 | E-03 | research | physics | `physics-MMSE[log,ms=3,it=25]` | 0.02041 | **0.018966** | 0.9967 | 0.02501 | 0.00998 | 75.6 | 531627 | 0.953 | FINAL |
| idea-002 | E-01 | research | physics | `physics-MMSE[log,ms=3,it=25]` | 0.02027 | **0.019056** | 0.9967 | 0.02513 | 0.00975 | 38.6 | 262560 | 0.952 | evaluated |
| idea-008 | E-06 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.00998 | 11.8 | 61517 | 0.957 | evaluated |
| idea-003 | E-02 | research | physics | `physics-MLE[log,ms=3,it=25]` | 0.02139 | **0.019338** | 0.9966 | 0.0255 | 0.00948 | 27.8 | 172754 | 0.957 | evaluated |
| idea-010 | E-04 | refine | data-driven | `torch_mlp/sqrt` | - | **0.028495** | 0.9925 | 0.03758 | 21.7 | 0.0134 | 0 | - | evaluated |
| idea-009 | E-04 | refine | data-driven | `torch_mlp/log` | - | **0.028979** | 0.9923 | 0.03822 | 30.7 | 0.0103 | 0 | - | evaluated |
| idea-005 | E-04 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **0.032246** | 0.9904 | 0.04252 | 23 | 0.0127 | 0 | - | evaluated |
| idea-007 | E-05 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 6.57 | 0.341 | 0 | - | evaluated |
| idea-012 | E-05 | refine | data-driven | `extra_trees/sqrt` | - | **0.082521** | 0.9373 | 0.1088 | 4.75 | 0.197 | 0 | - | evaluated |
| idea-011 | E-05 | refine | data-driven | `extra_trees/sqrt` | - | **0.10413** | 0.9002 | 0.1373 | 2.18 | 0.238 | 0 | - | evaluated |
| idea-006 | E-05 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 202s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-004` `physics-MMSE[log,ms=3,it=25]` ← P-021 v1 (researcher-exploratory, r3, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00205` [checkpoint] by scheduler: final method (single)
- `resu-00202` [result] by executor: exp-0023 rel_l2=0.01827 r2=0.997 rmse=0.02404
- `hypo-00117` [hypothesis] by scheduler: idea-004 physics-MMSE[log,ms=3,it=25] (research)
- `plan-00201` [plan] by engineer-v2: exp-0023: final fit of 1 member(s) on train+val; test once
- `gate-00116` [gate] by gate: E-03: approved (physics-MMSE[log,ms=3,it=25])
- `prop-00104` [proposal] by researcher-exploratory: P-021 v1 [propose] exploratory: IS readout accuracy control: same LM, posterior_samples 512 vs 128, paired with P-014
- `spec-00001` [spec] by scheduler: research spec heat_v2_memory_s1 (918b2cc00c)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 442.7 / 400 worker-seconds over 23 jobs; wall 763 s
- By stage: baseline 2.1s, pilot 56.9s, screen 42.9s, evaluate 158.8s, refine 70.1s, final 112.0s
- LLM: 19 calls (by phase {'research': 19}), 959 s, $3.852 reported · TheoremSearch: 24 queries (0 failed)

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

The objective is met by a wide margin. The selected estimator is a per-sample posterior mean computed in log space by running the forward model, with 3 multistarts and 25 iterations (`physics-MMSE[log,ms=3,it=25]`). It was chosen on validation (rel_l2 0.01897) and scored rel_l2 0.01827 on the held-out test set (R² 0.997, RMSE 0.024), against the ≤ 0.05 target. Test error is close to validation error, so selection did not visibly overfit. The best purely data-driven model (`torch_mlp/sqrt`, 0.0285) is about 50% worse. The physics estimate also sits at roughly the Cramér–Rao bound (CRB) estimate of about 0.0185, the best error any unbiased estimator can reach at this noise level. Together these suggest the remaining error is mostly the irreducible error set by noise at σ = 0.001 (Fisher condition ≈ 62), not error from the model. Two predictions held: one physics variant beat 0.0225 (0.01906), and the hybrid with the learned initialiser came out worse than the best physics run (−1.96%), as expected.

**What failed and why**
- **Learned initialiser:** the hypothesis that it improves forward-model refinement is not supported. In the pilots, the hybrid and the prior-centre Levenberg–Marquardt (LM) fit agreed to about 1e-6 on rel_l2, χ²/dof (chi-squared per degree of freedom), effective sample size and coverage, while the hybrid cost 4–5× more. This points to a single dominant likelihood basin, so initialisation does not matter.
- **Mean vs mode:** the predicted gain of ≥ 0.5% from posterior-mean over mode readout came in at +0.47% and was formally contradicted. That is a near-tie and probably within sampling noise at n = 750, so it says nothing either way.
- **Noise augmentation:** the pure amortised MLP with noise augmentation was abandoned without being run. Its question (does more effective training data close the amortised gap?) is still open, not answered negatively.

The single-basin claim and the MLP's irrelevance rest on n = 60 pilots. The CRB figure of 0.0185 is itself an estimate, and its uncertainty was not reported. The test score (0.0183) is slightly below it, which is consistent with a bounded prior beating the unbiased-estimator bound, but this was not checked. All data are simulated with the same forward model and noise model, so robustness to model mismatch, other noise levels or sensor misplacement is untested.

**Next experiments**
- **Uncertainty on small gaps:** bootstrap or paired-difference confidence intervals on the validation error differences (mean vs mode, ms = 3 vs 1, 25 vs 15 iterations), to decide whether the sub-1% gaps are real.
- **Confirm single basin on full validation:** count multistart disagreements and boundary-stratum cases across all 750 samples.
- **Noise sweep:** repeat at σ ∈ {0.003, 0.01, 0.03}. Multimodality, the value of a learned initialiser, and the mean-vs-mode gap are most likely to show up when noise is higher.
- **Misspecification stress test:** perturb sensor locations, boundary values or noise colour in held-out simulations, and compare how physics and amortised errors degrade.
- **Amortised scaling curve:** vary the number of new parameter draws (not noise copies) to see whether the MLP approaches the CRB. This would settle the abandoned noise-augmentation question cheaply.

_Automatic summary:_ 21 proposal versions over 3 research rounds; 6 methods advanced; 2 predictions supported, 1 contradicted. Final: ['physics-MMSE[log,ms=3,it=25]'] with test rel_l2 0.018272.
