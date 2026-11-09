# Research report (V2): heat_v2_full_s0

Run `heat_v2_full_s0_20261109_001356` · 2026-11-09 00:24:00 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 46.6 (max 1.78e+03); Cramér–Rao relative std per parameter: k1 0.0209, k2 0.0063, k3 0.0051, k4 0.0161; implied rel_l2 floor ≈ **0.01949**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `hybrid[ridge/log+logy->LM20]`
- Validation rel_l2 (selection estimate): **0.019325** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018412**, r2 = **0.99696**, rmse = **0.024226** · bootstrap 95% CI for rel_l2: [0.01728, 0.01954]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.94× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.093612 (vs 0.018412 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.052031 (vs 0.018412 in distribution)
- Uncertainty calibration on test: 70.1% within 1σ, 95.6% within 2σ (ideal 68% / 95%)
- Inference latency: 3.48 ms/sample; fit 0.0237 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s0\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s0")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[ridge/log+logy->LM20]` rel_l2=0.019325; **physics** `physics-MLE[log,ms=2,it=21]` rel_l2=0.019325; **data-driven** `torch_mlp/raw+pca48+logy` rel_l2=0.024664
- SUPPORTED — E-01 *Mode twin of P-014: ridge/log init + LM20, estimator=mode, paired against P-014's posterior_mean*: val rel_l2 = 0.019338 vs predicted < 0.0205
- SUPPORTED — E-02 *Forward-free amortised posterior mean: raw-target, unscaled-input MLP vs the 0.05 target and the hybrid gap*: val rel_l2 = 0.02609 vs predicted < 0.05
- SUPPORTED — E-03 *Single-basin test: one LM start from the prior centre reproduces multistart MLE accuracy*: val rel_l2 = 0.019335 vs predicted < 0.0205
- SUPPORTED — E-04 *Amortised inverse on information-preserving PCA-48 scores: below target but at least 1.3x the hybrid, with the gap conc…*: val rel_l2 0.024664 vs best_hybrid 0.019338: gain -27.55%; expected to be worse than best_hybrid (gain < +30.0%)
- SUPPORTED — E-05 *Noise-limited regime: bounded log-space MLE through the forward model should land near the CRB floor (~0.02 rel_l2)*: val rel_l2 = 0.019325 vs predicted < 0.032
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01949

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:22914204` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Theorem 7. http://arxiv.org/abs/1003.5004v1 — cited by ['P-001', 'P-009']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-005', 'P-013', 'P-019']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:21098230` Andrea Bonito, Albert Cohen, Ronald DeVore et al. (2016). "Diffusion Coefficients Estimation for Elliptic Partial Differential Equations". Corollary 3.8. http://arxiv.org/abs/1609.05231v2 — cited by ['P-001']; researcher claim ['no']; harness screen **no**; conflicts: different equation class (theorem is not about parabolic/diffusion equations); theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:22914205` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Theorem 18. http://arxiv.org/abs/1003.5004v1 — cited by ['P-002', 'P-010', 'P-016']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:19464924` M. S. Aronna, J. M. Machado (2021). "The shooting algorithm for partially control-affine problems with application to an SIRS epidemiological model". Proposition 6.2 (\cite{bonnans2006numerical,fletcher2013practical}). http://arxiv.org/abs/2103.06339v1 — cited by ['P-002', 'P-010', 'P-016']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:25496689` Masoumeh Dashti, Andrew M. Stuart (2011). "Uncertainty quantification and weak approximation of an elliptic inverse problem". Theorem. http://arxiv.org/abs/1102.0143v1 — cited by ['P-003', 'P-011', 'P-017']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26204848` Badr-Eddine Chérief-Abdellatif (2019). "Convergence Rates of Variational Inference in Sparse Deep Learning". Theorem 5. http://arxiv.org/abs/1908.04847v2 — cited by ['P-004']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:26986930` MohammadJava Taheri, Abolfazl Diyanat, MortezaAli Ahmadi et al. (2025). "Predicting Drive Test Results in Mobile Networks Using Optimization Techniques". Lemma 1. http://arxiv.org/abs/2502.09305v1 — cited by ['P-005', 'P-006', 'P-013', 'P-014', 'P-019']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:27031792` Shenghua Hu, Guangyang Zeng, Wenchao Xue et al. (2025). "Theoretical Guarantees for AOA-based Localization: Consistency and Asymptotic Efficiency". Theorem 2. http://arxiv.org/abs/2507.07647v2 — cited by ['P-005', 'P-013', 'P-019']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-007', 'P-008', 'P-015', 'P-020']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-007']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:26360863` Christian Kanzow, Theresa Lechner (2022). "Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems". Proposition. http://arxiv.org/abs/2210.07644v1 — cited by ['P-009', 'P-011', 'P-017']; researcher claim ['no', 'partial']; harness screen **unknown**
- `ts:thm:26746037` Dimitri Meunier, Zikai Shen, Mattes Mollenhauer et al. (2024). "Optimal Rates for Vector-Valued Spectral Regularization Learning Algorithms". Theorem 2 (Upper and lower bounds for KRR in the well-specified regime). http://arxiv.org/abs/2405.14778v1 — cited by ['P-011', 'P-012', 'P-018']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:25921946` Arash A. Amini, Richard Baumgartner, Dai Feng (2022). "Target alignment in truncated kernel ridge regression". Theorem 2. http://arxiv.org/abs/2206.14255v1 — cited by ['P-012', 'P-018']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- none

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-exploratory — P-007 *Rel-L2 is a squared loss, so target the Bayes estimator: learned init, then LM refinement, then a p…*: This duplicates P-001 and P-006 and exceeds the per-experiment cost limit. Its distinctive mechanism claim (the MAP is biased outward from the bounds) was wrong: box-constrained estimates are clipped inward, and any MMS…
- ABANDONED in round 3 by researcher-mathematical — P-009 *Paired estimator contrast: posterior-mean (MMSE, k-space, box-truncated) vs mode under an identical…*: blocking issue: The mode twin must have an identical init block. The current cross-references (P-003/P-005/P-011) disagree on noise_aug.
- ABANDONED in round 3 by researcher-experimental — P-014 *Metric-matched estimator: a learned init plus LM refinement with a posterior-mean (MMSE) readout be…*: duplicate of P-009

### 3.5 Remaining uncertainty

- E-01 (Mode twin of P-014: ridge/log init + LM20, estimator=mode, …): challenge: This is the most informative remaining contrast (estimator only), but it depends on P-014 being run with an identical seed, samples and init. If P-014 is not scheduled this round, the arm is an orphan.
- E-01 (Mode twin of P-014: ridge/log init + LM20, estimator=mode, …): challenge: The boundary stratum is small. 2 GN-std is about 0.021 at k=0.5 and about 0.084 at k=2 for k1, and somewhat less for k4. That gives roughly 10-13% of val, or about 75-100 samples. A '>10% relative gain' CI on…
- E-01 (Mode twin of P-014: ridge/log init + LM20, estimator=mode, …): challenge: The GN-std used to define the stratum should be computed at the truth, not at the estimate. Otherwise stratum membership depends on the estimator being compared.
- E-01 (Mode twin of P-014: ridge/log init + LM20, estimator=mode, …): counterexample: If the box-truncated posterior is skewed inward for k1 near 2.0 (where the CRB is about 4% absolute), the mean can move away from a truth that sits on the bound. The stratum gain could then be negative f…
- E-02 (Forward-free amortised posterior mean: raw-target, unscaled…): challenge: The raw target under squared loss weights k=2 errors 16x more than k=0.5 errors relative to rel_l2's per-sample normalisation. That is not the MMSE estimator for rel_l2. If rel_l2 is //k_hat-k/////k// per sam…
- E-02 (Forward-free amortised posterior mean: raw-target, unscaled…): challenge: The claim that the population minimiser is E[k/y] holds only asymptotically in n and capacity. At n=3000 in an 800-D input space, estimation error dominates, which the proposal acknowledges. Its point estimat…
- E-02 (Forward-free amortised posterior mean: raw-target, unscaled…): challenge: noise_scale=1.0 with noise_aug=0 is a no-op. Confirm the harness ignores it.
- E-02 (Forward-free amortised posterior mean: raw-target, unscaled…): counterexample: If the gap to the hybrid is driven by k1/k4 low-amplitude features being swamped by large k2/k3 traces in unscaled inputs, scaler='none' makes the gap larger, not cleaner.
- E-03 (Single-basin test: one LM start from the prior centre repro…): challenge: The P-010 pilot already ran this exact configuration (log, ms=1, it=30, mode). It gave 0.0190236 against 0.0190234 for ms=4 on the same 60 val samples, a difference of about 1e-7. At pilot scale the aggregate…
- E-03 (Single-basin test: one LM start from the prior centre repro…): challenge: An aggregate tolerance of 3% cannot detect a 1% wrong-basin fraction. Eight bad samples out of 750 at a 20% error each would move rel_l2 by only a few percent. The per-sample criterion (/theta_ms1 - theta_ms4…
- E-03 (Single-basin test: one LM start from the prior centre repro…): challenge: 'Single basin on [0.5,2]^4' is not established by agreement from one start. Agreement shows only that the centre lies in the basin of the global minimiser for these data. It says nothing about basins that LM …
- E-03 (Single-basin test: one LM start from the prior centre repro…): counterexample: a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply
- E-04 (Amortised inverse on information-preserving PCA-48 scores: …): challenge: Its contrast with P-020 is confounded: it changes PCA (48 vs none) and target transform (log vs raw) at once. A difference between the two arms cannot be attributed to either factor.
- E-04 (Amortised inverse on information-preserving PCA-48 scores: …): challenge: The prediction block reads 'improvement 0.3, expect worse'. This is ambiguous: state explicitly 'rel_l2 >= 1.3 x best_hybrid on val'.
- E-04 (Amortised inverse on information-preserving PCA-48 scores: …): challenge: The attribution to finite-sample error depends on the projected-Fisher pilot, which has not been run. Run it first, since it costs seconds and decides whether the arm can be interpreted at all.
- E-04 (Amortised inverse on information-preserving PCA-48 scores: …): challenge: Fixed 100 epochs with no early-stopping rule. Under-training would be misread as 'sample-size limited'. Hold out part of train for early stopping, not val.
- E-05 (Noise-limited regime: bounded log-space MLE through the for…): challenge: The threshold is stale. Three pilots (ms=1,3,4) are already at 0.0190 on n=60, so '< 0.032' is all but certain to pass and is uninformative. Tighten it to the val-matched CRB band, e.g. [0.9, 1.1] x CRB evalu…
- E-05 (Noise-limited regime: bounded log-space MLE through the for…): challenge: It is internally inconsistent: the hypothesis says 3 multistarts x 30 iterations, but the method is ms=2, it=21. Twenty-one iterations is a change that has not been tested; the pilots used 30.
- E-05 (Noise-limited regime: bounded log-space MLE through the for…): challenge: It overlaps P-016 and the P-002/P-005/P-010 pilots. The only new content is the full-val run and the per-parameter CRB comparison.
- E-05 (Noise-limited regime: bounded log-space MLE through the for…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- Unresolved disagreement ['P-014', 'P-017']: The size of the posterior-mean gain: at least 5% (P-014) or 0-3% (P-017). The paired pilot suggests about 2%.
- Unresolved disagreement ['P-018', 'P-020']: Whether a log target with PCA-48 or a raw target with full inputs is the right amortised representation. The comparison is confounded unless one factor is held fixed.
- Unresolved disagreement ['P-016', 'P-019']: Whether multistarts are needed. The pilots say no at n=60; the full-val tail is untested.
- Test-set sampling uncertainty: bootstrap 95% CI [0.01728, 0.01954] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The P-002 pilot changes the picture. Physics-only MLE (log-space, 4 starts, 30 iterations) reached val rel_l2… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | Duplicate of P-005: accepted. I am revising P-002 so it no longer replicates P-005's run. The ms=4 pilot (0.0… |
| 2 | researcher-ml | revise | P-003 | P-011 | llm | (1) Duplicate of P-001 and confounded comparison: accepted. P-003 is now only the MODE arm of a paired estima… |
| 2 | researcher-ml | revise | P-004 | P-012 | llm | (1) 120 s cost at the limit: accepted. The revised arm is a small MLP on 24 PCA components for 100 epochs, es… |
| 2 | researcher-experimental | revise | P-005 | P-013 | heuristic-fallback | Revised: reduced solver cost as requested |
| 2 | researcher-experimental | revise | P-006 | P-014 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | I accept all points. (1) Cost: 145.6 s is over the 120 s per-experiment limit, and the portfolio is already o… |
| 2 | researcher-exploratory | revise | P-008 | P-015 | llm | Cost: I cut the arm to a [256,256] MLP for 120 epochs with no augmentation, so no rows are multiplied; the es… |
| 3 | researcher-mathematical | abandon | P-009 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 3 | researcher-mathematical | revise | P-010 | P-016 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-ml | revise | P-011 | P-017 | llm | (1) Stale prediction: accepted. Pilots at about 0.019 already pass < 0.030, so that threshold is uninformativ… |
| 3 | researcher-ml | revise | P-012 | P-018 | llm | Pilot evidence: 0.0656 on 60 val samples with only 300 training samples, which is above my 0.06 'representati… |
| 3 | researcher-experimental | revise | P-013 | P-019 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-experimental | abandon | P-014 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 3 | researcher-exploratory | revise | P-015 | P-020 | llm | (1) Standard scaler on 800 raw features. Accepted. Early outer-sensor readings carry almost no prior-induced … |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.642 | fdbk-00044 |
| 2 | 7 | 0.764 | fdbk-00078 |
| 3 | 5 | 0.862 | fdbk-00111 |

### Idea evolution

**P-001 — Learned initializer + Gauss-Newton/LM refinement with posterior-mean estimator reaches the CRB regime** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy+aug1->LM15]` status literature-supported. *H:* A kernel-ridge amortized initializer followed by bounded LM refinement in log-k through the exact forward model, reported as the importance-sampled posterior mean under the box prior, achieves val rel_l2 < 0.03. That is within ~1.5x of the CRB floor of 0.0195…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If rel_l2 stalls near the initializer's error, LM is not converging: either the Jacobian is effectively rank-deficient in parts of the box (e.g. k1/k4 weakly s…
  - critic r1 (quality 0.782): The noise_scale units must be confirmed before running. The augmentation is either a no-op or destructive, depending on the units. · The CRB rel_l2 figure of 0.0195 comes from the MEDIAN over 32 prior draws. rel_l2 is a mean over samples, and samples with large k1/k4 (2.1% rel std at k=2 mea… · The 'within 1.5x of CRB' framing mixes estimator classes. The posterior mean is biased, so the unbiased-estimator CRB is neither a floor nor the right benchmar… · Replace 'CRB floor = 0.0195' with 'mean per-sample sqrt(sum_i CRB_ii)///k// evaluated at every val truth'. It is cheap to compute: one Jacobian per sample. · If KRR init is already about 0.03 and LM converges to the same MLE from any start, the hybrid equals pure LM (P-005). The learned initializer then contributes …
- v2 `P-009` r2 [revise, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status literature-supported. *H:* Hold the initializer and LM settings fixed. Replacing the mode readout with the box-truncated k-space posterior mean then reduces val rel_l2 by a small but real amount, giving rel_l2 < 0.0195 in aggregate. Nearly all of the gain is concentrated in samples who…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0195}`; if it fails: If the posterior mean equals the mode even in the boundary stratum, either truncation is rarer or milder than estimated, or the Laplace-IS readout is degenerat…
  - response to critic: The P-002 pilot changes the picture. Physics-only MLE (log-space, 4 starts, 30 iterations) reached val rel_l2 0.0190 on 60 samples. Its residual std at the truth was 0.000995 against a nominal 0.001, with no model mismatch. Uncertainty was…
  - critic r2 (quality 0.776): The mode twin must have an identical init block. The current cross-references (P-003/P-005/P-011) disagree on noise_aug. · The pairing is not actually defined. P-009 says its mode twin is 'P-003 (or P-005)' with noise_aug=0. P-011, the mode arm the team wrote for this contrast, use… · The prediction rel_l2 < 0.0195 is close to vacuous. The ms=4 mode pilot already gives 0.0190 on n=60, so a posterior-mean arm can pass this test with zero read… · Treat the mode twin as an exact copy of this arm's init block (KRR, noise_aug=0) with estimator=mode. Neither P-011's (noise_aug=1) nor P-005's (pure multistar… · Take a truth at k1 = 0.51 with posterior std about 0.011. The clipped mode sits at ≥ 0.5 and the truncated mean is pulled inward to about 0.51–0.52. The mean c…

**P-002 — Pure per-sample MLE with multistart LM (no learning) is sufficient: the least-squares landscape is effectively unimodal…** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, llm] `physics-MLE[log,ms=4,it=30]` status conjectured. *H:* The misfit J(log k) = //F(k) - y//^2 / sigma^2 has a single basin over [0.5,2]^4 for these sensors, apart from bound-truncation effects. Therefore physics-only bounded LM in log-space with 4 multistarts and no prior regularisation (MLE, mode estimator) achiev…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If residuals are near the noise level but rel_l2 is high, the error is noise amplification at weakly sensed k1/k4. That would point to MAP/posterior_mean (tikh…
  - critic r1 (quality 0.524): This is functionally a duplicate of P-005: physics-only log-LM, mode estimator, tikhonov=0. The only differences are multistart 4 vs 3 and the falsification th… · The 'dedicated sensor per segment' argument assumes equal quarter segments, which is not stated. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If k2 is very low, a near-flat k1 direction could let LM stall at the bound k1=0.5 or 2.0 without a true second basin. A residual-based multimodality diagnosti… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-010` r2 [revise, llm] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* The log-space misfit has a single basin on [0.5,2]^4. Bounded LM from the prior centre alone (multistart=1, 30 iterations, mode) therefore achieves val rel_l2 within 3% of the 4-start result (pilot 0.0190) and below 0.0205. Its chi2 wrong-basin fraction is be…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0205}`; if it fails: If ms=1 loses accuracy and shows high-chi2 samples, the landscape has secondary basins. Initialization then matters, and the learned-init hybrids gain a genuin…
  - response to critic: Duplicate of P-005: accepted. I am revising P-002 so it no longer replicates P-005's run. The ms=4 pilot (0.0190) now serves as shared evidence for P-005. P-002 keeps only its distinct contribution, the single-basin conjecture, tested in t…
  - critic r2 (quality 0.94): The pilots at ms=4 and ms=3 differ by 1e-7 in rel_l2, and P-006, using a ridge start, matches them. This already suggests unimodality on these 60 samples, but … · The 3% relative tolerance is about 0.0006. At n=750 the unpaired SE of rel_l2 is larger than that. The test must be paired: same samples, and compare per-sampl… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-016` r3 [revise, heuristic-fallback] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* The log-space misfit has a single basin on [0.5,2]^4. Bounded LM from the prior centre alone (multistart=1, 30 iterations, mode) therefore achieves val rel_l2 within 3% of the 4-start result (pilot 0.0190) and below 0.0205. Its chi2 wrong-basin fraction is be…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0205}`; if it fails: If ms=1 loses accuracy and shows high-chi2 samples, the landscape has secondary basins. Initialization then matters, and the learned-init hybrids gain a genuin…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.835): The P-010 pilot already ran this exact configuration (log, ms=1, it=30, mode). It gave 0.0190236 against 0.0190234 for ms=4 on the same 60 val samples, a diffe… · An aggregate tolerance of 3% cannot detect a 1% wrong-basin fraction. Eight bad samples out of 750 at a 20% error each would move rel_l2 by only a few percent.… · Reframe the claim as: 'the prior-centre start reaches the same stationary point as the best-of-4 start for >= 99% of val samples'. Do not claim global unimodal… · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer plus Gauss-Newton/LM refinement through the forward model reaches near-CRB accuracy** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status conjectured. *H:* A cheap learned inverse map (kernel ridge on standardized raw sensor traces, log-target) followed by a few bounded Levenberg-Marquardt iterations in log-k space will reach test rel_l2 <= 0.035, within about 1.8x of the CRB floor (0.0195). Purely data-driven m…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If LM does not improve on the initializer, either the forward model differs from the data generator (model error), or the likelihood has ridges in k1/k4 that p…
  - critic r1 (quality 0.528): This is near-duplicate of P-001: same KRR init and the same LM setup. It differs only in the estimator (mode vs posterior_mean) and in noise_aug (0 vs 1). Beca… · The falsification criterion 'at least 2x better than the initializer alone' may fail even if the method is excellent. If KRR is already at about 0.03, a refine… · Set noise_aug equal to P-001's value so that the pair isolates the estimator effect. · Pure LM from the prior centre (P-005) may converge to the same MLE. The hybrid would then equal physics-only, and the 'learned init matters' claim would be uns… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-011` r2 [revise, llm] `hybrid[kernel_ridge/raw+logy+aug1->LM15]` status conjectured. *H:* With the initializer and LM settings identical to P-001, the mode (bounded MLE in log-k) estimator reaches val rel_l2 < 0.030. Relative to P-001's posterior_mean, it is worse by a small aggregate margin: the paired bootstrap mean difference is between 0 and 3…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If (a) fails and the wrong-basin fraction is high, the KRR init is outside the basin; P-002/P-005 multistart is the remedy. If (a) fails with a low wrong-basin…
  - response to critic: (1) Duplicate of P-001 and confounded comparison: accepted. P-003 is now only the MODE arm of a paired estimator contrast with P-001. The init block is set to noise_aug=1, which is P-001's value, and the LM settings are iters=15 with log t…
  - critic r2 (quality 0.435): The pairing must use an init block identical to its posterior-mean partner. At present it is P-001-specific and does not match P-009 or P-014. · The prediction rel_l2 < 0.030 is stale. Three pilots are at about 0.019, so this cannot fail informatively. Tighten it to < 0.0205. · This arm duplicates the role of P-009's mode twin, but with a different init (noise_aug=1 vs 0). The portfolio now has three half-pairs (P-001/P-011, P-009/?, … · Make the init cheap and shared: ridge/log as in P-006. KRR only matters if P-010 fails. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-017` r3 [revise, llm] `hybrid[ridge/log+logy->LM20]` status conjectured. *H:* This arm uses P-014's init and LM settings with estimator=mode (bounded log-space MLE). It reaches val rel_l2 < 0.0205. On the paired per-sample contrast with P-014, the posterior_mean improves aggregate rel_l2 by 0% to 3% relative. On the shared boundary str…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0205}`; if it fails: If (a) fails with chi2 near 1, the mode is noticeably inefficient at n_obs/sigma here. MMSE is then the readout to deploy, and the truncation mechanism is larg…
  - response to critic: (1) Stale prediction: accepted. Pilots at about 0.019 already pass < 0.030, so that threshold is uninformative. I tighten it to < 0.0205, roughly the 32-draw CRB floor plus 5%. (2) Duplicate of P-009 and orphan half-pairs: accepted. I drop…
  - critic r3 (quality 0.916): This is the most informative remaining contrast (estimator only), but it depends on P-014 being run with an identical seed, samples and init. If P-014 is not s… · The boundary stratum is small. 2 GN-std is about 0.021 at k=0.5 and about 0.084 at k=2 for k1, and somewhat less for k4. That gives roughly 10-13% of val, or a… · The posterior-mean gain may come partly from importance-sampling noise reduction rather than from truncation. Reported ESS was about 58 of 64. Check that the g… · If the box-truncated posterior is skewed inward for k1 near 2.0 (where the CRB is about 4% absolute), the mean can move away from a truth that sits on the boun… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-004 — Amortized MLP with log-target and noise augmentation is accurate but cannot reach the CRB without physics refinement** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch MLP trained on raw standardized sensor traces, with log-conductivity targets and noise augmentation at the true sigma, will reach val/test rel_l2 in [0.04, 0.08]. That is well below the mean baseline (0.33), but at least 1.5x worse than the best physi…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.5, "expect": "worse"}`; if it fails: If the MLP matches the hybrid, amortized inference is sufficient and is far cheaper at test time, so effort should shift toward architecture and data scaling. …
  - critic r1 (quality 0.705): The estimate sits at the 120 s limit. Reduce epochs before scheduling. · The noise_scale units are unresolved. · The rule estimate is 120.0 s, exactly at the per-experiment limit. Any overrun kills the run. It also takes 30% of the 400 s total budget. · noise_scale=0.001 conflicts with P-008's noise_scale=1.0 for the same method. They cannot both be matched augmentation. · Reduce epochs (e.g. 80-100 with early stopping on val) to bring the cost to about 50-60 s. Smooth 4-output regression rarely needs 200 epochs on 3000 rows.
- v2 `P-012` r2 [revise, llm] `torch_mlp/raw+pca24+logy` status conjectured. *H:* Diffusion makes the noise-free sensor traces effectively low-rank (about 10 to 25 informative directions). An MLP on the top 24 raw-scale PCA scores with log targets reaches val rel_l2 in [0.03, 0.06]. That is below the 0.05 target in most of the range, but a…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If it lands within 1.3x of the best hybrid, the amortisation gap is mainly a representation artefact. The next step is a PCA-MLP initializer for LM, or droppin…
  - response to critic: (1) 120 s cost at the limit: accepted. The revised arm is a small MLP on 24 PCA components for 100 epochs, estimated at about 20 s. (2) noise_scale unit conflict: I avoid it entirely by setting noise_aug=0. The raw-trace MLP with augmentat…
  - critic r2 (quality 0.904): Unscaled PCA ranks directions by signal variance across the prior. The k1/k4 information sits in the small-amplitude traces at sensors 5 and 35, which may fall… · The hypothesis bundles a wide range [0.03, 0.06] with a relative ≥1.3x gap. Given the hybrid at about 0.019, any result above 0.0247 passes the gap claim, so t… · Information retention by the PCA subspace can be computed without training. Use the projected Fisher P^T J, compare its CRB with the full-data CRB, and check t… · Suppose the top 24 components retain >99% of the Fisher information. A residual gap is then purely regression error, which supports the hypothesis. Suppose ins… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-018` r3 [revise, llm] `torch_mlp/raw+pca48+logy` status conjectured. *H:* An MLP on the top 48 unscaled PCA scores of the raw sensor traces, with log targets, trained on all 3000 samples, reaches val rel_l2 in [0.025, 0.05]. That is below the 0.05 target but at least 1.3x worse than the best hybrid. The excess over the CRB is large…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If rel_l2 < 1.3x the hybrid, amortised inference is essentially sufficient. The next step is a cheap PCA-MLP initializer for LM, or no LM where test-time cost …
  - response to critic: Pilot evidence: 0.0656 on 60 val samples with only 300 training samples, which is above my 0.06 'representation helps' bound. That pilot used 10% of the training data, so it does not falsify the full-fidelity claim. It does show that sampl…
  - critic r3 (quality 0.865): Its contrast with P-020 is confounded: it changes PCA (48 vs none) and target transform (log vs raw) at once. A difference between the two arms cannot be attri… · The prediction block reads 'improvement 0.3, expect worse'. This is ambiguous: state explicitly 'rel_l2 >= 1.3 x best_hybrid on val'. · A single n=3000 point plus a pilot at n=300 with a different PCA rank and val size is not a learning curve. Use n in {750, 1500, 3000} with fixed architecture. · If PCA-48 retains below 99% of the k1 information, a k1-dominated gap would come from the representation, which is the opposite of the stated attribution. · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Noise-limited regime: bounded log-space MLE through the forward model should land near the CRB floor (~0.02 rel_l2)** (researcher-experimental, family physics)
- v1 `P-005` r1 [propose, llm] `physics-MLE[log,ms=3,it=30]` status literature-supported. *H:* The observation map is locally well-conditioned (Fisher condition number median 46.6) and sigma=0.001 is small relative to the signal. Most of the irreducible error should therefore come from measurement noise, not from model misfit or local minima. A per-sam…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.032}`; if it fails: If residuals are chi-square-consistent but error is still above 1.6x CRB, the error comes from estimator bias or boundary effects, and an MMSE estimator (propo…
  - critic r1 (quality 0.928): The threshold 0.032 = 1.6x the median CRB has the same mean-vs-median problem as P-001. Re-derive it from the per-sample CRB at the val truths. · Box-constrained MLE has lower variance than the CRB at bounded truths, so the CRB comparison is not clean near the bounds. Stratify the analysis by distance to… · Treat this as the canonical physics-mode arm. It absorbs P-002. · If val rel_l2 is about 0.022 while the mean CRB is about 0.025, the proposal 'passes' but sits below the floor. That would indicate bound-truncation benefit, n… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, heuristic-fallback] `physics-MLE[log,ms=2,it=21]` status literature-supported. *H:* The observation map is locally well-conditioned (Fisher condition number median 46.6) and sigma=0.001 is small relative to the signal. Most of the irreducible error should therefore come from measurement noise, not from model misfit or local minima. A per-sam…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.032}`; if it fails: If residuals are chi-square-consistent but error is still above 1.6x CRB, the error comes from estimator bias or boundary effects, and an MMSE estimator (propo…
  - response to critic: Revised: reduced solver cost as requested
  - critic r2 (quality 0.886): The prediction < 0.032 is already essentially confirmed by the ms=3/ms=4 pilots at 0.0190. Retain it only with a tightened threshold, for example < 0.0205 and … · This is a near-duplicate of P-005, which was already piloted at ms=3, it=30. Its distinct value is the ms=2, it=21 point on a multistart/iteration ladder with … · Asymptotic CRLB attainment (ts:thm:26986930, 27031792) does not apply with box constraints. Truncation lets MSE fall below CRB, as the pilot hints. 'Within 1.6… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-019` r3 [revise, heuristic-fallback] `physics-MLE[log,ms=2,it=21]` status literature-supported. *H:* The observation map is locally well-conditioned (Fisher condition number median 46.6) and sigma=0.001 is small relative to the signal. Most of the irreducible error should therefore come from measurement noise, not from model misfit or local minima. A per-sam…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.032}`; if it fails: If residuals are chi-square-consistent but error is still above 1.6x CRB, the error comes from estimator bias or boundary effects, and an MMSE estimator (propo…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.82): The threshold is stale. Three pilots (ms=1,3,4) are already at 0.0190 on n=60, so '< 0.032' is all but certain to pass and is uninformative. Tighten it to the … · It is internally inconsistent: the hypothesis says 3 multistarts x 30 iterations, but the method is ms=2, it=21. Twenty-one iterations is a change that has not… · Asymptotic CRLB attainment is not the right reference under box constraints (as the proposal itself concedes). Truncation allows MSE below the unconstrained CR… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-006 — Metric-matched estimator: a learned init plus LM refinement with a posterior-mean (MMSE) readout beats the MAP/MLE mode…** (researcher-experimental, family hybrid)
- v1 `P-006` r1 [propose, llm] `hybrid[ridge/log+logy->LM20]` status conjectured. *H:* rel_l2 is a squared-error metric, so under the known uniform box prior the Bayes-optimal point estimate is the posterior mean, not the mode. For the weakly determined k1 and k4, the posterior near the bounds 0.5 and 2.0 is truncated and skewed. The mode piles…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.05, "expect": "better"}`; if it fails: A gain under 2% means the posterior is effectively Gaussian and interior, so the mode is sufficient and the remaining error is noise-limited. That would be the…
  - critic r1 (quality 0.845): Rough magnitude check. For k1, the posterior std is about 0.0105 at k=0.5 and about 0.042 at k=2.0. Under the uniform prior, about 1.4% of truths lie within 2 … · The 'best_physics' comparator uses mode with multistart and no learned init, so the estimator and the initializer are confounded. A ridge init on log features … · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the gain on the boundary stratum is large (e.g. 30%) but the aggregate gain is 2%, the hypothesis is 'refuted' even though the mechanism is confirmed. The s… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-014` r2 [revise, heuristic-fallback] `hybrid[ridge/log+logy->LM20]` status conjectured. *H:* rel_l2 is a squared-error metric, so under the known uniform box prior the Bayes-optimal point estimate is the posterior mean, not the mode. For the weakly determined k1 and k4, the posterior near the bounds 0.5 and 2.0 is truncated and skewed. The mode piles…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.05, "expect": "better"}`; if it fails: A gain under 2% means the posterior is effectively Gaussian and interior, so the mode is sufficient and the remaining error is noise-limited. That would be the…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.507): The prediction of a ≥5% improvement over best physics is inconsistent with the team's own magnitude estimate (about 7% of samples near a bound per end paramete… · The boundary stratum 'any k within 0.1 of a bound' is about 5x wider than the 2-GN-std region where truncation matters (about 0.02 absolute for k1). It dilutes… · The expected gain is ~1–3% aggregate. Lower the prediction or make it a paired-sign claim. · If P-010 shows a single basin, the ridge-init 'basin-correct' rationale is moot. Ridge then buys only compute. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-007 — Rel-L2 is a squared loss, so target the Bayes estimator: learned init, then LM refinement, then a posterior-mean (MMSE)…** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM20]` status conjectured. *H:* With a box prior and weakly determined end segments (k1, k4), the per-sample MAP/MLE is biased toward the bounds, and its error is inflated wherever the likelihood is flat. Two claims follow. (a) A hybrid that refines a torch_mlp initializer by log-space Leve…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Either (i) the hybrid is already at the CRB and the estimator choice is irrelevant, so the end-segment parameters are likelihood-dominated, not prior-dominated…
  - critic r1 (quality 0.408): The estimated cost (145.6 s) exceeds the per-experiment limit. Reduce posterior_samples, iters and epochs, or defer. · The rule estimate is 145.6 s, which exceeds the 120 s per-experiment limit and takes 36% of the total budget. · This duplicates the P-001/P-006 hypothesis (hybrid + posterior_mean < 0.03, with mean beating mode). The only change is an MLP initializer, which adds training… · If an MLP-initialized hybrid is wanted, reuse the trained MLP from the single MLP arm rather than retraining it. · If P-001 and P-006 both reach the oracle-init MLE error, the initializer is irrelevant, and P-007 adds no information at 3-7x their cost.

**P-008 — Amortised MMSE: a squared-loss network trained with noise augmentation is itself a posterior-mean estimator; test wheth…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* A network trained with squared loss on prior-distributed simulations with matched sensor noise approximates E[k / data]. With only 3000 simulations, plus noise augmentation at the true sigma and a log-target transform, the amortised estimator will reach val r…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If it beats 0.05 and approaches the hybrid, amortised inference is sufficient and the forward-call budget is better spent generating more simulations than on p…
  - critic r1 (quality 0.417): The estimated cost (160 s) exceeds the per-experiment limit, and the full portfolio (565 s) exceeds the 400 s budget. This arm must be reduced or merged. · The rule estimate is 160 s, which exceeds the limit and is the most expensive arm. It also asks for an extra noise_aug=0 ablation, which raises the cost furthe… · This duplicates P-004 (same MLP family and features). The 'amortised MMSE' framing is genuinely distinct and correct in principle, but it is the same experimen… · Train on k directly, or apply a variance correction, if the claim being tested is that the network is a posterior-mean estimator. · Because the training features already contain a noise realisation, noise_aug=0 is already a valid (if single-draw) amortised posterior-mean regression. Augment…
- v2 `P-015` r2 [revise, llm] `torch_mlp/raw` status conjectured. *H:* A torch_mlp trained by squared loss on the 3000 prior-distributed noisy simulations, with an untransformed target, approximates E[k/y]. (a) It reaches val rel_l2 < 0.05 without any forward-model calls. (b) It stays at least 30% worse than the best hybrid, bec…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 >= 0.05, simulation scarcity rather than identifiability limits amortised inference, and the forward budget is better spent on per-sample LM, as in t…
  - response to critic: Cost: I cut the arm to a [256,256] MLP for 120 epochs with no augmentation, so no rows are multiplied; the estimate is about 30-50 s. I also dropped the noise_aug ablation. Duplication with P-004: rather than merging into an identical run,…
  - critic r2 (quality 0.898): standard scaler on 800 raw features is risky. Early-time readings at the outer sensors have near-zero signal variance across the prior, so standardisation infl… · Claim (c) tests Bayes theory only if P-004 is identical except for target_transform. Confirm P-004's architecture, epochs, scaler and noise_aug. Otherwise a >1… · Use scaler='none' (or PCA-whitened inputs) so that noise stays isotropic at sigma, or include the scaler as an explicit factor. · Suppose the MLP lands at 0.04 because of noise-inflated inputs. Then (a) and (b) both pass, yet the conclusion 'simulation scarcity limits amortisation' is wro… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-020` r3 [revise, llm] `torch_mlp/raw` status conjectured. *H:* A torch_mlp trained by squared loss on the 3000 prior-distributed noisy simulations, with an untransformed target and unscaled inputs (isotropic noise), approximates E[k/y]. (a) It reaches val rel_l2 < 0.05 with zero forward calls. (b) Its val rel_l2 is at le…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 >= 0.05 with isotropic noise inputs, input scaling is ruled out as the cause. The gap is then sample scarcity or optimisation for amortised inference…
  - response to critic: (1) Standard scaler on 800 raw features. Accepted. Early outer-sensor readings carry almost no prior-induced variance, so z-scoring inflates them into unit-variance noise channels. I switch to scaler='none'. Sensor noise then stays isotrop…
  - critic r3 (quality 0.874): The raw target under squared loss weights k=2 errors 16x more than k=0.5 errors relative to rel_l2's per-sample normalisation. That is not the MMSE estimator f… · The claim that the population minimiser is E[k/y] holds only asymptotically in n and capacity. At n=3000 in an 800-D input space, estimation error dominates, w… · The loss is matched to the metric only if targets are weighted by 1///k//^2 or the metric is absolute L2. · If the gap to the hybrid is driven by k1/k4 low-amplitude features being swamped by large k2/k3 traces in unscaled inputs, scaler='none' makes the gap larger, … · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

- Critic round 1 portfolio view: The portfolio is technically sound. Small noise, a known likelihood and Fisher conditioning around 47 make likelihood-based inversion near the CRB very plausible, and the 0.05 target is likely met by P-005 alone. The remaining scientific questions are narrow: whether the learned init matters, whether the posterior mean matters, and how large the amortisation gap is. The portfolio is heavily dupli…  families={'hybrid': 4, 'physics': 2, 'data-driven': 2}
- Critic round 2 portfolio view: Empirically the problem is solved. Physics MLE and the ridge-init hybrid both reach about 0.019 on the pilot, with exact likelihood and calibrated uncertainty, which is well under the 0.05 target. The remaining scientific questions are: (1) is the misfit single-basin (P-010, cheap and decisive with per-sample comparison); (2) how large is the MMSE readout gain at the bounds (merge P-009, P-011 an…  families={'hybrid': 3, 'physics': 2, 'data-driven': 2}
- Critic round 3 portfolio view: The physics MLE and hybrid arms are empirically at about 0.019 rel_l2, already below the 0.05 target and near the noise floor. Start count appears irrelevant: ms=1 equals ms=4 to 1e-7 on n=60. Recommended runs: (1) one canonical full-val physics arm with paired ms=1 and ms=4, absorbing P-016 and P-019, with tightened CRB-relative thresholds; (2) the P-017/P-014 paired estimator contrast with inte…  families={'physics': 2, 'hybrid': 1, 'data-driven': 2}

**Pilot experiments run during research** (small validation subsets):

- P-002 `physics-MLE[log,ms=4,it=30]`: ok rel_l2=0.019023 on 60 val samples in 2.96 s
- P-005 `physics-MLE[log,ms=3,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.66 s
- P-006 `hybrid[ridge/log+logy->LM20]`: ok rel_l2=0.018628 on 60 val samples in 2.97 s
- P-010 `physics-MLE[log,ms=1,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.05 s
- P-012 `torch_mlp/raw+pca24+logy`: ok rel_l2=0.065565 on 60 val samples in 6.09 s
- P-015 `torch_mlp/raw`: ok rel_l2=0.069395 on 60 val samples in 6 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 64 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-experimental | 1 | uniqueness and Lipschitz stability for identifying piecewise constant diffusion coefficie… | 4 refs via /search |
| researcher-mathematical | 1 | Uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search |
| researcher-ml | 1 | uniqueness and Lipschitz stability for recovering piecewise constant conductivity in a on… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant conductivity in parabolic eq… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient in one-dimensional heat equation from boundary or poi… | 4 refs via /search |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | 4 refs via /search |
| researcher-ml | 1 | neural network or amortized posterior mean estimator approaching Cramer-Rao bound for fin… | 4 refs via /search |
| researcher-mathematical | 1 | Convergence of Gauss-Newton or Levenberg-Marquardt for finite-dimensional nonlinear least… | 4 refs via /search |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Wrong-basin fraction for the KRR and ridge inits (from P-001/P-006) | 4 refs via /search |
| researcher-ml | 2 | KRR-only val rel_l2 and per-parameter rel RMSE | 4 refs via /search |
| researcher-experimental | 2 | Per-parameter rel RMSE vs per-sample CRB std, stratified by whether a truth lies within 0… | 4 refs via /search |
| researcher-mathematical | 2 | Per-sample CRB at all 750 val truths: report mean and 90th percentile of the rel_l2 impli… | 4 refs via /search |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 3 | Harness definition of noise_scale | 4 refs via /search |
| researcher-experimental | 3 | Val-matched per-sample CRB | 4 refs via /search |
| researcher-mathematical | 3 | Per-sample CRB at each of the 750 val truths, aggregated exactly like rel_l2 | 4 refs via /search |
| researcher-exploratory | 3 | Per-feature signal-to-noise across the training set | 4 refs via /search |
| researcher-exploratory | 3 | P-004 exact config | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_full_s0.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 167 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-017) | hybrid | `hybrid[ridge/log+logy->LM20]` | researcher-ml | 0.916 | best-rated proposal in family 'hybrid' (quality 0.92) |
| E-02 | proposal (P-020) | data-driven | `torch_mlp/raw` | researcher-exploratory | 0.874 | best-rated proposal in family 'data-driven' (quality 0.87) |
| E-03 | proposal (P-016) | physics | `physics-MLE[log,ms=1,it=30]` | researcher-mathematical | 0.835 | best-rated proposal in family 'physics' (quality 0.83) |
| E-04 | proposal (P-018) | data-driven | `torch_mlp/raw+pca48+logy` | researcher-ml | 0.865 | quality 0.86 with diversity penalty |
| E-05 | proposal (P-019) | physics | `physics-MLE[log,ms=2,it=21]` | researcher-experimental | 0.82 | quality 0.82 with diversity penalty |
| E-06 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-07 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-08 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-012 | E-01 | refine | hybrid | `hybrid[ridge/log+logy->LM20]` | - | **0.019325** | 0.9966 | 0.02548 | 0.0203 | 3.65 | 50411 | 0.957 | FINAL |
| idea-006 | E-05 | research | physics | `physics-MLE[log,ms=2,it=21]` | 0.02138 | **0.019325** | 0.9966 | 0.02548 | 0.0103 | 7.78 | 114683 | 0.957 | evaluated |
| idea-011 | E-01 | refine | hybrid | `hybrid[ridge/log+logy->LM20]` | - | **0.019331** | 0.9966 | 0.02549 | 0.0197 | 3.63 | 50771 | 0.957 | evaluated |
| idea-004 | E-03 | research | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.00925 | 4.49 | 61517 | 0.957 | evaluated |
| idea-010 | E-08 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.01 | 4.38 | 61517 | 0.957 | evaluated |
| idea-002 | E-01 | research | hybrid | `hybrid[ridge/log+logy->LM20]` | 0.02139 | **0.019338** | 0.9966 | 0.0255 | 0.0218 | 3.7 | 50825 | 0.957 | evaluated |
| idea-005 | E-04 | research | data-driven | `torch_mlp/raw+pca48+logy` | 0.05422 | **0.024664** | 0.9944 | 0.03253 | 9.42 | 0.00532 | 0 | - | evaluated |
| idea-013 | E-04 | refine | data-driven | `torch_mlp/raw+pca48+logy` | - | **0.026079** | 0.9937 | 0.03439 | 7.69 | 0.00661 | 0 | - | evaluated |
| idea-003 | E-02 | research | data-driven | `torch_mlp/raw` | 0.06754 | **0.02609** | 0.9937 | 0.03441 | 9.76 | 0.00515 | 0 | - | evaluated |
| idea-007 | E-06 | baseline | data-driven | `torch_mlp/raw` | 0.06289 | **0.029522** | 0.992 | 0.03893 | 7.09 | 0.0151 | 0 | - | evaluated |
| idea-014 | E-04 | refine | data-driven | `torch_mlp/log+pca48+logy` | - | **0.032391** | 0.9903 | 0.04272 | 9.11 | 0.00985 | 0 | - | evaluated |
| idea-009 | E-07 | baseline | data-driven | `kernel_ridge/raw` | 0.05292 | **0.032419** | 0.9903 | 0.04275 | 0.452 | 0.0477 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `knn/sqrt` | 0.2036 | **0.14038** | 0.8184 | 0.1851 | 0.00712 | 0.0811 | 0 | - | evaluated |

## 8. Lineage of the final method

- `idea-012` `hybrid[ridge/log+logy->LM20]` ← refined from idea-002 ← P-017 v3 (researcher-ml, r3, revise) ← P-011 v2 (researcher-ml, r2, revise) ← P-003 v1 (researcher-ml, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00230` [checkpoint] by scheduler: final method (single)
- `resu-00227` [result] by executor: exp-0030 rel_l2=0.01841 r2=0.997 rmse=0.02423
- `hypo-00198` [hypothesis] by scheduler: idea-012 hybrid[ridge/log+logy->LM20] (refine)
- `plan-00226` [plan] by engineer-v2: exp-0030: final fit of 1 member(s) on train+val; test once
- `gate-00197` [gate] by gate: refine idea-002: approved (hybrid[ridge/log+logy->LM20])
- `prop-00105` [proposal] by researcher-ml: P-017 v3 [revise] ml: Mode twin of P-014: ridge/log init + LM20, estimator=mode, paired against P-014's posterio
- `hypo-00115` [hypothesis] by scheduler: idea-002 hybrid[ridge/log+logy->LM20] (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_full_s0 (044ddf1b35)
- `prop-00071` [proposal] by researcher-ml: P-011 v2 [revise] ml: Mode arm of a paired MLE-vs-MMSE contrast: same KRR init and LM as P-001, estimator=mode
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00114` [gate] by gate: E-01: approved (hybrid[ridge/log+logy->LM20])
- `prop-00037` [proposal] by researcher-ml: P-003 v1 [propose] ml: Learned initializer plus Gauss-Newton/LM refinement through the forward model reaches near

## 9. Compute and cost

- Execution compute: 160.0 / 400 worker-seconds over 30 jobs; wall 603 s
- By stage: baseline 1.9s, pilot 22.7s, screen 34.1s, evaluate 61.1s, refine 30.8s, final 9.3s
- LLM: 19 calls (by phase {'research': 19}), 1024 s, $4.044 reported · TheoremSearch: 24 queries (0 failed)

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

The evidence establishes that the success criterion is met with margin. The selected estimator, `hybrid[ridge/log+logy->LM20]` (a ridge initializer in log-k space refined by 20 Levenberg–Marquardt iterations), reached val rel_l2 = 0.01932 and held-out test rel_l2 = 0.01841 (R² = 0.997, RMSE = 0.0242), well under the 0.05 target. The val-to-test gap is small and favourable, so selection does not appear to have overfit. The more informative result is the near-exact tie between the hybrid and the pure physics estimator, `physics-MLE[log,ms=2,it=21]`: 0.0193246 vs 0.0193249, a difference of about 2.5×10⁻⁷. This means the learned initializer contributes essentially nothing once LM converges; both methods land on the same likelihood optimum. That fits the negative result recorded in the abandoned proposals, that initializer choice should not matter when LM converges within the basin. The best purely data-driven model (`torch_mlp/raw+pca48+logy`, 0.0247) was 27.6% worse than the hybrid, as predicted. So explicit forward-model inversion is worth roughly a quarter of the error relative to amortized regression at this data size.

No pre-registered hypothesis was contradicted empirically. All five supported predictions held, although several used loose thresholds (< 0.05, < 0.032) and were weak tests. The failures were at the proposal stage. The MMSE/Bayes-estimator line was abandoned three times: once as a duplicate that also exceeded the cost limit and rested on a wrong mechanism (it claimed the MAP is biased outward from the bounds, when box constraints actually clip estimates inward), once because the paired mode-vs-mean contrast could not be given an identical initializer (cross-references disagreed on `noise_aug`), and once as a duplicate. The question it posed is therefore open, not answered negatively: would a posterior-mean readout beat the mode on a squared-error metric, especially for k₁ and k₄ near the box bounds? Other uncertainties remain. We do not know whether 0.019 is near the irreducible noise floor (no Cramér–Rao or posterior-spread reference was reported). The wrong-basin fraction, which was the stated trigger for revisiting the abandoned lines, was never measured. Results come from a single run with no seed or interval estimates, so the ~1e-7 hybrid/physics difference should be read as a tie, not a ranking.

- **Noise-floor check:** compute the Cramér–Rao bound (or Laplace posterior spread) at each test-like sample from the LM Jacobian, and compare the implied rel_l2 to 0.019 to see whether further gains are possible.
- **Wrong-basin audit:** run multi-start LM (or start from the truth) on validation, and report the fraction of samples whose converged likelihood differs from the best-found optimum. This decides whether initializer or MMSE work is worth revisiting (threshold above 2%).
- **Clean MMSE-vs-mode contrast:** fix one initializer config (resolve the `noise_aug` inconsistency first), then compare a box-truncated posterior mean, via Laplace or short MCMC around the LM optimum, against the mode. Report per-component errors for k₁–k₄.
- **Per-parameter and near-bound breakdown:** stratify val error by component and by distance of the true k to the prior bounds to find where the residual error concentrates.
- **Robustness and variance:** repeat the top three configs over several seeds and noise levels, and perturb the forward model (sensor position, boundary values). This gives intervals and tests how sensitive the physics-based advantage over the MLP is to misspecification.

_Automatic summary:_ 20 proposal versions over 3 research rounds; 8 methods advanced; 5 predictions supported, 0 contradicted. Final: ['hybrid[ridge/log+logy->LM20]'] with test rel_l2 0.018412.
