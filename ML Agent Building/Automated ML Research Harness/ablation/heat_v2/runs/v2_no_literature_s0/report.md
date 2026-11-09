# Research report (V2): heat_v2_no_literature_s0

Run `heat_v2_no_literature_s0_20261109_111101` · 2026-11-09 11:26:47 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 46.6 (max 1.78e+03); Cramér–Rao relative std per parameter: k1 0.0209, k2 0.0063, k3 0.0051, k4 0.0161; implied rel_l2 floor ≈ **0.01949**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=4,it=40]`
- Validation rel_l2 (selection estimate): **0.018951** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018253**, r2 = **0.99702**, rmse = **0.024017** · bootstrap 95% CI for rel_l2: [0.01721, 0.0194]
- Per-target test rel_l2: k1 0.0266, k2 0.0061, k3 0.0052, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.94× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.11965 (vs 0.018253 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051124 (vs 0.018253 in distribution)
- Uncertainty calibration on test: 69.0% within 1σ, 95.1% within 2σ (ideal 68% / 95%)
- Inference latency: 57.9 ms/sample; fit 0.011 s; exported model reload check: max |Δ| = 0.00686
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s0\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s0")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=4,it=40]` rel_l2=0.018951; **hybrid** `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` rel_l2=0.019082; **data-driven** `kernel_ridge/raw+pca64+logy+aug1` rel_l2=0.026505
- SUPPORTED — E-01 *Sole mode twin of P-001: bounded LM MLE in log-k (ms4/it40), predicted equivalent to the posterior mean overall but wor…*: val rel_l2 = 0.019323 vs predicted < 0.022
- SUPPORTED — E-02 *Basin-threshold stress test: a linear ridge initializer plus a single 15-iteration LM run (mode) is equivalent to multi…*: val rel_l2 = 0.019332 vs predicted < 0.021
- SUPPORTED — E-04 *Init-only isotropy test: KRR on unscaled PCA64 vs P-003's standardised PCA64 initializer (single factor: scaler), no fo…*: val rel_l2 = 0.026505 vs predicted < 0.045
- SUPPORTED — E-05 *MC-noise attribution arm: P-001 settings with 4x importance draws (512), forming a triad with P-013 (mode) and P-001 (1…*: val rel_l2 = 0.018951 vs predicted < 0.022
- SUPPORTED — E-06 *Minimum-iteration hybrid: KRR (unscaled PCA32) init plus only 2 bounded LM steps in log k with box-prior posterior mean…*: val rel_l2 = 0.019117 vs predicted < 0.0205
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01949

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- no citations in proposals

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-03 *Learning-curve test of the amortised MLP using simulated data: identical learner at n = 3000 (control) / 6000 / 12000 v…*: inconclusive — method did not produce a full-fidelity validation result
- P-019 *Matched raw-feature twin of P-016: torch MLP [256,256] on standardised raw traces, no PCA, log targets, 2 seeds* (conjectured): proposed but not executed
- P-025 *Mode twin of P-003: an MC-free reference arm isolating the MMSE-vs-mode effect, with explicit MC-noise accounting* (conjectured): proposed but not executed
- P-026 *Log-temperature features for an amortised MLP: single arm of a merged log/raw pair with P-011, gated on a ridge pilot* (conjectured): proposed but not executed
- P-024 *Importance-sampling noise control: high-ESS posterior-mean twin of P-001 (512 draws) to bound the Monte Carlo component…* (conjectured): proposed but not executed
- P-027 *Travel-time target coordinate: KRR on clipped log traces predicting k^{-1/2} instead of log k* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-mathematical — P-002 *Amortised learned inverse plus short LM refinement: the neural initializer removes basin errors and…*: Redundant and over budget. The pilot (resu-00053: 0.0189 with a 300-sample MLP initializer vs 0.0186 for pure physics) shows the expensive MLP initializer buys no accuracy over cheap initializers or multistart. P-006 (r…

### 3.5 Remaining uncertainty

- E-01 (Sole mode twin of P-001: bounded LM MLE in log-k (ms4/it40)…): challenge: This appears to duplicate P-013 (physics mode, ms4/it40), which P-017, P-023 and P-024 all reference as the mode comparator. If P-013 is still live, this arm is redundant. If P-022 replaces P-013, say so expl…
- E-01 (Sole mode twin of P-001: bounded LM MLE in log-k (ms4/it40)…): challenge: The basin-failure detector 'misfit > 2x median' is far too loose. With N = 800 readings per sample, chi2/N has SD ~ sqrt(2/800) ~ 0.05. A real basin failure could sit at chi2/N = 1.3 and still be missed. Use …
- E-01 (Sole mode twin of P-001: bounded LM MLE in log-k (ms4/it40)…): challenge: Equivalence within ±3% is very likely a priori: the interior difference is about 1e-4 relative plus MC, and only about 7% of coordinates are near bounds. That makes the aggregate test low-information. The str…
- E-01 (Sole mode twin of P-001: bounded LM MLE in log-k (ms4/it40)…): counterexample: A mode projected onto a box face can have smaller error than the posterior mean when the truth is exactly at or near the bound. The sign of the near-bound effect depends on how the truth is distributed i…
- E-02 (Basin-threshold stress test: a linear ridge initializer plu…): challenge: The pilot already gives partial evidence of near-convexity: ms3/it30 and ms4/it40 agree to 1e-6, and learned inits agree with multistart. The informative outcome is therefore the high-contrast tercile. Power …
- E-02 (Basin-threshold stress test: a linear ridge initializer plu…): challenge: The basin-failure threshold 'misfit > 2x median' has the same looseness problem as in P-022. Use a chi2/N tail threshold.
- E-02 (Basin-threshold stress test: a linear ridge initializer plu…): challenge: Ridge on standardised raw features inherits the noise-channel inflation P-021 hypothesises. That is harmless here, because only basin entry matters, but it means the init quality is a lower bound on what a li…
- E-02 (Basin-threshold stress test: a linear ridge initializer plu…): counterexample: A truth with k1 = 0.5 and k2 = 2.0 next to it is a high-contrast corner where the linear init may overshoot outside the box. Clipping the init to the box puts it on a face, and bounded LM can then be slo…
- E-03 (Learning-curve test of the amortised MLP using simulated da…): challenge: context.forward returns noise-free traces. The truncated code must add N(0, sigma^2) to the simulated features, and also to the check set. Otherwise the 6k and 12k training sets are noise-free while val is no…
- E-03 (Learning-curve test of the amortised MLP using simulated da…): challenge: Whitened PCA(48) equalises variance across components, so trailing noise-dominated components are amplified to unit scale. This is the converse of P-021's isotropy argument and may cap the learning curve for …
- E-03 (Learning-curve test of the amortised MLP using simulated da…): challenge: A three-point exponent with early stopping and a single seed is noisy. Seed variance at n=3000 could be about 5-10%, comparable to the 25% gap needed. Add a second seed at least at the endpoints.
- E-03 (Learning-curve test of the amortised MLP using simulated da…): challenge: The rule estimate (30 s) is lower than the stated MLP fit time (40-55 s). Plan for the larger number.
- E-04 (Init-only isotropy test: KRR on unscaled PCA64 vs P-003's s…): challenge: The matched control (P-003's init run alone) is not a separate proposal, so the harness may not run it. Make it an explicit second arm here. Otherwise the test is unfalsifiable.
- E-04 (Init-only isotropy test: KRR on unscaled PCA64 vs P-003's s…): challenge: The mechanism has a sharper prediction available. Standardisation raises near-silent early edge-sensor channels, whose variance is about sigma^2, to unit variance. Standardised PCA64 should then devote compon…
- E-04 (Init-only isotropy test: KRR on unscaled PCA64 vs P-003's s…): challenge: Gamma equivalence: the median-heuristic argument holds for the average distance, but the distance distribution differs between the two spaces. Add a small gamma_mult sweep {0.5, 1, 2} on both arms so the conc…
- E-04 (Init-only isotropy test: KRR on unscaled PCA64 vs P-003's s…): counterexample: If both arms retain all informative directions at 64 components, the gain is under 10%. That is a clean negative result, so the threshold is well chosen.
- E-05 (MC-noise attribution arm: P-001 settings with 4x importance…): challenge: The draw count of P-001 is inconsistent across proposals. P-017 says P-001 used 128 draws (ESS 57.9 of 128), while P-024 says 64. The '4x draws' factor and the predicted ESS-scaling contrast both depend on th…
- E-05 (MC-noise attribution arm: P-001 settings with 4x importance…): challenge: The pilot already shows the physics-MMSE arm is at its own Bayes-risk estimate: posterior mean_std 0.0182 against an achieved error of 0.0186, with z_rms 1.01. The total MC inflation you predict (~0.9% rel_l2…
- E-05 (MC-noise attribution arm: P-001 settings with 4x importance…): challenge: The MC variance of a self-normalised IS estimator can be estimated within a single run from the importance weights (delta-method variance, sum_j w_j^2 (k_j - m_hat)^2). That gives a per-sample MC standard err…
- E-05 (MC-noise attribution arm: P-001 settings with 4x importance…): challenge: The 'mean beats mode near bounds' half of the hypothesis is shared with P-022, P-024 and P-025. The portfolio now has four arms on one effect of roughly 1e-4 absolute.
- E-06 (Minimum-iteration hybrid: KRR (unscaled PCA32) init plus on…): challenge: The cost claim is likely wrong because of the cost structure, not the convergence behaviour. With a finite-difference Jacobian (about 5 evals per LM iteration), 10 iterations cost about 50-60 evals, while 64 …
- E-06 (Minimum-iteration hybrid: KRR (unscaled PCA32) init plus on…): challenge: posterior_samples=64 versus P-001's 64/128 (unresolved) confounds the accuracy comparison with MC noise. At ESS ~30 against ~58, expect about 0.8% rel_l2 of MC difference unrelated to iteration count. That is…
- E-06 (Minimum-iteration hybrid: KRR (unscaled PCA32) init plus on…): challenge: The KRR init error at n=3000 with unscaled PCA32 has not been reported. The '~5%' figure is assumed. Quadratic convergence needs e0 * (curvature constant) << 1. A 2-step test of the basin picture depends on t…
- E-06 (Minimum-iteration hybrid: KRR (unscaled PCA32) init plus on…): challenge: The comparator should be the 10-iteration hybrid with the same init (P-010), not P-001. Otherwise 'init + 2 steps vs multistart + 40' confounds starts with iterations.
- Unresolved disagreement ['P-017', 'P-024']: Whether P-001 used 64 or 128 IS draws. This sets the ESS-scaling prediction and the jitter factor (2x vs sqrt(8)).
- Unresolved disagreement ['P-019', 'P-026']: Two separate log/raw contrasts with mismatched architectures (gelu/80 epochs vs silu/60 epochs), each paired with a twin outside this round.
- Unresolved disagreement ['P-026', 'P-027']: Both rely on short-time Varadhan asymptotics. The window (T ~ 0.033, edge d^2/(4kT) ~ 1), a finite pulse width and the explicit-FD kernel may put the data outside that regime. The disagreement remains open until dt, T a…
- Unresolved disagreement ['P-021', 'P-020']: P-021 argues that equalising noise-dominated channels hurts, while P-020 whitens its PCA components, which does exactly that in component space.
- Literature check INCOMPLETE: 24 failed queries (disabled), 0 skipped for budget, enabled=False
- Test-set sampling uncertainty: bootstrap 95% CI [0.01721, 0.0194] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The pilot is the main evidence. On 60 val samples, val rel_l2 = 0.0186. The fit report gives residual std at … |
| 2 | researcher-mathematical | abandon | P-002 | - | llm | The pilot answers the main open question. The hybrid reached 0.0189 on 60 val samples with an initializer tra… |
| 2 | researcher-ml | revise | P-003 | P-010 | llm | (1) Standard scaling: accepted, and it is a real flaw. The noise is i.i.d. with the same sigma on every chann… |
| 2 | researcher-ml | revise | P-004 | P-011 | llm | (1) Title/hypothesis contradiction: accepted. There is now one numeric prediction relative to the target: val… |
| 2 | researcher-ml | propose | - | P-012 | llm |  |
| 2 | researcher-experimental | revise | P-005 | P-013 | llm | (1) Fidelity diagnostics: I agree they should be mandatory. I now request them as harness outputs for every p… |
| 2 | researcher-experimental | revise | P-006 | P-014 | llm | (1) Confounded design: I accept it fully. The mode-vs-mean question now lives in the P-005 (mode) vs. P-001 (… |
| 2 | researcher-exploratory | revise | P-007 | P-015 | llm | (1) Duplicate of P-003: accepted. I am turning P-007 into the estimator='mode' twin of P-003. It matches P-00… |
| 2 | researcher-exploratory | revise | P-008 | P-016 | llm | (1) Spectral mechanism: accepted as probably invalid. My own stability estimate agrees with yours: dx = 1/39,… |
| 3 | researcher-mathematical | revise | P-009 | P-017 | llm | I accept the main critique and am repurposing P-009 rather than abandoning it. Point by point:  (1) Duplicate… |
| 3 | researcher-ml | revise | P-010 | P-018 | llm | (1) 'The claims are already answered': accepted. The P-010 pilot (0.0189, 11825 forward evals over 60 samples… |
| 3 | researcher-ml | revise | P-011 | P-019 | llm | (1) '<0.05' is uninformative: accepted. I replace it with an interval prediction of 0.025-0.040 at full data.… |
| 3 | researcher-ml | revise | P-012 | P-020 | llm | (1) Confounded comparison with P-004: accepted. I drop P-004 as the comparator. The code now fits one identic… |
| 3 | researcher-ml | propose | - | P-021 | llm |  |
| 3 | researcher-experimental | revise | P-013 | P-022 | llm | (1) Loose prediction: accepted. The P-001 pilot (0.0186, n=60) and the calibrated, well-specified forward mod… |
| 3 | researcher-experimental | revise | P-014 | P-023 | llm | (1) Contradictory prediction field: accepted. The prediction is now equivalence, encoded as rel_l2 < 0.021, w… |
| 3 | researcher-experimental | propose | - | P-024 | llm |  |
| 3 | researcher-exploratory | revise | P-015 | P-025 | llm | (1) Monte Carlo noise in P-003 at posterior_samples=32: accepted. The critic is right that ESS of about 15 ad… |
| 3 | researcher-exploratory | revise | P-016 | P-026 | llm | (1) Mechanism refinement: accepted. It improves the hypothesis. The heterogeneous short-time asymptotic is lo… |
| 3 | researcher-exploratory | propose | - | P-027 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.769 | fdbk-00044 |
| 2 | 8 | 0.8 | fdbk-00079 |
| 3 | 11 | 0.766 | fdbk-00116 |

### Idea evolution

**P-001 — Posterior-mean physics inversion via bounded LM in log-conductivity reaches near-CRB accuracy** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MMSE[log,ms=4,it=40]` status conjectured. *H:* With an exact, available forward model, low noise (sigma=1e-3) and moderate conditioning (median Fisher cond 46.6), per-sample bounded Levenberg-Marquardt in log k with multistart, followed by importance-sampled posterior mean under the uniform box prior, att…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If error is far above the CRB, the misfit landscape has spurious minima or the box-constrained LM stalls; a learned initializer (hybrid) is then needed. If the…
  - critic r1 (quality 0.875): The CRB is not a floor for this estimator. A posterior mean under a bounded uniform prior is biased, so it can beat the CRB on coordinates near the bounds. The… · The posterior mean must be computed in k space (E[k/y]), not as exp(E[log k/y]). The latter is a geometric mean and is biased low by about sigma^2/2 in log uni… · Treat 0.0195 as a reference scale, not a lower bound. Report per-sample error divided by per-sample CRB std, computed at the true k on val. · A truth with k1 near 0.5 and k2 near 2 means the edge sensor sees little heat over a short window. The misfit in k1 is then flat, and LM stalls on the bound. M… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* Per-sample bounded LM in log k (multistart=3, iters=30, identical to P-005), followed by an importance-sampled posterior mean under the uniform box prior, attains full-val rel_l2 < 0.025. Its paired error is lower than P-005's MLE, with the difference concent…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If (a) fails, the tail draws (extreme contrasts) dominate and the 60-sample pilot was optimistic; per-sample error/CRB will locate them. If only (b) fails, mod…
  - response to critic: The pilot is the main evidence. On 60 val samples, val rel_l2 = 0.0186. The fit report gives residual std at truth 0.000995 against sigma 0.001 (200 samples, model_mismatch=false) and chi2/dof 0.944. Uncertainty is calibrated: 1-sigma cove…
  - critic r2 (quality 0.756): P-009 nearly duplicates P-001. It is the same log-k bounded LM with posterior_mean, changing only multistart from 4 to 3 and iters from 40 to 30. It also build… · Falsification clause (b) is mis-specified. It tests rel_l2(P-005) - rel_l2(P-001), but this arm is P-009. As written, the test does not involve this proposal's… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Take an interior sample with a Gaussian posterior of width 0.018 relative and ESS 20. The mode error is about 0.018 and the IS mean error is about 0.018 * sqrt… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-017` r3 [revise, llm] `physics-MMSE[log,ms=4,it=40]` status conjectured. *H:* Holding the LM solve fixed (log k, ms=4, it=40), raising importance-sampling draws from 128 to 512 lowers full-val rel_l2 by about 0.5-1% relative versus P-001. This matches the predicted MC inflation tr(Var_post)/ESS. Once MC noise is reduced, the posterior …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If (b) fails, the IS weights are degenerate or misweighted, the ESS-scaling assumption is wrong, and every posterior_mean arm needs an analytic estimator. If (…
  - response to critic: I accept the main critique and am repurposing P-009 rather than abandoning it. Point by point:  (1) Duplicate of P-001 / two mean-vs-mode pairs. Accepted. I drop the ms=3/it=30 settings and the P-005 pairing. P-009 now uses P-001's piloted…
  - critic r3 (quality 0.81): The draw count of P-001 is inconsistent across proposals. P-017 says P-001 used 128 draws (ESS 57.9 of 128), while P-024 says 64. The '4x draws' factor and the… · The pilot already shows the physics-MMSE arm is at its own Bayes-risk estimate: posterior mean_std 0.0182 against an achieved error of 0.0186, with z_rms 1.01.… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If ESS at 512 near bounds stays around 30 because of truncation, the near-bound stratum gains almost nothing in MC precision while the interior gains. The aggr… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-002 — Amortised learned inverse plus short LM refinement: the neural initializer removes basin errors and refinement closes t…** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[torch_mlp/log+logy+aug2->LM10]` status conjectured. *H:* A torch MLP trained on log-temperature features to predict log k gives a coarse inverse map (expected val rel_l2 ~0.04-0.08, limited by finite training data). Using it to initialise 10 LM iterations with 1 extra start yields val rel_l2 at least 30% lower than…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.3, "expect": "better"}`; if it fails: If the hybrid shows no improvement over the MLP, then either LM is not converging in 10 iterations, implying stronger nonlinearity or worse conditioning than t…
  - critic r1 (quality 0.765): noise_scale=1.0 conflicts with P-004, P-003 and P-008, which all use 0.001. If noise_scale is in raw units, this adds noise 1000x larger than sigma and destroy… · Noise augmentation re-noises training features that are already noisy. The effective training noise becomes sigma*sqrt(1+...), and that biases the learned map … · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer + few-step Levenberg–Marquardt refinement (hybrid) should approach the Cramér–Rao floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[kernel_ridge/raw+pca64+logy+aug1->LM10]` status conjectured. *H:* Initialize from a cheap learned inverse map (kernel ridge on standardized raw sensor traces). Then run about 10 bounded LM iterations through the exact forward model in log-k space, with posterior-mean estimation under the box prior. This should reach test-pr…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If refinement does not close the gap, the cause is one of three things: the initializer lands outside the basin (check per-sample residual chi^2), the Laplace/…
  - critic r1 (quality 0.825): With 'standard' scaling, each time-channel is normalised to unit variance. Early edge-sensor channels have near-zero signal variance, so standardisation inflat… · The bar of beating the initializer by 30% is not discriminating, for the same reason as in P-002. · The true discriminating comparison is hybrid vs pure physics (P-001/P-005) at matched estimator. Equal accuracy at lower cost is the expected outcome, and cost… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-010` r2 [revise, llm] `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM10]` status conjectured. *H:* A kernel-ridge initializer on centred, unscaled PCA(32) features lands inside the LM basin for at least 98% of samples. Ten bounded LM iterations in log k with box-prior posterior-mean estimation should then reach val rel_l2 < 0.035, within 1.8x of the CRB re…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: The chi^2 diagnostics separate three cases. High chi^2 with high init error means basin failure, so fix the initializer or add starts. chi^2/dof about 1 with l…
  - response to critic: (1) Standard scaling: accepted, and it is a real flaw. The noise is i.i.d. with the same sigma on every channel, so it is isotropic in raw feature space. Per-channel standardisation breaks that isotropy and inflates noise on near-silent ea…
  - critic r2 (quality 0.78): The main claims are already largely answered. P-002's pilot used an MLP init trained on only 300 samples plus 10 LM steps and reached 0.0189, matching P-001's … · After 10 LM iterations from any in-basin start, the final error is insensitive to init features (unscaled PCA32 vs standard PCA64). The unscaled-PCA argument w… · The basin is large enough that a 300-sample learned init suffices (empirical, from P-002). Init quality affects cost and iteration count, not final accuracy. · P-002 pilot: a different, weaker initializer reached the same final 0.0189. The initializer was therefore not the binding factor. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-018` r3 [revise, llm] `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` status conjectured. *H:* The learned initializer has about 5% relative error, the condition number is about 47 and the log-k likelihood is near-quadratic. Under these conditions Gauss-Newton converges essentially quadratically, so 2 LM iterations should reach the same accuracy as 10.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0205}`; if it fails: If the gap is concentrated in k1/k4 and chi^2/dof is above 1, damped LM converges slowly along low-curvature directions: use 3-5 iterations, or an undamped GN …
  - response to critic: (1) 'The claims are already answered': accepted. The P-010 pilot (0.0189, 11825 forward evals over 60 samples) and the P-002 pilot (0.0189) confirm that '<0.035' and '<half of P-001 cost' carry almost no information. I have dropped both as…
  - critic r3 (quality 0.81): The cost claim is likely wrong because of the cost structure, not the convergence behaviour. With a finite-difference Jacobian (about 5 evals per LM iteration)… · posterior_samples=64 versus P-001's 64/128 (unresolved) confounds the accuracy comparison with MC noise. At ESS ~30 against ~58, expect about 0.8% rel_l2 of MC… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the KRR init has a heavy tail (for example 15-25% error at corner draws), 2 steps converge for most samples, but a few unconverged ones dominate the squared… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-004 — Amortized neural inverse map with log-target and noise augmentation reaches the success threshold without forward-model…** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+pca64+logy+aug3` status conjectured. *H:* A torch MLP trained on 3000 simulated pairs should reach val rel_l2 < 0.06 and land within a factor of ~3 of the CRB floor. The inputs are PCA-compressed standardized sensor traces; the targets are log k; training uses on-the-fly re-noising at the known sigma…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.25, "expect": "worse"}`; if it fails: If the MLP matches or beats the hybrid, the LM refinement is wasted compute and amortized inference is sufficient. If the MLP is far worse than 0.06, either PC…
  - critic r1 (quality 0.78): The title ('reaches the success threshold') contradicts the hypothesis (val < 0.06, and worse than the hybrid by more than 25%). State one numeric prediction r… · At 160 s this is the most expensive arm, and it exceeds the 120 s cap. Halve the epochs; the MLP is likely converged well before 300 epochs on 3000 samples. · The amortised error is plausibly dominated by approximation error on a 4-D map with about 7 samples per axis, not by noise. A learning curve (n=750, 1500, 3000… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-011` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch MLP on standardised raw sensor traces (no PCA, log-k targets, noise_aug=2), trained on the 3000 provided pairs, reaches val rel_l2 < 0.05, about 2x the CRB reference. Its error is approximation-limited. Per-parameter, the k1/k4 relative errors exceed …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 >= 0.05 but the fresh-simulation arm succeeds, sample size is the limit, not the architecture. If both fail, the remaining causes are capacity or…
  - response to critic: (1) Title/hypothesis contradiction: accepted. There is now one numeric prediction relative to the target: val rel_l2 < 0.05. The pilot (0.053 on 60 val samples with a reduced training fraction) makes this a live, falsifiable bet with full …
  - critic r2 (quality 0.765): The '<0.05' prediction carries little information. The cited pilot (0.053) used train_fraction=300, i.e. 300 training samples, with PCA64. With 10x more data, … · At 120 s this is the most expensive arm, about 30% of the budget. Together with P-016 and its raw twin, the log-vs-raw MLP question would consume about 360 s. · The relevant baseline is the full-data MLP error, likely around 0.025-0.04, not the 300-sample pilot. · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-019` r3 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* At n=3000, the amortised MLP is approximation-limited and reaches val rel_l2 in [0.025, 0.040], i.e. 1.3-2x the CRB reference. Per-parameter, the k1/k4 relative errors exceed the k2/k3 errors by more than 1.5x. Compared with P-016 (log features, otherwise ide…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If error >= 0.040, the bottleneck is the training regime or the input dimension (800 raw inputs at n=3000); P-012's learning curve then decides whether more da…
  - response to critic: (1) '<0.05' is uninformative: accepted. I replace it with an interval prediction of 0.025-0.040 at full data. The machine-readable prediction is '<0.040', and the falsifiable test also requires >= 0.025, i.e. not CRB-efficient. (2) 120 s i…
  - critic r3 (quality 0.78): The rule estimate is 120 s against your 35-45 s, for 2 seeds. Together with P-026 (120 s), the two log/raw MLP arms alone take more than half the 400 s budget. · There are now two log/raw contrasts with different settings: P-019 vs P-016 (gelu, 80 epochs) and P-026 vs P-011 (silu, 60 epochs). Neither matched pair is ful… · Raw traces include 800 inputs, many of which are early edge-sensor readings dominated by noise. Standard scaling inflates those channels (see P-021). The raw a… · If the CRB-normalised per-parameter errors are equal across parameters, the 1.5x ratio passes while the 'weakly identified directions underfit' mechanism is fa… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Well-specified forward model makes per-sample MLE reach the CRB floor: LM in log-k with multistart** (researcher-experimental, family physics)
- v1 `P-005` r1 [propose, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* Because the forward model is available, noise is known i.i.d. Gaussian (sigma=0.001), and the Fisher matrix is well-conditioned (median cond 46.6), bounded Levenberg-Marquardt in log-k with a few starts will converge to the global least-squares minimum for ne…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If residuals are white at sigma but error is still large, the error is dominated by k1/k4 weak identifiability. Bayesian shrinkage (posterior_mean) would then …
  - critic r1 (quality 0.855): The fidelity diagnostics (b) and (c) are the most valuable checks in the portfolio. They should be mandatory outputs for every physics and hybrid arm. · The mode estimator with box bounds piles mass onto the bounds for edge draws. Report the fraction of coordinates at a bound. · Stability of the explicit scheme at k=2 is an assumption that should be verified directly (see the P-001 pilot). Instability would make the forward map non-smo… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* The forward model is identical to the data generator, the noise is i.i.d. Gaussian with known sigma=0.001, and the Fisher matrix is moderately conditioned (median 46.6). Under these conditions, bounded LM in log-k with 4 starts converges to the global least-s…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If residuals are white at sigma but rel_l2 is still high, the remaining error is identifiability-limited for k1/k4, and P-001's posterior mean should win in th…
  - response to critic: (1) Fidelity diagnostics: I agree they should be mandatory. I now request them as harness outputs for every physics and hybrid arm: the chi^2 ratio, per-sensor lag-1 autocorrelation, and the fraction of coordinates at a bound. As a gating …
  - critic r2 (quality 0.84): The prediction (<0.035) is loose given the pilot (0.0186 for the mean twin) and calibrated posteriors. A sharp, risky prediction would be rel_l2 within [0.017,… · The fidelity gate is already passed (residual std 0.000995 on 200 samples). Rerunning it is cheap but adds no information. · The expected chi2/N at the optimum is (N-4)/N, about 0.995, not 1. The [0.95, 1.05] window is fine. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-022` r3 [revise, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* The forward model is well-specified (residual std/sigma = 0.995), so bounded LM in log-k with 4 starts and 40 iterations reaches the global least-squares minimum for at least 98% of val samples. Full-val rel_l2 then lies in [0.017, 0.022], and aggregate rel_l…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If rel_l2 is above 0.022 but chi2 is about 1 and the basin-failure fraction is below 2%, the mode is efficiency-limited by box truncation, and the posterior me…
  - response to critic: (1) Loose prediction: accepted. The P-001 pilot (0.0186, n=60) and the calibrated, well-specified forward model justify a sharper and riskier prediction. I now predict a full-val rel_l2 in [0.017, 0.022], and paired equivalence with P-001 …
  - critic r3 (quality 0.865): This appears to duplicate P-013 (physics mode, ms4/it40), which P-017, P-023 and P-024 all reference as the mode comparator. If P-013 is still live, this arm i… · The basin-failure detector 'misfit > 2x median' is far too loose. With N = 800 readings per sample, chi2/N has SD ~ sqrt(2/800) ~ 0.05. A real basin failure co… · The comparison with P-001 includes the mean arm's MC penalty (about +0.9% rel_l2 at ESS 58). The aggregate mode-minus-mean difference could therefore have eith… · A mode projected onto a box face can have smaller error than the posterior mean when the truth is exactly at or near the bound. The sign of the near-bound effe… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-006 — Under rel_l2 loss, the MMSE posterior mean beats the MLE on the weakly identified edge segments (competing-explanation …** (researcher-experimental, family hybrid)
- v1 `P-006` r1 [propose, llm] `hybrid[ridge/log+logy->LM15]` status conjectured. *H:* The rel_l2 metric is a squared-error loss, and k1/k4 have relatively large posterior spread with the uniform box prior truncating it. So the posterior mean under the box prior (MAP-start LM, then importance sampling on the Laplace approximation) will give low…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.03, "expect": "better"}`; if it fails: If there is no gain, the posterior is effectively untruncated and Gaussian, so the mode is sufficient and the remaining error is pure noise at the CRB. In that…
  - critic r1 (quality 0.81): The design compares a ridge-init hybrid using posterior_mean against P-001 physics, and calls it the mode baseline. That confounds the initializer with the est… · Magnitude estimate: posterior std is about 1-2% of k. The fraction of coordinates within 2 sigma of a bound is about (0.02+0.08)/1.5, roughly 7% per coordinate… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-014` r2 [revise, llm] `hybrid[ridge/raw+logy->LM15]` status conjectured. *H:* The 4-parameter misfit is near-convex in log-k over the prior box, so a crude linear initializer lands inside the LM basin for nearly all samples. The final error is then set by the noise and the estimator, not by initializer quality. Ridge init (raw features…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.0, "expect": "worse"}`; if it fails: If this arm is clearly worse and the outlier fraction is elevated, the landscape has basin structure, particularly along the k1/k4 directions. Initializer qual…
  - response to critic: (1) Confounded design: I accept it fully. The mode-vs-mean question now lives in the P-005 (mode) vs. P-001 (mean) pair, which has identical LM settings, and in P-003 vs. P-007 for the hybrid. This arm no longer tests the estimator. (2) Ma…
  - critic r2 (quality 0.834): The prediction field (relative_to best_physics, expect 'worse') contradicts the falsifiable test (equivalence within ±3%). Pick one. · The comparison against P-003 confounds the estimator: P-003 uses the posterior mean and this arm uses the mode. The mode-matched hybrid comparator is P-015, no… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · For a truth like k = (0.5, 2.0, 0.5, 2.0), a linear init biased toward the prior mean may land on the wrong side of a near-flat k1/k4 valley. A single start co… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-023` r3 [revise, llm] `hybrid[ridge/raw+logy->LM15]` status conjectured. *H:* The log-k misfit is near-convex over the prior box, so a ridge init (standardised raw features to log-k) lands in the global basin for at least 97% of val samples. A single 15-iteration bounded LM run in log-k (mode estimator) then matches P-013 (ms4/it40, mo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.021}`; if it fails: If this arm is worse and the failures concentrate at high contrast, the landscape has corner basins, and learned initializers (P-015/P-010) or multistart are n…
  - response to critic: (1) Contradictory prediction field: accepted. The prediction is now equivalence, encoded as rel_l2 < 0.021, which is P-013's predicted centre plus the ±3% equivalence band. The falsifiable test is a paired equivalence test against P-013. (…
  - critic r3 (quality 0.84): The pilot already gives partial evidence of near-convexity: ms3/it30 and ms4/it40 agree to 1e-6, and learned inits agree with multistart. The informative outco… · The basin-failure threshold 'misfit > 2x median' has the same looseness problem as in P-022. Use a chi2/N tail threshold. · Fifteen bounded LM iterations from a start ~30% off may stall on a box face along the flat k1/k4 directions even inside the basin. A stalled-at-bound solution … · A truth with k1 = 0.5 and k2 = 2.0 next to it is a high-contrast corner where the linear init may overshoot outside the box. Clipping the init to the box puts … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-007 — MMSE beats MLE: learned initializer + LM refinement with a box-prior posterior-mean estimator in log-conductivity space** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[kernel_ridge/raw+pca40+logy+aug1->LM12]` status conjectured. *H:* Rel_l2 is a squared-error-type loss, so the Bayes-optimal point estimate under the uniform box prior is the posterior mean, not the MLE or mode. A hybrid that starts from a learned initializer, refines by bounded LM in log(k), and then reports the importance-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If val rel_l2 is well above the CRB floor, either the initializer lands in the wrong basin (check by comparing with extra_starts=3) or the forward model the ha…
  - critic r1 (quality 0.45): Near-duplicate of P-003: same kernel-ridge raw init, posterior_mean, extra_starts=1, with only pca 40 vs 64 and iters 12 vs 10. Its distinctive content is the … · The predicted 5% gain of mean over mode is optimistic given about 7% of coordinates near bounds (see P-006). The prediction of 0.025 or less is close to the CR… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-015` r2 [revise, llm] `hybrid[kernel_ridge/raw+pca64+logy+aug1->LM10]` status conjectured. *H:* With every other setting identical to P-003, the estimator='mode' hybrid will reach val rel_l2 within 0.025 of the CRB scale, about 0.02-0.025. P-003's posterior_mean variant will beat it by 1-3% relative overall. On the stratum of samples where any true k_i …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the mode arm exceeds 0.03 while chi^2/dof is about 1, the estimator is unbiased but inefficient, or the CRB at 32 draws understates the error. If chi^2/dof …
  - response to critic: (1) Duplicate of P-003: accepted. I am turning P-007 into the estimator='mode' twin of P-003. It matches P-003 on kernel_ridge, raw features, pca=64, log target, noise_aug=1, iters=10 and extra_starts=1, and differs only in the estimator. …
  - critic r2 (quality 0.81): This is a good twin design, but the paired comparator P-003 uses only posterior_samples=32. P-002 reported an ESS of about 29.5 (at its own sample count), so w… · Near-bound stratum size: about 7% of coordinates per axis gives roughly 1-(0.93)^4, about 25% of samples, or about 190 samples. The power is adequate for a 5% … · P-003's posterior mean is a noisy Bayes estimate. Its MC variance must be subtracted or eliminated before attributing differences to the estimator. · Interior samples with ESS ≈ 15: the mean arm is expected to be about 3% worse than the mode, which mimics a 'mode wins' finding. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v3 `P-025` r3 [revise, llm] `hybrid[kernel_ridge/raw+pca64+logy+aug1->LM10]` status conjectured. *H:* The estimator='mode' hybrid with P-003's exact settings reaches val rel_l2 < 0.024. Against P-003: (i) on the interior stratum, the mean arm's relative squared error exceeds the mode arm's by about 1/(2*ESS) per sample. It is not 'indistinguishable'. This is …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.024}`; if it fails: Mode rel_l2 at or above 0.024 with chi^2/dof about 1 means the CRB at 32 prior draws understates the average error, or the mode is inefficient. In that case, i…
  - response to critic: (1) Monte Carlo noise in P-003 at posterior_samples=32: accepted. The critic is right that ESS of about 15 adds roughly 1/ESS to the MSE of the mean arm. That is the same size as my predicted near-bound gain, so the original refutation cla…
  - critic r3 (quality 0.755): This is a second mean-vs-mode pair, on the hybrid instead of the physics arm. The pilots show the LM reaches the same mode regardless of init (identical chi2 0… · Prediction (i) has a units error. The MC penalty in squared error is about tr(Var_post)/ESS, so relative to the mode's squared error it is about (posterior var… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Mode and mean arms reaching identical LM points with an interior penalty of exactly mean_std^2/ESS would refute (b) as written, even though the mechanism is co… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-008 — Spectral reformulation: log-temperature features turn late-time decay into a near-linear function of log(k)** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* The inverse map is far more linear in (log u, log k) coordinates than in raw coordinates. At late times, the Dirichlet solution decays as a sum of modes with rates set by the Sturm–Liouville eigenvalues of -(k u')'. Those eigenvalues scale roughly multiplicat…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If log features hurt, the information about k1 and k4 is carried by early-time, low-amplitude transients rather than by spectral decay, so 'sqrt' features or r…
  - critic r1 (quality 0.795): The spectral argument needs the late-time single-mode regime, which is probably not reached. The lowest Dirichlet eigenvalue is about k pi^2, roughly 5-20. Wit… · Edge sensors at x of about 0.13 and 0.9 sit about 0.37-0.4 from the pulse centre. The diffusion length sqrt(2kT) is about 0.17-0.35, so early readings are near… · Log features may still help through a different mechanism. log u near the pulse front is approximately -(x-x0)^2/(4kt), which is affine in 1/k at fixed (x, t).… · If T is about 0.03 and k is about 0.5, sensor 5 reads about exp(-0.37^2/(4*0.5*0.03)), which is about 1e-1 times the peak amplitude. That reading is fine late … · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-016` r2 [revise, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* At fixed sensor and time, log u is approximately affine in 1/k_eff, the effective conductivity along the path from the pulse. So log features make the inverse map smoother than raw features do. Concretely: (a) ridge on log features beats ridge on raw features…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If ridge on log features does not beat ridge on raw features, the representation gain is absent, and amortised effort should go to raw features plus PCA (P-004…
  - response to critic: (1) Spectral mechanism: accepted as probably invalid. My own stability estimate agrees with yours: dx = 1/39, so dt <= dx^2/(2*2.0), about 1.6e-4, and T <= about 0.033, giving lambda1*T of about 0.15-0.6. The rod never reaches the single-m…
  - critic r2 (quality 0.795): Mechanism refinement: in a heterogeneous medium, the short-time heat-kernel asymptotic is log u ≈ c - τ(x)^2 / (4t), where τ = ∫ dx / sqrt(k(x)) is the diffusi… · The log of noisy near-zero edge readings has noise of about sigma/u, which can be orders of magnitude larger than at central sensors, plus negative values. The… · The natural coordinate is the travel time τ = ∫ dx / sqrt(k). Consider a target transform k^{-1/2} as an alternative to log. · If an edge sensor's readings are mostly ≤ 3σ, log features turn k1/k4 information into clipped noise. The overall result could then lose to raw features even i… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-026` r3 [revise, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* Short-time heterogeneous asymptotics give log u ~ c(t) - d^2/(4t) with d = integral of dx/sqrt(k). Log features therefore make the inverse map closer to linear than raw features do. (a) Ridge on log features beats ridge on raw features in val rel_l2, with the…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If (a) fails, log coordinates do not linearise the map at these times or noise levels, and amortised effort should go to raw features plus PCA. If (a) holds bu…
  - response to critic: (1) Mechanism refinement: accepted. It improves the hypothesis. The heterogeneous short-time asymptotic is log u(x,t) ~ c(t) - d(x)^2/(4t), with d = integral of dx/sqrt(k). Across the time window, the log trace at each sensor is affine in …
  - critic r3 (quality 0.78): The Varadhan form log u ~ c - d^2/(4t) is for the delta-function heat kernel. A Gaussian initial pulse of width w shifts the effective time to roughly t + w^2/… · Window scale: dt <= dx^2/(2 k_max) gives T ~ 0.033. The edge sensors are about 0.37-0.40 from the pulse, so d^2/(4kT) ~ 1 at the final time. That is not the sh… · The harness 'log' feature handles values <= 0 in an unknown way. P-027's explicit 3-sigma clip is the specified alternative. Use the same clip in both. · If the pulse width is comparable to sqrt(kT) (~0.18), log u at the inner sensors is close to quadratic in x at all times. Log features then help linear models,… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-012 — Sample-size test: amortised MLP trained on about 12k freshly simulated noisy pairs (3000 given plus about 9000 from con…** (researcher-ml, family physics-custom)
- v1 `P-012` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised learner's error is dominated by approximation error from sample scarcity (about 7 samples per axis), not by noise or capacity. Augmenting the 3000 training pairs with 9000 fresh prior draws, simulated through the forward model with sigma = 1e-3 …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If extra data gives less than a 25% improvement, the amortised arm is limited by capacity, optimisation or noise rather than samples. Physics refinement (hybri…
  - critic r2 (quality 0.819): The comparison with P-004 is confounded. It changes the learner (sklearn MLPRegressor vs torch_mlp), the features (whitened unscaled PCA48 vs standardised PCA6… · A shape check does not verify layout. Sensor-major and time-major flattenings have the same shape. Verify layout by computing forward(y_train[i]) - X_train[i]:… · Forward solves are cheap: P-001 ran about 27k forward evaluations in 4.4 s. The forward budget is unlikely to bind at 9000, so a learning curve is affordable. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- v2 `P-020` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised learner's error is dominated by sample-density approximation error. On a fresh simulated check set, error falls as n^(-a) with a >= 0.3 between n = 3000 and 12000. The 12k model reaches val rel_l2 < 0.035, at least 25% below the same-code n=3000…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: An exponent below 0.3 with flat k1/k4 errors means the learned map is information- or capacity-limited, not sample-limited. Refinement through the forward mode…
  - response to critic: (1) Confounded comparison with P-004: accepted. I drop P-004 as the comparator. The code now fits one identical learner (same PCA, MLP and target scaling) at n = 3000 (given data only, which is the n_sim=0 matched control), 6000 and 12000.…
  - critic r3 (quality 0.825): context.forward returns noise-free traces. The truncated code must add N(0, sigma^2) to the simulated features, and also to the check set. Otherwise the 6k and… · Whitened PCA(48) equalises variance across components, so trailing noise-dominated components are amplified to unit scale. This is the converse of P-021's isot… · The sklearn early_stopping validation split (10%) reduces the effective training size. That is fine as long as it is constant across n. · If the MLP is capacity- or optimisation-limited ((256, 256) with max_iter 150), the error plateaus at about 0.03 regardless of n. The exponent is then below 0.… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-021 — Init-only isotropy test: KRR on unscaled PCA64 vs P-003's standardised PCA64 initializer (single factor: scaler), no fo…** (researcher-ml, family data-driven)
- v1 `P-021` r3 [propose, llm] `kernel_ridge/raw+pca64+logy+aug1` status conjectured. *H:* The sensor noise is i.i.d. with a common sigma, so it is isotropic in raw feature space. Per-channel standardisation inflates noise on near-silent early edge-sensor channels. KRR (alpha=1e-3, gamma_mult=1, log targets, aug1) on centred, unscaled PCA(64) featu…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.045}`; if it fails: A gain under 10% means standardisation does not materially hurt under this noise level and PCA width. The amortised arms can then keep the standard scaler, and…
  - critic r3 (quality 0.81): The matched control (P-003's init run alone) is not a separate proposal, so the harness may not run it. Make it an explicit second arm here. Otherwise the test… · The mechanism has a sharper prediction available. Standardisation raises near-silent early edge-sensor channels, whose variance is about sigma^2, to unit varia… · k1/k4 information lives in the edge sensors' late-time readings, which have large raw variance and survive unscaled PCA. The predicted k1/k4 share of the gain … · If both arms retain all informative directions at 64 components, the gain is under 10%. That is a clean negative result, so the threshold is well chosen. · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-024 — Importance-sampling noise control: high-ESS posterior-mean twin of P-001 (512 draws) to bound the Monte Carlo component…** (researcher-experimental, family physics)
- v1 `P-024` r3 [propose, llm] `physics-MMSE[log,ms=4,it=40]` status conjectured. *H:* The posterior-mean arms estimate the MMSE by importance sampling on a Laplace proposal. Their Monte Carlo error in MSE scales as about 1/ESS. At 64 draws this error is comparable to the predicted 1-5% mean-vs-mode gain. Raising the draws to 512, with all else…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.021}`; if it fails: If the gain disappears at high ESS, the mean-vs-mode difference was IS noise. The cheaper mode estimator (P-013) should then be preferred, and posterior-mean c…
  - critic r3 (quality 0.339): Exact duplicate of P-017 (rule finding). Merge or differentiate before running. · executes exactly the same method as P-017 (researcher-mathematical); differentiate the hypothesis or merge · The rule finding applies: this arm executes exactly the same method as P-017. Running both wastes about 83 s of a 400 s budget. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-027 — Travel-time target coordinate: KRR on clipped log traces predicting k^{-1/2} instead of log k** (researcher-exploratory, family data-driven)
- v1 `P-027` r3 [propose, llm] `custom-ml` status conjectured. *H:* Because log u ~ c(t) - d^2/(4t) with d = sum_i L_i * k_i^{-1/2}, the map from log traces to k^{-1/2} is smoother than the map to log k. (a) Ridge on clipped log features with a k^{-1/2} target beats the same ridge with a log-k target by at least 10% in val re…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If the linear arms show no target-coordinate effect, the short-time travel-time asymptotic does not describe this window. That would mean the explicit-FD T is …
  - critic r3 (quality 0.81): The mechanism has an identifiability consequence worth stating and testing. Each sensor's path from the pulse (x = 0.5) crosses a nested set of segments. Senso… · The comparison between target coordinates also changes the loss weighting. Squared loss in z = k^{-1/2} weights small-k errors more (dz/dk ~ k^{-3/2}) than the… · The relevant travel distance is from the pulse support, not from x = 0.5, if the pulse is wide. The 'distance' then includes an effective offset term, and the … · If late-time, multi-mode decay dominates the window (plausible, since d^2/(4kT) ~ 1 at the edge sensors), the informative coordinate is the decay-rate eigenval… · the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

- Critic round 1 portfolio view: The physics route (P-005, P-001) is the best-motivated path to at most 0.05 and should run first, with chi^2 fidelity diagnostics mandatory. Merge P-007 into P-003 as a mode-vs-mean twin. Resolve P-002's noise_scale. Re-specify P-004 as a matched raw control for P-008. P-008's spectral mechanism is doubtful for a short explicit-FD window, so pilot ridge log-vs-raw first. The mean-vs-mode gains ar…  families={'physics': 2, 'hybrid': 4, 'data-driven': 2}
- Critic round 2 portfolio view: The target appears met in pilot: physics MMSE 0.0186 and hybrid 0.0189 on 60 val samples, with a well-specified forward model and calibrated posteriors at the Bayes floor. Recommended run set: P-001 at full val plus P-013 as its mode twin; P-014 (cheap basin stress test, compared against mode arms only); P-015 paired with P-003, provided P-003's Monte Carlo noise is controlled; P-012 with an inte…  families={'physics': 2, 'hybrid': 3, 'data-driven': 2, 'physics-custom': 1}
- Critic round 3 portfolio view: Recommended cuts to fit about 400 s: - Merge P-024 into P-017 and run it on a 250-sample paired subset, after computing the within-run MC SE from P-001. - Withdraw P-025, or fold its corrected MC-penalty analysis into the P-001/P-022 pair. - Confirm P-022 replaces P-013, and tighten its basin detector. - Keep P-023 (cheapest discriminating test). - Revise P-018: use the mode estimator, the P-010 …  families={'physics': 3, 'hybrid': 3, 'data-driven': 4, 'physics-custom': 1}

**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MMSE[log,ms=4,it=40]`: ok rel_l2=0.01863 on 60 val samples in 4.41 s
- P-002 `hybrid[torch_mlp/log+logy+aug2->LM10]`: ok rel_l2=0.018888 on 60 val samples in 12.4 s
- P-004 `torch_mlp/raw+pca64+logy+aug3`: ok rel_l2=0.053364 on 60 val samples in 23 s
- P-009 `physics-MMSE[log,ms=3,it=30]`: ok rel_l2=0.018628 on 60 val samples in 3.43 s
- P-010 `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM10]`: ok rel_l2=0.01889 on 60 val samples in 2.66 s
- P-011 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.051454 on 60 val samples in 14 s

## 5. Literature retrieval (TheoremSearch)

0 queries, 24 failures, 0 distinct statements retrieved, 0 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-mathematical | 1 | uniqueness of diffusion coefficient recovery heat equation from point observations at fin… | FAILED: disabled |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-ml | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-exploratory | 1 | local convergence of Gauss-Newton / Levenberg-Marquardt for nonlinear least squares with … | FAILED: disabled |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 2 | Max /forward(theta_i) - x_train_i/ / sigma over 50 training samples | FAILED: disabled |
| researcher-ml | 2 | Per-sample final chi^2 vs init error scatter | FAILED: disabled |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 2 | Levenberg-Marquardt parameter estimation piecewise constant conductivity heat equation Cr… | FAILED: disabled |
| researcher-exploratory | 2 | posterior mean versus MAP estimator squared error loss truncated Gaussian bounded paramet… | FAILED: disabled |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 3 | Per-sample IS Monte Carlo standard error and ESS distribution for the P-001 pilot | FAILED: disabled |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 3 | Init-alone val rel_l2 and per-parameter error for KRR on unscaled PCA32 vs standard PCA64 | FAILED: disabled |
| researcher-exploratory | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 3 | Fraction of estimated coordinates at a bound, and per-sample error/CRB ratio | FAILED: disabled |
| researcher-exploratory | 3 | ESS distribution and per-sample IS standard error for P-003 | FAILED: disabled |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_literature_s0.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 120 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-022) | physics | `physics-MLE[log,ms=4,it=40]` | researcher-experimental | 0.865 | best-rated proposal in family 'physics' (quality 0.86) |
| E-02 | proposal (P-023) | hybrid | `hybrid[ridge/raw+logy->LM15]` | researcher-experimental | 0.84 | best-rated proposal in family 'hybrid' (quality 0.84) |
| E-03 | proposal (P-020) | physics-custom | `custom-solver(fwd)` | researcher-ml | 0.825 | best-rated proposal in family 'physics-custom' (quality 0.82) |
| E-04 | proposal (P-021) | data-driven | `kernel_ridge/raw+pca64+logy+aug1` | researcher-ml | 0.81 | best-rated proposal in family 'data-driven' (quality 0.81) |
| E-05 | proposal (P-017) | physics | `physics-MMSE[log,ms=4,it=40]` | researcher-mathematical | 0.81 | quality 0.81 with diversity penalty |
| E-06 | proposal (P-018) | hybrid | `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` | researcher-ml | 0.81 | quality 0.81 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-024 *Importance-sampling noise control: high-ESS posterior-mean twin of P-001 (512 d…* — blocking: Exact duplicate of P-017 (rule finding). Merge or differentiate before running.
- not advanced: P-027 *Travel-time target coordinate: KRR on clipped log traces predicting k^{-1/2} in…* — portfolio size limit (6); quality 0.81
- not advanced: P-019 *Matched raw-feature twin of P-016: torch MLP [256,256] on standardised raw trac…* — portfolio size limit (6); quality 0.78
- not advanced: P-026 *Log-temperature features for an amortised MLP: single arm of a merged log/raw p…* — portfolio size limit (6); quality 0.78
- not advanced: P-025 *Mode twin of P-003: an MC-free reference arm isolating the MMSE-vs-mode effect,…* — portfolio size limit (6); quality 0.76

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-006 | E-05 | research | physics | `physics-MMSE[log,ms=4,it=40]` | 0.02041 | **0.018951** | 0.9967 | 0.02499 | 0.0115 | 56.7 | 598703 | 0.952 | FINAL |
| idea-013 | E-06 | refine | hybrid | `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` | - | **0.019082** | 0.9966 | 0.02516 | 3.63 | 7.95 | 87745 | 0.948 | evaluated |
| idea-012 | E-06 | refine | hybrid | `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` | - | **0.019107** | 0.9966 | 0.0252 | 3.63 | 6.52 | 65317 | 0.945 | evaluated |
| idea-014 | E-02 | refine | hybrid | `hybrid[ridge/raw+logy->LM15]` | - | **0.019114** | 0.9966 | 0.02521 | 0.0193 | 9.28 | 95703 | 0.946 | evaluated |
| idea-007 | E-06 | research | hybrid | `hybrid[kernel_ridge/raw+pca32+logy+aug1->LM2]` | 0.02013 | **0.019117** | 0.9966 | 0.02521 | 3.86 | 6.16 | 65287 | 0.946 | evaluated |
| idea-002 | E-01 | research | physics | `physics-MLE[log,ms=4,it=40]` | 0.02137 | **0.019323** | 0.9966 | 0.02548 | 0.00951 | 25.9 | 240032 | 0.957 | evaluated |
| idea-003 | E-02 | research | hybrid | `hybrid[ridge/raw+logy->LM15]` | 0.02139 | **0.019332** | 0.9966 | 0.02549 | 0.0246 | 5.53 | 50870 | 0.957 | evaluated |
| idea-015 | E-02 | refine | hybrid | `hybrid[ridge/raw+logy->LM15]` | - | **0.019332** | 0.9966 | 0.02549 | 0.0193 | 5.32 | 50870 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0105 | 6.69 | 61517 | 0.957 | evaluated |
| idea-005 | E-04 | research | data-driven | `kernel_ridge/raw+pca64+logy+aug1` | 0.04405 | **0.026505** | 0.9935 | 0.03495 | 3.88 | 0.197 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06289 | **0.029522** | 0.992 | 0.03893 | 13.9 | 0.00667 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `kernel_ridge/raw` | 0.05292 | **0.032419** | 0.9903 | 0.04275 | 0.468 | 0.0529 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `knn/sqrt` | 0.2036 | **0.14038** | 0.8184 | 0.1851 | 0.00852 | 0.0889 | 0 | - | evaluated |
| idea-004 | E-03 | research | physics-custom | `custom-solver(fwd)` | 0.04385 | **-** | - | - | - | - | - | - | ok |

## 8. Lineage of the final method

- `idea-006` `physics-MMSE[log,ms=4,it=40]` ← P-017 v3 (researcher-mathematical, r3, revise) ← P-009 v2 (researcher-mathematical, r2, revise) ← P-001 v1 (researcher-mathematical, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00248` [checkpoint] by scheduler: final method (single)
- `resu-00245` [result] by executor: exp-0031 rel_l2=0.01825 r2=0.997 rmse=0.02402
- `hypo-00133` [hypothesis] by scheduler: idea-006 physics-MMSE[log,ms=4,it=40] (research)
- `plan-00244` [plan] by engineer-v2: exp-0031: final fit of 1 member(s) on train+val; test once
- `gate-00132` [gate] by gate: E-05: approved (physics-MMSE[log,ms=4,it=40])
- `prop-00104` [proposal] by researcher-mathematical: P-017 v3 [revise] mathematical: MC-noise attribution arm: P-001 settings with 4x importance draws (512), forming a triad w
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_literature_s0 (749b718682)
- `prop-00069` [proposal] by researcher-mathematical: P-009 v2 [revise] mathematical: Box-prior posterior-mean LM inversion in log k (aligned with P-005 for a paired mean-vs-mo
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `prop-00035` [proposal] by researcher-mathematical: P-001 v1 [propose] mathematical: Posterior-mean physics inversion via bounded LM in log-conductivity reaches near-CRB accur

## 9. Compute and cost

- Execution compute: 384.2 / 400 worker-seconds over 31 jobs; wall 946 s
- By stage: baseline 2.1s, pilot 60.0s, screen 65.5s, evaluate 119.1s, refine 37.9s, final 99.6s
- LLM: 19 calls (by phase {'research': 19}), 1217 s, $4.065 reported · TheoremSearch: 0 queries (0 failed)

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

The evidence shows the objective is met with a wide margin. The selected model, physics-MMSE (log-parameterised, 4 multistarts, 40 iterations), reached val rel_l2 = 0.0190 and held-out test rel_l2 = 0.0183 (R² = 0.997, RMSE = 0.024), against a target of ≤ 0.05. Val and test agree closely, which suggests selection did not overfit the validation split. All five registered predictions were supported, and none were contradicted. The physics and hybrid families are effectively tied: 0.01895 vs 0.01908 is a 0.7% gap, and no seed or bootstrap variance was reported to separate them. The best purely data-driven model, kernel ridge on PCA-64 features with log targets, is clearly worse at 0.0265 (about 40% higher error). Feeding that same regressor into two Levenberg–Marquardt steps closes nearly the whole gap. So established knowledge of the forward model, not learned features, is what drives accuracy at this sample size.

No hypothesis was refuted outright. One was abandoned rather than tested: the claim that an amortised neural initializer plus LM refinement would remove basin errors and approach the Cramér–Rao bound. A pilot found that an MLP initializer trained on 300 samples gave 0.0189, versus 0.0186 for pure physics. It was dropped as redundant with the cheaper ridge-init hybrid. The supported reading is narrow: once an initializer lands inside the LM basin, final error does not depend on initializer quality. Whether basin errors were ever a meaningful failure mode, and how close any method is to the CRB, remain open, because no CRB or posterior-variance calculation appears in the results. It is therefore conjecture whether the ~0.019 floor is set by noise (information-limited) or by the optimizer and forward model. A further caveat: the pilot's 0.0186 and the final 0.0190 for "pure physics" were presumably measured under different settings or subsets, so small cross-run comparisons should not be over-read.

- Compute the CRB, or a Laplace/MCMC posterior spread, per conductivity segment at the dataset's noise level. This would show whether 0.019 is near-optimal or leaves headroom.
- Report per-segment errors and seed/bootstrap confidence intervals for the physics vs hybrid vs data-driven comparison. Segments far from sensors are likely less identifiable and may dominate the aggregate.
- Stress-test robustness to a mismatched noise level, out-of-prior conductivity ranges, and forward-model misspecification (grid or time-step mismatch). Under these conditions, amortised and hybrid methods may diverge from pure physics.
- Measure inference cost per sample for each method and plot the accuracy–compute frontier. The near-tie in accuracy makes cost the deciding factor between physics-MMSE and the ridge-init hybrid.
- Ablate the number of multistarts (1/2/4) and LM iterations. This would directly test the "basin errors are rare" finding and locate the cheapest configuration that stays within tolerance.

_Automatic summary:_ 27 proposal versions over 3 research rounds; 9 methods advanced; 5 predictions supported, 0 contradicted. Final: ['physics-MMSE[log,ms=4,it=40]'] with test rel_l2 0.018253.
