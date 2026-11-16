# Research report (V2): heat_v2_no_literature_s1

Run `heat_v2_no_literature_s1_20261109_112717` · 2026-11-09 11:39:04 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 62.3 (max 1.08e+03); Cramér–Rao relative std per parameter: k1 0.0215, k2 0.0058, k3 0.0052, k4 0.0174; implied rel_l2 floor ≈ **0.01851**; least determined: k1, k4.

## 2. Outcome

- Final method: **ensemble** of `hybrid[kernel_ridge/raw+logy+aug1->LM9]`, `physics-MLE[log,ms=1,it=30]`, `hybrid[kernel_ridge/raw+logy+aug1->LM10]` (weights [1.0, 1.0, 1.0])
- Validation rel_l2 (selection estimate): **0.019019** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018339**, r2 = **0.99699**, rmse = **0.024131** · bootstrap 95% CI for rel_l2: [0.01729, 0.01953]
- Per-target test rel_l2: k1 0.0269, k2 0.0061, k3 0.0053, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.086922 (vs 0.018339 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.05156 (vs 0.018339 in distribution)
- Inference latency: 33.1 ms/sample; fit 14.6 s; exported model reload check: max |Δ| = 1.1e-10
- Ensemble members on test: `hybrid[kernel_ridge/raw+logy+aug1->LM9]` 0.0185, `physics-MLE[log,ms=1,it=30]` 0.01842, `hybrid[kernel_ridge/raw+logy+aug1->LM10]` 0.0185
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[kernel_ridge/raw+logy+aug1->LM9]` rel_l2=0.01911; **physics** `physics-MLE[log,ms=1,it=30]` rel_l2=0.019335; **physics-custom** `custom-solver(fwd)` rel_l2=0.026151; **data-driven** `extra_trees/sqrt` rel_l2=0.082482
- SUPPORTED — E-01 *Per-sample bounded LM MLE in log-conductivity is statistically efficient relative to the mean-aggregated per-sample CRB…*: val rel_l2 = 0.019337 vs predicted < 0.025
- SUPPORTED — E-02 *Is the amortization gap data-limited? Fresh simulated training data (4x) with a box-aware logit target, as the missing …*: val rel_l2 0.026151 vs best_data_driven 0.082482: gain +68.29%; expected gain >= +20.0% vs best_data_driven
- SUPPORTED — E-03 *Merged hybrid (kernel-ridge init -> bounded log-LM) with init-only, mode and posterior-mean reported on identical val s…*: val rel_l2 = 0.019111 vs predicted < 0.03
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01851

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- no citations in proposals

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-04 *Log-feature amortized MLP: shape of the amortization gap relative to the per-parameter CRB*: inconclusive — method did not produce a full-fidelity validation result
- E-05 *Small-basin test: the cached log-MLP init plus 5 LM steps reaches about 1.35x the CRB (cross-arm trace with P-015, P-01…*: inconclusive — method did not produce a full-fidelity validation result
- E-06 *In-box truncated-Gaussian MMSE: bounded log-LM mode plus a linear GN extrapolated centre, CRN-centred truncation, zero …*: inconclusive — method did not produce a full-fidelity validation result

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-mathematical — P-002 *Learned initializer + LM refinement with posterior-mean (MMSE) estimator beats pure MLE on rel_l2*: This is a duplicate of P-005. Its 10% MMSE gain is implausible (expected 2-7%). It confounds initializer, iteration count and estimator. The P-001 pilot (rel_l2 0.0190 ≈ 1.03x CRB, chi^2/dof 0.944) shows that learned in…
- ABANDONED in round 2 by researcher-exploratory — P-007 *Posterior-mean (MMSE) estimation beats MLE because rel_l2 is a squared-error loss and k1/k4 are wea…*: Duplicate of P-005, which is the merged hybrid arm. The extra_trees initializer is dominated, and the MMSE-vs-mode question is better answered by one paired, bound-stratified comparison inside P-005 using the adopted 3%…
- ABANDONED in round 3 by researcher-ml — P-011 *Cheap learned initializer lets 8 LM iterations match 30-iteration multistart LM (speed, not accurac…*: Duplicate of P-016 and over the per-run time cap. Its scientific question (how many LM iterations an amortized init needs, and at what N amortization pays) is fully answered by the merged per-iteration trace on P-016 th…
- ABANDONED in round 3 by researcher-ml — P-012 *MMSE (posterior-mean) vs MLE under the box prior: small aggregate gain concentrated on near-bound k…*: Expected aggregate effect (about 1.5-2% in rel_l2) is at the falsification threshold, so the test cannot discriminate. The mechanism is already covered at lower cost by P-013 (IS posterior mean vs mode on identical fits…
- ABANDONED in round 3 by researcher-experimental — P-014 *Control: raw-feature amortized MLP (twin of P-008) vs forward-model-refined hybrid separates approx…*: blocking issue: The controlled contrast is invalid until the hyperparameters match P-015.
- PRUNED — idea-008 `svm/log+pca8`: projected full-fidelity cost 220s exceeds the 120s per-experiment limit
- GATE-REJECTED — E-06 `custom-solver(fwd)`: code: code must define a top-level build_method(params, context)

### 3.5 Remaining uncertainty

- E-01 (Per-sample bounded LM MLE in log-conductivity is statistica…): challenge: The CRB lower-bounds unbiased estimators only. Bounded MLE is biased near the box faces and can beat the per-sample CRB there, so rel_l2/floor_mean < 1 on edge samples is possible and would not mean the floor…
- E-01 (Per-sample bounded LM MLE in log-conductivity is statistica…): challenge: floor_mean = sqrt(mean_i tr(CRB_i)///k_i//^2) matches rel_l2 only if the harness aggregates as RMS of per-sample relative errors. If rel_l2 is the mean of per-sample ratios, or a global //K_hat-K//_F///K//_F,…
- E-01 (Per-sample bounded LM MLE in log-conductivity is statistica…): challenge: The per-sample CRB is evaluated at the estimate (the final GN Jacobian), not at the truth. For shielded samples the Jacobian is sensitive to k, so floor_mean estimated at k_hat is itself noisy and biased.
- E-01 (Per-sample bounded LM MLE in log-conductivity is statistica…): counterexample: k1 = 1.98 with the truth near the upper bound: the bounded MLE clips to 2.0 roughly half the time, so its MSE is about half the CRB. Pooling such samples lowers the ratio without any efficiency gain in t…
- E-02 (Is the amortization gap data-limited? Fresh simulated train…): challenge: Confound: with fixed epochs=80, 12000 samples get 4x more gradient steps than the n_new=0 twin. A gain could come from optimisation rather than data. Either match the step count or add a third arm (n_new=0, 3…
- E-02 (Is the amortization gap data-limited? Fresh simulated train…): challenge: The logit target is a second change relative to P-021's log target. The twin isolates data size, but cross-comparison with P-021 is confounded, so do not attribute any k1/k4 gain to the logit target without a…
- E-02 (Is the amortization gap data-limited? Fresh simulated train…): challenge: Is the 30 s estimate for both arms or one? The text says about 35 s in total including the twin; confirm the twin is budgeted.
- E-02 (Is the amortization gap data-limited? Fresh simulated train…): challenge: The fallback on column mismatch silently reverts to the original data. Treat a fallback as a failed run, not a null result.
- E-03 (Merged hybrid (kernel-ridge init -> bounded log-LM) with in…): challenge: The hypothesis says 15 LM iterations, but method.params.iters = 10. State which value is tested. The 'gap concentrated in…' prediction depends on it.
- E-03 (Merged hybrid (kernel-ridge init -> bounded log-LM) with in…): challenge: The posterior mean minimises E//k_hat-k//^2, not relative error. Under rel_l2 the Bayes-optimal point estimate is E[k///k//^2]/E[1///k//^2], which is shifted toward small-norm k. For narrow posteriors the dif…
- E-03 (Merged hybrid (kernel-ridge init -> bounded log-LM) with in…): challenge: Training rows with noise_aug carry sqrt(2)·sigma noise while val carries sigma, so the KRR init is mildly mis-specified. This is harmless for an initializer, but it confounds the init-only number if that numb…
- E-03 (Merged hybrid (kernel-ridge init -> bounded log-LM) with in…): challenge: The per-parameter ranking test (k2/k3 worse than k1/k4 falsifies) is near-certain to pass given a 4x CRB gap. It is weakly informative.
- E-04 (Log-feature amortized MLP: shape of the amortization gap re…): challenge: Non-uniform r_j does not by itself show an amortization deficiency. The Bayes-optimal amortized estimator (the posterior mean) shrinks weakly identified k1/k4 most, and can even reach r<1 near the bounds. A l…
- E-04 (Log-feature amortized MLP: shape of the amortization gap re…): challenge: Log features: the Gaussian pulse at x=0.5 gives near-zero true temperatures at the x≈0.13 and x≈0.90 sensors at early times. With sigma=1e-3, noisy readings there are often ≤0. How the harness clips or shifts…
- E-04 (Log-feature amortized MLP: shape of the amortization gap re…): challenge: The scaling exponent from the n=300 pilot (60 val samples) to n=3000 (750 val samples) mixes a noisy 60-sample estimate with changed epochs and architecture. P-019 measures data-limitation cleanly, so this pa…
- E-04 (Log-feature amortized MLP: shape of the amortization gap re…): challenge: At 120 s this is 30% of the total budget, and P-022 trains an identical model.
- E-05 (Small-basin test: the cached log-MLP init plus 5 LM steps r…): challenge: The rule estimate of 125.3 s assumes retraining. Together with P-021 (120 s), that is 245 of 400 s for the same MLP. The arm only makes sense if the init is cached from P-021, and that dependency must be enfo…
- E-05 (Small-basin test: the cached log-MLP init plus 5 LM steps r…): challenge: The cross-arm trace references P-015, P-011 and P-009, none of which are in this round. The available paired references are P-021 (iteration 0), P-020 (KRR init, 10 iterations) and P-017 (30 iterations, multi…
- E-05 (Small-basin test: the cached log-MLP init plus 5 LM steps r…): challenge: With a ~4% init, Gauss-Newton contraction over 5 steps depends on the LM damping start. Without a per-iteration trace within this arm, a miss cannot be attributed to init or to damping.
- E-05 (Small-basin test: the cached log-MLP init plus 5 LM steps r…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- E-06 (In-box truncated-Gaussian MMSE: bounded log-LM mode plus a …): challenge: The previous version of this solver (P-010 pilot) scored 0.079, about 4x worse than physics MLE. The diagnosis (an out-of-box explicit-FD instability with a loose clamp) is plausible but unverified. Run a 60-…
- E-06 (In-box truncated-Gaussian MMSE: bounded log-LM mode plus a …): challenge: By the authors' own estimate, about 14% of samples are near a bound, which caps the aggregate gain at about 2%. The prediction 'improvement 0.01 relative to best_physics' is ambiguous: is it absolute or relat…
- E-06 (In-box truncated-Gaussian MMSE: bounded log-LM mode plus a …): challenge: The truncated Gaussian centred on a linearly extrapolated m uses the curvature at the box face. For k1 near 2 (weakest direction) the extrapolation can run several sd outside the box, where the first-order er…
- E-06 (In-box truncated-Gaussian MMSE: bounded log-LM mode plus a …): challenge: The posterior mean optimises squared error, not relative error (see P-020).
- Unresolved disagreement ['P-021', 'P-019']: P-021 attributes the amortization gap to approximation/shrinkage on k1/k4, while P-019 attributes it to data size. Only P-019 has a clean control. P-021's shape test cannot distinguish learner error from Bayes-optimal s…
- Unresolved disagreement ['P-018', 'P-020']: Both claim an MMSE gain over the mode, via truncated-Gaussian vs posterior samples. Both are bounded at about 2% by the fraction of near-bound samples, and neither optimises the relative-error metric.
- Literature check INCOMPLETE: 24 failed queries (disabled), 0 skipped for budget, enabled=False
- Test-set sampling uncertainty: bootstrap 95% CI [0.01729, 0.01953] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The pilot (60 val samples) supports the hypothesis. Results: val rel_l2 = 0.0190, which is about 1.03x the CR… |
| 2 | researcher-mathematical | abandon | P-002 | - | llm | I accept the critic's points. (1) The 10% target is unattainable. The bound-truncation estimate gives 2-7%, a… |
| 2 | researcher-mathematical | propose | - | P-010 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-011 | llm | (1) Time blocker: accepted. I cut the MLP to [256,256] and 60 epochs, LM to 8 iterations and extra_starts to … |
| 2 | researcher-ml | revise | P-004 | P-012 | llm | (1) The 10% threshold is likely too high: accepted. A bound-truncation argument also suggests only a few perc… |
| 2 | researcher-experimental | revise | P-005 | P-013 | llm | (1) Mixing estimator and initializer effects: accepted. This merged arm, which absorbs P-002 and P-007, will … |
| 2 | researcher-experimental | revise | P-006 | P-014 | llm | (1) noise_scale conflict: accepted. 0.001 was a units error under the likely semantics. I align with P-008 by… |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | I accept all three points. (1) P-007 duplicates P-005, and an extra_trees initializer is coarse on roughly 80… |
| 2 | researcher-exploratory | revise | P-008 | P-015 | llm | The pilot gave val rel_l2 = 0.0474 on 60 val samples, trained on only 300 samples (a tenth of the train split… |
| 2 | researcher-exploratory | propose | - | P-016 | llm |  |
| 3 | researcher-mathematical | revise | P-009 | P-017 | llm | I accept all three challenges. The method is unchanged; the test and the floor are tightened.  (1) chi^2 thre… |
| 3 | researcher-mathematical | revise | P-010 | P-018 | llm | The pilot is decisive negative evidence: rel_l2 0.0792 on n = 60, against 0.0190 for P-009 on the same pilot … |
| 3 | researcher-ml | abandon | P-011 | - | llm | All three challenges are accepted. (1) Budget: at 127 s the arm breaks the 120 s per-run cap and would take a… |
| 3 | researcher-ml | abandon | P-012 | - | llm | The critic's arithmetic is right, and it undermines my own test design. (1) Stratum sizing: truncation matter… |
| 3 | researcher-ml | propose | - | P-019 | llm |  |
| 3 | researcher-experimental | revise | P-013 | P-020 | heuristic-fallback | Revised: reduced solver cost as requested |
| 3 | researcher-experimental | abandon | P-014 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 3 | researcher-exploratory | revise | P-015 | P-021 | llm | (1) Missing twin. Agreed. P-006 is not in this round and P-014 uses different hyperparameters, so I withdraw … |
| 3 | researcher-exploratory | revise | P-016 | P-022 | llm | (1) Over the cap (125.3 s). Fixed. The init is now byte-identical to the revised P-015 spec (80 epochs, [256,… |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.715 | fdbk-00044 |
| 2 | 8 | 0.735 | fdbk-00076 |
| 3 | 6 | 0.833 | fdbk-00103 |

### Idea evolution

**P-001 — Per-sample bounded Gauss-Newton/LM MLE in log-conductivity attains near-CRB accuracy** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* The forward map F: k -> sensor time series is smooth and locally injective on [0.5,2]^4. Its Fisher condition number is moderate (median 62). The noise is small (sigma=1e-3). So the least-squares misfit is close to quadratic around the truth over most of the …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If chi^2 is near 1 but rel_l2 is high, the CRB is optimistic and the error is irreducible noise sensitivity in k1/k4, so MMSE shrinkage is needed (Proposal 2).…
  - critic r1 (quality 0.87): The chi^2 falsification threshold (more than 2x expected) is far too loose. With n_obs = 4*T (about 800 readings), chi^2/n has a standard deviation of about sq… · The prediction (<0.035) is weak relative to the stated floor. Combining the per-parameter median CRB values as an RMS gives about 0.014; the harness reports 0.… · Replace 'no model mismatch' as an assumption with a measured fact. Compute chi^2/n of forward(k_true) against the val features before any fitting. · If k1 is near 2.0 and k2 is near 0.5, the edge segment is thermally shielded from the pulse. The k1 direction becomes nearly flat, and LM can stall at a bound … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* The forward model matches the data (oracle residual std 0.000995 vs 0.001). The misfit is near-quadratic at sigma = 1e-3. Under these conditions, the bounded LM MLE in log k with 3 starts and 30 iterations is efficient.  Predictions on the full val split: - r…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: Each failure mode points to a different fix: - rel_l2 ≥ 0.025 with chi^2 near 1 and calibrated coverage: the tail draws are intrinsically harder than the media…
  - response to critic: The pilot (60 val samples) supports the hypothesis. Results: val rel_l2 = 0.0190, which is about 1.03x the CRB floor of 0.0185. chi^2/dof = 0.944. The oracle check gave residual std at truth 0.000995 against nominal 0.001 over 200 samples,…
  - critic r2 (quality 0.885): The chi^2/n <= 1.15 threshold for 95% of samples only works if n_obs is large. For Gaussian residuals, sd(chi^2/n) is about sqrt(2/n). At n = 800 (4 sensors x … · The CRB floor of 0.0185 is built from per-parameter medians. rel_l2 is a mean over samples. Shielded configurations (large k1, small k2) have per-sample CRBs w… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · k1 = 1.98, k2 = 0.55: k1 is shielded behind a low-conductivity segment. Its per-sample CRB rel std can be several times the 2.15% median, so the realised error… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-017` r3 [revise, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* The forward model matches the data and the misfit is near-quadratic at sigma = 1e-3. Under these conditions, the bounded log-LM MLE (3 starts, 30 iterations) is efficient.  Predictions on the full val split: - rel_l2 / floor_mean <= 1.15, where floor_mean = s…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: Each failure mode points to a different conclusion: - (i) fails with chi^2 calibrated: the excess is bias or a local-minimum problem, so inspect the start spre…
  - response to critic: I accept all three challenges. The method is unchanged; the test and the floor are tightened.  (1) chi^2 threshold. If every step is observed, n_obs = 4 sensors x 200 steps = 800 per sample. The facts only say 'discrete times, finite windo…
  - critic r3 (quality 0.888): The CRB lower-bounds unbiased estimators only. Bounded MLE is biased near the box faces and can beat the per-sample CRB there, so rel_l2/floor_mean < 1 on edge… · floor_mean = sqrt(mean_i tr(CRB_i)///k_i//^2) matches rel_l2 only if the harness aggregates as RMS of per-sample relative errors. If rel_l2 is the mean of per-… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · k1 = 1.98 with the truth near the upper bound: the bounded MLE clips to 2.0 roughly half the time, so its MSE is about half the CRB. Pooling such samples lower… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-002 — Learned initializer + LM refinement with posterior-mean (MMSE) estimator beats pure MLE on rel_l2** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[kernel_ridge/sqrt+logy+aug1->LM15]` status conjectured. *H:* Because rel_l2 is a squared-error metric, the Bayes-optimal estimator under the uniform box prior is the posterior mean, not the MLE/mode. For the weakly determined boundary segments k1 and k4, and for samples whose truth lies near the bounds 0.5 or 2.0, the …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.1, "expect": "better"}`; if it fails: If the hybrid matches physics-mode but does not beat it, the posterior is near-Gaussian and untruncated for most samples, so the remaining error is the noise f…
  - critic r1 (quality 0.477): The 10% improvement target is probably unattainable from estimator choice alone. Posterior std is about 2% of k for k1/k4. Truncation matters only for truths w… · The comparison is confounded. The proposal changes the initializer (kernel ridge), the iteration count (15 vs 30), the starts, and the estimator at once, relat… · Expect MMSE gain ≤ 5% relative unless near-bound samples dominate the error. · For interior truths the posterior is Gaussian with std ~2%, so the mean equals the mode to within about 1e-4 relative. An improvement can only come from roughl… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer + few-step LM refinement (log-space) reaches the CRB-limited regime** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status conjectured. *H:* A torch_mlp trained on log-conductivity from standardized raw sensor traces (with noise augmentation at sigma=0.001) gives an initial estimate with rel_l2 ~0.05-0.10; 10-15 bounded LM iterations in log-space through the forward model then converge to the loca…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If refinement fails to improve on the initializer, the likelihood is multimodal or the forward model is mismatched, which can be checked from Gauss-Newton resi…
  - critic r1 (quality 0.675): Estimated 135.9s exceeds the 120s per-experiment limit. Reduce epochs and/or LM iterations, or change the initializer, before running. · The rule estimate is 135.9s, which exceeds the 120s per-experiment limit. This is mainly MLP training plus LM over the val set. · The 30%-improvement-over-init criterion is degenerate if the MLP is poor (easy) or already near the floor (impossible). It does not test the stated hypothesis. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-011` r2 [revise, llm] `hybrid[torch_mlp/raw+logy+aug2->LM8]` status conjectured. *H:* A short-trained noise-augmented torch_mlp (log targets) lands inside the LM basin of attraction (init rel_l2 roughly 0.04-0.10). From there, 8 bounded LM iterations in log-space with no extra starts reach val rel_l2 within 5% (relative, paired) of P-001's 30-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each failure mode points to a different cause. Hybrid rel_l2 ≫ P-001 with a good init would mean 8 iterations are too few or multimodality is present; check th…
  - response to critic: (1) Time blocker: accepted. I cut the MLP to [256,256] and 60 epochs, LM to 8 iterations and extra_starts to 0. This roughly halves training and LM cost, so the estimate should fall well under 120s. A 100-sample timing pilot should still r…
  - critic r2 (quality 0.396): Estimated 127 s is over the 120 s per-run limit. Merge with P-016 and reuse a cached MLP, or cut it down. · The rule estimate is 127 s, which is over the 120 s per-run cap the proposals cite. It is also nearly a third of the 400 worker-second total budget. · The speed claim counts forward calls per sample and leaves out MLP training. P-009 solves all 750 val samples in about 23 s, while the MLP alone takes 10-60 s … · Cost metric: total wall-clock including training, plus the break-even N at which the hybrid beats P-009. · If the MLP init is about 4% off and LM converges quadratically, then 3 iterations already suffice. Fixing 8 iterations without a per-iteration trace means neit…

**P-004 — MMSE (posterior-mean) estimator beats MLE because rel_l2 is a squared-error loss under a box prior** (researcher-ml, family physics)
- v1 `P-004` r1 [propose, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* For the weakly determined edge segments k1/k4, whose CR std is about 2% and grows near the upper bound where diffusive sensitivity saturates, the posterior mean under the uniform box prior has lower expected squared error than the LM mode. A physics posterior…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If MMSE ≈ mode, the posterior is effectively Gaussian and interior, so the remaining error is irreducible noise, and effort should move to speed (amortized hyb…
  - critic r1 (quality 0.825): This is the cleanest MMSE test because it pairs with P-001 under identical physics settings. However, the 10% threshold is likely too high (see P-002), so a nu… · The absolute target of 0.03 is weak. If the mode already gives about 0.02, the absolute criterion is trivially satisfied, and only the relative claim is inform… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-012` r2 [revise, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* Run with settings identical to P-001 apart from the estimator. The importance-sampled posterior mean (128 samples on the Laplace approximation, box prior) reduces val rel_l2 by 2-7% relative to P-001's mode, with a paired CI excluding 0. The reduction is ≥10%…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: Each failure mode has a different reading. No gain anywhere means the posterior is Gaussian and interior, so effort should move to amortized speed. A gain loca…
  - response to critic: (1) The 10% threshold is likely too high: accepted. A bound-truncation argument also suggests only a few percent overall, because the posterior std (~2%) is small next to the box width. I now predict a 2-7% aggregate gain, falsified below …
  - critic r2 (quality 0.78): The near-bound stratum is mis-sized for the mechanism. Truncation only matters within about 2 posterior sd of a bound. k1's sd is about 0.043 at k = 2 and abou… · Expected gain: if MMSE cuts squared error by about 20-30% in the ~14% truncated subset, the aggregate MSE gain is about 3-4%, and the rel_l2 gain is about half… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Interior sample with ESS/N about 0.9: the IS mean differs from the mode by MC noise of about sigma/sqrt(ESS). That adds about 1/115 to the relative MSE, a pure… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-005 — Hybrid learned-initializer + LM refinement reaches the Cramér–Rao regime (MMSE estimator)** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status conjectured. *H:* The inverse map is locally well-posed: Fisher condition number is about 62 and the CRB rel_l2 floor is about 0.0185. With a cheap learned initializer, a few Levenberg–Marquardt iterations in log-k space through the exact forward model should converge to the b…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the residual norm is about the noise level but the error is above 0.03, the CRB is optimistic, likely due to nonlinearity or bound effects, and the Bayesian…
  - critic r1 (quality 0.81): The CRB-ranking falsifier is good. However, an efficiency target of 1.5x the CRB floor (<0.03) still mixes estimator choice with initializer choice. · Feasibility notes mention 1500 samples (val and test), but the test split must not be touched until final evaluation. Only val should be timed and used. · Use this arm as the canonical merged hybrid, absorbing P-002 and P-007. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, llm] `hybrid[kernel_ridge/raw+logy+aug1->LM15]` status conjectured. *H:* The inverse map is locally well-posed: Fisher condition number about 62 and CRB rel_l2 floor about 0.0185. A kernel-ridge initializer lands inside the basin of the true log-k for at least 95% of val samples. Fifteen bounded LM iterations in log-k then reach n…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each failure mode points to a different cause. Residual chi^2/n about 1 but error above 0.03 means the CRB is optimistic because of nonlinearity or bounds, so …
  - response to critic: (1) Mixing estimator and initializer effects: accepted. This merged arm, which absorbs P-002 and P-007, will report three numbers from the same val samples: init-only rel_l2, refined mode (MLE) rel_l2, and refined posterior_mean rel_l2. Th…
  - critic r2 (quality 0.84): Of the arms in this round, this one is the most informative per second: init-only, mode and mean from one fit. Its MMSE part, though, makes P-012 largely redun… · Several RMSE-ranking claims are ambiguous between absolute and relative error. Absolute error scales with k, so state relative per-parameter RMSE. · Augmentation noise adds to noise already in the features. Training rows then have sqrt(2) sigma noise while val has sigma. That is fine for an initializer, but… · If KRR init-only rel_l2 is about 0.15 (raw-feature RBF in 800-D with noisy low-signal columns), basins may still be hit given the condition number. The [0.04, … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-020` r3 [revise, heuristic-fallback] `hybrid[kernel_ridge/raw+logy+aug1->LM10]` status conjectured. *H:* The inverse map is locally well-posed: Fisher condition number about 62 and CRB rel_l2 floor about 0.0185. A kernel-ridge initializer lands inside the basin of the true log-k for at least 95% of val samples. Fifteen bounded LM iterations in log-k then reach n…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each failure mode points to a different cause. Residual chi^2/n about 1 but error above 0.03 means the CRB is optimistic because of nonlinearity or bounds, so …
  - response to critic: Revised: reduced solver cost as requested
  - critic r3 (quality 0.834): The hypothesis says 15 LM iterations, but method.params.iters = 10. State which value is tested. The 'gap concentrated in…' prediction depends on it. · The posterior mean minimises E//k_hat-k//^2, not relative error. Under rel_l2 the Bayes-optimal point estimate is E[k///k//^2]/E[1///k//^2], which is shifted t… · iters must be set consistently (10 or 15) in both the hypothesis and the spec · If the KRR init is about 0.06 and lands in a k1/k4 flat valley, 10 damped LM steps may stall short of the MLE. Then mode equals posterior_mean and both sit abo… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-006 — Control: a purely data-driven noise-augmented MLP is limited by approximation and sample error, not by noise** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* This is a competing explanation test for where the error comes from. A torch MLP trained on 3000 simulated samples with noise augmentation will reach val rel_l2 between 0.03 and 0.06. That is clearly worse, by at least 30% relative, than the forward-model-ref…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If the MLP matches the hybrid, the problem is noise-limited at 3000 samples, and the cheaper amortized model is preferred for deployment. If the MLP is far wor…
  - critic r1 (quality 0.795): noise_scale=0.001 conflicts with P-003 and P-008, which use 1.0. If noise_scale is in units of sigma (or of standardized features), this arm has essentially no… · The falsification compares against 'the hybrid from proposal 1', but P-001 is pure physics. Fix the reference arm. · The '~0.2 per axis' spacing argument ignores the smoothness of the forward map. The interpolation error can be much smaller than the spacing suggests. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-014` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* An amortized torch MLP on raw noisy temperatures reaches val rel_l2 in [0.028, 0.055], i.e. 1.5-3x the CRB floor. It is worse than the best hybrid by at least 30% relative, with the gap concentrated in k1 and k4. That would mean the data-driven error is domin…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If the MLP matches the hybrid, the problem is noise-limited at n=3000 and the amortized model is preferable for deployment cost. If the MLP is much worse than …
  - response to critic: (1) noise_scale conflict: accepted. 0.001 was a units error under the likely semantics. I align with P-008 by setting noise_aug=2 and noise_scale=1.0, read as 1 x sigma pending the harness definition. Run this way, the arm becomes the raw-…
  - critic r2 (quality 0.64): The controlled contrast is invalid until the hyperparameters match P-015. · It is not a twin of anything in this round. P-014 is [256,256] gelu wd 1e-4, while P-015/P-008 is [512,256,128] silu wd 1e-5. Features, architecture, activatio… · The comparison references (P-005, 'best_hybrid', P-008) are not arms in this round. Name the paired reference explicitly (P-009 mode or P-013 mean). · Use exactly P-015's hyperparameters with features = raw, so this arm becomes the raw twin P-015 says is P-006. · Raw features may beat log on k1 and k4: early, near-zero edge-sensor readings carry the shielding signal, and log on noisy values near 0 destroys it.

**P-007 — Posterior-mean (MMSE) estimation beats MLE because rel_l2 is a squared-error loss and k1/k4 are weakly identified near …** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[extra_trees/raw+logy+aug1->LM20]` status conjectured. *H:* Since the metric is rel_l2 (an L2 loss), the Bayes-optimal estimator is the posterior mean, not the MLE/mode. For the weakly identified edge segments (k1, k4, CRB rel std ~2%), a hybrid estimator (learned initializer + LM refinement in log space + importance-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If it misses 0.03, the forward-model budget, local minima, or model mismatch dominates rather than estimator choice. If mode equals posterior_mean, the posteri…
  - critic r1 (quality 0.459): This is the same pipeline as P-002 and P-005 with an extra_trees init. Tree ensembles produce piecewise-constant, coarse predictions on an 800-dimensional inpu… · A 3% relative gain is the realistic MMSE size, but it is within the noise of an unpaired comparison on 750 samples. It requires paired evaluation. · The 3% threshold is the right magnitude and should be adopted by P-002 and P-004. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-008 — Physics-informed feature reformulation: amortized regression on log-temperatures with noise augmentation reaches near-C…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* The inverse map from sensor data to k is smooth but highly nonlinear in raw temperatures. Diffusion makes the signal decay roughly exponentially with rate proportional to k, so log temperature makes the map closer to linear in log k. A torch MLP on log featur…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If log features lose to raw, near-zero readings make the log transform a noise amplifier, and sqrt or raw is preferable. If both miss 0.04, the amortized appro…
  - critic r1 (quality 0.81): The heat-kernel linearisation is heuristic. With piecewise k and Dirichlet ends, late-time decay is governed by the first eigenvalue of the full operator, whic… · Early-time readings at the far sensors (indices 5 and 35) are about 0 relative to sigma=1e-3, so log is undefined or noise-amplifying. How nonpositive values a… · Consider sqrt or log(u + c*sigma) features as a safer middle ground. · If most information about k1 comes from early-time sensor-5 readings near the noise floor, log features discard it, and k1 error rises relative to raw. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-015` r2 [revise, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* With the full 3000-sample train split, a noise-augmented torch MLP on log features with a log target reaches val rel_l2 < 0.04, about 2x the CRB floor of 0.0185. It beats the identical raw-feature MLP (P-006) on a paired comparison. Its residual error is domi…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If log loses to raw, especially on k1 and k4, the critic's counterexample holds: early near-zero readings carry edge information that log discards, so prefer s…
  - response to critic: The pilot gave val rel_l2 = 0.0474 on 60 val samples, trained on only 300 samples (a tenth of the train split). That is already near 0.05 with ten times less data. So the log transform is not catastrophic in practice, whatever the harness …
  - critic r2 (quality 0.825): The paired twin P-006 is not in this round. P-014 is the candidate twin but uses different hyperparameters, so as specified the 'log beats raw' test cannot be … · The secondary claim (ratio >= 1.5) is weak. The CRB ratio k1/k2 is about 3.7, so even an approximation-limited learner will probably exceed 1.5. Test whether t… · The main test should be the shape diagnostic: per-parameter (MLP RMSE / CRB sd). Roughly equal ratios mean a noise-limited shape; unequal ratios mean approxima… · Going from 300 to 3000 samples does not guarantee error below 0.04. If error is limited by the effective noise on edge segments (sqrt(2) sigma training noise f… · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-021` r3 [revise, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* A noise-augmented torch MLP on log features with a log target, trained on all 3000 samples, reaches val rel_l2 in [0.025, 0.04], i.e. below 0.04. Its per-parameter error-to-CRB ratios r_j are NOT uniform: r for k1/k4 exceeds r for k2/k3 by more than 1.5x. Tha…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If rel_l2 >= 0.04 with large r spread, the learned inverse shrinks k1/k4 toward the centre. Forward refinement or a heteroscedastic loss is required. If r is u…
  - response to critic: (1) Missing twin. Agreed. P-006 is not in this round and P-014 uses different hyperparameters, so I withdraw the 'log beats raw' claim from the primary test. I now use a deliberately plain configuration (hidden [256,256], silu, lr 1e-3, wd…
  - critic r3 (quality 0.816): Non-uniform r_j does not by itself show an amortization deficiency. The Bayes-optimal amortized estimator (the posterior mean) shrinks weakly identified k1/k4 … · Log features: the Gaussian pulse at x=0.5 gives near-zero true temperatures at the x≈0.13 and x≈0.90 sensors at early times. With sigma=1e-3, noisy readings th… · The model is trained once and cached for P-022. Budget accounting should charge it once. · The exact posterior mean for a sample with k1 near 0.5 (shielded) has RMSE below the CRB in k1 but equal to the CRB in k2. That gives r_k2/r_k1 > 1.5 with zero… · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-010 — Deterministic truncated-Gaussian MMSE in k-space: isolates the box-prior truncation gain with zero extra forward evalua…** (researcher-mathematical, family physics-custom)
- v1 `P-010` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* At sigma = 1e-3 the posterior under the uniform box prior is well approximated by the linearised Gaussian likelihood N(theta_hat, sigma^2 (J^T J)^-1) truncated to [0.5, 2]^4. Here theta_hat is the unconstrained Gauss-Newton estimate in k-space. The pilot's ca…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.03, "expect": "better"}`; if it fails: There are three outcomes to distinguish: - Mean ≈ mode, with the mode arm matching P-001: near-bound truncation is negligible at this noise level. The clipped …
  - critic r2 (quality 0.855): Explicit FD stability: the unconstrained GN estimate, and its Jacobian perturbations, can step outside [0.5, 2]. If dt was chosen so that k_max dt/dx^2 is near… · MC noise: a 512-draw truncated-Gaussian mean adds about sigma^2/512 variance per sample, which slightly hurts interior samples. Use common random numbers and c… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · theta_hat for k1 = 2.06 with an unstable forward at 2.06: J is wrong, the truncated mean is wrong, and the arm loses exactly where it predicts gains. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-018` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* Hypothesis 1 (failure attribution). The pilot failure (0.079) came from evaluating the explicit-FD forward outside [0.5, 2] together with a k-space LM. With forward calls restricted to the box and LM run in log k, the mode arm of this code reproduces P-009: f…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.01, "expect": "better"}`; if it fails: Three outcomes to distinguish: - Mode arm still far worse than P-009: the custom_solver forward/feature alignment or the FD Jacobian is broken. Abandon the cus…
  - response to critic: The pilot is decisive negative evidence: rel_l2 0.0792 on n = 60, against 0.0190 for P-009 on the same pilot protocol. That is about 4x worse, so a 2-7% truncation effect cannot explain it. The solver itself failed.  The most likely cause …
  - critic r3 (quality 0.795): The previous version of this solver (P-010 pilot) scored 0.079, about 4x worse than physics MLE. The diagnosis (an out-of-box explicit-FD instability with a lo… · By the authors' own estimate, about 14% of samples are near a bound, which caps the aggregate gain at about 2%. The prediction 'improvement 0.01 relative to be… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the true k1 = 0.52 and the MLE clips at 0.5, the extrapolated m may land at 0.47. The truncated mean then lies above 0.5 and closer to the truth, which is a… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-016 — Amortization gap is a small-basin error: the log-MLP initializer plus only 5 LM steps reaches the CRB regime** (researcher-exploratory, family hybrid)
- v1 `P-016` r2 [propose, llm] `hybrid[torch_mlp/log+logy+aug2->LM5]` status conjectured. *H:* The amortized log-MLP from P-008 lands within the quadratic-convergence basin of the per-sample MLE. Its error (about 0.04) is approximation error, not identifiability error. So a very short refinement (5 LM iterations in log space, no extra starts, mode esti…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If 5 steps leave error well above 0.025 but P-003 reaches it with 15, convergence is slow along weak k1 and k4 directions, which suggests the problem is flat-v…
  - critic r2 (quality 0.655): Estimated 125.3 s is over the per-run limit unless the MLP is cached from P-015. · The rule estimate is 125.3 s, over the cap, because it retrains the P-015 MLP. · The paired reference P-003 is not in this round. Use P-009. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Damped LM with initial lambda tuned for far starts may take 2-3 steps to shrink lambda. Five iterations then fail for optimizer-schedule reasons, not geometry.
- v2 `P-022` r3 [revise, llm] `hybrid[torch_mlp/log+logy+aug2->LM5]` status conjectured. *H:* The amortized log-MLP initializer (identical to P-015) lies within the near-quadratic basin of the per-sample MLE. So 5 bounded log-space LM iterations, with no extra starts, bring val rel_l2 below 0.025. The result is within 10% of P-009's 30-iteration multi…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If iteration 5 misses but P-011 at iteration 8 and P-009 reach the floor, and the residual sits in k1/k4 with steps still shrinking, convergence is limited by …
  - response to critic: (1) Over the cap (125.3 s). Fixed. The init is now byte-identical to the revised P-015 spec (80 epochs, [256,256]). Without caching that costs about 50 s of training plus about 10-15 s of LM for 5 iterations, roughly 65 s in total. With ca…
  - critic r3 (quality 0.81): The rule estimate of 125.3 s assumes retraining. Together with P-021 (120 s), that is 245 of 400 s for the same MLP. The arm only makes sense if the init is ca… · The cross-arm trace references P-015, P-011 and P-009, none of which are in this round. The available paired references are P-021 (iteration 0), P-020 (KRR ini… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-019 — Is the amortization gap data-limited? Fresh simulated training data (4x) with a box-aware logit target, as the missing …** (researcher-ml, family physics-custom)
- v1 `P-019` r3 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The gap between amortized MLPs (P-014/P-015) and the CRB-level physics MLE (P-009, about 0.019) is mostly estimation error from only 3000 training pairs, not intrinsic MLP approximation error. Augmenting with 9000 fresh prior draws simulated through context.f…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.2, "expect": "better"}`; if it fails: If the gain is <20% with a tight CI around 0, the amortization gap is capacity- or optimisation-limited, not data-limited. Effort should then go to forward-mod…
  - critic r3 (quality 0.852): Confound: with fixed epochs=80, 12000 samples get 4x more gradient steps than the n_new=0 twin. A gain could come from optimisation rather than data. Either ma… · The logit target is a second change relative to P-021's log target. The twin isolates data size, but cross-comparison with P-021 is confounded, so do not attri… · The fresh simulated features use the same noise std (sigma), so train and val noise match. This is unlike noise_aug. · If the gain is 25% but the step-matched twin also gains 15%, the data-limitation claim holds only weakly. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: The proposals are specific and falsifiable, but the portfolio is not ready. P-003 is blocked by its time estimate (135.9s > 120s). P-002 and P-007 duplicate P-005. The total budget is exceeded. Several thresholds are miscalibrated: P-001's chi^2 criterion is too loose, and the 10% MMSE gain is likely unattainable. The highest-value action is the cheap oracle chi^2 pilot, followed by the exact P-0…  families={'physics': 2, 'hybrid': 4, 'data-driven': 2}
- Critic round 2 portfolio view: The success criterion is already met: the physics MLE pilot scored 0.019, about 1.03x the CRB. What remains is efficiency and estimator science. Recommended budget-feasible slate (about 350 s): P-009 full val with saved fits (23 s); P-010 with a stability probe and CRN-centred draws, also applied post-hoc to P-009 (30 s); P-013 with extra_starts = 0 (about 45 s); P-015 and P-014 with aligned hype…  families={'physics': 2, 'hybrid': 3, 'data-driven': 2, 'physics-custom': 1}
- Critic round 3 portfolio view: P-017 is the anchor and is well specified; its only fix is metric-consistent, interior-stratified floor accounting. P-019 is the cleanest new control but needs a step-matched twin. P-021's shape claim needs an MMSE comparator and a log-feature sanity check, and its scaling claim should defer to P-019. P-018 must re-pilot after its 0.079 failure before taking full budget. P-022 is only worth runni…  families={'physics': 1, 'hybrid': 2, 'data-driven': 1, 'physics-custom': 2}

**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MLE[log,ms=3,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.72 s
- P-008 `torch_mlp/log+logy+aug2`: ok rel_l2=0.047425 on 60 val samples in 11.2 s
- P-010 `custom-solver(fwd)`: ok rel_l2=0.07925 on 60 val samples in 2.87 s

## 5. Literature retrieval (TheoremSearch)

0 queries, 24 failures, 0 distinct statements retrieved, 0 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-experimental | 1 | uniqueness and Lipschitz stability for recovering piecewise constant conductivity in one-… | FAILED: disabled |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-exploratory | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | FAILED: disabled |
| researcher-exploratory | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for finite-dimensional nonlinear least… | FAILED: disabled |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for identification of piecewise constant conductivity … | FAILED: disabled |
| researcher-mathematical | 1 | local convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares para… | FAILED: disabled |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-ml | 1 | convergence of Gauss-Newton or Levenberg-Marquardt iteration for nonlinear least squares … | FAILED: disabled |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 2 | chi^2/n of forward(k_true) vs val features over all 750 val samples | FAILED: disabled |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | Fraction of P-001 samples whose multistart solutions disagree | FAILED: disabled |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 2 | Init-only val rel_l2 of the kernel ridge | FAILED: disabled |
| researcher-exploratory | 2 | MMSE vs MLE for bounded parameters in inverse heat conduction | FAILED: disabled |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 3 | Per-sample CRB distribution over val (mean, 90th and 99th percentile of the rel std for k… | FAILED: disabled |
| researcher-ml | 3 | P-009 per-sample iteration count to convergence and the 3-start spread | FAILED: disabled |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 3 | Per-sample paired difference vs P-009 mode | FAILED: disabled |
| researcher-exploratory | 3 | Learning curve at 300 / 1000 / 3000 training samples | FAILED: disabled |
| researcher-exploratory | 3 | fraction of train feature entries with /u/ < 3 sigma per sensor | FAILED: disabled |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_literature_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 105 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-017) | physics | `physics-MLE[log,ms=3,it=30]` | researcher-mathematical | 0.888 | best-rated proposal in family 'physics' (quality 0.89) |
| E-02 | proposal (P-019) | physics-custom | `custom-solver(fwd)` | researcher-ml | 0.852 | best-rated proposal in family 'physics-custom' (quality 0.85) |
| E-03 | proposal (P-020) | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM10]` | researcher-experimental | 0.834 | best-rated proposal in family 'hybrid' (quality 0.83) |
| E-04 | proposal (P-021) | data-driven | `torch_mlp/log+logy+aug2` | researcher-exploratory | 0.816 | best-rated proposal in family 'data-driven' (quality 0.82) |
| E-05 | proposal (P-022) | hybrid | `hybrid[torch_mlp/log+logy+aug2->LM5]` | researcher-exploratory | 0.81 | quality 0.81 with diversity penalty |
| E-06 | proposal (P-018) | physics-custom | `custom-solver(fwd)` | researcher-mathematical | 0.795 | quality 0.80 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-012 | E-03 | refine | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM9]` | - | **0.01911** | 0.9966 | 0.0252 | 4.49 | 14.4 | 143475 | 0.947 | FINAL |
| idea-011 | E-03 | refine | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM10]` | - | **0.019111** | 0.9966 | 0.0252 | 4.52 | 9.73 | 92425 | 0.947 | FINAL |
| idea-004 | E-03 | research | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM10]` | 0.02049 | **0.019111** | 0.9966 | 0.0252 | 6.59 | 15 | 144015 | 0.947 | evaluated |
| idea-010 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0098 | 9.9 | 61517 | 0.957 | FINAL |
| idea-002 | E-01 | research | physics | `physics-MLE[log,ms=3,it=30]` | 0.02139 | **0.019337** | 0.9966 | 0.0255 | 0.00914 | 22.9 | 175409 | 0.957 | evaluated |
| idea-003 | E-02 | research | physics-custom | `custom-solver(fwd)` | 0.02718 | **0.026151** | 0.9937 | 0.03449 | 29.6 | 0.0111 | 9000 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 5.65 | 0.452 | 0 | - | evaluated |
| idea-014 | E-08 | refine | data-driven | `extra_trees/sqrt` | - | **0.08611** | 0.9317 | 0.1136 | 3.52 | 0.138 | 0 | - | evaluated |
| idea-013 | E-08 | refine | data-driven | `extra_trees/sqrt` | - | **0.10373** | 0.9009 | 0.1368 | 2.32 | 0.328 | 0 | - | evaluated |
| idea-005 | E-04 | research | data-driven | `torch_mlp/log+logy+aug2` | 0.04054 | **-** | - | - | - | - | - | - | ok |
| idea-006 | E-05 | research | hybrid | `hybrid[torch_mlp/log+logy+aug2->LM5]` | 0.02139 | **-** | - | - | - | - | - | - | ok |
| idea-007 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **-** | - | - | - | - | - | - | ok |
| idea-008 | E-08 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 220s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-012` `hybrid[kernel_ridge/raw+logy+aug1->LM9]` ← refined from idea-004 ← P-020 v3 (researcher-experimental, r3, revise) ← P-013 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)
- `idea-010` `physics-MLE[log,ms=1,it=30]` ← baseline/control (physics baseline: Levenberg-Marquardt MLE (defaults))
- `idea-011` `hybrid[kernel_ridge/raw+logy+aug1->LM10]` ← refined from idea-004 ← P-020 v3 (researcher-experimental, r3, revise) ← P-013 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00214` [checkpoint] by scheduler: final method (ensemble)
- `resu-00211` [result] by executor: exp-0023 rel_l2=0.01834 r2=0.997 rmse=0.02413
- `hypo-00181` [hypothesis] by scheduler: idea-012 hybrid[kernel_ridge/raw+logy+aug1->LM9] (refine)
- `hypo-00125` [hypothesis] by scheduler: idea-010 physics-MLE[log,ms=1,it=30] (baseline)
- `hypo-00179` [hypothesis] by scheduler: idea-011 hybrid[kernel_ridge/raw+logy+aug1->LM10] (refine)
- `plan-00210` [plan] by engineer-v2: exp-0023: final fit of 3 member(s) on train+val; test once
- `gate-00180` [gate] by gate: refine idea-004: approved (hybrid[kernel_ridge/raw+logy+aug1->LM9])
- `prop-00098` [proposal] by researcher-experimental: P-020 v3 [revise] experimental: Merged hybrid (kernel-ridge init -> bounded log-LM) with init-only, mode and posterior-mea
- `hypo-00111` [hypothesis] by scheduler: idea-004 hybrid[kernel_ridge/raw+logy+aug1->LM10] (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_literature_s1 (9f0f28a432)
- `gate-00124` [gate] by gate: E-09: approved (physics-MLE[log,ms=1,it=30])
- `gate-00178` [gate] by gate: refine idea-004: approved (hybrid[kernel_ridge/raw+logy+aug1->LM10])
- `prop-00070` [proposal] by researcher-experimental: P-013 v2 [revise] experimental: Merged hybrid (kernel-ridge init -> bounded log-LM) with init-only, mode and posterior-mea
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00110` [gate] by gate: E-03: approved (hybrid[kernel_ridge/raw+logy+aug1->LM10])
- `prop-00039` [proposal] by researcher-experimental: P-005 v1 [propose] experimental: Hybrid learned-initializer + LM refinement reaches the Cramér–Rao regime (MMSE estimator)

## 9. Compute and cost

- Execution compute: 301.6 / 400 worker-seconds over 23 jobs; wall 707 s
- By stage: baseline 2.0s, pilot 16.8s, screen 77.2s, evaluate 95.0s, refine 42.3s, final 68.3s
- LLM: 19 calls (by phase {'research': 19}), 1016 s, $3.514 reported · TheoremSearch: 0 queries (0 failed)

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

The evidence establishes that physics-based inversion solves this problem and pure data-driven regression does not. Multistart Levenberg–Marquardt maximum likelihood in log-conductivity (physics-MLE) reached val rel_l2 = 0.0193. The kernel-ridge-initialised hybrid reached 0.0191, and the custom forward solver reached 0.0262. All three beat the 0.05 success threshold and their pre-registered predictions (< 0.025, < 0.03, and ≥ 20% gain over data-driven; the custom solver's measured gain was 68%). The best data-driven model (extra_trees, 0.0825) missed the threshold. The final three-member ensemble generalised without degradation: val 0.0190, held-out test rel_l2 0.0183 (R² 0.997, RMSE 0.024). The test split was used only for this final report. The hybrid's edge over plain MLE is about 1% relative, and no seed variance or paired confidence interval is reported, so the two should be treated as statistically indistinguishable. The pilot diagnostics support a noise-limited regime: error ≈ 1.03× the Cramér–Rao bound and χ²/dof ≈ 0.94, a residual level consistent with the assumed noise model. This suggests the remaining error comes from measurement noise, not optimiser basin failures. That conclusion is still an inference from one pilot, not a dedicated test.

No hypothesis was contradicted. The five abandoned hypotheses were dropped for procedural reasons, not falsified. Three were duplicates of merged arms (P-005, P-016). One had an expected effect of about 1.5–2%, too close to the falsification threshold to discriminate. One control, raw-feature MLP vs the forward-refined hybrid, was blocked because its hyperparameters did not match P-015. The recorded negative result is that learned initialisation does not improve accuracy. It rests on the CRB argument, not on a direct paired experiment, so it is well-motivated conjecture. Three questions remain open:
- whether posterior-mean (MMSE) estimation gives measurable gains for the weakly identified, near-bound k1/k4;
- whether amortised initialisation buys speed at fixed accuracy;
- whether the near-CRB result holds under noise-model misspecification.

- **Seed-replicated paired comparison:** run physics-MLE vs hybrid on identical val instances with bootstrap CIs, to decide whether the ~1% hybrid gain is real.
- **Direct CRB/basin audit:** compute per-sample CRB and the fraction of multistart runs converging to non-global minima, to confirm or refute the "noise-limited, not basin-limited" claim.
- **Bound-stratified MMSE test (P-010/P-013):** compare exact truncated-Gaussian or importance-sampled posterior mean vs mode, reporting k1/k4 error stratified by distance to the box edge.
- **Iteration-trace amortisation study (P-016):** measure accuracy vs LM iteration count for learned vs multistart initialisation, and wall-clock break-even N, under the per-run time cap.
- **Robustness sweep:** test noise-level misspecification, sensor-position jitter, and out-of-prior k values on simulated data only, to check whether the near-CRB performance holds outside the training distribution.

_Automatic summary:_ 22 proposal versions over 3 research rounds; 9 methods advanced; 3 predictions supported, 0 contradicted. Final: ['hybrid[kernel_ridge/raw+logy+aug1->LM9]', 'physics-MLE[log,ms=1,it=30]', 'hybrid[kernel_ridge/raw+logy+aug1->LM10]'] with test rel_l2 0.018339.
