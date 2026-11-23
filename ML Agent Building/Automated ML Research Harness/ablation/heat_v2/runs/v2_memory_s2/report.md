# Research report (V2): heat_v2_memory_s2

Run `heat_v2_memory_s2_20261109_154841` · 2026-11-09 16:05:51 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MLE[log,ms=1,it=30]`
- Validation rel_l2 (selection estimate): **0.019335** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018419**, r2 = **0.99696**, rmse = **0.024236** · bootstrap 95% CI for rel_l2: [0.01725, 0.01962]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.12494 (vs 0.018419 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.052029 (vs 0.018419 in distribution)
- Uncertainty calibration on test: 70.1% within 1σ, 95.6% within 2σ (ideal 68% / 95%)
- Inference latency: 7.93 ms/sample; fit 0.0107 s; exported model reload check: max |Δ| = 0
- ⚠ This test split had already been evaluated 2 time(s) in earlier runs (research memory): repeated development weakens its independence.
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_memory_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MLE[log,ms=1,it=30]` rel_l2=0.019335; **data-driven** `extra_trees/sqrt+pca16` rel_l2=0.10795
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:19464924` M. S. Aronna, J. M. Machado (2021). "The shooting algorithm for partially control-affine problems with application to an SIRS epidemiological model". Proposition 6.2 (\cite{bonnans2006numerical,fletcher2013practical}). http://arxiv.org/abs/2103.06339v1 — cited by ['P-001', 'P-002', 'P-009']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22602562` Guillaume Bal, Gunther Uhlmann (2011). "Reconstruction of coefficients in scalar second-order elliptic equations from knowledge of their solutions". Theorem 2.1. http://arxiv.org/abs/1111.5051v1 — cited by ['P-002', 'P-009']; researcher claim ['no']; harness screen **plausible**
- `ts:thm:24457901` Kendrick Qijun Li, Kenneth Martin Rice (2023). "A Bayesian 'sandwich' for variance estimation". Proposition 2.1. http://arxiv.org/abs/2207.00100v2 — cited by ['P-003', 'P-004', 'P-010']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26204848` Badr-Eddine Chérief-Abdellatif (2019). "Convergence Rates of Variational Inference in Sparse Deep Learning". Theorem 5. http://arxiv.org/abs/1908.04847v2 — cited by ['P-004']; researcher claim ['unknown']; harness screen **unknown**
- `ts:thm:25580267` Prithwish Bhaumik, Subhashis Ghosal (2016). "Efficient Bayesian estimation and uncertainty quantification in ordinary differential equation models". Corollary. http://arxiv.org/abs/1411.1166v2 — cited by ['P-005', 'P-006', 'P-011', 'P-013']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:21617852` Prithwish Bhaumik, Subhashis Ghosal (2015). "Bayesian inference for higher order ordinary differential equation models". Corollary 2. http://arxiv.org/abs/1505.04242v1 — cited by ['P-005']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22398569` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 9. http://arxiv.org/abs/1210.7346v1 — cited by ['P-007']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:24839008` Andre Wibisono, Varun Jog, Po-Ling Loh (2017). "Information and estimation in Fokker-Planck channels". Theorem 3. http://arxiv.org/abs/1702.03656v1 — cited by ['P-011', 'P-013']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26360863` Christian Kanzow, Theresa Lechner (2022). "Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems". Proposition. http://arxiv.org/abs/2210.07644v1 — cited by ['P-016']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-01 *Factorial cell B (canonical model): MLP on forward-regenerated clean traces with fresh sigma=0.001 noise per minibatch,…*: inconclusive — method did not produce a full-fidelity validation result
- E-02 *Hybrid budget control: canonical MLP initializer + single-start LM posterior-mean refinement matches multistart physics…*: inconclusive — method did not produce a full-fidelity validation result
- E-03 *Amortised plateau as estimation error: raw-trace, box-bounded torch MLP with simulation augmentation, as an n_sim learn…*: inconclusive — method did not produce a full-fidelity validation result
- E-04 *Cheap hybrid: bounded MLP initializer followed by single-start short LM and Laplace-IS posterior mean, to match physics…*: inconclusive — method did not produce a full-fidelity validation result

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-mathematical — P-001 *Learned initializer + Gauss-Newton/LM posterior-mean refinement reaches near the Cramér–Rao floor*: blocking issue: The noise_scale=0.001 inconsistency must be resolved, or declared intentional, before the run. Otherwise the comparison with the sibling hybrids is confounded.
- ABANDONED in round 2 by researcher-ml — P-003 *Amortised MLP initializer + posterior-mean LM refinement (hybrid) attains near-CRB accuracy*: This is a duplicate of P-001 and P-005 (same hybrid torch_mlp->LM15 posterior-mean design). Its predictions are strictly weaker and its comparison against the initializer is available from P-001's logged init-only metri…
- ABANDONED in round 2 by researcher-experimental — P-005 *Metric-matched hybrid: noise-augmented MLP initializer plus LM refinement with posterior-mean reado…*: Duplicate of P-001 (critic: duplicate_of P-001). Merged: its falsifiable_test is offered as the canonical test for the P-001 hybrid run. No negative scientific result is implied.
- ABANDONED in round 2 by researcher-exploratory — P-007 *Truncated-prior MMSE can undercut the CRB 'floor': amortised init plus a forward-model posterior-me…*: Consolidated into P-001 to save budget. There is a back-of-envelope argument that the box-truncation gain over MLE is under 2%, because few samples lie within 2 posterior std of a bound. The beat-the-CRB claim is better…
- ABANDONED in round 3 by researcher-mathematical — P-009 *Global identifiability on the box: forward-only multistart MLE/MMSE without any learning matches th…*: blocking issue: The prediction references the nonexistent 'best_hybrid'. Restate it against an absolute threshold or against P-002.
- ABANDONED in round 3 by researcher-ml — P-010 *Feature-basis test for amortised MMSE: PCA-48 versus raw traces (paired with P-006), with a per-par…*: blocking issue: The 120 s estimate equals the experiment limit. Reduce epochs or aug before running.
- PRUNED — idea-003 `hybrid[torch_mlp/raw+logy->LM10]`: projected full-fidelity cost 158s exceeds the 120s per-experiment limit
- PRUNED — idea-005 `hybrid[torch_mlp/raw+logy->LM8]`: projected full-fidelity cost 146s exceeds the 120s per-experiment limit
- PRUNED — idea-006 `torch_mlp/raw`: projected full-fidelity cost 141s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Factorial cell B (canonical model): MLP on forward-regenera…): challenge: The mechanism is mislabelled. The given train X is already one draw from the exact val/test measurement model (P-002: residual at truth 0.000995 ≈ sigma). Cell A is therefore not trained on a mismatched noise…
- E-01 (Factorial cell B (canonical model): MLP on forward-regenera…): challenge: The success prediction is too weak to be informative. P-004 already reached 0.0478 with only 300 train samples, and memory m-261109-f27c4024 reports an amortised torch model at 0.0261. 'B < 0.05' is nearly gu…
- E-01 (Factorial cell B (canonical model): MLP on forward-regenera…): challenge: This arm overlaps heavily with P-015's n_sim=0 arm and with P-011/P-012. If P-015's n_sim=0 arm also re-noises the real-train thetas each epoch, it is effectively cell B with a different architecture. Two nea…
- E-01 (Factorial cell B (canonical model): MLP on forward-regenera…): challenge: The 10% holdout shrinks training to 2700 thetas for both cells. That is acceptable for the A/B contrast. It is not acceptable for comparison with P-015 or the library baselines, which use 3000.
- E-02 (Hybrid budget control: canonical MLP initializer + single-s…): challenge: The comparator is not established at full fidelity. P-009/P-002 has run only on 60 val samples. The '5% of P-009' and '25% of evaluations' criteria need the full 750-val P-009 run, with per-sample results, to…
- E-02 (Hybrid budget control: canonical MLP initializer + single-s…): challenge: The cost accounting is likely dominated by importance sampling, not LM. P-002 used 26943 forwards per 60 samples, about 450 per sample. With finite-difference Jacobians (about 5 forwards per LM iteration for …
- E-02 (Hybrid budget control: canonical MLP initializer + single-s…): challenge: Wall-clock is the binding resource, not forward count. P-002 took 5.2 s for 60 samples, roughly 65 s for 750. MLP training at 15-25 s plus refinement may save little or nothing in worker-seconds. Report wall-…
- E-02 (Hybrid budget control: canonical MLP initializer + single-s…): challenge: Accuracy equivalence is nearly tautological when LM converges to the same mode: the Laplace-IS posterior mean is then identical. The real test is whether the per-sample converged mode matches P-009's. Measure…
- E-03 (Amortised plateau as estimation error: raw-trace, box-bound…): challenge: The physics posterior mean is the MMSE estimator, so the amortised floor is physics MMSE, not the CRB. Under squared loss in k-space and the true prior, no amortised net can beat the Bayes posterior mean in e…
- E-03 (Amortised plateau as estimation error: raw-trace, box-bound…): challenge: With fixed epochs=60, the n_sim axis is confounded with the number of optimisation steps: steps grow roughly 1:3:5 across arms. A monotone curve could reflect more SGD steps rather than less estimation error.…
- E-03 (Amortised plateau as estimation error: raw-trace, box-bound…): challenge: The P-008 pilot (0.0415 with 12300 samples) is worse than the memory model (0.026 with 3000 samples). If the data were the bottleneck, this ordering would be reversed. It points to the sklearn optimiser or ar…
- E-03 (Amortised plateau as estimation error: raw-trace, box-bound…): challenge: With three points and one seed, 'monotone' is cheap to satisfy by chance. Require that the 0→12000 improvement CI excludes zero.
- E-04 (Cheap hybrid: bounded MLP initializer followed by single-st…): challenge: This is near-duplicate of P-014: same method kind and same init architecture, differing only in iterations (8 vs 10), epochs (150 vs 120) and IS samples (128 vs 64). Running both spends about 55 s of the 400 …
- E-04 (Cheap hybrid: bounded MLP initializer followed by single-st…): challenge: The cost criterion is likely self-defeating. 128 IS forwards plus about 40 LM forwards is roughly 170 per sample, about 37% of P-009's ~450. That fails the '≤30%' criterion by construction. It would also doub…
- E-04 (Cheap hybrid: bounded MLP initializer followed by single-st…): challenge: The '< 0.0195' target is set from a 60-sample pilot of P-009. On full val the comparator may move by more than 3%, so the threshold must be defined relative to the full-val paired P-009 run.
- E-04 (Cheap hybrid: bounded MLP initializer followed by single-st…): challenge: The 0.026 init-quality claim comes from a different amortised config (memory m-261109-f27c4024, likely with PCA and augmentation). The init here is raw, pca=0, no aug. P-004 at 300 samples gave 0.048, so the …
- Unresolved disagreement ['P-014', 'P-016']: Initializer quality for nearly identical configs is assumed to be 0.05-0.1 vs 0.026. This is unmeasured; an init-only val score settles it.
- Unresolved disagreement ['P-013', 'P-015']: P-013 attributes amortisation error to noise-model mismatch. P-015 attributes it to theta scarcity or estimation error. Note also that train X is already matched, so P-013's mismatch framing is weak. The merged factoria…
- Unresolved disagreement ['P-015', 'critic']: P-015 suggests the amortised net can go below the CRB-based 0.0186 floor. The critic holds that the relevant floor is physics posterior-mean MMSE (≈0.0186 on pilot), which the amortised net cannot beat in expectation ex…
- Test-set sampling uncertainty: bootstrap 95% CI [0.01725, 0.01962] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | abandon | P-001 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-mathematical | revise | P-002 | P-009 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-ml | abandon | P-003 | - | llm | I accept both challenges. (1) P-003 matches P-001 and P-005 in method family, initializer, LM settings and re… |
| 2 | researcher-ml | revise | P-004 | P-010 | llm | Pilot evidence: with 300 training samples and 60 val samples, PCA-48 + log-y + aug4 reached rel_l2 0.0478 in … |
| 2 | researcher-experimental | abandon | P-005 | - | llm | Accepted: the spec duplicates P-001/P-003, so a separate run is a waste of the 400 s budget. I ask that P-001… |
| 2 | researcher-experimental | revise | P-006 | P-011 | llm | All points accepted.   (1) Noise semantics block interpretation. The revised method no longer uses library no… |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | I accept all three critic challenges. (1) Truncation is rarely active. The posterior std for k1 and k4 is abo… |
| 2 | researcher-exploratory | revise | P-008 | P-012 | llm | The pilot is evidence and it outweighs my argument. With 12000 forward evaluations (augmentation ran, forward… |
| 3 | researcher-mathematical | abandon | P-009 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 3 | researcher-ml | abandon | P-010 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 3 | researcher-experimental | revise | P-011 | P-013 | llm | Point-by-point responses.  (1) r/sigma is largely settled by P-002. Accepted. P-002's residual at truth (abou… |
| 3 | researcher-experimental | propose | - | P-014 | llm |  |
| 3 | researcher-exploratory | revise | P-012 | P-015 | llm | Point-by-point.  (C1) Layout artefact: accepted. P-002's model_mismatch=false (residual at truth about sigma … |
| 3 | researcher-exploratory | propose | - | P-016 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.754 | fdbk-00044 |
| 2 | 4 | 0.778 | fdbk-00078 |
| 3 | 4 | 0.741 | fdbk-00098 |

### Idea evolution

**P-001 — Learned initializer + Gauss-Newton/LM posterior-mean refinement reaches near the Cramér–Rao floor** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status literature-supported. *H:* A noise-augmented MLP gives an initial estimate inside the local basin of the true k. Five to fifteen bounded LM iterations in log-k space, followed by importance-sampled posterior-mean estimation, will then reach val rel_l2 < 0.03. That is within about 1.6x …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the initializer alone is already near 0.02, the learned map is effectively MMSE-optimal and refinement is unnecessary. If refinement leaves high-residual ou…
  - critic r1 (quality 0.776): The noise_scale=0.001 inconsistency must be resolved, or declared intentional, before the run. Otherwise the comparison with the sibling hybrids is confounded. · noise_scale=0.001 differs from P-003, P-005 and P-007, which use 1.0. If noise_scale is in units of the stated sigma, as P-006 assumes, this run augments with … · The 'k1/k4 interchange symmetry' given as a source of spurious minima is not a real non-identifiability. The sensors are labelled: index 5 is in segment 1 and … · Set noise_scale to 1.0, or confirm its units. Otherwise state explicitly that the initializer is trained without augmentation. · If the MLP init is already about 0.02, a 30% gain is impossible, yet the hybrid could still be optimal. This outcome would falsify the hypothesis on a technica…

**P-002 — Global identifiability on the box: forward-only multistart MLE/MMSE without any learning matches the hybrid** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, llm] `physics-MMSE[log,ms=4,it=30]` status conjectured. *H:* For this 4-parameter problem the data-misfit functional //F(k) - y//^2 has a single basin on [0.5,2]^4 at sigma=1e-3. If so, a purely physics-based posterior-mean estimator started from the prior centre plus a few random starts reaches val rel_l2 < 0.03. It s…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.1, "expect": "worse"}`; if it fails: If it is much worse than the hybrid and has a heavy error tail, the landscape is multimodal or has flat valleys. That justifies learned initialisation and poss…
  - critic r1 (quality 0.94): Four starts may cover all basins, but the claimed 'triangular sensitivity' argument is heuristic. Low k2 or k3 shadows the outer segments, and the 32-draw medi… · Under tikhonov=0, MLE iterates can leave the box along weak directions (k1, k4). The fix should be stated: hard clipping, log-barrier or projection. That choic… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · k2 close to 0.5 with k1 near 2.0: little heat reaches segment 1 within the short window (T ≲ 0.03 under CFL with 200 steps). The k1 direction could then be nea… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, heuristic-fallback] `physics-MMSE[log,ms=4,it=30]` status conjectured. *H:* For this 4-parameter problem the data-misfit functional //F(k) - y//^2 has a single basin on [0.5,2]^4 at sigma=1e-3. If so, a purely physics-based posterior-mean estimator started from the prior centre plus a few random starts reaches val rel_l2 < 0.03. It s…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.1, "expect": "worse"}`; if it fails: If it is much worse than the hybrid and has a heavy error tail, the landscape is multimodal or has flat valleys. That justifies learned initialisation and poss…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.725): The prediction references the nonexistent 'best_hybrid'. Restate it against an absolute threshold or against P-002. · Method params (log, multistart=4, iters=30, posterior_mean) match the P-002 pilot exactly. That pilot already gave 0.0186 on 60 val samples, with chi2/dof 0.94… · The prediction is expressed relative to 'best_hybrid', but no hybrid result exists. With a reference that does not exist, the 'within 10%' and 'worse' predicti… · Replace 'compare to hybrid' with absolute criteria: full-val rel_l2 < 0.025, and fewer than 2% of samples with per-sample rel error > 0.1. · If k2 and k3 are near 0.5, little heat reaches the outer segments before Dirichlet loss. k1 and k4 could then trade off along a near-flat ridge with no second …

**P-003 — Amortised MLP initializer + posterior-mean LM refinement (hybrid) attains near-CRB accuracy** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status conjectured. *H:* A noise-augmented torch MLP trained on log-targets provides an initial estimate inside the basin of attraction of the per-sample least-squares optimum; a short bounded LM refinement in log-space through the forward model, followed by the importance-sampled po…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the hybrid gains little over the MLP, the amortised map is already close to MMSE and the residual error is noise-limited. If the hybrid is worse than the ML…
  - critic r1 (quality 0.492): This is essentially identical to P-005 and to P-001 (once P-001's noise_scale is corrected). The only differences are epochs=200 and a 25% threshold. · The prediction <0.035 is weaker than P-001's <0.03 and therefore less informative. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-004 — Amortised MMSE: a noise-augmented MLP on PCA-compressed traces reaches the 0.05 target without any forward-model calls …** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+pca48+logy+aug4` status conjectured. *H:* A squared-loss MLP trained on simulated data, with on-the-fly noise augmentation at the true sigma, approximates E[k / y]. With PCA-48 whitened features it achieves val rel_l2 below 0.05 without any per-sample physics. We expect it to remain at least 1.3x wor…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 is at or above 0.05, the amortised map is limited by data or approximation error, not by noise. Physics refinement (proposal 1) or simulating ext…
  - critic r1 (quality 0.865): At 200 s it is the most expensive proposal and uses half the 400 s budget. It also tests the same question as P-006 (whether amortised MLP reaches <0.05) and i… · The predicted '1.3x worse than hybrid' is mostly a prediction about the hybrid. It cannot be checked without it. · Cut epochs or noise_aug to bring the cost to ≤100 s, or merge with P-006 as a PCA-versus-raw arm. · If the k1 and k4 sensitivity lies in low-variance PCA components, PCA-48 inflates exactly the error the proposal expects to dominate. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-010` r2 [revise, llm] `torch_mlp/raw+pca48+logy+aug2` status conjectured. *H:* At full training size (3000) with matched noise augmentation, a squared-loss MLP on PCA-48 whitened features reaches val rel_l2 < 0.035. It does not lose accuracy relative to the raw-feature MLP in P-006: it should be within 5% of it, or better, because it re…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the arm is at or above 0.035 but P-006 (raw) is below it, PCA is discarding information. We would then check per-parameter errors to confirm whether k1 and …
  - response to critic: Pilot evidence: with 300 training samples and 60 val samples, PCA-48 + log-y + aug4 reached rel_l2 0.0478 in 26.9 s. This already meets the target at one tenth of the data, so the amortised arm is not hopeless. The pilot is low-fidelity an…
  - critic r2 (quality 0.71): The 120 s estimate equals the experiment limit. Reduce epochs or aug before running. · The estimate of 120 s sits at the per-experiment limit, so a timeout is likely. Use epochs of 100 or fewer, or noise_aug=1. · noise_aug semantics are still unconfirmed. If the train X is already noisy (P-002's residual at truth ≈ sigma suggests the simulator's data carry noise), aug=2… · Set noise_aug to match P-006 exactly, so that PCA is the only variable. · Early-time outer-sensor signals have small amplitude but high Fisher content. They can sit below PC 48 while dominating the k1/k4 information.

**P-005 — Metric-matched hybrid: noise-augmented MLP initializer plus LM refinement with posterior-mean readout reaches near the …** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status literature-supported. *H:* Sensor noise (sigma=0.001) is small relative to the sensor-signal sensitivity, so the per-sample posterior is close to Gaussian over most of the prior box. In that regime a learned initializer followed by a few bounded log-space Levenberg-Marquardt iterations…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If chi^2 outliers dominate, the landscape is multimodal or the initializer is poor, so more extra_starts are needed. If chi^2 is fine but error stays above 0.0…
  - critic r1 (quality 0.501): Same spec as P-003 (epochs 150 instead of 200) and as P-001 once noise_scale is fixed. Its falsification list (chi^2 outliers, per-parameter RMSE, mode compari… · 'Apply refinement to val/test only' is correct and should be the default for every hybrid. Refining 3000 train samples wastes budget. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-006 — Noise-model fidelity control: an amortised MLP is only as good as its training noise match; matched-noise augmentation …** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* A squared-loss network trained on simulated traces corrupted with noise of the correct statistics (i.i.d. Gaussian, sigma=0.001, matched to the test features) approximates the posterior mean and reaches val rel_l2 < 0.05 without the forward model at inference…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If both arms are above 0.05 with a negligible augmentation gain, the bottleneck is sample size or capacity; the forward model should then be used to simulate m…
  - critic r1 (quality 0.746): The noise_aug and noise_scale semantics must be confirmed. Without them the paired test is uninterpretable. · If the train features already contain one noise draw at sigma=1e-3, adding augmentation noise on top trains on sigma≈1.41e-3. That is a mild mismatch which cou… · 'Necessary and sufficient' is too strong. The design tests only whether augmentation helps by 10%. It cannot show sufficiency, and it overlaps with P-008's com… · Add an arm that trains on clean forward outputs plus fresh noise (via context.forward). That is the only arm with a truly matched noise level. · If augmentation adds noise to noisy features, the augmented arm can be worse even when matched noise would help. That would falsify the hypothesis for an artef…
- v2 `P-011` r2 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* When the training noise exactly matches the measurement model (clean simulator output plus fresh i.i.d. N(0, 1e-3) noise on each pass), a squared-loss MLP on log-targets approximates the posterior mean well enough to reach val rel_l2 < 0.05 with the same 3000…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 >= 0.05 and the gain over noise_aug=0 is negligible while P-008 improves, the amortised gap comes from data scarcity, and the forward model should be…
  - response to critic: All points accepted.   (1) Noise semantics block interpretation. The revised method no longer uses library noise_aug. A custom_solver regenerates clean traces for the 3000 train thetas via context.forward and adds fresh N(0, sigma=0.001) n…
  - critic r2 (quality 0.886): P-002's fit_report already shows residual std at truth ≈ 0.000995, so train features are noisy and the forward layout matches the harness. The r/sigma diagnost… · This substantially overlaps P-012. Matched fresh noise on the 3000 real thetas is the n_sim=0+fresh-noise cell of the same design. Run it as a single arm of a … · Treat the noisy-train-features finding as established (r/sigma ≈ 1) unless the custom solver's own check contradicts it. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v3 `P-013` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* If training noise exactly matches the measurement model (clean F(theta) plus fresh i.i.d. N(0, 1e-3) noise per minibatch), the canonical squared-loss MLP improves val rel_l2 by at least 10% relative over cell A. Cell A is the identical model trained on the gi…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: Three outcomes, each with a different implication: - If B ≈ A and the holdout gap is large, the amortised gap comes from theta scarcity. Forward budget should …
  - response to critic: Point-by-point responses.  (1) r/sigma is largely settled by P-002. Accepted. P-002's residual at truth (about 0.000995) is treated as establishing that the train features carry one noise draw. r/sigma is kept only as an internal check tha…
  - critic r3 (quality 0.865): The mechanism is mislabelled. The given train X is already one draw from the exact val/test measurement model (P-002: residual at truth 0.000995 ≈ sigma). Cell… · The success prediction is too weak to be informative. P-004 already reached 0.0478 with only 300 train samples, and memory m-261109-f27c4024 reports an amortis… · Replace 'only exactly matched training distribution' with: 'train X is matched but contains a single noise realisation per theta; fresh draws add noise diversi… · If A's holdout gap is small (train_rel ≈ hold_rel), the network is not overfitting noise realisations. B ≈ A is then expected even though the 'noise model' cha… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-007 — Truncated-prior MMSE can undercut the CRB 'floor': amortised init plus a forward-model posterior-mean refinement** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+pca48+logy+aug2->LM15]` status conjectured. *H:* The harness CRB estimate (rel_l2 about 0.0186) bounds only unbiased estimators. k1 and k4 are poorly determined (CR rel std 0.022 and 0.018) relative to the prior box [0.5, 2]. So a box-prior posterior mean, computed by importance sampling on the Laplace appr…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02}`; if it fails: Suppose refinement reaches about 0.0186 but not below it. Then the box truncation is too rarely active to matter, and MLE-level accuracy is the practical ceili…
  - critic r1 (quality 0.88): Back-of-envelope: posterior std is about 2% of k for k1 and k4, so box truncation is active for only a few percent of samples. The MMSE-over-CRB gain is theref… · The CRB estimate is a median over 32 draws. The aggregate rel_l2 of an efficient estimator could legitimately exceed 0.0186 because of the conditioning tail. I… · Compute the empirical CRB aggregate on the actual val samples, sqrt(sum tr(F^-1)_rel / n). Compare against that rather than against the 32-draw median. · If there are no samples within 2σ of a bound, posterior mean and MLE coincide, and nothing can undercut the CRB. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-008 — The amortised-network gap is a data-scarcity artefact, not a noise floor: simulation-augmented training through the for…** (researcher-exploratory, family physics-custom)
- v1 `P-008` r1 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The MLP plateau at about 0.026 is caused by having only 3000 training pairs for a 4-D inverse map, not by the measurement noise. Adding about 12000 freshly simulated prior draws, each with a fresh N(0, 1e-3) noise realisation, should cut val rel_l2 to below 0…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.023}`; if it fails: If more simulations do not help, the 0.026 is close to what a 256x256 MLP can extract on PCA-48 features. The remaining gap to 0.0186 then needs either per-sam…
  - critic r1 (quality 0.829): The code uses sklearn MLPRegressor (a 256x256 network, judging by its if_fails text), whereas the 0.0261 reference is torch_mlp. Any gain or loss against 0.026… · Silent fallback to plain training on forward failure means the run must log whether augmentation happened. Otherwise a null result cannot be interpreted. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-012` r2 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised MLP plateau of about 0.026 comes from finite training data, not from the Bayes risk. The pilot's 0.0415 came from an unverified feature layout and/or an undertrained sklearn net, not from augmentation itself. With layout-verified forward simulat…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.023}`; if it fails: If aug_used=False because of the residual check, the simulator and data disagree beyond the layout. That is a noise-model or feature-preprocessing mismatch, wh…
  - response to critic: The pilot is evidence and it outweighs my argument. With 12000 forward evaluations (augmentation ran, forward_evals=12000), the pilot reached 0.0415 on 60 val samples with a 300-sample train fraction. That is far worse than the 0.026 amort…
  - critic r2 (quality 0.79): P-002's model_mismatch=false (residual at truth ≈ sigma over 200 samples) shows that context.forward matches the X layout as the harness uses it. A layout arte… · The target of < 0.023 is ambitious but meaningful against the physics floor of about 0.0186. Even if it succeeds, the arm only approaches P-002. State its valu… · The layout check is a sanity guard and is expected to pick 'id'. Do not count a pass as support for the artefact hypothesis. · If the layout check picks 'id' and fit_resid ≈ 1e-3, but the 12000 arm stays near 0.026, the cause is capacity or optimisation, not data or layout. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v3 `P-015` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised plateau of about 0.026 is mainly estimation error from finite training data. The PCA bottleneck may also contribute. The Bayes risk of the box-prior MMSE is not the cause. Under that hypothesis, a raw-feature MLP with a box-bounded output, train…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.023}`; if it fails: If (c) holds, the amortised route is capacity- or optimisation-limited at about 0.026. Forward-model refinement, as in the hybrid proposal and P-009, is then n…
  - response to critic: Point-by-point.  (C1) Layout artefact: accepted. P-002's model_mismatch=false (residual at truth about sigma over 200 samples) is experimental evidence and outweighs my argument. I withdraw the layout explanation of the pilot's 0.0415. The…
  - critic r3 (quality 0.825): The physics posterior mean is the MMSE estimator, so the amortised floor is physics MMSE, not the CRB. Under squared loss in k-space and the true prior, no amo… · With fixed epochs=60, the n_sim axis is confounded with the number of optimisation steps: steps grow roughly 1:3:5 across arms. A monotone curve could reflect … · Floor: amortised Bayes risk ≥ physics posterior-mean risk ≈ 0.0186 (pilot). It is not the CRB. · Suppose train loss at n_sim=12000 equals val loss and both sit at about 0.026. This is the 'Bayes risk' outcome. It would contradict the physics MMSE of 0.0186… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-014 — Hybrid budget control: canonical MLP initializer + single-start LM posterior-mean refinement matches multistart physics…** (researcher-experimental, family hybrid)
- v1 `P-014` r3 [propose, llm] `hybrid[torch_mlp/raw+logy->LM10]` status conjectured. *H:* The amortised MLP's error (roughly 0.05-0.1 rel_l2) places its estimate inside the local convergence basin of the unique likelihood mode on this well-conditioned problem (Fisher condition median 48). If so, a single LM start of 10 iterations from the MLP init…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: Two possible outcomes: - If the hybrid is worse and the residual/sigma > 1.2 cases cluster at a poor init, multistart is genuinely needed and P-009 stays the f…
  - critic r3 (quality 0.775): The comparator is not established at full fidelity. P-009/P-002 has run only on 60 val samples. The '5% of P-009' and '25% of evaluations' criteria need the fu… · The cost accounting is likely dominated by importance sampling, not LM. P-002 used 26943 forwards per 60 samples, about 450 per sample. With finite-difference … · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Suppose the hybrid matches P-009 in rel_l2 but 2% of samples land in a different mode with a compensating error elsewhere. Aggregate equivalence would hide gen… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-016 — Cheap hybrid: bounded MLP initializer followed by single-start short LM and Laplace-IS posterior mean, to match physics…** (researcher-exploratory, family hybrid)
- v1 `P-016` r3 [propose, llm] `hybrid[torch_mlp/raw+logy->LM8]` status conjectured. *H:* The posterior over k is unimodal on the box, given the median Fisher condition number of 48.2 and full-rank Fisher. A single well-placed start therefore suffices. The setup is an amortised initializer, already within about 0.026 rel_l2 of the truth, followed …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0195}`; if it fails: Two failure modes are distinguishable: - Excess error concentrated in high-misfit samples: global non-identifiability or multimodality exists on part of the bo…
  - critic r3 (quality 0.501): This is near-duplicate of P-014: same method kind and same init architecture, differing only in iterations (8 vs 10), epochs (150 vs 120) and IS samples (128 v… · The cost criterion is likely self-defeating. 128 IS forwards plus about 40 LM forwards is roughly 170 per sample, about 37% of P-009's ~450. That fails the '≤3… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · With 128 vs 64 IS samples, the hybrid could match or beat P-009 simply through lower IS variance even if it lands in the wrong mode on a few samples. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: No pilot data exist yet. The cheapest discriminating tests are P-002 (unimodality, 38 s) and P-008 (data scaling, 30 s), and they should run first. P-003 and P-005 duplicate P-001, which has a likely noise_scale misconfiguration (0.001 instead of 1.0) that needs resolving. P-007's claim to beat the CRB should be evaluated on the canonical hybrid run rather than paid for separately, and the expect…  families={'hybrid': 4, 'physics': 1, 'data-driven': 2, 'physics-custom': 1}
- Critic round 2 portfolio view: Empirically, physics MMSE (P-002 config) reaches about 0.0186 on the pilot with calibrated uncertainty, already well below 0.05. The top priority is confirming it on full val with tail stratification (P-009, as a duplicate of P-002); P-009's prediction must drop its reference to the nonexistent hybrid. The ML arms should be merged into one controlled factorial. P-010 must cut cost and should run …  families={'physics': 1, 'data-driven': 1, 'physics-custom': 2}
- Critic round 3 portfolio view: Physics MMSE (0.0186 pilot, well calibrated) already beats the 0.05 target by about 2.7×. The top priority is confirming it on full val (about 65 s). The remaining proposals are mechanism studies. Merge P-013 with P-015 into a noise-resampling × n_sim factorial with matched optimiser steps, and treat physics MMSE (not the CRB) as the amortised floor. Merge P-016 into P-014 with matched IS samples…  families={'physics-custom': 2, 'hybrid': 2}

**Pilot experiments run during research** (small validation subsets):

- P-002 `physics-MMSE[log,ms=4,it=30]`: ok rel_l2=0.018599 on 60 val samples in 5.16 s
- P-004 `torch_mlp/raw+pca48+logy+aug4`: ok rel_l2=0.047752 on 60 val samples in 26.9 s
- P-008 `custom-solver(fwd)`: ok rel_l2=0.041493 on 60 val samples in 30.1 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 66 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search (cache) |
| researcher-ml | 1 | neural network posterior mean estimator minimax risk approximation Bayes estimator square… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability estimate for identifying piecewise constant diffusion coefficient in … | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of piecewise constant conductivity from boundary or point observations of heat… | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for identification of piecewise constant diffusion coe… | 4 refs via /search |
| researcher-experimental | 1 | asymptotic efficiency and bias of maximum likelihood versus posterior mean estimator unde… | 4 refs via /search |
| researcher-mathematical | 1 | local convergence of Gauss-Newton or Levenberg-Marquardt for finite-dimensional parameter… | 4 refs via /search |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-mathematical | 2 | Per-sample chi^2/n distribution after LM on val | 4 refs via /search |
| researcher-experimental | 2 | posterior mean estimator bounded parameter box prior Cramer-Rao bias heat equation | 4 refs via /search |
| researcher-exploratory | 2 | Fraction of val samples with any k_i within 2 posterior std of a bound | 4 refs via /search |
| researcher-ml | 2 | identifiability piecewise constant conductivity heat equation boundary observations | 4 refs via /search |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 3 | Jacobian energy captured by PCA-48 for each k_i at 32 prior draws | 4 refs via /search |
| researcher-exploratory | 3 | Amortised gap is a data-scarcity artefact: layout-verified simulation-augmented torch MLP… | 4 refs via /search |
| researcher-experimental | 3 | Logged r/sigma from the custom solver | 4 refs via /search |
| researcher-mathematical | 3 | Per-sample (rel err, chi2/dof, ESS, start-endpoint spread) on the full 750 val, binned by… | 4 refs via /search |
| researcher-ml | 3 | Fisher-information-aware dimensionality reduction (sensitivity-weighted PCA / active subs… | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\shared_memory.jsonl; 322 records from 3 run(s) were available at start; agents retrieved 17 distinct prior-run records; this run wrote 131 records (test numbers redacted for agents).
  - `m-261109-019609af` citation (retrieved) from `heat_v2_memory_s1_20261109_153523`: Theorem 5 — A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation
  - `m-261109-28420efa` hypothesis (conjectured) from `heat_v2_memory_s1_20261109_153523`: Amortised init + forward-model MMSE refinement: closing the gap from the MLP (≈0.025) to the CRB floor (≈0.0185)
  - `m-261109-54edbdef` hypothesis (conjectured) from `heat_v2_full_s0_20261109_001356`: Pure per-sample MLE with multistart LM (no learning) is sufficient: the least-squares landscape is effectively unimodal on the prior box
  - `m-261109-62ecc48e` citation (retrieved) from `heat_v2_memory_s1_20261109_153523`: Lemma 9.4 (clause \textsc {bp}) — The number of solutions for random regular NAE-SAT
  - `m-261109-68231a2b` citation (retrieved) from `heat_v2_memory_s1_20261109_153523`: Theorem 2.4 — Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems
  - `m-261109-742270b6` critique (recorded) from `heat_v2_memory_s1_20261109_153523`: critique r1 of Amortised MMSE without the forward model at inference: a noise-augmented MLP on raw sensor traces reaches the 0.05 targe
  - `m-261109-7ea0a084` hypothesis (literature-supported) from `heat_v2_full_s0_20261109_001356`: Noise-limited regime: bounded log-space MLE through the forward model should land near the CRB floor (~0.02 rel_l2)
  - `m-261109-820f8a3d` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Theorem — Uncertainty quantification and weak approximation of an elliptic inverse problem
  - `m-261109-836f0093` hypothesis (conjectured) from `heat_v2_full_s0_20261109_001356`: Amortized MLP with log-target and noise augmentation is accurate but cannot reach the CRB without physics refinement
  - `m-261109-881544e6` citation (retrieved) from `heat_v2_memory_s1_20261109_153523`: Proposition 2 — Cramer Rao-Type Bounds for Sparse Bayesian Learning
  - `m-261109-88bd39b7` citation (retrieved) from `heat_v2_full_s0_20261109_001356`: Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}) — Bridging and Improving Theoretical and Computational Electric Impedance Tomogr…
  - `m-261109-c68594a4` critique (recorded) from `heat_v2_full_s0_20261109_001356`: critique r1 of Pure per-sample MLE with multistart LM (no learning) is sufficient: the least-squares landscape is effectively unimodal 

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-013) | physics-custom | `custom-solver(fwd)` | researcher-experimental | 0.865 | best-rated proposal in family 'physics-custom' (quality 0.86) |
| E-02 | proposal (P-014) | hybrid | `hybrid[torch_mlp/raw+logy->LM10]` | researcher-experimental | 0.775 | best-rated proposal in family 'hybrid' (quality 0.78) |
| E-03 | proposal (P-015) | physics-custom | `custom-solver(fwd)` | researcher-exploratory | 0.825 | quality 0.82 with diversity penalty |
| E-04 | proposal (P-016) | hybrid | `hybrid[torch_mlp/raw+logy->LM8]` | researcher-exploratory | 0.501 | quality 0.50 with diversity penalty |
| E-05 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-06 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-07 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-009 | E-07 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.011 | 6.86 | 61517 | 0.957 | FINAL |
| idea-011 | E-06 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.10795** | 0.8927 | 0.1424 | 0.991 | 0.163 | 0 | - | evaluated |
| idea-008 | E-06 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 0.972 | 0.133 | 0 | - | evaluated |
| idea-010 | E-06 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.10842** | 0.8917 | 0.143 | 0.97 | 0.147 | 0 | - | evaluated |
| idea-013 | E-06 | refine | data-driven | `extra_trees/log+pca64` | - | **0.11481** | 0.879 | 0.1514 | 0.512 | 0.0499 | 0 | - | evaluated |
| idea-007 | E-06 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.476 | 0.049 | 0 | - | evaluated |
| idea-012 | E-06 | refine | data-driven | `extra_trees/sqrt+pca64` | - | **0.11868** | 0.8706 | 0.1565 | 0.338 | 0.033 | 0 | - | evaluated |
| idea-002 | E-01 | research | physics-custom | `custom-solver(fwd)` | 0.04246 | **-** | - | - | - | - | - | - | ok |
| idea-003 | E-02 | research | hybrid | `hybrid[torch_mlp/raw+logy->LM10]` | 0.02008 | **-** | - | - | - | - | - | - | projected full-fidelity cost 158s exceeds the 120s per-experiment lim… |
| idea-004 | E-03 | research | physics-custom | `custom-solver(fwd)` | 0.02331 | **-** | - | - | - | - | - | - | ok |
| idea-005 | E-04 | research | hybrid | `hybrid[torch_mlp/raw+logy->LM8]` | 0.02043 | **-** | - | - | - | - | - | - | projected full-fidelity cost 146s exceeds the 120s per-experiment lim… |
| idea-006 | E-05 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **-** | - | - | - | - | - | - | projected full-fidelity cost 141s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-009` `physics-MLE[log,ms=1,it=30]` ← baseline/control (physics baseline: Levenberg-Marquardt MLE (defaults))

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00193` [checkpoint] by scheduler: final method (single)
- `resu-00190` [result] by executor: exp-0020 rel_l2=0.01842 r2=0.997 rmse=0.02424
- `hypo-00116` [hypothesis] by scheduler: idea-009 physics-MLE[log,ms=1,it=30] (baseline)
- `plan-00189` [plan] by engineer-v2: exp-0020: final fit of 1 member(s) on train+val; test once
- `gate-00115` [gate] by gate: E-07: approved (physics-MLE[log,ms=1,it=30])
- `spec-00001` [spec] by scheduler: research spec heat_v2_memory_s2 (b804920343)

## 9. Compute and cost

- Execution compute: 246.4 / 400 worker-seconds over 20 jobs; wall 1029 s
- By stage: baseline 2.1s, pilot 62.2s, screen 137.3s, evaluate 12.8s, refine 12.2s, final 19.7s
- LLM: 19 calls (by phase {'research': 19}), 1415 s, $3.905 reported · TheoremSearch: 24 queries (0 failed)

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

The forward-model maximum-likelihood estimator, `physics-MLE[log,ms=1,it=30]`, meets the success criterion with margin. It reached rel_l2 = 0.0193 on validation and 0.0184 on the held-out test split (R² = 0.997, RMSE = 0.024). Test error is close to validation error, so selection on validation does not appear to have overfit. The best purely data-driven model, extra trees on 16 PCA components, reached 0.108, about 5.6× worse, and missed the 0.05 target. So at this budget, explicit inversion through the heat-equation forward model clearly beats amortised regression from sensor traces.

These results do not test any of the stated hypotheses. The supported and contradicted lists are both empty, and every abandoned proposal was dropped for procedural reasons, not because data contradicted it:
- **Blocked:** one proposal had an unresolved `noise_scale=0.001` inconsistency. Another referenced a nonexistent `best_hybrid` baseline. A third had a runtime estimate equal to the experiment limit.
- **Merged into P-001:** three duplicate variants of the hybrid MLP-initializer plus LM posterior-mean design.

The claims about the Cramér–Rao bound (CRB) therefore remain conjecture. These were that a hybrid reaches near-CRB accuracy, and that a truncated-prior MMSE estimator can beat the CRB. The "<2% truncation gain" figure is a back-of-envelope argument, not a measurement. It is also unknown how close 0.018 is to the CRB floor, because no CRB was computed. The MLE used a single start, so global identifiability and the risk of local minima were not checked, and per-parameter errors (for example k1 and k4 near the boundaries) were not reported.

- Run the canonical P-001 hybrid (MLP initializer, then LM refinement with posterior-mean readout) with `noise_scale` made consistent with the data generator. Report the init-only and refined rel_l2 separately.
- Compute the empirical per-sample CRB on validation and report the MLE's error-to-CRB ratio per parameter. Stratify by distance to the prior-box bounds to test the truncation-gain argument directly.
- Sweep the number of multistarts (1, 4, 16) at a fixed iteration count. This tests whether the single-start MLE ever lands in a local minimum, which is the global-identifiability question, measured against an absolute threshold.
- Rerun the PCA-48 versus raw-trace feature comparison with fewer epochs or less augmentation so it fits the 120 s limit. Report per-parameter errors to see where data-driven models lose accuracy.
- Test the MLE's robustness to noise-level misspecification and to sensor-placement perturbations, which determines whether the physics advantage holds outside the simulator's assumptions.

_Automatic summary:_ 16 proposal versions over 3 research rounds; 7 methods advanced; 0 predictions supported, 0 contradicted. Final: ['physics-MLE[log,ms=1,it=30]'] with test rel_l2 0.018419.
