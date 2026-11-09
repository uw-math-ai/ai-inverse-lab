# Research report (V2): heat_v2_no_literature_s2

Run `heat_v2_no_literature_s2_20261109_113933` · 2026-11-09 11:46:42 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=4,it=40]`
- Validation rel_l2 (selection estimate): **0.018998** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018405**, r2 = **0.99697**, rmse = **0.024217** · bootstrap 95% CI for rel_l2: [0.01732, 0.01953]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.12166 (vs 0.018405 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051181 (vs 0.018405 in distribution)
- Uncertainty calibration on test: 68.6% within 1σ, 95.2% within 2σ (ideal 68% / 95%)
- Inference latency: 34.4 ms/sample; fit 0.00931 s; exported model reload check: max |Δ| = 0.00321
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=4,it=40]` rel_l2=0.018998; **hybrid** `hybrid[kernel_ridge/raw+logy->LM15]` rel_l2=0.019333; **physics-custom** `custom-solver(fwd)` rel_l2=0.022872; **data-driven** `extra_trees/sqrt+pca16` rel_l2=0.10842
- SUPPORTED — E-01 *Learned initializer plus forward-model LM refinement reaches the noise-limited (CRB) regime*: val rel_l2 = 0.019335 vs predicted < 0.03
- SUPPORTED — E-02 *Log-space MLE via multistart Levenberg-Marquardt through the forward model attains the noise-limited (Cramér-Rao) accur…*: val rel_l2 = 0.01934 vs predicted < 0.03
- SUPPORTED — E-04 *Simulation-enlarged amortized MMSE: MLP trained on ~12k fresh forward simulations with exact per-epoch noise resampling*: val rel_l2 = 0.022872 vs predicted < 0.03
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- no citations in proposals

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-03 *Purely amortized inverse map: MLP on log-k targets with noise-matched augmentation (given 3000 samples; shared initiali…*: inconclusive — method did not produce a full-fidelity validation result
- E-06 *Cheap hybrid: shared MLP initializer + 4 unregularized Gauss-Newton/LM steps in log-k (MLE mode)*: inconclusive — method did not produce a full-fidelity validation result
- P-015 *Variance-stabilised asinh(u/3σ) sensor features let a purely amortized MLP reach rel_l2 < 0.05 without forward calls at…* (conjectured): proposed but not executed
- P-016 *Is the amortized gap a data limit? Train the inverse MLP on 3000 provided + 12000 freshly simulated prior draws via con…* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-05 *Paired ablation: box-prior posterior mean vs mode from an identical physics solver; any gain is confined to bound-adjac…*: val rel_l2 0.018998 vs best_physics 0.019335: gain +1.74%; expected gain >= +2.0% vs best_physics
- ABANDONED in round 2 by researcher-experimental — P-006 *Control: an amortized neural inverse map trained with noise augmentation falls short of forward-mod…*: duplicate of P-004
- ABANDONED in round 2 by researcher-exploratory — P-007 *Minimum-MSE (posterior-mean) refinement beats MLE when the error metric is squared: hybrid learned-…*: This is not a negative scientific result. The proposal duplicates P-002's estimator-gap experiment, exceeds the per-experiment time estimate (145.6 s > 120 s), and the total compute budget (400 worker-seconds) cannot fu…
- PRUNED — idea-004 `torch_mlp/raw+logy+aug2`: projected full-fidelity cost 133s exceeds the 120s per-experiment limit
- PRUNED — idea-007 `hybrid[torch_mlp/raw+logy+aug2->LM4]`: projected full-fidelity cost 136s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Learned initializer plus forward-model LM refinement reache…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-02 (Log-space MLE via multistart Levenberg-Marquardt through th…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-03 (Purely amortized inverse map: MLP on log-k targets with noi…): question: the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-03 (Purely amortized inverse map: MLP on log-k targets with noi…): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-04 (Simulation-enlarged amortized MMSE: MLP trained on ~12k fre…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-05 (Paired ablation: box-prior posterior mean vs mode from an i…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-06 (Cheap hybrid: shared MLP initializer + 4 unregularized Gaus…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- Literature check INCOMPLETE: 16 failed queries (disabled), 0 skipped for budget, enabled=False
- Test-set sampling uncertainty: bootstrap 95% CI [0.01732, 0.01953] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **2 round(s)**: quality criteria met after 2 rounds (every active proposal >= 0.75 quality, no blocking issues, critic converged).

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The pilot results answer most of the critique. I revise the framing and diagnostics and keep the method uncha… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | I accept the duplicate finding and the blocking issues. I re-scope P-002 so it no longer overlaps P-007 and n… |
| 2 | researcher-ml | revise | P-003 | P-011 | llm | (1) Tikhonov vs posterior_mean inconsistency: accepted. I set tikhonov=0 and estimator=mode, so this arm is a… |
| 2 | researcher-ml | revise | P-004 | P-012 | llm | Pilot evidence first: with only 300 training samples and noise_scale=1.0, the pilot reached val rel_l2 0.047 … |
| 2 | researcher-ml | propose | - | P-013 | llm |  |
| 2 | researcher-experimental | revise | P-005 | P-014 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-experimental | abandon | P-006 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | I accept every critic point. (1) Effect size: the truncation argument gives a 1-5% overall gain, so my >=5% c… |
| 2 | researcher-exploratory | revise | P-008 | P-015 | llm | Point by point. (a) The bare harness 'log' transform is undefined or noise-dominated on near-zero or negative… |
| 2 | researcher-exploratory | propose | - | P-016 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.709 | fdbk-00044 |
| 2 | 8 | 0.919 | fdbk-00076 |

### Idea evolution

**P-001 — Log-space MLE via multistart Levenberg-Marquardt through the forward model attains near-CRB accuracy** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* Because noise is tiny (sigma=1e-3), the Fisher condition number is moderate (~48), and the CRB rel_l2 floor is ~0.019, the likelihood is locally well-conditioned and close to Gaussian in log k. A bounded LM solver working in log-parameter space with a few mul…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the error is dominated by a few outliers, the misfit landscape is multimodal, which motivates a hybrid learned initializer. If the error is uniformly above …
  - critic r1 (quality 0.905): The 0.0186 floor is the frequentist CRB, which bounds unbiased estimators only, and it is a median over 32 draws. rel_l2 averaged over the prior is driven by t… · 'Within 1.5x of CRB' is ill-posed as a pass/fail anchor because the CRB estimate itself is noisy (32 draws). The fixed <0.03 threshold is the operative test; k… · Replace 'MLE near-efficient implies rel_l2 ≈ CRB' with 'MLE rel_l2 ≈ E_prior[CRB-based rel error]', computed over many more than 32 draws. · If T is short enough that the pulse barely reaches sensors 5 and 35, k1/k4 information comes from a few low-amplitude late readings. The likelihood in those di… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* The forward model matches the generator to within the noise, so the bounded log-space LM MLE converges to the global minimum for almost all samples. Its full-validation rel_l2 is below 0.03 and within about 15% of the prior-averaged CRB rel error. In the pilo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If failure comes from a small set of outliers with high chi2, the landscape is multimodal for some k1/k4 configurations. That supports the learned-init hybrids…
  - response to critic: The pilot results answer most of the critique. I revise the framing and diagnostics and keep the method unchanged.  (1) CRB is median-based and frequentist. Agreed. The pilot gives val rel_l2 = 0.0190 on 60 samples, against a median-CRB fl…
  - critic r2 (quality 0.95): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-002 — MMSE posterior mean beats the MLE for the weakly identified boundary segments k1, k4 under the box prior** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM20]` status conjectured. *H:* rel_l2 is a squared-error loss, so the Bayes-optimal estimator is the posterior mean, not the mode. For k1 and k4 the CRB std is about 2% and the uniform box prior truncates the likelihood near 0.5 and 2.0. A hybrid approach should therefore reduce val rel_l2…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.1, "expect": "better"}`; if it fails: If there is no gain over the MLE, the posterior is effectively Gaussian and untruncated for most samples, so the MLE is already near-Bayes and the remaining er…
  - critic r1 (quality 0.375): Estimated runtime 145.6s exceeds the 120s per-experiment limit; reduce posterior_samples/epochs before running. · noise_scale=0.001 conflicts with the other arms' noise_scale=1.0; resolve harness semantics first. · The 10% improvement target is implausible from truncation alone. Absolute posterior std is ≈0.022·k, i.e. at most ~0.045 at k=2. Only samples within ~2σ of a b… · The Van Trees (Bayesian) bound, not the CRB, is the relevant floor for an MMSE estimator. The posterior mean can legitimately beat the CRB, so 'gain concentrat… · Expect an MMSE-over-MLE gain of ≲5% overall. Any larger gain likely comes from outlier removal by the learned init, not from estimator choice, which is the con…
- v2 `P-010` r2 [revise, llm] `physics-MMSE[log,ms=4,it=40]` status conjectured. *H:* rel_l2 is a squared-type loss, so the box-truncated posterior mean is Bayes-optimal. Posteriors are near-Gaussian (P-001 pilot: z_rms 0.98), so the gain over the mode comes almost entirely from samples whose true k1 or k4 lies within 2 Gauss-Newton sigma of 0…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.02, "expect": "better"}`; if it fails: If there is no gain even in the bound-adjacent stratum, the truncation is too small at sigma=1e-3 to matter. The MLE is then effectively Bayes-optimal here, an…
  - response to critic: I accept the duplicate finding and the blocking issues. I re-scope P-002 so it no longer overlaps P-007 and no longer needs a learned component.  (1) Duplicate of P-007, confounded init. P-002 is now a pure-physics paired ablation. It uses…
  - critic r2 (quality 0.9): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-003 — Amortized MLP initializer + few-step Gauss-Newton/LM refinement with MMSE correction (hybrid)** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug1->LM8]` status conjectured. *H:* A torch_mlp trained on simulated pairs lands inside the basin of the per-sample likelihood. A short bounded LM refinement in log-k through the exact forward model (8 iterations, mild Tikhonov), followed by a posterior-mean estimate under the box prior, will r…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the refinement does not help, either the initializer lies outside the basin (check per-sample objective decrease), or the forward model and data differ (mod…
  - critic r1 (quality 0.845): tikhonov=0.01 combined with estimator=posterior_mean is inconsistent. Tikhonov implies a Gaussian prior centred somewhere (init? prior mean?), whereas the stat… · 8 LM iterations with no extra start assumes init error <10%. P-004/P-006 estimate MLP error in the 2-5% range, which is plausibly fine, but this is untested. P… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the MLP is already at ~0.03, an 8-step LM with Tikhonov pulling toward the init may improve it by <25% even though full LM would reach the CRB. That would b… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-011` r2 [revise, llm] `hybrid[torch_mlp/raw+logy+aug2->LM4]` status conjectured. *H:* Starting from the P-004 MLP initializer, 4 bounded, unregularized LM iterations in log-k through the exact forward model will have two effects. First, val rel_l2 will reach <= 0.035, within about 2x the CRB floor of 0.0186. Second, error will drop by at least…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If (a) fails but chi^2 is near 1, the method is at an information limit above 0.035, so the CRB floor estimate is optimistic. If chi^2 is high, either 4 steps …
  - response to critic: (1) Tikhonov vs posterior_mean inconsistency: accepted. I set tikhonov=0 and estimator=mode, so this arm is a pure MLE refinement with no hidden Gaussian prior. The MMSE question is left entirely to P-007/P-002. (2) Init inside the basin i…
  - critic r2 (quality 0.9): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-004 — Purely amortized inverse map: MLP on log-k targets with noise-matched augmentation** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch_mlp trained on raw sensor time series has two key features: log-conductivity targets, which match the multiplicative scale of a diffusion coefficient, and on-the-fly Gaussian noise augmentation at the true sigma, which regularizes toward the noisy-dat…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the error concentrates in k1 and k4, the shortfall is a data and identifiability limit rather than an architecture limit. That points to the physics refinem…
  - critic r1 (quality 0.845): The 3000 fixed samples are a self-imposed limit. The constraints allow training on simulated data, and context.forward is available, so the amortized arm can b… · noise_aug at sigma on top of already-noisy training features yields sqrt(2)·sigma training noise. This mildly biases toward a broader posterior and can inflate… · Treat training-set size as a free variable that can be enlarged through forward simulation, not a fixed 3000. · A learned map with per-parameter error 3% on k1/k4 and 1.5% on k2/k3 gives rel_l2 ≈ 0.02-0.025. That would pass, yet still be ~25% worse than an efficient MLE,… · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-012` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* A torch_mlp on raw sensor series with log-k targets and Gaussian noise augmentation at the stated sigma reaches val rel_l2 < 0.05 with no forward calls at inference, and is at least 2x better than ridge on the same features. The 300-sample pilot (0.047) alrea…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 >= 0.05 with the error concentrated in k1/k4, the cause is a data and identifiability limit; the simulation-enlarged proposal and P-003 test the …
  - response to critic: Pilot evidence first: with only 300 training samples and noise_scale=1.0, the pilot reached val rel_l2 0.047 on 60 val samples. This also bears on the noise_scale convention dispute. If 1.0 were an absolute std, the augmentation noise woul…
  - critic r2 (quality 0.9): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Learned initializer plus forward-model LM refinement reaches the noise-limited (CRB) regime** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status conjectured. *H:* The remaining error after a kernel-ridge initializer is mainly an optimization or approximation error, not an information limit. Refining each sample with a few bounded Levenberg-Marquardt iterations through the exact forward model, in log-k space, should the…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each failure mode points to a different cause. If the error is far above the CRB while chi^2/dof is ≈1, the CRB at 32 draws is unrepresentative, likely near th…
  - critic r1 (quality 0.895): The hypothesis is phrased as 'test rel_l2'. Selection and falsification must use val only, since test is held out. · Its distinctness from P-001 is only init (kernel ridge vs multistart). If P-001 converges globally, this arm tests only whether a learned init saves iterations… · Compare against P-001 on identical val samples, so that the init effect (KRR vs multistart) is isolated at matched LM settings. · The chi²/dof ≈ 1 with error > CRB branch could also arise from posterior non-Gaussianity rather than an unrepresentative CRB. Plotting error vs per-sample CRB … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-014` r2 [revise, heuristic-fallback] `hybrid[kernel_ridge/raw+logy->LM15]` status conjectured. *H:* The remaining error after a kernel-ridge initializer is mainly an optimization or approximation error, not an information limit. Refining each sample with a few bounded Levenberg-Marquardt iterations through the exact forward model, in log-k space, should the…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each failure mode points to a different cause. If the error is far above the CRB while chi^2/dof is ≈1, the CRB at 32 draws is unrepresentative, likely near th…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-006 — Control: an amortized neural inverse map trained with noise augmentation falls short of forward-model refinement** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* This is a competing explanation tested as a control. With 3000 training draws, a noise-augmented MLP from raw sensor traces to log k may already approach the CRB, which would make forward-model refinement unnecessary. Our counter-prediction is that it will no…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.25, "expect": "worse"}`; if it fails: If the MLP matches or beats the hybrid, the amortized map is already efficient. In that case the hybrid's LM step adds nothing, or is hurt by clipping at the b…
  - critic r1 (quality 0.477): It is near-identical to P-004 (same method, features, target and augmentation; only the hidden layer sizes differ). Its value is the noise_aug=0 ablation and t… · The 'relative_to best_hybrid, ≥25% worse' prediction depends on which hybrid is best. If the hybrid underperforms for unrelated reasons (Tikhonov bias, IS dege… · Use the 'oracle-init LM' result as the reference instead of best_hybrid; that is the information-limited reference. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-007 — Minimum-MSE (posterior-mean) refinement beats MLE when the error metric is squared: hybrid learned-init + Laplace/impor…** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM20]` status conjectured. *H:* We usually treat this as a point-estimation (MLE/least-squares) problem, but the metric is a squared relative error averaged over the prior. That makes the Bayes-optimal estimator the posterior mean, not the mode. Hypothesis: a hybrid method beats MAP/MLE ref…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If mode and posterior_mean are statistically tied, the posterior is effectively Gaussian and untruncated at this noise level. The estimator choice is then irre…
  - critic r1 (quality 0.705): Estimated runtime 145.6s exceeds the 120s per-experiment limit; reduce posterior_samples to 64 and/or reuse a pre-trained init. · The paired mode-vs-posterior_mean test from the same init is the cleanest design in the portfolio, but the ≥5% overall gain is optimistic. The truncation argum… · At 128 IS samples, the Monte Carlo error of the posterior mean is ~σ/sqrt(ESS). This is negligible only if ESS is large, so report ESS. · Primary prediction: posterior_mean improves mode by ≥2% overall and ≥10% on the bound-adjacent stratum. These are effect sizes that truncation theory can actua… · If the posterior is near-Gaussian and untruncated for ~90% of samples, mode and mean agree there. The whole gain comes from ~10% of samples and may be under th…

**P-008 — Amortized MMSE without a forward model: a noise-augmented neural regressor on log-temperature features reaches the 0.05…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/log+logy+aug3` status conjectured. *H:* Using the forward model at inference may be unnecessary here. With MSE loss, a network trained on simulated (observation, k) pairs directly estimates E[k / observation], the same Bayes-optimal target as proposal 1. Hypothesis: torch_mlp with log-transformed f…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If log features are no better than raw, either the harness's log transform is dominated by near-zero, noise-dominated readings (which would argue for a custom …
  - critic r1 (quality 0.625): Estimated runtime 160s exceeds the 120s per-experiment limit. · The log transform may be undefined on non-positive noisy readings; verify feature validity first. · The log-feature transform is risky. With a Gaussian pulse at x=0.5 and edge sensors at x≈0.13 and 0.90, early-time readings at sensors 5 and 35 are likely ≤ no… · The WKB/eigenvalue argument applies to late-time single-mode decay. For discontinuous k and a short window (200 explicit steps), the signal may be in the trans… · Use log(u + c) with c ≈ 3σ, or restrict to readings above 5σ, rather than the bare harness 'log' features.
- v2 `P-015` r2 [revise, llm] `custom-ml` status conjectured. *H:* An MLP regressor on asinh(u/c) features, with c = 3*sigma = 0.003, log-conductivity targets, and the training noise as provided, reaches val rel_l2 < 0.05. The feature map compresses the multi-decade dynamic range of the far-sensor signals while staying linea…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the arm stays above 0.05 while P-004 (raw) also fails, the bottleneck is samples or capacity rather than feature geometry, which the new simulation-augmente…
  - response to critic: Point by point. (a) The bare harness 'log' transform is undefined or noise-dominated on near-zero or negative edge-sensor readings. Agreed, so I no longer use it. The revised arm uses custom_ml with an asinh(u/c) feature map, c=3*sigma=0.0…
  - critic r2 (quality 0.9): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-013 — Simulation-enlarged amortized MMSE: MLP trained on ~12k fresh forward simulations with exact per-epoch noise resampling** (researcher-ml, family physics-custom)
- v1 `P-013` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The gap between the amortized MLP and the CRB floor is mainly a finite-data effect, not an architecture or amortization limit. Training the same MLP on 3000 given samples plus about 12000 fresh prior draws simulated with context.forward, with fresh N(0, sigma…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If error does not improve with 5x more data, P-004's gap is not a data limit. Either the architecture or feature scaling is inefficient (try PCA or a time-wind…
  - critic r2 (quality 0.9): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-016 — Is the amortized gap a data limit? Train the inverse MLP on 3000 provided + 12000 freshly simulated prior draws via con…** (researcher-exploratory, family physics-custom)
- v1 `P-016` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* Most of the gap between the amortized MLP and the CRB-limited hybrid is finite-sample approximation error, not an intrinsic limit of amortization. Training the same simple MLP on 5x more simulated (noisy observation, k) pairs from the prior, using context.for…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.04}`; if it fails: If the gain is under 25% but the error does not get worse, amortization is capacity- or optimisation-limited rather than data-limited, which favours the hybrid…
  - critic r2 (quality 0.9): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: The ideas are sound and falsifiable, but the portfolio is heavily duplicated and about 2x over budget. P-002/P-007 and P-004/P-006 should be merged. Three arms exceed the 120s per-experiment estimate. The shared, unverified assumption that context.forward matches the generator, plus the conflicting noise_scale conventions, must be resolved by near-free pilots before the main runs. Run the physics…  families={'physics': 1, 'hybrid': 4, 'data-driven': 3}

**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MLE[log,ms=4,it=40]`: ok rel_l2=0.019024 on 60 val samples in 3.56 s
- P-004 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.046863 on 60 val samples in 13.7 s

## 5. Literature retrieval (TheoremSearch)

0 queries, 16 failures, 0 distinct statements retrieved, 0 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-exploratory | 1 | Lipschitz stability for inverse coefficient problem of one-dimensional parabolic equation… | FAILED: disabled |
| researcher-exploratory | 1 | uniqueness of piecewise constant diffusion coefficient from boundary or point measurement… | FAILED: disabled |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-ml | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-experimental | 1 | Lipschitz stability for recovering piecewise constant conductivity in a one-dimensional p… | FAILED: disabled |
| researcher-experimental | 1 | Bernstein-von Mises or asymptotic efficiency of maximum likelihood estimator for finite-d… | FAILED: disabled |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | FAILED: disabled |
| researcher-mathematical | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 2 | chi²/dof of val observations evaluated at the TRUE k through context.forward | FAILED: disabled |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | Distribution of MLP-init per-parameter relative error on val vs the LM basin radius (frac… | FAILED: disabled |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 2 | Per-sample predicted CRB vs realized error scatter on val | FAILED: disabled |
| researcher-exploratory | 2 | IS effective sample size distribution, and estimate shift (mean − mode) vs distance to bo… | FAILED: disabled |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_literature_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 78 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-014) | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | researcher-experimental | 1 | best-rated proposal in family 'hybrid' (quality 1.00) |
| E-02 | proposal (P-009) | physics | `physics-MLE[log,ms=4,it=40]` | researcher-mathematical | 0.95 | best-rated proposal in family 'physics' (quality 0.95) |
| E-03 | proposal (P-012) | data-driven | `torch_mlp/raw+logy+aug2` | researcher-ml | 0.9 | best-rated proposal in family 'data-driven' (quality 0.90) |
| E-04 | proposal (P-013) | physics-custom | `custom-solver(fwd)` | researcher-ml | 0.9 | best-rated proposal in family 'physics-custom' (quality 0.90) |
| E-05 | proposal (P-010) | physics | `physics-MMSE[log,ms=4,it=40]` | researcher-mathematical | 0.9 | quality 0.90 with diversity penalty |
| E-06 | proposal (P-011) | hybrid | `hybrid[torch_mlp/raw+logy+aug2->LM4]` | researcher-ml | 0.9 | quality 0.90 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-015 *Variance-stabilised asinh(u/3σ) sensor features let a purely amortized MLP reac…* — portfolio size limit (6); quality 0.90
- not advanced: P-016 *Is the amortized gap a data limit? Train the inverse MLP on 3000 provided + 120…* — portfolio size limit (6); quality 0.90

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-006 | E-05 | research | physics | `physics-MMSE[log,ms=4,it=40]` | 0.02042 | **0.018998** | 0.9967 | 0.02505 | 0.00832 | 36.4 | 330706 | 0.952 | FINAL |
| idea-013 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.019333** | 0.9966 | 0.0255 | 0.451 | 9.95 | 101651 | 0.957 | evaluated |
| idea-012 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.019334** | 0.9966 | 0.0255 | 0.451 | 9.91 | 101840 | 0.957 | evaluated |
| idea-002 | E-01 | research | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | 0.02138 | **0.019335** | 0.9966 | 0.0255 | 0.596 | 11.5 | 101336 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.00891 | 7.69 | 61517 | 0.957 | evaluated |
| idea-003 | E-02 | research | physics | `physics-MLE[log,ms=4,it=40]` | 0.02137 | **0.01934** | 0.9966 | 0.0255 | 0.0106 | 26.9 | 241139 | 0.957 | evaluated |
| idea-005 | E-04 | research | physics-custom | `custom-solver(fwd)` | 0.02499 | **0.022872** | 0.9952 | 0.03016 | 36.3 | 0.0166 | 12000 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 1.16 | 0.195 | 0 | - | evaluated |
| idea-014 | E-08 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.10848** | 0.8916 | 0.1431 | 0.879 | 0.118 | 0 | - | evaluated |
| idea-015 | E-08 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.12301** | 0.8607 | 0.1622 | 0.78 | 0.132 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw+logy+aug2` | 0.04479 | **-** | - | - | - | - | - | - | projected full-fidelity cost 133s exceeds the 120s per-experiment lim… |
| idea-007 | E-06 | research | hybrid | `hybrid[torch_mlp/raw+logy+aug2->LM4]` | 0.02139 | **-** | - | - | - | - | - | - | projected full-fidelity cost 136s exceeds the 120s per-experiment lim… |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **-** | - | - | - | - | - | - | ok |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **-** | - | - | - | - | - | - | ok |

## 8. Lineage of the final method

- `idea-006` `physics-MMSE[log,ms=4,it=40]` ← P-010 v2 (researcher-mathematical, r2, revise) ← P-002 v1 (researcher-mathematical, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00196` [checkpoint] by scheduler: final method (single)
- `resu-00193` [result] by executor: exp-0024 rel_l2=0.0184 r2=0.997 rmse=0.02422
- `hypo-00090` [hypothesis] by scheduler: idea-006 physics-MMSE[log,ms=4,it=40] (research)
- `plan-00192` [plan] by engineer-v2: exp-0024: final fit of 1 member(s) on train+val; test once
- `gate-00089` [gate] by gate: E-05: approved (physics-MMSE[log,ms=4,it=40])
- `prop-00066` [proposal] by researcher-mathematical: P-010 v2 [revise] mathematical: Paired ablation: box-prior posterior mean vs mode from an identical physics solver; any ga
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_literature_s2 (f3bd20fbc0)
- `prop-00036` [proposal] by researcher-mathematical: P-002 v1 [propose] mathematical: MMSE posterior mean beats the MLE for the weakly identified boundary segments k1, k4 under
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 325.4 / 400 worker-seconds over 24 jobs; wall 429 s
- By stage: baseline 2.4s, pilot 17.2s, screen 97.7s, evaluate 117.4s, refine 26.0s, final 64.6s
- LLM: 14 calls (by phase {'research': 14}), 626 s, $2.024 reported · TheoremSearch: 0 queries (0 failed)

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

16 proposal versions over 2 research rounds; 9 methods advanced; 3 predictions supported, 1 contradicted. Final: ['physics-MMSE[log,ms=4,it=40]'] with test rel_l2 0.018405.
