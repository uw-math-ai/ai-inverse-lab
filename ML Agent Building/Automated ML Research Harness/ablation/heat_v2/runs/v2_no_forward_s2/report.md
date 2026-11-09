# Research report (V2): heat_v2_no_forward_s2

Run `heat_v2_no_forward_s2_20261109_104026` · 2026-11-09 10:55:32 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **single** of `custom-solver`
- Validation rel_l2 (selection estimate): **0.021924** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.019521**, r2 = **0.99659**, rmse = **0.025686** · bootstrap 95% CI for rel_l2: [0.01822, 0.02052]
- Per-target test rel_l2: k1 0.0283, k2 0.0065, k3 0.0060, k4 0.0258
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.099063 (vs 0.019521 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.052693 (vs 0.019521 in distribution)
- Inference latency: 2.66 ms/sample; fit 82.2 s; exported model reload check: max |Δ| = 1.53e-13
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **custom** `custom-solver` rel_l2=0.021924; **data-driven** `kernel_ridge/raw+pca30+logy+aug1` rel_l2=0.024748
- SUPPORTED — E-02 *Learned-emulator information arbiter: Jacobian CRLB for rel_l2, plus empirically gated emulator-LM refinement of a sigm…*: val rel_l2 0.021924 vs best_data_driven 0.024748: gain +11.41%; expected gain >= +3.0% vs best_data_driven
- SUPPORTED — E-05 *Preprocessing contrast for kernel ridge: unscaled PCA vs standardised PCA (P-006) on the same KRR cell*: val rel_l2 = 0.024748 vs predicted < 0.065

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-005', 'P-011']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:19860711` Mukhtar Karazym, Tohru Ozawa, Durvudkhan Suragan (2020). "Multidimensional inverse Cauchy problems for evolution equations". Theorem 3.2. http://arxiv.org/abs/2004.09195v1 — cited by ['P-005', 'P-011']; researcher claim ['no']; harness screen **plausible**
- `ts:thm:21621221` Guy Barles, Erwin Topp (2015). "Lipschitz Regularity for Censored Subdiffusive Integro-Differential Equations with Superfractional Gradient Terms". Lemma 4.10. http://arxiv.org/abs/1412.2621v2 — cited by ['P-006', 'P-012', 'P-018']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:17546409` Ali Feizmohammadi (2025). "Reconstruction of 1-D evolution equations and their initial data from one passive measurement". Theorem 1.2. http://arxiv.org/abs/2501.10590v2 — cited by ['P-007', 'P-013']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:21876495` Eva Sincich (2014). "Smoothness dependent stability in corrosion detection". Theorem 2.1. http://arxiv.org/abs/1406.5907v1 — cited by ['P-007']; researcher claim ['no']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:21098230` Andrea Bonito, Albert Cohen, Ronald DeVore et al. (2016). "Diffusion Coefficients Estimation for Elliptic Partial Differential Equations". Corollary 3.8. http://arxiv.org/abs/1609.05231v2 — cited by ['P-008', 'P-014']; researcher claim ['no']; harness screen **no**; conflicts: different equation class (theorem is not about parabolic/diffusion equations); theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:22602562` Guillaume Bal, Gunther Uhlmann (2011). "Reconstruction of coefficients in scalar second-order elliptic equations from knowledge of their solutions". Theorem 2.1. http://arxiv.org/abs/1111.5051v1 — cited by ['P-008']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:26758904` Andrea Montanari, Pierfrancesco Urbani (2025). "Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks". Lemma J.4. http://arxiv.org/abs/2502.21269v3 — cited by ['P-010', 'P-016']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-013', 'P-014', 'P-017', 'P-019', 'P-020']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:23892151` Jan Bohr (2022). "A Bernstein--von-Mises theorem for the Calderón problem with piecewise constant conductivities". Theorem 1.1. http://arxiv.org/abs/2206.08177v1 — cited by ['P-017']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:19588153` Florio M. Ciaglia, Jürgen Jost, Lorenz Schwachhöfer (2020). "Differential geometric aspects of parametric estimation theory for states on finite-dimensional C*-algebras". Proposition 2. http://arxiv.org/abs/2010.14394v2 — cited by ['P-020']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-01 *Augmentation-off control for the P-013 anchor: torch_mlp raw/log-k with noise_aug=0*: inconclusive — method did not produce a full-fidelity validation result
- E-03 *Anchor cell of the shared MLP factorial: log-k vs raw-k targets at fixed sigma-scale augmentation (1 seed per cell, ful…*: inconclusive — method did not produce a full-fidelity validation result
- E-04 *Architecture-matched deep ensemble MLP with exact raw-unit sigma=0.001 noise augmentation and k-space averaging*: inconclusive — method did not produce a full-fidelity validation result
- E-06 *Noise-floor PCA in the isotropic (raw-unit) noise metric with box-constrained k-space MMSE loss: whitened deep ensemble*: inconclusive — method did not produce a full-fidelity validation result

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-mathematical — P-001 *Log-linearisation of diffusive decay*: blocking issue: Missing evidence field (deterministic rule).
- ABANDONED in round 2 by researcher-mathematical — P-002 *Kernel ridge regression on log features*: blocking issue: Missing evidence field (deterministic rule).
- ABANDONED in round 2 by researcher-ml — P-004 *Kernel ridge on sqrt features with PCA denoising*: Duplicate of P-002 (same KRR + PCA-32 setup), and the sqrt feature transform is ill-defined for the near-zero/negative noisy readings expected at the end sensors. Its one informative contrast (feature transform) is bett…
- ABANDONED in round 3 by researcher-ml — P-009 *Factorial cell: torch_mlp on log features with log-k targets (contrast against raw-feature arms)*: Negative/null result. At pilot fidelity (300 train, 60 val), log features (0.0542) are indistinguishable from raw features (0.0526-0.0587, seed spread ~0.006). The predicted degradation from near-zero log-feature noise …
- PRUNED — idea-005 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted
- PRUNED — idea-007 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted

### 3.5 Remaining uncertainty

- E-01 (Augmentation-off control for the P-013 anchor: torch_mlp ra…): challenge: the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until tested
- E-01 (Augmentation-off control for the P-013 anchor: torch_mlp ra…): challenge: Rule finding: the evidence item calls local identifiability 'proven' for an elliptic Calderón analogue and marks it 'supports', while the citation is marked applicability 'no'. A theorem that does not apply c…
- E-01 (Augmentation-off control for the P-013 anchor: torch_mlp ra…): challenge: The ±10% relative equivalence band comes from a 60-row, 300-train pilot pair. It may not be the right yardstick at full data. On 750 shared val rows, a paired per-sample comparison has a much smaller SE (≈ pe…
- E-01 (Augmentation-off control for the P-013 anchor: torch_mlp ra…): challenge: The comparator is P-013 aug2, which is now operationalised as P-019 cell (a). P-017 should state that it reuses P-019(a) rather than a separate P-013 run. Otherwise the comparator might be the 300-row pilot.
- E-02 (Learned-emulator information arbiter: Jacobian CRLB for rel…): challenge: The emulator accuracy gate is in the wrong units. emu_rel_std < 2% of channel std can still be many times σ=0.001, depending on channel amplitude. Both the CRLB and the LM likelihood assume the residual is pu…
- E-02 (Learned-emulator information arbiter: Jacobian CRLB for rel…): challenge: Good value-accuracy does not imply good Jacobian accuracy. NN derivatives can be much worse than NN values. With n_emu=2, report the Jacobian disagreement between emulators (e.g. relative Frobenius difference…
- E-02 (Learned-emulator information arbiter: Jacobian CRLB for rel…): challenge: The unbiased CRLB is not a floor for the amortized learners. They are biased, prior-exploiting and box-constrained, and near box edges they can beat it. Compare the CRLB only on interior holdout samples, agai…
- E-02 (Learned-emulator information arbiter: Jacobian CRLB for rel…): challenge: The emulator is trained on noisy outputs. With 2550 pairs and a 192-wide net it will partially fit noise. Check residual whiteness.
- E-03 (Anchor cell of the shared MLP factorial: log-k vs raw-k tar…): challenge: The 0.006 'spread' mixes seed variance with 60-row val sampling noise. It is unclear whether P-005 and P-007 used the same 60 rows. At full data, compare (a) and (b) with a paired per-sample bootstrap on the …
- E-03 (Anchor cell of the shared MLP factorial: log-k vs raw-k tar…): challenge: The mechanism claim (heteroscedastic errors flattened by log) is testable. The prior-ratio-4 argument suggests a small effect, so a null gap is a likely and informative outcome. Pre-commit to dropping the tar…
- E-03 (Anchor cell of the shared MLP factorial: log-k vs raw-k tar…): challenge: Two hypotheses are conjoined: crossing 0.05, and log beating raw. Score them separately so that one failing does not obscure the other.
- E-03 (Anchor cell of the shared MLP factorial: log-k vs raw-k tar…): question: Confirm the rel_l2 definition (per-sample mean vs global ratio). It changes the expected log advantage and the CRLB in P-020.
- E-04 (Architecture-matched deep ensemble MLP with exact raw-unit …): challenge: 'Architecture-matched to P-013' needs evidence. Does the library torch_mlp use LayerNorm? If not, LayerNorm, fixed epochs vs early stopping, lr 0.002 vs 0.001, and scaler=none vs standard are all confounds al…
- E-04 (Architecture-matched deep ensemble MLP with exact raw-unit …): challenge: With scaler='none', the MLP sees raw temperatures. Channels have very different magnitudes, and some are near-zero (early times, end sensors). Unless the code standardises internally after noise injection, op…
- E-04 (Architecture-matched deep ensemble MLP with exact raw-unit …): challenge: Ensembling 3 members reduces variance by at most the between-seed component. That component is not yet measured at full data, so the hypothesis may hinge on a quantity that turns out small.
- E-04 (Architecture-matched deep ensemble MLP with exact raw-unit …): question: Is there internal standardisation in fit(), and is noise added before it?
- E-05 (Preprocessing contrast for kernel ridge: unscaled PCA vs st…): challenge: The only standardised-KRR evidence is the 300-row pilot (P-006 and P-012 are the same config and returned the identical value 0.07131). The proposal admits a matched full-data standardised run is needed. That…
- E-05 (Preprocessing contrast for kernel ridge: unscaled PCA vs st…): challenge: The hypothesis overlaps P-015's raw-unit PCA claim. That is useful as a cross-check, but P-018 should state that it is the KRR analogue, so its result can corroborate or contradict P-015.
- E-05 (Preprocessing contrast for kernel ridge: unscaled PCA vs st…): challenge: Without whitening, the RBF distance is dominated by the first 1-2 PCs, typically overall amplitude, which correlates with mean k. Then 30 PCs ≈ 2 effective PCs. This is a third explanation, beyond noise infla…
- E-05 (Preprocessing contrast for kernel ridge: unscaled PCA vs st…): challenge: KRR sits at about 0.071, so this arm cannot meet the success criterion. Its value is diagnostic only. Keep it cheap.
- E-06 (Noise-floor PCA in the isotropic (raw-unit) noise metric wi…): challenge: The prediction 4 ≤ r ≪ 64 is weak. A curved 4-D manifold of heat traces has a linear span that is much larger than 4, and smooth traces with O(0.1) amplitudes will exceed 4σ²=4e-6 for perhaps 10-30 components…
- E-06 (Noise-floor PCA in the isotropic (raw-unit) noise metric wi…): challenge: Components with signal variance just below 4σ² carry little information individually but non-trivial information jointly. Their Fisher contribution is J^T P J/σ², summed over components. Run a c=1 or c=2.3 (M…
- E-06 (Noise-floor PCA in the isotropic (raw-unit) noise metric wi…): challenge: The same √2σ effective-noise bias as P-016 applies when noise is added on top of already-noisy rows. Acknowledge it.
- E-06 (Noise-floor PCA in the isotropic (raw-unit) noise metric wi…): challenge: The response_to_critic field is empty. Confirm whether prior critiques were addressed.
- Unresolved disagreement ['P-020', 'P-016']: Is the 0.053-0.059 pilot floor approximation/variance-limited or information-limited on k1/k4? P-020's interior CRLB vs per-parameter learner error decides it.
- Unresolved disagreement ['P-015', 'P-018', 'P-019']: Does per-feature standardisation hurt by inflating noise channels (P-015/P-018), or is it harmless (the P-019 anchor uses a standard scaler)?
- Unresolved disagreement ['P-016', 'P-017']: Does extra noise injection help (P-016) or is it immaterial (P-017), given the √2σ over-shrinkage bias?
- Literature check INCOMPLETE: 0 failed queries (-), 4 skipped for budget, enabled=True
- Test-set sampling uncertainty: bootstrap 95% CI [0.01822, 0.02052] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **3 round(s)**: iteration cap reached (max_rounds=3); unresolved issues are documented.

| round | researcher | action | proposal | new version | origin | response to critic |
|---|---|---|---|---|---|---|
| 1 | researcher-mathematical | propose | - | P-001 | heuristic-fallback |  |
| 1 | researcher-mathematical | propose | - | P-002 | heuristic-fallback |  |
| 1 | researcher-ml | propose | - | P-003 | heuristic-fallback |  |
| 1 | researcher-ml | propose | - | P-004 | heuristic-fallback |  |
| 1 | researcher-experimental | propose | - | P-005 | llm |  |
| 1 | researcher-experimental | propose | - | P-006 | llm |  |
| 1 | researcher-exploratory | propose | - | P-007 | llm |  |
| 1 | researcher-exploratory | propose | - | P-008 | llm |  |
| 2 | researcher-mathematical | abandon | P-001 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-mathematical | abandon | P-002 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-ml | revise | P-003 | P-009 | llm | (1) Feature count: I wrongly stated 80. The facts give 4 sensors and 200 explicit steps, but not the output s… |
| 2 | researcher-ml | abandon | P-004 | - | llm | All points accepted. (1) It duplicates P-002 except for the sqrt transform, so its marginal information is on… |
| 2 | researcher-ml | propose | - | P-010 | llm |  |
| 2 | researcher-experimental | revise | P-005 | P-011 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-experimental | revise | P-006 | P-012 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-exploratory | revise | P-007 | P-013 | llm | (1) Duplicate of P-005: accepted. I am folding this arm into the shared factorial. It now owns only the targe… |
| 2 | researcher-exploratory | revise | P-008 | P-014 | llm | (1) Emulator bias in raw units: accepted. The revised code estimates per-channel emulator error on a 15% inte… |
| 3 | researcher-mathematical | propose | - | P-015 | llm |  |
| 3 | researcher-ml | abandon | P-009 | - | llm | Point by point. (1) Unmeasured clip/floor semantics and sub-5-sigma fraction: accepted. The pilot now answers… |
| 3 | researcher-ml | revise | P-010 | P-016 | llm | (1) Effective training noise of sqrt(2)*sigma: accepted, and I have relabelled it. It is a known bias, not a … |
| 3 | researcher-experimental | revise | P-011 | P-017 | llm | Accepted all points. (1) Duplicate and noise_scale semantics: the pilot evidence for the multiplier reading o… |
| 3 | researcher-experimental | revise | P-012 | P-018 | llm | Accepted. (1) Duplicate of P-006: the only distinct content is the preprocessing contrast, so the spec now us… |
| 3 | researcher-exploratory | revise | P-013 | P-019 | llm | (1) Two seeds give a weak spread estimate and do not fit the budget. Accepted. I adopt the critic's changed a… |
| 3 | researcher-exploratory | revise | P-014 | P-020 | llm | (1) Posterior mean vs MAP. Accepted in substance. LM toward a MAP point can only help if the amortized net is… |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.637 | fdbk-00048 |
| 2 | 6 | 0.746 | fdbk-00083 |
| 3 | 6 | 0.852 | fdbk-00109 |

### Idea evolution

**P-001 — Log-linearisation of diffusive decay** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, heuristic-fallback] `ridge/log+logy` status conjectured. *H:* Diffusive decay is approximately exponential in time, so log temperatures are near-linear in the conductivities; a regularised linear model in log space is a strong inverse.
  - prediction: `{"relative_to": "best_other", "improvement": -0.2, "expect": "better"}`; if it fails: the inverse map is strongly nonlinear; need a nonlinear learner
  - critic r1 (quality 0.539): Missing evidence field (deterministic rule). · The separation-of-variables argument covers late-time decay. With explicit FD and 200 steps under CFL (dt <= dx^2/(2*k_max) ~ 3.3e-4, so T is about 0.066 at mo… · Log features are undefined or heavy-tailed wherever u is about 0 plus noise (early times at outer sensors, near the Dirichlet ends). Clipping to epsilon create… · Treat this as a cheap lower-bound baseline. The realistic falsifiable claim is that it beats the mean baseline (0.33) by a stated margin, e.g. rel_l2 < 0.15. · A sensor reading of 0.0005 +/- 0.001 noise gives log features spanning several units from noise alone. A linear model with alpha=1e-3 will overfit these channe…

**P-002 — Kernel ridge regression on log features** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, heuristic-fallback] `kernel_ridge/log+pca32` status conjectured. *H:* The inverse map is smooth, so RBF kernel ridge on log features is data-efficient.
  - prediction: `{"relative_to": "baseline_mean", "improvement": 0.9, "expect": "better"}`; if it fails: bandwidth mis-specified or PCA discards information
  - critic r1 (quality 0.524): Missing evidence field (deterministic rule). · This overlaps heavily with P-004 and P-006: three kernel-ridge arms that differ only in feature transform and target. Consolidate them into one feature-transfo… · alpha=1e-3 and gamma_mult=0.5 are untuned. KRR performance is dominated by these two hyperparameters, so a single setting is not a test of 'smooth inverse map'. · Use log-k targets, matching the relative-error metric, rather than target_transform=none. · Does the harness tune gamma and alpha internally on val?

**P-003 — torch_mlp on log features with log-conductivity targets** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, heuristic-fallback] `torch_mlp/log+logy` status conjectured. *H:* A neural inverse on log features regressing log k captures the multiplicative structure of diffusive time scales and approaches the noise floor.
  - prediction: `{"relative_to": "baseline_mean", "improvement": 0.9, "expect": "better"}`; if it fails: the learner is sample- or capacity-limited; compare to the physics floor
  - critic r1 (quality 0.554): Missing evidence field (deterministic rule). · The feature dimension is stated as 80 here but about 800 in P-005 and P-007. Establish the true number of time samples. · Log features carry the same near-zero-signal hazard as P-001; an MLP will fit noise in the inflated channels. · Compare against an identical MLP on raw standardized features to isolate the effect of the feature transform. · What is the actual feature count: 4x20 or 4x200?
- v2 `P-009` r2 [revise, llm] `torch_mlp/log+logy` status conjectured. *H:* Log features hurt, they do not help. Information about k1/k4 is carried partly by low-amplitude readings at the near-boundary sensors (indices 5, 35). The log transform maps their N(0, 1e-3) noise onto large, heavy-tailed feature values, which lowers the effe…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.0, "expect": "worse"}`; if it fails: If log features match or beat raw features, then late-time log-linear decay information dominates and the near-zero hazard is immaterial. The team should then …
  - response to critic: (1) Feature count: I wrongly stated 80. The facts give 4 sensors and 200 explicit steps, but not the output sampling stride, so the true dimension is 4*T with T unknown (somewhere in 20 to 200). I removed every argument that depended on '8…
  - critic r2 (quality 0.71): The hazard mechanism depends on two unknowns: how the harness handles log of readings <= 0 (clip, floor or abs) and what fraction of readings fall below about … · The contradicting 'late-time slowest-mode decay' argument is probably inapplicable. With explicit FD, 40 points and 200 steps, stability requires dt <= dx^2/(2… · Replace 'late-time log-linear decay may favour log features' with 'the observation window likely ends before modal asymptotics (CFL-bounded T ~ 0.03)'. Check t… · If the pulse is wide (std of about 0.1 or more), the end sensors could start well above 5*sigma. Log features would then be benign everywhere except the earlie… · What exactly does features='log' do for non-positive values?

**P-004 — Kernel ridge on sqrt features with PCA denoising** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, heuristic-fallback] `kernel_ridge/sqrt+pca32` status conjectured. *H:* Diffusion makes the 80 readings highly correlated; PCA removes noise directions and an RBF kernel learns the smooth inverse map.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.1, "expect": "better"}`; if it fails: PCA truncation loses parameter information
  - critic r1 (quality 0.306): Missing evidence field (deterministic rule). · This is nearly a duplicate of P-002: same KRR, PCA-32 and hyperparameters, differing only in sqrt vs log features. Its marginal information is a single feature… · sqrt amplifies noise near zero (derivative 1/(2 sqrt u)) and is undefined for negative noisy readings. · Do PCA on unstandardized (or noise-whitened) features, since standardization destroys the 'noise in low-variance directions' premise. · How are negative readings handled under sqrt?

**P-005 — Smooth nonlinear map: noise-augmented MLP on standardized raw traces with log-k targets reaches rel_l2 <= 0.05** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The inverse map from 4-sensor time traces to (k1..k4) is smooth and well-conditioned over the prior box. Sensor noise (sigma=0.001) is small relative to the trace signal. The remaining error is therefore approximation error, not irreducible noise. A torch MLP…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 lands in 0.05-0.1 with errors concentrated on k1/k4, the limit is measurement information at the boundary segments rather than the learner. The n…
  - critic r1 (quality 0.91): noise_scale=0.001 conflicts with P-007's noise_scale=1.0 under the same harness. If noise_scale is a multiplier on sigma, this arm adds 1e-6 noise (an effectiv… · If training features already contain sigma=0.001 noise, augmentation trains at an effective sqrt(2)*sigma or more. This is mild, but the arm is not 'at the tru… · Sensor-segment mapping: verify whether segments are equal quarters (indices 0-9, 10-19, ...). If they are, sensors 5/15/25/35 sit mid-segment. · If the 200-step window is short (T ~ 0.066), the outer sensors see a weak signal that is mostly governed by k2/k3 transport. k1/k4 errors could then dominate e… · What are the exact semantics of noise_scale in the harness?
- v2 `P-011` r2 [revise, heuristic-fallback] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The inverse map from 4-sensor time traces to (k1..k4) is smooth and well-conditioned over the prior box. Sensor noise (sigma=0.001) is small relative to the trace signal. The remaining error is therefore approximation error, not irreducible noise. A torch MLP…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 lands in 0.05-0.1 with errors concentrated on k1/k4, the limit is measurement information at the boundary segments rather than the learner. The n…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.399): The executable spec (noise_scale=0.001) likely disables augmentation under the pilot-supported multiplier semantics, so it does not implement the stated hypoth… · This is essentially the P-013 log/aug2 cell with epochs 300 instead of 250. The pilot-backed reading (P-013) is that noise_scale is a multiplier on sigma. Unde… · 'Smooth and well-conditioned over the prior box' is asserted without evidence. The pulse starts at x=0.5, the k2/k3 interface, and with T possibly ~0.03 the ou… · Treat noise_scale as a multiplier of sigma, which the pilot supports. Set noise_scale=1.0 to match the hypothesis, or drop the arm and adopt the P-013 cell. · P-007 (presumably the same cell at 10% data) scored 0.0526 and P-005 scored 0.0587. A single full-data run below 0.05 would not separate the augmentation effec…
- v3 `P-017` r3 [revise, llm] `torch_mlp/raw+logy` status conjectured. *H:* At sigma=0.001 the sensor noise is small relative to trace variation over the prior box, except possibly at early times on end sensors. The anchor MLP's error is therefore dominated by approximation and information structure, not noise overfitting. Prediction…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If aug0 is much worse, noise regularisation is essential. Effort should then go to the explicit raw-unit ensemble (P-010) and posterior-mean-style shrinkage fo…
  - response to critic: Accepted all points. (1) Duplicate and noise_scale semantics: the pilot evidence for the multiplier reading outweighs my argument. Under that reading, noise_scale=0.001 means augmentation was effectively off, so the old spec did not test t…
  - critic r3 (quality 0.91): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · Rule finding: the evidence item calls local identifiability 'proven' for an elliptic Calderón analogue and marks it 'supports', while the citation is marked ap… · Replace the fixed ±10% band with a paired bootstrap CI on per-sample rel_l2 differences between aug0 and aug2 on the identical 750 val rows. Report seed varian… · With 3000 rows, a 512-wide net and 250 epochs, an unregularised net can memorise the single noise realisation. aug0 could then look worse without k1/k4 being i… · Does noise_aug=2 mean two noisy copies per row (data augmentation) or fresh noise every epoch? This changes the size of the predicted effect.

**P-006 — Competing explanation: traces lie on a low-dimensional manifold, so PCA-compressed kernel ridge matches or beats the MLP** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `kernel_ridge/raw+pca30+logy+aug1` status conjectured. *H:* The 4 sensor traces are smooth in time and driven by a 4-parameter family, so the observation manifold is about 4-dimensional and well captured by roughly 30 PCA components. Most of the remaining variance is noise. RBF kernel ridge on the PCA-whitened feature…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.2, "expect": "worse"}`; if it fails: Failure would show that the discriminative information lies in low-variance PCA directions, likely late-time or boundary-sensor tails, or that the map is too a…
  - critic r1 (quality 0.81): Standardize-then-PCA equalizes noise-dominated channels with signal channels. The '~4-D manifold, rest is noise' premise holds for unstandardized PCA, not this… · noise_aug=1 on KRR only increases n (O(n^3) solve) and duplicates near-identical points, which acts as a crude regularizer. It is not clearly informative. · Run PCA on raw unscaled features, or scale each channel by sigma, then whiten the retained components. · Does noise_aug double the KRR training set to 6000? If so, memory and time change.
- v2 `P-012` r2 [revise, heuristic-fallback] `kernel_ridge/raw+pca30+logy+aug1` status conjectured. *H:* The 4 sensor traces are smooth in time and driven by a 4-parameter family, so the observation manifold is about 4-dimensional and well captured by roughly 30 PCA components. Most of the remaining variance is noise. RBF kernel ridge on the PCA-whitened feature…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.2, "expect": "worse"}`; if it fails: Failure would show that the discriminative information lies in low-variance PCA directions, likely late-time or boundary-sensor tails, or that the map is too a…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.75): Its executable spec matches the already-piloted P-006 (kernel_ridge, pca30, log-y, aug1), which scored 0.0713 at 10% data. It is a duplicate unless the PCA pre… · Internal inconsistency: the assumptions say 'run PCA on raw unscaled features, or scale by sigma', but the spec uses scaler='standard'. Standard scaling inflat… · Use scaler='none', or a per-channel sigma scaler, before PCA, then whiten. This is the distinct contribution relative to P-006. · At 10% data the KRR pilot (0.0713) was already about 25% worse than the MLP pilots. Even a large data gain is unlikely to close the gap if gamma_mult=1 is mis-… · Can the harness run PCA without standardisation? If not, the hypothesis cannot be tested as stated.
- v3 `P-018` r3 [revise, llm] `kernel_ridge/raw+pca30+logy+aug1` status conjectured. *H:* Standardisation before PCA inflates noise-dominated, near-constant channels (early-time and boundary-sensor readings) to unit variance. PCA components are then wasted on noise, which degrades the isotropic RBF. With scaler='none', the 30 leading PCs capture t…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.065}`; if it fails: If unscaled is worse, the discriminative signal is in low-variance channels. Feature design should then emphasise per-sensor/per-time normalisation (log featur…
  - response to critic: Accepted. (1) Duplicate of P-006: the only distinct content is the preprocessing contrast, so the spec now uses scaler='none' before PCA. Everything else matches P-006 (pca 30, log-y, aug1, alpha 1e-3, gamma_mult 1), so the comparison isol…
  - critic r3 (quality 0.795): The only standardised-KRR evidence is the 300-row pilot (P-006 and P-012 are the same config and returned the identical value 0.07131). The proposal admits a m… · The hypothesis overlaps P-015's raw-unit PCA claim. That is useful as a cross-check, but P-018 should state that it is the KRR analogue, so its result can corr… · Treat 'scaler=none + PCA' as variance-weighted PCs unless whitening is confirmed. Add a whitened-PC variant, or a per-dimension lengthscale (ARD), as the discr… · Variance-weighted RBF on unscaled PCs ≈ RBF on PC1-PC2 only. In that case R ≥ 1.10 occurs even if no information lives in low-amplitude channels. · Does the harness PCA whiten?

**P-007 — Multiplicative-scale learned inverse: log-conductivity targets with physical-noise augmentation** (researcher-exploratory, family data-driven)
- v1 `P-007` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The inverse map from sensor time series to k is closer to smooth and homoscedastic when written in log k. Under the uniform prior, an MLP trained on log k with noise augmentation at the true sigma=0.001 will reach val rel_l2 < 0.05. The same MLP trained on ra…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If (a) is not better than (b), the scale argument is wrong, meaning sensitivity is not ~1/k at these sensors and times. If both fail with a large error on k1/k…
  - critic r1 (quality 0.528): The architecture, features, target and augmentation are the same as P-005. The only differences are noise_scale (1.0 vs 0.001) and the raw-target control. · The control (b) changes two factors at once (target transform and augmentation), so a 15% gap cannot be attributed to the log-scale argument. · Use a one-factor-at-a-time control: log vs none, both with the same noise_aug. · Is noise_scale=1.0 a multiplier on sigma (consistent with your assumption)? If so, P-005 is mis-specified.
- v2 `P-013` r2 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* With augmentation fixed (noise_aug=2, noise_scale=1.0, i.e. about sigma), training torch_mlp on log k gives lower val rel_l2 than training on raw k, by at least 10% and by more than 2-seed spread. The full-data log cell reaches val rel_l2 < 0.05. Mechanism: t…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If log does not beat none, the 1/k sensitivity argument is wrong at these sensors and time window. If both cells stall around 0.05-0.06 with errors concentrate…
  - response to critic: (1) Duplicate of P-005: accepted. I am folding this arm into the shared factorial. It now owns only the target-transform factor (log vs none). P-005 owns the augmentation factor. The executable spec is kept as the log/aug2 cell because the…
  - critic r2 (quality 0.904): The seed-spread criterion uses only 2 seeds, which is a weak spread estimate. Under the 400 worker-second budget, 4 full runs at about 60 s (240 s) would consu… · The mechanism test (absolute error rising with k under raw targets) is confounded by sensitivity. Even with log targets, absolute error can rise with k if sens… · Use the pilot pair P-005/P-007 as an empirical seed-plus-split spread (about 0.006) and run 1 seed per cell at full data. Add a second seed only if the gap fal… · Over [0.5, 2] the log-k transform is mild: log range 1.39 vs a raw range of 1.5, with max/min ratio 4. A 10% gain is plausible but not guaranteed, and a tie is… · Is rel_l2 averaged per sample or computed globally?
- v3 `P-019` r3 [revise, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* With augmentation fixed (noise_aug=2, noise_scale=1.0, i.e. about sigma), torch_mlp trained on log k reaches val rel_l2 < 0.05 on the full 750-row val set. It beats the raw-k target cell by at least max(5%, 0.006), where 0.006 is the empirical pilot spread. M…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If (a) stalls at 0.05-0.06 with error concentrated in k1/k4, the information-floor hypothesis gains weight, and the P-014 CRLB decides it. If log ties raw, the…
  - response to critic: (1) Two seeds give a weak spread estimate and do not fit the budget. Accepted. I adopt the critic's changed assumption. Each cell gets 1 seed at full data, and the gap is judged against the empirical seed-plus-split spread from the pilot p…
  - critic r3 (quality 0.905): The 0.006 'spread' mixes seed variance with 60-row val sampling noise. It is unclear whether P-005 and P-007 used the same 60 rows. At full data, compare (a) a… · The mechanism claim (heteroscedastic errors flattened by log) is testable. The prior-ratio-4 argument suggests a small effect, so a null gap is a likely and in… · The detectability threshold is set by the paired-CI width at n=750, not by the pilot spread. · Confirm the rel_l2 definition (per-sample mean vs global ratio). It changes the expected log advantage and the CRLB in P-020.

**P-008 — Simulator-free physics inversion: learn an emulator of the forward map from training pairs, then solve a noise-weighted…** (researcher-exploratory, family custom)
- v1 `P-008` r1 [propose, llm] `custom-solver` status conjectured. *H:* Training data contains (theta, observation) pairs, so the missing forward model can be learned as a differentiable emulator f: log k -> standardized observations. For each val sample, start from an amortized inverse-net estimate and minimise a residual weight…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If refinement hurts, the emulator's systematic bias exceeds the information gain. In that case the learned-forward route needs a higher-capacity or physics-str…
  - critic r1 (quality 0.925): Emulator bias is the crux. Standardized-unit residuals hide that sigma=0.001 is tiny in raw units, so the emulator RMS error in raw units must be compared agai… · With 4 parameters, Adam refinement for 150 steps is slower and less reliable than batched Gauss-Newton or Levenberg-Marquardt using the emulator Jacobian. · An emulator ensemble (e.g. 3 members) gives a calibrated per-channel error variance for the weighting, rather than a single global residual. · If the emulator RMSE is about 0.01 with structured (smooth-in-time) errors, the refined estimate converges to the emulator-optimal point, which can be biased b… · What is the emulator held-out RMSE in raw units relative to sigma=0.001?
- v2 `P-014` r2 [revise, llm] `custom-solver` status conjectured. *H:* A 2-member MLP emulator log k -> observations, trained on the training pairs, has raw-unit holdout RMSE within a small multiple (<5x) of sigma=0.001. Levenberg-Marquardt refinement of the inverse-net estimate under the emulator likelihood then reduces val rel…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If the gate rejects refinement with emu_rmse_raw >> sigma, a generic MLP emulator is the bottleneck. A physics-structured emulator (sum of decaying exponential…
  - response to critic: (1) Emulator bias in raw units: accepted. The revised code estimates per-channel emulator error on a 15% internal holdout of the training split, in raw units. It weights residuals by 1/(emu_var_raw + sigma^2), so the likelihood is honestly…
  - critic r2 (quality 0.865): For a squared-error-type metric, the Bayes-optimal estimator is the posterior mean. A well-trained, noise-augmented MLP approximates that mean. Emulator-LM con… · An emulator RMSE gate at 5*sigma = 0.005 is meaningless without the signal scale. If peak readings are O(0.1), then 0.005 is 5% error and the gate is loose. Re… · Compare against the posterior-mean benchmark: refinement should be judged against the noise-augmented MLP (P-010/P-013), not against its own un-augmented inver… · A sample whose posterior on k1 is nearly flat over [0.5, 2] has a posterior mean near 1.25. LM driven by a small prior_w can wander to any value within the lik… · Is the inverse net inside P-014 noise-augmented? If not, the refine_iters=0 baseline is weaker than the team's best arm and a 10% gain proves little.
- v3 `P-020` r3 [revise, llm] `custom-solver` status conjectured. *H:* (H1, primary) An MLP emulator log k -> traces trained on about 2550 pairs reaches holdout error well below the channel std (emu_rel_std < 2%). Its Jacobian-based CRLB for per-sample rel_l2 at sigma=0.001 is below 0.03. If so, the 0.053-0.059 pilot errors are …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.03, "expect": "better"}`; if it fails: If crlb_rel is small but learners stall near 0.05, the gap is approximation or optimisation, so invest in data efficiency (ensembles, physics-structured featur…
  - response to critic: (1) Posterior mean vs MAP. Accepted in substance. LM toward a MAP point can only help if the amortized net is far from Bayes. The revision makes three changes. First, the inverse initializer is now noise-augmented with exact raw-unit sigma…
  - critic r3 (quality 0.886): The emulator accuracy gate is in the wrong units. emu_rel_std < 2% of channel std can still be many times σ=0.001, depending on channel amplitude. Both the CRL… · Good value-accuracy does not imply good Jacobian accuracy. NN derivatives can be much worse than NN values. With n_emu=2, report the Jacobian disagreement betw… · Replace the emulator gate with: holdout residual RMSE ≤ 2σ per channel (in aggregate), and inter-emulator CRLB ratio within [0.8, 1.25]. Only then is the CRLB … · Suppose the emulator error is 5σ but well below 2% of channel std. The CRLB is then computed from a plausible J, but LM refinement chases emulator error. H2 wo… · How is the per-sample rel_l2 CRLB derived via the delta method? Is it sqrt(tr(W I⁻¹ W)) with W = diag(k)///k// in log-space?

**P-010 — Deep ensemble MLP with exact raw-unit sigma=0.001 noise augmentation (resolves the noise_scale ambiguity by constructio…** (researcher-ml, family data-driven)
- v1 `P-010` r2 [propose, llm] `custom-ml` status conjectured. *H:* Training on fresh N(0, 0.001^2) noise draws in raw sensor units at every minibatch makes the MLP approximate the posterior-mean map under the true observation model rather than interpolating a single noise realisation. Averaging 3 independently initialised me…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the sigma=0 and sigma=0.001 variants tie and both sit above 0.05 with k1/k4 errors dominating, the floor is information-limited rather than approximation-li…
  - critic r2 (quality 0.846): Training data already carry one noise realisation, so added noise makes the effective training noise sqrt(2)*sigma. The population minimiser is then E[log k / … · The ensemble average in log space followed by exp gives a geometric mean, which is biased low relative to the arithmetic posterior mean of k. For a rel_l2 on k… · Effective train noise is sqrt(2)*sigma, which gives a slightly over-smoothed posterior mean. Report this as a known bias, not a 'conservative' feature. · If the limit is an information floor for k1/k4, all augmented MLPs tie within seed spread and the secondary 'beats P-005/P-007' claim fails without the method … · Is the 3-member ensemble the same width and depth as P-013's [512, 256, 128]? If not, the comparison with P-005/P-007 confounds architecture with augmentation.
- v2 `P-016` r3 [revise, llm] `custom-ml` status conjectured. *H:* With architecture held equal to P-013 ([512,256,128], LayerNorm, GELU, log-k targets), two changes reduce val rel_l2 below 0.05 at full data: (a) fresh raw-unit N(0, 0.001^2) input noise at every minibatch and (b) arithmetic k-space averaging of 3 independent…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val > 0.05 and ties P-013 within seed spread while k1/k4 errors dominate, the floor is information-limited rather than variance- or approximation-limited. R…
  - response to critic: (1) Effective training noise of sqrt(2)*sigma: accepted, and I have relabelled it. It is a known bias, not a 'conservative feature'. The fitted map approximates E[log k / x + extra noise]. That posterior is slightly broader than the test-t…
  - critic r3 (quality 0.846): 'Architecture-matched to P-013' needs evidence. Does the library torch_mlp use LayerNorm? If not, LayerNorm, fixed epochs vs early stopping, lr 0.002 vs 0.001,… · With scaler='none', the MLP sees raw temperatures. Channels have very different magnitudes, and some are near-zero (early times, end sensors). Unless the code … · Report the single-member val score alongside the ensemble. This isolates the ensembling gain from the noise-injection gain within one run, at no extra cost. · Is there internal standardisation in fit(), and is noise added before it?

**P-015 — Noise-floor PCA in the isotropic (raw-unit) noise metric with box-constrained k-space MMSE loss: whitened deep ensemble** (researcher-mathematical, family data-driven)
- v1 `P-015` r3 [propose, llm] `custom-ml` status conjectured. *H:* The sensor-noise covariance is sigma^2 I in raw temperature units. Two consequences follow. (a) PCA computed on raw centred traces separates signal from noise cleanly: retain the components with eigenvalue > 4 sigma^2. That threshold sits above the Marchenko-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the result matches P-010/P-013 within noise, preprocessing and loss are not the bottleneck. The remaining gap is then either approximation-limited (resolve …
  - critic r3 (quality 0.77): The prediction 4 ≤ r ≪ 64 is weak. A curved 4-D manifold of heat traces has a linear span that is much larger than 4, and smooth traces with O(0.1) amplitudes … · Components with signal variance just below 4σ² carry little information individually but non-trivial information jointly. Their Fisher contribution is J^T P J/… · Treat c as a swept hyperparameter {2.3, 4}, selected on an internal holdout, not on val. · If r saturates at 64, the denoising claim is untested. The arm then reduces to 'MLP on 64 PCs'. · Are the retained PCs whitened (divided by sqrt(λ)) before the MLP? If so, injected noise in PC coordinates must be scaled to σ/sqrt(λ) accordingly.

- Critic round 1 portfolio view: The proposals are reasonable and the 0.05 target (about 0.066 RMS per parameter) looks attainable, but the portfolio is redundant and under-controlled. Priorities: (1) resolve the noise_scale semantics and the feature shape and positivity; (2) merge the MLP arms into a factorial and the KRR arms into one sweep with unstandardized PCA; (3) run P-008 with raw-unit emulator error and use its Jacobia… 7/8 active proposals are 'data-driven' families={'data-driven': 7, 'custom': 1}
- Critic round 2 portfolio view: Budget is binding: the claimed arms with seeds and ablations exceed 400 worker-seconds. Recommended priority order: 1. Zero-cost diagnostics: SNR per sensor, PCA spectrum, feature shape. 2. P-013 log/aug2 anchor plus the raw-target cell, 1 seed each, about 120 s, followed by the inference-time noise probe. 3. P-010 ensemble at sigma=0.001, about 30 s. 4. P-011 repurposed as the noise_aug=0 contro… 5/6 active proposals are 'data-driven' families={'data-driven': 5, 'custom': 1}
- Critic round 3 portfolio view: All six proposals are specific and falsifiable, and the remaining issues are execution-level rather than changes to what is worth testing. Recommended priority: P-019 (shared anchor), then P-020 (with its emulator gate re-expressed in σ units and an inter-emulator Jacobian check), then P-016 and P-015 (near-factorial on PCA+loss; log member scores and the spectrum), then P-017, then P-018 (diagno… 5/6 active proposals are 'data-driven' families={'data-driven': 5, 'custom': 1}

**Pilot experiments run during research** (small validation subsets):

- P-005 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.058722 on 60 val samples in 10.9 s
- P-006 `kernel_ridge/raw+pca30+logy+aug1`: ok rel_l2=0.07131 on 60 val samples in 2.23 s
- P-007 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.052602 on 60 val samples in 8.76 s
- P-009 `torch_mlp/log+logy`: ok rel_l2=0.054158 on 60 val samples in 9.71 s
- P-012 `kernel_ridge/raw+pca30+logy+aug1`: ok rel_l2=0.07131 on 60 val samples in 2.23 s

## 5. Literature retrieval (TheoremSearch)

22 queries, 0 failures, 76 distinct statements retrieved, 4 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search (cache) |
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant diffusion coefficient in par… | 4 refs via /search (cache) |
| researcher-ml | 1 | neural network approximation rates for Lipschitz inverse maps of finite-dimensional param… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of coefficient recovery in one-dimensional heat equation from boundary or poin… | 4 refs via /search |
| researcher-experimental | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search |
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-experimental | 1 | conditional stability estimate inverse coefficient problem heat equation finite-dimension… | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness of diffusion coefficient recovery in one-dimensional heat equation from measur… | 4 refs via /search |
| researcher-ml | 1 | neural network approximation of the solution operator of a parameter identification inver… | FAILED: budget |
| researcher-ml | 1 | learning rates of kernel ridge regression for smooth regression functions | FAILED: budget |
| researcher-mathematical | 1 | identifiability of piecewise-constant (discontinuous) conductivity in the parabolic (heat… | FAILED: budget |
| researcher-mathematical | 1 | Lipschitz stability for an inverse coefficient problem of a parabolic equation with point… | FAILED: budget |
| researcher-ml | 2 | torch_mlp on log features with log-conductivity targets | 4 refs via /search (cache) |
| researcher-ml | 2 | Training-set shape and the time sampling of features | 4 refs via /search |
| researcher-mathematical | 2 | Log-linearisation of diffusive decay | 4 refs via /search |
| researcher-experimental | 2 | Harness docs or code for noise_aug/noise_scale semantics | 4 refs via /search |
| researcher-exploratory | 2 | Binned abs error vs true k per segment, under raw vs log target | 4 refs via /search |
| researcher-experimental | 2 | Per-parameter val error and its correlation with k magnitude | 4 refs via /search |
| researcher-mathematical | 2 | Per-feature min/median of training observations, by sensor and time | 4 refs via /search |
| researcher-exploratory | 2 | identifiability of piecewise constant conductivity from interior point measurements heat … | 4 refs via /search |
| researcher-exploratory | 3 | Learning curve: val rel_l2 for the log/aug2 cell at 300, 1000, 3000 rows | 4 refs via /search |
| researcher-experimental | 3 | Harness source or doc string for noise_scale semantics | 4 refs via /search |
| researcher-experimental | 3 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search (cache) |
| researcher-ml | 3 | Factorial cell: torch_mlp on log features with log-k targets (contrast against raw-featur… | 4 refs via /search |
| researcher-exploratory | 3 | Fisher information / Cramer-Rao bound for piecewise-constant conductivity from interior p… | 4 refs via /search |
| researcher-ml | 3 | Per-sensor histogram of training readings / sigma, and the fraction of readings <= 0 at e… | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 177 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-017) | data-driven | `torch_mlp/raw+logy` | researcher-experimental | 0.91 | best-rated proposal in family 'data-driven' (quality 0.91) |
| E-02 | proposal (P-020) | custom | `custom-solver` | researcher-exploratory | 0.886 | best-rated proposal in family 'custom' (quality 0.89) |
| E-03 | proposal (P-019) | data-driven | `torch_mlp/raw+logy+aug2` | researcher-exploratory | 0.905 | quality 0.91 with diversity penalty |
| E-04 | proposal (P-016) | data-driven | `custom-ml` | researcher-ml | 0.846 | quality 0.85 with diversity penalty |
| E-05 | proposal (P-018) | data-driven | `kernel_ridge/raw+pca30+logy+aug1` | researcher-experimental | 0.795 | quality 0.80 with diversity penalty |
| E-06 | proposal (P-015) | data-driven | `custom-ml` | researcher-mathematical | 0.77 | quality 0.77 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-003 | E-02 | research | custom | `custom-solver` | 0.02753 | **0.021924** | 0.9956 | 0.02891 | 83.2 | 0.00802 | 0 | - | FINAL |
| idea-006 | E-05 | research | data-driven | `kernel_ridge/raw+pca30+logy+aug1` | 0.04536 | **0.024748** | 0.9944 | 0.03264 | 3.63 | 0.125 | 0 | - | evaluated |
| idea-011 | E-05 | refine | data-driven | `kernel_ridge/log+pca30+logy+aug1` | - | **0.05698** | 0.9701 | 0.07514 | 3.72 | 0.141 | 0 | - | evaluated |
| idea-012 | E-05 | refine | data-driven | `kernel_ridge/log+pca30+logy+aug1` | - | **0.081386** | 0.9391 | 0.1073 | 3.64 | 0.143 | 0 | - | evaluated |
| idea-013 | E-08 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.10797** | 0.8927 | 0.1424 | 1.05 | 0.167 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 1.33 | 0.191 | 0 | - | evaluated |
| idea-014 | E-08 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.1094** | 0.8898 | 0.1443 | 1.05 | 0.179 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.605 | 0.0511 | 0 | - | evaluated |
| idea-002 | E-01 | research | data-driven | `torch_mlp/raw+logy` | 0.06243 | **-** | - | - | - | - | - | - | ok |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw+logy+aug2` | 0.04946 | **-** | - | - | - | - | - | - | ok |
| idea-005 | E-04 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-007 | E-06 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **-** | - | - | - | - | - | - | ok |

## 8. Lineage of the final method

- `idea-003` `custom-solver` ← P-020 v3 (researcher-exploratory, r3, revise) ← P-014 v2 (researcher-exploratory, r2, revise) ← P-008 v1 (researcher-exploratory, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00226` [checkpoint] by scheduler: final method (single)
- `resu-00223` [result] by executor: exp-0026 rel_l2=0.01952 r2=0.9966 rmse=0.02569
- `hypo-00115` [hypothesis] by scheduler: idea-003 custom-solver (research)
- `plan-00222` [plan] by engineer-v2: exp-0026: final fit of 1 member(s) on train+val; test once
- `gate-00114` [gate] by gate: E-02: approved (custom-solver)
- `prop-00107` [proposal] by researcher-exploratory: P-020 v3 [revise] exploratory: Learned-emulator information arbiter: Jacobian CRLB for rel_l2, plus empirically gated emu
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s2 (8fafaa561b)
- `prop-00081` [proposal] by researcher-exploratory: P-014 v2 [revise] exploratory: Learned-forward-model inversion: emulator-weighted batched Levenberg-Marquardt refinement 
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `prop-00046` [proposal] by researcher-exploratory: P-008 v1 [propose] exploratory: Simulator-free physics inversion: learn an emulator of the forward map from training pairs

## 9. Compute and cost

- Execution compute: 342.6 / 400 worker-seconds over 23 jobs; wall 906 s
- By stage: baseline 2.6s, pilot 31.6s, screen 94.1s, evaluate 102.6s, refine 19.9s, final 91.7s
- LLM: 21 calls (by phase {'research': 19, 'execution': 2}), 1186 s, $4.244 reported · TheoremSearch: 22 queries (0 failed)

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

The evidence establishes that the objective is met. The selected `custom-solver` reached val rel_l2 = 0.0219 and held-out test rel_l2 = 0.0195 (R² = 0.9966, RMSE = 0.0257), well under the 0.05 success threshold. Because test was slightly better than val, there is no sign of selection overfitting. Two pre-registered predictions were supported. First, the custom method beat the best data-driven model (kernel ridge on raw features, PCA-30, log-k targets, aug1; val 0.0247) by 11.4%, above the expected ≥3% margin. Second, that KRR baseline came in well under its predicted ceiling of 0.065. So the data-driven family is also a viable solution here, and the custom method's advantage is real on val but modest in absolute terms (about 0.003 rel_l2). One caveat on the comparison: this is a single run (seed s2), the results do not describe what the custom solver internally does, and no seed spread is reported for the final arms. Whether the 11% gap exceeds run-to-run variance is therefore not established.

Only one hypothesis was actually tested and rejected. The log-feature MLP contrast was a null result at pilot fidelity (300 train / 60 val): log features scored 0.0542 against 0.0526–0.0587 for raw features, inside a seed spread of about 0.006. The predicted harm from near-zero noisy readings did not appear. The proposed counter-mechanism, late-time log-linear decay, did not apply because the CFL-bounded time window never reaches that regime. This is weak evidence, given the small pilot and a single architecture, that the feature transform is not what limits accuracy. The sqrt-feature KRR proposal was dropped as a duplicate and because sqrt is ill-posed for negative readings, so it was not a failed hypothesis. The log-linearisation and log-feature KRR proposals were abandoned only because a procedural rule fired on a missing evidence field. Their scientific merit remains untested and they should not be counted as negatives. What remains uncertain:
- how error is distributed across k1–k4 (identifiability of the end segments is unknown);
- how robust the result is to noise level and to seeds;
- whether the custom method's gain comes from physics structure or from tuning.

- Re-run the final custom solver and the best KRR over ≥3 seeds and report mean ± std, to test whether the 11% gain exceeds variance.
- Report per-parameter rel_l2 for k1–k4 and run the zero-cost SNR diagnostic on the sensor-5/35 readings, which would decide whether the log-feature caveat applies.
- Sweep the noise level (e.g. 0.5×, 2×, 4× the training σ) on simulated data to map where each method crosses the 0.05 threshold.
- Re-submit the log-linearisation and log-feature KRR proposals with their evidence fields filled in, so they get a real scientific verdict rather than a procedural one.
- Try a cheap hybrid: KRR prediction as a warm start or prior for the custom solver, to see whether the two families' errors are complementary.

_Automatic summary:_ 20 proposal versions over 3 research rounds; 8 methods advanced; 2 predictions supported, 0 contradicted. Final: ['custom-solver'] with test rel_l2 0.019521.
