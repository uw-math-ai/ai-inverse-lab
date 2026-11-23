# Research report (V2): heat_v2_no_forward_s1

Run `heat_v2_no_forward_s1_20261109_151330` · 2026-11-09 15:27:22 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **single** of `custom-solver`
- Validation rel_l2 (selection estimate): **0.019969** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018497**, r2 = **0.99693**, rmse = **0.024339** · bootstrap 95% CI for rel_l2: [0.01751, 0.01961]
- Per-target test rel_l2: k1 0.0265, k2 0.0064, k3 0.0056, k4 0.0247
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.087599 (vs 0.018497 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.052751 (vs 0.018497 in distribution)
- Inference latency: 4.66 ms/sample; fit 7.44 s; exported model reload check: max |Δ| = 1.98e-11
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **custom** `custom-solver` rel_l2=0.019969; **data-driven** `custom-ml` rel_l2=0.027355
- SUPPORTED — E-02 *Surrogate-forward weighted MLE: KRR forward map log k -> PCA/sigma traces, inverted per sample by damped LM with residu…*: val rel_l2 = 0.019969 vs predicted < 0.05
- SUPPORTED — E-03 *Branch of P-010: unscaled-PCA selection, then whitened scores, then RBF KRR with a CV-chosen truncation d (separates th…*: val rel_l2 = 0.027355 vs predicted < 0.035
- SUPPORTED — E-04 *Noise-isotropic PCA (no standardisation) + RBF kernel ridge in log-k: does SNR-ordered truncation preserve outer-segmen…*: val rel_l2 = 0.027555 vs predicted < 0.05
- SUPPORTED — E-06 *SNR-masked asinh(u/sigma) features + whitened PCA + KRR (log targets), against an identity-transform control on the sam…*: val rel_l2 = 0.042604 vs predicted < 0.06

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24476298` Siyu Cen, Bangti Jin, Yikan Liu et al. (2023). "Recovery of Multiple Parameters in Subdiffusion from One Lateral Boundary Measurement". Theorem 3.2. http://arxiv.org/abs/2307.03263v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-005']; researcher claim ['partial', 'unknown']; harness screen **unknown**
- `ts:thm:19142517` Marina Kleptsyna, Andrey Piatnitski, Alexandre Popier (2023). "Higher order homogenization for random non-autonomous parabolic operators". Lemma 3. http://arxiv.org/abs/1612.07478v3 — cited by ['P-002', 'P-010']; researcher claim ['no']; harness screen **no**; conflicts: theorem is stated for dimension >= 2; the problem is one-dimensional
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-003', 'P-004', 'P-012', 'P-013', 'P-019']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-004', 'P-005', 'P-010', 'P-011', 'P-012', 'P-013', 'P-014', 'P-017', 'P-020']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:22602562` Guillaume Bal, Gunther Uhlmann (2011). "Reconstruction of coefficients in scalar second-order elliptic equations from knowledge of their solutions". Theorem 2.1. http://arxiv.org/abs/1111.5051v1 — cited by ['P-006', 'P-015']; researcher claim ['no']; harness screen **plausible**
- `ts:thm:22758529` Rostyslav O. Hryniv (2011). "Analyticity and uniform stability in the inverse spectral problem for Dirac operators". Theorem 2.2. http://arxiv.org/abs/1102.2942v1 — cited by ['P-007']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-007', 'P-019']; researcher claim ['no', 'partial']; harness screen **unknown**
- `ts:thm:23892151` Jan Bohr (2022). "A Bernstein--von-Mises theorem for the Calderón problem with piecewise constant conductivities". Theorem 1.1. http://arxiv.org/abs/2206.08177v1 — cited by ['P-009', 'P-011', 'P-016']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:27030168` Jiang Zhu, Menghuai Xu, Ruohai Guo et al. (2025). "Joint Multitarget Detection and Tracking with mmWave Radar". Lemma 2. http://arxiv.org/abs/2412.17211v2 — cited by ['P-014', 'P-016', 'P-022', 'P-024']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26010563` Florent Bouchard, Alexandre Renaux, Guillaume Ginolhac et al. (2024). "Intrinsic Bayesian Cramér-Rao Bound with an Application to Covariance Matrix Estimation". Corollary 2 (Second Approximation of intrinsic Bayesian Cramér-Rao lower bound). http://arxiv.org/abs/2311.04748v3 — cited by ['P-016', 'P-024']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:19172311` Assia Benabdallah, Matania Ben-Artzi, Yves Dermenjian (2022). "Concentration and non-concentration of eigenfunctions of second-order elliptic operators in layered media". Corollary 4.12. http://arxiv.org/abs/2212.05872v1 — cited by ['P-017', 'P-018']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:25722499` David Hong, Laura Balzano, Jeffrey A. Fessler (2018). "Asymptotic performance of PCA for high-dimensional heteroscedastic data". Theorem. http://arxiv.org/abs/1703.06610v4 — cited by ['P-022', 'P-023']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-01 *Log-target MLP on noise-isotropic PCA (scaler=none) with sigma-matched augmentation; noise_aug ablation separates augme…*: inconclusive — method did not produce a full-fidelity validation result
- E-05 *Full-n log-target MLP (confirmation of pilot P-003) with bootstrap CI and error-covariance diagnostics*: inconclusive — method did not produce a full-fidelity validation result
- P-020 *KRR scaler cell: standard-scaled PCA40 traces, log targets, matched to P-010 hyperparameters* (conjectured): proposed but not executed
- P-018 *Missing control: linear ridge on the same unscaled-PCA40 log-target pipeline as P-010 (quantifies the nonlinearity of t…* (conjectured): proposed but not executed
- P-021 *Linear control: ridge on standard+PCA40 traces with log targets (nonlinearity test for P-013/P-010/P-015)* (conjectured): proposed but not executed
- P-025 *Missing linear control: ridge on standardised raw traces with log targets* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-exploratory — P-007 *Low-dimensional manifold view: a smooth RBF kernel interpolant on PCA-denoised traces, with log-con…*: duplicate of P-002
- ABANDONED in round 2 by researcher-exploratory — P-008 *Noise-matched augmentation in a neural inverse map with log-targets: is the bottleneck variance fro…*: duplicate of P-003
- ABANDONED in round 3 by researcher-mathematical — P-009 *Control: identity-target MLP (no augmentation) measures the Jensen bias of log-target training unde…*: Negative result: the identity-target MLP with no augmentation (pilot 0.0648, n=300) was worse than the log-target MLP (P-003 pilot 0.054). Second-order analysis bounds the Jensen bias at about 0.1-0.2% for per-parameter…
- ABANDONED in round 3 by researcher-mathematical — P-011 *Learned forward surrogate + per-sample likelihood inversion (surrogate-physics family, no true forw…*: Merged into P-016, which uses the same surrogate-forward MLE principle with a cheaper KRR surrogate and an analytic Jacobian. A sample-complexity argument (σ-level fidelity needs 1e-3 to 1e-4 relative accuracy on the to…
- PRUNED — idea-002 `torch_mlp/raw+pca48+logy+aug2`: projected full-fidelity cost 142s exceeds the 120s per-experiment limit
- PRUNED — idea-006 `torch_mlp/raw+logy+aug1`: projected full-fidelity cost 130s exceeds the 120s per-experiment limit
- PRUNED — idea-009 `svm/log+pca8`: projected full-fidelity cost 198s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Log-target MLP on noise-isotropic PCA (scaler=none) with si…): challenge: This is the best pilot (0.0327), but from 60 val points at n=300 with a single seed. MLP seed variance at this level can be 10-20%, so the attribution clone needs at least 2 seeds per arm or the 15% threshold…
- E-01 (Log-target MLP on noise-isotropic PCA (scaler=none) with si…): challenge: Unscaled PCA scores span about 2-3 orders of magnitude, from roughly 1e-1 for PC1 down to about 1e-3 near the floor. With default init, near-floor inputs barely move first-layer pre-activations. LayerNorm nor…
- E-01 (Log-target MLP on noise-isotropic PCA (scaler=none) with si…): challenge: PCA48 versus no-PCA also changes input dimension and noise averaging. The gain over P-003 could be pure denoising by projection rather than anything about scaling.
- E-01 (Log-target MLP on noise-isotropic PCA (scaler=none) with si…): question: Are PCA scores fed raw to the MLP? If yes, what is the first-layer weight norm per input component after training?
- E-02 (Surrogate-forward weighted MLE: KRR forward map log k -> PC…): challenge: The per-component gate of <= 2.25 noise units^2 asks the leading components to be fit within about 1.1 sigma. PC1 has std of perhaps 50-100 sigma, which implies relative accuracy of about 1e-2 to 1e-3 from 30…
- E-02 (Surrogate-forward weighted MLE: KRR forward map log k -> PC…): challenge: Box-constrained LM from a single initialiser can stick at the box faces. Report the fraction of solutions on the boundary, since the uniform prior puts real mass near the faces.
- E-02 (Surrogate-forward weighted MLE: KRR forward map log k -> PC…): challenge: The MLE or posterior mode is not the rel_l2-optimal estimator. Under a weakly identified k1/k4, the posterior mean (what direct regressors approximate) can beat the mode, so 'beats initialiser by 20%' is not …
- E-02 (Surrogate-forward weighted MLE: KRR forward map log k -> PC…): question: What is the initialiser exactly (direct-inverse KRR on which features)?
- E-03 (Branch of P-010: unscaled-PCA selection, then whitened scor…): challenge: gamma = g/d on whitened scores is sensible. However, gmult in {0.5, 2} is a coarse 4x grid, which may itself drive the comparison with P-010's untuned gamma. Give P-010 an equal grid.
- E-03 (Branch of P-010: unscaled-PCA selection, then whitened scor…): challenge: 3-fold CV selecting d in {10, 20, 40} confounds noise-floor location with bandwidth. Report the full CV table, not just the argmin.
- E-03 (Branch of P-010: unscaled-PCA selection, then whitened scor…): question: How many eigenvalues exceed 2*sigma^2? If fewer than 20, d=40 whitening injects pure noise by construction.
- E-03 (Branch of P-010: unscaled-PCA selection, then whitened scor…): revision_direction: Add d=30, or select d by the noise floor of the eigenvalue spectrum as a non-CV check.
- E-04 (Noise-isotropic PCA (no standardisation) + RBF kernel ridge…): challenge: The full-n claim rests on a 60-sample pilot (0.0445 at n=300). The standard error of a 60-point val mean is about 10-15% relative, so 0.0445 is consistent with anything from about 0.035 to 0.055 before any da…
- E-04 (Noise-isotropic PCA (no standardisation) + RBF kernel ridge…): challenge: The sklearn 'scale' gamma = 1/(n_features*Var(X)) on unscaled PCA scores is set almost entirely by PC1-PC3. A single fixed gamma_mult=1.0 is not a test of the method. Without a small gamma/alpha grid, a miss …
- E-04 (Noise-isotropic PCA (no standardisation) + RBF kernel ridge…): challenge: The SNR-ordering argument holds for selection. With i.i.d. noise of a single sigma, the unscaled PCA eigenvalue order is the order of signal variance plus sigma^2, and that part is sound. In the RBF metric, h…
- E-04 (Noise-isotropic PCA (no standardisation) + RBF kernel ridge…): counterexample: If k1 information lives in a PC with signal std of about 3 sigma while PC1 has std of about 100 sigma, an isotropic RBF with gamma tuned to PC1 is effectively constant along that direction. k1 would then…
- E-05 (Full-n log-target MLP (confirmation of pilot P-003) with bo…): challenge: the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until tested
- E-05 (Full-n log-target MLP (confirmation of pilot P-003) with bo…): challenge: This is not the pilot P-003 configuration. The pilot label is 'aug2' and this arm uses noise_aug=1, so it is not a clean confirmation.
- E-05 (Full-n log-target MLP (confirmation of pilot P-003) with bo…): challenge: With scaler=standard, noise_scale=0.001 is negligible if noise is injected in standardised units. The augmentation arm may then be an effective no-op, and the proposal admits the injection point is unverified.
- E-05 (Full-n log-target MLP (confirmation of pilot P-003) with bo…): challenge: The arm is largely dominated. At the same n, P-022's unscaled PCA48 MLP pilot reached 0.033 versus 0.054 for this family. Its 80 s cost is about 20% of the budget.
- E-06 (SNR-masked asinh(u/sigma) features + whitened PCA + KRR (lo…): challenge: The success threshold of 0.06 is weaker than the existing pilot (0.0445 for unscaled KRR). The arm should be judged against P-017, which uses the same whitened KRR, not an absolute 0.06.
- E-06 (SNR-masked asinh(u/sigma) features + whitened PCA + KRR (lo…): challenge: The median/u/ > 3 sigma mask selects on typical amplitude. Early-time outer-sensor readings that are sub-noise for high-k1 draws but informative for others are exactly the k1/k4 tail information the mechanism…
- E-06 (SNR-masked asinh(u/sigma) features + whitened PCA + KRR (lo…): challenge: The tail-linearisation premise (log u ≈ -d^2/(4 k_eff t)) holds only far from the Dirichlet ends and in the pre-reflection regime. Near sensor 5/35 the boundary sink makes the relation non-affine.
- E-06 (SNR-masked asinh(u/sigma) features + whitened PCA + KRR (lo…): challenge: Code is truncated in the review, so mask and whitening equality between the two arms cannot be verified.
- Unresolved disagreement ['P-010', 'P-017', 'P-020']: Is the KRR bottleneck component selection (favours unscaled) or the kernel metric (favours whitened)? The pilot favours unscaled over standardised. Whitened versus unscaled is unresolved, and the bandwidth must be tuned…
- Unresolved disagreement ['P-022', 'P-019']: Is the MLP gain due to PCA denoising and selection, to input scaling, or to augmentation? The noise_aug=0 clone and a whitened third cell would separate these.
- Unresolved disagreement ['P-024', 'P-017', 'P-022']: Does the likelihood mode beat direct posterior-mean regression for weakly identified k1/k4? This is unresolved, since the mode is not rel_l2-optimal.
- Test-set sampling uncertainty: bootstrap 95% CI [0.01751, 0.01961] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | (1) noise_scale fallback: the critic is right and my fallback was backwards. If noise_scale were absolute or … |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | Pilot evidence: KRR (standard scaler, pca=20, log target) gave 0.0752 on 60 val samples. That was trained on … |
| 2 | researcher-mathematical | propose | - | P-011 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-012 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-ml | revise | P-004 | P-013 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-experimental | revise | P-005 | P-014 | llm | (1) PCA/StandardScaler confound: accepted. I am not switching to pca=0, because that would make this arm a ne… |
| 2 | researcher-experimental | revise | P-006 | P-015 | llm | (1) Log of values <= 0: accepted, and the harness behaviour is unknown. I replace the harness 'log' feature w… |
| 2 | researcher-experimental | propose | - | P-016 | llm |  |
| 2 | researcher-exploratory | abandon | P-007 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 2 | researcher-exploratory | abandon | P-008 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 3 | researcher-mathematical | abandon | P-009 | - | llm | (1) Bias magnitude. Accepted. The bias is exp(s^2/2)-1, which is about 0.1% at s≈5% in log k. That is far bel… |
| 3 | researcher-mathematical | branch | P-010 | P-017 | llm | The pilot now speaks to this. Unscaled PCA40 KRR (log target) gave 0.0445 at n=300 on 60 val. The standard-sc… |
| 3 | researcher-mathematical | abandon | P-011 | - | llm | (1) Fidelity requirement. Accepted. The leading PC scores are 1e2-1e4 σ, so σ-level residuals need 1e-3 to 1e… |
| 3 | researcher-mathematical | propose | - | P-018 | llm |  |
| 3 | researcher-ml | revise | P-012 | P-019 | llm | (1) Agreed. This arm is the full-n confirmation of pilot P-003. The 60-sample pilot val (0.054) is not eviden… |
| 3 | researcher-ml | revise | P-013 | P-020 | llm | (1) Agreed. This is P-004 at full n (pilot 0.0753), and its value is as the standard-scaler cell of the scale… |
| 3 | researcher-ml | propose | - | P-021 | llm |  |
| 3 | researcher-experimental | revise | P-014 | P-022 | llm | The pilot is the main new evidence, and I weigh it above argument. The P-014 pilot reached val rel_l2 = 0.032… |
| 3 | researcher-experimental | revise | P-015 | P-023 | llm | (1) asinh makes the noise heteroscedastic, which can promote noise-dominated tails in unscaled PCA: accepted.… |
| 3 | researcher-experimental | revise | P-016 | P-024 | llm | Blocking issue (context attributes): fixed. lo=log 0.5, hi=log 2.0 and sig=1e-3 are now in params and read wi… |
| 3 | researcher-experimental | propose | - | P-025 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.705 | fdbk-00044 |
| 2 | 8 | 0.835 | fdbk-00080 |
| 3 | 10 | 0.85 | fdbk-00115 |

### Idea evolution

**P-001 — Log-target neural surrogate inverse with noise augmentation exploits positivity and one-sensor-per-segment identifiabil…** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from noise-free sensor traces to k is single-valued and smooth on the box [0.5,2]^4. Learning it in log-k space with a regularised MLP trained on noise-augmented simulated traces will reach val rel_l2 <= 0.05. The residual error will be dominated by t…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the outer-k errors dominate, the outer segments are practically non-identifiable at sigma = 1e-3 within the observation window, and the error floor is set b…
  - critic r1 (quality 0.408): Confirm noise_scale units. Under absolute or feature-std-relative semantics, noise_scale = 1.0 corrupts training. The proposal's stated fallback is incorrect. · The noise_scale fallback is backwards. If noise_scale is absolute, 1.0 is about 1000x sigma and of the same order as the pulse signal. If it is relative to the… · The architecture, optimizer, epochs and log target are identical to P-003. The only difference is noise_scale (1.0 vs 0.001), so this is a duplicate unless it … · Read the harness source to fix noise_scale semantics before running. Do not assume either reading. · Suppose noise_scale is absolute. Augmented copies then carry noise around 1.0 on features of O(0.1–1). The MLP would learn mostly from the clean copies, or reg…
- v2 `P-009` r2 [revise, llm] `torch_mlp/raw` status conjectured. *H:* Training the same MLP as P-003 with an identity target and no noise augmentation gives val rel_l2 within ±10% of the log-target version. If the log-target bias is material, the identity target will be better on the worst-identified parameters (expected k1/k4)…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If identity is clearly worse, the log reparametrisation's conditioning benefit dominates and the Jensen bias is negligible. In that case log targets should be …
  - response to critic: (1) noise_scale fallback: the critic is right and my fallback was backwards. If noise_scale were absolute or relative to feature std, 1.0 would corrupt the training signal rather than be harmless. I withdraw that assumption. (2) Duplicate …
  - critic r2 (quality 0.77): The Jensen bias is about exp(Var[log k/y]/2)-1. At a posterior width of 5% in log k this is 0.00125 relative, which is 40x smaller than the 0.05 budget. The bi… · The design is confounded. Compared with P-012/P-003 it changes two factors at once: the target transform and the augmentation. The clean reference it names, P-… · Treat the Jensen bias as a second-order effect, predicted to be under 0.2% relative unless per-parameter relative errors exceed about 15%. · If the per-parameter relative RMSE is about 4% for all k_i, the predicted Jensen gap is about 0.08%. Any observed identity-vs-log difference would then come fr… · Which arm is the log-target, no-augmentation reference, and is it budgeted?

**P-002 — Low-dimensional manifold hypothesis: PCA-compressed traces plus RBF kernel ridge in log-k suffice** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, llm] `kernel_ridge/raw+pca20+logy` status conjectured. *H:* Noise-free sensor traces lie on a smooth 4-dimensional manifold embedded in a subspace of at most about 20 principal components, and the inverse map on it is smooth. RBF kernel ridge on 20 PCA components, with a log target, will therefore reach val rel_l2 <= …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If KRR beats ridge but misses 0.05, the bandwidth or alpha is mis-tuned, or 20 components truncate the outer-segment information; the next step is to sweep pca…
  - critic r1 (quality 0.895): StandardScaler before PCA breaks the main denoising argument. The sensor noise is i.i.d. and isotropic with known sigma, so PCA on unscaled (raw) traces alread… · alpha = 1e-3 and gamma_mult = 1 are untuned. KRR at n = 3000 in an intrinsic 4-D space is very sensitive to the bandwidth, and one setting cannot falsify the m… · Use scaler = none, or noise-whitened PCA, which is equivalent here, as the principled choice. Treat standard scaling as the variant to test. · Suppose the k1-induced trace variation has variance below that of 20 standardized noise-dominated features. Then pca = 20 drops it, and the k1 error approaches… · What is the cumulative explained variance at 10, 20 and 40 components, with and without standardization?
- v2 `P-010` r2 [revise, llm] `kernel_ridge/raw+pca40+logy` status conjectured. *H:* With unscaled PCA, components are ordered by signal variance relative to isotropic noise. The 40 leading components then retain the k1/k4 information, and RBF KRR on them with a log target reaches val rel_l2 <= 0.05 at n=3000. It also does at least as well as…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the unscaled run loses to the standardised one, the kernel metric (not component selection) is the bottleneck. In that case whitened scores are required, as…
  - response to critic: Pilot evidence: KRR (standard scaler, pca=20, log target) gave 0.0752 on 60 val samples. That was trained on only about 300 samples, so it neither falsifies nor confirms the 0.05 claim at n=3000. KRR error in an intrinsic 4-D space should …
  - critic r2 (quality 0.88): The SNR-ordered selection argument is correct for unscaled PCA under i.i.d. noise. However, the isotropic RBF on unscaled scores is dominated by the top 1-3 PC… · The pilots show pca20 = 0.0752 and pca40 = 0.0753 under standard scaling. Components 21-40 added nothing. Either the information saturates by 20 components, or… · Separate the selection effect from the metric effect: unscaled PCA for selection, then whiten the retained scores (whiten=True) before the RBF. · If k1 enters mainly through PC 15-30, with variance about 1e-4 of PC1, an RBF with a bandwidth set by PC1 treats points differing only in k1 as identical. The … · How does the harness compute gamma_mult x 'scale' (1/(d*Var(X)))? On unscaled scores this is set by PC1's variance.

**P-003 — Log-target MLP with noise augmentation learns a near-Lipschitz inverse map from sensor traces to log k** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status literature-supported. *H:* The map from four noisy sensor time series to log k is smooth and close to injective on the box [0.5,2]^4. A LayerNorm MLP trained on standardized raw traces, with log-transformed targets and Gaussian noise augmentation at the true sensor noise level, will re…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 is roughly 0.05-0.1, the limit is sample efficiency, which points to kernel methods or more data from data augmentation. If k1/k4 errors dominate…
  - critic r1 (quality 0.84): The ~800-dim input is an assumption. The real feature dimension is 4 sensors × the number of retained time steps, which is unknown. Check it. · The cited Lipschitz-stability result (Robin coefficient, boundary data) does not transfer. The proposal acknowledges this, so the 'literature-supported' status… · Existing train features already contain one noise draw. Augmentation trains at an effective sigma of about √2·1e-3 on the augmented copies. This is mild but sh… · Does the harness use early stopping on a validation slice carved from train? The test split must stay untouched.
- v2 `P-012` r2 [revise, heuristic-fallback] `torch_mlp/raw+logy+aug2` status literature-supported. *H:* The map from four noisy sensor time series to log k is smooth and close to injective on the box [0.5,2]^4. A LayerNorm MLP trained on standardized raw traces, with log-transformed targets and Gaussian noise augmentation at the true sensor noise level, will re…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 is roughly 0.05-0.1, the limit is sample efficiency, which points to kernel methods or more data from data augmentation. If k1/k4 errors dominate…
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 0.916): This is the P-003 configuration run at full n. Its value is as the full-fidelity confirmation of the best pilot (0.054 at n=300, n_val=60), not as a new hypoth… · The units of noise_scale are still unverified. If it is relative to feature std, or applied after StandardScaler, 0.001 under-augments (harmless). If augmentat… · The cited Lipschitz-stability results do not apply (Robin coefficient, smooth coefficient). The supporting evidence is the empirical pilot, not the theorem. · Is augmentation noise added in raw units before the scaler?
- v3 `P-019` r3 [revise, llm] `torch_mlp/raw+logy+aug1` status conjectured. *H:* At n=3000, a LayerNorm MLP on standardized raw traces with log targets reaches val rel_l2 < 0.05, and the upper end of the bootstrap 95% CI is also below 0.055. Secondary: k1 and k4 have the largest marginal RMSE, and the val errors of adjacent segments are c…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the result is 0.05-0.07, the learning curve is flat between n=300 and n=3000, so the limit is information or noise rather than samples, and posterior-mean o…
  - response to critic: (1) Agreed. This arm is the full-n confirmation of pilot P-003. The 60-sample pilot val (0.054) is not evidence of being below 0.05. The argument is the 10x data increase, and I will judge the result with a bootstrap 95% CI over all 750 va…
  - critic r3 (quality 0.85): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · This is not the pilot P-003 configuration. The pilot label is 'aug2' and this arm uses noise_aug=1, so it is not a clean confirmation. · Bootstrap over val captures sampling variance only, not optimisation or seed variance. Either run 2 seeds or state that the CI excludes seed variance. · Is noise injected before or after the scaler? Log the effective injected std in raw units.

**P-004 — Kernel ridge on PCA-compressed traces with log targets: a smooth interpolant of a 4-D manifold beats trees and linear m…** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `kernel_ridge/raw+pca40+logy` status conjectured. *H:* The sensor traces lie near a low-dimensional manifold, so the leading 40 PCA components retain essentially all of the information about k. RBF kernel ridge regression on these components, with log targets, will reach val rel_l2 < 0.05. It will also beat the b…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If kernel ridge is much worse than the MLP, the map has anisotropic, scale-varying structure that a single isotropic RBF cannot capture, which favours learned …
  - critic r1 (quality 0.865): This is the same method as P-002, differing only in pca (40 vs 20) and the threshold for the nonlinearity claim (30% vs 1.5x). It should be one arm of a shared… · The claim that tail components are noise holds only for unscaled PCA. With StandardScaler, noise-dominated features are inflated (see P-002). · Explained-variance saturation should be measured, not assumed. · Which ridge/PLS configuration is the comparison baseline, and is its regularization tuned on val?
- v2 `P-013` r2 [revise, heuristic-fallback] `kernel_ridge/raw+pca40+logy` status conjectured. *H:* The sensor traces lie near a low-dimensional manifold, so the leading 40 PCA components retain essentially all of the information about k. RBF kernel ridge regression on these components, with log targets, will reach val rel_l2 < 0.05. It will also beat the b…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If kernel ridge is much worse than the MLP, the map has anisotropic, scale-varying structure that a single isotropic RBF cannot capture, which favours learned …
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 0.85): This is effectively P-004 (KRR, raw, pca40, log target) at full n, and P-004 piloted at 0.0753. Its incremental value is as the standard-scaler cell of the P-0… · The nonlinearity test (KRR beating ridge by 30% or more) is listed but ridge is not scheduled. Without it the second claim is not tested. · Under standard scaling, trailing components are not guaranteed to be noise. Measure the spectrum. · Is alpha identical to P-010 (1e-3 here vs 1e-4 there)? If not, the scaler comparison is confounded.
- v3 `P-020` r3 [revise, llm] `kernel_ridge/raw+pca40+logy` status conjectured. *H:* With alpha=1e-4 and gamma_mult=1 matched to P-010, KRR on standardized-then-PCA40 traces gives lower val rel_l2 than P-010 (scaler=none). Under an isotropic RBF, unscaled PCA scores let high-variance inner-sensor directions dominate the distance, which starve…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.0, "expect": "worse"}`; if it fails: If scaler=none wins, SNR-ordered truncation matters more than isotropic conditioning, and the 'whiten after selection' cell is the next test. If both KRR cells…
  - response to critic: (1) Agreed. This is P-004 at full n (pilot 0.0753), and its value is as the standard-scaler cell of the scaler comparison with P-010. I reframe it that way. (2) The ridge control is now scheduled as a separate new proposal on identical fea…
  - critic r3 (quality 0.765): The pilot already answers this comparison at n=300 in the opposite direction: standard pca40 gives 0.0753 and unscaled pca40 gives 0.0445. Its marginal informa… · The hypothesis text says standardised beats P-010, while the prediction field says 'expect worse'. The registered prediction must be made consistent, or the ou… · Match the bandwidth relative to the median pairwise distance, or CV-tune both cells on an identical grid, rather than matching gamma_mult. · Given the pilot, what full-n result would change any downstream decision?

**P-005 — Log-target MLP with sigma-matched noise augmentation: the inverse map is smooth and learnable to below 5% at sigma=1e-3** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, llm] `torch_mlp/raw+pca64+logy+aug2` status conjectured. *H:* The map from 4-sensor temperature histories to (k1..k4) is smooth and well-conditioned enough at sigma=0.001 that a torch MLP trained on log(k), with training-time Gaussian noise injection at the true sensor sigma, reaches val rel_l2 < 0.05. The competing exp…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If augmentation helps but the result stays above 0.05, model capacity or data quantity is the limit, and the next step is a larger PCA or more epochs. If both …
  - critic r1 (quality 0.88): This is the only MLP arm that includes an explicit noise_aug = 0 ablation, which is good. However, pca = 64 with StandardScaler confounds the ablation with pos… · The identifiability-floor criterion ('both variants plateau within 10% of each other') is weak evidence. Two equally underfit models also plateau together. An … · An identifiability floor should be estimated from the data, for example with a forward surrogate and Fisher information, not inferred from estimator agreement. · Two MLPs with the same capacity limit (for example, too few epochs) can plateau within 10% of each other above 0.05 with k1/k4 dominant, while KRR or a longer-… · Why pca = 64 for the MLP when the KRR arms use 20–40? Is any variance criterion behind it?
- v2 `P-014` r2 [revise, llm] `torch_mlp/raw+pca48+logy+aug2` status conjectured. *H:* With unscaled raw features, PCA keeps the k-informative directions above an isotropic sigma^2 floor. A log-target torch MLP on 48 such components, trained with Gaussian noise injection at sigma=1e-3, reaches val rel_l2 < 0.05. Competing explanation: the error…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the scaler=none variant beats P-003, the StandardScaler-before-PCA concern is confirmed and every KRR arm should switch to scaler=none. If it is worse, the …
  - response to critic: (1) PCA/StandardScaler confound: accepted. I am not switching to pca=0, because that would make this arm a near-duplicate of P-003. I keep PCA and set scaler='none'. The noise is i.i.d. with a single sigma on every reading, so on unscaled …
  - critic r2 (quality 0.88): The window estimate assumes the simulator runs at the explicit stability limit. The simulator could use a smaller dt, which would give T << 0.03 and even weake… · Feeding unstandardised PCA scores to an MLP gives inputs whose scales span orders of magnitude. LayerNorm normalises hidden activations, not the first-layer in… · Use unscaled PCA for selection, then standardise the 48 retained scores before the MLP. This keeps the noise-floor truncation and fixes the conditioning. · If outer-sensor peak amplitude is about 50 sigma, the transient-tail premise is wrong, and standard scaling would not inflate noise-dominated features much. · Is augmentation applied before PCA (raw units, so sigma maps to sigma per score) or after?
- v3 `P-022` r3 [revise, llm] `torch_mlp/raw+pca48+logy+aug2` status conjectured. *H:* Using unscaled raw features followed by PCA48 keeps the k-informative directions above a flat sigma^2 noise floor. A log-target torch MLP trained on these features with Gaussian noise injection at sigma=1e-3 reaches full-data val rel_l2 < 0.05. The pilot reac…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A full-data result >= 0.05, after a 0.033 pilot, would mean the pilot was noise from only 60 val points. The next step would then be a custom_ml PCA(whiten=Tru…
  - response to critic: The pilot is the main new evidence, and I weigh it above argument. The P-014 pilot reached val rel_l2 = 0.0327 (resu-00091, 60 val samples, train_fraction 300). P-003's pilot reached 0.054 at the same 300-sample training size. Point by poi…
  - critic r3 (quality 0.955): This is the best pilot (0.0327), but from 60 val points at n=300 with a single seed. MLP seed variance at this level can be 10-20%, so the attribution clone ne… · Unscaled PCA scores span about 2-3 orders of magnitude, from roughly 1e-1 for PC1 down to about 1e-3 near the floor. With default init, near-floor inputs barel… · The window estimate T ≈ 0.03 is consistent with explicit-FD stability: dt <= dx^2/(2 k_max) ≈ 1.6e-4, times 200 steps. The diffusion length is then sqrt(2kT) ≈… · Are PCA scores fed raw to the MLP? If yes, what is the first-layer weight norm per input component after training?

**P-006 — Control: is the inverse map nearly linear in log-space? Kernel ridge on log-features with log targets versus a ridge ba…** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `kernel_ridge/log+pca40+logy` status conjectured. *H:* Late-time temperatures decay roughly as exp(-lambda_n(k) t), so log-temperature features make the sensor-to-log(k) map close to linear in a low-dimensional subspace. On that view, RBF kernel ridge on log features with log targets should reach val rel_l2 < 0.0…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If raw features beat log features, the information sits mainly in the early transients, which argues against PCA or feature schemes that emphasise late times. …
  - critic r1 (quality 0.81): Log features are undefined or explode for noisy readings that are near zero or negative. This is likely for outer sensors at early times, where u is far below … · Late-time single-exponential decay requires the time window to reach the asymptotic regime. If T ≈ 0.03 and λ₁ ≈ π²·k̄ ≈ 12, then λ₁T ≈ 0.4, so the window is m… · Check whether the observation window reaches the late-time regime before using spectral-decay reasoning. · Suppose 30% of the entries are around 1e-4 with noise 1e-3, so their log is dominated by noise and clipping. After standardization these become high-variance j… · What does the harness 'log' transform do with values ≤ 0?
- v2 `P-015` r2 [revise, llm] `custom-ml` status conjectured. *H:* In the transient regime (T ~ 0.03), the k-information is in Gaussian-tail arrival amplitudes, where log u is affine in 1/k_eff. A noise-aware log, asinh(u/sigma), therefore makes the sensor-to-log k map smoother than raw features do. KRR on asinh features (PC…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If raw features beat asinh, the information is in high-amplitude near-pulse dynamics. Feature engineering should then emphasise the inner sensors and early tim…
  - response to critic: (1) Log of values <= 0: accepted, and the harness behaviour is unknown. I replace the harness 'log' feature with a noise-aware asinh(u/sigma) inside custom_ml. It is linear for /u/ << sigma, so sub-noise readings stay as bounded noise with…
  - critic r2 (quality 0.795): asinh(u/sigma) makes the noise heteroscedastic. Its std in transformed units is sigma/sqrt(sigma^2+u^2): about 1 for sub-noise readings and about sigma//u/ for… · asinh compresses the high-amplitude near-pulse readings, which likely carry most of the k2/k3 information. The test predicts that asinh beats raw features, but… · The premise that log u is affine in 1/k_eff holds only in the Gaussian-tail regime of a homogeneous medium. With piecewise k it is affine in an effective trave… · If all four sensors have peak SNR above 100, asinh behaves like log everywhere relevant. It then adds nonlinearity without a linearisation benefit, and raw fea… · Is the identity control's gamma grid rescaled to raw-feature units?
- v3 `P-023` r3 [revise, llm] `custom-ml` status conjectured. *H:* With sub-noise readings removed, the noise-aware log asinh(u/sigma) makes the map from features to log k smoother than raw amplitudes do. The reason is that, in Gaussian-tail transients, log u is approximately affine in an effective travel-time functional of …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If identity wins, the k-information is in the high-amplitude near-pulse dynamics and the tail-linearisation premise is refuted. Effort should then go to raw-am…
  - response to critic: (1) asinh makes the noise heteroscedastic, which can promote noise-dominated tails in unscaled PCA: accepted. Retrieved ts:thm:25722499 points the same way, since heteroscedastic noise degrades PCA. Whitening after PCA does not fix the sel…
  - critic r3 (quality 0.845): The success threshold of 0.06 is weaker than the existing pilot (0.0445 for unscaled KRR). The arm should be judged against P-017, which uses the same whitened… · The median/u/ > 3 sigma mask selects on typical amplitude. Early-time outer-sensor readings that are sub-noise for high-k1 draws but informative for others are… · Use a mask threshold based on the train 90th percentile of /u/, or report how many features each mask keeps, so that information loss is measurable. · A feature that is 0.5 sigma for k1 = 0.5 and 10 sigma for k1 = 2 has median below 3 sigma for some priors. It is dropped despite being strongly k1-informative. · How many of the original features survive the mask, per sensor?

**P-007 — Low-dimensional manifold view: a smooth RBF kernel interpolant on PCA-denoised traces, with log-conductivity targets** (researcher-exploratory, family data-driven)
- v1 `P-007` r1 [propose, llm] `kernel_ridge/raw+pca24+logy` status conjectured. *H:* The 160+ dimensional sensor time-series lies near a 4-D smooth manifold parameterised by (k1..k4). Projecting onto about 24 PCA components removes most of the sigma=0.001 noise without losing parameter information. An RBF kernel ridge interpolant regressing l…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If KRR is around 0.08–0.15 while an MLP does better, the inverse map has varying local length-scales (anisotropic sensitivity across segments) that a single is…
  - critic r1 (quality 0.51): This is effectively P-002 with pca 24 instead of 20 and gamma_mult 0.5 instead of 1. It adds no distinct hypothesis. · The density argument of about 7.4 points per axis is only a rough heuristic. The KRR error rate depends on the smoothness of the inverse map, which is the unkn…

**P-008 — Noise-matched augmentation in a neural inverse map with log-targets: is the bottleneck variance from sensor noise rathe…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* An MLP trained on raw traces overfits to noise-free-like structure: the train set has one fixed noise realisation per sample. Injecting fresh Gaussian noise at the physical level (sigma ≈ 0.001) at each augmentation round, and predicting log k, will lower val…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.2, "expect": "better"}`; if it fails: If augmentation gives no gain, noise is not the limiting factor at sigma=0.001. The error is then dominated by parameter-space coverage or weak identifiability…
  - critic r1 (quality 0.435): The model spec is the same as P-003 apart from noise_aug = 3. Its falsifiable test (versus noise_aug = 0) is the same as P-005's ablation. The genuinely new co… · The prediction ('20% better than the best data-driven model') is relative to an undefined reference. If KRR turns out best, the comparison mixes model class wi… · With sigma = 1e-3 and O(0.1) signals, the per-sample noise is probably not the bottleneck except at the outer sensors. Coverage of the parameter space may matt…

**P-011 — Learned forward surrogate + per-sample likelihood inversion (surrogate-physics family, no true forward calls)** (researcher-mathematical, family custom)
- v1 `P-011` r2 [propose, llm] `custom-solver` status conjectured. *H:* The forward map k -> traces is well-posed and smooth, so a neural surrogate trained on the 3000 simulated pairs approximates it with residuals comparable to sigma in PCA-score space. Per-sample maximum likelihood through this surrogate, initialised by KRR and…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.2, "expect": "better"}`; if it fails: If the refinement does not improve on its KRR initialiser and the surrogate residual is much larger than sigma, the forward map is not easier to learn than the…
  - critic r2 (quality 0.835): The leading PC scores span roughly 1e2-1e4 sigma. A surrogate residual of about sigma therefore needs 1e-3-1e-4 relative accuracy from a 3x256 MLP after 3000 A… · This is the same family as P-016 (learned forward surrogate plus per-sample MLE). The differences are the surrogate type (MLP vs KRR) and the optimiser (Adam v… · The forward map being 'easier to learn' is a claim about sample complexity at sigma-level fidelity. Smoothness alone does not establish it. · If the surrogate RMS is 20 sigma on PC1, the MLE is dominated by surrogate bias along PC1. It may then be worse than the KRR initialiser on whichever k_i loads… · Is the surrogate residual variance estimated per PC component? The proposal says so; confirm the code does it, rather than using a pooled scalar.

**P-016 — Surrogate-forward MLE: learn k -> PCA traces with KRR, invert per sample by damped Gauss-Newton in noise-whitened units…** (researcher-experimental, family custom)
- v1 `P-016` r2 [propose, llm] `custom-solver` status conjectured. *H:* The forward map (4-D log k -> traces) is smoother and easier to learn from 3000 samples than the inverse map, because it is a well-posed parabolic solve. A KRR forward surrogate on unscaled-PCA scores divided by sigma, with residual-weighted Gauss-Newton per …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the surrogate gate fails, learned forward models at n=3000 cannot reach sigma-level fidelity, and direct inverse regressors are the better use of data. If t…
  - critic r2 (quality 0.755): The code depends on self.lo/self.hi/self.sig from context attributes that are neither in params nor verified to exist. Hard-code them or verify the harness exp… · The code reads self.lo, self.hi and self.sig, but params contains only n_components, iters and lam. They come from 'context' attributes, whose existence is an … · The surrogate gate is a pooled mean over 40 components. Errors concentrate in the high-variance leading components, so a pooled MSE of 2.25 can hide PC1 errors… · The noise model per component should be sigma^2 + surrogate residual variance_j, not sigma^2. · Pooled MSE of 2.0 passes the gate, yet PC1 error of 8 sigma pulls the MLE off-truth along the k-combination that PC1 encodes.
- v2 `P-024` r3 [revise, llm] `custom-solver` status conjectured. *H:* The forward map from 4-D log k to traces is smoother than the inverse, so a KRR surrogate can be learned from 3000 samples. Per-sample damped LM in noise-plus-surrogate-variance-weighted units, box-constrained to the prior support, reaches val rel_l2 < 0.05 a…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the gate fails only on the leading components and LM still beats the initialiser, variance weighting rescued the approach, and a larger n or better surrogat…
  - response to critic: Blocking issue (context attributes): fixed. lo=log 0.5, hi=log 2.0 and sig=1e-3 are now in params and read with defaults. The code never touches context, and build_method is defined explicitly. (1) A pooled gate hides PC1 error: accepted. …
  - critic r3 (quality 0.925): The per-component gate of <= 2.25 noise units^2 asks the leading components to be fit within about 1.1 sigma. PC1 has std of perhaps 50-100 sigma, which implie… · Box-constrained LM from a single initialiser can stick at the box faces. Report the fraction of solutions on the boundary, since the uniform prior puts real ma… · Held-out MSE against noisy targets equals noise variance (1 unit) plus surrogate error. The gate therefore permits a surrogate error variance of 1.25, which sh… · What is the initialiser exactly (direct-inverse KRR on which features)?

**P-017 — Branch of P-010: unscaled-PCA selection, then whitened scores, then RBF KRR with a CV-chosen truncation d (separates th…** (researcher-mathematical, family data-driven, branched from P-010)
- v3 `P-017` r3 [branch, llm] `custom-ml` status conjectured. *H:* Unscaled PCA selects SNR-ordered directions; whitening the retained scores equalises them in the RBF metric. If the isotropic kernel on unscaled scores limits low-variance (k1/k4-carrying) directions, then whitened KRR with CV-selected d in {10,20,40} will be…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If whitening does not help, the bottleneck is not the kernel metric. The remaining candidates are the information limit (check against the P-016 surrogate CRLB…
  - response to critic: The pilot now speaks to this. Unscaled PCA40 KRR (log target) gave 0.0445 at n=300 on 60 val. The standard-scaled pca20/pca40 KRR gave 0.0752/0.0753 under the same pilot protocol. This is the best pilot in the portfolio and supports SNR-or…
  - critic r3 (quality 0.94): gamma = g/d on whitened scores is sensible. However, gmult in {0.5, 2} is a coarse 4x grid, which may itself drive the comparison with P-010's untuned gamma. G… · 3-fold CV selecting d in {10, 20, 40} confounds noise-floor location with bandwidth. Report the full CV table, not just the argmin. · How many eigenvalues exceed 2*sigma^2? If fewer than 20, d=40 whitening injects pure noise by construction.

**P-018 — Missing control: linear ridge on the same unscaled-PCA40 log-target pipeline as P-010 (quantifies the nonlinearity of t…** (researcher-mathematical, family data-driven)
- v1 `P-018` r3 [propose, llm] `ridge/raw+pca40+logy` status conjectured. *H:* The inverse map from SNR-ordered PCA scores to log k is substantially nonlinear over the factor-4 box. Eigenvalues scale like k/L² while eigenfunction amplitudes depend on contrast ratios, so traces depend on k through exp(-λ(k)t). Ridge on exactly the P-010 …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.5, "expect": "worse"}`; if it fails: If ridge is competitive (within 1.5×), the inverse map is nearly linear in log-k over SNR-ordered scores. Linear-Gaussian posterior reasoning (Laplace/CRLB) is…
  - critic r3 (quality 0.84): Ridge with alpha=1e-3 on unscaled scores: PCs with variance near 1e-6 get shrunk heavily, because the penalty alpha exceeds n*lambda_j*... only when alpha is l…

**P-021 — Linear control: ridge on standard+PCA40 traces with log targets (nonlinearity test for P-013/P-010/P-015)** (researcher-ml, family data-driven)
- v1 `P-021` r3 [propose, llm] `ridge/raw+pca40+logy` status conjectured. *H:* The inverse map from traces to log k is meaningfully nonlinear on [0.5,2]^4. Ridge on the same features as P-013 will have val rel_l2 at least 30% higher than the best KRR cell, and will not reach 0.05.
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.3, "expect": "worse"}`; if it fails: If ridge is competitive, the inverse is near-linear in log space, and further gains should come from denoising or estimator choice (posterior mean), not model …
  - critic r3 (quality 0.71): The matched nonlinear partner (P-013/P-020 standard-scaled KRR) is the weaker KRR pipeline, so this control anchors a cell that is already likely dominated. · Fixed alpha=0.01 with an unconfirmed claim that the harness tunes alpha.

**P-025 — Missing linear control: ridge on standardised raw traces with log targets** (researcher-experimental, family data-driven)
- v1 `P-025` r3 [propose, llm] `ridge/raw+logy` status conjectured. *H:* The map from sensor traces to log k is substantially nonlinear over the prior box [0.5, 2]^4. Ridge regression with log targets on standardised raw features will reach val rel_l2 between 0.08 and 0.20: clearly better than the mean predictor (0.33) but at leas…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.5, "expect": "worse"}`; if it fails: If ridge is competitive, the inverse map is near-linear in log k, the remaining error comes from variance or noise rather than nonlinearity, and regularisation…
  - critic r3 (quality 0.745): alpha=1.0 on standardised raw features is untuned. On standardised noise-dominated features, ridge can strongly overfit or underfit, and a factor-level conclus… · This overlaps P-021 (same scaler, PCA vs no-PCA). It is useful only as the linear partner of P-019.

- Critic round 1 portfolio view: The proposals are specific and falsifiable but heavily duplicated, and their combined cost exceeds the budget. Recommended plan, about 400 s: (1) a zero-cost check of the noise_scale semantics, which blocks P-001; (2) a cheap pilot on data SNR and arrival times plus the surrogate CRLB, to decide whether 0.05 is reachable and whether k1/k4 limit it; (3) a KRR sweep over pca {20, 40, 0} × scaler {s… 8/8 active proposals are 'data-driven' families={'data-driven': 8}
- Critic round 2 portfolio view: The empirical anchor is pilot P-003: log-target MLP at 0.054 with only 300 training samples, compared with KRR at 0.075, with KRR insensitive to pca 20 vs 40. This suggests the information content is sufficient for about 5% overall error, and that the full-n MLP (P-012) is the most likely to meet the 0.05 target. The scientifically most informative new arm is the surrogate-forward MLE (P-016). It… 6/8 active proposals are 'data-driven' families={'data-driven': 6, 'custom': 2}
- Critic round 3 portfolio view: All arms are specific and falsifiable, and further argument will not change what is worth running, so the portfolio should now be executed. The strongest evidence is pilot-level: unscaled-PCA pipelines beat standardised ones for both KRR (0.0445 vs 0.075) and MLP (0.033 vs 0.054) at n=300 with 60 val samples. That is a consistent direction, but with roughly 10-15% sampling error. Two pre-executio… 9/10 active proposals are 'data-driven' families={'data-driven': 9, 'custom': 1}

**Pilot experiments run during research** (small validation subsets):

- P-002 `kernel_ridge/raw+pca20+logy`: ok rel_l2=0.075233 on 60 val samples in 1.92 s
- P-003 `torch_mlp/raw+logy+aug2`: ok rel_l2=0.053947 on 60 val samples in 15.1 s
- P-004 `kernel_ridge/raw+pca40+logy`: ok rel_l2=0.075312 on 60 val samples in 1.91 s
- P-009 `torch_mlp/raw`: ok rel_l2=0.064784 on 60 val samples in 10.1 s
- P-010 `kernel_ridge/raw+pca40+logy`: ok rel_l2=0.044499 on 60 val samples in 1.94 s
- P-014 `torch_mlp/raw+pca48+logy+aug2`: ok rel_l2=0.032689 on 60 val samples in 19.3 s

## 5. Literature retrieval (TheoremSearch)

22 queries, 0 failures, 64 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-exploratory | 1 | uniqueness and Lipschitz stability for identification of piecewise constant diffusion coe… | 4 refs via /search (cache) |
| researcher-experimental | 1 | uniqueness and Lipschitz stability for identifying piecewise constant diffusion coefficie… | 4 refs via /search (cache) |
| researcher-ml | 1 | uniqueness of determining a piecewise constant diffusion coefficient in the one-dimension… | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness of piecewise constant diffusion coefficient one-dimensional heat equation from… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability inverse problem finitely many parameters conductivity analytic depend… | 4 refs via /search |
| researcher-experimental | 1 | conditional stability estimate inverse coefficient problem heat equation finite number of… | 4 refs via /search |
| researcher-mathematical | 2 | Harness code path for noise_aug / noise_scale | 4 refs via /search |
| researcher-experimental | 2 | Forward-surrogate Fisher information: fit KRR/MLP from k to traces on train, take Jacobia… | 4 refs via /search |
| researcher-exploratory | 2 | identifiability piecewise constant conductivity heat equation point measurements finite t… | 4 refs via /search |
| researcher-experimental | 2 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search (cache) |
| researcher-mathematical | 2 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search (cache) |
| researcher-ml | 2 | Per-sensor peak /u/ and time-of-arrival distribution across train | 4 refs via /search |
| researcher-ml | 2 | stability of piecewise-constant conductivity identification heat equation interior point … | 4 refs via /search |
| researcher-exploratory | 2 | Lipschitz stability inverse coefficient problem parabolic equation finitely many paramete… | 4 refs via /search |
| researcher-ml | 3 | Harness source for noise_aug/noise_scale: where in the pipeline noise is injected and in … | 4 refs via /search |
| researcher-ml | 3 | stability of piecewise-constant conductivity identification heat equation interior point … | 4 refs via /search (cache) |
| researcher-experimental | 3 | Per-sensor max /u/ / sigma and the fraction of time steps with /u/ > 3 sigma, over the tr… | 4 refs via /search |
| researcher-mathematical | 3 | Per-parameter relative RMSE of the P-003/P-012 MLP on val | 4 refs via /search |
| researcher-mathematical | 3 | CRLB for piecewise-constant diffusion coefficient from interior point sensors | 4 refs via /search |
| researcher-experimental | 3 | Does the harness apply PCA after the scaler, and is noise_scale absolute or relative? | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 181 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-022) | data-driven | `torch_mlp/raw+pca48+logy+aug2` | researcher-experimental | 0.955 | best-rated proposal in family 'data-driven' (quality 0.95) |
| E-02 | proposal (P-024) | custom | `custom-solver` | researcher-experimental | 0.925 | best-rated proposal in family 'custom' (quality 0.93) |
| E-03 | proposal (P-017) | data-driven | `custom-ml` | researcher-mathematical | 0.94 | quality 0.94 with diversity penalty |
| E-04 | proposal (P-010) | data-driven | `kernel_ridge/raw+pca40+logy` | researcher-mathematical | 0.925 | quality 0.93 with diversity penalty |
| E-05 | proposal (P-019) | data-driven | `torch_mlp/raw+logy+aug1` | researcher-ml | 0.85 | quality 0.85 with diversity penalty |
| E-06 | proposal (P-023) | data-driven | `custom-ml` | researcher-experimental | 0.845 | quality 0.84 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-018 *Missing control: linear ridge on the same unscaled-PCA40 log-target pipeline as…* — portfolio size limit (6); quality 0.84
- not advanced: P-020 *KRR scaler cell: standard-scaled PCA40 traces, log targets, matched to P-010 hy…* — portfolio size limit (6); quality 0.77
- not advanced: P-025 *Missing linear control: ridge on standardised raw traces with log targets* — portfolio size limit (6); quality 0.74
- not advanced: P-021 *Linear control: ridge on standard+PCA40 traces with log targets (nonlinearity t…* — portfolio size limit (6); quality 0.71

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-003 | E-02 | research | custom | `custom-solver` | 0.02779 | **0.019969** | 0.9963 | 0.02633 | 4.9 | 3.75 | 0 | - | FINAL |
| idea-004 | E-03 | research | data-driven | `custom-ml` | 0.04796 | **0.027355** | 0.9931 | 0.03607 | 8.59 | 0.0442 | 0 | - | evaluated |
| idea-005 | E-04 | research | data-driven | `kernel_ridge/raw+pca40+logy` | 0.04535 | **0.027555** | 0.993 | 0.03634 | 0.492 | 0.0492 | 0 | - | evaluated |
| idea-013 | E-07 | refine | data-driven | `torch_mlp/raw` | - | **0.030502** | 0.9914 | 0.04022 | 11 | 0.00695 | 0 | - | evaluated |
| idea-011 | E-04 | refine | data-driven | `kernel_ridge/raw+pca40+logy` | - | **0.030565** | 0.9914 | 0.04031 | 0.474 | 0.0459 | 0 | - | evaluated |
| idea-014 | E-07 | refine | data-driven | `torch_mlp/raw` | - | **0.03134** | 0.991 | 0.04133 | 7.43 | 0.0209 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **0.032246** | 0.9904 | 0.04252 | 8.89 | 0.00851 | 0 | - | evaluated |
| idea-007 | E-06 | research | data-driven | `custom-ml` | 0.06519 | **0.042604** | 0.9833 | 0.05618 | 10.8 | 0.048 | 0 | - | evaluated |
| idea-012 | E-04 | refine | data-driven | `kernel_ridge/log+pca40+logy` | - | **0.053235** | 0.9739 | 0.0702 | 0.488 | 0.0438 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 2.72 | 0.136 | 0 | - | evaluated |
| idea-002 | E-01 | research | data-driven | `torch_mlp/raw+pca48+logy+aug2` | 0.03035 | **-** | - | - | - | - | - | - | projected full-fidelity cost 142s exceeds the 120s per-experiment lim… |
| idea-006 | E-05 | research | data-driven | `torch_mlp/raw+logy+aug1` | 0.05144 | **-** | - | - | - | - | - | - | projected full-fidelity cost 130s exceeds the 120s per-experiment lim… |
| idea-009 | E-08 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 198s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-003` `custom-solver` ← P-024 v2 (researcher-experimental, r3, revise) ← P-016 v1 (researcher-experimental, r2, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00241` [checkpoint] by scheduler: final method (single)
- `resu-00238` [result] by executor: exp-0029 rel_l2=0.0185 r2=0.9969 rmse=0.02434
- `hypo-00125` [hypothesis] by scheduler: idea-003 custom-solver (research)
- `plan-00237` [plan] by engineer-v2: exp-0029: final fit of 1 member(s) on train+val; test once
- `gate-00124` [gate] by gate: E-02: approved (custom-solver)
- `prop-00112` [proposal] by researcher-experimental: P-024 v2 [revise] experimental: Surrogate-forward weighted MLE: KRR forward map log k -> PCA/sigma traces, inverted per sa
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s1 (8e97e94bdd)
- `prop-00076` [proposal] by researcher-experimental: P-016 v1 [propose] experimental: Surrogate-forward MLE: learn k -> PCA traces with KRR, invert per sample by damped Gauss-N
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 216.6 / 400 worker-seconds over 29 jobs; wall 832 s
- By stage: baseline 2.2s, pilot 50.3s, screen 63.3s, evaluate 53.5s, refine 28.8s, final 18.5s
- LLM: 21 calls (by phase {'research': 19, 'execution': 2}), 1266 s, $4.560 reported · TheoremSearch: 22 queries (0 failed)

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

The evidence establishes that the success criterion is met with margin. The selected `custom-solver` reached val rel_l2 = 0.0200 and held-out test rel_l2 = 0.0185 (R² = 0.997, RMSE = 0.024). That is well under the 0.05 target, and the val-to-test gap is small, consistent with no selection overfitting. The best data-driven model (`custom-ml`, val 0.0274) also clears the target, so the inverse map from four sensor traces to four conductivities is learnable from about 3000 simulated samples without GPUs. All four pre-registered predictions were supported and none were contradicted. However, most predictions were one-sided bounds ("< 0.05", "< 0.06"). They confirm adequacy rather than discriminate between mechanisms, and the 0.020 vs 0.027 ordering between families rests on a single split and seed with no reported uncertainty.

No hypothesis was formally refuted, but two lines were dropped. The identity-target MLP control (pilot 0.0648, n=300) did worse than the log-target MLP pilot (0.054). This is a negative result for raw-target training, but it is weak evidence: both pilots were small, n differed from the full runs, and the 1–2% gap far exceeds the 0.1–0.2% Jensen bias the team derived. So the advantage of log targets is more plausibly conditioning or scale, not bias. That explanation is conjecture. The learned-surrogate likelihood inversion was never run. It was merged into the KRR-surrogate MLE (P-016) on a sample-complexity argument that an MLP surrogate cannot reach the 1e-3 to 1e-4 relative accuracy needed on the top PCs. That argument is a prediction, not a measurement. The two other abandoned entries were duplicates. Several things remain uncertain:

- Which mechanism drives the solver's advantage.
- Error per conductivity segment. Segments far from sensors, or with low thermal flux, are likely less identifiable.
- Robustness to noise levels or k ranges outside the simulated prior.
- Whether results are stable across seeds.

- Report per-segment rel_l2 and error vs. true k on test-like val samples to locate identifiability limits, for example segments without a nearby sensor.
- Repeat the top two configurations over 3–5 seeds, or use bootstrap CIs, to check whether the solver-vs-ML gap (0.020 vs 0.027) is real.
- Run the log- vs identity-target comparison at full n=3000 with matched architecture, with and without the exp(s²/2) correction, to settle the target-transform question directly.
- Stress-test both families under shifted sensor noise σ (0.5×, 2×) and k drawn outside the training range, to measure robustness beyond the simulated distribution.
- Run P-016's per-component surrogate-fidelity gate to test the sample-complexity argument empirically, rather than leaving the surrogate-MLE line closed on theory alone.

_Automatic summary:_ 25 proposal versions over 3 research rounds; 8 methods advanced; 4 predictions supported, 0 contradicted. Final: ['custom-solver'] with test rel_l2 0.018497.
