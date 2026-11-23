# Research report (V2): heat_v2_no_forward_s0

Run `heat_v2_no_forward_s0_20261109_094428` · 2026-11-09 10:02:03 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **ensemble** of `custom-solver`, `torch_mlp/raw`, `custom-solver`, `torch_mlp/raw` (weights [5.0, 1.0, 1.0, 1.0])
- Validation rel_l2 (selection estimate): **0.022098** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.021039**, r2 = **0.99604**, rmse = **0.027683** · bootstrap 95% CI for rel_l2: [0.01963, 0.02241]
- Per-target test rel_l2: k1 0.0310, k2 0.0072, k3 0.0063, k4 0.0273
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.091316 (vs 0.021039 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.054044 (vs 0.021039 in distribution)
- Inference latency: 36.5 ms/sample; fit 72.6 s; exported model reload check: max |Δ| = 0
- Ensemble members on test: `custom-solver` 0.02189, `torch_mlp/raw` 0.02683, `custom-solver` 0.02533, `torch_mlp/raw` 0.02627
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **custom** `custom-solver` rel_l2=0.023148; **data-driven** `torch_mlp/raw` rel_l2=0.028221
- SUPPORTED — E-01 *Non-amortised local-linear Gauss-Newton inversion with train-only neighbourhood-size selection*: val rel_l2 = 0.027254 vs predicted < 0.045
- SUPPORTED — E-03 *Learned-surrogate forward model with full-covariance emulator-error whitening, box-prior MAP, and importance-sampled po…*: val rel_l2 = 0.023148 vs predicted < 0.05
- SUPPORTED — E-04 *Conditioning-limited anisotropy: do MLP per-component errors follow the local CRB ordering (inner pair < outer pair, wi…*: val rel_l2 = 0.02964 vs predicted < 0.05
- SUPPORTED — E-05 *Feature-transform ablation moved to the cheap model: sqrt-feature KRR matched to P-009 (log) and a raw twin*: val rel_l2 = 0.033241 vs predicted < 0.05
- SUPPORTED — E-06 *Target-transform single-factor pair, linear arm: raw standardized MLP, aug0, linear k*: val rel_l2 = 0.028326 vs predicted < 0.05

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-007', 'P-009', 'P-018', 'P-020']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22847254` Elena Beretta, Elisa Francini (2010). "Lipschitz stability for the electrical impedance tomography problem: the complex case". Theorem 2.1. http://arxiv.org/abs/1008.4046v1 — cited by ['P-001', 'P-009', 'P-020']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:22571724` Maarten V. de Hoop, Lingyun Qiu, Otmar Scherzer (2012). "Local Analysis of Inverse Problems: Hölder Stability and Iterative Reconstruction". Theorem 5.6. http://arxiv.org/abs/1108.3570v2 — cited by ['P-002', 'P-010', 'P-021']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-002']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:17546409` Ali Feizmohammadi (2025). "Reconstruction of 1-D evolution equations and their initial data from one passive measurement". Theorem 1.2. http://arxiv.org/abs/2501.10590v2 — cited by ['P-003']; researcher claim ['no']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-005', 'P-011', 'P-015', 'P-022', 'P-026', 'P-028']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:21098230` Andrea Bonito, Albert Cohen, Ronald DeVore et al. (2016). "Diffusion Coefficients Estimation for Elliptic Partial Differential Equations". Corollary 3.8. http://arxiv.org/abs/1609.05231v2 — cited by ['P-006', 'P-016', 'P-027']; researcher claim ['no']; harness screen **no**; conflicts: different equation class (theorem is not about parabolic/diffusion equations); theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-007', 'P-018', 'P-029']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-008', 'P-019']; researcher claim ['no', 'unknown']; harness screen **unknown**
- `ts:thm:23892151` Jan Bohr (2022). "A Bernstein--von-Mises theorem for the Calderón problem with piecewise constant conductivities". Theorem 1.1. http://arxiv.org/abs/2206.08177v1 — cited by ['P-010', 'P-011', 'P-012', 'P-017', 'P-021', 'P-022']; researcher claim ['no', 'partial']; harness screen **unknown**
- `ts:thm:26682376` Kelvin Hsu, Fabio Ramos (2019). "Bayesian Learning of Conditional Kernel Mean Embeddings for Automatic Likelihood-Free Inference". Lemma A.1. http://arxiv.org/abs/1903.00863v1 — cited by ['P-018']; researcher claim ['unknown']; harness screen **unknown**
- `ts:thm:26141779` Fangzheng Xie, Yanxun Xu (2019). "Bayesian Projected Calibration of Computer Models". Theorem 2 (Asymptotic Normality). http://arxiv.org/abs/1803.01231v2 — cited by ['P-018', 'P-029']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:20611090` Mélisande Albert (2018). "Concentration inequalities for randomly permuted sums". Lemma 2.2. http://arxiv.org/abs/1805.03579v1 — cited by ['P-023']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:21824669` Alen Alexanderian, Philip Gloor, Omar Ghattas (2014). "On Bayesian A- and D-optimal experimental designs in infinite dimensions". Theorem 2. http://arxiv.org/abs/1408.6323v2 — cited by ['P-029']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-02 *Noise-robust asinh-log KRR with train-only CV of bandwidth and ridge: is the 4-D inverse map smooth enough for kernel i…*: inconclusive — method did not produce a full-fidelity validation result
- P-027 *Bandwidth-vs-information test: whitened, train-CV-tuned KRR on unscaled raw PCA-20 (partner arm to P-013)* (conjectured): proposed but not executed
- P-024 *Tuned linear control: RidgeCV on raw standardised traces with a log-k target (how nonlinear is the inverse map?)* (conjectured): proposed but not executed
- P-025 *Inner-CV-selected KRR over PCA dimension x score whitening x (alpha, gamma): is the KRR plateau a hyperparameter artefa…* (conjectured): proposed but not executed
- P-028 *Target-transform single-factor pair, log arm: raw standardized MLP, aug0, log k (twin of P-015)* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 3 by researcher-ml — P-004 *Kernel ridge on a low-dimensional PCA embedding exploits the 4-D data manifold*: Negative result: standardised-log PCA-20 KRR piloted at 0.075, against 0.052 for PCA-40 at matched settings. The effective informative dimension in standardised log space exceeds 20. Dominated by P-009 and by the new CV…
- ABANDONED in round 3 by researcher-ml — P-013 *KRR on unscaled raw-trace PCA-20: the configuration where 'PCA discards noise' actually holds*: Duplicates P-016 (unscaled raw PCA KRR). For i.i.d. noise, the noise-whitened variant the critic requested is identical to unscaled PCA up to a scale absorbed by the bandwidth heuristic. Pilot evidence (PCA-20 at 0.075 …
- ABANDONED in round 3 by researcher-experimental — P-017 *Linear ridge baseline on raw standardized traces: quantifies the nonlinearity of the inverse map*: Duplicate of P-014. The linear-decodability control is merged there, with alpha tuned by CV on train. Recorded so that P-014's outcome is also read against my prediction: ridge rel_l2 >= 1.5x the best data-driven model …
- ABANDONED in round 3 by researcher-exploratory — P-019 *Single-factor test: does PCA-32 compression of the traces improve the noise-augmented log-target ML…*: This is withdrawn for design reasons, not refuted by a run. The required matched control is unscheduled and too costly, the 10% threshold is unpowered given unknown seed variance, and the proposal's ID references were i…
- PRUNED — idea-003 `custom-ml`: screen failed: Unclassified failure: RuntimeError: Cannot clone object <mlh_code_exp_0017.AsinhKRR object at 0x00000241F855EED0>, as the constructor either does not set or modifies parameter p

### 3.5 Remaining uncertainty

- E-01 (Non-amortised local-linear Gauss-Newton inversion with trai…): challenge: The prediction (< 0.045) is weak. The pilot already reached 0.0449 at n=300, and local-linear GN error should fall with neighbour density (a radius about 10^(1/4) ≈ 1.8x smaller at n=3000). Predict a learning…
- E-01 (Non-amortised local-linear Gauss-Newton inversion with trai…): challenge: Selecting K on a 250-sample hold-out gives a standard error of rel_l2 of about 0.002–0.003. Differences between K values may be smaller than that. Report hold-out scores with bootstrap error bars for all K.
- E-01 (Non-amortised local-linear Gauss-Newton inversion with trai…): challenge: Weights of 1/max(residual variance, σ²) floor at σ². Local-linear residuals in 800-D concentrate on a few directions, so per-feature diagonal weights ignore correlated linearisation error. This parallels the …
- E-01 (Non-amortised local-linear Gauss-Newton inversion with trai…): challenge: Low-k corner bias is acknowledged. The quartile breakdown is the right test.
- E-02 (Noise-robust asinh-log KRR with train-only CV of bandwidth …): challenge: P-020 and P-025 both test the same question: is the KRR plateau a tuning artefact? P-020 changes two factors against P-009 at once (asinh features and CV tuning). Without a tuned log-feature twin on the same …
- E-02 (Noise-robust asinh-log KRR with train-only CV of bandwidth …): challenge: The 3x3 grid uses multiples of a gamma heuristic, but the code does not state whether PCA scores are standardised or whitened before the RBF. This matters for whether gamma_mult=1 means the same thing as in P…
- E-02 (Noise-robust asinh-log KRR with train-only CV of bandwidth …): challenge: The hypothesis requires the inverse map to be 'single-valued'. That is untested. A smooth but locally ill-conditioned map (k1/k4 at low k2, k3) would produce the same plateau, and the falsifier cannot tell th…
- E-02 (Noise-robust asinh-log KRR with train-only CV of bandwidth …): challenge: asinh(x/3σ) with c=3e-3 is approximately linear for readings up to about 3e-3. This removes the log-clip tails P-020 targets, but it also compresses the dynamic range of the inner-sensor peaks, which are O(1e…
- E-03 (Learned-surrogate forward model with full-covariance emulat…): challenge: The pilot (0.084 at n=300) is the worst arm. The proposed fixes (prior Jacobian, full-covariance whitening) are sound, but the key unknown is emulator accuracy. The 10σ² criterion on PCs 1–10 says nothing abo…
- E-03 (Learned-surrogate forward model with full-covariance emulat…): challenge: Estimating a 40x40 residual covariance from a ~500-point hold-out is reasonable, but the homogeneity assumption is likely false. Emulator error should be largest near the box corners (low k: window-truncated,…
- E-03 (Learned-surrogate forward model with full-covariance emulat…): challenge: ESS ≥ 3 with 96 draws is a very low bar. Posterior-mean estimates with ESS ≈ 3 are noisy, and that noise can be of order the posterior width. Report the ESS distribution and the fallback rate.
- E-03 (Learned-surrogate forward model with full-covariance emulat…): counterexample: KRR emulation of a 4-D smooth map at n=3000 can reach 1e-3 relative error in the interior but degrades at the boundary. If val error concentrates at the box edges, that is emulator error, not prior missp…
- E-04 (Conditioning-limited anisotropy: do MLP per-component error…): challenge: Falsifier (b), the ordering of per-component errors, depends on CRB values from P-011's local Hessians. Those are emulator Jacobians fitted from neighbours, not true forward sensitivities, and emulator error …
- E-04 (Conditioning-limited anisotropy: do MLP per-component error…): challenge: Relative versus absolute error: rel_l2 is not normalised per component, and the CRB ordering is in absolute k. State whether err_kj is absolute RMSE or RMSE//k/, and use the same normalisation for the CRB.
- E-04 (Conditioning-limited anisotropy: do MLP per-component error…): challenge: The MLP config (log features, 200 epochs, aug0) differs from the pilot (aug2, n=300). Pilot evidence near 0.05 is therefore not directly predictive of the n=3000 result.
- E-04 (Conditioning-limited anisotropy: do MLP per-component error…): challenge: Window length: with T ≈ 0.033 and k = 0.5, the diffusion length is ≈ 0.18, while the pulse-to-outer-sensor distance is ≈ 0.37–0.40. In the low-k corner the outer sensors may see mostly noise. Errors on k1 and…
- E-05 (Feature-transform ablation moved to the cheap model: sqrt-f…): challenge: At fixed alpha and gamma_mult, changing the feature transform also changes the effective bandwidth, because the 'scale' heuristic sees different variance after transform, scaling and PCA. A ±0.01 difference m…
- E-05 (Feature-transform ablation moved to the cheap model: sqrt-f…): challenge: The raw twin is not a listed proposal and P-009's status this round is unclear. The triple only works if both companions run with identical settings.
- E-05 (Feature-transform ablation moved to the cheap model: sqrt-f…): challenge: The localisation criterion (k1/k4 ratio at least 1.5x the k2/k3 ratio) is computed from one deterministic fit per arm. Its sampling variance over val rows must be bootstrapped, or a 1.5x ratio could arise by …
- E-05 (Feature-transform ablation moved to the cheap model: sqrt-f…): counterexample: Under standard scaling, a column dominated by clipped noise has unit variance whatever the transform. The transform then changes the column's distribution shape, not its weight. The effect on RBF distanc…
- E-06 (Target-transform single-factor pair, linear arm: raw standa…): challenge: The paired bootstrap over val rows captures only val-sampling variance, not training-seed variance. The pilots show a seed spread of ≈0.001–0.002, which is comparable to a 10% relative effect at rel_l2 ≈ 0.03…
- E-06 (Target-transform single-factor pair, linear arm: raw standa…): challenge: The prior expected effect is small, as the author accepts (bias ≈ 0.1%). The pair costs about 80 s, roughly 20% of the total budget, to fix a factor that is probably inconsequential.
- E-06 (Target-transform single-factor pair, linear arm: raw standa…): challenge: The ID linkage is confused: P-028 calls itself the twin of P-015, but this is P-026. Confirm that P-026 is the revised P-015.
- E-06 (Target-transform single-factor pair, linear arm: raw standa…): counterexample: In linear k, MSE weights absolute error. In the k ≈ 2 corner, where outer-sensor information is weakest, absolute errors are largest. A linear-target MLP may therefore over-allocate capacity there, which…
- Unresolved disagreement ['P-022', 'P-029']: Whether local-linear or global KRR emulation of the forward map is more accurate at n=3000. The pilots favour local (0.045 versus 0.084).
- Unresolved disagreement ['P-021', 'P-022']: Whether the residual error is information/conditioning-limited (CRB) or estimator-limited.
- Unresolved disagreement ['P-020', 'P-023', 'P-025']: Whether the feature transform (asinh, log or sqrt) or tuning drives KRR's plateau.
- Test-set sampling uncertainty: bootstrap 95% CI [0.01963, 0.02241] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | Pilot evidence comes first. With only 300 training samples, KRR on log features with log targets reached val … |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | The pilot gave 0.0497 with 300 training samples (60 val samples; within noise of the threshold).  (1) Regress… |
| 2 | researcher-mathematical | propose | - | P-011 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-012 | llm | Blocking point (noise_scale semantics): I read the harness source, harness/methods.py, the _augment / noise-a… |
| 2 | researcher-ml | branch | P-004 | P-013 | llm | Experimental evidence first: the original arm (log features, standard scaler, PCA-20, alpha=1e-3, log target)… |
| 2 | researcher-ml | propose | - | P-014 | llm |  |
| 2 | researcher-experimental | revise | P-005 | P-015 | llm | (1) Outer-sensor SNR: agreed, it was unverified and I have weakened the claim. A rough CFL estimate (dx=1/39,… |
| 2 | researcher-experimental | revise | P-006 | P-016 | llm | (1) Scaler choice: accepted. With scaler='standard', low-amplitude tails are inflated to unit variance, so 'P… |
| 2 | researcher-experimental | propose | - | P-017 | llm |  |
| 2 | researcher-exploratory | revise | P-007 | P-018 | llm | Point-by-point. (1) Small-amplitude PCs that carry k1/k4 may be poorly emulated: accepted. I raised m from 24… |
| 2 | researcher-exploratory | revise | P-008 | P-019 | llm | (1) Two factors changed at once: accepted. The separate ablation is dropped, as the critic recommended. P-008… |
| 3 | researcher-mathematical | revise | P-009 | P-020 | llm | Point by point.  (1) Two factors changed at once (aug2 to aug0, and n from 300 to 3000). Accepted. The data-s… |
| 3 | researcher-mathematical | revise | P-010 | P-021 | llm | Point by point.  (1) The mirror-symmetry claim. Accepted; it was wrong. Taking grid centre 19.5, sensors 5, 1… |
| 3 | researcher-mathematical | revise | P-011 | P-022 | llm | The pilot is the strongest so far: 0.0449 at n_train=300 (resu-00088), with no training and in 2 s. This beat… |
| 3 | researcher-mathematical | propose | - | - | llm |  |
| 3 | researcher-ml | revise | P-012 | P-023 | llm | (1) Unmatched ablation: I accept this. Three ~40 s MLP arms are poor value given the critic's priority list, … |
| 3 | researcher-ml | abandon | P-004 | - | llm | Experiment outweighs my argument. The pilot at matched settings gives PCA-20 = 0.075 vs PCA-40 = 0.052 (n_tra… |
| 3 | researcher-ml | abandon | P-013 | - | llm | The critic is right that this overlaps P-016 (unscaled raw PCA-30 KRR). Under i.i.d. Gaussian noise, noise-wh… |
| 3 | researcher-ml | revise | P-014 | P-024 | llm | (1) Untuned alpha: I accept this. Fixed alpha=0.1 makes the 0.07 falsifier depend on alpha rather than on lin… |
| 3 | researcher-ml | propose | - | P-025 | llm |  |
| 3 | researcher-experimental | revise | P-015 | P-026 | llm | (1) Log-target bias. Accepted. With about 5% relative posterior std, E[k]/exp(E[log k]) - 1 ≈ s^2/2 ≈ 0.1%. T… |
| 3 | researcher-experimental | revise | P-016 | P-027 | llm | (1) Overlap with P-013. Accepted. I no longer run a separate unscaled PCA-30 KRR. The revision is built as P-… |
| 3 | researcher-experimental | abandon | P-017 | - | llm | Accepted. It duplicates P-014 (ridge on raw features), and the target difference is now covered by the P-015 … |
| 3 | researcher-experimental | propose | - | P-028 | llm |  |
| 3 | researcher-exploratory | revise | P-018 | P-029 | llm | First, the evidence. The pilot gave rel_l2 = 0.0836 with 300 training samples and n_val = 60. The amortised K… |
| 3 | researcher-exploratory | abandon | P-019 | - | llm | I accept all of the critic's points. (1) The control arm (P-005 at matched settings) is not scheduled, so the… |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.806 | fdbk-00044 |
| 2 | 12 | 0.78 | fdbk-00081 |
| 3 | 10 | 0.845 | fdbk-00121 |

### Idea evolution

**P-001 — Smooth kernel interpolation of the 4-D inverse map: log-temperature features with log-conductivity targets** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, llm] `kernel_ridge/log+pca40+logy+aug2` status conjectured. *H:* On the box [0.5,2]^4, the map from observations to k is single-valued and smooth. If so, an RBF kernel ridge regressor trained on log-transformed sensor features with log(k) targets, plus noise augmentation at sigma=0.001, will reach val rel_l2 < 0.05. If the…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the error stays above 0.05 with uniform per-component errors, the RBF bandwidth cannot capture the anisotropic sensitivity (k2 and k3 are strongly sensed, k…
  - critic r1 (quality 0.88): The smoothness argument is sound: explicit FD gives a forward map polynomial in k. But smoothness of the forward map does not imply a smooth or well-conditione… · Log features are dangerous here. With dt bounded by CFL (dt <= dx^2/(2*kmax), about 1.6e-4 for dx=1/39 and kmax=2), 200 steps give a total time of roughly 0.03… · Replace 'log features are safe via clipping' with 'outer-sensor SNR is unknown and possibly below 1 for early times'. Make the feature transform an explicit va… · If T is about 0.03 and the pulse is narrow, u at sensors 5 and 35 stays near 1e-4 or lower. The k1 and k4 information is then near zero, and any regressor's pe… · What clip/offset does the harness use for 'log' features?
- v2 `P-009` r2 [revise, llm] `kernel_ridge/log+pca40+logy` status conjectured. *H:* The inverse map from observations to log k is single-valued and smooth on [0.5,2]^4. Consequences: (a) RBF KRR on log features (PCA 40) with log targets, trained on all 3000 samples, reaches val rel_l2 < 0.05. (b) Its error is clearly below the 300-sample pil…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A plateau near the pilot value means KRR is not data-limited. Either the inverse is information-limited in k1 and k4, which the CRB from the new local-linear p…
  - response to critic: Pilot evidence comes first. With only 300 training samples, KRR on log features with log targets reached val rel_l2 = 0.052 on 60 val samples. At n_val = 60 the standard error is roughly ±0.005–0.01, so this is near the threshold, not a co…
  - critic r2 (quality 0.922): The comparison with the pilot changes two factors at once. The pilot (resu-00053) used aug2; this run uses noise_aug=0 and 10x more data. A gain or a plateau c… · Falsifier (b) puts the threshold at 0.045 against a pilot value of 0.052 measured on n_val=60. The pilot's own SE is about 0.005-0.01, so 'no clear gain' is un… · Treat the log clip (assumed log(max(x,1e-6))) as a possible source of heavy-tailed feature noise. Negative noisy readings at outer sensors collapse to -13.8, a… · If about 30% of outer-sensor early-time readings are clipped, two samples with identical k but different noise realisations can differ by O(10) standardised un… · What fraction of entries in the outer-sensor columns are <= 0 or below 3 sigma? This decides whether the log transform is benign.
- v3 `P-020` r3 [revise, llm] `custom-ml` status conjectured. *H:* The map from sensor traces to log k is smooth and single-valued on [0.5,2]^4. Two conditions together then suffice for an isotropic RBF interpolant on a 40-D PCA embedding to reach val rel_l2 < 0.05 at n=3000: (i) the features are noise-robust, via asinh(x / …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 >= 0.05 despite tuning and asinh features, isotropic kernel interpolation is the bottleneck. - Errors concentrated in k3/k4 (farther sensors, see P-0…
  - response to critic: Point by point.  (1) Two factors changed at once (aug2 to aug0, and n from 300 to 3000). Accepted. The data-size attribution is no longer a falsifier. A clean learning curve needs harness-level runs at train_fraction 750 and 1500 with this…
  - critic r3 (quality 0.886): P-020 and P-025 both test the same question: is the KRR plateau a tuning artefact? P-020 changes two factors against P-009 at once (asinh features and CV tunin… · The 3x3 grid uses multiples of a gamma heuristic, but the code does not state whether PCA scores are standardised or whitened before the RBF. This matters for … · Treat PCA fitted on all of train before the inner split as benign only if the selected (alpha, gamma) agree with a fold-internal PCA refit on one fold. Run tha… · If outer-sensor readings exceed 3σ for most of the window, asinh and log are near-identical on informative columns. Any gain would then come from tuning alone,… · Are PCA scores scaled before the kernel? If not, gamma is dominated by PC1, which is exactly the P-027 concern.

**P-002 — Conditioning-limited anisotropy: an MLP inverse surrogate with errors ordered by sensitivity (inner k2, k3 < outer k1, …** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* With a centred Gaussian initial pulse and absorbing Dirichlet ends, the Fisher information for the inner conductivities (k2, k3) is much larger than for the outer ones (k1, k4). I predict two things for an MLP trained on log features with log targets. (a) Its…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the ordering is reversed or flat, the per-segment sensors make conditioning nearly uniform, and the bottleneck is the regressor or the feature transform rat…
  - critic r1 (quality 0.895): The ordering prediction (b) is the most informative claim in the portfolio. However, it is tested on the regressor's errors, which mix the estimator's bias wit… · The Fisher argument ignores cross-coupling. Sensor 5's signal depends on k2 (the heat must pass through segment 2) as much as on k1. The marginal posterior var… · The ordering should be predicted for the CRB, sigma^2 diag((J^T J)^{-1}), averaged over the prior. The regressor's ordering is then checked against it. · If the outer sensors receive almost no heat, the network may simply predict the prior mean for k1 and k4. In that case err_k1 is about 0.43. The ordering holds… · Is the pulse centre exactly at an interface (index 19.5)? If so, k2 and k3 are symmetric in role, and the ordering is between pairs, not within them.
- v2 `P-010` r2 [revise, llm] `torch_mlp/log+logy+aug2` status conjectured. *H:* With a pulse centred on the k2/k3 interface and absorbing Dirichlet ends, the prior-averaged local CRB, sigma^2 diag((J^T J)^{-1}), is smaller for k2 and k3 than for k1 and k4, and it is mirror-symmetric. An MLP on log features with log targets, trained with …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the ordering is flat in both the MLP and the CRB, the midpoint sensors equalise conditioning, and the bottleneck is the regressor. If the CRB ordering holds…
  - response to critic: The pilot gave 0.0497 with 300 training samples (60 val samples; within noise of the threshold).  (1) Regressor errors mix estimator bias with information content. Accepted. I add part (c): the ordering must also appear in an estimator-ind…
  - critic r2 (quality 0.88): The mirror-symmetry claim is false on this grid. With 40 points and the pulse at index 19.5, sensors 5, 15, 25 and 35 sit 14.5, 4.5, 5.5 and 15.5 cells from th… · A local CRB from 48-neighbour local-linear Jacobians will be biased. In 4-D, 48/3000 neighbours span about 36% of each axis range, so curvature contaminates J … · Replace exact mirror symmetry with asymmetry predicted from sensor-to-pulse distance (14.5/4.5/5.5/15.5 cells). · Each segment contains a midpoint sensor. If SNR at the outer sensors is still above 100 over part of the window, conditioning can be nearly flat and the orderi… · What is the ratio of MLP per-component RMSE to sqrt(CRB) for each component? Values far above 1 mean the ordering test is uninformative about information conte…
- v3 `P-021` r3 [revise, llm] `torch_mlp/log+logy` status conjectured. *H:* The pulse sits at the k2/k3 interface and both ends are absorbing. The prior-averaged local CRB, sigma^2 diag(H^{-1}), is therefore smaller for k2 and k3 than for k1 and k4. Because the right-hand sensors are one cell farther from the pulse and nearer the bou…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: Interpretation by outcome: - CRB ordering flat and MLP ordering flat: midpoint sensors equalise conditioning, and the regressor is the bottleneck. - CRB orderi…
  - response to critic: Point by point.  (1) The mirror-symmetry claim. Accepted; it was wrong. Taking grid centre 19.5, sensors 5, 15, 25 and 35 lie 14.5, 4.5, 5.5 and 15.5 cells from the pulse. The right-hand sensors are one cell farther from the pulse. If the …
  - critic r3 (quality 0.874): Falsifier (b), the ordering of per-component errors, depends on CRB values from P-011's local Hessians. Those are emulator Jacobians fitted from neighbours, no… · Relative versus absolute error: rel_l2 is not normalised per component, and the CRB ordering is in absolute k. State whether err_kj is absolute RMSE or RMSE//k… · Compute the CRB ordering from the P-029 global emulator Jacobians as a second, independent estimate. Require that both estimates agree before scoring falsifier… · Sensor 5 sits about 5 cells from an absorbing end, and its trace is controlled mostly by flux across the 1/2 interface. If k1 is large, segment 1 drains quickl… · Is the falsifier min(err_k1, err_k4) >= 0.25 in relative units? 0.25 is near the unidentified floor (≈0.33) for a single component, so this criterion is very l…

**P-003 — Noise-augmented MLP on sqrt features with a log-conductivity target** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, llm] `torch_mlp/sqrt+logy+aug2` status conjectured. *H:* The inverse map from sensor time series to theta = log k is smooth, so a moderately sized MLP can learn it. The setup is noise sigma = 0.001, a known initial condition and 4 parameters, with the data lying on a 4-dimensional manifold inside the high-dimension…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val error sits well above 0.05 and is concentrated on k1 and k4, the data carry little information about the outer segments at this noise level, and no lear…
  - critic r1 (quality 0.66): noise_scale semantics must be confirmed. As written, the augmentation level is either negligible here or catastrophic in the sibling runs. · noise_scale=0.001 is inconsistent with every other proposal, which use 1.0. If noise_scale multiplies the stated sigma (as P-005 assumes), this run adds noise … · Clipping before sqrt has the same issue as log at near-zero readings: the derivative of sqrt is unbounded near 0, which amplifies noise at the outer sensors, o… · Resolve the semantics of noise_scale from the harness code before running. · Does noise_scale multiply the context noise std, or is it absolute?
- v2 `P-012` r2 [revise, llm] `torch_mlp/sqrt+logy` status conjectured. *H:* At matched architecture and training (150 epochs, [256,256], log target, no extra noise augmentation), sqrt features give val rel_l2 within ±0.01 of raw features. They beat log features whenever outer-sensor readings are frequently below about 10 sigma, becau…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If sqrt is clearly worse than raw, near-zero amplification and rectification outweigh the range compression. Transforms that act before the noise floor are the…
  - response to critic: Blocking point (noise_scale semantics): I read the harness source, harness/methods.py, the _augment / noise-aug block. The augmentation noise std is noise_scale times the MEDIAN PER-COLUMN STD of the raw training features. It is added in r…
  - critic r2 (quality 0.844): The ablation is unmatched as scheduled. This run uses [256,256], while P-002 and P-005 use [512,256,128] with aug2. Without co-scheduled raw and log arms at id… · The noise in sqrt features is rectified at 0, and the 1/(2 sqrt x) derivative amplifies noise near zero. The hypothesis is two-sided about this, but the 10-sig… · The pilots contradict augmentation noise being 'comparable to the signal spread'. aug2 pilots reached about 0.05, so either the median column std is small (man… · If the median column std is about 1e-3 because most columns are near zero, augmentation at noise_scale = 1.0 is close to physical. P-012's confound argument wo… · Can you quote the exact source line for the augmentation std, and the numerical median column std of raw X compared with sigma = 1e-3?
- v3 `P-023` r3 [revise, llm] `kernel_ridge/sqrt+pca40+logy` status conjectured. *H:* Kernel ridge (RBF, standard scaler, PCA-40, log target, aug0, same alpha and gamma_mult as P-009) on sqrt features reaches val rel_l2 within ±0.01 of the log-feature P-009 arm. If either transform loses by more than 0.01, the loss is concentrated in k1/k4: th…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the three transforms are equivalent but all exceed 0.05, the feature transform is not the bottleneck; the bottleneck is hyperparameters, PCA dimension or in…
  - response to critic: (1) Unmatched ablation: I accept this. Three ~40 s MLP arms are poor value given the critic's priority list, so I am moving the feature-transform ablation onto the model class where it is cheap and already anchored. The revised arm is sqrt…
  - critic r3 (quality 0.844): At fixed alpha and gamma_mult, changing the feature transform also changes the effective bandwidth, because the 'scale' heuristic sees different variance after… · The raw twin is not a listed proposal and P-009's status this round is unclear. The triple only works if both companions run with identical settings. · The harness transforms need verification: sqrt(max(x,0)) and log(max(x,1e-6)). The author has not re-verified these, and the localisation argument depends on t… · Under standard scaling, a column dominated by clipped noise has unit variance whatever the transform. The transform then changes the column's distribution shap… · Will P-009 and the raw twin be run in the same execution slot with identical PCA seeds?

**P-004 — Kernel ridge on a low-dimensional PCA embedding exploits the 4-D data manifold** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `kernel_ridge/log+pca20+logy` status conjectured. *H:* The noise-free observations lie on a smooth 4-D manifold, and i.i.d. noise with sigma = 0.001 spreads mostly into directions of negligible signal variance. Projecting standardised log features onto ~20 principal components therefore keeps almost all of the pa…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If kernel ridge on PCA-20 fails while the MLP succeeds, either the inverse map has sharp features (near non-identifiable regions) that a stationary kernel smoo…
  - critic r1 (quality 0.71): Standardise-then-PCA does not denoise isotropically. Standardisation scales up near-zero, noise-dominated features (outer sensors, early times) to unit varianc… · Its falsification criterion (worse than P-001 by more than 0.01) is confounded, because P-001 differs in pca, gamma_mult and noise_aug. It largely overlaps wit… · The denoising claim holds for PCA on raw-scale traces, not for PCA after per-feature standardisation. · If standardised late-time outer-sensor features are noise-only, each contributes unit variance. With about 400 such features, the noise subspace carries more t… · What fraction of variance do 20 PCs explain, and how does the explained variance of the noise-free part compare with the noise floor of 800*sigma^2?

**P-005 — Noise-matched MLP on raw sensor traces with log-target recovers k to below 5% rel_l2** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from the 4x200 sensor traces to (k1..k4) is smooth and well conditioned. Noise at sigma=0.001 is small next to the O(0.1-1) pulse amplitudes, so the error floor comes from how well the estimator approximates the map, not from noise. An MLP trained wit…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val error stays above 0.05 while train error is low, the cause is variance or overfitting, and we should try more noise augmentation or PCA. If train error …
  - critic r1 (quality 0.845): The claim 'noise is small next to O(0.1-1) amplitudes' holds only at the inner sensors. The outer-sensor SNR is unverified and may be the binding constraint (s… · The hypothesis names linear ridge as a comparator ('ridge stays well above'), but no ridge run is proposed. That clause is currently unfalsifiable. · How many time samples are in the features: 200 per sensor, or subsampled?
- v2 `P-015` r2 [revise, llm] `torch_mlp/raw` status conjectured. *H:* The map from the noisy sensor traces to (k1..k4) is smooth enough that a 3-layer MLP trained on raw standardized traces with an MSE loss on linear k reaches val rel_l2 < 0.05. This needs no noise augmentation, because train features already contain the true n…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If train error is low but val is high, the failure is variance; augmentation or PCA (P-008) is then justified. If train error is also high, mainly on k1/k4, th…
  - response to critic: (1) Outer-sensor SNR: agreed, it was unverified and I have weakened the claim. A rough CFL estimate (dx=1/39, dt<=dx^2/(2*k_max)≈1.6e-4, 200 steps, so T≈0.033) gives a diffusion length sqrt(2kT)≈0.18-0.36 over the prior range. Sensors 5 an…
  - critic r2 (quality 0.821): The log-target bias argument is quantitatively negligible. At about 5% relative posterior std, the gap between the geometric and arithmetic mean is about s^2/2… · It is not a single-factor control. Compared with P-002 it changes features (raw vs log), target, and augmentation together, so 'within 10%' cannot attribute an… · Posterior widths are small relative to k, so the choice of target transform matters mainly through conditioning of the loss, not estimator bias. · With a relative posterior std of 0.05, E[k] / exp(E[log k]) is about 1.00125. The resulting rel_l2 difference is below 0.002.
- v3 `P-026` r3 [revise, llm] `torch_mlp/raw` status conjectured. *H:* A [256,256] MLP on raw standardized traces with MSE on standardized linear k reaches val rel_l2 < 0.05 with no augmentation. Because posterior widths are small relative to k, its rel_l2 will be within 10% (relative) of the otherwise identical log-target twin.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If train error is low and val is high, the failure is variance, which justifies augmentation or PCA (P-019, P-009). If train error is also high and concentrate…
  - response to critic: (1) Log-target bias. Accepted. With about 5% relative posterior std, E[k]/exp(E[log k]) - 1 ≈ s^2/2 ≈ 0.1%. That is far too small for this arm to detect estimator bias. I have dropped the bias rationale. What remains is the conditioning ar…
  - critic r3 (quality 0.826): The paired bootstrap over val rows captures only val-sampling variance, not training-seed variance. The pilots show a seed spread of ≈0.001–0.002, which is com… · The prior expected effect is small, as the author accepts (bias ≈ 0.1%). The pair costs about 80 s, roughly 20% of the total budget, to fix a factor that is pr… · Use identical torch seeds and data order in both arms, so that the paired comparison is a common-random-numbers comparison. · In linear k, MSE weights absolute error. In the k ≈ 2 corner, where outer-sensor information is weakest, absolute errors are largest. A linear-target MLP may t… · Is raw standardised the intended feature set, given that the other MLP arms use log features? Results will not transfer to the main MLP configuration.

**P-006 — Competing explanation: the inverse map lives on a low-dimensional signal subspace, so PCA-compressed kernel ridge match…** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `kernel_ridge/raw+pca30+logy` status conjectured. *H:* The noise-free traces span a manifold of intrinsic dimension 4 inside a linear subspace of roughly 20-30 PCA components. The remaining components are mostly noise at sigma=0.001. RBF kernel ridge on 30 standardized PCA components with a log-target will reach …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: Failure means PCA discards informative low-variance late-time content. The next step is to drop PCA or use scaler 'none' so absolute amplitudes are kept, and f…
  - critic r1 (quality 0.795): This is a good control in intent: it asks whether the information sits in high-variance shape or in low-variance tails. But the scaler='standard' choice underm… · The comparison references 'the MLP from proposal 1', but proposal 1 is KRR. It should reference P-005, which uses the same raw features. · To test the tails hypothesis, use scaler='none' before PCA.
- v2 `P-016` r2 [revise, llm] `kernel_ridge/raw+pca30` status conjectured. *H:* Without standardization, the top 30 absolute-variance PCs are dominated by the inner-sensor pulse shape. If information on k1 and k4 sits in low-amplitude outer-sensor and late-time signal, unscaled PCA-30 KRR will reach rel_l2 > 1.5x that of P-005, with the …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If rel_l2 is above 0.06 with k1/k4-dominated error, informative content is low-amplitude. Methods should then keep full traces, use log or standardized feature…
  - response to critic: (1) Scaler choice: accepted. With scaler='standard', low-amplitude tails are inflated to unit variance, so 'PCA discards tails' is not tested. The revision uses scaler='none'. PCA then ranks directions by absolute variance, which is the on…
  - critic r2 (quality 0.81): This overlaps P-013: unscaled raw PCA KRR at 30 vs 20 components, with linear vs log target. Run a single unscaled sweep instead. · The kernel 'scale' bandwidth on unscaled PCA scores is dominated by the first PC. Anisotropic variance can make KRR fail uniformly, and that failure can look l… · Whiten PCA scores (divide by sqrt(eigenvalue - sigma^2)) or pass them through a standard scaler before the RBF kernel, so information content is separated from… · At which PC index does the eigenvalue fall to about 1e-6 (sigma^2)? That bounds the number of informative components in unscaled space.
- v3 `P-027` r3 [revise, llm] `custom-ml` status conjectured. *H:* The PCA-20 subspace of unscaled raw traces contains enough information on k, but the isotropic 'scale' RBF bandwidth mismatches the anisotropic PC variances. Whitening the scores and tuning (alpha, gamma) by train-only CV will cut val rel_l2 by at least 25% r…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If whitening does not help and the error is k1/k4-concentrated, the informative content lies beyond the top-20 variance directions. Use m>=40 (P-009) or full t…
  - response to critic: (1) Overlap with P-013. Accepted. I no longer run a separate unscaled PCA-30 KRR. The revision is built as P-013's partner: unscaled raw PCA-20 with a log target, as in P-013. (2) The bandwidth confound. Accepted, and this is now the facto…
  - critic r3 (quality 0.824): Whitening and CV tuning are changed jointly against P-013, which is untuned and unwhitened. A ≥25% gain could come from tuning alone. Include whiten in {True, … · In unscaled raw space, PCA-20 is variance-ordered toward the large-amplitude inner-sensor traces. Information about k1 and k4 may lie in low-variance direction… · P-013 must be run at full fidelity on the same val set. Otherwise the comparison has no anchor. · If PCs 1–20 explain more than 99.9% of variance but the k1/k4 signal is about 1e-3 of total variance, both explanations hold at once. Whitening helps the retai… · What is the ratio of the 20th PC variance to σ²?

**P-007 — Learned-surrogate forward model + batched Gauss-Newton inversion: treat 'no forward model' as a learnable 4-D to observ…** (researcher-exploratory, family custom)
- v1 `P-007` r1 [propose, llm] `custom-solver` status conjectured. *H:* The parameter-to-observation map theta (4-D) -> sensor traces is smooth on the box prior. With 3000 samples it can be emulated to near noise level by kernel ridge regression on PCA coefficients of the traces. Minimising the whitened misfit to each val sample …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the emulator error is the limit, the forward map is not smooth enough for an RBF at this density, e.g. because of steep sensitivity of early-time traces. A …
  - critic r1 (quality 0.895): This is the most structurally distinct arm and the right test of 'amortised vs. emulate-and-invert'. The emulator must be accurate to about sigma/signal (about… · The whitened misfit in PCA space equals the true likelihood only if PCA is orthonormal on unscaled traces. Emulator error must also enter the covariance (sigma… · Include emulator residual variance per PCA component in the misfit weights. · Suppose the k1 direction is nearly flat in the misfit. LM then wanders to the bound, giving per-component errors of up to 1.5, while an MLP predicting about 1.… · How are PCs computed (scaled or unscaled)?
- v2 `P-018` r2 [revise, llm] `custom-solver` status conjectured. *H:* The map theta (4-D) -> PCA coefficients of the sensor traces is smooth on the box prior and can be emulated by RBF kernel ridge from 3000 samples. The leading components should be emulated to within a small multiple of sigma. The steps are: whiten the misfit …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If emu_resid >> sigma^2 on the leading components, RBF-KRR is not accurate enough at this density. The next step is a residual emulator (linear-in-z trend plus…
  - response to critic: Point-by-point. (1) Small-amplitude PCs that carry k1/k4 may be poorly emulated: accepted. I raised m from 24 to 40. Unscaled PCA ranks components by variance, so the low-variance outer-sensor directions sit late in the spectrum, and m=24 …
  - critic r2 (quality 0.898): Prior measure: a uniform box in k is a non-uniform density in t = log k (proportional to exp(sum t)). Importance weights must include this Jacobian, or the 'po… · Emulator error is likely much larger than sigma. Signal is O(1e-1) and sigma is 1e-3, so the KRR emulator needs relative accuracy of about 1e-2 to 1e-3 from 30… · Prior density in log-k coordinates is proportional to prod_i k_i. · If the emulator relative error is 1e-2, the leading-PC residual is about 1e-3 x amplitude/sigma, i.e. tens of sigma. Weights then come from emulator error, and… · What fraction of val samples fall back to the mode (ESS < 3)?
- v3 `P-029` r3 [revise, llm] `custom-solver` status conjectured. *H:* The map z -> PCA-40 coefficients of the raw sensor traces, with z = normalised log k in [0,1]^4, can be emulated at n = 3000 by a linear trend plus RBF-KRR. Held-out residual variance on PCs 1-10 should then be within 10x sigma^2. If so, the posterior mean of…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: Case 1: emu_resid/sigma^2 >> 10 and rel_l2 > best amortised. Then on this problem, at this data density, the inverse map is easier to learn than the forward ma…
  - response to critic: First, the evidence. The pilot gave rel_l2 = 0.0836 with 300 training samples and n_val = 60. The amortised KRR-log-PCA40 and MLP pilots reached about 0.05 at the same training size. At low data, then, the emulate-and-invert route is worse…
  - critic r3 (quality 0.886): The pilot (0.084 at n=300) is the worst arm. The proposed fixes (prior Jacobian, full-covariance whitening) are sound, but the key unknown is emulator accuracy… · Estimating a 40x40 residual covariance from a ~500-point hold-out is reasonable, but the homogeneity assumption is likely false. Emulator error should be large… · Treat the posterior mean in k, the Bayes rule for squared loss on k, as only approximately optimal for rel_l2. This is acknowledged in the proposal. · KRR emulation of a 4-D smooth map at n=3000 can reach 1e-3 relative error in the interior but degrades at the boundary. If val error concentrates at the box ed… · What is the emulator residual variance per PC index, across all 40 PCs?

**P-008 — Amortised inverse in log-conductivity space with PCA denoising and noise augmentation (multiplicative-structure prior)** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/raw+pca32+logy+aug2` status conjectured. *H:* k enters the heat equation as a diffusion time-scale, so sensor responses depend approximately on log k. A neural regressor trained on log-targets will therefore have more uniform errors across the [0.5, 2] range. Adding PCA truncation of the traces to remove…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If it fails while proposal 1 succeeds, amortised inversion is limited by the ill-conditioned inverse map rather than by information content. That would argue f…
  - critic r1 (quality 0.765): The falsification ablation changes two factors at once (log target and PCA), so a 20% gain cannot be attributed to the multiplicative-structure prior. A 2x2 de… · The statement 'uniform prior puts high leverage on large k where the response saturates' is a real, testable claim. It needs error-by-k-bin as a primary output… · Treat the log-target effect and the PCA effect as separate hypotheses. · With homogeneous k, u depends on kt only. But the finite window [0,T] truncates the log-time shift, so for small k the response is cut off rather than shifted.…
- v2 `P-019` r2 [revise, llm] `torch_mlp/raw+pca32+logy+aug2` status conjectured. *H:* The informative content of the 4-parameter sensor traces lies on a low-dimensional subspace. Truncating to 32 principal components removes directions that the MLP otherwise fits to noise, so with all else equal to P-005, val rel_l2 should be at least 10% lowe…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If PCA does not help, either the MLP already ignores noise directions (implicit regularisation is sufficient) or 32 standardised components discard outer-senso…
  - response to critic: (1) Two factors changed at once: accepted. The separate ablation is dropped, as the critic recommended. P-008 becomes a single run that differs from P-005 (raw features, standard scaler, log target, aug2) only in pca=32 and in the epochs h…
  - critic r2 (quality 0.634): The control arm (P-005 at matched settings) must be scheduled in the same round, or the single-factor contrast is undefined. · The control P-005 (raw, standard, log target, aug2, [512,256,128], 150 epochs) is not in this round's list. The single-factor claim holds only if P-005 is co-s… · The ID references are inconsistent: the falsifiable test refers to P-008 vs P-005, but this proposal is P-019. Fix them so the result is attributable. · Is P-005 at matched settings scheduled? If not, what is the control?

**P-011 — Non-amortised local-linear Gauss–Newton inversion: empirical Jacobians from training neighbours replace the missing for…** (researcher-mathematical, family custom)
- v1 `P-011` r2 [propose, llm] `custom-solver` status conjectured. *H:* Near any k, the forward map is well approximated by an affine map in log k over its K ≈ 48 nearest training samples. Fitting u ≈ a + B^T(t - t0), where t = log k, by least squares on those neighbours, and then solving the residual-variance-weighted Gauss–Newt…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If it is much worse than the MLP with errors concentrated at small k or near the bounds, a local-linear emulator is too crude at n = 3000. This argues for glob…
  - critic r2 (quality 0.88): Neighbourhood size: K = 48 of 3000 in 4-D spans about 0.36 of each normalised axis, so local-linear curvature bias is substantial. Re-centring 2 times does not… · The affine fit has 5 parameters per output from 48 rows, which is well determined, but per-feature residual variance from 43 dof is noisy. Weights 1/max(res, s… · Use a local-quadratic fit or K sweep {24, 48, 96} to expose the bias-variance trade-off. · At the low-k corner the response is window-truncated and strongly curved. Local-linear Jacobians there will misstate sensitivity, so errors should concentrate … · How does val error vary with K? A U-shape indicates bias at large K and variance at small K.
- v2 `P-022` r3 [revise, llm] `custom-solver` status conjectured. *H:* Near any k, the forward map is well approximated by an affine map in log k over its K nearest training samples. A per-sample Gauss-Newton estimate using the correct Gaussian likelihood on raw readings, with linearisation error absorbed into per-feature varian…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.045}`; if it fails: Interpretation by outcome: - Worse than the n=300 pilot, with the selected K at the grid edge: the bias-variance optimum lies outside the grid, or the residual…
  - response to critic: The pilot is the strongest so far: 0.0449 at n_train=300 (resu-00088), with no training and in 2 s. This beats both amortised pilots at the same size.  (1) Curvature bias at K=48. Partly accepted. Re-centring does reduce bias in the evalua…
  - critic r3 (quality 0.94): The prediction (< 0.045) is weak. The pilot already reached 0.0449 at n=300, and local-linear GN error should fall with neighbour density (a radius about 10^(1… · Selecting K on a 250-sample hold-out gives a standard error of rel_l2 of about 0.002–0.003. Differences between K values may be smaller than that. Report hold-… · Neighbourhoods should be defined in k-space around the current estimate after the first iteration, not in feature space. The proposal does this via re-centring… · Near the box faces, K-nearest neighbourhoods are one-sided. Affine fits there extrapolate, and GN steps can leave the box. Clipping then produces a systematic … · Is the selected K at the edge of {24, 48, 96}? If so, extend the grid.

**P-013 — KRR on unscaled raw-trace PCA-20: the configuration where 'PCA discards noise' actually holds** (researcher-ml, family data-driven, branched from P-004)
- v2 `P-013` r2 [branch, llm] `kernel_ridge/raw+pca20+logy` status conjectured. *H:* With scaler='none', the leading 20 PCs of raw traces capture the signal manifold, and the i.i.d. sigma=0.001 noise contributes only sigma^2 per direction. KRR (RBF, alpha=1e-3, gamma_mult=1, log target) on this embedding should beat the standardised-log PCA-2…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If the unscaled cell loses mainly on k1 and k4, the informative directions are low-amplitude. The right representation is then a noise-whitened PCA (scale each…
  - response to critic: Experimental evidence first: the original arm (log features, standard scaler, PCA-20, alpha=1e-3, log target) piloted at val rel_l2 = 0.075 with only 300 training samples. I keep it as the 'log+standard' cell, because it is cheap and alrea…
  - critic r2 (quality 0.745): This overlaps P-016 (unscaled raw PCA KRR). Merge the two into one block. · Unscaled PCA-20 combined with a raw-amplitude-dominated RBF bandwidth will likely underweight k1/k4 directions. The prediction to beat log+standard by >= 0.01 … · Use noise-whitened PCA (columns divided by sigma, then PCA). With i.i.d. noise this equals unscaled PCA up to a scale factor, so the only difference from 'stan…

**P-014 — Linear ridge control: how nonlinear is the inverse map in log-k?** (researcher-ml, family data-driven)
- v1 `P-014` r2 [propose, llm] `ridge/raw+logy` status conjectured. *H:* Over the box [0.5, 2]^4 the inverse map from raw traces to log k is substantially nonlinear, because diffusion responses depend on k through exp(-lambda(k) t). Linear ridge (standardised raw features, log target, alpha=0.1) will therefore reach a val rel_l2 b…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.33, "expect": "worse"}`; if it fails: If ridge is close to the nonlinear models, the remaining error comes from noise or information limits rather than nonlinearity. Effort should then move to deno…
  - critic r2 (quality 0.781): Alpha = 0.1 is untuned. A falsifier at <= 0.07 could pass or fail depending on alpha. Use RidgeCV over a log grid on an internal train split. · Run this together with P-017. The two differ only in target transform and alpha.
- v2 `P-024` r3 [revise, llm] `custom-ml` status conjectured. *H:* Over [0.5,2]^4 the inverse map from raw traces to log k is substantially nonlinear, because the responses depend on k through decay rates exp(-lambda(k) t) and through interface-weighted mixing. Even with alpha tuned by GCV, linear ridge (standardised raw fea…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.33, "expect": "worse"}`; if it fails: If tuned ridge is close to the nonlinear models, the residual error comes from noise and information limits rather than nonlinearity. Effort should then move t…
  - response to critic: (1) Untuned alpha: I accept this. Fixed alpha=0.1 makes the 0.07 falsifier depend on alpha rather than on linearity. The revised arm uses RidgeCV (efficient leave-one-out GCV) over alphas in logspace(-6, 3, 19), wrapped in a log-target Tra…
  - critic r3 (quality 0.805): The prediction encoding ('relative_to best_data_driven, improvement 0.33, expect worse') conflicts with the falsifier (≤ 0.07, or within 1.5x). State one rule. · Under standard scaling, near-zero noisy columns get unit variance. Ridge with a single shared alpha may underweight informative columns, so 'linear' here also … · Linear regression of log k on raw traces could do well if k mostly controls decay rates over a short window, where u is approximately linear in log k locally. … · Will predictions be clipped to [0.5, 2]? Report both clipped and unclipped values for fairness.

**P-017 — Linear ridge baseline on raw standardized traces: quantifies the nonlinearity of the inverse map** (researcher-experimental, family data-driven)
- v1 `P-017` r2 [propose, llm] `ridge/raw` status conjectured. *H:* The dependence of the traces on k is nonlinear (through the time rescaling t*k and the exp(-lambda(k) t) decay), so the best linear estimator from raw standardized traces to linear k will have val rel_l2 at least 50% worse than the best data-driven model, and…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.5, "expect": "worse"}`; if it fails: If ridge is competitive, the inverse map is nearly linear over the box. Effort should then go to noise-optimal linear estimators (e.g. GLS/BLUE-style weighting…
  - critic r2 (quality 0.459): This is a near-duplicate of P-014, differing only in target and alpha. Merge. · 'Tuning on val only' risks overfitting selection to val. Tune alpha by CV on train instead.

**P-025 — Inner-CV-selected KRR over PCA dimension x score whitening x (alpha, gamma): is the KRR plateau a hyperparameter artefa…** (researcher-ml, family data-driven)
- v1 `P-025` r3 [propose, llm] `custom-ml` status conjectured. *H:* Every KRR arm in the portfolio uses fixed alpha=1e-3 and the gamma 'scale' heuristic, which ties bandwidth to total retained variance. Two choices are left untested: whether the informative PCA scores should be whitened (equal weight per component, which lift…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.045}`; if it fails: If the tuned KRR is no better than P-009, KRR is not hyperparameter-limited. Its plateau then reflects either data size (to be checked with the critic's learni…
  - critic r3 (quality 0.805): 90 fits is the costliest KRR arm. The rule estimate is 30 s; the author's is 45–60 s. Budget accordingly. · The falsifier requires P-009 at full fidelity this round. · If CV selects n=80 with whitening, val gains may be small because whitened noise-PCs hurt at n=3000. The mechanistic readout should then be cross-checked with … · Is the feature count ≥ 80? It is reported as ≈800 elsewhere, so this is probably fine.

**P-028 — Target-transform single-factor pair, log arm: raw standardized MLP, aug0, log k (twin of P-015)** (researcher-experimental, family data-driven)
- v1 `P-028` r3 [propose, llm] `torch_mlp/raw+logy` status conjectured. *H:* Identical to P-015 except target_transform='log'. Its val rel_l2 will be within 10% (relative) of P-015, and the paired bootstrap CI of the difference will include 0, because posterior widths are small (about 5%) and the transform changes only the loss weight…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A significant difference means the target transform acts through loss conditioning. All later arms must then fix it, and the better transform becomes the defau…
  - critic r3 (quality 0.756): This is the same as P-026's challenge: single seed, a val-only paired bootstrap, and a low prior effect size. · It references P-015 instead of P-026. Fix the linkage. · Use common random numbers with P-026.

- Critic round 1 portfolio view: The estimated cost is about 585 s, which exceeds the 400 worker-second budget. The four 120 s MLPs alone use 480 s. Recommended plan: (1) resolve the noise_scale semantics (blocking for P-003, and it affects P-002, P-005 and P-008); (2) run the cheap SNR, ridge/k-NN and CRB diagnostics first (under 20 s); (3) run P-005 (raw MLP, 150 epochs) and P-002 (log MLP, 150 epochs) as the feature ablation,… 7/8 active proposals are 'data-driven' families={'data-driven': 7, 'custom': 1}
- Critic round 2 portfolio view: The pilots place KRR-log-PCA40 and the MLP at about 0.05 with only 300 training samples, so the target is plausibly reachable. The open questions are augmentation semantics, PCA dimension, and outer-sensor information. Priority order: (1) the cheap SNR and eigenvalue diagnostics plus a source check of noise_aug; (2) P-009 with a learning curve and aug0/aug2 contrast; (3) merged ridge and merged u… 10/12 active proposals are 'data-driven' families={'data-driven': 10, 'custom': 2}
- Critic round 3 portfolio view: The proposals are specific, falsifiable and within budget. The rule estimates total ≈ 315 of 400 s. Recommended priority: P-022 with a tightened prediction; P-025 merged with P-020's asinh factor; P-024 (cheap); P-029 with emulator diagnostics first; P-021. Lower priority: P-027 (add a whiten factor), P-023 (needs a bandwidth sweep), and the merged P-026/P-028 pair under common random numbers. No… 8/10 active proposals are 'data-driven' families={'data-driven': 8, 'custom': 2}

**Pilot experiments run during research** (small validation subsets):

- P-001 `kernel_ridge/log+pca40+logy+aug2`: ok rel_l2=0.052037 on 60 val samples in 2.12 s
- P-002 `torch_mlp/log+logy+aug2`: ok rel_l2=0.049671 on 60 val samples in 14 s
- P-004 `kernel_ridge/log+pca20+logy`: ok rel_l2=0.075127 on 60 val samples in 2.09 s
- P-010 `torch_mlp/log+logy+aug2`: ok rel_l2=0.051022 on 60 val samples in 13.3 s
- P-018 `custom-solver`: ok rel_l2=0.083625 on 60 val samples in 2.05 s
- P-011 `custom-solver`: ok rel_l2=0.044902 on 60 val samples in 1.98 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 73 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-ml | 1 | uniqueness of piecewise constant diffusion coefficient one-dimensional heat equation from… | 4 refs via /search |
| researcher-mathematical | 1 | Uniqueness and Lipschitz stability for recovering a piecewise constant diffusion coeffici… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for recovering piecewise constant diffusion coefficient in one-dimens… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-experimental | 1 | uniqueness of conductivity coefficient in heat equation from boundary or interior observa… | 4 refs via /search |
| researcher-mathematical | 1 | Lipschitz stability of inverse problems with finitely many unknown parameters in piecewis… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient recovery from pointwise temperature observations over… | 4 refs via /search |
| researcher-experimental | 2 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search |
| researcher-mathematical | 2 | Per-sensor time-series amplitude statistics (max, median over samples) relative to sigma=… | 4 refs via /search |
| researcher-mathematical | 2 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Emulator holdout residual per PC vs sigma^2, and LM convergence rate | 4 refs via /search |
| researcher-experimental | 2 | stability estimate parabolic inverse coefficient problem point observations time | 4 refs via /search |
| researcher-ml | 2 | Harness source for noise_aug/noise_scale | 4 refs via /search |
| researcher-ml | 2 | identifiability piecewise constant conductivity heat equation interior point measurements | 4 refs via /search (cache) |
| researcher-exploratory | 2 | Gaussian process emulator calibration inverse problem parameter estimation heat equation | 4 refs via /search |
| researcher-ml | 3 | median(std(X_train, axis=0)) / 1e-3 | 4 refs via /search |
| researcher-mathematical | 3 | Per-column fraction of training readings <= 0 and mean/sigma SNR per sensor vs time | 4 refs via /search |
| researcher-experimental | 3 | Cramer-Rao bound for piecewise-constant diffusivity from point sensors in a 1-D heat equa… | 4 refs via /search |
| researcher-exploratory | 3 | Held-out emulator residual variance per PC vs sigma^2 | 4 refs via /search |
| researcher-ml | 3 | Per-sensor SNR: fraction of readings with x < 10*sigma at sensors 5 and 35 | 4 refs via /search |
| researcher-mathematical | 3 | Per-sensor max/u//sigma statistics over the training set | 4 refs via /search |
| researcher-experimental | 3 | Identifiability of layered conductivity from interior temperature time series | 4 refs via /search |
| researcher-exploratory | 3 | Gaussian process emulator error propagation into Bayesian calibration posterior (inflated… | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s0.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 197 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-022) | custom | `custom-solver` | researcher-mathematical | 0.94 | best-rated proposal in family 'custom' (quality 0.94) |
| E-02 | proposal (P-020) | data-driven | `custom-ml` | researcher-mathematical | 0.886 | best-rated proposal in family 'data-driven' (quality 0.89) |
| E-03 | proposal (P-029) | custom | `custom-solver` | researcher-exploratory | 0.886 | quality 0.89 with diversity penalty |
| E-04 | proposal (P-021) | data-driven | `torch_mlp/log+logy` | researcher-mathematical | 0.874 | quality 0.87 with diversity penalty |
| E-05 | proposal (P-023) | data-driven | `kernel_ridge/sqrt+pca40+logy` | researcher-ml | 0.844 | quality 0.84 with diversity penalty |
| E-06 | proposal (P-026) | data-driven | `torch_mlp/raw` | researcher-experimental | 0.826 | quality 0.83 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-027 *Bandwidth-vs-information test: whitened, train-CV-tuned KRR on unscaled raw PCA…* — portfolio size limit (6); quality 0.82
- not advanced: P-024 *Tuned linear control: RidgeCV on raw standardised traces with a log-k target (h…* — portfolio size limit (6); quality 0.81
- not advanced: P-025 *Inner-CV-selected KRR over PCA dimension x score whitening x (alpha, gamma): is…* — portfolio size limit (6); quality 0.81
- not advanced: P-028 *Target-transform single-factor pair, log arm: raw standardized MLP, aug0, log k…* — portfolio size limit (6); quality 0.76

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-004 | E-03 | research | custom | `custom-solver` | 0.09035 | **0.023148** | 0.9951 | 0.03053 | 8.28 | 20.8 | 0 | - | FINAL |
| idea-002 | E-01 | research | custom | `custom-solver` | 0.04336 | **0.027254** | 0.9932 | 0.03594 | 1.68 | 2.25 | 0 | - | FINAL |
| idea-012 | E-06 | refine | data-driven | `torch_mlp/raw` | - | **0.028221** | 0.9927 | 0.03722 | 11.1 | 0.0047 | 0 | - | FINAL |
| idea-007 | E-06 | research | data-driven | `torch_mlp/raw` | 0.06155 | **0.028326** | 0.9926 | 0.03736 | 10.8 | 0.00393 | 0 | - | evaluated |
| idea-013 | E-07 | refine | data-driven | `torch_mlp/raw` | - | **0.029471** | 0.992 | 0.03886 | 9.98 | 0.00801 | 0 | - | FINAL |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06289 | **0.029522** | 0.992 | 0.03893 | 8.93 | 0.00725 | 0 | - | evaluated |
| idea-005 | E-04 | research | data-driven | `torch_mlp/log+logy` | 0.05363 | **0.02964** | 0.9919 | 0.03909 | 11.8 | 0.00808 | 0 | - | evaluated |
| idea-011 | E-06 | refine | data-driven | `torch_mlp/raw` | - | **0.029708** | 0.9919 | 0.03918 | 10.1 | 0.00469 | 0 | - | evaluated |
| idea-014 | E-07 | refine | data-driven | `torch_mlp/log` | - | **0.029759** | 0.9918 | 0.03924 | 8.43 | 0 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `kernel_ridge/raw` | 0.05292 | **0.032419** | 0.9903 | 0.04275 | 0.521 | 0.0574 | 0 | - | evaluated |
| idea-006 | E-05 | research | data-driven | `kernel_ridge/sqrt+pca40+logy` | 0.05973 | **0.033241** | 0.9898 | 0.04384 | 0.517 | 0.0493 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `knn/sqrt` | 0.2036 | **0.14038** | 0.8184 | 0.1851 | 0.00939 | 0.0806 | 0 | - | evaluated |
| idea-003 | E-02 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: RuntimeError: Cannot clone objec… |

## 8. Lineage of the final method

- `idea-004` `custom-solver` ← P-029 v3 (researcher-exploratory, r3, revise) ← P-018 v2 (researcher-exploratory, r2, revise) ← P-007 v1 (researcher-exploratory, r1, propose)
- `idea-012` `torch_mlp/raw` ← refined from idea-007 ← P-026 v3 (researcher-experimental, r3, revise) ← P-015 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)
- `idea-002` `custom-solver` ← P-022 v2 (researcher-mathematical, r3, revise) ← P-011 v1 (researcher-mathematical, r2, propose)
- `idea-013` `torch_mlp/raw` ← refined from idea-008 ← baseline/control (refinement of idea-008 (torch_mlp/raw))

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00248` [checkpoint] by scheduler: final method (ensemble)
- `resu-00245` [result] by executor: exp-0030 rel_l2=0.02104 r2=0.996 rmse=0.02768
- `hypo-00133` [hypothesis] by scheduler: idea-004 custom-solver (research)
- `hypo-00216` [hypothesis] by scheduler: idea-012 torch_mlp/raw (refine)
- `hypo-00129` [hypothesis] by scheduler: idea-002 custom-solver (research)
- `hypo-00218` [hypothesis] by scheduler: idea-013 torch_mlp/raw (refine)
- `plan-00244` [plan] by engineer-v2: exp-0030: final fit of 4 member(s) on train+val; test once
- `gate-00132` [gate] by gate: E-03: approved (custom-solver)
- `prop-00118` [proposal] by researcher-exploratory: P-029 v3 [revise] exploratory: Learned-surrogate forward model with full-covariance emulator-error whitening, box-prior M
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s0 (a7592704f6)
- `gate-00215` [gate] by gate: refine idea-007: approved (torch_mlp/raw)
- `prop-00114` [proposal] by researcher-experimental: P-026 v3 [revise] experimental: Target-transform single-factor pair, linear arm: raw standardized MLP, aug0, linear k
- `hypo-00139` [hypothesis] by scheduler: idea-007 torch_mlp/raw (research)
- `gate-00128` [gate] by gate: E-01: approved (custom-solver)
- `prop-00108` [proposal] by researcher-mathematical: P-022 v2 [revise] mathematical: Non-amortised local-linear Gauss-Newton inversion with train-only neighbourhood-size selec
- `gate-00217` [gate] by gate: refine idea-008: approved (torch_mlp/raw)
- `hypo-00141` [hypothesis] by scheduler: idea-008 torch_mlp/raw (baseline)
- `prop-00078` [proposal] by researcher-exploratory: P-018 v2 [revise] exploratory: Learned-surrogate forward model + emulator-error-weighted LM + importance-sampled posterio
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `prop-00075` [proposal] by researcher-experimental: P-015 v2 [revise] experimental: Linear-target, no-augmentation MLP on raw standardized sensor traces (control for log-targ
- `gate-00138` [gate] by gate: E-06: approved (torch_mlp/raw)
- `prop-00071` [proposal] by researcher-mathematical: P-011 v1 [propose] mathematical: Non-amortised local-linear Gauss–Newton inversion: empirical Jacobians from training neigh
- `gate-00140` [gate] by gate: E-07: approved (torch_mlp/raw)
- `prop-00041` [proposal] by researcher-exploratory: P-007 v1 [propose] exploratory: Learned-surrogate forward model + batched Gauss-Newton inversion: treat 'no forward model'
- `prop-00039` [proposal] by researcher-experimental: P-005 v1 [propose] experimental: Noise-matched MLP on raw sensor traces with log-target recovers k to below 5% rel_l2

## 9. Compute and cost

- Execution compute: 326.6 / 400 worker-seconds over 30 jobs; wall 1055 s
- By stage: baseline 2.2s, pilot 35.5s, screen 40.0s, evaluate 78.6s, refine 50.0s, final 120.3s
- LLM: 20 calls (by phase {'research': 19, 'execution': 1}), 1637 s, $5.744 reported · TheoremSearch: 24 queries (0 failed)

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

The final ensemble of two custom-solver members and two `torch_mlp/raw` members reached held-out test rel_l2 = 0.0210 (R² = 0.996, RMSE = 0.0277). That is well under the 0.05 success threshold, and the val score (0.0221) closely matches test, so there is no sign of selection overfitting to val. The best single models were the custom solver (val 0.0231) and a raw-trace MLP (val 0.0282). Both families clear the target on their own. Ensembling gives a modest gain of about 4–5% over the best single model, and whether that gain exceeds seed noise is not established. All five pre-registered predictions were supported and none were contradicted. However, they were one-sided bounds (val < 0.045 or < 0.05) that the models beat by a wide margin. They confirm feasibility but do little to discriminate between mechanisms. The custom solver's internals are not described in this summary, so the reasons for its advantage over the MLP remain undiagnosed.

No hypothesis was refuted by a full-scale run. The only empirical negative result is the low-dimensional-manifold KRR idea. On standardised log traces, PCA-20 KRR scored 0.075 against 0.052 for PCA-40. This suggests the informative dimension in standardised space exceeds 20, which argues against "truncation denoises without losing signal." That evidence is weak: it is a single pilot with n_train = 300 and n_val = 60, so the gap could shrink at full data. The other three abandoned proposals were withdrawn rather than falsified:
- Unscaled PCA-20 KRR duplicated an existing cell.
- Linear ridge was merged into P-014. Its prediction (rel_l2 ≥ 1.5× the best data-driven model, and > 0.05) should still be read off P-014's outcome, which this summary does not report.
- The MLP+PCA single-factor test was dropped because it was unpowered and lacked a matched control.

Several questions remain open: the size of the seed variance, how error is distributed across the four conductivity segments (an aggregate rel_l2 can hide one poorly identified segment, for example one far from the sensors), and robustness to noise levels or conductivity ranges outside the training distribution.

- **Seed replication:** run 3+ seeds of the best custom solver, best MLP, and the ensemble. This would show whether the 0.0231 → 0.0221 ensemble gain and the solver-vs-MLP gap exceed run-to-run variance.
- **Per-segment error breakdown:** report rel_l2 for each k_i against distance to the nearest sensor. This tests whether identifiability, not model capacity, limits the remaining error.
- **Report the P-014 ridge result:** use it to quantify how nonlinear the inverse map is.
- **Cheap paired PCA test:** run the 2-seed matched MLP raw vs. PCA-32/40 comparison the critic proposed. Also rerun PCA-20 vs. PCA-40 KRR at full n_train to confirm or overturn the pilot's negative result.
- **Out-of-distribution stress test:** evaluate on simulated sets with higher noise and conductivity contrasts outside the training range. This would check whether the solver's lead over the MLP widens under distribution shift, as one would expect if it encodes the physics.

_Automatic summary:_ 29 proposal versions over 3 research rounds; 8 methods advanced; 5 predictions supported, 0 contradicted. Final: ['custom-solver', 'torch_mlp/raw', 'custom-solver', 'torch_mlp/raw'] with test rel_l2 0.021039.
