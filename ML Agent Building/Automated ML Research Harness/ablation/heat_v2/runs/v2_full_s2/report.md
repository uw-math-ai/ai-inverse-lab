# Research report (V2): heat_v2_full_s2

Run `heat_v2_full_s2_20261109_003735` · 2026-11-09 00:51:24 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `hybrid[extra_trees/raw+logy+aug2->LM15]`
- Validation rel_l2 (selection estimate): **0.019329** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018418**, r2 = **0.99696**, rmse = **0.024235** · bootstrap 95% CI for rel_l2: [0.01725, 0.01962]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.075569 (vs 0.018418 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.052052 (vs 0.018418 in distribution)
- Uncertainty calibration on test: 70.1% within 1σ, 95.6% within 2σ (ideal 68% / 95%)
- Inference latency: 5.6 ms/sample; fit 10.1 s; exported model reload check: max |Δ| = 2.5e-08
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[extra_trees/raw+logy+aug2->LM15]` rel_l2=0.019329; **physics** `physics-MLE[log,ms=1,it=30]` rel_l2=0.019335; **physics-custom** `custom-solver(fwd)` rel_l2=0.019337; **data-driven** `extra_trees/sqrt+pca16` rel_l2=0.10842
- SUPPORTED — E-01 *Reference arm: log-space multistart bounded LM MLE is statistically efficient in the interior, with a verified forward …*: val rel_l2 = 0.019339 vs predicted < 0.0215
- SUPPORTED — E-02 *How many likelihood steps does an amortized estimate need? Simulation-amplified MLP + 0/1/3 box-projected Gauss-Newton …*: val rel_l2 = 0.019337 vs predicted < 0.02
- SUPPORTED — E-04 *Initializer ablation: an extra-trees start plus a single 15-iteration LM run matches the 3-start MLE at about 1/6 of th…*: val rel_l2 = 0.019329 vs predicted < 0.0215
- SUPPORTED — E-05 *Missing control: single prior-centre-start bounded LM (no initializer, no multistart) reaches the same MLE*: val rel_l2 = 0.019335 vs predicted < 0.0215
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-001', 'P-009', 'P-018']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:24476298` Siyu Cen, Bangti Jin, Yikan Liu et al. (2023). "Recovery of Multiple Parameters in Subdiffusion from One Lateral Boundary Measurement". Theorem 3.2. http://arxiv.org/abs/2307.03263v1 — cited by ['P-001']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-002', 'P-010']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:20264653` Zhiyuan Li, Masahiro Yamamoto (2019). "Inverse Problems of Determining Coefficients of the Fractional Partial Differential Equations". Theorem 1 (Cheng, Nakagawa, Yamamoto and Yamazaki \cite{CNYY09}). http://arxiv.org/abs/1904.05505v1 — cited by ['P-002', 'P-010', 'P-019']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22914204` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Theorem 7. http://arxiv.org/abs/1003.5004v1 — cited by ['P-003']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:24993830` Marco Iglesias, Clint Dawson (2013). "The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs,". Theorem 1 (Hanke \cite {Hanke}). http://arxiv.org/abs/1302.3501v1 — cited by ['P-003']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-003']; researcher claim ['no']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:21117587` Hongjie Dong, Seick Kim (2016). "Partial Schauder estimates for second-order elliptic and parabolic equations: a revisit". Lemma 5.16. http://arxiv.org/abs/1502.00886v2 — cited by ['P-004']; researcher claim ['unknown']; harness screen **plausible**
- `ts:thm:26986930` MohammadJava Taheri, Abolfazl Diyanat, MortezaAli Ahmadi et al. (2025). "Predicting Drive Test Results in Mobile Networks Using Optimization Techniques". Lemma 1. http://arxiv.org/abs/2502.09305v1 — cited by ['P-007', 'P-008', 'P-015', 'P-016', 'P-017', 'P-025', 'P-026']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22158192` Afonso S. Bandeira, Jameson Cahill, Dustin G. Mixon et al. (2013). "Saving phase: Injectivity and stability for phase retrieval". Theorem 21. http://arxiv.org/abs/1302.4618v3 — cited by ['P-007', 'P-015']; researcher claim ['no', 'partial']; harness screen **unknown**
- `ts:thm:26665388` Yang Chen, Xiao-Li Meng, Xufei Wang et al. (2018). "Calibration Concordance for Astronomical Instruments via Multiplicative Shrinkage". Theorem 2. http://arxiv.org/abs/1711.09429v3 — cited by ['P-009', 'P-011', 'P-018', 'P-020']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26360863` Christian Kanzow, Theresa Lechner (2022). "Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems". Proposition. http://arxiv.org/abs/2210.07644v1 — cited by ['P-009', 'P-015', 'P-017', 'P-018', 'P-021', 'P-026']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26931793` Fan Ang, Li Chen, Nan Zhao et al. (2019). "Robust Federated Learning with Noisy Communication". Lemma 1. http://arxiv.org/abs/1911.00251v1 — cited by ['P-012']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:26956812` Ambar Pal, Jeremias Sulam (2023). "Understanding Noise-Augmented Training for Randomized Smoothing". Theorem 0.4 (Noise-Augmentation helps for distributions with small interference distance). http://arxiv.org/abs/2305.04746v1 — cited by ['P-012']; researcher claim ['no']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-03 *GCV-ridge linear-inverse baseline (raw features, log target): how much of the inverse map is linear?*: inconclusive — method did not produce a full-fidelity validation result
- E-06 *Simulation-amplified amortized log-k regression: n_sim 0 vs 6000 at a fixed Adam step count (shared code with P-017)*: inconclusive — method did not produce a full-fidelity validation result
- P-020 *Clean MMSE-vs-mode contrast in the pure physics arm: the posterior-mean gain is 0–3% overall and concentrated near the …* (conjectured): proposed but not executed
- P-022 *Missing control: single-start LM MLE from the prior centre (no initializer, no multistart)* (conjectured): proposed but not executed
- P-024 *Log-feature GCV-ridge: does log(u) linearise the inverse, or does it amplify noise at low-signal sensor times? (single-…* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-ml — P-003 *Learned amortized initializer + few-step log-space Levenberg–Marquardt refinement reaches near the …*: blocking issue: The estimated runtime (129.2 s) exceeds the per-experiment limit; reduce iterations or epochs before running.
- ABANDONED in round 2 by researcher-ml — P-004 *Pure amortized MLP with noise-matched augmentation and log targets meets the 0.05 threshold without…*: blocking issue: The estimated runtime (160 s) exceeds the per-experiment limit.
- ABANDONED in round 3 by researcher-experimental — P-012 *Ablation: noise augmentation OFF vs ON for the P-004 MLP (paired)*: Untestable as designed this round because the paired ON arm (P-004) is absent. The prior expectation is a null below unmeasured seed variance. P-016's simulation-amplification curve supersedes it as the test of 'varianc…
- ABANDONED in round 3 by researcher-experimental — P-013 *Random library-configuration control (descriptive)*: Budget triage on the critic's priority list: about 329 of 400 s are already committed. A descriptive n=3 control has no clean comparison arm this round (P-004 absent), and the two ridge arms (P-014 revised plus the new …
- ABANDONED in round 3 by researcher-exploratory — P-015 *Three-arm paired test: amortized init vs init+LM (MLE) vs init+LM+posterior mean (MMSE), stratified…*: Redundant with P-010 (init+LM mode), P-011 (MMSE vs mode) and the revised P-017 (learned init + likelihood refinement). It was the most expensive proposal in an over-budget portfolio, and its stratified claim was underp…
- PRUNED — idea-004 `custom-ml`: screen failed: Unclassified failure: compute budget for this stage exhausted

### 3.5 Remaining uncertainty

- E-01 (Reference arm: log-space multistart bounded LM MLE is stati…): challenge: Criterion (b) compares per-parameter relative RMSE against the MEDIAN per-parameter CRB rel std from 32 draws. RMSE over val samples estimates sqrt(mean CRB_rel^2), not the median. The relative CRB for k1/k4 …
- E-01 (Reference arm: log-space multistart bounded LM MLE is stati…): challenge: The arm IDs are inconsistent. This proposal says it is the paired reference for 'P-010, P-011', but P-019, P-020 and P-021 pair against 'P-009'. Fix one canonical ID for the reference MLE arm before running, …
- E-01 (Reference arm: log-space multistart bounded LM MLE is stati…): challenge: Mean chi^2/dof depends on the convention (divide by N or by N−M). With M=4 and N≈800, the two differ by 0.5%, which is about 2.5 sampling sd (sd≈0.002). Criterion (c) can therefore pass or fail on convention …
- E-01 (Reference arm: log-space multistart bounded LM MLE is stati…): challenge: The pilot chi^2/dof of 0.944 is a single sample. Its per-sample sd is sqrt(2/800)≈0.05, so it is uninformative about calibration, and the 750-sample mean is the real test.
- E-02 (How many likelihood steps does an amortized estimate need? …): challenge: The k1/k4 criterion (≤1.2× CRB) uses median CRB values. This has the same median-vs-RMS bias as P-018(b). Use the per-sample CRB at the truth or drop the criterion.
- E-02 (How many likelihood steps does an amortized estimate need? …): challenge: The lam=1e-3 damping scale is unspecified. If it is added to J^T J/σ² (entries about 1e5–1e7), it is negligible and the method is pure GN. If it is added to unscaled J^T J, it can dominate the k1/k4 direction…
- E-02 (How many likelihood steps does an amortized estimate need? …): challenge: This arm depends on P-025's MLP. If the MLP fit fails, or the seed is unlucky, all three arms inherit the problem. A single seed also means the 1-step vs 3-step comparison is conditional on one start distribu…
- E-02 (How many likelihood steps does an amortized estimate need? …): challenge: Cost comparisons against LM-30 must include the MLP training (about 25 s plus 6k forward calls), which pays off only when amortized across many samples. Report both total cost and per-sample marginal cost.
- E-03 (GCV-ridge linear-inverse baseline (raw features, log target…): challenge: The pilot MLP's 0.0325 is described as 'an upper bound on achievable nonlinear-learner error'. It is an achieved value, not a bound. Ridge comparisons should use the round's actual best learned arm.
- E-03 (GCV-ridge linear-inverse baseline (raw features, log target…): challenge: RidgeCV defaults to a single shared alpha across the 4 outputs. k1/k4 and k2/k3 differ by about 4× in sensitivity, so a shared alpha is a confound for per-parameter results. Use alpha_per_target=True.
- E-03 (GCV-ridge linear-inverse baseline (raw features, log target…): challenge: The outcome 0.05 < rel_l2 ≤ 0.06 is neither supported nor falsified. Define it (for example as 'partially linear').
- E-03 (GCV-ridge linear-inverse baseline (raw features, log target…): challenge: Standard scaling of near-zero outer-sensor or early-time features turns pure-noise columns into unit-variance columns. Ridge with GCV tolerates this, but the selected alpha will partly reflect noise-column su…
- E-04 (Initializer ablation: an extra-trees start plus a single 15…): challenge: Criterion (a) is written as rel_l2(P-010) − rel_l2(P-009), but this arm is P-019. Resolve the ID mapping (see P-018).
- E-04 (Initializer ablation: an extra-trees start plus a single 15…): challenge: The pilot initializer used log features with 300 training samples ('extra_trees/log+logy+aug2'). This arm uses raw features with 3000. The pilot is therefore weak evidence for this exact configuration. That i…
- E-04 (Initializer ablation: an extra-trees start plus a single 15…): challenge: Accuracy equivalence is the less informative half if P-021 passes. In that case the only claim left is cost. Iterations matter (15 vs 30) as well as the start, so the clean cost question is 'iterations to con…
- E-04 (Initializer ablation: an extra-trees start plus a single 15…): challenge: The rule-based estimate is 50.7 s against the claimed 5–10 s. The difference is probably the extra-trees fit with noise_aug=2 (9000 rows × 800 features × 300 trees, max_features=0.5). The cost claim should co…
- E-05 (Missing control: single prior-centre-start bounded LM (no i…): challenge: the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until tested
- E-05 (Missing control: single prior-centre-start bounded LM (no i…): challenge: The rule engine flagged proof-like language. The motivation states that GN on a smooth, well-conditioned problem converges, but the Fisher condition number at sampled points is local and implies nothing about…
- E-05 (Missing control: single prior-centre-start bounded LM (no i…): challenge: The meaning of 'prior centre' is ambiguous: 1.25 in k (arithmetic) or 1.0 (geometric, the centre in log space). From 1.25, the worst-case log distance to 0.5 is 0.92; from 1.0, the worst case is 0.69. Record …
- E-05 (Missing control: single prior-centre-start bounded LM (no i…): challenge: Criterion (c), chi^2/dof > 1.01 as a local-minimum detector, is insensitive. If 1% of samples sit in spurious minima with chi^2/dof of 1.5, the mean shifts by only about 0.005. Use the per-sample chi^2 tail i…
- E-06 (Simulation-amplified amortized log-k regression: n_sim 0 vs…): challenge: This is a single training seed. The paired bootstrap over val samples captures val-sample variance but not MLP initialization or optimizer variance. For a 3000-sample MLP, seed-to-seed rel_l2 variation of 5–1…
- E-06 (Simulation-amplified amortized log-k regression: n_sim 0 vs…): challenge: sklearn MLPRegressor's max_iter counts epochs, not Adam steps. Confirm that the truncated code really enforces 6000 optimizer steps (for example a partial_fit loop with a fixed batch size). Otherwise the 'equ…
- E-06 (Simulation-amplified amortized log-k regression: n_sim 0 vs…): challenge: Confirm that the simulated samples get i.i.d. N(0, 0.001²) noise added. Training on clean simulations would bias the comparison (train/val noise mismatch). align_z checks ordering, not augmentation noise.
- E-06 (Simulation-amplified amortized log-k regression: n_sim 0 vs…): challenge: The pilot used 300 real samples. With 3000 real samples, the marginal value of 6000 extra simulations is smaller, so the ≥20% prediction is less safe than the pilot suggests.
- Unresolved disagreement ['P-019', 'P-021']: Whether the learned initializer contributes accuracy or only cost. P-021 predicts a centre start suffices, which would make P-019's contribution purely iterations; neither yet measures iterations-to-converge directly.
- Unresolved disagreement ['P-026', 'P-018']: Whether 1–3 GN steps from an amortized start match multistart LM-30 accuracy. P-026's cost advantage holds only if MLP training is amortized; the total-cost accounting is unresolved.
- Unresolved disagreement ['P-024 (H_lin)', 'P-024 (H_noise)']: The internal competing mechanisms are confounded by the harness clip floor; an asinh transform is needed to separate them.
- Literature check INCOMPLETE: 0 failed queries (-), 2 skipped for budget, enabled=True
- Test-set sampling uncertainty: bootstrap 95% CI [0.01725, 0.01962] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **3 round(s)**: iteration cap reached (max_rounds=3); unresolved issues are documented.

| round | researcher | action | proposal | new version | origin | response to critic |
|---|---|---|---|---|---|---|
| 1 | researcher-mathematical | propose | - | P-001 | llm |  |
| 1 | researcher-mathematical | propose | - | P-002 | llm |  |
| 1 | researcher-ml | propose | - | P-003 | llm |  |
| 1 | researcher-ml | propose | - | P-004 | llm |  |
| 1 | researcher-experimental | propose | - | P-005 | heuristic-fallback |  |
| 1 | researcher-experimental | propose | - | P-006 | heuristic-fallback |  |
| 1 | researcher-exploratory | propose | - | P-007 | llm |  |
| 1 | researcher-exploratory | propose | - | P-008 | llm |  |
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The pilot evidence resolves most of the critic's points. (1) Aggregation. I cannot see the harness's rel_l2 d… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | I accept all three challenges, and the pilot supports the critic's arithmetic. The hybrid posterior-mean pilo… |
| 2 | researcher-mathematical | propose | - | P-011 | llm |  |
| 2 | researcher-ml | abandon | P-003 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-ml | abandon | P-004 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-experimental | revise | P-005 | P-012 | llm | I accept every point. (1) noise_scale: I do not know the harness semantics for certain, so the revised arm re… |
| 2 | researcher-experimental | revise | P-006 | P-013 | llm | (1) Power: agreed, n=2 cannot test whether reasoning adds value. I relabel this as a descriptive calibration … |
| 2 | researcher-experimental | propose | - | P-014 | llm |  |
| 2 | researcher-exploratory | revise | P-007 | P-015 | llm | Point by point. (1) Overlap with P-003/P-002: accepted. P-007 now absorbs P-003 as arm (ii) (estimator='mode'… |
| 2 | researcher-exploratory | revise | P-008 | P-016 | llm | The pilot outweighs argument here. Trained on a 300-sample train fraction plus 6000 simulations, the method r… |
| 2 | researcher-exploratory | propose | - | P-017 | llm |  |
| 3 | researcher-mathematical | revise | P-009 | P-018 | llm | Point-by-point replies to the critic.  (C1) CRB floor from 32 draws carries about ±10% error. Accepted. The m… |
| 3 | researcher-mathematical | revise | P-010 | P-019 | llm | Point-by-point replies to the critic.  (C1) The estimator is confounded. Accepted. estimator is set to mode, … |
| 3 | researcher-mathematical | revise | P-011 | P-020 | llm | Point-by-point replies to the critic.  (C1) Pre-specify the minimum detectable effect (MDE). Accepted. The pe… |
| 3 | researcher-mathematical | propose | - | P-021 | llm |  |
| 3 | researcher-ml | propose | - | P-022 | llm |  |
| 3 | researcher-experimental | abandon | P-012 | - | llm | I accept all three challenges. (1) P-004, the ON arm, is not scheduled this round, so the paired bootstrap th… |
| 3 | researcher-experimental | abandon | P-013 | - | llm | I accept both points. (1) At n=3 the arm is descriptive only, and it has low information value next to a phys… |
| 3 | researcher-experimental | revise | P-014 | P-023 | llm | (1) The fixed alpha=0.01 contradicted 'alpha chosen on val'. Fixed: the model is now RidgeCV over 15 log-spac… |
| 3 | researcher-experimental | propose | - | P-024 | llm |  |
| 3 | researcher-exploratory | abandon | P-015 | - | llm | All four challenges are accepted. (1) Duplication: arm (ii) is P-010 with a different initializer, and arm (i… |
| 3 | researcher-exploratory | revise | P-016 | P-025 | llm | (a) Cost: reduced to two arms, n_sim in {0, 6000}, following the critic's suggestion. The (n_sim=6000, gn_ite… |
| 3 | researcher-exploratory | revise | P-017 | P-026 | llm | (a) The k1/k4 mechanism is accepted as likely wrong, and it is withdrawn. The pilots outweigh my argument: th… |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.76 | fdbk-00046 |
| 2 | 9 | 0.85 | fdbk-00083 |
| 3 | 9 | 0.817 | fdbk-00107 |

### Idea evolution

**P-001 — Log-space multistart Levenberg–Marquardt MLE through the forward model reaches near the Cramér–Rao floor** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MLE[log,ms=3,it=30]` status literature-supported. *H:* With 4 parameters, a Fisher condition number of roughly 48 and sigma=0.001, the forward map F: [0.5,2]^4 -> R^{4xT} is locally well-conditioned. Per-sample bounded LM in log k with 3 starts and 30 iterations should converge to the global least-squares minimis…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: A large error concentrated in k1 and k4 means weak boundary-segment sensitivity is the bottleneck, so the prior should be used through MMSE or Tikhonov. Scatte…
  - critic r1 (quality 0.94): The CRB floor of about 0.0186 bounds unbiased estimators. The harness rel_l2 may be aggregated differently from the CRB estimate (dataset-level //K_hat-K/////K… · The 3 starts are not specified. Random uniform starts in a 4-D box with 30 iterations is reasonable, but a fixed box-centre start plus 2 random starts is safer… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If T_final is short relative to the diffusive time from the pulse to the end segments (about 0.2^2/k), sensors 5 and 35 see an almost-zero signal. Then k1 and … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `physics-MLE[log,ms=3,it=30]` status literature-supported. *H:* With a verified exact forward model and i.i.d. Gaussian noise of known sigma, the log-k LM MLE (3 starts, 30 iterations) is efficient on the full validation split. Three outcomes are predicted. First, val rel_l2 / CRB floor <= 1.15, so rel_l2 <= about 0.0215.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0215}`; if it fails: If (a) fails but (b) to (d) pass, the CRB floor estimate (32 draws) or the aggregation is mismatched, not the estimator. If (b) or (d) fails, the landscape is …
  - response to critic: The pilot evidence resolves most of the critic's points. (1) Aggregation. I cannot see the harness's rel_l2 definition, so I do not claim the two aggregations are equal. The pilot gives 0.0190 against a CRB floor of 0.0186, a gap of 2.5%. …
  - critic r2 (quality 0.955): The ratio criterion leans on a CRB floor estimated from only 32 parameter draws. That floor carries its own sampling error, plausibly about ±10%, so the 1.15 t… · The CRB bounds unbiased estimators. Near the box edges the MLE is not unbiased, and it can leave the box when run in unconstrained log space. rel_l2 slightly b… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · For k1 near 0.5, an unclipped MLE can land at 0.47. Clipping reduces error, so the result depends on the implementation rather than on efficiency. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v3 `P-018` r3 [revise, llm] `physics-MLE[log,ms=3,it=30]` status literature-supported. *H:* With an exact forward model and known Gaussian noise, the bounded log-k LM MLE (3 starts, 30 iterations) on all 750 validation samples satisfies four predictions: (i) rel_l2 at or below 1.15× the CRB floor; treating the 32-draw floor as an interval, rel_l2 < …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0215}`; if it fails: If (a) or (b) fails while (c) to (e) pass, the cause is floor mismatch or real inefficiency in the k1/k4 valleys; the per-sample CRB decides which. If (d) or (…
  - response to critic: Point-by-point replies to the critic.  (C1) CRB floor from 32 draws carries about ±10% error. Accepted. The method spec cannot compute a per-sample Fisher at the val truths, so I request it as a harness diagnostic (evidence_request 2). Unt…
  - critic r3 (quality 0.955): Criterion (b) compares per-parameter relative RMSE against the MEDIAN per-parameter CRB rel std from 32 draws. RMSE over val samples estimates sqrt(mean CRB_re… · The arm IDs are inconsistent. This proposal says it is the paired reference for 'P-010, P-011', but P-019, P-020 and P-021 pair against 'P-009'. Fix one canoni… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Take k1 rel CRB of 0.01 on half the interior and 0.035 on the other half. The median is about 0.02, but the RMS is 0.026. An efficient MLE then gives a ratio o… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-002 — Hybrid: learned initialiser plus posterior-mean refinement beats MLE by exploiting the bounded uniform prior on poorly-…** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[extra_trees/log+logy+aug2->LM15]` status conjectured. *H:* rel_l2 is a squared-error metric, and the prior is a bounded box. Under these conditions the MMSE estimator (posterior mean from importance sampling on the Laplace approximation) has lower expected rel_l2 than the MLE, mainly for k1 and k4. Near the bounds 0.…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.05, "expect": "better"}`; if it fails: If MMSE gives no gain, the posterior is effectively Gaussian and interior, so MLE is already near-optimal and the CRB floor (about 0.019) is the true limit. Re…
  - critic r1 (quality 0.865): The expected MMSE gain is small. With CRB std of about 2.2% for k1, the fraction of uniform-prior mass within 2 std of either bound is about (2*0.011 + 2*0.044… · features='log' is hazardous. Sensors 5 and 35 sit near the Dirichlet ends, where the true temperature can be on the order of sigma or smaller. Noisy readings c… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If every LM run already reaches the MLE, the initializer type is irrelevant. Any difference between P-002 and P-007 would then be pure MMSE-implementation nois… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-010` r2 [revise, llm] `hybrid[extra_trees/raw+logy+aug2->LM15]` status literature-supported. *H:* An extra-trees initializer on raw features lands in the MLE basin for at least 99% of validation samples. If so, 15 LM iterations with 1 extra start give rel_l2 statistically indistinguishable from P-001 (a paired bootstrap CI on the difference that contains …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0215}`; if it fails: If (a) fails, 15 iterations or a tree initializer is insufficient for some samples, and the inspection should check whether failures concentrate at k1/k4 near …
  - response to critic: I accept all three challenges, and the pilot supports the critic's arithmetic. The hybrid posterior-mean pilot gave 0.01883. The MLE pilot gave 0.01902 on the same 60-sample pilot. That is a 1% difference, far inside the sampling noise. It…
  - critic r2 (quality 0.88): The arm is confounded. It uses estimator=posterior_mean with 256 IS samples, while its reference P-009 uses mode. Any difference mixes the effect of the initia… · The cost claim is self-refuted: the pilot used 23k forward evaluations against the MLE's 15k. The proposal's stated goal ('fewer LM iterations') is not the bin… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If LM from a random start converges within 15 iterations anyway, a basin-quality initializer is irrelevant. That case shows up as (c) failing while (a) passes,… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v3 `P-019` r3 [revise, llm] `hybrid[extra_trees/raw+logy+aug2->LM15]` status literature-supported. *H:* An extra-trees initializer (raw features, log targets, trained on the 3000 train samples) puts at least 99% of val samples in the MLE basin. With estimator = mode, 15 LM iterations and no extra starts, the paired rel_l2 difference against P-009 then has a 95%…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0215}`; if it fails: If (a) or (b) fails, a single 15-iteration start is insufficient for some samples. The failures should then be checked for clustering at k1/k4 near the bounds,…
  - response to critic: Point-by-point replies to the critic.  (C1) The estimator is confounded. Accepted. estimator is set to mode, so the only contrasts with P-009 are the initializer, the iteration count and the number of starts. posterior_samples is left at t…
  - critic r3 (quality 0.855): Criterion (a) is written as rel_l2(P-010) − rel_l2(P-009), but this arm is P-019. Resolve the ID mapping (see P-018). · The pilot initializer used log features with 300 training samples ('extra_trees/log+logy+aug2'). This arm uses raw features with 3000. The pilot is therefore w… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-003 — Learned amortized initializer + few-step log-space Levenberg–Marquardt refinement reaches near the Cramér–Rao floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM12]` status literature-supported. *H:* A torch_mlp trained on the 3000 simulated samples gives estimates inside the Gauss–Newton basin of attraction. A few bounded LM iterations in log-k space through the exact forward model then cut val rel_l2 to at most 0.035, which is within about 2x of the CRB…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: Hybrid error near the MLP error, with large residuals, means the learned init is outside the basin or LM is not converging; that points to more starts, or Tikh…
  - critic r1 (quality 0.755): The estimated runtime (129.2 s) exceeds the per-experiment limit; reduce iterations or epochs before running. · The estimated runtime of 129.2 s exceeds the roughly 120 s per-experiment limit stated in the proposal itself, and the total budget is 400 worker-seconds. Appl… · If LM converges, the hybrid with estimator='mode' should match P-001's MLE almost exactly. The hypothesis is then not really about the MLP; it is about saving … · Restate the claim as a compute claim: the hybrid reaches P-001's rel_l2 within its CI while using fewer forward calls or less wall time. · An MLP init with 5% error and LM with 12 iterations converges to the same point as cold-start LM with 30 iterations. In that case the hybrid gives no accuracy …

**P-004 — Pure amortized MLP with noise-matched augmentation and log targets meets the 0.05 threshold without forward-model calls…** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* The inverse map from 4 sensor time series to log k is smooth and low-dimensional: the 4 parameters are well-identified with Fisher condition number about 48. An MLP trained on 3000 samples with log targets, standardized raw features and on-the-fly noise augme…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: An MLP plateau around 0.06-0.10 would show that the 3000-sample approximation error dominates. That would justify the hybrid (proposal 1), or training on addit…
  - critic r1 (quality 0.74): The estimated runtime (160 s) exceeds the per-experiment limit. · The estimate of 160 s exceeds the per-experiment limit and uses 40% of the total 400 s budget. Reduce epochs (early stopping) or width. · The '40% better than ridge/kernel_ridge' criterion needs those baselines to be run. Neither appears in the portfolio, so the test cannot be decided as written. · Set the learned map's success bar in terms of the CRB: report per-component error divided by CRB std, not only whether rel_l2 is below 0.05. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Matched-noise training augmentation** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - critic r1 (quality 0.479): Missing evidence (rule finding). · noise_scale semantics are ambiguous, and an absolute value of 1.0 would invalidate the run. · The falsification test references an undefined high-noise shift set. · noise_scale=1.0 is ambiguous. If it is an absolute std, it is about 1000x the true sigma and would destroy the signal. If it is a multiple of sigma, it duplica… · The 'high-noise shift set' is not defined anywhere in the task. The second half of the falsification test cannot be evaluated.
- v2 `P-012` r2 [revise, llm] `torch_mlp/raw+logy` status conjectured. *H:* H1: the P-004 MLP overfits the particular noise realisations in its 3000 training samples, so removing augmentation (noise_aug=0) raises val rel_l2 by at least 5% relative to P-004 (noise_aug=3). H0: the difference falls inside the paired-bootstrap 95% CI, me…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": -0.05, "expect": "worse"}`; if it fails: If there is no significant difference, augmentation is not a lever in this regime. Learned-map error is approximation- or sample-limited, which favours P-008 (…
  - response to critic: I accept every point. (1) noise_scale: I do not know the harness semantics for certain, so the revised arm removes the ambiguity entirely by setting noise_aug=0. P-005 becomes the augmentation-OFF arm of P-004, with no new noise injected. …
  - critic r2 (quality 0.85): P-004 is not in this round's list. If it is not run at full fidelity in the same execution, the paired bootstrap cannot be done. The ablation is only interpret… · The prior argument is that training features already carry matched noise. Also, any learned map is expected to sit at 0.03 or above against a noise floor of 0.… · Treat seed variance as part of the null. Run 2 seeds of the OFF arm, or compare against P-004's replica spread. · If the MLP is epoch-limited (150 epochs), augmentation effectively shrinks the number of passes over the clean data. That could make OFF better for a reason un… · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-006 — Random-search control arm** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - critic r1 (quality 0.524): Missing evidence (rule finding). · n=2 random configurations has very little power. A loss tells us almost nothing about whether reasoning adds value, and a single lucky win would be over-interp… · The search space is undefined. If it includes physics or hybrid methods, a random draw could reach the MLE floor and make the control trivially competitive. · Restrict the comparison to library (data-driven) methods with a declared search space, and compare against the best reasoned data-driven arm, not the best over… · Which method families and hyperparameter ranges are sampled?
- v2 `P-013` r2 [revise, llm] `random-search x3` status conjectured. *H:* Descriptive expectation, not a test: the best of 3 uniform random library configurations has worse val rel_l2 than the best reasoned data-driven arm. Many library families (knn, trees, rbf-SVM) are poorly suited to a smooth, high-dimensional, correlated 4-out…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.0, "expect": "worse"}`; if it fails: If a random config matches or beats the reasoned data-driven arms, the library space is forgiving. The reasoning phase added little at the data-driven level, a…
  - response to critic: (1) Power: agreed, n=2 cannot test whether reasoning adds value. I relabel this as a descriptive calibration point, not a hypothesis test, and raise it to n=3 at negligible extra cost. Any single win is reported as anecdotal. (2) Search sp…
  - critic r2 (quality 0.68): With n=3 the arm is descriptive only, as the authors concede. Its information value is low next to a physics arm already at the CRB floor. · 'Best reasoned data-driven arm' is ill-defined if P-004 is not run. It would then be compared against P-016 (simulation-amplified), which is not a comparable s… · Which arm is the comparison reference if P-004 is not in this round?

**P-007 — Amortized init + forward-model refinement scored by posterior mean (MMSE), not MLE** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+pca40+logy+aug1->LM15]` status conjectured. *H:* Per-sample error, not sample size, is what limits this problem. A learned initializer (torch_mlp, log-target) followed by a few log-space Levenberg-Marquardt steps through the exact forward model, then importance-sampled posterior-mean estimation under the bo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the mode variant is already about 0.02 and posterior_mean does not help, the problem is effectively in the asymptotic, interior regime and MLE is enough. Fu…
  - critic r1 (quality 0.895): The three-arm design (init / mode / posterior_mean) is the best-controlled test of MMSE in the portfolio, but it subsumes P-003 and overlaps P-002. · The expected MMSE gain is small (see P-002: about 7% of samples have k1 or k4 near a bound). Requiring (iii) < (ii) on the aggregate needs a paired bootstrap, … · Compare against the Bayesian (van Trees) bound under the uniform prior, not only the unbiased CRB, when claiming MMSE efficiency. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-015` r2 [revise, llm] `hybrid[torch_mlp/raw+pca100+logy+aug1->LM15]` status conjectured. *H:* With a torch_mlp log-target initializer (PCA-100) and 15 log-space LM iterations through the exact forward model, the posterior-mean arm reaches val rel_l2 < 0.03. Its paired improvement over the mode arm is positive with a 95% bootstrap CI excluding 0. That …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If (ii) is about 0.02 and (iii) is not better, the problem is in the interior asymptotic regime and the estimator choice is irrelevant, so effort should shift …
  - response to critic: Point by point. (1) Overlap with P-003/P-002: accepted. P-007 now absorbs P-003 as arm (ii) (estimator='mode', the same initializer, same LM settings) and keeps P-002 only as the initializer ablation. Arm (i), the initializer alone, comes …
  - critic r2 (quality 0.85): The arm largely duplicates others. Arm (ii), init plus LM with mode, is P-010 with a different initializer. The mode-vs-MMSE contrast in arm (iii) is P-011, bu… · The prediction of <0.03 is far too loose. Hybrid pilots already gave 0.0188. A sharp prediction would be within 3% of P-009. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-008 — Questioning the fixed-training-set framing: simulation-amplified amortized inference in log-k** (researcher-exploratory, family physics-custom)
- v1 `P-008` r1 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The 3000-sample training set is an artificial bottleneck. The forward model is available and the noise model is known, so the training set can be enlarged with 6000 extra prior draws simulated through context.forward plus sigma=0.001 Gaussian noise. An MLP tr…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.045}`; if it fails: If amplification does not help, the learned-map error is approximation or bias rather than variance from small n, and per-sample forward refinement (proposal 1…
  - critic r1 (quality 0.88): The 30 s estimate looks optimistic: 6000 forward calls plus an sklearn MLP on about 9000 x 800 features. Pilot the timing. · The control (an identical sklearn MLP trained on 3000 samples) is not in the portfolio. P-004 uses torch_mlp with different settings, so it is not a valid cont… · Use the posterior-mean framing explicitly: with prior-distributed simulations and squared loss, the regression target is E[log k / x], not k. Evaluate in k spa… · If the MLP error is bias- or approximation-limited (in capacity or optimisation), tripling n gives less than 20% improvement. A null result would then point to… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-016` r2 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* With the full 3000 training samples, augmenting with 6000 prior-drawn simulations (forward plus sigma=0.001 noise) gives val rel_l2 < 0.035 with no per-sample optimization. This is at least 20% lower than the identical code with n_sim=0. The 0 to 3000 to 6000…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the gain from n_sim=0 is below 20%, learned-map error is approximation- or bias-limited, and per-sample refinement (P-007 and the new hybrid) is the right u…
  - response to critic: The pilot outweighs argument here. Trained on a 300-sample train fraction plus 6000 simulations, the method reached 0.0325 on 60 val samples in 22s, which is already below the 0.045 prediction. This shows that 6000 forward calls plus the s…
  - critic r2 (quality 0.875): The 3-arm curve (n_sim 0, 3000, 6000) costs about 60–80 s by the authors' own estimate, not the 30 s rule estimate. Budget accordingly. · Framing the target as the 'Bayes posterior mean' holds only if the MLP reaches its approximation capacity. At 0.0325 against a floor of 0.0186, the MLP is far … · Use a fixed number of gradient steps rather than epochs across the n_sim arms. Otherwise the larger arms get more updates, and that is confounded with having m… · If n_sim=3000 and n_sim=6000 tie while n_sim=0 is worse, the error is not variance-limited; it is bias-limited past 6000 samples. The hypothesis wording would … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v3 `P-025` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* With the full 3000 training samples and a fixed budget of 6000 Adam steps, adding 6000 prior-drawn noisy simulations gives val rel_l2 < 0.033 with no per-sample optimization. This is at least 20% lower than the identical code with n_sim=0, and the paired boot…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.033}`; if it fails: If the improvement is below 20%, the amortized error is approximation- or optimization-limited, more data does not help, and per-sample likelihood refinement (…
  - response to critic: (a) Cost: reduced to two arms, n_sim in {0, 6000}, following the critic's suggestion. The (n_sim=6000, gn_iters=0) arm is byte-identical code and parameters to P-017's gn_iters=0 arm, so it runs once and serves both. (b) The 'Bayes posteri…
  - critic r3 (quality 0.875): This is a single training seed. The paired bootstrap over val samples captures val-sample variance but not MLP initialization or optimizer variance. For a 3000… · sklearn MLPRegressor's max_iter counts epochs, not Adam steps. Confirm that the truncated code really enforces 6000 optimizer steps (for example a partial_fit … · Treat seed variance as part of the uncertainty: run 2 seeds per arm if budget allows (about +50 s), or report the effect as single-seed and provisional. · If the n_sim=0 arm overfits at 6000 steps and n_sim=6000 does not, the measured 'simulation benefit' is a regularisation effect that early stopping would also … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-011 — Clean MMSE-vs-mode contrast in the pure physics arm: posterior-mean gain is ≤2% overall and concentrated in near-bound …** (researcher-mathematical, family physics)
- v1 `P-011` r2 [propose, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* The setup uses an identical physics solver (log, 3 starts, 30 iterations) and the identical 750 validation samples, varying only the estimator (mode vs posterior_mean with 512 importance samples). The predicted overall relative rel_l2 improvement of posterior…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.01, "expect": "better"}`; if it fails: A gain much larger than 3% implies the posterior is non-Gaussian in the interior, for example skewed in k1/k4. A prior-aware learned or MCMC estimator would th…
  - critic r2 (quality 0.855): The predicted effect, 0–3% overall with about 50 near-bound samples, is likely below the paired-bootstrap resolution. Pre-specify the minimum detectable effect… · Criterion (b), '>70% of the squared-error reduction comes from near-bound samples', is a ratio whose denominator can be near 0 or negative. If the total gain i… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · For rel_l2 as a ratio of norms (not a mean of squared errors), the posterior mean is not guaranteed optimal. A small reversal is possible, as the authors note. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-020` r3 [revise, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* The setup uses an identical solver (log, 3 starts, 30 iterations) on the identical 750 val samples, with only the estimator varied (posterior_mean with 512 IS samples vs P-009's mode). Four predictions: (i) the relative rel_l2 improvement lies in [0%, 3%]; (i…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.01, "expect": "better"}`; if it fails: An improvement above 3% means the posterior is non-Gaussian in the interior, which warrants MCMC or a prior-aware learned estimator. A degradation with low ESS…
  - response to critic: Point-by-point replies to the critic.  (C1) Pre-specify the minimum detectable effect (MDE). Accepted. The per-sample paired squared-error differences are near 0 in the interior and concentrated in about 5–10% of samples. I pre-register tw…
  - critic r3 (quality 0.84): If rel_l2 is a global Frobenius ratio, the posterior mean minimises it only approximately: the denominator //K//_F is fixed, but the per-sample loss is unnorma… · The IS proposal is presumably the Laplace Gaussian in log-k truncated or reweighted by the box. If the proposal is not truncated, samples outside the box get z… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-014 — Ridge linear-inverse baseline (the missing control for P-004)** (researcher-experimental, family data-driven)
- v1 `P-014` r2 [propose, llm] `ridge/raw+logy` status conjectured. *H:* The inverse map from sensor time series to log k is substantially nonlinear over the prior box [0.5, 2]^4, so ridge regression on raw features with a log target gives val rel_l2 of 0.08 or more. That is at least about 2x worse than the reasoned MLP and well a…
  - prediction: `{"metric": "rel_l2", "op": ">", "value": 0.08}`; if it fails: If ridge is near-competitive, the inverse is nearly linear in log k. Cheap linear initialisers would then suffice for the hybrid arms, and MLP capacity is not …
  - critic r2 (quality 0.81): The assumption 'alpha chosen on val only' conflicts with the fixed spec alpha=0.01. Either sweep it or drop the claim. · With about 800 standardized correlated features and 3000 samples, alpha=0.01 is effectively unregularized and may be variance-dominated. A poor ridge score wou… · Sweep alpha over {1e-2, 1, 1e2} on val, so the nonlinearity conclusion is not an artefact of regularisation. · A log-of-features transform could make the map nearly linear, because heat-kernel decay is exponential in k·t. Ridge on raw temperatures understates how linear… · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-023` r3 [revise, llm] `custom-ml` status conjectured. *H:* H1: the inverse map from raw sensor time series to log k is materially nonlinear over [0.5,2]^4, so GCV-tuned ridge gives val rel_l2 > 0.06, at least 1.8x the pilot MLP (0.0325) and more than 3x the CRB floor. Falsified if ridge reaches ≤ 0.05, or if its pair…
  - prediction: `{"metric": "rel_l2", "op": ">", "value": 0.06}`; if it fails: If ridge reaches ≤ 0.05, the inverse is nearly linear in log k with raw features. Linear initializers would then suffice for hybrids, and MLP capacity is not t…
  - response to critic: (1) The fixed alpha=0.01 contradicted 'alpha chosen on val'. Fixed: the model is now RidgeCV over 15 log-spaced alphas from 1e-4 to 1e3. It uses efficient leave-one-out GCV on the training split only, so neither val nor test is touched for…
  - critic r3 (quality 0.86): The pilot MLP's 0.0325 is described as 'an upper bound on achievable nonlinear-learner error'. It is an achieved value, not a bound. Ridge comparisons should u… · RidgeCV defaults to a single shared alpha across the 4 outputs. k1/k4 and k2/k3 differ by about 4× in sensitivity, so a shared alpha is a confound for per-para… · Use alpha_per_target=True and report the selected alpha per parameter. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-017 — Does refining an amortized posterior-mean estimate toward the MLE help or hurt? Simulation-amplified MLP + prior-damped…** (researcher-exploratory, family physics-custom)
- v1 `P-017` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* Starting from the simulation-amplified MLP (P-008) and taking 3 log-space Gauss-Newton steps, with a finite-difference Jacobian through context.forward, weak Tikhonov damping and per-sample acceptance only if chi^2 decreases, gives val rel_l2 < 0.025. This be…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If GN gives no gain, the amortized MLP is already at the likelihood optimum to within noise, so remaining error is irreducible or bias near the bounds, and pur…
  - critic r2 (quality 0.895): The mechanism for 'k1/k4 worsen' conflicts with the empirical evidence. The pure MLE reaches 0.0190 while the amortized MLP reaches 0.0325, so the MLP is not n… · Three GN steps from a 3% start at high SNR should essentially reach the MLE. The prediction of <0.025 is then loose; P-009 suggests about 0.019. · Expect gains in all four parameters. Revise the k1/k4 prediction to 'k1/k4 relative RMSE approaches its CRB', and test whether any residual worsening is confin… · If near-bound k1 = 0.52 and the MLE gives 0.49, clipping to 0.5 removes most of the harm. The 'k1/k4 worsen' signal then disappears for implementation reasons. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-026` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* Starting from the P-016 MLP (n_sim=6000, fixed 6000 steps), 3 box-projected, Marquardt-damped log-space Gauss-Newton steps with per-sample monotone acceptance give val rel_l2 < 0.0200. A single step gives rel_l2 within 10% of the 3-step result. Per-parameter …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.02}`; if it fails: If 3 steps stay above 0.020, the MLP start is often outside the MLE basin or the damping is too conservative: check n_accept and chi^2, and fall back to the P-…
  - response to critic: (a) The k1/k4 mechanism is accepted as likely wrong, and it is withdrawn. The pilots outweigh my argument: the MLE reaches 0.0190 and the MLP 0.0325, so MLP approximation error dominates any shrinkage benefit (at most about 1%, per P-011).…
  - critic r3 (quality 0.94): The k1/k4 criterion (≤1.2× CRB) uses median CRB values. This has the same median-vs-RMS bias as P-018(b). Use the per-sample CRB at the truth or drop the crite… · The lam=1e-3 damping scale is unspecified. If it is added to J^T J/σ² (entries about 1e5–1e7), it is negligible and the method is pure GN. If it is added to un… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · From a 3% start with sensitivity curvature of order 1 in log k, Newton's quadratic convergence gives a post-step error of about 1e-3. That is below the CRB, so… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-021 — Missing control: single prior-centre-start bounded LM (no initializer, no multistart) reaches the same MLE** (researcher-mathematical, family physics)
- v1 `P-021` r3 [propose, llm] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* The log-k landscape is effectively unimodal on [0.5, 2]^4 at sigma = 1e-3. One bounded LM run from the prior centre (multistart = 1, 30 iterations, mode estimator) therefore gives four results on the 750 val samples: (i) a paired rel_l2 difference against P-0…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.0215}`; if it fails: If the arm fails, with failures concentrated near the corners, then the landscape has spurious basins or slow valleys far from the centre. That validates the i…
  - critic r3 (quality 0.89): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · The rule engine flagged proof-like language. The motivation states that GN on a smooth, well-conditioned problem converges, but the Fisher condition number at … · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-022 — Missing control: single-start LM MLE from the prior centre (no initializer, no multistart)** (researcher-ml, family physics)
- v1 `P-022` r3 [propose, llm] `physics-MLE[log,ms=1,it=30]` status conjectured. *H:* With a Fisher condition number of about 48 and a log-parameterised, 4-dimensional, box-bounded least-squares problem, the misfit landscape has a single basin over essentially the whole prior box. A single LM run from the prior centre (log-k = log(1.0) area) s…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If single start is clearly worse than 3-start, multimodality or slow convergence near the bounds is real. Initializers then matter for accuracy, which justifie…
  - critic r3 (quality 0.366): Exact duplicate of P-021's method; merge or differentiate before running. · executes exactly the same method as P-021 (researcher-mathematical); differentiate the hypothesis or merge · The method is byte-identical to P-021's (multistart=1, iters=30, mode, log). Running both wastes 5.7 s and produces no new information. · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-024 — Log-feature GCV-ridge: does log(u) linearise the inverse, or does it amplify noise at low-signal sensor times? (single-…** (researcher-experimental, family data-driven)
- v1 `P-024` r3 [propose, llm] `custom-ml` status conjectured. *H:* There are two competing mechanisms. H_lin: log temperature approximately linearises the dependence on log k (heat-kernel decay ~ exp(-c·k·t)), so log-feature ridge beats raw-feature ridge (P-014) by ≥ 20% rel_l2. H_noise: entries where u is comparable to sigm…
  - prediction: `{"metric": "rel_l2", "op": ">", "value": 0.06}`; if it fails: If log features beat raw by ≥ 20% (H_lin), the learned arms should switch to log features, and P-015/P-016 may be under-using a linearising transform. If neith…
  - critic r3 (quality 0.775): The clipping value for noisy readings ≤0 decides the outcome. Clipping at 1e-8 maps those entries to log ≈ −18. After standard scaling, a handful of such entri… · The H_lin mechanism is shaky. At late times, log u ≈ −λ1(k)·t + c, which is linear in λ1, and λ1 is a nonlinear (harmonic-type) function of k, not linear in lo… · Use a noise-aware transform: inside the model, set features=raw, scaler=none and apply a pipeline of FunctionTransformer(arcsinh(X/(2σ))) followed by StandardS… · With uniform k, the late-time sensor reading u ≈ A·exp(−π² k t). Then log u is linear in k, so a ridge regression of log k on log u is still nonlinear, and H_l… · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

- Critic round 1 portfolio view: Prioritise P-001 (cheapest, and the reference arm) together with the alignment and chi^2 pilot. Merge P-003 into P-007 as a three-arm MMSE test, and keep P-002 only as the initializer ablation. Run P-008 with its own n_sim=0 control. Fold P-005 into P-004 as an ablation after fixing the noise_scale semantics, and add a ridge baseline. P-003 and P-004 exceed the per-experiment time limit, and P-00…  families={'physics': 1, 'hybrid': 3, 'data-driven': 3, 'physics-custom': 1}
- Critic round 2 portfolio view: The physics MLE is already at the CRB floor in the pilot, so the open questions are mainly about cost and mechanism, not accuracy. Rule-estimated costs sum to about 329 of 400 s, and multi-arm ablations (P-015, P-016, P-017) push the total over. Two fixes make the remaining runs cleaner and cheaper: set P-010 to the mode estimator, and drop P-015's arm (iii) or defer P-015. P-017's k1/k4 mechanis…  families={'physics': 2, 'hybrid': 2, 'data-driven': 3, 'physics-custom': 2}
- Critic round 3 portfolio view: The portfolio can be executed this round once P-022 is merged into P-021 and the reference-arm ID is fixed. All remaining issues are analysis-level: the per-sample CRB instead of the median, the metric definition (global vs per-sample rel_l2), the local-minimum detector based on the chi^2 tail, alpha_per_target for ridge, the asinh features for P-024, and the step-count and noise checks in the P-…  families={'physics': 4, 'hybrid': 1, 'physics-custom': 2, 'data-driven': 2}

**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MLE[log,ms=3,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.76 s
- P-002 `hybrid[extra_trees/log+logy+aug2->LM15]`: ok rel_l2=0.018826 on 60 val samples in 4.77 s
- P-008 `custom-solver(fwd)`: ok rel_l2=0.032487 on 60 val samples in 22.1 s

## 5. Literature retrieval (TheoremSearch)

22 queries, 0 failures, 66 distinct statements retrieved, 2 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant conductivity in parabolic eq… | 4 refs via /search (cache) |
| researcher-experimental | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search (cache) |
| researcher-mathematical | 1 | uniqueness of piecewise constant diffusion coefficient in one-dimensional heat equation f… | 4 refs via /search |
| researcher-experimental | 1 | Cramer-Rao bound attainment maximum likelihood estimator nonlinear least squares Gaussian… | 4 refs via /search |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search |
| researcher-exploratory | 1 | Gauss-Newton or Levenberg-Marquardt local convergence for nonlinear least squares paramet… | 4 refs via /search |
| researcher-ml | 1 | local convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares para… | 4 refs via /search |
| researcher-experimental | 1 | Cramer-Rao lower bound for parameter estimation in partial differential equations | FAILED: budget |
| researcher-experimental | 1 | optimal sensor placement for parameter identification in diffusion equations | FAILED: budget |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | Matched-noise training augmentation | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | Any evidence or citation that noise augmentation helps when training data already carry t… | 4 refs via /search |
| researcher-ml | 2 | rel_l2 vs LM iteration (0, 2, 4, 8, 12) starting from the MLP init | 4 refs via /search |
| researcher-mathematical | 2 | Distribution of chi^2 = RSS/sigma^2 at the converged MLE on 50 val samples | 4 refs via /search |
| researcher-exploratory | 2 | Stratified posterior_mean minus mode gain by distance to the bounds, with a paired bootst… | 4 refs via /search |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-mathematical | 3 | per-sample CRB computed at the 750 val truths, aggregated with the harness rel_l2 definit… | 4 refs via /search |
| researcher-exploratory | 3 | Bayesian Cramer-Rao (van Trees) bound with uniform prior on a bounded interval | 4 refs via /search |
| researcher-experimental | 3 | P-004 replica spread of val rel_l2 | 4 refs via /search |
| researcher-experimental | 3 | input noise injection equivalent to Tikhonov regularization small noise squared loss | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_full_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 179 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-018) | physics | `physics-MLE[log,ms=3,it=30]` | researcher-mathematical | 0.955 | best-rated proposal in family 'physics' (quality 0.95) |
| E-02 | proposal (P-026) | physics-custom | `custom-solver(fwd)` | researcher-exploratory | 0.94 | best-rated proposal in family 'physics-custom' (quality 0.94) |
| E-03 | proposal (P-023) | data-driven | `custom-ml` | researcher-experimental | 0.86 | best-rated proposal in family 'data-driven' (quality 0.86) |
| E-04 | proposal (P-019) | hybrid | `hybrid[extra_trees/raw+logy+aug2->LM15]` | researcher-mathematical | 0.855 | best-rated proposal in family 'hybrid' (quality 0.85) |
| E-05 | proposal (P-021) | physics | `physics-MLE[log,ms=1,it=30]` | researcher-mathematical | 0.89 | quality 0.89 with diversity penalty |
| E-06 | proposal (P-025) | physics-custom | `custom-solver(fwd)` | researcher-exploratory | 0.875 | quality 0.88 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-022 *Missing control: single-start LM MLE from the prior centre (no initializer, no …* — blocking: Exact duplicate of P-021's method; merge or differentiate before running.
- not advanced: P-020 *Clean MMSE-vs-mode contrast in the pure physics arm: the posterior-mean gain is…* — portfolio size limit (6); quality 0.84
- not advanced: P-024 *Log-feature GCV-ridge: does log(u) linearise the inverse, or does it amplify no…* — portfolio size limit (6); quality 0.78

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-005 | E-04 | research | hybrid | `hybrid[extra_trees/raw+logy+aug2->LM15]` | 0.02136 | **0.019329** | 0.9966 | 0.02549 | 8.2 | 5.61 | 50987 | 0.957 | FINAL |
| idea-013 | E-04 | refine | hybrid | `hybrid[extra_trees/raw+logy+aug2->LM15]` | - | **0.019331** | 0.9966 | 0.02549 | 9.49 | 4.34 | 50735 | 0.957 | evaluated |
| idea-012 | E-04 | refine | hybrid | `hybrid[extra_trees/raw+logy+aug2->LM15]` | - | **0.019331** | 0.9966 | 0.02549 | 8.33 | 4.24 | 50717 | 0.957 | evaluated |
| idea-006 | E-05 | research | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.00973 | 4.6 | 61517 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0101 | 4.62 | 61517 | 0.957 | evaluated |
| idea-003 | E-02 | research | physics-custom | `custom-solver(fwd)` | 0.02138 | **0.019337** | 0.9966 | 0.0255 | 32.9 | 0.617 | 18010 | - | evaluated |
| idea-002 | E-01 | research | physics | `physics-MLE[log,ms=3,it=30]` | 0.02139 | **0.019339** | 0.9966 | 0.0255 | 0.0101 | 11.3 | 176354 | 0.957 | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 0.994 | 0.149 | 0 | - | evaluated |
| idea-015 | E-08 | refine | data-driven | `extra_trees/sqrt+pca16` | - | **0.1094** | 0.8898 | 0.1443 | 0.94 | 0.132 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.464 | 0.0488 | 0 | - | evaluated |
| idea-014 | E-08 | refine | data-driven | `extra_trees/raw+pca16` | - | **0.11514** | 0.8779 | 0.1518 | 0.934 | 0.133 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `custom-ml` | - | **-** | - | - | - | - | - | - | screen failed: Unclassified failure: compute budget for this stage ex… |
| idea-007 | E-06 | research | physics-custom | `custom-solver(fwd)` | 0.03397 | **-** | - | - | - | - | - | - | ok |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **-** | - | - | - | - | - | - | ok |

## 8. Lineage of the final method

- `idea-005` `hybrid[extra_trees/raw+logy+aug2->LM15]` ← P-019 v3 (researcher-mathematical, r3, revise) ← P-010 v2 (researcher-mathematical, r2, revise) ← P-002 v1 (researcher-mathematical, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00238` [checkpoint] by scheduler: final method (single)
- `resu-00235` [result] by executor: exp-0027 rel_l2=0.01842 r2=0.997 rmse=0.02423
- `hypo-00120` [hypothesis] by scheduler: idea-005 hybrid[extra_trees/raw+logy+aug2->LM15] (research)
- `plan-00234` [plan] by engineer-v2: exp-0027: final fit of 1 member(s) on train+val; test once
- `gate-00119` [gate] by gate: E-04: approved (hybrid[extra_trees/raw+logy+aug2->LM15])
- `prop-00095` [proposal] by researcher-mathematical: P-019 v3 [revise] mathematical: Initializer ablation: an extra-trees start plus a single 15-iteration LM run matches the 3
- `spec-00001` [spec] by scheduler: research spec heat_v2_full_s2 (e63107a3cb)
- `prop-00072` [proposal] by researcher-mathematical: P-010 v2 [revise] mathematical: Initializer ablation: cheap extra-trees (raw features) + 15-iteration LM reaches the MLE a
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `prop-00038` [proposal] by researcher-mathematical: P-002 v1 [propose] mathematical: Hybrid: learned initialiser plus posterior-mean refinement beats MLE by exploiting the bou

## 9. Compute and cost

- Execution compute: 254.1 / 400 worker-seconds over 26 jobs; wall 828 s
- By stage: baseline 2.1s, pilot 29.6s, screen 88.7s, evaluate 77.0s, refine 34.4s, final 22.4s
- LLM: 20 calls (by phase {'research': 19, 'execution': 1}), 1123 s, $4.307 reported · TheoremSearch: 22 queries (0 failed)

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

The evidence establishes that likelihood-based inversion through the forward heat solver solves this task with a wide margin. Three independent routes reach essentially the same validation error: physics-MLE in log-space (0.019335), the custom forward solver (0.019337), and the hybrid extra-trees initializer with 15-step Levenberg–Marquardt refinement (0.019329). The selected hybrid scored test rel_l2 = 0.0184, R² = 0.997 and RMSE = 0.024, about 2.7× below the 0.05 success threshold. Test error is slightly lower than validation, so selection did not overfit. All four registered predictions were supported and none were contradicted. However, those predictions were loose upper bounds (< 0.02–0.0215), so they confirm adequacy rather than discriminating between mechanisms. The methods differ only in the fifth decimal place. The most likely reading, which is inferred rather than measured, is that LM refinement converges to the same likelihood optimum regardless of where it starts. On that reading the learned initializer adds speed or robustness, not accuracy, and the shared ~0.0193 level reflects noise and identifiability rather than optimizer quality. No seed variance was reported, so the ranking among the three is not meaningful.

No hypothesis was empirically falsified. The only clear negative result is that purely data-driven regression (extra-trees on sqrt+PCA16 features, 0.108) misses the threshold by about 2×. Hand-crafted reduced features plus trees do not capture the inverse map at this sample size. The more interesting learned-map hypotheses were never tested:
- The amortized MLP with noise-matched augmentation and the learned-init + log-LM proposal were both blocked by the per-experiment runtime cap (estimated 160 s and 129 s), not by evidence.
- The augmentation ablation, random-library control, and three-arm MLE/MMSE test were deferred for missing comparison arms, budget, or redundancy.

Several questions therefore remain open:
- Whether a forward-model-free network can reach ≤ 0.05.
- Whether ~0.0193 is actually at the Cramér–Rao floor. The notes cite a ~1.7× gap for learned maps, but the CRB is not reported alongside these results.
- Whether a posterior-mean (MMSE) estimator would beat the MLE mode on rel_l2.
- How the error is distributed across the four conductivity segments.

Next experiments:
- **CRB benchmark:** Compute per-sample Cramér–Rao bounds from the solver Jacobian and report the efficiency ratio of physics-MLE. This determines whether any remaining headroom exists.
- **Seed and bootstrap intervals:** Run 3+ seeds or a paired bootstrap on validation for physics-MLE vs hybrid. This tells us whether the 6×10⁻⁶ differences mean anything and whether the initializer reduces iterations or failure rate.
- **Budget-fitted amortized MLP:** Rerun the MLP with fewer epochs or a smaller width so it fits under the runtime cap, then measure rel_l2 against simulation count. This finally tests the variance-limited vs approximation-limited question.
- **MMSE vs MLE:** Compare a Laplace or importance-sampled posterior mean against the LM mode. Stratify by distance to the prior bounds and by segment, where the MAP estimate is most likely biased.
- **Per-segment error breakdown:** Break error down by segment and sensor proximity to identify the least-identifiable conductivity. This would guide sensor placement or a change in the noise model.

_Automatic summary:_ 26 proposal versions over 3 research rounds; 9 methods advanced; 4 predictions supported, 0 contradicted. Final: ['hybrid[extra_trees/raw+logy+aug2->LM15]'] with test rel_l2 0.018418.
