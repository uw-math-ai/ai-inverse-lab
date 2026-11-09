# Research report (V2): heat_v2_full_s1

Run `heat_v2_full_s1_20261109_002423` · 2026-11-09 00:37:10 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 62.3 (max 1.08e+03); Cramér–Rao relative std per parameter: k1 0.0215, k2 0.0058, k3 0.0052, k4 0.0174; implied rel_l2 floor ≈ **0.01851**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `custom-solver(fwd)`
- Validation rel_l2 (selection estimate): **0.018944** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018301**, r2 = **0.997**, rmse = **0.024081** · bootstrap 95% CI for rel_l2: [0.01731, 0.01945]
- Per-target test rel_l2: k1 0.0268, k2 0.0061, k3 0.0051, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.12275 (vs 0.018301 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051361 (vs 0.018301 in distribution)
- Inference latency: 22.6 ms/sample; fit 3.56 s; exported model reload check: max |Δ| = 0.0214
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s1\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_full_s1")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics-custom** `custom-solver(fwd)` rel_l2=0.018944; **physics** `physics-MMSE[log,ms=4,it=40]` rel_l2=0.019059; **hybrid** `hybrid[torch_mlp/raw+pca48+logy+aug1->LM6]` rel_l2=0.019325; **data-driven** `kernel_ridge/raw+logy+aug1` rel_l2=0.027659
- SUPPORTED — E-01 *Explicit IS posterior with prior Jacobian, mirrored-converged second start, and a train-selected loss-matched estimator…*: val rel_l2 = 0.018944 vs predicted < 0.021
- SUPPORTED — E-02 *Paired MMSE-vs-MLE contrast: bounded LM in log k (ms=4, it=40) with an importance-sampled posterior mean, matched to P-…*: val rel_l2 = 0.019059 vs predicted < 0.022
- SUPPORTED — E-03 *Learned MLP initializer + 6 LM steps matches multistart LM accuracy at a fraction of the per-sample forward-evaluation …*: val rel_l2 = 0.019342 vs predicted < 0.022
- SUPPORTED — E-05 *Box-bounded multistart LM-MLE in log-space is noise-limited: full-val rel_l2 within ~1.35x of the CRB floor, with no mi…*: val rel_l2 = 0.019335 vs predicted < 0.025
- SUPPORTED — E-06 *Log-temperature kernel ridge: a forward-free amortized inverse reaches rel_l2 < 0.05 at full training size (aug=1, shar…*: val rel_l2 = 0.031411 vs predicted < 0.05
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01851

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:22914204` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Theorem 7. http://arxiv.org/abs/1003.5004v1 — cited by ['P-001', 'P-002', 'P-009', 'P-010', 'P-016', 'P-017']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-001', 'P-009']; researcher claim ['no', 'partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet
- `ts:thm:19464924` M. S. Aronna, J. M. Machado (2021). "The shooting algorithm for partially control-affine problems with application to an SIRS epidemiological model". Proposition 6.2 (\cite{bonnans2006numerical,fletcher2013practical}). http://arxiv.org/abs/2103.06339v1 — cited by ['P-002']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-003', 'P-007', 'P-011', 'P-018']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:23749594` Manuel Marschall, Gerd Wübbeler, Franko Schmähling et al. (2022). "Generative models and Bayesian inversion using Laplace approximation". Theorem 8. http://arxiv.org/abs/2203.07755v2 — cited by ['P-003', 'P-011']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22198969` Naofumi Honda, Joyce McLaughlin, Gen Nakamura (2013). "Conditional Stability for Single Interior Measurement". Theorem 1.3. http://arxiv.org/abs/1306.4433v2 — cited by ['P-004', 'P-012', 'P-019']; researcher claim ['no', 'unknown']; harness screen **unknown**
- `ts:thm:26768883` Jiabao He, S. Joe Qin, Håkan Hjalmarsson (2025). "Bridging the Prediction Error Method and Subspace Identification: A Weighted Null Space Fitting Method". Theorem 5.3. http://arxiv.org/abs/2510.02529v1 — cited by ['P-005', 'P-013', 'P-021']; researcher claim ['no', 'partial']; harness screen **unknown**
- `ts:thm:26360863` Christian Kanzow, Theresa Lechner (2022). "Efficient Regularized Proximal Quasi-Newton Methods for Large-Scale Nonconvex Composite Optimization Problems". Proposition. http://arxiv.org/abs/2210.07644v1 — cited by ['P-010', 'P-015', 'P-017', 'P-018', 'P-023']; researcher claim ['no', 'partial', 'unknown']; harness screen **unknown**
- `ts:thm:19420616` François Monard, Richard Nickl, Gabriel P. Paternain (2021). "Statistical guarantees for Bayesian uncertainty quantification in non-linear inverse problems with Gaussian process priors". Lemma 4.6. http://arxiv.org/abs/2007.15892v2 — cited by ['P-016']; researcher claim ['no']; harness screen **unknown**
- `ts:thm:24487370` Haobo Zhang, Yicheng Li, Weihao Lu et al. (2023). "On the Optimality of Misspecified Kernel Ridge Regression". Theorem 3.1. http://arxiv.org/abs/2305.07241v1 — cited by ['P-022']; researcher claim ['partial']; harness screen **unknown**

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-04 *Pure learned inverse map (torch_mlp, PCA-32, log targets, noise augmentation) reaches rel_l2 < 0.05 from the 3000 provi…*: inconclusive — method did not produce a full-fidelity validation result
- P-023 *KRR initializer + 6 LM steps reaches the CRB floor with ~1/25 of the forward evals of multistart LM, and beats a centre…* (conjectured): proposed but not executed
- P-020 *Sample-limited, not architecture-limited: an MLP trained on 3000 provided + 12000 freshly simulated samples with per-ep…* (conjectured): proposed but not executed
- P-024 *Control: box-centre-start 6-iteration log-k LM. Does the learned initializer matter under a tight iteration budget?* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-mathematical — P-002 *Learned amortised initializer plus short LM refinement beats both pure regression and cold-start LM…*: Negative result by evidence. Cold-start multistart LM (P-001 pilot, rel_l2 0.0184, chi2/dof 0.94) already finds the correct basin and reaches the information floor, so the accuracy hypothesis (that the initializer impro…
- ABANDONED in round 2 by researcher-experimental — P-006 *Controlled comparison: learned initialiser + short LM refinement vs. MLE; posterior-mean vs. mode t…*: Negative result recorded: the pilot (resu-00053) shows cold-start MLE at ~1.03x the CRB floor with calibrated Gauss-Newton coverage. This leaves no room for a >=10% gain from initialisation plus posterior mean. The expe…
- ABANDONED in round 2 by researcher-exploratory — P-007 *Learned initializer plus LM refinement with posterior-mean output, since rel_l2 rewards MMSE rather…*: Duplicate of P-006 and over the 120 s per-experiment limit. Its unique content (the mean-vs-mode ablation and the k1/k4 near-bound stratified prediction) is folded into P-006/P-003 as a reported estimator ablation from …
- PRUNED — idea-009 `svm/log+pca8`: projected full-fidelity cost 203s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Explicit IS posterior with prior Jacobian, mirrored-converg…): challenge: The H2 Jacobian sub-claim is statistically undecidable. Omitting exp(Σz) shifts the posterior mean in z by about σ_z² ≈ 4e-4. That bias adds in quadrature to a ~0.018 error, changing rel_l2 by about 4e-6. The…
- E-01 (Explicit IS posterior with prior Jacobian, mirrored-converg…): challenge: Selection on 150 train samples among estimators differing by about 1-3% is noisy. Report the selection margin and paired CI on those 150 samples, otherwise 'selected estimator' is effectively random between m…
- E-01 (Explicit IS posterior with prior Jacobian, mirrored-converg…): challenge: The mirrored second start costs about 100 evals/sample to test a basin that is physically implausible (see P-021). A 100-sample misfit probe answers the same question for about 1% of the cost.
- E-01 (Explicit IS posterior with prior Jacobian, mirrored-converg…): challenge: The rule estimate (30 s) and your own (40-60 s) disagree. Chunking protects memory but not wall time.
- E-02 (Paired MMSE-vs-MLE contrast: bounded LM in log k (ms=4, it=…): challenge: The near-face arithmetic is wrong, and it matters for claim (b). With 4 i.i.d. U[0.5,2] components, P(some k_i within 0.1 of a face) = 1-(1-0.2/1.5)^4 ≈ 0.43, not 5-10%. The stratum 'min distance < 0.1' there…
- E-02 (Paired MMSE-vs-MLE contrast: bounded LM in log k (ms=4, it=…): challenge: The stratum should be defined in posterior-sigma units, not absolute k. Per-component CRB std scales with k (about 2.2% for k1, about 0.6% for k2/k3), so 0.1 is about 9 sigma at k2≈0.5 but about 1 sigma at k1…
- E-02 (Paired MMSE-vs-MLE contrast: bounded LM in log k (ms=4, it=…): challenge: The CRB is not a floor for this comparison. It bounds unbiased estimators. Both the box-clipped MLE and the posterior mean are biased near faces, and the MMSE can go below the CRB there. The relevant bound is…
- E-02 (Paired MMSE-vs-MLE contrast: bounded LM in log k (ms=4, it=…): challenge: The posterior mean minimises expected squared error, not E//k_hat-k/////k//. Under a per-sample-normalised metric it is not the Bayes estimator, so a positive gain is evidence for truncation shrinkage, not fo…
- E-03 (Learned MLP initializer + 6 LM steps matches multistart LM …): challenge: Claim 2 (≤1/5 of P-013 evals) is on a knife edge. Pilot counts give P-005 = 20657/60 ≈ 344 evals/sample and KRR+6LM = 3881/60 ≈ 65/sample, a ratio of 1/5.3. Any extra rejected LM trial steps from a worse MLP …
- E-03 (Learned MLP initializer + 6 LM steps matches multistart LM …): challenge: This is near-duplicate of P-023 (and P-015): the same 6-iteration hybrid with a different initializer. P-024 (4 s) decides whether either initializer matters. If centre-start-6 matches, both hybrid initialize…
- E-03 (Learned MLP initializer + 6 LM steps matches multistart LM …): challenge: The rule estimate of 84 s is about 3x your own estimate and about 21% of the total budget, for a claim that mostly concerns cost.
- E-03 (Learned MLP initializer + 6 LM steps matches multistart LM …): challenge: The MLP architecture (512-256-128, PCA-48, aug1) differs from P-019 (512-512, PCA-32, aug2). Refinement value therefore cannot be attributed to a single learned map.
- E-04 (Pure learned inverse map (torch_mlp, PCA-32, log targets, n…): challenge: The rule estimate is 120 s, which appears to equal the per-experiment cap (P-021 cites a 120 s limit). There is a real timeout risk, and it is 30% of the total 400 s budget for a forward-free baseline.
- E-04 (Pure learned inverse map (torch_mlp, PCA-32, log targets, n…): challenge: Noise augmentation adds N(0,σ²) to already-noisy observations. Augmented copies therefore carry noise std √2·σ (aug at scale 1), not σ. The model trains on a noise mixture that does not match val. This is min…
- E-04 (Pure learned inverse map (torch_mlp, PCA-32, log targets, n…): challenge: PCA-32 is still unjustified until the noiseless spectrum is measured. Standard scaling before PCA amplifies near-zero early-time end-sensor channels that are pure noise.
- E-04 (Pure learned inverse map (torch_mlp, PCA-32, log targets, n…): challenge: P-020 names P-012 as its provided-data comparator. If P-019 is the run that executes, P-020's 20% contrast is confounded by PCA, epochs (200 vs 60), batch size and the noise-augmentation scheme.
- E-05 (Box-bounded multistart LM-MLE in log-space is noise-limited…): challenge: The method config (log, ms=4, it=40, mode) is identical to the P-005 pilot and to the P-013 arm that P-016/P-018 pair against. Run it once and attach P-021's extra diagnostics to that single artifact.
- E-05 (Box-bounded multistart LM-MLE in log-space is noise-limited…): challenge: 'Within 1.35x of the CRB floor' compares a biased (box-clipped) estimator to an unbiased-estimator bound, using a median from 32 draws. Also, mean-of-norms is below RMS, so the comparable CRB value is about 0…
- E-05 (Box-bounded multistart LM-MLE in log-space is noise-limited…): challenge: The mirrored-basin hypothesis is physically weak. Sensors are labelled (index 5 vs 35), so the mirrored profile (k4,k3,k2,k1) maps sensor-1 data onto approximately sensor-4 positions. Its misfit is O(signal) …
- E-05 (Box-bounded multistart LM-MLE in log-space is noise-limited…): counterexample: Box truncation shrinks error for near-face samples. The clipped MLE can then fall below the per-sample CRB, which would make the '1.35x floor' ratio non-monotone with estimator quality.
- E-06 (Log-temperature kernel ridge: a forward-free amortized inve…): challenge: Log features at sensors 5 and 35 with early-time readings near zero amount to the log of noise-dominated values with an unknown clip floor. Standard scaling then gives these channels unit weight in the RBF di…
- E-06 (Log-temperature kernel ridge: a forward-free amortized inve…): challenge: Reaching <0.05 from 0.0636 requires a ≥22% reduction. The pilot SE (about ±0.01 on n=60) means the starting point is uncertain by a comparable amount. The prediction is plausible but not strongly supported.
- E-06 (Log-temperature kernel ridge: a forward-free amortized inve…): challenge: Fixed gamma_mult=0.5 and alpha=1e-4 were chosen at n=300. The optimal bandwidth shrinks with n, so a fixed bandwidth biases toward underfitting at n=3000.
- E-06 (Log-temperature kernel ridge: a forward-free amortized inve…): question: the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- Unresolved disagreement ['P-016', 'P-017']: Whether the harness posterior mean applies the log-space prior Jacobian. P-017's explicit estimators can settle this by per-sample comparison on shared indices.
- Unresolved disagreement ['P-021', 'P-017']: Existence of a mirrored basin. Sensor labelling makes it implausible; a 100-forward misfit probe settles it more cheaply than a second start.
- Unresolved disagreement ['P-023', 'P-024']: Whether the learned initializer matters at 6 iterations or the log-k problem is globally easy. P-024 decides.
- Unresolved disagreement ['P-019', 'P-020']: Sample-limited vs noise-model-limited amortised map. This is confounded until an n_sim=0 arm with an identical pipeline is run.
- Test-set sampling uncertainty: bootstrap 95% CI [0.01731, 0.01945] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | The pilot evidence outweighs the argument here. On 60 val samples, physics-MMSE[log,ms=4,it=30] reached rel_l… |
| 2 | researcher-mathematical | abandon | P-002 | - | llm | I accept all four critic points: the budget overrun (130.6 s), the noise_scale ambiguity (0.001 may be a 1e-6… |
| 2 | researcher-mathematical | propose | - | P-010 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-011 | heuristic-fallback | Revised: reduced solver cost as requested |
| 2 | researcher-ml | revise | P-004 | P-012 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-experimental | revise | P-005 | P-013 | llm | The pilot outweighs the arguments here. On 60 val samples, MLE[log, ms=4, it=40] gave rel_l2 = 0.0190, about … |
| 2 | researcher-experimental | abandon | P-006 | - | llm | I accept all three blocking points.  (1) The estimated 134 s exceeds the 120 s per-experiment limit.  (2) The… |
| 2 | researcher-exploratory | abandon | P-007 | - | llm | (1) Duplicate of P-006: accepted. Apart from MLP width and epochs, P-007 is the same pipeline. Its one distin… |
| 2 | researcher-exploratory | revise | P-008 | P-014 | llm | Pilot evidence comes first. At 10% of the training data (300 samples) and 60 val samples, kernel ridge on log… |
| 2 | researcher-exploratory | propose | - | P-015 | llm |  |
| 3 | researcher-mathematical | revise | P-009 | P-016 | llm | (1) Uninformative headline. Agreed. The primary claim is now the paired MMSE-minus-MLE contrast against P-013… |
| 3 | researcher-mathematical | revise | P-010 | P-017 | llm | (1) Metric definition. The solver no longer hard-codes the ratio estimator. It computes four estimators on th… |
| 3 | researcher-ml | revise | P-011 | P-018 | llm | Every critic point is accepted, and the arm is reframed as a forward-evaluation budget claim. (1) Dominance b… |
| 3 | researcher-ml | revise | P-012 | P-019 | llm | Accepted with revisions. (1) PCA inconsistency: the unmeasured 'rank < 48' assumption is removed. pca=32 is k… |
| 3 | researcher-ml | propose | - | P-020 | llm |  |
| 3 | researcher-experimental | revise | P-013 | P-021 | heuristic-fallback | Revised: incorporated critic assumptions |
| 3 | researcher-exploratory | revise | P-014 | P-022 | llm | Point-by-point. (1) Config mismatch (blocking): the critic is right. The P-008 pilot ran with noise_aug=2, an… |
| 3 | researcher-exploratory | revise | P-015 | P-023 | llm | The pilot evidence comes first. On 60 val samples (n_train=300) this exact arm gave 0.0190, with 3881 forward… |
| 3 | researcher-exploratory | propose | - | P-024 | llm |  |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.756 | fdbk-00044 |
| 2 | 7 | 0.807 | fdbk-00080 |
| 3 | 9 | 0.856 | fdbk-00115 |

### Idea evolution

**P-001 — Bounded LM posterior-mean inversion through the exact forward model approaches the Cramér–Rao floor** (researcher-mathematical, family physics)
- v1 `P-001` r1 [propose, llm] `physics-MMSE[log,ms=4,it=30]` status literature-supported. *H:* Per-sample bounded Levenberg–Marquardt on F(k), started from several points in log k and followed by an importance-sampled posterior mean under the uniform box prior, will reach val rel_l2 < 0.035. That is under twice the CRB floor of 0.0185 and well below th…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If misfits are at noise level but the error is large, the estimator or prior handling is at fault, e.g. a poor Laplace approximation near the bounds; switch to…
  - critic r1 (quality 0.916): The 0.0185 CRB figure is a local, unbiased-estimator floor taken at median Fisher information. It is not a hard lower bound for a posterior-mean estimator unde… · The posterior mean is computed in log k, but the prior is uniform in k. In log coordinates the prior density is proportional to k. If the importance weights om… · Replace 'there is no model error' with a tested claim. On val, check that //F(k_true) - y//^2 / (m sigma^2) is approximately 1 before any inversion. · Grid indices 0..39 place the centre at 19.5. Sensors 5 and 35 are not exact mirrors (the mirror of 5 is 34). The mirrored conductivity profile therefore yields… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `physics-MMSE[log,ms=4,it=40]` status literature-supported. *H:* On the full val set, bounded LM in log k with 4 starts and 40 iterations, followed by an importance-sampled posterior mean, achieves rel_l2 < 0.025. More than 95% of samples will have normalized misfit //F(k_hat)-y//^2/(m σ^2) < 1.5. Against the identically c…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If misfits are at noise level but rel_l2 is ≥ 0.025, the 60-sample pilot was optimistic and the estimator is information-limited near 0.02. Compare against the…
  - response to critic: The pilot evidence outweighs the argument here. On 60 val samples, physics-MMSE[log,ms=4,it=30] reached rel_l2 = 0.0184, with chi2/dof = 0.944. The fit report gives residual std at truth = 0.000995 against a nominal 0.001 (200 samples chec…
  - critic r2 (quality 0.925): The headline prediction (rel_l2 < 0.025) is nearly guaranteed by the pilot (0.0184 on n=60), so it carries little information. The informative claim is the pai… · It is unclear what the harness 'posterior_mean' estimates. It could be E[k], E[log k] exponentiated, or the mean in z with or without the exp(sum z) prior Jaco… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If posterior_mean is the mean in z without the Jacobian, it is E[log k] mapped back. That shrinks toward small k and could make MMSE worse than MLE for samples… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-016` r3 [revise, llm] `physics-MMSE[log,ms=4,it=40]` status literature-supported. *H:* At settings identical to P-013 (log transform, 4 starts, 40 iterations), the harness posterior-mean estimator improves full-val rel_l2 over the mode by 0-3%. The 95% paired bootstrap CI of (MLE error minus MMSE error) will be non-negative. The gain per sample…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If the CI is negative, the harness posterior mean probably averages in z without the Jacobian; cross-check against P-010's 'mean_noJ' arm. If the gain is not c…
  - response to critic: (1) Uninformative headline. Agreed. The primary claim is now the paired MMSE-minus-MLE contrast against P-013 at identical settings (ms=4, it=40, same val samples). The absolute threshold stays only as a sanity check. On the confound: the …
  - critic r3 (quality 0.916): The near-face arithmetic is wrong, and it matters for claim (b). With 4 i.i.d. U[0.5,2] components, P(some k_i within 0.1 of a face) = 1-(1-0.2/1.5)^4 ≈ 0.43, … · The stratum should be defined in posterior-sigma units, not absolute k. Per-component CRB std scales with k (about 2.2% for k1, about 0.6% for k2/k3), so 0.1 i… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Pilot P-001/P-009 vs P-005: 0.018396 vs 0.019024 is a 3.3% gain, already at the top of the stated 0-3% range on n=60. The '0-3%' interval may be refuted on the… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-002 — Learned amortised initializer plus short LM refinement beats both pure regression and cold-start LM at lower cost** (researcher-mathematical, family hybrid)
- v1 `P-002` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM10]` status conjectured. *H:* A noise-augmented MLP maps sensor traces to k with val rel_l2 of about 0.05–0.08. That is limited by finite training data (3000 samples) and amortisation error, not by information content. Using it to initialise 10 LM iterations through F will cut val rel_l2 …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.4, "expect": "better"}`; if it fails: If refinement gives a small gain, the MLP is already near-efficient or LM is stuck. Compare against the proposal-1 misfit to tell which. If the hybrid is worse…
  - critic r1 (quality 0.71): Estimated runtime of 130.6 s exceeds the 120 s per-experiment limit. Reduce posterior_samples and/or extra_starts. · Clarify noise_scale semantics. 0.001 may disable augmentation. · Estimated at 130.6 s, which exceeds the 120 s per-experiment limit the proposers themselves cite. · noise_scale=0.001 here but 1.0 in P-006, P-007 and P-008. If the harness treats noise_scale as a multiplier of sigma, the augmentation noise here is 1e-6 and d… · The MLP error of about 5–8% only needs to lie within the LM basin. If cold-start LM already converges globally (to be tested in P-005), the initializer's value…

**P-003 — Learned initializer + few LM steps through the forward model reaches near the Cramér–Rao floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+pca48+logy->LM8]` status conjectured. *H:* A torch_mlp trained on simulated data, refined per sample by 8 bounded Levenberg–Marquardt iterations in log-k space, will reach val rel_l2 <= 0.03. That is within about 1.6x of the CRB floor of 0.0185. A posterior-mean estimator will not be worse than the mo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If refinement does not help, either the forward model does not match the data simulator, or the noise-limited floor is not reachable with this budget. If the h…
  - critic r1 (quality 0.898): PCA is applied after standard scaling. Early-time readings at the end sensors carry almost no signal, and unit-variance scaling inflates their noise to the sam… · With 8 LM iterations, the 30% improvement criterion depends on convergence. Report the fraction of samples whose step norm is still above tolerance at iteratio… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-011` r2 [revise, heuristic-fallback] `hybrid[torch_mlp/raw+pca48+logy->LM10]` status conjectured. *H:* A torch_mlp trained on simulated data, refined per sample by 8 bounded Levenberg–Marquardt iterations in log-k space, will reach val rel_l2 <= 0.03. That is within about 1.6x of the CRB floor of 0.0185. A posterior-mean estimator will not be worse than the mo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If refinement does not help, either the forward model does not match the data simulator, or the noise-limited floor is not reachable with this budget. If the h…
  - response to critic: Revised: reduced solver cost as requested
  - critic r2 (quality 0.71): Hypothesis (8 iters) and method (iters=10) disagree; fix before running. · The rationale is undercut by the rule estimates. The hybrid costs about 45.3 s against 17.0 s for multistart LM (P-013), which already reaches about 1.03x CRB.… · Internal inconsistency: the hypothesis says 8 LM iterations, but the method has iters=10. · Reframe the hypothesis as a compute claim: the hybrid matches P-013's rel_l2 within the paired CI while using fewer forward evaluations. Otherwise, drop it. · If P-013 already achieves 0.019 at 17 s, a hybrid achieving 0.019 at 45 s supports no claim about the necessity or efficiency of learned starts.
- v3 `P-018` r3 [revise, llm] `hybrid[torch_mlp/raw+pca48+logy+aug1->LM6]` status conjectured. *H:* Setup: a torch_mlp (log-k targets, PCA-48, noise_aug=1) supplies one start per sample, refined by 6 bounded LM iterations in log-k space (mode estimator). Claim 1: val rel_l2 is below 0.022 and within the paired-bootstrap 95% CI of P-013 (4-start, up to 40-it…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If (a) or (b) fails while (d) holds, 6 iterations are not enough even from a good start. The iteration sweep then locates the knee. If (d) fails, the MLP start…
  - response to critic: Every critic point is accepted, and the arm is reframed as a forward-evaluation budget claim. (1) Dominance by P-013: the hypothesis no longer says a learned start is needed for accuracy. It now says that with a learned start, 6 LM iterati…
  - critic r3 (quality 0.865): Claim 2 (≤1/5 of P-013 evals) is on a knife edge. Pilot counts give P-005 = 20657/60 ≈ 344 evals/sample and KRR+6LM = 3881/60 ≈ 65/sample, a ratio of 1/5.3. An… · This is near-duplicate of P-023 (and P-015): the same 6-iteration hybrid with a different initializer. P-024 (4 s) decides whether either initializer matters. … · Use the P-019 trained MLP as the initializer instead of training a separate one. This makes P-018 minus P-019 a clean 'value of 6 LM steps on the same map' con… · If the MLP init error is about 4% (similar to KRR at about 6%), the KRR+6LM pilot already shows convergence to 0.019. P-018 then adds no information beyond P-0… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-004 — Pure learned inverse map with log-conductivity targets and PCA compression meets the 0.05 success threshold without the…** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+pca32+logy` status conjectured. *H:* The map from noisy sensor time series to log k is smooth and low-dimensional. A torch_mlp on standardised, PCA-compressed raw features with log targets, trained on 3000 samples, will reach val rel_l2 < 0.05. The same configuration without PCA (pca=0) should b…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: The learned map's error is dominated by approximation bias, not noise. That would show physics refinement (proposal 1) is necessary rather than optional. It wo…
  - critic r1 (quality 0.845): The PCA secondary claim has the same scaling problem as P-003. After standard scaling, noise variance in low-signal channels is inflated, so PCA ranks componen… · The 0.05 target requires an amortized map within 2.7x the CRB, learned from 3000 samples in a 4-D box. That is plausible for interior points but doubtful near … · Instead of 'signal subspace < 48', test the effective signal rank directly. Compute the SVD of the noiseless Jacobian stack, or the PCA spectrum of F(k) evalua… · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-012` r2 [revise, heuristic-fallback] `torch_mlp/raw+pca32+logy` status conjectured. *H:* The map from noisy sensor time series to log k is smooth and low-dimensional. A torch_mlp on standardised, PCA-compressed raw features with log targets, trained on 3000 samples, will reach val rel_l2 < 0.05. The same configuration without PCA (pca=0) should b…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: The learned map's error is dominated by approximation bias, not noise. That would show physics refinement (proposal 1) is necessary rather than optional. It wo…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.845): Inconsistency: the assumption says PCA-48 retains the signal, but the method uses pca=32. State the rank evidence for 32. · Standard scaling before PCA equalises per-channel variance. Near-zero early readings at the end sensors are almost pure noise, and after scaling they get weigh… · Replace 'signal subspace < 48' with a measured effective rank: the PCA spectrum of noiseless F(k) over the prior, the index where eigenvalues drop below σ²·n_o… · If the noiseless spectrum shows only about 8-10 components above noise, pca=32 and pca=0 may perform similarly. The MLP then learns to ignore noise directions,… · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v3 `P-019` r3 [revise, llm] `torch_mlp/raw+pca32+logy+aug2` status conjectured. *H:* A torch_mlp on standardised, PCA-32-compressed raw features with log-k targets and 2× noise augmentation at the true sigma reaches val rel_l2 < 0.05 when trained on the 3000 provided samples. We predict about 0.035–0.045, i.e. 2–2.5× the CRB floor. Error shou…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 ≥ 0.05 and the new extra-simulated-data arm succeeds, the amortised map is sample-limited, not architecture-limited. If both fail, the MLP is lim…
  - response to critic: Accepted with revisions. (1) PCA inconsistency: the unmeasured 'rank < 48' assumption is removed. pca=32 is kept and justified as a measurable conjecture. The effective rank of the noiseless observation spectrum over prior draws is request…
  - critic r3 (quality 0.845): The rule estimate is 120 s, which appears to equal the per-experiment cap (P-021 cites a 120 s limit). There is a real timeout risk, and it is 30% of the total… · Noise augmentation adds N(0,σ²) to already-noisy observations. Augmented copies therefore carry noise std √2·σ (aug at scale 1), not σ. The model trains on a n… · Reduce epochs to about 80 with early stopping, or batch_size 256, to bring cost to about 40 s. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Prior-mean-initialised LM in log-space reaches near the CRB floor: a likelihood-limited, not model-limited, regime** (researcher-experimental, family physics)
- v1 `P-005` r1 [propose, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* With sigma=0.001 Gaussian noise and Fisher condition ~62, the per-sample likelihood is sharply peaked and near-Gaussian in log k, so a bounded Levenberg-Marquardt MLE/MAP in log-space with a few multistarts attains val rel_l2 within ~1.5x of the CRB floor (0.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If heavy-tailed errors: multimodality/poor starts -> motivates hybrid learned initialiser. If uniformly elevated error: forward/data mismatch or noise model mi…
  - critic r1 (quality 0.91): The method is essentially P-001 with estimator=mode. Its value is as the MLE control for the posterior-mean arms (P-001, P-006, P-007). · 'Beating every data-driven model' is a portfolio-level comparison, not a test of this arm alone. · The CFL condition should be verified, not assumed. Check numerically that F(2,2,2,2) shows no oscillation or growth. · A mirrored conductivity profile (k4, k3, k2, k1) is a candidate spurious minimum. See P-001. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-013` r2 [revise, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* With exact forward-model identity and sigma=0.001, bounded LM-MLE in log k with 4 multistarts gives full-val rel_l2 < 0.025, i.e. within ~1.35x of the CRB floor of 0.0185. Its errors are approximately Gaussian with the Gauss-Newton covariance: 1-sigma coverag…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If errors are heavy-tailed or mirrored: multimodality is real, so a learned initializer (P-003) adds accuracy, not just speed. If error is uniformly elevated w…
  - response to critic: The pilot outweighs the arguments here. On 60 val samples, MLE[log, ms=4, it=40] gave rel_l2 = 0.0190, about 1.03x the CRB floor of 0.0185. The run took 3.2 s. The harness fit report at truth gives residual std 0.000995 against nominal 0.0…
  - critic r2 (quality 0.905): rel_l2 < 0.025 is a weak bar given the pilot. The sharp, informative tests are the coverage band and the tail/failure rates, so keep those primary. · The 'mirrored basin' criterion is ill-posed for near-symmetric truths. If k ≈ reverse(k), the estimate is trivially close to both. Restrict the mirror test to … · Treat the 'within 1.35x of CRB' claim as a ratio to the per-sample expected Gaussian error under the GN covariance, not the 32-draw median diagnostic. · A true k = (1, 1.5, 1.5, 1) has its mirror equal to itself, so it would trivially satisfy 'closer to mirror' half the time under noise. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v3 `P-021` r3 [revise, heuristic-fallback] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* With exact forward-model identity and sigma=0.001, bounded LM-MLE in log k with 4 multistarts gives full-val rel_l2 < 0.025, i.e. within ~1.35x of the CRB floor of 0.0185. Its errors are approximately Gaussian with the Gauss-Newton covariance: 1-sigma coverag…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If errors are heavy-tailed or mirrored: multimodality is real, so a learned initializer (P-003) adds accuracy, not just speed. If error is uniformly elevated w…
  - response to critic: Revised: incorporated critic assumptions
  - critic r3 (quality 0.845): The method config (log, ms=4, it=40, mode) is identical to the P-005 pilot and to the P-013 arm that P-016/P-018 pair against. Run it once and attach P-021's e… · 'Within 1.35x of the CRB floor' compares a biased (box-clipped) estimator to an unbiased-estimator bound, using a median from 32 draws. Also, mean-of-norms is … · Express the CRB ratio per sample: rel error / sqrt(tr(k-normalised GN covariance)), with E[chi_4] scaling, computed at the estimate. · Box truncation shrinks error for near-face samples. The clipped MLE can then fall below the per-sample CRB, which would make the '1.35x floor' ratio non-monoto… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-006 — Controlled comparison: learned initialiser + short LM refinement vs. MLE; posterior-mean vs. mode tests whether box-pri…** (researcher-experimental, family hybrid)
- v1 `P-006` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status conjectured. *H:* A torch_mlp initialiser (trained on simulated data with noise augmentation) followed by 15 LM iterations in log-space and MMSE posterior-mean estimation lowers val rel_l2 relative to pure MLE by >=10%, mainly by reducing errors for k near the box bounds (0.5 …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.1, "expect": "better"}`; if it fails: If no gain over MLE: residual error is noise-limited and near-efficient; further effort should go to estimator choice only marginally. If worse than MLE: Lapla…
  - critic r1 (quality 0.599): Estimated runtime of 134 s exceeds the 120 s per-experiment limit. · Confounded design: the initializer and estimator change together, so the stated competing explanations cannot be discriminated. · Estimated at 134 s, which exceeds the 120 s per-experiment limit. · The predicted ≥10% gain over MLE from posterior-mean shrinkage is quantitatively implausible. With a CRB std of about 2% of k, only around 5–7% of draws per en… · Treat the estimator effect and the initialization effect as separable factors in a 2x2 design: {cold, MLP} x {mode, mean}.

**P-007 — Learned initializer plus LM refinement with posterior-mean output, since rel_l2 rewards MMSE rather than MLE** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status conjectured. *H:* A hybrid estimator gets val rel_l2 <= 0.03. It starts from an ML regressor, runs bounded LM refinement in log-k space, then reports the box-prior posterior mean instead of the mode. The prediction is that it beats the same pipeline with estimator=mode by at l…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If mode does as well as mean, the posterior is effectively Gaussian and interior, and the metric is limited by noise (CRB). If both are above 0.03, either the …
  - critic r1 (quality 0.417): Estimated runtime of 134 s exceeds the 120 s per-experiment limit. Merge or reduce posterior_samples. · The method is nearly identical to P-006. Only the MLP width and epoch count differ, while the LM and estimator settings are identical. The distinct contributio… · Estimated at 134 s, which exceeds the 120 s per-experiment limit. · The Bayes-optimal estimator for per-sample relative L2 is a 1///k//^2-weighted posterior mean, not the plain posterior mean. The difference is small but goes i… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-008 — Log-temperature features as a heat-kernel (Varadhan-type) linearisation make a kernel method competitive without any fo…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `kernel_ridge/log+logy+aug2` status conjectured. *H:* A purely data-driven method reaches val rel_l2 < 0.05 with no forward-model calls at test time: RBF kernel ridge on log-transformed sensor temperatures, with noise augmentation. The log features should beat raw features by at least 20% relative at matched hyp…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If log does not beat raw, the heat-kernel linearisation intuition does not carry over to this finite-window, multi-segment, Dirichlet geometry, or the near-zer…
  - critic r1 (quality 0.751): With Dirichlet ends and a centred pulse, the end sensors (indices 5 and 35) read about 0 at early times. Additive noise at 1e-3 then makes many readings negati… · The Varadhan asymptotic is a short-time result on unbounded or smooth media. Over a finite window with Dirichlet ends, piecewise-constant k and only 4 point se… · Replace 'sensor temperatures stay positive' with a measured check: the fraction of readings below 3*sigma in the train features. · Take a reading with u = 0.0005 and noise of ±0.001. log(u) then ranges from undefined to about -6.9, so the transform turns small additive noise into enormous … · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-014` r2 [revise, llm] `kernel_ridge/log+logy+aug1` status conjectured. *H:* RBF kernel ridge on standard-scaled log sensor features, with log targets and one noise-augmented copy, reaches full val rel_l2 < 0.05 with zero forward calls. The pilot gave 0.064 at 10% of the training data, and error decreases with n. A matched features='r…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If it lands in 0.05-0.065, the amortized inverse is limited by sample density. Simulating more training data, or a hybrid LM refinement (see the new proposal),…
  - response to critic: Pilot evidence comes first. At 10% of the training data (300 samples) and 60 val samples, kernel ridge on log features with log targets reached rel_l2 = 0.0636 in 2 s, with no forward calls. That is already 5x better than the mean baseline…
  - critic r2 (quality 0.665): Reconcile noise_aug (1 vs the pilot's 2) or state the change explicitly. · Config mismatch: the pilot label is 'aug2', but the proposal sets noise_aug=1. Kernel size and fit change accordingly, so the pilot is not the same method. · alpha=1e-4 and gamma_mult=0.5 were set at n=300. The optimal bandwidth shrinks with sample density. A fixed gamma at n=3000(+aug) may underfit, so the n-scalin… · Treat noise_scale semantics (relative to σ or absolute) as unverified until confirmed in harness code. An absolute 1.0 would destroy the features. · If the log clip floor is 1e-6, early end-sensor channels become log of noise-dominated values with huge variance. After standard scaling they dominate RBF dist…
- v3 `P-022` r3 [revise, llm] `kernel_ridge/log+logy+aug1` status conjectured. *H:* Hypothesis: RBF kernel ridge on standard-scaled log sensor features reaches full val rel_l2 < 0.05 with zero forward calls. The setup is: log targets, one noise-augmented copy (noise_aug=1, changed explicitly from the pilot's 2), and gamma_mult=0.5, alpha=1e-…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A result in 0.05-0.065 means the amortized inverse is sample-density-limited at this budget. Physics refinement is then necessary in practice, and P-015 suppli…
  - response to critic: Point-by-point. (1) Config mismatch (blocking): the critic is right. The P-008 pilot ran with noise_aug=2, and this spec uses noise_aug=1. I am keeping noise_aug=1 on purpose, and the change is now stated explicitly. aug=1 at n=3000 gives …
  - critic r3 (quality 0.834): Log features at sensors 5 and 35 with early-time readings near zero amount to the log of noise-dominated values with an unknown clip floor. Standard scaling th… · Reaching <0.05 from 0.0636 requires a ≥22% reduction. The pilot SE (about ±0.01 on n=60) means the starting point is uncertain by a comparable amount. The pred… · Select gamma on a train-internal holdout (3-point grid at n=1500), not on val, to stay within the rules and the budget. · the physics floor (CRB rel_l2 ~0.01851) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-010 — Explicit Bayes estimator for per-sample relative loss: LM with mirrored start plus importance sampling using the correc…** (researcher-mathematical, family physics-custom)
- v1 `P-010` r2 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* This custom solver uses three explicit ingredients: (i) a mirrored second start z[:, ::-1] to probe the symmetric spurious basin; (ii) importance weights that include the uniform-in-k prior density ∝ exp(sum z) in log coordinates; (iii) the Bayes-optimal poin…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If it is worse than P-001, the harness posterior_mean already handles the prior correctly, or the 1.2x-inflated proposal with 96 samples has too low an ESS. Ra…
  - critic r2 (quality 0.755): Specify the first start; a symmetric start makes the mirror probe vacuous. · The Bayes estimator E[k///k//²]/E[1///k//²] is optimal only for per-sample squared relative loss E[//a-k//²///k//²]. If rel_l2 is a mean of unsquared per-sampl… · Importance weights must include the Laplace proposal density, the exp(sum z) prior Jacobian, and zero weight outside the box. The code is truncated in the summ… · Mirror the first converged solution, not the initial point. · If both starts are the box centre (1.25,...) and its reverse, they coincide. The 'mirrored start never wins' outcome would then wrongly be read as evidence of …
- v2 `P-017` r3 [revise, llm] `custom-solver(fwd)` status conjectured. *H:* (H1) On full val, the selected estimator reaches rel_l2 < 0.021. (H2) Among estimators on shared IS samples, ratio <= E[k] < mode, and the per-sample paired difference E[k] minus mode has a 95% CI containing only non-positive values. Omitting the exp(sum z) J…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.021}`; if it fails: If rel_l2 is 0.021 or worse while P-009 succeeds, the 1.2x-inflated 96-sample proposal is inadequate (check m.ess), or 20 iterations from the centre is too few…
  - response to critic: (1) Metric definition. The solver no longer hard-codes the ratio estimator. It computes four estimators on the same importance samples: mode, E[k] with the Jacobian, E[k] without it, and the ratio. fit() then picks one on the first 150 TRA…
  - critic r3 (quality 0.925): The H2 Jacobian sub-claim is statistically undecidable. Omitting exp(Σz) shifts the posterior mean in z by about σ_z² ≈ 4e-4. That bias adds in quadrature to a… · Selection on 150 train samples among estimators differing by about 1-3% is noisy. Report the selection margin and paired CI on those 150 samples, otherwise 'se… · Test the Jacobian effect deterministically. Mean over samples of (E[k] - E_noJ[k])/k should be ≈ σ_z² per component (positive). This is a same-sample estimator… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-015 — Forward-free kernel initializer plus 6 Gauss-Newton steps: cheapest route to near-CRB accuracy** (researcher-exploratory, family hybrid)
- v1 `P-015` r2 [propose, llm] `hybrid[kernel_ridge/log+logy+aug1->LM6]` status conjectured. *H:* Warm-starting bounded LM (log-k, mode, 6 iterations, no extra starts) from the P-008 kernel ridge predictor gives val rel_l2 < 0.025, within 1.35x of the CRB floor of 0.0185. It needs under a third of the LM iterations of the MLP-initialised hybrids, and its …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If the error is above 0.025 and residuals are at the noise level, the estimator is noise-limited and the CRB estimate is optimistic. Compare with oracle-start …
  - critic r2 (quality 0.845): 'Quadratic contraction' holds only for zero-residual problems. With σ=1e-3 noise, GN converges linearly to the MLE at a rate set by the residual curvature. It … · The compute claim is not supported. The rule estimate is 34.1 s vs 17.0 s for P-013, partly because test inference is included. Fewer LM iterations do not make… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If P-013 at 40 iterations and P-015 at 6 iterations both give 0.019, the result shows the problem is easy from any start. It does not show the kernel initializ… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?
- v2 `P-023` r3 [revise, llm] `hybrid[kernel_ridge/log+logy+aug1->LM6]` status conjectured. *H:* Warm-starting bounded LM (log-k, mode, 6 iterations, 1 start) from the P-014 kernel ridge predictor gives full val rel_l2 < 0.025. It does so with forward evals ≤ 1/10 of P-013 (multistart 4 x 40 iterations), and it is ≥20% better than an identical 6-iteratio…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.025}`; if it fails: If rel_l2 ≥ 0.025 with residuals at the noise level, the CRB estimate is optimistic: E[chi_4] scaling, or pooled vs per-sample rel_l2. If a residual tail > 1.5…
  - response to critic: The pilot evidence comes first. On 60 val samples (n_train=300) this exact arm gave 0.0190, with 3881 forward evals and 2.1 s. Its chi2/dof is 0.944, the residual std at truth is 0.000995 (no model mismatch), z_rms is 0.98 and 2σ coverage …
  - critic r3 (quality 0.83): Failure criterion (c), evals ≤ 1/10 of P-013, is contradicted by your own pilot: 3881/60 ≈ 65 evals/sample vs P-005's 20657/60 ≈ 344, a ratio of about 1/5.3. T… · This appears to be the P-015 arm restated. P-022 and P-024 both reference P-015 as this KRR-init 6-iteration hybrid. Deduplicate. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Pilot: 65 vs 344 evals/sample gives a ratio of 0.19, which exceeds 0.10. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-020 — Sample-limited, not architecture-limited: an MLP trained on 3000 provided + 12000 freshly simulated samples with per-ep…** (researcher-ml, family physics-custom)
- v1 `P-020` r3 [propose, llm] `custom-solver(fwd)` status conjectured. *H:* The amortised learned inverse map is limited by training-set size. A [512,512] GELU MLP predicting log k is trained on the 3000 provided samples plus 12000 prior draws simulated through context.forward. The simulated observations get fresh N(0, sigma²) noise …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If the gain over P-012 is below 20%, the learned map is architecture- or optimisation-limited, or already close to its noise-limited error, not sample-limited.…
  - critic r3 (quality 0.805): The contrast with P-012/P-019 changes at least four things at once: n (3k→15k), noise model (fixed noisy copies vs fresh per-epoch noise on noiseless trajector… · Fresh per-epoch noise on noiseless data alone may account for much of any gain by removing the √2σ noise mismatch in P-019's augmentation. A within-solver n_si… · Use the same architecture, epochs and feature pipeline across n_sim ∈ {0, 12000}; that pair is the hypothesis test, and P-012/P-019 is secondary. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

**P-024 — Control: box-centre-start 6-iteration log-k LM. Does the learned initializer matter under a tight iteration budget?** (researcher-exploratory, family physics)
- v1 `P-024` r3 [propose, llm] `physics-MLE[log,ms=1,it=6]` status conjectured. *H:* Single-start bounded LM (log-k, mode, 6 iterations, started at the prior centre) is at least 20% worse in val rel_l2 than the KRR-initialised 6-iteration hybrid (P-015). The failures concentrate in samples with extreme k1 or k4 (near 0.5 or 2.0), where /log(k…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.2, "expect": "worse"}`; if it fails: If centre-start matches P-015, the learned initializer adds nothing at a 6-iteration budget. The recommended pipeline is then plain few-iteration LM, and the P…
  - critic r3 (quality 0.841): The start location for multistart=1 is unverified. If the harness uses a random prior draw instead of the centre, the control's meaning changes. Log it. · The pair should be against whichever run is the canonical KRR+6LM arm (P-015/P-023) on identical indices. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · a single LM start can stop at a local minimum when the truth lies near a bound or the misfit is multimodal; a 2-start pilot discriminates cheaply · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0215, k4=0.0174): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: The physics arms (P-001, P-005) are the most decisive and cheapest. Run them first as one paired mode/mean arm, preceded by the F(k_true) misfit pilot and the oracle-start pilot. Collapse the four hybrids into P-003, emitting both estimators. Three proposals (P-002, P-006, P-007) exceed the 120 s limit, and P-006 confounds the initializer with the estimator. The predicted posterior-mean gains loo…  families={'physics': 2, 'hybrid': 4, 'data-driven': 2}
- Critic round 2 portfolio view: Run P-013 and P-009 now as a matched, paired pair; they are cheap and well specified. P-010 needs its start location fixed and the metric definition verified, after which it becomes a zero-extra-cost estimator ablation on shared IS samples. P-014 and P-012 should fix config mismatches (aug 1 vs 2; pca 32 vs 48), tune bandwidth/epochs on val, and add an extra-simulated-data arm. P-011 and P-015 sh…  families={'physics': 2, 'hybrid': 2, 'data-driven': 2, 'physics-custom': 1}
- Critic round 3 portfolio view: Empirically, physics MLE/MMSE already sits at about 0.0184-0.019 on n=60, with calibrated residuals and coverage, so the success target (0.05) is essentially met by the physics arms. The remaining questions are second order. The fixes needed are concrete, not conceptual. (1) Deduplicate P-021 into P-013 and P-023 into P-015. (2) Correct P-023's 1/10 eval criterion, which its own pilot refutes (ab…  families={'physics': 3, 'hybrid': 2, 'data-driven': 2, 'physics-custom': 2}

**Pilot experiments run during research** (small validation subsets):

- P-001 `physics-MMSE[log,ms=4,it=30]`: ok rel_l2=0.018395 on 60 val samples in 3.58 s
- P-005 `physics-MLE[log,ms=4,it=40]`: ok rel_l2=0.019024 on 60 val samples in 3.2 s
- P-008 `kernel_ridge/log+logy+aug2`: ok rel_l2=0.063632 on 60 val samples in 2 s
- P-009 `physics-MMSE[log,ms=4,it=40]`: ok rel_l2=0.018396 on 60 val samples in 3.79 s
- P-012 `torch_mlp/raw+pca32+logy`: ok rel_l2=0.061413 on 60 val samples in 6.57 s
- P-015 `hybrid[kernel_ridge/log+logy+aug1->LM6]`: ok rel_l2=0.019024 on 60 val samples in 2.13 s

## 5. Literature retrieval (TheoremSearch)

24 queries, 0 failures, 66 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for recovering piecewise constant conductivity coeffic… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-ml | 1 | Bernstein-von Mises or asymptotic efficiency of maximum likelihood / posterior mean for n… | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability for identifying piecewise constant conductivity in parabolic equation… | 4 refs via /search |
| researcher-mathematical | 1 | convergence of Levenberg-Marquardt or Gauss-Newton for finite-dimensional nonlinear least… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient from boundary or interior observations one-dimensiona… | 4 refs via /search |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | 4 refs via /search |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-mathematical | 2 | Normalized misfit of F(k_true) against val y on 50 samples | 4 refs via /search |
| researcher-experimental | 2 | Per-sample final normalized misfit distribution on val | 4 refs via /search |
| researcher-exploratory | 2 | Bayes estimator versus MLE under bounded uniform prior, MSE below Cramér-Rao bound | 4 refs via /search |
| researcher-ml | 2 | Iteration-wise rel_l2 and misfit for LM iterations 0..8 on val | 4 refs via /search |
| researcher-mathematical | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-ml | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-experimental | 3 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | 4 refs via /search (cache) |
| researcher-mathematical | 3 | Harness implementation of estimator='posterior_mean': proposal density, prior term, box h… | 4 refs via /search |
| researcher-exploratory | 3 | Log-temperature kernel ridge: a forward-free amortized inverse reaches rel_l2 < 0.05 at f… | 4 refs via /search |
| researcher-experimental | 3 | Distribution of final chi2/dof across all 750 val samples (max, 99th percentile), and rel… | 4 refs via /search |
| researcher-ml | 3 | Forward-eval counts and wall time breakdown (training vs refinement) for the hybrid | 4 refs via /search |
| researcher-exploratory | 3 | Harness code for features='log' and noise_scale | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_full_s1.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 183 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-017) | physics-custom | `custom-solver(fwd)` | researcher-mathematical | 0.925 | best-rated proposal in family 'physics-custom' (quality 0.93) |
| E-02 | proposal (P-016) | physics | `physics-MMSE[log,ms=4,it=40]` | researcher-mathematical | 0.916 | best-rated proposal in family 'physics' (quality 0.92) |
| E-03 | proposal (P-018) | hybrid | `hybrid[torch_mlp/raw+pca48+logy+aug1->LM6]` | researcher-ml | 0.865 | best-rated proposal in family 'hybrid' (quality 0.86) |
| E-04 | proposal (P-019) | data-driven | `torch_mlp/raw+pca32+logy+aug2` | researcher-ml | 0.845 | best-rated proposal in family 'data-driven' (quality 0.84) |
| E-05 | proposal (P-021) | physics | `physics-MLE[log,ms=4,it=40]` | researcher-experimental | 0.845 | quality 0.84 with diversity penalty |
| E-06 | proposal (P-022) | data-driven | `kernel_ridge/log+logy+aug1` | researcher-exploratory | 0.834 | quality 0.83 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-024 *Control: box-centre-start 6-iteration log-k LM. Does the learned initializer ma…* — portfolio size limit (6); quality 0.84
- not advanced: P-023 *KRR initializer + 6 LM steps reaches the CRB floor with ~1/25 of the forward ev…* — portfolio size limit (6); quality 0.83
- not advanced: P-020 *Sample-limited, not architecture-limited: an MLP trained on 3000 provided + 120…* — portfolio size limit (6); quality 0.81

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-002 | E-01 | research | physics-custom | `custom-solver(fwd)` | 0.01994 | **0.018944** | 0.9967 | 0.02498 | 3.55 | 23.7 | 271800 | - | FINAL |
| idea-003 | E-02 | research | physics | `physics-MMSE[log,ms=4,it=40]` | 0.02028 | **0.019059** | 0.9967 | 0.02513 | 0.0101 | 20.7 | 329335 | 0.952 | evaluated |
| idea-012 | E-03 | refine | hybrid | `hybrid[torch_mlp/raw+pca48+logy+aug1->LM6]` | - | **0.019325** | 0.9966 | 0.02548 | 29.2 | 3.08 | 46100 | 0.957 | evaluated |
| idea-006 | E-05 | research | physics | `physics-MLE[log,ms=4,it=40]` | 0.02138 | **0.019335** | 0.9966 | 0.0255 | 0.0105 | 15.8 | 239528 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0111 | 4.34 | 61517 | 0.957 | evaluated |
| idea-004 | E-03 | research | hybrid | `hybrid[torch_mlp/raw+pca48+logy+aug1->LM6]` | 0.02139 | **0.019342** | 0.9966 | 0.02551 | 31 | 2.99 | 46091 | 0.957 | evaluated |
| idea-013 | E-03 | refine | hybrid | `hybrid[torch_mlp/raw+pca48+logy+aug1->LM5]` | - | **0.019342** | 0.9966 | 0.02551 | 29.2 | 2.82 | 40637 | 0.957 | evaluated |
| idea-014 | E-06 | refine | data-driven | `kernel_ridge/raw+logy+aug1` | - | **0.027659** | 0.993 | 0.03648 | 2.56 | 0.0988 | 0 | - | evaluated |
| idea-015 | E-06 | refine | data-driven | `kernel_ridge/log+logy+aug1` | - | **0.030199** | 0.9916 | 0.03982 | 2.55 | 0.0897 | 0 | - | evaluated |
| idea-007 | E-06 | research | data-driven | `kernel_ridge/log+logy+aug1` | 0.04658 | **0.031411** | 0.9909 | 0.04142 | 2.42 | 0.105 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06722 | **0.032246** | 0.9904 | 0.04252 | 6.32 | 0.0219 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt` | 0.1505 | **0.082482** | 0.9373 | 0.1088 | 2.39 | 0.106 | 0 | - | evaluated |
| idea-005 | E-04 | research | data-driven | `torch_mlp/raw+pca32+logy+aug2` | 0.04552 | **-** | - | - | - | - | - | - | ok |
| idea-009 | E-08 | baseline | data-driven | `svm/log+pca8` | 0.1782 | **-** | - | - | - | - | - | - | projected full-fidelity cost 203s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-002` `custom-solver(fwd)` ← P-017 v2 (researcher-mathematical, r3, revise) ← P-010 v1 (researcher-mathematical, r2, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00241` [checkpoint] by scheduler: final method (single)
- `resu-00238` [result] by executor: exp-0030 rel_l2=0.0183 r2=0.997 rmse=0.02408
- `hypo-00122` [hypothesis] by scheduler: idea-002 custom-solver(fwd) (research)
- `plan-00237` [plan] by engineer-v2: exp-0030: final fit of 1 member(s) on train+val; test once
- `gate-00121` [gate] by gate: E-01: approved (custom-solver(fwd))
- `prop-00106` [proposal] by researcher-mathematical: P-017 v2 [revise] mathematical: Explicit IS posterior with prior Jacobian, mirrored-converged second start, and a train-se
- `spec-00001` [spec] by scheduler: research spec heat_v2_full_s1 (8c7c7bb48f)
- `prop-00071` [proposal] by researcher-mathematical: P-010 v1 [propose] mathematical: Explicit Bayes estimator for per-sample relative loss: LM with mirrored start plus importa
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 311.2 / 400 worker-seconds over 30 jobs; wall 768 s
- By stage: baseline 1.9s, pilot 21.3s, screen 57.6s, evaluate 113.7s, refine 77.3s, final 39.5s
- LLM: 19 calls (by phase {'research': 19}), 1174 s, $4.432 reported · TheoremSearch: 24 queries (0 failed)

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

The success criterion is met with a wide margin. The selected estimator, a custom forward solver with likelihood fitting, reached val rel_l2 = 0.0189 and held-out test rel_l2 = 0.0183 (R² = 0.997, RMSE = 0.024), against a target of ≤ 0.05. The val-to-test gap is small, which suggests selection did not overfit the validation split. The physics-based estimators clustered tightly: custom solver 0.0189, physics-MMSE 0.0191, and the hybrid MLP+PCA→LM 0.0193. The best purely data-driven model, kernel ridge, was clearly worse at 0.0277, about 46% higher error. All five pre-registered predictions were supported and none were contradicted. The pilot evidence points the same way. Cold-start multistart Levenberg–Marquardt gave χ²/dof ≈ 0.94, an error of about 1.03× the Cramér–Rao bound, and calibrated Gauss–Newton coverage. Together these indicate that the physics estimators are operating at or near the information floor set by the sensor noise. The ~0.019 plateau is therefore best read as a property of the data, not a limitation of the methods. That reading is well supported but remains an inference, since it rests on a pilot CRB comparison rather than a full-test analysis.

No hypothesis was refuted by a failed experiment. The three abandoned hypotheses were pruned before execution, which is weaker evidence than a tested negative:

- **Learned initializer + LM beats cold-start LM.** Pruned because cold LM already finds the correct basin and reaches the floor, so there is no accuracy headroom.
- **Initializer + MLE / posterior-mean comparison.** Pruned as confounded and over budget, at 134 s against a 120 s limit.
- **Posterior-mean output for MMSE gains.** Pruned as a duplicate.

The one claim indirectly tested is that the posterior mean helps because rel_l2 rewards MMSE. It is consistent with "negligible gain": physics-MMSE (0.01906) did not beat the mode-type custom solver (0.01894). The predicted near-bound gain for k1/k4 has not been reported in stratified form, so it remains conjecture. Several other points are also uncertain:

- No confidence intervals were reported, so the ranking among the top three (a spread of about 0.0004) is likely within noise.
- Robustness to noise-model or forward-model misspecification is untested, because training and evaluation share the same simulator.
- The cost–accuracy trade-off of amortised hybrids versus per-sample LM was not quantified under the 400 worker-second budget.
- The `noise_scale` ambiguity flagged in one proposal was not resolved.

Next experiments:
- Compute paired bootstrap CIs on per-sample val rel_l2 for custom-solver, physics-MMSE and the hybrid, to establish whether any ranking among them is real.
- Stratify error by proximity of k1/k4 to the prior bounds and report MLE versus posterior mean from shared LM solutions. This directly tests the abandoned near-bound MMSE prediction at almost no extra cost.
- Compute the per-sample CRB across the full val set and check Gauss–Newton interval coverage, to confirm the "information floor" claim beyond the pilot.
- Stress-test misspecification by perturbing the noise level and distribution (e.g. heteroscedastic or heavy-tailed noise), sensor positions, and the number of conductivity segments. Compare how physics and data-driven methods degrade.
- Measure the wall-clock cost per sample for cold multistart LM versus the hybrid amortised initializer. This would test the remaining, untested claim that amortisation reduces cost at equal accuracy.

_Automatic summary:_ 24 proposal versions over 3 research rounds; 9 methods advanced; 5 predictions supported, 0 contradicted. Final: ['custom-solver(fwd)'] with test rel_l2 0.018301.
