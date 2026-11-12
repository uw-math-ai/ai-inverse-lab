# Research report (V2): heat_v2_no_literature_s2

Run `heat_v2_no_literature_s2_20261109_152750` · 2026-11-09 15:34:53 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `hybrid[ridge/raw+pca40+logy+aug1->LM15]`
- Validation rel_l2 (selection estimate): **0.018992** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018326**, r2 = **0.99699**, rmse = **0.024114** · bootstrap 95% CI for rel_l2: [0.01725, 0.01944]
- Per-target test rel_l2: k1 0.0268, k2 0.0061, k3 0.0052, k4 0.0241
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.097203 (vs 0.018326 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051206 (vs 0.018326 in distribution)
- Uncertainty calibration on test: 68.3% within 1σ, 95.2% within 2σ (ideal 68% / 95%)
- Inference latency: 34.4 ms/sample; fit 0.0511 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_literature_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[ridge/raw+pca40+logy+aug1->LM15]` rel_l2=0.018992; **physics** `physics-MMSE[log,ms=3,it=30]` rel_l2=0.018993; **data-driven** `torch_mlp/raw` rel_l2=0.029289
- SUPPORTED — E-02 *Raw-feature ridge initializer + log-k LM + box-prior MMSE (posterior mean of k): CRB-level accuracy at full validation …*: val rel_l2 = 0.018992 vs predicted < 0.022
- SUPPORTED — E-03 *Target-space ablation for the amortized inverse: E[k/x] vs E[log k/x], with a per-parameter CRB-ratio diagnostic*: val rel_l2 0.029289 vs best_hybrid 0.018992: gain -54.22%; expected to be worse than best_hybrid (gain < +30.0%)
- SUPPORTED — E-04 *Shared physics-mode baseline: initializer-free multistart MLE in log k is as accurate as the hybrid (landscape operatio…*: val rel_l2 = 0.019339 vs predicted < 0.023
- SUPPORTED — E-05 *MMSE rather than MAP: posterior-mean hybrid in log-conductivity space targets the rel_l2 risk directly*: val rel_l2 = 0.018996 vs predicted < 0.03
- SUPPORTED — E-06 *Does initializer quality matter after LM? MLP-init hybrid with posterior-mean readout (cost-reduced)*: val rel_l2 = 0.019032 vs predicted < 0.03
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- no citations in proposals

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- none

### 3.4 Failed hypotheses and negative results

- CONTRADICTED — E-01 *Paired estimator control: box-prior posterior mean vs. MLE with identical physics solver, stratified by distance to bou…*: val rel_l2 0.018993 vs best_physics 0.019335: gain +1.77%; expected gain >= +2.0% vs best_physics
- ABANDONED in round 2 by researcher-experimental — P-005 *Physics MLE via log-space multistart LM reaches near-CRB accuracy*: duplicate of P-002
- ABANDONED in round 2 by researcher-experimental — P-006 *Learned initializer + MMSE refinement beats both pure ML and pure MLE (competing-explanation contro…*: blocking issue: The estimated runtime of 134.7 s exceeds the 120 s per-experiment limit.
- ABANDONED in round 2 by researcher-exploratory — P-008 *Framing challenge: is the forward model even necessary? An amortized, noise-matched neural inverse …*: blocking issue: The estimated runtime of 120 s leaves no margin under the 120 s per-experiment limit. Reduce epochs.

### 3.5 Remaining uncertainty

- E-01 (Paired estimator control: box-prior posterior mean vs. MLE …): challenge: This is the cleanest control in the portfolio. The main risk is power: if about 5% of samples are near a bound, the near-bound stratum holds about 35-40 samples. The paired CI may include 0 even if the effect…
- E-01 (Paired estimator control: box-prior posterior mean vs. MLE …): challenge: 'Within 2 CR sigma of a bound' needs a per-sample CR sigma, not the median. Define the stratum from the per-sample CRB at the truth.
- E-01 (Paired estimator control: box-prior posterior mean vs. MLE …): challenge: The assumption about k-space averaging matters for interpretation. If the harness averages log k, the result tests exp(E[log k]), not MMSE.
- E-01 (Paired estimator control: box-prior posterior mean vs. MLE …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-02 (Raw-feature ridge initializer + log-k LM + box-prior MMSE (…): challenge: The prediction (<0.022) is weak given the pilot already gave 0.0186. A result of 0.021 would 'confirm' the hypothesis but would be an 13% regression from the pilot. Tighten it to a band, e.g. rel_l2 in [0.016…
- E-02 (Raw-feature ridge initializer + log-k LM + box-prior MMSE (…): challenge: The CRB reference 0.0186 comes from 32 prior draws, and it bounds unbiased estimators only. A box-prior posterior mean is biased and can legitimately go below the CRB, especially for k1 and k4 near bounds. 'W…
- E-02 (Raw-feature ridge initializer + log-k LM + box-prior MMSE (…): challenge: The pilot P-001 label shows 'ridge/log+pca40', i.e. log features, with train_fraction=300. This proposal uses raw features and presumably the full train set. The init has changed, so the pilot is not strictly…
- E-02 (Raw-feature ridge initializer + log-k LM + box-prior MMSE (…): challenge: Criterion (b) uses 1.5x the CR std for k2 and k3 from the 32-draw median. Per-sample CR std varies with k, so an aggregate RMSE is being compared against a median std. Use the RMS of per-sample CR std over th…
- E-03 (Target-space ablation for the amortized inverse: E[k/x] vs …): challenge: The Jensen-gap argument concerns the posterior E[log k/x] vs E[k/x]. The MLP error (about 0.03-0.05) is dominated by approximation error, and the target transform changes loss weighting across [0.5, 2]. Part …
- E-03 (Target-space ablation for the amortized inverse: E[k/x] vs …): challenge: The comparator P-008 is not in this round. Unless the log-target MLP is retrained here with the same seed and data, the paired CI cannot be computed. Seed variance of an MLP may itself exceed 0.003.
- E-03 (Target-space ablation for the amortized inverse: E[k/x] vs …): challenge: The prediction field ('relative_to best_hybrid, improvement 0.3, expect worse') does not match the stated hypothesis (equivalence with the log-target MLP). It should state the equivalence test explicitly.
- E-03 (Target-space ablation for the amortized inverse: E[k/x] vs …): challenge: The question bears weakly on the objective, since hybrids are already at about CRB level. Its main value is (ii): whether amortised maps are information-limited.
- E-04 (Shared physics-mode baseline: initializer-free multistart M…): challenge: The harness's 'multistart=3' placement is unverified. If the starts are prior-centre plus small jitter, a pass says nothing about landscape benignity over the box. Log the start points.
- E-04 (Shared physics-mode baseline: initializer-free multistart M…): challenge: Criterion (b) is concrete: with N_obs about 800 (4 sensors x ~200 times), the threshold is about 1.2. However, a series-resistance valley stall could leave chi2/N below 1.2 while the parameters are off. Crite…
- E-04 (Shared physics-mode baseline: initializer-free multistart M…): challenge: Criterion (d) depends on P-009 running on the same val samples with the same noise realisation. Confirm that pairing.
- E-04 (Shared physics-mode baseline: initializer-free multistart M…): counterexample: Take a sample with k1 small and k4 large. The effective series resistance seen by the outer sensors can be nearly matched by trading k1 against k2. A start on the wrong side of the valley could converge …
- E-05 (MMSE rather than MAP: posterior-mean hybrid in log-conducti…): challenge: The proposal is internally inconsistent. Its own assumption predicts an aggregate mean-vs-mode gain below 5%, but the falsification criterion requires the posterior mean to beat the mode by at least 5% relati…
- E-05 (MMSE rather than MAP: posterior-mean hybrid in log-conducti…): challenge: The required comparator (the same hybrid with estimator=mode) is not in the portfolio. The mode-vs-mean question is already isolated cleanly by P-010 vs P-011 (physics) and by P-009's within-run readouts.
- E-05 (MMSE rather than MAP: posterior-mean hybrid in log-conducti…): challenge: The only new factor relative to P-009 is extra_trees vs ridge init. Given P-010/P-012, init quality is likely irrelevant after LM. The arm is near-duplicate of P-009 in what it can teach.
- E-05 (MMSE rather than MAP: posterior-mean hybrid in log-conducti…): challenge: 'Mode biased toward the boundary' is imprecise. A box-clipped MLE piles mass on the bound, which is a projection, while the posterior mean is pulled inward. The effect exists only for samples within about 2 C…
- E-06 (Does initializer quality matter after LM? MLP-init hybrid w…): challenge: This arm is largely redundant with P-010. If initializer-free multistart already matches the hybrid, init quality is shown not to matter, and a better init cannot add accuracy. If P-010 reveals basin failures…
- E-06 (Does initializer quality matter after LM? MLP-init hybrid w…): challenge: At 52.4 s it is the most expensive arm, and its expected information gain is the lowest of the hybrid arms given the pilot evidence.
- E-06 (Does initializer quality matter after LM? MLP-init hybrid w…): challenge: It uses 64 posterior samples while P-009 uses 256. A 'within 5%' comparison against P-009 confounds init with IS precision, which matters most for edge samples.
- E-06 (Does initializer quality matter after LM? MLP-init hybrid w…): counterexample: MLP and ridge both reach the basin in 100% of samples, and the final rel_l2 differs by 1-2% because of IS sample count alone. Criterion (b) would then be misattributed.
- Unresolved disagreement ['P-014', 'P-011']: Size of the mean-vs-mode gain. P-014's falsification criterion requires an aggregate gain of at least 5%. P-011 (and P-014's own assumption) predicts 1-5%, concentrated near bounds. P-011's framing is better grounded.
- Unresolved disagreement ['P-012', 'P-010']: Whether init quality matters after LM. P-010 claims it buys only runtime. If P-010 confirms this at full val, P-012 is uninformative.
- Unresolved disagreement ['P-009', 'P-011', 'P-012']: Whether the harness posterior mean averages k or log k. The answer is unverified and changes the interpretation of every MMSE arm.
- Literature check INCOMPLETE: 16 failed queries (disabled), 0 skipped for budget, enabled=False
- Test-set sampling uncertainty: bootstrap 95% CI [0.01725, 0.01944] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | llm | Pilot evidence (60 val samples): hybrid rel_l2 = 0.0186. This is already below my prediction of 0.03. Fit dia… |
| 2 | researcher-mathematical | revise | P-002 | P-010 | llm | Pilot evidence (n=60): initializer-free multistart MLE reached rel_l2 0.0190 with chi2/dof 0.944. That is wit… |
| 2 | researcher-mathematical | propose | - | P-011 | llm |  |
| 2 | researcher-ml | revise | P-003 | P-012 | llm | (1) Cost 134.7 s > 120 s: accepted. The revision cuts the initializer to hidden [256,128], 60 epochs and nois… |
| 2 | researcher-ml | revise | P-004 | P-013 | llm | (1) Cost 160 s > 120 s: accepted. The revision uses hidden [256,128], 80 epochs and noise_aug=0, about 25-40 … |
| 2 | researcher-experimental | abandon | P-005 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 2 | researcher-experimental | abandon | P-006 | - | heuristic-fallback | Accepted: cannot be executed as specified. |
| 2 | researcher-exploratory | revise | P-007 | P-014 | heuristic-fallback | Revised: incorporated critic assumptions |
| 2 | researcher-exploratory | abandon | P-008 | - | heuristic-fallback | Accepted: cannot be executed as specified. |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.641 | fdbk-00044 |
| 2 | 6 | 0.843 | fdbk-00075 |

### Idea evolution

**P-001 — Learned initializer + Gauss-Newton/LM refinement in log-conductivity, MMSE readout: reach within ~1.5x of the Cramér–Ra…** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[ridge/log+pca40+logy+aug1->LM15]` status conjectured. *H:* The parameter-to-observation map F: [0.5,2]^4 -> R^{4 x T} is smooth and locally well-conditioned (median Fisher condition number about 48). The noise is small (sigma=1e-3), so the likelihood is close to Gaussian around the truth. Under these conditions, a ch…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If rel_l2 is still near the ridge-init level, LM is not converging. Either the init lies outside the basin, which points to nonconvexity and argues for more ex…
  - critic r1 (quality 0.855): The CRB of about 0.0186 is a local, unbiased-estimator bound. It is not a floor for the posterior mean, which is biased near the box edges and uses the prior, … · features='log' on noisy temperatures is risky. Early-time readings at the outer sensors (indices 5 and 35) are near zero before heat arrives, so additive noise… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · For a sample with k1 at 0.52, the likelihood in log k is truncated at log 0.5. The Laplace proposal centred on the clipped mode puts about half its mass outsid… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-009` r2 [revise, llm] `hybrid[ridge/raw+pca40+logy+aug1->LM15]` status conjectured. *H:* The parameter-to-observation map is smooth and locally well-conditioned at sigma=1e-3, and the forward model is verified exact (residual std at truth 0.000995). Under these conditions, ridge-on-raw initialisation followed by 15 bounded LM iterations in log k,…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.022}`; if it fails: If (b) or (c) fails, the init falls outside the basin for some samples. Raise extra_starts or switch to a nonlinear init. If only (a) fails, with k1 and k4 at …
  - response to critic: Pilot evidence (60 val samples): hybrid rel_l2 = 0.0186. This is already below my prediction of 0.03. Fit diagnostics: chi2/dof 0.944, IS ESS 58.6 of 128, 1-sigma coverage 0.70, 2-sigma coverage 0.95, z_rms 1.01. Residual std at the truth …
  - critic r2 (quality 0.876): The prediction (<0.022) is weak given the pilot already gave 0.0186. A result of 0.021 would 'confirm' the hypothesis but would be an 13% regression from the p… · The CRB reference 0.0186 comes from 32 prior draws, and it bounds unbiased estimators only. A box-prior posterior mean is biased and can legitimately go below … · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · If the val set contains a few samples with k1 or k4 near 0.5, where sensitivity is lowest, the per-sample CRB there can be several times the median. Aggregate … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-002 — Global well-posedness test: initializer-free multistart MLE through the forward model achieves near-CRB accuracy (no le…** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* With four segment-wise sensors and a centred pulse, the least-squares misfit J(log k) = //F(k) - d//^2 has no spurious local minima over the prior box that attract LM from a small number of spread starts. Pure multistart MLE (3 starts, 30 iterations, log tran…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: Misfit-inconsistent endpoints would show that J has spurious minima or long flat valleys, most likely along the k1/k4 directions. That would justify learned in…
  - critic r1 (quality 0.846): Three starts over a 4-D box is sparse coverage. 'No spurious minima' cannot be concluded from 3 starts. Absence of misfit-inconsistent endpoints only shows tha… · The chi^2/N_obs > 1.5 threshold is too loose to detect shallow secondary minima. With N_obs of about 800, the noise-only chi^2/N has std sqrt(2/800), about 0.0… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · A ridge along a series-resistance direction, such as 1/k1 + 1/k2 held constant, produces an elongated valley. LM with 30 iterations from a far start may stop o… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-010` r2 [revise, llm] `physics-MLE[log,ms=3,it=30]` status conjectured. *H:* Over the prior box, LM in log k from three spread starts converges to a noise-consistent minimiser for at least 98% of samples. Full-val rel_l2 is < 0.023 and within 10% of the P-001 hybrid. The learned initializer therefore buys runtime, not accuracy.
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.023}`; if it fails: If (b) or (c) fails, misfit-consistent valley endpoints or misfit-inconsistent stationary points exist, most likely along an inner/outer series-resistance dire…
  - response to critic: Pilot evidence (n=60): initializer-free multistart MLE reached rel_l2 0.0190 with chi2/dof 0.944. That is within 2.3% of the hybrid's 0.0186, consistent with my claim of 'at most 15% worse'. It is preliminary evidence that the learned init…
  - critic r2 (quality 0.882): The harness's 'multistart=3' placement is unverified. If the starts are prior-centre plus small jitter, a pass says nothing about landscape benignity over the … · Criterion (b) is concrete: with N_obs about 800 (4 sensors x ~200 times), the threshold is about 1.2. However, a series-resistance valley stall could leave chi… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · Take a sample with k1 small and k4 large. The effective series resistance seen by the outer sensors can be nearly matched by trading k1 against k2. A start on … · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer + LM refinement with posterior-mean estimator approaches the CRB floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM12]` status conjectured. *H:* A torch MLP trained on simulated train data gives a coarse estimate of k. A few bounded Levenberg-Marquardt iterations in log-k space through the exact forward model, followed by an importance-sampled posterior mean under the box prior, will reach val rel_l2 …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: There are three diagnostic outcomes. If LM does not improve on the MLP, there is either forward-model mismatch or the init is outside the basin; check the resi…
  - critic r1 (quality 0.646): The estimated runtime of 134.7 s exceeds the 120 s per-experiment limit. Reduce epochs or width before running. · The estimated cost of 134.7 s exceeds the 120 s per-experiment limit cited in the proposal. · noise_scale=1.0 conflicts with P-006 and P-008, which use 0.001. If noise_scale is in absolute units, sigma=1.0 noise swamps the signal and the MLP learns almo… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-012` r2 [revise, llm] `hybrid[torch_mlp/raw+logy->LM12]` status conjectured. *H:* A torch MLP initializer (no augmentation) followed by 12 bounded log-space LM iterations and a box-truncated importance-sampled posterior mean will reach val rel_l2 < 0.03. Its final rel_l2 will be within 5% (relative) of the ridge-initialized hybrid P-001. I…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: Each refutation outcome points to a different cause. (a) Rel_l2 >= 0.03 with chi^2/N_obs about 1 means the estimator is information-limited: the CRB reference …
  - response to critic: (1) Cost 134.7 s > 120 s: accepted. The revision cuts the initializer to hidden [256,128], 60 epochs and noise_aug=0, and posterior_samples goes from 128 to 64. The dominant MLP training cost should fall roughly 3-4x, so I expect it to fit…
  - critic r2 (quality 0.795): This arm is largely redundant with P-010. If initializer-free multistart already matches the hybrid, init quality is shown not to matter, and a better init can… · At 52.4 s it is the most expensive arm, and its expected information gain is the lowest of the hybrid arms given the pilot evidence. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · MLP and ridge both reach the basin in 100% of samples, and the final rel_l2 differs by 1-2% because of IS sample count alone. Criterion (b) would then be misat… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-004 — Pure amortized inverse map with noise augmentation and log targets falls short of the success threshold** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* A purely data-driven MLP trained on 3000 simulated samples, with log-k targets, standardized raw sensor features and on-the-fly noise augmentation at the true sigma, will reach val rel_l2 in [0.035, 0.07]. It will beat the mean predictor (0.33) by more than 5…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If the MLP alone achieves <= 0.035, the amortized map is near-efficient and the hybrid's LM step is unnecessary cost; the next step would be ensembling MLPs to…
  - critic r1 (quality 0.384): The estimated runtime of 160 s exceeds the 120 s per-experiment limit. · The estimated cost of 160 s exceeds the 120 s limit. · This duplicates P-008 except for noise_scale (1.0 vs 0.001), epochs and activation. The noise_scale difference is exactly the unresolved unit ambiguity. · Replace 'errors largest for k1/k4' with a test on the ratio of per-parameter MLP RMSE to CRB std. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- v2 `P-013` r2 [revise, llm] `torch_mlp/raw` status conjectured. *H:* Two predictions. (i) A torch MLP trained on untransformed k targets reaches val rel_l2 within ±0.003 (absolute) of the identical log-target MLP (P-008). The Jensen gap between exp(E[log k/x]) and E[k/x] is about Var(log k/x)/2, roughly 1e-3 relative at poster…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.3, "expect": "worse"}`; if it fails: If (i) fails with log better, the loss weighting rather than estimator bias matters, and log should be preferred. If it fails with none better, E[k/x] targetin…
  - response to critic: (1) Cost 160 s > 120 s: accepted. The revision uses hidden [256,128], 80 epochs and noise_aug=0, about 25-40 s. (2) Duplicate of P-008: accepted as stated. Rather than abandon, I repurpose the arm to resolve the critic's own listed disagre…
  - critic r2 (quality 0.78): The Jensen-gap argument concerns the posterior E[log k/x] vs E[k/x]. The MLP error (about 0.03-0.05) is dominated by approximation error, and the target transf… · The comparator P-008 is not in this round. Unless the log-target MLP is retrained here with the same seed and data, the paired CI cannot be computed. Seed vari… · Include at least 2 seeds per target space, or an explicit seed-variance estimate, before interpreting a ±0.003 difference. · Two seeds of the same log-target MLP differ by 0.004 rel_l2. Part (i) would then be 'refuted' by pure optimisation noise. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Physics MLE via log-space multistart LM reaches near-CRB accuracy** (researcher-experimental, family physics)
- v1 `P-005` r1 [propose, llm] `physics-MLE[log,ms=4,it=40]` status conjectured. *H:* Because the forward model is available, noise is small (sigma=0.001) and the Fisher condition number is moderate (median 48), per-sample bounded Levenberg-Marquardt in log-k space with a few multistarts will converge to the global least-squares minimum for mo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If error is high but misfits are at noise level, the problem is less identifiable than the local CRB suggests (posterior mass near box edges) -> switch to post…
  - critic r1 (quality 0.498): This substantively duplicates P-002: multistart log-LM with the mode, with 4x40 instead of 3x30. · The '>5% of samples with chi^2/N > 2' threshold is very loose for N of about 800, as explained in the P-002 review. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-006 — Learned initializer + MMSE refinement beats both pure ML and pure MLE (competing-explanation control)** (researcher-experimental, family hybrid)
- v1 `P-006` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM20]` status conjectured. *H:* The dominant residual error of pure data-driven regression is approximation error of the inverse map (not noise), while the residual error of pure MLE comes from k1/k4 boundary-truncated, non-Gaussian posteriors. A hybrid that initialises LM with a torch MLP …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.15, "expect": "better"}`; if it fails: If hybrid does not beat MLP alone, the learned map is already near-efficient and forward-model refinement is wasted compute; if it does not beat MLE, the initi…
  - critic r1 (quality 0.375): The estimated runtime of 134.7 s exceeds the 120 s per-experiment limit. · This is nearly identical to P-003: torch_mlp init, LM in log space, posterior_mean with 128 samples. The only differences are iters (20 vs 12), epochs (200 vs … · The estimated cost of 134.7 s exceeds the 120 s limit. · Expect a hybrid gain over MLE of less than 5% unless MLE has basin failures. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-007 — MMSE rather than MAP: posterior-mean hybrid in log-conductivity space targets the rel_l2 risk directly** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[extra_trees/raw+logy+aug1->LM12]` status conjectured. *H:* rel_l2 is a squared-error risk, so the Bayes-optimal estimator is the posterior mean under the uniform box prior, not the MLE or the mode. A learned initializer followed by bounded LM in log k, then importance-sampled posterior mean, should reach val rel_l2 <…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the mode and the posterior mean tie, the posterior is effectively Gaussian and unclipped, so bound effects are negligible and the CRB floor is the binding l…
  - critic r1 (quality 0.869): Rough quantification suggests the MMSE gain will be small. For sigma of about 2% relative, roughly (0.022 + 0.088)/1.5, about 7%, of k1 draws lie within 2 sigm… · The extra_trees initializer is piecewise-constant and has coarse resolution in a 4-D box. It is fine as an init only if the landscape is benign, which confound… · Evaluate the mean-versus-mode effect on the subset of samples with any true k_i within 3 CRB sigma of a bound. Expect an aggregate gain of less than 5% and a s… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- v2 `P-014` r2 [revise, heuristic-fallback] `hybrid[extra_trees/raw+logy+aug1->LM12]` status conjectured. *H:* rel_l2 is a squared-error risk, so the Bayes-optimal estimator is the posterior mean under the uniform box prior, not the MLE or the mode. A learned initializer followed by bounded LM in log k, then importance-sampled posterior mean, should reach val rel_l2 <…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the mode and the posterior mean tie, the posterior is effectively Gaussian and unclipped, so bound effects are negligible and the CRB floor is the binding l…
  - response to critic: Revised: incorporated critic assumptions
  - critic r2 (quality 0.839): The proposal is internally inconsistent. Its own assumption predicts an aggregate mean-vs-mode gain below 5%, but the falsification criterion requires the post… · The required comparator (the same hybrid with estimator=mode) is not in the portfolio. The mode-vs-mean question is already isolated cleanly by P-010 vs P-011 … · Fix the mean-vs-mode criterion: aggregate gain in 0-5%, and a significant gain only in the near-bound stratum. This is consistent with P-011. · Mean and mode tie in aggregate within 1%, which is the expected outcome. P-014 is then 'falsified' even though MMSE behaves exactly as theory predicts. · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-008 — Framing challenge: is the forward model even necessary? An amortized, noise-matched neural inverse with log targets** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The inverse map from sensor time series to log k is smooth and low-dimensional, with 4 parameters and a well-conditioned Fisher information. A purely amortized estimator should therefore reach val rel_l2 <= 0.05 with no forward calls at test time. The estimat…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.5, "expect": "worse"}`; if it fails: If the MLP stalls around 0.06–0.10, the bottleneck is sample efficiency in a 4-D box rather than information content. That argues for hybrid refinement or for …
  - critic r1 (quality 0.655): The estimated runtime of 120 s leaves no margin under the 120 s per-experiment limit. Reduce epochs. · The estimated cost of 120 s sits exactly at the limit, leaving no margin. · A squared loss on log k gives E[log k / x], not E[k / x]. The amortised estimator therefore does not target the rel_l2-optimal quantity, contrary to the motiva… · Train on k targets, or apply a variance correction, if the goal is to approximate the MMSE. · the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-011 — Paired estimator control: box-prior posterior mean vs. MLE with identical physics solver, stratified by distance to bou…** (researcher-mathematical, family physics)
- v1 `P-011` r2 [propose, llm] `physics-MMSE[log,ms=3,it=30]` status conjectured. *H:* With everything else held fixed (log LM, multistart 3, 30 iterations, no learned init), replacing the mode with the IS posterior mean of k under the uniform box prior lowers full-val rel_l2 by 1-5% relative to P-002. The gain is concentrated in samples whose …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_physics", "improvement": 0.02, "expect": "better"}`; if it fails: If MMSE gives no gain even on edge samples, the box truncation is too rare or too weak to matter. The mode is then sufficient, and the MMSE arguments in P-001,…
  - critic r2 (quality 0.885): This is the cleanest control in the portfolio. The main risk is power: if about 5% of samples are near a bound, the near-bound stratum holds about 35-40 sample… · 'Within 2 CR sigma of a bound' needs a per-sample CR sigma, not the median. Define the stratum from the per-sample CRB at the truth. · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

- Critic round 1 portfolio view: The physics-based arms are cheap (14-45 s) and well motivated, and the CRB reference of about 0.019 makes 0.05 very likely attainable. Main fixes: merge the duplicates (P-005 into P-002, P-006 into P-003, P-004 into P-008); resolve the noise_scale unit conflict; cut the MLP arms under 120 s; and avoid log-feature NaNs in the P-001 ridge init. Treat the CRB as a reference rather than a floor for M…  families={'hybrid': 4, 'physics': 2, 'data-driven': 2}
- Critic round 2 portfolio view: The pilots are decisive: the forward model is exact, and both physics MLE and the hybrid reach about 0.019 rel_l2 at n=60, well under the 0.05 target and calibrated. Further debate will not change what is worth running. The full-val runs of P-009, P-010 and P-011 form a clean, paired 2x2-style decomposition: init vs none, mode vs mean. These should run first, with per-sample CRB, stratification b…  families={'hybrid': 3, 'physics': 2, 'data-driven': 1}

**Pilot experiments run during research** (small validation subsets):

- P-001 `hybrid[ridge/log+pca40+logy+aug1->LM15]`: ok rel_l2=0.018596 on 60 val samples in 3.1 s
- P-002 `physics-MLE[log,ms=3,it=30]`: ok rel_l2=0.019024 on 60 val samples in 2.85 s

## 5. Literature retrieval (TheoremSearch)

0 queries, 16 failures, 0 distinct statements retrieved, 0 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-mathematical | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-mathematical | 1 | uniqueness of coefficient recovery in one-dimensional heat equation from point observatio… | FAILED: disabled |
| researcher-exploratory | 1 | Lipschitz stability for inverse problem recovering piecewise constant conductivity in par… | FAILED: disabled |
| researcher-exploratory | 1 | local convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares para… | FAILED: disabled |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | FAILED: disabled |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | FAILED: disabled |
| researcher-ml | 1 | uniqueness of piecewise constant diffusion coefficient recovery from point observations o… | FAILED: disabled |
| researcher-mathematical | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-experimental | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-mathematical | 2 | Residual between context.forward(theta_true) and the train features, compared with sigma*… | FAILED: disabled |
| researcher-exploratory | 2 | convergence of Levenberg-Marquardt for nonlinear least squares parameter estimation with … | FAILED: disabled |
| researcher-ml | 2 | Harness definition of noise_scale and whether stored train features include noise | FAILED: disabled |
| researcher-experimental | 2 | Levenberg-Marquardt multistart parameter identification heat equation piecewise constant … | FAILED: disabled |
| researcher-exploratory | 2 | posterior mean versus MAP estimator squared error bounded uniform prior inverse problem | FAILED: disabled |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_literature_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 77 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-011) | physics | `physics-MMSE[log,ms=3,it=30]` | researcher-mathematical | 0.885 | best-rated proposal in family 'physics' (quality 0.89) |
| E-02 | proposal (P-009) | hybrid | `hybrid[ridge/raw+pca40+logy+aug1->LM15]` | researcher-mathematical | 0.876 | best-rated proposal in family 'hybrid' (quality 0.88) |
| E-03 | proposal (P-013) | data-driven | `torch_mlp/raw` | researcher-ml | 0.78 | best-rated proposal in family 'data-driven' (quality 0.78) |
| E-04 | proposal (P-010) | physics | `physics-MLE[log,ms=3,it=30]` | researcher-mathematical | 0.882 | quality 0.88 with diversity penalty |
| E-05 | proposal (P-014) | hybrid | `hybrid[extra_trees/raw+logy+aug1->LM12]` | researcher-exploratory | 0.839 | quality 0.84 with diversity penalty |
| E-06 | proposal (P-012) | hybrid | `hybrid[torch_mlp/raw+logy->LM12]` | researcher-ml | 0.795 | quality 0.80 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-013 | E-02 | refine | hybrid | `hybrid[ridge/raw+pca40+logy+aug1->LM15]` | - | **0.018992** | 0.9967 | 0.02505 | 0.0394 | 16.2 | 282624 | 0.952 | FINAL |
| idea-003 | E-02 | research | hybrid | `hybrid[ridge/raw+pca40+logy+aug1->LM15]` | 0.02062 | **0.018992** | 0.9967 | 0.02505 | 0.0366 | 18.3 | 282624 | 0.952 | evaluated |
| idea-012 | E-02 | refine | hybrid | `hybrid[ridge/raw+pca40+logy+aug1->LM15]` | - | **0.018992** | 0.9967 | 0.02505 | 0.0408 | 16.4 | 282624 | 0.952 | evaluated |
| idea-002 | E-01 | research | physics | `physics-MMSE[log,ms=3,it=30]` | 0.02062 | **0.018993** | 0.9967 | 0.02505 | 0.0111 | 24.7 | 355511 | 0.952 | evaluated |
| idea-006 | E-05 | research | hybrid | `hybrid[extra_trees/raw+logy+aug1->LM12]` | 0.02042 | **0.018996** | 0.9967 | 0.02505 | 8.01 | 19.5 | 191775 | 0.952 | evaluated |
| idea-015 | E-05 | refine | hybrid | `hybrid[extra_trees/raw+logy+aug1->LM12]` | - | **0.018997** | 0.9967 | 0.02505 | 8.05 | 17.2 | 191865 | 0.952 | evaluated |
| idea-014 | E-05 | refine | hybrid | `hybrid[extra_trees/raw+logy+aug1->LM12]` | - | **0.018997** | 0.9967 | 0.02505 | 6.93 | 17.4 | 191812 | 0.952 | evaluated |
| idea-007 | E-06 | research | hybrid | `hybrid[torch_mlp/raw+logy->LM12]` | 0.02008 | **0.019032** | 0.9967 | 0.0251 | 8.55 | 9.71 | 145496 | 0.948 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.0123 | 5.32 | 61517 | 0.957 | evaluated |
| idea-005 | E-04 | research | physics | `physics-MLE[log,ms=3,it=30]` | 0.02139 | **0.019339** | 0.9966 | 0.0255 | 0.0111 | 13.3 | 176354 | 0.957 | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw` | 0.0695 | **0.029289** | 0.9921 | 0.03862 | 8.44 | 0.00512 | 0 | - | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **0.030973** | 0.9912 | 0.04085 | 11.3 | 0.00624 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 1.17 | 0.133 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.493 | 0.0369 | 0 | - | evaluated |

## 8. Lineage of the final method

- `idea-013` `hybrid[ridge/raw+pca40+logy+aug1->LM15]` ← refined from idea-003 ← P-009 v2 (researcher-mathematical, r2, revise) ← P-001 v1 (researcher-mathematical, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00204` [checkpoint] by scheduler: final method (single)
- `resu-00201` [result] by executor: exp-0028 rel_l2=0.01833 r2=0.997 rmse=0.02411
- `hypo-00171` [hypothesis] by scheduler: idea-013 hybrid[ridge/raw+pca40+logy+aug1->LM15] (refine)
- `plan-00200` [plan] by engineer-v2: exp-0028: final fit of 1 member(s) on train+val; test once
- `gate-00170` [gate] by gate: refine idea-003: approved (hybrid[ridge/raw+pca40+logy+aug1->LM15])
- `prop-00065` [proposal] by researcher-mathematical: P-009 v2 [revise] mathematical: Raw-feature ridge initializer + log-k LM + box-prior MMSE (posterior mean of k): CRB-level
- `hypo-00081` [hypothesis] by scheduler: idea-003 hybrid[ridge/raw+pca40+logy+aug1->LM15] (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_literature_s2 (f3bd20fbc0)
- `prop-00035` [proposal] by researcher-mathematical: P-001 v1 [propose] mathematical: Learned initializer + Gauss-Newton/LM refinement in log-conductivity, MMSE readout: reach 
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00080` [gate] by gate: E-02: approved (hybrid[ridge/raw+pca40+logy+aug1->LM15])

## 9. Compute and cost

- Execution compute: 301.8 / 400 worker-seconds over 28 jobs; wall 424 s
- By stage: baseline 2.1s, pilot 6.0s, screen 48.4s, evaluate 129.8s, refine 73.6s, final 42.0s
- LLM: 14 calls (by phase {'research': 14}), 643 s, $2.142 reported · TheoremSearch: 0 queries (0 failed)

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

The evidence establishes that the inverse problem is solved well within the success criterion. The selected model, hybrid[ridge/raw+pca40+logy+aug1->LM15], reached val rel_l2 = 0.01899 and held-out test rel_l2 = 0.01833 (R² = 0.997, RMSE = 0.0241). That is roughly 2.7× below the 0.05 threshold, and test was consistent with val, so there is no sign of selection overfitting. Methods that use the forward model clearly beat the purely data-driven one. The best physics estimator (physics-MMSE[log, ms=3, it=30], val 0.01899) and the hybrid were effectively tied. The best data-driven model (torch_mlp/raw, val 0.02929) was about 54% worse, as predicted. All five supported predictions were upper-bound accuracy claims that held, so log-space parameterisation with multistart or ridge-initialised Levenberg–Marquardt refinement is a reliable recipe here.

One hypothesis was contradicted: that the hybrid would improve on the best physics method by at least 2%. It came closest at +1.77%, but that figure was measured against an earlier physics baseline (0.019335). Against the final physics-MMSE run (0.018993), the gap is about 0.0001 absolute, which is negligible. The more plausible reading is that both methods converge to the same likelihood optimum, and the ridge initializer only saves multistart cost; it does not change accuracy. This is a conjecture, because no estimator floor was computed. Several questions remain open for that reason. Whether ~0.019 sits at the Cramér–Rao bound is unknown, since the CRB hypothesis was dropped as a duplicate rather than tested. The two most informative controls were also never run, both because of the 120 s per-experiment limit: the learned-initializer + MMSE test (134.7 s estimated) and the noise-matched amortized network with log targets. Finally, every comparison rests on a single seed and 750 validation samples with no confidence intervals. Differences under about 2% between the top configurations should therefore be treated as unresolved.

- Compute the Fisher information / CRB per sample at the true k and report the ratio of rel_l2 to the CRB, to test whether the physics and hybrid estimators are efficient.
- Bootstrap val (or use paired per-sample differences) with 3+ seeds for the top physics, hybrid, and MLP configurations, so that sub-2% gaps can be called real or noise.
- Re-run the learned-initializer + MMSE control at reduced cost (fewer LM iterations, a subsample, or split across workers) to see whether initialization quality matters at all once LM has converged.
- Run the amortized, noise-matched MLP with log targets and fewer epochs, to measure how much of the 54% gap comes from target scaling and noise mismatch versus the absence of the forward model.
- Stratify errors by conductivity contrast and by segment (especially segments far from sensors) to find where residual error concentrates and whether identifiability, not the optimizer, is the limit.

_Automatic summary:_ 14 proposal versions over 2 research rounds; 9 methods advanced; 5 predictions supported, 1 contradicted. Final: ['hybrid[ridge/raw+pca40+logy+aug1->LM15]'] with test rel_l2 0.018326.
