# Research report (V2): heat_v2_no_critic_s2

Run `heat_v2_no_critic_s2_20261109_110702` · 2026-11-09 11:10:34 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 48.2 (max 2.32e+03); Cramér–Rao relative std per parameter: k1 0.0223, k2 0.0062, k3 0.0051, k4 0.0179; implied rel_l2 floor ≈ **0.01856**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `physics-MMSE[log,ms=3,it=25]`
- Validation rel_l2 (selection estimate): **0.018999** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018405**, r2 = **0.99697**, rmse = **0.024218** · bootstrap 95% CI for rel_l2: [0.01732, 0.01953]
- Per-target test rel_l2: k1 0.0270, k2 0.0061, k3 0.0052, k4 0.0242
- Relative to the Cramér–Rao floor estimate: 0.99× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.1155 (vs 0.018405 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051227 (vs 0.018405 in distribution)
- Uncertainty calibration on test: 68.6% within 1σ, 95.2% within 2σ (ideal 68% / 95%)
- Inference latency: 36.1 ms/sample; fit 0.00801 s; exported model reload check: max |Δ| = 0.00255
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s2\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s2")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **physics** `physics-MMSE[log,ms=3,it=25]` rel_l2=0.018999; **hybrid** `hybrid[kernel_ridge/raw+logy->LM15]` rel_l2=0.01903; **data-driven** `torch_mlp/raw` rel_l2=0.030973
- SUPPORTED — E-01 *Learned initializer + log-space Gauss-Newton/LM refinement reaches near the Cramér–Rao floor*: val rel_l2 = 0.019337 vs predicted < 0.035
- SUPPORTED — E-02 *Bayesian MMSE (posterior mean under the box prior) beats the MLE on rel_l2 because CRB-limited segments k1 and k4 benef…*: val rel_l2 = 0.018999 vs predicted < 0.03
- SUPPORTED — E-04 *Learned initializer + LM refinement with posterior-mean estimator reaches the CRB floor*: val rel_l2 = 0.01903 vs predicted < 0.03
- SUPPORTED — E-05 *Posterior-mean (MMSE) under the box prior beats the MLE on the poorly determined end segments*: val rel_l2 0.018999 vs best_data_driven 0.030973: gain +38.66%; expected gain >= +8.0% vs best_data_driven
- SUPPORTED — E-06 *Learned initializer plus Gauss-Newton/LM refinement reaches the Cramér–Rao floor*: val rel_l2 = 0.01934 vs predicted < 0.03
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01856

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24993830` Marco Iglesias, Clint Dawson (2013). "The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs,". Theorem 1 (Hanke \cite {Hanke}). http://arxiv.org/abs/1302.3501v1 — cited by ['P-001']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:25601259` Stefania Bellavia, Benedetta Morini, Elisa Riccietti (2015). "On an adaptive regularization for ill-posed nonlinear systems and its trust-region implementation". Theorem. http://arxiv.org/abs/1504.03442v1 — cited by ['P-001', 'P-002']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:17785646` Rafaela Filippozzi, Everton Boos, Douglas S. Gonçalves et al. (2025). "On the regularization property of Levenberg-Marquardt method with Singular Scaling for nonlinear inverse problems". Theorem 4. http://arxiv.org/abs/2506.00190v1 — cited by ['P-002']; researcher claim ['unknown']; harness screen **plausible**
- `ts:thm:26986930` MohammadJava Taheri, Abolfazl Diyanat, MortezaAli Ahmadi et al. (2025). "Predicting Drive Test Results in Mobile Networks Using Optimization Techniques". Lemma 1. http://arxiv.org/abs/2502.09305v1 — cited by ['P-003', 'P-005']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:27031792` Shenghua Hu, Guangyang Zeng, Wenchao Xue et al. (2025). "Theoretical Guarantees for AOA-based Localization: Consistency and Asymptotic Efficiency". Theorem 2. http://arxiv.org/abs/2507.07647v2 — cited by ['P-003', 'P-006']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:25959707` Guangyang Zeng, Biqiang Mu, Ling Shi et al. (2023). "Consistent and Asymptotically Efficient Localization from Range-Difference Measurements". Lemma 4 ({\cite [Theorem 4.3]{lehmann2006theory}}). http://arxiv.org/abs/2302.03311v3 — cited by ['P-005']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22000821` Wolfgang Arendt, Dominik Dier, Hafida Laasri et al. (2014). "Maximal Regularity for Evolution Equations Governed by Non-Autonomous Forms". Corollary 5.2. http://arxiv.org/abs/1303.1166v2 — cited by ['P-007']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:21621221` Guy Barles, Erwin Topp (2015). "Lipschitz Regularity for Censored Subdiffusive Integro-Differential Equations with Superfractional Gradient Terms". Lemma 4.10. http://arxiv.org/abs/1412.2621v2 — cited by ['P-008']; researcher claim ['no']; harness screen **partial**; conflicts: theorem uses final-time (full spatial) data; the problem observes point sensors over time

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-03 *A pure amortised MLP in log-k space with noise-matched augmentation can beat the 0.05 target without per-sample solves*: inconclusive — method did not produce a full-fidelity validation result
- P-007 *Skip learning entirely: likelihood-based MMSE through the forward model, in log-conductivity coordinates* (conjectured): proposed but not executed
- P-008 *Amortised-then-exact: an MLP on log features as a global initialiser, LM posterior-mean refinement as the corrector* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- PRUNED — idea-004 `torch_mlp/raw+logy+aug2`: projected full-fidelity cost 155s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Learned initializer + log-space Gauss-Newton/LM refinement …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-02 (Bayesian MMSE (posterior mean under the box prior) beats th…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-03 (A pure amortised MLP in log-k space with noise-matched augm…): question: the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-03 (A pure amortised MLP in log-k space with noise-matched augm…): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-04 (Learned initializer + LM refinement with posterior-mean est…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-05 (Posterior-mean (MMSE) under the box prior beats the MLE on …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- E-06 (Learned initializer plus Gauss-Newton/LM refinement reaches…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?
- Test-set sampling uncertainty: bootstrap 95% CI [0.01732, 0.01953] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **1 round(s)**: iteration cap reached (max_rounds=1); unresolved issues are documented.

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

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.931 | fdbk-00043 |

### Idea evolution

**P-001 — Learned initializer + log-space Gauss-Newton/LM refinement reaches near the Cramér–Rao floor** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM10]` status literature-supported. *H:* The map theta -> F(theta) is smooth and locally well-conditioned (median Fisher condition 48), with noise sigma=1e-3. So a few LM steps started from a cheap data-driven estimate converge to the per-sample MLE. The MLE's error should approach the CRB rel_l2 fl…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If rel_l2 stays near the init, the learned initializer is outside the LM basin or the forward model is mismatched with the data. In that case increase extra_st…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-002 — Bayesian MMSE (posterior mean under the box prior) beats the MLE on rel_l2 because CRB-limited segments k1 and k4 benef…** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* rel_l2 is a squared-error-type loss. The prior is known exactly (i.i.d. uniform on [0.5, 2]). So the posterior mean is the Bayes-optimal estimator and is not bound by the unbiased CRB. Pure physics LM in log space with multistart 3, followed by importance-sam…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If mode and posterior_mean tie, the posterior is so concentrated that the CRB/MLE is effectively Bayes-optimal and the hybrid approach (proposal 1) is the chea…
  - critic r1 (quality 1.0): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer + LM refinement with posterior-mean estimator reaches the CRB floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status literature-supported. *H:* A kernel-ridge inverse map in log-k space, followed by bounded Levenberg-Marquardt through the forward model with a posterior-mean (MMSE) readout, will reach val rel_l2 within about 1.5x of the harness CRB floor (0.0186), i.e. <= 0.03. The refinement step, no…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the error stays near the initializer's level, LM is not converging: there may be a forward-model mismatch with the data simulator or a Jacobian or budget pr…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-004 — A pure amortised MLP in log-k space with noise-matched augmentation can beat the 0.05 target without per-sample solves** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* Because the map from 4 conductivities to sensor traces is smooth and low-dimensional, a torch MLP trained on 3000 simulated pairs will reach val rel_l2 < 0.05. It should use log targets, standardized raw features and noise augmentation matched to sigma=0.001.…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 > 0.05 with errors concentrated in k1/k4, the learned map is data-limited on weakly identified directions. That argues for simulating more training d…
  - critic r1 (quality 0.85): the physics floor (CRB rel_l2 ~0.01856) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Learned initializer plus Gauss-Newton/LM refinement reaches the Cramér–Rao floor** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy+aug1->LM12]` status literature-supported. *H:* If the simulator that generated the data matches context.forward, then a cheap learned initializer (kernel ridge on standardized raw sensor traces) followed by about 12 bounded LM iterations will give an approximate MLE. Its val rel_l2 should be within 1.5x o…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the chi-square is large, the forward model and data generator differ (e.g. time-step, sensor indexing or initial condition), so physics refinement cannot he…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-006 — Posterior-mean (MMSE) under the box prior beats the MLE on the poorly determined end segments** (researcher-experimental, family physics)
- v1 `P-006` r1 [propose, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* Because rel_l2 is a squared-error metric and the prior is a bounded uniform box, the importance-sampled posterior mean should beat the LS mode, especially on k1 and k4. Prediction: at least 8% lower val rel_l2 than the best data-driven method and below 0.025 …
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.08, "expect": "better"}`; if it fails: If the posterior mean is no better than the mode, the estimator choice is irrelevant at this noise level and remaining error comes from optimization failures o…
  - critic r1 (quality 1.0): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-007 — Skip learning entirely: likelihood-based MMSE through the forward model, in log-conductivity coordinates** (researcher-exploratory, family physics)
- v1 `P-007` r1 [propose, llm] `physics-MMSE[log,ms=3,it=25]` status conjectured. *H:* Training on simulated pairs is not needed to reach the target. Per-sample inference through the known forward model, using log(k) coordinates and the posterior-mean (MMSE) estimator under the uniform box prior, will reach val rel_l2 < 0.03. That is within abo…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the result is far above the CRB floor (> 0.05), the likely causes are a mismatch between the simulator and the data, or multimodality. A per-sample residual…
  - critic r1 (quality 0.6): executes exactly the same method as P-006 (researcher-experimental); differentiate the hypothesis or merge · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?

**P-008 — Amortised-then-exact: an MLP on log features as a global initialiser, LM posterior-mean refinement as the corrector** (researcher-exploratory, family hybrid)
- v1 `P-008` r1 [propose, llm] `hybrid[torch_mlp/log+logy+aug1->LM10]` status conjectured. *H:* A purely data-driven regressor (torch MLP on log-temperature features, with log targets) will stall well above the CRB floor, at val rel_l2 around 0.04-0.08, because 3000 samples are too few to learn the inverse map to noise-limited precision in 4D. Using tha…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.4, "expect": "better"}`; if it fails: (a) If the hybrid does not beat 0.03 but proposal 1 does, the learned initialiser is placing LM in wrong basins, which points to a systematic MLP bias near the…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0223, k4=0.0179): does your estimator exploit the prior bounds for them?


## 5. Literature retrieval (TheoremSearch)

8 queries, 0 failures, 25 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-exploratory | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search (cache) |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for identification of piecewise constant diffusion coe… | 4 refs via /search |
| researcher-exploratory | 1 | local convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares para… | 4 refs via /search |
| researcher-experimental | 1 | convergence and bias of Gauss-Newton or maximum likelihood estimator for nonlinear least … | 4 refs via /search |
| researcher-mathematical | 1 | convergence of Levenberg-Marquardt or Gauss-Newton for finite-dimensional parameter ident… | 4 refs via /search |
| researcher-ml | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_critic_s2.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 71 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | hybrid | `hybrid[kernel_ridge/raw+logy->LM10]` | researcher-mathematical | 1 | best-rated proposal in family 'hybrid' (quality 1.00) |
| E-02 | proposal (P-002) | physics | `physics-MMSE[log,ms=3,it=25]` | researcher-mathematical | 1 | best-rated proposal in family 'physics' (quality 1.00) |
| E-03 | proposal (P-004) | data-driven | `torch_mlp/raw+logy+aug2` | researcher-ml | 0.85 | best-rated proposal in family 'data-driven' (quality 0.85) |
| E-04 | proposal (P-003) | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-05 | proposal (P-006) | physics | `physics-MMSE[log,ms=3,it=25]` | researcher-experimental | 1 | quality 1.00 with diversity penalty |
| E-06 | proposal (P-005) | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM12]` | researcher-experimental | 1 | quality 1.00 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-008 *Amortised-then-exact: an MLP on log features as a global initialiser, LM poster…* — portfolio size limit (6); quality 1.00
- not advanced: P-007 *Skip learning entirely: likelihood-based MMSE through the forward model, in log…* — portfolio size limit (6); quality 0.60

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-003 | E-02 | research | physics | `physics-MMSE[log,ms=3,it=25]` | 0.02042 | **0.018999** | 0.9967 | 0.02505 | 0.00964 | 44.4 | 263087 | 0.952 | FINAL |
| idea-006 | E-05 | research | physics | `physics-MMSE[log,ms=3,it=25]` | 0.02042 | **0.018999** | 0.9967 | 0.02505 | 0.00964 | 43.7 | 263087 | 0.952 | evaluated |
| idea-005 | E-04 | research | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | 0.02008 | **0.01903** | 0.9967 | 0.0251 | 0.519 | 29.1 | 146099 | 0.948 | evaluated |
| idea-013 | E-04 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.019033** | 0.9967 | 0.0251 | 0.553 | 20.3 | 146414 | 0.948 | evaluated |
| idea-012 | E-04 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.019034** | 0.9967 | 0.0251 | 0.552 | 20.5 | 146604 | 0.948 | evaluated |
| idea-015 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM10]` | - | **0.019331** | 0.9966 | 0.02549 | 0.54 | 14.1 | 98213 | 0.957 | evaluated |
| idea-014 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM10]` | - | **0.019335** | 0.9966 | 0.0255 | 0.551 | 14.3 | 99905 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.01 | 11 | 61517 | 0.957 | evaluated |
| idea-002 | E-01 | research | hybrid | `hybrid[kernel_ridge/raw+logy->LM10]` | 0.02138 | **0.019337** | 0.9966 | 0.0255 | 0.523 | 20.6 | 99203 | 0.957 | evaluated |
| idea-007 | E-06 | research | hybrid | `hybrid[kernel_ridge/raw+logy+aug1->LM12]` | 0.02138 | **0.01934** | 0.9966 | 0.0255 | 7.34 | 21.2 | 100274 | 0.957 | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.0605 | **0.030973** | 0.9912 | 0.04085 | 39.7 | 0.0166 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `extra_trees/sqrt+pca16` | 0.1766 | **0.10842** | 0.8917 | 0.143 | 1.44 | 0.275 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `extra_trees/log+pca64` | 0.1898 | **0.11491** | 0.8787 | 0.1515 | 0.485 | 0.0773 | 0 | - | evaluated |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw+logy+aug2` | 0.05219 | **-** | - | - | - | - | - | - | projected full-fidelity cost 155s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-003` `physics-MMSE[log,ms=3,it=25]` ← P-002 v1 (researcher-mathematical, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00171` [checkpoint] by scheduler: final method (single)
- `resu-00168` [result] by executor: exp-0025 rel_l2=0.01841 r2=0.997 rmse=0.02422
- `hypo-00051` [hypothesis] by scheduler: idea-003 physics-MMSE[log,ms=3,it=25] (research)
- `plan-00167` [plan] by engineer-v2: exp-0025: final fit of 1 member(s) on train+val; test once
- `gate-00050` [gate] by gate: E-02: approved (physics-MMSE[log,ms=3,it=25])
- `prop-00036` [proposal] by researcher-mathematical: P-002 v1 [propose] mathematical: Bayesian MMSE (posterior mean under the box prior) beats the MLE on rel_l2 because CRB-lim
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_critic_s2 (75a02c865b)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4

## 9. Compute and cost

- Execution compute: 384.7 / 400 worker-seconds over 25 jobs; wall 212 s
- By stage: baseline 2.4s, screen 56.5s, evaluate 199.1s, refine 62.6s, final 64.1s
- LLM: 8 calls (by phase {'research': 8}), 203 s, $0.765 reported · TheoremSearch: 8 queries (0 failed)

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

The evidence establishes that the success criterion was met with margin. The selected model, physics-MMSE[log, ms=3, it=25], reached val rel_l2 = 0.0190 and held-out test rel_l2 = 0.0184 (R² = 0.997, RMSE = 0.0242). Test was slightly better than val, which is consistent with no selection overfitting, since test was used only once. Physics-based inversion clearly beat the best purely data-driven model (torch_mlp/raw, 0.0310), a 38.7% relative gain against a pre-registered expectation of at least 8%. However, the hybrid (kernel-ridge initialisation refined by 15 Levenberg–Marquardt steps, 0.01903) and the pure physics MMSE estimator (0.01900) are statistically indistinguishable on this evidence. The data therefore do not support claiming that the learned prior adds value once a physics-based refinement step is applied. Plausibly, both are approaching a noise-limited floor, but this is conjecture: no Cramér–Rao or posterior-variance bound was computed to confirm it.

No hypothesis was contradicted and none was abandoned, so there are no failures to diagnose. This should be read cautiously, because the supported predictions were loose upper bounds (rel_l2 < 0.03 or < 0.035) that all methods in the physics and hybrid families cleared easily. They confirm feasibility but discriminate weakly between mechanisms. The data-driven family did not fail the success threshold (0.031 ≤ 0.05), but it was dominated. Whether that gap reflects limited training data (n=3000), limited budget (400 worker-seconds, CPU only), or a structural advantage of forward-model inversion is not resolved. Several other points remain uncertain:
- Results appear to come from single runs, with no seed variance or confidence intervals reported.
- Per-segment error was not broken down; conductivities in segments far from sensors, or with small k, are likely less identifiable.
- Robustness to noise levels or sensor placements outside the simulated distribution is untested.

Next experiments:
- **Uncertainty on the headline numbers:** Repeat the top three configurations over ≥5 seeds and bootstrap the val/test sets to put confidence intervals on the 0.0190 vs 0.01903 vs 0.0193 differences.
- **Per-segment error:** Report rel_l2 and bias for each of the four conductivities, and compare against the Fisher-information (CRLB) bound from the forward model to test the noise-floor conjecture.
- **Noise and misspecification sweep:** Vary the sensor noise σ and perturb the sensor locations and boundary conditions to see where physics-MMSE degrades relative to the hybrid and MLP.
- **Data-scaling ablation for the MLP:** Train on 1k, 3k, and 10k simulated samples to determine whether the 0.031 gap is data-limited or structural.
- **Calibration of the MMSE posterior:** Check the coverage of its credible intervals, since a calibrated uncertainty estimate is the main practical advantage it would hold over the hybrid point estimate.

_Automatic summary:_ 8 proposal versions over 1 research rounds; 9 methods advanced; 5 predictions supported, 0 contradicted. Final: ['physics-MMSE[log,ms=3,it=25]'] with test rel_l2 0.018405.
