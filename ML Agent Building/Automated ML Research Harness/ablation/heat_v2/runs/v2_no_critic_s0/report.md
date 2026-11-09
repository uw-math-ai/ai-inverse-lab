# Research report (V2): heat_v2_no_critic_s0

Run `heat_v2_no_critic_s0_20261109_105558` · 2026-11-09 11:00:53 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: AVAILABLE via context.forward(theta) | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

**Numerical identifiability (harness-computed, 32 prior draws).** Fisher condition number median 46.6 (max 1.78e+03); Cramér–Rao relative std per parameter: k1 0.0209, k2 0.0063, k3 0.0051, k4 0.0161; implied rel_l2 floor ≈ **0.01949**; least determined: k1, k4.

## 2. Outcome

- Final method: **single** of `hybrid[torch_mlp/raw+logy+aug1->LM12]`
- Validation rel_l2 (selection estimate): **0.018947** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.018251**, r2 = **0.99702**, rmse = **0.024014** · bootstrap 95% CI for rel_l2: [0.01713, 0.01928]
- Per-target test rel_l2: k1 0.0267, k2 0.0060, k3 0.0052, k4 0.0240
- Relative to the Cramér–Rao floor estimate: 0.94× (the floor is an estimate at random prior draws, not at the test parameters)
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.097548 (vs 0.018251 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.051178 (vs 0.018251 in distribution)
- Uncertainty calibration on test: 68.5% within 1σ, 95.1% within 2σ (ideal 68% / 95%)
- Inference latency: 13.9 ms/sample; fit 26.8 s; exported model reload check: max |Δ| = 0
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s0\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_critic_s0")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **hybrid** `hybrid[torch_mlp/raw+logy+aug1->LM12]` rel_l2=0.018947; **physics** `physics-MMSE[log,ms=4,it=30]` rel_l2=0.019047; **data-driven** `torch_mlp/raw` rel_l2=0.029522
- SUPPORTED — E-01 *Hybrid: learned initializer + few LM steps through forward model reaches near-CRB accuracy*: val rel_l2 = 0.019335 vs predicted < 0.035
- SUPPORTED — E-03 *Posterior-mean (MMSE) estimation under box prior beats MLE in rel_l2 for poorly-determined end segments*: val rel_l2 = 0.019047 vs predicted < 0.03
- SUPPORTED — E-05 *Framing challenge: the inverse map is nearly linear in (log observation, log k) coordinates, so a forward-model-free ke…*: val rel_l2 = 0.029661 vs predicted < 0.05
- SUPPORTED — E-06 *Learned initializer + LM refinement reaches the Cramér–Rao regime; MMSE beats MLE on the weak parameters k1 and k4*: val rel_l2 = 0.019048 vs predicted < 0.035
- Identifiability diagnostics (numerical): full-rank Fisher information; CRB floor ≈ 0.01949

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24993830` Marco Iglesias, Clint Dawson (2013). "The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs,". Theorem 1 (Hanke \cite {Hanke}). http://arxiv.org/abs/1302.3501v1 — cited by ['P-001']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:25601259` Stefania Bellavia, Benedetta Morini, Elisa Riccietti (2015). "On an adaptive regularization for ill-posed nonlinear systems and its trust-region implementation". Theorem. http://arxiv.org/abs/1504.03442v1 — cited by ['P-001']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-005']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:17785646` Rafaela Filippozzi, Everton Boos, Douglas S. Gonçalves et al. (2025). "On the regularization property of Levenberg-Marquardt method with Singular Scaling for nonlinear inverse problems". Theorem 4. http://arxiv.org/abs/2506.00190v1 — cited by ['P-002']; researcher claim ['unknown']; harness screen **plausible**
- `ts:thm:22914204` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Theorem 7. http://arxiv.org/abs/1003.5004v1 — cited by ['P-003']; researcher claim ['partial']; harness screen **plausible**
- `ts:thm:22914184` O. P. Ferreira, M. L. N. Goncalves, P. R. Oliveira (2010). "Local convergence analysis of Gauss-Newton's method under majorant condition". Corollary 8. http://arxiv.org/abs/1003.5004v1 — cited by ['P-003']; researcher claim ['no']; harness screen **plausible**
- `ts:thm:26986930` MohammadJava Taheri, Abolfazl Diyanat, MortezaAli Ahmadi et al. (2025). "Predicting Drive Test Results in Mobile Networks Using Optimization Techniques". Lemma 1. http://arxiv.org/abs/2502.09305v1 — cited by ['P-005', 'P-006']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:21098230` Andrea Bonito, Albert Cohen, Ronald DeVore et al. (2016). "Diffusion Coefficients Estimation for Elliptic Partial Differential Equations". Corollary 3.8. http://arxiv.org/abs/1609.05231v2 — cited by ['P-006', 'P-008']; researcher claim ['no']; harness screen **no**; conflicts: different equation class (theorem is not about parabolic/diffusion equations); theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-007']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-02 *Control: a purely amortized learned inverse (no forward model at inference) stalls above the target because sensor geom…*: inconclusive — method did not produce a full-fidelity validation result
- E-04 *Learned initializer + Gauss-Newton/LM refinement with posterior-mean estimator approaches the Cramér-Rao floor*: inconclusive — method did not produce a full-fidelity validation result
- P-004 *A purely amortized MLP with log-target and noise augmentation reaches the 0.05 target without physics at inference* (conjectured): proposed but not executed
- P-007 *Hybrid amortized-init + LM with posterior-mean (MMSE) estimator in log-k space reaches near-CRB accuracy* (literature-supported): proposed but not executed

### 3.4 Failed hypotheses and negative results

- PRUNED — idea-003 `torch_mlp/raw+logy+aug2`: projected full-fidelity cost 173s exceeds the 120s per-experiment limit
- PRUNED — idea-005 `hybrid[torch_mlp/raw+logy+aug2->LM15]`: projected full-fidelity cost 121s exceeds the 120s per-experiment limit

### 3.5 Remaining uncertainty

- E-01 (Hybrid: learned initializer + few LM steps through forward …): challenge: the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until tested
- E-01 (Hybrid: learned initializer + few LM steps through forward …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- E-02 (Control: a purely amortized learned inverse (no forward mod…): question: the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-02 (Control: a purely amortized learned inverse (no forward mod…): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-03 (Posterior-mean (MMSE) estimation under box prior beats MLE …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- E-04 (Learned initializer + Gauss-Newton/LM refinement with poste…): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- E-05 (Framing challenge: the inverse map is nearly linear in (log…): question: the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?
- E-05 (Framing challenge: the inverse map is nearly linear in (log…): revision_direction: consider whether the forward model could correct this learner (e.g. as an initializer for LM) — a hybrid tests the 'approximation error' explanation directly
- E-06 (Learned initializer + LM refinement reaches the Cramér–Rao …): question: the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?
- Test-set sampling uncertainty: bootstrap 95% CI [0.01713, 0.01928] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

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
| 1 | 8 | 0.975 | fdbk-00043 |

### Idea evolution

**P-001 — Hybrid: learned initializer + few LM steps through forward model reaches near-CRB accuracy** (researcher-mathematical, family hybrid)
- v1 `P-001` r1 [propose, llm] `hybrid[kernel_ridge/raw+logy->LM15]` status literature-supported. *H:* A kernel-ridge initializer on raw sensor traces lands within the basin of attraction of the MLE (log-transformed k), and 10-15 bounded LM iterations through F then drive per-sample error to near the Cramér–Rao floor, giving val/test rel_l2 < 0.035 (floor ~0.0…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If error stays well above CRB despite converged residuals, the likelihood is multimodal or flat for k1/k4 and the MMSE estimator (proposal 2) is needed; if res…
  - critic r1 (quality 1.0): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-002 — Posterior-mean (MMSE) estimation under box prior beats MLE in rel_l2 for poorly-determined end segments** (researcher-mathematical, family physics)
- v1 `P-002` r1 [propose, llm] `physics-MMSE[log,ms=4,it=30]` status conjectured. *H:* Because rel_l2 is a squared-error metric and k1,k4 have the largest posterior spread, the posterior mean under the uniform box prior (importance-sampled Laplace) has lower expected rel_l2 than the MLE/mode; physics-only multistart LM with estimator=posterior_…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If mode and posterior_mean tie, the posterior is near-Gaussian and the MLE is already efficient — prefer the cheaper hybrid. If multistart physics is far worse…
  - critic r1 (quality 0.95): noise model (from the simulator): additive i.i.d. Gaussian on every sensor reading; the likelihood must match it · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-003 — Learned initializer + Gauss-Newton/LM refinement with posterior-mean estimator approaches the Cramér-Rao floor** (researcher-ml, family hybrid)
- v1 `P-003` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug2->LM15]` status literature-supported. *H:* A torch MLP trained on simulated data, with noise augmentation, gives a starting point inside the local convergence basin of bounded LM through the forward model for nearly every sample. A few LM iterations in log-k space, followed by importance-sampled poste…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If the error stays well above the CRB, one of three things is happening: the initializer often lands outside the LM basin (check by per-sample residual chi^2 v…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-004 — A purely amortized MLP with log-target and noise augmentation reaches the 0.05 target without physics at inference** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `torch_mlp/raw+logy+aug3` status conjectured. *H:* The inverse map from sensor time series to log k is smooth and low-dimensional: 4 parameters, a well-conditioned Fisher matrix, and noise of sigma = 0.001 against signals of order 0.1–1. Under that assumption, a torch MLP trained on 3000 simulated pairs, with…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If rel_l2 >= 0.05 while the hybrid succeeds, amortization error, not information, is the bottleneck. That justifies using physics refinement for the final subm…
  - critic r1 (quality 0.9): the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-005 — Learned initializer + LM refinement reaches the Cramér–Rao regime; MMSE beats MLE on the weak parameters k1 and k4** (researcher-experimental, family hybrid)
- v1 `P-005` r1 [propose, llm] `hybrid[torch_mlp/raw+logy+aug1->LM12]` status literature-supported. *H:* A hybrid estimator gets val rel_l2 <= 0.035, within about 1.8x of the CRB floor of 0.0195. It uses a torch_mlp initializer trained on simulated (theta, noisy observation) pairs, followed by bounded Levenberg–Marquardt in log-k space through the exact forward …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.035}`; if it fails: If chi^2 per datum is about 1 but rel_l2 exceeds 0.035, the gap is intrinsic. The CRB is then optimistic because of nonlinearity or bound effects, and we shoul…
  - critic r1 (quality 1.0): the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-006 — Control: a purely amortized learned inverse (no forward model at inference) stalls above the target because sensor geom…** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The best amortized data-driven model will reach val rel_l2 of at least 1.4x the best hybrid. We expect this model to be a noise-augmented torch_mlp on log-k targets trained on the 3000 simulated samples, and it should land in the range 0.03–0.06. Its excess e…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_hybrid", "improvement": 0.4, "expect": "worse"}`; if it fails: If the amortized model matches the hybrid, the inverse map is learnable from 3000 samples. The cheaper amortized estimator then wins on cost, and LM adds nothi…
  - critic r1 (quality 1.0): the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?

**P-007 — Hybrid amortized-init + LM with posterior-mean (MMSE) estimator in log-k space reaches near-CRB accuracy** (researcher-exploratory, family hybrid)
- v1 `P-007` r1 [propose, llm] `hybrid[kernel_ridge/log+logy+aug1->LM25]` status literature-supported. *H:* A learned initializer (kernel ridge on log-features, log targets) followed by bounded LM refinement in log-conductivity space, reporting the box-prior posterior mean rather than the mode, will reach test rel_l2 <= 0.03. That is within about 1.5x the CRB floor…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.03}`; if it fails: If rel_l2 > 0.03 but mode ≈ posterior_mean, the bottleneck is basin capture or convergence. Inspect the residual chi^2 distribution: bimodal residuals indicate…
  - critic r1 (quality 1.0): the proposal uses proof language ('proved'/'formally verified'); retrieval of a related theorem is not a proof of this claim — treated as conjecture until test… · the least determined parameters are ['k1', 'k4'] (CRB rel std k1=0.0209, k4=0.0161): does your estimator exploit the prior bounds for them?

**P-008 — Framing challenge: the inverse map is nearly linear in (log observation, log k) coordinates, so a forward-model-free ke…** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `kernel_ridge/log+logy+aug2` status conjectured. *H:* In 1-D diffusion, sensor temperatures depend on conductivities mainly through decay rates and transit times that scale like exp(-c k t) and 1/k. After log-transforming the features and targets, the inverse map becomes smooth and close to low-order. A purely d…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If val rel_l2 lands between 0.05 and 0.1, amortization error dominates. A hybrid refinement such as proposal 1 is then needed, and this model is still useful a…
  - critic r1 (quality 0.95): the physics floor (CRB rel_l2 ~0.01949) is known: what error do you expect relative to it, and what limits a learned inverse here (samples, capacity, noise)?


## 5. Literature retrieval (TheoremSearch)

8 queries, 0 failures, 28 distinct statements retrieved, 0 skipped for budget; check complete: **True**.

| agent | round | query | result |
|---|---|---|---|
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search (cache) |
| researcher-experimental | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search (cache) |
| researcher-experimental | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for identifying piecewise constant diffusion coefficie… | 4 refs via /search |
| researcher-ml | 1 | convergence of Gauss-Newton or Levenberg-Marquardt for nonlinear least squares parameter … | 4 refs via /search |
| researcher-exploratory | 1 | Lipschitz stability for recovering piecewise constant conductivity coefficient in parabol… | 4 refs via /search |
| researcher-mathematical | 1 | convergence of Levenberg-Marquardt or Gauss-Newton for finite-dimensional parameter ident… | 4 refs via /search |
| researcher-exploratory | 1 | uniqueness of diffusion coefficient from boundary or point observation of heat equation v… | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_critic_s0.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 74 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-001) | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | researcher-mathematical | 1 | best-rated proposal in family 'hybrid' (quality 1.00) |
| E-02 | proposal (P-006) | data-driven | `torch_mlp/raw+logy+aug2` | researcher-experimental | 1 | best-rated proposal in family 'data-driven' (quality 1.00) |
| E-03 | proposal (P-002) | physics | `physics-MMSE[log,ms=4,it=30]` | researcher-mathematical | 0.95 | best-rated proposal in family 'physics' (quality 0.95) |
| E-04 | proposal (P-003) | hybrid | `hybrid[torch_mlp/raw+logy+aug2->LM15]` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-05 | proposal (P-008) | data-driven | `kernel_ridge/log+logy+aug2` | researcher-exploratory | 0.95 | quality 0.95 with diversity penalty |
| E-06 | proposal (P-005) | hybrid | `hybrid[torch_mlp/raw+logy+aug1->LM12]` | researcher-experimental | 1 | quality 1.00 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
| E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | scheduler | - | conventional optimisation through the forward model |
- not advanced: P-007 *Hybrid amortized-init + LM with posterior-mean (MMSE) estimator in log-k space …* — portfolio size limit (6); quality 1.00
- not advanced: P-004 *A purely amortized MLP with log-target and noise augmentation reaches the 0.05 …* — portfolio size limit (6); quality 0.90

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-013 | E-06 | refine | hybrid | `hybrid[torch_mlp/raw+logy+aug1->LM12]` | - | **0.018947** | 0.9967 | 0.02499 | 91.3 | 10.4 | 160518 | 0.951 | FINAL |
| idea-004 | E-03 | research | physics | `physics-MMSE[log,ms=4,it=30]` | 0.02024 | **0.019047** | 0.9967 | 0.02512 | 0.0101 | 30.4 | 322430 | 0.95 | evaluated |
| idea-007 | E-06 | research | hybrid | `hybrid[torch_mlp/raw+logy+aug1->LM12]` | 0.02024 | **0.019048** | 0.9967 | 0.02512 | 36.2 | 19.6 | 190087 | 0.95 | evaluated |
| idea-012 | E-06 | refine | hybrid | `hybrid[torch_mlp/raw+logy+aug1->LM12]` | - | **0.019051** | 0.9967 | 0.02512 | 91.5 | 12.1 | 192373 | 0.95 | evaluated |
| idea-014 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.01911** | 0.9966 | 0.0252 | 0.461 | 13.7 | 146170 | 0.946 | evaluated |
| idea-015 | E-01 | refine | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | - | **0.019333** | 0.9966 | 0.02549 | 0.461 | 10.3 | 101480 | 0.957 | evaluated |
| idea-002 | E-01 | research | hybrid | `hybrid[kernel_ridge/raw+logy->LM15]` | 0.02138 | **0.019335** | 0.9966 | 0.0255 | 0.623 | 10.1 | 101336 | 0.957 | evaluated |
| idea-011 | E-09 | baseline | physics | `physics-MLE[log,ms=1,it=30]` | 0.02139 | **0.019335** | 0.9966 | 0.0255 | 0.00974 | 6.61 | 61517 | 0.957 | evaluated |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06289 | **0.029522** | 0.992 | 0.03893 | 14.1 | 0.00978 | 0 | - | evaluated |
| idea-006 | E-05 | research | data-driven | `kernel_ridge/log+logy+aug2` | 0.04575 | **0.029661** | 0.9919 | 0.03912 | 10.5 | 0.216 | 0 | - | evaluated |
| idea-010 | E-08 | baseline | data-driven | `kernel_ridge/raw` | 0.05292 | **0.032419** | 0.9903 | 0.04275 | 0.604 | 0.0694 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `knn/sqrt` | 0.2036 | **0.14038** | 0.8184 | 0.1851 | 0.00911 | 0.0835 | 0 | - | evaluated |
| idea-003 | E-02 | research | data-driven | `torch_mlp/raw+logy+aug2` | 0.04789 | **-** | - | - | - | - | - | - | projected full-fidelity cost 173s exceeds the 120s per-experiment lim… |
| idea-005 | E-04 | research | hybrid | `hybrid[torch_mlp/raw+logy+aug2->LM15]` | 0.02024 | **-** | - | - | - | - | - | - | projected full-fidelity cost 121s exceeds the 120s per-experiment lim… |

## 8. Lineage of the final method

- `idea-013` `hybrid[torch_mlp/raw+logy+aug1->LM12]` ← refined from idea-007 ← P-005 v1 (researcher-experimental, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00168` [checkpoint] by scheduler: final method (single)
- `resu-00165` [result] by executor: exp-0024 rel_l2=0.01825 r2=0.997 rmse=0.02401
- `hypo-00135` [hypothesis] by scheduler: idea-013 hybrid[torch_mlp/raw+logy+aug1->LM12] (refine)
- `plan-00164` [plan] by engineer-v2: exp-0024: final fit of 1 member(s) on train+val; test once
- `gate-00134` [gate] by gate: refine idea-007: approved (hybrid[torch_mlp/raw+logy+aug1->LM12])
- `prop-00039` [proposal] by researcher-experimental: P-005 v1 [propose] experimental: Learned initializer + LM refinement reaches the Cramér–Rao regime; MMSE beats MLE on the w
- `hypo-00059` [hypothesis] by scheduler: idea-007 hybrid[torch_mlp/raw+logy+aug1->LM12] (research)
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_critic_s0 (105283f344)
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `gate-00058` [gate] by gate: E-06: approved (hybrid[torch_mlp/raw+logy+aug1->LM12])

## 9. Compute and cost

- Execution compute: 491.2 / 400 worker-seconds over 24 jobs; wall 295 s
- By stage: baseline 2.2s, screen 76.8s, evaluate 132.2s, refine 227.9s, final 52.1s
- LLM: 8 calls (by phase {'research': 8}), 198 s, $0.755 reported · TheoremSearch: 8 queries (0 failed)

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

The evidence establishes that the objective is met with margin. The selected hybrid model (MLP initialization from raw sensor traces with log-target and augmentation, refined by 12 Levenberg–Marquardt iterations against the forward solver) reached val rel_l2 = 0.0189 and held-out test rel_l2 = 0.0183 (R² = 0.997, RMSE = 0.024). That is well under the 0.05 threshold. The test score is in line with validation, which suggests selection did not overfit the validation split. All three method families cleared the bar. The physics-only MMSE estimator (log-parameterized, 4 multistarts, 30 iterations) scored 0.0190, essentially tied with the hybrid. The purely data-driven MLP scored 0.0295, about 50% higher error than either physics-informed approach. All four pre-registered predictions were supported, for example 0.0193 against a predicted bound of < 0.035, and 0.0297 against < 0.05.

No hypotheses were contradicted or abandoned, so this run reports no failures. That is partly because the predictions were loose upper bounds rather than sharp, falsifiable claims. The most informative result is a near-null one: the hybrid beats pure physics inversion by only about 0.0001 rel_l2 (0.01895 vs 0.01905). This is almost certainly within seed and sample noise, so the data do not show that the learned prior adds accuracy beyond a well-initialized physics fit. A plausible reading, not yet tested, is that the LM refinement drives the final accuracy and the MLP mainly supplies a cheaper or more reliable starting point. Several things remain uncertain:
- Whether 0.019 is near the noise-limited floor for this sensor layout.
- How error is distributed across the four conductivity segments; interior segments far from sensors may be less identifiable.
- Run-to-run variance, since results come from a single seed.
- The relative compute cost of hybrid versus multistart physics inversion.

Next experiments:
- **Seed and bootstrap replicates.** Run 3–5 seeds per family and compute paired bootstrap CIs on per-sample val rel_l2, to test whether the hybrid vs physics gap is real.
- **Noise-floor estimate.** Run LM started from the true k, or compute a Cramér–Rao bound from the forward-model Jacobian, to see how much headroom remains below 0.019.
- **Per-segment error breakdown.** Measure error for each k segment, and correlate it with distance to the sensors and with contrast between adjacent segments, to locate identifiability limits.
- **Compute–accuracy trade-off.** Sweep LM iterations (0, 3, 6, 12) on MLP initializations against multistarts (1, 2, 4) for physics-MMSE, and measure worker-seconds per sample. This isolates what the learned initialization actually buys.
- **Robustness under shift.** Re-evaluate on simulated data with higher noise, perturbed sensor positions, or a mildly misspecified forward model, since the physics-heavy methods may degrade differently from the MLP.

_Automatic summary:_ 8 proposal versions over 1 research rounds; 9 methods advanced; 4 predictions supported, 0 contradicted. Final: ['hybrid[torch_mlp/raw+logy+aug1->LM12]'] with test rel_l2 0.018251.
