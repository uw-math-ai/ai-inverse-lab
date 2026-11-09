# Research report (V2): heat_v2_no_forward_s0

Run `heat_v2_no_forward_s0_20261109_005148` · 2026-11-09 00:57:54 · pipeline **v2 (research-first)** · LLM backend **claude_code** · workers 6 × 2 threads · GPUs none · remote off

## 1. Problem

**Problem.** Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

**Data.** 1-D heat equation u_t = (k(x) u_x)_x on [0,1] with Dirichlet ends; unknown piecewise-constant conductivity k (targets); features are noisy temperatures at sensor points over time. (train/val/test = 3000/750/750; the test split is used once, at the end).

**Problem context available to methods.** forward model: not available | parameters: ['k1', 'k2', 'k3', 'k4'] | bounds: lower=[0.5, 0.5, 0.5, 0.5] upper=[2.0, 2.0, 2.0, 2.0] | noise std: 0.001 (additive i.i.d. Gaussian on every sensor reading) | prior: i.i.d. uniform on [0.5, 2.0] per segment | facts: dimension=1; pde=parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x; coefficient_regularity=piecewise-constant (discontinuous) conductivity, one value per segment; boundary_conditions=homogeneous Dirichlet at both ends; initial_condition=known Gaussian pulse centred at x = 0.5; observations=interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window; discretization=explicit finite differences, 40 grid points, 200 steps; noise=Gaussian, sigma = 0.001; parameter_count=4; positivity=conductivity is strictly positive

## 2. Outcome

- Final method: **ensemble** of `torch_mlp/raw+logy+aug2`, `torch_mlp/log+logy+aug1`, `torch_mlp/raw+logy+aug2`, `torch_mlp/raw+logy+aug1` (weights [2.0, 1.0, 1.0, 1.0])
- Validation rel_l2 (selection estimate): **0.02373** · mean-predictor baseline 0.3307
- **Test (held out, evaluated once):** rel_l2 = **0.022481**, r2 = **0.99546**, rmse = **0.029581** · bootstrap 95% CI for rel_l2: [0.02129, 0.02368]
- Per-target test rel_l2: k1 0.0328, k2 0.0100, k3 0.0092, k4 0.0281
- Success criterion (rel_l2 <= 0.05): **MET**
- Robustness, shift set `k_low_0.3`: rel_l2 = 0.079585 (vs 0.022481 in distribution)
- Robustness, shift set `noise_x3`: rel_l2 = 0.059111 (vs 0.022481 in distribution)
- Inference latency: 0.0147 ms/sample; fit 120 s; exported model reload check: max |Δ| = 0
- Ensemble members on test: `torch_mlp/raw+logy+aug2` 0.02551, `torch_mlp/log+logy+aug1` 0.02576, `torch_mlp/raw+logy+aug2` 0.02674, `torch_mlp/raw+logy+aug1` 0.02459
- Model: `C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0\final\model.pkl` — `from mlharness.export import load_model; m = load_model(r"C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\runs\v2_no_forward_s0")` (manifest records seeds, versions, data hash, lineage)

## 3. What we learned — evidence classes

### 3.1 Established evidence (experiments on the validation split)

- Best full-fidelity validation result per method family: **data-driven** `torch_mlp/raw+logy+aug2` rel_l2=0.025766
- SUPPORTED — E-01 *Log-conductivity MLP inverse map with noise-matched augmentation*: val rel_l2 = 0.030074 vs predicted < 0.05
- SUPPORTED — E-03 *Log-target MLP with noise augmentation learns the smooth 4-D inverse map*: val rel_l2 = 0.026564 vs predicted < 0.05
- SUPPORTED — E-06 *Matched-noise training augmentation*: val rel_l2 0.027873 vs best_data_driven 0.026564: gain -4.93%; expected gain >= -5.0% vs best_data_driven

### 3.2 Literature-supported statements (retrieved, with applicability screen)

- `ts:thm:24153014` Sunghoon Kim, Ki-Ahm Lee (2019). "System of Porous Medium Equations". Lemma 2.1 (Uniqueness of solutions). http://arxiv.org/abs/1812.11007v2 — cited by ['P-001', 'P-007', 'P-009']; researcher claim ['partial']; harness screen **unknown**
- `ts:thm:22000821` Wolfgang Arendt, Dominik Dier, Hafida Laasri et al. (2014). "Maximal Regularity for Evolution Equations Governed by Non-Autonomous Forms". Corollary 5.2. http://arxiv.org/abs/1303.1166v2 — cited by ['P-001', 'P-002', 'P-009', 'P-010']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:17546409` Ali Feizmohammadi (2025). "Reconstruction of 1-D evolution equations and their initial data from one passive measurement". Theorem 1.2. http://arxiv.org/abs/2501.10590v2 — cited by ['P-003', 'P-011']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:25849792` Tan Bui-Thanh, Qin Li, Leonardo Zepeda-Núñez (2021). "Bridging and Improving Theoretical and Computational Electric Impedance Tomography via Data Completion". Theorem 3.4 (\cite {allessandrini_lipschitz,Rondi_e_to_N}). http://arxiv.org/abs/2105.00554v1 — cited by ['P-004', 'P-012']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem uses boundary measurements; the problem uses interior point sensors
- `ts:thm:21098230` Andrea Bonito, Albert Cohen, Ronald DeVore et al. (2016). "Diffusion Coefficients Estimation for Elliptic Partial Differential Equations". Corollary 3.8. http://arxiv.org/abs/1609.05231v2 — cited by ['P-008', 'P-015']; researcher claim ['no']; harness screen **no**; conflicts: different equation class (theorem is not about parabolic/diffusion equations); theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly
- `ts:thm:22398568` Adriano De Cezaro, B. Tomas Johansson (2012). "A note on uniqueness in the identification of a spacewise dependent source and diffusion coefficient for the heat equation". Theorem 5. http://arxiv.org/abs/1210.7346v1 — cited by ['P-009', 'P-010', 'P-011', 'P-012', 'P-015']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses final-time (full spatial) data; the problem observes point sensors over time
- `ts:thm:21223425` Jiang Daijun, Feng Hui, Zou Jun (2016). "Quadratic Convergence of Levenberg-Marquardt Method for Elliptic and Parabolic Inverse Robin Problems". Theorem 2.4. http://arxiv.org/abs/1507.01711v2 — cited by ['P-009', 'P-010', 'P-015']; researcher claim ['partial']; harness screen **partial**; conflicts: theorem assumes a smooth/regular coefficient; the problem's conductivity is piecewise-constant (discontinuous) — the result may not transfer directly; theorem uses Neumann/Robin boundary conditions; problem has Dirichlet

_Retrieval of an existing statement supports reasoning; it is not a proof of any proposal's claim and nothing here is formally verified._

### 3.3 Conjectures (not established by experiment in this run)

- E-02 *Kernel ridge (RBF) as a near-Bayes-optimal smooth interpolant on a 4-D data manifold*: inconclusive — method did not produce a full-fidelity validation result
- E-04 *Kernel ridge on PCA-compressed histories as a smooth-interpolant competitor*: inconclusive — method did not produce a full-fidelity validation result
- E-05 *Thermal-resistance target space: regress r = 1/k instead of k*: inconclusive — method did not produce a full-fidelity validation result
- P-014 *Random-search control arm* (conjectured): proposed but not executed

### 3.4 Failed hypotheses and negative results

- ABANDONED in round 2 by researcher-exploratory — P-007 *Log-conductivity regression with noise-augmented neural net (diffusion-timescale reparametrisation)*: duplicate of P-003
- PRUNED — idea-006 `custom-ml`: not meaningfully better than the mean baseline at screen fidelity (0.372 vs 0.3307)

### 3.5 Remaining uncertainty

- E-06 (Matched-noise training augmentation): revision_direction: state the evidence explicitly
- E-06 (Matched-noise training augmentation): revision_direction: the portfolio is dominated by 'data-driven' methods: branch into an under-represented family
- Literature check INCOMPLETE: 0 failed queries (-), 2 skipped for budget, enabled=True
- Test-set sampling uncertainty: bootstrap 95% CI [0.02129, 0.02368] for rel_l2; one seed, one dataset — between-method differences inside this interval are not established

## 4. Research phase: hypotheses, criticism and revision

Stopped after **2 round(s)**: quality criteria met after 2 rounds (every active proposal >= 0.75 quality, no blocking issues, critic converged).

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
| 2 | researcher-mathematical | revise | P-001 | P-009 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-mathematical | revise | P-002 | P-010 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-ml | revise | P-003 | P-011 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-ml | revise | P-004 | P-012 | heuristic-fallback | Revised: added requested literature with applicability notes |
| 2 | researcher-experimental | revise | P-005 | P-013 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-experimental | revise | P-006 | P-014 | heuristic-fallback | Revised: filled missing fields |
| 2 | researcher-exploratory | abandon | P-007 | - | heuristic-fallback | Accepted: merged into the duplicate. |
| 2 | researcher-exploratory | revise | P-008 | P-015 | heuristic-fallback | Revised: added requested literature with applicability notes |

| round | active proposals | mean critic quality | critique artifact |
|---|---|---|---|
| 1 | 8 | 0.835 | fdbk-00046 |
| 2 | 7 | 0.93 | fdbk-00068 |

### Idea evolution

**P-001 — Log-conductivity MLP inverse map with noise-matched augmentation** (researcher-mathematical, family data-driven)
- v1 `P-001` r1 [propose, llm] `torch_mlp/raw+pca48+logy+aug2` status conjectured. *H:* The map from sensor time series to theta = log k is smooth and injective on the prior box. With 3000 samples in only 4 parameters, a moderately sized MLP regressing log k on standardized, PCA-compressed sensor traces will reach val rel_l2 <= 0.05. Errors will…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If error is uniform across parameters, the limit is model capacity or sample size, so try kernel methods, no PCA, or longer training. If error is concentrated …
  - critic r1 (quality 1.0): 
- v2 `P-009` r2 [revise, heuristic-fallback] `torch_mlp/raw+pca48+logy+aug2` status conjectured. *H:* The map from sensor time series to theta = log k is smooth and injective on the prior box. With 3000 samples in only 4 parameters, a moderately sized MLP regressing log k on standardized, PCA-compressed sensor traces will reach val rel_l2 <= 0.05. Errors will…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If error is uniform across parameters, the limit is model capacity or sample size, so try kernel methods, no PCA, or longer training. If error is concentrated …
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 1.0): 

**P-002 — Kernel ridge (RBF) as a near-Bayes-optimal smooth interpolant on a 4-D data manifold** (researcher-mathematical, family data-driven)
- v1 `P-002` r1 [propose, llm] `kernel_ridge/raw+pca24+logy` status conjectured. *H:* The noise-free sensor traces form a 4-dimensional smooth manifold in feature space. RBF kernel ridge on PCA-compressed standardized features, regressing log k, will match or beat the MLP, with val rel_l2 < 0.06. It will also beat linear ridge by at least a fa…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If KRR is clearly worse than the MLP, the inverse map has anisotropic or multi-scale structure that an isotropic RBF cannot capture. Learned features would the…
  - critic r1 (quality 0.9): 
- v2 `P-010` r2 [revise, heuristic-fallback] `kernel_ridge/raw+pca24+logy` status conjectured. *H:* The noise-free sensor traces form a 4-dimensional smooth manifold in feature space. RBF kernel ridge on PCA-compressed standardized features, regressing log k, will match or beat the MLP, with val rel_l2 < 0.06. It will also beat linear ridge by at least a fa…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.06}`; if it fails: If KRR is clearly worse than the MLP, the inverse map has anisotropic or multi-scale structure that an isotropic RBF cannot capture. Learned features would the…
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 1.0): 

**P-003 — Log-target MLP with noise augmentation learns the smooth 4-D inverse map** (researcher-ml, family data-driven)
- v1 `P-003` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from 4-sensor temperature histories to log k is smooth and low-dimensional. A LayerNorm-GELU MLP on standardised raw features, trained on log-targets with sensor-noise augmentation at the true sigma, will reach val rel_l2 <= 0.05. That is about a 6.6x…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A per-parameter error breakdown separates two causes. Uniformly high errors point to optimisation or capacity, so try a wider net, more epochs or an ensemble. …
  - critic r1 (quality 0.9): 
- v2 `P-011` r2 [revise, heuristic-fallback] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from 4-sensor temperature histories to log k is smooth and low-dimensional. A LayerNorm-GELU MLP on standardised raw features, trained on log-targets with sensor-noise augmentation at the true sigma, will reach val rel_l2 <= 0.05. That is about a 6.6x…
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: A per-parameter error breakdown separates two causes. Uniformly high errors point to optimisation or capacity, so try a wider net, more epochs or an ensemble. …
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 1.0): 

**P-004 — Kernel ridge on PCA-compressed histories as a smooth-interpolant competitor** (researcher-ml, family data-driven)
- v1 `P-004` r1 [propose, llm] `kernel_ridge/raw+pca20+logy` status conjectured. *H:* The sensor histories lie near a low-dimensional manifold, with roughly 20 PCA components capturing nearly all non-noise variance. Kernel ridge (RBF) on 20 PCA components with log targets will reach val rel_l2 <= 0.07, within 1.5x of the best MLP. This would s…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.5, "expect": "worse"}`; if it fails: If KRR is far worse than the MLP, the inverse map has anisotropic or nonlinear structure that an isotropic RBF cannot capture, which argues for learned represe…
  - critic r1 (quality 0.9): 
- v2 `P-012` r2 [revise, heuristic-fallback] `kernel_ridge/raw+pca20+logy` status conjectured. *H:* The sensor histories lie near a low-dimensional manifold, with roughly 20 PCA components capturing nearly all non-noise variance. Kernel ridge (RBF) on 20 PCA components with log targets will reach val rel_l2 <= 0.07, within 1.5x of the best MLP. This would s…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.5, "expect": "worse"}`; if it fails: If KRR is far worse than the MLP, the inverse map has anisotropic or nonlinear structure that an isotropic RBF cannot capture, which argues for learned represe…
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 1.0): 

**P-005 — Matched-noise training augmentation** (researcher-experimental, family data-driven)
- v1 `P-005` r1 [propose, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - critic r1 (quality 0.756): 
- v2 `P-013` r2 [revise, heuristic-fallback] `torch_mlp/log+logy+aug1` status conjectured. *H:* Training with re-noised replicas at the sensor noise level regularises the learned inverse and makes it more robust to measurement noise without hurting accuracy.
  - prediction: `{"relative_to": "best_data_driven", "improvement": -0.05, "expect": "better"}`; if it fails: the clean-noise regime does not benefit from augmentation
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-006 — Random-search control arm** (researcher-experimental, family data-driven)
- v1 `P-006` r1 [propose, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - critic r1 (quality 0.756): 
- v2 `P-014` r2 [revise, heuristic-fallback] `random-search x2` status conjectured. *H:* If reasoning adds value, reasoned proposals beat uniformly random configurations evaluated under the same budget.
  - prediction: `{"relative_to": "best_other", "improvement": 0.0, "expect": "worse"}`; if it fails: the search space is forgiving; reasoning did not add value at proposal stage
  - response to critic: Revised: filled missing fields
  - critic r2 (quality 0.756): 

**P-007 — Log-conductivity regression with noise-augmented neural net (diffusion-timescale reparametrisation)** (researcher-exploratory, family data-driven)
- v1 `P-007` r1 [propose, llm] `torch_mlp/raw+logy+aug2` status conjectured. *H:* The map from sensor time series to log k is closer to linear and closer to having uniform error than the map to k itself. A torch MLP trained on standardized raw features, with log targets and noise augmentation at the true sensor noise level, will reach val …
  - prediction: `{"metric": "rel_l2", "op": "<", "value": 0.05}`; if it fails: If error stalls well above 0.05 and is dominated by k1 and k4, the limit is information content rather than function approximation, because outer segments are …
  - critic r1 (quality 0.57): executes exactly the same method as P-003 (researcher-ml); differentiate the hypothesis or merge

**P-008 — Thermal-resistance target space: regress r = 1/k instead of k** (researcher-exploratory, family data-driven)
- v1 `P-008` r1 [propose, llm] `custom-ml` status conjectured. *H:* Thermal resistances in series add, and the slow decay modes of a layered rod depend on segment resistances r_i = 1/k_i through near-additive combinations. Regressing r (clipped to [0.5, 2]) and inverting will therefore give a lower val rel_l2 on k than the sa…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If 1/k is no better than identity or log, the observation window is dominated by transient local-diffusion information rather than series-resistance slow modes…
  - critic r1 (quality 0.9): 
- v2 `P-015` r2 [revise, heuristic-fallback] `custom-ml` status conjectured. *H:* Thermal resistances in series add, and the slow decay modes of a layered rod depend on segment resistances r_i = 1/k_i through near-additive combinations. Regressing r (clipped to [0.5, 2]) and inverting will therefore give a lower val rel_l2 on k than the sa…
  - prediction: `{"metric": "rel_l2", "relative_to": "best_data_driven", "improvement": 0.1, "expect": "better"}`; if it fails: If 1/k is no better than identity or log, the observation window is dominated by transient local-diffusion information rather than series-resistance slow modes…
  - response to critic: Revised: added requested literature with applicability notes
  - critic r2 (quality 1.0): 

- Critic round 1 portfolio view:  8/8 active proposals are 'data-driven' families={'data-driven': 8}
- Critic round 2 portfolio view:  7/7 active proposals are 'data-driven' families={'data-driven': 7}

## 5. Literature retrieval (TheoremSearch)

16 queries, 0 failures, 47 distinct statements retrieved, 2 skipped for budget; check complete: **False**.

| agent | round | query | result |
|---|---|---|---|
| researcher-exploratory | 1 | uniqueness and Lipschitz stability for recovering piecewise constant diffusion coefficien… | 4 refs via /search (cache) |
| researcher-exploratory | 1 | conditional stability estimate inverse coefficient problem heat equation finitely many un… | 4 refs via /search |
| researcher-ml | 1 | Lipschitz stability for identification of piecewise constant conductivity in parabolic eq… | 4 refs via /search |
| researcher-experimental | 1 | Lipschitz stability for identification of piecewise constant conductivity coefficient in … | 4 refs via /search |
| researcher-mathematical | 1 | uniqueness and Lipschitz stability for identifying piecewise constant diffusion coefficie… | 4 refs via /search |
| researcher-ml | 1 | uniqueness of piecewise constant diffusion coefficient from finitely many interior point … | 4 refs via /search |
| researcher-experimental | 1 | uniqueness of diffusion coefficient recovery from boundary or interior flux measurements … | 4 refs via /search |
| researcher-mathematical | 1 | conditional stability estimate inverse coefficient problem heat equation finite-dimension… | 4 refs via /search |
| researcher-experimental | 1 | Cramer-Rao lower bound for parameter estimation in partial differential equations | FAILED: budget |
| researcher-experimental | 1 | optimal sensor placement for parameter identification in diffusion equations | FAILED: budget |
| researcher-experimental | 2 | Matched-noise training augmentation | 4 refs via /search (cache) |
| researcher-experimental | 2 | Random-search control arm | 4 refs via /search (cache) |
| researcher-ml | 2 | identifiability piecewise constant conductivity heat equation interior point sensors | 4 refs via /search |
| researcher-mathematical | 2 | identifiability piecewise constant conductivity point measurements heat equation | 4 refs via /search |
| researcher-mathematical | 2 | Lipschitz stability inverse coefficient problem parabolic finitely many parameters | 4 refs via /search |
| researcher-ml | 2 | neural network surrogate inverse map parabolic parameter identification noise augmentation | 4 refs via /search |
| researcher-exploratory | 2 | identifiability of piecewise constant conductivity from interior point measurements heat … | 4 refs via /search |
| researcher-exploratory | 2 | Lipschitz stability inverse coefficient problem parabolic piecewise constant finite dimen… | 4 refs via /search |

## 6. Research memory

- Backend jsonl:C:\Users\kusha\Research Code\UW Math AI Lab\ai-inverse-lab\ML Agent Building\Automated ML Research Harness\ablation\heat_v2\memory\v2_no_forward_s0.jsonl; 0 records from 1 run(s) were available at start; agents retrieved 0 distinct prior-run records; this run wrote 115 records (test numbers redacted for agents).

## 7. Portfolio, gate and experiments

| entry | source | family | method | author | quality | why selected |
|---|---|---|---|---|---|---|
| E-01 | proposal (P-009) | data-driven | `torch_mlp/raw+pca48+logy+aug2` | researcher-mathematical | 1 | best-rated proposal in family 'data-driven' (quality 1.00) |
| E-02 | proposal (P-010) | data-driven | `kernel_ridge/raw+pca24+logy` | researcher-mathematical | 1 | quality 1.00 with diversity penalty |
| E-03 | proposal (P-011) | data-driven | `torch_mlp/raw+logy+aug2` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-04 | proposal (P-012) | data-driven | `kernel_ridge/raw+pca20+logy` | researcher-ml | 1 | quality 1.00 with diversity penalty |
| E-05 | proposal (P-015) | data-driven | `custom-ml` | researcher-exploratory | 1 | quality 1.00 with diversity penalty |
| E-06 | proposal (P-013) | data-driven | `torch_mlp/log+logy+aug1` | researcher-experimental | 0.756 | quality 0.76 with diversity penalty |
| E-07 | baseline | data-driven | `torch_mlp/raw` | scheduler | - | strong conventional learned-inverse default |
| E-08 | baseline | data-driven | `random-search x2` | scheduler | - | basic random search control |
- not advanced: P-014 *Random-search control arm* — portfolio size limit (6); quality 0.76

| idea | entry | origin | family | method | screen rel_l2 | **full val rel_l2** | val r2 | val rmse | fit s | ms/sample | fwd evals | 2σ cover | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idea-011 | E-03 | refine | data-driven | `torch_mlp/raw+logy+aug2` | - | **0.025766** | 0.9939 | 0.03398 | 23.8 | 0 | 0 | - | FINAL |
| idea-004 | E-03 | research | data-driven | `torch_mlp/raw+logy+aug2` | 0.04219 | **0.026564** | 0.9935 | 0.03503 | 35.7 | 0.00879 | 0 | - | FINAL |
| idea-012 | E-03 | refine | data-driven | `torch_mlp/raw+logy+aug2` | - | **0.026644** | 0.9935 | 0.03514 | 36.8 | 0.0235 | 0 | - | evaluated |
| idea-014 | E-06 | refine | data-driven | `torch_mlp/raw+logy+aug1` | - | **0.027387** | 0.9931 | 0.03612 | 15 | 0.0187 | 0 | - | FINAL |
| idea-013 | E-06 | refine | data-driven | `torch_mlp/log+logy+aug1` | - | **0.027847** | 0.9929 | 0.03672 | 20.7 | 0.00819 | 0 | - | FINAL |
| idea-007 | E-06 | research | data-driven | `torch_mlp/log+logy+aug1` | 0.05058 | **0.027873** | 0.9928 | 0.03676 | 14.1 | 0.00895 | 0 | - | evaluated |
| idea-002 | E-01 | research | data-driven | `torch_mlp/raw+pca48+logy+aug2` | 0.04864 | **0.030074** | 0.9917 | 0.03966 | 69.1 | 0.018 | 0 | - | evaluated |
| idea-009 | E-08 | baseline | data-driven | `knn/sqrt` | 0.2036 | **0.14038** | 0.8184 | 0.1851 | 0.00703 | 0.0763 | 0 | - | evaluated |
| idea-003 | E-02 | research | data-driven | `kernel_ridge/raw+pca24+logy` | 0.1118 | **-** | - | - | - | - | - | - | ok |
| idea-005 | E-04 | research | data-driven | `kernel_ridge/raw+pca20+logy` | 0.06893 | **-** | - | - | - | - | - | - | ok |
| idea-006 | E-05 | research | data-driven | `custom-ml` | 0.372 | **-** | - | - | - | - | - | - | not meaningfully better than the mean baseline at screen fidelity (0.… |
| idea-008 | E-07 | baseline | data-driven | `torch_mlp/raw` | 0.06289 | **-** | - | - | - | - | - | - | ok |
| idea-010 | E-08 | baseline | data-driven | `kernel_ridge/raw` | 0.05292 | **-** | - | - | - | - | - | - | ok |

## 8. Lineage of the final method

- `idea-011` `torch_mlp/raw+logy+aug2` ← refined from idea-004 ← P-011 v2 (researcher-ml, r2, revise) ← P-003 v1 (researcher-ml, r1, propose)
- `idea-013` `torch_mlp/log+logy+aug1` ← refined from idea-007 ← P-013 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)
- `idea-004` `torch_mlp/raw+logy+aug2` ← P-011 v2 (researcher-ml, r2, revise) ← P-003 v1 (researcher-ml, r1, propose)
- `idea-014` `torch_mlp/raw+logy+aug1` ← refined from idea-007 ← P-013 v2 (researcher-experimental, r2, revise) ← P-005 v1 (researcher-experimental, r1, propose)

Artifact trace (checkpoint → result → plan → hypothesis → gate → proposal → …):

- `chec-00173` [checkpoint] by scheduler: final method (ensemble)
- `resu-00170` [result] by executor: exp-0019 rel_l2=0.02248 r2=0.9955 rmse=0.02958
- `hypo-00139` [hypothesis] by scheduler: idea-011 torch_mlp/raw+logy+aug2 (refine)
- `hypo-00143` [hypothesis] by scheduler: idea-013 torch_mlp/log+logy+aug1 (refine)
- `hypo-00077` [hypothesis] by scheduler: idea-004 torch_mlp/raw+logy+aug2 (research)
- `hypo-00145` [hypothesis] by scheduler: idea-014 torch_mlp/raw+logy+aug1 (refine)
- `plan-00169` [plan] by engineer-v2: exp-0019: final fit of 4 member(s) on train+val; test once
- `gate-00138` [gate] by gate: refine idea-004: approved (torch_mlp/raw+logy+aug2)
- `prop-00061` [proposal] by researcher-ml: P-011 v2 [revise] ml: Log-target MLP with noise augmentation learns the smooth 4-D inverse map
- `spec-00001` [spec] by scheduler: research spec heat_v2_no_forward_s0 (a7592704f6)
- `gate-00142` [gate] by gate: refine idea-007: approved (torch_mlp/log+logy+aug1)
- `prop-00063` [proposal] by researcher-experimental: P-013 v2 [revise] experimental: Matched-noise training augmentation
- `hypo-00083` [hypothesis] by scheduler: idea-007 torch_mlp/log+logy+aug1 (research)
- `gate-00076` [gate] by gate: E-03: approved (torch_mlp/raw+logy+aug2)
- `gate-00144` [gate] by gate: refine idea-007: approved (torch_mlp/raw+logy+aug1)
- `prop-00039` [proposal] by researcher-ml: P-003 v1 [propose] ml: Log-target MLP with noise augmentation learns the smooth 4-D inverse map
- `data-00004` [dataset] by simulator: python: 4500 samples, 80 features, target dim 4
- `prop-00041` [proposal] by researcher-experimental: P-005 v1 [propose] experimental: Matched-noise training augmentation
- `gate-00082` [gate] by gate: E-06: approved (torch_mlp/log+logy+aug1)

## 9. Compute and cost

- Execution compute: 414.9 / 400 worker-seconds over 19 jobs; wall 366 s
- By stage: baseline 1.9s, screen 56.7s, evaluate 128.0s, refine 106.2s, final 122.2s
- LLM: 14 calls (by phase {'research': 14}), 189 s, $0.729 reported · TheoremSearch: 16 queries (0 failed)

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

15 proposal versions over 2 research rounds; 8 methods advanced; 3 predictions supported, 0 contradicted. Final: ['torch_mlp/raw+logy+aug2', 'torch_mlp/log+logy+aug1', 'torch_mlp/raw+logy+aug2', 'torch_mlp/raw+logy+aug1'] with test rel_l2 0.022481.
