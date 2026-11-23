# V3 research product: heat_v3_smoke_s0_20261113_163324

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: action limit reached

Model export certified: **True**. Studies complete: **1/3**.

Model performance and study completion are separate outcomes.

## Actual exported method

- physics-MLE[log,ms=4,it=30] (kind `physics`, method hash `c247693e4c47`, source hash `None`, forward access `True`).
  Observed fit diagnostics: `{"fit_report": {"checked_samples": 200, "model_mismatch": false, "nominal_noise_std": 0.001, "residual_std_at_truth": 0.0009970299536016798}}`

Validation selection value: `0.005529704324879796`. Final metrics: `{"r2": 0.9997051257578653, "rel_l2": 0.005490560987682502, "rmse": 0.007440878039646911}`.

Export certificate: `{"atol": 1e-10, "certified": true, "checks": {"finite_shape": true, "partition": true, "reload": true, "repeat": true}, "input_role": "development", "max_abs_differences": {"partition": 2.100711604668959e-09, "reload": 0.0, "repeat": 0.0}, "n_rows": 80, "output_shape": [80, 2], "passed": true, "protocol": "pristine-repeat-reload-partition-v1", "reasons": [], "rtol": 1e-08, "schema_version": 3}`.

Lineage: `{"idea-004": {"arm_id": "S-Q1-002.physics", "baseline_identity": false, "gate": "gate-00062", "label": "physics-MLE[log,ms=4,it=30]", "origin": "proposal", "study_id": "S-Q1-002"}}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-Q1-001 | 0/4 | superseded |  |
| S-Q1-002 | 4/4 | incomplete | required work remains unresolved |
| S-Q1-003 | 4/4 | complete | all registered required work completed |

### S-Q1-001: Does forward-model inversion (physics MAP/MLE with log transform and multistart) achieve lower validation rel_l2 than the best library regressor (ridge baseline and a kernel/tree treatment) on the 40 sensor features?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-001.ridge` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.kernel_ridge` (treatment, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.extra_trees` (treatment, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.physics` (treatment, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-001.paired_phys_vs_ridge`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-001.paired_phys_vs_krr`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-001.paired_krr_vs_ridge`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-001.per_target_physics`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-002: On the 40 sensor features with 240 training rows, does physics MAP/MLE inversion (log transform, 4 multistarts, verified forward model) achieve lower validation rel_l2 than each of three fixed-configuration library regressors (ridge alpha=1, kernel ridge alpha=0.01 gamma_mult=1, extra_trees 300 trees)?

Owner: `researcher-003`; registered evidence: `['resu-00071', 'resu-00073', 'resu-00077', 'resu-00075']`.

- Arm `S-Q1-002.ridge` (control, required=True): `ok`; method `ridge/raw`; metrics `{"r2": 0.9990386738477597, "rel_l2": 0.010104799607854287, "rmse": 0.013289870408463697}`; effective hash `968c77ed472593a1999d9d08d2f22447840eecd9dcc7bab194debf78a0e17737`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.kernel_ridge` (control, required=True): `ok`; method `kernel_ridge/raw`; metrics `{"r2": 0.9978963508294071, "rel_l2": 0.014848759376192715, "rmse": 0.019529144119065353}`; effective hash `c65c72b1dcbc43d6aaae0e1f6ee3ce7b76c4025c1d0264ec7814ab908b37b32f`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.extra_trees` (control, required=True): `ok`; method `extra_trees/raw`; metrics `{"r2": 0.9972798283057571, "rel_l2": 0.016890225361442593, "rmse": 0.022214087852750992}`; effective hash `8b17af31216305d6bbdfbc4fe0b2d295ad2511f1be58e5794b6f88174dbde58f`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.physics` (treatment, required=True): `ok`; method `physics-MLE[log,ms=4,it=30]`; metrics `{"r2": 0.999712444260544, "rel_l2": 0.005529704324879796, "rmse": 0.007272687903444611}`; effective hash `68b205f4db4d0390f8a69855071c528319fd0704d31990a3b813486bcf9fd1a7`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-002.paired_phys_vs_ridge`: `ok`; artifact `resu-00079`; reason `registered computation returned`.
- `S-Q1-002.paired_phys_vs_krr`: `ok`; artifact `resu-00080`; reason `registered computation returned`.
- `S-Q1-002.paired_phys_vs_et`: `ok`; artifact `resu-00081`; reason `registered computation returned`.
- `S-Q1-002.paired_krr_vs_ridge`: `ok`; artifact `resu-00082`; reason `registered computation returned`.
- `S-Q1-002.per_target_phys_vs_krr`: `complete`; artifact `resu-00083`; reason `registered computation returned`.
- `S-Q1-002.per_target_phys_vs_ridge`: `complete`; artifact `resu-00084`; reason `registered computation returned`.
- `S-Q1-002.per_sample_physics`: `complete`; artifact `resu-00085`; reason `registered computation returned`.

Claim `C-Q1-A`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-Q1-002.ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics"}`; values `{"control": 0.010104799607854287, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `None`; interval `None`.
Evidence `['resu-00075', 'resu-00071']`; unmet requirements `['unregistered varying factor:features', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'unregistered varying factor:pca', 'unregistered varying factor:scaler', 'matching:features', 'matching:noise_aug', 'matching:noise_scale', 'matching:pca', 'matching:scaler']`.

Claim `C-Q1-B`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-Q1-002.kernel_ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics"}`; values `{"control": 0.014848759376192715, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `None`; interval `None`.
Evidence `['resu-00075', 'resu-00073']`; unmet requirements `['unregistered varying factor:features', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'unregistered varying factor:pca', 'unregistered varying factor:scaler', 'matching:features', 'matching:noise_aug', 'matching:noise_scale', 'matching:pca', 'matching:scaler']`.

Claim `C-Q1-C`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-Q1-002.extra_trees", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics"}`; values `{"control": 0.016890225361442593, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `None`; interval `None`.
Evidence `['resu-00075', 'resu-00077']`; unmet requirements `['unregistered varying factor:features', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'unregistered varying factor:pca', 'unregistered varying factor:scaler', 'matching:features', 'matching:noise_aug', 'matching:noise_scale', 'matching:pca', 'matching:scaler']`.

Claim `C-Q1-D`: **contradicted**. registered predicate not met; this does not establish absence of an effect.
Predicate: `{"control": "S-Q1-002.ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.kernel_ridge"}`; values `{"control": 0.010104799607854287, "metric": "rel_l2", "treatment": 0.014848759376192715}`; effect `-0.46947588793854256`; interval `[-0.8371639641593845, -0.13455926414533212]`.
Evidence `['resu-00073', 'resu-00071']`; unmet requirements `[]`.

### S-Q1-003: On the 40 sensor features with 240 training rows, does physics MAP/MLE inversion (log transform, 4 multistarts) achieve lower validation rel_l2 than each of three fixed-configuration library regressors (ridge alpha=1, kernel ridge alpha=0.01 gamma_mult=1, extra_trees 300 trees)? Re-registration of S-Q1-002: the comparison is unchanged, the matching protocol is corrected, and implementation-verification outputs are made explicit.

Owner: `researcher-003`; registered evidence: `['resu-00150', 'resu-00152', 'resu-00156', 'resu-00154']`.

- Arm `S-Q1-003.ridge` (control, required=True): `ok`; method `ridge/raw`; metrics `{"r2": 0.9990386738477597, "rel_l2": 0.010104799607854287, "rmse": 0.013289870408463697}`; effective hash `968c77ed472593a1999d9d08d2f22447840eecd9dcc7bab194debf78a0e17737`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.kernel_ridge` (control, required=True): `ok`; method `kernel_ridge/raw`; metrics `{"r2": 0.9978963508294071, "rel_l2": 0.014848759376192715, "rmse": 0.019529144119065353}`; effective hash `c65c72b1dcbc43d6aaae0e1f6ee3ce7b76c4025c1d0264ec7814ab908b37b32f`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.extra_trees` (control, required=True): `ok`; method `extra_trees/raw`; metrics `{"r2": 0.9972798283057571, "rel_l2": 0.016890225361442586, "rmse": 0.022214087852750985}`; effective hash `8b17af31216305d6bbdfbc4fe0b2d295ad2511f1be58e5794b6f88174dbde58f`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.physics` (treatment, required=True): `ok`; method `physics-MLE[log,ms=4,it=30]`; metrics `{"r2": 0.999712444260544, "rel_l2": 0.005529704324879796, "rmse": 0.007272687903444611}`; effective hash `68b205f4db4d0390f8a69855071c528319fd0704d31990a3b813486bcf9fd1a7`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-003.paired_phys_vs_ridge`: `ok`; artifact `resu-00158`; reason `registered computation returned`.
- `S-Q1-003.paired_phys_vs_krr`: `ok`; artifact `resu-00159`; reason `registered computation returned`.
- `S-Q1-003.paired_phys_vs_et`: `ok`; artifact `resu-00160`; reason `registered computation returned`.
- `S-Q1-003.paired_krr_vs_ridge`: `ok`; artifact `resu-00161`; reason `registered computation returned`.
- `S-Q1-003.per_target_phys_vs_ridge`: `complete`; artifact `resu-00162`; reason `registered computation returned`.
- `S-Q1-003.per_target_phys_vs_krr`: `complete`; artifact `resu-00163`; reason `registered computation returned`.
- `S-Q1-003.per_target_phys_vs_et`: `complete`; artifact `resu-00164`; reason `registered computation returned`.
- `S-Q1-003.per_sample_phys_vs_ridge`: `complete`; artifact `resu-00165`; reason `registered computation returned`.

Claim `C-Q1-A2`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-003.ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-003.physics"}`; values `{"control": 0.010104799607854287, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `0.45276457332398246`; interval `[0.354872874195872, 0.521803253038282]`.
Evidence `['resu-00154', 'resu-00150']`; unmet requirements `['objection-028', 'objection-029', 'objection-030', 'objection-031']`.

Claim `C-Q1-B2`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-003.kernel_ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-003.physics"}`; values `{"control": 0.014848759376192715, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `0.6275982265733479`; interval `[0.5015978434292101, 0.7040029280785912]`.
Evidence `['resu-00154', 'resu-00152']`; unmet requirements `['objection-028', 'objection-029', 'objection-030', 'objection-031']`.

Claim `C-Q1-C2`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-003.extra_trees", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-003.physics"}`; values `{"control": 0.016890225361442586, "metric": "rel_l2", "treatment": 0.005529704324879796}`; effect `0.6726092040486837`; interval `[0.5799060418423404, 0.7383873962779354]`.
Evidence `['resu-00154', 'resu-00156']`; unmet requirements `['objection-028', 'objection-029', 'objection-030', 'objection-031']`.

Claim `C-Q1-D2`: **contradicted**. registered predicate not met; this does not establish absence of an effect.
Predicate: `{"control": "S-Q1-003.ridge", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-003.kernel_ridge"}`; values `{"control": 0.010104799607854287, "metric": "rel_l2", "treatment": 0.014848759376192715}`; effect `-0.46947588793854256`; interval `[-0.8371639641593845, -0.13455926414533212]`.
Evidence `['resu-00152', 'resu-00150']`; unmet requirements `[]`.

## Independent critic and uncertainty

- `objection-001` (S-Q1-001, open): Resolve the contract inconsistency before running any arm. The objective says 'four unknown piecewise-constant thermal conductivities', but parameter_count, target_count and target_shape all say 2. State the true dimensionality of the parameter, and the segment boundaries in grid units, and use those values in both the physics forward model and the label checks.. Resolution: `unresolved`.
- `objection-002` (S-Q1-001, open): Document how the 40 flattened features map to the observation operator. Four sensors over 60 steps gives 240 readings, so 40 features implies 10 observation times. List the exact time indices. The physics arm's forward model must evaluate at the same sensor and time indices, and in the same feature order, as the stored observations.. Resolution: `unresolved`.
- `objection-003` (S-Q1-001, open): Verify that the physics arm's forward solver matches the data-generating simulator. Check the grid, dt, Dirichlet BCs, Gaussian IC parameters, how k is averaged at cell interfaces, and the segment map. Then reproduce noise-free training observations from the training labels with residuals well below sigma = 0.001. All claims require implementation verification, and this is the core check for that.. Resolution: `unresolved`.
- `objection-004` (S-Q1-001, open): Confirm that the explicit FD scheme is stable at the upper bound k = 2.0. Report the CFL ratio k_max*dt/dx^2 ≤ 0.5. Also confirm that the optimizer cannot step outside [0.5, 2.0] in log space and trigger an unstable solve. Unstable solves would silently corrupt the MAP estimates.. Resolution: `unresolved`.
- `objection-005` (S-Q1-001, open): Specify the physics estimator unambiguously. The params give estimator='mode', posterior_samples=128 and tikhonov=0. Under a uniform prior with no Tikhonov term, MAP equals bounded MLE. State whether the reported point estimate is the optimizer's mode or a posterior summary, explain what the 128 samples are used for, and say how the log transform interacts with the prior. A uniform prior on k is not uniform on log k, so dropping the Jacobian changes the MAP.. Resolution: `unresolved`.
- `objection-006` (S-Q1-001, open): Report physics optimizer convergence diagnostics for each validation row: final misfit compared with the expected chi-square level (about 40·sigma^2), the number of starts that converged, and the spread across starts. With only 30 iterations and 4 starts, rows that fail to converge, rather than the method itself, could drive the rel_l2 result.. Resolution: `unresolved`.
- `objection-007` (S-Q1-001, open): Account for the forward-evaluation budget. Compute validation rows × 4 starts × 30 iterations × evaluations per iteration, including finite-difference or adjoint gradients and the 128 posterior samples. Show that the total is at most 3,000,000, and log the actual count.. Resolution: `unresolved`.
- `objection-008` (S-Q1-001, open): Tune the library baselines fairly, or narrow the claim. Ridge (alpha=1) and KRR (alpha=0.01, gamma_mult=1) use fixed, untuned hyperparameters. The question asks about the 'best library regressor', which requires hyperparameter selection by cross-validation on training data only, never on validation. Otherwise, restate C-Q1-1 and C-Q1-2 as comparisons against these specific configurations.. Resolution: `unresolved`.
- `objection-009` (S-Q1-001, open): Remove the presupposition in C-Q1-2 or make it conditional. It calls kernel ridge 'the stronger library regressor' before C-Q1-3 has been evaluated. Extra trees is listed as optional, yet the question names a 'kernel/tree treatment'. Either make extra_trees required and define the best-library reference as the best of all library arms, chosen without using validation outcomes, or drop the tree from the question.. Resolution: `unresolved`.
- `objection-010` (S-Q1-001, open): Disclose the information asymmetry between arms. The physics arm uses the exact forward model and the prior but no training labels. The library arms use only 240 labelled rows. State whether noise augmentation is applied to the library arms and with what noise_scale and number of copies. Note that target_transform differs (log for physics, none for library), which is permitted under allowed_varying_factors but must be reported as part of the intervention.. Resolution: `unresolved`.
- `objection-011` (S-Q1-001, open): Define the paired uncertainty procedure. Specify the unit (validation row), the resampling method (for example a paired bootstrap on the rel_l2 ratio, with B stated), the confidence level, and the decision rule for 'at least 1% relative better'. For example: the upper CI bound of rel_l2_treat/rel_l2_ctrl must be below 0.99. Report the validation sample size. A single seed of 0 gives no variability estimate across training runs for the library arms.. Resolution: `unresolved`.
- `objection-012` (S-Q1-001, open): Adjust for multiplicity, or pre-declare the decision hierarchy, across C-Q1-1 to C-Q1-3, which share arms and share the validation set.. Resolution: `unresolved`.
- `objection-013` (S-Q1-001, open): Make the per-target diagnostic required, or report per-target rel_l2 and RMSE alongside the global Frobenius rel_l2. A global aggregate can hide a failure on the less identifiable segment, for example the segment whose sensors see weaker signal from the centred pulse.. Resolution: `unresolved`.
- `objection-014` (S-Q1-001, open): Keep the scope of any Fisher or identifiability statement local, as the contract's diagnostic_limits require. Do not present physics-arm success as evidence of global identifiability. Do not consult the test or stress splits; confirm that the test fingerprint is untouched.. Resolution: `unresolved`.
- `objection-015` (S-Q1-001, open): Supply the evidence. evidence_ids is empty and the status is 'planned'. No claim can be adjudicated until run artifacts exist, linked by split fingerprint: predictions, metrics, paired diagnostics, verification logs, and forward-evaluation counts.. Resolution: `unresolved`.
- `objection-016` (S-Q1-002, open): Resolve the parameter-dimension contradiction before any arm is run. contract.objective says 'four unknown piecewise-constant thermal conductivities', but contract.facts.parameter_count = 2, target_count = 2, target_shape = [2], and bounds have length 2. State the true number of segments and give the segment-to-grid map: interface location(s) in grid units over the 20 points, and which cell or face each k applies to. Show that the physics arm's forward model uses the same map and the same interface discretization as the data generator (harmonic or arithmetic averaging of k at the discontinuity). If the physics arm estimates a different number of parameters than the labels contain, every comparison is invalid.. Resolution: `unresolved`.
- `objection-017` (S-Q1-002, open): Restate the forward-verification gate for C-Q1-A through C-Q1-D. The claims are 'adjudicated only if forward-model verification residuals are well below sigma=0.001', but training observations already carry N(0, 0.001^2) noise. A correct forward model evaluated at the true labels should therefore give RMS residual ≈ sigma, not well below it. Specify one of two tests: (a) compare against noise-free simulator output, where the residual should be at machine or discretization precision; or (b) compare against noisy observations, with a chi-square test that the sum of squared residuals over 40 features is consistent with 40·sigma². Also give a pass/fail threshold. As written, a correct implementation would fail the gate, or the gate cannot be evaluated.. Resolution: `unresolved`.
- `objection-018` (S-Q1-002, open): Report dt and dx, and show that the CFL condition k_max·dt/dx² ≤ 0.5 holds at k = 2.0 for the explicit scheme. With 20 points, dx = 1/19. Also confirm that every forward call made by the optimizer, the multistarts and the 128 Laplace samples is clipped to [0.5, 2.0] before simulation. Laplace samples drawn around a mode near the upper bound can leave the bounds and violate CFL unless they are clipped or rejected. State how out-of-bound samples are handled.. Resolution: `unresolved`.
- `objection-019` (S-Q1-002, open): No evidence is attached yet: evidence_ids is empty, verdicts is empty, and status is 'planned'. Every claim sets requires_implementation_verification = true and requires_uncertainty = true. No claim may be marked as holding until the following artifacts exist:
- paired bootstrap outputs
- per-target diagnostics
- forward-reproduction residuals
- CFL check
- convergence logs
- forward-evaluation counts

The validation n must also be reported.. Resolution: `unresolved`.
- `objection-020` (S-Q1-002, open): Per-row convergence evidence for the physics arm. iters = 30 with 4 multistarts is a fixed, unjustified budget. For each row report:
- final misfit vs the expected 40·sigma²
- how many of the 4 starts converged to the same basin
- spread of k estimates across starts
- whether the estimate sits at a bound (0.5 or 2.0)

State whether iters or multistart were chosen by looking at validation results; if so, the validation comparison is no longer pre-registered for the physics arm.. Resolution: `unresolved`.
- `objection-021` (S-Q1-002, open): Check near-symmetry and local minima. The initial pulse is centred at x = 0.5, between grid indices 9 and 10. The sensors [3, 8, 12, 17] are nearly but not exactly mirror-symmetric (the mirror images would be [16, 11, 7, 2]). If the segment interface is at the centre, swapping the two conductivities may produce a nearly identical observation pattern, so the multistart optimizer may land in a swapped basin. Report the local Fisher condition number per row and the frequency of swapped or alternate-basin solutions. Remember that local Fisher is not a global identifiability guarantee.. Resolution: `unresolved`.
- `objection-022` (S-Q1-002, open): Make the diagnostics consistent across the claim hierarchy:
- C-Q1-C (vs extra_trees) has no per-target diagnostic, while C-Q1-A and C-Q1-B do. Rel_l2 is a global Frobenius norm and can hide one target dominating or failing, so add per_target_phys_vs_et or justify its omission.
- The per_sample_physics diagnostic lists kernel_ridge as its control without explanation. State what it compares and why that control was chosen.. Resolution: `unresolved`.
- `objection-023` (S-Q1-002, open): Specify the bootstrap precisely:
- the resampling unit (validation rows, with both targets kept together)
- whether the ratio rel_l2_phys/rel_l2_ctrl is recomputed from resampled Frobenius norms in each replicate
- the CI method (percentile or BCa)
- the RNG seed

State explicitly that this CI covers only validation-row sampling. It does not cover training-set resampling, the extra_trees seed, or the physics multistart initialization seed, all of which use seed 0. Synthesis claims must be scoped to 'this validation split and these seeds'.. Resolution: `unresolved`.
- `objection-024` (S-Q1-002, open): Scope the inverse-crime advantage. If the physics arm inverts with the same simulator (same grid, dt and interface rule) that generated the data, its advantage over label-trained regressors is a best-case result under no model error. The synthesis must not generalize it to settings with model mismatch, or to regressors that are tuned or given more training rows. The library configurations are fixed and untuned, and the claims already say so; any summary must keep that caveat.. Resolution: `unresolved`.
- `objection-025` (S-Q1-002, open): Define the kernel_ridge gamma_mult = 1.0 convention: what quantity it multiplies, for example 1/n_features or a median-distance heuristic. Confirm that the standard scaler for all library arms is fit on training rows only. Confirm that the physics arm uses raw, unscaled observations against the forward model.. Resolution: `unresolved`.
- `objection-026` (S-Q1-002, open): Report the total forward-evaluation count against the 3,000,000 budget. Count all of: n_val × 4 starts × 30 iterations × evaluations per iteration (including finite-difference gradients if they are used), plus the Jacobian or Fisher computation for the 128 Laplace samples, plus the forward-reproduction check on 240 training rows. Confirm that no forward evaluation touched the test or stress splits.. Resolution: `unresolved`.
- `objection-027` (S-Q1-002, open): Fix the uncertainty-reporting claim. The Laplace/local-Fisher samples are described as 'for uncertainty reporting only', but no claim uses them. Either drop them from this study's evidence, or state that they are a local approximation under Gaussian noise with sigma = 0.001, per the diagnostic limits. They must not be presented as calibrated posterior intervals, especially for modes at the boundary, where a Laplace approximation is invalid under a bounded uniform prior.. Resolution: `unresolved`.
- `objection-028` (S-Q1-003, open): Resolve the contract inconsistency before execution. The objective text says 'four unknown piecewise-constant thermal conductivities', but parameter_count=2, target_count=2, target_shape=[2], and bounds are 2-D. The physics arm's forward(theta) must be verified to use exactly the 2-segment parameterization that generated the data, including segment breakpoint location. A mismatched parameterization would silently create model error and invalidate the 'no model error' scope.. Resolution: `unresolved`.
- `objection-029` (S-Q1-003, open): Make the implementation-verification outputs explicit as registered, required diagnostics. The question promises this, but the diagnostics list contains only paired, per_target and per_sample comparisons. At minimum, register these: (a) a forward-reproduction check, where forward(theta_true) on training rows gives residuals consistent with sigma=0.001 (mean squared residual per row ≈ 40·1e-6); (b) explicit-scheme stability (dt ≤ dx²/(2·k_max) with k_max=2.0 and dx=1/19) across the full bound box; (c) per-row optimizer convergence (final objective, gradient or step norm, iterations used vs the 30 cap); (d) multistart agreement (spread of the 4 terminal estimates and the fraction reaching the best objective); (e) the fraction of estimates at or near the bounds [0.5, 2.0]. All four claims set requires_implementation_verification=true, yet no artifact is defined that could satisfy it.. Resolution: `unresolved`.
- `objection-030` (S-Q1-003, open): Pre-specify the bootstrap replicate count B for S-Q1-003 itself. The claim only says 'B is reported as executed', which leaves B free after the fact. Given the S-Q1-002 deviation (B=200 vs 2000 registered), register B≥2000 or justify a smaller B. Report the Monte Carlo error of the decision quantile. At B=200 the tail percentile rests on about 5 replicates, which is too unstable for a hard 0.99 threshold.. Resolution: `unresolved`.
- `objection-031` (S-Q1-003, open): Disambiguate the decision bound. 'Upper 95% bound of rel_l2_phys/rel_l2_ridge' could mean the 95th percentile (one-sided 95%) or the 97.5th percentile (upper end of a two-sided 95% CI). These imply different error rates for the A→B→C gatekeeping sequence. State which percentile is used.. Resolution: `unresolved`.
- `objection-032` (S-Q1-003, open): Report the number of validation rows, and confirm that the bootstrap resamples identical row indices for both arms in each replicate (a true paired design). Power and CI width cannot be assessed without n_val.. Resolution: `unresolved`.
- `objection-033` (S-Q1-003, open): Define in advance how the required per_target diagnostics affect interpretation. Example: physics wins on Frobenius rel_l2 but is worse on one target. As written, these diagnostics are required but have no stated role in the decision rule or synthesis. The same applies to per_sample_phys_vs_ridge, which no claim references.. Resolution: `unresolved`.
- `objection-034` (S-Q1-003, open): Disclose that this re-registration reuses the same validation split, and the same arm settings, as S-Q1-002. S-Q1-002's results on that split were already observed, and C-Q1-D was returned 'contradicted'. S-Q1-003 must not be presented as independent confirmation or replication of S-Q1-002. The S-Q1-002 outcomes for C-Q1-A/B/C should be cited, and the protocol corrections should be shown not to depend on those outcomes.. Resolution: `unresolved`.
- `objection-035` (S-Q1-003, open): Disclose the information asymmetry in the controls and restrict the synthesis accordingly. The physics arm uses the exact generating simulator, known IC/BCs and the noise model, with up to 3M forward evaluations (an inverse-crime setting). The library arms use only 240 labelled rows, even though forward simulation was authorized and could have been used to enlarge their training sets. The claims compare against fixed, untuned configurations. Any synthesis must not generalize to 'physics inversion beats learned regressors' or to settings with model error.. Resolution: `unresolved`.
- `objection-036` (S-Q1-003, open): Report the total forward-evaluation count actually consumed by the physics arm, including finite-difference gradients if used, and confirm it is within max_forward_evals=3,000,000.. Resolution: `unresolved`.
- `objection-037` (S-Q1-003, open): Fix the naming. Claims call the arm 'Physics MAP inversion', but the point estimate is a bounded MLE (uniform prior, tikhonov=0, no log-Jacobian). Use consistent terminology (e.g., 'bounded MLE / uniform-prior mode in k') so the synthesis does not imply prior regularization or Bayesian estimation. Also state that the 128 Laplace samples are not used and need no calibration check.. Resolution: `unresolved`.
- `objection-038` (S-Q1-003, open): Address single-seed dependence. Extra_trees and the physics multistart initializations depend on seed 0. The CI covers only validation-row resampling, conditional on fitted models. Either add a small seed-sensitivity check (especially for extra_trees and for multistart initialization) or explicitly confine claims C-Q1-C2 and C-Q1-A2 to seed 0.. Resolution: `unresolved`.
- `objection-039` (S-Q1-003, open): Acknowledge that the matching is nearly vacuous. For the physics comparisons, allowed_varying_factors includes every pipeline factor, so only split, seed and fidelity are held fixed. State plainly that these are method-vs-method comparisons on a shared split, not matched-ablation comparisons, and that no attribution to any single factor (e.g., the log reparameterization) is supported.. Resolution: `unresolved`.
- `objection-040` (S-Q1-003, open): Check local identifiability per row. Use the local Fisher diagnostic, within its stated limits, to flag validation rows whose 2×2 information matrix is ill-conditioned. This shows whether any physics win or loss is concentrated in poorly identified rows rather than reflecting the estimator in general.. Resolution: `unresolved`.
- `objection-041` (S-Q1-002, open): Resolve the parameter-dimensionality contradiction. The objective says 'four unknown piecewise-constant conductivities', but facts.parameter_count=2, target_count=2 and target_shape=[2]. Before any physics claim is adjudicated, report the segment map in grid units and confirm the physics arm inverts exactly the 2 parameters the labels encode, in the same order (verification output objection-001).. Resolution: `unresolved`.
- `objection-042` (S-Q1-002, open): Fix the matching protocol before adjudicating C-Q1-A/B/C. All three verdicts are 'inconclusive' because features, noise_aug, noise_scale, pca and scaler vary without being registered. These fields are structurally absent from the physics arm, not tuned confounders. A versioned claim amendment should register them as allowed (or as N/A for kind=physics), state the reason, and timestamp it before re-adjudication. The registered claim text should not be reinterpreted after the fact.. Resolution: `unresolved`.
- `objection-043` (S-Q1-002, open): Use the registered bootstrap size. Every claim registers B=2000, but every paired diagnostic reports n_bootstrap=200. Rerun all paired analyses (including krr_vs_ridge, which already has a 'contradicted' verdict) with B=2000 and the registered seed.. Resolution: `unresolved`.
- `objection-044` (S-Q1-002, open): Report the registered decision statistic. The rule is a 95% CI on the ratio rel_l2_treatment/rel_l2_control with upper bound < 0.99. The diagnostics report relative_improvement and absolute_effect intervals instead. Report the paired-bootstrap ratio CI directly. Do not derive it by transforming a relative-improvement interval unless that transform is shown to be identical to the bootstrap percentile of the ratio.. Resolution: `unresolved`.
- `objection-045` (S-Q1-002, open): Supply the implementation-verification evidence the physics claims require. method_contract.diagnostics shows declared=[], observed=[], verified=false, yet every claim has requires_implementation_verification=true and C-Q1-A is adjudicable only after verification. Required outputs: (a) forward reproduction of noise-free training observations from training labels, with max-abs and RMS residual compared with sigma=0.001; (b) the CFL ratio k_max*dt/dx^2 at k=2.0, plus evidence that log-space iterates are clipped to [0.5, 2.0]; (c) the observation operator (sensor indices [3,8,12,17] × time indices) and its feature order matched to the 40 input columns; (d) per-row convergence, i.e. final misfit vs 40*sigma^2, number of the 4 starts that converged, and spread across starts; (e) total forward evaluations vs the 3,000,000 budget.. Resolution: `unresolved`.
- `objection-046` (S-Q1-002, open): The per-target diagnostics need actual comparison content. S-Q1-002.per_target_phys_vs_ridge, per_target_phys_vs_krr and per_sample_physics have identical payloads containing only physics-arm values (per_target_rel_l2=[0.00478, 0.00620]) and no control per-target rel_l2 or per-target paired interval. These are required diagnostics for C-Q1-A and C-Q1-B. Regenerate them with treatment and control per-target rel_l2, paired per-target intervals, and distinct artifact content.. Resolution: `unresolved`.
- `objection-047` (S-Q1-002, open): Recompute rel_l2 independently from the per-row data. Verify that the reported arm rel_l2 values (physics 0.00553, ridge 0.01010, krr 0.01485, ET 0.01689) equal sqrt(sum per_sample_squared_error / sum target_squared_norm), using global Frobenius aggregation over the 80 validation rows. State whether per_sample_squared_error is summed or averaged over the 2 targets.. Resolution: `unresolved`.
- `objection-048` (S-Q1-002, open): Respect the hierarchy's gatekeeping. C-Q1-B may be tested only if C-Q1-A holds, and C-Q1-C only if C-Q1-B holds. Since C-Q1-A is currently inconclusive, B and C must not be reported as supported, and their point estimates must not be used in synthesis as directional evidence, even though their observed ratios look favorable.. Resolution: `unresolved`.
- `objection-049` (S-Q1-002, open): Make C-Q1-D's status consistent. It is called 'hierarchy rank 4' and also 'descriptive and outside the primary hierarchy'. Pick one. If it is descriptive, its 'contradicted' verdict should not be presented as a confirmatory finding. It should also be redone with B=2000 and the registered ratio statistic.. Resolution: `unresolved`.
- `objection-050` (S-Q1-002, open): Document the noise_scale normalization. Each library arm requested noise_scale=0.0 but the effective value is 1.0. With noise_aug=0 this should be inert. Confirm the training data was unchanged, e.g. by showing training-feature hashes are identical under requested and effective configs, so the effective_hash difference does not signal hidden augmentation.. Resolution: `unresolved`.
- `objection-051` (S-Q1-002, open): State the inverse-crime caveat. The physics arm presumably uses the same 20-point, 60-step explicit FD simulator and the exact Gaussian sigma that generated the data, while the library arms see only 240 labeled rows. Any synthesis must say the advantage is measured under a matched model with known noise. It does not show robustness to model misspecification, discretization mismatch, or unknown noise.. Resolution: `unresolved`.
- `objection-052` (S-Q1-002, open): Scope the controls as fixed and untuned. Ridge (alpha=1), kernel ridge (alpha=0.01, gamma_mult=1) and extra_trees were not tuned on train/CV. Kernel ridge underperforming ridge suggests it is poorly configured. Synthesis must not claim physics beats 'data-driven methods' or 'library regressors' in general, only these three configurations.. Resolution: `unresolved`.
- `objection-053` (S-Q1-002, open): Report the uncertainty scope and its limits. Intervals are conditional on fitted models from a single seed (0) and exclude training variation, extra_trees randomness and physics multistart-initialization randomness. Either add seed replication (several seeds per arm, especially for extra_trees and the physics multistarts) or state explicitly that the CIs reflect validation-row sampling only.. Resolution: `unresolved`.
- `objection-054` (S-Q1-002, open): Align the estimator name. The claims say 'Physics MAP inversion', but the arm is a bounded MLE (tikhonov=0, uniform prior on k, no log-Jacobian). Under the uniform prior these coincide on the box, but claim text, method_label and synthesis should consistently say bounded MLE. Also state that the 128 Laplace/local-Fisher samples are used only for uncertainty reporting and carry no global identifiability guarantee (per diagnostic_limits.local_fisher).. Resolution: `unresolved`.
- `objection-055` (S-Q1-002, open): Disclose the supersession history. S-Q1-002 supersedes S-Q1-001. Report what changed and why, and confirm that S-Q1-001 validation results did not drive the choice of control configurations, margin or physics settings. Otherwise the 80-row validation set has effectively been reused for both design and confirmation.. Resolution: `unresolved`.
- `objection-056` (S-Q1-002, open): Confirm the held-out data was untouched. Provide evidence (access logs or the pipeline config) that the test split (fingerprint f7409783…) and any stress observations were not loaded during S-Q1-002 development or adjudication, as diagnostic_limits.held_out requires.. Resolution: `unresolved`.
- `objection-057` (S-Q1-002, open): Make the completion bookkeeping consistent. completion_fraction=1.0 and completed_work=4/4 are reported alongside status='incomplete' and completed_claim_work=1/4. Report completion against claim requirements so the study is not read as finished.. Resolution: `unresolved`.
- `objection-058` (S-Q1-002, open): Investigate the physics-arm tail errors. per_sample_l2 ranges from 0.0005 to 0.031, with several rows above 0.02. Report the misfit and multistart agreement for those rows to rule out local minima or bound-active solutions, since a mean rel_l2 near 0.0055 could hide optimizer failures that matter for a stronger claim.. Resolution: `unresolved`.
- `objection-059` (program, open): Final synthesis must not report any physics-vs-library comparison (C-Q1-A/B/C, C-Q1-A2/B2/C2) as supported or holding. Every harness verdict is 'inconclusive': S-Q1-002 failed the matching gate, and S-Q1-003 has unresolved objections 028-031. The program stopped on the action limit before any verification artifact or resolve_objection action existed. The point values (physics 0.00553 vs ridge 0.01010, KRR 0.01485, ET 0.01689) and the relative-improvement intervals may be reported only as descriptive, uncertified measurements on this split at seed 0. An orchestrator closing objections by 'acknowledged_uncertainty' does not turn these verdicts into support.. Resolution: `unresolved`.
- `objection-060` (program, open): Synthesis must not claim recovery of 'four conductivities'. All executed arms, labels and bounds are 2-dimensional. The per-row target_squared_norm values (0.63 to 7.22) are consistent only with two parameters in [0.5, 2.0], since four would give a minimum of 1.0. State explicitly that the objective text conflicts with the contract, and that results cover only the 2-parameter contract task.. Resolution: `unresolved`.
- `objection-061` (program, open): The required per_target and per_sample diagnostics in S-Q1-003 (resu-00162 to resu-00165) are byte-identical, physics-only payloads, as were resu-00083 to resu-00085 in S-Q1-002. They contain no control per-target values and no paired per-target interval, yet the harness counted them toward completion_fraction=1.0 and status 'complete'. Treat these diagnostics as contentless. Any per-target statement ('physics lower on both targets') must cite the arm-level val_per_target_rel_l2 fields (resu-00150/152/154/156) and be labelled descriptive, with no interval. Also flag the completion accounting as unreliable.. Resolution: `unresolved`.
- `objection-062` (program, open): Q-2 (hybrid initializer + LM) and Q-3 (log target transform and noise augmentation for learned regressors) were never studied. The synthesis must list them as unanswered. It must not use the Q-1 arms (all with noise_aug=0 and target_transform=none) to draw conclusions about either question.. Resolution: `unresolved`.
- `objection-063` (program, open): S-Q1-003 is not independent evidence. Its arms are cache-equivalent re-runs of S-Q1-002 on the same row_hash and seed, with identical metrics to about 1e-17. Synthesis must count this as one measurement, not two concurring studies. The study record also lists S-Q1-003 as superseded_by S-Q1-004, a study that was never created because action-005 was rejected. Correct or annotate that stale pointer so the export does not treat S-Q1-003 as obsolete or treat S-Q1-004 as existing.. Resolution: `unresolved`.
- `objection-064` (program, open): Weigh the indirect forward-model evidence correctly instead of treating objections 001/003/016/028/041 as wholly unaddressed. fit_report (resu-00075/00154) gives residual_std_at_truth = 1.00683e-3 over 200 training rows (about 8,000 readings) against sigma = 1e-3, and the standard error of a std estimate there is about 0.8%. That is consistent with forward(theta_true) reproducing the generator up to noise, which a wrong parameter count or wrong segment map would be unlikely to do. However, this is a harness self-report, not independently recomputed, and it does not give the noise-free max-abs residual, the segment/interface map, the observation time indices, or the CFL ratio. It may support partial resolution, but only as uncertified evidence.. Resolution: `unresolved`.
- `objection-065` (program, open): Do not present optimizer failure or swapped basins as excluded, and do not present them as likely. The aggregate local-uncertainty diagnostics (z_rms 0.985; 2-sigma coverage 0.969 over 160 row-target values) argue against widespread gross basin errors. But chi^2/dof is reported only for the last row (0.93), and the largest physics rows (row 35 at 0.0310, row 74 at 0.0279, row 23 at 0.0234) have no per-row misfit, multistart agreement or bound-active flag. Per-row z or chi^2 for those rows is the minimal missing check (objections 020, 021, 058).. Resolution: `unresolved`.
- `objection-066` (program, open): Downgrade the bootstrap-size objections (023, 030, 043) to non-decision-relevant, and disclose the deviation rather than block on it. The decision-relevant lower bounds of relative improvement (0.355, 0.502 and 0.580 against a 0.01 margin) sit more than 30 margin-units away from the threshold. Monte Carlo error in a B=200 percentile cannot plausibly close that gap. The B=2000 registration was still not honoured, and that must be stated.. Resolution: `unresolved`.
- `objection-067` (program, open): Resolve objection-044 (ratio vs relative-improvement statistic) by confirming the per-replicate definition. If relative_improvement is computed per replicate as 1 - rel_l2_t/rel_l2_c, then because percentiles commute with monotone transforms, the ratio's upper bound is exactly 1 - (lower bound): 0.645, 0.498 and 0.420, all below 0.99. The point estimates check out (1 - 0.00553/0.01010 = 0.4528). The per-replicate definition is undocumented and must be stated, not assumed.. Resolution: `unresolved`.
- `objection-068` (program, open): Objection-047 is approximately satisfied by recomputation and should be recorded as such. For physics, sum(per_sample_squared_error) is about 8.5e-3 (it equals rmse^2 x 160, so per_sample values are summed over the 2 targets). sum(target_squared_norm) is about 272. sqrt(ratio) is about 0.0056, which matches the reported 0.00553 within the precision of this check. An exact recomputation from the stored arrays should be attached before citing four significant figures.. Resolution: `unresolved`.
- `objection-069` (program, open): Disclose that the physics arm's 'uses_labels=false' is not strictly true for diagnostics: fit_report consumes 200 training labels (forward_evals_fit=200) to compute residual_std_at_truth. This does not affect predictions but should be stated in the information-asymmetry caveat.. Resolution: `unresolved`.
- `objection-070` (program, open): Report the delegate failures as infrastructure failures, not as findings about the method. specialist-004 ran 900 s against a 60 s limit with no output, and specialist-005 reported 'treatment predictions unavailable' even though val_pred_path files exist for every arm. These leave objections 023, 025 and 050 open by default. The scaler-fit scope and the gamma_mult base remain undocumented.. Resolution: `unresolved`.
- `objection-071` (program, open): Keep every scope caveat in any exported summary:
- Inverse-crime setting: the same simulator and the known sigma.
- Fixed, untuned library controls trained on 240 rows. Forward simulation could have enlarged their training data but was not used.
- A single seed (0) for extra_trees and for multistart initialisation.
- Intervals cover validation-row resampling only, with n_val = 80.
- C-Q1-D/D2 ('contradicted') is descriptive only. It shows that this KRR configuration is worse than ridge (-0.47 [-0.84, -0.13]), not that kernel methods are worse in general.. Resolution: `unresolved`.
- `objection-072` (S-Q1-001, open): S-Q1-001 remains superseded. Resolution: `unresolved`.
- `objection-073` (S-Q1-002, open): S-Q1-002 remains incomplete. Resolution: `unresolved`.

test_bootstrap: `{"ci95": [0.004870637077438388, 0.006197424139505603], "mean": 0.005459914683370376, "n_boot": 200, "std": 0.00033033083853570126}`.

test_per_target_rel_l2: `[0.005215525779419974, 0.005737746709956274]`.

test_uncertainty: `{"coverage_1sigma": 0.65, "coverage_2sigma": 0.95, "mean_std": 0.006956507281874569, "z_rms": 1.0805524721871205}`.

shift: `{}`.

## Scientific contract

```json
{
  "bounds": {
    "lower": [
      0.5,
      0.5
    ],
    "upper": [
      2.0,
      2.0
    ]
  },
  "diagnostic_limits": {
    "held_out": "final test and stress observations cannot inform research decisions",
    "local_fisher": "local sensitivity under specified noise/prior assumptions; not a global identifiability or Bayes guarantee"
  },
  "facts": {
    "boundary_conditions": "homogeneous Dirichlet at both ends",
    "coefficient_regularity": "piecewise-constant (discontinuous) conductivity, one value per segment",
    "dimension": 1,
    "discretization": "explicit finite differences, 20 grid points, 60 steps",
    "initial_condition": "known Gaussian pulse centred at x = 0.5",
    "noise": "Gaussian, sigma = 0.001",
    "observations": "interior point sensors at grid indices [3, 8, 12, 17], discrete times, finite window",
    "parameter_count": 2,
    "pde": "parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x",
    "positivity": "conductivity is strictly positive"
  },
  "flattened_feature_count": 40,
  "input_shape": [
    40
  ],
  "metadata_error": null,
  "metrics": {
    "r2": {
      "aggregation": "metric_defined",
      "definition": "mean target-wise R^2 over targets with nonzero target variance",
      "direction": "max",
      "name": "r2"
    },
    "rel_l2": {
      "aggregation": "global_frobenius",
      "definition": "||prediction - target||_F / ||target||_F; denominator is 1 when target norm is zero",
      "direction": "min",
      "name": "rel_l2"
    },
    "rmse": {
      "aggregation": "metric_defined",
      "definition": "sqrt(mean((prediction - target)^2)) over all rows and targets",
      "direction": "min",
      "name": "rmse"
    }
  },
  "noise": {
    "augmentation": "copies add independent N(0,(noise_scale*noise_std)^2) to already noisy training observations",
    "model": "additive i.i.d. Gaussian on every sensor reading",
    "original_observations_retained": true,
    "replica_variance_if_independent": "noise_std^2 * (1 + noise_scale^2)",
    "std": 0.001,
    "units": "raw observation units"
  },
  "objective": "Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.",
  "primary_metric": "rel_l2",
  "prior": "i.i.d. uniform on [0.5, 2.0] per segment",
  "privileges": {
    "custom_code": true,
    "forward_authorized": true,
    "max_forward_evals": 3000000,
    "simulator_has_forward": true
  },
  "schema_version": 3,
  "splits": {
    "test": {
      "fingerprint": "f7409783f1e852fbb42dec8ce2ec55abef7b88e3d047c0766a62d7b9b626f94e",
      "role": "final_held_out"
    },
    "train": {
      "fingerprint": "1a36da0686b153acbdb42c4cdeea03503a4821022cbbe96977adc1208caae87c",
      "role": "training"
    },
    "val": {
      "fingerprint": "1a64dda07d161f67b29ce7300e34718eae5c43b346932cca5b734a2ae0b7c53a",
      "role": "development"
    }
  },
  "target_count": 2,
  "target_shape": [
    2
  ],
  "target_transform": "method-specific: optional log before estimator fit, exp after prediction",
  "task": "regression",
  "training_rows": 240,
  "transform_order": [
    "observation_noise_augmentation",
    "features",
    "scaler",
    "pca",
    "estimator"
  ]
}
```

## Delegation, fallback attribution and costs

Agent instances: `5`; specialist tasks: `2`.

Fallback reasons (role calls): `{"service_error": 1}`.

Costs: `{"by_stage_s": {"final": 7.49, "study": 16.46}, "compute_budget_s": 120.0, "compute_remaining_s": 96.06, "compute_used_s": 23.94, "gpus": [], "jobs": 9, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {"execution": 17}, "llm_calls": 17, "llm_calls_max": 24, "llm_cost_usd": 9.8793, "llm_seconds": 1472.85, "wall_budget_s": 2400.0, "wall_elapsed_s": 1490.19, "workers": 2}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 30.0, "confirmation_used_seconds": 7.486, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-002:4", "seconds": 13.1}, {"action": "release", "actual_seconds": 8.746, "group_id": "study:S-Q1-002:4", "overrun_seconds": 0.0, "reserved_seconds": 13.1}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-003:6", "seconds": 13.1}, {"action": "release", "actual_seconds": 7.714, "group_id": "study:S-Q1-003:6", "overrun_seconds": 0.0, "reserved_seconds": 13.1}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 30.0}, {"action": "release", "actual_seconds": 7.486, "group_id": "final-export", "overrun_seconds": 0.0, "reserved_seconds": 30.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 96.054, "reserved_seconds": 0, "total_seconds": 120.0, "used_seconds": 23.946}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
