# V3 research product: heat_v3_opus_top_s0_20261114_145516

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: Why stop. The study limit has been reached: action-012, a revise to S-Q1-007, was rejected with 'ValueError: study limit'. No new study or revision can be registered. The remaining open blocking objections cannot be closed with the actions still allowed:
- 073 and 074 are redacted and cannot be addressed.
- 030, 100, 101 and 102 require 2000 bootstrap replicates and labelled interval levels. The harness has capped n_bootstrap at 200 in every paired diagnostic (resu-00109, 00110, 00125, 00129, 00266, 00267), and a delegated re-run also returned 200.
- 103 requires the treatment's printed diagnostics, which the harness discarded (runtime_diagnostics is empty).
- 105 requires re-running mlp_fixed, which needs a new study.
Another diagnose or delegate would reproduce the same capped outputs, and delegation is blocked by the depth/children limit (action-006). Stopping now is better than spending budget on actions that cannot change any verdict.

Outcome under the pre-registered rules. Every claim stays inconclusive / not evaluable, and none is self-certified:
- C-Q1-1b and C-Q1-2b (S-Q1-002): blocking implementation objections are unresolved, and the bootstrap ran 200 replicates against 2000 registered.
- C-Q1-5a and C-Q1-5c (S-Q1-005): the bootstrap ran 200 replicates against 2000 registered, the interval level is not reported, the reporting criteria were not observed, and redacted objections 073 and 074 are open.
- C-Q1-5b: the control arm failed. The harness wraps custom_ml in TransformedTargetRegressor, so the arm's positivity guard raised (resu-00257). No MLP comparison exists for this study, and resu-00269 holds treatment-side values only.

Descriptive development evidence. The validation split was reused across several validation-informed revisions, so none of this is confirmatory. n_val = 750; row_hash 260db427 and train_row_hash 4dea572e are identical across all arms.
- Validation rel_l2:
  - transparent_mle (resu-00261, source_hash ef4142b6, fit-time gates G1-G7 passed because the arm completed): 0.019322.
  - harness physics_mle (resu-00107/00263): 0.019333.
  - torch_mlp log-target: 0.02805 at seed 0, 0.03016 at seed 1.
  - extra_trees log-target: 0.07741 at seed 0, 0.07711 at seed 1.
- Paired relative improvement 1 - R (200 replicates, interval level unstated):
  - transparent_mle vs ET: 0.750, interval [0.729, 0.770].
  - harness MLE vs MLP: 0.311, interval [0.272, 0.352].
  - transparent_mle vs harness reference: 0.00054, interval [-0.00026, 0.00133]. This is aggregate agreement only. It does not show row-level identity or identify the estimator (objections 071, 092, 108).
- Per-segment rel_l2 for the likelihood arms is about [0.027, 0.006, 0.005, 0.026]. The outer segments 1 and 4 carry roughly 5x the error of the inner segments, and the same pattern appears in both learned arms. This is an early exploratory signal for Q-3 and is not a local-Fisher or identifiability result.
- Cost asymmetry: transparent_mle used about 113k forward evaluations (about 139 per row in predict, 11.4 ms per row). The harness MLE used about 177k (about 20 ms per row). The learners used no forward evaluations and took 0.01-0.17 ms per row. About 0.47M forward evaluations were used in total, against a cap of 3M.

Final export. The best executed full-fidelity arm is S-Q1-005.transparent_mle (val rel_l2 0.019322). Its procedure is auditable code with source_hash ef4142b6. Its result is statistically indistinguishable from the harness physics_mle (0.019333).

Scope and caveats:
- Inverse-crime condition: the forward model is exactly specified, with known sigma = 0.001 and a U[0.5,2]^4 prior.
- The learned controls are fixed and untuned, with 1-2 seeds. Uncertainty covers row sampling only.
- The output is the best-objective projected-LM iterate, not a certified MLE (objections 063, 111).
- The CFL number and interface averaging are unverified (018, 040, 067, 080, 094).
- Each comparison stands on its own; there is no family-wise claim.

Q-2 (hybrid initializer) and Q-3 (Fisher identifiability) were not studied because the study limit was reached. They remain open.

Everything above is for independent review.

Model export certified: **False**. Studies complete: **1/6**.

Model performance and study completion are separate outcomes.

## Actual exported method

- custom-solver(fwd) (kind `custom_solver`, method hash `f47bba887760`, source hash `caae064c48785e14ff12e75f7f32a1fd665ca52734db6d8510a6b4cf9baf3fec`, forward access `True`).
  Observed fit diagnostics: `{}`

Validation selection value: `0.019322146939217813`. Final metrics: `{}`.

Export certificate: `{"atol": 1e-10, "certified": false, "checks": {"finite_shape": true, "partition": false, "reload": true, "repeat": true}, "input_role": "development", "max_abs_differences": {"partition": 0.0010889326404079913, "reload": 0.0, "repeat": 0.0}, "n_rows": 750, "output_shape": [750, 4], "passed": false, "protocol": "pristine-repeat-reload-partition-v1", "reasons": ["partition check failed"], "rtol": 1e-08, "schema_version": 3}`.

Lineage: `{"idea-016": {"arm_id": "S-Q1-005.transparent_mle", "baseline_identity": false, "gate": "gate-00241", "label": "custom-solver(fwd)", "origin": "proposal", "study_id": "S-Q1-005"}}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-Q1-001 | 0/3 | superseded |  |
| S-Q1-002 | 3/3 | complete | all registered required work completed |
| S-Q1-003 | 0/4 | superseded |  |
| S-Q1-004 | 0/4 | superseded |  |
| S-Q1-005 | 3/4 | incomplete | required work remains unresolved |
| S-Q1-006 | 0/4 | superseded |  |

### S-Q1-001: Does forward-model inversion (physics MAP/MLE with a log transform and multistart) beat a strong learned regressor (extra_trees or torch MLP on standardized raw features with a log target) on validation rel_l2 for the four conductivities?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-001.et_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.mlp_log` (baseline, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.physics_map` (treatment, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-001.paired_phys_vs_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-001.paired_phys_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-001.per_target_phys`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-002: On this exactly specified simulator (explicit FD, 40 points, 200 steps, Dirichlet ends, known Gaussian IC, sensors [5,15,25,35], sigma=0.001, U[0.5,2]^4 prior), does box-constrained MLE inversion in log-conductivity with 3 starts achieve lower validation rel_l2 than two specific fixed, untuned learned regressors (extra_trees and torch MLP, standardized raw features, log target, no noise augmentation)?

Owner: `researcher-003`; registered evidence: `['resu-00101', 'resu-00105', 'resu-00107', 'resu-00100', 'resu-00103']`.

- Arm `S-Q1-002.et_log` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.mlp_log` (control, required=True): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9927600207712438, "rel_l2": 0.028045400632179675, "rmse": 0.0369844869642745}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.physics_mle` (treatment, required=True): `ok`; method `physics-MLE[log,ms=3,it=30]`; metrics `{"r2": 0.9965576397109494, "rel_l2": 0.019332653763499383, "rmse": 0.02549467167463335}`; effective hash `0e0748889aca79572e090ef4e40e2bff3bfc7052d56540f5d5fa4bbbd6243618`.
  Normalization changes: `[{"effective": 32, "effective_present": true, "field": "params.posterior_samples", "requested": 0, "requested_present": true}]`.
- Arm `S-Q1-002.et_log_seed1` (diagnostic, required=False): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.9452383805062798, "rel_l2": 0.07710818941173594, "rmse": 0.10168536593714962}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.
- Arm `S-Q1-002.mlp_log_seed1` (diagnostic, required=False): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9916315078180268, "rel_l2": 0.030156978716347062, "rmse": 0.03976910156658218}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[{"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": 0.0, "requested_present": true}]`.

Diagnostics and absences:

- `S-Q1-002.paired_mle_vs_et`: `ok`; artifact `resu-00109`; reason `registered computation returned`.
- `S-Q1-002.paired_mle_vs_mlp`: `ok`; artifact `resu-00110`; reason `registered computation returned`.
- `S-Q1-002.per_target_mle_vs_et`: `complete`; artifact `resu-00111`; reason `registered computation returned`.
- `S-Q1-002.per_target_mle_vs_mlp`: `complete`; artifact `resu-00112`; reason `registered computation returned`.
- `S-Q1-002.per_sample_mle_vs_et`: `complete`; artifact `resu-00113`; reason `registered computation returned`.

Claim `C-Q1-1b`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-002.et_log", "margin": 0.05, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics_mle"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019332653763499383}`; effect `0.7502464857509381`; interval `[0.7283495937967271, 0.7702705032022985]`.
Evidence `['resu-00107', 'resu-00101']`; unmet requirements `['objection-017', 'objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-024', 'objection-029']`.

Claim `C-Q1-2b`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-002.mlp_log", "margin": 0.05, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics_mle"}`; values `{"control": 0.028045400632179675, "metric": "rel_l2", "treatment": 0.019332653763499383}`; effect `0.31066580160324636`; interval `[0.2721292524004297, 0.3518783363059486]`.
Evidence `['resu-00107', 'resu-00105']`; unmet requirements `['objection-017', 'objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-024', 'objection-029']`.

### S-Q1-003: On the exactly specified simulator (inverse-crime condition disclosed; sigma=0.001; U[0.5,2]^4 prior), does an auditable box-constrained log-k Gaussian MLE (a) reproduce the harness physics_mle within 5% relative validation rel_l2, and (b) beat the fixed, untuned extra_trees and torch MLP log-target controls by at least 5% relative, (c) with documented forward identity, stability, objective-only start selection and convergence diagnostics?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-003.transparent_mle` (treatment, required=True): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.physics_mle_ref` (baseline, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.et_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-003.mlp_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-003.paired_tmle_vs_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-003.paired_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-003.paired_tmle_vs_ref`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-003.per_target_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-004: On the exactly specified simulator (inverse-crime condition disclosed; sigma=0.001; U[0.5,2]^4 prior), does an auditable box-constrained log-k Gaussian MLE (a) reproduce the harness physics_mle aggregate validation rel_l2 within a pre-specified equivalence band, and (b) beat the fixed, untuned extra_trees and torch MLP log-target controls by at least 5% relative validation rel_l2?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-004.transparent_mle` (treatment, required=True): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-004.physics_mle_ref` (baseline, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-004.et_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-004.mlp_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-004.paired_tmle_vs_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-004.paired_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-004.paired_tmle_vs_ref`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-004.per_sample_tmle_vs_ref`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-004.per_target_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-005: On the exactly specified simulator (inverse-crime condition disclosed; sigma=0.001; U[0.5,2]^4 prior), does an auditable box-constrained log-k Gaussian MLE procedure (a) reproduce the harness physics_mle aggregate validation rel_l2 within a pre-specified equivalence band, and (b) beat a fixed, untuned extra_trees log-target control and an auditable fixed-epoch MLP log-target control by at least 5% relative validation rel_l2? Development evidence only.

Owner: `researcher-003`; registered evidence: `['resu-00261', 'resu-00263', 'resu-00259', 'resu-00257', 'resu-00256']`.

- Arm `S-Q1-005.transparent_mle` (treatment, required=True): `ok`; method `custom-solver(fwd)`; metrics `{"r2": 0.9965613757532694, "rel_l2": 0.019322146939217813, "rmse": 0.025480815944391732}`; effective hash `8547565eaa6d38d38df1d4e1a88a38658828962242fe3d252f5dd33c6d797cdf`.
  Normalization changes: `[]`.
- Arm `S-Q1-005.physics_mle_ref` (baseline, required=True): `ok`; method `physics-MLE[log,ms=3,it=30]`; metrics `{"r2": 0.9965576397109494, "rel_l2": 0.019332653763499383, "rmse": 0.02549467167463335}`; effective hash `0e0748889aca79572e090ef4e40e2bff3bfc7052d56540f5d5fa4bbbd6243618`.
  Normalization changes: `[]`.
- Arm `S-Q1-005.et_log` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-005.mlp_fixed` (control, required=True): `error`; method `custom_ml`; metrics `{}`; effective hash `bb6af0fa97f0781674f48e149f48024cc0082d22ff3c28286241f7ca7032ded1`.
  Normalization changes: `[{"effective": "none", "effective_present": true, "field": "target_transform", "requested": null, "requested_present": false}]`.
- Arm `S-Q1-005.mlp_log_harness` (diagnostic, required=False): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9927600228848563, "rel_l2": 0.028045396586525308, "rmse": 0.036984481629123364}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-005.paired_tmle_vs_et`: `ok`; artifact `resu-00266`; reason `registered computation returned`.
- `S-Q1-005.paired_tmle_vs_mlp`: `missing`; artifact `None`; reason `control predictions unavailable`.
- `S-Q1-005.paired_tmle_vs_ref`: `ok`; artifact `resu-00267`; reason `registered computation returned`.
- `S-Q1-005.per_sample_tmle_vs_ref`: `complete`; artifact `resu-00268`; reason `registered computation returned`.
- `S-Q1-005.per_target_tmle_vs_mlp`: `complete`; artifact `resu-00269`; reason `registered computation returned`.

Claim `C-Q1-5a`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-005.et_log", "margin": 0.05, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-005.transparent_mle"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019322146939217813}`; effect `0.7503822206748664`; interval `[0.7286620264557506, 0.7703376937235787]`.
Evidence `['resu-00261', 'resu-00259']`; unmet requirements `['objection-073', 'objection-074']`.

Claim `C-Q1-5b`: **inconclusive**. required evidence is missing or unmatched.
Predicate: `{"control": "S-Q1-005.mlp_fixed", "margin": 0.05, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-005.transparent_mle"}`; values `{"control": null, "metric": "rel_l2", "treatment": 0.019322146939217813}`; effect `None`; interval `None`.
Evidence `['resu-00261', 'resu-00257']`; unmet requirements `['diagnostic:S-Q1-005.paired_tmle_vs_mlp', 'control:S-Q1-005.mlp_fixed', 'verified implementation contract', 'finite control metric']`.

Claim `C-Q1-5c`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-005.physics_mle_ref", "margin": 0.0476, "metric": "rel_l2", "operator": "equivalent", "treatment": "S-Q1-005.transparent_mle"}`; values `{"control": 0.019332653763499383, "metric": "rel_l2", "treatment": 0.019322146939217813}`; effect `0.0005434755316110651`; interval `[-0.0002633017565882629, 0.0013339233030709066]`.
Evidence `['resu-00261', 'resu-00263']`; unmet requirements `['objection-073', 'objection-074']`.

### S-Q1-006: Development evidence only; this is a validation-informed revision of S-Q1-005. On the exactly specified simulator (inverse-crime condition, sigma=0.001, U[0.5,2]^4 prior), does the auditable projected-LM log-k likelihood procedure, with its predict-time reporting criteria enforced as hard gates, (a) reproduce the harness physics_mle aggregate validation rel_l2 within the pre-specified equivalence band, and (b) beat a fixed untuned extra_trees log-target control and an auditable fixed-epoch MLP log-target control by at least 5% relative validation rel_l2?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-006.transparent_mle` (treatment, required=True): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-006.physics_mle_ref` (baseline, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-006.et_log` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-006.mlp_fixed` (control, required=True): `missing`; method `custom_ml`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-006.paired_tmle_vs_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-006.paired_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-006.paired_tmle_vs_ref`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-006.per_target_tmle_vs_mlp`: `missing`; artifact `None`; reason `registered computation returned`.

## Independent critic and uncertainty

- `objection-001` (S-Q1-001, open): Define the decision rule for 'at least 5% relative better' before running any arm: improvement = (rel_l2_control - rel_l2_treatment) / rel_l2_control, and the claim holds only if the lower bound of a pre-specified paired CI (for example a 95% row-resampling bootstrap) exceeds 0.05. A point estimate above 0.05 must not be enough.. Resolution: `unresolved`.
- `objection-002` (S-Q1-001, open): rel_l2 is a global Frobenius ratio and does not decompose per row. The paired uncertainty analysis must resample whole validation rows jointly for treatment and control, and recompute both numerator and denominator norms on each replicate. Averaging per-row ratios is not acceptable. Report the validation row count, since the split size is not stated anywhere.. Resolution: `unresolved`.
- `objection-003` (S-Q1-001, open): Implementation verification of the physics forward model. Show that the inversion's forward operator matches the data generator in every respect: 40-point grid, 200 explicit steps, Dirichlet ends, divergence-form flux discretization at segment interfaces, Gaussian IC at x=0.5, sensor indices [5,15,25,35], and the exact time indices behind the 80 features (80 = 4 sensors × 20 times). Required check: simulating a sample of validation rows at their true k should leave residuals consistent with N(0, 0.001²), for example a per-row chi-square near 80.. Resolution: `unresolved`.
- `objection-004` (S-Q1-001, open): Verify explicit-scheme stability over the whole prior box. With k up to 2.0, the condition dt·k_max/dx² ≤ 0.5 must hold. If it fails for some k, the optimizer can enter unstable regions or exploit numerical artifacts. Report the CFL number at k=2.0 and confirm that no NaN or Inf forward evaluations occurred.. Resolution: `unresolved`.
- `objection-005` (S-Q1-001, open): Disclose the inverse-crime condition. If the inversion uses the identical simulator and noise model that generated the data, the comparison favors physics by construction. The claim text must say the forward model is exactly specified. If feasible, add a non-blocking robustness check: a perturbed grid or model mismatch, using validation data only.. Resolution: `unresolved`.
- `objection-006` (S-Q1-001, open): Resolve the physics estimator specification. The arm is labelled MAP and sets estimator='mode', but it also sets posterior_samples=128 and tikhonov=0. State what is actually returned (MAP, MLE, or posterior mean) and what the 128 samples are used for. With a uniform prior on k and optimization in log k, a log-space 'MAP' without the Jacobian term is the constrained MLE, not the MAP. The label and the claim text must match the implementation.. Resolution: `unresolved`.
- `objection-007` (S-Q1-001, open): Optimizer convergence diagnostics for physics_map. Report: the distribution of final negative log-likelihood or chi-square per row; how often the 3 starts disagree; the fraction of rows where the optimizer stopped at the 30-iteration cap without meeting a tolerance; and the fraction of solutions pinned at the bounds [0.5, 2.0]. State how bounds are enforced in log space (clipping, projection, or reparameterization) and the gradient method used (adjoint, autodiff, or finite differences). If gradients are adjoint or autodiff, include a gradient check.. Resolution: `unresolved`.
- `objection-008` (S-Q1-001, open): Forward-evaluation budget accounting. Count the forward evaluations physics_map uses on the validation split: starts × iterations × evaluations per iteration, plus line searches, finite-difference gradients and posterior samples. Show the total is ≤ 3,000,000, and confirm that the final test evaluation fits inside any remaining budget. Report per-row failures and how failed rows are scored. They must not be silently dropped.. Resolution: `unresolved`.
- `objection-009` (S-Q1-001, open): Baseline strength and tuning parity. The question calls ET and the MLP 'strong learned regressors', but each is a single untuned configuration run once. The MLP has no early stopping, validation monitoring or LR schedule. Either (a) give each arm an equal, pre-registered tuning budget, using a split carved from train or internal CV and not the claim-bearing validation set, or (b) reword the claims to refer to these specific fixed configurations only.. Resolution: `unresolved`.
- `objection-010` (S-Q1-001, open): Seed variance. Every arm uses only seed 0. For the stochastic learners (ET bootstrap and feature subsampling; MLP initialization and batching), run at least 3–5 seeds. The uncertainty statement must include seed variability, not just row resampling. Report the multistart seeds for physics_map.. Resolution: `unresolved`.
- `objection-011` (S-Q1-001, open): Specify noise augmentation per arm. The transform order lists observation_noise_augmentation, but no arm gives noise_scale or a copy count. State whether augmentation is applied to ET and the MLP. If it is, confirm the noise is added only to training rows and never to validation rows.. Resolution: `unresolved`.
- `objection-012` (S-Q1-001, open): Confirm the scale on which metrics are computed. All three arms must be scored on the original conductivity scale (exp applied after log-target fitting), and on the same validation rows in the same order. Verify that the val split fingerprint edfcc32d... matches the data actually loaded for every arm.. Resolution: `unresolved`.
- `objection-013` (S-Q1-001, open): Held-out discipline. Do not compute any test-split fingerprint aa0964... metric before the claim verdicts are frozen. That includes the 'final' check of physics_map. Log any access to the test split.. Resolution: `unresolved`.
- `objection-014` (S-Q1-001, open): Per-target breakdown. Promote the per_target_phys diagnostic to cover both controls, and add local Fisher sensitivities at representative k. Segments far from the central pulse may be weakly informed within the finite window, and an aggregate rel_l2 win could hide a loss on poorly identified segments. Interpret Fisher results only as local sensitivity, as the contract requires.. Resolution: `unresolved`.
- `objection-015` (S-Q1-001, open): Missing evidence. evidence_ids is empty and the study status is 'planned'. No verdict on C-Q1-1 or C-Q1-2 can be issued until arm outputs, the paired diagnostics, and the implementation-verification artifacts above are attached.. Resolution: `unresolved`.
- `objection-016` (S-Q1-001, open): Synthesis scope. Even if both claims pass, the conclusion must be limited to: this exactly specified simulator, sigma=0.001 noise, the uniform prior on [0.5, 2], and these specific baseline configurations. It must not generalize to 'forward-model inversion beats learned regressors'. The write-up should also report the compute asymmetry: per-row inversion cost against amortized inference.. Resolution: `unresolved`.
- `objection-017` (S-Q1-002, open): Implementation verification of forward-model identity: show that the MLE's internal forward operator matches the data-generating simulator exactly in grid (40 points), time step and 200 steps, observation-time subsampling, sensor indices [5,15,25,35], Gaussian IC centre and width, Dirichlet ends, and segment-to-grid mapping. The check should evaluate the MLE forward at the true k for a sample of validation rows and compare with the noiseless generator output (max abs difference ~ machine precision). Without this, 'exactly specified forward model' is asserted, not demonstrated.. Resolution: `unresolved`.
- `objection-018` (S-Q1-002, open): Report the explicit-FD stability condition dt <= dx^2 / (2 * k_max) at k_max = 2.0, including the divergence-form harmonic or arithmetic interface averaging used. Confirm that no training, validation, or MLE-trial trajectory is unstable or non-finite inside the box [0.5, 2]^4. An unstable corner would corrupt data and the MLE objective differently for different arms.. Resolution: `unresolved`.
- `objection-019` (S-Q1-002, open): Matched evaluation inputs: verify that the MLE consumes the identical stored noisy validation observations (same noise realization, same row order and row IDs) that the ET and MLP arms are scored on, and does not re-simulate or re-noise them. The paired bootstrap is invalid if rows are not aligned. Provide row-ID hash equality across arms.. Resolution: `unresolved`.
- `objection-020` (S-Q1-002, open): Pre-specify and report MLE failure handling. Report: rows with non-converged optimizer, NaN/inf, or iteration cap hit; how many estimates land on the box boundary; and the fallback rule (e.g., best of 3 starts by objective, never by validation error against truth). Selecting among starts using target knowledge would be leakage.. Resolution: `unresolved`.
- `objection-021` (S-Q1-002, open): Optimization adequacy evidence. 30 iterations and 3 starts are asserted, not justified. Report the final objective relative to the noise floor (chi^2 / (n_obs) ~ 1 at sigma = 0.001), the fraction of rows where the three starts disagree beyond tolerance, and a noiseless sanity check (recovery of true k to small error). Without this, an MLE loss could reflect optimizer failure rather than the estimator.. Resolution: `unresolved`.
- `objection-022` (S-Q1-002, open): State the validation set size and confirm the test split is untouched. Confirm that MLE settings (iters = 30, multistart = 3, log transform), the 0.05 margin, and the decision rule were fixed before seeing validation results. This matters because this study supersedes S-Q1-001, which may have used the same validation split. If S-Q1-001 informed these choices, disclose it and treat the validation CI as optimistically biased.. Resolution: `unresolved`.
- `objection-023` (S-Q1-002, open): Forward-evaluation budget accounting: report the total forward solves consumed by MLE (rows × starts × iterations × evals per iteration, including finite-difference gradients and line search) and confirm it stays within max_forward_evals = 3,000,000. Report wall-clock as a cost disclosure alongside accuracy.. Resolution: `unresolved`.
- `objection-024` (S-Q1-002, open): Learned-control hygiene: verify the scaler is fit on training rows only, the MLP uses the final epoch-150 weights with no validation-based checkpoint selection or early stopping, and both learned arms use the same 3000 training rows (fingerprint cdad4859…). Verify that the log/exp target transform is applied as specified, so that scores are computed on the original conductivity scale for all arms.. Resolution: `unresolved`.
- `objection-025` (S-Q1-002, open): Bootstrap specification: state the CI method (percentile vs BCa), the RNG seed, and that the improvement statistic is recomputed per replicate as 1 - ||E_MLE||_F / ||E_ET||_F. Both numerators and the shared denominator must come from the same resampled row set. Also report the point estimate and the full CI, not only pass/fail.. Resolution: `unresolved`.
- `objection-026` (S-Q1-002, open): Multiplicity and synthesis scope: the two claims are each tested at 95% with no correction. Any synthesized statement such as 'MLE beats learned regressors' requires both claims to pass, which should use a familywise-adjusted level (e.g., Bonferroni 97.5% lower bounds) or be explicitly worded as two separate claims.. Resolution: `unresolved`.
- `objection-027` (S-Q1-002, open): Generalization limits must be stated in any verdict. The result is conditional on the inverse-crime setting, on seed 0 of fixed untuned controls (seed-1 arms are optional, so seed variability may be unquantified), and on validation data only. It must not be extrapolated to model misspecification, tuned learners, other noise levels, or held-out test performance.. Resolution: `unresolved`.
- `objection-028` (S-Q1-002, open): Estimator-target mismatch caveat: rel_l2 is an MSE-type loss, favoring a posterior-mean estimator, while the MLE/mode is not MSE-optimal under the U[0.5,2]^4 prior. The log-trained learners exp-transform a conditional log-mean, which is also biased on the original scale. Per-target diagnostics (optional per_target arms) should be run and reported. They would show whether any advantage concentrates in well-identified segments and whether weakly sensed outer segments (far from the central pulse) behave differently.. Resolution: `unresolved`.
- `objection-029` (S-Q1-002, open): Disclose the observation layout behind the 80 features (presumably 4 sensors × 20 sampled times out of 200 steps). Confirm that the MLE likelihood uses exactly those 80 observations with the known sigma, not the full time series. Giving the MLE denser observations than the learners would break the comparison.. Resolution: `unresolved`.
- `objection-030` (S-Q1-002, open): Both claims pre-specify a 95% paired bootstrap with 2000 replicates, but both reported diagnostics (resu-00109, resu-00110) show n_bootstrap = 200. Recompute both intervals with exactly 2000 joint whole-row replicates, recomputing the Frobenius numerator and denominator per replicate. Report the CI method (percentile or other) and the bootstrap RNG seed. A decision rule run with a different replicate count from the one registered does not satisfy that rule, whatever the margin.. Resolution: `unresolved`.
- `objection-031` (S-Q1-002, open): Both claims set requires_implementation_verification = true, yet every arm has method_contract.diagnostics.verified = false, empty declared/observed diagnostics, and source_hash = null. Provide a source hash for the code that produced resu-00101, resu-00105 and resu-00107, and independent verification of each arm's fit/predict path. Gate checks such as no_leakage and metric_correct are self-reported approvals, not verification.. Resolution: `unresolved`.
- `objection-032` (S-Q1-002, open): Physics arm normalization: posterior_samples was requested as 0 and silently clipped to 32 under estimator = 'mode'. Show from code or an ablation that the reported point predictions are the multistart best-objective mode in log k, mapped by exp, and not a posterior mean or median over the 32 samples. If the samples affect the predictions, the evaluated method differs from the claimed 'MLE inversion' and the claim text must change.. Resolution: `unresolved`.
- `objection-033` (S-Q1-002, open): Physics arm leakage: the interface is fit(X_train, y_train)/predict(X). Document what, if anything, the MLE takes from training data, including initialization, box, and sigma. Confirm that multistart initial points are drawn independently of validation targets, for example from the prior or a fixed design with a recorded seed. Confirm that no validation labels enter the optimizer or the start selection.. Resolution: `unresolved`.
- `objection-034` (S-Q1-002, open): Report optimizer health for the physics arm on the 750 validation rows: the convergence or termination reason for each start, the final negative log-likelihood against its expected value under sigma = 0.001 (about 80/2 per row), the fraction of rows with any coordinate at the [0.5, 2.0] bound, the number of rows where the starts disagree, and the total forward evaluations used against the 3,000,000 budget. With 30 iterations and 3 starts, the per-sample l2 tail (max 0.209) may reflect non-converged or local-minimum rows rather than intrinsic ill-conditioning.. Resolution: `unresolved`.
- `objection-035` (S-Q1-002, open): Data-driven arms: noise_scale was normalized from 0.0 to 1.0. Confirm that with noise_aug = 0 this is inert, meaning no augmented copies were generated and the training matrix holds exactly 3000 rows. A short code reference or an equality check of the training-set fingerprint is enough.. Resolution: `unresolved`.
- `objection-036` (S-Q1-002, open): The seed-1 diagnostic arms (et_log_seed1, mlp_log_seed1) completed, and resu-00100 and resu-00103 are listed in evidence_ids, but no result or synthesis uses them. Report their validation rel_l2 and the paired effect against the MLE. The registered uncertainty explicitly excludes training-seed variation, so the synthesis should state how much the margin moves across seeds, especially for the MLP control. Without this, the claims hold only for the seed-0 models, as worded.. Resolution: `unresolved`.
- `objection-037` (S-Q1-002, open): The per-target and per-sample diagnostics (resu-00111, resu-00112, resu-00113) report identical per_target_rel_l2 and per_sample_l2 arrays for the ET and MLP comparisons. They appear to contain treatment-side values only. Relabel them as treatment-only, or add control-side and paired per-target values, before any synthesis cites them as per-target comparisons.. Resolution: `unresolved`.
- `objection-038` (S-Q1-002, open): Synthesis scope: the MLE uses the identical simulator, noise model, and known sigma that generated the data (the inverse-crime condition). The synthesis must limit the conclusion to this exact simulator and these two fixed, untuned regressor configurations. It must not generalize to physics-based versus learned inversion, to model misspecification, to different discretizations, or to the held-out test split. The test and stress splits must remain unused.. Resolution: `unresolved`.
- `objection-039` (S-Q1-002, open): Supersession and forking: S-Q1-002 supersedes S-Q1-001 on the same validation split, with revised claims C-Q1-1b and C-Q1-2b. State what changed (controls, margin, decision rule, or configuration) and confirm that the changes were not chosen after seeing S-Q1-001 validation results. If they were, disclose this as a validation-informed revision in the synthesis.. Resolution: `unresolved`.
- `objection-040` (S-Q1-002, open): Forward-model validity across the prior box: confirm that the explicit finite-difference scheme (40 points, 200 steps) satisfies its stability condition for all k in [0.5, 2.0]^4, using the actual dt/dx^2 and the interface averaging of k. An unstable corner would distort both the generated data and the MLE likelihood landscape near the upper bound.. Resolution: `unresolved`.
- `objection-041` (S-Q1-003, open): Input-pipeline verification (blocking for C-Q1-3a/b/c): the custom_solver arm does not declare features/scaler/pca/target_transform, yet transform_order lists scaler, pca and an optional log target transform before the estimator. Show from harness code or a logged check that TransparentLogMLE.fit/predict receive raw, unscaled 80-dim observations in the same sensor-by-time ordering that ctx.forward returns, and receive y on the original k scale. A standard-scaled X, a PCA-reduced X, or a log-transformed y would make the chi-square residuals, the forward identity and the noiseless recovery meaningless. The harness might also apply exp() a second time to predictions on the original k scale.. Resolution: `unresolved`.
- `objection-042` (S-Q1-003, open): Forward identity must gate the run rather than only be printed. obs_dim_forward is compared with obs_dim_features but never asserted. Before predictions are scored, require obs_dim_forward == obs_dim_features == 80. Also require chi2_at_truth mean to be about 80 ± a few times sqrt(160), with residual_std_at_truth about 1e-3. A mean near 160 would mean ctx.forward adds its own noise, which would corrupt the FD Jacobian. A much larger value would mean an ordering, time-grid or scale mismatch. Pre-specify the acceptance band.. Resolution: `unresolved`.
- `objection-043` (S-Q1-003, open): Confirm ctx.forward is deterministic and noise-free: identical inputs must give bit-identical outputs. Stochastic forward output with FD step h=1e-4 in log k would make J dominated by noise. A two-call identity check on a few rows should be logged.. Resolution: `unresolved`.
- `objection-044` (S-Q1-003, open): C-Q1-3c has no pre-specified equivalence decision rule. Unlike 3a/3b, it does not say what 'within ±5% relative' means. Specify the statistic (e.g. ratio rel_l2_TMLE/rel_l2_REF), the interval (a 90% paired bootstrap CI for TOST at alpha=0.05, or a 95% CI), and the bounds ([0.95, 1.05] or [1/1.05, 1.05]). State that the claim fails if the interval crosses either bound.. Resolution: `unresolved`.
- `objection-045` (S-Q1-003, open): The inference in C-Q1-3c ('equivalence is indirect evidence that harness predictions are MLE modes') does not follow from aggregate rel_l2 equivalence. Two estimators can match in rel_l2 while differing row by row, and a posterior mean or median could also land within 5%. Either drop the mode-identification sentence, or add a direct per-row comparison. That comparison should report quantiles of |log k_TMLE − log k_REF| and the chi-square of each prediction under the observed data, and require near-identity, e.g. p90 < 1e-3. Also note that the start sets, LM schedule and seeds of the two arms are not matched. A 5% gap could reflect optimizer differences rather than estimator type.. Resolution: `unresolved`.
- `objection-046` (S-Q1-003, open): Part (c) of the study question (forward identity, stability, objective-only start selection, convergence) has no corresponding claim and no acceptance criteria. Examples of missing criteria: maximum frac_cap_hit_without_tol, maximum frac_nonfinite_objective, stability flags must be true, and a noiseless_rel_l2 threshold. Either register (c) as a claim with pre-specified thresholds, or remove it from the question so synthesis cannot assert it informally.. Resolution: `unresolved`.
- `objection-047` (S-Q1-003, open): Bound handling is projection of an unconstrained LM step (np.clip after solve), not a true box-constrained method such as projected gradient with active-set freezing. When a component is at a bound and the gradient points outward, that component still enters the linear solve. The clipped step can then distort or block updates to the free components. Many rows hit the 1e9 lambda 'stuck' exit and are recorded as converged. Under the U[0.5,2] prior, a nontrivial fraction of truths lie near bounds. Report the 'stuck' exits separately from tolerance convergence. Also report validation error stratified by pinned versus unpinned rows, so that bound artifacts are visible.. Resolution: `unresolved`.
- `objection-048` (S-Q1-003, open): Convergence labeling is ambiguous. The 'conv' flag merges 'small decrease' with 'stuck at lam ≥ 1e9', so frac_cap_hit_without_tol understates non-convergence. Add separate fractions for tolerance-converged, stuck, and iteration-capped rows at the selected start.. Resolution: `unresolved`.
- `objection-049` (S-Q1-003, open): Multistart adequacy: three starts (midpoint plus two uniform draws) may not detect multimodality in a 4-parameter problem with 4 point sensors. Report frac_starts_disagree_gt_1e-3_logk on validation. If it is substantial, show that the selected chi2_per_dof is near 1. A best chi2/m well above 1 would indicate local minima. No added starts may be tuned on validation error.. Resolution: `unresolved`.
- `objection-050` (S-Q1-003, open): Explicit FD stability: with 40 points and 200 steps, the scheme's CFL condition depends on k_max = 2. The corner and 1000-draw finiteness checks are logged but do not gate the run. They should be required to be true. Finiteness also does not rule out oscillatory but finite solutions near the stability limit, so a monotonicity or positivity check on corner solutions is desirable.. Resolution: `unresolved`.
- `objection-051` (S-Q1-003, open): Bootstrap specification: record the bootstrap RNG seed. Confirm that resampling is of whole validation rows, shared across arms within each replicate. Confirm that rel_l2 is recomputed as a global Frobenius ratio inside each replicate. Report the validation row count. Use a percentile or BCa interval and state which.. Resolution: `unresolved`.
- `objection-052` (S-Q1-003, open): Multiplicity: C-Q1-3a and C-Q1-3b are two one-sided superiority tests at 95% on the same treatment. Either state that each claim is judged separately (no family-wise claim that 'MLE beats ML baselines'), or apply a Bonferroni/Holm adjustment before any combined synthesis statement.. Resolution: `unresolved`.
- `objection-053` (S-Q1-003, open): Scope of the ML-control comparisons: the controls are single-seed (seed 0), untuned, use no noise augmentation, and the MLP has no early stopping. The bootstrap captures only row-sampling variance, not training-seed variance. Synthesis must not generalize beyond 'these fixed configurations', and must restate the inverse-crime condition (exact simulator, known sigma) wherever the result is cited.. Resolution: `unresolved`.
- `objection-054` (S-Q1-003, open): Forward-budget accounting: log the actual forward calls for both forward-using arms (transparent_mle, physics_mle_ref) against max_forward_evals = 3,000,000. The estimate for the transparent arm is about 3 × (1 + 30 × 5) ≈ 453 calls per validation row plus about 46k in fit. Confirm the combined total fits the budget at the actual validation size, and that the budget is not silently truncated.. Resolution: `unresolved`.
- `objection-055` (S-Q1-003, open): Robustness of the batched linear solve: np.linalg.solve over the whole batch will raise and abort every row if any single A is singular or non-finite, for example after non-finite forward output. Either guard per row (e.g. fall back to lstsq or reject the step), or log that no exceptions occurred.. Resolution: `unresolved`.
- `objection-056` (S-Q1-003, open): Dependency S-Q1-002: state its status and which of its results (e.g. the physics_mle_ref configuration or forward-identity evidence) this study relies on. If it is unresolved, the reference arm's interpretation in C-Q1-3c is unsupported.. Resolution: `unresolved`.
- `objection-057` (S-Q1-003, open): Held-out discipline: confirm that no diagnostics, thresholds, start counts, iteration counts or fd_step values were chosen after viewing validation or test results. Confirm that the final test split is not scored for any arm in this study.. Resolution: `unresolved`.
- `objection-058` (S-Q1-004, open): Make the paired diagnostics for C-Q1-4a and C-Q1-4b match their decision rules. These two diagnostic specs (paired_tmle_vs_et, paired_tmle_vs_mlp) set no interval_level, sidedness or statistic. The claims need the 5th percentile of the bootstrap distribution of 1 - ||E_MLE||_F/||E_ctrl||_F, with numerator and denominator recomputed on the same resampled rows in each replicate. A harness default such as a two-sided 95% interval (2.5th percentile) or a mean-of-row-ratios statistic would test a different rule. Before any verdict, state the statistic, the quantile, the bootstrap RNG seed and the validation row count n_val in each diagnostic spec.. Resolution: `unresolved`.
- `objection-059` (S-Q1-004, open): Make the equivalence analysis for C-Q1-4c use the stated asymmetric ratio band. The claim record gives margin = 0.05 with operator 'equivalent', which a generic harness may read as |1 - R| <= 0.05 or |difference| <= 0.05. The pre-specified rule is a 90% CI on R inside [1/1.05, 1.05]. Encode R = rel_l2_TMLE/rel_l2_REF and the bounds [0.95238, 1.05] explicitly in the paired_tmle_vs_ref diagnostic, or show that the harness applies them.. Resolution: `unresolved`.
- `objection-060` (S-Q1-004, open): Pre-register what each claim's verdict becomes in these cases: a GateError in G1–G6 (the treatment produces no predictions), a failed physics_mle_ref arm, or a failed predict-time 'reporting_criteria' check (nonfinite objective, iteration_capped > 0.10, chi2/dof p50 > 1.2). Choose between 'not supported' and 'inconclusive / not evaluable'. The reporting criteria are computed on validation data, but their effect on a verdict is undefined. Leaving it open allows post-hoc discretion after seeing validation results.. Resolution: `unresolved`.
- `objection-061` (S-Q1-004, open): Show from the mlp_log arm configuration or harness code that it trains for exactly 150 epochs, keeps final-epoch weights, and does no early stopping or checkpoint selection on validation data. The claim text asserts all three, but the arm params do not specify them. If the harness default uses the validation split for early stopping, the control is tuned on the scoring split and C-Q1-4b's comparison is invalid.. Resolution: `unresolved`.
- `objection-062` (S-Q1-004, open): Report whether batched and single-row forward calls give identical outputs, and gate on it. The fit records batch_consistency_max_abs_diff but does not gate on it. G3 only shows that the same batch repeats bit-identically. LM evaluates the Jacobian perturbations for all active rows in one p*n batch. If the forward output depends on batch composition, the objective for a row depends on which other rows share its batch. Add a gate requiring exact equality or a small tolerance relative to sigma*h, or justify why it is unnecessary.. Resolution: `unresolved`.
- `objection-063` (S-Q1-004, open): Do not describe the treatment output as 'the MLE' without supporting evidence. It is the lowest-cost iterate from 3 starts and 30 projected-LM iterations. Clipping the step to the box (instead of projecting the gradient or using active-set handling) can stall at a bound. The 'small' criterion (max |zn - za| < 1e-9) can then mark a non-KKT point as tolerance-converged. With the validation results, report: the fraction iteration-capped, the fraction stuck at lambda_max, the fraction pinned at a bound, and the fraction of rows where starts disagree by more than 1e-3 in log k. Also report a check of projected-gradient KKT residuals on the best points.. Resolution: `unresolved`.
- `objection-064` (S-Q1-004, open): Fix the multistart draws per row. Random starts come from rng.uniform(size=(n, p)) with one seed, so a row's starts depend on its position and on the batch size n. The same observation vector can therefore get different estimates in validation, test or re-batched runs. Derive start draws per row, for example by seeding from the row index or a hash of the row, or report that estimates are invariant to row permutation.. Resolution: `unresolved`.
- `objection-065` (S-Q1-004, open): Show that the forward-evaluation budget is sufficient. The fit uses about 1.2k gate evaluations plus about 45k for G6 noiseless recovery (100 rows x 3 starts x (1 + 30 x 5) = 453 per row). Prediction costs about 453 x n_val, and the physics_mle_ref arm uses additional evaluations. State n_val and the projected total against max_forward_evals = 3,000,000, including any later test evaluation, so the arm cannot fail from budget exhaustion partway through a run.. Resolution: `unresolved`.
- `objection-066` (S-Q1-004, open): Justify the gate thresholds and disclose their history. Why is noiseless rel_l2 <= 0.01 on 100 training rows the threshold for G6? Why chi2 band [74, 86] and residual std band [0.95e-3, 1.05e-3]? Were any of them changed after the superseded S-Q1-003 ran? Under correct specification, mean chi2 over 200 rows of 80 observations has standard deviation of about 0.9, so the G4 band is roughly ±6.7 SD and catches only gross mismatch. State that G4 does not test the noise model beyond its first two moments.. Resolution: `unresolved`.
- `objection-067` (S-Q1-004, open): Check that the explicit scheme is stable, not just finite. G5 only checks that outputs are finite at the 16 corners and 1000 random draws. An explicit FD scheme near its CFL limit (k up to 2, 40 points, 200 steps) can produce bounded oscillatory artefacts. Report max k*dt/dx^2 against 0.5 at k = 2, or show that solutions at the corners are non-negative and monotone in time.. Resolution: `unresolved`.
- `objection-068` (S-Q1-004, open): Report the comparison of computational resources. The MLE uses the exact simulator with up to about 453 forward solves per validation row. The ET and MLP controls see only 3000 fixed training rows, although forward evaluation is authorized. C-Q1-4a and C-Q1-4b therefore compare inference with simulator access against untuned learners limited to the training data. Report forward-call and wall-clock costs per arm. Do not generalize to 'likelihood beats learned surrogates'. A learned control trained on simulator-augmented data would be needed for that wider claim.. Resolution: `unresolved`.
- `objection-069` (S-Q1-004, open): Keep the synthesis within scope. Question (b) is phrased as beating 'the controls', but C-Q1-4a and C-Q1-4b are separate claims without family-wise correction. A combined statement such as 'the MLE beats both ML baselines' needs either a pre-specified correction (for example, Bonferroni with each one-sided bound at 97.5%) or an explicit label as two separate results. The single training seed (0) for ET and MLP means seed variance is not quantified. Report the margins as conditional on seed 0, or run additional control seeds as a sensitivity check.. Resolution: `unresolved`.
- `objection-070` (S-Q1-004, open): Disclose how validation data was used in earlier studies. This study depends on S-Q1-002 and supersedes S-Q1-003. State what validation metrics those studies observed and which design elements changed afterwards: the gates, the iteration count, the start count, and the choice of controls. Claims made on a development split that has already informed design choices should be labelled as development evidence, not confirmatory.. Resolution: `unresolved`.
- `objection-071` (S-Q1-004, open): Prevent C-Q1-4c from being read as validating the harness physics estimator. The claim compares aggregate rel_l2 only. With sigma = 0.001, both arms may sit near the noise floor while differing on rows that have multiple local minima or bound-pinned solutions, and a few such outliers can dominate the Frobenius norm. Report the optional per-sample comparison alongside the claim: the rows with the largest estimate differences and the agreement of their chi2 values. A pass should not be interpreted as row-level agreement between the two estimators.. Resolution: `unresolved`.
- `objection-072` (S-Q1-004, open): Carry the inverse-crime scope into every reported result. The same simulator generates the data and is inverted, the noise sigma is known exactly, and the prior box equals the inversion box. Any verdict text must say that the margins over ET and MLP are not evidence of robustness to model or noise misspecification. Held-out test and stress observations must not be used to refine these claims.. Resolution: `unresolved`.
- `objection-073` (S-Q1-005, open): Tie the reporting criteria and verdict inputs to the validation predict() call only. predict() overwrites self.diagnostics['predict'] on every call and prints TRANSPARENT_MLE_PREDICT_DIAGNOSTICS each time. If the harness also calls predict() on train or on the final held-out test split, then the 'any failed reporting criterion' verdict mapping could be decided by test-split statistics (chi2_per_dof, iteration caps, pinned row indices). That violates the held-out rule. Pre-specify that only the val-split record counts, identified by split fingerprint (edfcc32d…) rather than by x_shape. Either suppress test-split diagnostics or state explicitly that they are excluded from all decisions.. Resolution: `unresolved`.
- `objection-074` (S-Q1-005, open): Account for forward evaluations against max_forward_evals = 3,000,000 before running. Per row, the treatment costs 3 starts × (1 + 30 × (4 FD perturbations + 1 trial)) = 453 evals. fit() adds about 46k evals (200 + 12 + 1016 + 100 × 453 + 100). The val and test row counts are not stated. It is also not stated whether the harness calls predict() on train (3000 × 453 ≈ 1.36M), val or test. It is not stated whether physics_mle_ref (30 iterations, 3 starts, 32 posterior samples) draws on the same budget. A plausible total exceeds 3M. State the split sizes and which calls are made, and show the projected total fits the budget.. Resolution: `unresolved`.
- `objection-075` (S-Q1-005, open): Extend the pre-registered verdict mapping to failure modes it does not currently cover. These include: a treatment exception that is not a GateError (budget exhaustion, LinAlgError outside the try block, timeout); a failed et_log or mlp_fixed control arm; and bootstrap replicates that are non-finite or degenerate. Each of these should map explicitly to 'inconclusive / not evaluable' rather than being left to post-hoc judgment.. Resolution: `unresolved`.
- `objection-076` (S-Q1-005, open): Resolve the inconsistency in the C-Q1-5c equivalence bounds. The diagnostic registers equivalence_bounds [0.952381, 1.05]. The claim says the stricter of that and |1-R| <= 0.0476 decides, which gives an effective upper bound of 1.0476. An automated pass flag computed from the diagnostic's bounds would accept 1.0476 < CI_upper <= 1.05. Report the raw 90% CI endpoints and have the verdict apply [0.952381, 1.0476] explicitly, or register the diagnostic with those bounds.. Resolution: `unresolved`.
- `objection-077` (S-Q1-005, open): Provide evidence that the custom_solver arm receives raw, unscaled, un-PCA'd 80-dimensional observations. The transform_order is features → scaler → pca → estimator. The treatment arm does not declare scaler, pca or features, so the harness defaults are unknown. A default standard scaler would make G4 fail, because chi2 is computed against forward(y) in raw units, and would waste the run as inconclusive. Also confirm that y reaches fit() on the original k scale with no default target_transform; G2 guards this, but only by failing.. Resolution: `unresolved`.
- `objection-078` (S-Q1-005, open): Show that the C-Q1-5c matching check passes. Its allowed_varying_factors exclude features, scaler, pca, target_transform, noise_aug and noise_scale. physics_mle_ref sets transform='log' in params, while the custom_solver arm has whatever harness defaults apply to those fields. If the matcher compares resolved defaults, the pair may be flagged as unmatched.. Resolution: `unresolved`.
- `objection-079` (S-Q1-005, open): Provide a dry-run or prior evidence, e.g. from S-Q1-002 or S-Q1-004 on train rows only, that the reporting criteria are attainable. The tolerance rule requires an accepted step with relative decrease <= 1e-9·max(cost,1), or a step under 1e-9 in log k, within 30 iterations. The criteria are frac_iteration_capped <= 0.10 and chi2/dof p50 <= 1.2. Without that evidence, a likely outcome is 'inconclusive' from conservative convergence bookkeeping rather than from estimator quality. Note also that rows whose best start ended at λ_max (status 2) are not counted as capped, so status-2 rows need their own reporting.. Resolution: `unresolved`.
- `objection-080` (S-Q1-005, open): G5 checks finiteness only. It is not a stability check for the explicit FD scheme with 40 grid points and 200 steps at k up to 2. A CFL-violating explicit scheme can produce bounded-but-oscillatory, non-physical outputs that stay finite. Under the inverse-crime condition the comparison remains internally consistent, but the scope statement should disclose this. Better, add a discrete max-principle or CFL check: dt <= dx²/(2·k_max), and outputs bounded by the initial-condition maximum.. Resolution: `unresolved`.
- `objection-081` (S-Q1-005, open): Justify the clipping-based projection in the LM step. When a component is at a bound and the step points outward, clipping zn changes the step direction without re-solving on the free variables. This can cause repeated rejections and λ blow-up (status 2) for truths near the prior boundary. The procedure is not an exact projected or active-set method. Report row-level stats for pinned versus unpinned rows on val, analogous to noiseless_rel_l2_pinned, so that boundary rows driving R in C-Q1-5c can be identified.. Resolution: `unresolved`.
- `objection-082` (S-Q1-005, open): Provide a precision or power rationale for C-Q1-5c. The equivalence band of about ±5% on an aggregate Frobenius ratio may not be resolvable at the unstated val size. Both estimators' rel_l2 can be dominated by a few multimodal or boundary rows where the start sets differ. Without this, a non-equivalence or inconclusive result is uninterpretable. State the val row count and the expected CI width, e.g. from S-Q1-002 numbers.. Resolution: `unresolved`.
- `objection-083` (S-Q1-005, open): Verify the paired bootstrap's row alignment. All four arms' validation predictions must be indexed to the same val rows in the same order (fingerprint edfcc32d…). Each replicate must recompute both Frobenius numerators and denominators over the same resampled rows. Report the replicate count actually achieved, since fewer than 2000 is registered as inconclusive.. Resolution: `unresolved`.
- `objection-084` (S-Q1-005, open): Disclose what changed from the superseded S-Q1-004 and whether any val-split results from S-Q1-002 or S-Q1-004 informed the current gate bands, iteration count, starts, margin or control configurations. If they did, C-Q1-5a, 5b and 5c are tested on reused development data and should be labelled as such in the synthesis; it does not support confirmatory language.. Resolution: `unresolved`.
- `objection-085` (S-Q1-005, open): Keep the synthesis within scope. C-Q1-5a and 5b compare against single-seed, untuned, fixed-configuration learned controls under an inverse-crime simulator with σ = 0.001. A pass supports only 'beats these specific configurations on val', not 'likelihood methods beat learned regressors'. The MLP comparison excludes training-seed variance and torch CPU nondeterminism, so it should be reported as row-sampling uncertainty only. The treatment's start_seed (20261114) is separate from the arm seed (0), and this should be stated.. Resolution: `unresolved`.
- `objection-086` (S-Q1-005, open): requires_implementation_verification is true for all three claims, yet no independent verification evidence is listed and evidence_ids is empty. The gates are self-checks written by the same author. At minimum, supply one externally checked case: a hand-constructed k with known forward output, or agreement of FD Jacobian columns against a central-difference reference at h = 1e-5. This would confirm the LM implementation and the forward-call accounting independently of the gates.. Resolution: `unresolved`.
- `objection-087` (S-Q1-006, open): Make the required paired diagnostics compute the statistic each claim actually tests. For C-Q1-6a and C-Q1-6b, the diagnostics S-Q1-006.paired_tmle_vs_et and S-Q1-006.paired_tmle_vs_mlp set no 'statistic' field. A harness default could therefore bootstrap a difference or the raw metrics instead of 1 - rel_l2_MLE/rel_l2_control. Specify the relative-improvement statistic explicitly in both diagnostic records, or show that the harness default equals it.. Resolution: `unresolved`.
- `objection-088` (S-Q1-006, open): Reconcile the interval definition for C-Q1-6a and C-Q1-6b. The claim text requires a one-sided 95% percentile lower bound, which is the 5th percentile. The diagnostics only state interval_level 0.95 with ci_method percentile, which a harness would normally compute as a two-sided interval with a 2.5th-percentile lower bound. State which endpoint is compared to the 0.05 margin, and record the bootstrap RNG seed so the replicates can be reproduced.. Resolution: `unresolved`.
- `objection-089` (S-Q1-006, open): Fix the asymmetric verdict mapping created by the predict-time hard gates. The gates are computed on validation outputs: chi2/dof p50 <= 1.2, iteration-capped fraction <= 0.10, and no non-finite objective. These quantities are directly tied to treatment performance. When the treatment fits validation rows poorly, the claim becomes 'inconclusive' instead of 'not supported'. As a result, C-Q1-6a and C-Q1-6b can be supported or left unevaluated, but never refuted, because of treatment fit failure. Pre-specify that a predict-time gate failure on the treatment arm is recorded as a treatment failure (claim not supported), separately from infrastructure exceptions.. Resolution: `unresolved`.
- `objection-090` (S-Q1-006, open): Supply a forward-evaluation budget accounting against max_forward_evals = 3,000,000, with validation (and later test) row counts stated. The worst case per row is about 3 starts × (1 + 30 × (4 Jacobian + 1 trial)) ≈ 453 evaluations. Jacobians are recomputed even after rejected steps. Fit-time costs are about 1,228 evaluations for the gates plus about 45,300 for G6 noiseless recovery. The harness physics_mle_ref arm also uses forward calls. The validation set size is not given, so feasibility cannot be checked. The code counts forward_calls but never enforces the cap.. Resolution: `unresolved`.
- `objection-091` (S-Q1-006, open): Disclose what changed from S-Q1-005 and how many validation-informed revisions preceded this one. That includes the gates, the margins (0.05 and 0.0476), the iteration and start counts, and the thresholds. Validation rows have already informed the design, so the bootstrap intervals understate selection uncertainty. Synthesis must treat these intervals as descriptive development evidence, not confirmatory. The S-Q1-005 results themselves are missing from the supplied evidence.. Resolution: `unresolved`.
- `objection-092` (S-Q1-006, open): The equivalence claim C-Q1-6c compares against harness physics 'mode' with posterior_samples = 32 and tikhonov = 0. The harness estimator's definition (optimizer, starts, seeds, whether samples affect the point estimate) is not supplied. Without it, aggregate rel_l2 equivalence is not interpretable as reproduction. Two different procedures could match in aggregate while differing systematically row by row. Provide the harness estimator specification. Also report row-level agreement descriptively, for example the median absolute log-k difference between arms, even though it is not claimed.. Resolution: `unresolved`.
- `objection-093` (S-Q1-006, open): The predict-time chi2 criterion (median chi2/dof <= 1.2) would not detect local-minimum failures affecting up to half the rows. Report the treatment's rel_l2 restricted to rows where starts disagree (n_rows_starts_disagree_gt_1e-3_logk) and to pinned rows, so readers can see whether the aggregate is driven by optimization failures. Consider pre-specifying a gate on frac_starts_disagree or on the chi2/dof p90.. Resolution: `unresolved`.
- `objection-094` (S-Q1-006, open): G5 checks only that outputs are finite at corners and random draws. It does not check numerical stability of the explicit scheme (40 points, 200 steps, k up to 2): an unstable explicit scheme can stay finite while oscillating. Report the CFL number max(k)·dt/dx² at the upper bound, or check for sign oscillations or non-physical growth. Because this is an inverse-crime setup, data and inversion would share any such artifact silently.. Resolution: `unresolved`.
- `objection-095` (S-Q1-006, open): Confirm what X the custom_solver receives. Under transform_order, noise augmentation, scaler and PCA could be applied before estimator.fit and predict. The likelihood requires raw, unscaled 80-dim observations. G1 and G4 would catch scaling at fit time, but the harness behaviour for custom_solver at predict time should be stated, not inferred from gate passage.. Resolution: `unresolved`.
- `objection-096` (S-Q1-006, open): The superiority claims against ET and MLP compare a method with exact simulator access (inverse crime, known sigma) against learned regressors trained on 3,000 rows. Both controls are untuned and single-seed, and the MLP's seed variance is excluded from the uncertainty. Synthesis must not generalize beyond 'beats these specific fixed configurations at seed 0 under inverse-crime conditions'. A seed sweep of at least 3 seeds for ET and MLP would show whether the 5% margin is robust to control seed.. Resolution: `unresolved`.
- `objection-097` (S-Q1-006, open): G6 checks recovery on noiseless forward outputs of the first 100 training targets (rel_l2 <= 0.01). It does not validate behaviour at sigma = 0.001, and a 30-iteration, 3-start budget may itself fail this gate. Report G6 sub-statistics (pinned vs unpinned rel_l2, frac_iteration_capped) whether the gate passes or fails. If any threshold is adjusted after a failure, that adjustment is a further validation-informed revision and must be logged.. Resolution: `unresolved`.
- `objection-098` (S-Q1-006, open): The projected LM handles bounds by clipping the step after solving the unconstrained system. This is not a proper bound-constrained solver: at an active bound the clipped step may be rejected repeatedly until lambda reaches 1e9 (status 2). The 'MLE' label already disclaims exactness, but report the fraction of best-start rows with status 2 that are pinned. That separates boundary stalls from true convergence.. Resolution: `unresolved`.
- `objection-099` (S-Q1-006, open): Held-out protection: predict() raises GateError and prints full diagnostics on any input, including test. State that test-time predictions and diagnostics are not computed or inspected during this development study. Otherwise the held_out contract limit could be breached through gate outcomes.. Resolution: `unresolved`.
- `objection-100` (S-Q1-005, open): Re-run the bootstrap with the registered 2000 replicates for C-Q1-5a and C-Q1-5c. Both paired diagnostics (resu-00266, resu-00267) report n_bootstrap = 200, but the diagnostics spec registers 2000. The pre-registered verdict mapping says fewer than 2000 replicates gives 'inconclusive / not evaluable', so neither claim can be evaluated until the analysis is re-run at 2000.. Resolution: `unresolved`.
- `objection-101` (S-Q1-005, open): For C-Q1-5c, compute and report the registered statistic and interval. The registered statistic is the ratio R = rel_l2_TMLE / rel_l2_REF, with a two-sided 90% paired whole-row percentile CI, checked against [0.952381, 1.05] and also against |1 - R| <= 0.0476. resu-00267 instead reports estimand 'relative_improvement' (1 - R) and does not state the interval level, so it cannot be read as the registered 90% TOST interval. Do not substitute an algebraic transform of a 200-replicate interval at an unstated level.. Resolution: `unresolved`.
- `objection-102` (S-Q1-005, open): For C-Q1-5a, state the interval level and sidedness, and report the 95% one-sided percentile lower bound on 1 - rel_l2_MLE/rel_l2_ET. resu-00266 gives only an unlabeled interval [0.7287, 0.7703]. The registered decision rule uses a one-sided 95% lower bound, so it must be shown that this exact quantity was computed.. Resolution: `unresolved`.
- `objection-103` (S-Q1-005, open): Export and attach the treatment arm's TRANSPARENT_MLE_FIT_DIAGNOSTICS and TRANSPARENT_MLE_PREDICT_DIAGNOSTICS logs for the validation run. These cover the G1-G7 gate outcomes, chi2_at_truth, noiseless_rel_l2, frac_iteration_capped, the chi2_per_dof p50, frac_nonfinite_objective, the pinned fraction, start disagreement and forward-call counts. The method_contract shows diagnostics declared = [], observed = [] and verified = false. The pre-registered verdict mapping depends on gates and reporting criteria, and claims C-Q1-5a/b/c all set requires_implementation_verification = true. An arm status of 'ok' does not establish that the reporting criteria passed.. Resolution: `unresolved`.
- `objection-104` (S-Q1-005, open): Confirm that total forward evaluations across all fit and predict calls stayed within max_forward_evals = 3,000,000, using the logged forward_calls. A rough estimate for 3 starts × up to 30 iterations × (1 + 4 + 1) evaluations × 750 rows, plus fit-time gates on 100 rows, is about 0.4M+ for validation alone. This is likely within budget, but it is unverified.. Resolution: `unresolved`.
- `objection-105` (S-Q1-005, open): For C-Q1-5b, attach the error log for the failed S-Q1-005.mlp_fixed arm (status 'error'), diagnose the root cause, and re-run the arm unchanged. Also record the logged epochs_run, because the claim is inconclusive if epochs_run != 150. Without the cause, nobody can tell whether the failure is an infrastructure fault or a defect in the method, such as the custom estimator receiving scaled or float64 inputs, the torch dependency, or the build_model signature. Any code change after seeing validation results would make it a new, re-registered arm.. Resolution: `unresolved`.
- `objection-106` (S-Q1-005, open): Relabel or withdraw the diagnostic S-Q1-005.per_target_tmle_vs_mlp (resu-00269). It is marked 'complete' and cites resu-00257 as a source even though the MLP control produced no predictions. Its per_sample_l2 and per_target_rel_l2 are byte-identical to per_sample_tmle_vs_ref, which shows they are treatment-only statistics with no control content. It must not be cited as MLP-comparison evidence.. Resolution: `unresolved`.
- `objection-107` (S-Q1-005, open): Resolve the bookkeeping inconsistency where orphaned_arm_ids lists the three successful arms (transparent_mle, physics_mle_ref, et_log) while those same arms are the sources of the paired diagnostics. The provenance link from arm results (resu-00261/00263/00259) to the diagnostics and verdicts needs to be confirmed via row_hash and the effective_hash of each arm.. Resolution: `unresolved`.
- `objection-108` (S-Q1-005, open): Explain or check the near row-level identity between the treatment and the harness reference. Their per-row squared errors agree to about 1e-7 on the first rows, and their per-target RMSEs agree to about 1e-5, even though the claim text says the arms differ in start sets, LM schedule and seeds. Provide evidence that the transparent arm is an independent implementation rather than a call path into the harness solver. Otherwise C-Q1-5c is a self-comparison with no reproduction value. Per its own text, C-Q1-5c also must not be synthesised as row-level agreement or as identifying the harness estimator as a mode.. Resolution: `unresolved`.
- `objection-109` (S-Q1-005, open): Limit synthesis to the declared scope. Results come from the inverse-crime condition, with the same forward model and discretization used for generation and inversion, at sigma = 0.001 under the U[0.5,2]^4 prior. The comparison is against a single untuned extra_trees configuration on seed 0, and uncertainty covers row resampling only, conditional on fitted models. No claim about model misspecification, other noise levels, tuned or other ML methods, training-seed variability, or held-out test performance is supported.. Resolution: `unresolved`.
- `objection-110` (S-Q1-005, open): Report per-segment results in any synthesis. Segments 1 and 4 have about 5× the RMSE and relative error of segments 2 and 3 (per-target rel_l2 0.027/0.006/0.005/0.026). This is consistent with weaker local sensitivity near the Dirichlet boundaries relative to sensors at indices 5 and 35. An aggregate rel_l2 should not be read as uniform recoverability, and local-Fisher arguments for this pattern are not a global identifiability guarantee.. Resolution: `unresolved`.
- `objection-111` (S-Q1-005, open): Document that the selection rule (argmin chi2 over 3 starts, 30-iteration cap) gives the best-objective iterate, not a certified global MLE. Report on validation the fraction of rows where starts disagree by more than 1e-3 in log k, and the fraction pinned at the bounds. The treatment must not be described as 'the MLE' in synthesis.. Resolution: `unresolved`.
- `objection-112` (program, open): The final synthesis must not carry MLE-versus-MLP superiority over to the auditable arm. The only paired MLE-vs-MLP result (0.311, resu-00110) uses the opaque harness physics_mle. That arm has source_hash = null, an unresolved posterior_samples normalization (032), and open implementation objections 017-029. For the verified arm S-Q1-005.transparent_mle, no paired comparison against any MLP exists: mlp_fixed failed and resu-00269 is treatment-only. The stop text lists transparent_mle-vs-ET and harness-MLE-vs-MLP next to each other and exports transparent_mle on the grounds that it matches the harness arm. That presentation suggests transitivity the evidence does not support. Any statement that the likelihood procedure beats the MLP must be attributed to the harness arm alone, together with its unresolved objections.. Resolution: `unresolved`.
- `objection-113` (program, open): Drop the phrase 'statistically indistinguishable' for transparent_mle vs physics_mle_ref, or justify it. It rests on a 200-replicate interval with no stated level and the wrong estimand (1 - R, not the registered ratio R with a 90% TOST CI; see objections 100 and 101). Report it descriptively instead, e.g. 'aggregate rel_l2 differs by 0.05%'. The final-export rationale also contradicts itself. It selects transparent_mle because its val rel_l2 is lower (0.019322 vs 0.019333) while also calling the two arms indistinguishable. State that the choice between them is arbitrary at this resolution and that the tie was broken by auditability, not by validation performance.. Resolution: `unresolved`.
- `objection-114` (program, open): The stop rationale says no remaining action could add evidence. That is not accurate. A 'diagnose' action on the already-executed S-Q1-005, with treatment transparent_mle and control mlp_log_harness (resu-00256, same row_hash), needs no new study or delegation. It would have given a descriptive paired comparison between the auditable arm and an MLP. It could also have supplied control-side per-target values, which would fix the treatment-only per-target artifacts (037, 106). The comparison is non-claim-bearing because the harness torch_mlp hygiene question (061) is still open. Even so, the synthesis should either report it or state that it was not obtained.. Resolution: `unresolved`.
- `objection-115` (program, open): The Q-3 per-segment synthesis overstates consistency across arms. It claims outer segments carry about 5x the inner error and that 'the same pattern appears in both learned arms'. The measured outer/inner rel_l2 ratios are about 4.5-5.4x for the likelihood arms, about 3.2-4.3x for ET [0.117, 0.028, 0.029, 0.092], and only about 2.1-2.7x for the MLP [0.039, 0.016, 0.014, 0.034]. Report the ratios per arm. Note that sensors at indices 5 and 35 sit inside the outer segments, so 'far from sensors' does not explain the pattern. Keep any explanation (distance from the central pulse, proximity to the Dirichlet ends) labeled as an untested hypothesis, since no local-Fisher computation was run.. Resolution: `unresolved`.
- `objection-116` (program, open): Gate G6 (noiseless recovery rel_l2 <= 0.01) is too loose to separate optimizer error from noise-limited error. Validation rel_l2 is 0.0193, so an optimizer error of up to about half the reported error would still pass G6. For a 4-parameter problem with an exact forward model, noiseless recovery should be near FD/tolerance precision. The actual noiseless_rel_l2 value was never captured. The synthesis must not cite gate passage as evidence that the residual 0.019 error is noise-limited rather than partly due to optimization.. Resolution: `unresolved`.
- `objection-117` (program, open): There is no noise-floor benchmark. No Cramér-Rao or local-Fisher expected error at sigma = 0.001 was computed, so the synthesis cannot say whether 0.019 is near the attainable limit or whether the 31% gap to the MLP is large relative to what is achievable. The harness arm's posterior-sample UQ (mean_std 0.019, z_rms 0.98, 2-sigma coverage 0.957 on validation) hints that the error is consistent with posterior spread. That is a harness-internal, unverified quantity. If cited, label it so, and do not present it as an identifiability or efficiency result.. Resolution: `unresolved`.
- `objection-118` (program, open): Clarify what C-Q1-5c agreement can and cannot validate. Both arms call the same ctx.forward, so near row-level agreement (per-row squared errors match to about 1e-5 relative) shows only that two optimizers reached the same optimum of the same objective. It provides no independent check of the forward operator, the discretization, CFL stability, or interface averaging. The only forward-related evidence is the statistical chi-square check G4 at truth, whose values were not captured. Under inverse-crime conditions this does not bias the comparison, but synthesis must not present cross-implementation agreement as verification of the simulator.. Resolution: `unresolved`.
- `objection-119` (program, open): The final export is an arm whose predict-time reporting criteria (iteration-capped fraction <= 0.10, chi2/dof p50 <= 1.2, no non-finite objective) were computed but not observed, because runtime_diagnostics is empty. Its convergence behavior on validation is therefore undocumented. The export record should state this. It should not cite the arm's forward-call count (about 139 per row, implying early termination of most starts) as evidence of efficient convergence, since early stops include rows stuck at lambda_max (status 2), which counts as neither converged nor capped.. Resolution: `unresolved`.
- `objection-120` (program, open): Seed-sensitivity disclosure is incomplete. The synthesis reports the MLP seed-1 rel_l2 (0.03016) but not the implied effect against the MLE (about 0.36 vs 0.31 at seed 0), and gives no paired analysis. With two seeds per learner, training-seed variance is effectively unquantified. State that the margin is conditional on 1-2 control seeds and that the reported intervals cover row resampling only.. Resolution: `unresolved`.
- `objection-121` (program, open): Restate the resource-asymmetry caveat in the conclusions, not only in the cost section. The learned controls used 3000 fixed training rows and zero forward calls, while about 2.5M of the 3M authorized forward evaluations went unused. An amortized learner trained on simulator-generated data was feasible and was never tested. The evidence supports only 'likelihood inversion beats these fixed, untuned, data-limited learner configurations', not anything about learned inversion in general. Q-2 (hybrid) was not studied, so no statement about cost-accuracy trade-offs is supported.. Resolution: `unresolved`.
- `objection-122` (S-Q1-001, open): S-Q1-001 remains superseded. Resolution: `unresolved`.
- `objection-123` (S-Q1-003, open): S-Q1-003 remains superseded. Resolution: `unresolved`.
- `objection-124` (S-Q1-004, open): S-Q1-004 remains superseded. Resolution: `unresolved`.
- `objection-125` (S-Q1-005, open): S-Q1-005 remains incomplete. Resolution: `unresolved`.
- `objection-126` (S-Q1-006, open): S-Q1-006 remains superseded. Resolution: `unresolved`.

test_bootstrap: `null`.

test_per_target_rel_l2: `null`.

test_uncertainty: `null`.

shift: `{}`.

## Scientific contract

```json
{
  "bounds": {
    "lower": [
      0.5,
      0.5,
      0.5,
      0.5
    ],
    "upper": [
      2.0,
      2.0,
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
    "discretization": "explicit finite differences, 40 grid points, 200 steps",
    "initial_condition": "known Gaussian pulse centred at x = 0.5",
    "noise": "Gaussian, sigma = 0.001",
    "observations": "interior point sensors at grid indices [5, 15, 25, 35], discrete times, finite window",
    "parameter_count": 4,
    "pde": "parabolic (heat / diffusion equation), divergence form u_t = (k u_x)_x",
    "positivity": "conductivity is strictly positive"
  },
  "flattened_feature_count": 80,
  "input_shape": [
    80
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
      "fingerprint": "aa096404521859ad248e66b22e880c498febc8667a2f15b4782a20afdec0ed2c",
      "role": "final_held_out"
    },
    "train": {
      "fingerprint": "cdad48590f63a82658396a12757300627418529ccf5554130ff22dda5163a173",
      "role": "training"
    },
    "val": {
      "fingerprint": "edfcc32d6aa195b668c2a60ae6e89879ebb15b4bddf26f47584bee6d2f7c0543",
      "role": "development"
    }
  },
  "target_count": 4,
  "target_shape": [
    4
  ],
  "target_transform": "method-specific: optional log before estimator fit, exp after prediction",
  "task": "regression",
  "training_rows": 3000,
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

Agent instances: `7`; specialist tasks: `4`.

Fallback reasons (role calls): `{}`.

Costs: `{"by_stage_s": {"final": 54.38, "study": 191.06}, "compute_budget_s": 760.0, "compute_remaining_s": 514.55, "compute_used_s": 245.45, "gpus": [], "jobs": 19, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {"execution": 30}, "llm_calls": 30, "llm_calls_max": 80, "llm_cost_usd": 9.3324, "llm_seconds": 1504.69, "wall_budget_s": 7200.0, "wall_elapsed_s": 1653.38, "workers": 6}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 360.0, "confirmation_used_seconds": 54.382, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-002:4", "seconds": 152.6}, {"action": "release", "actual_seconds": 0.0, "group_id": "study:S-Q1-002:4", "overrun_seconds": 0.0, "reserved_seconds": 152.6}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-002:5", "seconds": 152.6}, {"action": "release", "actual_seconds": 51.056000000000004, "group_id": "study:S-Q1-002:5", "overrun_seconds": 0.0, "reserved_seconds": 152.6}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-005:10", "seconds": 157.6}, {"action": "release", "actual_seconds": 0.0, "group_id": "study:S-Q1-005:10", "overrun_seconds": 0.0, "reserved_seconds": 157.6}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-005:11", "seconds": 157.6}, {"action": "release", "actual_seconds": 42.433, "group_id": "study:S-Q1-005:11", "overrun_seconds": 0.0, "reserved_seconds": 157.6}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 360.0}, {"action": "release", "actual_seconds": 54.382, "group_id": "final-export", "overrun_seconds": 0.0, "reserved_seconds": 360.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 612.129, "reserved_seconds": 0, "total_seconds": 760.0, "used_seconds": 147.871}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
