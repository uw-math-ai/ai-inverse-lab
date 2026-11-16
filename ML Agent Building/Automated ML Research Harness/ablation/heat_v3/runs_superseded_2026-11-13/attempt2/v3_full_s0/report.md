# V3 research product: heat_v3_full_s0_20261113_212624

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: registered work resolved or explicitly deferred; no justified next study

Model export certified: **False**. Studies complete: **1/6**.

Model performance and study completion are separate outcomes.

Run error: RuntimeError: final export failed: AttributeError: Can't pickle local object 'build_method.<locals>.BoxLMVec'

## Actual exported method

No final model was exported.

Validation selection value: `None`. Final metrics: `{}`.

Export certificate: `{}`.

Lineage: `{}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-Q1-1 | 0/4 | superseded |  |
| S-Q1-2 | 0/4 | superseded |  |
| S-Q1-3 | 0/7 | superseded |  |
| S-Q1-4 | 0/7 | superseded |  |
| S-Q1-5 | 4/5 | deferred | required evidence missing; preserve uncertainty |
| S-Q1-6 | 5/5 | complete | all registered required work completed |

### S-Q1-1: Does forward-model inversion (physics MAP/MLE with log-parameter transform and multistart) achieve lower validation rel_l2 than the strongest library regressors (extra_trees, torch_mlp on raw 80-dim sensor features) for recovering the four segment conductivities?

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-1.physics` (treatment, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-1.extra_trees` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-1.torch_mlp` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-1.mean` (baseline, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-1.paired_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-1.paired_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-1.per_target_phys`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-2: Does box-constrained forward-model MLE in log-conductivity space (3 seeded prior-draw starts, 30 LM iterations) achieve at least 10% lower validation rel_l2 in original conductivity space than fixed-configuration extra_trees and torch_mlp regressors trained on the 3000 raw 80-dim sensor rows? Pre-registration: (a) the optimizer clips log-k iterates to [log 0.5, log 2.0]; explicit-scheme stability dt*k_max/dx^2 at k=2.0 is computed from facts (40 grid points, 200 steps) and reported, and the run is aborted if it exceeds 0.5. (b) Objective is sigma-weighted SSE (sigma=0.001) over the 80 observations, box-constrained MLE, no log-Jacobian term, tikhonov=0; posterior_samples has no role in the point estimate. (c) Starts are 3 i.i.d. draws from the uniform prior with seed 0 per row; the row estimate is the start with lowest final SSE; if all starts are non-finite the prior centre k=1.25 is used; failed rows remain in the metric; counts of non-finite, non-converged (relative step > 1e-6 at iteration 30) and bound-stuck (any component within 1e-6 of a bound) rows are reported together with the final sigma-normalized SSE distribution against the 80/2 floor and the fraction of rows whose three starts disagree by more than 1% in any component. (d) All arms are scored in original k-space after exp for log-target arms, using global Frobenius rel_l2 on the identical validation rows. (e) Uncertainty: paired bootstrap over validation rows, B=2000, seed 0, recomputing both global Frobenius rel_l2 values per resample; success requires the 97.5th percentile of rel_l2_phys/rel_l2_control to be below 0.9. (f) Controls use fixed hyperparameters with no tuning, noise augmentation off, scaler fit on training rows only, no early stopping or checkpoint selection; they are 'fixed-configuration' baselines, not the strongest possible regressors. (g) The physics arm has privileged forward-model access and ignores the training rows; the comparison is model-based vs data-driven under unequal information. (h) Per-target rel_l2 and R2 are reported for every arm; mean_baseline is required as a sanity floor that both controls must beat.

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-2.physics` (treatment, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-2.extra_trees` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-2.torch_mlp` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-2.mean` (baseline, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-2.paired_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.paired_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.per_target_phys_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.per_target_phys_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.per_target_et_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.per_target_mlp_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-2.per_sample_phys`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-3: Does box-constrained forward-model MLE in log-conductivity space (3 seeded prior-draw starts, 30 LM iterations) achieve at least 10% lower validation rel_l2 in original conductivity space than fixed-configuration extra_trees and torch_mlp regressors trained on the 3000 raw 80-dim sensor rows? Pre-registration: (a) the optimizer clips log-k iterates to [log 0.5, log 2.0]; dx, dt, final time T and dt*k_max/dx^2 at k=2.0 are read from the simulator configuration via context.facts and reported by the fidelity arm; the run is aborted if the ratio exceeds 0.5. (a2) Forward-model fidelity: the fidelity arm calls the authorized context.forward (not a re-implementation) on the true k of the first 200 training rows and reports mean, median and 95th percentile of sum(r^2/sigma^2) against the chi^2_80 expectation of about 80; values far from 80 invalidate the physics arm's verdict. The physics arm itself uses the harness physics kind, which calls the same authorized forward. (b) Objective is sum(r^2/sigma^2) with sigma=0.001 over the 80 observations, box-constrained MLE, no log-Jacobian term, tikhonov=0; posterior_samples has no role in the point estimate. (b2) Budget: Jacobian is finite-difference in log k (1+4 forward calls per LM iteration plus rejected damping steps), about 450 to 1000 calls per row over 3 starts and 30 iterations; n_val and total forward calls are reported; if max_forward_evals=3,000,000 is reached mid-run, unprocessed rows take the prior centre k=1.25 and remain in the metric. (c) Starts are 3 i.i.d. draws from the uniform prior with seed 0 per row; the row estimate is the start with lowest final SSE; if all starts are non-finite the prior centre k=1.25 is used; failed rows remain in the metric; counts of non-finite, non-converged (relative step > 1e-6 at iteration 30), bound-stuck (any component within 1e-6 of a bound) and budget-fallback rows are reported, together with the final SSE distribution compared to chi^2_76 (expected about 76 = 80 - 4) and the fraction of rows whose three starts disagree by more than 1% in any component; rows with SSE above the chi^2_76 99th percentile are counted separately as suspected local minima or forward mismatch. (d) All arms are scored in original k-space after exp for log-target arms, using global Frobenius rel_l2 on the identical validation rows; no arm is scored in log space. (d2) The fraction of control predictions outside [0.5, 2.0] is reported, and a sensitivity ratio with controls clipped to the box is reported alongside the primary unclipped ratio. (e) Uncertainty: paired bootstrap over validation rows, B=2000, seed 0, recomputing both global Frobenius rel_l2 values per resample; success requires the 97.5th percentile of rel_l2_phys/rel_l2_control to be below 0.9. The bootstrap conditions on a single seed-0 training realization of each control; seed-1 exploratory replicates are reported as a non-decisive robustness check. (f) Controls use fixed hyperparameters with no tuning, noise augmentation off, scaler fit on training rows only, no early stopping or checkpoint selection; they are 'fixed-configuration' baselines, not the strongest possible regressors. (g) The physics arm has privileged forward-model access and ignores the training rows, and fits data generated by the identical discretization with exactly known sigma, initial and boundary conditions (inverse crime); any advantage is an upper bound for model-based inversion under perfect specification, not evidence of robustness to model error. (h) Per-target rel_l2 and R2 are reported for every arm; mean_baseline is a required sanity floor. (i) Dependency: if C-Q1-3c fails, C-Q1-3a is uninformative; if C-Q1-3d fails, C-Q1-3b is uninformative; a joint statement that physics beats both controls requires both C-Q1-3a and C-Q1-3b to pass and is framed as two one-sided tests. (j) The per_sample diagnostic reports per-row k-space error versus final SSE, start disagreement and bound-stuck status for the physics arm against the extra_trees control.

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-3.physics` (treatment, required=True): `missing`; method `physics`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.fidelity` (diagnostic, required=False): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.extra_trees` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.torch_mlp` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.extra_trees_seed1` (exploratory, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.torch_mlp_seed1` (exploratory, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-3.mean` (baseline, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-3.paired_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.paired_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_target_phys_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_target_phys_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_target_et_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_target_mlp_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_sample_phys`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_sample_et_seeds`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-3.per_sample_mlp_seeds`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-4: Does box-constrained forward-model MLE in log-conductivity space (3 seeded prior-draw starts, 30 LM iterations, authorized context.forward) achieve at least 10% lower validation rel_l2 in original conductivity space than fixed-configuration extra_trees and torch_mlp regressors trained on the 3000 raw 80-dim sensor rows? Pre-registration: (a) iterates are clipped to [log 0.5, log 2.0]; finite-difference step 1e-4 in log k, one-sided toward the interior when th+h would exceed the upper bound, so no forward call leaves the box. The explicit-scheme stability ratio dt*k_max/dx^2 is NOT computable from context.facts (no dt, T, dx exposed); the previous abort rule is removed and the fidelity arm prints the facts verbatim. (a2) Fidelity: the fidelity arm calls context.forward on the true k of the first 200 training rows after asserting y is in raw k bounds and X is not standardised; it evaluates both sensor-major and time-major layouts and uses the one with lower SSE, reporting both. Acceptance rule: |sse_mean-80| < 4, |sse_median - chi2_80 median| < 4, sse_max < chi2_80 quantile at 1-0.01/200; fidelity_pass must be true or C-Q1-4a and C-Q1-4b are uninformative. A val-split fidelity check is not possible because the custom_solver predict interface receives no targets; this is an acknowledged limitation. The physics arm determines the feature layout from 20 training rows (layout only, 40 forward calls, no parameter estimation). (b) Objective is sum(r^2/sigma^2), sigma=0.001, box-constrained MLE, no log-Jacobian, no Tikhonov. (b2) Budget: per-arm cap 2.8M forward calls; 1 initial + p perturbations + accepted/rejected trial steps per LM iteration per start (worst case about (p+6)*30*3 per row); rows processed in validation order; if remaining budget cannot cover a worst-case row, that row takes the prior centre 1.25 and is counted as budget_fallback and kept in the metric. n_val, total calls and calls per row are printed in PHYSICS_LM_REPORT. (c) Starts: 3 draws from the uniform prior with numpy default_rng([0, row_index]); row estimate is the finite start with lowest final SSE; if all starts are non-finite the row takes 1.25 and counts as non-finite. Reported counts: non-finite, non-converged (relative log-step > 1e-6 at the final accepted iteration), stalled (no accepted step at final iteration), bound-stuck (any log component within 1e-6 of a bound), budget_fallback; final SSE mean, median, p05, p95, max versus chi2_76 and the count above its 99th percentile (local minima or forward mismatch); fraction of rows where any finite start differs from the best by more than 1% in any component; SSE medians split by bound-stuck status. (d) All arms scored in original k-space after exp for log-target arms, global Frobenius rel_l2 on identical validation rows. (d2) Fraction of control predictions outside [0.5, 2.0] and a clipped-control sensitivity ratio are reported from the diagnostics. (e) Paired bootstrap over validation rows, B=2000, seed 0, recomputing both Frobenius norms per resample; success requires the 97.5th percentile of rel_l2_phys/rel_l2_control below 0.9; the bootstrap conditions on one seed-0 training realization per control, and seed-1 replicates are reported non-decisively; if seed 0 passes but seed 1 fails the result is reported as fragile to control seed. (f) Controls are fixed-configuration baselines, no tuning, no augmentation, scaler fit on train only, no early stopping. (g) Inverse crime: the physics arm fits data from the identical discretization with known sigma, IC and BC, using privileged forward access and ignoring training rows; any advantage is an upper bound under perfect specification. (h) Per-target rel_l2 and R2 for every arm; mean_baseline is the sanity floor; sanity passes with ratio between 0.65 and 0.75 are flagged as borderline. (i) If C-Q1-4c fails, C-Q1-4a is uninformative; if C-Q1-4d fails, C-Q1-4b is uninformative; a joint statement requires both to pass as two one-sided tests. (j) per_sample_phys reports per-row k-space error of physics vs extra_trees; the physics arm's own report supplies per-row SSE, bound-stuck and disagreement summaries. (k) The fidelity arm predicts the prior centre and is excluded from any ranking or synthesis of predictive performance. (l) No S-Q1-2 or S-Q1-3 run occurred; all design changes respond to critic objections, none to validation results.

Owner: `researcher-003`; registered evidence: `[]`.

- Arm `S-Q1-4.physics` (treatment, required=True): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.fidelity` (diagnostic, required=True): `missing`; method `custom_solver`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.extra_trees` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.torch_mlp` (control, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.extra_trees_seed1` (exploratory, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.torch_mlp_seed1` (exploratory, required=False): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.
- Arm `S-Q1-4.mean` (baseline, required=True): `missing`; method `library`; metrics `{}`; effective hash `None`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-4.paired_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.paired_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_target_phys_et`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_target_phys_mlp`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_target_et_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_target_mlp_mean`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_target_fidelity`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_sample_phys`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_sample_et_seeds`: `missing`; artifact `None`; reason `registered computation returned`.
- `S-Q1-4.per_sample_mlp_seeds`: `missing`; artifact `None`; reason `registered computation returned`.

### S-Q1-5: Does box-constrained forward-model MLE in log-conductivity space (custom LM, 3 seeded prior-draw starts, 30 iterations, authorized context.forward) achieve at least 10% lower validation rel_l2 in original conductivity space than fixed-configuration extra_trees and torch_mlp regressors trained on the 3000 raw 80-dim sensor rows? Pre-registration (supersedes S-Q1-4; no S-Q1-2/3/4 run occurred, all changes respond to critic objections): (a) iterates clipped to [log 0.5, log 2.0]; FD step 1e-4 in log k, one-sided toward the interior at the upper bound; no forward call leaves the box. dt*k_max/dx^2 is not computable from context.facts; no abort rule; facts printed verbatim. (a2) Fidelity arm: context.forward on true k of the first 200 training rows after asserting y in raw bounds and X not standardised; both layouts evaluated; acceptance |sse_mean-80|<4, |sse_median-chi2_80 median|<4, sse_max<chi2_80 ppf(1-0.01/200); fidelity_pass false makes C-Q1-5a/b uninformative. (a3) Solver verification (objection 043): the physics arm's fit() runs the identical LM path on the first 20 training rows (seed namespace seed+10^6, separate from validation seeds) and prints SOLVER_VERIFICATION; solver_pass requires recovery Frobenius rel_l2 < 0.10 on those 20 rows, |median SSE - chi2_76 median| < 10, at most 1 of 20 rows with SSE above the chi2_76 99th percentile, and start-disagreement fraction <= 0.25; solver_pass false makes C-Q1-5a/b uninformative (a solver bug cannot be read as a physics loss, and a lucky landing cannot be read as a win). (a4) Layout gate (044): the physics arm asserts X is not standardised; layout_ok requires |min(sse_mean_as_returned, sse_mean_transposed) - 80| < 12 over 20 rows; the claims additionally require transpose_used == (fidelity ordering_used == 'transposed_sensor_time'); otherwise uninformative. (b) Objective sum(r^2/sigma^2), sigma=0.001, no log-Jacobian, no Tikhonov. (b2) Budget: per-arm cap 2.8M calls; worst case 3*(1+30*(4+5)) = 813 calls per validation row, plus about 16k for the recovery check and 40 for layout; fidelity arm uses 200; total stays under the contract 3M only if n_val*813 + 17k < 2.8M (about 3400 rows); n_val is not known in advance, so n_budget_fallback is reported and any value > 0 flags the claims as biased against physics. Rows processed in validation order; fallback rows take the prior centre 1.25 and stay in the metric. (c) Starts: 3 uniform-prior draws with default_rng([0, global_row_index]) where the row index is a running offset across predict calls (051); predict_calls is reported. Row estimate is the finite start with lowest final SSE; all-non-finite rows take 1.25. Convergence (050): non-converged = final accepted absolute log-step > 1e-6 AND relative cost decrease > 1e-8; stalled = no accepted step at the final iteration; joint counts (converged&not stalled, converged&stalled, nonconverged&not stalled, nonconverged&stalled) reported; bound-stuck = any log component within 1e-6 of a bound. Final SSE mean/median/p05/p95/max vs chi2_76 and count above its 99th percentile; start-disagreement fraction; SSE medians split by bound-stuck. (c2) Secondary validation check (055): if the validation SSE median exceeds chi2_76 median + 15, C-Q1-5a/b are uninformative (mismatch or local minima), not a loss. (c3) Local Fisher diagnostic (054): per-target sd in log k from (J^T J)^-1 at the last computed Jacobian of the best start and the condition number of J^T J are printed as medians; local only, not a global identifiability claim, used only to contextualise per-target gaps. (d) All arms scored in original k-space after exp for log-target arms, global Frobenius rel_l2 on identical validation rows. (d2) No diagnostic computes a clipped-control ratio (046); the asymmetry is stated: the physics arm is box-constrained, torch_mlp exp predictions may leave [0.5,2.0], extra_trees cannot. (e) Paired bootstrap over validation rows, B=2000, seed 0, recomputing both Frobenius norms per resample; success requires the 97.5th percentile of rel_l2_phys/rel_l2_control below 0.9; conditions on one seed-0 control realization. (e2) Seed-1 arms (048): if run, the identical bootstrap rule is applied against them and a seed-0 pass with seed-1 fail is reported as fragile; if not run, fragility is reported as 'not assessed'. (f) Controls are fixed-configuration baselines, no tuning, no augmentation, scaler fit on train only, no early stopping. (g) Inverse crime: identical discretization, known sigma, IC and BC, privileged forward access; any advantage is an upper bound under perfect specification against these specific fixed baselines (053) and must not be generalised. (h) Per-target rel_l2 and R2 for every arm; mean_baseline sanity floor; sanity pass requires ratio < 0.70; a pass in [0.65, 0.70) is flagged 'weak control' and the gated comparison is reported as a pass against a weak control, not uninformative (047). (i) If C-Q1-5c fails, C-Q1-5a is uninformative; if C-Q1-5d fails, C-Q1-5b is uninformative; a joint statement requires both as two one-sided tests. (j) per_sample_phys_et and per_sample_phys_mlp (052) report per-row k-space error; the physics report supplies per-row SSE, bound-stuck and disagreement summaries. (k) The fidelity arm predicts the prior centre and is excluded from any ranking.

Owner: `researcher-003`; registered evidence: `['resu-00182', 'resu-00173', 'resu-00176', 'resu-00180', 'resu-00178', 'resu-00174', 'resu-00175']`.

- Arm `S-Q1-5.physics` (treatment, required=True): `timeout`; method `custom_solver`; metrics `{}`; effective hash `72153ad76ea232ba3d1eb2574050d440ab16f663e0fd602efd9f437431e12076`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.fidelity` (diagnostic, required=True): `ok`; method `custom-solver(fwd)`; metrics `{"r2": -0.0008909775527933261, "rel_l2": 0.3306557378688622, "rmse": 0.4360477137503121}`; effective hash `2e2c6157d50337dff422b9229f0639bafe053e0f2b2dacb17187c2f8c64129ba`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.extra_trees` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.torch_mlp` (control, required=True): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9927600207712438, "rel_l2": 0.028045400632179675, "rmse": 0.0369844869642745}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.extra_trees_seed1` (exploratory, required=False): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.945239129945993, "rel_l2": 0.07710766974217785, "rmse": 0.10168468063005599}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.torch_mlp_seed1` (exploratory, required=False): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9916315067445958, "rel_l2": 0.03015698062612236, "rmse": 0.03976910408507212}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[]`.
- Arm `S-Q1-5.mean` (baseline, required=True): `ok`; method `mean_baseline/raw`; metrics `{"r2": -0.0011333917959855389, "rel_l2": 0.330697038745562, "rmse": 0.4361021787143162}`; effective hash `641ba5b24826a458558d45e0e13b0787745f0e9028669de61ac3dd1b2c8fc9e2`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-5.paired_et`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.paired_mlp`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.per_target_phys_et`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.per_target_phys_mlp`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.per_target_et_mean`: `complete`; artifact `resu-00185`; reason `registered computation returned`.
- `S-Q1-5.per_target_mlp_mean`: `complete`; artifact `resu-00186`; reason `registered computation returned`.
- `S-Q1-5.per_target_fidelity`: `complete`; artifact `resu-00187`; reason `registered computation returned`.
- `S-Q1-5.per_sample_phys_et`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.per_sample_phys_mlp`: `missing`; artifact `None`; reason `treatment predictions unavailable`.
- `S-Q1-5.per_sample_et_seeds`: `complete`; artifact `resu-00188`; reason `registered computation returned`.
- `S-Q1-5.per_sample_mlp_seeds`: `complete`; artifact `resu-00189`; reason `registered computation returned`.

Claim `C-Q1-5a`: **inconclusive**. required evidence is missing or unmatched.
Predicate: `{"control": "S-Q1-5.extra_trees", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-5.physics"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": null}`; effect `None`; interval `None`.
Evidence `['resu-00182', 'resu-00176']`; unmet requirements `['diagnostic:S-Q1-5.paired_et', 'diagnostic:S-Q1-5.per_target_phys_et', 'diagnostic:S-Q1-5.per_sample_phys_et', 'treatment:S-Q1-5.physics', 'verified implementation contract', 'finite treatment metric']`.

Claim `C-Q1-5b`: **inconclusive**. required evidence is missing or unmatched.
Predicate: `{"control": "S-Q1-5.torch_mlp", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-5.physics"}`; values `{"control": 0.028045400632179675, "metric": "rel_l2", "treatment": null}`; effect `None`; interval `None`.
Evidence `['resu-00182', 'resu-00180']`; unmet requirements `['diagnostic:S-Q1-5.paired_mlp', 'diagnostic:S-Q1-5.per_target_phys_mlp', 'diagnostic:S-Q1-5.per_sample_phys_mlp', 'treatment:S-Q1-5.physics', 'verified implementation contract', 'finite treatment metric']`.

Claim `C-Q1-5c`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-5.mean", "margin": 0.3, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-5.extra_trees"}`; values `{"control": 0.330697038745562, "metric": "rel_l2", "treatment": 0.07740693387889735}`; effect `0.7659279497254459`; interval `None`.
Evidence `['resu-00176', 'resu-00175']`; unmet requirements `['objection-056', 'objection-057']`.

Claim `C-Q1-5d`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-5.mean", "margin": 0.3, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-5.torch_mlp"}`; values `{"control": 0.330697038745562, "metric": "rel_l2", "treatment": 0.028045400632179675}`; effect `0.9151930699513829`; interval `None`.
Evidence `['resu-00180', 'resu-00175']`; unmet requirements `['objection-056', 'objection-057']`.

### S-Q1-6: Does box-constrained forward-model MLE in log-conductivity space (vectorised LM, 3 seeded prior-draw starts, 40 iterations, authorized context.forward) achieve at least 10% lower validation rel_l2 in original conductivity space than fixed-configuration extra_trees and torch_mlp regressors trained on 3000 raw rows?

This supersedes S-Q1-5. Its physics arm timed out at 120 s with no output. Control results from S-Q1-5 are known, but the physics design changes respond only to the timeout and to critic objections, not to any physics result, because none exists.

Pre-registration changes from S-Q1-5; everything else is carried over unchanged:

(A) Throughput. All rows and starts are solved jointly. Each iteration makes one batched FD-Jacobian call (N*p rows) and one batched trial step. If the trial is rejected, the iterate is kept and lambda is multiplied by 4; if accepted, lambda is divided by 3. FD step is 1e-4 in log k, one-sided toward the interior at the upper bound. Iterates are clipped to [log 0.5, log 2.0]. Cost is 3*(1+40*5+4)=615 calls per row.

(B) Deadline. Recovery stops iterating 20 s after fit start; validation stops 95 s after fit start. If either is hit, timed_out=true, and timed_out=true in SOLVER_VERIFICATION or PHYSICS_LM_REPORT makes C-Q1-6a/b uninformative.

(C) Budget (objection 056). Rows that cannot be covered by the remaining per-arm 2.8M budget take the prior centre. Any n_budget_fallback>0 on validation makes C-Q1-6a/b uninformative. calls_before_predict is reported.

(D) Seeds (objections 051 and 059). Starts use default_rng([seed_base, row index within the call]); validation uses seed_base=0 and recovery uses 10^6. The result therefore does not depend on earlier predict calls. predict_calls must equal 1, otherwise uninformative.

(E) Layout (objection 058). All three permutations are tried: as_returned, sensor_major_to_time_major and time_major_to_sensor_major. layout_ok requires |min SSE mean - 80| < 12 over 20 rows. Agreement with the fidelity arm: physics as_returned requires fidelity as_returned; physics sensor_major_to_time_major requires fidelity transposed_sensor_time; physics time_major_to_sensor_major cannot be confirmed by the fidelity arm and is uninformative.

(F) Final-iterate diagnostics (objections 060 and 061). The Jacobian is recomputed at the final iterate. Local Fisher sd in log k and its condition number are reported at that point. The projected gradient norm (components blocked by an active bound are zeroed) is reported, including its median for converged-but-stalled rows. Conv&stalled rows are not described as optima.

(G) Recovery (objection 062). solver_pass requires all of: Frobenius rel_l2 < 0.10, |SSE median - chi2_76 median| < 10, at most 1 of 20 rows above chi2_76 q99, disagreement <= 0.25, no timeout, no fallback, no non-finite rows. The per-target log-k RMSE is reported next to the local-Fisher sd. The 0.10 threshold is acknowledged as lax; the SSE conditions carry the verification weight.

(H) Validation SSE (objections 055 and 063). If the median exceeds the chi2_76 median + 15, or is missing, the claims are uninformative. n_sse_above_q99 is reported against its binomial expectation 0.01*n. If it exceeds 3x that expectation, the per-sample comparison is reported as 'driven partly by high-SSE rows'.

(I) Predict-time standardisation guard (objection 064) for batches of 50 rows or more.

(J) Missing reports (objection 057). If PHYSICS_LAYOUT_REPORT, SOLVER_VERIFICATION, PHYSICS_LM_REPORT or FORWARD_FIDELITY_REPORT is absent, unparsable, or from another run, the claims it gates are uninformative. arm_gates_ok printed by the physics arm is a necessary condition, not a sufficient one.

Unchanged from S-Q1-5:
- Fidelity acceptance rule (|mean-80|<4, |median-chi2_80 median|<4, max<chi2_80 ppf(1-0.01/200)).
- Objective sum(r^2/sigma^2), with no log-Jacobian and no Tikhonov term.
- k-space global Frobenius scoring.
- Paired bootstrap, B=2000, seed 0; the 97.5th percentile of the ratio must be < 0.9.
- Sanity floors < 0.70, with [0.65, 0.70) flagged as 'weak control'.
- Seed-1 fragility reporting.
- Inverse-crime scoping: upper bound under perfect specification against these specific untuned baselines only.
- Box asymmetry stated (physics clipped; torch_mlp unclipped; extra_trees inherently in range).
- Intersection-union rule for any joint statement.

Owner: `researcher-003`; registered evidence: `['resu-00305', 'resu-00296', 'resu-00299', 'resu-00303', 'resu-00301', 'resu-00297', 'resu-00298']`.

- Arm `S-Q1-6.physics` (treatment, required=True): `ok`; method `custom-solver(fwd)`; metrics `{"r2": 0.9965601178821133, "rel_l2": 0.019325695574256917, "rmse": 0.025485495658119698}`; effective hash `a3d5ada3e88b7e86a37b980161a053f5aea45adb7a1057a38b79f2f937ec36e3`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.fidelity` (diagnostic, required=True): `ok`; method `custom-solver(fwd)`; metrics `{"r2": -0.0008909775527933261, "rel_l2": 0.3306557378688622, "rmse": 0.4360477137503121}`; effective hash `2e2c6157d50337dff422b9229f0639bafe053e0f2b2dacb17187c2f8c64129ba`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.extra_trees` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.torch_mlp` (control, required=True): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9927600207712438, "rel_l2": 0.028045400632179675, "rmse": 0.0369844869642745}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.extra_trees_seed1` (exploratory, required=False): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.945239129945993, "rel_l2": 0.07710766974217785, "rmse": 0.10168468063005599}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.torch_mlp_seed1` (exploratory, required=False): `ok`; method `torch_mlp/raw+logy`; metrics `{"r2": 0.9916315067445958, "rel_l2": 0.03015698062612236, "rmse": 0.03976910408507212}`; effective hash `397d747b933257d4675eb1a80606ace7ea4f652ee41229b29313d78ccd5a4a94`.
  Normalization changes: `[]`.
- Arm `S-Q1-6.mean` (baseline, required=True): `ok`; method `mean_baseline/raw`; metrics `{"r2": -0.0011333917959855389, "rel_l2": 0.330697038745562, "rmse": 0.4361021787143162}`; effective hash `641ba5b24826a458558d45e0e13b0787745f0e9028669de61ac3dd1b2c8fc9e2`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-6.paired_et`: `ok`; artifact `resu-00307`; reason `registered computation returned`.
- `S-Q1-6.paired_mlp`: `ok`; artifact `resu-00308`; reason `registered computation returned`.
- `S-Q1-6.per_target_phys_et`: `complete`; artifact `resu-00309`; reason `registered computation returned`.
- `S-Q1-6.per_target_phys_mlp`: `complete`; artifact `resu-00310`; reason `registered computation returned`.
- `S-Q1-6.per_target_et_mean`: `complete`; artifact `resu-00311`; reason `registered computation returned`.
- `S-Q1-6.per_target_mlp_mean`: `complete`; artifact `resu-00312`; reason `registered computation returned`.
- `S-Q1-6.per_target_fidelity`: `complete`; artifact `resu-00313`; reason `registered computation returned`.
- `S-Q1-6.per_sample_phys_et`: `complete`; artifact `resu-00314`; reason `registered computation returned`.
- `S-Q1-6.per_sample_phys_mlp`: `complete`; artifact `resu-00315`; reason `registered computation returned`.
- `S-Q1-6.per_sample_et_seeds`: `complete`; artifact `resu-00316`; reason `registered computation returned`.
- `S-Q1-6.per_sample_mlp_seeds`: `complete`; artifact `resu-00317`; reason `registered computation returned`.

Claim `C-Q1-6a`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-6.extra_trees", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-6.physics"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019325695574256917}`; effect `0.75033637678387`; interval `[0.7286088137639991, 0.7702855222196127]`.
Evidence `['resu-00305', 'resu-00299']`; unmet requirements `['objection-069', 'objection-070', 'objection-071']`.

Claim `C-Q1-6b`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-6.torch_mlp", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-6.physics"}`; values `{"control": 0.028045400632179675, "metric": "rel_l2", "treatment": 0.019325695574256917}`; effect `0.31091390607262887`; interval `[0.2726341318965123, 0.3518126426715412]`.
Evidence `['resu-00305', 'resu-00303']`; unmet requirements `['objection-069', 'objection-070', 'objection-071']`.

Claim `C-Q1-6c`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-6.mean", "margin": 0.3, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-6.extra_trees"}`; values `{"control": 0.330697038745562, "metric": "rel_l2", "treatment": 0.07740693387889735}`; effect `0.7659279497254459`; interval `None`.
Evidence `['resu-00299', 'resu-00298']`; unmet requirements `['objection-069', 'objection-070', 'objection-071']`.

Claim `C-Q1-6d`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-6.mean", "margin": 0.3, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-6.torch_mlp"}`; values `{"control": 0.330697038745562, "metric": "rel_l2", "treatment": 0.028045400632179675}`; effect `0.9151930699513829`; interval `None`.
Evidence `['resu-00303', 'resu-00298']`; unmet requirements `['objection-069', 'objection-070', 'objection-071']`.

## Independent critic and uncertainty

- `objection-001` (S-Q1-1, open): Before any comparison, verify that the physics arm's forward solver matches the simulator. It must use the same 40-point grid, the same 200 explicit steps and the same dt. It must use the same interface averaging for the discontinuous coefficient in (k u_x)_x (harmonic or arithmetic) and the same segment-to-grid mapping. It must use the same Dirichlet handling, Gaussian initial condition, sensor indices [5,15,25,35] and observation times, and the same ordering of the 80-dim feature vector. Test: on a sample of training rows, simulate with the true k and report the residual RMS against the observations. It must be about sigma = 0.001 (chi^2/dof ≈ 1). A noise-free round trip (simulate, then invert) must also recover k to near machine or optimizer tolerance. Without this, a misspecified forward model is confounded with the estimator.. Resolution: `unresolved`.
- `objection-002` (S-Q1-1, open): Guarantee explicit-scheme stability inside the optimizer. The log-parameter search must stay within the box [log 0.5, log 2.0], by clipping or a bounded reparameterization. Check and report dt*max(k)/dx^2 <= 0.5 at k = 2.0. Iterates outside the bounds can make the solver blow up or produce non-finite values.. Resolution: `unresolved`.
- `objection-003` (S-Q1-1, open): Pre-register how optimizer failures are handled. Report the count of non-finite results, non-converged rows and rows stuck at a bound. Failed rows stay in the metric with a pre-declared fallback, such as the best start or the prior mean. They must never be silently dropped.. Resolution: `unresolved`.
- `objection-004` (S-Q1-1, open): Report convergence evidence for iters = 30 and multistart = 3. Show the distribution of final negative log-likelihood against the noise floor (≈ 80/2 in sigma-normalized units). Show the fraction of rows where the three starts disagree. Also state how starts are chosen (fixed points or seeded prior draws) and the seed. If convergence is incomplete, the claim measures an under-iterated optimizer, not physics inversion.. Resolution: `unresolved`.
- `objection-005` (S-Q1-1, open): Account for forward evaluations against max_forward_evals = 3,000,000. Count evaluations per row as iters × starts × (1 + gradient cost: finite-difference evaluations per parameter, or adjoint/autodiff), multiplied by the number of validation rows. Reserve budget for any later test-set run. Log the actual count.. Resolution: `unresolved`.
- `objection-006` (S-Q1-1, open): Disambiguate the estimator. With tikhonov = 0 and a uniform prior on k, the 'MAP' is a box-constrained MLE. A mode taken in log-k space includes the Jacobian term unless it is explicitly excluded. State the exact objective: sigma-weighted SSE, with or without the log-Jacobian. State whether posterior_samples = 128 plays any role in a point estimate with estimator = 'mode'.. Resolution: `unresolved`.
- `objection-007` (S-Q1-1, open): Compute rel_l2 in the original conductivity space, after exp for log-transformed arms, using the global Frobenius definition on the identical validation rows for all arms. Confirm that no arm is scored in log space.. Resolution: `unresolved`.
- `objection-008` (S-Q1-1, open): Pre-specify the uncertainty analysis for the 'at_least_better' claim with margin 0.1. Define success as rel_l2_phys <= 0.9 × rel_l2_control. Use a paired bootstrap over validation rows that recomputes both global Frobenius ratios in each resample, because rel_l2 is a ratio of norms and not a per-row mean. Fix the bootstrap seed and count, and require the upper CI bound of rel_l2_phys / rel_l2_control to be below 0.9.. Resolution: `unresolved`.
- `objection-009` (S-Q1-1, open): Account for control seed variance. Each control runs only seed 0. The MLP's initialization and batch order, and extra-trees' randomization, add variance that row bootstrap does not capture. Run at least 3–5 control seeds and use the control's seed-mean, or its best seed for a conservative test, or else explicitly restrict the claim to the seed-0 configuration.. Resolution: `unresolved`.
- `objection-010` (S-Q1-1, open): Specify whether observation_noise_augmentation (noise_scale, number of copies) is active for the library controls. Confirm that the scaler is fit on training data only, and that MLP training has no early stopping or checkpoint selection on the validation set used for the claim.. Resolution: `unresolved`.
- `objection-011` (S-Q1-1, open): Do not let synthesis call the controls 'the strongest library regressors'. Both use fixed, untuned hyperparameters: no search, and 150 MLP epochs with no schedule. Any positive result supports only 'better than these specific configurations'. If tuning is added, it needs a split separate from the claim's validation rows.. Resolution: `unresolved`.
- `objection-012` (S-Q1-1, open): Report per-target rel_l2 and R² for all arms, including extra_trees, not only physics vs torch_mlp. The pulse is centred at x = 0.5, so the outer segments may be weakly excited within the finite window. Pair this with a local Fisher or sensitivity check at a few prior points, presented only as a local diagnostic, to explain any per-segment failures. Do not average them away.. Resolution: `unresolved`.
- `objection-013` (S-Q1-1, open): Include the mean_baseline result as a sanity floor. Both controls should clearly beat it, so that a 'physics wins' result does not come from broken controls.. Resolution: `unresolved`.
- `objection-014` (S-Q1-1, open): State in the claim text that the physics arm uses privileged forward-model access and ignores the 3000 training rows, while the controls learn from data. The comparison is model-based versus data-driven, not estimator versus estimator under equal information.. Resolution: `unresolved`.
- `objection-015` (S-Q1-1, open): Leakage controls: the physics arm must read only validation observations and the known noise sigma, never validation targets. Test and stress observations must not inform the choice of iters, multistart or the fallback rules. Log a fingerprint check of each split at run time.. Resolution: `unresolved`.
- `objection-016` (S-Q1-2, open): Physics-arm forward-model fidelity must be verified before any verdict. Pre-registration (a) says the stability number dt*k_max/dx^2 is 'computed from facts', but the facts give only 40 grid points and 200 steps. They give no final time T, no dt and no domain length, so the number cannot be computed from the facts alone. The run must report dx, dt and T as read from the simulator configuration, plus the resulting ratio at k=2.0. The physics arm must also call the authorized simulator forward function, not a re-implementation. A re-implementation could differ in interface averaging for the discontinuous k (harmonic vs arithmetic), in segment-to-grid assignment (10 points per segment?), in how Dirichlet nodes are counted, or in the sensor-time sampling that maps 200 steps to 80 = 4x20 features. Required check: on a subset of training rows, forward(true k) residuals against the noisy observations should have a sigma-normalized SSE consistent with chi^2_80, i.e. about 80. Using training rows for this check is verification, not estimation, so it does not contradict (g).. Resolution: `unresolved`.
- `objection-017` (S-Q1-2, open): Leakage audit for the physics arm. Evidence must show that the per-row estimator receives only that row's noisy 80-dim validation observation. It must not receive the true k, the noise-free trajectory, the noise realization, or any test or stress rows. Arms must be shown to score identical validation row IDs in identical order, checked against the val fingerprint, because the paired bootstrap depends on row alignment. Claims C-Q1-2a/b require implementation verification, and none is currently present (evidence_ids is empty).. Resolution: `unresolved`.
- `objection-018` (S-Q1-2, open): The forward-eval budget must be accounted for. LM with a finite-difference Jacobian costs roughly (1 + 4) or (1 + 8) forward calls per iteration, plus rejected damping steps. Across 30 iterations and 3 starts that is about 450 to more than 1000 calls per row. The number of validation rows is not stated, so whether the run fits under max_forward_evals = 3,000,000 cannot be checked. The run must report n_val, total forward calls, and the Jacobian method (analytic, adjoint, or FD with step size). It must also pre-declare what happens if the budget is hit mid-run. Recommendation: rows that were not processed count as fallback (k = 1.25) and stay in the metric, consistent with (c).. Resolution: `unresolved`.
- `objection-019` (S-Q1-2, open): The inverse-crime caveat must be stated in the synthesis. The physics arm fits data generated by the exact same discretization, with exactly known noise sigma, initial condition and boundary conditions. Any advantage over the controls is therefore an upper bound for model-based inversion under perfect model specification. It is not evidence of robustness to model error. Pre-registration (g) acknowledges unequal information but not this zero-model-error advantage.. Resolution: `unresolved`.
- `objection-020` (S-Q1-2, open): The start-seeding protocol is ambiguous and must be fixed. 'Seed 0 per row' would give identical 3 starting points for every row, so the starts are not i.i.d. across rows. State whether the generator is seeded once with 0 and then drawn sequentially, or seeded per row as f(0, row_id). Report the actual starting points used. Also confirm that posterior_samples = 128 consumes neither forward evaluations nor RNG state that would shift the starts.. Resolution: `unresolved`.
- `objection-021` (S-Q1-2, open): The SSE reference floor is mis-specified. If the objective is sum(r^2/sigma^2), its expected value at a correct-model MLE is about 80 - 4 = 76. If the objective is 0.5*sum(r^2/sigma^2), it is about 38. Neither matches '80/2' exactly. Pre-declare the normalization and compare against the chi^2_{76} distribution, scaled consistently. Rows far above the floor indicate local minima or forward mismatch and should be counted separately.. Resolution: `unresolved`.
- `objection-022` (S-Q1-2, open): The constrained-optimizer semantics need definition. Clipping log-k iterates inside LM is projected LM, not a true box-constrained MLE solver. The 'non-converged' test (relative step > 1e-6) needs a defined norm, measured before or after projection. Bound-stuck rows should be accompanied by a projected-gradient / KKT check, so that true boundary optima can be told apart from iterates stalled against the clip.. Resolution: `unresolved`.
- `objection-023` (S-Q1-2, open): Post-processing is asymmetric between arms. Physics estimates are guaranteed to lie in [0.5, 2.0], while control predictions after exp are not clipped. Report the fraction of control predictions outside the bounds. Add a sensitivity analysis with controls clipped to the same box, so that the ratio is not partly an artifact of clipping.. Resolution: `unresolved`.
- `objection-024` (S-Q1-2, open): Control training stochasticity is outside the uncertainty model. The paired bootstrap resamples only validation rows. The extra_trees and torch_mlp controls are each a single seed-0 fit, so the 97.5th-percentile ratio is conditional on one training realization. The synthesis must state this limitation. Preferably, also report control rel_l2 across a few training seeds as a non-decisive robustness check.. Resolution: `unresolved`.
- `objection-025` (S-Q1-2, open): The dependency between the sanity floors and the comparisons must be pre-declared. If C-Q1-2c or C-Q1-2d fails, meaning a control is not at least 30% better than mean_baseline, then C-Q1-2a or C-Q1-2b respectively should be declared uninformative rather than counted as a success. Any synthesized statement such as 'physics beats data-driven regressors' requires both C-Q1-2a and C-Q1-2b to pass. That joint statement should be framed with the multiplicity of two one-sided tests in mind.. Resolution: `unresolved`.
- `objection-026` (S-Q1-2, open): Per-segment identifiability must be reported and must not be hidden by the global metric. The initial pulse is centred at x = 0.5, and with sigma = 0.001 the outer segments (sensors at indices 5 and 35) may carry weak information. The per-target diagnostics must show whether a global-Frobenius win is driven by the inner segments while the outer segments are no better than the controls. The per_sample_phys diagnostic should be defined before running; currently its content is unspecified. It should cover, for example, per-row error versus final SSE, start disagreement, and bound-stuck status.. Resolution: `unresolved`.
- `objection-027` (S-Q1-2, open): Held-out discipline relative to the superseded study S-Q1-1 must be documented. State which S-Q1-1 observations motivated the changes in this design, and confirm that they were validation-only. Confirm that the final test and stress splits remain untouched by this study's decisions and evaluation.. Resolution: `unresolved`.
- `objection-028` (S-Q1-3, open): Pre-registration (a) requires reading dx, dt, final time T and dt*k_max/dx^2 at k=2.0 from context.facts, and aborting if the ratio exceeds 0.5. The contract facts give only 'explicit finite differences, 40 grid points, 200 steps'. They contain no dt, T or dx convention (for example 40 vs 39 intervals). The fidelity arm code only prints context.facts. It computes no ratio and implements no abort. Either source dt and T from the simulator configuration and compute and enforce the 0.5 rule in code, or remove the rule from the pre-registration.. Resolution: `unresolved`.
- `objection-029` (S-Q1-3, open): Claims C-Q1-3a and C-Q1-3b are 'uninformative if the forward-fidelity check departs materially from chi^2_80', but 'materially' has no numeric threshold, which leaves room for a post-hoc judgement. Pre-specify an acceptance rule. For example: the mean of 200 chi^2_80 draws has SD of about 0.89, so require |sse_mean - 80| < 4, sse_median within the chi^2_80 sampling band, and sse_max below a stated chi^2_80 quantile with Bonferroni correction over 200 rows.. Resolution: `unresolved`.
- `objection-030` (S-Q1-3, open): The fidelity arm (S-Q1-3.fidelity) is marked required:false and is absent from required_arm_ids. No diagnostic id references it, and neither claim lists it in required_diagnostics. Yet both comparison claims are conditioned on its result. Make the arm required and add it as a required diagnostic for C-Q1-3a and C-Q1-3b, so a verdict cannot be issued without it.. Resolution: `unresolved`.
- `objection-031` (S-Q1-3, open): The fidelity code assumes three things: (1) fit() receives raw, unscaled observations X, with no scaler, PCA or augmentation from transform_order applied to custom_solver input; (2) y is in raw k units, not log-transformed; (3) context.forward(theta).reshape(n,-1) uses the same feature ordering (sensor-major vs time-major) as X. None of these is verified. If any fails, the SSE is meaningless. If it passes trivially, it still says nothing about the val split. Verify all three, or assert them in code, for example by checking that X is not standardised and y lies within [0.5, 2.0].. Resolution: `unresolved`.
- `objection-032` (S-Q1-3, open): Implementation verification of the harness 'physics' kind is required (requires_implementation_verification=true), but no evidence or test is specified showing that it does what the intervention describes. Specifically, it should be shown to:
- run LM in log k with clipping to the box and a finite-difference Jacobian costing 1+4 calls per iteration;
- use an objective of SSE/sigma^2 with no log-Jacobian;
- treat estimator='mode' as pure MLE, with posterior_samples=128 not affecting the point estimate;
- select the best start by SSE;
- fall back to the prior centre when all starts are non-finite or the budget is exhausted.
Add a code-level check, or a recovery test on synthetic rows with known k, before treating physics outputs as evidence.. Resolution: `unresolved`.
- `objection-033` (S-Q1-3, open): The diagnostics required by (b2), (c) and (j) are not produced by any declared arm or diagnostic kind:
- forward-call totals and n_val;
- non-finite, non-converged, bound-stuck and budget-fallback counts;
- the final SSE distribution against chi^2_76 and the 99th-percentile exceedance count;
- start-disagreement fraction;
- per-row SSE and bound-stuck status in per_sample_phys.
The generic per_sample and per_target kinds probably report only errors. Specify where each quantity is emitted, or the pre-registered failure accounting cannot be audited.. Resolution: `unresolved`.
- `objection-034` (S-Q1-3, open): The finite-difference Jacobian near the box faces is unspecified. If perturbations are clipped, the Jacobian column becomes zero and iterates stick at the bound, biasing the bound-stuck counts. If they are not clipped, forward calls are made at k>2, which may break the explicit-scheme stability limit. State the step size and how it behaves at the bounds.. Resolution: `unresolved`.
- `objection-035` (S-Q1-3, open): Start seeding is ambiguous: '3 i.i.d. draws from the uniform prior with seed 0 per row' could mean identical starts on every row. Specify the RNG scheme, for example seed = f(0, row index). Also state whether the forward budget is global or per arm, and the row-processing order, since budget-fallback rows depend on that order.. Resolution: `unresolved`.
- `objection-036` (S-Q1-3, open): Fidelity is checked only on training rows, but the physics arm is scored on val rows. Add the same chi^2_80 check on val true k (val is a development split, so this is permitted). That would rule out a generation mismatch between splits.. Resolution: `unresolved`.
- `objection-037` (S-Q1-3, open): The bootstrap conditions on a single seed-0 training realization of each control. Report the paired ratio against the seed-1 controls alongside it. If the 0.9 criterion passes for seed 0 but fails for seed 1, the synthesis must state that the result is fragile to control seed, even though seed 1 is non-decisive.. Resolution: `unresolved`.
- `objection-038` (S-Q1-3, open): The sanity floors C-Q1-3c and C-Q1-3d use point estimates only (requires_uncertainty=false). They gate the main claims, so report a bootstrap interval for the control/mean ratio, or at least flag borderline passes near 0.7.. Resolution: `unresolved`.
- `objection-039` (S-Q1-3, open): This study supersedes S-Q1-2. Disclose which design changes were made and whether any were informed by S-Q1-2 validation results. If they were, the val-split comparison is partly adaptive, and the claims should be framed as development-stage rather than confirmatory.. Resolution: `unresolved`.
- `objection-040` (S-Q1-3, open): The fidelity arm emits prior-centre predictions. Exclude it explicitly from any ranking, leaderboard or synthesis of predictive performance.. Resolution: `unresolved`.
- `objection-041` (S-Q1-3, open): Synthesis framing must keep the comparison's asymmetry explicit. The physics arm uses the exact simulator, known sigma, and known initial and boundary conditions (inverse crime). The controls learn from 3000 rows with fixed, untuned hyperparameters. A pass supports only 'model-based inversion under perfect specification beats these fixed-configuration regressors on val'. It does not support claims about learned surrogates in general, robustness to model error, or held-out test performance.. Resolution: `unresolved`.
- `objection-042` (S-Q1-3, open): No evidence exists yet: evidence_ids and verdicts are empty and the status is 'planned'. No claim may be adjudicated until all required arms, the fidelity check and the required diagnostics have run and been recorded.. Resolution: `unresolved`.
- `objection-043` (S-Q1-4, open): Both claims set requires_implementation_verification=true, but there is no pre-registered pass/fail criterion for the solver itself. The fidelity arm verifies the forward model at the true k only. It does not verify the LM solver. PHYSICS_LM_REPORT prints the SSE distribution against chi2_76, n_sse_above_q99, disagreement and bound-stuck counts, but no threshold turns these into a verdict. Before running, pre-register an acceptance rule, for example: SSE median within a stated tolerance of the chi2_76 median, n_sse_above_q99 at most a stated fraction of n_val, and frac_start_disagree below a bound. Alternatively or additionally, run a truth-recovery check: apply the same predict path to a small block of training rows whose k is known, and set a recovery tolerance. Without either, a 'physics wins' result cannot be separated from a solver that happens to land near good answers, and a 'physics loses' result cannot be separated from a solver bug.. Resolution: `unresolved`.
- `objection-044` (S-Q1-4, open): The physics arm picks its feature layout from 20 training rows. It neither aborts nor gates when the SSE of the chosen layout is far from 80. It also has no X-standardisation guard, which the fidelity arm does have. Pre-register two things: (1) the physics arm aborts or is declared uninformative if min(sse_mean_as_returned, sse_mean_transposed) falls outside a chi2_80 tolerance; and (2) physics transpose_used must equal the fidelity arm's ordering_used. Otherwise fidelity_pass=true does not establish that the physics arm consumed X the same way.. Resolution: `unresolved`.
- `objection-045` (S-Q1-4, open): The layout check only tests one permutation: it reshapes F as (4, m/4) and transposes, which turns sensor-major forward output into time-major. If the forward output is time-major and X is sensor-major, the inverse permutation is never tried. Both arms would then fail, and the whole study becomes uninformative rather than correct. Test both permutations, or confirm the forward's layout before the run. Also take the sensor count (4) from context rather than hard-coding it.. Resolution: `unresolved`.
- `objection-046` (S-Q1-4, open): Pre-registration item (d2) promises the fraction of control predictions outside [0.5, 2.0] and a clipped-control sensitivity ratio 'from the diagnostics'. No listed diagnostic (paired, per_target, per_sample) computes either. The physics arm enforces the prior box through clipping, while the controls do not. Torch_mlp with exp(log-target) can predict outside the box; extra_trees cannot. This asymmetric use of prior knowledge goes unmeasured. Either add a diagnostic that computes the clipped-control rel_l2 and ratio, or remove the item and state the asymmetry in the synthesis.. Resolution: `unresolved`.
- `objection-047` (S-Q1-4, open): The sanity-floor rules contradict each other. Passing requires a ratio below 0.70 (30% lower), yet ratios between 0.65 and 0.75 are 'flagged as borderline'. The 0.70–0.75 part of that band is already a failure. State whether a borderline flag only applies to passes in 0.65–0.70, and say what a borderline flag does to the informativeness of C-Q1-4a and C-Q1-4b.. Resolution: `unresolved`.
- `objection-048` (S-Q1-4, open): C-Q1-4a reports 'a seed-0 pass with seed-1 fail as fragile', but the seed-1 control arms and their per_sample diagnostics are marked required=false. If they are not run, fragility cannot be assessed. In that case the synthesis must say 'not assessed', not imply robustness. Either make the seed-1 arms required, or state the reporting rule for when they are absent. Also, the seed-1 check only compares the physics arm against the seed-1 control at seed 0's bootstrap. Specify that the identical bootstrap rule is applied to it.. Resolution: `unresolved`.
- `objection-049` (S-Q1-4, open): Budget adequacy depends on n_val, which is not stated. Worst case is about 3 + 3·30·(p+5) = 813 calls per row, against a reserve check of 900. The 2.8M cap therefore covers roughly 3,100 rows before budget_fallback rows (prior centre) start entering the metric. Those fallback rows would bias the comparison against physics. State n_val ahead of the run, or confirm n_val·900 + 40 < 2.8M. Also confirm that this cap plus the fidelity arm's 200 calls stays within the contract-wide 3M.. Resolution: `unresolved`.
- `objection-050` (S-Q1-4, open): The convergence flag is not scale-free. It uses relative step = max|Δlog k| / max|log k|, and |log k| approaches 0 as k approaches 1, so rows with all components near k≈1 inflate the flag. In addition, a final-iteration rejection sets last_rel=0, so those rows count as 'converged' while also being flagged 'stalled'. The non-converged and stalled counts are therefore not interpretable separately. Use an absolute log-step tolerance and/or a gradient-norm or relative cost-decrease criterion, and report the joint counts.. Resolution: `unresolved`.
- `objection-051` (S-Q1-4, open): Starts are seeded with default_rng([seed, i]), where i is the row index within one predict call. If the harness calls predict in batches, start seeds repeat across batches and break the stated scheme. Key the seed on a global validation row index, or assert that predict is called exactly once.. Resolution: `unresolved`.
- `objection-052` (S-Q1-4, open): The diagnostics are asymmetric. C-Q1-4a requires per_sample_phys, but C-Q1-4b has no per-sample physics-vs-MLP diagnostic. Without one, the claim against the MLP cannot be checked for being driven by a few rows, such as bound-stuck or local-minimum rows. Add a per_sample physics-vs-torch_mlp diagnostic.. Resolution: `unresolved`.
- `objection-053` (S-Q1-4, open): The synthesis is close to tautological and needs to be scoped tightly. Under the inverse crime (identical discretization, known sigma, IC and BC, sigma=1e-3, exact forward access), the physics arm is expected to win. The fixed-configuration controls are deliberately untuned: MLP with 150 epochs, no schedule, no early stopping; ET with no tuning. Any pass therefore says nothing about model-based versus learned inversion under misspecification, or against tuned learned baselines. The synthesis must not generalise beyond 'upper bound under perfect specification versus these specific fixed baselines'. A pass margin attributable mainly to weak controls should be stated as such.. Resolution: `unresolved`.
- `objection-054` (S-Q1-4, open): Per-target interpretation is missing an identifiability diagnostic. With the Gaussian centred at x=0.5 and Dirichlet ends, the outer segments (sensors at indices 5 and 35) are plausibly much less sensitive than the inner ones. The contract allows a local Fisher diagnostic, but none is registered. If per-target rel_l2 differences are discussed, add a local Fisher or Jacobian-conditioning summary at the estimates, within the contract's stated limits: local, not global. Otherwise do not attribute per-target gaps to identifiability.. Resolution: `unresolved`.
- `objection-055` (S-Q1-4, open): Fidelity is checked only on training rows; the predict interface receives no targets. Validation rows therefore have no direct check that they come from the same forward model and noise. The physics SSE distribution on validation against chi2_76 is the only indirect evidence. Make this an explicit, pre-specified secondary check: a val SSE median far above chi2_76 should be interpreted as mismatch or local minima, and should make C-Q1-4a and C-Q1-4b uninformative rather than counting as a loss.. Resolution: `unresolved`.
- `objection-056` (S-Q1-5, open): Budget exhaustion must not be able to produce a recorded loss. The physics arm's cumulative call counter (self.calls) is shared across fit and every predict call. If the harness calls predict on the 3000 training rows before validation (for example to compute train metrics), about 3000*813 ≈ 2.44M calls are spent first. Most validation rows would then fall back to the prior centre 1.25 and stay in the metric, and the registered rule only 'flags' this as biased. Either (i) make n_budget_fallback > 0 on the validation predict render C-Q1-5a/b uninformative, or (ii) obtain and record harness evidence of the predict-call order and split sizes. Also report how many calls were consumed before the validation predict started.. Resolution: `unresolved`.
- `objection-057` (S-Q1-5, open): Missing gate reports must default to 'uninformative'. The fidelity_pass, solver_pass, layout_ok, transpose_used/ordering agreement and val_sse_median_mismatch_flag gates exist only as stdout prints (FORWARD_FIDELITY_REPORT, SOLVER_VERIFICATION, PHYSICS_LAYOUT_REPORT, PHYSICS_LM_REPORT). The required_diagnostics list for C-Q1-5a/b does not include them: S-Q1-5.per_target_fidelity only compares the fidelity arm's prior-centre predictions with the mean baseline, which carries no fidelity information. Pre-register that if any of these reports is absent, unparsable, or comes from a different run or seed, the claims are uninformative, not passed.. Resolution: `unresolved`.
- `objection-058` (S-Q1-5, open): The layout detection handles only one of the two possible sensor/time permutations. fwd() and the fidelity arm both apply reshape(n,4,m//4).transpose(0,2,1), which maps sensor-major output to time-major. If context.forward returns time-major output and X is sensor-major, the needed inverse is reshape(n,m//4,4).transpose(0,2,1), which is never tried. The gates would correctly report layout_ok=False/fidelity fail, but the study would then be uninformative for a fixable reason. Evaluate both permutations, or record in advance that this case is accepted as a dead end.. Resolution: `unresolved`.
- `objection-059` (S-Q1-5, open): Validation results depend on prior predict calls. Starting points are seeded with default_rng([seed, global_row_offset + i]), and row_offset accumulates across predict calls. Validation estimates therefore depend on whether, and on how many rows, predict was called earlier (train, stress, chunked calls). Report row_offset at the start of the validation call, or reset the offset per split so the validation result can be reproduced exactly.. Resolution: `unresolved`.
- `objection-060` (S-Q1-5, open): The convergence labels misclassify stalled rows. A row whose best start had no accepted step in the final iteration gets last_step = 0 and is therefore labelled 'converged'. Under box clipping this can happen at a non-stationary point: the gradient pushes outward, the clipped proposal never decreases the cost, and the damping escalates. The joint table helps, but synthesis must not describe conv&stalled rows as converged optima. Add a projected-gradient-norm check (gradient components zeroed where the active bound blocks them) for the best start.. Resolution: `unresolved`.
- `objection-061` (S-Q1-5, open): The local Fisher diagnostic is computed at a stale point. Jlast is the Jacobian taken at the start of the final iteration, before that iteration's accepted step, not at the returned estimate. The reported per-target sd and condition number therefore describe a slightly different point than the estimate, and are distorted most for bound-stuck rows. Either recompute J at the final iterate (p*1 extra calls per row) or label the diagnostic as evaluated at the pre-final iterate.. Resolution: `unresolved`.
- `objection-062` (S-Q1-5, open): The 0.10 recovery threshold in the solver check is lax relative to the controls. The mean baseline's expected Frobenius rel_l2 under the U[0.5,2]^4 prior is about 0.33. A recovery rel_l2 below 0.10 is therefore within reach of a regressor-quality answer and does not by itself show a correct optimiser at sigma = 0.001. The SSE-median and q99 conditions do the real work. State this in synthesis, and also report recovery rel_l2 against the local-Fisher-predicted error scale.. Resolution: `unresolved`.
- `objection-063` (S-Q1-5, open): The validation SSE gate checks only the median. A minority of rows in local minima or affected by model mismatch (for example 10–30%) would not trip median > chi2_76 median + 15, yet could dominate the Frobenius error. Add n_sse_above_q99 on validation, compared with its binomial expectation, as a reported diagnostic. Also pre-register how an excess of high-SSE rows qualifies the per-sample comparison.. Resolution: `unresolved`.
- `objection-064` (S-Q1-5, open): Pipeline handling of the custom_solver arms is unspecified. The contract's transform_order (augmentation → features → scaler → pca → estimator) and target_transform are not specified for the custom_solver arms. The fit-time guards catch a standardised X or log-transformed y, but predict has no guard: a scaler applied only at predict time, or an exp applied to raw-k predictions, would go undetected. Specify scaler='none', target_transform='none' and noise_aug=0 for the physics and fidelity arms, and add the standardisation check to predict.. Resolution: `unresolved`.
- `objection-065` (S-Q1-5, open): API and forward-call accounting are unverified. Both custom arms call context.prior_center() as a function and pass batched arrays to context.forward, and the code counts each batch row as one call. Before attributing any outcome, record evidence that (i) prior_center is callable, (ii) the contract's max_forward_evals counts rows the same way, and (iii) the contract cap is per arm rather than shared across all arms.. Resolution: `unresolved`.
- `objection-066` (S-Q1-5, open): Synthesis must not read the result as a methods comparison. The physics arm uses the exact data-generating simulator, known sigma, IC and BC (inverse crime), and essentially no training data. The controls are untuned fixed configurations: torch_mlp runs 150 epochs with no schedule and no early stopping. A ratio far below 0.9 is the expected outcome, so it supports only 'privileged exact-model MLE beats these specific fixed regressors under perfect specification'. It must not be generalised to learned inversion or to any misspecified setting, and the bounding-box asymmetry (physics clipped; torch_mlp unclipped; extra_trees inherently in range) must be stated next to any number.. Resolution: `unresolved`.
- `objection-067` (S-Q1-5, open): The joint statement needs both sanity floors to pass and has no effect-size anchor. Any 'beats both controls' statement must require C-Q1-5c and C-Q1-5d to pass, and must use the intersection-union rule, under which no multiplicity correction is needed. Because the sanity floors are point estimates with no uncertainty, a ratio near 0.70 should be reported with its bootstrap interval rather than as a bare pass/fail.. Resolution: `unresolved`.
- `objection-068` (S-Q1-5, open): No outcome-dependent analysis changes. Final test and stress observations, and their SSE or fallback statistics, must not be used to revise thresholds, iteration counts, multistart counts or gates. Any change after seeing validation output requires a new registered version.. Resolution: `unresolved`.
- `objection-069` (S-Q1-6, open): Add a forward-throughput pilot before running. S-Q1-5 died at the 120 s harness limit and produced no output, and nothing here shows that the vectorised design fits. On training rows only, measure context.forward wall time at the batch sizes the arm actually uses: N*p = 3*4*n_val rows for the Jacobian and 3*n_val rows for the trial step. Report n_val, calls/s, and the projected time for 615*n_val calls plus 20*615 recovery calls. Show that this fits inside the 95 s deadline with margin, and inside whatever the harness kill limit is.. Resolution: `unresolved`.
- `objection-070` (S-Q1-6, open): Bound the deadline overshoot and protect report emission. The deadline is checked only at the top of each LM iteration. After it come one in-flight iteration (5N forward rows), the final Jacobian (4N rows) and a per-row Python SVD/pinv loop, so actual wall time can run well past 95 s. If the harness kills the arm at about 120 s, no PHYSICS_LM_REPORT is printed and the result repeats S-Q1-5. Fix this in one of two ways: (a) check time per iteration using a measured per-iteration cost and stop early enough that the final Jacobian and diagnostics finish under the harness limit, or (b) print a partial report before the final Jacobian.. Resolution: `unresolved`.
- `objection-071` (S-Q1-6, open): Specify and verify the harness's predict-call order and count: train, val, test and stress. Three parts of the arm depend on it. (1) predict_calls==1 is a gate, so any train-set or pre-val predict call makes the claims uninformative and burns budget: a 3000-row train predict costs about 1.85M of the 2.8M calls. (2) The 95 s deadline counts from fit start, not per call, so any predict after validation (final test, stress) hits the deadline at iteration 0. It then returns the best of 3 random prior draws, or the prior centre once the budget runs out. The physics arm's held-out evaluation would be silently degenerate. (3) Budget for test/stress is not reserved. Either give each predict call its own deadline and budget allocation, or declare in advance that the physics test/stress scores are invalid and must not be reported as method performance.. Resolution: `unresolved`.
- `objection-072` (S-Q1-6, open): Gate on, or at least pre-register handling of, validation n_nonfinite > 0. Rows where every start is non-finite silently take the prior centre. This is in neither arm_gates_ok nor the claim's uninformative list, whereas recovery does gate on it. The effect is conservative for the claim, but it should be stated, and the count should be reported next to the per-sample analysis.. Resolution: `unresolved`.
- `objection-073` (S-Q1-6, open): Tighten or justify the validation SSE mismatch threshold of chi2_76 median + 15. With hundreds or more validation rows, the sampling sd of the median is under 1. A +15 offset allows roughly a 20% excess residual variance, meaning substantial forward/noise misspecification between train and val, to pass undetected. This matters because fidelity is checked only on training rows. A threshold of about +5, or a threshold derived from the sampling distribution of the median at the actual n_val, is more defensible. Also report the validation SSE mean.. Resolution: `unresolved`.
- `objection-074` (S-Q1-6, open): Address the reference distribution for bound-stuck rows. The chi2_76 references (the recovery |median - 75.3| < 10 rule and the q99 counts) assume an interior optimum and local linearity. Rows with active bounds have an effective df between 76 and 80. Report the SSE median separately for bound-stuck and interior rows (partly present), along with the fraction of bound-stuck rows. Do not read q99 excess on bound-stuck rows as misfit.. Resolution: `unresolved`.
- `objection-075` (S-Q1-6, open): Clarify that clipping after an unconstrained LM solve is not a projected/active-set method. When a bound is active and the gradient points outward, the clipped step can be repeatedly rejected while lambda grows toward 1e12, so the row stalls without having converged. The proj_grad_norm diagnostics help, but the synthesis must not count conv&stalled or bound-stuck rows as MLE optima. If the stalled fraction is large, attribute any physics shortfall to the optimiser rather than to the MLE estimator.. Resolution: `unresolved`.
- `objection-076` (S-Q1-6, open): Report the forward-output dtype and an FD-Jacobian accuracy check. One-sided FD with h = 1e-4 in log k is adequate in float64 but marginal if context.forward returns float32. Compare h = 1e-4 against a central-difference or h = 1e-3 Jacobian on a few recovery rows, and report the relative discrepancy.. Resolution: `unresolved`.
- `objection-077` (S-Q1-6, open): Verify that the controls reproduce S-Q1-5. Control results are already known and the configurations and seeds are identical. Report whether the seed-0 extra_trees and torch_mlp validation rel_l2 match S-Q1-5 exactly, and investigate any mismatch, since it would indicate nondeterminism that the paired bootstrap does not capture. State explicitly that the margin and gate thresholds were not changed after seeing the control values.. Resolution: `unresolved`.
- `objection-078` (S-Q1-6, open): Limit the scope of the bootstrap uncertainty in the synthesis. The paired bootstrap resamples validation rows only. It does not capture control training variability (seed, MLP optimisation) or physics start randomness. Seed-1 runs are exploratory and non-gating. The conclusion must stay conditional on seed-0 controls. If either seed-1 control would reverse the decision under the identical rule, report it prominently.. Resolution: `unresolved`.
- `objection-079` (S-Q1-6, open): Reconcile the layout check with the fidelity check under a richer layout mismatch. Both arms test only the 4x20 sensor/time transpositions. If the forward output differs in size or ordering in some other way (for example, different time indices or sensor subset), the result is a broadcast crash or a high-SSE failure, not a clean diagnosis. Explicitly map such a crash to 'missing report → uninformative' and do not treat it as a method result.. Resolution: `unresolved`.
- `objection-080` (S-Q1-6, open): Report per-target identifiability evidence alongside the per-target comparison. The pulse is centred at x = 0.5 and there is one sensor per segment, so the outer-segment conductivities may be weakly informed. Pair the per-target physics-vs-control rel_l2 with the local-Fisher sd and the bound-stuck rate for each target. That way, any aggregate win or loss can be attributed to specific segments rather than presented as uniform.. Resolution: `unresolved`.
- `objection-081` (S-Q1-5, open): The treatment arm S-Q1-5.physics ended with status 'timeout', so it has no predictions, no finite treatment metric and no verified implementation contract. C-Q1-5a and C-Q1-5b must stay inconclusive. No synthesis may describe this as a physics loss, a physics failure, or evidence about the method's accuracy. A timeout is an execution failure, not an outcome.. Resolution: `unresolved`.
- `objection-082` (S-Q1-5, open): The 30 s wall-time estimate for the physics arm is not credible. On 750 validation rows the worst case is 813 forward calls per row, about 610k calls, plus about 16k for recovery and 40 for layout. The arm also runs a pure-Python per-row, per-start, per-iteration loop with up to 5 trial solves. The pre-registered budget analysis (b2) covers forward-call count but not wall time. A rerun requires: a measured per-forward-call latency, a wall-time budget derived from it, and a resource allocation that covers the worst case. Alternatively, vectorise the LM across rows. Any code or parameter change must be registered as a new superseding study version, not edited in place.. Resolution: `unresolved`.
- `objection-083` (S-Q1-5, open): No partial logs from the timed-out run are in evidence: PHYSICS_LAYOUT_REPORT, SOLVER_VERIFICATION, or the forward_calls and row count reached before the timeout. Before rerunning, recover or regenerate these logs to show whether the timeout happened in fit(), in the 20-row recovery check, or in predict(). The fix cannot be targeted without this.. Resolution: `unresolved`.
- `objection-084` (S-Q1-5, open): Document the harness call order for predict(). BoxLM.calls is cumulative and never reset, and row_offset advances on every predict call. If the harness calls predict on the 3000 training rows before validation (for example, for train metrics), that alone could consume about 2.4M of the 2.8M budget. Validation rows would then silently take the 1.25 prior-centre fallback, and seed indices would shift. Either guarantee predict is called only on validation rows, or reset or partition budget and offsets per split. Report predict_calls and n_budget_fallback for the validation call specifically.. Resolution: `unresolved`.
- `objection-085` (S-Q1-5, open): The fidelity gate result is missing from the evidence. fidelity_pass, ordering_used and the SSE statistics exist only in the printed FORWARD_FIDELITY_REPORT, which no result artifact records. The diagnostic S-Q1-5.per_target_fidelity only scores the fidelity arm's constant prior-centre predictions against mean_baseline (rel_l2 about 0.33 per target). That comparison says nothing about forward-model fidelity, so it cannot satisfy the 'per_target_fidelity' requirement for C-Q1-5a/b. The FORWARD_FIDELITY_REPORT fields must be captured as a structured, hashed artifact and checked against the pre-registered acceptance thresholds.. Resolution: `unresolved`.
- `objection-086` (S-Q1-5, open): Sanity-floor claims C-Q1-5c and C-Q1-5d are blocked by unresolved objections 056 and 057, whose content is not in the supplied evidence. The point values themselves are clear: ET/mean ratio 0.0774/0.3307 ≈ 0.234 and MLP/mean ratio 0.0280/0.3307 ≈ 0.085, both well below 0.70 and outside the weak-control band. Even so, neither claim can be resolved until 056 and 057 are addressed and their resolution is recorded.. Resolution: `unresolved`.
- `objection-087` (S-Q1-5, open): The completion record lists the fidelity, extra_trees, torch_mlp and mean arms as 'orphaned_arm_ids', even though they completed with status ok and feed computed diagnostics. Explain and fix the linkage, so that control results used for the C-Q1-5a/b verdict values (0.0774, 0.0280) are traceably bound to this study version and its row_hash 260db427….. Resolution: `unresolved`.
- `objection-088` (S-Q1-5, open): Layout detection tests only one alternative permutation: reshape(4, m//4) then transpose. That is the correct inverse only if forward returns sensor-major output and X is time-major. In the opposite case the 'transposed' candidate is a different, wrong permutation. The layout_ok gate would then fail and make the claims uninformative, rather than corrupt them. Either determine the forward output ordering from the simulator contract, or test both reshape orders (4×20 and 20×4) in both the physics and fidelity arms.. Resolution: `unresolved`.
- `objection-089` (S-Q1-5, open): The local Fisher diagnostic uses Jlast, the Jacobian computed at the start of the final iteration, before that iteration's accepted step. It is not evaluated at the returned estimate. Label it as such, or recompute J at the final iterate (cost: p extra calls per row).. Resolution: `unresolved`.
- `objection-090` (S-Q1-5, open): The convergence classification sets last_step = 0 whenever the final iteration has no accepted step, so stalled rows are automatically counted as 'converged'. Synthesis must use the joint counts (n_conv_stalled, etc.) and must not cite 'converged' alone as evidence that the solver reached a minimum.. Resolution: `unresolved`.
- `objection-091` (S-Q1-5, open): The physics arm sets pass_rel at recovery rel_l2 < 0.10 on 20 training rows. That is loose relative to the controls it must beat by a 10% margin: torch_mlp validation rel_l2 is 0.028, so the physics arm needs roughly < 0.025 at the bootstrap 97.5th percentile. solver_pass therefore cannot be read as evidence that the solver is competitive, only that it is not grossly broken. Synthesis should state this.. Resolution: `unresolved`.
- `objection-092` (S-Q1-5, open): Seed-1 robustness evidence exists only for the controls. ET seed-1 per-target rel_l2 is about [0.116, 0.027, 0.029, 0.092], against about [0.117, 0.028, 0.029, 0.092] at seed 0. MLP seed-1 is about [0.042, 0.016, 0.016, 0.036], against about [0.039, 0.016, 0.014, 0.034] at seed 0. The controls look stable, but the seed-0 vs seed-1 bootstrap for the physics treatment (e2) cannot be computed. Fragility must be reported as 'not assessed' until the physics arm completes.. Resolution: `unresolved`.
- `objection-093` (S-Q1-5, open): Synthesis must not carry forward any interpretation from the inconclusive verdicts beyond the observed control numbers. Permitted: ET rel_l2 0.0774, MLP 0.0280, mean 0.3307 on 750 validation rows, single seed-0 realization, no uncertainty interval. Not permitted: any statement about whether physics-based inversion beats or loses to data-driven regressors. Also note the per-target heterogeneity in the controls: targets 1 and 4 have errors 3–4× larger than targets 2 and 3. This is relevant context for interpreting any future physics per-target gaps, not a claim in itself.. Resolution: `unresolved`.
- `objection-094` (S-Q1-1, open): S-Q1-1 remains superseded. Resolution: `unresolved`.
- `objection-095` (S-Q1-2, open): S-Q1-2 remains superseded. Resolution: `unresolved`.
- `objection-096` (S-Q1-3, open): S-Q1-3 remains superseded. Resolution: `unresolved`.
- `objection-097` (S-Q1-4, open): S-Q1-4 remains superseded. Resolution: `unresolved`.
- `objection-098` (S-Q1-5, open): S-Q1-5 remains deferred. Resolution: `unresolved`.

test_bootstrap: `null`.

test_per_target_rel_l2: `null`.

test_uncertainty: `null`.

shift: `null`.

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

Agent instances: `6`; specialist tasks: `2`.

Fallback reasons (role calls): `{"service_error": 8}`.

Costs: `{"by_stage_s": {"final": 2.88, "study": 358.22}, "compute_budget_s": 760.0, "compute_remaining_s": 398.9, "compute_used_s": 361.1, "gpus": [], "jobs": 23, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {"execution": 29}, "llm_calls": 29, "llm_calls_max": 80, "llm_cost_usd": 15.3129, "llm_seconds": 1700.89, "wall_budget_s": 7200.0, "wall_elapsed_s": 1934.45, "workers": 6}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 360.0, "confirmation_used_seconds": 2.879, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-5:7", "seconds": 204.0}, {"action": "release", "actual_seconds": 0.0, "group_id": "study:S-Q1-5:7", "overrun_seconds": 0.0, "reserved_seconds": 204.0}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-5:8", "seconds": 204.0}, {"action": "release", "actual_seconds": 150.81300000000002, "group_id": "study:S-Q1-5:8", "overrun_seconds": 0.0, "reserved_seconds": 204.0}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-6:10", "seconds": 204.0}, {"action": "release", "actual_seconds": 0.0, "group_id": "study:S-Q1-6:10", "overrun_seconds": 0.0, "reserved_seconds": 204.0}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-6:12", "seconds": 204.0}, {"action": "release", "actual_seconds": 66.163, "group_id": "study:S-Q1-6:12", "overrun_seconds": 0.0, "reserved_seconds": 204.0}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 360.0}, {"action": "release", "actual_seconds": 2.879, "group_id": "final-export", "overrun_seconds": 0.0, "reserved_seconds": 360.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 540.145, "reserved_seconds": 0, "total_seconds": 760.0, "used_seconds": 219.855}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
