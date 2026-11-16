# V3 research product: heat_v3_full_s0_20261113_170700

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: registered work resolved or explicitly deferred; no justified next study

Model export certified: **False**. Studies complete: **1/2**.

Model performance and study completion are separate outcomes.

Run error: RuntimeError: final export failed: timeout after 100s

## Actual exported method

No final model was exported.

Validation selection value: `None`. Final metrics: `{}`.

Export certificate: `{}`.

Lineage: `{}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-1 | 3/3 | deferred | required evidence missing; preserve uncertainty |
| S-2 | 3/3 | complete | all registered required work completed |

### S-1: Does a forward-model physics fit (MAP in log-conductivity space, multistart=8, Levenberg-Marquardt) achieve lower validation rel_l2 than the strongest library regressor (extra_trees with standard scaling and log target transform) on the 80-dim sensor time series?

Owner: `researcher-003`; registered evidence: `['resu-00060', 'resu-00058', 'resu-00062']`.

- Arm `S-1.library_et` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-1.library_kr` (baseline, required=True): `ok`; method `kernel_ridge/raw+logy`; metrics `{"r2": 0.991468392913482, "rel_l2": 0.030427018542519214, "rmse": 0.04012521287916009}`; effective hash `8dc2fc3c9aedd09566e89d61f86658b5fb49dc2a6f9326e45c0e1cc69e293f1e`.
  Normalization changes: `[]`.
- Arm `S-1.physics_map` (treatment, required=True): `ok`; method `physics-MLE[log,ms=6,it=30]`; metrics `{"r2": 0.996560013322493, "rel_l2": 0.019325980337639572, "rmse": 0.025485871185920207}`; effective hash `2b8fd20de45f63459bfd6a5f5e6efc6ef2177189a30fda70bb776a7b288ebd67`.
  Normalization changes: `[{"effective": 6, "effective_present": true, "field": "params.multistart", "requested": 8, "requested_present": true}]`.

Diagnostics and absences:

- `S-1.paired`: `ok`; artifact `resu-00064`; reason `registered computation returned`.
- `S-1.paired_kr`: `ok`; artifact `resu-00065`; reason `registered computation returned`.
- `S-1.per_target`: `complete`; artifact `resu-00066`; reason `registered computation returned`.

Claim `C-1`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-1.library_et", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-1.physics_map"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019325980337639572}`; effect `None`; interval `None`.
Evidence `['resu-00062', 'resu-00060']`; unmet requirements `['unregistered varying factor:kind', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'matching:kind', 'matching:noise_aug', 'matching:noise_scale']`.

Claim `C-2`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-1.library_kr", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-1.physics_map"}`; values `{"control": 0.030427018542519214, "metric": "rel_l2", "treatment": 0.019325980337639572}`; effect `None`; interval `None`.
Evidence `['resu-00062', 'resu-00058']`; unmet requirements `['unregistered varying factor:kind', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'matching:kind', 'matching:noise_aug', 'matching:noise_scale']`.

### S-2: Under a forward model that is correctly specified, does the bounded log-conductivity physics fit (multistart=6, 30 LM iterations, tikhonov=0) achieve lower validation rel_l2 than (a) extra_trees and (b) kernel_ridge? Both library regressors use raw features, standard scaling, a log target and no noise augmentation. Registration fix: S-1 was ruled inconclusive only because the method fields kind, noise_aug and noise_scale necessarily differ between the physics and library families and were not registered as varying factors.

Owner: `researcher-003`; registered evidence: `['resu-00121', 'resu-00119', 'resu-00123']`.

- Arm `S-2.library_et` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-2.library_kr` (baseline, required=True): `ok`; method `kernel_ridge/raw+logy`; metrics `{"r2": 0.991468392913482, "rel_l2": 0.030427018542519214, "rmse": 0.04012521287916009}`; effective hash `8dc2fc3c9aedd09566e89d61f86658b5fb49dc2a6f9326e45c0e1cc69e293f1e`.
  Normalization changes: `[]`.
- Arm `S-2.physics_mle` (treatment, required=True): `ok`; method `physics-MLE[log,ms=6,it=30]`; metrics `{"r2": 0.996560013322493, "rel_l2": 0.019325980337639572, "rmse": 0.025485871185920207}`; effective hash `2b8fd20de45f63459bfd6a5f5e6efc6ef2177189a30fda70bb776a7b288ebd67`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-2.paired_et`: `ok`; artifact `resu-00125`; reason `registered computation returned`.
- `S-2.paired_kr`: `ok`; artifact `resu-00126`; reason `registered computation returned`.
- `S-2.per_target_et`: `complete`; artifact `resu-00127`; reason `registered computation returned`.
- `S-2.per_target_kr`: `complete`; artifact `resu-00128`; reason `registered computation returned`.

Claim `C-3`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-2.library_et", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-2.physics_mle"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019325980337639572}`; effect `0.7503326979999628`; interval `[0.7286589476422367, 0.7702869545331144]`.
Evidence `['resu-00123', 'resu-00121']`; unmet requirements `['objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-023', 'objection-024']`.

Claim `C-4`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-2.library_kr", "margin": 0.1, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-2.physics_mle"}`; values `{"control": 0.030427018542519214, "metric": "rel_l2", "treatment": 0.019325980337639572}`; effect `0.36484147105530135`; interval `[0.30800771591144827, 0.41582957972968076]`.
Evidence `['resu-00123', 'resu-00119']`; unmet requirements `['objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-023', 'objection-024']`.

## Independent critic and uncertainty

- `objection-001` (S-1, open): No evidence exists yet: status is 'planned', evidence_ids is empty and verdicts is empty. Neither C-1 nor C-2 can receive a verdict until all three required arms (S-1.library_et, S-1.library_kr, S-1.physics_map) have run on the validation split and the required diagnostics (S-1.paired, S-1.paired_kr, S-1.per_target) are produced from those runs.. Resolution: `unresolved`.
- `objection-002` (S-1, open): Implementation verification of the physics arm's forward model is required by both claims and has not been specified. The forward solver used inside the MAP fit must be shown to be identical to the data-generating simulator. That means the same 40-point grid, the same 200 explicit steps and dt, homogeneous Dirichlet ends, the known Gaussian IC at x=0.5, the same face/interface averaging of piecewise-constant k, the same segment-to-grid-index assignment, and the same 20 observation times per sensor at indices [5,15,25,35]. Acceptance test: on a sample of training rows, forward(k_true) reproduces the noiseless or noisy observations with residual RMS consistent with sigma=0.001 (≈0.001, not ≫0.001).. Resolution: `unresolved`.
- `objection-003` (S-1, open): Synthetic recovery check: on noiseless data generated from known k (including corners of [0.5,2]^4), the multistart LM must recover k to near machine or solver tolerance. Under sigma=0.001 noise, the per-target error spread should be roughly consistent with the local Fisher prediction. This must be reported before validation results, so that optimizer bugs are not confounded with the method comparison.. Resolution: `unresolved`.
- `objection-004` (S-1, open): Explicit-scheme stability must be checked. With k up to 2.0 on a 40-point grid and 200 steps, confirm that dt*k_max/dx^2 ≤ 0.5 holds for every k the optimizer may visit. LM iterates outside [0.5,2] (e.g., before projection or in log-space line searches) could produce unstable or NaN forward solves. The study must state how bounds are enforced in log space: clipping, a reparameterisation, or a bounded LM.. Resolution: `unresolved`.
- `objection-005` (S-1, open): The physics arm's estimator definition is ambiguous. params specify estimator='mode' together with posterior_samples=128, multistart=8 and tikhonov=0. Specify exactly what prediction is scored: the best-of-8 LM minimiser, or a posterior-sample-based mode. Also state whether the uniform-in-k prior's Jacobian term is included when optimising in log k. With tikhonov=0 and a flat prior, 'MAP' reduces to bounded maximum likelihood. Note that rel_l2/RMSE is minimised by the posterior mean, not the mode, so the arm as specified may not be the strongest physics estimator. This doesn't invalidate C-1/C-2, but it limits any synthesis claim of the form 'physics fitting beats learning'.. Resolution: `unresolved`.
- `objection-006` (S-1, open): The forward-evaluation budget must be accounted for and shown to fit within max_forward_evals=3,000,000. Rough cost per row: 8 starts × 30 iterations × (1 + 4 Jacobian columns, if finite differences are used), about 1,200 evaluations, plus any posterior sampling. This allows only about 2,500 rows in total, which may not cover the full validation split plus later test and stress use. Report evaluations used per row and in total. If budget truncation causes rows to be skipped or iterations to be reduced, the arm is not run as registered.. Resolution: `unresolved`.
- `objection-007` (S-1, open): Failure handling must be pre-specified. Every validation row must receive a prediction from every arm. Rows where LM fails to converge, returns NaN, or hits a bound must be counted and scored, for example by falling back to the best finite iterate or the prior mean. They must not be dropped. Report the count of such rows. Dropping hard rows from the treatment arm only would bias the paired comparison.. Resolution: `unresolved`.
- `objection-008` (S-1, open): The uncertainty quantification for the 'at_least_better' margin test is unspecified. Both claims set requires_uncertainty=true with margin 0.1. Pre-register the following:
- the statistic: rel_l2_treatment / rel_l2_control on the same validation rows;
- the method: a paired row-level bootstrap with a stated number of resamples and confidence level;
- the decision rule: the upper CI bound of the ratio must be below 0.9.

A point estimate below 0.9 is not sufficient.. Resolution: `unresolved`.
- `objection-009` (S-1, open): Seed and stochasticity coverage is thin. All arms use seed 0 only. Extra-trees and the multistart initialisation are stochastic. Either report results over several seeds per stochastic arm, or justify why single-seed variability is negligible relative to the 10% margin. The paired bootstrap captures row-sampling variance but not algorithmic seed variance.. Resolution: `unresolved`.
- `objection-010` (S-1, open): The study question asserts that extra_trees is 'the strongest library regressor', but no evidence supports this. The KR baseline uses fixed, untuned alpha=0.01 and gamma_mult=1.0, and the ET hyperparameters are also fixed. Either drop the 'strongest' wording from any synthesis claim, or provide hyperparameter selection done by cross-validation within the training split only. Tuning on the validation split and then comparing on that same split would leak information and is not allowed.. Resolution: `unresolved`.
- `objection-011` (S-1, open): Comparison asymmetry must be disclosed. The physics arm uses the exact simulator and prior knowledge: known IC, BCs, noise sigma and discretisation. The library arms see only 3,000 training rows. C-1 and C-2 are valid as stated, but any synthesis must restrict its conclusion to the well-specified-forward-model setting. It must not generalise to a misspecified or unknown forward model.. Resolution: `unresolved`.
- `objection-012` (S-1, open): The per-target diagnostic (S-1.per_target) is required for C-1 and must be reported for all three arms, not only physics vs ET. It should include per-segment RMSE and R^2 with CIs. Global Frobenius rel_l2 can be dominated by large-k segments and by well-observed segments. With sensors at indices 5, 15, 25 and 35, one per segment, and the pulse centred at x=0.5, the outer segments near the Dirichlet ends may be weakly identified. An aggregate win must not hide a per-target loss. If one occurs, state it in the synthesis.. Resolution: `unresolved`.
- `objection-013` (S-1, open): Weakly identifiable directions need a diagnostic. Report the local Fisher information or posterior spread per target at representative k. This shows whether any segment's error is noise-limited for every estimator and would therefore compress achievable relative gains. Fisher results must be described only as local sensitivity, per the contract's diagnostic_limits, not as a global identifiability guarantee.. Resolution: `unresolved`.
- `objection-014` (S-1, open): Multiplicity: two comparison claims (C-1 and C-2) share the same treatment arm and validation rows. Pre-specify a family-wise correction, such as Holm across the two margin tests. Alternatively, state explicitly that each claim is evaluated at its own level without correction.. Resolution: `unresolved`.
- `objection-015` (S-1, open): Held-out discipline must be confirmed. All tuning, multistart settings, iteration counts, failure-handling rules and the choice of estimator='mode' must be fixed before any test or stress observation is touched. Both claims concern validation rel_l2 only. Any test-split numbers reported later must not feed back into S-1 decisions.. Resolution: `unresolved`.
- `objection-016` (S-1, open): Matching checks for the library arms must be shown. ET and KR must both use the same training rows (train fingerprint cdad4859…), noise_aug=0, standard scaling, pca=0, a log target with exp back-transform, and the same validation rows (fingerprint edfcc32d…). The physics arm must be scored on exactly those validation rows, with predictions in original conductivity units. Report the row counts and fingerprints alongside the results.. Resolution: `unresolved`.
- `objection-017` (S-1, open): The physics likelihood weighting should be stated. The LM residual should be the unweighted (or 1/sigma-weighted, which is equivalent under i.i.d. noise) residual across all 80 readings, using sigma=0.001. Confirm that no extra noise augmentation or feature scaling is applied to the physics arm's inputs, so that the observations seen by each arm differ only in the allowed factors.. Resolution: `unresolved`.
- `objection-018` (S-2, open): Missing evidence: evidence_ids is empty, verdicts is empty and status is 'planned'. Neither C-3 nor C-4 can be assessed until all three required arms (S-2.library_et, S-2.library_kr, S-2.physics_mle) are run on the validation split with fingerprint edfcc32d…c0543. All four required diagnostics (paired_et, paired_kr, per_target_et, per_target_kr) must also be produced and attached as evidence.. Resolution: `unresolved`.
- `objection-019` (S-2, open): Implementation verification of 'correctly specified': show by direct comparison that the physics arm's forward model reproduces the data-generating simulator to numerical precision. This covers 40 grid points, 200 explicit steps, the same dt, and the same divergence-form interface treatment of k at segment boundaries (harmonic vs arithmetic averaging). It also covers the same segment-to-grid mapping, homogeneous Dirichlet ends, the same Gaussian initial condition at x=0.5, sensor indices [5,15,25,35], and the same observation-time subsampling. The 80 features imply 20 times per sensor, so the time indices and the feature ordering must match exactly. Report the max |forward(theta_true) - noiseless observation| on a sample of rows.. Resolution: `unresolved`.
- `objection-020` (S-2, open): Optimizer specification mismatch: the claim text says 'LM' and 'MLE', but the arm params list estimator='mode', posterior_samples=128 and no optimizer field. Registration or the logged config must state the optimizer (Levenberg–Marquardt or something else). It must also state how bounds [0.5, 2.0] are enforced in log space (projection, clipping, or a bounded transform), what role posterior_samples plays when the estimator is the mode, and confirm that mode with a uniform prior and tikhonov=0 is equivalent to bounded MLE.. Resolution: `unresolved`.
- `objection-021` (S-2, open): Uncertainty procedure must be pre-specified, because requires_uncertainty=true. Specify: a paired bootstrap over validation rows that recomputes the global-Frobenius rel_l2 ratio (treatment/control) in each resample; the number of resamples; the CI level; and the decision rule for 'at_least_better' with margin 0.1. For example, require the upper CI bound of rel_l2_T/rel_l2_C to be below 0.9, rather than only the point estimate. Also report the validation row count.. Resolution: `unresolved`.
- `objection-022` (S-2, open): Physics-fit convergence diagnostics: for each validation row, report final weighted residual (chi^2 relative to sigma=0.001 and 80 observations), iterations used, number of multistarts converging to the best basin, and the fraction of estimates at or near the bounds. Provide a failure/fallback policy decided before seeing results. Without this, a rel_l2 advantage or deficit cannot be attributed to the estimator rather than to optimization failures.. Resolution: `unresolved`.
- `objection-023` (S-2, open): Forward-evaluation budget accounting: log the total forward evaluations used by the physics arm, including those spent on the Jacobian (finite-difference or adjoint), and show it stays within max_forward_evals=3,000,000. With 6 starts × 30 iterations × (1 + 4 or more Jacobian evals) per row, this depends on the validation size.. Resolution: `unresolved`.
- `objection-024` (S-2, open): Held-out discipline: certify that no test or stress observations (fingerprint aa0964…ed2c) were used to choose the margin, multistart count, iteration count, or library hyperparameters.. Resolution: `unresolved`.
- `objection-025` (S-2, open): Revision provenance: S-2 revises S-1, which was run before. State whether the S-1 validation results were seen before fixing margin=0.1, multistart=6, iters=30 and the library hyperparameters. If they were, disclose this as a post-hoc registration.. Resolution: `unresolved`.
- `objection-026` (S-2, open): Inverse-crime scoping: if the data were generated with the identical discretization used for inversion, the result only shows superiority under exact model knowledge. The synthesis must not generalize it to model misspecification, a different discretization, or real data.. Resolution: `unresolved`.
- `objection-027` (S-2, open): Matching is effectively vacuous: allowed_varying_factors covers every method field, so any difference cannot be attributed to a specific design element. The claim wording credits 'bounded log-space', 'multistart=6' and 'LM', but no ablation is registered for these. Synthesis should state only that this physics configuration beats this specific baseline configuration.. Resolution: `unresolved`.
- `objection-028` (S-2, open): Baseline strength: kernel ridge (alpha=0.01, gamma_mult=1.0) and extra_trees (300 trees, max_features=0.5) use fixed, untuned hyperparameters. State how they were chosen. Consider a train-internal CV-tuned baseline so the 10% margin is not inflated by a weak control.. Resolution: `unresolved`.
- `objection-029` (S-2, open): Seed variability: every arm uses only seed 0. Extra-trees fitting and multistart initialisation are stochastic, so report sensitivity across a few seeds, or state that this source of variance is excluded from the CI.. Resolution: `unresolved`.
- `objection-030` (S-2, open): Per-target interpretation: the global-Frobenius rel_l2 is weighted toward large-k rows. The per_target diagnostics should report rel_l2 and R^2 for each of the 4 segments. They should flag segments where the physics fit is worse, for example outer segments that are weakly informed by the central pulse under Dirichlet conditions. The local Fisher diagnostic must not be presented as global identifiability.. Resolution: `unresolved`.
- `objection-031` (S-2, open): Multiplicity: C-3 and C-4 share the same treatment and validation set. State whether the two comparisons are adjusted for (for example Bonferroni or Holm on the CI level) or are interpreted independently.. Resolution: `unresolved`.
- `objection-032` (S-2, open): Log-target back-transform: the library arms fit in log space and predict with exp. This targets the conditional geometric mean, not the arithmetic mean that is optimal for rel_l2. Note this as a possible handicap to the controls, or report the arithmetic-space alternative.. Resolution: `unresolved`.
- `objection-033` (S-1, open): Resolve the matching-protocol failure before any verdict. Both claims register allowed_varying_factors = [method, params, features, scaler, pca, target_transform], but the arms also differ in 'kind' (library vs physics) and in 'noise_aug'/'noise_scale', which the physics arm does not have at all. The verdict engine correctly returned 'inconclusive' for C-1 and C-2. The fix must be a documented, pre-specified amendment, not a post-hoc relabel. Either register 'kind' as an allowed factor, since a library-vs-physics comparison cannot hold it fixed, or define how absent fields are matched (for example, physics noise_aug treated as 0, meaning no augmentation). Then re-run verdicts against the amended protocol.. Resolution: `unresolved`.
- `objection-034` (S-1, open): Satisfy requires_implementation_verification for both claims. Every arm has method_contract.diagnostics declared=[], observed=[], verified=false and source_hash=null. The inference policy says repeatability requires exported-artifact certification, and none is present. Required evidence: a source hash for the physics LM/multistart code, and an independent check that the forward solver used for inversion is the authorised simulator. Also confirm that each method's predict() returns one finite, in-bounds prediction per validation row.. Resolution: `unresolved`.
- `objection-035` (S-1, open): Reconcile the executed treatment with the registered question. The study question specifies multistart=8. The effective config was clipped to multistart=6 (requested_hash != effective_hash). Either restate the question and claim text to multistart=6, or re-run at 8 within the forward-evaluation budget. Until then, the evidence answers a different question than the one registered.. Resolution: `unresolved`.
- `objection-036` (S-1, open): Correct the false premise in the study question and claim framing. The question calls extra_trees 'the strongest library regressor', but on the same paired rows kernel_ridge has rel_l2 0.0304 against 0.0774 for extra_trees. C-1 is therefore a comparison against a weaker baseline. Any synthesis must lead with C-2, where the physics arm improves on kernel ridge by a relative effect of 0.365 (bootstrap CI [0.308, 0.416]). It must not present the 75% gain over extra_trees as the headline improvement over data-driven methods.. Resolution: `unresolved`.
- `objection-037` (S-1, open): Clarify the estimator identity: MAP vs MLE. The claim text says 'MAP'. The gate label says 'physics-MLE[log,...]', with estimator='mode' and tikhonov=0.0. Under the stated i.i.d. Uniform[0.5, 2.0] prior on k, the posterior mode in log-k space includes a Jacobian term (density proportional to k). It equals the box-constrained MLE only if that term is omitted or the mode is taken in k-space. State which objective was optimised and how the bounds were enforced (clipping, reparameterisation, or penalty), and fix the claim wording to match.. Resolution: `unresolved`.
- `objection-038` (S-1, open): Expand uncertainty beyond conditional row bootstrap. The intervals (n_bootstrap=200) are explicitly 'row sampling conditional on fitted models; excludes training/task seed variation', and every arm uses seed=0 only. Variation from extra_trees and kernel ridge training seeds and from physics multistart initialisation is unquantified. The margins look large relative to plausible seed effects (lower CI bounds 0.729 and 0.308 vs margin 0.1), but that is an expectation, not evidence. Run at least a small multi-seed replication, or explicitly scope the claim to seed 0.. Resolution: `unresolved`.
- `objection-039` (S-1, open): Disclose the inverse-crime / well-specified-model condition. The physics arm inverts with the same 40-point, 200-step explicit finite-difference simulator and the same known Gaussian noise model that generated the data. The advantage measured here is therefore an upper bound under zero model error. No synthesis may generalise to misspecified physics, a different discretisation, or unknown noise without a stress test, and stress observations are held out from research decisions.. Resolution: `unresolved`.
- `objection-040` (S-1, open): Report baseline tuning and matching of effort. The extra_trees (n_estimators=300, max_features=0.5) and kernel_ridge (alpha=0.01, gamma_mult=1.0) hyperparameters appear untuned. Neither uses noise augmentation (noise_aug=0). The physics arm, by contrast, has forward access up to 3,000,000 evaluations and a well-specified noise model. The claims as written compare specific configurations, which is acceptable, but any wording such as 'physics beats library regressors' needs a tuned baseline. Tuning must use train/val only.. Resolution: `unresolved`.
- `objection-041` (S-1, open): Provide per-row optimiser diagnostics for the physics arm. Needed: convergence or iteration-cap rates at iters=30, how often each parameter hits the [0.5, 2.0] bounds, how far apart the multistart optima are, and the actual forward-evaluation count against the 3M cap. None of these are logged; est_seconds=76.4 is an estimate, not a measurement.. Resolution: `unresolved`.
- `objection-042` (S-1, open): Complete the per-target diagnostic symmetrically. S-1.per_target reports per_target_rel_l2 for the treatment only: [0.0268, 0.0060, 0.0050, 0.0263]. Matching control values are needed. The treatment errors are concentrated in end segments 1 and 4, roughly 5x the interior segments. This is consistent with weaker sensitivity near the Dirichlet boundaries relative to sensors at indices 5 and 35. Report this as a limitation, and optionally add local Fisher sensitivity, interpreted only as local and assumption-conditional per the contract.. Resolution: `unresolved`.
- `objection-043` (S-1, open): Confirm that validation-set reuse did not inform the physics configuration. If iters, multistart, or transform were chosen by looking at validation rel_l2, the claim is a development-set result with selection bias. State that explicitly, and reserve any confirmatory statement for a single pre-registered test evaluation.. Resolution: `unresolved`.
- `objection-044` (S-1, open): Fix the internal inconsistency in the completion state. completion_fraction=1.0 and completed_work=3/3 are reported alongside status='incomplete' and completed_claim_work=0/2. The synthesis must report the study as incomplete with both claims inconclusive. The raw effect sizes must not be described as supporting C-1 or C-2 until the blocking items above are resolved.. Resolution: `unresolved`.
- `objection-045` (S-1, open): S-1 remains deferred. Resolution: `unresolved`.

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

Agent instances: `7`; specialist tasks: `3`.

Fallback reasons (role calls): `{"service_error": 5}`.

Costs: `{"by_stage_s": {"final": 100.06, "study": 94.18}, "compute_budget_s": 400.0, "compute_remaining_s": 205.76, "compute_used_s": 194.24, "gpus": [], "jobs": 7, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {"execution": 22}, "llm_calls": 22, "llm_calls_max": 80, "llm_cost_usd": 8.4288, "llm_seconds": 454.92, "wall_budget_s": 7200.0, "wall_elapsed_s": 624.51, "workers": 6}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 100.0, "confirmation_used_seconds": 100.059, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-1:4", "seconds": 112.4}, {"action": "release", "actual_seconds": 52.871, "group_id": "study:S-1:4", "overrun_seconds": 0.0, "reserved_seconds": 112.4}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-2:5", "seconds": 112.4}, {"action": "release", "actual_seconds": 41.309, "group_id": "study:S-2:5", "overrun_seconds": 0.0, "reserved_seconds": 112.4}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 100.0}, {"action": "release", "actual_seconds": 100.059, "group_id": "final-export", "overrun_seconds": 0.0589999999999975, "reserved_seconds": 100.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 205.761, "reserved_seconds": 0, "total_seconds": 400.0, "used_seconds": 194.239}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
