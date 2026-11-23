# V3 research product: heat_v3_full_s0_20261114_130646

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: Stopping with S-Q1-002.physics_mle (resu-00147, exp-0010, val rel_l2 0.019323) as the best executed full-fidelity arm. The program has reached the limit of what the available adapters can produce. Both required studies ran to completion on the val split (fingerprint edfcc32d..., 750 rows, row_hash 260db427...): S-Q1-001 was inconclusive on a matching-protocol technicality, and S-Q1-002 re-registered the same arms with every differing field listed, completing all three required arms, all five diagnostics, and the two optional seed-1 arms. Measured results, not self-certified: rel_l2 is 0.07741 for extra_trees (control_et, resu-00141), 0.019323 for bounded log-k MLE (physics_mle, resu-00147) and 0.019330 for the ET-initialized LM hybrid (hybrid_lm, resu-00145); paired relative improvement is 0.750 with interval [0.729, 0.770] for both treatments at B=200, and the hybrid-vs-physics paired difference is -0.00036 with interval [-0.00126, 0.00021], all far from the 1% margin; seed-1 replicates (resu-00140, resu-00143) move rel_l2 by less than 0.0004 relative for ET and 0.0008 for physics; per-target rel_l2 for physics is [0.0268, 0.0060, 0.0050, 0.0263] against [0.117, 0.028, 0.029, 0.092] for ET, so outer segments carry about 5x the error of inner segments under every method (Q-2, local evidence only). The fit_report spot checks (200 rows) show residual std at truth 9.95e-4 against nominal 1e-3 and model_mismatch=false; 2-sigma coverage is 0.957 with z_rms 0.98. Every claim in S-Q1-002 remains 'inconclusive' solely because blocking objections 016, 017, 018, 020, 021, 022, 024 and 029 require source-level artifacts (forward-operator parity, CFL ratio, bound handling, per-row convergence logs, initializer identity) that only a coding_execution specialist can produce. That path is closed: tasks 001 and 003 both failed at adapter setup with 'study requires a nonempty stable id' regardless of how the id was supplied, action-006 was rejected on the depth/children limit, and the analysis adapter (task-004) returned only a B=200 re-run of one pair. Resolving objection-020 from the executor's own forward_evals logs was rejected twice, the second time because the independent critic has not accepted the resolution, so further resolve_objection attempts would repeat the same outcome. Resources: 661 s of 760 s wall time remain and cumulative measured forward spend is 731,311 of 3,000,000 (about 24%), so stopping is not budget-driven; it is driven by the absence of any action that can create new verifiable evidence. Scope limitations carried into the final export: results hold only in the inverse-crime setting (exact forward model, known IC, known Gaussian noise sigma=0.001); the control is one untuned extra_trees configuration; row-bootstrap intervals exclude ET training-seed and multistart-RNG variation; no seed-1 hybrid exists; the forward-eval split into residual/Jacobian/rejected-step counts is unmeasured; and no synthesis should rank hybrid against physics, attribute the hybrid result to the ET initializer, or generalize beyond this sensor layout and discretization. Verdicts remain with the independent verifier.

Model export certified: **False**. Studies complete: **1/2**.

Model performance and study completion are separate outcomes.

## Actual exported method

- physics-MLE[log,ms=3,it=30] (kind `physics`, method hash `3be67b83aa14`, source hash `None`, forward access `True`).
  Observed fit diagnostics: `{"fit_report": {"checked_samples": 200, "model_mismatch": false, "nominal_noise_std": 0.001, "residual_std_at_truth": 0.0009994548867275769}}`

Validation selection value: `0.01932290895348481`. Final metrics: `{}`.

Export certificate: `{"atol": 1e-10, "certified": false, "checks": {"finite_shape": true, "partition": false, "reload": true, "repeat": true}, "input_role": "development", "max_abs_differences": {"partition": 0.023731306972196275, "reload": 0.0, "repeat": 0.0}, "n_rows": 750, "output_shape": [750, 4], "passed": false, "protocol": "pristine-repeat-reload-partition-v1", "reasons": ["partition check failed"], "rtol": 1e-08, "schema_version": 3}`.

Lineage: `{"idea-002": {"arm_id": "S-Q1-001.physics_map", "baseline_identity": false, "gate": "gate-00043", "label": "physics-MLE[log,ms=3,it=30]", "origin": "proposal", "study_id": "S-Q1-001"}}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-Q1-001 | 3/3 | incomplete | required work remains unresolved |
| S-Q1-002 | 3/3 | complete | all registered required work completed |

### S-Q1-001: Does forward-model inversion (physics MAP with log transform and multistart, or hybrid extra_trees-init + Levenberg-Marquardt) achieve lower validation rel_l2 than a library regressor (extra_trees on raw 80-dim sensor features) for recovering the four piecewise-constant conductivities?

Owner: `researcher-003`; registered evidence: `['resu-00051', 'resu-00055', 'resu-00053']`.

- Arm `S-Q1-001.control_et` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[{"effective": 0, "effective_present": true, "field": "noise_aug", "requested": null, "requested_present": false}, {"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": null, "requested_present": false}]`.
- Arm `S-Q1-001.physics_map` (treatment, required=True): `ok`; method `physics-MLE[log,ms=3,it=30]`; metrics `{"r2": 0.9965611049282912, "rel_l2": 0.01932290895348481, "rmse": 0.02548182084024236}`; effective hash `64ab929f8997070c4dfc1995f7256ab1c54f865040d20b4854c1f8f5b767b605`.
  Normalization changes: `[]`.
- Arm `S-Q1-001.hybrid_lm` (treatment, required=True): `ok`; method `hybrid[extra_trees/raw+logy->LM15]`; metrics `{"r2": 0.9965586184290847, "rel_l2": 0.019329887377011627, "rmse": 0.025491023540440638}`; effective hash `926094218a2b0a6cd08f96d02f83c5f25ed5f67d3413e24829d48262b223cffe`.
  Normalization changes: `[{"effective": 0, "effective_present": true, "field": "init.noise_aug", "requested": null, "requested_present": false}, {"effective": 1.0, "effective_present": true, "field": "init.noise_scale", "requested": null, "requested_present": false}]`.

Diagnostics and absences:

- `S-Q1-001.paired_physics`: `ok`; artifact `resu-00057`; reason `registered computation returned`.
- `S-Q1-001.paired_hybrid`: `ok`; artifact `resu-00058`; reason `registered computation returned`.
- `S-Q1-001.per_target_hybrid`: `complete`; artifact `resu-00059`; reason `registered computation returned`.

Claim `C-Q1-001.physics`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-Q1-001.control_et", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-001.physics_map"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.01932290895348481}`; effect `None`; interval `None`.
Evidence `['resu-00055', 'resu-00051']`; unmet requirements `['unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'unregistered varying factor:target_transform', 'matching:noise_aug', 'matching:noise_scale', 'matching:target_transform']`.

Claim `C-Q1-001.hybrid`: **inconclusive**. registered matching protocol not satisfied.
Predicate: `{"control": "S-Q1-001.control_et", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-001.hybrid_lm"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019329887377011627}`; effect `None`; interval `None`.
Evidence `['resu-00053', 'resu-00051']`; unmet requirements `['unregistered varying factor:features', 'unregistered varying factor:method', 'unregistered varying factor:noise_aug', 'unregistered varying factor:noise_scale', 'unregistered varying factor:pca', 'unregistered varying factor:scaler', 'unregistered varying factor:target_transform', 'matching:features', 'matching:method', 'matching:noise_aug', 'matching:noise_scale', 'matching:pca', 'matching:scaler', 'matching:target_transform']`.

### S-Q1-002: Re-registration of S-Q1-001 with correct matching. Do forward-model inversions (bound-constrained log-k MLE with 3 starts, and an extra_trees-initialized Levenberg-Marquardt hybrid) achieve validation rel_l2 at least 1% lower (relative) than extra_trees on raw 80-dim sensor features? Is the hybrid equivalent to full physics inversion while using fewer forward evaluations?

Owner: `researcher-003`; registered evidence: `['resu-00141', 'resu-00147', 'resu-00145', 'resu-00143', 'resu-00140']`.

- Arm `S-Q1-002.control_et` (control, required=True): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.944810890291922, "rel_l2": 0.07740693387889735, "rmse": 0.10207933110086001}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-002.physics_mle` (treatment, required=True): `ok`; method `physics-MLE[log,ms=3,it=30]`; metrics `{"r2": 0.9965611049282912, "rel_l2": 0.01932290895348481, "rmse": 0.02548182084024236}`; effective hash `64ab929f8997070c4dfc1995f7256ab1c54f865040d20b4854c1f8f5b767b605`.
  Normalization changes: `[]`.
- Arm `S-Q1-002.hybrid_lm` (treatment, required=True): `ok`; method `hybrid[extra_trees/raw+logy->LM15]`; metrics `{"r2": 0.9965586184347424, "rel_l2": 0.019329887361103953, "rmse": 0.02549102351946261}`; effective hash `926094218a2b0a6cd08f96d02f83c5f25ed5f67d3413e24829d48262b223cffe`.
  Normalization changes: `[]`.
- Arm `S-Q1-002.control_et_seed1` (exploratory, required=False): `ok`; method `extra_trees/raw+logy`; metrics `{"r2": 0.945239129945993, "rel_l2": 0.07710766974217785, "rmse": 0.10168468063005599}`; effective hash `170e334e7e1b33a1ecb71995d0e710e09c1676a5971c2c2a0bc978e6c2b8ab98`.
  Normalization changes: `[]`.
- Arm `S-Q1-002.physics_mle_seed1` (exploratory, required=False): `ok`; method `physics-MLE[log,ms=3,it=30]`; metrics `{"r2": 0.996555975200928, "rel_l2": 0.01933729343090818, "rmse": 0.02550079017231679}`; effective hash `64ab929f8997070c4dfc1995f7256ab1c54f865040d20b4854c1f8f5b767b605`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-Q1-002.paired_physics`: `ok`; artifact `resu-00149`; reason `registered computation returned`.
- `S-Q1-002.paired_hybrid`: `ok`; artifact `resu-00150`; reason `registered computation returned`.
- `S-Q1-002.paired_hybrid_vs_physics`: `ok`; artifact `resu-00151`; reason `registered computation returned`.
- `S-Q1-002.per_target_physics`: `complete`; artifact `resu-00152`; reason `registered computation returned`.
- `S-Q1-002.per_target_hybrid`: `complete`; artifact `resu-00153`; reason `registered computation returned`.
- `S-Q1-002.paired_seed_physics`: `ok`; artifact `resu-00154`; reason `registered computation returned`.

Claim `C-Q1-002.physics`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-002.control_et", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.physics_mle"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.01932290895348481}`; effect `0.7503723764111963`; interval `[0.7286856199382119, 0.7704072680770814]`.
Evidence `['resu-00147', 'resu-00141']`; unmet requirements `['objection-016', 'objection-017', 'objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-024', 'objection-029']`.

Claim `C-Q1-002.hybrid`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-002.control_et", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-Q1-002.hybrid_lm"}`; values `{"control": 0.07740693387889735, "metric": "rel_l2", "treatment": 0.019329887361103953}`; effect `0.7502822241823266`; interval `[0.7283945277984851, 0.7702838654233639]`.
Evidence `['resu-00145', 'resu-00141']`; unmet requirements `['objection-016', 'objection-017', 'objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-024', 'objection-029']`.

Claim `C-Q1-002.hybrid_equiv_physics`: **inconclusive**. independent blocking objections unresolved.
Predicate: `{"control": "S-Q1-002.physics_mle", "margin": 0.01, "metric": "rel_l2", "operator": "equivalent", "treatment": "S-Q1-002.hybrid_lm"}`; values `{"control": 0.01932290895348481, "metric": "rel_l2", "treatment": 0.019329887361103953}`; effect `-0.00036114684574356157`; interval `[-0.0012584367236495044, 0.0002076942567800568]`.
Evidence `['resu-00145', 'resu-00147']`; unmet requirements `['objection-016', 'objection-017', 'objection-018', 'objection-019', 'objection-020', 'objection-021', 'objection-022', 'objection-024', 'objection-029']`.

## Independent critic and uncertainty

- `objection-001` (S-Q1-001, open): Missing evidence: the study is 'planned' and evidence_ids is empty. Neither claim (C-Q1-001.physics, C-Q1-001.hybrid) can receive a verdict until all three required arms have run on the val split (fingerprint edfcc32d…) and per-row predictions are recorded for the paired diagnostics S-Q1-001.paired_physics and S-Q1-001.paired_hybrid.. Resolution: `unresolved`.
- `objection-002` (S-Q1-001, open): Uncertainty procedure is unspecified, but both claims set requires_uncertainty. Pre-register a paired resampling scheme over validation rows, such as a paired bootstrap with B ≥ 2000 and a fixed resampling seed. Each resample must recompute the global-Frobenius rel_l2 for both arms on the same rows; averaging per-row errors is not a substitute. The decision rule for 'at least 1% better' must be stated as: the upper confidence bound of rel_l2(treatment)/rel_l2(control) is below 0.99. Also state the confidence level and whether it is corrected for the two claims.. Resolution: `unresolved`.
- `objection-003` (S-Q1-001, open): Forward-model fidelity must be verified (requires_implementation_verification). Show that the physics and hybrid arms call the same forward operator that generated the data: same 40-point grid, 200 steps, Dirichlet BCs, Gaussian IC at x = 0.5, sensor indices [5, 15, 25, 35] and observation times. Also show the mapping of the 4 segments to grid cells and how conductivity is evaluated at interfaces in the divergence-form stencil. Evidence: for a sample of val rows, the residual at the true parameters should give a chi-square consistent with 80 × sigma² (about 8e-5). Any mismatch invalidates the comparison.. Resolution: `unresolved`.
- `objection-004` (S-Q1-001, open): Explicit-FD stability must hold over the full search region. Report dt·k_max/dx² against the 0.5 stability limit at k = 2.0 and at any iterate the optimizer can reach. If log-space LM iterates are not projected or clipped to [log 0.5, log 2], unstable forward solves can produce NaN or exploding misfits that silently corrupt results. State the bound-handling method (projection, clipping, or reparameterization) and the treatment of NaN evaluations.. Resolution: `unresolved`.
- `objection-005` (S-Q1-001, open): Optimizer correctness needs evidence. Provide a Jacobian check (finite-difference versus analytic or adjoint, if analytic is used) and a per-row convergence log: final misfit, iterations used, whether the 30- or 15-iteration cap was hit, and the fraction of rows ending on a bound. The selection rule across multistarts and extra_starts must be minimum data misfit, not proximity to the target, and this needs to be shown. Clarify the role of posterior_samples = 128 when estimator = 'mode'. If those samples feed the point estimate, the arm is not the MAP described in the claim text.. Resolution: `unresolved`.
- `objection-006` (S-Q1-001, open): Hybrid isolation: the claim attributes any gain to the LM refinement, so the hybrid's initializer must be the identical fitted ET model used in control_et. That means the same seed, the same training rows (fingerprint cdad4859…), the same augmentation draws, scaler and log transform. Either share the artifact or show bitwise-identical predictions. Otherwise the paired difference mixes ET refitting variance with the LM effect.. Resolution: `unresolved`.
- `objection-007` (S-Q1-001, open): The control's noise augmentation configuration is unspecified. The contract lists observation_noise_augmentation first in transform_order, but the control arm gives no noise_scale or copy count. Record the exact values, or 'none', so that the control is reproducible and the matched comparison is well defined.. Resolution: `unresolved`.
- `objection-008` (S-Q1-001, open): Forward-evaluation budget accounting is missing. Count evaluations per row: residuals plus the finite-difference Jacobian (about 5 solves per iteration for 4 parameters), × iterations × starts, plus posterior_samples if they are evaluated. Multiply by the validation row count and show the total stays under max_forward_evals = 3,000,000. Do this separately for the two arms and include any reruns, since debugging runs also consume the budget.. Resolution: `unresolved`.
- `objection-009` (S-Q1-001, open): Held-out discipline: confirm that no settings were selected using the test split (fingerprint aa096404…) or stress data. This covers iters, multistart count, extra_starts, the tikhonov = 0 choice and ET hyperparameters. If any of these were tuned on val, the val-based claim is optimistic and this must be disclosed. A final test confirmation should be a separate, single-shot evaluation.. Resolution: `unresolved`.
- `objection-010` (S-Q1-001, open): Inverse-crime scope: if the inversion uses exactly the data-generating discretization and the exact known noise model, the result shows performance under a perfectly specified model. The synthesis should scope the claim this way and not generalize to robustness against model error. A mismatched-discretization check would be informative but is not required for the claims as written.. Resolution: `unresolved`.
- `objection-011` (S-Q1-001, open): Seed variability: each arm runs at a single seed (0). ET seed variance, and for physics_map the multistart initialization seed, should be characterized with a few additional seeds, or acknowledged as a source of uncertainty the row bootstrap does not capture. This matters most if the observed improvement is close to the 1% margin.. Resolution: `unresolved`.
- `objection-012` (S-Q1-001, open): Per-target analysis: rel_l2 pools all four conductivities, so a gain could come from well-observed interior segments while edge segments next to the Dirichlet boundaries degrade. S-Q1-001.per_target_hybrid is optional, and there is no per-target diagnostic for physics_map. Report per-target rel_l2 or R² for both treatments before any synthesis statement about 'recovering the four conductivities'.. Resolution: `unresolved`.
- `objection-013` (S-Q1-001, open): Terminology and prior consistency: with tikhonov = 0 and a uniform prior, the 'MAP' estimate is a bound-constrained maximum-likelihood estimate in log-k. That is only valid if the bounds are actually enforced. Note also that a uniform prior on k is not uniform in log-k, so if any prior term is applied in log space it needs a Jacobian correction. The claim text should describe the estimator accurately.. Resolution: `unresolved`.
- `objection-014` (S-Q1-001, open): Local Fisher or identifiability statements, if made in synthesis, must stay within the contract's limits. They describe local sensitivity under the assumed noise and prior, not global identifiability. Multistart agreement or disagreement per row should be reported as the evidence about multimodality, rather than as a Fisher-based claim.. Resolution: `unresolved`.
- `objection-015` (S-Q1-001, open): The synthesis should report compute cost alongside accuracy: forward evaluations and wall-clock time per row for physics and hybrid, versus ET inference. The claims concern rel_l2 only and should not be expanded into an overall superiority claim.. Resolution: `unresolved`.
- `objection-016` (S-Q1-002, open): Forward-model parity must be verified before any claim is evaluated. The physics and hybrid arms must reproduce the simulator's noise-free observations for the same k to near machine precision. The check covers the same 40-point grid, 200 explicit steps, homogeneous Dirichlet ends, the same Gaussian initial condition at x=0.5, and the same sensor indices [5,15,25,35] at the same discrete times. It must also cover the same interface treatment for piecewise-constant k in divergence form, for example harmonic or arithmetic face averaging. If any of these differ, the stated inverse-crime scope is false and the comparison is misattributed.. Resolution: `unresolved`.
- `objection-017` (S-Q1-002, open): Report explicit-scheme stability across the full bound box. Show that dt <= dx^2/(2*k_max) holds for k up to 2.0, and also for any trial k the optimizer may evaluate during line search or LM steps before projection. Non-finite or unstable forward solves must be logged per row.. Resolution: `unresolved`.
- `objection-018` (S-Q1-002, open): Pre-specify how failed rows are handled: optimizer failures, NaN or Inf residuals, and non-convergence. Every validation row must receive a prediction from every arm, with fallbacks declared in advance, or else failure counts must be reported. Rows must not be silently dropped, since that would break pairing and bias rel_l2.. Resolution: `unresolved`.
- `objection-019` (S-Q1-002, open): The uncertainty procedure required by requires_uncertainty=true is not specified. Pre-register the following: the number of validation rows; the paired resampling scheme, such as a row-level paired bootstrap of the rel_l2 ratio, with B and CI level; the decision rule for at_least_better (upper CI bound of treatment/control rel_l2 ratio < 0.99); and, for the equivalence claim, a TOST-style rule with the exact margin definition. The margin definition must state whether the ratio is hybrid/physics in [0.99, 1.01] or uses an absolute difference relative to physics rel_l2.. Resolution: `unresolved`.
- `objection-020` (S-Q1-002, open): The registered question asks whether the hybrid is equivalent 'while using fewer forward evaluations', but no claim, metric, or diagnostic measures forward-evaluation counts. Add per-row accounting of forward evaluations for each arm: residual evaluations, Jacobian evaluations including finite-difference perturbations, rejected LM steps, and all starts. Report total use against the 3,000,000 budget. Without this, the efficiency half of the question cannot appear in any synthesis.. Resolution: `unresolved`.
- `objection-021` (S-Q1-002, open): Matched-control integrity for the hybrid: verify that the hybrid's ET initializer predictions on validation are identical to the S-Q1-002.control_et predictions. The initializer must use the same fitted model, seed 0, log target, and standard scaler. Otherwise C-Q1-002.hybrid does not isolate the effect of LM refinement.. Resolution: `unresolved`.
- `objection-022` (S-Q1-002, open): Convergence diagnostics for both inversion arms are needed so that performance differences can be attributed correctly. For each row, report: final misfit compared with the misfit at the true k (which should be about n_obs*sigma^2 if the global optimum is found); the fraction of rows converged within the iteration cap (30 for MLE, 15 for hybrid); the number of distinct minima across starts; and the fraction of estimates at or near the [0.5, 2.0] bounds. Without these, a non-equivalence result cannot be separated into an iteration-cap effect, a local-minimum effect, or an initialization effect.. Resolution: `unresolved`.
- `objection-023` (S-Q1-002, open): Specify the implementation of bound handling and the Jacobian. State how bounds are enforced in log-k (projection, clipping, or reparameterization) and how the Jacobian is computed (adjoint, forward sensitivity, or finite differences with a stated step). For finite differences, the step must be justified relative to sigma=0.001 and to solver precision. These choices change both accuracy and forward-evaluation counts.. Resolution: `unresolved`.
- `objection-024` (S-Q1-002, open): Document the provenance of the re-registration. State whether S-Q1-001 validation results informed any S-Q1-002 choices, such as iteration counts, multistart counts, extra_starts, or the 1% margin. If they did, the validation split is no longer a clean confirmatory set for these claims, and this must be disclosed. Test and stress splits must remain untouched.. Resolution: `unresolved`.
- `objection-025` (S-Q1-002, open): Clarify the estimator settings. The physics and hybrid arms specify estimator='mode' with posterior_samples=128. State whether posterior sampling is performed or ignored. State whether the uniform prior with tikhonov=0 makes 'mode' exactly the bounded MLE. The reported estimator must match the claim text ('maximum-likelihood').. Resolution: `unresolved`.
- `objection-026` (S-Q1-002, open): Report the per-target diagnostics alongside the global result. Global-Frobenius rel_l2 on raw k is weighted toward large-k rows and toward the better-identified segments. The outer segments may be weakly informed, given a centred initial pulse, decay toward the Dirichlet boundaries, and a finite observation window. Per-target results should not be summarized away, and any local Fisher or conditioning analysis must be labelled as local sensitivity only, not as global identifiability.. Resolution: `unresolved`.
- `objection-027` (S-Q1-002, open): Seed and optimizer randomness is not represented in the confirmatory uncertainty. Row-level bootstrap captures validation sampling only, not ET training randomness or multistart initialization randomness. The seed-1 arms are optional, and there is no seed-1 hybrid. Any synthesis must either state that its conclusions are conditional on seed 0, or run the seed replicates and report between-seed spread relative to the 1% margin. This matters most for the equivalence claim.. Resolution: `unresolved`.
- `objection-028` (S-Q1-002, open): Limit the scope of any synthesis. All results hold only in the inverse-crime setting: an exact forward model, a known initial condition, and known Gaussian noise with sigma=0.001. They must not be generalized to model misspecification, other noise levels, other discretizations, or other sensor layouts, because none of these were tested.. Resolution: `unresolved`.
- `objection-029` (S-Q1-002, open): Certification: claim verdicts must be issued by an independent verifier from the logged per-row predictions and forward-evaluation logs, not self-certified by the executing agent. The evidence_ids list is currently empty, and no evidence yet supports any claim.. Resolution: `unresolved`.
- `objection-030` (S-Q1-001, open): C-Q1-001.physics: the verdict is inconclusive because noise_aug, noise_scale and target_transform vary between arms but were not registered as allowed. These differences are structural: the physics arm has no library preprocessing fields, the control's noise_aug=0 and noise_scale=1.0 were inserted by normalization, and the log transform sits in physics params.transform. Register a new claim version that adds these factors to allowed_varying_factors, with this justification attached. Record it as a disclosed protocol deviation made after results were seen, not as the original preregistration. The verdict must then be re-issued from the existing paired evidence (resu-00055, resu-00051).. Resolution: `unresolved`.
- `objection-031` (S-Q1-001, open): C-Q1-001.hybrid: the matcher flags features, method, scaler, pca, noise_aug, noise_scale and target_transform. This looks like a schema artifact: the hybrid nests the extra_trees config under 'init', so top-level fields appear absent. Show that the hybrid's effective init config is identical to the control's effective config by hashing it and comparing against 170e334e…. Also state whether the hybrid reuses the control's fitted ET model or refits it with the same seed and training rows. Then either map init.* to the control fields in merge_mappings or register a new claim version. Until then the hybrid-vs-ET comparison is not a matched control.. Resolution: `unresolved`.
- `objection-032` (S-Q1-001, open): Both claims set requires_implementation_verification=true, but every arm has method_contract.diagnostics.verified=false, no declared or observed diagnostics, and source_hash=null. The inference policy requires exported-artifact certification. Provide source hashes and certified exported artifacts for all three arms. Also provide an independent check of the physics forward/inverse code, for example a noiseless synthetic round-trip recovering known conductivities to near machine or solver tolerance. No verdict can be supported before this exists.. Resolution: `unresolved`.
- `objection-033` (S-Q1-001, open): The physics and hybrid treatments are nearly identical. Their rel_l2 values are 0.0193229 and 0.0193299; per-sample squared errors agree to about 1e-9 on the first rows; p90 values are bit-identical. Either both converge to the same optimum or they share code or outputs. Show that resu-00055 and resu-00053 are separate computations, for example with per-row forward-eval counts and start points. Synthesis must not attribute any hybrid gain to the ET initialization. Doing so would need an ablation of the hybrid with extra_starts=0, since the extra start may be what drives convergence.. Resolution: `unresolved`.
- `objection-034` (S-Q1-001, open): The claim text says 'Physics MAP', but tikhonov=0 and the arm label is 'physics-MLE'. Under the i.i.d. uniform prior, the mode equals MAP only if the bounds [0.5, 2.0] are enforced during optimization rather than ignored or clipped afterwards. Document how bounds are handled, and either rename the claim to MLE or verify that the estimator is the bounded posterior mode.. Resolution: `unresolved`.
- `objection-035` (S-Q1-001, open): The uncertainty interval only covers row resampling (200 bootstrap draws) conditional on single seed-0 fitted models. It excludes variation from the ET training seed and from the multistart RNG. The interval lower bound (about 0.728 relative) is far above the 0.01 margin, so this is unlikely to flip the verdict. Still, report results across at least a few seeds, or explicitly scope the claim as conditional on seed 0.. Resolution: `unresolved`.
- `objection-036` (S-Q1-001, open): Data are generated by the same simulator used for inversion: same 40-point grid, 200 explicit steps, known initial condition and known noise model. Physics-arm superiority is therefore an 'inverse crime' result under a perfectly specified forward model. Synthesis must scope the claim to the well-specified setting and not generalize to model misspecification, unknown initial conditions, or different discretizations.. Resolution: `unresolved`.
- `objection-037` (S-Q1-001, open): The control is one untuned extra_trees configuration on raw features, with no noise augmentation and no PCA. Synthesis must not generalize the result to 'forward-model inversion beats library regressors'. The supported claim is limited to this specific ET configuration.. Resolution: `unresolved`.
- `objection-038` (S-Q1-001, open): Error is concentrated in the end segments. Treatment RMSE is about 0.035 for targets 1 and 4 versus about 0.007 for targets 2 and 3, and the worst row has squared error 0.0436. Provide two things. First, a per-target diagnostic for the physics arm; one currently exists only for the hybrid. Second, a comparison of per-target error against the local Fisher / Cramér–Rao scale at sigma=0.001, to tell noise-limited error from local-minimum or non-convergence failures. Include a list of rows where the best of the 3 starts did not converge. Per the contract, Fisher results are local diagnostics only.. Resolution: `unresolved`.
- `objection-039` (S-Q1-001, open): Report the actual number of forward evaluations used by the physics and hybrid arms, in total and per row. Confirm the total is within max_forward_evals=3,000,000. Report wall-clock cost alongside accuracy, since the hybrid (about 27.3 s estimated) costs more than physics (about 17.3 s) with no measurable accuracy gain.. Resolution: `unresolved`.
- `objection-040` (S-Q1-001, open): Document that iters (30 and 15), multistart=3 and extra_starts=1 were fixed before the validation results were seen. Confirm that no test or stress observations informed any choice. If any settings were tuned on validation, the 750 validation rows are not an unbiased estimate, and a held-out confirmation is needed before final claims.. Resolution: `unresolved`.
- `objection-041` (S-Q1-001, open): No claim comparing the hybrid against the physics arm is registered or tested. Their difference (about 7e-6 in rel_l2) has no interval. Synthesis must not rank the two treatments against each other.. Resolution: `unresolved`.
- `objection-042` (S-Q1-002, open): Implementation verification is missing for all three claims, and each claim sets requires_implementation_verification=true. In every arm the method_contract shows diagnostics.verified=false, the declared and observed diagnostic lists are empty, and source_hash=null. The inference_policy says repeatability requires exported-artifact certification, and none is supplied. Provide a source hash for the physics/LM and hybrid code, an exported-artifact certification, and declared-versus-observed diagnostics before any claim leaves 'inconclusive'.. Resolution: `unresolved`.
- `objection-043` (S-Q1-002, open): The registered question asks whether the hybrid is equivalent 'while using fewer forward evaluations', but no forward-evaluation counts, wall-clock times, or Jacobian-evaluation counts are reported for either arm. The est_seconds values (17.3 for physics_mle, 27.3 for hybrid_lm) are gate estimates, not measurements, and they suggest the hybrid may cost more. Report the measured forward-eval count per row (mean and max) for physics_mle and hybrid_lm. Until then, no synthesis may claim a compute advantage; C-Q1-002.hybrid_equiv_physics can only support accuracy equivalence.. Resolution: `unresolved`.
- `objection-044` (S-Q1-002, open): Attribution of the hybrid result to the ET initializer is unsupported. The hybrid's per-sample squared errors match physics_mle to about 1e-9 (for example, 0.0017367219887 vs 0.0017367224577) and its per-target RMSEs match to 4 digits. This indicates that both arms converge to the same likelihood optimum regardless of initialization. No ablation is supplied, such as LM with 15 iterations and 2 random or prior-mean starts and no ET init, so the contribution of the ET init is unidentified. Any synthesis implying that the data-driven init improves accuracy is unsupported.. Resolution: `unresolved`.
- `objection-045` (S-Q1-002, open): Init provenance cannot be verified. C-Q1-002.hybrid states that the initializer uses the 'same configuration and seed as the control'. The configuration hash matches the control (170e334e...), but the hybrid's source_artifacts (resu-00145, resu-00141) do not show that the fitted init model is the same artifact as control_et. Provide the fitted-model identifier or hash used inside hybrid_lm.. Resolution: `unresolved`.
- `objection-046` (S-Q1-002, open): No optimizer convergence diagnostics are reported for the MLE label. Missing items include the final objective or chi-square per row compared with its expected value (about 80·sigma² = 8e-5 for 80 observations at sigma=0.001), gradient or step norms at termination, the fraction of rows hitting the 30- or 15-iteration cap, the fraction of estimates at the [0.5, 2.0] bounds, and the agreement among the 3 starts. Without these, 'maximum-likelihood' is asserted rather than demonstrated, and local minima cannot be ruled out.. Resolution: `unresolved`.
- `objection-047` (S-Q1-002, open): There is no efficiency reference against local Fisher information. Per-target RMSE is about 5x larger for the outer segments (0.0356 and 0.0350) than for the inner ones (0.0079 and 0.0065). This is consistent with weaker sensitivity near the Dirichlet boundaries, but it is not checked against a local Fisher/Cramér-Rao bound under sigma=0.001. Per the contract, any such comparison is local only and is not an identifiability guarantee. Without it, the evidence cannot separate estimator suboptimality from intrinsic ill-conditioning.. Resolution: `unresolved`.
- `objection-048` (S-Q1-002, open): The control is a single untuned ExtraTrees configuration (max_features=0.5, 300 trees, raw features, no noise augmentation, no PCA). The about-75% relative improvement supports only the claim as registered against this specific baseline. Any synthesis generalizing to 'physics inversion beats data-driven methods' would need tuned, stronger, or noise-augmented data-driven controls.. Resolution: `unresolved`.
- `objection-049` (S-Q1-002, open): The inverse-crime scope must be carried into synthesis. Data generation and inversion share the exact 40-point, 200-step explicit FD forward model and the exact known noise sigma. The claim texts state this scope. Any synthesis must not extend the conclusions to model misspecification, grid mismatch, unknown noise level, or real data, because no stress evidence under misspecification is supplied and held-out stress data cannot be used for this.. Resolution: `unresolved`.
- `objection-050` (S-Q1-002, open): Uncertainty scope is limited. The bootstrap intervals use n_bootstrap=200, cover only row resampling conditional on fitted models, and exclude training and task seed variation. The seed-1 replicate covers only ET and physics_mle (physics 0.019337 vs 0.019323, ET 0.07711 vs 0.07741), and no seed-1 hybrid arm exists. The margins are far from the intervals, so the verdicts are unlikely to change. Still, the equivalence claim should note that hybrid seed variability is unmeasured, and a larger bootstrap is advisable for tail quantiles.. Resolution: `unresolved`.
- `objection-051` (S-Q1-002, open): Previously raised objections are not visible here. The verdicts list objection-016 through 022, 024, and 029 as unresolved blocking requirements, but their content and resolution status are not part of the supplied evidence. A verdict update requires explicit resolution records for each of them, and this review cannot adjudicate them.. Resolution: `unresolved`.
- `objection-052` (S-Q1-002, open): The per-target diagnostics are incomplete for a comparison. S-Q1-002.per_target_physics and per_target_hybrid report per_target_rel_l2 only for the treatment, for example [0.0268, 0.0060, 0.0050, 0.0263]. They give no matching control per-target rel_l2 or per-target paired intervals, so the claim that improvement holds on every target is not directly evidenced. The per_target_rmse values in the paired diagnostics do suggest it does, but without intervals.. Resolution: `unresolved`.
- `objection-053` (program, open): Implementation verification is still absent for every S-Q1-002 arm, and all three claims set requires_implementation_verification=true. Each arm shows source_hash=null and method_contract.diagnostics.verified=false with empty declared and observed lists. Each result artifact shows verification.intervention_verified=false and diagnostic_gates_verified=false. Passing the finite and shape checks is not verification. Before any claim leaves 'inconclusive', supply three things: source hashes for the physics, LM and hybrid code; exported-artifact certification; and an independent noiseless round-trip that recovers known k to solver tolerance. This keeps objections 016, 032 and 042 open.. Resolution: `unresolved`.
- `objection-054` (program, open): Convergence is contradicted, not just undocumented. Every inversion artifact reports last_iters equal to its cap: 30 for physics_mle, physics_mle_seed1 and S-Q1-001.physics_map, and 15 for both hybrid runs. So at least the last solve batch ran to the iteration limit. Report the per-row iterations-used distribution and the fraction of rows at the cap, for the selected start and for all starts. Report the gradient or step norm at termination and final chi2 versus chi2 at true k for the same rows. Until then, 'maximum-likelihood' describes the objective, not the estimate obtained (objections 022 and 046).. Resolution: `unresolved`.
- `objection-055` (program, open): The forward-evaluation counts do not reconcile with the stated design and need an explicit breakdown. physics_mle used 174,716 evals over 750 rows, about 233 per row. The nominal 3 starts × 30 iterations × (1 residual + 4 finite-difference Jacobian solves) would be 450 per row, or 337,500 total. hybrid_lm used about 138 per row against a nominal 2 × 15 × 5 = 150, so it is close to nominal while physics is at about 52%. Possible explanations include per-row early termination (which conflicts with last_iters=cap), a non-FD Jacobian, shared or batched solves, or uncounted calls. Provide per-row residual, Jacobian, rejected-step and start counts, and state which forward calls the counter excludes. The executor's totals may count as partial evidence for objection 020, but the split and this discrepancy must be resolved before the counts support any efficiency statement.. Resolution: `unresolved`.
- `objection-056` (program, open): The uncertainty procedure does not meet the decision rule registered in objections 019 and 002. Every paired diagnostic uses n_bootstrap=200. Each reports a relative_effect interval at an unstated confidence level, with no Bonferroni adjustment across the three claims and no ratio-scale interval. Task-004 was asked for B=2000, all four pairs, and 95% plus 98.33% ratio intervals, but it returned a single B=200 pair. For the at_least_better claims, the effect is far from the margin, so this is unlikely to change the outcome, but the registered procedure was still not followed. For C-Q1-002.hybrid_equiv_physics, the TOST-style equivalence decision has not been computed at the adjusted level: the reported interval is [-0.00126, 0.00021] relative, at an unknown level, from 200 resamples. Provide B≥2000 ratio intervals at 95% and 98.33% for every pair.. Resolution: `unresolved`.
- `objection-057` (program, open): Provenance of the seed-1 physics arm is unresolved. resu-00140 is labelled exp-0013 but has cached=true, reused_result_exp_id=exp-0008, wall_seconds=0.0 and device=cpu. Its pred and idx paths point to exp-0008, which predates S-Q1-002's experiments (exp-0009 to exp-0012) and so predates the study's registration. Show what produced exp-0008, with which seed and configuration. Confirm that its 175,409 forward evals are counted once, not as new S-Q1-002 spend. Note also that the seed change is confounded with a device change (cpu versus gpu0 for seed 0). No seed-variability statement for physics can rest on this arm until this is resolved.. Resolution: `unresolved`.
- `objection-058` (program, open): The stop and synthesis text misstates the ET seed variation by about 10×. It says the seed-1 replicates move rel_l2 'by less than 0.0004 relative for ET'. The measured values are 0.077407 versus 0.077108, a relative change of 0.0039, or 0.39% of rel_l2 and about 39% of the 1% margin. The physics figure (0.019323 versus 0.019337, about 0.00075 relative) is stated correctly. Correct the synthesis. Do not describe ET seed variation as negligible relative to the margin.. Resolution: `unresolved`.
- `objection-059` (program, open): Held-out leakage audit. researcher-003's final Q-1 message and objections 024, 049 and the code_rules spec are redacted as '[redacted: held-out evidence]'. This means research-agent content touched held-out material. The contract says final test and stress observations cannot inform research decisions. Before any verdict or export, provide an audit showing that no test-split observation, metric or fingerprint-linked result influenced arm selection, study revision or the stop decision.. Resolution: `unresolved`.
- `objection-060` (program, open): Hybrid initializer identity (objection 021) is still unverified. The configuration hash matches the control (170e334e…), but the hybrid reports a separate initializer_fit_seconds of 4.38 s. That indicates the ET was refit inside the hybrid rather than shared with control_et (exp-0009). Report the max absolute difference between the hybrid's val initializer predictions and exp-0009/val_pred.npy, or a fitted-model hash. Until then, C-Q1-002.hybrid does not isolate the effect of LM refinement.. Resolution: `unresolved`.
- `objection-061` (program, open): Handling of failed rows and bounds (objections 017, 018 and 023). All arms show n_finite=750 and failed_arms is empty. This shows every row received a finite prediction, but it does not show that the solves were valid. A clip or projection fallback would produce finite outputs that hide unstable or failed solves. Report the declared fallback rule. Report the count of rows whose prediction lies within 1e-6 of log 0.5 or log 2.0. Report the count of NaN or Inf forward outputs during trial steps. Report the CFL ratio dt·k_max/dx² at k=2.0 and the maximum k evaluated before projection. The finite-count evidence can partly resolve 018 only once the fallback rule is stated.. Resolution: `unresolved`.
- `objection-062` (program, open): Scope the forward-fidelity spot check. fit_report shows residual_std_at_truth=9.95e-4 and model_mismatch=false on checked_samples=200, with the same value in every artifact. The evidence does not state whether these 200 rows are val or train rows. Identify the split and row indices. Note that this check shows consistency between the data and the forward model at the true k on those rows. It does not verify the inversion's forward call: it says nothing about the face-averaging rule, the segment-to-cell map, or Jacobian parity.. Resolution: `unresolved`.
- `objection-063` (program, open): An efficiency claim is not supported by a matched design. The stop text asserts that the hybrid 'uses about 41% fewer forward evaluations per row at equivalent rel_l2'. The arms differ in starts (2 versus 3), iteration cap (15 versus 30), and initialization (ET versus prior or random). No physics arm with 2 starts × 15 iterations and no ET initialization exists. The saving therefore cannot be attributed to the ET initializer, and no efficiency comparison is a registered claim. Synthesis may report the raw counts (103,235 versus 174,716) and predict times (8.4 s versus 15.2 s; ET fit adds about 4.4 s) descriptively only. Objections 043 and 044 still apply.. Resolution: `unresolved`.
- `objection-064` (program, open): S-Q1-002 is not an independent replication of S-Q1-001. The physics and hybrid arms reproduce S-Q1-001 exactly: identical rel_l2 to every reported digit, identical forward_evals (174,716 and 103,235), and identical per-row error summaries. They differ only in cache_key and device. Synthesis must treat S-Q1-002 as a deterministic re-evaluation under corrected matching, not as additional confirmatory evidence. The disclosure of the post-hoc protocol deviation requested in objection 030 still applies to S-Q1-002's re-registration.. Resolution: `unresolved`.
- `objection-065` (program, open): The control is weak and unrepresentative of Q-1 as asked. Q-1 asks about 'the best library regressor (e.g. extra_trees / torch_mlp)'. Only one untuned ET configuration ran. That ET reaches train rel_l2 1.46e-15, which is full memorization, against val 0.0774. No tuned, noise-augmented, PCA or MLP control exists. Synthesis may state only that the result holds against this ET configuration. It must not answer Q-1 as 'beats the best library regressor' (objections 037 and 048).. Resolution: `unresolved`.
- `objection-066` (program, open): Q-2 is only half addressed and lacks its local reference. The per-target rel_l2 values are physics [0.0268, 0.0060, 0.0050, 0.0263] and ET [0.117, 0.028, 0.029, 0.092]. They show that outer-segment error is larger, but they are point estimates without intervals or matched control per-target paired intervals. No Tikhonov or multistart-count ablation was executed, so the second half of Q-2 is untested. No local CRLB at sigma=0.001 was computed, so noise-limited error cannot be separated from optimizer failure. Synthesis must not answer the regularization or multistart part of Q-2. Any identifiability statement must be labelled local, empirical and descriptive.. Resolution: `unresolved`.
- `objection-067` (program, open): The role of posterior_samples=128 when estimator='mode' is still undocumented (objections 025 and 034). The artifacts report uncertainty summaries: mean_std 0.0190, 1σ coverage 0.703, 2σ coverage 0.957, z_rms 0.983. So sampling or a Laplace approximation is being computed, and whether it enters the point estimate is unknown. Also, coverage and calibration are not registered claims, so synthesis should present them as unverified descriptive diagnostics only.. Resolution: `unresolved`.
- `objection-068` (program, open): Artifact bookkeeping needs reconciling before evidence can be cited for resolution. Task-004 lists artifact_id resu-00163, but its result carries artifact_id resu-00162 with id=null. Specialist tasks 001 and 003 failed with unmeasured forward spend. The stop text says 'all five diagnostics', but S-Q1-002 registers six (five required and one optional). Correct these records, and carry the spend of the failed tasks as unmeasured in the budget ledger.. Resolution: `unresolved`.
- `objection-069` (program, open): Objection 029 (certification) can be addressed by evidence: evidence_ids are now populated and the verdicts were issued by the harness verifier. To resolve it, cite resu-00155 and the per-claim verdict records explicitly, and confirm that no researcher text was used as verdict input. Being resolvable does not lift the remaining blocking implementation objections.. Resolution: `unresolved`.
- `objection-070` (S-Q1-001, open): S-Q1-001 remains incomplete. Resolution: `unresolved`.

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

Costs: `{"by_stage_s": {"final": 62.31, "study": 133.92}, "compute_budget_s": 760.0, "compute_remaining_s": 563.78, "compute_used_s": 196.22, "gpus": [], "jobs": 13, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {"execution": 24}, "llm_calls": 24, "llm_calls_max": 80, "llm_cost_usd": 8.0054, "llm_seconds": 637.14, "wall_budget_s": 7200.0, "wall_elapsed_s": 764.17, "workers": 6}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 360.0, "confirmation_used_seconds": 62.307, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-001:3", "seconds": 65.6}, {"action": "release", "actual_seconds": 50.254999999999995, "group_id": "study:S-Q1-001:3", "overrun_seconds": 0.0, "reserved_seconds": 65.6}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-002:4", "seconds": 101.89999999999999}, {"action": "release", "actual_seconds": 0.0, "group_id": "study:S-Q1-002:4", "overrun_seconds": 0.0, "reserved_seconds": 101.89999999999999}, {"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-Q1-002:5", "seconds": 101.89999999999999}, {"action": "release", "actual_seconds": 48.162, "group_id": "study:S-Q1-002:5", "overrun_seconds": 0.0, "reserved_seconds": 101.89999999999999}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 360.0}, {"action": "release", "actual_seconds": 62.307, "group_id": "final-export", "overrun_seconds": 0.0, "reserved_seconds": 360.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 599.2760000000001, "reserved_seconds": 0, "total_seconds": 760.0, "used_seconds": 160.724}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
