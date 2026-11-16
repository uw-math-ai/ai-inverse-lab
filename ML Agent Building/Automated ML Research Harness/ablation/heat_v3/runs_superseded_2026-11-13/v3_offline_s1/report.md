# V3 research product: heat_v3_offline_s1_20261113_171736

Given noisy temperature measurements at four sensor locations over time, recover the four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation.

Program stop: registered work resolved or explicitly deferred; no justified next study

Model export certified: **True**. Studies complete: **1/1**.

Model performance and study completion are separate outcomes.

## Actual exported method

- physics-MLE[log,ms=1,it=30] (kind `physics`, method hash `74dff9ec37c0`, source hash `None`, forward access `True`).
  Observed fit diagnostics: `{"fit_report": {"checked_samples": 200, "model_mismatch": false, "nominal_noise_std": 0.001, "residual_std_at_truth": 0.0009994548867275769}}`

Validation selection value: `0.019335308875384715`. Final metrics: `{"r2": 0.9969618386959722, "rel_l2": 0.01841886806236584, "rmse": 0.024235620252272574}`.

Export certificate: `{"atol": 1e-10, "certified": true, "checks": {"finite_shape": true, "partition": true, "reload": true, "repeat": true}, "input_role": "development", "max_abs_differences": {"partition": 0.0, "reload": 0.0, "repeat": 0.0}, "n_rows": 750, "output_shape": [750, 4], "passed": true, "protocol": "pristine-repeat-reload-partition-v1", "reasons": [], "rtol": 1e-08, "schema_version": 3}`.

Lineage: `{"idea-003": {"arm_id": "S-1.physics", "baseline_identity": true, "gate": "gate-00039", "label": "physics-MLE[log,ms=1,it=30]", "origin": "baseline", "study_id": "S-1"}}`.

## Registered studies and exact claim outcomes

| Study | Required work | State | Reason |
|---|---:|---|---|
| S-1 | 3/3 | complete | all registered required work completed |

### S-1: Which validated method improves on frozen conventional controls?

Owner: `researcher-003`; registered evidence: `['resu-00046', 'resu-00050', 'resu-00048']`.

- Arm `S-1.control` (baseline, required=True): `ok`; method `ridge/raw`; metrics `{"r2": 0.9565820640182872, "rel_l2": 0.06864850713267057, "rmse": 0.09052927609998045}`; effective hash `968c77ed472593a1999d9d08d2f22447840eecd9dcc7bab194debf78a0e17737`.
  Normalization changes: `[{"effective": 0, "effective_present": true, "field": "noise_aug", "requested": null, "requested_present": false}, {"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": null, "requested_present": false}, {"effective": "none", "effective_present": true, "field": "target_transform", "requested": null, "requested_present": false}]`.
- Arm `S-1.treatment` (treatment, required=True): `ok`; method `extra_trees/raw`; metrics `{"r2": 0.9469883455915696, "rel_l2": 0.07584817000093681, "rmse": 0.1000237326417435}`; effective hash `8b17af31216305d6bbdfbc4fe0b2d295ad2511f1be58e5794b6f88174dbde58f`.
  Normalization changes: `[{"effective": 0, "effective_present": true, "field": "noise_aug", "requested": null, "requested_present": false}, {"effective": 1.0, "effective_present": true, "field": "noise_scale", "requested": null, "requested_present": false}, {"effective": "none", "effective_present": true, "field": "target_transform", "requested": null, "requested_present": false}]`.
- Arm `S-1.physics` (baseline, required=True): `ok`; method `physics-MLE[log,ms=1,it=30]`; metrics `{"r2": 0.9965566885417642, "rel_l2": 0.019335308875384715, "rmse": 0.02549817306697214}`; effective hash `08c8e0bba677e0b0dc6d70ddd07271e00f43f3c9177d3b2885b783cda17f0a03`.
  Normalization changes: `[]`.

Diagnostics and absences:

- `S-1.paired`: `ok`; artifact `resu-00052`; reason `registered computation returned`.

Claim `C-1`: **contradicted**. registered predicate not met; this does not establish absence of an effect.
Predicate: `{"control": "S-1.control", "margin": 0.01, "metric": "rel_l2", "operator": "at_least_better", "treatment": "S-1.treatment"}`; values `{"control": 0.06864850713267057, "metric": "rel_l2", "treatment": 0.07584817000093681}`; effect `-0.1048771949891369`; interval `None`.
Evidence `['resu-00050', 'resu-00046']`; unmet requirements `[]`.

## Independent critic and uncertainty


test_bootstrap: `{"ci95": [0.0174123046407792, 0.01963229174281298], "mean": 0.018493489118195608, "n_boot": 200, "std": 0.0005562121605088046}`.

test_per_target_rel_l2: `[0.0269799001673662, 0.006088333683040254, 0.005218525678847445, 0.02419188207713548]`.

test_uncertainty: `{"coverage_1sigma": 0.701, "coverage_2sigma": 0.956, "mean_std": 0.018403128846459282, "z_rms": 0.98157672398061}`.

shift: `{"k_low_0.3": {"r2": 0.8926730245849349, "rel_l2": 0.12494414929093481, "rmse": 0.15560486806487978}, "noise_x3": {"r2": 0.97642188876497, "rel_l2": 0.0520294521303912, "rmse": 0.0693233392838876}}`.

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

Agent instances: `4`; specialist tasks: `0`.

Fallback reasons (role calls): `{"offline_choice": 13}`.

Costs: `{"by_stage_s": {"final": 37.35, "study": 28.48}, "compute_budget_s": 400.0, "compute_remaining_s": 334.17, "compute_used_s": 65.83, "gpus": [], "jobs": 4, "literature_failures": 0, "literature_queries": 0, "llm_by_phase": {}, "llm_calls": 0, "llm_calls_max": 80, "llm_cost_usd": 0.0, "llm_seconds": 0.0, "wall_budget_s": 7200.0, "wall_elapsed_s": 49.4, "workers": 6}`.

Reservation policy: `{"active_groups": {}, "confirmation_seconds": 100.0, "confirmation_used_seconds": 37.352, "hard": false, "history": [{"action": "reserve", "admitted": true, "confirmation": false, "group_id": "study:S-1:5", "seconds": 32.4}, {"action": "release", "actual_seconds": 28.479, "group_id": "study:S-1:5", "overrun_seconds": 0.0, "reserved_seconds": 32.4}, {"action": "reserve", "admitted": true, "confirmation": true, "group_id": "final-export", "seconds": 100.0}, {"action": "release", "actual_seconds": 37.352, "group_id": "final-export", "overrun_seconds": 0.0, "reserved_seconds": 100.0}], "overrun_seconds": 0.0, "policy": "soft_estimates", "remaining_seconds": 334.169, "reserved_seconds": 0, "total_seconds": 400.0, "used_seconds": 65.83099999999999}`.

## Limits

- No discovery or transfer campaign was run during development.
- Shared fitted-state/trajectory adapters are unsupported; immutable prediction reuse is supported.
- The default local Fisher diagnostic is a local prior-sample calculation, not a universal Bayes floor.
- Specialist callbacks have observed soft deadlines; execution/retrieval adapters enforce their own timeouts. Arbitrary in-process callbacks cannot be forcibly preempted.

Evidence bundle: `evidence_bundle.json`. Program trace: `research_trace_v3.json`. Evaluation readiness metrics: `program_evaluation.json`.
