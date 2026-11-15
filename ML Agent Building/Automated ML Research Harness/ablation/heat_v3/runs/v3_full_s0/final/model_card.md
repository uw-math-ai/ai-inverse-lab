# V3 exported model

Selected methods: physics-MMSE[log,ms=3,it=25].

Export certified: **True**.

Scientific studies complete: 4/6. Passing model accuracy does not certify unfinished scientific claims.

```json
{
  "atol": 1e-10,
  "certified": true,
  "checks": {
    "finite_shape": true,
    "partition": true,
    "reload": true,
    "repeat": true
  },
  "input_role": "development",
  "max_abs_differences": {
    "partition": 0.0,
    "reload": 0.0,
    "repeat": 0.0
  },
  "n_rows": 750,
  "output_shape": [
    750,
    4
  ],
  "passed": true,
  "protocol": "pristine-repeat-reload-partition-v1",
  "reasons": [],
  "rtol": 1e-08,
  "schema_version": 3
}
```

Validation: `0.019046710309041506`. Final metrics: `{"r2": 0.996989327623733, "rel_l2": 0.018334935911737203, "rmse": 0.0241251819928364}`.

See ../report.md and ../evidence_bundle.json for exact claims, applicability, controls, diagnostics, unresolved objections, provenance and costs.
