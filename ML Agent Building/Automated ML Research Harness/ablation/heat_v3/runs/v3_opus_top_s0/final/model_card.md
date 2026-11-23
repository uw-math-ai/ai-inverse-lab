# V3 exported model

Selected methods: custom-solver(fwd).

Export certified: **False**.

Scientific studies complete: 1/6. Passing model accuracy does not certify unfinished scientific claims.

```json
{
  "atol": 1e-10,
  "certified": false,
  "checks": {
    "finite_shape": true,
    "partition": false,
    "reload": true,
    "repeat": true
  },
  "input_role": "development",
  "max_abs_differences": {
    "partition": 0.0010889326404079913,
    "reload": 0.0,
    "repeat": 0.0
  },
  "n_rows": 750,
  "output_shape": [
    750,
    4
  ],
  "passed": false,
  "protocol": "pristine-repeat-reload-partition-v1",
  "reasons": [
    "partition check failed"
  ],
  "rtol": 1e-08,
  "schema_version": 3
}
```

Validation: `0.019322146939217813`. Final metrics: `{}`.

See ../report.md and ../evidence_bundle.json for exact claims, applicability, controls, diagnostics, unresolved objections, provenance and costs.
