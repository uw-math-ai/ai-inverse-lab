:robot: Additional source review using Codex at head `c35572b5b104da58ecc1732742e851381bc497aa`.

For the new infinite-visits measurability proof, Mathlib already provides
`MeasurableSet.setOfPred_infinite` and `Measurable.of_mem`. Pulling back along the
visit-set map avoids rebuilding this through successive visits:

```lean
  MeasurableSet.setOfPred_infinite.preimage
    (Measurable.of_mem fun n => by
      simpa only [Set.mem_ofPred_eq] using
        (measurable_pi_apply n : Measurable fun w : ℕ → α => w n).eq_const y)
```

I checked this expression in the expected type
`MeasurableSet {w : ℕ → α | {n | w n = y}.Infinite}` with the pinned Mathlib
`1b1ccf4b088b32fa3f23f2de1127433254a81e1e`; Lean compiled it without warnings.
This can shorten the helper or be used directly at its consumers, without
introducing another public process-level declaration.

This is a narrow source/reuse review with a separately checked replacement
expression, not an independent build of the whole PR or a TauCetiReview rubric
verdict. The official CLI could not run on this Windows host (`fcntl` import).
