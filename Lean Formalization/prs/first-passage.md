Roadmap: MarkovChainsAndRenewal

Add finite-time first-passage identities using the existing kernel, path-law, and
`hittingAfter` interfaces. On the event of first entering a state at time `k`,
the law `n` steps later factors through the kernel power. Partitioning by that
entrance time gives a finite convolution for the time-`n` singleton mass.

This advances [MarkovChainsAndRenewal, Layer 1.3–4](https://github.com/TauCetiProject/TauCetiRoadmap/blob/3090489d1fdfca8af8ae56626a7d44cd51295bf2/TauCetiRoadmap/MarkovChainsAndRenewal/README.md#layer-1-deterministic-and-strong-markov-properties):
event identities for the strong Markov property and interfaces for entrances and
returns. The cutoff is explicit, so `m = 1` covers positive-time first returns
even when the chain starts at the target. Infinite hitting times contribute no
terminal mass. The statements need neither recurrence nor a countable or finite
state space, and accept any finite initial measure.

Validation against the pinned Lean 4.36.0-rc2 / Mathlib revision:

- `lake build TauCeti.Probability.MarkovChain.FirstPassage` passed with repository
  warnings treated as errors.
- The pinned Tools source/import guards and header-style check passed.
- Mathlib's text-style checker passed for the new module and the empty root.
- The pinned Tools `LocalLint.lean` checked five compiled declarations with all
  14 registered environment linters; no findings or `nolint` entries.
- The pinned Tools `LocalAxioms.lean` checked all five compiled declarations;
  no local violations. Transitive/global audits remain for CI.

The Lean checkers were invoked directly on Windows because the Python wrappers
require POSIX-only `fcntl`; the checker sources and repository settings were
not modified.
