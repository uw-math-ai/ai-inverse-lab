Roadmap: OperatorTheory

Local draft only: NOT SUBMITTED. Independent scope review found no specific
unfinished target authorizing this new capability. Preserve the checked candidate
pending a valid target; do not use the draft rationale below as scope approval.

Extend the existing Moore–Penrose API with exact recovery from consistent normal
equations: if `y` lies in the range of `A`, then
`A ((A† A)⁺ (A† y)) = y`. This permits a singular Gram operator and works over
`RCLike` fields without requiring `A` to be surjective.

This is a bounded improvement of the already implemented
[`OperatorTheory/PolarDecomposition` Moore–Penrose interface](https://github.com/TauCetiProject/TauCetiRoadmap/blob/3090489d1fdfca8af8ae56626a7d44cd51295bf2/TauCetiRoadmap/OperatorTheory/PolarDecomposition/README.md#the-moorepenrose-interface),
under the repository's existing-code improvement policy. PD-C19–PD-C27 are
already implemented; this does not claim a missing named roadmap target. The
proof combines the existing Penrose equation with Mathlib's
`LinearMap.ker_adjoint_comp_self`, without new definitions or imports.

Validation: the isolated proof candidate compiled with the pinned dependency;
the integrated module build and scoped checks are pending.
