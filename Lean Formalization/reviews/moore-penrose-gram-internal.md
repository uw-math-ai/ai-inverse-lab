# Internal review: consistent Gram normal equations

Reviewed 2026-11-09 against the checked-out TauCetiReview rubrics and
`TauCeti/AGENTS.md`. Candidate:
`LinearMap.apply_gram_moorePenroseInverse_adjoint_of_mem_range` in
`TauCeti/Analysis/InnerProductSpace/MoorePenrose.lean`, together with
`prs/moore-penrose-gram.md`. This is an independent local review, not an upstream
review verdict. The root agent owns compilation and submission.

## Finding: scope justification needs resolution before submission

**Scope: block under the strict new-capability test.** The PR description calls
the new theorem an existing-code improvement and explicitly says that no missing
named roadmap target is claimed. However, `rubrics/scope.md` exempts refactoring,
re-proving, modestly generalizing, or documenting mathematics already present on
main, and explicitly treats a new theorem that adds a capability as new
mathematical content. This change adds exact recovery by the Gram pseudoinverse;
it does not replace or generalize an existing declaration in the diff.

The cited `OperatorTheory/PolarDecomposition/README.md`, heading
`The Moore–Penrose interface`, specifies PD-C19–PD-C27: the Penrose relation,
canonical construction, verification, uniqueness, characterization, adjoint
compatibility, and injective/surjective/invertible specializations. These are
already implemented. None states the Gram normal-equation result. Merely
combining already-implemented PD-C21 with a Mathlib kernel theorem does not make
a specific remaining target closer. The scope rubric expressly asks for
advancement rather than membership in a mathematical topic.

Resolve this by identifying a real, proximate existing target that needs the
result, or by giving an actual existing-statement generalization rather than
only an API-extension description. Otherwise retain the candidate locally for
the paper-specific project and do not submit it under a completed target.
This is a conservative interpretation of the explicit rubric; there is no
mathematical correctness objection.

## Remaining review angles

- **Correctness: approve.** For `y = A x`, Penrose gives
  `A†A ((A†A)⁺ (A†A x)) = A†A x`. The difference from `x` lies in
  `ker (A†A) = ker A`, hence its image under `A` vanishes. The claim handles zero
  maps, zero-dimensional spaces, nontrivial kernels, rectangular maps, and real
  and complex scalars. Consistency is appropriately explicit as range membership.
- **Reuse: approve.** Searched TauCeti for `moorePenroseInverse`, `pseudoInverse`,
  normal equations and Gram inverses; inspected `MoorePenrose`,
  `Symmetric/Range`, `CovariancePseudoInverse`, and `ConstantRank`. No located
  theorem directly replaces this statement. The algebraic semisimple-module
  pseudo-inverse is an existence result, not this canonical Gram construction.
  Pinned Mathlib has no Moore–Penrose development; its
  `LinearMap.ker_adjoint_comp_self` is reused directly.
- **API, generality, naming, placement: approve.** One public theorem extends
  the relevant `LinearMap` API in its canonical file, with the finite-dimensional
  assumptions needed by the existing `LinearMap` adjoint and inverse interfaces.
  No compatibility alias, redundant import, or alternate construction is added.
- **Documentation, attribution, proof quality: approve.** The docstring accurately
  describes exact recovery without ambient invertibility. The short proof uses
  named Penrose and kernel facts and introduces no unchecked hypotheses or
  complicated unfolding. Existing file references cover the conventional result.

The PR body's validation paragraph is intentionally provisional and should be
updated with the integrated build and scoped-check results before publication.
