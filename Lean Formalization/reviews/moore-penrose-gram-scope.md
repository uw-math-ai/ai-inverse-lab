# Scope and reuse assessment: Gram pseudoinverse recovery

Date: 2026-11-09. Isolated candidate: `.tools/MoorePenroseGram.lean`.

The candidate proves, for a linear map between finite-dimensional real or complex inner-product spaces and `y` in its range,

```text
A ((A* A)^+ (A* y)) = y.
```

This includes singular Gram operators and rectangular maps. It supplies the exact-control property of the paper's Proposition 3.1 once `W` is known surjective. It does not itself prove the minimum-norm assertion or graph controllability, and should not be described as doing either.

## Existing mathematics consumed

Tau Ceti `Analysis/InnerProductSpace/MoorePenrose.lean` already constructs the canonical inverse, proves the four Penrose equations, uniqueness, adjoint compatibility, and injective/surjective specializations. Mathlib `Analysis/InnerProductSpace/Adjoint.lean` already proves `LinearMap.ker_adjoint_comp_self`. The candidate combines these: the Penrose equation for the Gram map puts the normal-equation residual in its kernel, and the Mathlib kernel equality makes that residual invisible under `A`.

Searches covered all `moorePenroseInverse`/`IsMoorePenroseInverse` uses in Tau Ceti, the surrounding inner-product-space and linear-algebra APIs, and specifically `Analysis/Matrix/CovariancePseudoInverse.lean` and `Analysis/InnerProductSpace/Symmetric/Range.lean`. The covariance file has covariance projection identities and the bridge to the same inverse; the symmetric-range file inverts a symmetric map on its range. Neither exposes the rectangular Gram recovery formula. Mathlib already has `range_adjoint_comp_self` and normal-equation least-squares characterizations, but none of these directly replaces the canonical inverse expression in this candidate. The underlying Penrose inverse must be reused; no alternate construction is warranted.

## Scope assessment

**Resolution after independent review:** the proposed API-improvement classification
below is not sufficient under the stricter new-capability test in `rubrics/scope.md`.
The candidate adds a capability without generalizing an existing statement, and no
specific unfinished target was found. It is retained locally in `candidates/`, not
submitted upstream. See `moore-penrose-gram-internal.md`. The earlier reasoning below
is preserved to explain the disputed classification, not as authorization.

`TauCetiRoadmap/OperatorTheory/PolarDecomposition/README.md`, PD-C19–PD-C27, specifies the Penrose relation, finite-dimensional construction, uniqueness, adjoint compatibility, and injective/surjective/invertible cases. Those named targets already have implementations. **The roadmap does not explicitly name the Gram recovery formula or minimum-norm controls.** Do not present this as a missing PD-C milestone or infer permission for graph/control theory from it.

The defensible scope claim is a small improvement to the existing Moore–Penrose API: it stays entirely within the already merged finite-dimensional inverse theory, uses the same map and hypotheses, adds no new carrier, introduces no graph/control vocabulary, and derives the standard consistent-normal-equation consequence from the existing Penrose theorem. This is comparable to the existing injective/surjective consequences and supports reuse of the canonical inverse. Under `AGENTS.md`'s policy that modest improvements of existing mathematics do not need fresh roadmap scope, this is a reasonable bounded API improvement.

That classification is an assessment, not an invented explicit roadmap target. The candidate is a new theorem, rather than a purely mechanical refactor, so its PR should explain the bounded existing-API improvement honestly. If a scope reviewer rejects that classification, do not manufacture a dependency on PD-C19–PD-C27 or on the paper. Contest only with concrete existing-code evidence, and follow the repository's bounded repair/closure policy if disagreement persists.

Suggested attribution is `Roadmap: OperatorTheory`, with prose explaining that this improves the already implemented Moore–Penrose interface chiefly motivated by PolarDecomposition PD-C19–PD-C27. The roadmap is attribution, not a claim that the paper's inverse reconstruction is already authorized.

## Candidate and validation limits

The isolated proof and temporary library integration both compiled against the pinned revisions without warnings. The integrated file also passed all 14 scoped environment linters (55 compiled declarations, no findings) and the local compiled-declaration audit. The parent preserved the standalone candidate and integration patch in `candidates/`, then removed the temporary library edit. No upstream PR, push, or merge was made for this candidate. It proves an exact preimage property only; minimum-norm uniqueness would require an additional theorem or identifying `(A* A)^+ A*` with `A^+` and proving its minimum-norm property.
