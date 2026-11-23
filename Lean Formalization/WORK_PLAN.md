# Work and completion evidence

This plan preserves the full requested objective. Passing a local Lean build is
not completion: accepted upstream contributions, CI, review, and the eventual
roadmap tutorial remain required. `STATUS` records the current PRs and estimate.

## Prerequisite work

| Stage | Existing foundation | Remaining obligation and current boundary |
| --- | --- | --- |
| Finite weighted Hilbert spaces | Mathlib finite inner-product spaces, `L2`, adjoints and projections | Choose a weighted representation and prove restriction/zero-extension formulas. No specific graph-analysis roadmap target located. |
| Weighted graph operator | Mathlib unweighted positive Laplacian | Conductance/mass Laplacian, energy identity and Dirichlet theory; keep the paper's opposite sign explicit. Requires a suitable human-owned roadmap. |
| Markov realization | Tau Ceti PMF/kernel and finite matrix adapters; detailed balance | Specialize the existing APIs to graph transition weights in the downstream construction. Do not submit a duplicate carrier. |
| Positive first passage | Mathlib `hittingAfter`; Tau Ceti strong Markov/path-law APIs | PR #34660 gives entrance-slice factorization and finite-time convolution. It must pass CI, review, and merge. |
| First-passage data to heat kernel | Finite-time convolution from the preceding stage | Establish finite-data recursive determination, including diagonal positive returns. Check scope after the entrance API review; do not open a dependent PR first. |
| Forced discrete evolution | Finite sums, operator powers, linear maps | Global recurrence, finite Duhamel sum, finite observation/data dependence. No graph heat/control target located. |
| Control/response Gram identity | Existing adjoints and symmetric operators | Define the operators and prove the finite anti-diagonal identity with weighted pairings. No boundary-control target located. |
| Unique continuation to controllability | Mathlib Cayley–Hamilton and symmetric spectral theory | Restrict to the unobservable subspace and prove exact controllability. Do not assume a cited proposition as an axiom. No explicit control target located. |
| Harmonic response pairing | Existing projections, adjoints and finite time-index equivalences | Graph-specific telescoping and time reversal; explicit temporal extension. Requires the missing graph/control interfaces. |
| Gram normal-equation recovery | Tau Ceti canonical Moore–Penrose inverse; Mathlib Gram-kernel identity | Checked candidate in `candidates/`. Independent review found its new capability is not authorized by a specific unfinished existing target. Not submitted. Minimum norm remains a separate obligation. |
| Harmonic moments and reconstruction | Finite-dimensional spans, rank and least squares | Correct the counting-measure versus mass-weighted projection convention; preserve boundary mass factors in coordinates. No existing inverse-graph target located. |
| Generic full rank | Existing polynomial-zero-locus null theorem | Rational Dirichlet coordinates and one nonzero minor numerator; intersect admissibility afterward. No existing graph inverse-problem target located. |

The proof audit gives the exact source revisions, declarations, counterexamples,
and edge cases. This table is a work ledger, not a new Tau Ceti roadmap or an
authorization to add off-roadmap mathematics.

## Submission and infrastructure work

1. Shepherd #34660. Inspect the current head, content build, review threads, and
   merge state. Address actual findings and rerun checks affected by each edit.
2. Repair its infrastructure obstacle through the existing upstream fix:
   TauCetiTools #210 already removes the 2,000-record open-PR ceiling while
   preserving request/byte/time budgets. Check for a consuming-repository update
   before proposing a minimal workflow pin refresh. Such a PR is independent of
   the mathematics and must retain the human-review gate.
3. Continue useful review work in proportion to contributions. One separately
   Lean-checked source/reuse review was posted on #34367. Official TauCetiReview
   could not start on this Windows host because its coordination code imports
   `fcntl`; do not fabricate official rubric verdicts.
4. Keep dependent work local until its prerequisite is merged. Independent PRs
   must start from upstream main and contain no other unmerged PR's content.
5. Keep all task source, checkouts, caches, credentials used by local tools,
   reports, and patches inside `Lean Formalization/`; do not alter unrelated work.

## Completion checks

- [x] Read the paper and record mathematical dependencies and proof issues.
- [x] Inspect pinned Mathlib, Tau Ceti, and relevant authoritative roadmaps.
- [x] Implement and locally validate an eligible preliminary contribution.
- [ ] Obtain successful remote content CI and resolve its reported failures.
- [ ] Address all upstream review findings and verify merge of eligible PRs.
- [ ] Complete subsequent eligible prerequisites without stacking PRs.
- [x] Record opened/landed PRs and a central estimate in `STATUS`.
- [x] Contribute a concrete, checked review suggestion on another PR.
- [ ] Recheck that no required prerequisite supported by an existing roadmap
      remains unimplemented, unreviewed, or unmerged.
- [ ] At that point, deliver the requested tutorial on human-owned roadmap
      contributions, grounded in current `CONTRIBUTING.md` and README guidance.

The goal remains active until these conditions are verified or an external
blocker meets the prescribed blocked-audit threshold. No upstream merge, roadmap
exhaustion, or complete formalization is currently claimed.
