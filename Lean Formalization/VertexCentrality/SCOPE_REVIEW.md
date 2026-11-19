# Scope review — 19 November 2026

Reviewed against the 24 blueprint statements and the approved correction register.
This is a semantic review record; kernel checking and axiom inspection are separately
recorded in `handoff/scope-integration.json`. All listed declarations are checked by
`scripts/audit_scope.lean` through the public umbrella module.

The exact mathematical route is implemented: finite weighted graph → stochastic
kernel → Tau Ceti infinite path law → Mathlib positive hitting time → finite-path
renewal/composition expansion → observed heat responses → weighted control and
harmonic adjoint identities → data-derived controls → moments → corrected Euclidean
projection and full-rank uniqueness → generic recovery from one witness.

| Node | Reviewed coverage | Representative declaration |
| --- | --- | --- |
| 1 | Finite weighted graph | `VertexCentrality.WeightedGraph` |
| 2 | Weighted spaces and restriction | `VertexCentrality.WeightedGraph.space_inner` |
| 3 | Laplacian and energy identity | `VertexCentrality.WeightedGraph.laplacian_const` |
| 4 | Stochastic transition and heat evolution | `VertexCentrality.WeightedGraph.transition_nonneg` |
| 5 | Positive-time first passage | `VertexCentrality.WeightedGraph.positivePathLaw_map_prefix` |
| 6 | Finite renewal identity | `VertexCentrality.WeightedGraph.transition_pow_eq_hitting_compositionSum` |
| 7 | Forced heat recurrence and Duhamel sum | `VertexCentrality.forcedHeat_unique` |
| 8 | Response reconstructed from first passages | `VertexCentrality.WeightedGraph.recoveredResponse_hitting_eq_heatSolution` |
| 9 | Unique continuation hypothesis | `VertexCentrality.WeightedGraph.boundary_nonempty_of_uniqueContinuation` |
| 10 | Exact controllability | `VertexCentrality.reachability_surjective` |
| 11 | Blagovescenskii Gram identity | `VertexCentrality.forcedHeat_gram` |
| 12 | Interior harmonic functions | `VertexCentrality.WeightedGraph.harmonicOn_iff` |
| 13 | Harmonic extension and basis | `VertexCentrality.WeightedGraph.existsUnique_harmonicExtension` |
| 14 | Time reversal and response adjoint | `VertexCentrality.WeightedGraph.timeReverse_timeReverse` |
| 15 | Adjoint on harmonic functions | `VertexCentrality.WeightedGraph.heatSolution_harmonic_adjoint` |
| 16 | Gram pseudoinverse exact preimage | `VertexCentrality.moorePenrose` |
| 17 | Data-derived exact controls | `VertexCentrality.apply_gram_moorePenrose_adjoint_of_surjective` |
| 18 | Interior harmonic-product moments | `VertexCentrality.WeightedGraph.gramControl_harmonic_moment` |
| 19 | Harmonic product space and rank | `VertexCentrality.WeightedGraph.harmonicProductSpace` |
| 20 | Euclidean projection reconstruction | `VertexCentrality.WeightedGraph.projection_eq_of_hittingProbability_eq` |
| 21 | Full-rank reconstruction | `VertexCentrality.WeightedGraph.mass_eq_of_hittingProbability_eq_of_rank` |
| 22 | Rational dependence on conductances | `VertexCentrality.WeightedGraph.conductanceDirichletMatrix_det_ne_zero` |
| 23 | Generic full rank from one witness | `VertexCentrality.WeightedGraph.ae_mass_eq_of_hittingProbability_eq` |
| 24 | Algorithm 1 in consistent coordinates | `VertexCentrality.WeightedGraph.exists_data_normal_solution` |

## Statement qualifications

- The graph structures allow empty types where meaningful; connectedness and nonempty
  boundary are explicit for Dirichlet bases. UCP forces a nonempty boundary on a nonempty
  graph (`boundary_nonempty_of_uniqueContinuation`).
- Positive path coordinates start at time one. `pathProcess` prepends the fixed initial
  vertex; `hittingAfter ... 1` therefore counts positive returns, never a time-zero hit.
  Prefix measures equal the normalized finite PMFs, not merely assumed path weights.
- Projection recovery is equality of the counting-measure Euclidean projections for
  models with equal known input. It does not assert equality of the printed weighted
  projections. The product space itself is independent of the unknown vertex masses.
- Algorithm correctness is exact mathematical correctness of recurrence, responses,
  consistent normal equations, moments, and the unique full-rank solve. It does not
  assert executable numerical efficiency or floating-point stability.
- Genericity uses a Gram determinant certificate rather than choosing a minor.
  Polynomial conductance parameterizations permit one variable per undirected edge;
  the theorem does not take an ambient ordered-pair null set and assume it remains null
  on a symmetry subspace. The full-rank witness is supplied explicitly.
- Generic recovery retains admissibility and UCP for both candidate graphs. No generic
  UCP claim, unconditional full-rank claim, or existence of a witness for every graph
  is made. The boundary-pair count alone is not sufficient.
- Imported Tau Ceti and Mathlib closures are checked at the recorded pins; all project
  source is free of `sorry`, `admit`, and added axioms. Historical parent-folder work
  is excluded. Numerical Section 5, Monte Carlo convergence, stability, and complexity
  estimates are outside the approved initial scope.

## Independent ZIP integration review

The supplied ZIP (SHA-256
`34da8cd872fb97523a2b0323701f4b882a5aa4980a21da5b0e599413b719dc46`)
was compared against checkpoint `c08f181` and independently rebuilt on the desktop.
The pinned Mathlib and TauCeti revisions were retrieved without modification.
`lake build` passed all 3746 jobs. `scripts/check_axioms.py` independently checked
all 105 public audit endpoints against the standard-axiom allowlist; Lean LSP
also checked the full-rank hitting-probability recovery endpoint. The project
source scan found no `sorry`, `admit`, or added axioms.

Review traced positive-return event indexing, stochastic path realization,
observed response determination, weighted control transfer, Euclidean product
projection, polynomial Gram rank certificate, and normal-equation correctness.
No additional missing proof premise was found in that route. The final projection
statement is data-invariance/identifiability, combined with exact algebraic control
and moment-solving theorems; this is not executable numerical software. Generic
recovery still requires the explicitly supplied full-rank witness and UCP.

The dashboard tests and both exact coordinate examples passed. Blueprint
PDF/HTML generation passed after escaping plain metadata for LaTeX (the imported
`SCOPE_REVIEW.md` note previously caused `Missing $ inserted`). The detached-HEAD
assumption in the dashboard test was removed for CI. A GitHub Actions workflow
now repeats the build, full namespace axiom audit, endpoint checks, dashboard
checks, and blueprint generation. Historical cloud audit JSON files remain
provenance records, not substitutes for these checks. No coordinator was restarted.

The initial GitHub run passed the Linux Lean build and full namespace/endpoint
axiom audits, but its blueprint job could not authenticate to the private
swarmalize repository. Read-only tooling now uses a public PyYAML parser; its
output was compared with swarmalize and matched all 24 nodes exactly. Optional
coordinator support still uses swarmalize when installed. CI installs only
`requirements-blueprint.txt`, without private repository credentials.
