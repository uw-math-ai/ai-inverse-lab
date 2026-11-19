# Vertex centrality reconstruction

Lean formalization of the corrected exact reconstruction results in Gao–Li–Yang,
*Vertex Centrality Reconstruction in an Inverse Problem for Information Diffusion*,
arXiv:2603.20710v1, Sections 1–4 and Appendix A.

The library now connects infinite-path positive hitting-time probabilities to
response recovery, weighted boundary controls, harmonic moments, Euclidean
projection recovery, full-rank mass uniqueness, and generic recovery from one
full-rank witness. The [correction register](CORRECTIONS.md) records the projection
convention, missing boundary mass factor, and genericity restrictions.
See [SCOPE_REVIEW.md](SCOPE_REVIEW.md) for the statement-to-declaration review and
[HANDOFF.md](HANDOFF.md) for historical checkpoints.

## Build and audit

From this directory, with elan installed:

```sh
lake exe cache get
lake build
lake env lean scripts/audit_scope.lean
```

`VertexCentrality.lean` publicly re-exports the library. The scope audit checks all
blueprint-linked declarations plus the integration endpoints (105 declarations).
The recorded build and axiom results are in `handoff/scope-integration.json`.
A successful build alone is not an axiom audit; the latter checks for `sorryAx`
and unexpected axioms. The accepted foundational axioms are `propext`,
`Classical.choice`, and `Quot.sound`.

## Library foundations

- Lean 4.35.0; Mathlib `56a005855ea17e0b0958f73ec801ae140b0a1679` (v4.35.0).
- Tau Ceti `5a01f5fe3621126cccf9ba22e700493cbc7b7156`.
- Direct Tau Ceti uses: the canonical Moore–Penrose inverse, homogeneous
  Markov-chain path law, and nullity of nonzero polynomial zero loci.
- Mathlib supplies weighted inner-product machinery, finite-dimensional spectral
  and matrix results, PMFs, hitting times, polynomials, and measure theory.

The root Mathlib pin overrides Tau Ceti's historical Mathlib pin. The imported
dependency closure is checked with both dependency working trees pristine; no
claim is made that every module in that Tau Ceti revision builds with this override.
Historical candidates outside this Lake project are not included in its proof claims.

## Main endpoints

In `VertexCentrality.WeightedGraph`:

- `positivePathLaw_hittingAfter`: infinite-path and finite-path probabilities agree.
- `transition_pow_eq_hitting_compositionSum`: the visit-gap expansion.
- `recoveredResponse_hitting_eq_heatSolution`: explicit boundary response recovery.
- `projection_eq_of_hittingProbability_eq`: corrected Euclidean projection recovery.
- `mass_eq_of_hittingProbability_eq_of_rank`: full-rank mass uniqueness.
- `ae_mass_eq_of_hittingProbability_eq`: generic recovery from one witness.
- `reconstruction_controls_exist_unique_mass`: consistent normal equations and
  exact full-rank moment solving for the corrected algorithm.

Admissibility, unique continuation, the observation horizon, known conductances
and boundary masses, and the full-rank witness remain explicit where required.
The dimension count is only necessary for full rank. First passage starts at
time one, including first returns. Numerical experiments, Monte Carlo convergence,
floating-point stability, and Remark 2.2 complexity estimates are excluded.

## Blueprint and dashboard

`blueprint/nodes/*.tex` contains the mathematical statements, declaration links,
and progress metadata. `proof_revision` records the checked code revision;
statuses are review metadata, not independent certificates. The generated
blueprint assets must be rebuilt after changing nodes:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-blueprint.txt
.venv/bin/python scripts/build_blueprint.py
.venv/bin/python scripts/check.py
.venv/bin/python scripts/test_dashboard.py
.venv/bin/python scripts/serve.py --port 8766
```

The dashboard binds to loopback and reads source-bound build evidence from ignored
`.local/checks.json`; it does not independently certify proofs. The ZIP integration
review rebuilt the PDF/HTML and passed the dashboard tests. No dashboard or coordinator
is currently running.

## Continuous integration

The `Vertex Centrality` GitHub Actions workflow builds this nested project, audits
all declarations under the `VertexCentrality` namespace, checks all 105 blueprint
audit endpoints, runs the exact coordinate examples, tests the dashboard, and
rebuilds the blueprint. `scripts/check_axioms.py` fails on missing audit results or
axioms outside `propext`, `Classical.choice`, and `Quot.sound`.

This is exact mathematical reconstruction, with the approved corrections and explicit
hypotheses. It is not a verified floating-point implementation of Algorithm 1.

Read-only dashboard and blueprint rendering use public dependencies in
`requirements-blueprint.txt`. Coordinator users may additionally install
`requirements-tools.txt` with access to the pinned swarmalize repository. CI does
not require its credentials; node files remain compatible with swarmalize.
