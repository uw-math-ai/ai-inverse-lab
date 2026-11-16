# Internal review: first-passage decompositions

Date: 2026-11-09. Private review; nothing posted externally.

Reviewed `TauCeti/Probability/MarkovChain/FirstPassage.lean`, the two-declaration version ending at line 140, against the pinned Mathlib/Tau Ceti sources and TauCetiReview rubrics for scope, correctness, reuse, generality, API, proof quality, documentation, naming, and placement. This is independent source review, not the TauCetiReview command-line pipeline and not a compilation certificate.

## Findings from the initial reviewed version

Follow-up source inspection confirmed that both findings below have been resolved: the roadmap/layer paragraph was removed from the module docstring and the `change` line was deleted. No remaining source-review request is identified. Compilation and full repository checks remain outstanding.

1. **Documentation — roadmap/process text in module documentation** (`FirstPassage.lean`, opening docstring, lines 18–20). `rubrics/documentation.md` explicitly excludes roadmap targets and layer numbers from module docstrings. Remove the reference to the Markov chains and renewal roadmap and Layers 1.3–4/2.2. Preserve the mathematical explanation of finite-time renewal decomposition. Put the exact roadmap attribution and target argument in the PR description.

2. **Proof quality — undocumented `change`** (approximately line 82). `rubrics/proof-quality.md` asks for an explanation of necessary definitional manipulation. The `change` only folds the local abbreviation `tau` into the event; either remove it if rewriting finds the event definition directly, use an explicit conversion, or add a short comment explaining that local folding makes the event equality usable. The surrounding restriction/map calculation is the standard strong Markov proof shape used by existing files; it is otherwise coherent.

## Scope assessment and required PR description

Canonical attribution: `Roadmap: MarkovChainsAndRenewal`.

The best exact source is `TauCetiRoadmap/MarkovChainsAndRenewal/README.md`, **Layer 1, item 3**, requiring event identities for the strong Markov property, together with **Layer 1, item 4**, requiring entrances/returns and stopping-time/measurability interfaces. The first theorem realizes the event identity at the entrance of a singleton, with its entrance time pinned to `k` and an arbitrary measurable future-state event. The second theorem is the finite partition over those entrance events. This is a coherent, standard first-passage API, directly adjacent to those requirements; no graph inverse-problem scope extension is needed.

Do **not** say this is a missing necessary prerequisite for the already implemented Layer 2, item 2 Green-function identity. `Probability/MarkovChain/Visits.lean` already proves that target via successive visits and the geometric sum, independently of the new finite-time convolution. The new result gives genuinely different **finite-time** information, but citing the existing Green identity alone would leave a weak satellite-work justification under `scope.md`. State the direct Layer 1 event/entrance target and precise finite-time capability. No dedicated finite-convolution milestone was located; the fit is the stopped-event/entrance interface, not a claim that the roadmap explicitly names this formula.

Suggested scope explanation: “This provides the entrance/return specialization of the strong Markov event identities required by Layer 1.3–4 of `TauCetiRoadmap/MarkovChainsAndRenewal/README.md`. It factors a fixed-time future-state event on an entrance-time slice and sums the disjoint slices to obtain the first-passage transition convolution, retaining the infinity and positive-return conventions.”

The patch is one topic. No stacking issue appears in the local single-file patch. This assessment does not authorize the separate weighted-graph/control stages of the paper.

## Correctness assessment

No semantic defect found in either theorem.

- `hittingAfter ... {y} m` searches at or **after** `m`. Specializing `m=1` matches the paper's positive first passage even when the initial state is `y`; specializing `m=0` includes initial entrance.
- The helper factors the event `tau=k AND X_(k+n) in s`. On `tau=k`, the stopped state is `y`; the future-state marginal is `(kappa^n) y s`. The factorization uses the existing strong Markov law rather than assuming its conclusion.
- No requirement `m<=k` is missing: when `k<m` the entrance slice is empty and both sides vanish. The helper is valid for future lag `n=0` and for zero initial measure.
- In the convolution theorem, `m<=n` is necessary and present. A path at `y` at time `n` has an attained entrance at some `k` in `[m,n]`; distinct entrance slices are disjoint. Paths whose entrance is infinity cannot contribute. Terms with `k>n` are excluded by `Icc`, so truncated natural subtraction causes no hidden time-index error.
- For `m=n`, the formula reduces to the identity for the single entrance event at `n`, with `(kappa^0) y {y}=1`. For `m=n=0` it includes initial mass. For a one-state identity chain the unique nonzero slice is `k=m`.
- Arbitrary finite initial measures are appropriate: the path-law slice carries their total mass, while the future kernel remains normalized. The result does not mistakenly square the initial mass.
- Countability, recurrence, almost-sure finite hitting, irreducibility, and finiteness of the state space are unnecessary and correctly absent. Measurable singletons are required for the singleton entrance time and are present. General future event `s` is explicitly measurable.

These were mathematical and source-level checks. Subsequent validation of commit `c325d5de1d` passed: the module build with warnings as errors, trusted source/import guards, header style, all 14 scoped environment linters (five compiled declarations, zero findings), and the local compiled-declaration axiom/module audit. The source was updated for pinned `untopA`, cast-order, and deprecation APIs without changing the mathematical statements. PR #34660 is open; full remote CI and external review remain pending.

## Reuse, API, and generality assessment

Searches of pinned Mathlib's probability sources and Tau Ceti's probability sources for first-passage/entrance decompositions, hitting-time sums, convolution identities, and the distinctive stopped-shift law found no direct replacement for either new statement. The nearby `Visits.lean` Green identities sum over **all** times and do not replace the finite-time convolution.

The proof correctly consumes `Measure.markovChainLaw_restrict_map_stoppedValue_shift_iterate_eq_smul`, Mathlib `hittingAfter`, stopped values, finite disjoint-union additivity, and the existing path-law marginal theorem. It does not construct a second Markov-chain or hitting-time carrier. The helper is a useful exported event identity and is consumed by the main result. A public extra `m=1` wrapper is unnecessary because the main theorem specializes directly.

The namespace is consistent with nearby path-law results: the first explicit argument is the initial `Measure`. Placement in `Probability/MarkovChain/FirstPassage.lean` and the single focused strong-Markov import are reasonable. Names describe the event factorization and finite sum. No gratuitous notation, compatibility alias, custom definition, or excessive assumptions were introduced.

The two proofs are readable stopping-time and finite-partition arguments; neither hides a new standalone general-purpose construction that clearly needs its own module. The documentation/process and `change` findings were fixed before submission. This internal review does not replace official upstream rubric review.
