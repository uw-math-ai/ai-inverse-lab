# Vertex centrality reconstruction: proof and library audit

Date: 2026-11-09, America/Los_Angeles.

Source: the local 20-page `2603.20710v1.pdf`, *Vertex centrality reconstruction in an inverse problem for information diffusion*, Gao–Li–Yang. Page and equation numbers below refer to that version. This is an audit and implementation plan, not a claim that the paper or its prerequisites have been formalized.

## Revisions inspected

| Source | Exact revision |
| --- | --- |
| Tau Ceti | `1e668f5e55bc20f2eb87f4ba49b48833470290d1` |
| TauCetiRoadmap | `3090489d1fdfca8af8ae56626a7d44cd51295bf2` |
| Mathlib, as pinned by Tau Ceti | `1b1ccf4b088b32fa3f23f2de1127433254a81e1e` |

Paths beginning `Mathlib/` are relative to `TauCeti/.lake/packages/mathlib/`; paths beginning `TauCeti/` below are library paths relative to the Tau Ceti checkout. Roadmap paths are relative to the TauCetiRoadmap checkout. Sources were inspected directly, including the authoritative roadmap READMEs. Generated roadmap STATUS files are useful historical clues and are not current completion certificates.

## Mathematical dependency graph

Use a finite nonempty vertex type `X`, a finite simple undirected graph, positive vertex masses `mu`, and symmetric nonnegative off-diagonal conductances `w`, positive precisely on edges. Write `M = diagonal mu`, `L = D_w - W`, `Delta = -M^-1 L`, and `P = I + Delta`. Keeping these signs explicit prevents confusing the paper's nonpositive Laplacian with Mathlib's nonnegative graph Laplacian.

1. **Finite weighted analysis.** Construct the weighted inner product `sum_x mu_x f_x g_x`; restriction to `B`; zero extension from `B`; their adjoint relationship; finite space-time products and time reversal. Positivity of `mu` is essential. Reuse finite inner-product spaces and orthogonal projection, rather than inventing an independent Hilbert-space theory.
2. **Weighted graph operator.** Prove linearity, constants in the kernel, weighted symmetry, and the energy identity for `Delta`. The energy identity yields the maximum principle and Dirichlet uniqueness when `B` meets every connected component. On a connected nonempty graph it suffices that `B` is nonempty. Prove existence by finite-dimensional injectivity/surjectivity or positivity of the interior Dirichlet block. Harmonic extension is independent of `mu`.
3. **Markov realization.** Under `mu_x >= sum_y w_xy`, prove that `P` is row stochastic, build its existing kernel adapter, and establish detailed balance `mu_x P_xy = mu_y P_yx`. Identify kernel powers with the homogeneous heat equation `u(t) = P^t g`. No aperiodicity is needed.
4. **Positive-time first passage.** Define the paper's `tau(x,y)` using `hittingAfter` with starting index **one**, even if `x != y`; use `WithTop Nat`, retaining infinity. Prove measurable first-passage events, the one-step formula, and renewal decomposition

   `P^t(x,y) = sum_{s=1}^t r(s,x,y) P^(t-s)(y,y)` for `t > 0`.

   The diagonal case is a first-return identity, not a time-zero entrance identity. Recursively expanding the diagonal term gives (2.5); equivalently, a recursive renewal algorithm recovers the same finite data without the exponential composition sum.
5. **Forced recurrence.** For boundary injection `J`, define `U(0)=0`, `U(t+1)=P U(t)+J f(t)`, with `f` zero after `T`. Prove uniqueness, linearity, and the finite Duhamel sum `U(t)=sum_{s<t} P^(t-1-s) J f(s)`. Combine with item 4 to obtain Lemma 2.1. Record explicit finite-time data dependence; an equality that assumes access to the unknown full transition matrix does not yet establish reconstruction from observed first-passage data.
6. **Control and observation operators.** Define `W f=U(T)` and the short and extended response maps separately. With weighted adjoints, prove the finite Hankel/Gram identity of Proposition 2.4. Its induction is an elementary finite anti-diagonal telescoping identity valid for every self-adjoint `P`; graph assumptions are unnecessary for that identity.
7. **Unique continuation to controllability.** Prove Lemma 2.3 rather than treating the cited result as an axiom. For `n=card X`, `ker W*` is the intersection of `ker(C P^k)` for `k<T`, where `C` is restriction. Cayley–Hamilton shows that the intersection for `k<n` is `P` invariant. If nonzero, the symmetric restriction has an eigenvector vanishing on `B`, contradicting the assumed unique continuation. Conversely such an eigenvector obstructs controllability. This is finite self-adjoint observability/control theory, not an infinite-dimensional PDE theorem.
8. **Harmonic response pairing.** Prove time reversal is a weighted isometry and `Lambda* = R Lambda R` (Appendix A). Telescope `inner(U(t),phi)` for interior-harmonic `phi` to obtain Proposition 2.7. Temporal constant extension of `P_B phi` and `P_B Delta phi` must be explicit in types. Computing `Delta phi` on `B` uses only known `mu_B` and `w`.
9. **Pseudoinverse controls.** Prove the general identity `A ((A* A)^+ (A* y)) = y` for `y` in `range A`, plus orthogonality to `ker A` and minimum-norm uniqueness. Apply to surjective `W` for Proposition 3.1. Weighted Hilbert-space Moore–Penrose inverse must not silently become the ordinary matrix inverse in unnormalized coordinates.
10. **Moment reconstruction.** Harmonic basis products span `Q`; (3.2) supplies the **unweighted** moments `sum_I mu_x q_x`. Set up the product matrix and prove its column range is `Q`. Recover the Euclidean orthogonal projection using minimum-norm solution, and recover all of `mu_I` when its row rank is full. See the projection convention issue below.
11. **Generic rank.** Show the interior Dirichlet matrix has nonzero determinant for positive edge weights, harmonic coordinates are rational functions of those weights, and so are minors of the harmonic-product matrix. One full-rank witness supplies one numerator polynomial that is not zero. Its zero locus is Lebesgue-null. The exceptional rank-deficient set is a subset of that locus. This is enough; no analysis of a finite intersection of all minors is required.

The main proof does not need Monte Carlo simulation, floating-point QR, stability/error estimates, or a formal complexity analysis. These are separate possible developments and should not inflate the prerequisites for the exact theorem.

## Existing library material to consume

| Need | Inspected existing source and declarations | Remaining bridge |
| --- | --- | --- |
| Finite graph Laplacian | Mathlib `Combinatorics/SimpleGraph/LapMatrix.lean`: `SimpleGraph.lapMatrix`, `posSemidef_lapMatrix`, kernel/component theory | This is the nonnegative **unweighted** Laplacian. Do not identify it directly with the paper's `Delta`; a conductance/mass specialization and sign bridge are needed. |
| Finite Hilbert spaces | Mathlib `Analysis/InnerProductSpace/PiL2.lean`; projection modules `Projection/Basic`, `Projection/FiniteDimensional`, `Projection/Minimal` | Finite weighted space, or transport through multiplication by `sqrt(mu)`, plus restriction/extension adapters |
| Adjoints and normal equations | Mathlib `Analysis/InnerProductSpace/Adjoint.lean`: `LinearMap.orthogonal_range`, `ker_adjoint_comp_self`, `range_adjoint_comp_self`, `range_self_comp_adjoint`; `ContinuousLinearMap.norm_eq_iInf_range_iff_adjoint_apply_eq_zero` | Compose these with Penrose equations for a reusable Gram pseudoinverse control/minimum-norm theorem |
| Spectral decomposition | Mathlib `Analysis/InnerProductSpace/Spectrum.lean`: `LinearMap.IsSymmetric.eigenvectorBasis`, `hasEigenvector_eigenvectorBasis`, `apply_eigenvectorBasis` | Finite observability criterion, using an invariant kernel intersection and Cayley–Hamilton |
| Symmetric invariant subspaces | Tau Ceti `Analysis/InnerProductSpace/InvariantSubspace.lean`: `LinearMap.IsSymmetric.isReducing_iff_mem_invtSubmodule`, `starProjection_apply_comm` | Restrict to the unobservable subspace; avoid rebuilding reduction theory |
| Moore–Penrose inverse | Tau Ceti `Analysis/InnerProductSpace/MoorePenrose.lean`: `LinearMap.IsMoorePenroseInverse`, `.unique`, `.apply_apply_apply`, `LinearMap.moorePenroseInverse`, `isMoorePenroseInverse_moorePenroseInverse`, `comp_moorePenroseInverse_of_surjective`, `moorePenroseInverse_adjoint` | General normal-equation/minimum-norm application described above. The core inverse and singular-system construction are already present. `LinearAlgebra/LinearMap/PseudoInverse.lean` supplies only a weaker algebraic generalized inverse, unsuitable as a replacement for the canonical weighted minimum-norm claim. |
| Range projection formula | Tau Ceti `Analysis/InnerProductSpace/RangeProjection.lean`: `ContinuousLinearMap.starProjection_range_eq`, `isUnit_adjoint_comp_self_iff` | Its invertible Gram hypothesis does not directly cover a surjective control map with a nontrivial kernel. |
| Probability kernels from weights | Tau Ceti `Probability/Kernel/PMF.lean`: `Kernel.ofPMF`, `toPMF`, `existsUnique_isMarkovKernel_of_tsum_eq_one`, `pow_add_apply_singleton` | Specialize transition weights; do not introduce a second probability-vector carrier |
| Finite transition matrices | Tau Ceti `Probability/Kernel/Matrix/Basic.lean`: `Kernel.toMatrix`, `toMatrix_comp`, `toMatrix_pow`, `toMatrix_mem_rowStochastic`, `Matrix.rowStochastic.toKernel`, `toKernel_apply_singleton` | Bridge from nonnegative real graph transition entries to the existing `NNReal` matrix/kernel API |
| Finite detailed balance | Tau Ceti `Probability/Kernel/Matrix/Invariance.lean`: `Kernel.isReversible_iff_isSymm_diagonal_mul`, `invariant_iff_vecMul` | Weighted conductance specialization; the finite measure need not be normalized to a probability |
| Path laws and Markov properties | Tau Ceti `Probability/Process/MarkovChain.lean`, `Probability/MarkovChain/MarkovProperty.lean`, `StrongMarkov/Basic.lean`: `markovChainLaw`, `Measure.markovChainLaw_restrict_map_stoppedValue_shift_iterate_eq_smul` | Finite-time first-passage/renewal identities with the paper's convention |
| Hitting times | Mathlib `Probability/Process/HittingTime.lean`: `MeasureTheory.hittingAfter`, `hittingAfter_eq_top_iff`, `le_hittingAfter`, `hittingAfter_le_iff`, `Adapted.isStoppingTime_hittingAfter` | Use starting index one for positive first passages; a first-passage event is never replaced by time-zero entrance on the diagonal |
| Successive visits/returns | Tau Ceti `Probability/MarkovChain/Returns.lean`, `Visits.lean`, `ReturnGaps.lean`: return probabilities, successive-visit laws, strong Markov decomposition, Green-function identities `Kernel.tsum_pow_apply_singleton_self` and `tsum_pow_comp_apply_singleton` | These are substantial existing recurrence/renewal infrastructure; finite-time first-passage distributions still require the explicit adapter/identity rather than merely citing infinite Green sums. |
| Generic polynomial nonvanishing | Tau Ceti `MeasureTheory/Measure/MvPolynomialZeroLocus.lean`: `MvPolynomial.measure_pi_setOfPred_eval_eq_zero`, `volume_setOfPred_eval_eq_zero` | Dirichlet rationality, nonzero minor numerator witness, and inclusion of the exceptional rank locus |

Searches inspected the probability MarkovChain/Kernel directories, inner-product-space modules, graph directories, relevant filenames across Tau Ceti, and authoritative roadmap READMEs/Suggested files. No claim of mathematical absence is inferred merely from failure to find a particular filename.

## Roadmap authorization and boundaries

Tau Ceti's `AGENTS.md` permits new mathematics only for a specific existing roadmap target or a prerequisite that target needs. A theorem being useful for this paper is not, by itself, Tau Ceti roadmap authorization. Existing-code fixes and modest improvements have their separate always-in-scope policy.

1. **`MarkovChainsAndRenewal`**, `TauCetiRoadmap/MarkovChainsAndRenewal/README.md`:
   - Layer 0, item 1: transition weights, singleton evaluation, PMF and finite stochastic-matrix adapters, compatibility with powers. Most of this is already implemented in the modules above.
   - Layer 0, item 2: time-n marginals, finite cylinders, measurable starting-state dependence.
   - Layer 1, items 3–4: strong Markov property, stopping-time/measurability lemmas for entrances, returns, successive visits, shifted paths. Positive first-passage distribution lemmas are a natural narrow library addition here.
   - Layer 2, item 2: return/occupation/Green-function identities. A finite-time renewal identity is relevant infrastructure for this target and for the paper; the PR must explain the concrete connection, not claim that arbitrary graph inverse problems are covered.
   - Layer 3, item 1: finite stationary/reversible matrix adapters, already present.
   The roadmap explicitly chooses `Kernel`, `markovChainLaw`, `WithTop Nat`, and Mathlib `hittingAfter`; the implementation should follow those choices. Ordinary continuous-time renewal equations in Layer 6 are not automatically an owner for every discrete matrix recurrence.
2. **`OperatorTheory`**, sub-roadmap `TauCetiRoadmap/OperatorTheory/PolarDecomposition/README.md`:
   - PD-C19–PD-C27 specify the Penrose relation, construction, uniqueness, adjoint compatibility, and injective/surjective/invertible cases. These core targets are already represented by `MoorePenrose.lean`.
   - A general consequence proving exact recovery from Gram normal equations improves the existing inverse API and may be justifiable as a modest extension. It must not be presented as an unimplemented named roadmap milestone: none of PD-C19–PD-C27 explicitly states the minimum-norm control formula.
3. **`GraphConnectivityAndFlows`** is about connectivity, cuts, flows, Menger, orientations, and related finite graph infrastructure. Its weighted capacities do not constitute a target for weighted graph Laplacians, graph heat equations, harmonic extension, or inverse boundary control.
4. **`OperatorTheory/SelfAdjointSpectralTheory`** supplies spectral/partial-operator theory; its title does not authorize all finite control theory. The paper needs ordinary finite-dimensional symmetric spectral theory already available in Mathlib, rather than Stone's theorem or unbounded operators.
5. The existing polynomial-null-locus theorem is usable without a new contribution. Its existence is not permission to add the paper's entire generic inverse problem under an unrelated algebraic-geometry roadmap.

A search of roadmap READMEs and Suggested files for graph Laplacians, weighted graph Laplacians, discrete heat, vertex centrality, Kalman/controllability, and graph Dirichlet problems did not identify a dedicated roadmap target for the graph-analysis/control/reconstruction stages. Incidental matches concern continuous PDE/geometric graphs or unrelated arithmetic Dirichlet problems. Treat the graph-specific stages as requiring a human-curated roadmap unless a maintainer points to a more precise existing target. Do not claim existing-roadmap work exhausted until the finite-time probability and inverse-API gaps have actually been checked, implemented where warranted, and merged.

## Corrections and edge cases to settle before formalization

### Confirmed coordinate error: equation (4.3) omits the boundary mass matrix

The paper's space-time inner product is weighted by `mu_B`. With the paper's ordinary value-vector coordinates, the first term on the right of (4.3) must be

`h^T (I_T tensor diagonal(mu_B)) [W*] phi`,

not just `h^T [W*] phi`. An alternative valid presentation uses coordinates normalized by square roots of the masses everywhere, but that is not the value-vector map (3.3).

Here is an exact rational counterexample satisfying the graph assumptions with a nonempty interior. Take the path `b--i`, conductance one, `B={b}`, masses `(mu_b,mu_i)=(2,1)`, and `T=2`. Then

```text
P = [[1/2, 1/2], [1, 0]],
W(f0,f1) = (f0/2 + f1, f0),
W* = [[1/2,1/2], [1,0]],
W*W = [[3/4,1/2], [1/2,1]].
```

Take `psi=phi=(1,1)`, harmonic on the interior. The unique exact control is `h=(1,1/2)`, which also equals `(W*W)^+ W* psi`, and `W* phi=(1,1)`. The weighted pairing is `2*(1+1/2)=3`. Subtracting the boundary term `mu_b=2` gives the correct interior moment `mu_i=1`. The printed unweighted dot product instead gives `3/2-2=-1/2`. Unique continuation holds: an eigenfunction zero at `b` has value zero at `i` by the eigen-equation at `b`. Both vertex masses dominate their weighted degrees. Thus the mismatch is not caused by a failed hypothesis.

### Confirmed convention mismatch in the projection derivation

Equation (3.2) determines the linear functional `q -> sum_I mu_x q_x`. Its Riesz vector on `Q` in **counting-measure Euclidean** inner product is the Euclidean orthogonal projection of the vector `mu_I`. The paper defined `ell^2(I)` using the **unknown mu-weighted** inner product. Projection of `mu_I` in that inner product instead satisfies moments involving `sum_I mu_x^2 q_x`. These are different objects.

For an exact example take the path `b--i--j`, conductance one on both edges, `B={b}`, and masses `(1,2,3)`. Every function harmonic at `i,j` is constant: the equation at `j` gives `phi_j=phi_i`, and that at `i` then gives `phi_i=phi_b`. Hence `Q=span{(1,1)}` on the two interior vertices. All graph assumptions hold; unique continuation from endpoint `b` propagates successively to `i,j`. Choose `T>=3`. The two projections of `mu_I=(2,3)` are

```text
Euclidean projection:       ((2+3)/2) (1,1) = (5/2,5/2).
mu-weighted projection:     ((2^2+3^2)/(2+3)) (1,1) = (13/5,13/5).
```

This demonstrates that the moment/least-squares derivation proves the Euclidean projection statement, not the weighted statement as introduced. It does **not** prove that the full first-passage data could not determine the weighted projection by some different argument. A faithful formal theorem should explicitly choose the Euclidean projection supported by the displayed reconstruction procedure. The full-rank case `Q=all` has no projection ambiguity.

### Genericity proof: use one nonzero minor and separate admissibility

In Corollary 3.3, `S_beta` includes both a determinant-zero condition and `mu_x >= sum_y w_xy`. The assertion that each such `S_beta` is either the entire positive orthant or null does not follow from rationality: if its minor is identically zero, `S_beta` is the admissible weight region, which generally is a proper set of positive measure. This is a gap in that sentence of the proof, with an easy repair. Use a full-rank witness to select a minor whose numerator polynomial is nonzero; the rank-deficient locus is contained in that one polynomial zero set. Intersect with admissibility and with any unique-continuation assumptions only afterward. The generic **rank** conclusion does not itself remove the theorem's controllability hypothesis.

### Other necessary conventions and boundary checks

- `T` must be positive wherever `Fin T` reversal and the paper's nonempty interval convention are used. For a nonempty graph `T>=card X` supplies this; earlier identities should state positivity or handle `T=0` separately.
- Define the recurrence on all natural times with controls extended by zero. The paper writes the forcing equation on `Z_(2T)` although a forward difference at its last time uses `U(2T)`. The measured response only needs values through `2T-1`; a global recurrence avoids an ill-typed endpoint.
- `R` acts on time indices `0,...,T-1`. Encode with `Fin.rev` or a proved equivalence, avoiding unguarded truncated subtraction. Spatial functions in Proposition 2.7 require an explicit constant-in-time map.
- In a nonempty graph, unique continuation excludes `B=empty`: the nonzero constant function is a zero-eigenvalue eigenfunction. Prove this consequence before using Dirichlet uniqueness. More generally, Dirichlet uniqueness needs `B` to meet each connected component.
- The connected one-vertex graph is valid, has no edges, and has `P=1`. If unique continuation holds then its vertex lies in `B`; the interior reconstruction is empty. No positive degree or division by the number of edges should be assumed.
- Positive `mu` prevents division by zero. The diagonal transition is `1-sum w/mu`; `w_xx=0` follows from simple-loopless support, not from symmetry alone.
- `P` may have negative or repeated eigenvalues and may be periodic. The arguments need weighted self-adjointness and finite-dimensional spectral theory, not strict positive definiteness, simple spectrum, laziness, mixing, or convergence of `P^t`.
- The boundary control space has dimension `T*card B`; `W*W` is typically singular. Surjectivity of `W` does not make this Gram matrix invertible. The existing Moore–Penrose construction handles this.
- Ordinary matrix Moore–Penrose inversion changes under nonorthonormal coordinates. In Step 4 it still yields some solution of the consistent normal equations, and hence an exact control, but its asserted **weighted** minimum-norm property requires proof or normalized coordinates.
- Empty interior is a valid trivial full-rank case. The `0`-by-`0` determinant is one; the generic-rank proof should not assume a positive minor size.
- The pair-product count is `card B*(card B+1)/2`; the rank/cardinality bound is necessary, not sufficient. A full-rank weight witness must be an actual hypothesis or example, not inferred solely from that count.
- The computational and empirical error claims in Section 5 should not be used as hypotheses or exact correctness guarantees. Exact reconstruction assumes genuine first-passage probabilities and exact linear algebra.

## Human roadmap contribution preparation

Read `TauCetiRoadmap/CONTRIBUTING.md` and the root README's sections on writing, prototyping, and filing roadmaps. A future human-owned graph-analysis roadmap should settle the two coordinate conventions above and specify reusable layers for weighted graph operators, Dirichlet extension, finite observability/control, and inverse reconstruction, with explicit dependencies on the probability and operator APIs listed here. Its endpoint must say whether it includes only exact reconstruction or also genericity and quantitative numerical stability.

The contribution guide asks the human author to inspect nearby merged roadmaps and the Lean Zulip roadmap discussion, prototype precise interfaces importing existing modules, include `metadata.toml` with a primary arXiv topic, and submit a PR touching only the new roadmap's own directory. The README remains definitive; `Suggested.lean` illustrates interfaces rather than replacing the specification. Roadmap content requires approval from `TauCetiProject/roadmap-reviewers` and a passing build. AI assistance must be attributed, and the human must read and endorse the proposal before posting it. The guide's `awaiting-review`/`awaiting-author` labels track review handoff.

No roadmap PR or issue is opened by this audit. This is preparation for the eventual tutorial requested by the user, not a claim that the existing-roadmap contribution work is complete.
