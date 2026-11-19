# Approved statement corrections

The user approved corrected targets with documented deviations on 2026-11-19.
These are project decisions supported by the existing [proof audit](../PROOF_AUDIT.md),
not author-confirmed errata. All affected claims still need Lean proofs.

| Paper location | Project target | Reason |
| --- | --- | --- |
| Theorem 3.2 | Euclidean/counting-measure orthogonal projection of the interior mass vector onto the harmonic-product span | Equation (3.2) provides unweighted moments of the mass vector. The paper's earlier mass-weighted inner product would define a different projection. |
| Equation (4.3) | `hᵀ (I_T ⊗ diag(μ_B)) [W*] φ` in ordinary value coordinates | The space-time pairing includes the boundary masses; the printed dot product omits them. |
| Corollary 3.3 proof | Select one minor nonzero at a full-rank witness, clear its nonvanishing denominator, then use the null zero locus of its nonzero numerator polynomial | The printed zero-set dichotomy need not hold after intersecting with admissibility. Generic rank does not imply unique continuation or admissibility. |

For the pseudoinverse, the Hilbert-space control uses the weighted inner product.
Ordinary matrix pseudoinversion of the consistent normal equations still gives an
exact control, but does not automatically give the weighted minimum-norm control.
The algorithm correctness target requires exact controls; the minimum-norm theorem
is stated in the correct weighted spaces.

The existing exact-rational examples were rerun successfully for this setup:

```sh
python3 ../scripts/check_paper_coordinates.py
```

They return `1` versus `-1/2` for the corrected/printed moment formula and `5/2`
versus `13/5` for the Euclidean/weighted projection coefficients. They are arithmetic
checks, not Lean proofs or a refutation of every possible interpretation of the paper.

The first-passage definition always starts at positive time, including on the
diagonal. The full-rank witness remains an explicit hypothesis. Numerical accuracy,
Monte Carlo convergence, and complexity estimates are outside the initial scope.
