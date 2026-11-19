module

public import VertexCentrality.PolynomialRank

/-! # Generic rank of the actual harmonic-product matrix

An explicit full-rank graph in the conductance family is required. Exceptional
parameters lie in the zero set of a nonzero Gram determinant polynomial.
-/

@[expose] public section

namespace VertexCentrality

open MeasureTheory

variable {X σ : Type*} [Fintype X] [DecidableEq X]

/-- Polynomial numerators of the symmetric harmonic-product coordinates. -/
noncomputable def polynomialProductMatrix (w : X → X → MvPolynomial σ ℝ)
    (B : Finset X) : Matrix (Finset.univ \ B : Finset X) (Sym2 B) (MvPolynomial σ ℝ) :=
  fun x ↦ Sym2.lift ⟨fun j k ↦
    (polynomialDirichletMatrix w B).adjugate x j *
      (polynomialDirichletMatrix w B).adjugate x k,
    fun _ _ ↦ mul_comm _ _⟩

namespace WeightedGraph

/-- Clearing the common squared determinant gives the polynomial product matrix. -/
theorem polynomialProductMatrix_eval (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) :
    (polynomialProductMatrix w B).map (MvPolynomial.eval z) =
      MvPolynomial.eval z ((polynomialDirichletMatrix w B).det ^ 2) •
        G.harmonicProductMatrix B hconn hB := by
  ext x p
  induction p using Sym2.inductionOn with
  | _ j k =>
    change MvPolynomial.eval z ((polynomialDirichletMatrix w B).adjugate x j *
      (polynomialDirichletMatrix w B).adjugate x k) =
      MvPolynomial.eval z ((polynomialDirichletMatrix w B).det ^ 2) *
        G.harmonicProductMatrix B hconn hB x (Sym2.mk j k)
    rw [G.harmonicProductMatrix_eq_polynomial_ratio B hconn hB w z hw]
    exact (mul_div_cancel₀ _ (by
      rw [map_pow]
      exact pow_ne_zero _ (G.polynomialDirichletMatrix_det_eval_ne_zero B hconn hB w z hw))).symm

/-- Clearing the denominator preserves the actual harmonic-product rank. -/
theorem polynomialProductMatrix_rank (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) :
    ((polynomialProductMatrix w B).map (MvPolynomial.eval z)).rank =
      (G.harmonicProductMatrix B hconn hB).rank := by
  rw [G.polynomialProductMatrix_eval B hconn hB w z hw]
  apply Matrix.rank_smul_of_mem_nonZeroDivisors
  apply mem_nonZeroDivisors_of_ne_zero
  rw [map_pow]
  exact pow_ne_zero _ (G.polynomialDirichletMatrix_det_eval_ne_zero B hconn hB w z hw)

/-- A full-rank witness makes the exceptional parameter set null. Only parameters
realized by connected graphs are included; further admissibility/UCP restrictions
can be imposed by taking subsets of this set. -/
theorem volume_harmonicProduct_rank_deficient_eq_zero [Fintype σ]
    (B : Finset X) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ)
    (G₀ : WeightedGraph X) (hconn₀ : G₀.toSimpleGraph.Connected) (z₀ : σ → ℝ)
    (hw₀ : ∀ x y, MvPolynomial.eval z₀ (w x y) = G₀.conductance x y)
    (hrank₀ : (G₀.harmonicProductMatrix B hconn₀ hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    volume {z : σ → ℝ | ∃ (G : WeightedGraph X) (hconn : G.toSimpleGraph.Connected),
      (∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) ∧
      (G.harmonicProductMatrix B hconn hB).rank ≠
        Fintype.card (Finset.univ \ B : Finset X)} = 0 := by
  apply measure_mono_null _ (volume_rank_deficient_eq_zero (polynomialProductMatrix w B) z₀
    ((G₀.polynomialProductMatrix_rank B hconn₀ hB w z₀ hw₀).trans hrank₀))
  rintro z ⟨G, hconn, hw, hrank⟩
  change ((polynomialProductMatrix w B).map (MvPolynomial.eval z)).rank ≠ _
  rw [G.polynomialProductMatrix_rank B hconn hB w z hw]
  exact hrank

/-- Almost every conductance parameter gives full harmonic-product rank whenever
it is realized by a connected graph. -/
theorem ae_harmonicProduct_full_rank [Fintype σ]
    (B : Finset X) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ)
    (G₀ : WeightedGraph X) (hconn₀ : G₀.toSimpleGraph.Connected) (z₀ : σ → ℝ)
    (hw₀ : ∀ x y, MvPolynomial.eval z₀ (w x y) = G₀.conductance x y)
    (hrank₀ : (G₀.harmonicProductMatrix B hconn₀ hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    ∀ᵐ z : σ → ℝ, ∀ (G : WeightedGraph X) (hconn : G.toSimpleGraph.Connected),
      (∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) →
      (G.harmonicProductMatrix B hconn hB).rank =
        Fintype.card (Finset.univ \ B : Finset X) := by
  rw [ae_iff]
  simpa only [not_forall, Classical.not_imp, exists_prop] using
    volume_harmonicProduct_rank_deficient_eq_zero B hB w G₀ hconn₀ z₀ hw₀ hrank₀

/-- Generic finite-path mass recovery, conditional on admissibility and UCP
for both candidate graphs. These restrictions are not dropped in passing to an
almost-everywhere rank statement. -/
theorem ae_mass_eq_of_firstPassageProbability_eq [Fintype σ]
    (B : Finset X) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ)
    (G₀ : WeightedGraph X) (hconn₀ : G₀.toSimpleGraph.Connected) (z₀ : σ → ℝ)
    (hw₀ : ∀ x y, MvPolynomial.eval z₀ (w x y) = G₀.conductance x y)
    (hrank₀ : (G₀.harmonicProductMatrix B hconn₀ hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    ∀ᵐ z : σ → ℝ, ∀ (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
      (_hconn : G.toSimpleGraph.Connected) (T : ℕ),
      (∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) →
      G.conductance = H.conductance → (∀ x : B, G.mass x = H.mass x) →
      (∀ k, k + 1 < 2 * T → ∀ x y : B,
        G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y) →
      G.UniqueContinuation B → H.UniqueContinuation B → Fintype.card X ≤ T →
      G.mass = H.mass := by
  filter_upwards [ae_harmonicProduct_full_rank B hB w G₀ hconn₀ z₀ hw₀ hrank₀] with z hz
  intro G H hG hH hconn T hw hc hm hr hucG hucH hT
  exact G.mass_eq_of_firstPassageProbability_eq_of_rank H hG hH B T hconn hB
    hc hm hr hucG hucH hT (hz G hconn hw)

end WeightedGraph
end VertexCentrality
