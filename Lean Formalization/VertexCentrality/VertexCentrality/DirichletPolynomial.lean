module

public import VertexCentrality.DirichletMatrix
public import Mathlib.Algebra.MvPolynomial.Eval

/-!
# Polynomial coordinates for the Dirichlet problem

The parameter family is arbitrary, allowing one variable per undirected edge.
Evaluation is linked to the actual conductances, not assumed rational coordinates.
-/

@[expose] public section

namespace VertexCentrality

variable {X σ : Type*} [Fintype X] [DecidableEq X]

/-- Polynomial version of the mass-cleared Dirichlet matrix. -/
noncomputable def polynomialDirichletMatrix (w : X → X → MvPolynomial σ ℝ)
    (B : Finset X) : Matrix X X (MvPolynomial σ ℝ) := fun x y ↦
  if x ∈ B then (if x = y then 1 else 0)
  else w x y - if x = y then ∑ z, w x z else 0

namespace WeightedGraph

/-- Evaluation of the polynomial matrix gives the graph's Dirichlet matrix. -/
theorem polynomialDirichletMatrix_eval (G : WeightedGraph X) (B : Finset X)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) :
    (MvPolynomial.eval z).mapMatrix (polynomialDirichletMatrix w B) =
      G.conductanceDirichletMatrix B := by
  ext x y
  by_cases hxy : x = y
  · subst y
    by_cases hx : x ∈ B <;>
      simp [polynomialDirichletMatrix, conductanceDirichletMatrix, hx, hw, degree]
  · by_cases hx : x ∈ B <;>
      simp [polynomialDirichletMatrix, conductanceDirichletMatrix, hx, hxy, hw]

/-- The polynomial denominator evaluates to a nonzero real number on every
connected graph with nonempty boundary in the conductance family. -/
theorem polynomialDirichletMatrix_det_eval_ne_zero (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y) :
    MvPolynomial.eval z (polynomialDirichletMatrix w B).det ≠ 0 := by
  rw [RingHom.map_det, G.polynomialDirichletMatrix_eval B w z hw]
  exact G.conductanceDirichletMatrix_det_ne_zero B hconn hB

/-- Every harmonic basis coordinate is the ratio of two explicitly constructed
polynomials in the conductance parameters. -/
theorem harmonicBasis_eq_polynomial_ratio (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y)
    (j : B) (x : X) :
    (G.harmonicBasis B hconn hB j).val x =
      MvPolynomial.eval z ((polynomialDirichletMatrix w B).adjugate x j) /
        MvPolynomial.eval z (polynomialDirichletMatrix w B).det := by
  rw [G.harmonicBasis_eq_adjugate_div_det B hconn hB, RingHom.map_det,
    G.polynomialDirichletMatrix_eval B w z hw]
  have h := congrArg (fun M : Matrix X X ℝ ↦ M x j)
    ((MvPolynomial.eval z).map_adjugate (polynomialDirichletMatrix w B))
  simp only [G.polynomialDirichletMatrix_eval B w z hw] at h
  exact congrArg (fun a ↦ a / (G.conductanceDirichletMatrix B).det) h.symm

/-- Products use the common squared determinant denominator. -/
theorem harmonicBasis_product_eq_polynomial_ratio (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y)
    (j k : B) (x : X) :
    (G.harmonicBasis B hconn hB j).val x * (G.harmonicBasis B hconn hB k).val x =
      MvPolynomial.eval z ((polynomialDirichletMatrix w B).adjugate x j *
        (polynomialDirichletMatrix w B).adjugate x k) /
        MvPolynomial.eval z ((polynomialDirichletMatrix w B).det ^ 2) := by
  rw [G.harmonicBasis_eq_polynomial_ratio B hconn hB w z hw j x,
    G.harmonicBasis_eq_polynomial_ratio B hconn hB w z hw k x,
    map_mul, map_pow, pow_two, div_mul_div_comm]

/-- The rational formula applies to the actual symmetric harmonic-product matrix. -/
theorem harmonicProductMatrix_eq_polynomial_ratio (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (w : X → X → MvPolynomial σ ℝ) (z : σ → ℝ)
    (hw : ∀ x y, MvPolynomial.eval z (w x y) = G.conductance x y)
    (j k : B) (x : (Finset.univ \ B : Finset X)) :
    G.harmonicProductMatrix B hconn hB x (Sym2.mk j k) =
      MvPolynomial.eval z ((polynomialDirichletMatrix w B).adjugate x j *
        (polynomialDirichletMatrix w B).adjugate x k) /
        MvPolynomial.eval z ((polynomialDirichletMatrix w B).det ^ 2) := by
  exact G.harmonicBasis_product_eq_polynomial_ratio B hconn hB w z hw j k x

end WeightedGraph
end VertexCentrality
