module

public import VertexCentrality.ProductSpan
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# A mass-independent Dirichlet matrix

Use identity rows on the boundary and the unnormalized conductance Laplacian
on the interior. This square matrix has polynomial entries in conductances.
Its inverse gives the existing harmonic extension; its nonzero determinant
provides the denominator for rational Dirichlet coordinates.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- Dirichlet constraints with the interior vertex masses cleared from each row. -/
noncomputable def conductanceDirichletMatrix (G : WeightedGraph X) (B : Finset X) :
    Matrix X X ℝ := fun x y ↦
  if x ∈ B then (if x = y then 1 else 0)
  else G.conductance x y - if x = y then G.degree x else 0

/-- The matrix imposes boundary values and mass-cleared interior Laplacian values. -/
theorem conductanceDirichletMatrix_mulVec (G : WeightedGraph X) (B : Finset X)
    (u : X → ℝ) (x : X) :
    (G.conductanceDirichletMatrix B).mulVec u x =
      if x ∈ B then u x else G.mass x * G.laplacian u x := by
  by_cases hx : x ∈ B
  · simp [conductanceDirichletMatrix, Matrix.mulVec, dotProduct, hx]
  · simp only [conductanceDirichletMatrix, Matrix.mulVec, dotProduct, hx, ite_false,
      sub_mul, ite_mul, zero_mul, Finset.sum_sub_distrib]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    rw [laplacian, mul_div_cancel₀ _ (ne_of_gt (G.mass_pos x))]
    simp [mul_sub, Finset.sum_sub_distrib, degree, Finset.sum_mul]

/-- Connectedness and nonempty boundary make the mass-independent matrix injective. -/
theorem conductanceDirichletMatrix_injective (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    Function.Injective (G.conductanceDirichletMatrix B).mulVec := by
  intro u v huv
  apply G.dirichletOperator_injective B hconn hB
  funext x
  have h := congrFun huv x
  rw [G.conductanceDirichletMatrix_mulVec, G.conductanceDirichletMatrix_mulVec] at h
  by_cases hx : x ∈ B
  · rw [G.dirichletOperator_apply_of_mem B u hx, G.dirichletOperator_apply_of_mem B v hx]
    simpa only [hx, ite_true] using h
  · rw [G.dirichletOperator_apply_of_notMem B u hx, G.dirichletOperator_apply_of_notMem B v hx]
    exact mul_left_cancel₀ (ne_of_gt (G.mass_pos x)) (by simpa only [hx, ite_false] using h)

/-- The denominator of the Dirichlet inverse never vanishes on connected positive graphs. -/
theorem conductanceDirichletMatrix_det_ne_zero (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    (G.conductanceDirichletMatrix B).det ≠ 0 := by
  exact isUnit_iff_ne_zero.mp ((Matrix.isUnit_iff_isUnit_det _).mp
    (Matrix.mulVec_injective_iff_isUnit.mp (G.conductanceDirichletMatrix_injective B hconn hB)))

/-- The existing harmonic extension solves the mass-independent matrix equation. -/
theorem conductanceDirichletMatrix_harmonicExtension (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) (g : B → ℝ) :
    (G.conductanceDirichletMatrix B).mulVec (G.harmonicExtension B hconn hB g) =
      boundaryExtend B g := by
  funext x
  rw [G.conductanceDirichletMatrix_mulVec]
  by_cases hx : x ∈ B
  · simp only [boundaryExtend, hx, ite_true, dite_true]
    exact G.harmonicExtension_boundary B hconn hB g ⟨x, hx⟩
  · simp only [boundaryExtend, hx, ite_false, dite_false]
    rw [G.harmonicExtension_harmonic B hconn hB g x hx, mul_zero]

/-- The actual harmonic extension equals the inverse-matrix Dirichlet solution. -/
theorem harmonicExtension_eq_inverse (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) (g : B → ℝ) :
    G.harmonicExtension B hconn hB g =
      (G.conductanceDirichletMatrix B)⁻¹.mulVec (boundaryExtend B g) := by
  rw [← G.conductanceDirichletMatrix_harmonicExtension B hconn hB g,
    Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _
      (isUnit_iff_ne_zero.mpr (G.conductanceDirichletMatrix_det_ne_zero B hconn hB)),
    Matrix.one_mulVec]

/-- A Dirichlet basis coordinate is a column of the mass-independent inverse. -/
theorem harmonicBasis_eq_inverse_entry (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) (j : B) (x : X) :
    (G.harmonicBasis B hconn hB j).val x = (G.conductanceDirichletMatrix B)⁻¹ x j := by
  rw [G.harmonicBasis_apply, G.harmonicExtension_eq_inverse B hconn hB]
  have hg : boundaryExtend B (Pi.basisFun ℝ B j) = Pi.single (j : X) 1 := by
    funext y
    by_cases hy : y ∈ B
    · simp [boundaryExtend, hy, Pi.basisFun_apply, Pi.single_apply, Subtype.ext_iff]
    · have hj : (j : X) ≠ y := fun h ↦ hy (h ▸ j.property)
      simp [boundaryExtend, hy, hj]
  change (G.conductanceDirichletMatrix B)⁻¹.mulVec (boundaryExtend B (Pi.basisFun ℝ B j)) x = _
  exact (congrArg (fun u ↦ (G.conductanceDirichletMatrix B)⁻¹.mulVec u x) hg).trans (by
    simp [Matrix.mulVec, dotProduct, Pi.single_apply])

/-- Dirichlet basis coordinates have an explicit adjugate-over-determinant formula. -/
theorem harmonicBasis_eq_adjugate_div_det (G : WeightedGraph X) (B : Finset X)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) (j : B) (x : X) :
    (G.harmonicBasis B hconn hB j).val x =
      (G.conductanceDirichletMatrix B).adjugate x j / (G.conductanceDirichletMatrix B).det := by
  rw [G.harmonicBasis_eq_inverse_entry B hconn hB, Matrix.inv_def]
  simp [Ring.inverse_eq_inv, div_eq_mul_inv, mul_comm]

end VertexCentrality.WeightedGraph
