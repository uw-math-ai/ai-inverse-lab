/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Authors: Owen Parks
-/
module

public import Mathlib.Algebra.Module.Pi
public import Mathlib.Basic.Real.Basic
public import Mathlib.Combinatorics.SimpleGraph.Basic
public import Mathlib.LinearAlgebra.BilinearMap
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic

/-!
# Weighted graph operators

Finite positive vertex masses and symmetric nonnegative conductances realize the graph
Laplacian and transition matrix in Gao–Li–Yang, arXiv:2603.20710v1. Connectivity is not
needed for these algebraic identities. It must be imposed on `WeightedGraph.toSimpleGraph`
when using the later Dirichlet and reconstruction results.
-/

@[expose] public section

open scoped BigOperators

namespace VertexCentrality

/-- Positive vertex masses and symmetric, nonnegative, loopless conductances.
An edge is present exactly when its conductance is positive. -/
structure WeightedGraph (X : Type*) where
  mass : X → ℝ
  conductance : X → X → ℝ
  mass_pos : ∀ x, 0 < mass x
  conductance_nonneg : ∀ x y, 0 ≤ conductance x y
  conductance_symm : ∀ x y, conductance x y = conductance y x
  conductance_diag : ∀ x, conductance x x = 0

namespace WeightedGraph

variable {X : Type*} (G : WeightedGraph X)

/-- The underlying simple graph, with positive-conductance adjacency. -/
def toSimpleGraph : SimpleGraph X where
  Adj x y := 0 < G.conductance x y
  symm := ⟨by intro x y h; simpa only [G.conductance_symm y x] using h⟩
  loopless := ⟨by intro x h; simp only [G.conductance_diag, lt_self_iff_false] at h⟩

variable [Fintype X]

/-- Weighted degree, including a zero diagonal contribution. -/
noncomputable def degree (x : X) : ℝ := ∑ y, G.conductance x y

/-- The mass-weighted spatial pairing. -/
noncomputable def weightedPair (u v : X → ℝ) : ℝ := ∑ x, G.mass x * u x * v x

/-- The graph Laplacian with the paper's nonpositive sign convention. -/
noncomputable def laplacian (u : X → ℝ) (x : X) : ℝ :=
  (∑ y, G.conductance x y * (u y - u x)) / G.mass x

/-- The holding transition kernel in ordinary value coordinates. -/
noncomputable def transition (x y : X) : ℝ := by
  classical
  exact (if x = y then 1 else 0) + G.conductance x y / G.mass x -
    (if x = y then G.degree x / G.mass x else 0)

/-- The mass domination needed for nonnegative holding probabilities. -/
def Admissible : Prop := ∀ x, G.degree x ≤ G.mass x

/-- Harmonicity on an arbitrary set of unobserved vertices. -/
def HarmonicOn (I : Set X) (u : X → ℝ) : Prop := ∀ x ∈ I, G.laplacian u x = 0

/-- The weighted pairing is symmetric. -/
theorem weightedPair_comm (u v : X → ℝ) : G.weightedPair u v = G.weightedPair v u := by
  unfold weightedPair
  apply Finset.sum_congr rfl
  intro x _
  ring

/-- The pairing is additive in its first argument. -/
theorem weightedPair_add_left (u v z : X → ℝ) :
    G.weightedPair (u + v) z = G.weightedPair u z + G.weightedPair v z := by
  simp [weightedPair, mul_add, add_mul, Finset.sum_add_distrib]

/-- The pairing respects real scaling in its first argument. -/
theorem weightedPair_smul_left (c : ℝ) (u v : X → ℝ) :
    G.weightedPair (c • u) v = c * G.weightedPair u v := by
  simp [weightedPair, Finset.mul_sum, mul_left_comm, mul_assoc]

/-- The pairing is additive in its second argument. -/
theorem weightedPair_add_right (u v z : X → ℝ) :
    G.weightedPair u (v + z) = G.weightedPair u v + G.weightedPair u z := by
  simp [weightedPair, mul_add, Finset.sum_add_distrib]

/-- The pairing respects real scaling in its second argument. -/
theorem weightedPair_smul_right (c : ℝ) (u v : X → ℝ) :
    G.weightedPair u (c • v) = c * G.weightedPair u v := by
  simp [weightedPair, Finset.mul_sum, mul_left_comm, mul_assoc]

/-- The weighted spatial pairing as a bilinear map. -/
noncomputable def weightedPairLinear : (X → ℝ) →ₗ[ℝ] (X → ℝ) →ₗ[ℝ] ℝ :=
  LinearMap.mk₂ ℝ G.weightedPair G.weightedPair_add_left G.weightedPair_smul_left
    G.weightedPair_add_right G.weightedPair_smul_right

/-- Evaluation of the bundled pairing. -/
@[simp] theorem weightedPairLinear_apply (u v : X → ℝ) :
    G.weightedPairLinear u v = G.weightedPair u v := rfl

/-- The weighted quadratic pairing is nonnegative. -/
theorem weightedPair_self_nonneg (u : X → ℝ) : 0 ≤ G.weightedPair u u := by
  apply Finset.sum_nonneg
  intro x _
  simpa only [mul_assoc] using mul_nonneg (le_of_lt (G.mass_pos x)) (mul_self_nonneg (u x))

/-- Positive masses make the weighted pairing positive definite. -/
@[simp] theorem weightedPair_self_eq_zero_iff (u : X → ℝ) :
    G.weightedPair u u = 0 ↔ u = 0 := by
  constructor
  · intro h
    have hx : ∀ x, G.mass x * u x * u x = 0 := by
      intro x
      apply (Finset.sum_eq_zero_iff_of_nonneg ?_).mp h x (Finset.mem_univ x)
      intro y _
      simpa only [mul_assoc] using
        mul_nonneg (le_of_lt (G.mass_pos y)) (mul_self_nonneg (u y))
    ext x
    have hm := ne_of_gt (G.mass_pos x)
    simpa [mul_assoc, hm] using hx x
  · rintro rfl
    simp [weightedPair]

/-- Mass factors cancel in a Laplacian pairing. -/
theorem weightedPair_laplacian (u v : X → ℝ) :
    G.weightedPair (G.laplacian u) v =
      ∑ x, ∑ y, G.conductance x y * (u y - u x) * v x := by
  unfold weightedPair laplacian
  apply Finset.sum_congr rfl
  intro x _
  rw [← Finset.sum_mul]
  field_simp [ne_of_gt (G.mass_pos x)]

/-- The discrete integration-by-parts identity, with both ordered edge orientations. -/
theorem energy_identity (u v : X → ℝ) :
    G.weightedPair (G.laplacian u) v =
      -(1 / 2 : ℝ) * ∑ x, ∑ y,
        G.conductance x y * (u y - u x) * (v y - v x) := by
  rw [G.weightedPair_laplacian]
  have hswap : (∑ x, ∑ y, G.conductance x y * (u y - u x) * v y) =
      -(∑ x, ∑ y, G.conductance x y * (u y - u x) * v x) := by
    rw [Finset.sum_comm]
    simp only [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro x _
    apply Finset.sum_congr rfl
    intro y _
    rw [G.conductance_symm y x]
    ring
  have hsplit : (∑ x, ∑ y, G.conductance x y * (u y - u x) * (v y - v x)) =
      (∑ x, ∑ y, G.conductance x y * (u y - u x) * v y) -
      (∑ x, ∑ y, G.conductance x y * (u y - u x) * v x) := by
    simp only [mul_sub, Finset.sum_sub_distrib]
  rw [hsplit, hswap]
  ring

/-- The graph Laplacian is self-adjoint for the mass-weighted pairing. -/
theorem laplacian_selfAdjoint (u v : X → ℝ) :
    G.weightedPair (G.laplacian u) v = G.weightedPair u (G.laplacian v) := by
  rw [G.weightedPair_comm u, G.energy_identity, G.energy_identity]
  congr 1
  apply Finset.sum_congr rfl
  intro x _
  apply Finset.sum_congr rfl
  intro y _
  ring

/-- The Laplacian's quadratic form is nonpositive. -/
theorem laplacian_nonpos (u : X → ℝ) : G.weightedPair (G.laplacian u) u ≤ 0 := by
  rw [G.energy_identity]
  apply mul_nonpos_of_nonpos_of_nonneg (by norm_num)
  apply Finset.sum_nonneg
  intro x _
  apply Finset.sum_nonneg
  intro y _
  nlinarith [G.conductance_nonneg x y, sq_nonneg (u y - u x)]

/-- Every constant function is harmonic everywhere. -/
@[simp] theorem laplacian_const (c : ℝ) : G.laplacian (fun _ ↦ c) = 0 := by
  ext x
  simp [laplacian]

/-- Laplacian linearity under addition. -/
theorem laplacian_add (u v : X → ℝ) : G.laplacian (u + v) = G.laplacian u + G.laplacian v := by
  ext x
  simp only [laplacian, Pi.add_apply]
  simp_rw [show ∀ y, G.conductance x y * (u y + v y - (u x + v x)) =
    G.conductance x y * (u y - u x) + G.conductance x y * (v y - v x) by intro y; ring]
  rw [Finset.sum_add_distrib, add_div]

/-- Laplacian linearity under real scaling. -/
theorem laplacian_smul (c : ℝ) (u : X → ℝ) : G.laplacian (c • u) = c • G.laplacian u := by
  ext x
  simp only [laplacian, Pi.smul_apply, smul_eq_mul]
  simp_rw [show ∀ y, G.conductance x y * (c * u y - c * u x) =
    c * (G.conductance x y * (u y - u x)) by intro y; ring]
  rw [← Finset.mul_sum]
  ring

/-- The Laplacian as a linear operator. -/
noncomputable def laplacianLinear : (X → ℝ) →ₗ[ℝ] (X → ℝ) where
  toFun := G.laplacian
  map_add' := G.laplacian_add
  map_smul' := G.laplacian_smul

/-- The one-step heat evolution operator. -/
noncomputable def transitionLinear : (X → ℝ) →ₗ[ℝ] (X → ℝ) :=
  LinearMap.id + G.laplacianLinear

/-- Evaluation of the heat evolution operator. -/
@[simp] theorem transitionLinear_apply (u : X → ℝ) :
    G.transitionLinear u = u + G.laplacian u := rfl

/-- Weighted self-adjointness of one-step heat evolution in the bilinear-map interface. -/
theorem transitionLinear_selfAdjoint (u v : X → ℝ) :
    G.weightedPairLinear (G.transitionLinear u) v =
      G.weightedPairLinear u (G.transitionLinear v) := by
  simp only [weightedPairLinear_apply, transitionLinear_apply,
    G.weightedPair_add_left, G.weightedPair_add_right, G.laplacian_selfAdjoint]

/-- Vanishing of the Laplacian is independent of the positive mass. -/
theorem laplacian_eq_zero_iff (u : X → ℝ) (x : X) :
    G.laplacian u x = 0 ↔ ∑ y, G.conductance x y * (u y - u x) = 0 := by
  simp [laplacian, ne_of_gt (G.mass_pos x)]

/-- Harmonicity is described using conductances alone. -/
theorem harmonicOn_iff (I : Set X) (u : X → ℝ) :
    G.HarmonicOn I u ↔ ∀ x ∈ I, ∑ y, G.conductance x y * (u y - u x) = 0 := by
  simp only [HarmonicOn, G.laplacian_eq_zero_iff]

/-- The transition probability to a distinct vertex is conductance divided by mass. -/
theorem transition_of_ne {x y : X} (h : x ≠ y) :
    G.transition x y = G.conductance x y / G.mass x := by
  simp [transition, h]

/-- The holding probability is one minus weighted degree divided by mass. -/
@[simp] theorem transition_self (x : X) :
    G.transition x x = 1 - G.degree x / G.mass x := by
  simp [transition, G.conductance_diag]

/-- Admissibility makes every transition probability nonnegative. -/
theorem transition_nonneg (h : G.Admissible) (x y : X) : 0 ≤ G.transition x y := by
  by_cases hxy : x = y
  · subst y
    rw [G.transition_self]
    exact sub_nonneg.mpr ((div_le_one (G.mass_pos x)).mpr (h x))
  · rw [G.transition_of_ne hxy]
    exact div_nonneg (G.conductance_nonneg x y) (le_of_lt (G.mass_pos x))

/-- Each transition row has total mass one. -/
theorem sum_transition (x : X) : ∑ y, G.transition x y = 1 := by
  classical
  simp only [transition, Finset.sum_sub_distrib, Finset.sum_add_distrib]
  simp [← Finset.sum_div, degree]

/-- The transition kernel acts as one plus the graph Laplacian. -/
theorem transition_mul (u : X → ℝ) (x : X) :
    ∑ y, G.transition x y * u y = G.transitionLinear u x := by
  classical
  simp only [transition, sub_mul, add_mul, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    ite_mul, zero_mul, one_mul]
  simp [transitionLinear_apply, laplacian, degree, mul_sub, Finset.sum_sub_distrib,
    Finset.sum_div, Finset.sum_mul, div_mul_eq_mul_div]
  rw [← Finset.sum_div, ← Finset.sum_div, sub_div]
  ring

/-- Detailed balance with respect to the vertex masses. -/
theorem detailed_balance (x y : X) :
    G.mass x * G.transition x y = G.mass y * G.transition y x := by
  by_cases h : x = y
  · subst y; rfl
  · rw [G.transition_of_ne h, G.transition_of_ne (Ne.symm h), G.conductance_symm x y]
    field_simp [ne_of_gt (G.mass_pos x), ne_of_gt (G.mass_pos y)]

end WeightedGraph
end VertexCentrality
