module

public import VertexCentrality.GraphHeat
public import Mathlib.LinearAlgebra.Matrix.ToLin
public import Mathlib.LinearAlgebra.Matrix.Stochastic

/-!
# Boundary transition powers determine the heat response

This is the deterministic part of the first-passage/data bridge. We prove
that equality of observed transition powers implies equality of the concrete
short and extended heat responses. The probabilistic renewal theorem must still
supply these powers from positive-time first-passage laws.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- The graph transition entries as a matrix in ordinary vertex value coordinates. -/
noncomputable def transitionMatrix (G : WeightedGraph X) : Matrix X X ℝ := G.transition

/-- Admissibility places the concrete transition in Mathlib's stochastic submonoid. -/
theorem transitionMatrix_mem_rowStochastic (G : WeightedGraph X) (hG : G.Admissible) :
    G.transitionMatrix ∈ Matrix.rowStochastic ℝ X :=
  Matrix.mem_rowStochastic_iff_sum.mpr ⟨G.transition_nonneg hG, G.sum_transition⟩

/-- Every finite transition power is stochastic, by the existing submonoid API. -/
theorem transitionMatrix_pow_mem_rowStochastic (G : WeightedGraph X)
    (hG : G.Admissible) (n : ℕ) :
    G.transitionMatrix ^ n ∈ Matrix.rowStochastic ℝ X :=
  (Matrix.rowStochastic ℝ X).pow_mem (G.transitionMatrix_mem_rowStochastic hG) n

/-- Mathlib's matrix-to-map conversion gives the actual heat transition operator. -/
theorem transitionMatrix_toLin (G : WeightedGraph X) :
    G.transitionMatrix.toLin' = G.transitionLinear := by
  apply LinearMap.ext
  intro u
  funext x
  exact G.transition_mul u x

/-- Matrix powers act as the iterated heat transition map. -/
theorem transitionMatrix_pow_apply (G : WeightedGraph X) (n : ℕ) (u : X → ℝ) (x : X) :
    ((G.transitionLinear ^ n) u) x = ∑ y, (G.transitionMatrix ^ n) x y * u y := by
  rw [← G.transitionMatrix_toLin, ← Matrix.toLin'_pow]
  rfl

/-- Propagating a boundary source uses only columns indexed by the boundary. -/
theorem transition_pow_boundaryExtend (G : WeightedGraph X) (B : Finset X)
    (n : ℕ) (f : B → ℝ) (x : X) :
    ((G.transitionLinear ^ n) (boundaryExtend B f)) x =
      ∑ y : B, (G.transitionMatrix ^ n) x y * f y := by
  rw [G.transitionMatrix_pow_apply]
  calc
    _ = ∑ y ∈ B, (G.transitionMatrix ^ n) x y * boundaryExtend B f y := by
      symm
      apply Finset.sum_subset (Finset.subset_univ B)
      intro y _ hy
      simp [boundaryExtend_not_mem B f y hy]
    _ = ∑ y : B, (G.transitionMatrix ^ n) x y * boundaryExtend B f y :=
      Finset.sum_subtype B (fun _ ↦ Iff.rfl) _
    _ = _ := by simp only [boundaryExtend_apply]

/-- The explicit finite-data Duhamel formula at any time, including after the
source switches off. Only observed matrix entries occur when `x` is on the boundary. -/
theorem heatSolution_eq_boundary_sum (G : WeightedGraph X) (B : Finset X)
    {T : ℕ} (f : Fin T → B → ℝ) (t : ℕ) (x : X) :
    G.heatSolution B f t x =
      ∑ s ∈ Finset.range t, if hs : s < T then
        ∑ y : B, (G.transitionMatrix ^ (t - 1 - s)) x y * f ⟨s, hs⟩ y else 0 := by
  classical
  rw [heatSolution, forcedHeat_eq_sum]
  simp only [Finset.sum_apply]
  apply Finset.sum_congr rfl
  intro s _
  by_cases hs : s < T
  · simp only [extendControl, dite_eq_left hs]
    exact G.transition_pow_boundaryExtend B _ _ x
  · simp [extendControl, hs]

/-- Equal observed powers through time `N` give equal boundary solutions through
`N + 1`, for every source horizon. No equality of interior transition entries is required. -/
theorem heatSolution_boundary_eq_of_powers_eq (G H : WeightedGraph X) (B : Finset X)
    (N : ℕ) (hp : ∀ n ≤ N, ∀ x y : B,
      (G.transitionMatrix ^ n) x y = (H.transitionMatrix ^ n) x y)
    {T : ℕ} (f : Fin T → B → ℝ) (t : ℕ) (ht : t ≤ N + 1) (x : B) :
    G.heatSolution B f t x = H.heatSolution B f t x := by
  rw [G.heatSolution_eq_boundary_sum, H.heatSolution_eq_boundary_sum]
  apply Finset.sum_congr rfl
  intro s hs
  by_cases hsT : s < T
  · simp only [dite_eq_left hsT]
    apply Finset.sum_congr rfl
    intro y _
    rw [hp (t - 1 - s) (by omega) x y]
  · simp only [dite_eq_right hsT]

/-- Equality of boundary transition powers determines the short response operator. -/
theorem responseLinear_eq_of_powers_eq (G H : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hp : ∀ n < T, ∀ x y : B,
      (G.transitionMatrix ^ n) x y = (H.transitionMatrix ^ n) x y) :
    G.responseLinear B T = H.responseLinear B T := by
  apply LinearMap.ext
  intro f
  funext t x
  exact G.heatSolution_boundary_eq_of_powers_eq H B t
    (fun n hn ↦ hp n (lt_of_le_of_lt hn t.isLt)) f t (by omega) x

/-- The extended measured response has the same finite source horizon as `W`. -/
noncomputable def extendedResponseLinear (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    (Fin T → B → ℝ) →ₗ[ℝ] (Fin (2 * T) → B → ℝ) :=
  LinearMap.pi fun t ↦ (boundaryRestrictLinear B).comp (G.heatSolutionLinear B T t)

/-- Powers strictly below `2T` suffice for the extended response used by the Gram identity. -/
theorem extendedResponseLinear_eq_of_powers_eq (G H : WeightedGraph X)
    (B : Finset X) (T : ℕ) (hp : ∀ n < 2 * T, ∀ x y : B,
      (G.transitionMatrix ^ n) x y = (H.transitionMatrix ^ n) x y) :
    G.extendedResponseLinear B T = H.extendedResponseLinear B T := by
  apply LinearMap.ext
  intro f
  funext t x
  exact G.heatSolution_boundary_eq_of_powers_eq H B t
    (fun n hn ↦ hp n (lt_of_le_of_lt hn t.isLt)) f t (by omega) x

end VertexCentrality.WeightedGraph
