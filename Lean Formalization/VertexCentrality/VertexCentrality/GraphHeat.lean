/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Authors: Owen Parks
-/
module

public import VertexCentrality.Graph
public import VertexCentrality.Heat
public import Mathlib.Data.Fintype.BigOperators
public import Mathlib.LinearAlgebra.Pi
import Mathlib.Tactic

/-!
# Boundary-controlled graph heat evolution

Finite boundary controls are extended by zero in space and time. The weighted boundary
pairings retain the known vertex masses, including in the time reversal and harmonic
response identities. The graph-specific identities specialize the actual forced recurrence.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

open Finset

variable {X : Type*} [Fintype X]

/-- Restriction of a spatial function to the observation set. -/
def boundaryRestrict (B : Finset X) (u : X → ℝ) : B → ℝ := fun x ↦ u x

/-- Extension of an observed-vertex function by zero. -/
noncomputable def boundaryExtend (B : Finset X) (f : B → ℝ) (x : X) : ℝ := by
  classical
  exact if h : x ∈ B then f ⟨x, h⟩ else 0

/-- The weighted boundary pairing, with the same masses as the full spatial pairing. -/
noncomputable def boundaryPair (G : WeightedGraph X) (B : Finset X) (f g : B → ℝ) : ℝ :=
  ∑ x : B, G.mass x * f x * g x

omit [Fintype X] in
@[simp]
theorem boundaryExtend_apply (B : Finset X) (f : B → ℝ) (x : B) :
    boundaryExtend B f x = f x := by
  simp [boundaryExtend, x.property]

omit [Fintype X] in
@[simp]
theorem boundaryExtend_not_mem (B : Finset X) (f : B → ℝ) (x : X) (hx : x ∉ B) :
    boundaryExtend B f x = 0 := by
  simp [boundaryExtend, hx]

/-- Zero extension and restriction are adjoint in the weighted sum pairings. -/
theorem weightedPair_boundaryExtend (G : WeightedGraph X) (B : Finset X)
    (f : B → ℝ) (u : X → ℝ) :
    G.weightedPair (boundaryExtend B f) u =
      G.boundaryPair B f (boundaryRestrict B u) := by
  classical
  calc
    _ = ∑ x ∈ B, G.mass x * boundaryExtend B f x * u x := by
      symm
      apply sum_subset (subset_univ B)
      intro x _ hx
      simp [boundaryExtend_not_mem B f x hx]
    _ = ∑ x : B, G.mass x * boundaryExtend B f x * u x :=
      sum_subtype B (fun _ ↦ Iff.rfl) _
    _ = _ := by simp [boundaryPair, boundaryRestrict]

omit [Fintype X] in
/-- A function supported on the observation set equals its restricted zero extension. -/
theorem boundaryExtend_restrict (B : Finset X) (u : X → ℝ)
    (hu : ∀ x, x ∉ B → u x = 0) : boundaryExtend B (boundaryRestrict B u) = u := by
  classical
  funext x
  by_cases hx : x ∈ B
  · simp [boundaryExtend, boundaryRestrict, hx]
  · simp [boundaryExtend, hx, hu x hx]

/-- Restricting a pairing is valid when its first factor vanishes off the boundary. -/
theorem weightedPair_eq_boundaryPair (G : WeightedGraph X) (B : Finset X)
    (u v : X → ℝ) (hu : ∀ x, x ∉ B → u x = 0) :
    G.weightedPair u v = G.boundaryPair B (boundaryRestrict B u) (boundaryRestrict B v) := by
  calc
    G.weightedPair u v = G.weightedPair (boundaryExtend B (boundaryRestrict B u)) v := by
      rw [boundaryExtend_restrict B u hu]
    _ = _ := G.weightedPair_boundaryExtend B _ v

/-- A control on the finite time interval, extended by zero in both space and time. -/
noncomputable def extendControl (B : Finset X) {T : ℕ} (f : Fin T → B → ℝ)
    (t : ℕ) : X → ℝ :=
  if h : t < T then boundaryExtend B (f ⟨t, h⟩) else 0

omit [Fintype X] in
@[simp]
theorem extendControl_apply (B : Finset X) {T : ℕ} (f : Fin T → B → ℝ) (t : Fin T) :
    extendControl B f t = boundaryExtend B (f t) := by
  simp [extendControl, t.isLt]

omit [Fintype X] in
/-- The source is zero at and after the terminal time. -/
theorem extendControl_late (B : Finset X) {T : ℕ} (f : Fin T → B → ℝ)
    (t : ℕ) (ht : T ≤ t) : extendControl B f t = 0 := by
  simp [extendControl, Nat.not_lt.mpr ht]

/-- The actual graph heat solution with a finite observed-vertex source. -/
noncomputable def heatSolution (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f : Fin T → B → ℝ) : ℕ → X → ℝ :=
  forcedHeat G.transitionLinear (extendControl B f)

/-- The short source-to-solution response, evaluated on the observed vertices. -/
noncomputable def response (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f : Fin T → B → ℝ) : Fin T → B → ℝ :=
  fun t ↦ boundaryRestrict B (G.heatSolution B f t)

/-- Reversal on a finite time interval uses the involution `Fin.rev`. -/
def timeReverse {T : ℕ} {V : Type*} (f : Fin T → V) : Fin T → V := fun t ↦ f t.rev

@[simp]
theorem timeReverse_timeReverse {T : ℕ} {V : Type*} (f : Fin T → V) :
    timeReverse (timeReverse f) = f := by
  funext t
  simp [timeReverse]

/-- The boundary space-time pairing. -/
noncomputable def controlPair (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f g : Fin T → B → ℝ) : ℝ := ∑ t, G.boundaryPair B (f t) (g t)

/-- The graph-specific Blagovescenskii identity, including all boundary mass factors. -/
theorem heatSolution_gram (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f g : Fin T → B → ℝ) :
    G.weightedPair (G.heatSolution B f T) (G.heatSolution B g T) =
      ∑ t : Fin T, G.boundaryPair B (f t)
        (boundaryRestrict B (G.heatSolution B g (2 * T - 1 - t))) := by
  change G.weightedPairLinear
    (forcedHeat G.transitionLinear (extendControl B f) T)
    (forcedHeat G.transitionLinear (extendControl B g) T) = _
  rw [forcedHeat_gram G.transitionLinear G.weightedPairLinear
    G.transitionLinear_selfAdjoint _ _ T (extendControl_late B g)]
  rw [← Fin.sum_univ_eq_sum_range]
  apply sum_congr rfl
  intro t _
  simp only [extendControl_apply, weightedPairLinear_apply, weightedPair_boundaryExtend,
    heatSolution]


omit [Fintype X] in
/-- The boundary pairing is symmetric. -/
theorem boundaryPair_comm (G : WeightedGraph X) (B : Finset X) (f g : B → ℝ) :
    G.boundaryPair B f g = G.boundaryPair B g f := by
  unfold boundaryPair
  apply sum_congr rfl
  intro x _
  ring

omit [Fintype X] in
/-- Reversing a control twice at a valid time recovers its original boundary value. -/
theorem extendControl_reverse (B : Finset X) {T : ℕ} (f : Fin T → B → ℝ)
    (t : Fin T) :
    extendControl B (timeReverse f) (T - 1 - t) = boundaryExtend B (f t) := by
  have ht : T - 1 - (t : ℕ) = (t.rev : ℕ) := by simp [Fin.val_rev]; omega
  rw [ht, extendControl_apply]
  simp [timeReverse]

/-- Time reversal of the measured response is its adjoint for the boundary mass pairing.
This is the concrete pairing form of Appendix A's operator identity. -/
theorem response_adjoint (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f g : Fin T → B → ℝ) :
    G.controlPair B (G.response B f) g =
      G.controlPair B f (timeReverse (G.response B (timeReverse g))) := by
  have h := forcedHeat_pairing_eq_sum G.transitionLinear G.weightedPairLinear
    G.transitionLinear_selfAdjoint (extendControl B f)
    (extendControl B (timeReverse g)) T 0
  simp only [forcedHeat_zero, map_zero, Nat.zero_add] at h
  rw [← Fin.sum_univ_eq_sum_range] at h
  have hsum : (∑ j : Fin T,
      (G.weightedPairLinear (extendControl B f j)
        (forcedHeat G.transitionLinear (extendControl B (timeReverse g)) (T - 1 - j)) -
      G.weightedPairLinear (forcedHeat G.transitionLinear (extendControl B f) j)
        (extendControl B (timeReverse g) (T - 1 - j)))) =
      G.controlPair B f (timeReverse (G.response B (timeReverse g))) -
        G.controlPair B (G.response B f) g := by
    unfold controlPair
    rw [← sum_sub_distrib]
    apply sum_congr rfl
    intro t _
    simp only [extendControl_apply, extendControl_reverse, weightedPairLinear_apply]
    rw [G.weightedPair_boundaryExtend, G.weightedPair_comm _ (boundaryExtend B (g t)),
      G.weightedPair_boundaryExtend, G.boundaryPair_comm B (g t)]
    have ht : T - 1 - (t : ℕ) = (t.rev : ℕ) := by simp [Fin.val_rev]; omega
    simp only [timeReverse, response, heatSolution, ht]
  rw [hsum] at h
  linarith


private theorem forcedHeat_pairing_time_sum (G : WeightedGraph X) (f : ℕ → X → ℝ)
    (φ : X → ℝ) (n : ℕ) :
    G.weightedPair (forcedHeat G.transitionLinear f n) φ =
      ∑ t ∈ range n, (G.weightedPair (forcedHeat G.transitionLinear f t) (G.laplacian φ) +
        G.weightedPair (f t) φ) := by
  induction n with
  | zero => simp [weightedPair]
  | succ n ih =>
    rw [forcedHeat_succ, G.weightedPair_add_left, G.transitionLinear_apply,
      G.weightedPair_add_left, G.laplacian_selfAdjoint, ih, sum_range_succ]
    ring

/-- Pairing the controlled solution with an interior-harmonic function only uses boundary
measurements, boundary masses, and the boundary Laplacian of that function. -/
theorem heatSolution_harmonic_pairing (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f : Fin T → B → ℝ) (φ : X → ℝ)
    (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    G.weightedPair (G.heatSolution B f T) φ =
      G.controlPair B (G.response B f) (fun _ ↦ boundaryRestrict B (G.laplacian φ)) +
        G.controlPair B f (fun _ ↦ boundaryRestrict B φ) := by
  unfold heatSolution
  rw [forcedHeat_pairing_time_sum, ← Fin.sum_univ_eq_sum_range]
  unfold controlPair
  rw [← sum_add_distrib]
  apply sum_congr rfl
  intro t _
  rw [extendControl_apply, G.weightedPair_boundaryExtend]
  congr 1
  rw [G.weightedPair_comm _ (G.laplacian φ),
    G.weightedPair_eq_boundaryPair B (G.laplacian φ) _ (fun x hx ↦ hφ x hx),
    G.boundaryPair_comm]
  rfl

omit [Fintype X] in
/-- Additivity of the weighted control pairing in its second argument. -/
theorem controlPair_add_right (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f g h : Fin T → B → ℝ) :
    G.controlPair B f (g + h) = G.controlPair B f g + G.controlPair B f h := by
  simp [controlPair, boundaryPair, mul_add, sum_add_distrib]

/-- The concrete data-side formula in Proposition 2.7, expressed as an adjoint pairing.
The spatial boundary functions are explicitly extended constantly in time. -/
theorem heatSolution_harmonic_adjoint (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f : Fin T → B → ℝ) (φ : X → ℝ)
    (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    G.weightedPair (G.heatSolution B f T) φ =
      G.controlPair B f
        (timeReverse (G.response B (timeReverse (fun _ ↦ boundaryRestrict B (G.laplacian φ)))) +
          fun _ ↦ boundaryRestrict B φ) := by
  rw [G.controlPair_add_right, G.heatSolution_harmonic_pairing B f φ hφ,
    G.response_adjoint]


/-- Boundary restriction as a linear map. -/
def boundaryRestrictLinear (B : Finset X) : (X → ℝ) →ₗ[ℝ] (B → ℝ) where
  toFun := boundaryRestrict B
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

/-- Boundary zero extension as a linear map. -/
noncomputable def boundaryExtendLinear (B : Finset X) : (B → ℝ) →ₗ[ℝ] (X → ℝ) where
  toFun := boundaryExtend B
  map_add' f g := by
    classical
    funext x
    by_cases hx : x ∈ B <;> simp [boundaryExtend, hx]
  map_smul' c f := by
    classical
    funext x
    by_cases hx : x ∈ B <;> simp [boundaryExtend, hx]

/-- Finite controls inject linearly into natural-time spatial sources. -/
noncomputable def extendControlLinear (B : Finset X) (T : ℕ) :
    (Fin T → B → ℝ) →ₗ[ℝ] (ℕ → X → ℝ) where
  toFun := extendControl B
  map_add' f g := by
    classical
    funext t x
    by_cases ht : t < T <;> by_cases hx : x ∈ B <;>
      simp [extendControl, boundaryExtend, ht, hx]
  map_smul' c f := by
    classical
    funext t x
    by_cases ht : t < T <;> by_cases hx : x ∈ B <;>
      simp [extendControl, boundaryExtend, ht, hx]

/-- The boundary-control-to-state map at any specified time. At time `T` this is `W`. -/
noncomputable def heatSolutionLinear (G : WeightedGraph X) (B : Finset X) (T t : ℕ) :
    (Fin T → B → ℝ) →ₗ[ℝ] (X → ℝ) :=
  (forcedHeatLinear G.transitionLinear t).comp (extendControlLinear B T)

/-- Evaluation of the bundled control-to-state map. -/
@[simp] theorem heatSolutionLinear_apply (G : WeightedGraph X) (B : Finset X) (T t : ℕ)
    (f : Fin T → B → ℝ) : G.heatSolutionLinear B T t f = G.heatSolution B f t := rfl

/-- The measured short response as a linear map. -/
noncomputable def responseLinear (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    (Fin T → B → ℝ) →ₗ[ℝ] (Fin T → B → ℝ) :=
  LinearMap.pi fun t ↦ (boundaryRestrictLinear B).comp (G.heatSolutionLinear B T t)

/-- Evaluation of the bundled short response. -/
@[simp] theorem responseLinear_apply (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f : Fin T → B → ℝ) : G.responseLinear B T f = G.response B f := rfl


omit [Fintype X] in
/-- Time reversal preserves the weighted boundary space-time pairing. -/
theorem controlPair_timeReverse (G : WeightedGraph X) (B : Finset X) {T : ℕ}
    (f g : Fin T → B → ℝ) :
    G.controlPair B (timeReverse f) (timeReverse g) = G.controlPair B f g := by
  exact Equiv.sum_comp Fin.revPerm (fun t ↦ G.boundaryPair B (f t) (g t))

end VertexCentrality.WeightedGraph
