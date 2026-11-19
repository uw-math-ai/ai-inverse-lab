/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Authors: Owen Parks
-/
module

public import Mathlib.Algebra.Module.LinearMap.End
public import Mathlib.Algebra.Module.Pi
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring

/-!
# Discrete forced heat evolution

The recurrence is defined on all natural times, avoiding endpoint conventions for finite
measurements. The pairing identities apply to any bilinear form for which the evolution
operator is self-adjoint, in particular the graph's mass-weighted pairing.
-/

public section

namespace VertexCentrality

open Finset

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- Zero-initial-data forced evolution for a linear one-step operator. -/
@[expose] def forcedHeat (P : E →ₗ[ℝ] E) (f : ℕ → E) : ℕ → E
  | 0 => 0
  | t + 1 => P (forcedHeat P f t) + f t

@[simp]
theorem forcedHeat_zero (P : E →ₗ[ℝ] E) (f : ℕ → E) : forcedHeat P f 0 = 0 := rfl

/-- The defining heat recurrence. -/
theorem forcedHeat_succ (P : E →ₗ[ℝ] E) (f : ℕ → E) (t : ℕ) :
    forcedHeat P f (t + 1) = P (forcedHeat P f t) + f t := rfl

/-- The recurrence and initial value determine the solution. -/
theorem forcedHeat_unique (P : E →ₗ[ℝ] E) (f u : ℕ → E) (h0 : u 0 = 0)
    (hs : ∀ t, u (t + 1) = P (u t) + f t) : u = forcedHeat P f := by
  funext t
  induction t with
  | zero => exact h0
  | succ t ih => rw [hs, forcedHeat_succ, ih]

/-- Superposition of sources. -/
theorem forcedHeat_add (P : E →ₗ[ℝ] E) (f g : ℕ → E) (t : ℕ) :
    forcedHeat P (f + g) t = forcedHeat P f t + forcedHeat P g t := by
  induction t with
  | zero => simp
  | succ t ih => simp only [forcedHeat_succ, ih, map_add, Pi.add_apply]; abel

/-- Scaling a source scales its solution. -/
theorem forcedHeat_smul (P : E →ₗ[ℝ] E) (c : ℝ) (f : ℕ → E) (t : ℕ) :
    forcedHeat P (c • f) t = c • forcedHeat P f t := by
  induction t with
  | zero => simp
  | succ t ih => simp [forcedHeat_succ, ih, smul_add]

/-- The source-to-state map at a fixed time is linear. -/
@[expose] def forcedHeatLinear (P : E →ₗ[ℝ] E) (t : ℕ) : (ℕ → E) →ₗ[ℝ] E where
  toFun f := forcedHeat P f t
  map_add' f g := forcedHeat_add P f g t
  map_smul' c f := forcedHeat_smul P c f t

/-- The discrete Duhamel representation. -/
theorem forcedHeat_eq_sum (P : E →ₗ[ℝ] E) (f : ℕ → E) (t : ℕ) :
    forcedHeat P f t = ∑ s ∈ range t, (P ^ (t - 1 - s)) (f s) := by
  induction t with
  | zero => simp
  | succ t ih =>
    rw [forcedHeat_succ, ih, map_sum, sum_range_succ]
    congr 1
    · apply sum_congr rfl
      intro s hs
      have hst : s < t := mem_range.mp hs
      have he : t + 1 - 1 - s = t - 1 - s + 1 := by omega
      rw [he, pow_succ']
      rfl
    · simp

/-- The difference of adjacent pairings along an anti-diagonal is the forcing pairing. -/
theorem forcedHeat_pairing_step (P : E →ₗ[ℝ] E)
    (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (hP : ∀ u v, B (P u) v = B u (P v))
    (f g : ℕ → E) (s t : ℕ) :
    B (forcedHeat P f (s + 1)) (forcedHeat P g t) =
      B (forcedHeat P f s) (forcedHeat P g (t + 1)) +
        B (f s) (forcedHeat P g t) - B (forcedHeat P f s) (g t) := by
  simp only [forcedHeat_succ, map_add, LinearMap.add_apply]
  rw [hP]
  ring

/-- Finite anti-diagonal telescoping, with no support assumption on the sources. -/
theorem forcedHeat_pairing_eq_sum (P : E →ₗ[ℝ] E)
    (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (hP : ∀ u v, B (P u) v = B u (P v))
    (f g : ℕ → E) (s t : ℕ) :
    B (forcedHeat P f s) (forcedHeat P g t) =
      ∑ j ∈ range s, (B (f j) (forcedHeat P g (t + s - 1 - j)) -
        B (forcedHeat P f j) (g (t + s - 1 - j))) := by
  induction s generalizing t with
  | zero => simp
  | succ s ih =>
    rw [forcedHeat_pairing_step P B hP, ih (t + 1), sum_range_succ]
    have htime : t + (s + 1) - 1 - s = t := by omega
    rw [htime]
    have hsum : (∑ j ∈ range s,
        (B (f j) (forcedHeat P g (t + 1 + s - 1 - j)) -
          B (forcedHeat P f j) (g (t + 1 + s - 1 - j)))) =
        ∑ j ∈ range s, (B (f j) (forcedHeat P g (t + (s + 1) - 1 - j)) -
          B (forcedHeat P f j) (g (t + (s + 1) - 1 - j))) := by
      apply sum_congr rfl
      intro j _
      congr 3 <;> omega
    rw [hsum]
    ring

/-- The discrete Blagovescenskii identity. The second source is zero after the control
horizon, so only measured late-time values of its solution remain. -/
theorem forcedHeat_gram (P : E →ₗ[ℝ] E)
    (B : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (hP : ∀ u v, B (P u) v = B u (P v))
    (f g : ℕ → E) (T : ℕ) (hg : ∀ t, T ≤ t → g t = 0) :
    B (forcedHeat P f T) (forcedHeat P g T) =
      ∑ j ∈ range T, B (f j) (forcedHeat P g (2 * T - 1 - j)) := by
  rw [forcedHeat_pairing_eq_sum P B hP]
  apply sum_congr rfl
  intro j hj
  have hjT : j < T := mem_range.mp hj
  have htime : T + T - 1 - j = 2 * T - 1 - j := by omega
  have hlate : T ≤ 2 * T - 1 - j := by omega
  rw [htime, hg _ hlate]
  simp

end VertexCentrality
