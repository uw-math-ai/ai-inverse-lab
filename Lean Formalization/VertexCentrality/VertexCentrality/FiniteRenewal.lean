module

public import VertexCentrality.ResponseData
import Mathlib.Tactic

/-!
# Finite renewal algebra

Killing a transition on arrival at a chosen vertex records avoidance at all
positive intermediate times. Its powers give finite first-passage weights,
including positive returns on the diagonal. The renewal identity below is
proved algebraically, not postulated. Identification with a path probability
law is a separate obligation.
-/

@[expose] public section

namespace VertexCentrality

/-- Noncommutative telescoping for two transition operators. -/
theorem pow_eq_pow_add_sum {R : Type*} [Ring R] (P Q : R) (n : ℕ) :
    P ^ n = Q ^ n + ∑ k ∈ Finset.range n, Q ^ k * (P - Q) * P ^ (n - 1 - k) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ P, ih, add_mul, Finset.sum_mul, Finset.sum_range_succ]
    have hs : (∑ k ∈ Finset.range n, Q ^ k * (P - Q) * P ^ (n - 1 - k) * P) =
        ∑ k ∈ Finset.range n, Q ^ k * (P - Q) * P ^ (n + 1 - 1 - k) := by
      apply Finset.sum_congr rfl
      intro k hk
      have h : n + 1 - 1 - k = (n - 1 - k) + 1 := by
        simp only [Finset.mem_range] at hk
        omega
      rw [h, pow_succ, mul_assoc]
    rw [hs, pow_succ Q]
    simp only [Nat.add_sub_cancel, Nat.sub_self, pow_zero, mul_one]
    noncomm_ring

namespace WeightedGraph

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- Transition killed on arrival at `y`; time zero is deliberately not tested. -/
noncomputable def avoidingTransition (G : WeightedGraph X) (y : X) : Matrix X X ℝ :=
  fun x z ↦ if z = y then 0 else G.transitionMatrix x z

/-- Finite first-passage weight: avoid `y` for `n` steps, then arrive there.
The argument `n` indexes the positive hitting time `n + 1`. -/
noncomputable def firstPassageWeight (G : WeightedGraph X) (n : ℕ) (x y : X) : ℝ :=
  ((G.avoidingTransition y ^ n) * G.transitionMatrix) x y

/-- At positive time one, first passage is exactly the one-step transition,
including the diagonal first-return case. -/
@[simp] theorem firstPassageWeight_zero (G : WeightedGraph X) (x y : X) :
    G.firstPassageWeight 0 x y = G.transition x y := by
  simp only [firstPassageWeight, pow_zero, one_mul]
  rfl

/-- Algebraic first-passage weights obey the killed first-step recursion. -/
theorem firstPassageWeight_succ (G : WeightedGraph X) (n : ℕ) (x y : X) :
    G.firstPassageWeight (n + 1) x y =
      ∑ z, if z = y then 0 else G.transition x z * G.firstPassageWeight n z y := by
  unfold firstPassageWeight
  rw [pow_succ', mul_assoc, Matrix.mul_apply]
  apply Finset.sum_congr rfl
  intro z _
  by_cases hz : z = y <;> simp [avoidingTransition, transitionMatrix, hz]

/-- Admissibility makes the finite first-passage weights nonnegative. -/
theorem firstPassageWeight_nonneg (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) : 0 ≤ G.firstPassageWeight n x y := by
  induction n generalizing x with
  | zero => simpa using G.transition_nonneg hG x y
  | succ n ih =>
    rw [G.firstPassageWeight_succ]
    apply Finset.sum_nonneg
    intro z _
    split_ifs
    · exact le_rfl
    · exact mul_nonneg (G.transition_nonneg hG x z) (ih z)

/-- A killed power has zero mass at the killed vertex after any positive time. -/
theorem avoidingTransition_pow_succ (G : WeightedGraph X) (n : ℕ) (x y : X) :
    (G.avoidingTransition y ^ (n + 1)) x y = 0 := by
  rw [pow_succ, Matrix.mul_apply]
  simp [avoidingTransition]

/-- The telescoping term factors into first passage followed by return. -/
theorem renewal_term (G : WeightedGraph X) (k n : ℕ) (x y : X) :
    ((G.avoidingTransition y ^ k) * (G.transitionMatrix - G.avoidingTransition y) *
      G.transitionMatrix ^ n) x y =
      G.firstPassageWeight k x y * (G.transitionMatrix ^ n) y y := by
  have hd (z w : X) : (G.transitionMatrix - G.avoidingTransition y) z w =
      if w = y then G.transitionMatrix z y else 0 := by
    by_cases hw : w = y <;> simp [avoidingTransition, hw]
  rw [Matrix.mul_apply]
  simp_rw [Matrix.mul_apply, hd]
  simp [firstPassageWeight, Matrix.mul_apply]

/-- Finite positive-time renewal, valid also for diagonal first returns. -/
theorem transition_pow_renewal (G : WeightedGraph X) (n : ℕ) (x y : X) :
    (G.transitionMatrix ^ (n + 1)) x y =
      ∑ k ∈ Finset.range (n + 1),
        G.firstPassageWeight k x y * (G.transitionMatrix ^ (n - k)) y y := by
  have h := congrArg (fun M : Matrix X X ℝ ↦ M x y)
    (pow_eq_pow_add_sum G.transitionMatrix (G.avoidingTransition y) (n + 1))
  simpa only [Matrix.add_apply, G.avoidingTransition_pow_succ, zero_add,
    Matrix.sum_apply, Nat.add_sub_cancel, G.renewal_term] using h

/-- Finite first-passage weights recursively determine all observed transition
powers through the same positive-time horizon. -/
theorem transition_powers_eq_of_firstPassageWeight_eq (G H : WeightedGraph X)
    (B : Finset X) (N : ℕ) (hr : ∀ k < N, ∀ x y : B,
      G.firstPassageWeight k x y = H.firstPassageWeight k x y) :
    ∀ n ≤ N, ∀ x y : B,
      (G.transitionMatrix ^ n) x y = (H.transitionMatrix ^ n) x y := by
  intro n
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro hn x y
    cases n with
    | zero => simp
    | succ n =>
      rw [G.transition_pow_renewal, H.transition_pow_renewal]
      apply Finset.sum_congr rfl
      intro k hk
      have hk' := Finset.mem_range.mp hk
      rw [hr k (by omega) x y, ih (n - k) (by omega) (by omega) y y]

/-- The extended response is determined by finite first-passage weights.
The separate path-law identification is required to interpret these as observations. -/
theorem extendedResponseLinear_eq_of_firstPassageWeight_eq (G H : WeightedGraph X)
    (B : Finset X) (T : ℕ) (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageWeight k x y = H.firstPassageWeight k x y) :
    G.extendedResponseLinear B T = H.extendedResponseLinear B T := by
  apply G.extendedResponseLinear_eq_of_powers_eq H B T
  intro n hn
  exact G.transition_powers_eq_of_firstPassageWeight_eq H B (2 * T - 1)
    (fun k hk ↦ hr k (by omega)) n (by omega)

end WeightedGraph
end VertexCentrality
