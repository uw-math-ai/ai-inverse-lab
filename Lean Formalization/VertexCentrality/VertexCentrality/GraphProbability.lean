module

public import VertexCentrality.FiniteRenewal
public import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
# Finite graph path laws

The graph transition is realized using Mathlib's `PMF`, so normalization is
part of the type. A finite path records the vertices at positive times only:
its coordinate zero is time one. This convention retains diagonal first returns.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph
open scoped ENNReal

variable {X : Type*} [Fintype X]

/-- The normalized one-step law of an admissible weighted graph. -/
noncomputable def transitionPMF (G : WeightedGraph X) (hG : G.Admissible) (x : X) : PMF X :=
  PMF.ofFintype (fun y ↦ ENNReal.ofReal (G.transition x y)) (by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun y _ ↦ G.transition_nonneg hG x y),
      G.sum_transition, ENNReal.ofReal_one])

/-- One-step probabilities agree with the graph transition entries. -/
@[simp] theorem transitionPMF_apply (G : WeightedGraph X) (hG : G.Admissible) (x y : X) :
    G.transitionPMF hG x y = ENNReal.ofReal (G.transition x y) := rfl

/-- In real coordinates the PMF gives precisely the original transition. -/
theorem transitionPMF_toReal (G : WeightedGraph X) (hG : G.Admissible) (x y : X) :
    (G.transitionPMF hG x y).toReal = G.transition x y := by
  rw [G.transitionPMF_apply, ENNReal.toReal_ofReal (G.transition_nonneg hG x y)]

/-- The finite path law records times `1, …, n`, starting from `x` at time zero. -/
noncomputable def finitePathLaw (G : WeightedGraph X) (hG : G.Admissible) :
    (n : ℕ) → X → PMF (Fin n → X)
  | 0, _ => PMF.pure Fin.elim0
  | n + 1, x => (G.transitionPMF hG x).bind fun z ↦
      (G.finitePathLaw hG n z).map (Fin.cons z)

/-- The probability of a path factors into its first step and its remaining path. -/
theorem finitePathLaw_cons (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x z : X) (w : Fin n → X) :
    G.finitePathLaw hG (n + 1) x (Fin.cons z w) =
      G.transitionPMF hG x z * G.finitePathLaw hG n z w := by
  classical
  simp [finitePathLaw, PMF.bind_apply, PMF.map_apply, Fin.cons_inj, tsum_fintype, ite_and, mul_ite]

/-- First arrival at the last positive time of a finite path. In particular the
starting vertex is not tested, so the diagonal event is first return. -/
def FirstPassageAtEnd {n : ℕ} (y : X) (w : Fin (n + 1) → X) : Prop :=
  w (Fin.last n) = y ∧ ∀ i : Fin n, w i.castSucc ≠ y

/-- Observed positive-time first-passage probability, indexed by time `n + 1`. -/
noncomputable def firstPassageProbability (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) : ℝ≥0∞ := by
  classical
  exact ∑ w : Fin (n + 1) → X,
    if FirstPassageAtEnd y w then G.finitePathLaw hG (n + 1) x w else 0

omit [Fintype X] in
/-- A longer first-passage path avoids the target on its first step and first
arrives at the end of its suffix. -/
theorem firstPassageAtEnd_cons (n : ℕ) (y z : X) (w : Fin (n + 1) → X) :
    FirstPassageAtEnd y (Fin.cons z w) ↔ z ≠ y ∧ FirstPassageAtEnd y w := by
  simp [FirstPassageAtEnd, Fin.forall_fin_succ, and_left_comm]

/-- The event probability at time one is the transition probability, even for a return. -/
theorem firstPassageProbability_zero (G : WeightedGraph X) (hG : G.Admissible) (x y : X) :
    G.firstPassageProbability hG 0 x y = G.transitionPMF hG x y := by
  classical
  unfold firstPassageProbability
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin 1 ↦ X)), Fintype.sum_prod_type]
  change (∑ z, ∑ w : Fin 0 → X, if FirstPassageAtEnd y (Fin.cons z w) then
    G.finitePathLaw hG 1 x (Fin.cons z w) else 0) = _
  simp only [finitePathLaw_cons]
  have he : ∀ w : Fin 0 → X, w = Fin.elim0 := fun _ ↦ Subsingleton.elim _ _
  simp [FirstPassageAtEnd, finitePathLaw, he]

/-- Finite first-passage probabilities satisfy the killed first-step recursion. -/
theorem firstPassageProbability_succ [DecidableEq X]
    (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) :
    G.firstPassageProbability hG (n + 1) x y =
      ∑ z, if z = y then 0 else
        G.transitionPMF hG x z * G.firstPassageProbability hG n z y := by
  classical
  unfold firstPassageProbability
  rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (n + 2) ↦ X)),
    Fintype.sum_prod_type]
  change (∑ z, ∑ w : Fin (n + 1) → X,
    if FirstPassageAtEnd y (Fin.cons z w) then
      G.finitePathLaw hG (n + 2) x (Fin.cons z w) else 0) = _
  apply Finset.sum_congr rfl
  intro z _
  simp only [firstPassageAtEnd_cons, finitePathLaw_cons]
  by_cases hz : z = y
  · simp [hz]
  · simp [hz, Finset.mul_sum, mul_ite]

/-- The actual finite-path first-passage event has precisely the killed-matrix
weight. This includes positive-time first returns when the initial and target
vertices coincide. -/
theorem firstPassageProbability_eq_weight [DecidableEq X]
    (G : WeightedGraph X) (hG : G.Admissible) (n : ℕ) (x y : X) :
    G.firstPassageProbability hG n x y = ENNReal.ofReal (G.firstPassageWeight n x y) := by
  induction n generalizing x with
  | zero => rw [G.firstPassageProbability_zero, G.firstPassageWeight_zero,
      G.transitionPMF_apply]
  | succ n ih =>
    rw [G.firstPassageProbability_succ, G.firstPassageWeight_succ,
      ENNReal.ofReal_sum_of_nonneg]
    · apply Finset.sum_congr rfl
      intro z _
      by_cases hz : z = y
      · simp [hz]
      · simp only [ite_eq_right hz, ih, G.transitionPMF_apply,
          ENNReal.ofReal_mul (G.transition_nonneg hG x z)]
    · intro z _
      split_ifs
      · exact le_rfl
      · exact mul_nonneg (G.transition_nonneg hG x z) (G.firstPassageWeight_nonneg hG n z y)

/-- Real first-passage probabilities equal the algebraic weights in finite renewal. -/
theorem firstPassageProbability_toReal [DecidableEq X]
    (G : WeightedGraph X) (hG : G.Admissible) (n : ℕ) (x y : X) :
    (G.firstPassageProbability hG n x y).toReal = G.firstPassageWeight n x y := by
  rw [G.firstPassageProbability_eq_weight,
    ENNReal.toReal_ofReal (G.firstPassageWeight_nonneg hG n x y)]

/-- Finite-path first-passage probabilities determine observed matrix powers. -/
theorem transition_powers_eq_of_firstPassageProbability_eq [DecidableEq X]
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (N : ℕ) (hr : ∀ k < N, ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y) :
    ∀ n ≤ N, ∀ x y : B,
      (G.transitionMatrix ^ n) x y = (H.transitionMatrix ^ n) x y := by
  apply G.transition_powers_eq_of_firstPassageWeight_eq H B N
  intro k hk x y
  have h := congrArg ENNReal.toReal (hr k hk x y)
  simpa only [firstPassageProbability_toReal] using h

/-- Observed first-passage probabilities also determine the short heat response. -/
theorem responseLinear_eq_of_firstPassageProbability_eq [DecidableEq X]
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y) :
    G.responseLinear B T = H.responseLinear B T := by
  apply G.responseLinear_eq_of_powers_eq H B T
  intro n hn
  exact G.transition_powers_eq_of_firstPassageProbability_eq H hG hH B (2 * T - 1)
    (fun k hk ↦ hr k (by omega)) n (by omega)

/-- Observed first-passage probabilities determine the extended heat response
within the paper's strict positive-time observation horizon. -/
theorem extendedResponseLinear_eq_of_firstPassageProbability_eq [DecidableEq X]
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y) :
    G.extendedResponseLinear B T = H.extendedResponseLinear B T := by
  apply G.extendedResponseLinear_eq_of_firstPassageWeight_eq H B T
  intro k hk x y
  have h := congrArg ENNReal.toReal (hr k hk x y)
  simpa only [firstPassageProbability_toReal] using h

end VertexCentrality.WeightedGraph
