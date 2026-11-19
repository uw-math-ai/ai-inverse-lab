module

public import VertexCentrality.GraphProbability
public import Mathlib.Combinatorics.Enumerative.Composition

/-!
# Recovering transition powers by renewal and compositions

The input `r k x y` is the positive-time first-passage probability at time `k+1`.
The renewal recurrence computes transition entries without interior information.
The composition sum expands repeated returns, with Mathlib compositions recording
the successive positive gaps between visits.
-/

@[expose] public section

namespace VertexCentrality

variable {X : Type*} [DecidableEq X]

/-- The finite renewal reconstruction algorithm, including the time-zero identity. -/
def recoverTransition (r : ℕ → X → X → ℝ) : ℕ → X → X → ℝ
  | 0, x, y => if x = y then 1 else 0
  | n + 1, x, y => ∑ k : Fin (n + 1), r k x y * recoverTransition r (n - k) y y
termination_by n => n

/-- Recovering through time `N` only reads positive first-passage times through `N`. -/
theorem recoverTransition_congr (r s : ℕ → X → X → ℝ) (N : ℕ)
    (hrs : ∀ k < N, ∀ x y, r k x y = s k x y) :
    ∀ n ≤ N, ∀ x y, recoverTransition r n x y = recoverTransition s n x y := by
  intro n
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro hn x y
    cases n with
    | zero => rw [recoverTransition, recoverTransition]
    | succ n =>
      rw [recoverTransition, recoverTransition]
      apply Finset.sum_congr rfl
      intro k _
      rw [hrs k (by have := k.isLt; omega), ih (n - k) (by omega) (by omega)]

/-- The contribution of successive positive gaps between visits to the target. -/
def compositionWeight (r : ℕ → X → X → ℝ) : X → X → List ℕ → ℝ
  | x, y, [] => if x = y then 1 else 0
  | x, y, a :: as => r (a - 1) x y * compositionWeight r y y as

/-- The paper's visit-gap expansion, expressed using Mathlib's compositions. -/
noncomputable def compositionSum (r : ℕ → X → X → ℝ) (n : ℕ) (x y : X) : ℝ :=
  ∑ c : Composition n, compositionWeight r x y c.blocks

/-- On a nonempty composition the first factor is a first arrival and every
remaining factor is a positive return, exactly as in the paper's visit expansion. -/
theorem compositionWeight_cons_eq_prod (r : ℕ → X → X → ℝ) (x y : X)
    (a : ℕ) (as : List ℕ) :
    compositionWeight r x y (a :: as) =
      r (a - 1) x y * (as.map (fun b ↦ r (b - 1) y y)).prod := by
  have hd : ∀ bs : List ℕ, compositionWeight r y y bs =
      (bs.map (fun b ↦ r (b - 1) y y)).prod := by
    intro bs
    induction bs with
    | nil => simp [compositionWeight]
    | cons b bs ih => simp [compositionWeight, ih]
  rw [compositionWeight, hd]

/-- Prepending the first positive gap to a composition of the remaining time. -/
def prependComposition (n : ℕ) (p : Σ k : Fin (n + 1), Composition (n - k)) :
    Composition (n + 1) where
  blocks := (p.1.val + 1) :: p.2.blocks
  blocks_pos := by
    intro i hi
    rcases List.mem_cons.mp hi with rfl | hi
    · omega
    · exact p.2.blocks_pos hi
  blocks_sum := by
    simp only [List.sum_cons, Composition.blocks_sum]
    have := p.1.isLt
    omega

/-- Every positive-time composition has a unique first gap and remaining composition. -/
theorem prependComposition_bijective (n : ℕ) : Function.Bijective (prependComposition n) := by
  constructor
  · rintro ⟨i, c⟩ ⟨j, d⟩ h
    have hb := congrArg Composition.blocks h
    change (i.val + 1) :: c.blocks = (j.val + 1) :: d.blocks at hb
    have hij : i = j := Fin.ext (by have := (List.cons.inj hb).1; omega)
    subst j
    have hcd : c = d := Composition.ext (List.cons.inj hb).2
    subst d
    rfl
  · intro c
    rcases c with ⟨blocks, hp, hs⟩
    cases blocks with
    | nil => simp at hs
    | cons a as =>
      have ha : 0 < a := hp (by simp)
      have hsum : a + as.sum = n + 1 := by simpa using hs
      let k : Fin (n + 1) := ⟨a - 1, by omega⟩
      let d : Composition (n - k) := ⟨as, fun hi ↦ hp (List.mem_cons_of_mem a hi), by
        dsimp [k]; omega⟩
      refine ⟨⟨k, d⟩, ?_⟩
      apply Composition.ext
      change (a - 1 + 1) :: as = a :: as
      congr 1
      omega

/-- First-gap decomposition of Mathlib compositions. -/
noncomputable def prependCompositionEquiv (n : ℕ) :
    (Σ k : Fin (n + 1), Composition (n - k)) ≃ Composition (n + 1) :=
  Equiv.ofBijective (prependComposition n) (prependComposition_bijective n)

/-- The zero-time composition contributes the identity entry. -/
theorem compositionSum_zero (r : ℕ → X → X → ℝ) (x y : X) :
    compositionSum r 0 x y = if x = y then 1 else 0 := by
  have hb (c : Composition 0) : c.blocks = [] := c.blocks_eq_nil.mpr rfl
  simp [compositionSum, hb, compositionWeight, composition_card]

/-- Splitting off the first gap turns the composition expansion into finite renewal. -/
theorem compositionSum_succ (r : ℕ → X → X → ℝ) (n : ℕ) (x y : X) :
    compositionSum r (n + 1) x y =
      ∑ k : Fin (n + 1), r k x y * compositionSum r (n - k) y y := by
  unfold compositionSum
  rw [← Equiv.sum_comp (prependCompositionEquiv n), Fintype.sum_sigma]
  simp [prependCompositionEquiv, prependComposition, compositionWeight, Finset.mul_sum]

/-- The recursive recovery algorithm equals the full positive-gap composition sum. -/
theorem recoverTransition_eq_compositionSum (r : ℕ → X → X → ℝ) (n : ℕ) (x y : X) :
    recoverTransition r n x y = compositionSum r n x y := by
  induction n using Nat.strong_induction_on generalizing x y with
  | h n ih =>
    cases n with
    | zero => rw [recoverTransition, compositionSum_zero]
    | succ n =>
      rw [recoverTransition, compositionSum_succ]
      apply Finset.sum_congr rfl
      intro k _
      rw [ih (n - k) (by omega)]

namespace WeightedGraph

variable [Fintype X]

omit [Fintype X] in
/-- Recovery on a subset only uses first-passage entries on that subset. -/
theorem recoverTransition_restrict (r : ℕ → X → X → ℝ) (B : Finset X)
    (n : ℕ) (x y : B) :
    recoverTransition (fun (k : ℕ) (a b : B) ↦ r k a b) n x y = recoverTransition r n x y := by
  induction n using Nat.strong_induction_on generalizing x y with
  | h n ih =>
    cases n with
    | zero =>
      by_cases h : x = y
      · subst y; simp [recoverTransition]
      · have hv : (x : X) ≠ y := fun he ↦ h (Subtype.ext he)
        simp only [recoverTransition, ite_eq_right h, ite_eq_right hv]
    | succ n =>
      rw [recoverTransition, recoverTransition]
      apply Finset.sum_congr rfl
      intro k _
      rw [ih (n - k) (by omega)]

/-- The finite renewal algorithm recovers the actual graph transition powers. -/
theorem recoverTransition_eq_transition_pow (G : WeightedGraph X) (n : ℕ) (x y : X) :
    recoverTransition G.firstPassageWeight n x y = (G.transitionMatrix ^ n) x y := by
  induction n using Nat.strong_induction_on generalizing x y with
  | h n ih =>
    cases n with
    | zero => simp [recoverTransition, Matrix.one_apply]
    | succ n =>
      rw [recoverTransition, G.transition_pow_renewal, ← Fin.sum_univ_eq_sum_range]
      apply Finset.sum_congr rfl
      intro k _
      rw [ih (n - k) (by omega)]

/-- The normalized finite-path probability input gives exactly the same reconstruction. -/
theorem recoverTransition_firstPassageProbability (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) :
    recoverTransition (fun k a b ↦ (G.firstPassageProbability hG k a b).toReal) n x y =
      (G.transitionMatrix ^ n) x y := by
  simp only [G.firstPassageProbability_toReal]
  exact G.recoverTransition_eq_transition_pow n x y

/-- The visit-gap composition formula for finite-path first-passage observations. -/
theorem transition_pow_eq_compositionSum (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) :
    (G.transitionMatrix ^ n) x y =
      compositionSum (fun k a b ↦ (G.firstPassageProbability hG k a b).toReal) n x y := by
  rw [← G.recoverTransition_firstPassageProbability hG, recoverTransition_eq_compositionSum]

/-- Boundary-only first-passage input recovers the observed transition matrix entries. -/
theorem recoverTransition_boundary_probability (G : WeightedGraph X) (hG : G.Admissible)
    (B : Finset X) (n : ℕ) (x y : B) :
    recoverTransition (fun k (a b : B) ↦ (G.firstPassageProbability hG k a b).toReal) n x y =
      (G.transitionMatrix ^ n) x y := by
  exact (recoverTransition_restrict
    (fun k (a b : X) ↦ (G.firstPassageProbability hG k a b).toReal) B n x y).trans
    (G.recoverTransition_firstPassageProbability hG n x y)

/-- Explicit response reconstruction from the observed first-passage array only. -/
noncomputable def recoveredResponse {B : Finset X} (r : ℕ → B → B → ℝ)
    {T : ℕ} (f : Fin T → B → ℝ) (t : ℕ) (x : B) : ℝ :=
  ∑ s ∈ Finset.range t, if hs : s < T then
    ∑ y : B, recoverTransition r (t - 1 - s) x y * f ⟨s, hs⟩ y else 0

omit [Fintype X] in
/-- Finite data suffice for response reconstruction through time `N+1`. -/
theorem recoveredResponse_congr {B : Finset X} (r s : ℕ → B → B → ℝ) (N : ℕ)
    (hrs : ∀ k < N, ∀ x y, r k x y = s k x y)
    {T : ℕ} (f : Fin T → B → ℝ) (t : ℕ) (ht : t ≤ N + 1) (x : B) :
    recoveredResponse r f t x = recoveredResponse s f t x := by
  unfold recoveredResponse
  apply Finset.sum_congr rfl
  intro k hk
  by_cases hkT : k < T
  · simp only [dite_eq_left hkT]
    apply Finset.sum_congr rfl
    intro y _
    rw [recoverTransition_congr r s N hrs (t - 1 - k) (by omega)]
  · simp only [dite_eq_right hkT]

/-- The reconstructed response equals the actual boundary heat solution. -/
theorem recoveredResponse_eq_heatSolution (G : WeightedGraph X) (hG : G.Admissible)
    (B : Finset X) {T : ℕ} (f : Fin T → B → ℝ) (t : ℕ) (x : B) :
    recoveredResponse (fun k (a b : B) ↦ (G.firstPassageProbability hG k a b).toReal) f t x =
      G.heatSolution B f t x := by
  rw [G.heatSolution_eq_boundary_sum]
  unfold recoveredResponse
  simp only [G.recoverTransition_boundary_probability]

end WeightedGraph
end VertexCentrality
