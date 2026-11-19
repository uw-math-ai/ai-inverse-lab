module

public import VertexCentrality.InfinitePath
public import VertexCentrality.GenericRank
public import VertexCentrality.RecoveryRecurrence

/-! # Reconstruction from positive hitting times on infinite paths -/

@[expose] public section

namespace VertexCentrality.WeightedGraph

open MeasureTheory
open scoped ENNReal

variable {X σ : Type*} [Fintype X] [DecidableEq X]
  [MeasurableSpace X] [MeasurableSingletonClass X]

/-- The observed probability of first arrival (or return) at positive time `n+1`,
using Mathlib's hitting time on Tau Ceti's infinite path law. -/
noncomputable def hittingProbability (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) : ℝ≥0∞ :=
  G.positivePathLaw hG x {ω | hittingAfter (pathProcess x) {y} 1 ω = (n + 1 : ℕ)}

omit [DecidableEq X] in
theorem hittingProbability_eq_firstPassageProbability (G : WeightedGraph X)
    (hG : G.Admissible) (n : ℕ) (x y : X) :
    G.hittingProbability hG n x y = G.firstPassageProbability hG n x y :=
  G.positivePathLaw_hittingAfter hG n x y

/-- The explicit visit-gap recurrence computes transition probabilities from
positive hitting-time observations, including return probabilities. -/
theorem transition_pow_eq_hitting_compositionSum (G : WeightedGraph X)
    (hG : G.Admissible) (n : ℕ) (x y : X) :
    (G.transitionMatrix ^ n) x y =
      compositionSum (fun k a b ↦ (G.hittingProbability hG k a b).toReal) n x y := by
  simp only [hittingProbability_eq_firstPassageProbability]
  exact G.transition_pow_eq_compositionSum hG n x y

/-- The response-recovery algorithm uses only boundary hitting-time observations. -/
theorem recoveredResponse_hitting_eq_heatSolution (G : WeightedGraph X)
    (hG : G.Admissible) (B : Finset X) {T : ℕ}
    (f : Fin T → B → ℝ) (t : ℕ) (x : B) :
    recoveredResponse (fun k (a b : B) ↦ (G.hittingProbability hG k a b).toReal) f t x =
      G.heatSolution B f t x := by
  simp only [hittingProbability_eq_firstPassageProbability]
  exact G.recoveredResponse_eq_heatSolution hG B f t x

/-- Corrected Euclidean projection recovery from the infinite-path observations. -/
theorem projection_eq_of_hittingProbability_eq
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.hittingProbability hG k x y = H.hittingProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) :
    (G.harmonicProductSpace B).starProjection (G.interiorMass B) =
      (G.harmonicProductSpace B).starProjection (H.interiorMass B) := by
  apply G.projection_eq_of_firstPassageProbability_eq H hG hH B T hc hm _ hucG hucH hT
  simpa only [hittingProbability_eq_firstPassageProbability] using hr

/-- Full-rank mass uniqueness from the infinite-path observations. -/
theorem mass_eq_of_hittingProbability_eq_of_rank
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ) (hconn : G.toSimpleGraph.Connected)
    (hB : (↑B : Set X).Nonempty)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.hittingProbability hG k x y = H.hittingProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T)
    (hrank : (G.harmonicProductMatrix B hconn hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) : G.mass = H.mass := by
  apply G.mass_eq_of_firstPassageProbability_eq_of_rank H hG hH B T hconn hB
    hc hm _ hucG hucH hT hrank
  simpa only [hittingProbability_eq_firstPassageProbability] using hr

/-- Generic recovery from infinite-path observations, with a full-rank witness
and the admissibility and UCP restrictions explicitly preserved. -/
theorem ae_mass_eq_of_hittingProbability_eq [Fintype σ]
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
        G.hittingProbability hG k x y = H.hittingProbability hH k x y) →
      G.UniqueContinuation B → H.UniqueContinuation B → Fintype.card X ≤ T →
      G.mass = H.mass := by
  filter_upwards [ae_harmonicProduct_full_rank B hB w G₀ hconn₀ z₀ hw₀ hrank₀] with z hz
  intro G H hG hH hconn T hw hc hm hr hucG hucH hT
  exact G.mass_eq_of_hittingProbability_eq_of_rank H hG hH B T hconn hB
    hc hm hr hucG hucH hT (hz G hconn hw)

end VertexCentrality.WeightedGraph
