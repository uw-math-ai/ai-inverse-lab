module

public import VertexCentrality.GraphProbability
public import TauCeti.Probability.Process.MarkovChain
public import Mathlib.Probability.Process.HittingTime

/-! # Infinite graph paths from Tau Ceti's Markov-chain law

The path coordinates record positive times; time zero is the fixed initial
vertex. This agrees with the existing finite-path indexing.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {X : Type*} [Fintype X] [MeasurableSpace X] [MeasurableSingletonClass X]

/-- The graph transition as a normalized measurable kernel. -/
noncomputable def transitionKernel (G : WeightedGraph X) (hG : G.Admissible) : Kernel X X :=
  Kernel.ofFunOfCountable fun x ↦ (G.transitionPMF hG x).toMeasure

instance transitionKernel_isMarkov (G : WeightedGraph X) (hG : G.Admissible) :
    IsMarkovKernel (G.transitionKernel hG) where
  isProbabilityMeasure x := by
    change IsProbabilityMeasure (G.transitionPMF hG x).toMeasure
    infer_instance

@[simp] theorem transitionKernel_singleton (G : WeightedGraph X) (hG : G.Admissible)
    (x y : X) : G.transitionKernel hG x {y} = G.transitionPMF hG x y :=
  PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _)

/-- Tau Ceti's infinite Markov law on the states at times `1, 2, …`. -/
noncomputable def positivePathLaw (G : WeightedGraph X) (hG : G.Admissible) (x : X) :
    Measure (ℕ → X) :=
  TauCeti.Probability.markovChainLaw (G.transitionPMF hG x).toMeasure (G.transitionKernel hG)

instance positivePathLaw_isProbability (G : WeightedGraph X) (hG : G.Admissible) (x : X) :
    IsProbabilityMeasure (G.positivePathLaw hG x) := by
  unfold positivePathLaw
  infer_instance

omit [MeasurableSpace X] [MeasurableSingletonClass X] in
/-- Finite path probabilities have the usual product formula. -/
theorem finitePathLaw_eq_product (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x : X) (w : Fin (n + 1) → X) :
    G.finitePathLaw hG (n + 1) x w =
      G.transitionPMF hG x (w 0) *
        ∏ i : Fin n, G.transitionPMF hG (w i.castSucc) (w i.succ) := by
  induction n generalizing x with
  | zero =>
    rw [← Fin.cons_self_tail w, G.finitePathLaw_cons]
    have ht : Fin.tail w = Fin.elim0 := Subsingleton.elim _ _
    simp [finitePathLaw, ht]
  | succ n ih =>
    rw [← Fin.cons_self_tail w, G.finitePathLaw_cons, ih]
    simp [Fin.prod_univ_succ, Fin.tail]

/-- Every positive-time prefix of the infinite law has exactly the existing
finite-path mass. -/
theorem positivePathLaw_prefix_singleton (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x : X) (w : Fin (n + 1) → X) :
    ((G.positivePathLaw hG x).map fun ω (i : Fin (n + 1)) ↦ ω i.val) {w} =
      G.finitePathLaw hG (n + 1) x w := by
  rw [positivePathLaw, TauCeti.Probability.markovChainLaw_map_prefix_apply_singleton,
    PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _), G.finitePathLaw_eq_product]
  simp only [transitionKernel_singleton]

/-- Prefix laws agree as measures, not only at individual paths. -/
theorem positivePathLaw_map_prefix (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x : X) :
    (G.positivePathLaw hG x).map (fun ω (i : Fin (n + 1)) ↦ ω i.val) =
      (G.finitePathLaw hG (n + 1) x).toMeasure := by
  apply Measure.ext_of_singleton
  intro w
  rw [G.positivePathLaw_prefix_singleton, PMF.toMeasure_apply_singleton _ _
    (measurableSet_singleton _)]

/-- The coordinate process with its prescribed time-zero vertex. -/
def pathProcess (x : X) : ℕ → (ℕ → X) → X
  | 0, _ => x
  | n + 1, ω => ω n

omit [Fintype X] [MeasurableSpace X] [MeasurableSingletonClass X] in
/-- Mathlib's hitting time after time one is precisely our positive first-passage
prefix event. This includes a first return when the target is the initial vertex. -/
theorem hittingAfter_pathProcess_iff (x y : X) (n : ℕ) (ω : ℕ → X) :
    hittingAfter (pathProcess x) {y} 1 ω = (n + 1 : ℕ) ↔
      FirstPassageAtEnd y (fun i : Fin (n + 1) ↦ ω i.val) := by
  classical
  rw [hittingAfter_def]
  dsimp only
  split_ifs with h
  · change (WithTop.some (sInf {j : ℕ | 1 ≤ j ∧ pathProcess x j ω ∈ ({y} : Set X)}) = WithTop.some (n + 1)) ↔ _
    have hs : ({j : ℕ | 1 ≤ j ∧ pathProcess x j ω ∈ ({y} : Set X)} : Set ℕ).Nonempty := h
    rw [WithTop.coe_inj, csInf_eq_iff hs]
    constructor
    · rintro ⟨hm, hmin⟩
      refine ⟨hm.2, fun i hi ↦ ?_⟩
      have := hmin (i.val + 1) ⟨by omega, hi⟩
      omega
    · rintro ⟨hlast, hprev⟩
      refine ⟨⟨by omega, hlast⟩, ?_⟩
      intro j hj
      have hjpos : 1 ≤ j := hj.1
      obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : j ≠ 0)
      by_contra hlt
      exact hprev ⟨k, by omega⟩ hj.2
  · constructor
    · intro ht
      exact (WithTop.natCast_ne_top (n + 1) ht.symm).elim
    · intro he
      exact (h ⟨n + 1, by omega, he.1⟩).elim

/-- Infinite-path positive hitting-time probabilities equal the finite-path
observations used in all reconstruction theorems. -/
theorem positivePathLaw_hittingAfter (G : WeightedGraph X) (hG : G.Admissible)
    (n : ℕ) (x y : X) :
    G.positivePathLaw hG x {ω | hittingAfter (pathProcess x) {y} 1 ω = (n + 1 : ℕ)} =
      G.firstPassageProbability hG n x y := by
  classical
  have hs : {ω | hittingAfter (pathProcess x) {y} 1 ω = (n + 1 : ℕ)} =
      (fun ω (i : Fin (n + 1)) ↦ ω i.val) ⁻¹' {w | FirstPassageAtEnd y w} := by
    ext ω
    exact hittingAfter_pathProcess_iff x y n ω
  rw [hs, ← Measure.map_apply (by fun_prop) (Set.to_countable _).measurableSet,
    G.positivePathLaw_map_prefix, PMF.toMeasure_apply _ (Set.to_countable _).measurableSet,
    tsum_fintype]
  rfl

end VertexCentrality.WeightedGraph
