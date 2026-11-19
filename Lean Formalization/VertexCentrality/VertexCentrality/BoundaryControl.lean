module

public import VertexCentrality.GraphControl
public import VertexCentrality.Pseudoinverse

/-!
# Weighted boundary controls

The terminal heat map is realized between the paper's weighted Hilbert spaces.
Its Gram inverse uses Tau Ceti's canonical Moore–Penrose operator. The formulas
below identify the Gram operator and harmonic adjoint with concrete responses;
the first-passage-to-response bridge remains a separate obligation.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X]

/-- Finite boundary-time controls with the mass-weighted Hilbert structure below. -/
def ControlSpace (_G : WeightedGraph X) (B : Finset X) (T : ℕ) := Fin T → B → ℝ

instance (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    AddCommGroup (G.ControlSpace B T) := inferInstanceAs (AddCommGroup (Fin T → B → ℝ))
instance (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    Module ℝ (G.ControlSpace B T) := inferInstanceAs (Module ℝ (Fin T → B → ℝ))
instance (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    FiniteDimensional ℝ (G.ControlSpace B T) :=
  inferInstanceAs (FiniteDimensional ℝ (Fin T → B → ℝ))

omit [Fintype X] in
/-- The weighted space-time quadratic form is nonnegative. -/
theorem controlPair_self_nonneg (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f : G.ControlSpace B T) : 0 ≤ G.controlPair B f f := by
  apply Finset.sum_nonneg
  intro t _
  apply Finset.sum_nonneg
  intro x _
  simpa only [mul_assoc] using mul_nonneg (G.mass_pos x).le (mul_self_nonneg (f t x))

omit [Fintype X] in
/-- Positive boundary masses make the finite space-time pairing definite. -/
theorem controlPair_self_eq_zero (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f : G.ControlSpace B T) (h : G.controlPair B f f = 0) : f = 0 := by
  have hn : ∀ (t : Fin T) (x : B), 0 ≤ G.mass x * f t x * f t x := by
    intro t x
    simpa only [mul_assoc] using mul_nonneg (G.mass_pos x).le (mul_self_nonneg (f t x))
  have ht := (Finset.sum_eq_zero_iff_of_nonneg
    (fun t _ ↦ Finset.sum_nonneg (fun x _ ↦ hn t x))).mp h
  funext t x
  have hx := (Finset.sum_eq_zero_iff_of_nonneg (fun x _ ↦ hn t x)).mp
    (ht t (Finset.mem_univ t)) x (Finset.mem_univ x)
  change f t x = 0
  simpa [mul_assoc, ne_of_gt (G.mass_pos x)] using hx

/-- The weighted control inner product in ordinary boundary value coordinates. -/
@[instance_reducible] noncomputable def controlSpaceCore (G : WeightedGraph X)
    (B : Finset X) (T : ℕ) : InnerProductSpace.Core ℝ (G.ControlSpace B T) where
  inner f g := G.controlPair B f g
  conj_inner_symm f g := by
    simp only [controlPair, map_sum, conj_trivial]
    exact Finset.sum_congr rfl (fun t _ ↦ G.boundaryPair_comm B (g t) (f t))
  re_inner_nonneg f := controlPair_self_nonneg G B T f
  add_left f g h := by
    change G.controlPair B (fun t x ↦ f t x + g t x) h = _
    simp [controlPair, boundaryPair, add_mul, mul_add, Finset.sum_add_distrib]
  smul_left f g a := by
    change G.controlPair B (fun t x ↦ a * f t x) g = _
    simp [controlPair, boundaryPair, Finset.mul_sum, mul_left_comm, mul_assoc]
  definite f h := G.controlPair_self_eq_zero B T f h

noncomputable instance (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    NormedAddCommGroup (G.ControlSpace B T) := by
  letI := controlSpaceCore G B T
  exact @InnerProductSpace.Core.toNormedAddCommGroup ℝ (G.ControlSpace B T) _ _ _
    (controlSpaceCore G B T)

noncomputable instance (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    InnerProductSpace ℝ (G.ControlSpace B T) := by
  letI := controlSpaceCore G B T
  exact InnerProductSpace.ofCore (inferInstance : PreInnerProductSpace.Core ℝ (G.ControlSpace B T))

omit [Fintype X] in
/-- The Hilbert inner product retains all boundary mass factors. -/
theorem controlSpace_inner (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f g : G.ControlSpace B T) : inner ℝ f g = G.controlPair B f g := rfl

/-- The concrete terminal heat map between weighted Hilbert spaces. -/
noncomputable def controlToState (G : WeightedGraph X) (B : Finset X) (T : ℕ) :
    G.ControlSpace B T →ₗ[ℝ] G.Space := G.heatSolutionLinear B T T

/-- Unique continuation makes the weighted terminal map onto. -/
theorem controlToState_surjective (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T) :
    Function.Surjective (G.controlToState B T) := G.heatSolutionLinear_surjective B huc T hT

/-- The Gram operator is the reversed extended response on the control interval. -/
theorem controlToState_gram (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (g : G.ControlSpace B T) :
    (G.controlToState B T).adjoint (G.controlToState B T g) =
      (fun t ↦ boundaryRestrict B (G.heatSolution B g (2 * T - 1 - t)) :
        G.ControlSpace B T) := by
  apply ext_inner_left ℝ
  intro f
  rw [LinearMap.adjoint_inner_right]
  exact G.heatSolution_gram B f g

/-- The harmonic adjoint is computed from the short response and boundary values. -/
theorem controlToState_harmonic_adjoint (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (φ : G.Space) (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    (G.controlToState B T).adjoint φ =
      (timeReverse (G.response B (timeReverse
        (fun _ ↦ boundaryRestrict B (G.laplacian φ)))) +
        fun _ ↦ boundaryRestrict B φ : G.ControlSpace B T) := by
  apply ext_inner_left ℝ
  intro f
  rw [LinearMap.adjoint_inner_right]
  exact G.heatSolution_harmonic_adjoint B f φ hφ

/-- Exact Gram control using Tau Ceti's canonical inverse in the weighted spaces. -/
noncomputable def gramControl (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (φ : G.Space) : G.ControlSpace B T :=
  ((G.controlToState B T).adjoint ∘ₗ G.controlToState B T).moorePenroseInverse
    ((G.controlToState B T).adjoint φ)

/-- The concrete Gram formula reaches every target under unique continuation. -/
theorem gramControl_exact (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T) (φ : G.Space) :
    G.controlToState B T (G.gramControl B T φ) = φ := by
  unfold gramControl
  rw [← moorePenrose_eq_tauceti]
  exact apply_gram_moorePenrose_adjoint_of_surjective _
    (G.controlToState_surjective B T huc hT) φ

/-- The Gram control minimizes the weighted norm among exact controls. -/
theorem gramControl_norm_le (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (φ : G.Space) (f : G.ControlSpace B T) (hf : G.controlToState B T f = φ) :
    ‖G.gramControl B T φ‖ ≤ ‖f‖ := by
  unfold gramControl
  rw [← moorePenrose_eq_tauceti]
  exact gram_moorePenrose_adjoint_norm_le _ hf

/-- The corrected interior moment formula with the actual weighted Gram control.
This establishes the response-side identity, not recovery from first-passage laws. -/
theorem gramControl_harmonic_moment [DecidableEq X]
    (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (ψ φ : G.Space) (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    (∑ x ∈ Finset.univ \ B, G.mass x * ψ x * φ x) =
      G.controlPair B (G.gramControl B T ψ)
        (timeReverse (G.response B (timeReverse
          (fun _ ↦ boundaryRestrict B (G.laplacian φ)))) +
          fun _ ↦ boundaryRestrict B φ) -
        G.boundaryPair B (boundaryRestrict B ψ) (boundaryRestrict B φ) := by
  classical
  have hp := G.heatSolution_harmonic_adjoint B (G.gramControl B T ψ) φ hφ
  change G.weightedPair (G.controlToState B T (G.gramControl B T ψ)) φ = _ at hp
  rw [G.gramControl_exact B T huc hT] at hp
  rw [← hp]
  have hb : G.boundaryPair B (boundaryRestrict B ψ) (boundaryRestrict B φ) =
      ∑ x ∈ B, G.mass x * ψ x * φ x := by
    exact (Finset.sum_subtype B (fun _ ↦ Iff.rfl)
      (fun x ↦ G.mass x * ψ x * φ x)).symm
  rw [hb]
  exact eq_sub_iff_add_eq.mpr (Finset.sum_sdiff (Finset.subset_univ B))

end VertexCentrality.WeightedGraph
