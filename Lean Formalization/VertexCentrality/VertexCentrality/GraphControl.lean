module

public import VertexCentrality.Graph
public import VertexCentrality.GraphHeat
public import VertexCentrality.Observability

/-!
# The weighted graph Hilbert space

The paper's weighted pairing is realized as an actual finite-dimensional inner
product space. This construction allows the spectral controllability theorem to
apply to the concrete graph operator rather than assuming a Hilbert realization.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X]

/-- Vertex functions equipped below with the mass-weighted Hilbert structure. -/
def Space (_G : WeightedGraph X) := X → ℝ

instance (G : WeightedGraph X) : AddCommGroup G.Space := inferInstanceAs (AddCommGroup (X → ℝ))
instance (G : WeightedGraph X) : Module ℝ G.Space := inferInstanceAs (Module ℝ (X → ℝ))
instance (G : WeightedGraph X) : FiniteDimensional ℝ G.Space :=
  inferInstanceAs (FiniteDimensional ℝ (X → ℝ))

/-- Positive-definite weighted inner product data. -/
@[instance_reducible] noncomputable def spaceCore (G : WeightedGraph X) : InnerProductSpace.Core ℝ G.Space where
  inner u v := G.weightedPair u v
  conj_inner_symm u v := by simpa using G.weightedPair_comm v u
  re_inner_nonneg u := G.weightedPair_self_nonneg u
  add_left u v w := G.weightedPair_add_left u v w
  smul_left u v a := by
    change G.weightedPair (fun x => a * u x) v = _
    simp [weightedPair, Finset.mul_sum, mul_left_comm, mul_assoc]
  definite u h := (G.weightedPair_self_eq_zero_iff u).mp h

noncomputable instance (G : WeightedGraph X) : NormedAddCommGroup G.Space := by
  letI := spaceCore G
  exact @InnerProductSpace.Core.toNormedAddCommGroup ℝ G.Space _ _ _ (spaceCore G)

noncomputable instance (G : WeightedGraph X) : InnerProductSpace ℝ G.Space :=
  by
    letI := spaceCore G
    exact InnerProductSpace.ofCore (inferInstance : PreInnerProductSpace.Core ℝ G.Space)

/-- The new Hilbert structure has exactly the paper's inner product. -/
theorem space_inner (G : WeightedGraph X) (u v : G.Space) :
    inner ℝ u v = G.weightedPair u v := rfl

/-- The actual transition operator on the weighted Hilbert space. -/
noncomputable def spaceTransition (G : WeightedGraph X) : G.Space →ₗ[ℝ] G.Space :=
  G.transitionLinear

/-- Weighted self-adjointness supplies the symmetry required by spectral theory. -/
theorem spaceTransition_isSymmetric (G : WeightedGraph X) : G.spaceTransition.IsSymmetric := by
  intro u v
  exact G.transitionLinear_selfAdjoint u v

/-- Boundary-supported injection, allowing vertex coordinates for the controls. -/
noncomputable def boundaryMask (G : WeightedGraph X) (B : Set X) : G.Space →ₗ[ℝ] G.Space where
  toFun u x := @ite ℝ (x ∈ B) (Classical.propDecidable _) (u x) 0
  map_add' u v := by
    classical
    funext x
    change (if x ∈ B then u x + v x else 0) =
      (if x ∈ B then u x else 0) + (if x ∈ B then v x else 0)
    by_cases hx : x ∈ B <;> simp [hx]
  map_smul' a u := by
    classical
    funext x
    change (if x ∈ B then a * u x else 0) = a * (if x ∈ B then u x else 0)
    by_cases hx : x ∈ B <;> simp [hx]

/-- The boundary mask is its own weighted adjoint. -/
theorem boundaryMask_selfAdjoint (G : WeightedGraph X) (B : Set X) (u v : G.Space) :
    inner ℝ (G.boundaryMask B u) v = inner ℝ u (G.boundaryMask B v) := by
  classical
  change (∑ x, G.mass x * (if x ∈ B then u x else 0) * v x) =
    ∑ x, G.mass x * u x * (if x ∈ B then v x else 0)
  apply Finset.sum_congr rfl
  intro x _
  by_cases hx : x ∈ B <;> simp [hx]

/-- The paper's unique continuation assumption for eigenfunctions of the negative Laplacian. -/
def UniqueContinuation (G : WeightedGraph X) (B : Set X) : Prop :=
  ∀ (a : ℝ) (u : X → ℝ), (∀ x, -G.laplacian u x = a * u x) →
    (∀ x ∈ B, u x = 0) → u = 0

/-- On a nonempty graph, unique continuation excludes an empty boundary. -/
theorem boundary_nonempty_of_uniqueContinuation [Nonempty X]
    (G : WeightedGraph X) (B : Set X) (huc : G.UniqueContinuation B) : B.Nonempty := by
  by_contra hB
  have hu := huc 0 (fun _ ↦ 1) (by intro x; simp) (by
    intro x hx
    exact (hB ⟨x, hx⟩).elim)
  obtain ⟨x⟩ := ‹Nonempty X›
  have hx := congrFun hu x
  norm_num at hx

/-- Unique continuation also holds for the actual transition operator. -/
theorem transition_uniqueContinuation (G : WeightedGraph X) (B : Set X)
    (huc : G.UniqueContinuation B) (a : ℝ) (u : G.Space)
    (he : G.spaceTransition u = a • u) (hb : G.boundaryMask B u = 0) : u = 0 := by
  classical
  apply huc (1 - a) u
  · intro x
    have h := congrFun he x
    change u x + G.laplacian u x = a * u x at h
    linarith
  · intro x hx
    have h := congrFun hb x
    change (if x ∈ B then u x else 0) = 0 at h
    simpa [hx] using h

/-- Exact boundary-supported controllability of the graph transition evolution. -/
theorem boundary_reachability_surjective (G : WeightedGraph X) (B : Set X)
    (huc : G.UniqueContinuation B) (T : ℕ) (hT : Fintype.card X ≤ T) :
    Function.Surjective (reachability G.spaceTransition (G.boundaryMask B) T) := by
  apply reachability_surjective G.spaceTransition (G.boundaryMask B) (G.boundaryMask B)
    G.spaceTransition_isSymmetric (G.boundaryMask_selfAdjoint B)
    (G.transition_uniqueContinuation B huc) T
  change Module.finrank ℝ (X → ℝ) ≤ T
  simpa using hT

omit [Fintype X] in
/-- Restriction followed by zero extension is the boundary mask. -/
theorem boundaryExtend_restrict_eq_mask (G : WeightedGraph X) (B : Finset X)
    (u : G.Space) : boundaryExtend B (boundaryRestrict B u) = G.boundaryMask B u := by
  classical
  have hm : G.boundaryMask (↑B : Set X) u =
      (fun x ↦ @ite ℝ (x ∈ (↑B : Set X)) (Classical.propDecidable _) (u x) 0 :
        G.Space) := rfl
  rw [hm]
  funext x
  by_cases hx : x ∈ B <;> simp [boundaryExtend, boundaryRestrict, hx]

/-- Reversing power-indexed vertex controls and restricting to the boundary gives
the same terminal state for the concrete heat recurrence. -/
theorem heatSolution_reverse_restrict (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f : Fin T → G.Space) :
    G.heatSolution B (fun t ↦ boundaryRestrict B (f t.rev)) T =
      reachability G.spaceTransition (G.boundaryMask B) T f := by
  classical
  rw [heatSolution, forcedHeat_eq_sum, ← Fin.sum_univ_eq_sum_range,
    reachability_apply]
  calc
    _ = ∑ t : Fin T, (G.spaceTransition ^ (t.rev : ℕ))
        (G.boundaryMask B (f t.rev)) := by
      apply Finset.sum_congr rfl
      intro t _
      rw [extendControl_apply, G.boundaryExtend_restrict_eq_mask]
      have ht : T - 1 - (t : ℕ) = (t.rev : ℕ) := by simp [Fin.val_rev]; omega
      rw [ht]
      rfl
    _ = _ := Equiv.sum_comp Fin.revPerm
      (fun k : Fin T ↦ (G.spaceTransition ^ (k : ℕ)) (G.boundaryMask B (f k)))

/-- Unique continuation implies exact controllability for the actual finite
boundary-time heat map, including the empty-time/empty-vertex edge case. -/
theorem heatSolutionLinear_surjective (G : WeightedGraph X) (B : Finset X)
    (huc : G.UniqueContinuation B) (T : ℕ) (hT : Fintype.card X ≤ T) :
    Function.Surjective (G.heatSolutionLinear B T T) := by
  intro u
  obtain ⟨f, hf⟩ := G.boundary_reachability_surjective B huc T hT u
  exact ⟨fun t ↦ boundaryRestrict B (f t.rev),
    (G.heatSolution_reverse_restrict B T f).trans hf⟩

end VertexCentrality.WeightedGraph
