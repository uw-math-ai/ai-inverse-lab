module

public import VertexCentrality.Graph
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

end VertexCentrality.WeightedGraph
