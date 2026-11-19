/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Owen Parks
-/
module

public import Mathlib.Analysis.InnerProductSpace.Projection.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Reconstruction from harmonic-product moments

These results establish the linear-algebra step of the corrected Theorem 3.2 of
Gao–Li–Yang, arXiv:2603.20710v1. Equality of moments against a family determines
exactly the orthogonal projection onto its span. If that span is the whole space,
the moments determine the vector itself.

The Euclidean specializations use counting measure, not the unknown vertex-mass
inner product. The separate graph/control argument must still construct these
moments from observed first-passage data; this file does not assume that bridge
has been proved.
-/

@[expose] public section

namespace VertexCentrality

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The projection onto a subspace preserves all moments against that subspace. -/
theorem projection_inner_eq (Q : Submodule ℝ E) [Q.HasOrthogonalProjection]
    (u : E) {q : E} (hq : q ∈ Q) :
    inner ℝ (Q.starProjection u) q = inner ℝ u q := by
  rw [Q.inner_starProjection_left_eq_right,
    (Submodule.starProjection_eq_self_iff (K := Q)).mpr hq]

/-- Equal moments against a subspace are equivalent to equal orthogonal projections. -/
theorem projection_eq_iff_moments_eq (Q : Submodule ℝ E) [Q.HasOrthogonalProjection]
    {u v : E} :
    Q.starProjection u = Q.starProjection v ↔
      ∀ q ∈ Q, inner ℝ u q = inner ℝ v q := by
  constructor
  · intro h q hq
    rw [← projection_inner_eq Q u hq, ← projection_inner_eq Q v hq, h]
  · intro h
    apply ext_inner_right ℝ
    intro z
    rw [Q.inner_starProjection_left_eq_right, Q.inner_starProjection_left_eq_right]
    exact h _ (Q.starProjection_apply_mem z)

/-- Equality of moments on a generating family extends to its entire linear span. -/
theorem moments_eq_on_span {J : Type*} (q : J → E) {u v : E}
    (h : ∀ j, inner ℝ u (q j) = inner ℝ v (q j)) :
    ∀ z ∈ Submodule.span ℝ (Set.range q), inner ℝ u z = inner ℝ v z := by
  intro z hz
  induction hz using Submodule.span_induction with
  | mem z hz =>
    obtain ⟨j, rfl⟩ := hz
    exact h j
  | zero => simp
  | add x y hx hy ihx ihy => simp only [inner_add_right, ihx, ihy]
  | smul a x hx ih => simp only [inner_smul_right, ih]

/-- A family of observed moments determines the projection onto the family's span. -/
theorem projection_span_eq_iff {J : Type*} (q : J → E)
    [(Submodule.span ℝ (Set.range q)).HasOrthogonalProjection] {u v : E} :
    (Submodule.span ℝ (Set.range q)).starProjection u =
        (Submodule.span ℝ (Set.range q)).starProjection v ↔
      ∀ j, inner ℝ u (q j) = inner ℝ v (q j) := by
  rw [projection_eq_iff_moments_eq]
  exact ⟨fun h j ↦ h _ (Submodule.subset_span ⟨j, rfl⟩), moments_eq_on_span q⟩

/-- A spanning family determines the whole vector from its moments. -/
theorem eq_of_moments_eq_of_span_eq_top {J : Type*} (q : J → E)
    (hspan : Submodule.span ℝ (Set.range q) = ⊤) {u v : E}
    (h : ∀ j, inner ℝ u (q j) = inner ℝ v (q j)) : u = v := by
  apply ext_inner_right ℝ
  intro z
  exact moments_eq_on_span q h z (by rw [hspan]; trivial)

/-- The projection is the unique vector in the subspace with the given moments. -/
theorem existsUnique_moment_representative (Q : Submodule ℝ E)
    [Q.HasOrthogonalProjection] (u : E) :
    ∃! v : E, v ∈ Q ∧ ∀ q ∈ Q, inner ℝ v q = inner ℝ u q := by
  refine ⟨Q.starProjection u, ⟨Q.starProjection_apply_mem u,
    fun q hq ↦ projection_inner_eq Q u hq⟩, ?_⟩
  intro v hv
  have h := (projection_eq_iff_moments_eq Q).mpr hv.2
  simpa only [Submodule.starProjection_eq_self_iff.mpr hv.1] using h

section Euclidean

variable {I J : Type*} [Fintype I]

/-- In counting-measure Euclidean coordinates the moment is the unweighted dot product. -/
theorem euclidean_inner_eq_sum (u v : EuclideanSpace ℝ I) :
    inner ℝ u v = ∑ i, u i * v i := by
  simp [PiLp.inner_apply, mul_comm]

/-- The finite moment equations determine the Euclidean projection onto the product span. -/
theorem euclidean_projection_eq_iff (q : J → EuclideanSpace ℝ I)
    {u v : EuclideanSpace ℝ I} :
    (Submodule.span ℝ (Set.range q)).starProjection u =
        (Submodule.span ℝ (Set.range q)).starProjection v ↔
      ∀ j, (∑ i, u i * q j i) = ∑ i, v i * q j i := by
  simpa only [euclidean_inner_eq_sum] using projection_span_eq_iff q (u := u) (v := v)

/-- Full span of the product columns makes the finite moment system injective. -/
theorem euclidean_eq_of_moments_eq (q : J → EuclideanSpace ℝ I)
    (hspan : Submodule.span ℝ (Set.range q) = ⊤) {u v : EuclideanSpace ℝ I}
    (h : ∀ j, (∑ i, u i * q j i) = ∑ i, v i * q j i) : u = v := by
  apply eq_of_moments_eq_of_span_eq_top q hspan
  simpa only [euclidean_inner_eq_sum] using h

/-- The Euclidean column span used by the moment reconstruction system. -/
def euclideanColumnSpan (H : Matrix I J ℝ) : Submodule ℝ (EuclideanSpace ℝ I) :=
  Submodule.span ℝ (Set.range fun j ↦ WithLp.toLp 2 (H.col j))

/-- In ordinary value coordinates the moment equations are multiplication by the transpose. -/
theorem matrix_projection_eq_iff (H : Matrix I J ℝ) (u v : I → ℝ) :
    (euclideanColumnSpan H).starProjection (WithLp.toLp 2 u) =
        (euclideanColumnSpan H).starProjection (WithLp.toLp 2 v) ↔
      H.transpose.mulVec u = H.transpose.mulVec v := by
  simpa [euclideanColumnSpan, Matrix.mulVec, dotProduct, funext_iff, mul_comm] using
    euclidean_projection_eq_iff (fun j ↦ WithLp.toLp 2 (H.col j))
      (u := WithLp.toLp 2 u) (v := WithLp.toLp 2 v)

/-- Full row rank of the product matrix gives full span of its Euclidean columns. -/
theorem euclideanColumnSpan_eq_top_of_rank [Fintype J] (H : Matrix I J ℝ)
    (hH : H.rank = Fintype.card I) : euclideanColumnSpan H = ⊤ := by
  have hspan : Submodule.span ℝ (Set.range H.col) = ⊤ := by
    rw [← Matrix.range_mulVecLin]
    exact LinearMap.range_eq_top.mpr (Matrix.mulVec_surjective_iff_rank_eq_card.mpr hH)
  let e := (WithLp.linearEquiv 2 ℝ (I → ℝ)).symm
  have hmap := congrArg (Submodule.map e.toLinearMap) hspan
  simpa [Submodule.map_span, ← Set.range_comp, Function.comp_def, e,
    euclideanColumnSpan] using hmap

/-- Full row rank gives uniqueness for the corrected interior-mass moment equations. -/
theorem matrix_eq_of_moments_eq_of_rank [Fintype J] (H : Matrix I J ℝ)
    (hH : H.rank = Fintype.card I) {u v : I → ℝ}
    (h : H.transpose.mulVec u = H.transpose.mulVec v) : u = v := by
  have hp := (matrix_projection_eq_iff H u v).mpr h
  rw [euclideanColumnSpan_eq_top_of_rank H hH] at hp
  simpa only [Submodule.starProjection_top, ContinuousLinearMap.id_apply,
    WithLp.ofLp_toLp] using congrArg WithLp.ofLp hp

end Euclidean

end VertexCentrality
