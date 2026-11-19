/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Authors: Owen Parks
-/
module

public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.Projection.Basic

/-!
# Moore–Penrose inverses and exact Gram controls

For a linear map between finite-dimensional real inner-product spaces, restrict the map
from the orthogonal complement of its kernel to its range. This restriction is bijective.
Its inverse, composed with the orthogonal range projection, constructs the Moore–Penrose
inverse as a linear map, without a surjectivity or inverse-existence assumption.

The Gram-control theorem is the linear-algebra step of Proposition 3.1 in
Gao–Li–Yang, arXiv:2603.20710v1. Its application to the graph still requires the
separate controllability theorem and data-derived Gram and adjoint operators.
-/

@[expose] public section

namespace VertexCentrality

variable {E F : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- A linear map restricted from the orthogonal complement of its kernel to its range. -/
def orthogonalRestriction (A : E →ₗ[ℝ] F) : A.kerᗮ →ₗ[ℝ] A.range where
  toFun x := ⟨A x, ⟨x, rfl⟩⟩
  map_add' x y := Subtype.ext (A.map_add x y)
  map_smul' c x := Subtype.ext (A.map_smul c x)

/-- Removing the kernel component gives the unique orthogonal preimage in the range. -/
theorem orthogonalRestriction_bijective (A : E →ₗ[ℝ] F) :
    Function.Bijective (orthogonalRestriction A) := by
  constructor
  · intro x y h
    apply Subtype.ext
    apply sub_eq_zero.mp
    apply (inner_self_eq_zero (𝕜 := ℝ)).mp
    have hker : (x : E) - (y : E) ∈ A.ker := by
      change A ((x : E) - (y : E)) = 0
      rw [map_sub]
      exact sub_eq_zero.mpr (congrArg Subtype.val h)
    exact (A.kerᗮ.sub_mem x.property y.property) _ hker
  · rintro ⟨y, x, rfl⟩
    refine ⟨⟨x - A.ker.starProjection x, A.ker.sub_starProjection_mem_orthogonal x⟩, ?_⟩
    apply Subtype.ext
    change A (x - A.ker.starProjection x) = A x
    rw [map_sub, LinearMap.mem_ker.mp (A.ker.starProjection_apply_mem x), sub_zero]

/-- The canonical equivalence between the orthogonal kernel complement and the range. -/
noncomputable def orthogonalRangeEquiv (A : E →ₗ[ℝ] F) : A.kerᗮ ≃ₗ[ℝ] A.range :=
  LinearEquiv.ofBijective (orthogonalRestriction A) (orthogonalRestriction_bijective A)

/-- The Moore–Penrose inverse, constructed using the orthogonal restriction inverse. -/
noncomputable def moorePenrose (A : E →ₗ[ℝ] F) : F →ₗ[ℝ] E :=
  A.kerᗮ.subtype ∘ₗ (orthogonalRangeEquiv A).symm.toLinearMap ∘ₗ
    A.range.orthogonalProjectionOnto.toLinearMap

/-- Applying the map after its inverse is orthogonal projection onto the range. -/
theorem apply_moorePenrose (A : E →ₗ[ℝ] F) (y : F) :
    A (moorePenrose A y) = A.range.starProjection y := by
  have h := congrArg Subtype.val
    ((orthogonalRangeEquiv A).apply_symm_apply (A.range.orthogonalProjectionOnto y))
  exact h

/-- The constructed inverse always takes values orthogonal to the kernel. -/
theorem moorePenrose_mem_orthogonal (A : E →ₗ[ℝ] F) (y : F) :
    moorePenrose A y ∈ A.kerᗮ :=
  ((orthogonalRangeEquiv A).symm (A.range.orthogonalProjectionOnto y)).property

/-- The inverse gives an exact preimage whenever the right hand side is in the range. -/
theorem apply_moorePenrose_of_mem_range (A : E →ₗ[ℝ] F) {y : F} (hy : y ∈ A.range) :
    A (moorePenrose A y) = y := by
  rw [apply_moorePenrose, Submodule.starProjection_eq_self_iff.mpr hy]

omit [FiniteDimensional ℝ E] in
/-- Orthogonal preimages of the same vector are equal. -/
theorem eq_of_apply_eq_of_mem_orthogonal (A : E →ₗ[ℝ] F) {x z : E}
    (hx : x ∈ A.kerᗮ) (hz : z ∈ A.kerᗮ) (h : A x = A z) : x = z := by
  apply sub_eq_zero.mp
  apply (inner_self_eq_zero (𝕜 := ℝ)).mp
  exact (A.kerᗮ.sub_mem hx hz) _ (by simpa only [LinearMap.mem_ker, map_sub, sub_eq_zero])

/-- Applying the inverse after the map projects onto the orthogonal kernel complement. -/
theorem moorePenrose_apply (A : E →ₗ[ℝ] F) (x : E) :
    moorePenrose A (A x) = A.kerᗮ.starProjection x := by
  apply eq_of_apply_eq_of_mem_orthogonal A (moorePenrose_mem_orthogonal A _)
    (A.kerᗮ.starProjection_apply_mem x)
  rw [apply_moorePenrose_of_mem_range A ⟨x, rfl⟩,
    A.ker.starProjection_orthogonal_val, map_sub,
    LinearMap.mem_ker.mp (A.ker.starProjection_apply_mem x), sub_zero]

/-- The first Penrose equation, evaluated at a vector. -/
theorem apply_moorePenrose_apply (A : E →ₗ[ℝ] F) (x : E) :
    A (moorePenrose A (A x)) = A x :=
  apply_moorePenrose_of_mem_range A ⟨x, rfl⟩

/-- The second Penrose equation, evaluated at a vector. -/
theorem moorePenrose_apply_moorePenrose (A : E →ₗ[ℝ] F) (y : F) :
    moorePenrose A (A (moorePenrose A y)) = moorePenrose A y := by
  rw [moorePenrose_apply,
    Submodule.starProjection_eq_self_iff.mpr (moorePenrose_mem_orthogonal A y)]

/-- The third Penrose condition: the range projection is self-adjoint. -/
theorem self_comp_moorePenrose_isSymmetric (A : E →ₗ[ℝ] F) :
    (A ∘ₗ moorePenrose A).IsSymmetric := by
  intro x y
  simpa only [LinearMap.comp_apply, apply_moorePenrose] using
    A.range.inner_starProjection_left_eq_right x y

/-- The fourth Penrose condition: the kernel-complement projection is self-adjoint. -/
theorem moorePenrose_comp_self_isSymmetric (A : E →ₗ[ℝ] F) :
    (moorePenrose A ∘ₗ A).IsSymmetric := by
  intro x y
  simpa only [LinearMap.comp_apply, moorePenrose_apply] using
    A.kerᗮ.inner_starProjection_left_eq_right x y

/-- The constructed preimage has norm no greater than any other exact preimage. -/
theorem moorePenrose_norm_le (A : E →ₗ[ℝ] F) {y : F} {z : E} (hz : A z = y) :
    ‖moorePenrose A y‖ ≤ ‖z‖ := by
  rw [← hz, moorePenrose_apply]
  exact A.kerᗮ.norm_starProjection_apply_le z

/-- Equality in the minimum-norm bound characterizes the Moore–Penrose preimage. -/
theorem eq_moorePenrose_of_norm_eq (A : E →ₗ[ℝ] F) {y : F} {z : E}
    (hz : A z = y) (hnorm : ‖z‖ = ‖moorePenrose A y‖) : z = moorePenrose A y := by
  have hzorth : z ∈ A.kerᗮ := by
    apply (Submodule.mem_iff_norm_starProjection A.kerᗮ z).mpr
    rw [← moorePenrose_apply, hz, ← hnorm]
  exact eq_of_apply_eq_of_mem_orthogonal A hzorth (moorePenrose_mem_orthogonal A y)
    (hz.trans (apply_moorePenrose_of_mem_range A ⟨z, hz⟩).symm)

/-- Every vector in the range has a unique exact preimage of minimum norm. -/
theorem existsUnique_minimumNorm_preimage (A : E →ₗ[ℝ] F) {y : F} (hy : y ∈ A.range) :
    ∃! x : E, A x = y ∧ ∀ z : E, A z = y → ‖x‖ ≤ ‖z‖ := by
  refine ⟨moorePenrose A y,
    ⟨apply_moorePenrose_of_mem_range A hy, fun z hz ↦ moorePenrose_norm_le A hz⟩, ?_⟩
  intro x hx
  apply eq_moorePenrose_of_norm_eq A hx.1
  exact le_antisymm (hx.2 _ (apply_moorePenrose_of_mem_range A hy))
    (moorePenrose_norm_le A hx.1)

variable [FiniteDimensional ℝ F]

/-- Any solution of consistent Gram normal equations is an exact preimage. -/
theorem apply_eq_of_normal_eq (A : E →ₗ[ℝ] F) {y : F} (hy : y ∈ A.range) {x : E}
    (h : A.adjoint (A x) = A.adjoint y) : A x = y := by
  obtain ⟨z, rfl⟩ := hy
  have hker : x - z ∈ (A.adjoint ∘ₗ A).ker := by
    simpa only [LinearMap.mem_ker, map_sub, LinearMap.comp_apply, sub_eq_zero] using h
  rw [A.ker_adjoint_comp_self] at hker
  simpa only [LinearMap.mem_ker, map_sub, sub_eq_zero] using hker

/-- Inverting the Gram operator on the adjoint data gives an exact control. -/
theorem apply_gram_moorePenrose_adjoint_of_mem_range (A : E →ₗ[ℝ] F)
    {y : F} (hy : y ∈ A.range) :
    A (moorePenrose (A.adjoint ∘ₗ A) (A.adjoint y)) = y := by
  apply apply_eq_of_normal_eq A hy
  exact apply_moorePenrose_of_mem_range (A.adjoint ∘ₗ A) (by
    rw [A.range_adjoint_comp_self]
    exact ⟨y, rfl⟩)

/-- The Gram formula is the same canonical minimum-norm preimage as the inverse of the map. -/
theorem gram_moorePenrose_adjoint_eq (A : E →ₗ[ℝ] F) {y : F} (hy : y ∈ A.range) :
    moorePenrose (A.adjoint ∘ₗ A) (A.adjoint y) = moorePenrose A y := by
  apply eq_of_apply_eq_of_mem_orthogonal A
  · simpa only [A.ker_adjoint_comp_self] using
      moorePenrose_mem_orthogonal (A.adjoint ∘ₗ A) (A.adjoint y)
  · exact moorePenrose_mem_orthogonal A y
  · rw [apply_gram_moorePenrose_adjoint_of_mem_range A hy,
      apply_moorePenrose_of_mem_range A hy]

/-- The Gram control has minimum norm among all exact preimages. -/
theorem gram_moorePenrose_adjoint_norm_le (A : E →ₗ[ℝ] F) {y : F} {z : E}
    (hz : A z = y) : ‖moorePenrose (A.adjoint ∘ₗ A) (A.adjoint y)‖ ≤ ‖z‖ := by
  rw [gram_moorePenrose_adjoint_eq A ⟨z, hz⟩]
  exact moorePenrose_norm_le A hz

/-- Surjectivity removes the range condition in the exact-control formula. -/
theorem apply_gram_moorePenrose_adjoint_of_surjective (A : E →ₗ[ℝ] F)
    (hA : Function.Surjective A) (y : F) :
    A (moorePenrose (A.adjoint ∘ₗ A) (A.adjoint y)) = y :=
  apply_gram_moorePenrose_adjoint_of_mem_range A (hA y)

end VertexCentrality
