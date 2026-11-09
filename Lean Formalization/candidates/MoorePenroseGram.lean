/-
Copyright (c) 2026 The Tau Ceti contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: The Tau Ceti contributors
-/
module

public import TauCeti.Analysis.InnerProductSpace.MoorePenrose

/-!
# Consistent normal equations and the Moore–Penrose inverse

The Moore–Penrose solution of the normal equations recovers any vector in the
range of the original operator. The original operator may have a nontrivial
kernel, so its Gram operator need not be invertible.

This is an isolated proof candidate, not part of the Tau Ceti library.
-/

public section

namespace LinearMap

variable {𝕜 E F : Type*} [RCLike 𝕜]
  [NormedAddCommGroup E] [InnerProductSpace 𝕜 E] [FiniteDimensional 𝕜 E]
  [NormedAddCommGroup F] [InnerProductSpace 𝕜 F] [FiniteDimensional 𝕜 F]

/-- Solving the consistent normal equations with the Moore–Penrose inverse of
the Gram operator gives an exact preimage under the original operator. -/
theorem apply_gram_moorePenroseInverse_adjoint_of_mem_range
    (A : E →ₗ[𝕜] F) {y : F} (hy : y ∈ A.range) :
    A ((A.adjoint ∘ₗ A).moorePenroseInverse (A.adjoint y)) = y := by
  obtain ⟨x, rfl⟩ := hy
  have hnormal := (A.adjoint ∘ₗ A).isMoorePenroseInverse_moorePenroseInverse.apply_apply_apply x
  have hker : (A.adjoint ∘ₗ A).moorePenroseInverse ((A.adjoint ∘ₗ A) x) - x ∈
      (A.adjoint ∘ₗ A).ker := by
    rw [mem_ker, map_sub, hnormal, sub_self]
  rw [A.ker_adjoint_comp_self] at hker
  simpa only [mem_ker, map_sub, sub_eq_zero, comp_apply] using hker

end LinearMap
