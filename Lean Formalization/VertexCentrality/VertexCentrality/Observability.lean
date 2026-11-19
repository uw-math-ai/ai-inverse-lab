module

public import Mathlib.Analysis.InnerProductSpace.Spectrum
public import Mathlib.LinearAlgebra.Charpoly.Basic

/-!
# Unique continuation and observability

Finite-dimensional symmetric operators have no nonzero unobservable invariant
subspace if no eigenvector can vanish under the observation map. This is the
spectral core of Lemma 2.3; the graph and finite-time control bridges remain separate.
-/

@[expose] public section

namespace VertexCentrality

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [AddCommGroup F] [Module ℝ F]

/-- The states invisible to an observation map at every nonnegative time. -/
def unobservable (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F) : Submodule ℝ E :=
  ⨅ k : ℕ, LinearMap.ker (C.comp (P ^ k))

/-- Membership means vanishing of every observed iterate. -/
theorem mem_unobservable_iff (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F) (v : E) :
    v ∈ unobservable P C ↔ ∀ k : ℕ, C ((P ^ k) v) = 0 := by
  simp [unobservable]

/-- The unobservable states form an invariant subspace. -/
theorem unobservable_invariant (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F)
    (v : E) (hv : v ∈ unobservable P C) : P v ∈ unobservable P C := by
  rw [mem_unobservable_iff] at hv ⊢
  intro k
  simpa [pow_succ, Module.End.mul_apply] using hv (k + 1)

variable [FiniteDimensional ℝ E]

/-- Unique continuation for eigenvectors rules out invisible states. -/
theorem unobservable_eq_bot (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F)
    (hP : P.IsSymmetric)
    (huc : ∀ (a : ℝ) (v : E), P v = a • v → C v = 0 → v = 0) :
    unobservable P C = ⊥ := by
  have hinv := unobservable_invariant P C
  have hres := hP.restrict_invariant hinv
  have hnone : ∀ a : ℝ, Module.End.eigenspace (P.restrict hinv) a = ⊥ := by
    intro a
    apply Module.End.eigenspace_restrict_eq_bot hinv
    rw [Submodule.disjoint_def]
    intro v he hu
    apply huc a v
    · exact Module.End.mem_eigenspace_iff.mp he
    · simpa using (mem_unobservable_iff P C v).mp hu 0
  have := hres.subsingleton_of_no_eigenvalue_finiteDimensional hnone
  exact Submodule.eq_bot_of_subsingleton

/-- Cayley--Hamilton reduces all observations to fewer than `finrank E` steps. -/
theorem observations_all_of_lt_finrank (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F) (v : E)
    (hv : ∀ k < Module.finrank ℝ E, C ((P ^ k) v) = 0) :
    ∀ k : ℕ, C ((P ^ k) v) = 0 := by
  intro k
  let q : Polynomial ℝ := Polynomial.X ^ k %ₘ P.charpoly
  rw [P.pow_eq_aeval_mod_charpoly k]
  change C ((Polynomial.aeval P q) v) = 0
  by_cases hq : q = 0
  · simp [hq]
  have hdeg : q.natDegree < Module.finrank ℝ E := by
    rw [← P.charpoly_natDegree]
    exact Polynomial.natDegree_lt_natDegree hq
      (Polynomial.degree_modByMonic_lt _ P.charpoly_monic)
  rw [Polynomial.aeval_endomorphism]
  change C (∑ i ∈ q.support, q.coeff i • (P ^ i) v) = 0
  rw [map_sum]
  apply Finset.sum_eq_zero
  intro i hi
  rw [map_smul, hv i ((Polynomial.le_natDegree_of_mem_supp i hi).trans_lt hdeg), smul_zero]

/-- Eigenvector unique continuation determines a state from `finrank E` observations. -/
theorem eq_zero_of_observations_lt_finrank (P : E →ₗ[ℝ] E) (C : E →ₗ[ℝ] F)
    (hP : P.IsSymmetric)
    (huc : ∀ (a : ℝ) (v : E), P v = a • v → C v = 0 → v = 0)
    (v : E) (hv : ∀ k < Module.finrank ℝ E, C ((P ^ k) v) = 0) : v = 0 := by
  have hmem : v ∈ unobservable P C :=
    (mem_unobservable_iff P C v).mpr (observations_all_of_lt_finrank P C v hv)
  simpa [unobservable_eq_bot P C hP huc] using hmem

section Control

variable {U : Type*} [NormedAddCommGroup U] [InnerProductSpace ℝ U]

/-- Reachability with controls indexed by the power of the evolution operator. -/
def reachability (P : E →ₗ[ℝ] E) (J : U →ₗ[ℝ] E) (T : ℕ) :
    (Fin T → U) →ₗ[ℝ] E :=
  ∑ k : Fin T, ((P ^ (k : ℕ)).comp J).comp (LinearMap.proj k)

omit [FiniteDimensional ℝ E] in
/-- The reachability map is the sum of propagated controls. -/
theorem reachability_apply (P : E →ₗ[ℝ] E) (J : U →ₗ[ℝ] E) (T : ℕ)
    (f : Fin T → U) : reachability P J T f = ∑ k : Fin T, (P ^ (k : ℕ)) (J (f k)) := by
  simp [reachability, LinearMap.sum_apply]

omit [FiniteDimensional ℝ E] in
/-- A single control produces one iterated injection. -/
theorem reachability_single (P : E →ₗ[ℝ] E) (J : U →ₗ[ℝ] E) (T : ℕ)
    (k : Fin T) (u : U) :
    reachability P J T (Pi.single k u) = (P ^ (k : ℕ)) (J u) := by
  classical
  simp [reachability_apply, Pi.single_apply, apply_ite]

/-- A symmetric evolution satisfying eigenvector unique continuation is exactly controllable.
The injection and observation must satisfy the displayed adjoint identity. -/
theorem reachability_surjective (P : E →ₗ[ℝ] E) (J : U →ₗ[ℝ] E) (C : E →ₗ[ℝ] U)
    (hP : P.IsSymmetric)
    (hJC : ∀ u v, inner ℝ (J u) v = inner ℝ u (C v))
    (huc : ∀ (a : ℝ) (v : E), P v = a • v → C v = 0 → v = 0)
    (T : ℕ) (hT : Module.finrank ℝ E ≤ T) :
    Function.Surjective (reachability P J T) := by
  apply LinearMap.range_eq_top.mp
  apply Submodule.orthogonal_eq_bot_iff.mp
  apply le_antisymm _ bot_le
  intro v hv
  change v = 0
  apply eq_zero_of_observations_lt_finrank P C hP huc v
  intro k hk
  have hkT : k < T := lt_of_lt_of_le hk hT
  let u : U := C ((P ^ k) v)
  have hz : inner ℝ ((P ^ k) (J u)) v = 0 := by
    apply (Submodule.mem_orthogonal _ _).mp hv
    exact ⟨Pi.single (⟨k, hkT⟩ : Fin T) u, reachability_single P J T _ _⟩
  rw [(hP.pow k) (J u) v, hJC] at hz
  exact (inner_self_eq_zero.mp hz)

end Control

end VertexCentrality
