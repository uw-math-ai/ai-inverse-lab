module

public import VertexCentrality.BoundaryControl
public import VertexCentrality.GraphProbability
public import VertexCentrality.Dirichlet
public import VertexCentrality.Reconstruction

/-!
# Equal observations give equal harmonic moments

All operators below are the concrete graph heat operators. Equal first-passage
data gives equal responses; equal responses transfer exact controls between
models with the same conductances and boundary masses. Consequently their
interior mass vectors have the same harmonic-product moments.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X]

/-- Harmonicity depends only on conductances, not on the unknown masses. -/
theorem harmonicOn_iff_of_conductance_eq (G H : WeightedGraph X)
    (hc : G.conductance = H.conductance) (I : Set X) (φ : X → ℝ) :
    G.HarmonicOn I φ ↔ H.HarmonicOn I φ := by
  rw [G.harmonicOn_iff, H.harmonicOn_iff, hc]

/-- The boundary Laplacian is known from conductances and boundary masses. -/
theorem boundary_laplacian_eq (G H : WeightedGraph X) (B : Finset X)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (φ : X → ℝ) : boundaryRestrict B (G.laplacian φ) =
      boundaryRestrict B (H.laplacian φ) := by
  funext x
  simp only [boundaryRestrict, laplacian, hc, hm x]

/-- The measured Gram formula in ordinary boundary-time value coordinates. -/
noncomputable def gramData (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (f : Fin T → B → ℝ) : Fin T → B → ℝ :=
  fun t ↦ boundaryRestrict B (G.heatSolution B f (2 * T - 1 - t))

/-- The measured harmonic adjoint formula in ordinary value coordinates. -/
noncomputable def harmonicData (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (φ : X → ℝ) : Fin T → B → ℝ :=
  timeReverse (G.response B (timeReverse
    (fun _ ↦ boundaryRestrict B (G.laplacian φ)))) + fun _ ↦ boundaryRestrict B φ

/-- Equal extended responses give the same Gram formula for every control. -/
theorem gramData_eq (G H : WeightedGraph X) (B : Finset X) (T : ℕ)
    (he : G.extendedResponseLinear B T = H.extendedResponseLinear B T)
    (f : Fin T → B → ℝ) : G.gramData B T f = H.gramData B T f := by
  funext t
  have ht : 2 * T - 1 - (t : ℕ) < 2 * T := by have := t.isLt; omega
  exact congrFun (LinearMap.congr_fun he f) ⟨2 * T - 1 - t, ht⟩

/-- Equal short responses and known boundary coefficients give the same harmonic data. -/
theorem harmonicData_eq (G H : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hs : G.responseLinear B T = H.responseLinear B T) (φ : X → ℝ) :
    G.harmonicData B T φ = H.harmonicData B T φ := by
  unfold harmonicData
  rw [G.boundary_laplacian_eq H B hc hm φ]
  have h := LinearMap.congr_fun hs
    (timeReverse (fun _ ↦ boundaryRestrict B (H.laplacian φ)))
  rw [G.responseLinear_apply, H.responseLinear_apply] at h
  rw [h]

/-- A control reaching a harmonic target in one model reaches it in any model
with the same responses, conductances and boundary masses and with UCP. -/
theorem exact_control_transfer (G H : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hs : G.responseLinear B T = H.responseLinear B T)
    (he : G.extendedResponseLinear B T = H.extendedResponseLinear B T)
    (hucH : H.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (ψ : X → ℝ) (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ)
    (f : Fin T → B → ℝ) (hf : G.heatSolution B f T = ψ) :
    H.heatSolution B f T = ψ := by
  let fG : G.ControlSpace B T := f
  let fH : H.ControlSpace B T := f
  let ψG : G.Space := ψ
  let ψH : H.Space := ψ
  have hψH := (G.harmonicOn_iff_of_conductance_eq H hc _ ψ).mp hψ
  have hdata : H.gramData B T f = H.harmonicData B T ψ := by
    rw [← G.gramData_eq H B T he f, ← G.harmonicData_eq H B T hc hm hs ψ]
    exact (G.controlToState_gram B T fG).symm.trans
      ((congrArg (G.controlToState B T).adjoint hf).trans
        (G.controlToState_harmonic_adjoint B T ψG hψ))
  exact apply_eq_of_normal_eq (H.controlToState B T)
    (H.controlToState_surjective B T hucH hT ψH)
    ((H.controlToState_gram B T fH).trans
      (hdata.trans (H.controlToState_harmonic_adjoint B T ψH hψH).symm))

omit [Fintype X] in
/-- Boundary space-time pairings agree when the known boundary masses agree. -/
theorem controlPair_eq_of_mass_eq (G H : WeightedGraph X) (B : Finset X)
    (hm : ∀ x : B, G.mass x = H.mass x) {T : ℕ} (f g : Fin T → B → ℝ) :
    G.controlPair B f g = H.controlPair B f g := by
  simp only [controlPair, boundaryPair]
  simp_rw [hm]

/-- Equal response operators imply equal full weighted pairings of harmonic functions. -/
theorem harmonic_pairing_eq_of_responses_eq (G H : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hs : G.responseLinear B T = H.responseLinear B T)
    (he : G.extendedResponseLinear B T = H.extendedResponseLinear B T)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) (ψ φ : X → ℝ)
    (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ) (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    G.weightedPair ψ φ = H.weightedPair ψ φ := by
  obtain ⟨f, hf⟩ := G.heatSolutionLinear_surjective B hucG T hT ψ
  change G.heatSolution B f T = ψ at hf
  have hfH := G.exact_control_transfer H B T hc hm hs he hucH hT ψ hψ f hf
  have hφH := (G.harmonicOn_iff_of_conductance_eq H hc _ φ).mp hφ
  have hpG := G.heatSolution_harmonic_adjoint B f φ hφ
  have hpH := H.heatSolution_harmonic_adjoint B f φ hφH
  rw [hf] at hpG
  rw [hfH] at hpH
  change G.weightedPair ψ φ = G.controlPair B f (G.harmonicData B T φ) at hpG
  change H.weightedPair ψ φ = H.controlPair B f (H.harmonicData B T φ) at hpH
  rw [hpG, hpH, G.harmonicData_eq H B T hc hm hs φ,
    G.controlPair_eq_of_mass_eq H B hm]

variable [DecidableEq X]

/-- The measured finite first-passage laws determine interior harmonic-product
moments, assuming UCP and the known conductances and boundary masses. -/
theorem harmonic_moments_eq_of_firstPassageProbability_eq
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) (ψ φ : X → ℝ)
    (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ) (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) :
    (∑ x ∈ Finset.univ \ B, G.mass x * ψ x * φ x) =
      ∑ x ∈ Finset.univ \ B, H.mass x * ψ x * φ x := by
  have hp := G.harmonic_pairing_eq_of_responses_eq H B T hc hm
    (G.responseLinear_eq_of_firstPassageProbability_eq H hG hH B T hr)
    (G.extendedResponseLinear_eq_of_firstPassageProbability_eq H hG hH B T hr)
    hucG hucH hT ψ φ hψ hφ
  have hb : (∑ x ∈ B, G.mass x * ψ x * φ x) = ∑ x ∈ B, H.mass x * ψ x * φ x := by
    apply Finset.sum_congr rfl
    intro x hx
    rw [hm ⟨x, hx⟩]
  have hsumG := Finset.sum_sdiff (f := fun x ↦ G.mass x * ψ x * φ x)
    (Finset.subset_univ B)
  have hsumH := Finset.sum_sdiff (f := fun x ↦ H.mass x * ψ x * φ x)
    (Finset.subset_univ B)
  change (∑ x, G.mass x * ψ x * φ x) = ∑ x, H.mass x * ψ x * φ x at hp
  linarith

/-- All products of interior harmonic functions, in Euclidean interior coordinates. -/
noncomputable def harmonicProducts (G : WeightedGraph X) (B : Finset X) :
    (G.harmonicSubmodule B × G.harmonicSubmodule B) →
      EuclideanSpace ℝ (Finset.univ \ B : Finset X) :=
  fun p ↦ WithLp.toLp 2 (fun x ↦ p.1.val x * p.2.val x)

/-- The counting-measure harmonic-product span from the corrected Theorem 3.2. -/
noncomputable def harmonicProductSpace (G : WeightedGraph X) (B : Finset X) :
    Submodule ℝ (EuclideanSpace ℝ (Finset.univ \ B : Finset X)) :=
  Submodule.span ℝ (Set.range (G.harmonicProducts B))

/-- The reconstruction subspace is determined by conductances alone. -/
theorem harmonicProductSpace_eq_of_conductance_eq (G H : WeightedGraph X)
    (B : Finset X) (hc : G.conductance = H.conductance) :
    G.harmonicProductSpace B = H.harmonicProductSpace B := by
  unfold harmonicProductSpace
  congr 1
  ext q
  constructor
  · rintro ⟨⟨ψ, φ⟩, rfl⟩
    have hψ : ψ.val ∈ H.harmonicSubmodule B := (H.mem_harmonicSubmodule B _).mpr
      ((G.harmonicOn_iff_of_conductance_eq H hc _ _).mp
        ((G.mem_harmonicSubmodule B _).mp ψ.property))
    have hφ : φ.val ∈ H.harmonicSubmodule B := (H.mem_harmonicSubmodule B _).mpr
      ((G.harmonicOn_iff_of_conductance_eq H hc _ _).mp
        ((G.mem_harmonicSubmodule B _).mp φ.property))
    exact ⟨(⟨ψ.val, hψ⟩, ⟨φ.val, hφ⟩), rfl⟩
  · rintro ⟨⟨ψ, φ⟩, rfl⟩
    have hψ : ψ.val ∈ G.harmonicSubmodule B := (G.mem_harmonicSubmodule B _).mpr
      ((G.harmonicOn_iff_of_conductance_eq H hc _ _).mpr
        ((H.mem_harmonicSubmodule B _).mp ψ.property))
    have hφ : φ.val ∈ G.harmonicSubmodule B := (G.mem_harmonicSubmodule B _).mpr
      ((G.harmonicOn_iff_of_conductance_eq H hc _ _).mpr
        ((H.mem_harmonicSubmodule B _).mp φ.property))
    exact ⟨(⟨ψ.val, hψ⟩, ⟨φ.val, hφ⟩), rfl⟩

/-- The interior mass vector uses Euclidean coordinates, not its unknown weighted metric. -/
noncomputable def interiorMass (G : WeightedGraph X) (B : Finset X) :
    EuclideanSpace ℝ (Finset.univ \ B : Finset X) := WithLp.toLp 2 (fun x ↦ G.mass x)

/-- Corrected projection reconstruction as data invariance: any two admissible
models with the same known inputs and finite first-passage data have the same
Euclidean projection of their interior mass vectors onto the harmonic-product span. -/
theorem projection_eq_of_firstPassageProbability_eq
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) :
    (G.harmonicProductSpace B).starProjection (G.interiorMass B) =
      (G.harmonicProductSpace B).starProjection (H.interiorMass B) := by
  apply (euclidean_projection_eq_iff (G.harmonicProducts B)).mpr
  intro p
  have hp := G.harmonic_moments_eq_of_firstPassageProbability_eq H hG hH B T hc hm hr
    hucG hucH hT p.1.val p.2.val
    ((G.mem_harmonicSubmodule B _).mp p.1.property)
    ((G.mem_harmonicSubmodule B _).mp p.2.property)
  have he (K : WeightedGraph X) :
      (∑ x ∈ Finset.univ \ B, K.mass x * p.1.val x * p.2.val x) =
      ∑ x : (Finset.univ \ B : Finset X), K.mass x * p.1.val x * p.2.val x :=
    Finset.sum_subtype (Finset.univ \ B) (fun _ ↦ Iff.rfl) _
  rw [he G, he H] at hp
  simpa [interiorMass, harmonicProducts, mul_assoc] using hp

/-- Full harmonic-product span upgrades the data invariant projection to unique
interior mass recovery. The span hypothesis is explicit, not inferred from dimension. -/
theorem interiorMass_eq_of_firstPassageProbability_eq
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) (hspan : G.harmonicProductSpace B = ⊤) :
    G.interiorMass B = H.interiorMass B := by
  have h := G.projection_eq_of_firstPassageProbability_eq H hG hH B T hc hm hr hucG hucH hT
  simpa only [hspan, Submodule.starProjection_top, ContinuousLinearMap.id_apply] using h

/-- Full recovery in ordinary vertex coordinates: equal known inputs and
first-passage laws force equality of all vertex masses under the full-span condition. -/
theorem mass_eq_of_firstPassageProbability_eq
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T) (hspan : G.harmonicProductSpace B = ⊤) :
    G.mass = H.mass := by
  have h := G.interiorMass_eq_of_firstPassageProbability_eq H hG hH B T hc hm hr
    hucG hucH hT hspan
  funext x
  by_cases hx : x ∈ B
  · exact hm ⟨x, hx⟩
  · have hi : x ∈ Finset.univ \ B := Finset.mem_sdiff.mpr ⟨Finset.mem_univ x, hx⟩
    exact congrArg (fun m : EuclideanSpace ℝ (Finset.univ \ B : Finset X) ↦ m ⟨x, hi⟩) h

end VertexCentrality.WeightedGraph
