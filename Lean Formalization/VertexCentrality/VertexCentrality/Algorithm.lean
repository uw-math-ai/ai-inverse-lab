module

public import VertexCentrality.ProductSpan

/-!
# Exact reconstruction by normal equations

Any solution of the measured control normal equations reaches its harmonic
target. The weighted boundary formula then produces the true interior moments.
Under full product-matrix rank, solving those moment equations has the unique
answer given by the interior masses. These are exact algebraic guarantees, not
claims about numerical conditioning or a floating-point implementation.
-/

@[expose] public section

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X] [DecidableEq X]

omit [DecidableEq X] in
/-- Any solution of the actual data normal equations reaches its harmonic target. -/
theorem heatSolution_eq_of_data_normal_eq (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (ψ : X → ℝ) (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ) (f : Fin T → B → ℝ)
    (hf : G.gramData B T f = G.harmonicData B T ψ) : G.heatSolution B f T = ψ := by
  let fG : G.ControlSpace B T := f
  let ψG : G.Space := ψ
  exact apply_eq_of_normal_eq (G.controlToState B T)
    (G.controlToState_surjective B T huc hT ψG)
    ((G.controlToState_gram B T fG).trans
      (hf.trans (G.controlToState_harmonic_adjoint B T ψG hψ).symm))

omit [DecidableEq X] in
/-- The measured normal equations are consistent for every harmonic target. -/
theorem exists_data_normal_solution (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (ψ : X → ℝ) (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ) :
    ∃ f : Fin T → B → ℝ, G.gramData B T f = G.harmonicData B T ψ := by
  let ψG : G.Space := ψ
  refine ⟨G.gramControl B T ψG, ?_⟩
  exact (G.controlToState_gram B T (G.gramControl B T ψG)).symm.trans
    ((congrArg (G.controlToState B T).adjoint (G.gramControl_exact B T huc hT ψG)).trans
      (G.controlToState_harmonic_adjoint B T ψG hψ))

/-- Corrected equation (4.3) computes the interior moment from any normal-equation
solution; `controlPair` includes the boundary mass matrix. -/
theorem moment_eq_of_data_normal_eq (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (ψ φ : X → ℝ) (hψ : G.HarmonicOn (↑B : Set X)ᶜ ψ)
    (hφ : G.HarmonicOn (↑B : Set X)ᶜ φ) (f : Fin T → B → ℝ)
    (hf : G.gramData B T f = G.harmonicData B T ψ) :
    (∑ x : (Finset.univ \ B : Finset X), G.mass x * (ψ x * φ x)) =
      G.controlPair B f (G.harmonicData B T φ) -
        G.boundaryPair B (boundaryRestrict B ψ) (boundaryRestrict B φ) := by
  have hex := G.heatSolution_eq_of_data_normal_eq B T huc hT ψ hψ f hf
  have hp := G.heatSolution_harmonic_adjoint B f φ hφ
  rw [hex] at hp
  change G.weightedPair ψ φ = G.controlPair B f (G.harmonicData B T φ) at hp
  rw [← hp]
  have hb : G.boundaryPair B (boundaryRestrict B ψ) (boundaryRestrict B φ) =
      ∑ x ∈ B, G.mass x * (ψ x * φ x) := by
    simpa only [boundaryPair, boundaryRestrict, mul_assoc] using
      (Finset.sum_subtype B (fun _ ↦ Iff.rfl) (fun x ↦ G.mass x * (ψ x * φ x))).symm
  have hi : (∑ x : (Finset.univ \ B : Finset X), G.mass x * (ψ x * φ x)) =
      ∑ x ∈ Finset.univ \ B, G.mass x * (ψ x * φ x) :=
    (Finset.sum_subtype (Finset.univ \ B) (fun _ ↦ Iff.rfl)
      (fun x ↦ G.mass x * (ψ x * φ x))).symm
  rw [hb, hi]
  apply eq_sub_iff_add_eq.mpr
  simpa only [weightedPair, mul_assoc] using
    (Finset.sum_sdiff (f := fun x ↦ G.mass x * (ψ x * φ x)) (Finset.subset_univ B))

/-- The moment values obtained from a family of solved basis-control equations. -/
noncomputable def computedBasisMoment (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (f : B → Fin T → B → ℝ) (j k : B) : ℝ :=
  G.controlPair B (f j) (G.harmonicData B T (G.harmonicBasis B hconn hB k).val) -
    G.boundaryPair B (boundaryRestrict B (G.harmonicBasis B hconn hB j).val)
      (boundaryRestrict B (G.harmonicBasis B hconn hB k).val)

/-- A candidate interior mass vector solves the reconstructed basis moment system. -/
def SolvesBasisMoments (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (f : B → Fin T → B → ℝ) (m : (Finset.univ \ B : Finset X) → ℝ) : Prop :=
  ∀ j k : B, (∑ x, m x * ((G.harmonicBasis B hconn hB j).val x *
    (G.harmonicBasis B hconn hB k).val x)) = G.computedBasisMoment B T hconn hB f j k

/-- The true masses solve the reconstructed equations for any solved basis controls. -/
theorem mass_solvesBasisMoments (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (f : B → Fin T → B → ℝ)
    (hf : ∀ j, G.gramData B T (f j) =
      G.harmonicData B T (G.harmonicBasis B hconn hB j).val) :
    G.SolvesBasisMoments B T hconn hB f (fun x ↦ G.mass x) := by
  intro j k
  exact G.moment_eq_of_data_normal_eq B T huc hT _ _
    ((G.mem_harmonicSubmodule B _).mp (G.harmonicBasis B hconn hB j).property)
    ((G.mem_harmonicSubmodule B _).mp (G.harmonicBasis B hconn hB k).property) (f j) (hf j)

/-- Under full row rank, the reconstructed system has exactly one solution,
namely the true interior mass vector. -/
theorem reconstruction_unique (G : WeightedGraph X) (B : Finset X) (T : ℕ)
    (hconn : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (huc : G.UniqueContinuation B) (hT : Fintype.card X ≤ T)
    (f : B → Fin T → B → ℝ)
    (hf : ∀ j, G.gramData B T (f j) =
      G.harmonicData B T (G.harmonicBasis B hconn hB j).val)
    (hrank : (G.harmonicProductMatrix B hconn hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    ∃! m : (Finset.univ \ B : Finset X) → ℝ, G.SolvesBasisMoments B T hconn hB f m := by
  have hmass := G.mass_solvesBasisMoments B T hconn hB huc hT f hf
  refine ⟨fun x ↦ G.mass x, hmass, ?_⟩
  intro m hm
  apply matrix_eq_of_moments_eq_of_rank (G.harmonicProductMatrix B hconn hB) hrank
  funext p
  induction p using Sym2.inductionOn with
  | _ j k =>
    change (∑ x, (G.harmonicProductMatrix B hconn hB) x (Sym2.mk j k) * m x) =
      ∑ x, (G.harmonicProductMatrix B hconn hB) x (Sym2.mk j k) * G.mass x
    simpa only [harmonicProductMatrix, symmetricBasisProducts, Sym2.lift_mk,
      harmonicProducts, PiLp.toLp_apply, mul_comm] using (hm j k).trans (hmass j k).symm

/-- The control systems are consistent and the resulting full-rank moment
system has a unique solution. This supplies existence as well as correctness
of the exact normal-equation reconstruction steps. -/
theorem reconstruction_controls_exist_unique_mass (G : WeightedGraph X)
    (B : Finset X) (T : ℕ) (hconn : G.toSimpleGraph.Connected)
    (hB : (↑B : Set X).Nonempty) (huc : G.UniqueContinuation B)
    (hT : Fintype.card X ≤ T)
    (hrank : (G.harmonicProductMatrix B hconn hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    ∃ f : B → Fin T → B → ℝ,
      (∀ j, G.gramData B T (f j) = G.harmonicData B T (G.harmonicBasis B hconn hB j).val) ∧
      ∃! m : (Finset.univ \ B : Finset X) → ℝ, G.SolvesBasisMoments B T hconn hB f m := by
  choose f hf using fun j : B ↦ G.exists_data_normal_solution B T huc hT
    (G.harmonicBasis B hconn hB j).val
    ((G.mem_harmonicSubmodule B _).mp (G.harmonicBasis B hconn hB j).property)
  exact ⟨f, hf, G.reconstruction_unique B T hconn hB huc hT f hf hrank⟩

end VertexCentrality.WeightedGraph
