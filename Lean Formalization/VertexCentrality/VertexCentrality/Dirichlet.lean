/-
Copyright (c) 2026 Owen Parks. All rights reserved.
Authors: Owen Parks
-/
module

public import VertexCentrality.Graph
public import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
public import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Tactic

/-!
# Harmonic Dirichlet extension on a finite weighted graph

The energy identity gives uniqueness for nonempty boundary in a connected graph.
Finite-dimensional injectivity then constructs the actual harmonic extension, rather
than assuming existence as part of a structure.
-/

public section

open scoped BigOperators

namespace VertexCentrality.WeightedGraph

variable {X : Type*} [Fintype X] (G : WeightedGraph X)

/-- Zero Laplacian energy forces equal values across every edge. -/
theorem eq_of_adj_of_energy_zero (u : X → ℝ) (hu : G.weightedPair (G.laplacian u) u = 0)
    {x y : X} (hxy : G.toSimpleGraph.Adj x y) : u x = u y := by
  have hnonneg (a b : X) :
      0 ≤ G.conductance a b * (u b - u a) * (u b - u a) := by
    simpa only [mul_assoc] using
      mul_nonneg (G.conductance_nonneg a b) (mul_self_nonneg (u b - u a))
  have hsum : (∑ a, ∑ b, G.conductance a b * (u b - u a) * (u b - u a)) = 0 := by
    have h := G.energy_identity u u
    rw [hu] at h
    linarith
  have hx := (Finset.sum_eq_zero_iff_of_nonneg
    (fun a _ ↦ Finset.sum_nonneg (fun b _ ↦ hnonneg a b))).mp hsum x (Finset.mem_univ x)
  have hterm := (Finset.sum_eq_zero_iff_of_nonneg
    (fun b _ ↦ hnonneg x b)).mp hx y (Finset.mem_univ y)
  have hpos : 0 < G.conductance x y := hxy
  have hd : (u y - u x) * (u y - u x) = 0 :=
    (mul_eq_zero.mp (by simpa only [mul_assoc] using hterm)).resolve_left (ne_of_gt hpos)
  exact (sub_eq_zero.mp (mul_self_eq_zero.mp hd)).symm

/-- A zero-boundary interior-harmonic function vanishes on a connected graph. -/
theorem harmonic_zero_of_boundary_zero (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (u : X → ℝ) (hu : G.HarmonicOn Bᶜ u)
    (hboundary : ∀ x ∈ B, u x = 0) : u = 0 := by
  have he : G.weightedPair (G.laplacian u) u = 0 := by
    apply Finset.sum_eq_zero
    intro x _
    by_cases hx : x ∈ B
    · simp [hboundary x hx]
    · simp [hu x hx]
  have hwalk {x y : X} (p : G.toSimpleGraph.Walk x y) : u x = u y := by
    induction p with
    | nil => rfl
    | @cons a b c hab p ih => exact (G.eq_of_adj_of_energy_zero u he hab).trans ih
  obtain ⟨b, hb⟩ := hB
  ext x
  obtain ⟨p⟩ := hG x b
  exact (hwalk p).trans (hboundary b hb)

/-- The square operator imposing boundary values and interior Laplacian values. -/
noncomputable def dirichletOperator (B : Set X) : (X → ℝ) →ₗ[ℝ] (X → ℝ) := by
  classical
  exact
    { toFun := fun u x ↦ if x ∈ B then u x else G.laplacian u x
      map_add' := by
        intro u v
        ext x
        by_cases hx : x ∈ B <;> simp [hx, G.laplacian_add]
      map_smul' := by
        intro c u
        ext x
        by_cases hx : x ∈ B <;> simp [hx, G.laplacian_smul] }

/-- Dirichlet operator evaluation at a boundary vertex. -/
theorem dirichletOperator_apply_of_mem (B : Set X) (u : X → ℝ) {x : X} (hx : x ∈ B) :
    G.dirichletOperator B u x = u x := by
  simp [dirichletOperator, hx]

/-- Dirichlet operator evaluation at an interior vertex. -/
theorem dirichletOperator_apply_of_notMem (B : Set X) (u : X → ℝ) {x : X} (hx : x ∉ B) :
    G.dirichletOperator B u x = G.laplacian u x := by
  simp [dirichletOperator, hx]

/-- The Dirichlet operator is injective for connected graphs with nonempty boundary. -/
theorem dirichletOperator_injective (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) : Function.Injective (G.dirichletOperator B) := by
  apply (G.dirichletOperator B).ker_eq_bot.mp
  apply LinearMap.ker_eq_bot'.mpr
  intro u hu
  apply G.harmonic_zero_of_boundary_zero B hG hB u
  · intro x hx
    change x ∉ B at hx
    have h := congrFun hu x
    simpa [dirichletOperator, hx] using h
  · intro x hx
    have h := congrFun hu x
    simpa [dirichletOperator, hx] using h

/-- The actual invertible Dirichlet operator, obtained from finite dimensionality. -/
noncomputable def dirichletEquiv (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) : (X → ℝ) ≃ₗ[ℝ] (X → ℝ) :=
  LinearEquiv.ofInjectiveEndo (G.dirichletOperator B) (G.dirichletOperator_injective B hG hB)

private noncomputable def boundaryData (B : Set X) : (B → ℝ) →ₗ[ℝ] (X → ℝ) := by
  classical
  exact
    { toFun := fun g x ↦ if hx : x ∈ B then g ⟨x, hx⟩ else 0
      map_add' := by intro g h; ext x; by_cases hx : x ∈ B <;> simp [hx]
      map_smul' := by intro c g; ext x; by_cases hx : x ∈ B <;> simp [hx] }

/-- Harmonic extension constructed by inverting the Dirichlet operator. -/
noncomputable def harmonicExtension (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) : (B → ℝ) →ₗ[ℝ] (X → ℝ) :=
  (G.dirichletEquiv B hG hB).symm.toLinearMap.comp (boundaryData B)

/-- The constructed extension satisfies its complete Dirichlet equation. -/
private theorem dirichletOperator_harmonicExtension (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (g : B → ℝ) :
    G.dirichletOperator B (G.harmonicExtension B hG hB g) = boundaryData B g := by
  exact (G.dirichletEquiv B hG hB).apply_symm_apply (boundaryData B g)

/-- Harmonic extension takes the prescribed boundary values. -/
@[simp] theorem harmonicExtension_boundary (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (g : B → ℝ) (x : B) :
    G.harmonicExtension B hG hB g x = g x := by
  have h := congrFun (G.dirichletOperator_harmonicExtension B hG hB g) x
  simpa [dirichletOperator, boundaryData, x.property] using h

/-- Harmonic extension has zero Laplacian on the interior. -/
theorem harmonicExtension_harmonic (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (g : B → ℝ) : G.HarmonicOn Bᶜ (G.harmonicExtension B hG hB g) := by
  intro x hx
  change x ∉ B at hx
  have h := congrFun (G.dirichletOperator_harmonicExtension B hG hB g) x
  simpa [dirichletOperator, boundaryData, hx] using h

/-- Every solution of the same Dirichlet problem equals the constructed extension. -/
theorem harmonicExtension_unique (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (g : B → ℝ) (u : X → ℝ) (hu : G.HarmonicOn Bᶜ u)
    (hb : ∀ x : B, u x = g x) : u = G.harmonicExtension B hG hB g := by
  apply G.dirichletOperator_injective B hG hB
  ext x
  by_cases hx : x ∈ B
  · rw [G.dirichletOperator_apply_of_mem B u hx,
      G.dirichletOperator_apply_of_mem B _ hx]
    exact (hb ⟨x, hx⟩).trans (G.harmonicExtension_boundary B hG hB g ⟨x, hx⟩).symm
  · rw [G.dirichletOperator_apply_of_notMem B u hx,
      G.dirichletOperator_apply_of_notMem B _ hx, hu x hx,
      G.harmonicExtension_harmonic B hG hB g x hx]

/-- The Dirichlet problem has exactly one solution. -/
theorem existsUnique_harmonicExtension (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (g : B → ℝ) :
    ∃! u : X → ℝ, G.HarmonicOn Bᶜ u ∧ ∀ x : B, u x = g x := by
  refine ⟨G.harmonicExtension B hG hB g, ⟨G.harmonicExtension_harmonic B hG hB g,
    G.harmonicExtension_boundary B hG hB g⟩, ?_⟩
  intro u hu
  exact G.harmonicExtension_unique B hG hB g u hu.1 hu.2

/-- Interior harmonic functions form a linear subspace. -/
def harmonicSubmodule (B : Set X) : Submodule ℝ (X → ℝ) where
  carrier := {u | G.HarmonicOn Bᶜ u}
  zero_mem' := by intro x _; simp [laplacian]
  add_mem' := by
    intro u v hu hv x hx
    simp [G.laplacian_add, hu x hx, hv x hx]
  smul_mem' := by
    intro c u hu x hx
    simp [G.laplacian_smul, hu x hx]

/-- Membership in the harmonic subspace is exactly interior harmonicity. -/
@[simp] theorem mem_harmonicSubmodule (B : Set X) (u : X → ℝ) :
    u ∈ G.harmonicSubmodule B ↔ G.HarmonicOn Bᶜ u := Iff.rfl

/-- Boundary restriction and harmonic extension are inverse linear isomorphisms. -/
noncomputable def harmonicEquiv (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) : (B → ℝ) ≃ₗ[ℝ] G.harmonicSubmodule B where
  toFun g := ⟨G.harmonicExtension B hG hB g, G.harmonicExtension_harmonic B hG hB g⟩
  map_add' g h := Subtype.ext ((G.harmonicExtension B hG hB).map_add g h)
  map_smul' c g := Subtype.ext ((G.harmonicExtension B hG hB).map_smul c g)
  invFun u := fun x ↦ u.val x
  left_inv g := funext (G.harmonicExtension_boundary B hG hB g)
  right_inv u := Subtype.ext
    (G.harmonicExtension_unique B hG hB (fun x ↦ u.val x) u.val u.property
      (fun _ ↦ rfl)).symm

/-- Extending the boundary coordinate basis gives a basis of all interior harmonic functions. -/
noncomputable def harmonicBasis (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) : Module.Basis B ℝ (G.harmonicSubmodule B) := by
  classical
  exact (Pi.basisFun ℝ B).map (G.harmonicEquiv B hG hB)

/-- The harmonic basis is given by extensions of the standard boundary basis. -/
theorem harmonicBasis_apply (B : Set X) (hG : G.toSimpleGraph.Connected)
    (hB : B.Nonempty) (j : B) :
    (G.harmonicBasis B hG hB j).val =
      G.harmonicExtension B hG hB (Pi.basisFun ℝ B j) := by
  classical
  rfl

/-- Harmonic extension is independent of vertex masses when conductances agree. -/
theorem harmonicExtension_eq_of_conductance_eq (H : WeightedGraph X) (B : Set X)
    (hG : G.toSimpleGraph.Connected) (hH : H.toSimpleGraph.Connected) (hB : B.Nonempty)
    (hw : G.conductance = H.conductance) (g : B → ℝ) :
    G.harmonicExtension B hG hB g = H.harmonicExtension B hH hB g := by
  symm
  apply G.harmonicExtension_unique B hG hB g
  · rw [G.harmonicOn_iff]
    rw [hw]
    exact (H.harmonicOn_iff Bᶜ _).mp (H.harmonicExtension_harmonic B hH hB g)
  · exact H.harmonicExtension_boundary B hH hB g

end VertexCentrality.WeightedGraph
