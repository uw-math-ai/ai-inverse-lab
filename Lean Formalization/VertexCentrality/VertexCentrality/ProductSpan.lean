module

public import VertexCentrality.DataInvariance
public import Mathlib.Algebra.Module.Submodule.Bilinear
public import Mathlib.Data.Sym.Card

/-!
# The finite harmonic-product matrix

Mathlib's bilinear-image span theorem reduces all harmonic products to products
of a Dirichlet basis. Symmetric pairs remove duplicate columns without choosing
an order on the boundary. Full row rank is exactly full harmonic-product span.
-/

@[expose] public section

namespace VertexCentrality

/-- A bilinear image is spanned by its values on pairs of basis vectors. -/
theorem bilinear_span_basis {E F J : Type*} [AddCommGroup E] [Module ℝ E]
    [AddCommGroup F] [Module ℝ F] (b : Module.Basis J ℝ E) (A : E →ₗ[ℝ] E →ₗ[ℝ] F) :
    Submodule.span ℝ (Set.range (fun p : E × E ↦ A p.1 p.2)) =
      Submodule.span ℝ (Set.range (fun p : J × J ↦ A (b p.1) (b p.2))) := by
  have h := Submodule.map₂_span_span ℝ A (Set.range b) (Set.range b)
  rw [b.span_eq, Submodule.map₂_eq_span_image2] at h
  have hu : Set.image2 (fun x y ↦ A x y) Set.univ Set.univ =
      Set.range (fun p : E × E ↦ A p.1 p.2) := by
    simpa only [Set.range_id, id_eq] using Set.image2_range (fun x y ↦ A x y) id id
  change Submodule.span ℝ (Set.image2 (fun x y ↦ A x y) Set.univ Set.univ) = _ at h
  simpa only [hu, Set.image2_range] using h

/-- Full Euclidean column span is equivalent to full row rank. -/
theorem euclideanColumnSpan_eq_top_iff_rank {I J : Type*} [Fintype I] [Fintype J]
    (A : Matrix I J ℝ) : euclideanColumnSpan A = ⊤ ↔ A.rank = Fintype.card I := by
  classical
  refine ⟨fun h ↦ ?_, euclideanColumnSpan_eq_top_of_rank A⟩
  let e := WithLp.linearEquiv 2 ℝ (I → ℝ)
  have hm := congrArg (Submodule.map e.toLinearMap) h
  have hs : Submodule.span ℝ (Set.range A.col) = ⊤ := by
    simpa [euclideanColumnSpan, Submodule.map_span, ← Set.range_comp,
      Function.comp_def, e] using hm
  apply Matrix.mulVec_surjective_iff_rank_eq_card.mp
  change Function.Surjective A.mulVecLin
  apply LinearMap.range_eq_top.mp
  simpa only [Matrix.range_mulVecLin] using hs

namespace WeightedGraph

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- Pointwise harmonic products, restricted to the interior, as a bilinear map. -/
noncomputable def harmonicProductBilinear (G : WeightedGraph X) (B : Finset X) :
    G.harmonicSubmodule B →ₗ[ℝ] G.harmonicSubmodule B →ₗ[ℝ]
      EuclideanSpace ℝ (Finset.univ \ B : Finset X) :=
  LinearMap.mk₂ ℝ (fun ψ φ ↦ G.harmonicProducts B (ψ, φ))
    (by intro a b c; ext x; change (a.val x + b.val x) * c.val x = _; simp [harmonicProducts, add_mul])
    (by intro a b c; ext x; change (a * b.val x) * c.val x = _; simp [harmonicProducts, mul_assoc])
    (by intro a b c; ext x; change a.val x * (b.val x + c.val x) = _; simp [harmonicProducts, mul_add])
    (by intro a b c; ext x; change b.val x * (a * c.val x) = _; simp [harmonicProducts]; ring)

/-- Harmonic products are symmetric in their two factors. -/
theorem harmonicProducts_swap (G : WeightedGraph X) (B : Finset X)
    (ψ φ : G.harmonicSubmodule B) :
    G.harmonicProducts B (ψ, φ) = G.harmonicProducts B (φ, ψ) := by
  ext x
  exact mul_comm _ _

/-- Products of boundary basis extensions, indexed by unordered pairs with repetition. -/
noncomputable def symmetricBasisProducts (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    Sym2 B → EuclideanSpace ℝ (Finset.univ \ B : Finset X) :=
  Sym2.lift ⟨fun j k ↦ G.harmonicProducts B
    (G.harmonicBasis B hG hB j, G.harmonicBasis B hG hB k),
    fun _ _ ↦ G.harmonicProducts_swap B _ _⟩

/-- The finite harmonic-product matrix in ordinary interior value coordinates. -/
noncomputable def harmonicProductMatrix (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    Matrix (Finset.univ \ B : Finset X) (Sym2 B) ℝ :=
  fun x p ↦ G.symmetricBasisProducts B hG hB p x

/-- Passing from ordered basis pairs to symmetric pairs does not change the product range. -/
theorem symmetricBasisProducts_range (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    Set.range (G.symmetricBasisProducts B hG hB) =
      Set.range (fun p : B × B ↦ G.harmonicProducts B
        (G.harmonicBasis B hG hB p.1, G.harmonicBasis B hG hB p.2)) := by
  ext q
  constructor
  · rintro ⟨p, rfl⟩
    induction p using Sym2.inductionOn with
    | _ j k => exact ⟨(j, k), rfl⟩
  · rintro ⟨⟨j, k⟩, rfl⟩
    exact ⟨Sym2.mk j k, rfl⟩

/-- The finite symmetric basis-product columns span all harmonic products. -/
theorem harmonicProductSpace_eq_columnSpan (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    G.harmonicProductSpace B = euclideanColumnSpan (G.harmonicProductMatrix B hG hB) := by
  have h := bilinear_span_basis (G.harmonicBasis B hG hB) (G.harmonicProductBilinear B)
  change G.harmonicProductSpace B = Submodule.span ℝ
    (Set.range (fun p : B × B ↦ G.harmonicProducts B
      (G.harmonicBasis B hG hB p.1, G.harmonicBasis B hG hB p.2))) at h
  rw [← G.symmetricBasisProducts_range B hG hB] at h
  exact h

/-- Full product span is exactly the paper's full-row-rank condition. -/
theorem harmonicProductSpace_eq_top_iff_rank (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty) :
    G.harmonicProductSpace B = ⊤ ↔
      (G.harmonicProductMatrix B hG hB).rank = Fintype.card (Finset.univ \ B : Finset X) := by
  rw [G.harmonicProductSpace_eq_columnSpan B hG hB, euclideanColumnSpan_eq_top_iff_rank]

/-- Full row rank requires enough symmetric products; it is not implied by this count. -/
theorem interior_card_le_choose_of_rank (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (hrank : (G.harmonicProductMatrix B hG hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    Fintype.card (Finset.univ \ B : Finset X) ≤ (Fintype.card B + 1).choose 2 := by
  simpa only [hrank, Sym2.card] using (G.harmonicProductMatrix B hG hB).rank_le_card_width

/-- The column-count obstruction in the paper's ordinary arithmetic form. -/
theorem interior_card_le_boundary_products_of_rank (G : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hB : (↑B : Set X).Nonempty)
    (hrank : (G.harmonicProductMatrix B hG hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) :
    Fintype.card (Finset.univ \ B : Finset X) ≤
      Fintype.card B * (Fintype.card B + 1) / 2 := by
  simpa only [Nat.choose_two_right, Nat.add_sub_cancel, Nat.mul_comm] using
    G.interior_card_le_choose_of_rank B hG hB hrank

/-- The actual finite product matrix depends only on the known conductances. -/
theorem harmonicProductMatrix_eq_of_conductance_eq (G H : WeightedGraph X) (B : Finset X)
    (hG : G.toSimpleGraph.Connected) (hH : H.toSimpleGraph.Connected)
    (hB : (↑B : Set X).Nonempty) (hc : G.conductance = H.conductance) :
    G.harmonicProductMatrix B hG hB = H.harmonicProductMatrix B hH hB := by
  have hb (j : B) : (G.harmonicBasis B hG hB j).val = (H.harmonicBasis B hH hB j).val := by
    rw [G.harmonicBasis_apply, H.harmonicBasis_apply]
    exact G.harmonicExtension_eq_of_conductance_eq H B hG hH hB hc _
  funext x p
  induction p using Sym2.inductionOn with
  | _ j k =>
    change (G.harmonicBasis B hG hB j).val x * (G.harmonicBasis B hG hB k).val x =
      (H.harmonicBasis B hH hB j).val x * (H.harmonicBasis B hH hB k).val x
    rw [hb j, hb k]

/-- Finite-path first-passage data uniquely determines the vertex masses under
the full row rank condition for the actual symmetric Dirichlet-product matrix. -/
theorem mass_eq_of_firstPassageProbability_eq_of_rank
    (G H : WeightedGraph X) (hG : G.Admissible) (hH : H.Admissible)
    (B : Finset X) (T : ℕ) (hconn : G.toSimpleGraph.Connected)
    (hB : (↑B : Set X).Nonempty)
    (hc : G.conductance = H.conductance) (hm : ∀ x : B, G.mass x = H.mass x)
    (hr : ∀ k, k + 1 < 2 * T → ∀ x y : B,
      G.firstPassageProbability hG k x y = H.firstPassageProbability hH k x y)
    (hucG : G.UniqueContinuation B) (hucH : H.UniqueContinuation B)
    (hT : Fintype.card X ≤ T)
    (hrank : (G.harmonicProductMatrix B hconn hB).rank =
      Fintype.card (Finset.univ \ B : Finset X)) : G.mass = H.mass :=
  G.mass_eq_of_firstPassageProbability_eq H hG hH B T hc hm hr hucG hucH hT
    ((G.harmonicProductSpace_eq_top_iff_rank B hconn hB).mpr hrank)

end WeightedGraph
end VertexCentrality
