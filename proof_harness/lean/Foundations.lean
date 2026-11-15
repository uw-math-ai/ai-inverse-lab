import Mathlib.Tactic
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.LinearAlgebra.Matrix.Charpoly.Basic

/- Trusted baseline facts. Each is proved, not postulated. The pilot tests
small algebraic proof obligations, not the complete reconstruction theorem. -/
namespace Baseline

theorem recurrence_step (a u f v : ℝ) (h : v = a * u + f) :
    v - f = a * u := by rw [h]; ring

theorem renewal_two (r1 r2 p1 p2 : ℝ) (h1 : p1 = r1)
    (h2 : p2 = r2 + r1 * p1) : p2 = r2 + r1 * r1 := by rw [h2, h1]

theorem weighted_symmetry (m1 m2 a12 a21 : ℝ)
    (h : m1 * a12 = m2 * a21) (u1 u2 v1 v2 : ℝ) :
    m1 * (a12 * u2) * v1 + m2 * (a21 * u1) * v2 =
    m1 * u1 * (a12 * v2) + m2 * u2 * (a21 * v1) := by
  have scaled := congrArg (fun z : ℝ => z * (u2*v1-u1*v2)) h
  nlinarith [scaled]

theorem cancel_positive_weight (m a : ℝ) (hm : 0 < m) :
    a / m = 0 ↔ a = 0 := by simp [div_eq_zero_iff, ne_of_gt hm]

theorem annihilate_span {V : Type*} [AddCommGroup V] [Module ℝ V]
    (L : V →ₗ[ℝ] ℝ) (S : Set V) (h : ∀ v ∈ S, L v = 0) :
    ∀ v ∈ Submodule.span ℝ S, L v = 0 := by
  intro v hv
  have hs : Submodule.span ℝ S ≤ LinearMap.ker L :=
    Submodule.span_le.mpr (by intro x hx; exact h x hx)
  exact hs hv

theorem injective_cancel {V U : Type*} (W : V → U)
    (hW : Function.Injective W) (x y : V) (h : W x = W y) : x = y := hW h

theorem normal_equation {F V : Type*} [AddCommGroup F] [Module ℝ F]
    [AddCommGroup V] [Module ℝ V] (W : F →ₗ[ℝ] V) (J : V →ₗ[ℝ] F)
    (hJ : Function.Injective J) (h : F) (p : V)
    (eqn : J (W h) = J p) : W h = p := hJ eqn

theorem gram_zero {F V : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (W : F →ₗ[ℝ] V) (J : V →ₗ[ℝ] F)
    (adj : ∀ x y, inner ℝ (W x) y = inner ℝ x (J y))
    (h : F) (hz : J (W h) = 0) : W h = 0 := by
  have hh : inner ℝ (W h) (W h) = (0 : ℝ) := by rw [adj, hz, inner_zero_right]
  exact (inner_self_eq_zero).mp hh

theorem moments_unique {V : Type*} [AddCommGroup V] [Module ℝ V]
    (L : V →ₗ[ℝ] ℝ) (hL : Function.Injective L)
    (x y : V) (b : ℝ) (hx : L x = b) (hy : L y = b) : x = y :=
  hL (hx.trans hy.symm)

theorem scalar_recover (c m d : ℝ) (hm : m ≠ 0) (hd : d ≠ 0)
    (h : d = c / m) : m = c / d := by
  have hc : d * m = c := (eq_div_iff hm).mp h
  apply (eq_div_iff hd).mpr
  nlinarith [hc]

theorem difference_product (a b x y : ℝ) :
    a * b - x * y = (a - x) * b + x * (b - y) := by ring

end Baseline
