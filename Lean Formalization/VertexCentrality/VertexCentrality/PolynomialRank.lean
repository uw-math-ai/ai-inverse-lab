module

public import VertexCentrality.DirichletPolynomial
public import TauCeti.MeasureTheory.Measure.MvPolynomialZeroLocus

/-! # Generic full row rank from one polynomial witness

The determinant of the row Gram matrix is a single polynomial certificate.
This avoids choosing a minor while retaining the explicit nonzero-witness
requirement. Tau Ceti supplies the nullity theorem for polynomial zero loci.
-/

@[expose] public section

namespace VertexCentrality

open MeasureTheory

variable {ι κ σ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ]

/-- Full row rank over the reals is equivalent to a nonzero Gram determinant. -/
theorem rank_eq_card_iff_gram_det_ne_zero (A : Matrix ι κ ℝ) :
    A.rank = Fintype.card ι ↔ (A * A.transpose).det ≠ 0 := by
  rw [← Matrix.rank_self_mul_transpose A]
  constructor
  · intro h
    exact isUnit_iff_ne_zero.mp ((Matrix.isUnit_iff_isUnit_det _).mp
      (Matrix.mulVec_surjective_iff_isUnit.mp
        (Matrix.mulVec_surjective_iff_rank_eq_card.mpr h)))
  · exact Matrix.rank_of_det_ne_zero

/-- A single polynomial whose nonvanishing certifies full row rank. -/
noncomputable def polynomialRankCertificate (A : Matrix ι κ (MvPolynomial σ ℝ)) :
    MvPolynomial σ ℝ := (A * A.transpose).det

/-- Evaluation commutes with the Gram determinant certificate. -/
theorem polynomialRankCertificate_eval (A : Matrix ι κ (MvPolynomial σ ℝ))
    (z : σ → ℝ) :
    MvPolynomial.eval z (polynomialRankCertificate A) =
      (A.map (MvPolynomial.eval z) *
        (A.map (MvPolynomial.eval z)).transpose).det := by
  simp [polynomialRankCertificate, RingHom.map_det, RingHom.mapMatrix_apply,
    Matrix.map_mul]
  rfl

/-- One full-rank evaluation proves that the certificate polynomial is nonzero. -/
theorem polynomialRankCertificate_ne_zero (A : Matrix ι κ (MvPolynomial σ ℝ))
    (z : σ → ℝ) (hz : (A.map (MvPolynomial.eval z)).rank = Fintype.card ι) :
    polynomialRankCertificate A ≠ 0 := by
  intro h
  have hn := (rank_eq_card_iff_gram_det_ne_zero _).mp hz
  rw [← polynomialRankCertificate_eval A z, h, map_zero] at hn
  exact hn rfl

/-- The rank-deficient locus of a polynomial matrix is null if one full-rank
witness exists. There is no assumption that every parameter is a valid graph. -/
theorem volume_rank_deficient_eq_zero [Fintype σ]
    (A : Matrix ι κ (MvPolynomial σ ℝ)) (z : σ → ℝ)
    (hz : (A.map (MvPolynomial.eval z)).rank = Fintype.card ι) :
    volume {w : σ → ℝ | (A.map (MvPolynomial.eval w)).rank ≠ Fintype.card ι} = 0 := by
  have hs : {w : σ → ℝ | (A.map (MvPolynomial.eval w)).rank ≠ Fintype.card ι} =
      {w : σ → ℝ | MvPolynomial.eval w (polynomialRankCertificate A) = 0} := by
    ext w
    simp [rank_eq_card_iff_gram_det_ne_zero, polynomialRankCertificate_eval]
  rw [hs]
  exact MvPolynomial.volume_setOfPred_eval_eq_zero _
    (polynomialRankCertificate_ne_zero A z hz)

end VertexCentrality
