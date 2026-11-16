import Lean

/-!
Algebraic foundations for potential recovery. These definitions, but none of
the proofs in Proofs.lean, are included in the agent's public packet.

The forward simulator has diagonal K(x,x) = 1 - h * (rate(x) + q(x)),
where rate(x) = sum_y w(x,y) / mu(x). For first POSITIVE-time return,
F_xx(1) = K(x,x). Here we formalize the algebra of that diagonal relation;
we do not yet formalize probability, graph geometry, or hidden vertices.

The scalar type is an arbitrary field satisfying Lean.Grind.Field, not
floating-point arithmetic. No graph assumption or positivity is needed for
these algebraic identities; a probabilistic interpretation additionally
requires positive masses, h > 0, q >= 0 and a substochastic kernel.
-/
namespace MathAILab

def oneStepReturn {K : Type} [Lean.Grind.Field K] (h rate q : K) : K :=
  1 - h * (rate + q)

def recoverPotential {K : Type} [Lean.Grind.Field K] (h rate observed : K) : K :=
  (1 - observed) / h - rate

def RecoveryGoal : Prop :=
  ∀ (K : Type) [Lean.Grind.Field K] (V : Type) (B : V → Prop)
    (h : K) (rate q observed : V → K),
    h ≠ 0 →
    (∀ x, B x → observed x = oneStepReturn h (rate x) (q x)) →
    ∀ x, B x → recoverPotential h (rate x) (observed x) = q x

def FullObservationGoal : Prop :=
  ∀ (K : Type) [Lean.Grind.Field K] (V : Type)
    (h : K) (rate q₁ q₂ : V → K),
    h ≠ 0 →
    (∀ x, oneStepReturn h (rate x) (q₁ x) =
          oneStepReturn h (rate x) (q₂ x)) →
    q₁ = q₂

end MathAILab
