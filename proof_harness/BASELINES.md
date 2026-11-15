# Pilot baseline reference

This is the exact implemented pilot catalogue. All custom lemmas are proved in `lean/Foundations.lean`.
The task and baseline descriptions are sent to the agents; this document itself is not.

## B0 — Elementary foundations

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0

## B1 — Recurrences

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.recurrence_step`: (a u f v : R) (h : v = a*u+f) : v-f=a*u
- `Baseline.renewal_two`: (r1 r2 p1 p2 : R) (h1 : p1=r1) (h2 : p2=r2+r1*p1) : p2=r2+r1*r1

## B2 — Weighted graph algebra

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.weighted_symmetry`: (m1 m2 a12 a21 : R) (h : m1*a12=m2*a21) (u1 u2 v1 v2 : R) : m1*(a12*u2)*v1+m2*(a21*u1)*v2=m1*u1*(a12*v2)+m2*u2*(a21*v1)
- `Baseline.cancel_positive_weight`: (m a : R) (hm : 0<m) : a/m=0 iff a=0

## B3 — Span and injectivity

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.annihilate_span`: {V : Type} [AddCommGroup V] [Module R V] (L : V ->_l[R] R) (S : Set V) (h : forall v in S, L v=0) : forall v in Submodule.span R S, L v=0
- `Baseline.injective_cancel`: {V U : Type} (W : V -> U) (hW : Function.Injective W) (x y : V) (h : W x=W y) : x=y

## B4 — Gram and adjoint algebra

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.normal_equation`: (W : F ->_l[R] V) (J : V ->_l[R] F) (hJ : Function.Injective J) (h : F) (p : V) (eqn : J (W h)=J p) : W h=p
- `Baseline.gram_zero`: (W : F ->_l[R] V) (J : V ->_l[R] F) (adj : forall x y, inner ℝ (W x) y=inner ℝ x (J y)) (h : F) (hz : J (W h)=0) : W h=0; F,V are real inner product spaces

## B5 — Moment recovery

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.moments_unique`: (L : V ->_l[R] R) (hL : Function.Injective L) (x y : V) (b : R) (hx : L x=b) (hy : L y=b) : x=y
- `Baseline.scalar_recover`: (c m d : R) (hm : m!=0) (hd : d!=0) (h : d=c/m) : m=c/d
- `Baseline.difference_product`: (a b x y : R) : a*b-x*y=(a-x)*b+x*(b-y)

## B6 — All supplied packs

- `Eq.symm`: a = b -> b = a
- `Eq.trans`: a = b -> b = c -> a = c
- `congrArg`: h : x = y -> f x = f y
- `id`: identity
- `Function.Injective`: forall x y, f x = f y -> x = y
- `LinearMap.map_sub`: L (x-y) = L x - L y
- `LinearMap.map_add`: L (x+y) = L x + L y
- `LinearMap.map_zero`: L 0 = 0
- `inner_zero_right`: inner x 0 = 0
- `inner_self_eq_zero`: inner x x = 0 iff x = 0
- `sub_eq_zero`: a-b=0 iff a=b
- `div_eq_iff`: b != 0 -> (a/b=c iff a=c*b)
- `eq_div_iff`: b != 0 -> (a=c/b iff a*b=c)
- `ne_of_gt`: 0 < x -> x != 0
- `Baseline.recurrence_step`: (a u f v : R) (h : v = a*u+f) : v-f=a*u
- `Baseline.renewal_two`: (r1 r2 p1 p2 : R) (h1 : p1=r1) (h2 : p2=r2+r1*p1) : p2=r2+r1*r1
- `Baseline.weighted_symmetry`: (m1 m2 a12 a21 : R) (h : m1*a12=m2*a21) (u1 u2 v1 v2 : R) : m1*(a12*u2)*v1+m2*(a21*u1)*v2=m1*u1*(a12*v2)+m2*u2*(a21*v1)
- `Baseline.cancel_positive_weight`: (m a : R) (hm : 0<m) : a/m=0 iff a=0
- `Baseline.annihilate_span`: {V : Type} [AddCommGroup V] [Module R V] (L : V ->_l[R] R) (S : Set V) (h : forall v in S, L v=0) : forall v in Submodule.span R S, L v=0
- `Baseline.injective_cancel`: {V U : Type} (W : V -> U) (hW : Function.Injective W) (x y : V) (h : W x=W y) : x=y
- `Baseline.normal_equation`: (W : F ->_l[R] V) (J : V ->_l[R] F) (hJ : Function.Injective J) (h : F) (p : V) (eqn : J (W h)=J p) : W h=p
- `Baseline.gram_zero`: (W : F ->_l[R] V) (J : V ->_l[R] F) (adj : forall x y, inner ℝ (W x) y=inner ℝ x (J y)) (h : F) (hz : J (W h)=0) : W h=0; F,V are real inner product spaces
- `Baseline.moments_unique`: (L : V ->_l[R] R) (hL : Function.Injective L) (x y : V) (b : R) (hx : L x=b) (hy : L y=b) : x=y
- `Baseline.scalar_recover`: (c m d : R) (hm : m!=0) (hd : d!=0) (h : d=c/m) : m=c/d
- `Baseline.difference_product`: (a b x y : R) : a*b-x*y=(a-x)*b+x*(b-y)

## Fixed proof targets

### renewal3 (centrality)

Derive the third return probability from the positive-time renewal recurrence. r1,r2,r3 are first-return masses; p1,p2,p3 are occupancy masses.

```lean
(r1 r2 r3 p1 p2 p3 : ℝ) (h1 : p1 = r1) (h2 : p2 = r2 + r1*p1) (h3 : p3 = r3 + r2*p1 + r1*p2)
⊢ p3 = r3 + 2*r1*r2 + r1^3
```

### forced3 (centrality)

Three-step forced heat recurrence, in a scalar mode. Derive the response without external facts.

```lean
(a f0 f1 f2 u1 u2 u3 : ℝ) (h1 : u1 = f0) (h2 : u2 = a*u1+f1) (h3 : u3 = a*u2+f2)
⊢ u3 = a^2*f0 + a*f1 + f2
```

### weighted_pair (centrality)

Prove symmetry for a two-coordinate weighted transition operator including arbitrary diagonal entries. This is a local self-adjointness calculation.

```lean
(m1 m2 a12 a21 d1 d2 u1 u2 v1 v2 : ℝ) (h : m1*a12 = m2*a21)
⊢ m1*(d1*u1+a12*u2)*v1 + m2*(a21*u1+d2*u2)*v2 = m1*u1*(d1*v1+a12*v2) + m2*u2*(a21*v1+d2*v2)
```

### span_observable (centrality)

A boundary observation functional vanishing on a set also vanishes on its linear span. Apply it to a sum of two reachable states.

```lean
{V : Type} [AddCommGroup V] [Module ℝ V] (L : V →ₗ[ℝ] ℝ) (S : Set V) (h : ∀ v ∈ S, L v = 0) (x y : V) (hx : x ∈ Submodule.span ℝ S) (hy : y ∈ Submodule.span ℝ S)
⊢ L (x+y) = 0
```

### control_pairing (centrality)

A control satisfies a normal equation. Assuming the adjoint is injective, show its reached state has the required measured pairing. Injectivity is an explicit hypothesis, not a proved controllability theorem.

```lean
{F V : Type} [AddCommGroup F] [Module ℝ F] [AddCommGroup V] [Module ℝ V] (W : F →ₗ[ℝ] V) (J : V →ₗ[ℝ] F) (hJ : Function.Injective J) (h : F) (p : V) (L : V →ₗ[ℝ] ℝ) (b : ℝ) (eqn : J (W h) = J p) (hp : L p = b)
⊢ L (W h) = b
```

### gram_unique (centrality)

Use the adjoint relation to show two controls with the same Gram image reach the same state.

```lean
{F V : Type} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [NormedAddCommGroup V] [InnerProductSpace ℝ V] (W : F →ₗ[ℝ] V) (J : V →ₗ[ℝ] F) (adj : ∀ x y, inner ℝ (W x) y = inner ℝ x (J y)) (h k : F) (hk : J (W h) = J (W k))
⊢ W h = W k
```

### path_recovery (centrality)

On path 0-1-2 with conductances a,b and known endpoint mass m0, p2(0,0)=(1-a/m0)^2+a^2/(m0*m1). Recover unknown m1 from excess d. This task checks the algebra after that path identity, not its probabilistic derivation.

```lean
(a m0 m1 d : ℝ) (hm0 : m0 ≠ 0) (hm1 : m1 ≠ 0) (hd : d ≠ 0) (h : d = (a^2/m0)/m1)
⊢ m1 = a^2 / (m0*d)
```

### moment_unique (centrality)

If a linear observation map is injective, two parameter vectors with the same measured moments agree. Apply this after subtracting a known boundary contribution.

```lean
{V : Type} [AddCommGroup V] [Module ℝ V] (L : V →ₗ[ℝ] ℝ) (hL : Function.Injective L) (x y : V) (b c : ℝ) (hx : L x + c = b) (hy : L y + c = b)
⊢ x = y
```

### absorption_difference (absorption)

Scalar analogue of the discrete Duhamel difference identity. This is a preparatory algebraic task; no absorption uniqueness assumption is implied.

```lean
(a q r : ℝ)
⊢ (a-q)^3-(a-r)^3 = (r-q)*((a-q)^2+(a-q)*(a-r)+(a-r)^2)
```
