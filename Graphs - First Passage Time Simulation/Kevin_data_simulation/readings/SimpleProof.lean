import Lean

def IsOdd (n : Nat) : Prop := ∃ k, n = 2 * k + 1

def IsEven (n : Nat) : Prop := ∃ k, n = 2 * k

theorem odd_add_odd_is_even {a b : Nat}
    (ha : IsOdd a) (hb : IsOdd b) : IsEven (a + b) := by
  obtain ⟨m, rfl⟩ := ha
  obtain ⟨n, rfl⟩ := hb
  refine ⟨m + n + 1, ?_⟩
  omega

#check odd_add_odd_is_even
