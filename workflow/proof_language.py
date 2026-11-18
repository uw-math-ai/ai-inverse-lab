"""Shared, explicit language contract for model prompts and the Lean gate."""
import re

VERSION = 3
ALLOWED_WORDS = frozenset("""
intro intros have let show from by exact apply refine simpa simp simp_all only
rw rfl constructor funext calc at all_goals assumption trivial change dsimp
grind first next case obtain rcases with h₁ h₂
a b c d y u v ha hc hd h1 h2 hy
K inst V B h rate q observed hh data x hx hxdata q₁ q₂
oneStepReturn recoverPotential RecoveryGoal FullObservationGoal
MathAILab.oneStepReturn MathAILab.recoverPotential MathAILab.RecoveryGoal
MathAILab.FullObservationGoal
Eq Eq.refl Eq.symm Eq.trans congrArg congrFun And.intro Or.inl Or.inr this
True False True.intro False.elim fun
""".split())
# Binding one of these names must never make a command executable.
FORBIDDEN_WORDS = frozenset("""
sorry admit sorryAx axiom axioms constant opaque theorem def abbrev example
import namespace end section variable variables universe universes
set_option option open export include omit attribute instance deriving
syntax macro macro_rules elab elab_rules initialize builtin_initialize
unsafe partial mutual termination_by decreasing_by run_tac run_elab
native_decide eval trace run_meta by_elab elabTerm do return IO Lean Meta Elab
""".split())
WORDS = re.compile(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*", re.UNICODE)
SAFE_CHARACTERS = re.compile(r"[A-Za-z0-9_\s.,:(){}\[\]=+*/<>!?|;^\-₁₂→←↔≠≤≥∀∃λ⟨⟩·∧∨]*\Z")
LOCAL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_₁₂]*\Z")
TACTIC_HEAD = re.compile(r"(?=(?:^|[\n;|]|\b(?:by|all_goals|first)\b)\s*[>(]*\s*([A-Za-z][A-Za-z0-9_₁₂]*))")


def introduced_names(proof: str) -> set[str]:
    """Only simple intro lists and have/let binders; Lean checks their scopes."""
    names = set()
    for match in re.finditer(r"\bintros?\s+([^\n;]+)", proof):
        tokens = match[1].strip().split()
        if tokens and all(LOCAL_NAME.fullmatch(token) for token in tokens):
            names.update(tokens)
    for match in re.finditer(r"\b(?:have|let)\s+([A-Za-z][A-Za-z0-9_₁₂]*)\s*(?::|:=)", proof):
        names.add(match[1])
    return names - FORBIDDEN_WORDS


def public_language() -> dict:
    return {"version": VERSION, "allowed_identifiers": sorted(ALLOWED_WORDS),
            "local_names": "Simple names introduced by intro/intros/have/let are allowed; Lean checks binding and scope.",
            "library": "Lean core only, not Mathlib. Do not invent field lemma names. grind is available for field and propositional reasoning.",
            "syntax": "Lean 4 tactic body only. Use have name : proposition := by followed by an indented proof. Lean 3 brace-style proofs are not supported. Put explanation in the prose proof field, never here.",
            "restrictions": "No imports, commands, comments, strings, new axioms, proof holes, or executable metaprogramming."}
