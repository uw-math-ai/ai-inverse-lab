"""Small fixed development suite. Reference proofs live separately from packets."""
from copy import deepcopy
from pathlib import Path
import json
from .proof_language import public_language

SUITE_ID = "research-trajectories-v1"
TASKS = {
    "bench-affine-injective": ("Affine injectivity", "construction", "AffineGoal",
        "Over a field K, prove that a*x+b=a*y+b and a≠0 imply x=y.",
        "∀ (K : Type) [Lean.Grind.Field K] (a b x y : K), a ≠ 0 → a*x+b=a*y+b → x=y"),
    "bench-two-measurements": ("Two measurements identify two unknowns", "construction", "MeasurementsGoal",
        "Over a field K, two linear measurements with determinant a*d-b*c≠0 identify both unknowns: if a*x+b*y=a*u+b*v and c*x+d*y=c*u+d*v, prove x=u and y=v.",
        "∀ (K : Type) [Lean.Grind.Field K] (a b c d x y u v : K), a*d-b*c ≠ 0 → a*x+b*y=a*u+b*v → c*x+d*y=c*u+d*v → x=u ∧ y=v"),
    "bench-sum-counterexample": ("Refute recovery from one sum", "counterexample", "SumCounterexampleGoal",
        "Refute injectivity of observing only x+y: over any field K construct x,y,u,v with x+y=u+v and x≠u.",
        "∀ (K : Type) [Lean.Grind.Field K], ∃ x y u v : K, x+y=u+v ∧ x≠u"),
    "bench-product-repair": ("Repair an invalid cancellation argument", "repair", "ProductGoal",
        "Over a field K, if a≠0, c≠0 and a*(b*x+c)=a*(b*y+c), prove b=0 or x=y. Repair the supplied argument without adding b≠0.",
        "∀ (K : Type) [Lean.Grind.Field K] (a b c x y : K), a≠0 → c≠0 → a*(b*x+c)=a*(b*y+c) → b=0 ∨ x=y"),
    "bench-affine-formalize": ("Formalize a supplied correct argument", "formalization", "AffineFormalGoal",
        "Formalize the argument: subtract b from a*x+b=a*y+b, giving a*x=a*y; cancellation of nonzero a gives x=y. Over a field K assume a≠0 and prove x=y.",
        "∀ (K : Type) [Lean.Grind.Field K] (a b x y : K), a ≠ 0 → a*x+b=a*y+b → x=y"),
}
SUITE_BENCHMARK_IDS = tuple(TASKS)


def frozen_problem() -> dict:
    return json.loads(Path(__file__).with_name("q_problem.json").read_text())


def suite_packet(identifier: str, lean_version: str) -> dict:
    title, family, target, statement, goal = TASKS[identifier]
    packet = {
        "id": identifier, "title": title, "short_title": title, "suite_id": SUITE_ID,
        "family": family, "split": "development",
        "statement": statement,
        "assumptions": ["K is an arbitrary field satisfying Lean.Grind.Field.",
                        "Use exactly the quantified hypotheses in the fixed Lean target; do not add hypotheses."],
        "definitions": {"domain": "Exact scalar field arithmetic; no floating-point evidence."},
        "allowed_results": ["Field identities and propositional reasoning from the pinned Lean core library."],
        "visibility": {"reference_proofs": "withheld", "published_proofs": "withheld",
                       "agent_retrieval_tools": False, "agent_filesystem_tools": False,
                       "same_packet_in_both_arms": True,
                       "limitation": "Small development/calibration suite, not a held-out research leaderboard. Related affine tasks share a family and are not independent samples."},
        "review": {"specification": "implementation_review", "reference": "Lean reference check required before suite campaigns",
                   "independent_human_review": "pending"},
        "source": {"title": "Local mathematical-agent calibration tasks", "version": "1"},
        "scope_notes": ["These tasks calibrate proof development; none solves hidden-q recovery.",
                        "Correctness of critique diagnoses still requires independent review. No automatic credit for verbosity or claimed repairs."],
        "milestone_ids": ["bench-affine-injective"] if identifier == "bench-two-measurements" else [],
        "verification_contract": {
            "target_binding": "repository_owned",
            "proof_language": public_language(),
            "kind": "lean4", "status": "ready", "target_id": identifier,
            "target": "MathAILab." + target, "toolchain": "leanprover/lean4:v" + lean_version,
            "specification": f"import Lean\nnamespace MathAILab\ndef {target} : Prop :=\n  {goal}\nend MathAILab\n",
            "instructions": "Submit only a tactic body. Use intro, have, exact, apply, constructor, refine, by, simp only, rw, calc, grind. Local names: K inst a b c d x y u v ha hc hd h1 h2 hx hy. No imports, axioms, sorry, comments or metaprogramming. Optional milestone_attempts may prove only the public milestone_ids; these are recorded separately and never imply the main theorem.",
        },
    }
    if family == "repair":
        packet["seed_argument"] = "Step 1: cancel a. Step 2: subtract c. Step 3: since c≠0, cancel b to get x=y. Hence the disjunction. Identify and repair any unjustified step."
    return deepcopy(packet)
