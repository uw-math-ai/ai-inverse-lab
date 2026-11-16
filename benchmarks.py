"""Pinned, proof-blind benchmark packets shared by every experimental arm.

This module contains statements and definitions, never a published proof or
expected fixture answers. Source identifiers support audit, not runtime access:
the workflow gives model agents no retrieval or filesystem tools.
"""

from copy import deepcopy
from pathlib import Path


CONTROLLABILITY_ID = "arxiv-2603.20710-lemma-2.3"
DEFAULT_BENCHMARK_ID = "q-recovery-one-step"
UNIQUENESS_ID = "q-uniqueness-full-observation"
BENCHMARK_IDS = (DEFAULT_BENCHMARK_ID, UNIQUENESS_ID, CONTROLLABILITY_ID)
LEAN_DIR = Path(__file__).resolve().parent / "lean"
LEAN_VERSION = "4.35.0"


_BENCHMARK = {
    "id": CONTROLLABILITY_ID,
    "title": "Exact controllability of discrete graph heat flow",
    "source": {
        "arxiv_id": "2603.20710",
        "version": "v1",
        "version_date": "2026-03-21",
        "title": "Vertex centrality reconstruction in an inverse problem for information diffusion",
        "url": "https://arxiv.org/html/2603.20710v1#S2.SS2",
        "statement_locator": "Lemma 2.3; Assumption 1; equation (2.1); definition of W in Section 2.2",
        "checked_on": "2026-11-08",
        "evidence": "Statement and definitions checked against the official arXiv v1 HTML.",
        "upstream_result": "arXiv:2107.00494v2, Proposition 5.3 (withheld from agents)",
        "numbering_note": "In arXiv:2603.20710v1, result 2.4 is a Proposition, not a Lemma.",
    },
    "statement": (
        "Under the assumptions below, the map W: f -> U^f(T, .) from "
        "ell^2({0,...,T-1} x B) onto ell^2(X) is surjective. "
        "Equivalently, every real target g on X can be reached at time T "
        "by a source f supported in {0,...,T-1} x B. Prove this for every "
        "graph and every T satisfying the assumptions."
    ),
    "assumptions": [
        "X is a nonempty finite vertex set; the graph is connected, undirected, and simple.",
        "The masses mu_x are positive; w_xy = w_yx is positive exactly on edges and zero otherwise.",
        "B is a nonempty subset of X.",
        "For every x, mu_x >= sum_{y adjacent to x} w_xy.",
        "No nonzero eigenfunction of -Delta vanishes at every vertex of B; this concerns every vector in every eigenspace, including repeated eigenvalues.",
        "T is an integer with T >= |X|.",
    ],
    "definitions": {
        "laplacian": "(Delta u)(x) = (1/mu_x) sum_y w_xy (u(y)-u(x)).",
        "inner_product": "<u,v>_X = sum_{x in X} mu_x u(x)v(x); functions are real valued.",
        "time_derivative": "D_t U(t,x) = U(t+1,x)-U(t,x).",
        "source": "f(t,x) is arbitrary for 0 <= t < T and x in B, and zero otherwise.",
        "heat_solution": "U^f(0,x)=0 and D_t U^f(t,x)-Delta U^f(t,x)=f(t,x).",
        "W": "Wf = U^f(T, .).",
        "matrix_convention": "Vertices are 0,...,n-1; states are columns. A=I+Delta acts on functions; E has coordinate columns for B. Controls enter as E f_t without a mass factor.",
    },
    "allowed_results": [
        "Elementary finite-dimensional real linear algebra, including rank-nullity, orthogonal complements, and adjoints.",
        "The spectral theorem for a self-adjoint linear operator on a finite-dimensional real inner-product space.",
        "The Cayley-Hamilton theorem, polynomial interpolation, and elementary induction.",
        "Definitions and assumptions in this packet; other intermediate claims must be proved.",
    ],
    "visibility": {
        "published_proofs": "withheld",
        "upstream_controllability_results": "withheld, including Proposition 5.3 of arXiv:2107.00494",
        "other_paper_results": "withheld, including Lemmas 2.1/2.3 and Proposition 2.4 as proof aids",
        "agent_retrieval_tools": False,
        "agent_filesystem_tools": False,
        "verifier_expected_answers": "withheld",
        "same_packet_in_both_arms": True,
        "limitation": "Prompt withholding cannot rule out prior model exposure to published material.",
    },
    "scope_notes": [
        "This is a Task 2 workflow benchmark, not a proof of potential recovery or stability.",
        "The source lemma has q=0. Vertex mass/centrality mu and a Schrödinger potential q are different quantities.",
        "Positive transition probabilities are part of the published assumptions; investigate weaker assumptions only as separately labelled conjectures.",
        "Finite examples can falsify a submitted rank assertion but cannot prove the universal theorem or validate a natural-language proof.",
    ],
    "verification_contract": {
        "kind": "lean4", "status": "formalization_pending", "target_id": None,
        "instructions": "The full controllability statement has not been formalized. "
            "Submit certificate=null. No finite ranks or unrelated Lean theorem can verify it.",
    },
}


def load_benchmark(benchmark_id: str = DEFAULT_BENCHMARK_ID) -> dict:
    """Return the fixed statement and specification, never a reference proof."""
    if benchmark_id == CONTROLLABILITY_ID:
        packet = deepcopy(_BENCHMARK)
        packet["verification_contract"] = {
            "kind": "lean4", "status": "formalization_pending", "target_id": None,
            "instructions": "The full controllability statement has not been formalized. "
                "Submit certificate=null. No finite ranks or unrelated Lean theorem can verify it.",
        }
        packet["short_title"] = "Lemma 2.3"
        packet["scope_notes"][-1] = "The published controllability theorem remains a separate, unformalized benchmark."
        return packet
    if benchmark_id not in (DEFAULT_BENCHMARK_ID, UNIQUENESS_ID):
        raise ValueError(f"Unknown benchmark: {benchmark_id}")
    recovery = benchmark_id == DEFAULT_BENCHMARK_ID
    target = "MathAILab.RecoveryGoal" if recovery else "MathAILab.FullObservationGoal"
    packet = {
        "id": benchmark_id,
        "short_title": "Recover q on B" if recovery else "Full observation uniqueness",
        "title": "Recover the potential at observed vertices" if recovery else "Uniqueness with every vertex observed",
        "source": {
            "title": "Local algebraic foundation for the forward simulator's killing kernel",
            "statement_locator": "One-step diagonal recovery" if recovery else "Injectivity of full diagonal observations",
            "implementation": "forwardSimulation/forward_sim/model.py:build_kernel",
            "version": "1", "checked_on": "2026-11-08",
        },
        "statement": (
            r"For a scalar field, a known nonzero step $h$, known rates $r(x)$, and exact observations "
            r"$d(x)=1-h(r(x)+q(x))$ for $x\in B$, prove "
            r"$q(x)=(1-d(x))/h-r(x)$ for every $x\in B$."
        ) if recovery else (
            r"For a scalar field, fixed $h\ne0$ and known rates $r(x)$, if "
            r"$1-h(r(x)+q_1(x))=1-h(r(x)+q_2(x))$ at every vertex $x$, prove $q_1=q_2$."
        ),
        "assumptions": [
            r"Scalars belong to an arbitrary field (Lean.Grind.Field); arithmetic is exact.",
            r"The vertex type $X$ is arbitrary; these algebraic lemmas do not require finiteness.",
            r"The time step $h\ne0$ is fixed and known; $r:X\to K$ is known.",
            r"Exact diagonal observations are given at each vertex in $B$." if recovery
                else r"All vertices are observed ($B=X$); both potentials use the same $h$ and $r$.",
        ],
        "definitions": {
            "exit_rate": r"For the graph application, $r(x)=\mu_x^{-1}\sum_y w_{xy}$; masses and weights are known.",
            "one_step_return": r"$d(x)=F_{xx}(1)=K_q(x,x)=1-h(r(x)+q(x))$. Return means first positive-time return, including a holding step.",
            "reconstruction": r"$\widehat q(x)=\frac{1-d(x)}{h}-r(x)$.",
            "formal_domain": "The Lean target quantifies over fields through Lean.Grind.Field. A concrete real-number/probability model is not imported in this foundation.",
        },
        "allowed_results": [
            "Field identities and equality reasoning from the pinned Lean core library.",
            "Function extensionality for full-observation uniqueness.",
            "Only the definitions and fixed goal in the supplied Lean specification; reference proofs are withheld.",
        ],
        "visibility": {
            "published_proofs": "withheld", "reference_proofs": "withheld",
            "agent_retrieval_tools": False, "agent_filesystem_tools": False,
            "same_packet_in_both_arms": True,
            "limitation": "This elementary identity is a formalization milestone, not a difficult discovery benchmark. Prompt withholding cannot prevent prior knowledge.",
        },
        "scope_notes": [
            "Lean verifies the fixed algebraic target, not the accuracy of all accompanying prose or the modelling of an experiment.",
            r"This only recovers $q$ on $B$. Full recovery follows when $B=X$; hidden vertices and weakest geometric conditions remain open work.",
            "The diagonal relation matches the simulator's source code. The Markov chain/first-return interpretation is not yet formalized in Lean.",
            r"The probabilistic application additionally needs $h>0$, positive masses, $q\ge0$, and $h(r(x)+q(x))\le1$.",
            r"Potential $q$ differs from mass $\mu$. No reconstruction stability bound, noisy-data guarantee, or ML inverse model is claimed.",
        ],
        "verification_contract": {
            "kind": "lean4", "status": "ready", "target_id": benchmark_id,
            "target": target, "toolchain": "leanprover/lean4:v" + LEAN_VERSION,
            "specification": (LEAN_DIR / "MathAILab" / "Spec.lean").read_text(encoding="utf-8"),
            "declaration": f"theorem MathAILab.candidate : {target} := by\n  <proof body>",
            "certificate": {"kind": "lean4", "target_id": benchmark_id, "proof": "Lean tactic body only, without the enclosing by"},
            "instructions": "Copy the claim and assumptions exactly. Submit only a proof body for the fixed goal. "
                "The verifier supplies imports, definitions, declaration and axiom audit. Never change the target or add hypotheses. "
                "No sorry/admit, new axioms, commands, comments, strings, macros, imports, IO, native_decide or metaprogramming. "
                "Use intro, have, simp only, rw, exact, apply, funext, rfl, constructor, calc or grind. "
                "Available local names: K inst V B h rate q observed hh data x hx hxdata q₁ q₂. "
                "Available definitions: oneStepReturn, recoverPotential, RecoveryGoal, FullObservationGoal, qualified by MathAILab if needed. "
                "Certificate may be null when no formal proof is available. Compiler errors are feedback, not counterexamples.",
        },
    }
    return packet
