"""Evaluator-only reference fixtures. Never included in live agent packets."""
REFERENCE_PROOFS = {
    "bench-affine-injective": "intro K inst a b x y ha h1\ngrind",
    "bench-two-measurements": "intro K inst a b c d x y u v hd h1 h2\nconstructor <;> grind",
    "bench-sum-counterexample": "intro K inst\nexact ⟨0, 1, 1, 0, by constructor <;> grind⟩",
    "bench-product-repair": "intro K inst a b c x y ha hc h1\ngrind",
    "bench-affine-formalize": "intro K inst a b x y ha h1\ngrind",
}


def check_suite() -> dict:
    from .benchmarks import load_benchmark
    from .verification import verify
    results = {}
    for identifier, proof in REFERENCE_PROOFS.items():
        packet = load_benchmark(identifier)
        proposal = {"claim": packet["statement"], "assumptions": packet["assumptions"],
                    "certificate": {"kind": "lean4", "target_id": identifier, "proof": proof}}
        results[identifier] = verify(proposal, packet)
    return {"suite_id": "research-trajectories-v1", "passed": all(r["status"] == "verified" for r in results.values()),
            "independent_human_review": "pending", "results": results}
