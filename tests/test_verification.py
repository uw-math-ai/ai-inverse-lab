"""Real Lean checks plus fail-closed proof, target, and receipt regressions."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from workflow.backends import DemoBackend
from workflow.benchmarks import CONTROLLABILITY_ID, DEFAULT_BENCHMARK_ID, UNIQUENESS_ID, load_benchmark
from workflow.evaluation import score_iteration
from workflow.schemas import Proposal
from workflow.verification import is_verified_receipt, verify, _compiler_environment


def candidate(benchmark_id=DEFAULT_BENCHMARK_ID, stage="Repair"):
    packet = load_benchmark(benchmark_id)
    proposal = DemoBackend().invoke(stage, Proposal, [{"content": json.dumps({"benchmark": packet})}])["parsed"]
    return packet, proposal


class VerificationTests(unittest.TestCase):
    def test_real_lean_proves_both_fixed_targets(self):
        for benchmark_id in (DEFAULT_BENCHMARK_ID, UNIQUENESS_ID):
            with self.subTest(benchmark_id=benchmark_id):
                packet, proposal = candidate(benchmark_id)
                result = verify(proposal, packet)
                self.assertEqual(result["status"], "verified", result)
                self.assertTrue(result["independently_verified"])
                self.assertTrue(is_verified_receipt(proposal, result))
                self.assertEqual(result["compiler"]["return_code"], 0)
                self.assertEqual(set(result["axioms"]), {"propext", "Classical.choice", "Quot.sound"})
                self.assertIn("#print axioms MathAILab.candidate", result["source"])
                self.assertEqual(score_iteration(proposal, result)["score"], 3)

    def test_wrong_proof_is_not_a_counterexample(self):
        packet, proposal = candidate(stage="CA")
        result = verify(proposal, packet)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["independently_verified"])
        self.assertIn("does not refute", result["summary"])
        self.assertIn('"severity":"error"', result["compiler"]["output"])
        self.assertEqual(score_iteration(proposal, result)["score"], 1)

    def test_missing_proof_remains_unverified(self):
        packet, proposal = candidate()
        proposal["certificate"] = None
        with patch("workflow.verification._compile") as compiler:
            result = verify(proposal, packet)
        compiler.assert_not_called()
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["independently_verified"])

    def test_rejects_holes_commands_and_metaprogramming_before_compilation(self):
        packet, proposal = candidate()
        attacks = [
            "sorry", "admit", "exact sorryAx _ false", "native_decide",
            "run_tac Lean.logInfo \"fake success\"",
            "#eval IO.println \"x\"", "import MathAILab.Proofs",
            "axiom falseProof : False", "set_option debug.skipKernelTC true in rfl",
            "rfl\nend MathAILab\ntheorem fake : True := by trivial",
            "rfl /- comment -/", "rfl -- comment", "rfl\n\u202e",
            "exact MathAILab.recoveryFormula", "__import__('os').system('false')",
        ]
        for body in attacks:
            with self.subTest(body=body), patch("workflow.verification._compile") as compiler:
                proposal["certificate"]["proof"] = body
                result = verify(proposal, packet)
                self.assertEqual(result["status"], "rejected")
                self.assertFalse(result["independently_verified"])
                compiler.assert_not_called()

    def test_fixed_target_and_assumptions_cannot_be_substituted(self):
        packet, proposal = candidate()
        variants = [
            (packet, dict(proposal, claim="True")),
            (packet, dict(proposal, assumptions=[])),
            (dict(packet, statement="True"), proposal),
            (packet, dict(proposal, certificate={**proposal["certificate"], "target_id": UNIQUENESS_ID})),
            (packet, dict(proposal, certificate={"kind": "fixture_ranks", "ranks": []})),
            (packet, dict(proposal, certificate={**proposal["certificate"], "imports": ["MathAILab.Proofs"]})),
        ]
        modified = deepcopy(packet)
        modified["verification_contract"]["target"] = "True"
        variants.append((modified, proposal))
        for benchmark, attempt in variants:
            with self.subTest(attempt=attempt), patch("workflow.verification._compile") as compiler:
                self.assertEqual(verify(attempt, benchmark)["status"], "rejected")
                compiler.assert_not_called()

    def test_axiom_audit_is_required_and_sorry_axioms_are_rejected(self):
        packet, proposal = candidate()
        for message in (
            "", "'MathAILab.candidate' depends on axioms: [sorryAx]",
            "'MathAILab.candidate' depends on axioms: [hiddenAxiom]",
            "'Another.theorem' depends on axioms: [propext]",
        ):
            output = json.dumps({"severity": "information", "data": message})
            fake = {"return_code": 0, "output": output, "timed_out": False, "output_truncated": False}
            with self.subTest(message=message), patch("workflow.verification._compile", return_value=fake):
                result = verify(proposal, packet)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(score_iteration(proposal, result)["score"], 1)

    def test_serialized_or_mutated_receipts_do_not_certify(self):
        packet, proposal = candidate()
        result = verify(proposal, packet)
        self.assertTrue(is_verified_receipt(proposal, result))
        forged = json.loads(json.dumps(result))
        self.assertFalse(is_verified_receipt(proposal, forged))
        self.assertEqual(score_iteration(proposal, forged)["score"], 1)
        changed = {**proposal, "proof": "A different proof"}
        self.assertFalse(is_verified_receipt(changed, result))
        result["source"] += "\n-- changed"
        self.assertFalse(is_verified_receipt(proposal, result))

    def test_missing_compiler_timeout_and_truncated_output_fail_closed(self):
        packet, proposal = candidate()
        unavailable = {"available": False, "reason": "Synthetic missing Lean"}
        with patch("workflow.verification.lean_status", return_value=unavailable):
            self.assertEqual(verify(proposal, packet)["status"], "unavailable")
        for timeout, truncated, expected in ((True, False, "timeout"), (False, True, "failed")):
            with patch("workflow.verification._compile", return_value={
                "return_code": 0, "output": "", "timed_out": timeout, "output_truncated": truncated,
            }):
                result = verify(proposal, packet)
                self.assertEqual(result["status"], expected)
                self.assertFalse(result["independently_verified"])

    def test_compiler_does_not_inherit_api_credentials_or_lean_path(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret", "LEAN_PATH": "/bad/path"}):
            environment = _compiler_environment()
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("LEAN_PATH", environment)

    def test_controllability_is_preserved_but_not_falsely_formalized(self):
        packet = load_benchmark(CONTROLLABILITY_ID)
        self.assertEqual(packet["source"]["version"], "v1")
        self.assertIn("surjective", packet["statement"])
        with patch("workflow.verification._compile") as compiler:
            result = verify({}, packet)
        self.assertEqual(result["status"], "formalization_pending")
        self.assertFalse(result["independently_verified"])
        compiler.assert_not_called()

    def test_public_packet_contains_goals_but_no_reference_proof(self):
        packet = load_benchmark()
        self.assertEqual(packet["verification_contract"]["toolchain"], "leanprover/lean4:v4.35.0")
        self.assertEqual(packet["visibility"]["reference_proofs"], "withheld")
        self.assertFalse(packet["visibility"]["agent_filesystem_tools"])
        specification = packet["verification_contract"]["specification"]
        self.assertIn("def RecoveryGoal", specification)
        self.assertNotIn("theorem recoveryFormula", specification)
        self.assertNotIn("intro K inst", specification)
        with self.assertRaises(ValueError):
            load_benchmark("unknown")


if __name__ == "__main__":
    unittest.main()
