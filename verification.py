"""Lean verification of fixed targets using a closed, bounded tactic language.

Agents supply a proof body, never a file, import, theorem statement or command.
The allowlist excludes executable metaprogramming. Lean must close the fixed
goal and report only standard foundational axioms. Failure is not a counterexample.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .benchmarks import LEAN_DIR, LEAN_VERSION, load_benchmark

TIMEOUT_SECONDS = 30
MAX_OUTPUT_BYTES = 128 * 1024
MAX_PROOF_CHARS = 16000
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
# Extend only after reviewing the elaboration behavior of new identifiers.
ALLOWED_WORDS = frozenset("""
intro intros have let show from by exact apply refine simpa simp simp_all only
rw rfl constructor funext calc at all_goals assumption trivial change dsimp
grind first next case obtain rcases with h₁ h₂
K inst V B h rate q observed hh data x hx hxdata q₁ q₂
oneStepReturn recoverPotential RecoveryGoal FullObservationGoal
MathAILab.oneStepReturn MathAILab.recoverPotential MathAILab.RecoveryGoal
MathAILab.FullObservationGoal
Eq Eq.refl Eq.symm Eq.trans congrArg congrFun And.intro
True False True.intro False.elim fun
""".split())
WORDS = re.compile(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*", re.UNICODE)
SAFE_CHARACTERS = re.compile(r"[A-Za-z0-9_\s.,:(){}\[\]=+*/<>!?|;^\-₁₂→←↔≠≤≥∀∃λ⟨⟩·]*\Z")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _LeanReceipt(dict):
    """In-process authority; a JSON flag cannot award a verified-proof score."""

    def __init__(self, result: dict, proposal: dict):
        super().__init__(result)
        self._result_digest = _digest(result)
        self._proposal_digest = _digest(proposal)


def is_verified_receipt(proposal: dict, result: dict) -> bool:
    return (isinstance(result, _LeanReceipt)
            and result.get("status") == "verified"
            and result._result_digest == _digest(result)
            and result._proposal_digest == _digest(proposal))


def _compiler_environment() -> dict:
    # No API credentials, LEAN_PATH, or project-specific plugin settings.
    allowed = ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG")
    return {key: os.environ[key] for key in allowed if key in os.environ}


@lru_cache(maxsize=1)
def lean_status() -> dict:
    """Discover a local compiler; never install or download at runtime."""
    configured = os.environ.get("MATHAI_LEAN_BIN")
    installed = Path.home() / ".elan" / "toolchains" / f"leanprover--lean4---v{LEAN_VERSION}" / "bin" / "lean"
    executable = configured or (str(installed) if installed.is_file() else shutil.which("lean"))
    result = {"engine": "lean4", "required_version": LEAN_VERSION,
              "available": False, "executable": executable, "version": None}
    if not executable:
        return {**result, "reason": "Lean is not installed. Install the pinned toolchain or set MATHAI_LEAN_BIN."}
    try:
        checked = subprocess.run([executable, "--version"], cwd=LEAN_DIR,
                                 env=_compiler_environment(), capture_output=True,
                                 text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {**result, "reason": f"Lean could not start: {exc}"}
    version = checked.stdout.strip()
    if checked.returncode or not re.search(rf"\bversion {re.escape(LEAN_VERSION)}(?:,|\))", version):
        return {**result, "version": version, "reason": "Lean does not match the pinned toolchain."}
    return {**result, "available": True, "version": version, "reason": None}


def _proof_error(proof: object) -> str | None:
    if not isinstance(proof, str) or not proof.strip() or len(proof) > MAX_PROOF_CHARS:
        return "Supply a nonempty Lean tactic body of at most 16000 characters."
    if any(token in proof for token in ("--", "/-", "-/")) or not SAFE_CHARACTERS.fullmatch(proof):
        return "Only the documented tactic body is allowed; comments, strings and commands are rejected."
    unknown = set(WORDS.findall(proof)) - ALLOWED_WORDS
    if unknown:
        return "Unsupported Lean identifiers: " + ", ".join(sorted(unknown)) + ". Use the public proof-language contract."
    return None


def _source(contract: dict, proof: str) -> str:
    # Indent every line; agent text cannot supply a top-level declaration.
    body = "\n".join("  " + line for line in proof.strip().splitlines())
    return (contract["specification"] + "\nopen MathAILab\n"
            "set_option maxHeartbeats 200000\nset_option maxRecDepth 512\n"
            f"theorem MathAILab.candidate : {contract['target']} := by\n{body}\n\n"
            "#print axioms MathAILab.candidate\n")


def _compile(source: str, executable: str) -> dict:
    started = time.monotonic()
    # File-backed diagnostics avoid unbounded in-memory subprocess output.
    with tempfile.TemporaryDirectory(prefix="mathai-lean-") as temporary:
        directory = Path(temporary)
        (directory / "Candidate.lean").write_text(source, encoding="utf-8")
        with (directory / "diagnostics.txt").open("w+b") as output:
            process = subprocess.Popen(
                [executable, "--json", "-j1", "-M2048", "Candidate.lean"],
                cwd=directory, env=_compiler_environment(), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, shell=False,
            )
            timed_out = False
            try:
                process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=5)
            output.seek(0)
            raw = output.read(MAX_OUTPUT_BYTES + 1)
    return {"return_code": process.returncode, "timed_out": timed_out,
            "output_truncated": len(raw) > MAX_OUTPUT_BYTES,
            "output": raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            "elapsed_seconds": round(time.monotonic() - started, 4)}


def _axioms(output: str) -> list[str] | None:
    # Parse compiler JSON info, not arbitrary stdout or source text.
    audits = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(item, dict) or item.get("severity") != "information":
            continue
        message = item.get("data", "")
        if not isinstance(message, str):
            continue
        if message == "'MathAILab.candidate' does not depend on any axioms":
            audits.append([])
        match = re.fullmatch(r"'MathAILab\.candidate' depends on axioms: \[([^\]]*)\]", message, re.DOTALL)
        if match:
            audits.append([name.strip() for name in match[1].split(",") if name.strip()])
    return audits[0] if len(audits) == 1 else None


def verify(proposal: dict, benchmark: dict) -> dict:
    result = {
        "engine": "lean4", "status": "inconclusive", "target": "formal_proof",
        "scope": "Only the repository-owned Lean statement; prose and experimental modelling require review.",
        "independently_verified": False, "certificate_valid": None,
        "checks": [], "gaps": [], "summary": "No formal proof has been verified.",
    }
    try:
        trusted = load_benchmark(benchmark["id"])
    except (KeyError, TypeError, ValueError):
        result["gaps"] = ["This verifier does not support the requested benchmark."]
        return result
    if benchmark != trusted:
        result.update(status="rejected", gaps=["Benchmark packet differs from the repository-owned specification."])
        return result
    contract = trusted["verification_contract"]
    result["benchmark_sha256"] = _digest(trusted)
    result["target_id"] = contract.get("target_id")
    result["formal_target"] = contract.get("target")
    if contract.get("status") != "ready":
        result.update(status="formalization_pending", summary="This benchmark's exact theorem has not yet been formalized in Lean.")
        result["gaps"] = ["Complete and review the formal statement before submitting a proof. No substitute theorem is accepted."]
        return result
    if not isinstance(proposal, dict):
        result["gaps"] = ["Proposal must be an object."]
        return result
    if proposal.get("claim") != trusted["statement"] or proposal.get("assumptions") != trusted["assumptions"]:
        result.update(status="rejected", gaps=["Claim and assumptions must match the fixed benchmark exactly; copy them from the packet."])
        return result
    certificate = proposal.get("certificate")
    if certificate is None:
        result["gaps"] = ["No Lean certificate was submitted. Natural-language reasoning alone is not checked by Lean."]
        return result
    if (not isinstance(certificate, dict) or set(certificate) != {"kind", "target_id", "proof"}
            or certificate["kind"] != "lean4" or certificate["target_id"] != contract["target_id"]):
        result.update(status="rejected", gaps=["Supply a lean4 certificate for this exact target_id."])
        return result
    error = _proof_error(certificate["proof"])
    if error:
        result.update(status="rejected", gaps=[error])
        return result
    source = _source(contract, certificate["proof"])
    result.update(source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest())
    status = lean_status()
    result["toolchain"] = dict(status)
    if not status["available"]:
        result.update(status="unavailable", gaps=[status["reason"]], summary="Lean is unavailable; no proof check ran.")
        return result
    try:
        compilation = _compile(source, status["executable"])
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(status="error", gaps=[f"Lean could not run: {exc}"])
        return result
    result["compiler"] = compilation
    if compilation["timed_out"]:
        result.update(status="timeout", gaps=["Lean exceeded the 30-second verification limit."],
                      summary="Proof check timed out; no conclusion about the claim.")
        return result
    if compilation["return_code"] or compilation["output_truncated"]:
        result.update(status="failed", certificate_valid=False,
                      gaps=["Lean did not close the fixed goal. Inspect compiler diagnostics and repair the proof."],
                      summary="Formal proof attempt failed to check; this does not refute the mathematical statement.")
        return result
    axioms = _axioms(compilation["output"])
    result["axioms"] = axioms
    if axioms is None or set(axioms) - STANDARD_AXIOMS:
        result.update(status="rejected", certificate_valid=False,
                      gaps=["Missing axiom audit or nonstandard axioms (including sorryAx); proof rejected."])
        return result
    result.update(status="verified", independently_verified=True, certificate_valid=True,
                  checks=[{"kind": "lean_kernel", "passed": True}, {"kind": "axiom_audit", "passed": True, "axioms": axioms}],
                  summary="Lean verified the fixed formal target with only standard foundational axioms. "
                          "This does not solve hidden-vertex recovery or certify the prose/model interpretation.")
    return _LeanReceipt(result, proposal)
