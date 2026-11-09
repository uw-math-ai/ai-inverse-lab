"""Post-research code generation and repair for agent-written methods (V2).

Turns a research proposal whose method is ``custom_ml`` / ``custom_solver`` but lacks code into code
that follows the contract, and repairs code that crashes in the screen (one LLM call per attempt).
Every generated or repaired file goes back through the static safety checks before it can run, and
always runs inside the sandboxed runner process. Offline (no LLM) the engineer cannot write code:
such proposals are rejected at the gate with that reason.
"""
from __future__ import annotations

from ..agents.base import Agent
from ..safety import check_code_v2

ENGINEER_SYSTEM = ("You are the research engineer of an autonomous ML research team. You turn a research proposal "
                   "into a short, robust, self-contained Python implementation that follows the given contract "
                   "exactly. Reply with a single JSON object only.")

CONTRACT = {
    "custom_ml": ("define build_model(params: dict, task: str, seed: int) returning a scikit-learn compatible "
                  "estimator (fit(X, y) / predict(X)); features/scaler/PCA are applied before it"),
    "custom_solver": ("define build_method(params: dict, context) returning an object with fit(X, y) and predict(X) "
                      "-> numpy array of shape (n, n_params). context offers forward(theta (n,p)) -> noise-free "
                      "observations (counts against a budget; batch your calls), jacobian(theta, h, transform), lower, "
                      "upper, noise_std, param_names, n_params, facts, clip(theta), prior_center()"),
}
RULES = ("Allowed imports: numpy, scipy, sklearn, torch, math. No file, network or process access; no getattr/"
         "setattr/eval/exec/open; no dunder attributes except __init__/__call__; never reference test data.")


class CodeEngineer(Agent):
    role = "engineer"

    def generate(self, proposal: dict, ms: dict, problem_text: str) -> tuple[str | None, str]:
        kind = ms.get("kind")
        if kind not in CONTRACT:
            return None, "not a code method"
        if not self.uses_llm:
            return None, "agent-written code requires an LLM backend"
        prompt = (f"{self.spec.brief()}\nProblem context: {problem_text}\n\nProposal {proposal.get('id')}: "
                  f"{proposal.get('title')}\nHypothesis: {proposal.get('hypothesis')}\nMethod request: "
                  f"{ms}\n\nContract: {CONTRACT[kind]}. {RULES}\n"
                  'Respond as JSON: {"code": "...full python file...", "explanation": str}')
        out = self.ask_json(ENGINEER_SYSTEM, prompt, purpose=f"codegen {proposal.get('id')}", stage="engineering")
        code = (out or {}).get("code") if isinstance(out, dict) else None
        if not code:
            return None, "the engineer returned no code"
        probs = check_code_v2(code, "build_model" if kind == "custom_ml" else "build_method")
        if probs:
            return None, "generated code failed static checks: " + "; ".join(probs)[:300]
        return code, str((out or {}).get("explanation") or "")[:300]

    def repair(self, ms: dict, error: str, traceback: str, problem_text: str) -> str | None:
        kind = ms.get("kind")
        if kind not in CONTRACT or not self.uses_llm or not ms.get("code"):
            return None
        prompt = (f"{self.spec.brief()}\nProblem context: {problem_text}\n\nCurrent code:\n```python\n{ms['code']}\n```"
                  f"\nIt failed with:\n{error}\n{(traceback or '')[-2500:]}\n\nContract: {CONTRACT[kind]}. {RULES}\n"
                  'Keep the idea, fix the bug. Respond as JSON: {"code": "...full fixed file...", "explanation": str}')
        out = self.ask_json(ENGINEER_SYSTEM, prompt, purpose="repair agent code", stage="screen")
        code = (out or {}).get("code") if isinstance(out, dict) else None
        if not code or check_code_v2(code, "build_model" if kind == "custom_ml" else "build_method"):
            return None
        return code
