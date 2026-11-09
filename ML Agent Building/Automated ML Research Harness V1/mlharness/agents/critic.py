"""Critic agent: keeps the research aligned with the spec and explains failures.

* review_proposal : zero-compute pre-screen (alignment with task/constraints, validity, duplicates,
                    code safety), optionally followed by an LLM review of the whole batch.
* review_result   : post-experiment critique (failure diagnosis, overfitting, worse-than-baseline,
                    leakage suspicion).
* final_audit     : test vs validation consistency and success criterion.
"""
from __future__ import annotations

import ast
import json
import math

from .. import artifacts as A
from .. import methods as ML
from .. import metrics as M
from ..features import valid_feature_set
from .base import Agent, log

BANNED_MODULES = {"subprocess", "socket", "requests", "urllib", "http", "shutil", "ctypes", "ftplib",
                  "smtplib", "webbrowser", "pickle", "multiprocessing", "asyncio"}
BANNED_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}   # bare calls only
DANGEROUS_BASES = {"os", "shutil", "signal", "sys"}            # any call like os.remove(...), sys.exit(...)
BANNED_ATTRS = {"system", "popen", "rmtree", "unlink", "fork", "execv", "execve", "write_text", "write_bytes"}

FAILURE_PATTERNS = [
    ("timeout", "Exceeded the per-experiment time limit; the method is too slow at this data size."),
    ("MemoryError", "Ran out of memory; reduce model size or use a method with lower memory complexity."),
    ("non-finite", "Produced NaN/inf predictions; usually divergence (learning rate too high) or bad scaling."),
    ("Input contains NaN", "NaN/inf inside the pipeline; a feature transform or the model diverged."),
    ("ModuleNotFoundError", "A required library is not installed in the experiment environment."),
    ("n_components", "Invalid number of components for the data dimension."),
    ("shape", "Shape mismatch between model output and targets (multi-output handling)."),
    ("build_model", "Agent-written code does not follow the build_model(params, task, seed) contract."),
    ("SyntaxError", "Agent-written code has a syntax error."),
]


def explain_failure(error: str | None) -> str:
    error = error or ""
    for pat, expl in FAILURE_PATTERNS:
        if pat.lower() in error.lower():
            return expl
    return "Unclassified failure: " + error[:200]


def check_code(code: str) -> list[str]:
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body):
        problems.append("code must define a top-level build_model(params, task, seed)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for nm in names:
                if nm.split(".")[0] in BANNED_MODULES:
                    problems.append(f"forbidden import {nm}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_BUILTINS:
                problems.append(f"forbidden call {f.id}()")
            elif isinstance(f, ast.Attribute) and (
                    f.attr in BANNED_ATTRS or (isinstance(f.value, ast.Name) and f.value.id in DANGEROUS_BASES)):
                base = f.value.id if isinstance(f.value, ast.Name) else "…"
                problems.append(f"forbidden call {base}.{f.attr}()")  # model.eval(), list.remove() stay allowed
    return problems


class CriticAgent(Agent):
    role = "critic"

    def __init__(self, *a, available: dict, feature_sets: list, n_features: int, **kw):
        super().__init__(*a, **kw)
        self.available = available
        self.feature_sets = feature_sets
        self.n_features = n_features

    # -- proposals -----------------------------------------------------------------------
    def review_proposal(self, idea: dict, existing_hashes: set, stage: str) -> bool:
        cfg = idea["config"]
        notes, reasons = [], []
        key = cfg.get("method")
        cons = self.spec.constraints
        if key == "custom":
            if not cons.get("allow_custom_code", True):
                reasons.append("custom code is disabled by the spec constraints")
            else:
                reasons += check_code(cfg.get("code") or "")
        elif key not in ML.LIBRARY:
            reasons.append(f"unknown method {key!r}")
        elif key not in self.available:
            m = ML.LIBRARY[key]
            if self.spec.task not in m.tasks:
                reasons.append(f"{key} does not support {self.spec.task} (misaligned with the objective)")
            else:
                reasons.append(f"{key} is excluded by the spec constraints or unavailable on this machine")
        else:
            cfg["params"], notes = ML.clip_params(self.available[key], cfg.get("params") or {})
        if not valid_feature_set(cfg.get("features", "raw"), self.feature_sets):
            notes.append(f"feature set {cfg.get('features')!r} invalid for this simulator; using 'raw'")
            cfg["features"] = "raw"
        if cfg.get("scaler") not in ML.SCALERS:
            notes.append(f"scaler {cfg.get('scaler')!r} invalid; using 'standard'")
            cfg["scaler"] = "standard"
        if cfg.get("pca", 0) and not 0 < int(cfg["pca"]) < min(self.n_features, self.spec.data["n_train"]):
            notes.append(f"pca={cfg['pca']} invalid; disabled")
            cfg["pca"] = 0
        if not reasons:
            h = ML.config_hash(cfg) if key != "custom" else None
            if h and h in existing_hashes:
                reasons.append("duplicate of an existing configuration")
            elif h:
                existing_hashes.add(h)
        approved = not reasons
        self.emit(A.CRITIQUE, {"type": "proposal_review", "approved": approved, "reasons": reasons,
                               "notes": notes, "config": cfg}, idea_id=idea["id"], parents=[idea["hyp_artifact"]],
                  stage=stage, summary=("approved" if approved else "rejected: " + "; ".join(reasons))[:160])
        return approved

    def llm_batch_review(self, ideas: list[dict], stage: str) -> dict[str, str]:
        """Ask the model to flag ideas misaligned with the objective. Returns {idea_id: reason}."""
        if not ideas or not self.uses_llm or not self.spec.search.get("llm_critic", True):
            return {}
        listing = "\n".join(f"- {i['id']}: {i['title']} | {i['hypothesis'][:200]} | "
                            f"{json.dumps({k: v for k, v in i['config'].items() if k != 'code'}, default=str)[:300]}"
                            for i in ideas)
        prompt = (f"{self.spec.brief()}\n\nProposed experiments:\n{listing}\n\n"
                  "As the critic, reject only ideas that are clearly misaligned with the objective, violate "
                  "a constraint, would leak test information, or are certain to fail. Respond as JSON: "
                  '{"reject": [{"id": idea_id, "reason": str}], "advice": str}')
        out = self.ask_json("You are the critic agent of an autonomous ML research team. Be concise and strict "
                            "about alignment with the research specification. Reply with JSON only.",
                            prompt, purpose="critic batch review", stage=stage)
        if not isinstance(out, dict):
            return {}
        rej = {r["id"]: r.get("reason", "") for r in out.get("reject", []) if isinstance(r, dict) and "id" in r}
        if out.get("advice"):
            self.emit(A.CRITIQUE, {"type": "batch_advice", "advice": out["advice"], "rejected": rej},
                      stage=stage, summary=f"critic advice: {str(out['advice'])[:140]}")
        return rej

    # -- results -------------------------------------------------------------------------
    def review_result(self, idea: dict, res: dict, baseline_value, result_art_id: str, stage: str,
                      reference_value=None) -> dict:
        """reference_value: best validation value seen so far at the same fidelity (for calibration)."""
        primary = self.spec.primary
        crit = {"type": "result_review", "status": res.get("status"), "flags": []}
        if res.get("status") != "ok":
            crit["explanation"] = explain_failure(res.get("error") or res.get("status"))
            crit["flags"].append("failed")
            self.emit(A.FAILURE, {"error": res.get("error"), "status": res.get("status"),
                                  "explanation": crit["explanation"], "method": idea["config"]["method"],
                                  "fraction": res.get("fraction")},
                      idea_id=idea["id"], parents=[result_art_id], stage=stage,
                      summary=f"{idea['config']['method']} failed: {crit['explanation'][:120]}")
        else:
            val = res["val"].get(primary)
            trn = (res.get("train") or {}).get(primary)
            vs, ts = M.to_score(val, primary), M.to_score(trn, primary)
            if math.isfinite(vs) and math.isfinite(ts) and abs(vs) > 1e-12:
                gap = (ts - vs) / abs(vs)
                crit["overfit_gap"] = round(gap, 4)
                # tree ensembles / distance-weighted kNN interpolate the training set by design
                interpolating = ML.LIBRARY.get(idea["config"]["method"]) and (
                    ML.LIBRARY[idea["config"]["method"]].family == "ensemble"
                    or (idea["config"]["method"] == "knn" and idea["config"]["params"].get("weights") == "distance"))
                # a train/val gap only matters if it also costs validation performance: models within 25 %
                # of the best result at this fidelity are not flagged (gap is recorded either way)
                competitive = reference_value is not None and M.relative_gain(
                    vs, M.to_score(reference_value, primary)) > -0.25
                if gap > 0.5 and not interpolating and not competitive:
                    crit["flags"].append("overfitting")
            if baseline_value is not None and vs <= M.to_score(baseline_value, primary):
                crit["flags"].append("no_better_than_baseline")
            if (primary == "rel_l2" and val is not None and val < 1e-6) or (primary == "r2" and val and val > 0.99999):
                crit["flags"].append("suspiciously_perfect")
            if res.get("warnings"):
                crit["warnings"] = res["warnings"][:3]
            parts = []
            if "overfitting" in crit["flags"]:
                parts.append(f"train {primary}={trn:.4g} vs val {val:.4g}: strong overfitting, more regularisation "
                             f"or data needed")
            if "no_better_than_baseline" in crit["flags"]:
                parts.append("no better than predicting the mean")
            if "suspiciously_perfect" in crit["flags"]:
                parts.append("near-perfect score: check for leakage")
            crit["explanation"] = "; ".join(parts) or "ok"
        if crit["flags"]:
            self.emit(A.CRITIQUE, crit, idea_id=idea["id"], parents=[result_art_id], stage=stage,
                      summary=f"{idea['id']}: {crit.get('explanation', '')[:140]}")
        return crit

    def final_audit(self, final_res: dict, val_value, members: list, stage="final") -> dict:
        primary = self.spec.primary
        test = (final_res.get("test") or {}).get(primary)
        audit = {"type": "final_audit", "test": final_res.get("test"), "val_estimate": val_value,
                 "success_metric": self.spec.success["metric"], "threshold": self.spec.success["threshold"]}
        sm = (final_res.get("test") or {}).get(self.spec.success["metric"])
        audit["success"] = bool(self.spec.meets_success(sm)) if self.spec.success["threshold"] is not None else None
        issues = []
        if test is None:
            issues.append("final evaluation failed")
        elif val_value is not None:
            vs, ts = M.to_score(val_value, primary), M.to_score(test, primary)
            rel = (vs - ts) / max(abs(vs), 1e-12)
            audit["test_vs_val_degradation"] = round(rel, 4)
            if rel > 0.2:
                issues.append(f"test {primary} is {rel:.0%} worse than validation: selection may have "
                              f"overfit the validation split")
        audit["issues"] = issues
        audit["aligned_with_objective"] = not issues
        self.emit(A.CRITIQUE, audit, stage=stage, parents=members,
                  summary=f"final audit: test {primary}={test}, success={audit['success']}, issues={len(issues)}")
        return audit
