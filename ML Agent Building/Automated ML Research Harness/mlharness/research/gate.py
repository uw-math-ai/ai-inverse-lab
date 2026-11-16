"""Independent implementation / safety gate (V2) — deterministic, outside the research loop.

Every method that reaches execution (research proposals, baselines, refinement children, pilots)
passes these checks; none of them involve an LLM or the research critic's opinion:

1. method validity      known kind, library method available, parameters clipped to their spaces
2. task compatibility   physics/hybrid need regression targets with one value per model parameter
3. forward authorisation  forward-model use requires ``constraints.allow_forward_model`` and a
                        simulator that provides a forward operator
4. code safety          agent code: custom code allowed by the spec + static AST checks
                        (``safety.check_code_v2``); dynamic sandbox is applied by the runner
5. data leakage         no reference to held-out splits anywhere in the method spec or code; the
                        runner structurally never passes test/shift arrays to development jobs
6. metric correctness   primary metric is defined for the task with a known direction; log targets
                        require strictly positive training targets
7. resource controls    a-priori cost within the per-experiment limit, forward evaluations within
                        ``constraints.max_forward_evals``, GPU requests noted
"""
from __future__ import annotations

from .. import artifacts as A
from .. import methods_v2 as M2
from .. import metrics as M
from ..safety import check_code_v2, references_test_data


class ImplementationGate:
    def __init__(self, spec, store, available: dict, feature_sets: list, n_features: int, problem: dict,
                 data_stats: dict, has_remote: bool = False):
        self.spec = spec
        self.store = store
        self.available = available
        self.feature_sets = feature_sets
        self.n_features = n_features
        self.problem = problem
        self.stats = data_stats
        self.has_remote = has_remote

    def check(self, raw_method: dict, *, label: str, parents=(), idea_id: str | None = None) -> dict:
        c = self.spec.constraints
        checks, reasons, notes = {}, [], []
        ms, nnotes, errors = M2.normalize(raw_method, self.available, self.feature_sets, self.n_features,
                                          int(self.spec.data["n_train"]))
        notes += nnotes
        checks["method_valid"] = not errors
        reasons += errors
        kind = ms.get("kind")
        if kind == "random_search":
            checks["method_valid"] = False
            reasons.append("random_search must be expanded into concrete library specs before the gate")
        # 2 task compatibility
        ok = True
        if kind in ("physics", "hybrid") or (kind == "custom_solver" and ms.get("uses_forward")):
            if self.spec.task != "regression":
                ok = False
                reasons.append("forward-model inversion requires a regression task")
            npar = self.problem.get("n_params")
            if npar and int(self.stats.get("target_dim", npar)) != int(npar):
                ok = False
                reasons.append(f"targets have {self.stats.get('target_dim')} columns but the forward model has "
                               f"{npar} parameters")
        checks["task_compatible"] = ok
        # 3 forward authorisation
        fwd = M2.uses_forward(ms)
        auth = True
        if fwd and not c.get("allow_forward_model", True):
            auth = False
            reasons.append("forward-model access is not authorised by the research spec (constraints.allow_forward_model)")
        elif fwd and not self.problem.get("simulator_has_forward"):
            auth = False
            reasons.append("the simulator does not provide a forward operator")
        checks["forward_authorised"] = auth
        # 4 code safety
        safe = True
        if kind in ("custom_ml", "custom_solver"):
            if not c.get("allow_custom_code", True):
                safe = False
                reasons.append("agent-written code is disabled by the spec constraints")
            elif not ms.get("code"):
                safe = False
                reasons.append("agent-written method has no code (code generation failed or no LLM backend)")
            else:
                probs = check_code_v2(ms.get("code") or "", "build_model" if kind == "custom_ml" else "build_method")
                if probs:
                    safe = False
                    reasons += [f"code: {p}" for p in probs]
        checks["code_safe"] = safe
        # 5 leakage
        leaks = references_test_data({k: v for k, v in ms.items() if k != "code"}) +             references_test_data({k: v for k, v in (raw_method or {}).items() if k != "code"})
        leaks = sorted(set(leaks))
        checks["no_leakage"] = not leaks
        if leaks:
            reasons.append(f"method spec references held-out data: {leaks}")
        # 6 metric correctness
        prim = self.spec.primary
        mok = prim in M.METRICS and M.METRICS[prim][2] == self.spec.task
        lib = ms.get("init") if kind == "hybrid" else ms
        if lib and lib.get("target_transform") == "log" and float(self.stats.get("target_min", 1.0)) <= 0:
            mok = False
            reasons.append("log target transform requires strictly positive targets")
        checks["metric_correct"] = mok
        # 7 resources
        n_val = int(self.spec.data["n_val"])
        est = M2.estimate_seconds(ms, n_val, int(self.spec.data["n_train"]), int(self.problem.get("n_params") or 1),
                                  self.problem.get("sec_per_forward")) if checks["method_valid"] else 0.0
        limit = float(self.spec.budget["experiment_timeout"])
        rok = True
        if est > 3 * limit:
            rok = False
            reasons.append(f"a-priori cost estimate {est:.0f}s exceeds 3x the per-experiment limit ({limit:.0f}s)")
        elif est > limit:
            notes.append(f"a-priori cost estimate {est:.0f}s exceeds the per-experiment limit; the screen will measure it")
        max_fwd = c.get("max_forward_evals")
        if fwd and max_fwd:
            need = M2.forward_evals_per_sample(ms, int(self.problem.get("n_params") or 1)) * n_val
            if need > float(max_fwd):
                rok = False
                reasons.append(f"needs ~{need:.0f} forward evaluations > constraints.max_forward_evals={max_fwd}")
        if M2.wants_gpu(ms) and not self.spec.hardware.get("gpu_devices") and not self.has_remote:
            notes.append("requests a GPU but none is available locally or remotely; will run on CPU")
        checks["resources_ok"] = rok
        approved = all(checks.values())
        verdict = {"approved": approved, "label": label, "method": ms, "checks": checks, "reasons": reasons,
                   "notes": notes, "forward_access": bool(fwd and auth), "est_seconds": round(est, 1),
                   "family": M2.family(ms), "method_label": M2.label(ms)}
        art = self.store.put(A.GATE, verdict, author="gate", stage="gate", parents=parents, idea_id=idea_id,
                             summary=(f"{label}: approved ({M2.label(ms)})" if approved else
                                      f"{label}: REJECTED — {'; '.join(reasons)[:150]}"))
        verdict["artifact_id"] = art.id
        return verdict
