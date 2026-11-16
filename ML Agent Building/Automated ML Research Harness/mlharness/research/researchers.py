"""The four parallel research subagents (V2).

Each researcher has an intellectual emphasis but may propose ANY executable method (library ML,
physics inversion, hybrid, agent-written ML or solver code, random-search control). Researchers:

1. read the research memory (prior successes, failures, critiques, citations) before proposing;
2. choose TheoremSearch queries (LLM) or use problem-derived query templates (offline) and receive
   the retrieved statements with ref ids and the harness's applicability screen;
3. propose hypotheses with assumptions, evidence, a falsifiable (machine-checkable) prediction,
   feasibility and "what we learn if it fails";
4. after each critique, respond to EVERY piece of feedback on their proposals: revise, defend,
   abandon (with reason) or branch, and may add one new proposal per round.

With an LLM backend a round costs one call (two in round 1: queries, then proposals); without one
(or when the LLM budget is exhausted / a call fails) the researcher falls back to deterministic,
problem-aware heuristic policies and records ``origin = heuristic(-fallback)``.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from .. import methods as ML
from ..agents.base import Agent, log

ROLES = {
    "mathematical": {
        "title": "mathematical researcher",
        "emphasis": "governing equations, forward/inverse operators, identifiability, conditioning, applicable "
                    "theorems, constraints, and analytical or numerical inversion approaches",
        "queries": ["identifiability of {coef} in the {pde} from {obs}",
                    "Lipschitz stability for an inverse coefficient problem of a parabolic equation with point measurements",
                    "Levenberg-Marquardt method convergence for nonlinear ill-posed inverse problems"]},
    "ml": {
        "title": "machine-learning researcher",
        "emphasis": "learned inverse maps, representations, architectures, estimators, training regimes and "
                    "ML-plus-physics hybrids",
        "queries": ["neural network approximation of the solution operator of a parameter identification inverse problem",
                    "learning rates of kernel ridge regression for smooth regression functions",
                    "operator learning for inverse problems in partial differential equations"]},
    "experimental": {
        "title": "experimental researcher",
        "emphasis": "simulation fidelity, sensor/measurement design, noise, data distributions, testable competing "
                    "explanations, controls and evaluation design",
        "queries": ["Cramer-Rao lower bound for parameter estimation in partial differential equations",
                    "optimal sensor placement for parameter identification in diffusion equations",
                    "robustness of regression estimators to measurement noise misspecification"]},
    "exploratory": {
        "title": "exploratory researcher",
        "emphasis": "challenging the framing assumptions, nonstandard formulations, and combining ideas across "
                    "research traditions",
        "queries": ["posterior mean estimator for Bayesian inverse problems with a uniform prior",
                    "learned initialization for Gauss-Newton iterations in nonlinear inverse problems",
                    "minimum mean square error estimation versus maximum likelihood for nonlinear models"]},
}

SYSTEM = ("You are the {title} in a four-person autonomous research team investigating an inverse problem. Your "
          "intellectual emphasis: {emphasis}. You may nevertheless propose ANY mathematical, ML, numerical, physics-based "
          "or hybrid technique. You are a scientist, not a hyper-parameter sampler: state hypotheses that experiments "
          "can falsify, make assumptions explicit, cite retrieved statements only by their ref id and say whether "
          "their assumptions actually hold here. Retrieval of a related theorem is NOT a proof of your claim. "
          "Reply with a single JSON object only.")

PROPOSAL_SCHEMA = (
    '{"title": str, "hypothesis": str, "motivation": "reasoning that motivates it", '
    '"assumptions": [str], "evidence": [{"claim": str, "stance": "supports|contradicts", '
    '"source": "ref id | memory id | diagnostics | reasoning"}], '
    '"citations": [{"ref_id": "ts:...", "claimed_applicability": "yes|partial|no|unknown", '
    '"note": "which of its assumptions hold or fail for THIS problem"}], '
    '"method": <executable method spec>, "falsifiable_test": str, '
    '"prediction": {"metric": str, "op": "<", "value": float} or {"relative_to": '
    '"best_data_driven|best_physics|best_hybrid|best_other|baseline_mean", "metric": str, "improvement": float, '
    '"expect": "better|worse"}, "feasibility": {"cost": "low|medium|high", "notes": str}, '
    '"if_fails": "what we learn if it fails", "status": "conjectured|literature-supported", '
    '"next_queries": [str]}')


def method_menu(available: dict, feature_sets: list, forward_available: bool, allow_custom: bool,
                n_params: int | None) -> str:
    fwd = "AVAILABLE" if forward_available else "NOT AVAILABLE (physics / hybrid / forward-using solvers will be rejected)"
    lines = [
        "EXECUTABLE METHOD SPECS (the 'method' field; the harness executes these after the research phase):",
        '1. {"kind":"library","method":KEY,"params":{...},"features":FEATURES,"scaler":"standard|robust|none",'
        '"pca":int,"target_transform":"none|log","noise_aug":0-4,"noise_scale":float}',
        "   library keys:\n" + "\n".join("     " + ln for ln in ML.catalog_text(available).splitlines()),
        f"   feature sets: {feature_sets}",
        '2. {"kind":"physics","params":{"transform":"log|none","multistart":1-6,"iters":5-80,"tikhonov":0-10,'
        '"estimator":"mode|posterior_mean","posterior_samples":32-512}}',
        "   per-sample bounded Levenberg-Marquardt through the forward model F(theta): MLE (tikhonov=0), MAP toward "
        "the prior centre (tikhonov>0, in transformed space), or posterior_mean = MMSE under the box prior by "
        f"importance sampling on the Laplace approximation; reports Gauss-Newton uncertainty. Forward model: {fwd}.",
        '3. {"kind":"hybrid","init":<library spec>,"params":{"transform","iters":1-40,"tikhonov","extra_starts":0-3,'
        '"estimator","posterior_samples"}}  learned initializer, then LM refinement through the forward model.',
        '6. {"kind":"random_search","n":1-8}  control arm: uniform random library configurations.',
    ]
    if allow_custom:
        lines += [
            '4. {"kind":"custom_ml","code":"<python>","params":{},"features","scaler","pca"}  code defines '
            "build_model(params, task, seed) -> scikit-learn style estimator (fit/predict).",
            '5. {"kind":"custom_solver","code":"<python>","params":{},"uses_forward":bool}  code defines '
            f"build_method(params, context) -> object with fit(X, y) and predict(X) -> array (n, {n_params or 'p'}). "
            "context: forward(theta (n,p)) -> noise-free observations (budgeted), jacobian(theta), lower, upper, "
            "noise_std, param_names, n_params, facts, clip(theta), prior_center().",
            "   Agent code: imports only numpy, scipy, sklearn, torch, math; no file/network/process access, no "
            "dunder attributes, no getattr/eval/exec; never reference test data. Keep code short and robust.",
        ]
    return "\n".join(lines)


class ResearcherAgent(Agent):
    role = "researcher"

    def __init__(self, *a, persona: str, literature, memory, problem: dict, available: dict, feature_sets: list,
                 n_features: int, seed: int, context: dict, run_id: str, rcfg: dict, **kw):
        super().__init__(*a, **kw)
        self.persona = persona
        self.r = ROLES[persona]
        self.literature = literature
        self.memory = memory
        self.problem = problem           # ProblemContext.public_summary() + diagnostics + describe text
        self.available = available
        self.feature_sets = feature_sets
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        self.mem_context = context
        self.run_id = run_id
        self.rcfg = rcfg
        self.memory_refs: list[str] = []
        self.literature_seen: list = []

    # -- shared helpers ---------------------------------------------------------------------
    @property
    def fwd(self) -> bool:
        return bool(self.problem.get("forward_available"))

    def _system(self) -> str:
        return SYSTEM.format(title=self.r["title"], emphasis=self.r["emphasis"])

    def _menu(self) -> str:
        return method_menu(self.available, self.feature_sets, self.fwd,
                           bool(self.spec.constraints.get("allow_custom_code", True)), self.problem.get("n_params"))

    def _memory(self) -> str:
        q = f"{self.spec.problem} {self.r['emphasis']}"
        text = self.memory.digest(q, context=self.mem_context, k=int(self.rcfg.get("memory_top_k", 8)),
                                  exclude_run=self.run_id, agent=self.name)
        hits = self.memory.retrievals[-1]["ids"] if self.memory.retrievals else []
        self.memory_refs = list(dict.fromkeys(self.memory_refs + hits))
        return text

    def _problem_text(self) -> str:
        d = self.problem.get("diagnostics") or {}
        diag = ""
        if d.get("available"):
            diag = (f"\nNumerical identifiability diagnostics (harness-computed at {d['n_samples']} prior draws): "
                    f"Fisher condition number median {d['fisher_condition_median']:.3g}; Cramér–Rao relative std per "
                    f"parameter {json.dumps({k: round(v, 4) for k, v in d['crb_rel_median'].items()})}; CRB rel_l2 "
                    f"floor estimate {d['crb_rel_l2_estimate']:.4g}; least determined {d['least_determined']}.")
        base = self.problem.get("baseline_value")
        return (f"{self.spec.brief()}\nProblem context: {self.problem.get('describe')}{diag}\n"
                f"Mean-predictor baseline val {self.spec.primary}: {base if base is None else round(base, 5)}")

    def _queries_text(self, results) -> str:
        return self.literature.digest(results) if self.literature is not None else "(literature tool unavailable)"

    def _search(self, queries, round_idx, purpose):
        out = []
        if self.literature is None:
            return out
        for q in queries[: int(self.rcfg.get("queries_per_round", 2))]:
            if q:
                out.append(self.literature.search(q, self.name, round_idx, purpose))
        self.literature_seen += out
        return out

    # -- round 1 ----------------------------------------------------------------------------
    def propose(self, rc: dict) -> list[dict]:
        """Returns [{'action': 'propose', 'raw': proposal-dict, 'origin': ...}]"""
        mem = self._memory()
        n = int(self.rcfg.get("proposals_per_researcher", 2))
        if self.uses_llm:
            out = self._llm_propose(rc, mem, n)
            if out:
                return out
            origin = "heuristic-fallback"
        else:
            origin = "heuristic"
        results = self._search(self._template_queries(), rc["round"], "background")
        return [{"action": "propose", "raw": r, "origin": origin} for r in self._heuristic_proposals(results)[:n]]

    def _llm_propose(self, rc, mem, n):
        qp = (f"{self._problem_text()}\n\nPrior research memory (other runs; test results redacted):\n{mem}\n\n"
              f"Before proposing, choose up to {self.rcfg.get('queries_per_round', 2)} queries for TheoremSearch, a "
              "semantic search engine over theorem/lemma statements from arXiv and other sources. Ask for existing "
              "results that would support, constrain or refute approaches from your perspective. "
              'Respond as JSON: {"queries": [{"query": str, "why": str}]}')
        q = self.ask_json(self._system(), qp, purpose=f"literature queries ({self.persona})", stage="research")
        queries = [x.get("query") if isinstance(x, dict) else str(x) for x in (q or {}).get("queries", [])] \
            if isinstance(q, dict) else []
        if q is None:
            return None
        results = self._search(queries or self._template_queries(), rc["round"], "background")
        prompt = (f"{self._problem_text()}\n\n{self._menu()}\n\nPrior research memory (other runs; test results "
                  f"redacted):\n{mem}\n\nRetrieved literature (cite by ref id; the harness screen is a keyword check, "
                  f"not a proof check):\n{self._queries_text(results)}\n\nResearch budget: {rc.get('budget_text')}\n\n"
                  f"Propose {n} meaningfully different, falsifiable hypotheses from your perspective "
                  f"({self.r['emphasis']}). Each must name an executable method. Respond as JSON: "
                  f'{{"proposals": [{PROPOSAL_SCHEMA}]}}')
        out = self.ask_json(self._system(), prompt, purpose=f"propose ({self.persona})", stage="research")
        props = out.get("proposals") if isinstance(out, dict) else None
        if not props:
            return None
        return [{"action": "propose", "raw": p, "origin": "llm"} for p in props[:n] if isinstance(p, dict)]

    # -- revision rounds ----------------------------------------------------------------------
    def revise(self, rc: dict, mine: list[dict]) -> list[dict]:
        """Respond to the critic. Returns [{'action', 'target' (proposal id or None), 'raw', 'origin',
        'response_to_critic', 'abandon_reason'}]."""
        fb = rc.get("feedback", {})
        queries = []
        for p in mine:
            for er in (fb.get(p["id"], {}).get("evidence_requests") or []):
                queries.append(er.get("query") if isinstance(er, dict) else str(er))
            queries += p.get("next_queries") or []
        results = self._search(list(dict.fromkeys(q for q in queries if q)), rc["round"], "evidence request")
        if self.uses_llm:
            out = self._llm_revise(rc, mine, results)
            if out is not None:
                return out
            origin = "heuristic-fallback"
        else:
            origin = "heuristic"
        return self._heuristic_revise(rc, mine, results, origin)

    def _llm_revise(self, rc, mine, results):
        fb = rc.get("feedback", {})
        own = []
        from .proposals import compact
        for p in mine:
            f = {k: (v[:6] if isinstance(v, list) else v) for k, v in fb.get(p["id"], {}).items()
                 if k != "llm_score" and v not in (None, [], "", False)}
            own.append({"proposal": compact(p, long=True), "critic": f, "pilot": rc.get("pilots", {}).get(p["id"])})
        others = "\n".join(f"  {o['id']} [{o['author']}] {o['family']} {o['method_label']}: {o['title']}"
                           for o in rc.get("others", []) if o["author"] != self.name) or "  none"
        prompt = (f"{self._problem_text()}\n\n{self._menu()}\n\nYour current proposals with the shared critic's review "
                  f"(and pilot-experiment results where run):\n{json.dumps(own, default=str)}\n\n"
                  f"Critic's portfolio-level notes: {json.dumps(rc.get('portfolio_feedback', {}), default=str)[:2500]}\n\n"
                  f"Other researchers' current proposals (avoid duplicating them; you may build on them):\n{others}\n\n"
                  f"Newly retrieved literature:\n{self._queries_text(results)}\n\n"
                  "Respond to EVERY critic point for EVERY proposal. For each proposal choose an action: 'revise' "
                  "(return the full updated proposal), 'defend' (keep it; explain why the criticism does not apply), "
                  "'abandon' (give the reason; it is recorded as a negative result) or 'branch' (keep the original AND "
                  "return a new variant). Experimental evidence (pilots, memory) outweighs argument. You may add at "
                  "most one new proposal. Respond as JSON: "
                  '{"responses": [{"proposal_id": str, "action": "revise|defend|abandon|branch", '
                  '"response_to_critic": str, "abandon_reason": str, "proposal": ' + PROPOSAL_SCHEMA + '}], '
                  '"new_proposals": [' + PROPOSAL_SCHEMA + ']}')
        out = self.ask_json(self._system(), prompt, purpose=f"revise ({self.persona})", stage="research")
        if not isinstance(out, dict) or not isinstance(out.get("responses"), list):
            return None
        ids = {p["id"] for p in mine}
        res, answered = [], set()
        for r in out["responses"]:
            if not isinstance(r, dict) or r.get("proposal_id") not in ids:
                continue
            act = str(r.get("action", "defend")).lower()
            act = act if act in ("revise", "defend", "abandon", "branch") else "defend"
            answered.add(r["proposal_id"])
            res.append({"action": act, "target": r["proposal_id"], "raw": r.get("proposal") or {}, "origin": "llm",
                        "response_to_critic": r.get("response_to_critic", ""),
                        "abandon_reason": r.get("abandon_reason", "")})
        for p in mine:      # every proposal gets a response, even if the model skipped one
            if p["id"] not in answered:
                res.append({"action": "defend", "target": p["id"], "raw": {}, "origin": "llm",
                            "response_to_critic": "(no explicit response returned; kept unchanged)"})
        for np_ in (out.get("new_proposals") or [])[:1]:
            if isinstance(np_, dict):
                res.append({"action": "propose", "target": None, "raw": np_, "origin": "llm"})
        return res

    # -- heuristic policies -------------------------------------------------------------------
    def _template_queries(self) -> list[str]:
        f = self.problem.get("facts") or {}
        fill = {"coef": f.get("coefficient_regularity", "unknown coefficient").split(",")[0],
                "pde": f.get("pde", self.spec.problem[:80]).split(",")[0],
                "obs": f.get("observations", "noisy measurements").split(",")[0]}
        out = []
        for q in self.r["queries"]:
            try:
                out.append(q.format(**fill))
            except (KeyError, IndexError):
                out.append(q)
        return out

    def _cite(self, results, k=2):
        cites, evidence, supported = [], [], False
        for res in results or []:
            for ref in res.refs:
                entry = self.literature.get(ref.ref_id) if self.literature else None
                a = (entry or {}).get("assessment") or {}
                relevant = any(m.startswith("same equation class") for m in a.get("matched") or []) or                     any(n.startswith("method-level") for n in a.get("notes") or [])
                if a.get("applies") in ("plausible", "partial") and relevant and len(cites) < k:
                    note = ("screen: " + a["applies"] + ("; conflicts: " + "; ".join(a["conflicts"]) if a.get("conflicts") else "")
                            + ("; matched: " + ", ".join(a["matched"]) if a.get("matched") else ""))
                    cites.append({"ref_id": ref.ref_id, "claimed_applicability": "partial", "note": note})
                    evidence.append({"claim": f"{ref.name or 'statement'} in '{(ref.paper_title or '')[:80]}' — "
                                              f"{(ref.slogan or '')[:220]}",
                                     "stance": "supports", "source": ref.ref_id})
                    supported = supported or a["applies"] == "plausible"
        return cites, evidence, supported

    def _diag_evidence(self):
        d = self.problem.get("diagnostics") or {}
        if not d.get("available"):
            return [], None
        crb = d["crb_rel_l2_estimate"]
        ev = [{"claim": f"Fisher information has full rank at sampled parameters (median condition number "
                        f"{d['fisher_condition_median']:.3g}); Cramér–Rao rel_l2 floor ~{crb:.4g}; least determined "
                        f"parameters {d['least_determined']}", "stance": "supports", "source": "diagnostics"}]
        return ev, crb

    def _ml_key(self):
        return "torch_mlp" if "torch_mlp" in self.available else ("mlp_sklearn" if "mlp_sklearn" in self.available else
                                                                  next(k for k in self.available if k != "mean_baseline"))

    def _feat(self, pref):
        for f in pref:
            if f in self.feature_sets:
                return f
        return "raw"

    def _lib(self, key, feats, **kw):
        m = self.available.get(key)
        params = dict(m.defaults) if m else {}
        params.update(kw.pop("params", {}))
        return {"kind": "library", "method": key, "params": params, "features": feats, "scaler": "standard",
                "pca": kw.pop("pca", 0), "target_transform": kw.pop("target_transform", "none"),
                "noise_aug": kw.pop("noise_aug", 0), "noise_scale": kw.pop("noise_scale", 1.0)}

    def _heuristic_proposals(self, results) -> list[dict]:
        cites, lit_ev, lit_ok = self._cite(results)
        diag_ev, crb = self._diag_evidence()
        noise = self.problem.get("noise_model") or "unspecified"
        lo, hi = self.problem.get("lower"), self.problem.get("upper")
        least = (self.problem.get("diagnostics") or {}).get("least_determined") or []
        status = "literature-supported" if lit_ok else "conjectured"
        P, fwd = [], self.fwd
        common = {"citations": cites, "status": status}
        if self.persona == "mathematical":
            if fwd:
                P.append({**common, "title": "Maximum-likelihood inversion through the forward operator",
                          "hypothesis": "Per-sample MLE through the exact forward model is (near) efficient: it reaches "
                                        "the Cramér–Rao floor and beats every learned inverse at this training-set size.",
                          "motivation": "The data-generating operator is available and the noise is Gaussian, so the "
                                        "least-squares fit IS the likelihood; learned inverses add approximation error.",
                          "assumptions": [f"noise model: {noise}", "the forward operator is the data-generating discretization",
                                          "local identifiability: full-rank Jacobian near the truth",
                                          f"truth lies inside the prior box {lo}..{hi}", "LM from the prior centre reaches the global minimum"],
                          "evidence": diag_ev + lit_ev, "method": {"kind": "physics", "params": {"transform": "log", "multistart": 1, "iters": 30}},
                          "falsifiable_test": "Validation rel_l2 at least 10% below the best data-driven method and within 15% of the CRB floor.",
                          "prediction": {"relative_to": "best_data_driven", "improvement": 0.10},
                          "feasibility": {"cost": "low", "notes": "~100 forward evaluations per sample; batched"},
                          "if_fails": "model mismatch (chi2/dof >> 1), multimodality (single start insufficient) or "
                                      "non-identifiability; check chi2 and multistart"})
                P.append({**common, "title": "MAP estimate regularised toward the prior centre",
                          "hypothesis": f"Weakly determined parameters {least} dominate the error; a Gaussian (Tikhonov) "
                                        "prior toward the box centre trades a little bias for less variance and lowers rel_l2.",
                          "motivation": "Bias–variance trade-off for poorly conditioned directions of the Fisher matrix.",
                          "assumptions": ["prior is centred in the box", f"noise model: {noise}", "Laplace approximation adequate"],
                          "evidence": diag_ev, "method": {"kind": "physics", "params": {"transform": "log", "multistart": 1,
                                                                                         "iters": 30, "tikhonov": 1.0}},
                          "falsifiable_test": "MAP validation rel_l2 lower than the unregularised MLE.",
                          "prediction": {"relative_to": "best_physics", "improvement": 0.0},
                          "feasibility": {"cost": "low", "notes": "same cost as MLE"},
                          "if_fails": "the prior box already constrains the estimate; regularisation only adds bias"})
            else:
                P.append({**common, "title": "Log-linearisation of diffusive decay",
                          "hypothesis": "Diffusive decay is approximately exponential in time, so log temperatures are "
                                        "near-linear in the conductivities; a regularised linear model in log space is a strong inverse.",
                          "motivation": "Separation-of-variables solution of the heat equation: modes decay as exp(-lambda(k) t).",
                          "assumptions": ["observations positive enough for log features", "smooth inverse map"],
                          "evidence": lit_ev, "method": self._lib("ridge", self._feat(["log", "raw"]),
                                                                  params={"alpha": 1e-3}, target_transform="log"),
                          "falsifiable_test": "Within 20% of the best nonlinear learner on validation.",
                          "prediction": {"relative_to": "best_other", "improvement": -0.2},
                          "feasibility": {"cost": "low", "notes": "linear model"},
                          "if_fails": "the inverse map is strongly nonlinear; need a nonlinear learner"})
                P.append({**common, "title": "Kernel ridge regression on log features",
                          "hypothesis": "The inverse map is smooth, so RBF kernel ridge on log features is data-efficient.",
                          "motivation": "Smooth inverse map; kernel methods attain minimax rates for smooth functions.",
                          "assumptions": ["smooth inverse map", "training set covers the prior box"],
                          "evidence": lit_ev, "method": self._lib("kernel_ridge", self._feat(["log", "sqrt", "raw"]),
                                                                  params={"alpha": 1e-3, "gamma_mult": 0.5}, pca=32),
                          "falsifiable_test": "Beats the mean baseline by >90%.",
                          "prediction": {"relative_to": "baseline_mean", "improvement": 0.9},
                          "feasibility": {"cost": "medium", "notes": "O(n^2) memory"},
                          "if_fails": "bandwidth mis-specified or PCA discards information"})
        elif self.persona == "ml":
            key = self._ml_key()
            P.append({**common, "title": f"{key} on log features with log-conductivity targets",
                      "hypothesis": "A neural inverse on log features regressing log k captures the multiplicative "
                                    "structure of diffusive time scales and approaches the noise floor.",
                      "motivation": "Time scales scale as 1/k; errors are multiplicative, so log targets equalise scales.",
                      "assumptions": ["3000 training samples suffice for an 80-dimensional smooth map", "positive targets"],
                      "evidence": lit_ev, "method": self._lib(key, self._feat(["log", "raw"]), target_transform="log"),
                      "falsifiable_test": (f"Validation rel_l2 within 1.3x of the CRB floor ({crb:.4g})." if crb else
                                           "Beats the mean baseline by >90%."),
                      "prediction": ({"metric": self.spec.primary, "op": "<", "value": round(1.3 * crb, 5)} if crb else
                                     {"relative_to": "baseline_mean", "improvement": 0.9}),
                      "feasibility": {"cost": "medium", "notes": "minutes on CPU"},
                      "if_fails": "the learner is sample- or capacity-limited; compare to the physics floor"})
            P.append({**common, "title": "Kernel ridge on sqrt features with PCA denoising",
                      "hypothesis": "Diffusion makes the 80 readings highly correlated; PCA removes noise directions "
                                    "and an RBF kernel learns the smooth inverse map.",
                      "motivation": "Smoothing forward operators have rapidly decaying singular values.",
                      "assumptions": ["low effective dimension of the observations", "smooth inverse map"],
                      "evidence": lit_ev, "method": self._lib("kernel_ridge", self._feat(["sqrt", "log", "raw"]),
                                                              params={"alpha": 1e-3, "gamma_mult": 0.5}, pca=32),
                      "falsifiable_test": "Within 10% of the best data-driven method.",
                      "prediction": {"relative_to": "best_data_driven", "improvement": -0.10},
                      "feasibility": {"cost": "medium", "notes": "O(n^2)"},
                      "if_fails": "PCA truncation loses parameter information"})
        elif self.persona == "experimental":
            key = self._ml_key()
            P.append({**common, "title": "Matched-noise training augmentation",
                      "hypothesis": "Training with re-noised replicas at the sensor noise level regularises the learned "
                                    "inverse and makes it more robust to measurement noise without hurting accuracy.",
                      "motivation": "Augmentation matched to the measurement model acts as Tikhonov regularisation.",
                      "assumptions": [f"noise model: {noise}", "noise level known"],
                      "evidence": lit_ev, "method": self._lib(key, self._feat(["log", "raw"]), noise_aug=1, target_transform="log"),
                      "falsifiable_test": "Validation error no more than 5% worse than the best data-driven method; "
                                          "smaller degradation on the high-noise shift set.",
                      "prediction": {"relative_to": "best_data_driven", "improvement": -0.05},
                      "feasibility": {"cost": "medium", "notes": "2x training data"},
                      "if_fails": "the clean-noise regime does not benefit from augmentation"})
            P.append({**common, "title": "Random-search control arm",
                      "hypothesis": "If reasoning adds value, reasoned proposals beat uniformly random configurations "
                                    "evaluated under the same budget.",
                      "motivation": "A control that separates the value of the research phase from the search pipeline.",
                      "assumptions": ["same execution pipeline and budget for all arms"],
                      "evidence": [], "method": {"kind": "random_search", "n": 2},
                      "falsifiable_test": "Random configurations are worse than the best reasoned proposal.",
                      "prediction": {"relative_to": "best_other", "improvement": 0.0, "expect": "worse"},
                      "feasibility": {"cost": "medium", "notes": "two random library configs"},
                      "if_fails": "the search space is forgiving; reasoning did not add value at proposal stage"})
        else:   # exploratory
            if fwd:
                P.append({**common, "title": "Learned initializer + Gauss–Newton/LM polishing (hybrid)",
                          "hypothesis": "A cheap learned inverse places LM in the right basin, so a few iterations reach "
                                        "MLE accuracy without multistart; the hybrid combines ML speed with physics accuracy.",
                          "motivation": "Combine the learning and inversion traditions: amortised initialisation.",
                          "assumptions": ["the initializer error is inside the basin of attraction",
                                          "the forward model is exact", f"noise model: {noise}"],
                          "evidence": diag_ev + lit_ev,
                          "method": {"kind": "hybrid", "init": self._lib("ridge", self._feat(["log", "raw"]),
                                                                         params={"alpha": 1e-3}, target_transform="log"),
                                     "params": {"transform": "log", "iters": 10}},
                          "falsifiable_test": "Within 2% of the physics MLE on validation.",
                          "prediction": {"relative_to": "best_physics", "improvement": -0.02},
                          "feasibility": {"cost": "low", "notes": "one start, few iterations"},
                          "if_fails": "the initializer lands outside the basin; multistart needed"})
                P.append({**common, "title": "Posterior-mean (MMSE) estimator under the known prior",
                          "hypothesis": "The metric is squared-error based and the prior is known (uniform box), so the "
                                        "posterior mean — not the likelihood mode — is the optimal estimator.",
                          "motivation": "Challenge the framing: MLE is not MSE-optimal under a known prior.",
                          "assumptions": ["the prior used for data generation is the box-uniform prior", "Laplace proposal covers the posterior"],
                          "evidence": diag_ev, "method": {"kind": "physics", "params": {"transform": "log", "multistart": 1,
                                                                                         "iters": 30, "estimator": "posterior_mean",
                                                                                         "posterior_samples": 128}},
                          "falsifiable_test": "Posterior-mean validation rel_l2 lower than the MLE.",
                          "prediction": {"relative_to": "best_physics", "improvement": 0.0},
                          "feasibility": {"cost": "low", "notes": "+128 forward evaluations per sample"},
                          "if_fails": "posterior is nearly Gaussian and symmetric; mode = mean"})
            else:
                P.append({**common, "title": "Partial least squares on log features",
                          "hypothesis": "A few latent directions explain the observations (smoothing operator), so PLS "
                                        "in log space is a competitive, cheap inverse.",
                          "motivation": "Low-rank structure of diffusion data.",
                          "assumptions": ["low effective rank"], "evidence": lit_ev,
                          "method": self._lib("pls", self._feat(["log", "raw"]), params={"n_components": 20}),
                          "falsifiable_test": "Within 30% of the best data-driven method.",
                          "prediction": {"relative_to": "best_data_driven", "improvement": -0.3},
                          "feasibility": {"cost": "low", "notes": "linear"},
                          "if_fails": "nonlinearity dominates"})
                P.append({**common, "title": "Gradient boosting on sqrt features",
                          "hypothesis": "Tree boosting captures interactions between segments without feature engineering.",
                          "motivation": "Different inductive bias from kernels/MLPs; useful for ensembling.",
                          "assumptions": ["enough data for per-target boosters"], "evidence": lit_ev,
                          "method": self._lib("hist_gbm", self._feat(["sqrt", "raw"])),
                          "falsifiable_test": "Within 40% of the best data-driven method.",
                          "prediction": {"relative_to": "best_data_driven", "improvement": -0.4},
                          "feasibility": {"cost": "medium", "notes": "one booster per target"},
                          "if_fails": "piecewise-constant trees approximate a smooth map poorly"})
        return P

    def _heuristic_revise(self, rc, mine, results, origin) -> list[dict]:
        fb_all = rc.get("feedback", {})
        cites, lit_ev, lit_ok = self._cite(results)
        out = []
        for p in mine:
            fb = fb_all.get(p["id"], {})
            pilot = rc.get("pilots", {}).get(p["id"])
            blocking = fb.get("blocking") or []
            if blocking:
                out.append({"action": "abandon", "target": p["id"], "raw": {}, "origin": origin,
                            "abandon_reason": "blocking issue: " + "; ".join(blocking)[:400],
                            "response_to_critic": "Accepted: cannot be executed as specified."})
                continue
            if fb.get("duplicate_of"):
                out.append({"action": "abandon", "target": p["id"], "raw": {}, "origin": origin,
                            "abandon_reason": f"duplicate of {fb['duplicate_of']}",
                            "response_to_critic": "Accepted: merged into the duplicate."})
                continue
            if pilot and pilot.get("status") not in (None, "ok"):
                if p["method"].get("kind") in ("physics", "hybrid") and p["method"].get("params"):
                    raw = copy.deepcopy(p)
                    raw["method"]["params"] = {k: v for k, v in raw["method"]["params"].items() if k in ("transform",)}
                    out.append({"action": "revise", "target": p["id"], "raw": raw, "origin": origin,
                                "response_to_critic": f"Pilot failed ({pilot.get('error', '')[:120]}); reverted to defaults."})
                else:
                    out.append({"action": "abandon", "target": p["id"], "raw": {}, "origin": origin,
                                "abandon_reason": f"pilot experiment failed: {pilot.get('error', '')[:200]}",
                                "response_to_critic": "Accepted: the pilot contradicts feasibility."})
                continue
            if pilot and pilot.get("worse_than_baseline"):
                out.append({"action": "abandon", "target": p["id"], "raw": {}, "origin": origin,
                            "abandon_reason": f"pilot: no better than the mean baseline ({pilot.get('value')})",
                            "response_to_critic": "Accepted: empirical evidence against the hypothesis."})
                continue
            raw = copy.deepcopy(p)
            changed, notes = False, []
            for ca in fb.get("changed_assumptions") or []:
                if ca not in raw["assumptions"]:
                    raw["assumptions"].append(str(ca)[:300])
                    changed = True
            if fb.get("missing"):
                if "falsifiable test" in fb["missing"] and not raw.get("falsifiable_test"):
                    raw["falsifiable_test"] = "Validation primary metric better than the best data-driven method."
                    raw["prediction"] = raw.get("prediction") or {"relative_to": "best_data_driven", "improvement": 0.0}
                if "if fails" in fb["missing"] and not raw.get("if_fails"):
                    raw["if_fails"] = "the hypothesis' mechanism does not dominate the error"
                if "assumptions" in fb["missing"] and not raw.get("assumptions"):
                    raw["assumptions"] = ["the simulator is the data-generating process"]
                changed = True
                notes.append("filled missing fields")
            if cites:
                raw["citations"] = (raw.get("citations") or []) + [c for c in cites if c["ref_id"] not in
                                                                   {x["ref_id"] for x in raw.get("citations") or []}][:2]
                raw["evidence"] = (raw.get("evidence") or []) + lit_ev[:2]
                if lit_ok:
                    raw["status"] = "literature-supported"
                changed = True
                notes.append("added requested literature with applicability notes")
            dirs = " ".join(fb.get("revision_directions") or []).lower()
            ms = raw.get("method") or {}
            if ("cost" in dirs or "cheaper" in dirs) and ms.get("kind") in ("physics", "hybrid"):
                prm = ms.setdefault("params", {})
                prm["multistart"] = max(1, int(prm.get("multistart", 1)) - 1)
                prm["iters"] = max(10, int(prm.get("iters", 30) * 0.7))
                if prm.get("posterior_samples"):
                    prm["posterior_samples"] = max(64, int(prm["posterior_samples"]) // 2)
                changed = True
                notes.append("reduced solver cost as requested")
            if "multistart" in dirs and ms.get("kind") == "physics":
                ms.setdefault("params", {})["multistart"] = min(3, int(ms["params"].get("multistart", 1)) + 1)
                changed = True
                notes.append("added a second start to test multimodality")
            if fb.get("diversify") and ms.get("kind") == "library":
                alt = self._branch_variant(p)
                if alt is not None:
                    out.append({"action": "branch", "target": p["id"], "raw": alt, "origin": origin,
                                "response_to_critic": "Branched into an under-represented method family as the critic asked."})
                    continue
            if changed:
                out.append({"action": "revise", "target": p["id"], "raw": raw, "origin": origin,
                            "response_to_critic": "Revised: " + "; ".join(notes or ["incorporated critic assumptions"])})
            else:
                out.append({"action": "defend", "target": p["id"], "raw": {}, "origin": origin,
                            "response_to_critic": "Defended: assumptions and test are explicit; the experiment decides."})
        return out

    def _branch_variant(self, p):
        if self.fwd:
            raw = copy.deepcopy(p)
            raw["title"] = "Hybrid branch: " + p["title"][:100]
            raw["method"] = {"kind": "hybrid", "init": copy.deepcopy(p["method"]), "params": {"transform": "log", "iters": 8}}
            raw["hypothesis"] = ("Branch: the learned inverse from '" + p["title"][:80] + "' used as initializer for LM "
                                 "polishing removes its residual approximation error.")
            raw["prediction"] = {"relative_to": "best_data_driven", "improvement": 0.05}
            return raw
        return None
