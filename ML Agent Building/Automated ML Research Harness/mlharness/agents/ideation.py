"""Ideation agents (AI-Scientist style): independently propose and refine methodologies.

Each ideator has a research persona. With an LLM backend it reasons over the spec, the method
library and the artifact record (leaderboard, failures, critiques); without one it uses a seeded
heuristic policy (persona-specific sampling for proposals, local mutation for refinement).
"""
from __future__ import annotations

import json

import numpy as np

from .. import artifacts as A
from .. import methods as ML
from .base import Agent, log

PERSONAS = {
    "classical": {"desc": "classical statistical learning: linear, kernel and instance-based models with "
                          "careful regularisation",
                  "methods": ["ridge", "pls", "logistic", "kernel_ridge", "svm", "knn"]},
    "ensembles": {"desc": "tree ensembles and gradient boosting for robust nonlinear function approximation",
                  "methods": ["extra_trees", "random_forest", "hist_gbm"]},
    "neural": {"desc": "neural networks: architecture, regularisation and optimisation choices",
               "methods": ["torch_mlp", "mlp_sklearn"]},
    "domain": {"desc": "domain/physics-informed feature engineering that exploits the structure of the "
                       "simulator's observations, paired with suitable learners",
               "methods": ["ridge", "kernel_ridge", "torch_mlp", "extra_trees", "knn", "logistic", "svm"],
               "domain_features": True},
    # Ablation control: uniform random sampling over the whole search space. Never calls the LLM and
    # ignores all feedback (refinement draws fresh random configs instead of building on parents).
    "random": {"desc": "uniform random sampling of methods, hyper-parameters, features, scaler and PCA",
               "methods": [], "random": True},
}

IDEA_SYSTEM = ("You are the {persona} research agent in an autonomous machine-learning research team "
               "(focus: {desc}). You propose concrete, testable methodologies that are aligned with the "
               "research objective. Experiments are cheap to screen but compute is limited, so prefer "
               "ideas with a clear rationale and avoid near-duplicates. Reply with a single JSON object only.")

CUSTOM_HELP = ("Method 'custom' lets you write your own model in Python: the code must define "
               "`build_model(params: dict, task: str, seed: int)` returning a scikit-learn compatible "
               "estimator (fit(X, y) / predict(X)). Features, scaler and PCA are applied before it and "
               "regression targets are standardised for it. Allowed imports: numpy, scipy, sklearn, torch. "
               "No file, network or subprocess access. Propose at most one custom idea, only when the library "
               "cannot express it.")


class IdeationAgent(Agent):
    role = "ideation"

    def __init__(self, *a, persona: str, available: dict, feature_sets: list, suggested_features: list,
                 n_features: int, new_id, seed: int, base_parents=(), **kw):
        super().__init__(*a, **kw)
        self.base_parents = list(base_parents)   # spec + dataset artifacts every idea derives from
        self.persona = persona
        self.p = PERSONAS.get(persona, {"desc": persona, "methods": list(available)})
        self.available = available
        self.feature_sets = feature_sets
        self.suggested = [f for f in suggested_features if f in feature_sets]
        self.n_features = n_features
        self.new_id = new_id
        self.rng = np.random.default_rng(seed)
        mine = [m for m in self.p["methods"] if m in available]
        self.my_methods = mine or [m for m in available if m != "mean_baseline"]

    # -- public API ----------------------------------------------------------------------
    def propose(self, n: int, context: dict, round_idx: int = 0) -> list[dict]:
        if self.p.get("random"):
            return [self._register(i, "random", round_idx, None) for i in self._random_propose(n, context)]
        ideas = self._llm_propose(n, context) if self.uses_llm else None
        origin = "llm"
        if not ideas:
            ideas, origin = self._heuristic_propose(n, context), "heuristic"
        return [self._register(i, origin, round_idx, None) for i in ideas[:n]]

    def refine(self, parents: list[dict], n_children: int, context: dict, round_idx: int) -> list[dict]:
        """parents: [{'idea': idea, 'score': val metric, 'critique': str}]"""
        out = []
        if self.p.get("random"):   # feedback-free: same number of children, but fresh random draws
            fresh = self._random_propose(n_children * len(parents), context)
            return [self._register(c, "random", round_idx, None) for c in fresh]
        llm_children = self._llm_refine(parents, n_children, context) if self.uses_llm else None
        if llm_children:
            by_id = {p["idea"]["id"]: p["idea"] for p in parents}
            for c in llm_children[: n_children * len(parents)]:
                parent = by_id.get(c.get("parent")) or parents[0]["idea"]
                out.append(self._register(c, "llm", round_idx, parent))
            return out
        for p in parents:
            for _ in range(n_children):
                child = self._mutate(p["idea"], context, round_idx)
                if child:
                    out.append(self._register(child, "heuristic", round_idx, p["idea"]))
        return out

    # -- registration --------------------------------------------------------------------
    def _register(self, raw: dict, origin: str, round_idx: int, parent: dict | None) -> dict:
        cfg = {"method": raw.get("method"), "params": raw.get("params") or {},
               "features": raw.get("features") or "raw", "scaler": raw.get("scaler") or "standard",
               "pca": int(raw.get("pca") or 0)}
        if raw.get("method") == "custom":
            cfg["code"] = raw.get("code") or ""
        idea = {"id": self.new_id(), "title": raw.get("title") or f"{cfg['method']} / {cfg['features']}",
                "hypothesis": raw.get("hypothesis") or "", "persona": self.persona, "origin": origin,
                "round": round_idx, "parent": parent["id"] if parent else None, "config": cfg}
        parents = ([parent["hyp_artifact"]] if parent else []) + self.base_parents
        art = self.emit(A.HYPOTHESIS, {k: v for k, v in idea.items()}, idea_id=idea["id"], parents=parents,
                        summary=idea["title"], stage="ideation" if not parent else "refine")
        idea["hyp_artifact"] = art.id
        return idea

    # -- heuristic policies --------------------------------------------------------------
    def _pick_features(self, i):
        if self.p.get("domain_features") and self.suggested:
            return self.suggested[i % len(self.suggested)]
        if i == 0:
            return "raw"
        opts = ["raw", "log"] if "log" in self.feature_sets else ["raw"]
        return opts[int(self.rng.integers(len(opts)))]

    def _heuristic_propose(self, n, context):
        out, used = [], set()
        order = list(self.my_methods)
        for i in range(n):
            key = order[i % len(order)]
            m = self.available[key]
            params = dict(m.defaults) if key not in used else ML.sample_params(m, self.rng)
            used.add(key)
            feats = self._pick_features(i)
            pca = 0
            if key in ("knn", "kernel_ridge", "svm") and self.n_features > 100 and self.rng.random() < 0.4:
                pca = int(self.rng.choice([16, 32, 64]))
            out.append({"title": f"{key} on '{feats}' features" + (f" + PCA{pca}" if pca else ""),
                        "hypothesis": f"{m.description} Using '{feats}' features tests whether a {m.family} "
                                      f"learner captures the target map under the {self.p['desc']} lens.",
                        "method": key, "params": params, "features": feats, "scaler": "standard", "pca": pca})
        return out

    def _random_propose(self, n, context):
        """Uniform random search: method, every hyper-parameter, feature set, scaler and PCA."""
        keys = [k for k in self.available if k != "mean_baseline"]
        existing = set(context.get("hashes", set()))
        out = []
        for _ in range(n):
            for _try in range(20):
                key = keys[int(self.rng.integers(len(keys)))]
                m = self.available[key]
                cfg = {"method": key, "params": ML.sample_params(m, self.rng),
                       "features": self.feature_sets[int(self.rng.integers(len(self.feature_sets)))],
                       "scaler": ML.SCALERS[int(self.rng.integers(len(ML.SCALERS)))],
                       "pca": 0 if self.rng.random() < 0.5 else
                       int(self.rng.choice([c for c in (8, 16, 32, 64) if c < self.n_features] or [0]))}
                h = ML.config_hash(cfg)
                if h not in existing:
                    existing.add(h)
                    break
            cfg["title"] = f"random {key} on '{cfg['features']}' ({cfg['scaler']}, pca={cfg['pca']})"
            cfg["hypothesis"] = "Random-search control: configuration drawn uniformly from the search space."
            out.append(cfg)
        return out

    def _mutate(self, parent, context, round_idx):
        cfg = parent["config"]
        if cfg["method"] not in self.available:
            return None
        m = self.available[cfg["method"]]
        existing = context.get("hashes", set())
        for _ in range(8):
            strength = max(0.15, 0.6 / (1 + round_idx))
            child = {"method": cfg["method"], "params": ML.mutate_params(m, cfg["params"], self.rng, strength),
                     "features": cfg["features"], "scaler": cfg["scaler"], "pca": cfg["pca"]}
            if self.rng.random() < 0.25 and len(self.feature_sets) > 1:
                child["features"] = str(self.rng.choice([f for f in self.feature_sets if f != cfg["features"]]))
            if self.rng.random() < 0.1 and self.n_features > 64:
                child["pca"] = 0 if cfg["pca"] else int(self.rng.choice([32, 64]))
            if ML.config_hash(child) not in existing:
                diff = {k: v for k, v in child["params"].items() if cfg["params"].get(k) != v}
                child["title"] = f"{cfg['method']} refinement of {parent['id']}: {json.dumps(diff, default=str)[:80]}"
                child["hypothesis"] = (f"Local search around {parent['id']} (val score {context.get('scores', {}).get(parent['id'])}):"
                                       f" perturbing {sorted(diff) or 'features'} may improve validation error.")
                return child
        return None

    # -- LLM policies --------------------------------------------------------------------
    def _context_text(self, context):
        lb = context.get("leaderboard") or []
        lines = [f"  {r['idea_id']} {r['method']} features={r['features']} params={json.dumps(r['params'], default=str)} "
                 f"val_{self.spec.primary}={r['value']:.5g} (fraction={r['fraction']})" for r in lb[:12]]
        fails = [f"  {f['idea_id']} {f['method']}: {f['reason']}" for f in (context.get("failures") or [])[:8]]
        base = context.get("baseline")
        return ("Baseline (predict mean): " + (f"{base:.5g}" if base is not None else "n/a") + "\n"
                "Results so far:\n" + ("\n".join(lines) or "  none yet") + "\n"
                "Failures / pruned so far:\n" + ("\n".join(fails) or "  none") + "\n"
                f"Already proposed (do not duplicate): {', '.join(context.get('proposed', [])[:40]) or 'none'}\n"
                f"Remaining compute: {context.get('budget', {}).get('compute_remaining_s', '?')} worker-seconds")

    def _library_text(self):
        allow_custom = self.spec.constraints.get("allow_custom_code", True)
        return (f"Method library (use these keys; params from each search space):\n"
                f"{ML.catalog_text(self.available)}\n"
                f"Feature sets (applied before scaling): {self.feature_sets}\n"
                f"Scalers: standard | robust | none.  pca: 0 (off) or a number of components "
                f"(raw feature dimension {self.n_features}).\n" + (CUSTOM_HELP if allow_custom else ""))

    def _llm_propose(self, n, context):
        system = IDEA_SYSTEM.format(persona=self.persona, desc=self.p["desc"])
        prompt = (f"{self.spec.brief()}\n\n{self._library_text()}\n\n{self._context_text(context)}\n\n"
                  f"Propose {n} diverse ideas from your perspective ({self.p['desc']}). Respond as JSON: "
                  '{"ideas": [{"title": str, "hypothesis": "why this should work for THIS problem", '
                  '"method": key or "custom", "params": {}, "features": str, "scaler": str, "pca": int, '
                  '"code": "only for custom"}]}')
        out = self.ask_json(system, prompt, purpose=f"propose ideas ({self.persona})", stage="ideation")
        ideas = out.get("ideas") if isinstance(out, dict) else out if isinstance(out, list) else None
        if not ideas:
            return None
        return [i for i in ideas if isinstance(i, dict) and i.get("method")]

    def _llm_refine(self, parents, n_children, context):
        system = IDEA_SYSTEM.format(persona=self.persona, desc=self.p["desc"])
        ptxt = "\n".join(
            f"  parent {p['idea']['id']}: config={json.dumps(p['idea']['config'], default=str)[:600]} "
            f"val_{self.spec.primary}={p['value']:.5g}; critic: {p.get('critique') or 'none'}" for p in parents)
        prompt = (f"{self.spec.brief()}\n\n{self._library_text()}\n\n{self._context_text(context)}\n\n"
                  f"Best full-fidelity ideas so far:\n{ptxt}\n\n"
                  f"Propose {n_children} improved variant(s) for EACH parent (hyper-parameters, features, "
                  "regularisation, or a closely related method), informed by the results and critiques. "
                  'Respond as JSON: {"ideas": [{"parent": parent_id, "title": str, "hypothesis": str, '
                  '"method": key or "custom", "params": {}, "features": str, "scaler": str, "pca": int, '
                  '"code": "only for custom"}]}')
        out = self.ask_json(system, prompt, purpose=f"refine ideas ({self.persona})", stage="refine")
        ideas = out.get("ideas") if isinstance(out, dict) else out if isinstance(out, list) else None
        if not ideas:
            return None
        return [i for i in ideas if isinstance(i, dict) and i.get("method")]
