"""Engineer agent: turns approved hypotheses into executable experiment plans and code.

For library methods the plan is a declarative job. For 'custom' ideas the agent-written code is
materialised as a versioned file (CODE artifact) and, if it crashes during screening, the engineer
asks the LLM to repair it from the traceback (AI-Scientist-style debug loop).
"""
from __future__ import annotations

import itertools
import json
import threading

from .. import artifacts as A
from .base import Agent, log
from .critic import check_code

REPAIR_SYSTEM = ("You are the engineer agent of an autonomous ML research team. You fix Python model code "
                 "so it runs. Keep the idea, fix the bug. Reply with a JSON object only.")


class EngineerAgent(Agent):
    role = "engineer"

    def __init__(self, *a, data_path: str, feature_shape, **kw):
        super().__init__(*a, **kw)
        self.data_path = str(data_path)
        self.feature_shape = feature_shape
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def new_exp_id(self) -> str:
        with self._lock:
            return f"exp-{next(self._ids):04d}"

    def materialize(self, idea: dict, version: int = 1) -> None:
        cfg = idea["config"]
        if cfg.get("method") != "custom":
            return
        path = self.store.workspace("ideas", idea["id"]) / f"model_v{version}.py"
        path.write_text(cfg.get("code") or "", encoding="utf-8")
        cfg["code_path"] = str(path)
        idea["code_version"] = version
        art = self.emit(A.CODE, {"path": str(path), "version": version, "chars": len(cfg.get("code") or "")},
                        idea_id=idea["id"], parents=[idea["hyp_artifact"]], files=[str(path)],
                        summary=f"model code v{version} for {idea['id']}", stage="engineering")
        idea["code_artifact"] = art.id

    def plan(self, idea: dict, fraction: float, stage: str, save_model: bool = False) -> dict:
        exp_id = self.new_exp_id()
        out_dir = self.store.workspace("experiments", exp_id)
        cfg = {k: v for k, v in idea["config"].items() if k != "code"}
        job = {"mode": "train_eval", "exp_id": exp_id, "idea_id": idea["id"], "idea": cfg,
               "fraction": float(fraction), "task": self.spec.task, "primary": self.spec.primary,
               "secondary": self.spec.metric_names[1:], "data_path": self.data_path,
               "data_seed": int(self.spec.data["seed"]), "seed": int(self.spec.seed),
               "feature_shape": self.feature_shape, "save_model": save_model, "stage": stage,
               "threads": int(self.spec.hardware["threads_per_worker"]),
               "wants_gpu": bool(cfg["method"] == "torch_mlp")}
        parents = [idea["hyp_artifact"]] + ([idea["code_artifact"]] if idea.get("code_artifact") else [])
        art = self.emit(A.PLAN, {k: v for k, v in job.items() if k != "data_path"}, idea_id=idea["id"],
                        parents=parents, stage=stage,
                        summary=f"{exp_id}: {cfg['method']} at {fraction:.0%} of training data")
        job["plan_artifact"] = art.id
        job["job_dir"] = str(out_dir)
        return job

    def plan_final(self, members: list[dict], weights: list[float], final_dir, run_id: str) -> dict:
        exp_id = self.new_exp_id()
        out_dir = self.store.workspace("experiments", exp_id)
        job = {"mode": "final", "exp_id": exp_id, "idea_id": members[0]["id"] if len(members) == 1 else "ensemble",
               "members": [{k: v for k, v in m["config"].items() if k != "code"} for m in members],
               "member_ids": [m["id"] for m in members], "weights": weights, "task": self.spec.task,
               "primary": self.spec.primary, "secondary": self.spec.metric_names[1:],
               "data_path": self.data_path, "seed": int(self.spec.seed), "feature_shape": self.feature_shape,
               "final_dir": str(final_dir), "run_id": run_id, "stage": "final", "fraction": 1.0,
               "threads": int(self.spec.hardware["threads_per_worker"]),
               "wants_gpu": any(m["config"]["method"] == "torch_mlp" for m in members)}
        art = self.emit(A.PLAN, {k: v for k, v in job.items() if k != "data_path"},
                        parents=[m["hyp_artifact"] for m in members], stage="final",
                        summary=f"{exp_id}: final fit of {len(members)} member(s) on train+val")
        job["plan_artifact"] = art.id
        job["job_dir"] = str(out_dir)
        return job

    def repair(self, idea: dict, error: str, traceback: str) -> bool:
        cfg = idea["config"]
        if cfg.get("method") != "custom" or not self.uses_llm:
            return False
        prompt = (f"{self.spec.brief()}\n\nIdea: {idea['title']}\n{idea['hypothesis']}\n\nCurrent code:\n"
                  f"```python\n{cfg.get('code')}\n```\n\nIt failed with:\n{error}\n{traceback[-2500:]}\n\n"
                  "Contract: define build_model(params: dict, task: str, seed: int) returning an sklearn-"
                  "compatible estimator. Allowed imports: numpy, scipy, sklearn, torch. No file/network/subprocess. "
                  'Respond as JSON: {"code": "...full fixed file...", "explanation": str}')
        out = self.ask_json(REPAIR_SYSTEM, prompt, purpose="repair code", idea_id=idea["id"], stage="screen")
        if not isinstance(out, dict) or not out.get("code") or check_code(out["code"]):
            return False
        cfg["code"] = out["code"]
        self.materialize(idea, idea.get("code_version", 1) + 1)
        log.info("[%s] repaired code for %s: %s", self.name, idea["id"], str(out.get("explanation"))[:120])
        return True
