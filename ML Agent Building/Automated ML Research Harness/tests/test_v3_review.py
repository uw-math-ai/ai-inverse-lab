"""Independent adversarial review fixtures; no network or scientific campaign."""
from __future__ import annotations

import tempfile
import unittest
import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlharness.knowledge_v3 import ExposureRegistry, validate_evaluation_protocol
from mlharness.research.contracts_v3 import evaluate_claim
from mlharness.research.program_v3 import DelegationManager
from mlharness.research.studies_v3 import study_completion


@contextmanager
def temporary_harness_directory():
    with tempfile.TemporaryDirectory() as directory:
        try:
            yield directory
        finally:
            logger = logging.getLogger("mlharness")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


class IndependentScientificReview(unittest.TestCase):
    def test_study_missing_uncertainty_differs_from_overlapping_interval(self):
        claim = {"id": "threshold", "kind": "performance", "treatment": "t",
                 "metric": "rel_l2", "operator": "<=", "threshold": .2,
                 "requires_uncertainty": True}
        study = {"id": "study", "question": "Is the threshold met?", "claims": [claim],
                 "arms": [{"id": "t", "method": {"kind": "library", "method": "ridge"}}]}
        results = {"t": {"status": "ok", "val": {"rel_l2": .1}, "row_hash": "rows",
                         "seed": 0, "fidelity": 1}}
        self.assertEqual(study_completion(study, results)["status"], "incomplete")
        diagnostics = {"scalar:t:rel_l2": {"status": "ok", "kind": "scalar_metric", "metric": "rel_l2",
                                            "treatment": "t", "row_hash": "rows", "metric_interval": [.1, .3]}}
        self.assertEqual(evaluate_claim(claim, results, diagnostics)["outcome"], "inconclusive")
        self.assertEqual(study_completion(study, results, diagnostics)["status"], "complete")

    def test_critic_objection_cannot_resolve_with_unrelated_work(self):
        from mlharness.scheduler_v3 import ResearchSchedulerV3
        for resolution in ("evidence", "revision"):
            with self.subTest(resolution=resolution):
                scheduler = object.__new__(ResearchSchedulerV3)
                scheduler.lead_id = "lead"
                scheduler.program = {"studies": {
                    "scope": {"id": "scope", "status": "complete", "verdicts": []},
                    "unrelated": {"id": "unrelated", "status": "complete", "verdicts": []}},
                    "objections": [{"id": "objection", "scope": "scope", "blocking": True,
                                    "status": "open", "requirement": "Independent calibration evidence"}],
                    "evidence": {"foreign-artifact": {"study_id": "unrelated", "status": "ok"}}}
                action = {"type": "resolve_objection", "target": "objection", "resolution": resolution,
                          "evidence_ids": ["foreign-artifact"], "revision_id": "unrelated"}
                with self.assertRaises(ValueError):
                    scheduler._dispatch_action(action)
                self.assertEqual(scheduler.program["objections"][0]["status"], "open")

    def test_paired_evidence_scenarios_change_subsequent_action(self):
        from mlharness.scheduler_v3 import ResearchSchedulerV3
        scheduler = object.__new__(ResearchSchedulerV3)
        scheduler.cfg = {"max_agents": 8, "max_active_agents": 6, "auto_followup": True}
        scheduler.spec = SimpleNamespace(research={"researchers": ["ml"]})
        scheduler.delegation = SimpleNamespace(agents={
            "r1": {"role": "researcher"}, "r2": {"role": "researcher"}})
        scheduler.program = {"questions": [{"id": "q", "question": "effect?"}],
                             "studies": {"s": {"id": "s", "status": "complete",
                                                "verdicts": [{"outcome": "supported"}]}}}
        positive = scheduler._offline_action()
        scheduler.program["studies"]["s"]["verdicts"][0]["outcome"] = "contradicted"
        negative = scheduler._offline_action()
        self.assertEqual(positive["type"], "stop")
        self.assertEqual(negative["type"], "diagnose")

    def test_scalar_uncertainty_is_an_evidence_obligation(self):
        claim = {"id": "threshold", "kind": "performance", "treatment": "t",
                 "metric": "rel_l2", "operator": "<=", "threshold": .2,
                 "requires_uncertainty": True}
        result = {"status": "ok", "val": {"rel_l2": .1}}
        verdict = evaluate_claim(claim, {"t": result})
        self.assertNotEqual(verdict["outcome"], "supported", verdict)
        self.assertTrue(verdict["unmet_requirements"] or verdict["outcome"] == "untestable", verdict)

    def test_comparison_disallows_unregistered_method_changes(self):
        claim = {"id": "comparison", "kind": "comparison", "treatment": "t", "control": "c",
                 "metric": "rel_l2", "operator": "at_least_better", "margin": .1,
                 "allowed_varying_factors": ["params.alpha"]}
        common = {"status": "ok", "row_hash": "same", "seed": 0, "fidelity": 1}
        arms = {"t": {**common, "val": {"rel_l2": .1},
                       "effective_method": {"features": "log", "params": {"alpha": 1}}},
                "c": {**common, "val": {"rel_l2": .2},
                       "effective_method": {"features": "raw", "params": {"alpha": 2}}}}
        verdict = evaluate_claim(claim, arms)
        self.assertNotEqual(verdict["outcome"], "supported", verdict)

    def test_comparison_uncertainty_must_describe_registered_pair(self):
        claim = {"id": "comparison", "kind": "comparison", "treatment": "t", "control": "c",
                 "metric": "rel_l2", "operator": "at_least_better", "margin": .1,
                 "requires_uncertainty": True, "analysis_id": "unrelated"}
        common = {"status": "ok", "row_hash": "same", "seed": 0, "fidelity": 1}
        arms = {"t": {**common, "val": {"rel_l2": .1}}, "c": {**common, "val": {"rel_l2": .2}}}
        diagnostics = {"unrelated": {"status": "ok", "metric": "accuracy", "treatment": "other-t",
                                     "control": "other-c", "row_hash": "other-rows",
                                     "relative_effect_interval": [.3, .6]}}
        verdict = evaluate_claim(claim, arms, diagnostics)
        self.assertNotEqual(verdict["outcome"], "supported", verdict)

    def test_malformed_protocol_returns_invalid_record(self):
        for protocol in ({"comparisons": None}, {"comparisons": [{"arms": None}]},
                         {"instances": [{"id": [], "dataset_seed": [], "search_seeds": [1]}]},
                         {"split_roles": []}):
            with self.subTest(protocol=protocol):
                result = validate_evaluation_protocol(protocol)
                self.assertFalse(result["valid"], result)
                self.assertTrue(result["errors"], result)

    def test_duplicate_persona_instances_and_real_nested_callbacks(self):
        manager = DelegationManager({"max_agents": 8, "max_active_agents": 8,
                                     "max_depth": 3, "max_children": 4})
        lead = manager.spawn("orchestrator")
        first = manager.spawn("researcher", persona="mathematical", parent_id=lead["id"], question="q1")
        second = manager.spawn("researcher", persona="mathematical", parent_id=lead["id"], question="q2")
        self.assertNotEqual(first["id"], second["id"])
        seen = []

        def callback(agent, request):
            nested = manager.dispatch(agent["id"], "analysis", {"question": "nested"},
                                      lambda child, req: {"status": "ok", "value": 3})
            seen.append(nested["result"]["value"])
            return {"status": "ok", "value": sum(seen), "uncertainty": ["fixture only"]}

        task = manager.dispatch(first["id"], "coding_execution", {"question": "compute"}, callback)
        self.assertEqual(task["result"]["value"], 3)
        self.assertEqual(first["messages"][0]["result"]["value"], 3)
        self.assertEqual(len(manager.tasks), 2)
        self.assertEqual(second["messages"], [])

    def test_concurrent_begin_consumes_exposure_once(self):
        with temporary_harness_directory() as d:
            path = Path(d) / "exposure.sqlite"
            registry = ExposureRegistry(path)
            token = registry.reserve("data", "split", "run", "selection", strict=True)

            def begin(_):
                try:
                    ExposureRegistry(path).begin(token)
                    return True
                except PermissionError:
                    return False

            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(sum(pool.map(begin, range(8))), 1)
            self.assertEqual(registry.get(token)["status"], "attempted")
            with self.assertRaises(PermissionError):
                ExposureRegistry(path).begin(token)

    def test_integrated_scripted_researcher_and_certified_resume(self):
        from mlharness.llm import LLMBackend, LLMResponse, NullLLM
        from mlharness.memory import InMemoryBackend, ResearchMemory
        from mlharness.knowledge_v3 import MethodPackageStore
        from mlharness.scheduler_v3 import ResearchSchedulerV3
        from mlharness.spec import compile_spec

        class Researcher(LLMBackend):
            name, model = "independent-fixture", "scripted-researcher"

            def __init__(self):
                self.investigations = 0

            def complete(self, system, prompt, timeout=None):
                context = json.loads(prompt)
                if "task" in context:
                    self.investigations += 1
                    value = {"interpretation": "Only development evidence is available."}
                    if self.investigations == 1:
                        value["specialist_requests"] = [{"kind": "source", "request": {
                            "type": "local", "key": "noise", "question": "Inspect local noise contract"}}]
                else:
                    value = {"interpretation": "The local source returned the noise definition.",
                             "evidence_ids": []}
                return LLMResponse(json.dumps(value), 0, self.name, self.model)

        with temporary_harness_directory() as d:
            root = Path(d)
            method = {"kind": "library", "method": "ridge", "params": {"alpha": 1.0},
                      "features": "raw", "scaler": "standard", "pca": 0}
            study = {"id": "independent-study", "question": "Does alpha change validation error?",
                     "arms": [{"id": "independent-control", "role": "baseline", "method": method},
                              {"id": "independent-control-copy", "role": "control", "method": method},
                              {"id": "independent-treatment", "role": "treatment",
                               "depends_on": ["independent-control-copy"],
                               "method": {**method, "params": {"alpha": 2.0}}}],
                     "matching": {"allowed_varying_factors": ["params.alpha"]},
                     "diagnostics": [{"id": "independent-paired", "kind": "paired",
                                      "treatment": "independent-treatment", "control": "independent-control"}],
                     "claims": [{"id": "independent-claim", "kind": "comparison", "metric": "rel_l2",
                                 "treatment": "independent-treatment", "control": "independent-control",
                                 "operator": "at_least_better", "margin": 100.0,
                                 "required_diagnostics": ["independent-paired"]}]}
            spec = compile_spec({"pipeline": "v3", "name": "independent_fixture",
                "problem": "Tiny offline software validation",
                "simulator": {"name": "friedman", "params": {"n_features": 6, "noise": .1}},
                "data": {"n_train": 30, "n_val": 12, "n_test": 10, "seed": 13213},
                "budget": {"compute_seconds": 90, "wall_seconds": 120, "experiment_timeout": 15,
                           "llm_calls": 20}, "hardware": {"workers": 2, "threads_per_worker": 1},
                "constraints": {"allowed_methods": ["ridge"], "allow_forward_model": False},
                "research": {"researchers": ["ml"]}, "memory": {"enabled": False},
                "literature": {"enabled": False}, "llm": {"backend": "none"},
                "execution": {"ensemble": False, "bootstrap": 10},
                "evaluation": {"shift_sets": [], "strict_holdout": True,
                               "exposure_registry": str(root / "exposures.sqlite")},
                "program": {"studies": [study], "max_actions": 12}}, hardware={
                    "cpus": 2, "gpus": [], "accelerator": "cpu", "torch": False,
                    "platform": "fixture", "python": "3.11"})
            memory = ResearchMemory(backend=InMemoryBackend(), enabled=False)
            scheduler = ResearchSchedulerV3(spec, root / "run", llm=NullLLM(),
                memory=memory, role_llms={"researcher": Researcher()}, verbose=False)
            scheduler.package_store = MethodPackageStore(root / "packages")
            result = scheduler.run()
            bundle = json.loads((root / "run/evidence_bundle.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "run/final/manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["export_certificate"]["certified"])
            self.assertEqual(manifest["export_certificate"], result["final"]["certificate"])
            self.assertEqual(bundle["studies"][0]["status"], "complete")
            self.assertTrue(bundle["arm_results"]["independent-control-copy"]["reused"])
            self.assertEqual(bundle["studies"][0]["verdicts"][0]["outcome"], "contradicted")
            self.assertTrue(any(a["type"] == "diagnose" for a in bundle["program"]["actions"]))
            people = [a for a in bundle["agents"].values() if a["role"] == "researcher"]
            self.assertEqual([a["persona"] for a in people], ["ml", "ml"])
            self.assertTrue(bundle["specialist_tasks"])
            self.assertEqual(bundle["specialist_tasks"][0]["result"]["status"], "ok")
            self.assertTrue((root / "run/report.md").exists())
            self.assertTrue((root / "run/program_evaluation.json").exists())
            self.assertEqual(scheduler.exposures.count(), 1)
            self.assertEqual(scheduler.exposures.entries()[0]["status"], "completed")
            final_before = (root / "run/final/model.pkl").read_bytes()
            resumed = ResearchSchedulerV3(spec, root / "run", llm=NullLLM(),
                memory=memory, resume=True, verbose=False)
            resumed.run()
            self.assertEqual(resumed.exposures.count(), 1)
            self.assertEqual(final_before, (root / "run/final/model.pkl").read_bytes())


if __name__ == "__main__":
    unittest.main()
