"""Small software integration fixtures; no scientific campaign or live services."""
import json
import logging
import copy
import tempfile
import unittest
from pathlib import Path

from mlharness.agents.reporter_v3 import write_report_v3
from mlharness.llm import NullLLM
from mlharness.knowledge_v3 import MethodPackageStore
from mlharness.scheduler_v3 import ResearchSchedulerV3
from mlharness.spec import compile_spec


HW = {"cpus": 2, "gpus": [], "accelerator": "cpu", "torch": False, "platform": "fixture", "python": "3"}


class RuntimeTests(unittest.TestCase):
    def test_offline_study_export_resume_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = compile_spec({"name": "v3_software_fixture", "pipeline": "v3",
                "simulator": {"name": "regression", "params": {"n_features": 4, "n_informative": 3}},
                "data": {"n_train": 40, "n_val": 12, "n_test": 12, "seed": 923},
                "hardware": {"workers": 1, "threads_per_worker": 1},
                "constraints": {"allowed_methods": ["ridge"], "allow_forward_model": False},
                "budget": {"compute_seconds": 80, "experiment_timeout": 20},
                "llm": {"backend": "none"}, "memory": {"enabled": False}, "literature": {"enabled": False},
                "execution": {"ensemble": False, "bootstrap": 20},
                "evaluation": {"shift_sets": [], "exposure_registry": str(root/"exposure.sqlite")},
                "program": {"max_actions": 12}}, hardware=HW)
            run = root/"run"
            scheduler = ResearchSchedulerV3(spec, run, llm=NullLLM(), verbose=False)
            scheduler.package_store = MethodPackageStore(root/"packages")
            out = scheduler.run()
            self.assertTrue(out["final"]["certificate"]["certified"])
            bundle = json.loads((run/"evidence_bundle.json").read_text())
            self.assertEqual(bundle["studies"][0]["status"], "complete")
            self.assertTrue(bundle["contract"]["splits"]["test"]["fingerprint"])
            count = scheduler.exposures.count()
            original = (run/"report.md").read_text(encoding="utf-8")
            self.assertEqual(write_report_v3(run), original)
            resumed = ResearchSchedulerV3(spec, run, llm=NullLLM(), resume=True, verbose=False).run()
            self.assertEqual(resumed, out)
            self.assertEqual(scheduler.exposures.count(), count)
            changed = copy.deepcopy(spec)
            changed.data["seed"] += 1
            with self.assertRaisesRegex(ValueError, "identical compiled specification"):
                ResearchSchedulerV3(changed, run, llm=NullLLM(), resume=True, verbose=False).run()
            self.assertEqual(scheduler.exposures.count(), count)
            # Simulate interruption after the final result was persisted but before report
            # completion. The ledger can recover the final artifact without another access.
            final_bytes = (run/"final/model.pkl").read_bytes()
            (run/"v3_result.json").unlink()
            (run/"v3_final.json").unlink()
            recovering = ResearchSchedulerV3(spec, run, llm=NullLLM(), resume=True, verbose=False)
            recovering.package_store = MethodPackageStore(root/"packages")
            recovered = recovering.run()
            self.assertEqual(recovered["final"]["test"], out["final"]["test"])
            self.assertEqual((run/"final/model.pkl").read_bytes(), final_bytes)
            self.assertEqual(recovering.exposures.count(), count)
            self.assertGreater(int(recovering.engineer.new_exp_id().split("-")[1]), 2)
            logging.shutdown()

    def test_random_search_and_real_shared_prediction_computation(self):
        method = {"kind": "library", "method": "ridge", "params": {"alpha": 1.0},
                  "features": "raw", "scaler": "standard", "pca": 0}
        studies = [{"id": "search", "question": "Can the bounded search execute all candidates?",
                    "arms": [{"id": "search.control", "role": "baseline", "method": method},
                             {"id": "search.policy", "role": "treatment", "method": {"kind": "random_search", "n": 2}}],
                    "claims": [{"id": "search.claim", "kind": "performance", "metric": "rel_l2",
                                "treatment": "search.policy", "operator": "<=", "threshold": 1}]},
                   {"id": "sharing", "question": "Reuse one computation without fitting duplicate arms",
                    "arms": [{"id": "sharing.a", "role": "baseline", "method": method},
                             {"id": "sharing.b", "role": "treatment", "method": method}],
                    "shared_computation": {"id": "shared-fixture", "protocol": "persisted_prediction_analysis"},
                    "claims": [{"id": "sharing.claim", "kind": "comparison", "metric": "rel_l2",
                                "treatment": "sharing.b", "control": "sharing.a", "operator": "equivalent", "margin": 0}],
                    "matching": {"allowed_varying_factors": []}}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                spec = compile_spec({"pipeline": "v3", "name": "search_sharing_fixture",
                    "simulator": {"name": "regression", "params": {"n_features": 4, "n_informative": 3}},
                    "data": {"n_train": 30, "n_val": 10, "n_test": 10, "seed": 932},
                    "constraints": {"allowed_methods": ["ridge"], "allow_forward_model": False},
                    "hardware": {"workers": 2, "threads_per_worker": 1},
                    "llm": {"backend": "none"}, "memory": {"enabled": False}, "literature": {"enabled": False},
                    "budget": {"compute_seconds": 100, "experiment_timeout": 20},
                    "execution": {"ensemble": False, "bootstrap": 10},
                    "evaluation": {"shift_sets": [], "exposure_registry": str(root/"exposures.sqlite")},
                    "program": {"studies": studies, "max_actions": 15}}, hardware=HW)
                scheduler = ResearchSchedulerV3(spec, root/"run", llm=NullLLM(), verbose=False)
                scheduler.package_store = MethodPackageStore(root/"packages")
                scheduler.run()
                search = scheduler.program["studies"]["search"]
                self.assertEqual(len(search["arms"]), 4)
                parent = scheduler.arm_results["search.policy"]
                self.assertTrue(parent["exploratory_selection"])
                self.assertIn(parent["selected_candidate"], parent["candidate_ids"])
                self.assertEqual(search["verdicts"][0]["outcome"], "inconclusive")
                shared = scheduler.program["studies"]["sharing"]
                self.assertEqual(shared["status"], "complete")
                self.assertEqual(scheduler.arm_ideas["sharing.a"], scheduler.arm_ideas["sharing.b"])
                self.assertEqual(scheduler.arm_results["sharing.b"]["shared_artifact_id"], "shared-fixture")
                self.assertEqual(shared["verdicts"][0]["outcome"], "supported")
            finally:
                logging.shutdown()


if __name__ == "__main__":
    unittest.main()
