import json
import unittest

from workflow.simulator import MAX_STEPS, run_simulator_request


class SimulatorTests(unittest.TestCase):
    def test_path_three_hand_computation(self):
        h, qx = 0.1, 0.4
        result = run_simulator_request({"graph": "path3", "T": 3, "h": h, "q": [0, qx, 0]})
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["r_exact"][1][0][1], h ** 2)
        self.assertAlmostEqual(result["r_exact"][2][0][1], h ** 2 * (2 - 3 * h - h * qx))
        self.assertLess(result["mass_max_error"], 1e-12)
        self.assertEqual(result["evidence_kind"], "numerical_only")
        json.dumps(result, allow_nan=False)

    def test_finite_horizon_match_is_not_all_time_match(self):
        request = {"operation": "compare_potentials", "graph": "path3", "T": 2,
                   "h": 0.1, "q": [0, 0, 0], "q_other": [0, 1, 0]}
        short = run_simulator_request(request)
        self.assertTrue(short["matches_within_tolerance"])
        self.assertTrue(short["potentials_distinct"])
        self.assertIsNone(short["distinguishing_observation"])
        self.assertIn("finite-time", " ".join(short["limitations"]))
        long = run_simulator_request({**request, "T": 3})
        self.assertFalse(long["matches_within_tolerance"])
        self.assertEqual(long["distinguishing_observation"]["t"], 3)
        self.assertAlmostEqual(long["max_absolute_difference"], 0.1 ** 3)

    def test_fixed_h_comparison_matches_independent_requests(self):
        request = {"operation": "compare_potentials", "graph": "paper9", "h": 0.2,
                   "T": 5, "q": 0.1, "q_other": 0.8, "observed": [2, 8]}
        result = run_simulator_request(request)
        self.assertEqual(result["status"], "completed")
        witness = result["distinguishing_observation"]
        self.assertIsNotNone(witness)
        base = {k: v for k, v in request.items() if k not in ("operation", "q_other")}
        left = run_simulator_request(base)
        right = run_simulator_request({**base, "q": request["q_other"]})
        t = witness["t"] - 1
        source = request["observed"].index(witness["source"])
        target = request["observed"].index(witness["target"])
        self.assertEqual(witness["r_q"], left["r_exact"][t][source][target])
        self.assertEqual(witness["r_q_other"], right["r_exact"][t][source][target])
        self.assertEqual(result["h"], 0.2)

    def test_inadmissible_h_is_not_silently_adjusted(self):
        result = run_simulator_request({"operation": "compare_potentials", "graph": "path3",
                                        "h": 0.4, "q": 0, "q_other": [0, 1, 0]})
        self.assertEqual(result["status"], "error")
        self.assertIn("negative holding probability", result["error"])

    def test_presets_and_selected_observations(self):
        for graph in ("path3", "paper8", "paper9"):
            with self.subTest(graph=graph):
                result = run_simulator_request({"graph": graph, "observed": [0], "T": MAX_STEPS})
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(result["r_exact"]), MAX_STEPS)
                self.assertEqual(result["observed"], [0])

    def test_invalid_or_executable_inputs_are_data_errors(self):
        bad = [None, [], {"code": "raise RuntimeError()"}, {"q": "__import__('os')"},
               {"graph": []}, {"operation": "eval"}, {"q": float("nan")},
               {"q": float("inf")}, {"q": 10 ** 1000}, {"q": -1}, {"q": True},
               {"q": [0, 0]}, {"h": 0}, {"h": 2}, {"h": "auto"}, {"T": True},
               {"T": 0}, {"T": MAX_STEPS + 1}, {"T": 1.5}, {"observed": []},
               {"observed": [0, 0]}, {"observed": [3]}, {"observed": [False]},
               {"observed": [[0]]}, {"q_other": 0}, {"operation": "compare_potentials"}]
        for request in bad:
            with self.subTest(request=request):
                result = run_simulator_request(request)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["evidence_kind"], "numerical_only")
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
