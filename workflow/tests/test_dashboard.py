"""Local HTTP, artifact isolation, persistence, and real offline job regressions."""

from copy import deepcopy
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from workflow import credentials
from workflow.dashboard import APIError, DashboardServer, DashboardStore, MAX_REQUEST_BYTES, WEB_DIR
from workflow.evaluation import proposal_digest
from workflow.records import write_json
from workflow.runtime import RunConfig, run_workflow


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="math-dashboard-tests-")
        self.base = Path(self.temporary.name).resolve()
        self.enterContext(patch.object(credentials, "KEY_FILE", self.base / ".secrets" / "openai.key"))
        self.data = self.base / "runs"
        self.web = self.base / "web"
        self.web.mkdir()
        (self.web / "index.html").write_text("<!doctype html><title>Test dashboard</title>")
        (self.web / "app.js").write_text("'use strict';")
        (self.web / "math-renderer.js").write_text("'use strict';")
        (self.web / "math-copy.js").write_text("'use strict';")
        (self.web / "styles.css").write_text("body { color: black; }")
        (self.web / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        self.store = DashboardStore(self.data, start_worker=False)
        self.server = DashboardServer(("127.0.0.1", 0), self.store, self.web)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, method, path, body=None, *, origin=True, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        request_headers = dict(headers or {})
        if body is not None:
            body = json.dumps(body).encode() if not isinstance(body, bytes) else body
            request_headers.setdefault("Content-Type", "application/json")
        if method == "POST" and origin:
            request_headers.setdefault("Origin", f"http://127.0.0.1:{self.server.server_port}")
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        result = response.read()
        content_type = response.getheader("Content-Type", "")
        if "application/json" in content_type:
            result = json.loads(result)
        values = response.status, result, dict(response.getheaders())
        connection.close()
        return values

    def make_run(self, parent=None, *, mode="multi"):
        return run_workflow(RunConfig(mode=mode, backend="demo", rounds=1), parent or self.data)

    def wait_job(self, identifier, terminal=True):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            workspace = self.request("GET", "/api/workspace")[1]
            job = next(job for job in workspace["jobs"] if job["id"] == identifier)
            if (job["status"] not in ("queued", "running")) if terminal else (job["status"] == "running"):
                return job
            time.sleep(0.03)
        self.fail("Background job did not reach its expected state.")

    def test_catalog_handles_nested_comparisons_and_partial_or_corrupt_runs(self):
        directory, summary = self.make_run(self.data / "comparison-fixture")
        partial = self.data / "partial-run"
        partial.mkdir()
        write_json(partial / "manifest.json", {"run_id": "partial-run", "mode": "multi", "backend": "demo"})
        partial_record = deepcopy(summary["records"][0])
        (partial / "iterations.jsonl").write_text(json.dumps(partial_record) + "\n{unfinished")
        (partial / "events.jsonl").write_text("\n".join(json.dumps({"sequence": i}) for i in range(125)) + "\n{unfinished")
        corrupt = self.data / "corrupt-run"
        corrupt.mkdir()
        (corrupt / "summary.json").write_text("{unfinished")
        status, workspace, _ = self.request("GET", "/api/workspace")
        self.assertEqual(status, 200)
        runs = {run["id"]: run for run in workspace["runs"]}
        self.assertEqual(runs[summary["run_id"]]["comparison_id"], "comparison-fixture")
        self.assertEqual(runs["partial-run"]["iteration_count"], 1)
        self.assertEqual(runs["partial-run"]["status"], "incomplete")
        self.assertTrue(runs["corrupt-run"]["warnings"])
        status, comparison, _ = self.request("GET", "/api/comparisons/comparison-fixture")
        self.assertEqual(status, 200)
        self.assertEqual(comparison["run_ids"], [summary["run_id"]])
        self.assertIn("by_mode", comparison["aggregate"])
        self.assertEqual(comparison["status"], "incomplete")
        write_json(directory.parent / "comparison.json", comparison["aggregate"])
        finalized = self.request("GET", "/api/comparisons/comparison-fixture")[1]
        self.assertEqual(finalized["status"], "completed")
        status, detail, _ = self.request("GET", "/api/runs/partial-run")
        self.assertEqual(status, 200)
        self.assertIsNone(detail["summary"])
        self.assertEqual(len(detail["events"]), 100)
        self.assertEqual(detail["events"][0]["sequence"], 25)

    def test_static_files_and_downloads_use_explicit_allowlists(self):
        _, summary = self.make_run()
        for asset in ("/", "/app.js", "/styles.css", "/favicon.svg", "/math-renderer.js", "/math-copy.js"):
            self.assertEqual(self.request("GET", asset)[0], 200)
        for path in ("/dashboard.py", "/../dashboard.py", "/%2e%2e/dashboard.py",
                     f"/api/runs/{summary['run_id']}/files/../../dashboard.py",
                     f"/api/runs/{summary['run_id']}/files/%2e%2e%2fsummary.json",
                     f"/api/runs/{summary['run_id']}/files/unknown.json"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        status, body, headers = self.request("GET", f"/api/runs/{summary['run_id']}/files/report.md")
        self.assertEqual(status, 200)
        self.assertIn(b"Mathematical research run", body)
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="report.md"')
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_pinned_katex_assets_and_all_referenced_fonts_are_served_locally(self):
        vendor = self.web / "vendor" / "katex"
        shutil.copytree(WEB_DIR / "vendor" / "katex", vendor)
        for filename, mime in (("katex.min.js", "text/javascript; charset=utf-8"),
                               ("contrib/auto-render.min.js", "text/javascript; charset=utf-8"),
                               ("katex.min.css", "text/css; charset=utf-8"),
                               ("LICENSE", "text/plain; charset=utf-8")):
            status, body, headers = self.request("GET", "/vendor/katex/" + filename)
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], mime)
            self.assertEqual(body, (vendor / filename).read_bytes())
        self.assertIn(b'0.19.1', (vendor / "katex.min.js").read_bytes())
        css = (vendor / "katex.min.css").read_text()
        fonts = set(re.findall(r"url\(([^)]+)\)", css))
        self.assertEqual(len(fonts), 60)
        signatures = {"woff2": b"wOF2", "woff": b"wOFF", "ttf": b"\x00\x01\x00\x00"}
        for filename in sorted(fonts):
            with self.subTest(font=filename):
                self.assertTrue(filename.startswith("fonts/"))
                status, body, headers = self.request("GET", "/vendor/katex/" + filename)
                self.assertEqual(status, 200)
                extension = filename.rsplit(".", 1)[1]
                self.assertEqual(headers["Content-Type"], "font/" + extension)
                self.assertTrue(body.startswith(signatures[extension]))
        policy = self.request("GET", "/")[2]["Content-Security-Policy"]
        self.assertIn("font-src 'self'", policy)
        self.assertIn("script-src 'self'", policy)
        self.assertNotIn("unsafe-eval", policy)

    def test_katex_routes_reject_extra_files_traversal_and_nested_symlinks(self):
        vendor = self.web / "vendor" / "katex"
        shutil.copytree(WEB_DIR / "vendor" / "katex", vendor)
        (vendor / "extra.js").write_text("unlisted content")
        (vendor / "fonts" / "KaTeX_Unlisted-Regular.woff2").write_bytes(b"unlisted font")
        for route in ("/vendor/katex/extra.js", "/vendor/katex/fonts/KaTeX_Unlisted-Regular.woff2",
                      "/vendor/katex/../app.js", "/vendor/katex/%2e%2e/%2e%2e/app.js",
                      "/vendor/katex/fonts/%2e%2e%2fLICENSE", "/vendor/katex/katex.min.js/extra"):
            with self.subTest(route=route):
                self.assertEqual(self.request("GET", route)[0], 404)
        outside = self.base / "outside-font.woff2"
        outside.write_bytes(b"secret external data")
        font = vendor / "fonts" / "KaTeX_Main-Regular.woff2"
        font.unlink()
        font.symlink_to(outside)
        self.assertEqual(self.request("GET", "/vendor/katex/fonts/KaTeX_Main-Regular.woff2")[0], 404)
        external_contrib = self.base / "external-contrib"
        (vendor / "contrib").rename(external_contrib)
        (vendor / "contrib").symlink_to(external_contrib, target_is_directory=True)
        self.assertEqual(self.request("GET", "/vendor/katex/contrib/auto-render.min.js")[0], 404)

    def test_symlink_artifacts_and_directories_cannot_escape_the_run_root(self):
        outside = self.base / "outside"
        outside.mkdir()
        secret = outside / "secret.json"
        secret.write_text('{"secret":"NEVER_EXPOSE_THIS"}')
        (self.data / "linked-directory").symlink_to(outside, target_is_directory=True)
        directory, summary = self.make_run()
        (directory / "report.md").unlink()
        (directory / "report.md").symlink_to(secret)
        (directory / "reviews").symlink_to(outside, target_is_directory=True)
        status, workspace, _ = self.request("GET", "/api/workspace")
        self.assertEqual(status, 200)
        self.assertNotIn("NEVER_EXPOSE_THIS", json.dumps(workspace))
        status, detail, _ = self.request("GET", f"/api/runs/{summary['run_id']}")
        self.assertEqual(status, 200)
        self.assertNotIn("report.md", detail["available_files"])
        self.assertEqual(self.request("GET", f"/api/runs/{summary['run_id']}/files/report.md")[0], 403)
        self.assertEqual(self.request("POST", f"/api/runs/{summary['run_id']}/reviews", {})[0], 403)
        self.assertFalse((outside / "summary_adjudicated.json").exists())

    def test_http_writes_enforce_host_origin_content_type_and_size(self):
        self.assertEqual(self.request("GET", "/api/workspace", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", {}, origin=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", {}, headers={"Origin": "http://attacker.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", {}, headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", {}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/jobs", b"{broken")[0], 400)
        self.assertEqual(self.request("POST", "/api/jobs", b"{" + b"x" * MAX_REQUEST_BYTES + b"}")[0], 413)

    def test_job_validation_never_accepts_paths_keys_or_unbounded_values(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertFalse(self.request("GET", "/api/workspace")[1]["settings"]["api_key_configured"])
            self.assertEqual(self.request("POST", "/api/jobs", {"backend": "openai"})[0], 409)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "TEST_KEY_MUST_NOT_REACH_BROWSER"}):
            workspace = self.request("GET", "/api/workspace")[1]
            self.assertTrue(workspace["settings"]["api_key_configured"])
            self.assertNotIn("TEST_KEY_MUST_NOT_REACH_BROWSER", json.dumps(workspace))
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            credentials.save_openai_key("LOCAL_KEY_MUST_NOT_REACH_BROWSER")
            workspace = self.request("GET", "/api/workspace")[1]
            self.assertTrue(workspace["settings"]["api_key_configured"])
            self.assertNotIn("LOCAL_KEY_MUST_NOT_REACH_BROWSER", json.dumps(workspace))
            self.assertEqual(self.request("GET", "/.secrets/openai.key")[0], 404)
        for payload in ({"output": "/tmp/arbitrary"}, {"api_key": "secret"}, {"rounds": True},
                        {"rounds": 21}, {"max_output_tokens": 1000000}, {"trials": 11},
                        {"kind": "compare", "seed": 2**31 - 1, "trials": 2},
                        {"kind": "shell"}, {"model": "$(touch marker)"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.request("POST", "/api/jobs", payload)[0], 400)
        self.assertEqual(self.store.jobs, {})

    def test_queue_limit_cancellation_and_restart_never_retry_automatically(self):
        jobs = []
        for _ in range(4):
            status, job, _ = self.request("POST", "/api/jobs", {"backend": "demo", "rounds": 1})
            self.assertEqual(status, 202)
            self.assertEqual(job["status"], "queued")
            jobs.append(job)
        self.assertEqual(self.request("POST", "/api/jobs", {})[0], 409)
        status, cancelled, _ = self.request("POST", f"/api/jobs/{jobs[0]['id']}/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.request("POST", "/api/jobs", {"rounds": 1})[0], 202)
        # A second process must not reconcile or cancel this live instance's jobs.
        with self.assertRaises(APIError):
            DashboardStore(self.data, start_worker=False)
        previous = deepcopy(self.store.jobs)
        self.store.close()
        # Restore stale metadata to model an abrupt process exit, after its
        # OS instance lock has been released.
        for old in previous.values():
            write_json(self.data / ".dashboard" / "jobs" / (old["id"] + ".json"), old)
        restarted = DashboardStore(self.data, start_worker=False)
        try:
            self.assertEqual(restarted.jobs[jobs[0]["id"]]["status"], "cancelled")
            for old in jobs[1:]:
                recovered = restarted.jobs[old["id"]]
                self.assertEqual(recovered["status"], "interrupted")
                self.assertIn("No automatic retry", recovered["error"])
        finally:
            restarted.close()

    def test_real_demo_job_runs_existing_cli_and_persists_catalog_links(self):
        self.store.worker.start()
        status, job, _ = self.request("POST", "/api/jobs", {"kind": "run", "mode": "multi", "backend": "demo", "rounds": 1})
        self.assertEqual(status, 202)
        completed = self.wait_job(job["id"])
        self.assertEqual(completed["status"], "completed", completed.get("error"))
        self.assertEqual(len(completed["run_ids"]), 1)
        detail = self.request("GET", f"/api/runs/{completed['run_ids'][0]}")[1]
        self.assertEqual(detail["run"]["job_id"], job["id"])
        self.assertEqual(detail["run"]["status"], "completed")
        self.assertEqual(detail["run"]["model_calls"], 2)
        saved = json.loads((self.data / ".dashboard" / "jobs" / (job["id"] + ".json")).read_text())
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(detail["records"][-1]["score"]["score"], 1)

    def test_real_demo_comparison_is_discovered_recursively(self):
        self.store.worker.start()
        _, job, _ = self.request("POST", "/api/jobs", {"kind": "compare", "backend": "demo", "rounds": 1, "trials": 1})
        completed = self.wait_job(job["id"])
        self.assertEqual(completed["status"], "completed", completed.get("error"))
        self.assertEqual(len(completed["run_ids"]), 2)
        self.assertTrue(completed["comparison_id"].startswith("comparison-"))
        status, comparison, _ = self.request("GET", "/api/comparisons/" + completed["comparison_id"])
        self.assertEqual(status, 200)
        self.assertEqual(set(comparison["run_ids"]), set(completed["run_ids"]))
        self.assertEqual(comparison["status"], "completed")
        self.assertEqual(comparison["aggregate"]["paired"]["matched_pairs"], 1)

    def test_running_cancellation_stops_child_and_keeps_partial_records(self):
        def slow_command(job):
            directory = self.data / "managed" / job["id"] / "partial-cancelled"
            script = (
                "import json,time; from pathlib import Path; "
                f"p=Path({str(directory)!r}); p.mkdir(); "
                "(p/'manifest.json').write_text(json.dumps({'run_id':'partial-cancelled','mode':'multi','backend':'demo'})); "
                "(p/'events.jsonl').write_text(json.dumps({'kind':'model_request','sequence':1})+'\\n'); "
                "time.sleep(30)"
            )
            return [sys.executable, "-c", script]

        with patch.object(self.store, "_command", side_effect=slow_command):
            self.store.worker.start()
            _, job, _ = self.request("POST", "/api/jobs", {"rounds": 1})
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                workspace = self.request("GET", "/api/workspace")[1]
                if any(run["id"] == "partial-cancelled" for run in workspace["runs"]):
                    break
                time.sleep(0.02)
            else:
                self.fail("Test child did not write its partial records.")
            child = self.store.process
            self.assertIsNotNone(child)
            status, cancelled, _ = self.request("POST", f"/api/jobs/{job['id']}/cancel", {})
            self.assertEqual(status, 200)
            self.assertEqual(cancelled["status"], "cancelled")
            child.wait(timeout=5)
        self.assertIsNotNone(child.returncode)
        status, detail, _ = self.request("GET", "/api/runs/partial-cancelled")
        self.assertEqual(status, 200)
        self.assertEqual(detail["run"]["status"], "cancelled")
        self.assertEqual(detail["events"][0]["kind"], "model_request")
        self.assertIsNone(detail["summary"])

    def test_review_uses_exact_proposal_hash_and_preserves_original_summary(self):
        directory, summary = self.make_run()
        original_bytes = (directory / "summary.json").read_bytes()
        record = summary["records"][-1]
        review = {"run_id": summary["run_id"], "iteration": record["iteration"],
                  "proposal_sha256": proposal_digest(record["proposal"]), "reviewer": "Offline test reviewer",
                  "target_preserved": True, "assumptions_preserved": True,
                  "complete_proof_validated": True, "repaired_gap_ids": [],
                  "evidence_type": "human_mathematical_review", "evidence": "Synthetic explicit review for API testing."}
        route = f"/api/runs/{summary['run_id']}/reviews"
        status, error, _ = self.request("POST", route, {**review, "proposal_sha256": "wrong"})
        self.assertEqual(status, 400)
        self.assertIn("proposal_sha256", error["error"])
        self.assertFalse((directory / "reviews").exists())
        status, saved, _ = self.request("POST", route, review)
        self.assertEqual(status, 201)
        self.assertEqual(saved["score"]["score"], 3)
        self.assertEqual(saved["review"], {**review, "assess_proof": True})
        self.assertEqual((directory / "summary.json").read_bytes(), original_bytes)
        detail = self.request("GET", f"/api/runs/{summary['run_id']}")[1]
        self.assertEqual(len(detail["reviews"]), 1)
        self.assertTrue(detail["run"]["reviewed"])
        self.assertFalse(detail["run"]["provisional"])
        self.assertEqual(detail["run"]["final_score"], 3)
        self.assertIn("summary_adjudicated.json", detail["available_files"])

    def test_invalid_saved_review_blocks_new_side_effects(self):
        directory, summary = self.make_run()
        reviews = directory / "reviews"
        reviews.mkdir()
        write_json(reviews / "invalid.json", {"review": {"iteration": 1, "proposal_sha256": "wrong"}})
        before = set(reviews.iterdir())
        record = summary["records"][-1]
        valid = {"run_id": summary["run_id"], "iteration": record["iteration"],
                 "proposal_sha256": proposal_digest(record["proposal"]), "reviewer": "Test reviewer",
                 "target_preserved": True, "assumptions_preserved": True,
                 "complete_proof_validated": False, "repaired_gap_ids": [],
                 "evidence_type": "human_mathematical_review", "evidence": "A test review."}
        status, _, _ = self.request("POST", f"/api/runs/{summary['run_id']}/reviews", valid)
        self.assertEqual(status, 409)
        self.assertEqual(set(reviews.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
