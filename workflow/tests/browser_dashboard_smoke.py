"""Optional Chromium regression for the complete local dashboard workflow.

Run from the project root::

    .venv/bin/python -m workflow.tests.browser_dashboard_smoke

Requires Playwright and its Chromium browser. All jobs and synthetic reviews
use a temporary run root. Only preview screenshots remain in the workspace.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading

from workflow.dashboard import DashboardServer, DashboardStore


def check_evaluation_edge_cases(page, base_url, original, expect):
    """Browser-only fixtures; never alter saved research evidence."""
    identifier = original["run_id"]
    template = page.request.get(f"{base_url}/api/runs/{identifier}").json()
    template["records"] = deepcopy(original["records"])
    template["summary"] = deepcopy(original)
    fixture = deepcopy(template)
    page.route(f"**/api/runs/{identifier}", lambda route: route.fulfill(json=fixture))
    page.set_viewport_size({"width": 1440, "height": 1000})

    def open_result():
        if page.locator("#run-dialog").is_visible():
            page.locator("#run-dialog [data-action=close-run]").click()
        page.locator(f'#view-overview [data-action="open-run"][data-id="{identifier}"]').first.click()
        expect(page.locator("#tab-evaluation")).to_have_attribute("aria-selected", "true")

    for verification, expected in [
        ({"engine": "lean4", "status": "unavailable"}, "Unscored"),
        ({"engine": "lean4", "status": "formalization_pending"}, "Unscored"),
        ({"engine": "lean4", "status": "inconclusive"}, "Unscored"),
        ({"engine": "lean4", "status": "timeout", "compiler": {"return_code": -9, "timed_out": True, "output_truncated": False}}, "Unscored"),
        ({"engine": "lean4", "status": "failed", "compiler": {"return_code": 1, "timed_out": False, "output_truncated": True}}, "Unscored"),
        ({"engine": "lean4", "status": "rejected"}, "0 / 1"),
        ({"engine": "lean4", "status": "verified"}, "Unscored"),
        ({"engine": "sympy", "status": "verified"}, "Unscored"),
    ]:
        fixture = deepcopy(template)
        fixture["records"][-1]["verification"] = verification
        open_result()
        expect(page.locator('[data-agent-role="Verification"] .score-pill')).to_have_text(expected)

    # A later partial iteration keeps its own proposal and failed SA call;
    # it must not borrow a previous iteration's critique or Lean success.
    fixture = deepcopy(template)
    fixture["records"] = fixture["records"][:1]
    fixture["summary"]["records"] = fixture["records"]
    proposer, skeptic = fixture["summary"]["agent_evaluations"][-2:]
    proposer["score"] = {"score": 1, "provisional": True, "rationale": "Synthetic partial proposal."}
    proposer["progress"] = {"direction": "unchanged", "score_delta": 0, "improvement_confirmed": False}
    skeptic.update(call_status="provider_error", output=None, dimensions={},
                   score={"score": None, "provisional": False, "evidence_type": "call_status", "rationale": "Synthetic provider failure."})
    open_result()
    expect(page.locator('.iteration-button[data-iteration="2"]')).to_contain_text("Incomplete attempt")
    expect(page.locator('[data-agent-role="Repair"] .score-pill')).to_have_text("1 / 3")
    expect(page.locator('[data-agent-role="SA"] .score-pill')).to_have_text("Unscored")
    expect(page.locator('[data-result-stage="skeptic"]')).not_to_contain_text("Scripted critique")
    expect(page.locator('[data-agent-role="Verification"] .score-pill')).to_have_text("Unscored")
    expect(page.locator("#tab-proof")).to_be_disabled()
    page.locator('.iteration-button[data-iteration="1"]').click()
    page.locator("#tab-proof").click()
    page.locator('.iteration-button[data-iteration="2"]').click()
    expect(page.locator("#tab-evaluation")).to_have_attribute("aria-selected", "true")

    # New output surfaces retain inert model text and mathematical typesetting.
    fixture = deepcopy(template)
    fixture["summary"]["agent_evaluations"][-1]["output"]["assessment"] = (
        r'<img src="/injected" onerror="window.__agentInjected=1"> Check $q(x)=\frac{1-d(x)}{h}-r(x)$.'
    )
    open_result()
    expect(page.locator('[data-result-stage="skeptic"] .katex-mathml math')).to_have_count(1)
    expect(page.locator('[data-result-stage="skeptic"] img')).to_have_count(0)
    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("<img")
    assert not page.evaluate("Boolean(window.__agentInjected)")

    # Historical output remains readable, without manufacturing role scores.
    fixture = deepcopy(template)
    fixture["summary"].pop("agent_evaluations")
    for record in fixture["records"]:
        record.pop("agent_evaluations", None)
    page.locator("#run-dialog [data-action=close-run]").click()
    page.locator(f'#view-overview [data-action="open-run"][data-id="{identifier}"]').first.click()
    page.locator("#tab-evaluation").click()
    expect(page.locator('[data-result-stage="proposal"]')).to_contain_text("Proposed proof")
    expect(page.locator('[data-agent-role="Repair"] .score-pill')).to_have_text("Unscored")
    expect(page.locator('[data-agent-role="SA"] .score-pill')).to_have_text("Unscored")
    page.locator("#run-dialog [data-action=close-run]").click()
    page.unroute(f"**/api/runs/{identifier}")


def main(argv=None):
    try:
        from playwright.sync_api import expect, sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit("Optional browser smoke test requires Playwright and Chromium in the selected environment.") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / ".dashboard")
    args = parser.parse_args(argv)
    expect.set_options(timeout=20000)
    with tempfile.TemporaryDirectory(prefix="math-dashboard-browser-") as temporary:
        root = Path(temporary) / "runs"
        store = DashboardStore(root)
        server = DashboardServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    context = browser.new_context(
                        viewport={"width": 1440, "height": 1000}, device_scale_factor=1,
                        accept_downloads=True,
                    )
                    page = context.new_page()
                    page.set_default_timeout(20000)
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    page.goto(base_url, wait_until="networkidle")
                    expect(page.locator("#view-overview")).to_contain_text("No runs yet")
                    expect(page.locator("#view-overview .kpi-value").first).to_have_text("0")

                    # Launch through the browser; no model API is called.
                    page.locator("#view-overview .section-heading [data-action=open-launch]").click()
                    form = page.locator("#launch-form")
                    expect(form.locator("[name=max_simulator_calls]")).to_have_value("0")
                    expect(form.locator("[name=stop_on_verified]")).to_have_value("false")
                    expect(form.locator("[name=rounds]")).to_have_attribute("max", "10")
                    expect(form.locator("[name=benchmark_id]")).to_have_value("bench-two-measurements")
                    form.locator("[name=benchmark_id]").select_option("q-recovery-one-step")
                    expect(form.locator("[name=benchmark_id] option")).to_have_count(8)
                    form.locator("[name=backend]").select_option("demo")
                    form.locator("[name=mode]").select_option("multi")
                    form.locator("[name=rounds]").fill("2")
                    form.locator("[name=max_simulator_calls]").fill("0")
                    form.get_by_role("button", name="Launch experiment").click()
                    expect(page.locator("#launch-dialog")).not_to_be_visible()
                    expect(page.locator("#view-overview .data-table tbody tr")).to_have_count(1)
                    expect(page.locator("#view-overview .status-completed")).to_have_count(1)
                    page.locator("#view-overview .data-table [data-action=open-run]").first.click()
                    expect(page.locator("#run-dialog .iteration-button")).to_have_count(2)
                    expect(page.locator('#run-dialog .iteration-button[data-iteration="2"]')).to_have_attribute("aria-current", "true")
                    expect(page.locator("#tab-evaluation")).to_have_attribute("aria-selected", "true")
                    expect(page.locator("#run-panel .agent-evaluation-card")).to_have_count(4)
                    assert page.locator("[data-result-stage]").evaluate_all("nodes => nodes.map(node => node.dataset.resultStage)") == ["proposal", "skeptic", "verification"]
                    expect(page.locator('[data-agent-role="Repair"] .score-pill')).to_have_text("3 / 3")
                    expect(page.locator('[data-agent-role="SA"] .score-pill')).to_have_text("1 / 3")
                    expect(page.locator('[data-agent-role="Verification"] .score-pill')).to_have_text("1 / 1")
                    expect(page.locator('[data-result-stage="proposal"]')).to_contain_text("Claim / conjecture")
                    expect(page.locator('[data-result-stage="proposal"]')).to_contain_text("Assumptions")
                    expect(page.locator('[data-result-stage="proposal"]')).to_contain_text("Proposed proof")
                    assert page.locator('[data-result-stage="proposal"] .katex-mathml math').count() > 0
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("Scripted critique.")
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("reported no gaps")
                    expect(page.locator('[data-result-stage="verification"]')).to_contain_text("MathAILab.RecoveryGoal")
                    expect(page.locator(".iteration-score-history tbody tr")).to_have_count(2)
                    expect(page.locator(".agent-history tbody tr")).to_have_count(4)
                    expect(page.locator("#run-panel")).to_contain_text("supported by independent evidence")
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(args.screenshots / "preview-agent-evaluation.png"))
                    page.locator('#run-dialog .iteration-button[data-iteration="1"]').click()
                    expect(page.locator('[data-agent-role="CA"] .score-pill')).to_have_text("1 / 3")
                    expect(page.locator('[data-result-stage="proposal"] .proof-text')).to_contain_text("claim follows by reflexivity")
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("Reflexivity does not use the observation equation")
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("Suggested repair:")
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("awaiting review")
                    expect(page.locator('[data-agent-role="Verification"] .score-pill')).to_have_text("0 / 1")
                    expect(page.locator('[data-result-stage="verification"] .checker-diagnostics')).to_be_visible()
                    expect(page.locator('[data-result-stage="verification"] details').first).not_to_have_attribute("open", "")
                    page.locator('#run-dialog .iteration-button[data-iteration="2"]').click()
                    page.locator("#tab-proof").click()
                    expect(page.locator("#run-panel")).to_contain_text("Proposed proof")
                    expect(page.locator(".attempt-summary")).to_contain_text("Lean verified")

                    page.locator("#tab-verification").click()
                    expect(page.locator("#run-panel")).to_contain_text("Formal proof verified")
                    expect(page.locator("#run-panel")).to_contain_text("MathAILab.RecoveryGoal")
                    expect(page.locator("#run-panel")).to_contain_text("propext")
                    page.locator("#tab-activity").click()
                    expect(page.locator("#run-panel .activity-item").first).to_be_visible()
                    expect(page.locator("#run-panel")).to_contain_text("run finished")
                    with page.expect_download() as download:
                        page.locator('#run-dialog a[href$="/summary.json"]').click()
                    downloaded = Path(download.value.path()).read_bytes()
                    original = json.loads(downloaded)
                    assert original["mode"] == "multi" and len(original["records"]) == 2
                    assert original["backend"] == "demo" and original["model_calls"] == 4
                    assert original["final_score"] == 3
                    assert original["stop_reason"] == "iteration_limit"
                    assert original["has_verified_proof"] is True
                    assert original["stop_on_verified"] is False
                    assert original["records"][0]["verification"]["status"] == "failed"
                    assert original["records"][1]["verification"]["status"] == "verified"
                    assert original["simulator_calls"] == 0
                    with page.expect_download() as training_download:
                        page.locator('#run-dialog a[href$="/agent_evaluations.jsonl"]').click()
                    training = [json.loads(line) for line in Path(training_download.value.path()).read_text().splitlines()]
                    assert len(training) == 4
                    assert [row["score"]["score"] for row in training] == [1, 1, 3, 1]
                    with page.expect_download() as lean_download:
                        page.locator('#run-dialog a[href$="/proof.lean"]').click()
                    assert Path(lean_download.value.path()).read_text() == original["records"][-1]["verification"]["source"]
                    summary_path = next(root.rglob("summary.json"))
                    assert summary_path.read_bytes() == downloaded

                    # Explicitly negative synthetic assessment tests sidecar
                    # persistence without certifying any mathematical claim.
                    page.locator("#tab-review").click()
                    review = page.locator("#review-form")
                    review.locator("[name=reviewer]").fill("Browser smoke test reviewer (synthetic)")
                    review.locator("[name=evidence]").fill(
                        "SYNTHETIC UI TEST ONLY. No mathematical proof has been checked; "
                        "all preservation and completeness attestations remain false."
                    )
                    for checkbox in review.locator("input[type=checkbox]").all():
                        expect(checkbox).not_to_be_checked()
                    review.locator("[name=assess_proof]").check()
                    review.get_by_role("button", name="Save independent review").click()
                    expect(page.locator(".attempt-summary")).to_contain_text("Independently reviewed")
                    expect(page.locator(".attempt-summary .score-pill")).to_have_text("1 / 3")
                    assert summary_path.read_bytes() == downloaded, "Review rewrote the automatic summary"
                    saved_reviews = list((summary_path.parent / "reviews").glob("*.json"))
                    assert len(saved_reviews) == 1
                    saved = json.loads(saved_reviews[0].read_text())
                    assert saved["review"]["complete_proof_validated"] is False
                    assert saved["review"]["target_preserved"] is False
                    assert saved["review"]["assumptions_preserved"] is False
                    assert saved["review"]["repaired_gap_ids"] == []
                    assert saved["score"]["score"] == 1
                    page.locator("#tab-evaluation").click()
                    expect(page.locator('[data-agent-role="Repair"] .score-pill')).to_have_text("1 / 3")
                    expect(page.locator("#run-panel")).not_to_contain_text("supported by independent evidence")
                    # Independent assessment of the skeptic is a distinct judgment.
                    page.locator('#run-dialog .iteration-button[data-iteration="1"]').click()
                    page.locator("#tab-review").click()
                    review = page.locator("#review-form")
                    review.locator("[name=reviewer]").fill("Synthetic skeptic reviewer")
                    review.locator("[name=evidence]").fill("SYNTHETIC UI TEST ONLY: verify skeptic adjudication is bound to the exact critique.")
                    review.locator('[data-skeptic-gap="field_algebra"]').select_option("confirmed")
                    expect(review.locator("[name=assess_proof]")).not_to_be_checked()
                    review.get_by_role("button", name="Save independent review").click()
                    expect(page.locator("#tab-proof")).to_have_attribute("aria-selected", "true")
                    page.locator("#tab-evaluation").click()
                    expect(page.locator('[data-agent-role="SA"] .score-pill')).to_have_text("3 / 3")
                    assert summary_path.read_bytes() == downloaded
                    page.locator("#run-dialog [data-action=close-run]").click()

                    # Filtering remains usable while the workspace polls.
                    page.locator('.nav-link[data-view="runs"]').click()
                    expect(page.locator("#run-table-body tr")).to_have_count(1)
                    page.locator("#run-search").fill("no-such-research-run")
                    expect(page.locator("#run-table-body tr")).to_have_count(0)
                    expect(page.locator("#run-empty")).to_contain_text("No matching runs")
                    page.locator("#run-search").fill("")
                    page.locator('[data-filter="mode"]').select_option("single")
                    expect(page.locator("#run-table-body tr")).to_have_count(0)
                    page.locator('[data-filter="mode"]').select_option("")
                    expect(page.locator("#run-table-body tr")).to_have_count(1)

                    # Blurring the search field must not replace a button between
                    # pointer-down and click, swallowing the Inspect action.
                    page.locator("#run-search").fill("q-recovery")
                    page.locator("#run-table-body").get_by_role("button", name="Inspect", exact=True).click()
                    expect(page.locator("#run-dialog")).to_be_visible()
                    page.locator("#run-dialog [data-action=close-run]").click()
                    page.locator("#run-search").fill("")

                    page.locator('.nav-link[data-view="comparisons"]').click()
                    page.locator("#view-comparisons .section-heading [data-action=open-launch]").click()
                    form = page.locator("#launch-form")
                    expect(form.locator("[name=kind]")).to_have_value("compare")
                    form.locator("[name=backend]").select_option("demo")
                    form.locator("[name=rounds]").fill("2")
                    form.locator("[name=trials]").fill("1")
                    form.locator("[name=max_simulator_calls]").fill("0")
                    form.get_by_role("button", name="Launch experiment").click()
                    expect(page.locator("#launch-dialog")).not_to_be_visible()
                    expect(page.locator("#view-overview .data-table tbody tr")).to_have_count(3)
                    expect(page.locator("#view-overview .status-completed")).to_have_count(3)
                    page.locator('.nav-link[data-view="comparisons"]').click()
                    expect(page.locator("#view-comparisons [data-action=open-comparison]")).to_have_count(1)
                    page.locator("#view-comparisons [data-action=open-comparison]").click()
                    expect(page.locator(".comparison-summary")).to_contain_text("1 matched pairs")
                    expect(page.locator("#view-comparisons")).to_contain_text("Agent contributions")
                    workspace = page.request.get(base_url + "/api/workspace").json()
                    assert len(workspace["runs"]) == 3
                    assert all(job["status"] == "completed" for job in workspace["jobs"])
                    comparison = workspace["comparisons"][0]["aggregate"]
                    assert comparison["paired"]["matched_pairs"] == 1
                    assert comparison["paired"]["excluded_pairs"] == []
                    assert all(mode["independently_verified_runs"] == 1 for mode in comparison["by_mode"].values())

                    # A baseline has no skeptic call or invented critique score.
                    single = next(run for run in workspace["runs"] if run["mode"] == "single")
                    page.locator('.nav-link[data-view="overview"]').click()
                    page.locator(f'#view-overview [data-action="open-run"][data-id="{single["id"]}"]').first.click()
                    expect(page.locator('[data-agent-role="SA"] .score-pill')).to_have_text("Not run")
                    expect(page.locator('[data-result-stage="skeptic"]')).to_contain_text("single-agent baseline does not run a skeptic")
                    expect(page.locator('[data-agent-role="Verification"] .score-pill')).to_have_text("1 / 1")
                    page.locator("#run-dialog [data-action=close-run]").click()

                    page.locator('.nav-link[data-view="benchmark"]').click()
                    expect(page.locator("#view-benchmark")).to_contain_text("Permitted assumptions")
                    expect(page.locator("#view-benchmark")).to_contain_text("Allowed prior results")
                    expect(page.locator("#view-benchmark")).to_contain_text("MathAILab.MeasurementsGoal")
                    expect(page.locator("#view-benchmark")).to_contain_text("Formalization milestones")
                    assert page.locator("#view-benchmark pre .katex").count() == 0
                    page.locator('.nav-link[data-view="overview"]').click()
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(args.screenshots / "preview-desktop.png"), full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    overflow = page.evaluate("({viewport: innerWidth, document: document.documentElement.scrollWidth})")
                    assert overflow["document"] <= overflow["viewport"], f"Mobile overview has horizontal overflow: {overflow}"
                    first_cell = page.locator("#view-overview .data-table tbody td").first.bounding_box()
                    assert first_cell and first_cell["width"] >= 100, "Mobile run names collapse into unreadable narrow columns"
                    page.screenshot(path=str(args.screenshots / "preview-mobile.png"), full_page=True)
                    page.locator("#view-overview .data-table [data-action=open-run]").first.click()
                    expect(page.locator("#run-dialog")).to_be_visible()
                    expect(page.locator("#run-panel")).to_contain_text("Progress across iterations")
                    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), "Mobile detail has horizontal overflow"
                    page.screenshot(path=str(args.screenshots / "preview-mobile-detail.png"))
                    # The actual stage cards must fit, even when a long formula
                    # or code block scrolls inside a card.
                    for stage in page.locator("[data-result-stage]").all():
                        stage.scroll_into_view_if_needed()
                        bounds = stage.bounding_box()
                        assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
                    page.locator("#run-dialog [data-action=close-run]").click()
                    check_evaluation_edge_cases(page, base_url, original, expect)
                    assert not errors, errors
                finally:
                    browser.close()
            print("PASS: temporary demo launch, final proof, verification, activity, export, negative synthetic review, immutable summary, filtering, paired comparison, benchmark, mobile layout; no browser exceptions.")
            print(f"Preview screenshots: {args.screenshots}")
        finally:
            server.shutdown()
            server.server_close()  # Also stops the worker and its child process.
            thread.join(timeout=5)
            assert not thread.is_alive(), "Dashboard server did not stop"
            assert not store.worker.is_alive(), "Dashboard job worker did not stop"
            assert store.process is None or store.process.poll() is not None, "Dashboard child did not stop"


if __name__ == "__main__":
    main()
