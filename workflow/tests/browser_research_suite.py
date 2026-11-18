"""Real browser coverage of campaign controls and persistent evidence views."""
import json
from pathlib import Path
import tempfile
import threading
import time

from playwright.sync_api import sync_playwright, expect
from workflow.dashboard import DashboardServer, DashboardStore


def main():
    with tempfile.TemporaryDirectory(prefix="math-suite-browser-") as directory:
        store = DashboardStore(Path(directory))
        server = DashboardServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.locator('.nav-link[data-view="benchmark"]').click()
                expect(page.locator("#view-benchmark")).to_contain_text("Frozen research reference")
                page.locator("#benchmark-selector").select_option("bench-product-repair")
                expect(page.locator("#view-benchmark")).to_contain_text("MathAILab.ProductGoal")
                page.get_by_role("button", name="Run benchmark suite", exact=True).click()
                form = page.locator("#launch-form")
                expect(form.locator('[name="rounds"]')).to_have_value("10")
                expect(form.locator('[name="budget_policy"]')).to_have_value("iterations")
                expect(page.locator("#launch-target-field")).to_be_hidden()
                form.locator('[name="trials"]').fill("1")
                form.locator('[name="ca_model"]').fill("demo-author")
                form.locator('[name="sa_model"]').fill("demo-skeptic")
                form.locator('[name="verifier_model"]').fill("demo-verifier")
                expect(page.locator("#launch-estimate")).to_contain_text("10 planned runs")
                expect(page.locator("#launch-estimate")).to_contain_text("300 model calls")
                form.get_by_role("button", name="Save settings", exact=True).click()
                form.locator('[name="ca_model"]').fill("changed")
                form.get_by_role("button", name="Load saved settings", exact=True).click()
                expect(form.locator('[name="ca_model"]')).to_have_value("demo-author")
                form.get_by_role("button", name="Launch experiment", exact=True).click()
                expect(page.locator("#launch-dialog")).not_to_be_visible()
                deadline = time.monotonic() + 90
                while True:
                    workspace = page.request.get(f"http://127.0.0.1:{server.server_port}/api/workspace").json()
                    if workspace["jobs"] and workspace["jobs"][0]["status"] in ("completed", "failed"):
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError("Campaign did not finish within 90 seconds")
                    time.sleep(0.2)
                assert workspace["jobs"][0]["status"] == "completed", workspace["jobs"]
                assert len(workspace["runs"]) == 10
                assert all(run["benchmark_id"].startswith("bench-") for run in workspace["runs"])
                page.locator('[data-action="refresh"]').click()
                page.locator('.nav-link[data-view="comparisons"]').click()
                page.locator('[data-action="open-comparison"]').click()
                expect(page.locator("#view-comparisons")).to_contain_text("5 matched pairs")
                expect(page.locator("#view-comparisons")).to_contain_text("Results by benchmark")
                page.locator('#view-comparisons [data-action="open-run"]').first.click()
                expect(page.locator('[data-agent-role="VA"]')).to_contain_text("Unscored / abstained")
                page.locator('#tab-ledger').click()
                expect(page.locator('#run-panel')).to_contain_text("formal target verified")
                expect(page.locator('#run-panel')).to_contain_text("claimed repairs")
                screenshots = Path("workflow/runs/ui-validation")
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots / "suite-ledger-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                page.screenshot(path=str(screenshots / "suite-ledger-mobile.png"))
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    print("PASS: suite launch, role settings, saved settings, paired results, adviser predictions, ledger, and mobile layout.")


if __name__ == "__main__":
    main()
