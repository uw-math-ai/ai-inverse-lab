"""UI fixtures for revision comparisons; no model calls or research results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading

from playwright.sync_api import sync_playwright, expect
from workflow.dashboard import DashboardStore, DashboardServer


def main():
    with tempfile.TemporaryDirectory(prefix="revision-ui-") as directory:
        store = DashboardStore(Path(directory), start_worker=False)
        server = DashboardServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proposal = {"claim": "Show $x=y$.", "assumptions": ["Assume $x=y$."],
                        "proof": "Use the hypothesis $x=y$.", "repairs": [], "remaining_gaps": [],
                        "certificate": {"kind": "lean4", "target_id": "test-only", "proof": "intro h\nexact h"},
                        "milestone_attempts": [], "simulator_requests": []}
            proposals = [deepcopy(proposal) for _ in range(5)]
            proposals[1]["certificate"]["proof"] = "intro h\nhave hx := h\nexact hx"
            proposals[1]["repairs"] = [{"gap_id": "G1", "change": "Name an intermediate hypothesis."}]
            proposals[2] = deepcopy(proposals[1])
            proposals[2]["proof"] = "The supplied equality $x=y$ closes the goal directly."
            proposals[3] = deepcopy(proposals[2])
            proposals[4] = deepcopy(proposals[3])
            proposals[4]["proof"] = '<img src="/bad" onerror="window.injected=1"> and $x=y$'
            proposals[4]["certificate"]["proof"] += '\n<img src="/bad">'
            records, actions = [], []
            for i, item in enumerate(proposals, 1):
                action = {"iteration": i, "role": "CA" if i == 1 else "Repair", "call_id": i,
                          "call_status": "completed", "model": "test-model", "output": item,
                          "score": {"score": 1, "provisional": True}, "dimensions": {},
                          "input": [{"role": "user", "content": json.dumps({"feedback": {
                              "skeptic": {"assessment": "Check $x=y$.", "gaps": []},
                              "verification": {"gaps": ["Synthetic checker feedback."]}}})}]}
                actions.append(action)
                records.append({"iteration": i, "proposal": item, "critique": None, "proof_gaps": [],
                                "verification": {"engine": "lean4", "status": "rejected", "gaps": []},
                                "score": {"score": 1, "provisional": True}, "agent_evaluations": [action]})
            run = {"id": "revision-fixture", "run_id": "revision-fixture", "mode": "multi", "backend": "demo",
                   "status": "completed", "model": "test-model", "iteration_count": 5, "final_score": 1,
                   "benchmark_id": "test-only", "provisional": True}
            detail = {"run": run, "manifest": {"mode": "multi", "backend": "demo"}, "records": records,
                      "summary": {"records": records, "agent_evaluations": actions}, "events": [], "available_files": []}
            workspace = store.workspace()
            workspace["runs"] = [run]
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/api/workspace", lambda route: route.fulfill(json=workspace))
                page.route("**/api/runs/revision-fixture", lambda route: route.fulfill(json=detail))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.locator('#view-overview [data-action="open-run"]').first.click()
                page.locator('.iteration-button[data-iteration="2"]').click()
                page.get_by_role("button", name="Compare revisions", exact=True).click()
                expect(page.locator('#run-panel')).to_contain_text("The Lean certificate changed; the written explanation is unchanged.")
                expect(page.locator('[data-revision-field=lean] .diff-removed code')).to_have_text("exact h")
                expect(page.locator('[data-revision-field=lean] .diff-added code')).to_have_text(["have hx := h", "exact hx"])
                expect(page.locator('.revision-columns .katex-mathml math')).to_have_count(2)
                page.locator('.iteration-button[data-iteration="3"]').click()
                expect(page.locator('#run-panel')).to_contain_text("The written explanation changed; the Lean certificate is unchanged.")
                expect(page.locator('#revision-baseline')).to_have_value("2")
                page.locator('#revision-baseline').select_option("1")
                expect(page.locator('#run-panel')).to_contain_text("Both the written explanation and Lean certificate changed.")
                expect(page.locator('#revision-baseline')).to_have_value("1")
                page.locator('.iteration-button[data-iteration="4"]').click()
                expect(page.locator('#run-panel')).to_contain_text("The written explanation and Lean certificate are unchanged.")
                expect(page.locator('#revision-baseline')).to_have_value("3")
                page.locator('.iteration-button[data-iteration="1"]').click()
                expect(page.locator('#run-panel')).to_contain_text("Initial CA proposal")
                page.locator('.iteration-button[data-iteration="5"]').click()
                assert page.locator('#run-panel img').count() == 0
                assert not page.evaluate("Boolean(window.injected)")
                expect(page.locator('.diff-added code')).to_contain_text(["<img", "<img"])
                page.set_viewport_size({"width": 390, "height": 844})
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    print("PASS: exact Lean diff, unchanged/changed prose, selectable baseline, initial attempt, inert HTML, and mobile layout.")


if __name__ == "__main__":
    main()
