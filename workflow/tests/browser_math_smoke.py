"""Optional Chromium checks for safe, local mathematical typesetting.

Run: .venv/bin/python -m workflow.tests.browser_math_smoke

Uses an empty temporary dashboard and browser-owned API/DOM fixtures. No
research runs or reviews are written. Only ignored QA screenshots persist.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import threading

from workflow.dashboard import DashboardServer, DashboardStore
from workflow.evaluation import proposal_digest


def _run_fixture():
    records = []
    for iteration in (1, 2):
        proposal = {
            "claim": r"For $u\in\mathbb{R}^X$, investigate $Wf=u$ under the stated assumptions.",
            "assumptions": [r"The masses satisfy $\mu_x>0$ and the graph has $n=|X|$ vertices."],
            "proof": (
                f"Synthetic typesetting fixture, attempt {iteration}; this is not a verified proof.\n\n"
                + rf"Define $q_{iteration}(x)=x^{iteration}$ and consider \(\alpha,\beta\in\mathbb{{R}}\)."
                + "\n\n" + r"$$\langle u,v\rangle_\mu=\sum_{x\in X}\mu_x u(x)v(x).$$"
                + "\n\n" + r"The finite matrix is \[A=\begin{pmatrix}1 & -1\\-1 & 1\end{pmatrix}.\]"
                + "\n\n" + r"Use $\frac{a+b}{c+d}$ only after checking $c+d\ne0$."
            ),
            "repairs": ([{"gap_id": "gap-algebra", "change": r"Check $c+d\ne0$ before division."}] if iteration == 2 else []),
            "remaining_gaps": [], "certificate": None, "simulator_requests": [],
        }
        records.append({
            "run_id": "math-renderer-fixture", "iteration": iteration,
            "proposal": proposal, "proposal_sha256": proposal_digest(proposal),
            "proof_gaps": ([{"gap_id": "gap-algebra", "scope": "general_proof", "severity": "major",
                            "description": r"Explain why $c+d\ne0$.",
                            "suggested_repair": r"State the restriction $c\ne-d$."}] if iteration == 1 else []),
            "verification": {"status": "inconclusive", "raw_formula": "$literal_json$"},
            "simulator_results": [],
            "score": {"score": 1, "provisional": True, "rationale": "Synthetic display fixture; no proof is certified.", "confirmed_repairs": []},
        })
    run = {
        "id": "math-renderer-fixture", "run_id": "math-renderer-fixture", "mode": "multi",
        "backend": "demo", "model": "scripted-fixture", "status": "completed",
        "created_at": "2026-11-08T12:00:00Z", "iteration_count": 2, "model_calls": 4,
        "max_model_calls": 4, "usage": {}, "final_score": 1, "provisional": True,
        "reviewed": False, "benchmark_id": "arxiv-2603.20710-lemma-2.3", "comparison_id": None,
    }
    return {"run": run, "manifest": {"mode": "multi", "backend": "demo"},
            "summary": {**run, "records": records}, "records": records,
            "events": [{"kind": "fixture_event", "iteration": 1, "message": "$literal_event$"}],
            "reviews": [], "available_files": []}


def main(argv=None):
    try:
        from playwright.sync_api import expect, sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit("Optional math browser test requires Playwright and Chromium.") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / ".dashboard")
    args = parser.parse_args(argv)
    expect.set_options(timeout=15000)
    with tempfile.TemporaryDirectory(prefix="math-renderer-browser-") as temporary:
        root = Path(temporary) / "runs"
        store = DashboardStore(root, start_worker=False)
        server = DashboardServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                    page = context.new_page()
                    page.set_default_timeout(15000)
                    errors, requests = [], []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on("request", lambda request: requests.append(request.url))
                    base_url = f"http://127.0.0.1:{server.server_port}"
                    fixture = _run_fixture()
                    workspace = deepcopy(store.workspace())
                    workspace["runs"] = [fixture["run"]]
                    workspace["benchmark"]["statement"] = r"Investigate $Wf=g$ for every $g\in\ell^2(X)$."
                    workspace["benchmark"]["assumptions"] = [r"Assume $\mu_x>0$ and $T\ge |X|$."]
                    workspace["benchmark"]["definitions"] = {"inner_product": r"$\langle u,v\rangle_\mu=\sum_x\mu_xu(x)v(x)$."}
                    page.route("**/api/workspace", lambda route: route.fulfill(json=workspace))
                    page.route("**/api/runs/math-renderer-fixture", lambda route: route.fulfill(json=fixture))
                    page.goto(base_url, wait_until="networkidle")
                    page.wait_for_function("typeof window.MathRenderer?.render === 'function'")

                    # Real app navigation re-creates marked text after every tab,
                    # iteration, and benchmark switch; math must render each time.
                    page.locator("#view-overview .data-table [data-action=open-run]").first.click()
                    expect(page.locator("#run-panel .katex").first).to_be_visible()
                    assert page.locator("#run-panel .katex-mathml math").count() >= 4
                    assert any("q_2" in text for text in page.locator("#run-panel annotation").all_text_contents())
                    page.locator('#run-dialog [data-iteration="1"]').click()
                    assert any("q_1" in text for text in page.locator("#run-panel annotation").all_text_contents())
                    page.locator("#tab-verification").click()
                    expect(page.locator("#run-panel pre").first).to_contain_text("$literal_json$")
                    assert page.locator("#run-panel pre .katex").count() == 0
                    page.locator("#tab-activity").click()
                    expect(page.locator("#run-panel pre").first).to_contain_text("$literal_event$")
                    assert page.locator("#run-panel pre .katex").count() == 0
                    page.locator("#tab-proof").click()
                    expect(page.locator("#run-panel .katex").first).to_be_visible()
                    page.locator('#run-dialog [data-iteration="2"]').click()
                    args.screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(args.screenshots / "preview-math.png"))
                    page.locator("#run-dialog [data-action=close-run]").click()
                    page.locator('.nav-link[data-view="benchmark"]').click()
                    expect(page.locator("#view-benchmark .katex:visible").first).to_be_visible()
                    assert page.locator("#view-benchmark .katex-mathml math").count() >= 4

                    samples = [
                        {"id": "inline-dollar", "text": r"The potential $q(x)=\alpha x^2$ is real."},
                        {"id": "display-dollar", "text": r"$$\sum_{k=0}^{n}\frac{1}{k+1}$$"},
                        {"id": "multiline-dollar", "text": "$$\n" + r"\frac{a}{b}" + "\n$$"},
                        {"id": "inline-paren", "text": r"Energy \(\lVert u\rVert_\mu^2\) is nonnegative."},
                        {"id": "display-bracket", "text": r"\[A=\begin{pmatrix}1&2\\3&4\end{pmatrix}\]"},
                        {"id": "multiline-bracket", "text": "\\[\n" + r"\frac{c}{d}" + "\n\\]"},
                        {"id": "align-environment", "text": r"\begin{align}Wf&=U^f(T,\cdot)\\q(x)&\ge0\end{align}"},
                        {"id": "bare-latex", "text": r"q(x)=\frac{a}{b}"},
                        {"id": "latex-fence", "text": "```latex\n" + r"\frac{a}{b}" + "\n```"},
                        {"id": "ordinary-fence", "text": "```python\nproof = '$x^2$'\n```"},
                        {"id": "inline-code", "text": "Keep `$x^2$` literal and render $y$."},
                        {"id": "html-inert", "text": r'<img src="https://invalid.example/image" onerror="window.__mathInjected=1"> and $x^2$'},
                        {"id": "unsafe-link", "text": r"$\href{javascript:window.__mathInjected=1}{unsafe}$"},
                        {"id": "unsafe-image", "text": r"$\includegraphics{https://invalid.example/image.png}$"},
                        {"id": "malformed", "text": r"Malformed $\frac{1}$ must stay visible."},
                        {"id": "unbalanced", "text": r"Unbalanced $\frac{1}{$ must stay visible."},
                        {"id": "long-display", "text": "$$" + "+".join(r"\frac{x_{" + str(index) + r"}^2+\alpha}{1+\beta}" for index in range(20)) + "$$"},
                    ]
                    samples += [
                        {"id": "bare-inline-algebra", "text": "Assume a*x+b=a*y+b and a≠0. Then x=y."},
                        {"id": "bare-inline-fraction", "text": r"Consider q(x)=\frac{1-d}{h} in the argument."},
                        {"id": "preserved-explicit-math", "text": r"Keep $a*x+b=a*y+b$ and `a*x+b` intact."},
                    ]
                    for sample in samples[-3:]:
                        sample["text"] = page.evaluate("text => window.formatResearchText(text)", sample["text"])
                    literal = r'{"proof":"$x^2$ and \\(y\\)"}'
                    page.evaluate("""({samples, literal}) => {
                        const board = document.createElement('section');
                        board.id = 'math-test-board'; board.className = 'card card-body';
                        for (const sample of samples) {
                            const node = document.createElement('div');
                            node.id = sample.id; node.dataset.math = ''; node.textContent = sample.text;
                            board.append(node);
                        }
                        for (const tag of ['code', 'pre', 'textarea']) {
                            const node = document.createElement(tag);
                            node.id = 'literal-' + tag; node.dataset.math = '';
                            node.textContent = literal;
                            board.append(node);
                        }
                        const raw = document.createElement('div'); raw.className = 'code-block';
                        raw.id = 'literal-json'; raw.dataset.math = ''; raw.textContent = '{"formula":"$x$"}';
                        board.append(raw);
                        document.querySelector('#main-content').append(board);
                        window.MathRenderer.render(board);
                    }""", {"samples": samples, "literal": literal})
                    for identifier in ("inline-dollar", "display-dollar", "multiline-dollar", "inline-paren", "display-bracket", "multiline-bracket", "align-environment", "bare-latex", "latex-fence"):
                        node = page.locator("#" + identifier)
                        expect(node.locator(".katex-mathml math")).to_have_count(1)
                        expect(node).to_have_attribute("data-math-rendered", "true")
                        visible = node.inner_text()
                        assert not any(delimiter in visible for delimiter in ("$$", r"\(", r"\)", r"\[", r"\]", r"\begin{align}")), visible
                    assert page.locator("#bare-inline-algebra .katex-mathml math").count() == 3
                    assert page.locator("#bare-inline-fraction .katex-mathml math").count() == 1
                    assert page.locator("#preserved-explicit-math .katex-mathml math").count() == 1
                    expect(page.locator("#preserved-explicit-math code")).to_have_text("a*x+b")
                    assert page.locator("#display-dollar .katex-display").count() == 1
                    assert page.locator("#display-bracket .katex-display").count() == 1
                    for identifier in ("multiline-dollar", "multiline-bracket", "bare-latex", "latex-fence"):
                        assert page.locator(f"#{identifier} .katex-display").count() == 1
                    expect(page.locator("#ordinary-fence pre")).to_have_text("proof = '$x^2$'\n")
                    assert page.locator("#ordinary-fence .katex").count() == 0
                    expect(page.locator("#inline-code code")).to_have_text("$x^2$")
                    assert page.locator("#inline-code code .katex").count() == 0
                    assert page.locator("#inline-code .katex-mathml math").count() == 1
                    assert page.locator("#html-inert img").count() == 0
                    expect(page.locator("#html-inert")).to_contain_text("<img")
                    assert page.locator("#html-inert .katex").count() == 1
                    assert page.locator("#unsafe-link a, #unsafe-image img").count() == 0
                    assert not page.evaluate("Boolean(window.__mathInjected)")
                    expect(page.locator("#malformed")).to_contain_text(r"\frac")
                    expect(page.locator("#malformed")).to_be_visible()
                    expect(page.locator("#unbalanced")).to_contain_text(r"\frac")
                    for identifier in ("literal-code", "literal-pre", "literal-textarea", "literal-json"):
                        node = page.locator("#" + identifier)
                        assert node.locator(".katex").count() == 0
                        assert "$" in node.text_content()
                        if identifier != "literal-json":
                            assert node.text_content() == literal
                    before = page.locator("#math-test-board").inner_html()
                    page.evaluate("window.MathRenderer.render(document.querySelector('#math-test-board'))")
                    assert page.locator("#math-test-board").inner_html() == before, "Repeated rendering changed existing markup"
                    direct = page.evaluate("""() => {
                        const node = document.createElement('div');
                        node.id = 'direct-math-root'; node.dataset.math = ''; node.textContent = '$z^2$';
                        document.querySelector('#math-test-board').append(node);
                        const renderer = window.renderMathInElement;
                        try {
                            window.renderMathInElement = undefined;
                            window.MathRenderer.render(node);
                            if (node.textContent !== '$z^2$' || node.dataset.mathUnavailable !== 'true') return false;
                        } finally { window.renderMathInElement = renderer; }
                        window.MathRenderer.render(node);
                        return node.dataset.mathRendered === 'true' && !node.dataset.mathUnavailable;
                    }""")
                    assert direct, "A directly marked root did not recover from an unavailable renderer"
                    expect(page.locator("#direct-math-root .katex-mathml math")).to_have_count(1)

                    # Wide equations scroll inside their own display container.
                    page.set_viewport_size({"width": 390, "height": 844})
                    dimensions = page.evaluate("({viewport: innerWidth, document: document.documentElement.scrollWidth})")
                    assert dimensions["document"] <= dimensions["viewport"], dimensions
                    scrollable = page.locator("#long-display .katex-display").evaluate("""element => {
                        for (let node = element; node && node.id !== 'math-test-board'; node = node.parentElement) {
                            if (['auto', 'scroll'].includes(getComputedStyle(node).overflowX) && node.scrollWidth > node.clientWidth) {
                                node.scrollLeft = 50; return node.scrollLeft > 0;
                            }
                        }
                        return false;
                    }""")
                    assert scrollable, "Long display math has no local horizontal scroll container"
                    page.locator("#long-display").scroll_into_view_if_needed()
                    page.screenshot(path=str(args.screenshots / "preview-math-mobile.png"))
                    assert not errors, errors
                    external = [url for url in requests if not url.startswith(base_url + "/")]
                    assert not external, f"Typesetting attempted external requests: {external}"
                    assert not list(root.rglob("summary.json")), "Browser fixtures unexpectedly created a research run"
                finally:
                    browser.close()
            print("PASS: real MathML, supported delimiters/environments, dynamic proof/benchmark rendering, idempotence, safe HTML/KaTeX commands, literal code/JSON, visible malformed fallback, local mobile scrolling; no external requests or browser errors.")
            print(f"Math previews: {args.screenshots}")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive(), "Temporary math dashboard did not stop"


if __name__ == "__main__":
    main()
