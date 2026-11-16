/* Local research dashboard. All dynamic content is escaped; no model text is executable. */
(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const h = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
  const mathText = value => h(window.formatResearchText ? window.formatResearchText(String(value ?? "")) : value);
  const number = value => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "—";
  const date = value => value && !Number.isNaN(Date.parse(value)) ? new Date(value).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}) : "—";
  const human = value => String(value ?? "unknown").replaceAll("_", " ");
  const active = job => ["queued", "running", "cancelling"].includes(job.status);
  const state = {view: "overview", workspace: null, detail: null, iteration: null, tab: "proof", comparison: null, filters: {search: "", mode: "", status: "", backend: ""}, drafts: {}, refreshing: false};
  const views = {overview: "Overview", runs: "Research runs", comparisons: "Comparisons", benchmark: "Benchmark"};
  const heading = (title, description, action = "Launch run", kind = "run") => `<div class="section-heading"><div><span class="eyebrow">Research workspace</span><h1>${h(title)}</h1><p>${h(description)}</p></div>${button(action, "open-launch", `data-kind="${kind}"`, "primary")}</div>`;
  const badge = (value, extra = "") => `<span class="status-pill status-${h(String(value || "unknown").replace(/[^a-z_-]/g, ""))} ${extra}">${h(human(value))}</span>`;
  const button = (label, action, attrs = "", className = "secondary") => `<button type="button" class="button ${className}" data-action="${action}" ${attrs}>${h(label)}</button>`;
  const empty = (title, message) => `<div class="empty-state"><h3>${h(title)}</h3><p>${h(message)}</p></div>`;
  const code = value => `<pre class="code-block">${h(typeof value === "string" ? value : JSON.stringify(value, null, 2))}</pre>`;
  const list = values => `<ul class="prose">${(values || []).map(value => `<li data-math>${mathText(value)}</li>`).join("")}</ul>`;
  const runID = run => String(run.id || run.run_id || "");
  const runLabel = run => state.workspace?.benchmarks?.find(item => item.id === run.benchmark_id)?.short_title || (run.benchmark_id?.includes("lemma-2.3") ? "Lemma 2.3" : "Research proof");
  const scoreLabel = score => score?.evidence_type === "lean_kernel" ? "Lean verified" : score?.provisional === false ? "Independently reviewed" : "Provisional assessment";

  async function api(path, options = {}) {
    const response = await fetch(path, {headers: {"Content-Type": "application/json"}, cache: "no-store", ...options});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    return body;
  }
  function announce(message, error = false) {
    const node = $("#workspace-status");
    node.textContent = message;
    node.classList.toggle("error-state", error);
  }
  function showView(view) {
    if (!(view in views)) return;
    state.view = view;
    for (const key of Object.keys(views)) $("#view-" + key).hidden = key !== view;
    document.querySelectorAll(".nav-link[data-view]").forEach(node => {
      const selected = node.dataset.view === view;
      node.classList.toggle("active", selected);
      if (selected) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current");
    });
    $("#page-title").textContent = views[view];
    renderView();
  }
  function renderView() {
    if (!state.workspace) return;
    ({overview: renderOverview, runs: renderRuns, comparisons: renderComparisons, benchmark: renderBenchmark})[state.view]();
    window.MathRenderer?.render($("#view-" + state.view));
  }
  function runRows(runs) {
    return runs.map(run => `<tr>
      <td><button type="button" class="text-button" data-action="open-run" data-id="${h(runID(run))}">${h(runLabel(run))} · ${h(runID(run).slice(-10))}</button><small>${h(run.backend === "demo" ? "Scripted responses" : run.model)}</small></td>
      <td>${badge(run.mode)} ${badge(run.backend)}</td><td>${badge(run.status)}</td>
      <td>${number(run.iteration_count ?? run.records?.length)}</td><td>${number(run.model_calls)}<span class="muted"> / ${number(run.max_model_calls)}</span></td>
      <td><span class="score-pill">${number(run.final_score)} / 3</span><small>${run.score_evidence === "lean_kernel" ? "Lean verified" : run.provisional === false ? "Independently reviewed" : "Provisional"}</small></td>
      <td>${h(date(run.created_at))}</td><td>${button("Inspect", "open-run", `data-id="${h(runID(run))}"`, "secondary small")}</td>
    </tr>`).join("");
  }
  function runTable(runs, bodyID = "") {
    return `<div class="table-wrap"><table class="data-table"><thead><tr><th>Run</th><th>Workflow</th><th>Status</th><th>Attempts</th><th>Calls</th><th>Final score</th><th>Started</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody ${bodyID ? `id="${bodyID}"` : ""}>${runRows(runs)}</tbody></table></div>`;
  }
  function renderOverview() {
    const {runs = [], jobs = [], comparisons = []} = state.workspace;
    const live = jobs.filter(active);
    const failedJobs = jobs.filter(job => ["failed", "interrupted"].includes(job.status)).slice(0, 3);
    const metrics = [["Total runs", runs.length, "Every attempt stays on record"], ["Active jobs", live.length, "Queued and running experiments"], ["Reviewed runs", runs.filter(run => run.reviewed).length, "Human assessment recorded"], ["Comparisons", comparisons.length, "Paired workflow experiments"]];
    const benchmark = state.workspace.benchmark || {};
    $("#view-overview").innerHTML = `${heading("Workflow overview", "Develop recovery arguments, check them in Lean, and repair unfinished proofs.")}
      <div class="kpi-grid">${metrics.map(([label, value, note]) => `<article class="kpi-card"><div class="kpi-label">${h(label)}</div><div class="kpi-value">${number(value)}</div><small>${h(note)}</small></article>`).join("")}</div>
      <div class="content-grid"><div><section class="card"><div class="card-header"><div><p class="eyebrow">Execution queue</p><h2 class="card-title">Active experiments</h2></div>${badge(live.length ? "running" : "idle")}</div>
      ${live.length ? `<div class="card-body job-list">${live.map(job => `<article class="job-card"><div><div class="tag-list">${badge(job.kind)} ${badge(job.status)} ${badge(job.config?.backend || "demo")}</div><h3>${h(job.kind === "compare" ? "Paired workflow comparison" : "Mathematical research run")}</h3><p class="muted">${h(job.id)} · ${h(date(job.created_at))}</p><p>${number(job.run_ids?.length || 0)} run records · ${h(job.config?.backend === "demo" ? "Scripted responses" : job.config?.model)}</p></div>${button("Cancel job", "cancel-job", `data-id="${h(job.id)}"`, "danger")}</article>`).join("")}</div>` : empty("Ready for the next question", "Launch a demo to inspect the workflow, or use a configured OpenAI model for a research run.")}${failedJobs.length ? `<div class="card-body">${failedJobs.map(job => `<div class="notice notice-warning"><strong>${h(human(job.status))}: ${h(job.kind)}</strong><p>${h(typeof job.error === "string" ? job.error : job.error?.message || "The job stopped. Available run records are preserved.")}</p><small>${h(job.id)}</small></div>`).join("")}</div>` : ""}</section>
      <section class="card"><div class="card-header"><div><p class="eyebrow">Research history</p><h2 class="card-title">Recent runs</h2></div>${button("View all runs", "view-runs")}</div>${runs.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Run</th><th>Workflow</th><th>Status</th><th>Score</th><th></th></tr></thead><tbody>${runs.slice(0, 5).map(run => `<tr><td><button data-action="open-run" data-id="${h(runID(run))}" type="button">${h(runLabel(run))} · ${h(runID(run).slice(-6))}</button><small>${h(date(run.created_at))}</small></td><td>${badge(run.mode)} ${badge(run.backend)}</td><td>${badge(run.status)}</td><td><span class="score-pill">${number(run.final_score)} / 3</span><small>${run.score_evidence === "lean_kernel" ? "Lean verified" : run.provisional === false ? "Reviewed" : "Provisional"}</small></td><td>${button("→", "open-run", `data-id="${h(runID(run))}" aria-label="Inspect run ${h(runID(run))}"`, "icon-button")}</td></tr>`).join("")}</tbody></table></div>` : empty("No runs yet", "Your claims, proof attempts, critiques, and verification evidence will appear here.")}</section></div>
      <aside><section class="card"><div class="card-header"><div><p class="eyebrow">Current benchmark</p><h2 class="card-title">${h(benchmark.short_title || "Formal benchmark")}</h2></div>${badge("proof blind")}</div><div class="card-body"><p class="benchmark-statement">${h(benchmark.title)}</p><div class="prose" data-math>${mathText(benchmark.statement)}</div><div class="notice">Both workflows receive the same fixed statement. Reference Lean proofs remain hidden.</div>${button("Explore benchmark →", "view-benchmark")}</div></section>
      <section class="card"><div class="card-header"><h2 class="card-title">What counts as progress?</h2></div><div class="card-body"><p class="muted">Lean must close the fixed goal with no proof holes or new axioms. A verified formal target earns score 3; independent review assesses its mathematical interpretation.</p><div class="notice">Compiler errors feed the next repair. Runs stop when Lean verifies the target.</div>${button("Compare workflows →", "view-comparisons")}</div></section></aside></div>
      <section class="card"><div class="card-body"><div class="tag-list">${badge(state.workspace.settings?.lean?.available ? "Lean ready" : "Lean unavailable")}${badge("formal mathematics")}</div><h2 class="card-title">Prove the reconstruction first</h2><p>Current path: fixed mathematical statement → proof attempt → skeptic → Lean → repair. Simulator calls default to zero. The separate ML inverse-model path remains future work.</p><p class="muted">${h(state.workspace.settings?.lean?.version || state.workspace.settings?.lean?.reason || "Checking the local Lean compiler…")}</p><div class="notice notice-warning">The initial formal targets recover q on observed vertices and prove uniqueness with full observation. Hidden-vertex recovery, weakest conditions, and stability remain research goals. Demo proofs are scripted.</div></div></section>`;
  }
  function renderRuns() {
    const root = $("#view-runs");
    if (!$("#run-search", root)) {
      root.innerHTML = `${heading("Runs & reviews", "Inspect every attempt, follow the evidence, and record an independent review.")}<section class="card"><div class="card-header"><div><p class="eyebrow">Full experiment history</p><h2 class="card-title">All research runs</h2></div><span id="run-count" class="muted"></span></div><div class="toolbar">
        <label class="field search-field"><span class="sr-only">Search runs</span><input id="run-search" data-filter="search" type="search" placeholder="Search by run, model, or benchmark" value="${h(state.filters.search)}"></label>
        ${[["mode", "All workflows", ["single", "multi"]], ["status", "All statuses", ["queued", "running", "completed", "failed", "cancelled", "interrupted"]], ["backend", "All backends", ["demo", "openai"]]].map(([key, label, values]) => `<label class="field"><span class="sr-only">${h(label)}</span><select data-filter="${key}"><option value="">${h(label)}</option>${values.map(value => `<option value="${value}" ${state.filters[key] === value ? "selected" : ""}>${h(human(value))}</option>`).join("")}</select></label>`).join("")}
        </div>${runTable([], "run-table-body")}<div id="run-empty"></div></section>`;
    }
    const runs = (state.workspace.runs || []).filter(run => {
      const query = [runID(run), run.model, run.benchmark_id, run.mode].join(" ").toLowerCase();
      return query.includes(state.filters.search.toLowerCase()) && ["mode", "status", "backend"].every(key => !state.filters[key] || run[key] === state.filters[key]);
    });
    $("#run-table-body").innerHTML = runRows(runs);
    $("#run-count").textContent = `${runs.length} ${runs.length === 1 ? "run" : "runs"}`;
    $("#run-empty").innerHTML = runs.length ? "" : empty("No matching runs", "Try different filters or launch your first experiment.");
  }
  function comparisonResults(detail) {
    const aggregate = detail.aggregate || {};
    const modes = Object.entries(aggregate.by_mode || {});
    const pairs = aggregate.paired || {};
    return `<div class="card-header"><div><p class="eyebrow">Paired evaluation</p><h2 class="card-title">${h(detail.id)}</h2></div>${badge("descriptive")}</div>
      <div class="notice notice-warning">Equal call and output caps do not guarantee equal token cost. Failed runs stay in the denominator. Demo results measure software behavior, not mathematical ability.</div>
      ${modes.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Workflow</th><th>Runs</th><th>Failed</th><th>Verified proofs</th><th>Provisional</th><th>Mean score</th><th>Model calls</th><th>Reported tokens</th></tr></thead><tbody>${modes.map(([mode, stats]) => `<tr><td>${badge(mode)}</td><td>${number(stats.runs)}</td><td>${number(stats.failed_runs)}</td><td>${number(stats.independently_verified_runs)}</td><td>${number(stats.provisional_runs)}</td><td>${typeof stats.mean_final_score === "number" ? stats.mean_final_score.toFixed(2) : "—"}</td><td>${number(stats.model_calls?.total)}</td><td>${number(stats.tokens?.total_tokens?.total)}</td></tr>`).join("")}</tbody></table></div>` : empty("Results are being collected", "The comparison appears when its paired runs finish.")}
      <div class="comparison-summary"><p><strong>${number(pairs.matched_pairs)} matched pairs</strong> · ${number(pairs.unpaired_runs ?? 0)} unpaired runs · ${number(pairs.excluded_pairs?.length || 0)} excluded pairs</p><p class="muted">${h(aggregate.interpretation || "Results are descriptive. Scores are ordinal and remain provisional without independent review.")}</p></div>
      ${aggregate.by_agent?.length ? `<h3>Agent contributions</h3><p class="muted">Grouped by workflow, backend, and model. Score distributions describe role-specific outcomes.</p><div class="table-wrap"><table class="data-table"><thead><tr><th>Configuration</th><th>Role</th><th>Calls</th><th>Failed</th><th>Score counts</th><th>Confirmed improvements</th><th>Reported tokens</th></tr></thead><tbody>${aggregate.by_agent.flatMap(group => Object.entries(group.roles).map(([role, metrics]) => `<tr><td>${h(group.mode)} · ${h(group.backend)}<small>${h(group.model)}</small></td><td>${h(role)}</td><td>${number(metrics.actions)}</td><td>${number(metrics.failed_calls)}</td><td>${h(Object.entries(metrics.score_counts).map(([score, count]) => `${score}: ${count}`).join(", "))}<small>${number(metrics.provisional_actions)} provisional · ${number(metrics.unscored_actions)} unscored</small></td><td>${number(metrics.confirmed_improvements)}</td><td>${number(metrics.reported_tokens)}<small>${number(metrics.missing_token_records)} missing records</small></td></tr>`)).join("")}</tbody></table></div>` : ""}
      ${pairs.excluded_pairs?.length ? `<details><summary>Why pairs were excluded</summary>${list(pairs.excluded_pairs.map(pair => `${pair.pair_id}: ${pair.reason}`))}</details>` : ""}
      <div class="tag-list">${(detail.run_ids || []).map(id => button("Inspect " + id, "open-run", `data-id="${h(id)}"`)).join("")}</div>`;
  }
  function renderComparisons() {
    const comparisons = state.workspace.comparisons || [];
    $("#view-comparisons").innerHTML = `${heading("Compare workflows", "Measure single-agent and multi-agent outcomes under matched conditions.", "New comparison", "compare")}<section class="card"><div class="card-header"><div><p class="eyebrow">Single agent versus feedback</p><h2 class="card-title">Paired experiments</h2></div></div>
      ${comparisons.length ? `<div class="table-wrap"><table class="data-table"><thead><tr><th>Comparison</th><th>Status</th><th>Created</th><th></th></tr></thead><tbody>${comparisons.map(item => `<tr><td>${h(item.id)}</td><td>${badge(item.status || (state.workspace.jobs || []).find(job => job.comparison_id === item.id)?.status || "unknown")}</td><td>${h(date(item.created_at))}</td><td>${button("View results", "open-comparison", `data-id="${h(item.id)}"`)}</td></tr>`).join("")}</tbody></table></div>` : empty("Compare the research process", "Run single-agent and multi-agent trials with the same model, benchmark, seed, and configured budgets.")}</section>
      ${state.comparison ? `<section class="card">${comparisonResults(state.comparison)}</section>` : ""}`;
  }
  function officialLink(value) {
    try { const url = new URL(value); return url.protocol === "https:" && ["arxiv.org", "www.arxiv.org"].includes(url.hostname) ? url.href : null; } catch { return null; }
  }
  function renderBenchmark() {
    const benchmark = state.workspace.benchmark || {};
    const source = benchmark.source || {};
    const url = officialLink(source.url);
    $("#view-benchmark").innerHTML = `${heading("The benchmark", "Exactly the statement, assumptions, and prior results available to the agents.")}<section class="card benchmark-intro"><div class="card-body"><p class="eyebrow">Proof-blind benchmark · ${h(source.statement_locator || benchmark.id)}</p><h2>${h(benchmark.title || "Research benchmark")}</h2><div class="prose" data-math>${mathText(benchmark.statement)}</div><div class="tag-list">${badge("proofs withheld")}${badge("same packet in both arms")}</div></div></section>
      <div class="content-grid"><section class="card"><div class="card-body"><h2 class="card-title">Permitted assumptions</h2>${list(benchmark.assumptions)}</div></section><section class="card"><div class="card-body"><h2 class="card-title">Allowed prior results</h2>${list(benchmark.allowed_results)}</div></section></div>
      <section class="card"><div class="card-body"><h2 class="card-title">Definitions available to agents</h2><dl class="definition-list">${Object.entries(benchmark.definitions || {}).map(([key, value]) => `<dt>${h(human(key))}</dt><dd data-math>${mathText(value)}</dd>`).join("")}</dl></div></section>
      <div class="content-grid"><section class="card"><div class="card-body"><h2 class="card-title">Scope and limits</h2>${list(benchmark.scope_notes)}<p class="muted">${h(benchmark.visibility?.limitation)}</p></div></section><section class="card"><div class="card-body"><h2 class="card-title">Source and scope</h2><p>${h(source.title)}</p><p class="muted">${h(source.arxiv_id || source.implementation || "")} · ${h(source.version)} · checked ${h(source.checked_on)}</p>${url ? `<a class="button secondary" href="${h(url)}" target="_blank" rel="noopener noreferrer">Read source on arXiv ↗</a>` : ""}<p class="muted">Reference proofs are withheld from agent prompts. Lean checks the formal statement; interpreting it as an observation model still requires mathematical review.</p></div></section></div>
      <section class="card"><div class="card-body"><h2 class="card-title">Fixed Lean target</h2><p>${h(benchmark.verification_contract?.target || "Formalization pending")}</p><p class="muted">${h(benchmark.verification_contract?.instructions)}</p>${code(benchmark.verification_contract?.specification || "No reviewed formal specification yet.")}</div></section>
      <section class="card"><div class="card-body"><h2 class="card-title">Formalization milestones</h2><ul class="prose">${(state.workspace.benchmarks || []).map(item => `<li>${h(item.title)} — ${h(human(item.formalization))}</li>`).join("")}</ul></div></section>`;
  }
  function bound(name, fallback) {
    const value = state.workspace?.settings?.limits?.[name];
    return Array.isArray(value) ? value : value && typeof value === "object" ? [value.min, value.max] : fallback;
  }
  function numberField(name, label, fallback, range, note = "") {
    const defaults = state.workspace?.settings?.defaults || {};
    const [min, max] = bound(name, range);
    return `<label class="field"><span>${h(label)}</span><input name="${name}" type="number" min="${h(min)}" max="${h(max)}" step="1" value="${h(defaults[name] ?? fallback)}" required>${note ? `<small>${h(note)}</small>` : ""}</label>`;
  }
  function openLaunch(kind = "run") {
    const settings = state.workspace?.settings || {};
    const defaults = settings.defaults || {};
    const dialog = $("#launch-dialog");
    dialog.innerHTML = `<form id="launch-form"><header class="dialog-header"><div><p class="eyebrow">New experiment</p><h2 id="launch-title">Configure a research run</h2></div>${button("×", "close-launch", 'aria-label="Close launch dialog"', "icon-button")}</header><div class="dialog-body"><div class="form-grid">
      <label class="field"><span>Experiment</span><select name="kind"><option value="run" ${kind === "run" ? "selected" : ""}>One research run</option><option value="compare" ${kind === "compare" ? "selected" : ""}>Paired comparison</option></select></label>
      <label class="field" id="launch-mode-field"><span>Workflow</span><select name="mode"><option value="multi">Multi-agent feedback loop</option><option value="single">Single-agent baseline</option></select></label>
      <label class="field"><span>Mathematical target</span><select name="benchmark_id">${(state.workspace.benchmarks || [state.workspace.benchmark]).filter(Boolean).map(item => `<option value="${h(item.id)}" ${item.id === defaults.benchmark_id ? "selected" : ""}>${h(item.short_title || item.title)}${item.formalization === "formalization_pending" ? " · formalization pending" : ""}</option>`).join("")}</select></label>
      <label class="field"><span>Backend</span><select name="backend"><option value="demo">Scripted demo · no API usage</option><option value="openai" ${settings.api_key_configured ? "" : "disabled"} ${defaults.backend === "openai" && settings.api_key_configured ? "selected" : ""}>OpenAI model</option></select></label>
      <label class="field"><span>Model</span><input name="model" value="${h(defaults.model || "gpt-4o-mini")}" maxlength="200" required></label>
      ${numberField("rounds", "Feedback rounds", 3, [1, 20], "Each arm allows twice this many model calls.")}${numberField("max_output_tokens", "Output tokens per call", 2400, [256, 16000])}${numberField("max_simulator_calls", "Simulator call cap", 0, [0, 80], "Zero keeps this run focused on Lean proofs.")}${numberField("seed", "Random seed", 2026, [0, 2147483647])}
      <div id="launch-trials-field">${numberField("trials", "Paired trials", 3, [1, 10], "Each trial launches one run of each workflow.")}</div></div>
      <p class="notice">Lean checks the selected fixed target after each attempt. Verification stops the loop early; comparisons use equal caps, not necessarily equal consumed calls.</p><p class="notice notice-warning" id="launch-backend-note"></p><p class="muted">${settings.api_key_configured ? "An OpenAI API key is configured on this server." : "No OpenAI API key is configured. Scripted demo runs are available."}</p><p class="error-state" id="launch-error" role="alert"></p></div><footer class="dialog-footer">${button("Cancel", "close-launch")}<button class="button primary" type="submit">Launch experiment</button></footer></form>`;
    dialog.setAttribute("aria-labelledby", "launch-title");
    updateLaunchFields();
    dialog.showModal();
  }
  function updateLaunchFields() {
    const form = $("#launch-form");
    if (!form) return;
    const paired = form.elements.kind.value === "compare";
    $("#launch-mode-field").hidden = paired;
    $("#launch-trials-field").hidden = !paired;
    $("#launch-title").textContent = paired ? "Configure a paired comparison" : "Configure a research run";
    $("#launch-backend-note").textContent = form.elements.backend.value === "demo" ? "Demo runs are scripted software checks. They do not measure a model’s mathematical ability." : "This experiment uses your configured OpenAI account. Actual input and output tokens are recorded; API usage may incur cost.";
  }
  async function launch(form) {
    const values = Object.fromEntries(new FormData(form));
    for (const key of ["rounds", "max_output_tokens", "max_simulator_calls", "seed", "trials"]) values[key] = Number(values[key]);
    const submit = $("button[type=submit]", form);
    submit.disabled = true;
    form.setAttribute("aria-busy", "true");
    try {
      await api("/api/jobs", {method: "POST", body: JSON.stringify(values)});
      $("#launch-dialog").close();
      await refresh();
      showView("overview");
      announce("Experiment queued. Progress and audit records will appear below.");
    } catch (error) { $("#launch-error").textContent = error.message; }
    finally { submit.disabled = false; form.removeAttribute("aria-busy"); }
  }
  function currentRecord() { return state.detail?.records?.find(record => record.iteration === state.iteration); }
  function currentScore(record) { return record.score; } // Server revalidates reviews; saved sidecar scores are not authoritative.
  function evaluationRows() {
    return state.detail?.summary?.agent_evaluations || (state.detail?.records || []).flatMap(record => record.agent_evaluations || []);
  }
  function iterationNumbers() {
    return [...new Set([...(state.detail?.records || []), ...evaluationRows()].map(item => item.iteration))].sort((a, b) => a - b);
  }
  async function openRun(id) {
    const dialog = $("#run-dialog");
    state.detail = null;
    dialog.innerHTML = `<header class="dialog-header"><h2>Loading research run</h2>${button("×", "close-run", 'aria-label="Close run details"', "icon-button")}</header><div class="dialog-body loading-state" role="status">Loading proof attempts and activity…</div>`;
    if (!dialog.open) dialog.showModal();
    try {
      state.detail = await api("/api/runs/" + encodeURIComponent(id));
      state.detail.id = id;
      state.iteration = iterationNumbers().at(-1) ?? null;
      state.tab = state.detail.summary?.agent_evaluations?.length || state.detail.records?.some(record => record.agent_evaluations?.length) ? "evaluation" : state.iteration === null ? "activity" : "proof";
      renderRunDetail();
      $("#run-dialog [data-action=close-run]")?.focus();
    } catch (error) { dialog.innerHTML = `<header class="dialog-header"><h2>Unable to load run</h2>${button("×", "close-run", 'aria-label="Close run details"', "icon-button")}</header><div class="dialog-body error-state" role="alert">${h(error.message)}</div>`; }
  }
  function proofPanel(record) {
    const proposal = record.proposal || {};
    return `<section class="proof-section"><h3>Claim</h3><div class="prose" data-math>${mathText(proposal.claim)}</div><h3>Assumptions</h3>${list(proposal.assumptions)}<h3>Proposed proof</h3><div class="prose proof-text" data-math>${mathText(proposal.proof || "No proof attempt was supplied.")}</div>${proposal.certificate?.kind === "lean4" ? `<h3>Lean proof body</h3>${code(proposal.certificate.proof)}` : ""}</section>
      <section class="proof-section"><h3>Identified gaps</h3>${record.proof_gaps?.length ? record.proof_gaps.map(gap => `<article class="gap-card"><div class="tag-list">${badge(gap.severity || "unassessed")}${badge(gap.scope || "general_proof")}</div><h4>${h(gap.gap_id)}</h4><div data-math>${mathText(gap.description)}</div>${gap.suggested_repair ? `<div class="muted" data-math>Suggested repair: ${mathText(gap.suggested_repair)}</div>` : ""}</article>`).join("") : '<p class="muted">No gaps recorded. This does not establish correctness.</p>'}<h3>Claimed repairs</h3>${proposal.repairs?.length ? `<ul class="prose">${proposal.repairs.map(repair => `<li><strong>${h(repair.gap_id)}:</strong> <span data-math>${mathText(repair.change)}</span></li>`).join("")}</ul>` : '<p class="muted">No repairs claimed in this attempt.</p>'}${proposal.remaining_gaps?.length ? `<h3>Remaining gaps</h3>${list(proposal.remaining_gaps)}` : ""}</section>`;
  }
  const agentScoreText = row => typeof row?.score?.score === "number" ? `${row.score.score} / 3` : "Unscored";
  const agentEvidence = row => !row?.score ? "No assessment recorded" : row.score.provisional ? "Provisional" : human(row.score.evidence_type || "independent_review");
  const yesNo = value => value === true ? "Yes" : value === false ? "No" : "Not assessed";
  function agentProgress(row) {
    const progress = row?.progress || {};
    return `${h(human(progress.direction || "unassessed"))}${typeof progress.score_delta === "number" ? ` (${progress.score_delta > 0 ? "+" : ""}${h(progress.score_delta)})` : ""}${progress.improvement_confirmed ? " · supported by independent evidence" : ""}`;
  }
  function agentAssessment(row) {
    if (!row) return '<p class="muted">No role-specific assessment was recorded. No score has been inferred from another agent.</p>';
    const dimensions = row.dimensions || {};
    const checks = row.role === "SA" ? [["Target check correct", dimensions.target_check_correct], ["Assumption check correct", dimensions.assumptions_check_correct]] : [["Target preserved", dimensions.target_preserved], ["Assumptions preserved", dimensions.assumptions_preserved]];
    return `<section class="agent-assessment"><h4>Score and evidence</h4><p class="muted">${h(agentEvidence(row))} · ${h(human(row.call_status))}</p><p data-math>${mathText(row.score?.rationale || "No score rationale recorded.")}</p>
      <dl class="definition-list">${checks.map(([label, value]) => `<dt>${h(label)}</dt><dd>${yesNo(value)}</dd>`).join("")}<dt>Progress</dt><dd>${agentProgress(row)}</dd></dl>
      ${row.role === "SA" && dimensions.reported_gap_count !== undefined ? `<p>${number(dimensions.reported_gap_count)} reported gaps · ${number(dimensions.confirmed_gap_ids?.length)} confirmed · ${number(dimensions.dismissed_gap_ids?.length)} dismissed · ${number(dimensions.unreviewed_gap_ids?.length)} awaiting review.</p>${typeof dimensions.reviewed_precision === "number" ? `<p>Precision on reviewed findings: ${(dimensions.reviewed_precision * 100).toFixed(0)}%</p>` : ""}` : ""}
      ${dimensions.repair_links?.length ? `<h4>Repair evidence</h4><ul>${dimensions.repair_links.map(repair => `<li><strong>${h(repair.gap_id)}</strong>: ${repair.independently_confirmed ? "independently confirmed" : repair.known_prior_gap ? "claimed; awaiting gap-level review" : "unknown prior gap"}</li>`).join("")}</ul>` : ""}
      <details><summary>Call metrics and assessment details</summary><dl class="definition-list"><dt>Tokens</dt><dd>${number(row.usage?.input_tokens)} input / ${number(row.usage?.output_tokens)} output</dd><dt>Response time</dt><dd>${typeof row.elapsed_seconds === "number" ? row.elapsed_seconds.toFixed(2) + " s" : "Unknown"}</dd></dl>${code(dimensions)}</details></section>`;
  }
  function stageHeader(step, role, title, score, scoreLabel, row) {
    return `<header class="agent-result-header"><span class="stage-number" aria-label="Step ${step}">${step}</span><div class="agent-result-title"><p class="eyebrow">${h(role)}${row ? ` · call ${number(row.call_id)}` : ""}</p><h3>${h(title)}</h3></div><div class="agent-score"><span class="score-pill" aria-label="${h(scoreLabel)}: ${h(score)}">${h(score)}</span><small>${h(scoreLabel)}</small></div></header>`;
  }
  function proposalResult(row, record) {
    const proposal = row ? row.output : record?.proposal;
    const role = row?.role || (record?.iteration > 1 ? "Repair" : "CA");
    return `<article class="agent-evaluation-card" data-agent-role="${h(role)}" data-result-stage="proposal">${stageHeader(1, role, role === "Repair" ? "Revised claim and proof" : "Conjecturing agent", agentScoreText(row), "Proof score", row)}
      <div class="agent-output">${role === "Repair" ? '<p class="muted">This iteration revises the earlier CA proposal using feedback.</p>' : ""}${proposal ? `<h4>Claim / conjecture</h4><div class="prose" data-math>${mathText(proposal.claim)}</div><h4>Assumptions</h4>${list(proposal.assumptions)}<h4>Proposed proof</h4><div class="prose proof-text" data-math>${mathText(proposal.proof || "No prose proof was supplied.")}</div>
      ${proposal.certificate?.kind === "lean4" ? `<details class="agent-proof-source"><summary>Submitted Lean proof</summary>${code(proposal.certificate.proof)}</details>` : ""}
      ${proposal.repairs?.length ? `<h4>Claimed repairs</h4><ul class="prose">${proposal.repairs.map(repair => `<li><strong>${h(repair.gap_id)}:</strong> <span data-math>${mathText(repair.change)}</span></li>`).join("")}</ul>` : ""}${proposal.remaining_gaps?.length ? `<h4>Remaining gaps reported by the author</h4>${list(proposal.remaining_gaps)}` : ""}` : '<p class="muted">No structured proposal was returned. See Activity for the call record.</p>'}</div>${agentAssessment(row)}</article>`;
  }
  function skepticResult(row, record) {
    const critique = row ? row.output : record?.critique;
    const single = (state.detail.run?.mode || state.detail.manifest?.mode) === "single";
    const dimensions = row?.dimensions || {};
    const judgment = gap => dimensions.confirmed_gap_ids?.includes(gap.gap_id) ? "confirmed" : dimensions.dismissed_gap_ids?.includes(gap.gap_id) ? "dismissed" : "awaiting review";
    return `<article class="agent-evaluation-card" data-agent-role="SA" data-result-stage="skeptic">${stageHeader(2, "SA", "Skeptical agent", !row && !critique ? "Not run" : agentScoreText(row), "Critique score", row)}
      <div class="agent-output">${critique ? `<h4>Assessment</h4><div class="prose proof-text" data-math>${mathText(critique.assessment)}</div><dl class="definition-list"><dt>SA says target preserved</dt><dd>${yesNo(critique.target_preserved)}</dd><dt>SA says assumptions preserved</dt><dd>${yesNo(critique.assumptions_preserved)}</dd></dl><h4>Reported gaps and suggested repairs</h4>
      ${critique.gaps?.length ? critique.gaps.map(gap => `<article class="gap-card"><div class="tag-list">${badge(gap.severity)}${badge(judgment(gap))}</div><h5>${h(gap.gap_id)}</h5><div class="prose" data-math>${mathText(gap.description)}</div>${gap.suggested_repair ? `<p data-math><strong>Suggested repair:</strong> ${mathText(gap.suggested_repair)}</p>` : ""}</article>`).join("") : '<p class="muted">The skeptic reported no gaps. This assessment still requires independent review.</p>'}` : `<p class="muted">${single ? "This single-agent baseline does not run a skeptic." : row ? "No structured critique was returned. See the call status and Activity for details." : "No skeptic action is recorded for this iteration."}</p>`}</div>${row || critique ? agentAssessment(row) : ""}</article>`;
  }
  function verificationOutcome(result) {
    // Display the recorded checking outcome only. Never award a proof score or
    // create a fresh verification receipt from saved JSON.
    if (result?.engine !== "lean4") return {score: null, reason: result?.engine ? "Historical checks are not Lean verification." : "No Lean check is recorded for this iteration."};
    const compiler = result.compiler;
    const compiled = compiler?.return_code === 0 && compiler.timed_out === false && compiler.output_truncated === false;
    const checks = ["lean_kernel", "axiom_audit"].every(kind => result.checks?.some(check => check.kind === kind && check.passed === true));
    if (result.status === "verified" && result.independently_verified === true && result.certificate_valid === true && compiled && checks) {
      return {score: 1, reason: "The recorded Lean check passed the fixed formal target and its axiom audit."};
    }
    if (result.status === "rejected" || (result.status === "failed" && Number.isInteger(compiler?.return_code) && compiler.return_code !== 0 && compiler.timed_out === false && compiler.output_truncated === false)) {
      return {score: 0, reason: "The submitted proof was rejected or failed to check. This does not refute the conjecture."};
    }
    return {score: null, reason: "No conclusive Lean result is recorded. Missing proofs, unavailable tools, timeouts and incomplete records stay unscored."};
  }
  function verifierResult(record) {
    const result = record?.verification;
    const outcome = verificationOutcome(result);
    const score = outcome.score === null ? "Unscored" : `${outcome.score} / 1`;
    return `<article class="agent-evaluation-card" data-agent-role="Verification" data-result-stage="verification">${stageHeader(3, "Lean", "Verification", score, "Verification pass score")}
      <div class="agent-output"><div class="tag-list">${badge(result?.status || "not_run")}</div><h4>Verification result</h4><div class="prose" data-math>${mathText(result?.summary || "No verification result is available.")}</div>${list(result?.gaps)}
      ${result?.engine === "lean4" ? `<dl class="definition-list"><dt>Formal target</dt><dd>${h(result.formal_target || "Formalization pending")}</dd><dt>Kernel check</dt><dd>${result.checks?.some(check => check.kind === "lean_kernel" && check.passed === true) ? "Passed" : "No pass recorded"}</dd><dt>Axiom audit</dt><dd>${result.checks?.some(check => check.kind === "axiom_audit" && check.passed === true) ? "Passed" : "No pass recorded"}${Array.isArray(result.axioms) ? ` · ${h(result.axioms.join(", ") || "No axioms")}` : ""}</dd></dl><p class="muted">${h(result.scope)}</p>` : ""}
      ${result?.compiler ? `<details ${outcome.score === 0 ? "open" : ""}><summary>Compiler diagnostics</summary>${code(result.compiler.output || "No diagnostics.")}</details>` : ""}
      ${result?.source ? `<details><summary>Exact Lean source checked</summary>${code(result.source)}</details>` : ""}</div>
      <section class="agent-assessment"><h4>Score and evidence</h4><p>${h(outcome.reason)}</p><p class="muted">Binary verification outcome: 1 = passed; 0 = rejected or failed. This measures the submitted proof’s check result, not a model’s ability. It is separate from the CA / SA scores and does not change their assessments.</p></section></article>`;
  }
  function evaluationPanel(record) {
    const rows = evaluationRows();
    const selected = rows.filter(row => row.iteration === state.iteration);
    const proposer = selected.find(row => row.role === "CA" || row.role === "Repair");
    const skeptic = selected.find(row => row.role === "SA");
    const rubric = state.detail.manifest?.agent_evaluation_rubric;
    if (!record && !selected.length) return empty("No agent results yet", "Results and individual scores appear as agent actions are recorded.");
    return `<div class="agent-results-intro"><h3>Iteration ${number(state.iteration)} · agent results</h3><p class="muted">Read the proposal, the skeptic’s response, then the Lean result. Each stage shows its own output, score and evidence.</p></div>
      ${(state.detail.run?.backend || state.detail.manifest?.backend) === "demo" ? '<p class="notice notice-warning">Scripted demo: these scores validate the workflow, not an AI model’s research ability.</p>' : ""}
      <ol class="agent-results"><li>${proposalResult(proposer, record)}</li><li>${skepticResult(skeptic, record)}</li><li>${verifierResult(record)}</li></ol>
      <h3>Progress across iterations</h3><p class="muted">CA / Repair: proof score out of 3. SA: critique score out of 3. Lean: recorded pass score out of 1. These measure different outcomes and are not added together.</p>
      <div class="table-wrap"><table class="data-table iteration-score-history"><thead><tr><th>Iteration</th><th>CA / Repair</th><th>SA</th><th>Lean</th></tr></thead><tbody>${iterationNumbers().map(iteration => {
        const actions = rows.filter(row => row.iteration === iteration);
        const saved = (state.detail.records || []).find(item => item.iteration === iteration);
        const proof = actions.find(row => row.role === "CA" || row.role === "Repair");
        const sa = actions.find(row => row.role === "SA");
        const outcome = verificationOutcome(saved?.verification);
        return `<tr><td>${button(`Iteration ${iteration}`, "select-iteration", `data-iteration="${h(iteration)}"`, "text-button")}</td><td>${h(proof ? proof.role + " · " + agentScoreText(proof) : saved?.proposal ? "Unscored" : "Not run")}</td><td>${h(sa ? agentScoreText(sa) : saved?.critique ? "Unscored" : "Not run")}</td><td>${outcome.score === null ? "Unscored" : outcome.score + " / 1"}</td></tr>`;
      }).join("")}</tbody></table></div>
      ${rows.length ? `<details class="evaluation-rubric"><summary>Agent progress and call history</summary><p class="muted">Repair is compared with the previous proof attempt. The skeptic is compared with its previous critique. A higher provisional score alone does not establish mathematical progress.</p><div class="table-wrap"><table class="data-table agent-history"><thead><tr><th>Iteration</th><th>Role</th><th>Score</th><th>Change</th><th>Evidence</th><th>Tokens</th></tr></thead><tbody>${rows.map(row => `<tr><td>${number(row.iteration)}</td><td>${h(row.role)}</td><td>${h(agentScoreText(row))}</td><td>${agentProgress(row)}</td><td>${h(agentEvidence(row))}</td><td>${number(row.usage?.total_tokens)}</td></tr>`).join("")}</tbody></table></div></details>` : ""}
      ${rubric ? `<details class="evaluation-rubric"><summary>Agent scoring rubric · ${h(rubric.version)}</summary><div class="table-wrap"><table class="data-table"><thead><tr><th>Score</th><th>Conjecture / repair</th><th>Skeptic</th></tr></thead><tbody>${[0, 1, 2, 3].map(value => `<tr><td>${value}</td><td>${h(rubric.CA_and_Repair[value])}</td><td>${h(rubric.SA[value])}</td></tr>`).join("")}</tbody></table></div><p class="muted">${h(rubric.limits)}</p></details>` : ""}
      <p class="muted">Download agent_evaluations.jsonl for the original agent training records. Reviewed scores appear here and in summary_adjudicated.json; export-training --reviewed creates a separate reviewed dataset. Lean’s display score summarizes the verification record saved in iterations.jsonl.</p>`;
  }
  function verificationPanel(record) {
    const result = record.verification || {};
    if (result.engine !== "lean4") return `<div class="notice notice-warning"><strong>Archived finite checks.</strong> This historical run used sample calculations, which did not verify the theorem. New runs use Lean.</div><div class="prose" data-math>${mathText(result.summary || "")}</div>${code(result)}`;
    return `<div class="notice ${result.status === "verified" ? "" : "notice-warning"}"><strong>${result.status === "verified" ? "Lean-verified formal target" : "Lean verification"}</strong><p>${h(result.scope)}</p></div><div class="tag-list">${badge(result.status)}${badge("Lean 4")}</div><h3>Verification result</h3><div class="prose" data-math>${mathText(result.summary || "")}</div>${list(result.gaps)}
      <dl class="definition-list"><dt>Fixed target</dt><dd>${h(result.formal_target || "Formalization pending")}</dd><dt>Toolchain</dt><dd>${h(result.toolchain?.version || result.toolchain?.reason || "No compiler check ran")}</dd><dt>Axiom audit</dt><dd>${h(result.axioms ? result.axioms.join(", ") || "No axioms" : "Not completed")}</dd><dt>Source SHA-256</dt><dd class="hash-text">${h(result.source_sha256 || "—")}</dd></dl>
      ${result.compiler ? `<h3>Compiler diagnostics</h3>${code(result.compiler.output || "No diagnostics.")}` : ""}
      ${result.source ? `<details><summary>Exact Lean source checked</summary>${code(result.source)}</details>` : ""}
      <details><summary>Complete verification record</summary>${code(result)}</details>
      <h3>Simulator observations</h3>${record.simulator_results?.length ? code(record.simulator_results) : '<p class="muted">No simulator probes. This attempt follows the formal mathematics path.</p>'}`;
  }
  function activityPanel() {
    const events = state.detail.events || [];
    return `<p class="muted">Most recent ${events.length} events. Download the event log for the full history.</p><div class="activity-list">${events.map(event => `<details class="activity-item"><summary><span>${h(human(event.kind))}</span><span class="muted">${h(event.stage || "")}${event.iteration ? " · attempt " + h(event.iteration) : ""} · ${h(date(event.timestamp))}</span></summary>${code(event)}</details>`).join("") || empty("No events yet", "Activity appears as the job starts.")}</div>`;
  }
  function reviewKey() { return `${state.detail.id}:${state.iteration}`; }
  function saveDraft() {
    const form = $("#review-form");
    if (!form || !state.detail) return;
    const fields = new FormData(form);
    state.drafts[reviewKey()] = {reviewer: fields.get("reviewer"), evidence: fields.get("evidence"), target_preserved: fields.has("target_preserved"), assumptions_preserved: fields.has("assumptions_preserved"), complete_proof_validated: fields.has("complete_proof_validated"), repaired_gap_ids: fields.getAll("repaired_gap_ids")};
    state.drafts[reviewKey()].assess_proof = fields.has("assess_proof");
    const record = currentRecord();
    if (record?.critique_sha256) state.drafts[reviewKey()].critique_review = {
      critique_sha256: record.critique_sha256,
      confirmed_gap_ids: [...form.querySelectorAll("[data-skeptic-gap]")].filter(input => input.value === "confirmed").map(input => input.dataset.skepticGap),
      dismissed_gap_ids: [...form.querySelectorAll("[data-skeptic-gap]")].filter(input => input.value === "dismissed").map(input => input.dataset.skepticGap),
      no_gap_assessment_validated: fields.has("no_gap_assessment_validated"),
    };
  }
  function eligibleRepairs(record) {
    const prior = (state.detail.records || []).filter(item => item.iteration < record.iteration).flatMap(item => item.proof_gaps || []);
    const forbidden = new Set(prior.filter(gap => gap.scope && gap.scope !== "general_proof").map(gap => gap.gap_id));
    const proposed = new Set((record.proposal?.repairs || []).filter(repair => repair.change?.trim()).map(repair => repair.gap_id));
    return [...new Map(prior.filter(gap => !forbidden.has(gap.gap_id) && proposed.has(gap.gap_id)).map(gap => [gap.gap_id, gap])).values()];
  }
  function reviewPanel(record) {
    if (!state.detail.summary) return '<div class="notice notice-warning">This partial run has no final summary yet. Independent review is unavailable; you can still inspect and export the existing records.</div>';
    const draft = state.drafts[reviewKey()] || {};
    const eligible = eligibleRepairs(record);
    return `<div class="notice notice-warning">Only record your own independent mathematical assessment. A model’s confidence or a finite test is insufficient. Your review is bound to this exact proof attempt and saved separately from the original record.</div>
      <form id="review-form" class="review-form"><label class="field"><span>Reviewer identity</span><input name="reviewer" value="${h(draft.reviewer || "")}" autocomplete="name" required maxlength="300"></label>
      <label class="checkbox-field"><input type="checkbox" name="assess_proof" ${draft.assess_proof ? "checked" : ""}><span>Evaluate the proof outcome in this review. Leave unchecked to review the skeptic alone and preserve the proof score.</span></label>
      ${[["target_preserved", "The original theorem target is preserved."], ["assumptions_preserved", "Only the permitted assumptions are used."], ["complete_proof_validated", "I independently checked a complete proof of the stated theorem."]].map(([name, label]) => `<label class="checkbox-field"><input type="checkbox" name="${name}" ${draft[name] ? "checked" : ""}><span>${h(label)}</span></label>`).join("")}
      <fieldset><legend>Specific proof gaps independently confirmed as repaired</legend>${eligible.length ? eligible.map(gap => `<label class="checkbox-field"><input type="checkbox" name="repaired_gap_ids" value="${h(gap.gap_id)}" ${draft.repaired_gap_ids?.includes(gap.gap_id) ? "checked" : ""}><span><strong>${h(gap.gap_id)}</strong> — <span data-math>${mathText(gap.description)}</span></span></label>`).join("") : '<p class="muted">No eligible prior general-proof gaps are addressed by this attempt. Finite-certificate corrections are not eligible.</p>'}</fieldset>
      ${record.critique_sha256 ? `<fieldset><legend>Evaluate the skeptic independently</legend><p class="muted">Confirm a finding only after checking the specific diagnosis. A failed Lean check alone is insufficient.</p>${record.critique.gaps.length ? record.critique.gaps.map(gap => `<label class="field"><span><strong>${h(gap.gap_id)}</strong> — <span data-math>${mathText(gap.description)}</span></span><select data-skeptic-gap="${h(gap.gap_id)}"><option value="unreviewed">Not reviewed</option><option value="confirmed" ${draft.critique_review?.confirmed_gap_ids?.includes(gap.gap_id) ? "selected" : ""}>Confirmed finding</option><option value="dismissed" ${draft.critique_review?.dismissed_gap_ids?.includes(gap.gap_id) ? "selected" : ""}>Dismissed finding</option></select></label>`).join("") : `<label class="checkbox-field"><input type="checkbox" name="no_gap_assessment_validated" ${draft.critique_review?.no_gap_assessment_validated ? "checked" : ""}><span>I independently validate the skeptic’s assessment that no gaps remain.</span></label>`}<p class="muted">Critique SHA-256: <span class="hash-text">${h(record.critique_sha256)}</span></p></fieldset>` : ""}
      <label class="field"><span>Review evidence and reasoning</span><textarea name="evidence" rows="6" required placeholder="Identify the proof steps checked, remaining issues, and why any selected gap is repaired.">${h(draft.evidence || "")}</textarea></label><p class="muted">Proposal SHA-256: <span class="hash-text">${h(record.proposal_sha256)}</span></p><p class="error-state" id="review-error" role="alert"></p><button class="button primary" type="submit">Save independent review</button></form>`;
  }
  function renderRunDetail() {
    const detail = state.detail;
    const run = detail.run || detail.summary || {};
    const records = detail.records || [];
    const record = currentRecord();
    const score = record ? currentScore(record) : null;
    const files = detail.available_files || [];
    const dialog = $("#run-dialog");
    dialog.setAttribute("aria-labelledby", "run-title");
    dialog.innerHTML = `<header class="dialog-header"><div><p class="eyebrow">Research record</p><h2 id="run-title">${h(detail.id)}</h2><div class="tag-list">${badge(run.mode || detail.manifest?.mode)}${badge(run.backend || detail.manifest?.backend)}${badge(run.status || "running")}</div></div>${button("×", "close-run", 'aria-label="Close run details"', "icon-button")}</header>
      <div class="detail-layout"><aside class="iteration-nav"><h3>Proof attempts</h3><div class="iteration-list">${iterationNumbers().map(iteration => {
        const item = records.find(item => item.iteration === iteration);
        return `<button type="button" data-action="select-iteration" data-iteration="${h(iteration)}" class="iteration-button ${iteration === state.iteration ? "active" : ""}" ${iteration === state.iteration ? 'aria-current="true"' : ""}><strong>Iteration ${h(iteration)}</strong><small>${item ? `Proof ${number(currentScore(item)?.score)} / 3 · ${number(item.proof_gaps?.length || 0)} gaps` : "Incomplete attempt · recorded actions"}</small></button>`;
      }).join("") || '<p class="muted">Waiting for the first recorded action.</p>'}</div><h3>Export records</h3><div class="export-links">${files.map(filename => `<a href="/api/runs/${encodeURIComponent(detail.id)}/files/${encodeURIComponent(filename)}" download>${h(filename)} ↗</a>`).join("") || '<p class="muted">No exports yet.</p>'}</div></aside>
      <main class="detail-main">${run.error ? `<div class="error-state">${h(typeof run.error === "string" ? run.error : run.error.message || JSON.stringify(run.error))}</div>` : ""}${run.warnings?.length ? `<div class="notice notice-warning">${list(run.warnings)}</div>` : ""}
      ${record ? `<div class="attempt-summary"><div><p class="eyebrow">Iteration ${h(record.iteration)}</p><h3>${h(scoreLabel(score))}</h3></div><span class="score-pill">${number(score?.score)} / 3</span></div><p class="muted" data-math>${mathText(score?.rationale)}</p>` : ""}
      <div class="tabs" role="tablist" aria-label="Run detail sections">${["evaluation", "proof", "verification", "activity", "review"].map(tab => `<button type="button" id="tab-${tab}" class="tab ${state.tab === tab ? "active" : ""}" role="tab" aria-selected="${state.tab === tab}" aria-controls="run-panel" data-action="select-tab" data-tab="${tab}" ${!record && !["activity", "evaluation"].includes(tab) ? "disabled" : ""}>${h(tab === "evaluation" ? "Agent results" : human(tab))}</button>`).join("")}</div><div id="run-panel" class="tab-panel" role="tabpanel" aria-labelledby="tab-${state.tab}">${state.tab === "activity" ? activityPanel() : state.tab === "evaluation" ? evaluationPanel(record) : !record ? empty("Waiting for an attempt", "Open Activity to see model calls and current progress.") : ({proof: proofPanel, verification: verificationPanel, review: reviewPanel})[state.tab](record)}</div></main></div>`;
    window.MathRenderer?.render(dialog);
  }
  async function submitReview(form) {
    saveDraft();
    const record = currentRecord();
    const payload = {...state.drafts[reviewKey()], run_id: state.detail.id, iteration: record.iteration, proposal_sha256: record.proposal_sha256, evidence_type: "human_mathematical_review"};
    const submit = $("button[type=submit]", form);
    submit.disabled = true;
    try {
      await api(`/api/runs/${encodeURIComponent(state.detail.id)}/reviews`, {method: "POST", body: JSON.stringify(payload)});
      delete state.drafts[reviewKey()];
      const id = state.detail.id;
      state.detail = {...await api("/api/runs/" + encodeURIComponent(id)), id};
      state.tab = "proof";
      renderRunDetail();
      $("#tab-proof")?.focus();
      await refresh();
      announce("Independent review saved. Original experiment records are unchanged.");
    } catch (error) { $("#review-error").textContent = error.message; submit.disabled = false; }
  }
  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true;
    const hadLive = (state.workspace?.jobs || []).some(active);
    try {
      state.workspace = await api("/api/workspace");
      renderView();
      const live = (state.workspace.jobs || []).filter(active).length;
      announce(live ? `${live} active ${live === 1 ? "job" : "jobs"} · updating automatically` : `Workspace updated ${new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})}`);
      if ($("#run-dialog").open && state.detail && state.tab !== "review" && (live || hadLive)) {
        const id = state.detail.id;
        const detail = await api("/api/runs/" + encodeURIComponent(id));
        if (JSON.stringify(detail.records) !== JSON.stringify(state.detail.records) || JSON.stringify(detail.run) !== JSON.stringify(state.detail.run) || state.tab === "activity") {
          state.detail = {...detail, id};
          state.iteration ??= detail.records?.at(-1)?.iteration ?? null;
          renderRunDetail();
        }
      }
      if (state.view === "comparisons" && state.comparison && (live || hadLive)) {
        state.comparison = await api("/api/comparisons/" + encodeURIComponent(state.comparison.id));
        renderComparisons();
      }
    } catch (error) {
      announce(error.message, true);
      if (!state.workspace) $("#view-" + state.view).innerHTML = `<div class="error-state"><h2>Unable to load the workspace</h2><p>${h(error.message)}</p>${button("Try again", "refresh")}</div>`;
    } finally { state.refreshing = false; }
  }
  async function poll() {
    await refresh();
    window.setTimeout(poll, (state.workspace?.jobs || []).some(active) ? 2000 : 6000);
  }
  document.addEventListener("click", async event => {
    const nav = event.target.closest("[data-view]");
    if (nav) { event.preventDefault(); showView(nav.dataset.view); return; }
    const target = event.target.closest("[data-action]");
    if (!target || target.disabled) return;
    try {
      switch (target.dataset.action) {
        case "refresh": await refresh(); break;
        case "view-runs": showView("runs"); break;
        case "view-benchmark": showView("benchmark"); break;
        case "view-comparisons": showView("comparisons"); break;
        case "open-launch": if (state.workspace) openLaunch(target.dataset.kind); else announce("Wait for the workspace to load.", true); break;
        case "close-launch": $("#launch-dialog").close(); break;
        case "close-run": saveDraft(); $("#run-dialog").close(); break;
        case "open-run": await openRun(target.dataset.id); break;
        case "cancel-job": target.disabled = true; await api(`/api/jobs/${encodeURIComponent(target.dataset.id)}/cancel`, {method: "POST", body: "{}"}); await refresh(); announce("Cancellation requested. Completed records remain available."); break;
        case "open-comparison": state.comparison = await api("/api/comparisons/" + encodeURIComponent(target.dataset.id)); renderComparisons(); break;
        case "select-iteration":
          saveDraft();
          state.iteration = Number(target.dataset.iteration);
          if (!currentRecord() && !["evaluation", "activity"].includes(state.tab)) state.tab = "evaluation";
          renderRunDetail();
          [...$("#run-dialog").querySelectorAll(".iteration-button")].find(node => Number(node.dataset.iteration) === state.iteration)?.focus();
          break;
        case "select-tab": saveDraft(); state.tab = target.dataset.tab; renderRunDetail(); $("#tab-" + state.tab)?.focus(); break;
      }
    } catch (error) { target.disabled = false; announce(error.message, true); }
  });
  document.addEventListener("input", event => {
    const key = event.target.dataset.filter;
    if (key) { state.filters[key] = event.target.value; renderRuns(); }
  });
  document.addEventListener("change", event => {
    if (event.target.closest("#launch-form")) updateLaunchFields();
    const key = event.target.dataset.filter;
    if (key) { state.filters[key] = event.target.value; renderRuns(); }
  });
  document.addEventListener("submit", event => {
    if (event.target.id === "launch-form") { event.preventDefault(); launch(event.target); }
    if (event.target.id === "review-form") { event.preventDefault(); submitReview(event.target); }
  });
  $("#run-dialog").addEventListener("cancel", saveDraft);
  $("#run-dialog").addEventListener("keydown", event => {
    if (!event.target.matches('[role="tab"]') || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...$("#run-dialog").querySelectorAll('[role="tab"]:not(:disabled)')];
    const index = tabs.indexOf(event.target);
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].click();
  });
  showView("overview");
  $("#view-overview").innerHTML = '<div class="loading-state" role="status">Loading the research workspace…</div>';
  poll();
})();
