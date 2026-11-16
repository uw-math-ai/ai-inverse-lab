# V2 to V3 migration plan

Prepared 2026-11-12 by the independent Astra planning lead, before implementation. This is a development plan, not a statement that the capabilities below already exist. The authoritative requirements are [V3_Research_Harness_Goals.md](V3_Research_Harness_Goals.md); the motivating evidence is [the V2 research review](../v2/RESEARCH_REVIEW_FOR_V3.md). No scientific experiments were requested or run for this plan. No credential files were read. No applicable `AGENTS.md` was found in the repository or checked ancestor paths.

## Scope and migration decision

Evolve the working V2 implementation in place. Add an explicit `pipeline: v3` and a V3 scheduler that reuses V2 data preparation, method factories, execution isolation, artifact storage, model selection, and export infrastructure. Keep `pipeline: v1` and `pipeline: v2` and their report/resume paths functional. Correct shared scientific defects where appropriate; version claim and result schemas so historical artifacts are not silently reinterpreted.

The V3 unit of work is a study with stable questions, claims, arms, controls, diagnostic requirements, and evidence links. A research program organizes studies. Its orchestrator decides the next bounded action from current evidence; researchers own questions and can commission bounded specialists. Neither an LLM assertion nor a passing baseline score can set study completion or certify scientific support. Deterministic evidence checks determine those states.

P0 and P1 are release gates for P2. P3 and P4 consume the same contracts and evidence ledger. Integrating APIs into an actual CLI-to-report path is required: unused helper classes, a fixed sequence called an orchestrator, or a configurable role list with no runtime spawning are insufficient.

## Existing implementation and precise gaps

Paths below are relative to the harness directory.

| Priority | Existing capability worth retaining | Concrete gap and change |
|---|---|---|
| P0 | `metrics.py` correctly implements global Frobenius `rel_l2`; `research/proposals.py` normalizes predictions and downgrades unsupported proof statuses. | Agents receive metric names instead of an exact machine-derived scientific contract. `check_prediction(expect="worse", improvement=.30)` accepts positive gains below .30. One scalar outcome upgrades an entire proposal. Add typed claims, finite-value validation, explicit effect semantics, per-claim outcomes, and conservative legacy conversion. |
| P0 | `research/gate.py`, `methods_v2.py`, `safety.py` provide normalization, forward authorization, static code checks and sandboxing. | Normalization can change the intervention; a static factory name is not a verified executable method; arbitrary diagnostic attributes are not collected. Record requested/effective specs and hashes, smoke-check factories and output contracts in isolation, and verify declared diagnostics/intervention evidence. |
| P0 | `runner.py` saves predictions and row indices, computes test bootstrap/per-target/uncertainty outputs, and exports custom modules and simulator copies. | Final prediction and shift calls precede pickle; reload checks only a slice and never gates certification. Final outputs lack explicit finite/shape checks and repeat/batch certification. Save a pristine fitted export before evaluation, verify actual exported behavior, and record certification in the manifest/report. |
| P0 | `agents/reporter_v2.py` renders tables and trace from artifacts. | Its discussion prompt omits implementation, controls and diagnostics and truncates serialized JSON. It can contradict tables or call a local CRB estimate a universal floor. Generate factual statements from one evidence bundle; unsupported discussion cannot become certified fact. |
| P1 | Proposal versions, lineage, portfolio diversity, pilots and method deduplication already exist. | `research/portfolio.py` selects individual methods, loses required partners, and marks deduplicated baselines only with `also_baseline`. Add stable study/arm/dependency IDs, retain baseline identity independently of proposal origin, and admit/defer required groups together. |
| P1 | `executor.py` has process isolation, timeouts, caching and resume; `resources.py` charges worker time. | Admission checks race with concurrent jobs; charging occurs after completion; final is exempt; cache identity omits V3 diagnostics/protocol. Add locked reservations, confirmation/export reserves, explicit hard/soft policies, and study/protocol-aware cache identity. |
| P2 | `research/loop.py` supports revisions, abandonment, branching and critic-requested pilots. | All full evaluation occurs after the loop. Pilots rerun the normal method rather than the requested diagnostic. `scheduler_v2.py` keys researchers by persona name, so duplicate personas overwrite. Add action-driven program control, unique agent instances, genuine specialist callbacks, typed diagnostics, and full-result feedback before the next decision. |
| P2 | LLM backends share `complete`; `Agent.ask` accounts for calls and service failures; the critic has independent implementation. | All agents share one backend; JSON/shape fallback lacks a causal record; critic objections are prose. Add injectable role routing/escalation, parent/task attribution, explicit fallback reasons and persistent objection records with evidence/revision/uncertainty resolutions. |
| P3 | `tools/literature.py` has retrieval, applicability screening, caches and source attribution. `memory/store.py` retains contradictions and artifact references. | Researchers route almost every request to TheoremSearch; registry presence can count as support despite unknown applicability. Memory strips custom code and redacts only selected top-level data keys. Add typed evidence routing, verified assumption maps, runnable packages and a safe agent-facing projection. |
| P4 | V2 records data/run seeds, costs, test access in memory, shifts, and artifact lineage. | Exposure tracking disappears with fresh/disabled memory; no program-level quality/completion/reliability/transfer metrics or independent-instance protocol. Add a separate exposure registry, evaluation protocol validation and metrics aggregation without launching a campaign. |

Historical review findings are evidence of past outcomes, not proof that current source reproduces every past defect. In particular, current `inversion.py` already contains per-observation seeding for posterior sampling and `tests/test_v2.py` tests batch independence. Extend verification; do not replace this working behavior with a weaker seed reset. Multistart and custom methods still require repeatability checks.

## Preservation constraints

1. Preserve all six V2 method kinds (`library`, `physics`, `hybrid`, `custom_ml`, `custom_solver`, `random_search`), target transforms, augmentation, authorized forward access, uncertainty, ensembles and refinement. The no-forward learned-surrogate path must remain expressible and runnable. Scientific truthfulness must not become a small fixed method menu.
2. Retain strong conventional baselines, separate their identity from who proposed them, and distinguish practical model quality from original scientific discovery. Refinement children must not retroactively replace registered treatment/comparator configurations.
3. Keep the append-only artifact ledger canonical. Preserve code, proposal versions, failures, negative findings, deferrals, merges and supersessions. Resume and report rebuild must use persisted evidence and must not rerun final test evaluation.
4. Keep deterministic implementation/safety checks separate from the scientific critic. All delegated code, including code from memory, goes through the same gate and isolated executor. Do not expose test/shift arrays, credentials, arbitrary file reads, or raw final artifacts to research agents.
5. Preserve offline behavior, fake LLM/client injection, remote fallback behavior and existing tests. Real LLM, retrieval, SSH or experimental campaign calls are not part of development validation.
6. Leave historical run directories, ablation outputs and the user's goal document unchanged. Write migration/progress/architecture/test/change records in this `v3/` directory.

## Interfaces to settle before concurrent coding

Use JSON-serializable versioned records (dataclasses with conversion or validated dictionaries fit the existing code). The implementation lead can adjust names, but must publish agreed fields and method signatures before partitioned work. Unknown/malformed evidence is an explicit error or unresolved state, never a default pass.

### Scientific contract and claims

- `ProblemContract`: schema version; actual input/target shapes and flattened feature count measured from training arrays; split roles/fingerprints; metric name, exact definition, direction and aggregation; transform order; noise units and augmentation semantics; forward/simulation privileges; bounds/prior; diagnostic limits. Expose this same object to researchers, critic, code engineer, execution and reporter.
- `MethodContract`: requested/effective normalized method specs and hashes; source hash; normalized changes; fit/predict/factory contract; available diagnostic names; deterministic/stochastic inference policy. Evidence says what ran, not what the proposal promised.
- `Claim`: stable ID, question/study ID, kind (`performance`, `comparison`, `mechanism`, `mathematical`), text, metric/estimand/aggregation, explicit treatment and control arm IDs, allowed varying factors, required evidence/diagnostics, margin/operator, fidelity/seed/uncertainty requirements, applicability and source references.
- `ClaimVerdict`: claim ID/version, outcome (`supported`, `contradicted`, `inconclusive`, `untestable`), exact predicate tested, values/effect/interval, evidence artifact IDs, unmet requirements, reason. A contradicted registered threshold is not evidence of no effect. A guarantee needs proof-specific evidence; empirical accuracy cannot establish it.
- For a lower-is-better nonnegative metric, signed relative improvement is `(control - treatment) / abs(control)`; reverse the numerator for higher-is-better metrics. "At least 30% worse" means improvement `<= -0.30`; "no more than 30% worse" means `>= -0.30`. Define equality and zero-reference behavior explicitly. Do not silently reinterpret legacy ambiguous `expect=worse` records as confirmed V3 findings.
- A legacy single scalar prediction may become a performance claim. It cannot stand in for mechanism or missing comparison claims. Dynamic `best_family` references are exploratory context unless frozen to explicit configuration/result IDs before the confirming test.

### Studies and execution

- `Study`: stable ID/version, question, owner and collaborators, priority, hypotheses/claim IDs, required and optional arm IDs, dependencies/DAG, matching policy, diagnostic requests, planned budgets, status and reason, evidence references, merge/supersession mappings. Distinguish `planned`, `running`, `complete`, `incomplete`, `deferred`, `failed_implementation`, `abandoned`, `superseded`. Completing a study means its registered required work was performed, not that its hypothesis won.
- `StudyArm`: ID; method contract; treatment/control/baseline role; required flag; train/validation row fingerprint, fidelity and seed policy; intervention factors; optional shared-fit/trajectory identity. Deduplication maps many scientific arms/claims to one identical execution without deleting obligations.
- `EvidenceRequest/DiagnosticRequest`: ID, source/tool type, study/claim IDs, inputs and required outputs, budget/fidelity, status, result artifact IDs. Unsupported diagnostics return `unsupported` and keep dependent claims unresolved. A boolean `pilot` is not a diagnostic implementation.
- `StudyExecutor.execute(study) -> StudyEvidence`: validate graph/contracts; reserve all required work; run admitted arms/diagnostics through existing gate/executor; reconcile actual costs; preserve individual failures; analyze matching and completion. Admission is atomic, execution need not succeed atomically. If an arm crashes after a partner finishes, retain both records and mark the study incomplete.
- Extend result/job metadata with schema/protocol version, study/arm/claim IDs, effective method/source hash, split and row hashes, fidelity, seeds, required diagnostic payload/status, runtime verification, parent task/agent IDs and costs. Returned prediction paths and row indices must remain available to paired analysis.
- Paired comparisons require the same validation rows and appropriate matching factors; bootstrap matched rows and recompute the actual registered global metric for each resample. Keep algorithm-seed variation separate from dataset/problem-instance variation. Missing rows, missing control or insufficient uncertainty returns inconclusive.
- Shared-fit/trajectory reuse requires an explicit adapter/protocol and immutable shared artifact identity. At minimum implement a real reusable prediction-analysis path plus one shared computation fixture; do not claim generic fitted-state sharing merely because method hashes match. Unsupported shared-state requirements defer the study instead of silently executing unmatched arms.

### Program control and delegation

- `ResearchProgram`: objective, unresolved questions and dependency links, studies, priorities, agent registry, shared evidence view, objections, remaining resources, decision history and stop reason.
- `ProgramAction`: validated action (`decompose`, `assign/spawn`, `propose_study`, `execute`, `diagnose`, `retrieve`, `revise`, `merge`, `defer`, `resolve_objection`, `stop`), target IDs, rationale, evidence IDs and expected decision value. Each action is persisted before/after execution.
- `AgentInstance`: unique ID independent of persona, parent ID, role/persona, assigned question/task, capability policy, depth, model tier, status and cumulative spend. Multiple simultaneous instances of one persona are valid. Orchestrator/critic ownership cannot be overwritten by a researcher.
- `SpecialistTask`: parent researcher, question/study, kind (`coding_execution`, `analysis`, `retrieval/source`), exact request, allowed evidence inputs, depth/call/time limits, result artifacts/observations/failures/uncertainty. Researchers interpret results and submit conclusions; the orchestrator integrates them.
- `DelegationManager`: bound total/active agents, depth, children per parent, turns/tasks and shared budgets. Reject cycles/unknown parents/unauthorized actions with ledger records. Route actual work through bounded tools; metadata-only tasks do not satisfy delegation.
- Role LLM configuration maps orchestrator/researcher/critic/specialist to backend/model/effort, with inherited defaults and an explicit stronger-tier escalation action. Use existing `LLMBackend` injection and `make_llm`; do not hard-code provider names or assume that differently named configurations guarantee a capability tier. Record selected tier/model and fallback reason.
- `CriticObjection`: stable ID, scope (study/program), blocking flag, requirement, status, evidence/revision/acknowledged-uncertainty resolution and responsible actor. The critic reviews design, evidence and final synthesis. Acknowledged uncertainty remains visible and cannot satisfy a missing claim requirement.

### Knowledge and evaluation

- `EvidenceRouter.resolve(request, safe_context)`: local contract/source metadata; allowlisted run artifact projection; diagnostic execution; research memory; theorem retrieval; empirical-literature adapter. Return exact source identity/hash, applicability map, uncertainties, cost and result status. Unknown empirical-provider capability is explicit; do not relabel theorem search as comprehensive methods-literature retrieval.
- `MethodPackage`: code/config plus content hashes, factory/interface version, environment requirements, provenance, validation evidence, supported claims, failure conditions and applicability. Copy reusable code into a durable package, verify hashes on load and reapply the gate. A path to a disappearing run folder is insufficient. Allow legacy records as unverified leads.
- `EvidenceBundle`: contract, all studies/claims/arms and their statuses, winning actual method identity/implementation evidence, exact metrics, diagnostics and absences, uncertainty, shifts, objections, fallback attribution, costs, export certificate and artifact links. This is the sole source for report facts, program metrics and scientific memory writeback.
- `EvaluationProtocol`: dataset/problem-instance IDs/seeds separated from training/search seeds, frozen baseline identity, component toggles and declared budgets, development/final stress split roles, memory mode/snapshot identity, campaign holdout policy. Validate comparability; do not run campaigns.
- `ExposureRegistry`: separate from mutable research memory, keyed by dataset/split fingerprint and purpose, records attempted/reserved and completed final access with run/selection IDs. Use an atomic local persistent implementation (SQLite is suitable). Fresh or disabled memory cannot reset exposure. Strict held-out mode blocks duplicate exposure; development mode records explicit reuse. Resume is idempotent, and uncertain interrupted access stays visible.
- `ProgramEvaluation`: model quality, required-work completion/orphan rate, claim evidence coverage/reliability checks, tested negative/inconclusive outcomes, novel method and transfer evidence, export certification, total/attributed cost and time to resolved questions. Unknown/untested quantities are null with reasons. A real transfer claim requires separate instance evidence; package availability alone is not transfer success.

## Sequenced implementation work

### Step 0 — Freeze seams and capture regression baseline

Implementation lead reads this plan, records the agreed minimal schemas, and creates `V3_PROGRESS.md` with P0–P4 acceptance tracking. Preserve V1/V2 CLI dispatch and add a V3 entry path without changing default historical behavior. The root coordinator reports that all 36 existing V1/V2 tests passed in 194.931 seconds using Python 3.11; evidence is `baseline_tests.log`. These are existing software fixtures, not a new research campaign. Establish fixture-only validation with fake LLM/retrieval/remote services and small synthetic arrays.

### Step 1 — P0 scientific truth and export/report integrity

Implement scientific contracts, typed claim normalization/verdicts, exact metric descriptions and ambiguity rejection. Add effect/finite/missing-control tests first. Supply effective normalized specs and implementation diagnostics to a bounded smoke runner and report bundle; distinguish declared from observed implementation properties. Validate diagnostic gates as actions (passed/failed plus enforced fallback), not merely the presence of an attribute.

Correct export order and certification: build/fit, save pristine inference state, load the exported object, perform canonical evaluation through that artifact, and verify repeated, full-versus-partitioned, and independent-reload predictions at a declared tolerance. A stochastic custom method may supply an explicit reproducibility protocol; otherwise report uncertified and do not advertise a validated winning model. Do not repair/refit/select using final test labels after an export failure. Prefer development-input certification before committing final exposure. Manifest/model card/report all carry the same certificate.

Create factual report rendering from `EvidenceBundle`; retain model labels, actual method/source hashes, merged lineage, all required measurements and unresolved claims. Optional LLM explanation must be structured against known evidence IDs and checked, or marked as unverified commentary; deterministic fallback must stay complete. Do not attempt to guarantee arbitrary prose truth with a few keyword checks. Report local Fisher/CRB diagnostics with their assumptions, not as universal Bayes limits.

Acceptance: a threshold-pass/orphan mechanism example remains scientifically unresolved; `.30 worse` rejects a method 10% better; NaN/inf/zero-reference cases cannot support a claim; misleading dimensions/augmentation assumptions are contradicted by contract facts; failing repeat/reload/batch behavior cannot get certification; report facts name the executed surrogate and available diagnostics from fixtures.

### Step 2 — P1 whole studies, diagnostics and budgeted confirmation

Implement study normalization/DAG validation, required-arm groups, stable comparator IDs and lossless dedup/merge mappings. The selector ranks studies for scientific value/feasibility and schedules the complete required set; a control that loses on accuracy must not be pruned out of a confirming comparison. Preserve optional exploratory search/refinement separately.

Add typed diagnostics for paired prediction analysis, per-target/per-sample errors and declared strata, plus a safe extension path for feature statistics, surrogate-fidelity and solver-trace computations. Validate available outputs and exact intervention differences before verdicts. Sharing/reanalysis must reuse persisted evidence rather than rerun a nominal pilot.

Introduce concurrency-safe admission reservations. Distinguish estimates/soft targets from enforceable hard caps; account for final/export work and reserve confirmation before optional refinement. A hard worker-time claim requires timeout-backed reservation against total and remaining wall time, not just predicted mean duration. Track startup/remote/failure costs and release reservation in `finally`. Exact limit semantics and allowed measured overhead must be documented. A legacy final-stage exemption may remain V2-only.

Acceptance: required pairs are admitted together or deferred together; a crash produces an incomplete study rather than a false negative; matching rows/seeds/configuration is checked; a merged study retains every measurement obligation; the same result can serve two claims with intact attribution; concurrent reservations cannot spend another study's confirmation reserve. Tiny paired/stratified fixtures verify global-metric bootstrap and missing-diagnostic behavior.

### Step 3 — P2 adaptive program and actual nested delegation

Add `scheduler_v3.py` with a program loop around reusable V2 execution primitives. Begin with a bounded deterministic policy and allow injectable structured LLM decisions. On each turn: inspect unresolved questions/resources/evidence, choose a valid action, obtain independent critique as required, execute the approved study/tool action, update evidence/claims, send results to its owner and orchestrator, then choose the next action. Full-result feedback must arrive before subsequent decisions, not merely be appended to a final report.

Enable program decomposition, priorities, dynamic unique researcher instances and shared evidence messages. Add researcher-originated specialist requests and returned artifacts for coding/execution, analysis and retrieval/source work. Enforce limits at spawn and dispatch, including nested costs, and preserve critic independence. Record service error, invalid JSON, invalid schema, budget refusal and offline choice as distinct fallback reasons.

Acceptance: a scripted orchestrator creates two same-persona researchers on different approaches, a researcher commissions a specialist, real fixture evidence returns to the parent, and the next program action changes when the measured result changes. A contradiction prompts targeted follow-up/defer/stop rather than another identical debate. Depth/concurrency/cost limits reject excess work; final synthesis includes unresolved critic objections. Role routing and escalation are testable with distinct fake backends.

### Step 4 — P3 verified and runnable accumulated knowledge

Connect evidence routing to researcher/critic/specialist requests. Source/config questions resolve locally; analysis questions invoke the diagnostic path; theorem requests preserve citations and exact applicability maps; unknown applicability supplies a lead, not certified support. Cache/deduplicate shared background without dropping request/decision attribution.

Persist method packages and claim/failure records after verified execution. Retrieve applicable findings and failures before discussion fragments and verify package hashes/interfaces on reuse. Apply recursive safe projection to memory, artifact views, nested diagnostics and package metadata; do not expose held-out metrics through an alternative retrieval path. Preserve frozen snapshot identity versus cumulative-memory behavior.

Acceptance: a local `noise_scale` question makes no theorem-service call; irrelevant/unknown-applicability citations do not elevate status; a temporary custom method package reloads and passes gate/fit/predict without its source run directory; tampering fails verification; a known invalid configuration is avoided or explicitly challenged based on cited failure evidence; nested held-out values cannot enter agent context.

### Step 5 — P4 evaluation readiness, held-out protocol and accounting

Add evaluation-protocol validators, central exposure registry and program evaluation output. Tie finalization to a frozen selection/manifest hash; separate final stress sets from development shifts. The pre-final research loop must never see final evidence. Rebuild reports and resume completed runs without new access. Expose capabilities through documented CLI/configuration, dry-run/protocol validation or offline artifact-summary commands as appropriate.

Acceptance: independent dataset seeds and model seeds are separately recorded; confounded component ablations are rejected/flagged; frozen-memory IDs persist; two schedulers using different memory stores still see the same final exposure; strict duplicate exposure is blocked; completion/reliability/reproducibility/cost metrics work on fixtures containing successful baselines and incomplete research. No experiment campaign is launched and no improvement across real tasks is claimed.

### Step 6 — Independent validation, repair and honest delivery

Independent Sol testing/review starts once the integrated P0–P4 path is runnable. It inspects source and tests outcomes rather than relying only on implementer-authored tests. Return findings by severity and acceptance ID; implementation lead repairs and the tester reruns affected checks. Complete documentation: `V3_ARCHITECTURE.md`, `V3_TEST_RESULTS.md`, `V3_CHANGELOG.md` and updated progress with verified versus unresolved requirements. Include reproducible offline commands and migration notes for V2 specs/artifacts.

Do not mark a priority complete solely because its schema or unit tests exist. The root coordinator should compare the integrated behavior against the goal document and this plan, and disclose unsupported extensions or empirical questions. Development validation demonstrates harness capability, not autonomous discovery quality on a new campaign.

## Recommended bounded implementation partitions

The Sol implementation lead owns interfaces, scheduler integration, CLI/spec changes, shared-file coordination, documentation and acceptance closure. With four live-agent slots, the root plus implementation lead leave two coding slots after the planning agent exits; use sequential waves, then finish coding children before the independent test agent needs a slot.

| Partition | Owned scope | Dependencies/handoff |
|---|---|---|
| A: scientific evidence core | New contract/claim/study/analysis modules and focused tests; P0 semantics, P1 validation, later bundle/evaluation functions. | Agree record schemas first. Return pure validation/verdict APIs to lead and execution worker. Avoid scheduler/runner edits. |
| B: execution and repeatability | `runner.py`, `executor.py`, `resources.py`, targeted `inversion.py`/export changes, runtime verification, diagnostic/job metadata, reservations and export certification tests. | Consume A contracts via agreed JSON records. Preserve V1/V2 mode behavior; lead integrates P1 scheduling. |
| Lead: orchestration integration | `scheduler_v3.py`, `spec.py`, `cli.py`, role routing/delegation/research-agent adapters, critic enforcement, report integration and V3 docs. | Integrate Step 1 then Step 2 before P2. Publish callback signatures for B and safe evidence APIs for next wave. |
| C: knowledge/evaluation wave | Evidence router, runnable packages, safe memory projection, exposure registry and evaluation-protocol fixtures. | After A/B finish or free slots, consume stable evidence and task APIs. Lead owns final scheduler/report wiring; one explicit owner for memory/CLI/shared modules. |
| Independent tester | Review all changes, add adversarial integration fixtures, run V1/V2/V3 suite, verify acceptance matrix and produce findings. | Separate agent from implementation and its coding children. Must not run scientific campaigns or live paid services. |

Do not assign simultaneous ownership of the same shared files. The implementation lead may shrink or reorder tasks within each priority, but must not skip integrated behaviors to finish a nominal ticket list.

## Acceptance matrix for independent testing

| ID | Required observable evidence |
|---|---|
| T0.1 | Exact global metric contract; actual dimensions; requested/effective method changes; noise-added-to-already-noisy semantics; finite predicate handling. |
| T0.2 | Correct better/worse/equivalence margins for min/max metrics; unsupported legacy semantics explicit; scalar performance never certifies absent mechanism/control/proof. |
| T0.3 | Full required diagnostics and implementation verification gate claim support; absent/failing surrogate gate cannot count as enforced. |
| T0.4 | Repeat/full-versus-batch/reload export fixture passes for deterministic methods and fails certification for stateful/noisy methods; serialized certificate/report agree; no nonfinite final success. |
| T0.5 | Canonical report facts cover actual winning algorithm, merged lineage, controls, diagnostics, uncertainty and fallback reasons; rebuild yields identical scientific facts. |
| T1.1 | Atomic group admission; incomplete arm/dependency/diagnostic retained; controls not quality-pruned; orphan count correct. |
| T1.2 | Exact comparator/config/row/fidelity matching; paired global-metric bootstrap; dedup/reuse preserves claims and baseline identity. |
| T1.3 | Reservation concurrency/failure release/final reserve; hard versus soft budget outcomes and parent attribution visible. |
| T2.1 | Duplicate personas retain unique identities; runtime spawn/reassignment/decomposition; collaboration evidence is accessible and cited. |
| T2.2 | Researcher-initiated nested specialist performs bounded actual work and returns artifacts/failures/uncertainty; cap breaches fail closed. |
| T2.3 | Full evidence changes a subsequent orchestrator decision in paired scripted scenarios; stopping depends on resolved work/resources, not citation volume. |
| T2.4 | Independent critic objections survive to study/program synthesis; unsupported resolution cannot certify support; role routing and escalation work. |
| T3.1 | Correct local/artifact/diagnostic/memory/theorem routing, applicability handling, source hashes and retrieval-decision attribution. |
| T3.2 | Durable custom package reuse, tamper rejection, known-failure retrieval, recursive held-out redaction, frozen/cumulative modes distinguished. |
| T4.1 | Dataset/task/train seeds separated; frozen baselines/component budgets validated; no unearned novelty or transfer claim. |
| T4.2 | Exposure independent of memory; duplicate/concurrent reservation prevented in strict mode; resume does not repeat final evaluation; held-out artifacts excluded from research. |
| T4.3 | Machine-readable model/completion/reliability/transfer/reproducibility/cost outcomes on incomplete and negative-study fixtures. |
| R1 | Existing V1/V2 tests and CLI/report paths pass or any deliberate scientific-semantic change is explained with regression coverage. |
| R2 | End-to-end V3 offline and scripted-agent runs on tiny software fixtures produce auditable studies, model artifact, certificate, evidence bundle, trace, report and program metrics. |

## Risks and explicit limits

- **Paper architecture:** the highest risk is implementing rich dictionaries while the runtime still follows V2's fixed sequence. Require T2.2/T2.3 and R2 before calling the migration complete.
- **Unsupported semantic inference:** source strings or self-reported diagnostics cannot prove causal mechanism, an analytic Jacobian, or a universally enforced fallback. Require declared executable checks and preserve uncertainty where verification is absent.
- **Legacy claim promotion:** never migrate an ambiguous historical prediction into a certified V3 claim. Do not rewrite historical research outputs during implementation.
- **Budget claims:** timeout-based reservations can constrain runtime but OS startup/termination and remote overhead complicate exact wall limits. Measure and report overshoot explicitly; soft budgets must be labeled. Do not assert strict guarantees from estimate-only admission.
- **Final holdout leakage:** code inspection, memory packages, reporting and resume create additional evidence paths. Central safe projections and split-aware access must cover every agent-facing path; final reporting is separate from research decisions.
- **Shared-fit correctness:** equal seeds do not prove equal fitted state; reuse must bind source/config/data/seed hashes. Initial limited adapters are acceptable only if unsupported study requirements remain visibly incomplete/deferred.
- **Custom model repeatability:** preserving custom freedom means some models will fail certification. Report a useful but uncertified candidate if necessary, rather than silently blessing it or changing algorithms after seeing test labels.
- **Literature and capability tiers:** provider coverage and model quality are configurable external capabilities. Software tests verify routing, contracts and fail-soft behavior, not frontier research ability or comprehensive literature coverage.
- **Empirical success:** no new heat ablation, task sweep, transfer campaign or live-model research benchmark belongs in this implementation. Better research outcomes remain an evaluation question after the harness passes its software acceptance gates.
