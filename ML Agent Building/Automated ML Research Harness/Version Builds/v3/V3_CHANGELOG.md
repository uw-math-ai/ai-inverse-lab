# V3 change and rationale log

2026-11-12. This records software changes, not new scientific findings. Original requirements: [V3_Research_Harness_Goals.md](V3_Research_Harness_Goals.md). Planning preceded implementation; [V3_MIGRATION_PLAN.md](V3_MIGRATION_PLAN.md) contains the sequenced handoff and acceptance criteria.

| What changed | Why it changed |
| --- | --- |
| Added a separate `pipeline: v3` runtime and CLI dispatch over existing V2 execution components. | Preserve working method freedom and artifact infrastructure while replacing the scientific control loop. |
| Added a machine-derived scientific contract and requested/effective method records. | Agents and reports need the implemented metric, actual dimensions, noise semantics, privileges and normalized intervention. |
| Added typed claims with conservative per-claim verdicts, signed effect margins and explicit evidence/uncertainty bindings. | A passing accuracy threshold must not certify an unmeasured mechanism, unmatched comparison or mathematical guarantee. |
| Made studies own required arms, dependencies, diagnostics and implicit claim obligations. | Controls and uncertainty work must survive selection, deduplication, delegation and partial failures. |
| Added atomic group reservations and explicit soft/timeout budget policies, including export capacity. | Concurrent work must not silently consume another study's confirmation resources; measured overhead remains visible. |
| Added dynamic unique researcher instances, role-sensitive backends/escalation and researcher-originated specialist callbacks. | Questions need scoped ownership and actual bounded technical work, with evidence returning before the next scientific decision. |
| Persisted independent critic objections and checked evidence/revision resolutions. | Criticism must change claim support and remain visible when unresolved. |
| Added paired/global-metric, per-sample, per-target, stratified and prediction-reuse diagnostics. | Diagnostic requests must produce the requested evidence rather than rerunning an ordinary model under a diagnostic label. |
| Expanded random-search policies into bounded candidate obligations and retained search-selection uncertainty. | Preserve the V2 control capability without certifying validation-selected candidates as confirmatory evidence. |
| Added an actual immutable prediction-sharing adapter and retained execution aliases. | Share computation without discarding claim/arm identity or pretending unsupported fitted-state sharing occurred. |
| Serialized pristine fitted models before inference and gated final evaluation on repeat/reload/partition checks. | Final metrics should describe the exported artifact; mutable or nondeterministic inference cannot silently receive a certificate. |
| Added deterministic factual reports and descriptive program metrics from one evidence bundle. | Narrative must agree with executed methods, controls, diagnoses, uncertainty and completion. Strong model accuracy must not hide unfinished research. |
| Added typed evidence routing, safe nested agent projections, durable hash-verified method packages and frozen-memory resume. | Accumulated knowledge should be applicable and runnable, while held-out results stay outside research context. |
| Added a SQLite exposure registry separate from memory and protocol validation separating dataset/search seeds. | Disabled/fresh memory must not reset held-out exposure, and repeated search seeds must not masquerade as independent instances. |
| Included exact split sizes in V3 dataset cache identity. | Equal total row counts with different train/validation/test allocations must not share a cached partition. |
| Added an offline example spec, usage guide, adversarial review fixtures and integration/regression checks. | Make the change reviewable and usable without requiring live services or a scientific campaign during development. |

## Review-driven repairs

Independent review found and prompted repairs for missing scalar uncertainty enforcement, undeclared varying factors, unrelated paired intervals, incomplete implicit study obligations, stale claim ownership on merge, mutable arm identities, dependency ordering through aliases, late critic objections, unrelated objection-resolution evidence, and final exposure/reservation ordering. These were software defects found during development; they are not experimental discoveries.

The initial V1/V2 baseline was 36 passing tests. The independent test agent added adversarial and end-to-end checks, including a researcher-requested source specialist, duplicate personas, measured-outcome-dependent follow-up, certified export, deterministic report rebuild and resume without another held-out exposure. Exact final commands/counts and repaired versus remaining items are recorded in [V3_TEST_RESULTS.md](V3_TEST_RESULTS.md); [V3_PROGRESS.md](V3_PROGRESS.md) tracks acceptance.

## Retained behavior and deliberate limits

V1/V2 remain available, including historical artifact interpretation and cache behavior. V3 uses explicit typed scientific verdicts; it does not retroactively relabel old results. V2 automatic tuning remains in that pipeline, while V3 scientific revision is represented as explicit program work. Existing method factories, forward-access control, sandbox, uncertainty, model selection and artifact provenance remain the foundation.

Generic shared fitted-state/solver-trajectory adapters, formal proof verification and unconfigured empirical-literature access remain unsupported. Callback deadlines are observed soft limits, not arbitrary in-process preemption. Export certification is limited to its recorded development-input protocol. Discovery/transfer benefit across real independent problems remains unmeasured. These limitations stay visible rather than being counted as completed scientific evidence.

No new scientific experiment campaign, live-model evaluation, external service campaign or historical-run rewrite was performed. See [V3_ARCHITECTURE.md](V3_ARCHITECTURE.md) for module responsibilities and preservation details, and [V3_USAGE.md](V3_USAGE.md) for commands and the example configuration.
