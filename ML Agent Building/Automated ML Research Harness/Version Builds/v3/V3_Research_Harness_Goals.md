# V3 Goal — Autonomous Scientific Research Harness

## Mission

Evolve V2 from a system that generates ideas and selects high-performing models into an **autonomous, evidence-driven research organization**. V3 must be able to define scientific questions, delegate and coordinate research, obtain and interpret experimental evidence, revise its direction, and produce a reproducible model and defensible scientific account. **This is a harness-development specification, not a request to run experiments during implementation.** Preserve V2's useful custom-method freedom, critic, artifact history, and model-evaluation capabilities.

## Intended research organization

**1. A stronger, research-driving orchestrator.** The orchestrator is the scientific lead, not merely a scheduler. It must interpret the overall inverse-problem objective, identify the most important unresolved questions, decompose them into smaller answerable research questions, set priorities, allocate resources, synthesize findings, resolve conflicting recommendations, and decide when more evidence or a change in direction is warranted. It should retain a coherent view of the complete research program and be accountable for the final research product.

**2. Dynamic, question-focused researcher teams.** The orchestrator must be able to **spawn multiple instances of any researcher persona**, not just one fixed set of four. Preserve mathematical, ML, experimental, and exploratory perspectives as available specializations rather than rigid agent slots. It should be possible to assign several researchers with the same persona to pursue independent approaches to one question, or assemble complementary personas around a difficult question. Researchers should collaborate, share evidence, challenge conclusions, and own clearly scoped scientific questions rather than all repeatedly revising the same proposal.

**3. Researcher-controlled specialist subagents.** Researchers must be able to spawn bounded subordinate agents when a question requires work outside their own reasoning loop: for example, a coding/execution agent to implement and run a diagnostic or model, an analysis agent to interpret results, or a retrieval/source-inspection agent to obtain evidence. These subagents return artifacts, observations, failures, and uncertainty to their parent researcher. Researchers remain responsible for interpreting the work; the orchestrator remains responsible for program-level decisions. The system must permit useful nested delegation without uncontrolled agent proliferation or untracked costs.

**4. Tiered model capabilities.** Support role-sensitive model selection: a highest-reasoning-capability tier for the orchestrator (e.g., Astra/Fable-class), a strong research-reasoning tier for researchers and critic (e.g., Opus/Sol-class), and an efficient execution-oriented tier for coding and other narrowly scoped specialists (e.g., Sonnet/GPT-class). These are illustrative role tiers, **not hard-coded provider or model requirements**. Model choice should reflect task difficulty, quality needs, and cost; crucial decisions should be escalatable to stronger reasoning.

**5. An independent, consequential critic.** Keep the critic as an adversarial scientific reviewer. It must challenge assumptions, research priorities, experimental design, matched controls, metric correctness, interpretation, and claims. Its actionable objections must remain visible and be resolved through evidence, explicit revision, or acknowledged uncertainty—not merely more discussion. It must be able to review individual studies and the orchestrator's synthesis.

## What the completed harness must achieve

- **Study-centered research:** A scientific question owns its hypotheses, required comparisons, controls, diagnostics, evidence, uncertainties, and outcomes. Related experiment arms cannot be silently separated or dropped; incomplete studies remain explicitly incomplete.
- **Short, iterative investigation:** Researchers can pursue small decisive questions, commission targeted computations, inspect preliminary findings, and either deepen, redirect, or stop a line of inquiry. Full experimental results must inform subsequent scientific decisions rather than arriving only after research deliberation ends.
- **Scientific truthfulness:** Distinguish model performance from comparative evidence, mechanism claims, and mathematical guarantees. Align claims with the actual metrics, implemented interventions, and measured results; permit contradicted and inconclusive outcomes without treating them as failures of the harness.
- **Reliable research product:** Produce a validated, reproducible winning model when justified, accompanied by a factual report explaining what was tested, what was discovered, what remains uncertain, and how conclusions follow from artifacts. Strong baseline accuracy must not mask an unfinished research study.
- **Usable accumulated knowledge:** Research agents can access verified prior results, failures, source evidence, theorem/literature findings, runnable method artifacts, and applicability limits. Route questions to the appropriate evidence source; retrieval should improve decisions, not just increase citations.
- **Controlled autonomy:** Delegation, execution, model usage, and resource consumption remain observable, bounded, and attributable. Dynamic teams should improve scientific coverage, not merely increase the volume of agent conversations.

## Development priorities (preserve this order)

**P0 — Trustworthy interpretation.** Correct the scientific contract, claim/verdict semantics, reporting fidelity, metric consistency, implementation verification, and exported-model repeatability. The system must not certify unsupported conclusions.

**P1 — Complete studies.** Make the full scientific study—not a standalone model proposal—the unit of work. Preserve required matched controls, diagnostics, dependencies, and completion status through agent delegation and execution.

**P2 — Adaptive, orchestrated research.** Give the orchestrator meaningful scientific authority; enable question decomposition, dynamic persona-based spawning, researcher collaboration, specialist subagents, and experimental feedback that changes subsequent research decisions. Preserve the critic's independence.

**P3 — Knowledge transfer.** Make literature/theorem retrieval, code inspection, research memory, and past run artifacts accessible as applicable, verifiable evidence. Store reusable, runnable findings and failure conditions for future research.

**P4 — Research-program evaluation readiness.** Equip the harness to assess model quality, scientific completion, claim reliability, discovery/transfer, reproducibility, and cost across independent problem instances. Protect held-out evidence and distinguish strong baselines from original discoveries. **This milestone concerns harness capability and evaluation readiness, not running a new experimental campaign as part of this change.**

## Codex implementation handoff (V2 → V3)

Use a **subagent-led development plan** to evolve the existing V2 codebase rather than rebuilding the harness from scratch. These are instructions for **developing V3**, distinct from the research agents that V3 itself will run.

- **Astra planning lead:** Spawn an Astra-level planning agent first to inspect V2, map existing capabilities to P0–P4, identify required changes and preservation constraints, and produce a sequenced V2-to-V3 migration plan. Have it identify interfaces, risks, dependencies, and acceptance criteria before implementation begins.
- **Sol implementation lead:** Assign a Sol-level agent to own implementation against that plan, refine its coding-task breakdown, and spawn/coordinate coding subagents for bounded tasks. It must integrate their work, resolve interface conflicts, and keep changes aligned with the scientific goals rather than simply completing isolated tickets.
- **Independent Sol test agent:** Spawn a separate Sol-level testing/review agent to inspect the implemented changes and run appropriate code tests and regression checks. It should verify P0–P4 acceptance criteria, identify breakages or missing requirements, and return actionable findings for repair and retesting. **This is software validation, not a request to run new scientific research experiments.**
- **V3 documentation and accountability:** Keep the migration plan, progress, architecture and behavior changes, testing results, outstanding limitations, and a concise **what changed / why it changed** log in the `v3/` folder. Document how V2 capabilities were retained, modified, or replaced, and distinguish completed work from unresolved items. The final implementation summary must point to those records.

If the requested model or subagent tier is unavailable, use the closest available equivalent while preserving the separation of planning, implementation, and independent testing responsibilities.

## Definition of success

V3 is successful when a capable orchestrator can organize a changing team of researchers around meaningful scientific questions; researchers can delegate bounded technical work, learn from evidence, and coordinate with an independent critic; and the system can turn that process into **completed, auditable studies and an accurately reported, reproducible research artifact**. Agent count, discussion volume, and nominal model accuracy alone are not measures of success.
