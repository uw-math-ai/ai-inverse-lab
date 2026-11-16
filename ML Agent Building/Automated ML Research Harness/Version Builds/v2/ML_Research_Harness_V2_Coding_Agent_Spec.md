# Autonomous ML Research Harness V2 — Coding Agent Implementation Brief

**Purpose:** Implement V2 by evolving the existing V1 repository. This is a goal- and behavior-oriented specification, not a mandate to rebuild the project or adopt a particular framework.

## 1. Mission

Transform the existing V1 harness from an agent-assisted model-search pipeline into a **research-first autonomous ML system**. It should investigate an inverse problem, generate meaningfully different scientific hypotheses, use mathematical literature and previous research evidence, improve hypotheses through structured criticism, and **only then** implement, train/solve, and evaluate a diverse set of candidate methods.

The principal deliverable is still **the best validated working reconstruction model/method**, accompanied by a traceable record of the research findings, failed attempts, sources, decisions, and experiments that led to it. Better scientific reasoning is a goal to **test**, not a claim to assume.

## 2. Preserve the useful V1 foundation

Work within the existing package and preserve, where possible:

- Research specifications and simulator plugins.
- Central scheduler, finite compute/LLM/wall-time budgets, parallel experiment execution and resumability.
- Artifact-first execution records and lineage, reproducible run directories, caching, checkpoints, reporting and export.
- Existing classical ML library, screening, successive halving, refinement and ensembles as available **execution/search tools**, not as constraints on what research can propose.
- Independent, hard validity and safety checks before execution.

**Do not rebuild the project around a new orchestration framework solely for V2.** Avoid adding SQLite as the canonical experiment store. The existing filesystem artifact ledger remains authoritative for experiment provenance. A separate queryable research-memory store may use whatever suitable persistence/indexing mechanism fits the project.

## 3. Required end-to-end workflow

```text
Research specification / simulator / available tools
                       |
                       v
         Four parallel research subagents
                       |
             Proposals + evidence
                       v
          Shared research-focused critic
                       |
       Challenges, revised assumptions,
       missing evidence, suggested tests
                       |
                       v
   Four researchers revise / branch hypotheses
                       |
       repeat until quality criteria OR
       maximum iterations OR research budget
                       |
                       v
     Select a DIVERSE, feasible research portfolio
                       |
                       v
       Independent implementation/safety gate
                       |
                       v
       Generate code / train or solve / evaluate
                       |
                       v
        Compare with appropriate baselines
                       |
                       v
    Best working model + research findings + memory
```

All four researchers must remain available across rounds. The critic is **inside the research iteration**, not merely a one-time veto immediately before model training. A separate deterministic execution gate remains outside that loop.

At the end of the research phase, advance **multiple distinct viable methods**, subject to budget; do not prematurely force consensus on one hypothesis. If the iteration cap is reached, advance the best valid proposals with their unresolved issues documented.

## 4. Four parallel research subagents

Each is a researcher, not just a parameter sampler. Give the agents different intellectual emphases while allowing every agent to propose **any** mathematical, ML, numerical, physics-based or hybrid technique.

1. **Mathematical researcher:** Examine governing equations, forward/inverse operators, identifiability, conditioning, useful theorems, constraints and analytical/numerical inversion approaches.
2. **ML researcher:** Investigate learned inverses, representations, architectures, estimators, training regimes and ML-plus-physics hybrids.
3. **Experimental researcher:** Investigate simulation fidelity, sensor/measurement design, noise, data distributions, testable competing explanations, controls and evaluation design.
4. **Exploratory researcher:** Challenge framing assumptions, seek nonstandard formulations and combine ideas across research traditions.

Each proposal should make explicit: **hypothesis, motivating reasoning, assumptions, supporting or contradictory evidence, proposed method, falsifiable test, computational feasibility, and what would be learned if it fails.** Agents must be able to produce approaches not represented in the fixed V1 estimator registry.

## 5. Iterative research critic (central V2 change)

After each parallel proposal round, invoke a **shared research critic** which reads all four sets of proposals and relevant memory. Its purpose is **to make hypotheses stronger and more informative**, not simply to reject unfamiliar ideas.

The critic should:

- Challenge mathematical and physical assumptions, validity, identifiability and applicability of cited theorems.
- Identify weaknesses, competing explanations, duplicate ideas and missing controls.
- Return **concrete changed assumptions, questions, counterexamples, evidence requests and revision directions** to the researchers.
- Encourage inexpensive discriminating tests for ambitious ideas instead of automatically dismissing them as costly.
- Preserve diverse lines of investigation; avoid converging all four agents onto the same method family.
- Separate supported facts from speculative conjectures; recognize genuine unresolved disagreement.

All four agents receive feedback and may **revise, defend, abandon, or branch** proposals. Store original and revised versions and the critic's reasoning. Default to a small configurable iteration count (e.g. 2–3 research/critique rounds), with explicit quality/feasibility/budget stopping criteria. Do not allow endless verbal self-approval; empirical experiments, when available, carry more weight than confident dialogue.

**Do not conflate the research critic with code security.** Retain independent checks for feasibility, code safety, budget, metric correctness, and data leakage prior to execution.

## 6. TheoremSearch integration

Make the service at **https://api.theoremsearch.com** available as a retrieval tool to **each** of the four research agents during every round, and to the critic when useful. Consult current provider documentation and available API/MCP access during implementation; do not hardcode speculative endpoint schemas.

Behavioral requirements:

- Search for potentially relevant existing mathematical statements, theorems, and source references using problem-specific queries.
- Return useful structured information to agents, with source/title/identifier/link and theorem assumptions where available.
- Allow agents to cite retrieved material in their proposals and explain **whether its assumptions apply** to the current problem.
- Record queries, useful results, attribution, and retrieval failures as research artifacts/memory.
- Distinguish retrieval of existing statements from original theorem discovery or formally verified proof. **Do not label an LLM conjecture “proved” merely because a nearby theorem was retrieved.**
- Recover gracefully from unavailable service, authentication or rate-limit errors while recording that the literature check was incomplete.

TheoremSearch augments scientific reasoning. It does not replace numerical validation or rigorous mathematical verification.

## 7. Persistent, accessible shared research memory

Add a **persistent, queryable research knowledge layer** accessible to all researchers and the critic, including across future runs. Keep it distinct from the artifact ledger:

- **Artifact store:** Canonical, immutable execution history, code and dataset/model lineage.
- **Research memory:** Searchable, structured scientific knowledge indexed back to artifacts and external citations.

Record at least:

- Original and revised hypotheses, parent/child relationships and authors.
- Assumptions, relevant mathematical results, citations and applicability notes.
- Critic feedback, disagreements, rejected and abandoned ideas **with reasons**.
- Experiment plans, observations, scores, failed runs, limitations and possible explanations.
- Confidence/status such as *conjectured*, *literature-supported*, *experimentally supported*, *contradicted* or *formally verified* (only when actually verified).
- Context of applicability: simulator, problem class, data/noise regime and methodological constraints.
- Pointers to source artifacts, code, datasets, checkpoints and reports.

Agents should **retrieve relevant prior successes and failures before proposing**, and update memory after criticism and after experiments. Memory should not only be a leaderboard: it must preserve **what was learned, why, where it applies, and what evidence disagrees**. Contradictory results should coexist with provenance instead of being silently overwritten.

Do not require a vector database or graph database for V2; make retrieval useful first and keep the backing implementation swappable.

## 8. Broaden what counts as an executable methodology

V1's scikit-learn-style `X -> y` estimator interface should no longer define the limits of scientific proposals. Support, through a controlled extensible interface:

- Data-driven predictors (existing ML methods).
- Numerical/analytical inversion using an authorized forward operator or simulator.
- Hybrid methods, e.g. learned initialization followed by forward-model-based optimization.
- Methods with different *fit*, *solve/predict*, and resource profiles, while retaining a common evaluation/reporting contract.

The heat-equation toy benchmark illustrates why: a separate Levenberg–Marquardt physics MLE achieved approximately **1.84%** relative L2 error while the best V1 harness arm achieved approximately **2.36%**. The physics method was run outside the harness. V2 should make such approaches expressible and testable, **not hardcode this specific solver as the answer**.

Allow research proposals to request controlled access to the forward model, observation operator and problem constraints when supplied by the research specification. Do not permit arbitrary access to sensitive files or tools.

## 9. Post-research execution and evaluation

Only after research iterations finish, the scheduler should:

1. Select a small **diverse portfolio** of well-defined, feasible approaches, retaining suitable baseline methods.
2. Turn research proposals into executable experiment plans/code, with reproducible parameters and explicit assumptions.
3. Apply independent static/dynamic safety, validity and resource controls; run untrusted generated code in genuine isolation.
4. Train or numerically solve the proposed methods and compare them on consistent validation datasets and metrics.
5. Use budget-aware screening without indiscriminately pruning methods that need more data or have different scaling characteristics.
6. Select and export the best legitimate method/model, evaluate on an untouched final test set, and report uncertainty and failure modes.
7. Write experimental evidence back to both the artifact store and shared research memory for later runs.

Evaluate reconstruction quality **and** runtime/compute cost, inference latency, robustness to measurement noise and distribution shift, and reproducibility, when applicable. Include basic heuristics/random search, conventional optimization and strong domain-appropriate physics baselines when available. Reserve test data so repeated development runs do not turn it into validation data.


## Aditional: Wire Access with Server GPU. 

Current AMATH Servers:
americano.amath.washington.edu
cubano.amath.washington.edu
doppio.amath.washington.edu
galao.amath.washington.edu
Server information:

Server
CPU
RAM
Size
GPU
americano
12-core (x2 6-core) Intel Xeon E5-2640 @2.5GHz
32 GB RAM
2 TB
x2 NVIDIA Titan RTX
cubano
24-core (x4 6-core) Intel Xeon X7460
128 GB RAM
500 GB
 
doppio
40-core (x2 20-core) Intel Xeon E5-2698 v4 @2.2GHz
512 GB RAM
6 TB
x2 TITAN X (Pascal) 12G RAM
galao
12-core (x2 6-core) Intel Xeon @2.66 GHz
32 GB RAM
2 TB
 



Step 2: Access information
Connect through the Command Line:
On a Mac: 
Open Terminal (Go → Utilities → Terminal)
Type: ssh uw_id@server.amath.washington.edu   
Log on with your UW password
When you want to log off, type: exit
uw_id = use your UW ID (the first part of your email address) 
server = the server name (i.e. americano, cubano, doppio, galao)
Password = UW password

On a PC:  
Option 1: Using Command Line:
Click the Windows icon and type cmd
Type: ssh uw_id@server.amath.washington.edu   
 Log on with your UW password
uw_id - located in .env file as SERVICE_USERNAME
password - located in .env file SERVICE_PASSWORD (you are not allowed to read this file, only use the variable names)


## 10. Deliverables for the coding agent

Implement V2 incrementally in the current repository and provide:

- A working four-researcher / shared-critic / revision loop, configurable with iteration and budget limits.
- A functioning TheoremSearch tool adapter used by the research agents, with citations and graceful failure behavior.
- Persistent cross-run research memory with agent-facing search/read/write behavior.
- An extended method/experiment interface for ML, physics-based and hybrid inversion.
- Post-research code generation, isolated execution, evaluation and model export using the existing scheduler.
- Clear artifacts showing idea evolution, assumptions, critic feedback, source evidence, experiments, failures and decisions.
- Configuration and CLI documentation, plus example run(s) using the current 1-D heat-equation simulator.
- Automated tests covering tool failures, memory retrieval, iteration termination, proposal provenance, budget enforcement, safety, cache correctness and end-to-end execution.
- A concise implementation report documenting decisions, limitations and evidence of successful behavior.

Prefer a working, testable vertical slice of the whole loop over a large number of partially implemented features.

## 11. V2 acceptance criteria

V2 is ready for evaluation when all of the following hold:

- [ ] Four independently specialized researchers generate hypotheses in parallel, and **all four** receive and respond to critic feedback across multiple rounds.
- [ ] The loop halts deterministically on quality criteria, maximum rounds or resource limits, preserving useful partial work.
- [ ] The critic produces actionable assumptions/revisions rather than merely a pass/fail filter, while maintaining diverse proposed approaches.
- [ ] TheoremSearch can be called by researchers; results have verifiable attribution and explicit applicability checks; outage behavior is tested.
- [ ] Findings, negative results, critiques and literature references persist and can be retrieved in another run.
- [ ] At least one executable hypothesis can use the simulator/forward operator rather than only a fixed sklearn estimator.
- [ ] Multiple methods can be safely executed, measured on comparable data, and selected under budget.
- [ ] A trained model or runnable numerical/hybrid method is exported with reproducibility metadata, quantitative results and research lineage.
- [ ] A final report distinguishes **established evidence, conjectures, failed hypotheses and remaining uncertainty**.

**Success is not defined as “more LLM calls” or “longer debate.”** It is defined by richer testable hypotheses, better evidence, reliable execution, and demonstrable scientific or predictive value per unit of compute. The goal is the push research horizon.

## 12. Explicitly out of scope for this V2

- Formal Lean proof automation.
- Unrestricted model-generated shell commands, unisolated code execution or broad autonomous access to the host machine.
- Large-scale model fine-tuning or expensive global optimization as a prerequisite.
- Mandatory vector-graph architecture or replacement of artifact provenance with a database.
- Increasing the number of researcher agents beyond the proposed four without evidence of benefit.

### TESTSING: 

RUNN ABLATION TEST from V1. for the heat.

## 13. Implementation guidance / definition of done

Make reasonable implementation choices without asking for a new detailed blueprint where the intended behavior is already clear. Preserve V1 compatibility, keep new interfaces narrow, include useful configuration defaults, and make failure modes explicit. Where TheoremSearch's current capabilities or provider constraints differ from expectations, implement the supported retrieval behavior and document gaps rather than inventing an API.

**Guiding principle:** Researchers propose and revise hypotheses; the critic strengthens their reasoning; theorem retrieval supplies prior knowledge; shared memory carries discoveries and failures forward; experiments, not argument alone, decide which approaches survive.
