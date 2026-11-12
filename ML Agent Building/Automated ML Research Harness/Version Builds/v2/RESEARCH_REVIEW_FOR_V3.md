# Heat V2 research review and priorities for V3

Reviewed 2026-11-11. Scope: **all 17 `report.md` files in `ablation/heat_v2/runs/`**. Archived degraded runs were excluded. The reports were analyzed alongside their research traces, experiment ledgers, completed status files, exported manifests, selected custom implementations, and relevant current harness source. No experiments were rerun and no existing results or implementation files were changed.

The accompanying [evidence snapshot](RESEARCH_REVIEW_EVIDENCE_v2.json) records every report's hash, section locations, results, proposal checks, portfolio, costs, and aggregate calculations. Cost and wall-time tables below use completed `status.json` values, which include reporting; report section 9 is captured earlier and can show slightly lower totals. Source inspection explains plausible design causes, but does not establish that today's source is identical to every historical run.

## Assessment

**V2 demonstrates useful research ability, especially the ability to invent and execute an alternative formulation when direct physics access is removed. Its main limitation is converting good scientific criticism into completed, interpretable experiments and a reliable final account. V3 should make a complete research study the unit of work.**

The best illustration is `v2_no_forward_s1`. Its generated solver learns a forward surrogate from labeled training examples, then inverts it. It reaches test rel_l2 **0.018497**, close to the forward-enabled methods. That is a substantive change in approach, beyond selecting a regressor from a menu. However, its discussion says the learned-surrogate inversion was never run, even though a merged implementation is the winning model. V2 can produce a stronger result than its own research narrative recognizes.

At the same time, forward-enabled runs mostly reproduce a result that the heuristic arm already obtains: about **0.0183–0.0184** test rel_l2. This benchmark offers limited accuracy headroom for separating research components. The appropriate response is to strengthen the scientific questions and evaluation tasks, preserve strong baselines, and improve experiment follow-through. More proposal rounds alone would not fix the observed problems.

For your priority—final research product before efficiency—I would spend additional compute on matched controls, replication, implementation verification, and stress tests before spending it on more prose revisions or tiny hyperparameter improvements.

## What the aggregate results establish

Lower rel_l2 is better. Standard deviations are across search/run seeds on the **same dataset**, not independent datasets.

| Arm | n | Test rel_l2, mean ± sample SD | LLM $/run | Completed wall min/run | Execution worker-s/run | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| full | 3 | 0.018377 ± 0.000066 | 4.33 | 12.55 | 241.8 | Strong final models; incremental scientific benefit over heuristic physics is not established. |
| offline | 3 | 0.018376 ± 0.000037 | 0 | 0.97 | 139.8 | Existing heuristic methods already solve the nominal task very well. |
| no critic cycle | 3 | 0.018320 ± 0.000078 | 0.81 | 4.84 | 436.0 | Same accuracy region with much less LLM work, but this arm also removes pilots and extra rounds. |
| no literature | 3 | 0.018306 ± 0.000047 | 3.29 | 11.85 | 329.2 | Retrieval has no demonstrated accuracy benefit on this task. |
| memory | 2 | 0.018345 ± 0.000104 | 3.95 | 15.26 | 344.6 | Retrieval works, but improved research decisions or efficiency are not demonstrated. |
| no forward model | 3 | 0.019686 ± 0.001279 | 4.91 | 15.79 | 295.3 | Most promising evidence of adaptive formulation and agent-written methods; results are more variable. |

All 17 runs meet the nominal 0.05 success threshold. All manifests have the same test-feature split hash, `5b49be1eb2715949`. All 14 forward-enabled runs export a forward-using method. Nominal accuracy therefore shows a successful inversion harness, but cannot by itself identify the causal value of critique, literature, memory, or multiple researchers.

Across the 96 selected proposal predictions, the automatic checker reports **65 supported, 9 contradicted, and 22 inconclusive**. Of those predictions, **69 are scalar threshold tests and 27 are relative comparisons**. These are checker outcomes, not an audited count of scientific hypotheses established: the checker often evaluates a much narrower statement than the proposal's title or mechanism.

## Where the research is working

### 1. It can change the formulation, not just tune a known model

In `no_forward_s1`, the exported `SurrInv` implementation fits PCA on the sensor traces, trains kernel ridge from normalized log-conductivity to sensor-PC scores, estimates componentwise surrogate errors on an internal training holdout, and performs batched damped LM inversion of the learned map. Its exported implementation uses no `context.forward` call. Development uses the provided training split; final export is refit on train plus validation under the standard final-stage protocol.

The achieved validation error is **0.019969**, versus **0.027355** for the best data-driven family result in that run. The final test error is **0.018497**. This is the clearest example to preserve and develop in V3: an explicit model of the observation process, learned when the true operator is unavailable, then used for inverse inference.

The other no-forward runs also export custom solvers. Seed 0 combines surrogate/local-solver and MLP predictions; seed 2's custom solver improves validation error by **11.4%** over its best data-driven comparison, exceeding its registered 3% margin. This is promising evidence across several outcomes, although it is not a controlled demonstration that any particular persona or critic caused the improvement.

### 2. The critic often identifies the right scientific problem

Useful criticisms recur throughout the reports:

- An initializer-versus-physics comparison changes iteration count, estimator readout, architecture, or augmentation as well as the initializer.
- A mean-versus-mode experiment needs the same fit and matched observations, not the best member of a moving method family.
- A scalar accuracy threshold cannot establish a claim about basin failures, parameter-specific gains, or information preservation.
- Pilot results from 300 training and 60 validation samples do not settle behavior at full fidelity.
- Near-boundary improvements need enough samples in a predeclared stratum.
- Full-rank local Fisher information does not establish global identifiability or global optimizer convergence.

These criticisms are useful research outputs. The missing step is enforcing them when scheduling, checking, and reporting the study. Retain the critic, but give its actionable requests executable representations and measure whether they were resolved.

### 3. Empirical evidence changes proposals

For example, `full_s0` pilots show multistart counts 1, 3, and 4 producing effectively the same aggregate error near **0.019024**. Several runs subsequently abandon accuracy claims for learned initialization or redirect them toward cost. In `no_forward_s1`, unscaled-PCA KRR improves from roughly **0.0752** to **0.0445** in the pilot configurations, and an unscaled-PCA MLP reaches **0.0327** versus **0.0539** for an earlier raw-input MLP configuration. The latter comparison changes more than one factor, so it motivates a study rather than proving that scaling alone explains the gain.

This is the beginning of an evidence-driven loop. V3 should preserve the observations and complete the controlled tests they motivate.

### 4. The execution foundation supports real research artifacts

V2 keeps proposal versions, criticism, abandonment reasons, experimental results, lineage, exported models, uncertainty summaries, and shift results. Most forward-enabled point estimates cluster tightly, and reported aggregate 2-sigma coverage is generally about **95%** where available. The physics interface makes a scientifically appropriate method expressible inside the harness.

These capabilities are worth keeping. The reproducibility and reporting qualifications below matter, but the artifact foundation is much stronger than an unlogged agent conversation.

## The largest weaknesses and what V3 should change

### A. A proposal's scientific claim is not its executable test

`full_s0` marks the "single-basin test" supported because error is below **0.0205**. That does not measure the fraction of samples converging to different minima. Its mode-versus-mean proposal is also supported by an absolute threshold even though its mean partner was abandoned. The report's conjecture list says "none," while its uncertainty and discussion sections correctly leave mechanism questions open.

`no_forward_s0` marks a parameter-anisotropy claim supported from aggregate error below **0.05**. `no_literature_s1` labels a simulation-amplification study supported relative to the best available data-driven model, but that reference is an **extra-trees** model at **0.08248**, rather than an otherwise identical learner trained without extra simulations. That comparison establishes an achieved score, not the causal effect of additional simulations.

There is also a specific checker defect. In `research/proposals.py::check_prediction`, `expect="worse"` checks `gain < improvement`. Thus `improvement=0.30` accepts a method that is **10% better**, because 0.10 < 0.30. It does not encode "at least 30% worse." This ambiguous pattern appears in four selected predictions, including `full_s0 E-04` and `no_literature_s2 E-03`.

**V3 change:** introduce stable claim records with explicit estimands, named controls, required outputs, intervention factors, aggregation rules, effect margins, and uncertainty requirements. Separate:

1. achieved performance;
2. evidence for a comparison;
3. evidence for a mechanism;
4. mathematical guarantees.

A compound proposal must retain an outcome for each subclaim. Completing an accuracy threshold must leave unmeasured mechanism claims unresolved. "Supported" should require the evidence specified for that particular claim.

### B. Controls and merged hypotheses are lost between discussion and execution

In `full_s0`, the final mode arm survives while its posterior-mean partner disappears. In `full_s1`, the cheap six-step center-start control `P-024` is proposed but not executed, weakening the claimed value of learned initialization. `no_forward_s0` executes a raw-target MLP while its intended log-target twin is not selected. Several reports use prose such as "merged into" or "answered by" without a durable link proving that the receiving experiment produced the required measurements.

The current portfolio selector ranks individual methods by family and critic quality. It has exact method deduplication, but no study object that owns a matched pair, a shared fit, or a set of downstream measurements. Selection can preserve method diversity while destroying the comparison needed to interpret the result.

**V3 change:** schedule studies and dependency graphs. A study owns its controls, matched configurations, observations, seed policy, shared intermediates, analysis, and claim IDs. Treat a required matched contrast as an atomic scheduling group. If a pair cannot fit, shrink both arms or report the study as deferred. A merge transfers all retained claims and measurement obligations to the receiving study; deduplication should share execution without discarding scientific questions.

This also saves compute: mode/mean readouts can share one likelihood fit, multiple iteration cutoffs can share a solver trajectory, and several hybrid studies can reuse an initializer when training data and configuration match.

### C. Good diagnostic requests become more argument instead of measurements

The pilot mechanism checks whether a suggested test has `pilot=true`, then runs that proposal's normal method at smaller fidelity. It does not implement the suggested test's description. A request for "compare 1 vs 2 starts," "measure feature SNR," or "inspect the projected Fisher information" does not automatically become that diagnostic.

The research loop also runs before the main evaluation. Once full-fidelity results arrive, researchers do not conduct another scientific decision round; the later refinement is local method mutation. This helps explain why reports repeatedly finish with the same uncompleted next experiments.

**V3 change:** provide typed experiment and diagnostic tools, including feature statistics, matched solver comparisons, per-sample errors, boundary-stratum analysis, surrogate fidelity, optimizer traces, and learning curves. Record whether each requested diagnostic ran and what decision it changed. Allow full-fidelity results to trigger a short follow-up study before finalization.

Do not force every request through a learned model's `fit/predict` interface. Some of the most valuable research actions here are source inspection, statistics, or reanalysis of existing predictions.

### D. Agents repeatedly reason about the wrong metric, dimensions, and augmentation semantics

The implementation defines

`rel_l2 = ||P - Y||_F / ||Y||_F`.

For a fixed evaluation set, its denominator is independent of predictions, so model rankings match total squared-error rankings. Several criticisms instead assume a mean of per-sample relative errors and derive an inverse-norm-weighted decision rule. That is not the implemented metric. The exact Bayes decision for an expected normalized, nonsquared loss is a separate question; V3 should specify the decision objective rather than silently substituting a different metric.

Some reports reason about **800 features** although the actual data and manifests have **80**. This changes claimed dimensionality, chi-square degrees of freedom, conditioning arguments, and cost estimates. The problem also reaches implementation: the custom `full_s1` winner selects its readout using mean per-sample relative error on a training subset, whereas the harness evaluates global Frobenius error. Executable custom code needs the same metric contract as the researchers.

`noise_scale` is another repeated source of confusion. The current `NoiseAugmented` wrapper adds noise of standard deviation `sigma * scale` to an already noisy input. A stored sample with independent sigma-noise, augmented at scale 1, has total noise standard deviation sqrt(2) * sigma in the augmented copy; the overall training set mixes original and augmented rows. The wrapper also retains originals and multiplies rows by `1 + copies`. Adding more noise copies is not equivalent to obtaining new parameter draws. Current normalization clips nonzero `noise_scale` below 0.1, so a proposal's literal 0.001 need not remain 0.001 in execution.

**V3 change:** publish a machine-generated problem and method contract: observed dimensions, exact metric formula, transforms and their order, noise units, effective normalized parameters, data-generation privileges, and what constitutes a new independent simulation. Let researchers query the relevant source or tool metadata. Add deterministic consistency checks before critique and code generation. Do not spend multiple LLM rounds debating facts the harness can answer directly.

### E. Literature retrieval is operational, but often poorly targeted

Enabled retrieval calls complete without recorded service failures in these runs, but successful retrieval is not evidence of relevance. Across report citation lists there are **133 citation appearances: 71 screened unknown, 35 partial, 22 plausible, and 5 no**. These counts include repeated appearances across runs and are keyword-screen labels, not a fresh assessment of the papers.

More concretely, `full_s0` sends requests such as **"P-004 exact config," "Harness definition of noise_scale,"** and **"Per-feature signal-to-noise across the training set"** to TheoremSearch. These require local artifacts, source, or numerical analysis. The memory report even retrieves an unrelated random-regular NAE-SAT statement. Neither a returned theorem nor a citation count resolves those questions.

The critic's quality function also rewards known citation IDs; existence in the retrieval registry is weaker than applicability to the claim.

**V3 change:** route evidence requests by type: local source, run artifact, diagnostic computation, memory, mathematical result, or empirical methods literature. Deduplicate shared background searches. For a relevant theorem, store the exact claim it supports and an assumptions-to-problem map. An unknown transfer stays unresolved; it should not increase confidence merely because it has a citation ID. Evaluate retrieval by decisions improved or assumptions resolved.

### F. Memory transfers records, but does not reliably transfer usable research

Both memory runs cite 17 distinct prior records in proposals. Yet `memory_s2` produces **zero full-fidelity results for its selected research proposals**; all four proposal predictions are inconclusive, and the final model is the conventional physics baseline. The run still succeeds numerically. That is a research-completion failure hidden by a good final score.

The two memory trials are also sequential: the reports show **167** records available before seed 1 and **322** before seed 2. Seed 2 can retrieve seed 1's records as well as `full_s0`'s. They are not independent trials of the same frozen memory. Cross-run retrieval should not be interpreted as a clean measured treatment effect here.

The current memory writeback excludes custom code from its method summaries, so a remembered custom-solver idea is not automatically a runnable method. Artifact references help, but retrieval needs to reconnect those references to implementation and evidence.

**V3 change:** retain validated method packages, configurations, environment and data provenance, measured effects with uncertainty, failed conditions, open questions, and stable claim status. Distinguish tested failure, implementation failure, missing evidence, duplicate, budget deferral, and superseded proposal. Retrieve high-confidence applicable findings and runnable artifacts before raw discussion fragments. For ablation, clone one frozen memory snapshot per trial; evaluate cumulative learning separately.

### G. The final scientific narrative is materially less reliable than the result tables

Examples:

- `full_s0` says no CRB reference or interval estimates were reported, although sections 1 and 2 contain a numerical CRB diagnostic and a test bootstrap interval.
- `full_s2` says per-segment errors and the CRB reference are missing; both appear earlier.
- `no_forward_s1` says the learned-surrogate inversion was never run and asks what the custom solver does; its winning implementation is precisely a learned-surrogate inversion.
- `memory_s1` describes the 512-versus-128 importance-sample result as a mean-versus-mode result. These are different interventions.
- `no_literature_s2` describes the contradicted physics mean-versus-mode prediction as a hybrid-versus-physics claim.
- Many discussions treat a favorable test-versus-validation gap as showing that selection did not overfit. One favorable comparison does not establish that.

The current reporter gives its LLM a small summary containing family bests, supported/contradicted strings, final scores, and a few abandonment reasons. It omits much of the available diagnostic, uncertainty, shift, implementation, and lineage evidence. This is an identifiable design cause of incomplete or inconsistent discussion.

**V3 change:** generate all factual results from a canonical evidence bundle, then let the reporter explain that bundle with claim-level artifact references. Include the actual winning algorithm, which controls ran, which diagnostics exist, uncertainty, shifts, and inherited claims from merged studies. Check assertions such as "not measured," "never run," and "all predictions" against the ledger. A summary should explain the discovery and its limits, rather than ask for analyses already completed.

### H. Export and resource accounting need stronger acceptance criteria

The existing aggregate writeup says reload differences are all zero. The actual reports and final-result artifacts include:

| Run | Reload max absolute prediction difference |
|---|---:|
| full s1 | 0.0214484 |
| memory s1 | 0.0081574 |
| no literature s0 | 0.0068562 |
| no critic s2 | 0.0025496 |

Other runs have zero or much smaller numerical differences. These discrepancies do not alone prove corrupted serialization. The custom `full_s1` solver uses a mutable random generator during prediction; the final runner predicts test and shift sets before saving, then compares a reloaded 50-row prediction against the earlier test prediction. Random state, batch composition, and call order can therefore confound its round-trip check. The current built-in posterior sampler already seeds draws from individual observations, which addresses part of this issue; V3 should validate the complete prediction contract rather than assume that fix covers every custom model and multistart path.

Execution also exceeds the stated 400 worker-second budget in **no_critic_s0 (491.21), no_critic_s1 (432.15), and memory_s1 (442.73)**. The resource manager charges jobs after completion, running workers do not reserve their expected consumption, and the final stage is deliberately exempt from budget exhaustion. Thus this is a soft execution budget, not a hard common cap. Conversely, useful controls are sometimes eliminated by rigid stage caps or cost projections while total budget remains.

**V3 change:** validate prediction repeatability, batch-size and row-order behavior, save/load equivalence under controlled random state, and stochastic-estimator Monte Carlo error. Assess the exact exported product. Separately meter CPU work, elapsed latency, forward calls, LLM usage, and total campaign time. Reserve compute for in-flight work and a final confirmation/export stage. Support an explicitly declared soft-budget mode, but mark overshoot and distinguish it from a hard-budget comparison. CPU time alone is not a complete replacement for wall time on multithreaded or accelerator jobs.

## A V3 research workflow

The central object should be a **study**: a question plus the evidence needed to answer it. Keep executable methods and artifacts underneath it.

1. **Establish the problem contract and reference results.** Compute dimensions, noise semantics, initial diagnostics, and strong baselines. Explain what remains unresolved and whether the requested accuracy target is already met.
2. **Choose a small set of discriminating questions.** Examples here: whether a surrogate reaches sensor-noise fidelity, whether the surrogate Jacobian is accurate enough for inversion, and whether initialization changes failure probability at a fixed solve budget.
3. **Design the study.** Bind claim IDs to matched arms, shared fits, metric and strata, required artifacts, effect margins, seed and data policy, and a feasible resource reservation. Let the critic challenge identifiability of the comparison and implementation fidelity.
4. **Run the cheapest decisive measurement.** A diagnostic can eliminate an explanation more cheaply than another trained model. A pilot calibrates cost and effect size; it should not silently redefine the claim.
5. **Update the scientific decision.** Inspect results, resolve or revise competing explanations, run a full-fidelity comparison or a follow-up when justified, and record why the next action is informative.
6. **Confirm and package the result.** Use matched comparisons, independent replication appropriate to the claim, stress tests, an audited report, and a validated export. Freeze the method and analysis before final held-out evaluation.
7. **Write back usable knowledge.** Store the supported scope, unresolved questions, effect estimates, and runnable method package. Keep the benchmark evaluation ledger separate from experiment-specific memory.

Start with broad perspectives once, then assign ownership of distinct questions. Let other researchers review studies across roles. The present four personas often all propose a similar hybrid; requiring every persona to revise every round can reward conversation volume rather than complementary scientific work. Use further LLM calls when they can change the experiment or interpretation. Do not remove useful adversarial review simply because this saturated task fails to show an accuracy gain.

Study priority should reflect the importance of the question, the ability to distinguish competing explanations, plausible final-product improvement, and cost. A cheap experiment with no interpretive value should not outrank an expensive decisive comparison merely because it improves throughput.

## Efficiency changes that improve the research product

The purpose of these savings is to fund better confirmation, controls, and robustness analysis.

- **Answer local questions once.** Cache a verified contract for metric, feature shape, transforms, and augmentation. Route source questions to source inspection. The current proposal path also requests an LLM-generated literature query list even when literature is disabled; skip that step when the tool cannot supply evidence.
- **Reuse scientifically identical work.** V2 already has exact experiment caching and method-hash deduplication. Extend reuse to shared initializer fits, solver trajectories, posterior samples, and diagnostic outputs, with keys that include data, preprocessing, seed, code, and environment. Share one execution across multiple claim records; retain deliberate replications as independent work.
- **Calibrate cost before eliminating a control.** Use measured scaling at more than one fidelity when needed and record machine contention. A one-point extrapolation that predicts 121 seconds against a 120-second cap should motivate a smaller matched study or a bounded measurement, rather than silently destroying the comparison. Retain a real timeout.
- **Move budget toward unanswered questions.** Reserve matched-arm and final-confirmation costs, account for running jobs, and release unused reservations. Fixed cumulative stage caps can starve an informative screen while later refinement spends budget on methods already tied within uncertainty.
- **Make additional debate conditional.** Most multi-round LLM runs hit the three-round cap; `no_literature_s2` is a useful exception that stops after two rounds. Ask which unresolved item another round can settle and execute a diagnostic when more argument will not change the study. A nominal task-success threshold need not end research if an important product question remains.
- **Repair the interface before expensive evaluation.** Check factory signatures, estimator cloning, output shape, effective configuration, and declared diagnostics on a tiny smoke input. Treat a failed research implementation as repairable engineering work when its scientific question is still valuable.
- **Send the evidence needed for the decision.** Use stable shared study/configuration objects and compact updates, while preserving exact required facts and links. Arbitrary truncation of a large summary is especially costly when it drops a control ID or a diagnostic the reporter needs.

These changes should be measured by cost to a confirmed research outcome. Their actual savings have not been benchmarked in this review.

## The highest-value follow-up studies on heat

| Question | Required design | Useful result for the final research product |
|---|---|---|
| Why does learned-surrogate inversion work? | Reproduce `no_forward_s1`; compare matched direct KRR, surrogate inversion without uncertainty weighting, and weighted surrogate inversion. Keep train-only tuning and common validation rows; include multiple training seeds. | Establish which part produces the gain, rather than only recording a winning score. |
| Does surrogate accuracy predict inverse accuracy? | Measure held-out forward residuals in sensor-noise units, parameter-direction derivative checks, optimization residuals, and resulting inverse errors. Test the declared fidelity gate. | Explain when a learned observation model is trustworthy and where it fails. |
| Does initialization buy accuracy, reliability, or speed? | Prior-center and learned starts at the same LM iteration/forward-call limits; reuse trajectories for multiple checkpoints. Measure per-sample failures, total training cost, and inference cost. | A measured accuracy/cost frontier and deployment break-even, with a clear mechanism. |
| Is a posterior-mean gain real and where? | One shared likelihood fit; mode and posterior-mean readouts; common random numbers where applicable; fixed boundary strata and paired effect intervals; quantify MC error. | A precise positive or negative result about prior-boundary effects. |
| What limits the learned inverse? | Cross new parameter-draw count with clean-trace noise resampling under a matched architecture and optimizer-step policy. Distinguish scientific fixed-step controls from operational fixed-compute controls. | Separate sample scarcity, augmentation mismatch, and optimization limits. |
| Why does performance degrade out of prior? | Explicitly compare known bounds and expanded bounds, with simulator stability checked; separate noise shift from a genuinely misspecified operator. | A documented applicability domain and evidence for robustness. |

For the surrogate gate, the exported `no_forward_s1` code computes `gate_pass_` but its prediction routine does not branch on that flag. Merely computing a diagnostic is therefore insufficient: record whether it passed and enforce the predeclared fallback or label the gated claim untested. The code also uses finite-difference surrogate derivatives, so a proposal's promise of an analytic Jacobian must not be accepted as implementation evidence.

These are proposed studies, not new experimental results. The current reports do not establish a universal Bayes floor, a globally single-basin landscape, or a general superiority of learned surrogates.

## How to evaluate V3's research ability

Keep this heat task as a regression and export check. Add tasks with meaningful room for scientific decisions: no-forward access, sparse or missing sensors, higher noise, nuisance parameters, and moderate forward-model mismatch. More segments can be useful, but verify the local rank and define an achievable inferential target; an unidentifiable point-estimation problem can otherwise punish the system for a property of the data. Some tasks should reward calibrated uncertainty or a correct non-identifiability diagnosis.

Do not remove a strong physics baseline from the main evaluation simply to make agent gains appear larger. A discovery-only arm can restrict automatic baselines to test whether the system rediscovers a method, but report that experiment separately from the practical comparison against the best available conventional approach.

Measure final-product quality using several auditable outcomes:

- **Model quality:** improvement or justified equivalence against a strong frozen baseline, robust performance, calibrated uncertainty when claimed, and export correctness.
- **Scientific completion:** primary questions resolved with the required controls and diagnostics; orphaned-control and incomplete-study rates.
- **Claim reliability:** agreement between registered claim, implemented intervention, measured endpoint, automatic verdict, and report explanation. Include correct inconclusive results and tested negatives as useful outputs.
- **Discovery and transfer:** a validated method outside the initial defaults, with demonstrated scope; successful reuse on a new task or dataset.
- **Efficiency conditional on quality:** time and cost to a confirmed result or resolved primary question, completed informative studies per budget, and inference cost for the exported product.

Use a holdout layer for the research process itself. Separate algorithm/training seeds from independently generated datasets and problem instances. Search-seed variability on one fixed dataset is informative but is not replication across scientific settings. For comparisons, bootstrap matched rows and recompute the actual global metric in each resample; use training-seed and task-level variation separately. A marginal interval on one method is not the interval for the paired method difference.

The existing `no_critic` arm changes critique, rounds, and pilots together, so it does not isolate the critic's effect. A future controlled ablation should vary those components separately, with a frozen method interface, equal declared budgets, and matched tasks. A component's effect on the next decision or on claim reliability may be informative even if its final nominal accuracy ties.

All 17 current manifests identify the same test-feature split. "Test once per run" is helpful protection, but repeated human development against that split still weakens its role as a final campaign holdout. Fresh-memory runs also cannot see test-access records in other stores. Track evaluation exposure centrally by split and dataset fingerprint; freeze V3 development before evaluating its final campaign holdout. The present evidence does **not** demonstrate that test labels were used for model selection.

The two current shift sets reveal useful limitations: tripled noise produces error around **0.051–0.054**, and the lower-bound shift produces roughly **0.076–0.125**. If robustness is part of the research objective, make an explicit development protocol for those shifts and preserve separate final stress tests. Do not repeatedly optimize against a nominally held-out shift set. Aggregate interval coverage on nominal data does not establish conditional or shifted calibration.

## Implementation order

| Priority | Work | Acceptance evidence |
|---|---|---|
| P0: trustworthy interpretation | Canonical problem/method contract; correct comparison predicates; per-subclaim outcomes; reporter evidence bundle and consistency checks; explicit fallback accounting; export checks. | The orphan mode arm cannot establish a mode/mean effect; "30% worse" has unambiguous semantics; the report recognizes the winning surrogate and existing diagnostics; exported prediction behavior is validated. |
| P1: complete studies | Study/dependency schema; stable comparison IDs; atomic matched-arm scheduling; shared-fit/trajectory reuse; typed diagnostics; reserved confirmation budget. | The heat initializer and mode/mean studies complete their specified controls and analyses, or are explicitly deferred together. No successful baseline masks completion failure. |
| P2: adaptive research | Feed main experiment results back into the scientific loop; choose the next study by unresolved alternatives and likely product value; verify implementation against the study. | A measured result changes the next action and resolves a primary question; no extra round is required when the decision is already determined. |
| P3: knowledge transfer | Evidence-type routing, claim-linked applicability, runnable memory packages, separate frozen-memory and cumulative-learning evaluations. | Reuse avoids a known invalid experiment or improves a new task; retrieved knowledge is supported and applicable. |
| P4: campaign evaluation | Independent task/data seeds; controlled component ablations; hard/soft budget distinction; protected campaign holdout; product-quality metrics. | V3 produces better confirmed results or more reliable scientific conclusions at a stated cost across tasks, not just more supported threshold predictions. |

Build P0 and P1 before increasing researcher count or round count. They address the largest observed loss of research value. P2 is the architectural step that makes full-scale experimental evidence drive the next research action.

## Per-run review

The automatic outcome counts below use S/C/I for supported/contradicted/inconclusive proposal predictions. They describe the current checker and are not independent scientific verdicts.

| Report | Test rel_l2 | S/C/I | Research contribution and main improvement |
|---|---:|---:|---|
| [full s0](../../ablation/heat_v2/runs/v2_full_s0/report.md) | 0.018412 | 5/0/0 | Strong criticism and useful start-count pilots. Best original research result improves the explicit physics baseline by only **0.054%** on validation. The mode/mean partner is abandoned, but the surviving mode arm is called supported; preserve the study dependency and its unresolved mechanism. |
| [full s1](../../ablation/heat_v2/runs/v2_full_s1/report.md) | 0.018301 | 5/0/1 | Custom posterior solver is the nominal winner; best original research validation is about **2.02%** below the physics baseline. Missing short-center-start control and nonzero reload difference prevent stronger mechanistic/reproducibility conclusions. Analyze paired effects and stochastic prediction behavior. |
| [full s2](../../ablation/heat_v2/runs/v2_full_s2/report.md) | 0.018418 | 4/0/2 | A custom simulation-amplified/few-step method executes and reaches physics-like accuracy, but required learning-curve/linear-control studies do not complete. Hybrid and physics are effectively tied. Reserve dependent controls before more variants. |
| [offline s0](../../ablation/heat_v2/runs/v2_offline_s0/report.md) | 0.018334 | 4/2/0 | A very strong, cheap control: heuristic physics-MMSE solves the task. The MLP threshold and centered-MAP comparison miss their registered targets. Do not attribute the physics win to LLM research. |
| [offline s1](../../ablation/heat_v2/runs/v2_offline_s1/report.md) | 0.018392 | 4/2/0 | Repeats the heuristic solution; best learned result is better after refinement but still above physics. Same two checker contradictions; measure matched uncertainty before treating small MAP/MMSE differences as established. |
| [offline s2](../../ablation/heat_v2/runs/v2_offline_s2/report.md) | 0.018402 | 4/2/0 | Confirms that the nominal forward-enabled result is available without an LLM. Useful for regression testing and establishing the remaining scientific questions before costly debate. |
| [no critic s0](../../ablation/heat_v2/runs/v2_no_critic_s0/report.md) | 0.018251 | 4/0/2 | A refined MLP-init hybrid wins, but proposed MLP/hybrid controls are projected out. Uses **491.21 worker-s** against the 400 target. Strong score does not isolate the critic effect or establish proposed mechanisms. |
| [no critic s1](../../ablation/heat_v2/runs/v2_no_critic_s1/report.md) | 0.018305 | 4/1/1 | Registers a sharper 5% improvement claim and misses it at **1.51%**. This is more informative than an easy threshold, but needs a named fixed comparator and paired interval. Uses **432.15 worker-s**. |
| [no critic s2](../../ablation/heat_v2/runs/v2_no_critic_s2/report.md) | 0.018405 | 5/0/1 | Physics-MMSE wins; the "beats MLE" claim is checked against a data-driven model in one prediction, so mechanism and comparator do not align. Reload difference **0.00255** needs explanation. |
| [no literature s0](../../ablation/heat_v2/runs/v2_no_literature_s0/report.md) | 0.018253 | 5/0/1 | Good ideas include an MC-sample-count control and a two-step hybrid. The simulation learning-curve study does not produce a full result. Several controls exist only in proposal prose; preserve and execute them as studies. Reload difference **0.00686**. |
| [no literature s1](../../ablation/heat_v2/runs/v2_no_literature_s1/report.md) | 0.018339 | 3/0/3 | A simulation-amplified custom learner executes, but comparison against weak extra trees does not isolate simulation count. Final ensemble is strong. One custom MMSE method is rejected for a missing top-level factory; add early contract checks and repair before allocating study budget. |
| [no literature s2](../../ablation/heat_v2/runs/v2_no_literature_s2/report.md) | 0.018326 | 5/1/0 | Stops on quality after **two** rounds, unlike most LLM multi-round runs. A 2% mean/mode prediction achieves **1.77%**; distinguish threshold miss from evidence of no effect. Its discussion misidentifies the intervention. |
| [memory s1](../../ablation/heat_v2/runs/v2_memory_s1/report.md) | 0.018272 | 2/1/0 | Executes physics readout/MC comparisons but none of the proposed learning-curve endpoints. The **0.47% vs 0.5%** near miss concerns sample count, not mode/mean. Uses **442.73 worker-s** and has a **0.00816** reload difference. |
| [memory s2](../../ablation/heat_v2/runs/v2_memory_s2/report.md) | 0.018419 | 0/0/4 | All selected research predictions are inconclusive; final winner is the physics baseline. Memory retrieval is functioning, but the research product is incomplete. Report this separately from task success, and schedule a complete feasible study using what memory already established. |
| [no forward s0](../../ablation/heat_v2/runs/v2_no_forward_s0/report.md) | 0.021039 | 5/0/1 | Rich custom-method exploration and a useful final ensemble. A KRR implementation fails cloning; several target/feature comparisons lose their twins. The final stage costs **120.3 worker-s**, making shared work and reserved confirmation especially valuable. |
| [no forward s1](../../ablation/heat_v2/runs/v2_no_forward_s1/report.md) | 0.018497 | 4/0/2 | Strongest formulation result: learned forward surrogate plus inversion. Make its actual algorithm and fidelity diagnostics central to the report. Replicate it and compare weighted/unweighted inversion with matched direct regression; do not describe the merged solver line as never executed. |
| [no forward s2](../../ablation/heat_v2/runs/v2_no_forward_s2/report.md) | 0.019521 | 2/0/4 | The custom emulator/inversion method beats its best data-driven result by **11.4%** on validation. Four planned controls have no full result; stage-budget failures and procedural abandonments must remain distinct from scientific negatives. |

## Qualifications to the existing aggregate writeup

The prior [ablation writeup](ABLATION_RESULTS_v2.md) captures the broad numerical result, but several statements need narrower interpretation:

- There are **17 reports/runs** in the reviewed set, consistent with five three-seed arms plus a two-seed memory arm; the introduction says 18.
- Reload differences are not all zero, as listed above.
- `memory_s2` reads an expanded store including `memory_s1`, not only the original seed run.
- The no-critic result combines several treatment changes and includes budget overshoot.
- Forward-enabled performance is **consistent with a near noise-limited regime**, not a demonstrated exact information limit. The CRB computation is a local, finite-draw diagnostic for an unbiased-estimator comparison; bounded and posterior-mean estimators need not obey it as a universal error floor. Its scalar aggregate is computed from summed variances, while per-parameter summaries use medians; these should not be conflated.
- `full_s2` has two literature requests skipped for budget and `no_forward_s2` has four. Service reliability and completeness of requested evidence are separate properties. Literature-disabled requests are expected disabled events, not service outages.
- There are **35 heuristic-fallback actions among 312 actions in LLM-configured runs**, distributed across eight runs, despite zero recorded LLM-service-failure artifacts in this reviewed set. The counts are actions, not failed calls. Parsing or response-shape handling can also produce fallback; record its actual reason and avoid describing every research response as LLM-authored.
- Best-baseline comparisons require attribution care: the portfolio can deduplicate a baseline into a proposal marked `also_baseline`. Counting only entries with `origin=baseline` can make a conventional method disappear from the baseline group. Separate method identity from who proposed it.

These are analysis findings; the existing writeup was left unchanged.

## Source locations relevant to the plan

| Current source | Relevant behavior |
|---|---|
| [metrics.py](../../mlharness/metrics.py) | Actual global Frobenius metric and gain convention. |
| [research/proposals.py](../../mlharness/research/proposals.py) | Single structured prediction and `expect=worse` comparison semantics. |
| [research/portfolio.py](../../mlharness/research/portfolio.py) | Individual-method family/quality selection and baseline deduplication. |
| [research/loop.py](../../mlharness/research/loop.py) | Pilot request handling, debate rounds, and stopping criteria. |
| [research/critic.py](../../mlharness/research/critic.py) | Citation/completeness quality score, review requests, and abbreviated shared context. |
| [research/researchers.py](../../mlharness/research/researchers.py) | Literature-first proposals, source-question routing, revision/fallback behavior, and abbreviated peer views. |
| [scheduler_v2.py](../../mlharness/scheduler_v2.py) | Full evaluation follows research; local refinement, family-reference prediction checks, final selection, and memory writeback. |
| [agents/reporter_v2.py](../../mlharness/agents/reporter_v2.py) | Report generation and limited LLM discussion summary. |
| [resources.py](../../mlharness/resources.py), [executor.py](../../mlharness/executor.py) | Stage caps, post-completion charging, concurrency, timing, final-stage exemption, and experiment cache. |
| [runner.py](../../mlharness/runner.py), [inversion.py](../../mlharness/inversion.py) | Final prediction/save/reload order, sampling behavior, and augmentation implementation. |
| [memory/store.py](../../mlharness/memory/store.py), [run_ablation_v2.py](../../ablation/heat_v2/run_ablation_v2.py) | Retrieval, local test-exposure tracking, and evolving shared-memory treatment. |

The review's central recommendation is to keep V2's method freedom, empirical pilots, criticism, and provenance, and make V3 accountable for the complete chain **question → implemented study → required evidence → justified conclusion → validated research artifact**.
