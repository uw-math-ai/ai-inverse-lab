# Using the V3 research harness

Run commands from `ML Agent Building/Automated ML Research Harness`. The migration adds `pipeline: v3`; V1 and V2 remain available. The example configuration below was inspected and compiled during development. It was not used to launch a scientific experiment.

## Validate the software

Use the same Python environment as V2. On this workstation:

```powershell
$py = 'C:/Users/kusha/AppData/Local/Programs/Python/Python311/python.exe'
& $py -m unittest discover -s tests -v
& $py -m mlharness research --help
```

The suite uses local software fixtures and fake services. See [V3_TEST_RESULTS.md](V3_TEST_RESULTS.md) for the independently verified result and remaining limits.

## Start a future research run

[v3_example.yaml](../../specs/v3_example.yaml) defines a two-segment heat inverse problem and an explicit matched ridge comparison. Both arms, the varying factor, full fidelity, seed, paired diagnostic, improvement margin and uncertainty requirement are registered together. An inconclusive or contradicted result is a valid study outcome.

```powershell
& $py -m mlharness research --pipeline v3 --spec specs/v3_example.yaml --backend none --run-dir runs/v3_example
# Equivalent dispatch from the pipeline field:
& $py -m mlharness run --spec specs/v3_example.yaml --backend none --run-dir runs/v3_example
```

Choose one invocation. These commands launch research and were not executed as part of implementation. The default offline policy exercises registered studies and evidence-dependent follow-up; it does not demonstrate the research quality of a frontier-model team. The example has one dataset instance and cannot establish transfer or component-level causal benefits.

`research` defaults to V2 unless `--pipeline v3` is supplied. `run` selects V3 from the spec's `pipeline` field. A V2 spec can be adapted by setting that field and adding explicit `program.studies`; historical proposal predictions are not silently promoted into V3 claim evidence.

The example denies forward-model access and disables network literature and memory. Enable these explicitly for a research program that needs them. Method kinds and physics access use V2's existing implementation gate and isolated runner. Custom code must satisfy the existing `build_model` or `build_method` interface and passes static and runtime checks.

## Configure the research program

`program.questions` supplies initial questions. `program.studies` supplies registered studies; model-driven researchers can propose more at runtime. `program.max_actions`, `max_studies`, `max_agents`, `max_active_agents`, `max_depth`, `max_children`, `max_agent_turns` and `max_specialist_tasks` bound the program. Agent instance IDs remain distinct from their personas, so multiple mathematical, ML, experimental or exploratory researchers can coexist.

`program.roles` maps role names to the existing backend/model/effort configuration. Supported roles include `orchestrator`, `researcher`, `critic` and `specialist`; `<role>_escalated` supplies an optional escalation configuration. With an empty map, roles inherit `llm`. Choose available model IDs in your configured backend; tier names do not assert model capability. The artifact ledger records the selected model, calls, cost and fallback reason.

The orchestrator can decompose questions, spawn/assign researchers, register/execute/revise/merge/defer studies, request evidence or diagnostics, delegate, resolve objections, escalate and stop. Researchers can commission coding/execution, analysis, retrieval and source specialists. Nested callbacks return evidence or failures to the parent. The critic retains its own identity and objections; missing evidence cannot be resolved by assertion.

`program.budget_mode: soft` admits groups using estimates and records measured overruns. `hard` reserves timeout bounds for required groups, including final export; process startup/termination overhead is still measured and may exceed those bounds. `confirmation_fraction` protects a portion of the total compute budget for export. Callback orchestration itself is synchronous; finite task counts and underlying tool/runner timeouts should not be described as a universal preemptive callback timeout.

## Read and reuse evidence

A study owns its required arms, diagnostics, claims and completion state. Required controls are retained even when they perform poorly. Failed or missing work remains visible. Exact repeated executions can share persisted results without dropping the scientific obligations of their consumers.

Evidence requests declare `source_type`: `local`, `artifact`, `diagnostic`, `memory`, `theorem`, `empirical_literature` or `method_package`. Local configuration questions resolve against canonical problem/source metadata. Artifact and memory views recursively withhold held-out results. Theorem retrieval supplies attributed leads and assumption information; unknown applicability does not certify a claim. Empirical-literature retrieval requires an explicit provider.

Runnable packages are content-addressed under `.research_memory/v3_packages/`. A `method_package` request includes `package_id`; loading verifies content and reapplies the implementation gate. Runtime fit/predict validation is still required. Packages retain provenance and applicability; availability is not evidence of successful transfer.

`memory.mode: frozen` persists a safe snapshot for resume. `cumulative` permits newly accumulated records. Evaluation protocols distinguish dataset seeds from training/search seeds and can declare frozen baseline identities, memory snapshots, component changes and split roles. Validate a protocol without running research:

```python
from mlharness.knowledge_v3 import validate_evaluation_protocol
result = validate_evaluation_protocol(protocol_dict)
assert result["valid"], result["errors"]
```

## Inspect the research product and resume

The run directory contains:

- `artifacts.jsonl`: append-only actions, agents, plans, checks, results and lineage.
- `evidence_bundle.json`: canonical problem contract, studies, claims, diagnostics, critic objections, costs and final artifact evidence.
- `research_trace_v3.json`: program decisions and development evidence.
- `report.md` and `program_evaluation.json`: deterministic factual reporting and model/completion/reliability/reproducibility/cost measures.
- `final/model.pkl` and `final/manifest.json`: the pristine fitted artifact, configuration/source identities and export certificate when finalization is reached.

Export certification checks repeat, reload and partition behavior on development inputs before final evaluation. Uncertified candidates do not receive final metrics. Certification describes the checked inputs and protocol; it is not a mathematical guarantee about all future inputs. A strong final score cannot make an incomplete study complete or establish a mechanism.

```powershell
& $py -m mlharness report runs/v3_example --rebuild
& $py -m mlharness research --pipeline v3 --spec specs/v3_example.yaml --backend none --run-dir runs/v3_example --resume
```

Resume requires the same compiled specification; changed settings are rejected. Completed-run resume returns persisted output without another final evaluation. If a completion pointer is missing but the final result is already in the ledger, recovery reuses that result. A separate SQLite registry at `.cache/v3_exposure.sqlite` tracks reserved, attempted and completed held-out exposure even when memory is disabled. `evaluation.exposure_registry` can select a shared registry path. The example uses `strict_holdout: true`, so a second independent run on the same held-out split is refused. Interrupted attempted access without a recoverable result remains uncertain and cannot be retried as a fresh holdout.

Generic shared fitted-state/solver-trajectory adapters and formal proof verification remain unsupported; studies requiring them retain unresolved obligations. Immutable prediction reanalysis is supported. No discovery, transfer advantage or improvement over V2 across real problem instances was established by this software validation.
