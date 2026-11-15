# Independent V3 software validation

The independent Sol review agent inspected the implementation separately from its authors, added adversarial fixtures in `tests/test_v3_review.py`, returned findings for repair, and retested the repairs. Validation uses Python 3.11 and tiny local software fixtures. No live LLM, literature, SSH or other service was called; no scientific research campaign was launched; no credentials were read. Historical runs and the user's goal document were preserved.

## Test results

| Run | Result | Evidence |
| --- | --- | --- |
| Supplied pre-migration baseline | 36 passed in 194.931 seconds; coordinator evidence | `baseline_tests.log` |
| Initial independent V3 module run | 45 passed in 2.533 seconds | Recorded during review before later integration cases |
| First independent complete regression | unittest reported 85 passed in 202.100 seconds | `independent_full_tests.log` |
| Final adversarial review retest | 10 passed in 13.672 seconds; process exit 0 | `independent_review_tests.log` |
| Complete regression before final resume hardening | **93 passed in 201.504 seconds; process exit 0** | `independent_final_tests.log` |
| Frozen-source V3 focused verification | **57 passed in 28.247 seconds; process exit 0** | `independent_final_focused_tests.log` |
| CLI help smoke check | Exit 0; exposes `research --pipeline {v2,v3}` | Independently invoked Python 3.11 CLI |

The first full run used PowerShell `*>` redirection, which wrapped normal unittest stderr in `NativeCommandError` records and returned shell exit 1 despite unittest's `OK`. The final runs use a Python subprocess wrapper that combines output in a UTF-8 log and propagates unittest's actual process exit. The earlier 85-test run predates the final uncertainty-identity, critic-scope, alias-dependency and implicit completion-obligation regressions; it is not substituted for the final run.

After the 93-test regression, the final V3-only resume patch added identical compiled-spec checks, continued experiment IDs and resource/role accounting, and recovery from persisted final ledger evidence when completion pointers are missing. The ensemble selection helper restores its temporary configuration override. These changes were inspected independently and are covered by the separate frozen-source V3 run; the report does not mislabel the earlier full run as testing later code.

All recorded independent regression findings have been repaired and the affected checks pass. No known failing test remains. `git diff --check` is clean for the independently owned test and results files. The delivery boundaries below remain applicable.

Reproduce from the harness directory:

```powershell
$py = 'C:/Users/kusha/AppData/Local/Programs/Python/Python311/python.exe'
& $py -m unittest discover -s tests -v
& $py -m unittest discover -s tests -p test_v3_review.py -v
& $py -m mlharness research --help
```

## Independent findings and repair verification

The initial review found unsupported scientific promotion. Scalar thresholds ignored required uncertainty; comparisons ignored unregistered varying factors; unrelated paired intervals could support another metric or pair. The repaired validators now reject these cases. Completion also distinguishes missing required uncertainty or verification from a genuine measured interval crossing its decision boundary: missing work is incomplete, while an executed inconclusive result can complete a study.

Runtime review found exposure reservation without attempted-access marking, incorrectly named contract fingerprint keys, critic objections added after support had been assigned, stale study IDs during merge, mutable arm identities, dependency edges treated as study IDs, and duplicate-control aliases preventing dependent arms from becoming ready. Source repairs were inspected. The independent integrated fixture directly verifies duplicate controls plus a dependency on an alias, adaptive follow-up, actual researcher-initiated source-specialist work, certified export and completed-run resume. Unrelated evidence or an unrelated revision cannot resolve a critic objection.

Protocol validation initially raised exceptions for malformed collections and unhashable instance identities. Those cases now yield invalid records. Package loading and frozen snapshot restoration are wired into the runtime, and final exposure reservations release through exception paths.

The independent integration fixture uses 30 training, 12 validation and 10 final rows, two same-persona researchers, two distinct ridge configurations and a deduplicated control. A fake researcher commissions a local source specialist and receives its result. A deliberately contradicted registered margin produces a subsequent diagnostic action. The fixture checks completed study status, actual callback results, manifest/result certificate equality, model/report/evaluation artifacts, and no additional exposure or model rewrite on completed-run resume. This is a software behavior fixture, not evidence of scientific discovery quality.

## Acceptance evidence and boundaries

| IDs | Observed evidence | Boundary |
| --- | --- | --- |
| T0.1-T0.2 | Actual dimensions and metric/noise contracts; normalization changes; min/max, worse, equivalence, finite and zero-reference semantics; per-claim results; uncertainty and intervention identity checks | Metrics and effects are checked on bounded fixtures, not all scientific problems |
| T0.3 | Absent diagnostics, unverifiable implementation, unenforced gates, mechanism and proof obligations cannot become support; missing claim work remains incomplete | Generic mechanism verification and formal proof adapters remain unsupported |
| T0.4 | Deterministic pristine export passes repeat/reload/partition checks; stateful, batch-sensitive and nonfinite fixtures fail; failed certification avoids final arrays; manifest/result certificates agree | Certification covers the declared inputs/tolerances, not all future inputs or an untested stochastic protocol |
| T0.5 | Factual reporting consumes one persisted bundle; actual methods, controls, diagnostics, uncertainty and fallbacks are rendered; rebuild is byte-identical in the runtime fixture | No live-model narrative or research-quality evaluation was performed |
| T1.1-T1.2 | Required failed controls remain visible; negative studies can complete; paired bootstrap recomputes the global metric; deduplication preserves control obligations and dependent execution; runtime fixture exercises immutable prediction sharing | Generic fitted-state and solver-trajectory sharing is deferred |
| T1.3 | Concurrent atomic reservations, protected confirmation budget, failure reconciliation, protocol-aware caching and explicit worker timeout fixtures | Callback orchestration is synchronous; startup/termination overhead can exceed timeout bounds; soft budgets remain estimates |
| T2.1-T2.2 | Unique duplicate-persona identities; actual nested callback API execution; researcher source specialist and parent interpretation through the scheduler; capability/count/depth caps | Fixture agents and fake models demonstrate plumbing, not improved research reasoning; there is no universal preemptive callback wall timeout |
| T2.3 | Paired supported/contradicted evidence views select different actions; full contradicted integration evidence triggers a later diagnostic | Adaptive behavior is verified for scripted/offline policies, not a frontier-model campaign |
| T2.4 | Independent critic identity and objection ledger; new blocking objections downgrade support; unrelated evidence/revisions are rejected; provider-neutral role routing/escalation is present | General scientific adequacy of a critic-approved prose resolution is not mechanically proved |
| T3.1 | Local queries avoid theorem calls; artifact/diagnostic/memory/theorem routes, source hashes, caches, failures and unknown applicability are exercised | Comprehensive empirical literature coverage requires a configured provider |
| T3.2 | Durable custom package runs after source removal; tampering and gate rejection fail; package evidence routing reapplies the gate; recursive held-out projection and frozen/cumulative views are exercised | No independent transfer-success or decision-benefit campaign was run; broad reuse of every custom dependency/environment is untested |
| T4.1-T4.3 | Separate dataset/model seeds, confounded protocol rejection, frozen identity, global persistent exposure reserve/begin/complete, concurrent strict refusal, resume idempotence and incomplete/negative program metrics | Discovery/transfer remain unknown with explicit reasons; independent-instance improvement is unmeasured |
| R1 | Full discovery includes existing V1/V2 software tests, including sandboxing, fake remote/retrieval, physics batch independence, report/export and resume | Prior scientific runs were not rerun or reinterpreted |
| R2 | Offline and fake-researcher integration fixtures produce studies, evidence bundle, trace, report, program metrics, exported model and certificate | CLI dispatch/help and artifact behavior are validated; no real service or new scientific campaign was used |

## Delivery limits

This validation supports the implemented software contracts and bounded integrated paths. It does not establish that V3 discovers better science than V2, that more agents improve outcomes, or that a stored package transfers successfully to an independent problem instance. Formal proof checking, arbitrary shared fitted-state/trajectory adapters, comprehensive empirical literature retrieval, and universal callback preemption are not delivered capabilities. Unsupported requirements remain explicit or unresolved instead of being certified.
