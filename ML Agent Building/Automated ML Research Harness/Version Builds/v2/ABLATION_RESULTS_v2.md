# Ablation study: the V2 research-first harness on the V1 heat benchmark

**Problem:** the same benchmark as V1. Recover the four piecewise-constant thermal conductivities of a 1-D
rod from noisy temperature readings at four sensors.
**Harness:** Automated ML Research Harness V2 (`mlharness research`, `pipeline: v2`).
**Runs:** 6 configurations, 18 valid harness runs, completed 2026-11-09. All runs used the V1 data, budget
and protocol. **Comparison:** the 15 V1 runs ([ABLATION_RESULTS_v1.md](../v1/ABLATION_RESULTS_v1.md)) and
the physics reference.

---

## TL;DR

| Configuration | What it removes | Test rel_l2 (mean ± sd) | vs V1 full | LLM $/run | Wall/run |
|---|---|---|---|---|---|
| `v2_full` | nothing (4 LLM researchers ⇄ LLM critic, TheoremSearch, pilots, forward model) | **0.01838 ± 0.00007** | −22.3 % (p = 0.008) | $4.33 | 12.6 min |
| `v2_offline` | the LLM (heuristic researchers + rule-based critic) | **0.01838 ± 0.00004** | −22.3 % (p = 0.008) | $0 | 1.0 min |
| `v2_no_critic` | the critique → revision cycle and the pilots (1 round) | 0.01832 ± 0.00008 | −22.5 % | $0.81 | 4.9 min |
| `v2_no_literature` | TheoremSearch | 0.01831 ± 0.00005 | −22.6 % | $3.29 | 11.9 min |
| `v2_memory` (n=2) | nothing: reads the memory `v2_full` s0 wrote | 0.01835 ± 0.00010 | −22.4 % | $3.95 | 15.3 min |
| `v2_no_forward` | forward-model access | 0.01969 ± 0.00128 | −16.7 % (p = 0.015) | $4.91 | 15.8 min |
| *V1 full harness* | — | *0.02365 ± 0.00083* | — | *$0.83* | — |
| *Physics reference (outside the harness)* | — | *MLE 0.01842 · Cramér–Rao floor 0.01923* | — | — | — |

Lower rel_l2 is better: ‖Ŷ − Y‖_F / ‖Y‖_F on the untouched 750-sample test set. Welch p-values, n = 3 per arm
(n = 2 for `v2_memory`).

**Main conclusions**

1. **V2 closes the V1 gap: every V2 arm beats every V1 arm, at p < 0.02.** The five arms with forward-model
   access all land at **0.0183–0.0184**. That is the level of the stand-alone physics MLE (0.01842), which
   in V1 had to run outside the harness. V2 achieves the stated goal of §8 of the spec: physics and hybrid
   methods are now expressible, testable and exportable in the harness. All 14 forward-arm runs exported a
   method that uses the forward model.
2. **On this benchmark, the gain comes from forward-model access, not from the research loop.**
   `v2_offline` (no LLM, $0, about 1 minute) ties `v2_full` (Welch p = 0.98). The physics-MLE baseline, which
   V2 adds automatically, already reaches the noise floor. In `v2_full` s0 the best research proposal and the
   best baseline differ by 0.05 %.
3. **Critic, literature and memory have no measurable effect on accuracy here, because the benchmark is
   saturated.** Removing any of them changes test error by ≤ 0.4 % (p ≥ 0.11). This is a ceiling effect,
   not evidence that these components are useless. Every arm sits at the information limit of the data,
   so no arm can be better. The critic's cost is visible: `v2_no_critic` reaches the same accuracy for
   one fifth of the LLM spend and about 40 % of the wall time.
4. **The one arm where research mattered is `v2_no_forward`. The agents found approaches outside the V1
   registry there.** All three of its final models are agent-written `custom_solver`s. In seed 1 the agent
   built a **learned surrogate forward model** (kernel ridge from log k to PCA-compressed sensor readings,
   using training data only), then ran batched Levenberg–Marquardt inversion through that surrogate. It
   reached **0.0185**, which is physics-MLE accuracy without access to the physics. The arm averages
   0.0197: 17 % better than V1 (p = 0.015), and 7 % worse than the forward arms (p = 0.22, seed variance is
   high).
5. **The improvement is concentrated where V1 was weakest: the inner segments.** Per-segment test error,
   V1 full → V2 full: k₁ 0.0335 → 0.0269, **k₂ 0.0114 → 0.0061, k₃ 0.0099 → 0.0052**, k₄ 0.0301 → 0.0242.
   The inner segments now match their per-segment Cramér–Rao values (0.0064 / 0.0052).

> **Caveats:** n = 3 per arm, and the benchmark is saturated. Conclusions 2–3 say "no detectable
> difference on this problem". They do not say "no difference". §5 proposes a harder benchmark that
> could separate the research components.

---

## 1. Setup

The data, metric and budget are identical to V1 ([§1–2 of the V1 report](../v1/ABLATION_RESULTS_v1.md)):

- Simulator [examples/heat_diffusion_simulator.py](../../examples/heat_diffusion_simulator.py), 4 segments,
  80 features, noise σ = 10⁻³.
- Data seed 5, split 3000 / 750 / 750.
- Budget: 400 worker-seconds of execution compute, 120 s per experiment, 6 workers, CPU only.
- Success threshold: rel_l2 ≤ 0.05.
- The search seed is the run seed (0, 1, 2), and runs execute one after another.

V2-specific settings were held constant across arms:

- Research: 3 rounds (minimum 2), 2 proposals per researcher, quality threshold 0.75, 30 research LLM
  calls, pilots on.
- TheoremSearch: up to 32 queries, 4 results each.
- Portfolio: 6 proposals plus baselines (conventional ML, random search, physics MLE when allowed). Then
  1 refinement round and an ensemble.
- Shift sets: 300 samples each, `noise_x3` (σ = 3·10⁻³) and `k_low_0.3`.

| Arm | LLM | Critic | Rounds | Pilots | Literature | Forward model | Memory |
|---|---|---|---|---|---|---|---|
| `v2_offline` | none (heuristic) | rule-based | ≤3 | ✓ | ✓ | ✓ | fresh |
| `v2_full` | Claude Code | LLM | ≤3 | ✓ | ✓ | ✓ | fresh |
| `v2_no_forward` | Claude Code | LLM | ≤3 | ✓ | ✓ | **✗** | fresh |
| `v2_no_critic` | Claude Code | **✗** | **1** | **✗** | ✓ | ✓ | fresh |
| `v2_no_literature` | Claude Code | LLM | ≤3 | ✓ | **✗** | ✓ | fresh |
| `v2_memory` | Claude Code | LLM | ≤3 | ✓ | ✓ | ✓ | **seeded from `v2_full` s0** |

The driver is [run_ablation_v2.py](../../ablation/heat_v2/run_ablation_v2.py) and the analysis is
[analyze_v2.py](../../ablation/heat_v2/analyze_v2.py). The raw results are in
[summary_v2.json](../../ablation/heat_v2/summary_v2.json), and each run's report, research trace and
exported model are in `ablation/heat_v2/runs/<arm>_s<seed>/`.

## 2. Per-run results

| Run | Test rel_l2 | Final method | Lit. queries / cited | Memory refs | Custom-code proposals |
|---|---|---|---|---|---|
| full s0 | 0.01841 | hybrid ridge → LM20 | 24 / 16 | 0 | 0 |
| full s1 | 0.01830 | custom solver (forward) | 24 / 10 | 0 | 3 |
| full s2 | 0.01842 | hybrid extra-trees → LM15 | 24 / 16 | 0 | 7 |
| offline s0–s2 | 0.01833 / 0.01839 / 0.01840 | physics-MMSE (all three) | 15 / 3 | 0 | 0 |
| no_critic s0 | 0.01825 | hybrid torch-MLP → LM12 | 8 / 9 | 0 | 0 |
| no_critic s1 | 0.01831 | hybrid torch-MLP → LM10 | 8 / 9 | 0 | 0 |
| no_critic s2 | 0.01841 | physics-MMSE | 8 / 8 | 0 | 0 |
| no_literature s0 | 0.01825 | physics-MMSE | 0 (disabled) | 0 | 3 |
| no_literature s1 | 0.01834 | ensemble: 2 hybrids + physics-MLE | 0 | 0 | 3 |
| no_literature s2 | 0.01833 | hybrid ridge → LM15 | 0 | 0 | 0 |
| memory s1 | 0.01827 | physics-MMSE | 24 / 12 | **17** | 4 |
| memory s2 | 0.01842 | physics-MLE | 24 / 10 | **17** | 5 |
| no_forward s0 | 0.02104 | ensemble: 2 custom solvers + 2 torch-MLPs | 24 / 15 | 0 | 9 |
| no_forward s1 | 0.01850 | custom solver (learned surrogate + LM) | 22 / 13 | 0 | 6 |
| no_forward s2 | 0.01952 | custom solver | 26 / 11 | 0 | 6 |

Every run met the success criterion. The exported models reload bit-exactly (`reload_max_abs_diff` = 0).

## 3. What the V2-specific components did

- **Four researchers and the critic.** In every multi-round run, all four personas responded to feedback
  in every round (`personas_responding_by_round`). `v2_full` averaged 23 proposal versions from 8
  originals, with 3.7 abandonments. Every abandonment has a recorded reason. `v2_offline` stopped early
  on the quality criterion after 2 rounds. The LLM arms ran to the 3-round cap, with unresolved issues
  documented.
- **Breadth of methods.** The LLM arms proposed physics, hybrid, data-driven and agent-coded methods
  (up to 9 custom-code proposals per run). The heuristic arm never wrote custom code. The gate approved
  them, and the runtime sandbox executed them.
- **TheoremSearch.** No service failures occurred in any run (0 failed queries outside the disabled arm).
  The full arms cited 10–16 retrieved results per run. In `v2_no_literature`, every query attempt is
  correctly logged as `disabled`.
- **Cross-run memory.** Both `v2_memory` runs retrieved 17 records written by `v2_full` s0 and cited them in
  their proposals. Accuracy did not change. This is expected: s0's own result was already at the floor.
- **Uncertainty.** Physics and hybrid exports report Gauss–Newton σ. The 2σ test coverage is 0.95–0.96
  (nominal 0.954), so the reported uncertainty is well calibrated.

## 4. Robustness and cost

| Arm | `noise_x3` rel_l2 | `k_low_0.3` rel_l2 | Inference latency | Compute used (of 400 s) |
|---|---|---|---|---|
| full | 0.0518 | 0.097 | 3–23 ms/sample | 242 s |
| offline | 0.0512 | 0.112 | 9 ms | 140 s |
| no_forward | 0.0532 | 0.093 | 3–36 ms | 295 s |
| no_critic | 0.0512 | 0.103 | 14–36 ms | 436 s |
| no_literature | 0.0513 | 0.101 | 33–58 ms | 329 s |
| memory | 0.0516 | 0.118 | 8–73 ms | 345 s |

- **Tripled sensor noise** raises error about 2.8× in every arm. That is close to the 3× expected for an
  estimator limited by noise.
- **The `k_low_0.3` shift draws conductivities below the prior's lower bound of 0.5.** The physics and
  hybrid methods use bounded optimization, so they cannot reach those values, and the error is structural
  (0.09–0.12). The data-heavy `no_forward` arm degrades least, but only slightly. Any future benchmark
  should state whether out-of-prior shift is in scope.

## 5. Threats to validity and what to run next

1. **The benchmark is saturated.** Every arm with forward-model access reaches the information limit, so
   accuracy cannot separate the critic, the literature or the memory. Mean test error (0.0183) is slightly
   *below* the Cramér–Rao floor (0.0192). This is not a contradiction: the CRB bounds unbiased estimators,
   while the bounded MMSE and MAP estimators trade a little bias for variance using the prior. To test the
   research components, use a problem where the obvious baseline does *not* reach the floor. Options:
   - 8 or more segments with the same 4 sensors (poorly identified);
   - forward-model misspecification, such as a wrong boundary condition or unknown initial-pulse width;
   - a no-forward-model setting like `v2_no_forward`, which is the only arm that showed spread between
     seeds;
   - dropping the automatic physics baseline, so that only research proposals can reach physics accuracy.
2. **Small n.** n = 3 per arm (n = 2 for memory), and LLM output is non-deterministic.
3. **Pruning uses wall-clock timing.** The screen prunes on projected cost, which is extrapolated from
   wall time. Background load can therefore change which method wins (see run log, below). Measuring CPU
   time instead of wall time would make the pruning robust to machine load.
4. **Disturbances during execution.** The ablation took four sessions. Every excluded or reviewed run is
   listed in the run log below, and the excluded ones are kept under
   `ablation/heat_v2/runs_degraded_2026-11-09/`.

### Run log (validity)

| Issue | Runs affected | Handling |
|---|---|---|
| Claude 5-hour usage limit hit during the first pass (2026-11-09 00:51 onward). LLM calls failed and the harness silently used heuristics. | 10 runs (no_forward ×3, no_critic ×3, no_literature s1–s2, memory ×2) | All redone. The driver now runs an LLM check before each LLM run, marks runs with failed LLM calls `LLM_DEGRADED.json`, and stops on a usage limit. The analyzer excludes marked runs. |
| Usage limit again during the second pass (11:44) | no_literature s2 (2 failed calls) | Redone after the reset. |
| Laptop lid closed (Modern Standby 10:05–10:21) | the first no_forward s1 attempt | Aborted and redone. The driver now reads the Windows Kernel-Power log and marks runs `SUSPENDED.json`. It also blocks idle sleep while it runs. |
| Windows Update and a Defender signature update (10:34–10:39) made every screen job about 10× slower | the second no_forward s1 attempt (test 0.0635, ridge final) | Excluded with a note, then redone (0.0185). |
| Short standbys during `v2_memory` (s1: 30 s in the final stage, after selection; s2: about 5 min while waiting on research LLM calls, with no experiment running) | memory s1, s2 | Reviewed and accepted. The stop reason and selection were unaffected, and a note is kept in `SUSPEND_REVIEWED.json`. |

## 6. Spec conformance (V2 acceptance criteria, §11)

| Criterion | Status | Evidence |
|---|---|---|
| Four specialized researchers propose in parallel; all four respond to the critic across rounds | ✓ | `personas_responding_by_round` lists all four personas in every round of every multi-round run |
| Halts deterministically on quality, max rounds or budget, preserving partial work | ✓ | Stop reasons: quality (offline), round cap (LLM arms). Tests `test_iteration_cap_and_quality_stop` and `test_llm_budget_halts_and_preserves_partial_work` |
| Critic gives actionable revisions and keeps proposals diverse | ✓ | Critique artifacts and memory records include challenges, changed assumptions and revision directions. Portfolios mix four families. |
| TheoremSearch callable, with attribution and applicability checks; outage behavior tested | ✓ | 0 live failures. Outage tests pass. |
| Findings, failures, critiques and references persist and are retrievable in another run | ✓ | `v2_memory` retrieved 17 prior records |
| A hypothesis can use the forward operator | ✓ | 14/14 forward-arm runs exported a method that uses the forward model |
| Multiple methods executed safely, compared and selected under budget | ✓ | Gate, sandbox and screen. 36/36 unit tests pass. |
| Model exported with reproducibility metadata and lineage | ✓ | `final/manifest.json`, `model_card.md`; reload diff 0 |
| Report separates established evidence, conjectures, failed hypotheses and uncertainty | ✓ | Each run's `report.md` |
| Remote AMATH GPU execution | ◐ | Implemented and tested against a fake SSH server only. Not run against a real host in this study (the benchmark is CPU-only). |
| "Concise implementation report" (§10 deliverable) | ✗ | Referenced in README_V2 but not yet written |
