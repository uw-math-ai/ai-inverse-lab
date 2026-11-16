# Ablation study: the V3 research harness on the V1/V2 heat benchmark

**Problem:** the same benchmark as V1 and V2. Recover the four piecewise-constant thermal conductivities of a 1-D
rod from noisy temperature readings at four sensors.
**Harness:** Automated ML Research Harness V3 (`mlharness research --pipeline v3`).
**Status: interim (2026-11-15).** 5 of the 7 planned core runs are valid. The remaining two Fable-orchestrated
seeds (`v3_full` s1, s2) are blocked by the weekly Fable usage limit until **2026-11-19 11:00 PT** (see §7).
**Scope:** chosen on 2026-11-14 to limit subscription use: Fable vs Opus as the highest tier, and LLM team
vs offline policy. No component ablations (no-critic / no-literature / no-forward) were run for V3.
**Execution route:** every development experiment ran on the UW AMATH GPU server `doppio` (2× TITAN X Pascal)
over SSH; final fit + held-out evaluation ran locally, by design.

---

## TL;DR

| Configuration | Top tier | Test rel_l2 | Certified | Studies complete | LLM calls | Input / output tokens | LLM $/run | Wall/run |
|---|---|---|---|---|---|---|---|---|
| `v3_offline` (n=3) | — (no LLM) | **0.01842 ± 0.00000** | 3/3 | 1/1 | 0 | 0 / 0 | $0 | 1.0 min |
| `v3_full` s0 (n=1) | **Fable 5.1** | **0.01833** (CI 0.0172–0.0194) | ✓ | 4/6 | 39 | 1.79 M / 101 k | $17.27 | 24 min (+64 min usage-limit pause) |
| `v3_opus_top` s0 (n=1) | **Opus 5.5** | — (export refused) | ✗ | 1/6 | 30 | 1.41 M / 167 k | $9.33 | 28 min |
| *V2 full (n=3)* | — | *0.01838 ± 0.00007* | — | — | — | — | *$4.33* | *12.6 min* |
| *V1 full (n=3)* | — | *0.02365 ± 0.00083* | — | — | — | — | *$0.83* | — |
| *Physics reference* | — | *MLE 0.01842 · Cramér–Rao floor 0.01923* | — | — | — | — | — | — |

Lower rel_l2 is better (‖Ŷ − Y‖_F / ‖Y‖_F on the untouched 750-sample test set). CI is the 95 % test bootstrap.
$ are Claude Code list-price equivalents (`total_cost_usd`), not invoices. Tokens include prompt-cache fields
and Claude Code's internal Haiku side calls are reported separately (§4).

**Main conclusions (provisional — n = 1 per LLM configuration)**

1. **V3 matches V2 on accuracy and stays at the information limit.** Every certified V3 model lands at
   0.0183–0.0184, the stand-alone physics MLE (0.01842), and 22 % below V1 (offline arm vs V1: p = 0.008).
   The Fable run (0.01833) and the offline arm (0.01842) differ by 0.5 %, well inside the test CI. The
   benchmark is saturated, as V2 already showed, so accuracy cannot separate V3 configurations here.
2. **What V3 changes is the research product, not the score.** The Fable team decomposed the objective into
   3 questions, registered 6 studies with 23 matched arms, completed 4 studies, and reported 6 inconclusive and
   1 contradicted claims instead of certifying improvements it could not demonstrate. The independent critic
   raised 155 objections (82 blocking). The offline policy runs one 3-arm study.
3. **Fable vs Opus as highest tier: no accuracy comparison is possible from this run.** The Opus-led team
   selected an agent-written `custom_solver` (validation 0.01932) whose predictions depend on batch
   composition (partition difference 1.1 × 10⁻³). V3's export certificate correctly refused it, so V3
   withholds the held-out metric by design. The Opus run cost **46 % less** ($9.33 vs $17.27) and
   produced more output (167 k vs 101 k tokens, more critic and orchestrator reasoning), but completed 1/6
   studies vs 4/6 and spent 4 of 13 actions on revisions. With n = 1 these are observations, not effects.
4. **V3 is 4× the LLM cost of V2 per run.** The orchestrator dominates: in the Fable run it used 66 % of
   input tokens and 77 % of cost (15 calls, 1.19 M input tokens, $13.32), because every action re-reads the
   whole program state.
5. **The live runs found ten harness defects that the fake-model test suite could not.** All were fixed with
   regression tests (§5). Four further gaps remain open (§6), chiefly the coding specialist (0/5 study
   implementations succeeded) and critic objections that are never resolved (0/281).

> **Caveats:** n = 1 per LLM configuration, a saturated benchmark, LLM non-determinism, and several harness
> fixes made between attempts (all valid runs use the same final code, §7). Do not read conclusion 3 as a
> model-quality ranking.

---

## 1. Setup

Data, metric and development budget are identical to V1/V2:

- Simulator [examples/heat_diffusion_simulator.py](../../examples/heat_diffusion_simulator.py), 4 segments,
  80 features, noise σ = 10⁻³. Data seed 5, split 3000 / 750 / 750. Success threshold rel_l2 ≤ 0.05.
- **Development compute: 400 worker-seconds**, 120 s per experiment, 6 workers; search seed = run seed.
- **V3-specific budget:** V3 reserves its final export out of `compute_seconds`. The default 25 % (100 s) is
  too small to export and certify a physics method, so the ablation uses `compute_seconds: 760` with
  `confirmation_seconds: 360` (400 s development + 360 s protected export, the same 3 × 120 s that V2 allowed
  its final fit). Wall budget 7200 s. Strict hold-out off (one final evaluation per run, as in V1/V2), with
  exposure recorded in `ablation/heat_v3/exposure_registry.sqlite`.
- Program limits: 24 actions, 10 agents (6 active), depth 2, 4 children, 8 specialist tasks, 8 turns per
  agent, 6 studies, 80 LLM calls, soft budget mode. Literature (TheoremSearch) and forward model enabled;
  fresh research memory per run. Shift sets `noise_x3` and `k_low_0.3` as in V2.

| Role tier | `v3_full` (tiered) | `v3_opus_top` |
|---|---|---|
| orchestrator | Fable 5.1, effort medium | **Opus 5.5**, effort medium |
| researcher, critic | Opus 5.5, medium | Opus 5.5, medium |
| specialist | Sonnet 5.5, medium (600 s timeout) | Sonnet 5.5, medium |
| `*_escalated` | Fable 5.1, high | Opus 5.5, high |

`v3_offline` uses no model: the offline program policy registers a default study (ridge control vs extra-trees
treatment, plus a physics-MLE baseline). All LLM calls go through the Claude Code CLI (`claude_code` backend)
with `DISABLE_PROMPT_CACHING=1`; every program view is unique, so the CLI's 1-hour cache writes (billed at 2×
input) were pure overhead (−47 % cost in a paired check).

Driver [run_ablation_v3.py](../../ablation/heat_v3/run_ablation_v3.py), analysis
[analyze_v3.py](../../ablation/heat_v3/analyze_v3.py), raw results
[summary_v3.json](../../ablation/heat_v3/summary_v3.json), per-call tokens in each run's `token_usage.json`,
run log `ablation/heat_v3/driver_log.jsonl`.

## 2. Per-run results

| Run | Test rel_l2 | Final method | Val rel_l2 | Per-segment test rel_l2 | `noise_x3` | `k_low_0.3` |
|---|---|---|---|---|---|---|
| offline s0–s2 | 0.01842 (all three identical) | physics-MLE [log, ms=1] | 0.01934 | 0.0270 / 0.0061 / 0.0052 / 0.0242 | 0.0520 | 0.125 |
| full s0 (Fable) | **0.01833** | physics-MMSE [log, ms=3] | 0.01905 | 0.0268 / 0.0061 / 0.0052 / 0.0241 | 0.0512 | 0.110 |
| opus_top s0 | withheld (uncertified) | custom_solver (forward) | 0.01932 | — | — | — |

The offline arm is deterministic: the search seed does not affect a single-start physics MLE. All certified
exports pass repeat / reload / partition checks with zero difference and reload bit-exactly. The per-segment
errors match the per-segment Cramér–Rao values (0.0286 / 0.0064 / 0.0052 / 0.0248), as in V2.

## 3. What the V3 research organization did

| | offline s0 | full s0 (Fable) | opus_top s0 |
|---|---|---|---|
| Program actions (rejected) | 7 (0) | 15 (5) | 13 (4) |
| Questions decomposed | 1 | 3 | 3 |
| Studies registered → complete / incomplete / superseded | 1 → 1 / 0 / 0 | 6 → 4 / 1 / 1 | 6 → 1 / 1 / 4 |
| Arms registered (executed OK) | 3 (3) | 23 (19) | 25 (9) |
| Method kinds | library 2, physics 1 | library 13, physics 9, hybrid 1 | library 13, physics 6, custom_solver 4, custom_ml 2 |
| Claim verdicts | 1 contradicted | 6 inconclusive, 1 contradicted | 5 inconclusive |
| Researchers spawned | 2 (math, ml) | 1 (mathematical) | 1 (mathematical) |
| Specialist tasks (completed) | 0 | 4 coding (0) | 1 source, 2 analysis, 1 coding (3) |
| Critic objections (blocking / resolved) | 0 | 155 (82 / 0) | 126 (50 / 0) |
| Stop reason | registered work resolved | orchestrator: "no remaining action can change a verdict or improve the export" | study limit reached (6) |

- **Orchestration.** Both LLM orchestrators followed the intended order (decompose → spawn → register →
  execute → stop) and wrote stop rationales that cite measured artifacts, intervals and remaining
  limitations. Neither used the dynamic-team capability: each spawned a single researcher.
- **Scientific truthfulness.** Neither team claimed an unsupported improvement. Physics vs extra-trees gains
  of ≈ 75 % with tight paired intervals stayed *inconclusive* because blocking critic objections (forward
  operator parity, CFL, convergence logs) were never resolved. This is V3 behaving as specified, but it also
  means no LLM claim reached *supported*.
- **Critic.** Very active and substantive (it caught a real problem-text/simulator mismatch in a smoke run),
  but objection volume is unbounded and no objection was ever resolved: `resolve_objection` attempts were
  rejected because the cited evidence belonged to a different study or the critic did not accept it.
- **Opus run.** More exploratory (6 custom-code arms vs 0) and more revision-heavy. Its best validation arm
  was an agent-written solver that is not batch-invariant; selection does not check certifiability, so the
  run ended with an uncertified export.

## 4. Token usage and cost

**By role tier** (calls · total input tokens · output tokens · thinking tokens · list $):

| Tier | full s0 (Fable top) | opus_top s0 (Opus top) |
|---|---|---|
| orchestrator | 15 · 1,186,212 · 22,671 · 5,454 · **$13.32** | 13 · 894,485 · 86,888 · 24,817 · **$5.55** |
| researcher | 8 · 284,565 · 21,687 · 5,793 · $1.64 | 7 · 213,450 · 28,424 · 6,559 · $1.49 |
| critic | 12 · 280,299 · 45,304 · 18,737 · $2.11 | 9 · 291,268 · 49,796 · 27,745 · $2.25 |
| specialist | 4 · 35,407 · 11,500 · 4,432 · $0.19 | 1 · 8,876 · 2,120 · 653 · $0.04 |
| **Total** | **39 · 1,786,483 · 101,162 · $17.27** | **30 · 1,408,079 · 167,228 · $9.33** |

**By served model** (from Claude Code `modelUsage`):

| Model | full s0 | opus_top s0 |
|---|---|---|
| claude-fable-5-1 | 1,186,212 in · 22,671 out · $13.00 | — |
| claude-opus-5-5 | 564,864 in · 66,991 out · $3.60 | 1,399,203 in · 165,108 out · $8.90 |
| claude-sonnet-5-5 | 35,407 in · 11,500 out · $0.19 | 8,876 in · 2,120 out · $0.04 |
| claude-haiku-5-5 (CLI internal side call) | 1,619,550 in · 4,628 out · $0.49 | 1,303,440 in · 2,510 out · $0.39 |

- The orchestrator's prompt is the whole program state and grows with every study: 11 k → 98 k tokens in the
  Fable run (it reached 287 k before the view compaction in §5).
- Fable input is ~1.8× the per-token price of Opus here ($11.2 vs $6.2 per M input-dominated tokens).
- Every Claude Code call also makes a Haiku side call that re-reads the prompt: it roughly doubles the
  *token count* but costs < 3 % of the money.
- One Fable-orchestrated run uses most of a 5-hour subscription window; the weekly Fable limit was reached
  after six Fable-orchestrated runs (including partial attempts and a smoke test) in two days.

**Total spend for this study**, all attempts including smoke tests, superseded and excluded runs:
≈ **$92** list-equivalent (valid runs $26.60; excluded/superseded $53.65; smoke tests $12.19). This excludes
the opus_top attempt stopped early in attempt 3 (no ledger) and the one-word preflight calls.

## 5. Harness defects found by the live runs (fixed, with regression tests)

| # | Symptom in a live run | Fix | Test |
|---|---|---|---|
| 1 | All 6 orchestrator actions rejected: the model was never told the study schema | `interface_guide` in the program view (action/study/claim schemas, method catalogue, example study, last rejection) | smoke run: 1 complete study, certified export |
| 2 | Token accounting showed 2 input tokens per call | record cache-creation / cache-read / thinking tokens and per-model usage | `test_v3_live_interface` |
| 3 | GPU route crashed: remote `val_idx_path` used locally | localize every downloaded path | `test_remote_result_paths_are_localized` |
| 4 | Export of a physics method timed out at 100 s | ablation config 400 s dev + 360 s protected export | — (configuration) |
| 5 | Prompt grew to 287 k tokens; one run exhausted the 5-hour window | model-facing view only: long numeric lists → summary statistics; superseded studies → stubs; actions reference studies; agent code stored once per arm (1.2 MB → 329 kB on the same program) | `test_v3_live_interface` |
| 6 | A usage limit mid-run discarded the whole run | `MLH_WAIT_ON_USAGE_LIMIT=1`: wait for the stated reset and retry; paused time excluded from the wall clock and logged | `UsageLimitTests` |
| 7 | Export crashed: agent class defined inside `build_method` cannot be pickled | gate rejects nested classes; V3 development jobs check picklability; codegen rules say so | `ExportabilityTests` |
| 8 | Multistart physics exports failed the partition certificate | random starts seeded per observation (as the posterior sampler already was); also affects V2 physics/hybrid | `test_multistart_mode_is_batch_independent` |
| 9 | Coding specialist studies rejected for a missing id | specialist receives the study schema; missing ids derived from the task | full suite |
| 10 | A per-call timeout marked a run as outage-degraded | driver counts timeouts separately | — (driver) |

Full suite after all fixes: **102 tests, OK** (`Version Builds/v3/ablation_relaunch3_tests.log`).

## 6. Open issues (not fixed; recommendations)

1. **Coding specialist never delivered a study** (0/5 in the valid runs): responses were either invalid JSON
   (code embedded in a JSON string) or valid JSON without a `study`. Responses are not persisted when the
   schema fails, so the cause cannot be inspected. Recommend logging raw responses on schema failure and
   letting the specialist return code in a fenced block.
2. **Critic objections are never resolved** (0/281) and are unbounded (155 in one run). Recommend
   de-duplication, a per-study cap, and a clearer evidence-resolution path the orchestrator can satisfy.
3. **Selection does not anticipate certification.** The best-validation arm can be uncertifiable (opus_top),
   leaving the run with no test metric. Recommend running the partition/repeat check on candidate arms during
   development, or falling back to the best certifiable arm.
4. **Wall time is not protected for export.** `v3_full` s1 finished its research but its export was refused
   because the wall budget was spent (inflated by two standbys). Recommend a wall-time export reserve.
5. Minor: 1–3 SSH channel failures per run (`Connect failed`) fell back to local execution, so a few
   development jobs ran on the laptop CPU (`devices` in `summary_v3.json`). Dynamic teams were not used (one
   researcher per LLM run).

## 7. Run log (validity)

| Attempt | Runs | Outcome | Handling |
|---|---|---|---|
| Smoke 1 (11-13) | 2-segment smoke | 6/6 actions rejected (defect 1) | fixed; not part of results |
| Smoke 2 (11-13) | 2-segment smoke | complete, certified ($9.88) | kept in `ablation/heat_v3/smoke/` |
| 1 (11-13 17:06) | full s0 | 5-hour limit after 17 calls; export timeout (defect 4) | superseded (`runs_superseded_2026-11-13/`) |
| 2 (11-13 21:25) | full s0 | nested-class pickle failure (defect 7); 5-hour limit | superseded (`.../attempt2/`) |
| 3 (11-14 13:05) | full s0, opus_top s0 (partial) | completed research but multistart export uncertified (defect 8); specialist schema (defect 9) | superseded (`.../attempt3/`); opus_top stopped early |
| 4 (11-14 13:25) | offline s0–s1, full s0, opus_top s0 | valid. full s0: one standby (13:38:45) during a usage-limit pause, no experiment in flight | reviewed and accepted (`SUSPEND_REVIEWED.json`, V2 precedent) |
| 4 (11-14 19:42) | full s1 | two standbys during LLM calls → wall budget spent → export refused; weekly Fable limit on the last call | excluded (`runs_excluded_2026-11-15/`) |
| 4 (11-15) | offline s2 | valid | — |
| pending | full s1, full s2 | weekly Fable limit, resets 2026-11-19 11:00 PT | run after reset (below) |

All valid runs use the same final harness code (every fix in §5 precedes attempt 4).

**To finish the study** (from the harness root, with the laptop on power and lid open):

```powershell
$py = 'C:/Users/kusha/AppData/Local/Programs/Python/Python311/python.exe'
# after fixing SERVICE_PASSWORD in the repository .env (it is one character too long):
& $py -u ablation/heat_v3/run_ablation_v3.py --remote doppio --seeds 1 2 --arms v3_full
& $py ablation/heat_v3/analyze_v3.py
```

Then update this document's TL;DR with n = 3 for `v3_full`.
