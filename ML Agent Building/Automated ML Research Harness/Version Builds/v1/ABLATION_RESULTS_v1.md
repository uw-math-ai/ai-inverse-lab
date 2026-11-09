# Ablation study: what each part of the V1 harness contributes

**Problem:** we have noisy temperature readings at four sensors over time. Can an ML model recover the
four unknown piecewise-constant thermal conductivities of a 1-D rod governed by the heat equation?
**Harness:** Automated ML Research Harness V1. **Date:** 2026-11-08. **Runs:** 5 configurations × 3
seeds = 15 complete harness runs, plus a physics-based reference.

---

## TL;DR

| Configuration | What it tests | Final test rel_l2 (mean ± sd, n=3) | Best single model, test | LLM $/run | Verdict |
|---|---|---|---|---|---|
| Random methodology search | Whether reasoning beats sampling | **0.0237 ± 0.0008** | 0.0268 ± 0.0004 (worst) | $0 | Random ideas were as good as LLM ideas. Random search fell behind only because it has no feedback-driven refinement. |
| Offline heuristic agents | Value of fixed search rules | **0.0237 ± 0.0006** | 0.0253 ± 0.0013 | $0 | Strong. Its local-mutation refinement was the most reliable source of improvement (+11 % in every seed). |
| LLM ideation only | Value of LLM-generated proposals | 0.0266 ± 0.0020 (worst) | 0.0266 ± 0.0020 | $0.39 | The first ideas were sensible but no better than 16 random draws. It also left about 58 % of the compute budget unused. |
| LLM ideation + refinement | Value of feedback-driven search | 0.0256 ± 0.0015 | 0.0256 ± 0.0015 | $0.72 | A small, inconsistent gain (+1.7 %). It improved in 1 of 3 seeds. |
| Full harness | Value of the complete pipeline | **0.0236 ± 0.0008** (best, tied) | **0.0250 ± 0.0014** (best) | $0.83 | Best on both measures, but by margins well inside the noise. The ensemble step provides most of its edge over the LLM ablations. |
| *Physics reference (not an arm)* | *What the data allows* | *MLE 0.0184 · Cramér–Rao floor 0.0192* | — | — | *Every ML model is at least 1.2× above the noise floor.* |

Lower rel_l2 is better: ‖Ŷ − Y‖_F / ‖Y‖_F on the held-out test set of 750 samples. Predicting the mean scores 0.330.

**Main conclusions**

1. **The three complete pipelines tie: full harness, offline heuristic agents and random proposals.**
   Their means are 0.0236, 0.0237 and 0.0237, with Welch p > 0.8 for every pair. On this problem, the
   source of the ideas (LLM, fixed rules or random draws) made no measurable difference to the final
   model. The pipeline around the ideas (screening, halving, refinement, ensemble) is what matters.
2. **Reasoning does not beat sampling at the proposal stage.** The best first-round idea had validation
   rel_l2 0.0273 with LLM ideators, 0.0274 with 16 uniform random draws, and 0.0299 with the heuristic
   personas. The LLM's ideas were physically well motivated (log features to linearise diffusive
   decay, PCA to denoise correlated sensor traces), but random draws found equally good regions of this
   small search space.
3. **Feedback-driven search helps, but the simple version was more reliable here.** The heuristic local
   mutation improved validation by **+11.0 % ± 1.8 %** in all three seeds. LLM refinement improved it
   by +1.7 % (ideation + refinement arm) and +4.0 % (full arm), and gained nothing in 3 of 6 runs. The
   heuristic arm did start from a worse point, so it had more room to improve.
4. **The ensemble is the most valuable single component.** Greedy ensembling cut test error by **11.5 %**
   for random search, 6 % for offline and 5 % for the full harness. It gains most where the candidate
   pool is most diverse (random) and it largely evens out the arms. The two LLM ablations were run
   without the ensemble, which explains most of their gap to the full harness.
5. **The real headroom is the physics, not the search.** Maximum-likelihood inversion through the known
   forward model reaches 0.0184, which beats every one of the 15 harness models (the best of the 15
   runs scored 0.0228). The ML models lose most on the **inner** conductivities: error there is about 2× the
   Cramér–Rao floor, against about 1.2× on the outer ones. None of the agents could propose a
   forward-model-based method, because the harness's method contract (a scikit-learn estimator on X)
   rules it out.

> **Caveat:** with n = 3 seeds per arm, none of the between-arm differences is statistically
> significant. The smallest p-values are about 0.11–0.17. Treat conclusions 2–4 as well-supported
> directions, not proven effects. §7 lists the threats to validity.

---

## 1. The problem

The simulator is [examples/heat_diffusion_simulator.py](../../examples/heat_diffusion_simulator.py):

- The equation is u_t = (k(x) u_x)_x on [0, 1] with u = 0 at both ends, solved by explicit finite
  differences (40 grid points, 200 time steps).
- The initial condition is a Gaussian heat pulse at x = 0.5.
- **Unknowns (targets):** k = (k₁, k₂, k₃, k₄), one conductivity per quarter of the rod, each drawn
  i.i.d. from U(0.5, 2.0).
- **Observations (features):** the temperature at 4 sensors (x ≈ 0.13, 0.38, 0.64, 0.90, one per
  segment), sampled every 10 steps, which gives 20 times × 4 sensors = **80 features**. Gaussian noise
  with σ = 10⁻³ is added.
- **Data:** a single draw, split 3000 / 750 / 750 into train / val / test (data seed 5). The identical
  dataset was used by all 15 runs (the cached dataset hash matches).
- **Success criterion** (from the spec): test rel_l2 ≤ 0.05. Every run in every arm met it by about 2×.
  This is a first sign that the problem is not hard enough to separate the arms (see §7).

## 2. Experimental design

All five configurations run through the same harness code, data, budget and evaluation protocol. Only
the switches in the table below differ. The driver is [run_ablation.py](run_ablation.py), and the
generated specs are in [specs/](specs/).

**Held constant across all arms**

- **Budget:** 400 worker-seconds of compute, `max_ideas` 34, a 120 s per-experiment limit, 6 workers ×
  2 threads, CPU only.
- **Search schedule:** 16 initial ideas; a screen at 10 % of the data; successive halving at 33 % and
  100 % (η = 3, `min_full` 4); up to 3 refinement rounds of 3 parents × 2 children; patience 2.
- **Selection and testing:** models are selected on validation only. The final model is retrained on
  train+val and scored **once** on test.
- **Execution:** runs were executed **one after another**, never concurrently, so that the harness's
  wall-time-based compute accounting is not distorted.
- **Seeds:** the search seed is 0, 1 or 2 for every arm. It drives the heuristic and random number
  generators and the model seeds. LLM output is also non-deterministic on its own.

| Arm | Proposals | Refinement | LLM critic | Crash repair | Ensemble | LLM backend |
|---|---|---|---|---|---|---|
| `random` | 16 uniform random configs* | fresh random configs (feedback ignored) | – | – | ✓ | none |
| `offline` | 4 heuristic personas | heuristic local mutation | – | – (no LLM) | ✓ | none |
| `llm_ideation` | 4 LLM personas | **off** (0 rounds) | off | off | off | Claude Code |
| `llm_refine` | 4 LLM personas | LLM refinement | off | off | off | Claude Code |
| `full` | 4 LLM personas | LLM refinement | ✓ | ✓ | ✓ | Claude Code |

\* The harness had no random-search mode, so I added one: a `random` persona in
[mlharness/agents/ideation.py](../../mlharness/agents/ideation.py). It samples the method uniformly
from all 10 regression methods, every hyper-parameter from the method's search space, the feature set
from {raw, log, sqrt}, the scaler from {standard, robust, none}, and PCA (off with probability ½,
otherwise 8, 16, 32 or 64 components). Its "refinement" step draws the same number of children
afresh, so it uses no feedback.

The random arm still goes through the harness's screen, halving and ensemble. It therefore isolates
the *idea source* rather than measuring "naked" random search. The 14 existing tests still pass after
this change.

**Design choice: the arm ladder.** The table in the request is a nested ladder (LLM ideation → +
refinement → full). I kept the two non-LLM arms on the *complete* pipeline with heuristic or random
agents. This makes `random` vs `offline` vs `full` a clean "who proposes the ideas" comparison. The
cost is that the step from `llm_refine` to `full` adds three components at once (LLM critic, repair,
ensemble). §4.4 separates them as far as the data allows.

**Extra analyses** (in [analyze.py](analyze.py) and [physics_reference.py](physics_reference.py))

- **Best-single-model test score.** After the fact, I retrained each run's best single full-data
  model on train+val and scored it on test. This removes the ensemble confound. The harness never sees
  these numbers.
- **Physics reference:**
  - The forward model was re-implemented and checked to match the simulator to 1e-14.
  - A batched Levenberg–Marquardt maximum-likelihood fit on every test sample, in log-k, bounded to
    the prior range [0.5, 2] and taking the best of 5 starts. Mean χ²/dof = 0.95, so the fit converged
    to the noise level.
  - The Cramér–Rao bound σ²(JᵀJ)⁻¹, evaluated at the true k.

## 3. Results

### 3.1 Per-seed results (final model, test rel_l2)

| Arm | seed 0 | seed 1 | seed 2 | mean ± sd | final model type |
|---|---|---|---|---|---|
| random | 0.0241 | 0.0228 | 0.0241 | 0.0237 ± 0.0008 | ensemble ×3 (2–5 members) |
| offline | 0.0234 | 0.0244 | 0.0234 | 0.0237 ± 0.0006 | ensemble ×3 (3–4 members, all torch MLP) |
| llm_ideation | 0.0247 | 0.0286 | 0.0266 | 0.0266 ± 0.0020 | single ×3 (ensemble disabled) |
| llm_refine | 0.0270 | 0.0258 | 0.0240 | 0.0256 ± 0.0015 | single ×3 (ensemble disabled) |
| full | 0.0232 | 0.0246 | 0.0232 | 0.0236 ± 0.0008 | ensemble ×3 (3–4 members) |

How large is test-set noise? The bootstrap standard deviation of a single run's test rel_l2 is
**0.00065**. The seed-to-seed spread within an arm is about the same size. Arm-mean differences
below about 0.001 are therefore noise.

Test error came out slightly *lower* than validation error in every arm (by 0.001–0.002). This is
expected, because the final model is retrained on train+val (25 % more data). No arm shows
validation-selection overfitting.

### 3.2 Search quality without the ensemble

| Arm | best val, initial ideas only | best val, after refinement | refinement gain | best single model, **test** |
|---|---|---|---|---|
| random | 0.0274 ± 0.0008 | 0.0274 | 0 % (random "children" never won) | 0.0268 ± 0.0004 |
| offline | 0.0299 ± 0.0010 | 0.0266 ± 0.0008 | **+11.0 % ± 1.8 %** (3/3 seeds) | 0.0253 ± 0.0013 |
| llm_ideation | 0.0276 ± 0.0010 | — | — | 0.0266 ± 0.0020 |
| llm_refine | 0.0273 ± 0.0009 | 0.0268 ± 0.0008 | +1.7 % ± 2.9 % (1/3 seeds) | 0.0256 ± 0.0015 |
| full | 0.0273 ± 0.0016 | 0.0261 ± 0.0006 | +4.0 % ± 3.5 % (2/3 seeds) | **0.0250 ± 0.0014** |

On single-model quality, the order is **full ≈ offline ≈ llm_refine < llm_ideation ≈ random**. The
arms with feedback-driven refinement beat the arms without it by about 5–7 %, in line with what the
table above is designed to test. The differences are not significant at n = 3: random vs full
p = 0.16, random vs offline p = 0.17.

### 3.3 Where the error is: per-conductivity test rel_l2

| | k₁ (outer) | k₂ (inner) | k₃ (inner) | k₄ (outer) |
|---|---|---|---|---|
| random | 0.0337 | 0.0110 | 0.0103 | 0.0302 |
| offline | 0.0329 | 0.0130 | 0.0121 | 0.0298 |
| llm_ideation | 0.0375 | 0.0137 | 0.0123 | 0.0331 |
| llm_refine | 0.0361 | 0.0121 | 0.0109 | 0.0325 |
| full | 0.0335 | 0.0114 | 0.0099 | 0.0301 |
| **Physics MLE** | **0.0270** | **0.0061** | **0.0052** | **0.0242** |
| **Cramér–Rao floor** | 0.0286 | 0.0064 | 0.0052 | 0.0248 |
| **best arm ÷ floor** | 1.17× | **1.72×** | **1.90×** | 1.20× |

- **The outer segments are intrinsically harder.** Their Cramér–Rao floor is 4–5× higher, because
  heat from the central pulse barely reaches the outer sensors within the observation window. The
  median Fisher-matrix condition number is 69.
- **The ML models are already near the floor on the outer segments, but about 2× above it on the
  inner ones.** The remaining headroom is in the *well-determined* parameters. This pattern suggests
  the learners are limited by training samples or capacity rather than by information. A physics
  step, such as an ML initial guess refined by Gauss–Newton through the forward model, would capture
  most of it.

(The MLE slightly beats the Cramér–Rao bound because the box constraint [0.5, 2] adds prior
information; the bound assumes an unconstrained unbiased estimator.)

### 3.4 Cost and effort

| Arm | ideas proposed | full-data evaluations | compute used (of 400 s) | wall time | LLM calls | LLM cost |
|---|---|---|---|---|---|---|
| random | 26 ± 3 | 14 ± 3 | 426 s | 267 s | 0 | $0 |
| offline | 34 | 14 ± 3 | 436 s | 235 s | 0 | $0 |
| llm_ideation | 16 | 4 | **169 s** | 139 s | 5 | $0.39 |
| llm_refine | 28 ± 6 | 10 ± 5 | 416 s | 350 s | 9.7 | $0.72 |
| full | 27 ± 4 | 10 ± 5 | 410 s | 374 s | 12 | $0.83 |

- **Totals:** 9 LLM runs cost **$5.81**. All 15 runs took about 1.1 h of wall time, run one at a time.
- **Over-budget runs:** compute can exceed 400 s because the final retrain is always allowed to run.
- **Under-used budget in `llm_ideation`:** with refinement switched off, it stops after halving and
  uses only about 42 % of the budget. Its weaker result is therefore partly a *less compute*
  effect, not only a *worse ideas* effect.
- **Wall-time overhead:** the LLM arms take about 1.5× longer, mostly from LLM latency (about 20 s
  per call).

## 4. Analysis by question

### 4.1 Does reasoning beat sampling? (`random` vs LLM arms)

**Not at the idea stage, on this problem.**

- **First-round quality was equal.** The best first-round idea had the same validation score under the
  LLM ideators and under random sampling (0.0273 vs 0.0274). Both found the same model families:
  torch MLP and RBF kernel ridge on log or sqrt features with PCA.
- **The LLM's reasoning was good:**
  - It consistently argued that diffusive time scales go as 1/k, so *log features linearise the
    exponential decay*.
  - It argued that *diffusion makes the 80 features highly correlated, so PCA removes noise*.
  - It wrote custom models that regress **log k**, because errors are multiplicative.
- **But the search space is small and forgiving.** Two of the ten methods (MLP and kernel ridge) win
  in essentially every configuration. Sixteen random draws therefore almost always include a good
  one, and the screen and halving stages pick it out.
- **Random search loses at the single-model level** (0.0268 vs 0.0250 for full) because it never
  *improves* on its first find. It only ties at the final-model level because the ensemble rescues it.

### 4.2 Value of fixed search rules (`offline`)

**High, relative to its cost (zero).**

- **Weak start:** the heuristic personas' first-round ideas were the weakest of all arms (0.0299),
  because they start from library defaults.
- **Most reliable improvement:** its local hyper-parameter mutation of the best torch-MLP gave
  +9–12 % in **every** seed. It ended up within noise of the full harness, both on the final model
  (0.0237 vs 0.0236) and on the best single model (0.0253 vs 0.0250).
- **Implication:** as long as the model family is right, cheap local search tunes hyper-parameters at
  least as well as an LLM.

### 4.3 Value of LLM proposals and feedback-driven refinement (`llm_ideation` → `llm_refine`)

- **Proposals only** (0.0266) is the weakest arm, but it is confounded by spending less than half the
  compute (see §3.4).
- **Adding LLM refinement** gives 0.0256, about 4 % better on test. Refinement improved on its parent
  in only 1 of 3 seeds. In the successful seed (s1), the LLM recognised that a sharper,
  lighter-regularised kernel ridge was the right direction and iterated it to the arm's best result.
  In the other seeds its children were near-duplicates of their parents or slightly worse.
- **LLM refinement varies more than heuristic mutation.** It makes larger, more "creative" moves:
  switching feature sets, changing architecture, adding custom code. Those moves occasionally find a
  new region but usually do not pay off within 2–3 rounds.

### 4.4 Value of the complete pipeline (`llm_refine` → `full`)

`full` adds the LLM critic, crash repair and the ensemble. The total gain is 0.0256 → 0.0236
(−8 %, p = 0.14). The pieces contribute as follows:

| Component | Evidence | Estimated contribution |
|---|---|---|
| Greedy ensemble | The full arm's ensemble beat its own best single model on test by 8.1 %, 6.3 % and 1.0 % | **Most of the gap** |
| LLM critic batch review | 1 rejection in 3 runs (a near-duplicate kernel-ridge idea) | Negligible |
| Crash repair | 2 repairs of agent-written code in 3 runs; one repaired custom model (log-target kernel ridge with per-output CV bandwidth) became the best single model in seed 2 | Occasional, not systematic |
| Refinement (same as `llm_refine`) | +4.0 % vs +1.7 %; within noise | — |

## 5. Answers in one line each

- **Random methodology search:** idea quality was the same as the LLM's. It lags only because it lacks
  feedback-driven refinement, and the ensemble hides that gap.
- **Offline heuristic agents:** fixed rules (persona defaults plus local mutation) were the best value
  for money. They matched the full harness at $0.
- **LLM ideation only:** good, well-reasoned first ideas, but no better than random draws, and not
  enough on its own.
- **LLM ideation + refinement:** feedback helps, but LLM refinement was less reliable than simple
  mutation on this problem.
- **Full harness:** best on both final and single-model test error, but tied with the offline and
  random pipelines within noise. The ensemble, not the LLM, is the decisive component.

## 6. Recommendations for the harness

1. **Add a physics-informed method family.** This is the largest lever: about 22 % lower error is
   available. Allow methods that use the simulator's forward model, for example an ML initial guess
   followed by Gauss–Newton or Levenberg–Marquardt (this is N4 in `v1_build_upd.md`). The agents
   cannot propose this under the current scikit-learn-estimator contract.
2. **Keep the ensemble on by default** and make it more diversity-aware. It was the most valuable
   component in every arm.
3. **Make refinement hybrid.** Use LLM refinement for structural moves (new families and features) and
   heuristic mutation for local hyper-parameter tuning. On this evidence the heuristic does local
   tuning more reliably and costs nothing.
4. **Spend unused budget.** When refinement is off or stops early, the remaining budget should go to
   more proposals or halving rungs. The `llm_ideation` arm wasted 58 % of its budget.
5. **Report the Cramér–Rao or MLE floor in the harness report** when a forward model is available.
   Here it shows that the success threshold (5 %) is 2.6× looser than what the data allows, which is
   why every arm "succeeded."

## 7. Threats to validity and limitations

- **Small n.** There are 3 seeds per arm, and no pairwise difference reaches p < 0.05. LLM
  non-determinism adds variance beyond the seed.
- **Ceiling effect.** Every arm is within 1.2–1.4× of the noise floor and 2× under the success
  threshold. An easier-to-separate benchmark would vary more segments (for example 8), raise the noise
  (σ = 10⁻²), use fewer training samples, or move the sensors away from the segments.
- **One dataset and one simulator.** The results may not carry over to the graph first-passage-time
  problem, where the earlier V1 runs found that Claude's agents *did* beat offline heuristics
  (1.19 % vs 1.38 %).
- **Confounds in the ladder.**
  - `llm_ideation` spent less compute.
  - `full` differs from `llm_refine` by three components at once.
  - The non-LLM arms keep the ensemble, while the two middle LLM arms do not. The best-single-model
    analysis (§3.2) controls for the ensemble.
- **Compute accounting** is wall-time × jobs, so it is sensitive to machine load. Runs were executed
  one at a time to limit this.
- **LLM backend:** `claude_code` with the CLI's default model, tools disabled, effort `medium`.
  Results may differ with another model or effort level.

## 8. Reproducing

```bash
cd "ML Agent Building/Automated ML Research Harness V1"
PY=C:/Users/kusha/AppData/Local/Programs/Python/Python311/python.exe
$PY ablation/heat/run_ablation.py --arms random offline llm_ideation llm_refine full --seeds 0 1 2
$PY ablation/heat/analyze.py --posthoc                       # -> ablation/heat/summary.json
$PY ablation/heat/physics_reference.py ablation/heat/runs/offline_s0/data/dataset.npz
```

| Path | Contents |
|---|---|
| `ablation/heat/runs/<arm>_s<seed>/` | Full harness output per run: `report.md`, `leaderboard.csv`, `artifacts.jsonl`, final model |
| `ablation/heat/summary.json` | Every per-run and per-arm number in this document, including best-so-far curves |
| `ablation/heat/physics_reference.json` | MLE and Cramér–Rao results |
| `ablation/heat/specs/` | The 15 generated specs |
| `mlharness/agents/ideation.py` | The `random` persona (the only change to the harness itself) |
