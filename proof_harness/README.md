# Local mathematical proof harness

This harness compares curated mathematical starting information in a creative-agent / skeptical-agent loop. It is inspired by the proposal revision, bounded resources and artifact logging in Kushal's `418c1f5` research harness. It is a standalone implementation on `harrison-working-branch`; it does not merge the ML harness or train ML predictors.

**Current experiment:** nine small Lean proof obligations: eight centrality prerequisites (including an algebraic path reconstruction formula) and one absorption identity. This is an instrumented proof pilot, not a formalization of the complete inverse reconstruction theorem. The original broad baseline design is narrowed to explicit, compiled facts for this first run. In particular, B3 currently tests span/injectivity facts, not a completed spectral-controllability proof.

**First pilot outcome:** 17 complete protocols and four usage-limit interruptions, from 98 attempted model calls (93 successful). Completed B3 and B6 runs verified 9/9 tasks; other completed runs verified 8/9. See the report before interpreting this as a baseline ranking: there is a strong ceiling effect and the span task is structurally helped by its supplied lemma. No full reconstruction theorem was proved.

The first pilot used the source frozen in `runs/first-pilot/source_snapshot/`, matching the original manifest hashes. After that run, the parenthesized-expression parser was fixed, empty proof submissions were classified as abstentions, usage-limit errors gained a shared circuit breaker, and Git checks were narrowed to actual branch/remote/tag refs. The live model experiment has **not** been rerun with those fixes; 24 passing local tests cover the implementation. Raw pilot responses and compiler results are unchanged. The current report separately identifies interrupted episodes rather than treating them as mathematical failures.

## Where to look first

- `runs/first-pilot/REPORT.md`: comparison report, updated as runs finish.
- `runs/first-pilot/summary.json`: detailed machine-readable results.
- `runs/first-pilot/B*_r*/`: individual experiment transcripts and proofs.
- `catalog.py`: exact tasks and allowed mathematical declarations.
- `lean/Foundations.lean`: proofs of every supplied custom baseline lemma.
- `gate.py`: restricted proof-command interface.
- `run.py`: creative → verifier → skeptic → revision loop and scoring.

`runs/` is excluded from Git. These links refer to local output after running; no outputs are uploaded or published.

## Run from the repository root

```bash
python3 -m proof_harness.setup
python3 -m unittest discover -s proof_harness/tests -v
python3 -m proof_harness.run --name my-pilot --repetitions 3 --rounds 3 --workers 3
```

For a smaller follow-up:

```bash
python3 -m proof_harness.run --name small-check --baselines B0 B6 --repetitions 1 --rounds 2 --workers 1
```

Run names must be new. Existing results are never overwritten by a new run. Interrupted runs remain on disk; automatic resume is deliberately absent, so partial runs cannot silently be mistaken for complete comparisons. The report states the number of finished runs. Each completed episode has `result.json`.

The runner refuses to start or continue a new round outside `harrison-working-branch`. It has no push, commit, merge or PR code. It records Git refs before and after a run. Uncommitted files are still part of the current working directory and can accompany a manual branch switch; keep this branch open while evaluating the harness.

## Dependencies and local privacy

- Python standard library only; no Python packages are installed.
- Lean `4.24.0`, installed inside `.tools/elan` without changing global defaults.
- Mathlib tag `v4.24.0`, exact commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`.
- Mathlib's pinned dependency manifest and precompiled cache.
- Signed-in Codex CLI (tested with `0.167.0-alpha.4.1`).

The toolchain and cache take several GB. `.tools`, compiled Lean objects and run logs are ignored by Git. The repository-local `.git/info/exclude` also excludes the existing `.env` and `.DS_Store`; that ignore file is local and is not shared by Git. The harness never reads `.env`.

“Local” means no publication to GitHub. Experimental prompts are still sent to the signed-in model service to generate responses. Calls consume the user's Codex allowance. The first pilot is capped at 126 agent calls; setup used two short interface checks. There is no automatic API fallback, paid API key usage or heuristic replacement for model failures.

## Experimental units and baselines

One episode is a baseline plus an independent repetition. Each episode starts with all nine tasks, then revises only unsolved ones for at most three rounds. Every round uses one creative call and one skeptical call, with Lean verification in between. Both agents see the same allowed mathematical declarations. The critic sees the actual compiler feedback.

| Arm | Extra information beyond B0 |
|---|---|
| B0 | Elementary equality, field and linear-map facts, fixed arithmetic tactics |
| B1 | Recurrence manipulation and two-step renewal |
| B2 | Weighted symmetry and positive-weight cancellation |
| B3 | Vanishing on a span and injective cancellation |
| B4 | Normal equations and a Gram-kernel consequence |
| B5 | Moment uniqueness, scalar recovery and a product-difference identity |
| B6 | Union of B1–B5 |

The packs differ in mathematical strength and length. They are not claimed to be equally informative. B6 tests the combined scaffolding. Baseline names refer to this implementation's manifest; they should not be conflated with fully implemented general-theorem packs from the initial design discussion.

All task statements and hypotheses are fixed before the run. The model and effort are held constant. Episode ordering is shuffled with a recorded scheduling seed. Repetition indices do **not** set the provider's sampling seed or make the LLM deterministic. No memory is shared between episodes. Accepted tasks remain accepted within an episode and cannot earn points again.

## Mathematical scope

The source documents motivate these tasks:

1. *Vertex Centrality Reconstruction in an Inverse Problem for Information Diffusion*, arXiv:2603.20710v1: first passage, heat response, weighted adjoints, controls, harmonic-product moments.
2. *Computing Potential from Random Walk on Graphs* (the user's absorption note): substochastic transition operator `A_q = I + Delta - diag(q)` and unknown interior killing probabilities.

The PDF texts, full paper proofs and this chat history are never placed in experimental prompts. `catalog.py` contains the complete mathematical task context supplied by the harness.

The nine tasks are:

1. `renewal3`: third return/occupancy renewal identity.
2. `forced3`: three-step forced scalar recurrence.
3. `weighted_pair`: local weighted self-adjointness with diagonal terms.
4. `span_observable`: vanishing of a linear functional on reachable spans.
5. `control_pairing`: a normal equation gives a specified pairing, assuming adjoint injectivity.
6. `gram_unique`: identical Gram images give identical reached states.
7. `path_recovery`: solve for the interior mass from an assumed path return-probability formula.
8. `moment_unique`: injective moments identify parameters after subtracting a known boundary contribution.
9. `absorption_difference`: scalar cubic difference identity motivating a Duhamel argument.

Tasks 4–8 expose their hypotheses explicitly. Proving them does not prove graph controllability, derive the harmonic-product measurement system, establish generic rank, or establish global absorption uniqueness/stability. The full theorem is always recorded as `full_reconstruction_proved: false` in this pilot.

Weighted coordinate formulas must retain their metric matrices: `[W*] = G_F^-1 [W]^T G_X`. A future full reconstruction suite must check this and distinguish Euclidean projection from projection in the unknown weighted metric.

## What is enforced

Agents output structured JSON proof commands. The harness supplies the theorem statement; agents cannot replace it. A lexical gate permits only the named mathematical facts, locally introduced variables, mathematical definitions, and a fixed set of tactics. No arbitrary Lean files, raw tactic scripts, custom imports, declarations, options, IO or metaprogramming are accepted.

`simp` is rendered as `simp only`. `ring`, `linarith`, `nlinarith`, `norm_num` and `field_simp` are trusted shared arithmetic automation and may internally use their Mathlib dependencies. The restriction is on explicitly supplied/used mathematical declarations, not an assertion that every transitive arithmetic dependency is listed in the prompt.

All custom baseline statements are theorems proved in `Foundations.lean`, not axioms. Lean compiles each submitted proof independently and prints its axiom dependencies. Any compilation failure, unresolved goal, timeout or `sorryAx` earns zero points. Candidate proofs cannot name earlier candidate theorems or hidden reference solutions. The trusted foundations module contains all pack facts, but the gate prevents naming facts outside the assigned pack; the permitted tactics do not perform general theorem search.

Codex calls run with user configuration and exec rules ignored, project instructions disabled, host skills disabled, shell/browsing/connectors/subagents/hooks/memory disabled, and read-only sandboxing. Calls use fresh ephemeral sessions. Runtime item events are audited; tool events invalidate the call. The CLI may emit a harmless diagnostic that Code Mode is unavailable because its host is intentionally disabled.

This is a controlled research interface, not a general sandbox for arbitrary hostile Lean programs. Pretrained model knowledge cannot be removed. The claim is that accepted derivations use the approved interface and trusted foundations.

## Scores and interpretation

- Creator: +10 for each newly Lean-verified task, at most 90.
- Critic: +2 when a previously unsuccessful task becomes verified after a round of critique. This is an operational feedback reward, not evidence that the critic caused the improvement.
- Prose claims, numerical evidence, agent agreement and unsupported objections earn no formal points.
- Usage and failures are recorded separately. Three repetitions are a descriptive pilot, not a statistically strong comparison.

The agents' model weights never change. This is reward-guided iterative revision, not RL training. A future learning experiment would need an explicit policy update and held-out evaluation.

## Artifact layout

Each run includes a manifest with source hashes, exact dependency versions, model settings, Git refs and scheduling order. Each round saves prompts, raw JSON events, responses, usage, generated Lean proofs and compiler logs. `progress.json` records interim results; `result.json` marks a completed episode. `calls.jsonl` counts attempted model calls, including failures. No silent retry is performed.

The harness is intentionally strict about proof syntax. A failure may reflect a missing mathematical premise, an incorrect proof, a Lean syntax error, the restricted command language, a timeout, or a model-service failure. Consult the per-task status before drawing conclusions.

## Extending the study

Add new trusted theorem statements/proofs to `Foundations.lean`, expose selected declarations through `catalog.py`, add fixed tasks and verifier regression tests, recompile, then start a **new** named experiment. Never change the task catalogue or proof gate during an active comparison. Preserve old manifests and reports.

The next substantive stage is a full finite-dimensional centrality reconstruction theorem with weighted adjoints and a separate absorption study with explicit observation-set assumptions. Its success must be measured separately from this checkpoint suite.
