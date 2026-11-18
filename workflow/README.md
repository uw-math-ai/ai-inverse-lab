# Lean-first mathematical research workflow

The active path is **formulate a precise statement → propose a proof → challenge it → check in Lean → repair from feedback**. Learning an inverse map from forward-simulation data is a separate future path. Simulator calls default to **zero**. No simulation campaign or ML training runs automatically.

The default backend is a scripted demonstration, not evidence of autonomous mathematical discovery. Live model runs require an explicit choice.

## Current formal results

The starting point is the killing kernel in the [forward simulator](../forwardSimulation/forward_sim/model.py):

$$
K_q=I+h(\Delta-\operatorname{diag}q),\qquad
r(x)=\frac{1}{\mu_x}\sum_y w_{xy}.
$$

First **positive-time** return includes a holding step. Its one-step diagonal observation is $d(x)=F_{xx}(1)=K_q(x,x)=1-h(r(x)+q(x))$. With known rates and known nonzero $h$,

$$
\boxed{q(x)=\frac{1-d(x)}{h}-r(x),\qquad x\in B.}
$$

[Spec.lean](lean/MathAILab/Spec.lean) fixes two algebraic goals. [Proofs.lean](lean/MathAILab/Proofs.lean) contains reference proofs checked by Lean 4.35.0.

| Target | Precisely what Lean checks |
| --- | --- |
| q-recovery-one-step (default) | The formula recovers $q(x)$ at every observed vertex, given the exact diagonal relation and $h\ne0$. |
| q-uniqueness-full-observation | Equal one-step diagonal observations at **all** vertices, with the same $h$ and rates, imply equal potentials. |
| arxiv-2603.20710-lemma-2.3 | Original discrete heat controllability benchmark retained; **formalization pending**. It cannot receive automatic Lean certification. |

The two proved results quantify over arbitrary fields through Lean.Grind.Field and arbitrary vertex types. They use exact algebra, not floating-point samples. They are elementary formalization milestones, not newly discovered recovery theorems.

The specification defines the diagonal algebra. It does **not** yet formalize Markov probabilities, graph geometry, a concrete real-number model, or the connection between a stochastic experiment and that algebra. The simulator source motivates that connection; a mathematical reviewer must check it. A probabilistic application additionally requires positive masses, $h>0$, $q\ge0$, and $h(r(x)+q(x))\le1$.

This recovers $q$ **on $B$**. Full observation gives full recovery. Recovering hidden vertices, finding the weakest conditions on $B$, and proving stability remain research tasks. Potential $q$ and mass $\mu$ are different unknowns.

## Run one formal workflow

From the repository root:

~~~bash
source .venv/bin/activate
python -m pip install -r workflow/requirements.txt
python -m workflow.response_call lean-status
python -m workflow.response_call benchmark
python -m workflow.response_call run --backend demo --rounds 2
~~~

[requirements.txt](requirements.txt) no longer includes SymPy. Lean is a separate pinned toolchain. If missing, install Lean/elan using the [official instructions](https://lean-lang.org/install/), then:

~~~bash
elan toolchain install leanprover/lean4:v4.35.0
cd workflow/lean
lake build
~~~

This project uses Lean's bundled library without Mathlib. Broader spectral/graph formalization will need additional reviewed definitions and likely Mathlib. The runner discovers the pinned toolchain in the usual elan directory, or lean on PATH. MATHAI_LEAN_BIN can specify a trusted local compiler; its version must match. Nothing downloads during verification. Missing Lean stops a formal run before any model call.

Other targets and the single-agent baseline:

~~~bash
python -m workflow.response_call run --benchmark q-uniqueness-full-observation --rounds 2
python -m workflow.response_call run --mode single --rounds 2
python -m workflow.response_call benchmark --benchmark arxiv-2603.20710-lemma-2.3
~~~

Configure a key privately from an interactive terminal, or use VS Code's **Workflow: configure OpenAI API key** task:

~~~bash
python -m workflow.response_call configure-openai
~~~

The prompt hides input, checks access to the selected model using its metadata endpoint, and saves the key in the ignored `.secrets/openai.key` file with owner-only permissions. The setup check does not generate model tokens or launch an experiment. Use `--model MODEL_ID` to check a different model; this does not change run defaults. Restricted keys need permission to read model metadata for the setup check.

The saved project key takes precedence over OPENAI_API_KEY in the environment. Both the CLI and dashboard read it automatically; the dashboard detects a newly saved key on its next refresh. Delete the local file to resume using the environment key. The key is never returned to the browser or included in run configuration. Check access again with `python -m workflow.response_call check-openai`.

Then explicitly request a live attempt:

~~~bash
python -m workflow.response_call run --backend openai --model gpt-4o-mini --rounds 3
~~~

Defaults: ten feedback rounds, 16,000 output tokens per call, **zero simulator calls**, seed 2026. Runs continue after verification by default, with a hard cap of ten iterations. Select early stopping explicitly when desired. A completed run may still contain unfinished proofs; inspect its verification status. Provider errors, refusals and truncated responses preserve available records.

## Verification contract

The public packet includes definitions, the fixed goal, assumptions and allowed proof language. Reference proofs are withheld. A structured lean4 certificate contains a target_id and tactic body. The agent cannot supply a replacement theorem statement or additional hypotheses.

[verification.py](verification.py):

1. Binds the certificate target ID to the repository-owned packet. Narrative wording is logged separately and cannot change the theorem or its hypotheses.
2. Accepts only the documented tactic/identifier subset. Commands, imports, comments, strings, new axioms, sorry, admit, native_decide, IO and metaprogramming are rejected.
3. Inserts the body into a fixed declaration and invokes the actual pinned Lean compiler.
4. Requires successful checking and a #print axioms audit containing only the standard foundations propext, Classical.choice, and Quot.sound.
5. Records exact source, source hash, compiler version, exit status, diagnostics, duration and axioms.

Limits: 30 seconds wall time, 2,048 MB Lean memory, heartbeats, bounded source size and bounded captured diagnostics. The compiler receives a minimal environment without API credentials. The proof language currently supports the initial algebraic targets; adding targets includes reviewing their specifications and allowed tactics.

**Failed** means the attempted proof did not check; it does not refute the theorem. Unavailable, timeout, rejected and formalization_pending never count as verification. **Verified** applies only to the fixed formal target, not every sentence of accompanying prose.

## Feedback and records

Multi-agent: **CA → SA → Lean → record → repair → SA …**. Single-agent: **CA → Lean → record → self-revision …**. Both receive identical public statements and configured caps. Compiler diagnostics, previous gaps and claimed repairs feed the next revision. See [runtime.py](runtime.py), [schemas.py](schemas.py) and [backends.py](backends.py).

Each run records:

| File | Contents |
| --- | --- |
| manifest.json | Configuration, public packet and hashes, dependencies and Lean toolchain. |
| events.jsonl | Prompts, schemas, raw responses, usage, checks and errors. |
| iterations.jsonl | Claims, assumptions, prose and Lean attempts, critiques, gaps, repairs, verification and scores. |
| agent_evaluations.jsonl | One self-contained training record per model call: role, claim, assumptions, proof, gaps, repairs, output hash, evidence-based score, progress, tokens and response time. Failed calls remain visible. |
| proof.lean | Standalone source for the latest accepted proof body, successful or failed; absent when the latest attempt supplied no accepted body. |
| lean-verification.json | Latest checker result and compiler diagnostics. |
| summary.json, report.md | Final outcomes and human-readable report. |
| reviews/, summary_adjudicated.json | Independent reviews stored separately from original records. |

Every iteration's complete source remains in its verification record. Recheck a saved attempt without trusting its stored success flag:

~~~bash
python -m workflow.response_call verify-lean "workflow/runs/<run-id>" --iteration 2
~~~

Rechecking uses the current trusted specification and rejects changed packets. Historical reproduction requires the recorded source and toolchain.

## Evaluation

| Score | Evidence |
| --- | --- |
| 0 | No prose attempt and no successful formal proof. |
| 1 | An attempted prose or Lean proof, not yet independently established. |
| 2 | Independent review confirms a specific prior mathematical gap was repaired. |
| 3 | A fresh Lean check verifies the unchanged fixed target, or independent review validates a complete proof. |

Automatic score 3 requires an in-process checker receipt bound to the exact proposal and result; a serialized success flag cannot create that receipt. Stored reports retain historical outcomes, and verify-lean reruns the checker. Reviews bind to run, iteration and proposal hash. They can supersede the displayed score without deleting original evidence; reviewers are responsible for their attestations.

A scripted proof can be checked by Lean, but that score says nothing about an AI's discovery ability. These easy targets validate integration before evaluating harder reasoning.

### Evaluate each agent at every iteration

New runs open on the dashboard's **Evaluation** tab. Each action has its own assessment and usage record. CA and Repair use the proof rubric above. SA uses a separate rubric:

| Score | Skeptic evidence |
| --- | --- |
| 0 | No usable critique, or independent review dismisses all reported gaps without another supported finding. |
| 1 | A critique is recorded, but its mathematical diagnoses remain unconfirmed. |
| 2 | Correctly detects an exact target/assumption mismatch, or independent review confirms some findings. |
| 3 | Independent review validates all gap findings (or explicitly validates an empty-gap assessment), and packet checks are correct. |

A failed Lean check does **not** confirm a skeptic's diagnosis. A successful Lean check establishes the formal proof outcome, but does **not** confirm each claimed repair or prove that the skeptic caused the success. The Review tab lets a reviewer confirm/dismiss each skeptic gap, attest a no-gap assessment, and confirm specific prior repairs. Select **Evaluate the proof outcome** to change the proof assessment; reviewing the skeptic alone preserves it. Reviews bind to the exact proposal and critique hashes. Unreviewed, confirmed and dismissed findings stay separate; precision is reported only over reviewed findings.

Every action records a score rationale and dimensions such as target/assumption preservation, proof changes, gap IDs linked to repairs, and per-call token counts/time. Repair compares with the previous proof attempt; SA compares with the previous critique. Changes may improve, stay flat or regress. An increase is labelled confirmed only when the new score has independent evidence. These ordinal differences are descriptive, not a continuous reward or a causal estimate. Provider failures are unscored; invalid structured output scores zero for usability. Missing token metadata is unknown rather than zero.

The dashboard's **Agent results** tab presents each iteration in order: **CA's claim, assumptions and proposed proof with its proof score → SA's assessment, gaps and suggested repairs with its critique score → Lean's verification result with its own pass score**. Later proposals are labelled Repair, and each stage includes its score rationale and evidence. A table compares the three outcomes across iterations. Failed calls and incomplete attempts remain inspectable; historical actions without role assessments stay unscored.

CA/Repair and SA retain their separate 0–3 rubrics. Lean has a **binary verification pass score**, not a model-performance score: **1/1** when the saved result reports a successful compiler run, kernel check and axiom audit; **0/1** for rejected submissions or conclusive compiler failures; **unscored** for missing proofs, unavailable tools, timeouts, truncated diagnostics, incomplete records or historical non-Lean checks. This display-only summary of the saved verification record does not re-run Lean, award a proof score, or modify training records. A failed proof check does not refute its mathematical claim. These different scores are not added together.

Single-agent runs explicitly show that SA was not run, with no invented skeptic score. Comparisons aggregate model-role outcomes separately by mode, backend and model. No extra model calls are used for evaluation, so the configured budget remains unchanged.

The original JSONL export is immutable. To produce a separate dataset that incorporates independently validated reviews:

~~~bash
python -m workflow.response_call export-training "workflow/runs/<run-id>" --reviewed --output "workflow/runs/training-reviewed.jsonl"
~~~

Omit `--reviewed` for the original automatic assessments. Existing destinations are rejected to preserve records. Historical runs without per-agent records remain available with their original iteration scores; new agent scores are not fabricated for them. See [agent_evaluation.py](agent_evaluation.py) for the versioned rubric and [evaluation.py](evaluation.py) for review validation.

~~~bash
python -m workflow.response_call review-template "workflow/runs/<run-id>" --reviewer "Independent reviewer"
python -m workflow.response_call review "workflow/runs/<run-id>" "<completed-review.json>"

# Optional after useful formal targets are established:
python -m workflow.response_call compare --backend demo --trials 1 --rounds 2
~~~

Paired trials share target, model, seed and caps. Multi-agent mode spends calls on critiques; single-agent mode can spend them on more proposals. Both arms continue after success by default; use --stop-on-verified for early stopping. No arm exceeds ten iterations. Compare actual calls, tokens, checker time, failures and formal success. A few descriptive trials do not establish multi-agent superiority. The trial seed labels paired trials and is not sent to the model; reproducible model sampling is not guaranteed.

## Dashboard

~~~bash
python -m workflow.dashboard --port 8766
~~~

Open **http://127.0.0.1:8766**, or use VS Code's **Workflow: management dashboard** task. The dashboard shows Lean availability, formalization milestones, target selection, attempt history, compiler errors, axiom audits, source, reviews and exports. Historical SymPy runs remain archived finite checks and are never relabelled as Lean results.

The server is local-only, with one worker, at most four queued/active jobs and a one-hour timeout. Cancellation preserves records; restarted unfinished jobs are interrupted and never automatically retried. Use --data-dir for a different run collection. API keys remain on the server, loaded from private local configuration or its environment.

Mathematical prose uses locally bundled [KaTeX](https://katex.org/docs/autorender.html): $…$, \(…\), $$…$$, \[…\], aligned equations and matrices. Lean source, ordinary code, raw JSON and exports retain literal text. Long equations scroll within their panels. See [web/](web/).

## Research direction

1. Review the exact observation model and formal goals. Extend the algebra to finite weighted graphs and first-passage/return processes.
2. Formalize the selected published lemmas as substantial benchmarks. The original [Lemma 2.3 statement](https://arxiv.org/html/2603.20710v1#S2.SS2) is retained; its spectral/controllability proof is not yet formalized.
3. For known finite connected graphs with all-time observations, determine which partial observation sets identify hidden potentials. Prove reconstruction lemmas or construct valid counterexamples; necessity of Two-Points Condition or unique continuation must be proved.
4. Derive and formally verify a reconstruction procedure, then prove stability under a specified noise model and observation horizon.
5. Separately, train an inverse model using forward-simulation data and compare it with the mathematical results. The [simulator adapter](simulator.py) remains available for explicitly enabled diagnostic probes, not proof certification.

## Checks

~~~bash
python -m unittest discover -s workflow/tests -v
cd workflow/lean
lake build
~~~

The suite includes real compiler success/failure, fixed-target enforcement, hole/axiom rejection, credential separation, feedback routing, early stopping, scoring, record preservation and dashboard lifecycle. It requires the pinned Lean compiler; missing Lean is a failed prerequisite, not a silently skipped proof test.

Optional Chromium checks:

~~~bash
python -m pip install -r workflow/requirements-dev.txt
python -m playwright install chromium
python -m workflow.tests.browser_dashboard_smoke
python -m workflow.tests.browser_math_smoke
~~~

Browser jobs and synthetic reviews use temporary directories. Screenshots go under the ignored run directory. No live model API calls are made.


## Benchmark campaign management (v2)

The [frozen q contract](q_problem.json) fixes first positive-time passage observations, known graph/masses/conductances, admissible nonnegative potential, and a finite observation horizon. It is a non-executable research reference. No campaign transfers results to hidden-q research.

The dashboard's **Benchmark** library exposes all public packets and the frozen contract. **Launch experiment** now supports a full five-task development suite, per-role OpenAI model identifiers (CA/repair, SA, optional verification adviser), output allowance, ten iterations, and two budget policies. Save/load launch settings in your browser; launch and cancel campaigns through the persisted job queue. API keys remain server-side. Blank author/skeptic overrides inherit the default model; a blank adviser disables that extra model call. Models must support the existing structured-output API and the chosen output allowance. Other providers are not yet implemented.

- **Iterations:** each arm gets up to the specified number of attempts. Different roles mean different call counts.
- **Calls:** both arms receive the same call ceiling, `(2 + adviser_enabled) * rounds`; the single-agent arm can use more attempts.

Both policies continue refinement after success by default and report actual usage. Neither can exceed ten iterations; --stop-on-verified restores optional early stopping. Neither guarantees equal monetary cost. The CLI retains call matching by default for compatibility; the frontend defaults to ten equal iterations. The model output allowance is configurable up to 131,072 tokens; provider-specific limits still apply. Prose accepts up to 200,000 characters. The executable Lean proof body remains bounded to 16,000 characters with the existing compiler limits and tactic allowlist.

### Development suite and review status

[research_suite.py](research_suite.py) provides five fixed tasks: affine injectivity, two-measurement uniqueness, a sum-observation counterexample, invalid-cancellation repair, and formalization of a supplied valid argument. A campaign checks every evaluator-only reference proof with the pinned Lean compiler before calling a model. The references are excluded from live prompts. Scripted demo mode deliberately uses fixtures and is never evidence of discovery.

These are small calibration tasks, not a novel or held-out research leaderboard. The affine tasks are correlated and should not be counted as independent research results. Specifications have implementation review and machine-checked references; **independent human mathematical review is pending**. This status is visible in the frontend. A formal proof validates its exact statement, not the intended interpretation or the supplied prose.

~~~bash
python -m workflow.response_call suite-check
python -m workflow.response_call suite --backend demo --rounds 10 --budget-policy iterations --trials 1
# Live: choose supported models and configure a valid key first.
python -m workflow.response_call suite --backend openai --model gpt-4o-mini --rounds 10 --budget-policy iterations --trials 1
~~~

### Persistent proof ledger and richer evaluations

Every completed iteration snapshots its proof ledger, also exported as `proof-ledger.json`. Repair receives the full ledger: previous attempts, all reported obligations, claimed repairs, and previously verified supporting milestones. A failed subsequent attempt cannot erase a verified milestone. Milestones must match repository-listed public targets and independently pass Lean; arbitrary lemma text cannot acquire proof credit. The two-measurement task lists affine cancellation as a supporting milestone, without claiming it proves the entire target. Ledger gaps remain unconfirmed unless reviewed separately; a successful final theorem does not retrospectively validate every critique.

The **Ledger** tab shows persistence and the best verified attempt. **Agent results** exposes each agent's exact input and output, input/output hashes, configured model, token counts, latency, and role-specific evidence. The optional verification adviser predicts success before seeing the current compiler result. Its prediction is compared only against conclusive checker outcomes; uncertainty, timeout, missing certificates and infrastructure failures do not become mathematical mistakes. Its explanations remain unreviewed and cannot award proof scores.

Comparisons include task-level results, per-role/model groups, verified outcomes by iteration, first-verification time/calls, repeated attempts and verified milestone counts. Pair matching includes role settings and budget policy. Original run evidence remains immutable; human review overlays are revalidated separately. The suite runner checkpoints its aggregate after every run and marks incomplete campaigns explicitly. No causal contribution claim is inferred from later success; checkpoint intervention/replay experiments and held-out task families remain future extensions.


## Readable results and checking stages

The frontend typesets the suite statements, frozen-problem notation, delimited LaTeX, and recognizable simple algebra. Agent prose, critique, repair notes and ledger descriptions use the same renderer. Raw Lean, compiler JSON, and original records stay available in collapsed technical disclosures. If a historical agent returned Lean code in the prose field, the UI explicitly says no written explanation was provided; it does not invent one. New prompts request prose and equations separately from the executable certificate.

Run details show the workflow mode and completed calls by role. Single-agent baselines intentionally omit SA, with a link to the paired multi-agent run where available. The verification adviser is a model making predictions; the Lean checker is separate software. A preflight rejection is labelled **Submission blocked before Lean**, while compiler failures, missing certificates and axiom-audit failures have separate explanations. A checker observation is not automatically a confirmed mathematical gap.

The version-3 proof-language contract is shared between prompts and the verifier. Simple local names introduced by `intro`, `have` and `let`, plus Lean's implicit `this`, are accepted; commands, metaprogramming and unknown global lemmas remain blocked. Lean checks binding, types and proof completion. Revised packets have new hashes; saved historical runs are never retroactively upgraded or silently compared with revised conditions.


## Formal target binding and progress (v3)

The formal certificate's `target_id` selects a repository-owned theorem. The agent supplies only its tactic body. Prose claim/assumption text is retained exactly for review; literal differences no longer block the compiler and never change the formal goal. Verification records include `narrative_alignment` with literal matching flags and `semantic_equivalence: not_assessed`. A Lean pass establishes only the fixed formal target, even if the prose is inaccurate. SA is no longer automatically rewarded for detecting harmless wording differences under this contract. Historical packets keep their earlier rubric and provenance.

Each new iteration records `progress_evidence`: whether Lean ran, whether the certificate repeated, newly verified supporting milestones, and whether the full goal was proved. Reaching the compiler is an interface improvement; it is not a mathematical proof milestone. The frontend shows these distinctions in Agent results and the Ledger tab. Repairs are prompted to change strategy when the same certificate or obstacle recurs. No synthetic percentage-of-proof-complete score is assigned.

`--reasoning-effort low|medium|high|none` is available for compatible models and in the launch form. Leave it empty for provider defaults or nonreasoning models. Settings are recorded in manifests/summaries and included in comparison matching. Example live attempt:

~~~bash
python -m workflow.response_call run --backend openai --mode multi --benchmark bench-two-measurements --model gpt-5.1 --verifier-model gpt-5.1 --reasoning-effort high --rounds 10 --budget-policy iterations --max-output-tokens 32768
~~~

This is a different model/configuration from the earlier gpt-4o-mini pilot, so any gain is not a controlled estimate of the benefit of multiple agents. Reference proofs remain withheld. Successful attempts are rechecked using the exact recorded certificate and the fixed specification.


## Continued refinement after verification

All runs have a hard ceiling of **10 iterations**, including single-agent call-matched baselines. By default, success does not terminate the loop. CA/repair continues auditing the explanation, simplifying the proof or proposing independently checked alternatives; SA and the optional verification adviser continue participating in each iteration. Repeating a valid proof is recorded but is not labelled new mathematical progress. Provider errors, cancellation and the dashboard wall-clock limit can still stop a run before the cap.

The first accepted certificate is retained in `verified-proof.lean` and `verified-checkpoint.json`. Every later repair receives this checkpoint. The ordinary `proof.lean` and latest verification record still describe the latest attempt, including failures; they cannot overwrite the retained successful proof. Run summaries distinguish `has_verified_proof` / `verified_iteration` from the latest score. Comparison success counts include a verified proof from any iteration, while the final score still describes the last attempt.

The frontend's **After a verified proof** selector defaults to **Continue reviewing and refining**. Choose **Stop on first verified proof**, or pass `--stop-on-verified` in the CLI, to use the previous early-stop behavior. `--rounds` accepts 1 through 10; call matching may grant the single-agent baseline more attempts than the selected rounds, but never more than ten.


## Inspect CA revisions

In run details, select iteration 2 or later and click **Compare revisions** on the CA/Repair card, or open the **revisions** tab. The default baseline is the preceding recorded iteration; the selector can compare against any earlier recorded attempt.

The comparison shows the recorded role, call ID, model, and checker outcome on both sides. It marks which proposal fields changed, typesets the two written proofs side by side, and highlights exact added/removed Lean lines with old/new line numbers. Expand **Highlight exact prose changes** for a textual diff or **Feedback supplied to this revision** for the recorded request feedback. Claimed repairs remain explicitly labelled as agent assertions. Unchanged prose or certificates are called out; neither textual changes nor self-reported repairs are credited as mathematical progress.

CA names the author role. Later `Repair` actions are fresh model calls using that role's configured model, previous proposal, feedback, ledger, and any retained verified checkpoint. The UI compares saved outputs without rewriting them or launching additional model calls. Historical actions without request input display that limitation instead of reconstructing missing context.
