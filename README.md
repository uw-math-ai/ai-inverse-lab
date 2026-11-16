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

Defaults: three feedback rounds, at most six model calls, 2,400 output tokens per call, **zero simulator calls**, seed 2026. Runs stop early when Lean verifies the target. A completed run may still contain unfinished proofs; inspect its verification status. Provider errors, refusals and truncated responses preserve available records.

## Verification contract

The public packet includes definitions, the fixed goal, assumptions and allowed proof language. Reference proofs are withheld. A structured lean4 certificate contains a target_id and tactic body. The agent cannot supply a replacement theorem statement or additional hypotheses.

[verification.py](verification.py):

1. Matches the claim and assumptions to the repository-owned packet.
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

Paired trials share target, model, seed and caps. Multi-agent mode spends calls on critiques; single-agent mode can spend them on more proposals. Early success stops either arm. Compare actual calls, tokens, checker time, failures and formal success. A few descriptive trials do not establish multi-agent superiority.

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
