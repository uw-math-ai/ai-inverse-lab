# Forward Simulation Lab

A reproducible data simulator for first-passage observations on fixed weighted graphs. The immediate goal is to reproduce the **forward-data stage of Section 5** of Gao, Li and Yang, *Vertex Centrality Reconstruction in an Inverse Problem for Information Diffusion*, then study how a time-independent vertex potential changes those observations.

## Project scope

For a fixed graph, edge weights `w`, vertex measures `μ`, and observed vertices `B`, we will:

1. Construct the transition kernel from the graph and a fixed potential `q`.
2. Compute the exact first-passage distribution `r` by the paper's recursion.
3. Simulate `N` independent walks **per ordered pair** `(x,y) ∈ B × B`, stopping at first arrival, killing, or the selected maximum step `T`.
4. Compare the empirical distribution `r′` against `r`, including the paper's Frobenius relative norm error (FRNE), probability mass accounting, and sampling uncertainty.
5. Manage reproducible experiments through a local browser dashboard, with graph inspection, potential controls, comparison plots, saved runs, and data export.

This is the data-generation foundation for later inverse reconstruction and the lab's conjecture/counterexample/verification experiments. Implementing Algorithm 1's inverse recovery of `μ`, or proving new identifiability results for nonzero `q`, is a later stage. The forward simulator must first provide trustworthy data and exact reference values.

## How the readings inform the project

- [VertexCentralityReconstuction.pdf](readings/VertexCentralityReconstuction.pdf): primary specification. Equation (1.1) defines the walk, Section 1.2 defines first passage/first return, and Section 5 supplies the recurrence, experiment graphs, and FRNE. [arXiv version 1](https://arxiv.org/abs/2603.20710v1).
- [INVERSE PROBLEMS FOR DISCRETE HEAT EQUATIONS AND RANDOM WALKS FOR A CLASS OF GRAPHS.pdf](readings/INVERSE%20PROBLEMS%20FOR%20DISCRETE%20HEAT%20EQUATIONS%20AND%20RANDOM%20WALKS%20FOR%20A%20CLASS%20OF%20GRAPHS.pdf): relates weighted graph Laplacians, heat evolution, partial observations, and the operator `−Δ + q`. Its potential equation motivates the explicitly documented extension below; the Section 5 walk itself uses `q = 0`.
- [Chapter11.pdf](readings/Chapter11.pdf): Markov transition matrices and absorbing chains, used to distinguish occupancy probabilities `(P^t)[x,y]` from first-passage probabilities. The supplied attachment is a read-only reference.
- [DiscoveringMathConceptsThruAMulti-AgentSystem.pdf](readings/DiscoveringMathConceptsThruAMulti-AgentSystem.pdf) and [MathAI_MathConcepts_MultiAgent.pptx](readings/MathAI_MathConcepts_MultiAgent.pptx): broader motivation for generating controlled observations, seeking counterexamples, and testing uniqueness. The slides' four-vertex path is a separate pilot, not one of the two Section 5 graphs.
- [SimpleProof.lean](readings/SimpleProof.lean): a small formal-verification demonstration; it does not specify the stochastic model.

## Step 1 — reproduce the fixed graphs

Vertices are labeled in **row-major order**, left to right and then top to bottom; the paper's figure vertices are unlabeled.

| Preset | Topology | Observed set B | Edge weight | Vertex measure |
| --- | --- | --- | --- | --- |
| `paper8` | 2 rows × 4 columns; horizontal and vertical nearest neighbors (10 edges) | `{1,2,3}` (top row except leftmost) | `w = 0.25` | `μ = 1` |
| `paper9` | 3 rows × 3 columns; horizontal and vertical nearest neighbors (12 edges) | `{0,1,2}` (top row) | `w = 1` | `μ(x) = degree(x)` |
| `path7` | Seven-vertex path | `{0,3,6}` | `w = 1` | `μ(x) = degree(x)` |
| `cycle8` | Eight-vertex cycle | `{0,3,6}` | `w = 1` | `μ(x) = degree(x)` |
| `star7` | Seven-vertex star, center vertex `0` | `{0,1,2}` | `w = 1` | `μ(x) = degree(x)` |
| `lollipop8` | Four-clique on `0,1,2,3` with path `3-4-5-6-7` | `{0,3,7}` | `w = 1` | `μ(x) = degree(x)` |

The first two presets were transcribed directly from Figures 1 and 4. No graph self-loops are added; a positive holding probability is a transition, not an extra graph edge.

The graph, weights, measures, and observation set remain fixed within each preset. `q`, maximum time, sample count, and random seed are experiment inputs.

## Step 2 — define the kernel and potential precisely

Write `s(x) = Σ_y w(x,y)/μ(x)`. The paper's unit-step kernel is

```text
P(x,y) = w(x,y)/μ(x)        for adjacent distinct vertices
P(x,x) = 1 − s(x).
```

The intended probability-preserving extension uses a **nonnegative, time-independent killing potential** `q(x)` in

```text
(u(t+h) − u(t))/h = Δu(t) − q u(t).
K(x,y) = h w(x,y)/μ(x)     for adjacent distinct vertices
K(x,x) = 1 − h(s(x) + q(x))
P(x, cemetery) = h q(x);   P(cemetery, cemetery) = 1.
```

`K` is the kernel restricted to the original vertices and has row sum `1 − h q(x)`. The augmented matrix including the cemetery state is stochastic. A walk killed before reaching its target never contributes an arrival.

Require `q(x) ≥ 0` and `h(s(x)+q(x)) ≤ 1` at every vertex. Automatically choose `h = min(1, 1/max_x(s(x)+q(x)))`, or validate an explicitly supplied step size. This keeps the topology, `w`, and `μ` unchanged. In particular, the nine-vertex normalized walk has zero holding probability at `h=1`, so adding any positive killing potential requires a smaller step size. Do not silently renormalize rows or modify the graph to hide this constraint.

At `q=0`, automatic `h=1` recovers the paper exactly. Signed potentials are not ordinary killing probabilities and are outside this probability simulator's initial scope.

## Step 3 — keep time conventions explicit

In this application, **`T` means the maximum number of simulated transitions**, and arrival bins are `1,2,…,T` (inclusive). Physical elapsed time is `t*h`; the terminal physical time is `T*h`.

The paper uses a reconstruction parameter also named `T` and simulates up to `2T−1`. Its `T_paper=9` therefore corresponds to **application `T=17`**, with `h=1`, `N=1,000,000` per ordered pair, and `q=0`. The dashboard will include a shortcut for these settings. This naming distinction avoids an off-by-one or factor-of-two comparison error.

## Step 4 — compute the exact reference independently

Define the strictly positive first-passage time

```text
τ(x,y) = inf { t ≥ 1 : H_t = y }.
r(1,x,y) = K(x,y)
r(t,x,y) = Σ_(z ≠ y) K(x,z) r(t−1,z,y),   t ≥ 2.
```

For each target `y`, evaluate the recursion for **every possible starting vertex**, then restrict the output to `B × B`. Intermediate states are not restricted to observed vertices. When `x=y`, this is the first return after time zero; a holding transition counts as a return at `t=1`.

A separate forward propagation with target absorption will account for arrivals, killing before arrival, and paths still alive without arrival at `T`. These mutually exclusive categories must sum to one. The finite-horizon no-hit bin combines killing and still-unarrived paths; it does **not** assert that every truncated path has infinite mathematical hitting time.

## Step 5 — Monte Carlo and comparison

For each ordered observed pair, generate `N` independent paths from `x`. Stop each path at its first arrival at `y`, death, or `T`. Count each trajectory once and divide finite-time arrival counts by **all `N` trials**, not only the successful trials. Otherwise the result would be conditional on arrival and would not estimate the paper's `r`.

Use seeded random streams, bounded batches, progress reporting, and cancellation. Save graph metadata and the full normalized experiment configuration with each result. Compute

```text
FRNE (%) = 100 ||r′ − r||_F / ||r||_F
```

on the finite-time `(T, |B|, |B|)` tensor, excluding the no-hit bin exactly as in Section 5. Also report per-pair total variation including the no-hit bin, maximum absolute error, and pointwise binomial confidence intervals. Confidence intervals are per bin, not a simultaneous guarantee for the whole tensor. Increasing `N` reduces typical Monte Carlo error at order `N^(−1/2)`, but no single seeded run is guaranteed to improve monotonically.

## Implementation sequence

1. Document scope, sources, graph transcription, time conventions, and the potential model (this document).
2. Implement graph presets, strict parameter validation, the exact recursion, independent mass accounting, and seeded vectorized Monte Carlo.
3. Add a local API with background jobs, cancellation, experiment storage, and JSON/CSV downloads.
4. Build the browser dashboard to configure runs and inspect the graph, distributions, errors, and experiment history.
5. Verify hand-computable chains, first returns, both graph topologies, killing and censoring, reproducibility, Monte Carlo agreement, API behavior, and the frontend workflow.
6. Record the finished run instructions, data schema, validation results, and limitations here.

## Original project note

### Here is the codespace for Math AI Lab

If you can edit this file, write your name below and save the file:
-
