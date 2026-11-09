# fptsim: first-passage-time data for random weighted graphs

`fptsim` is a command-line tool driven by one YAML config. For thousands of random weighted graphs it

1. generates the graph (family, size, weights, μ, observed set B, potential q),
2. computes the **exact** first-passage distribution r(t, x, y) for (x, y) ∈ B×B, t = 1..T,
3. simulates the **empirical** r′ by Monte Carlo on GPU (CUDA) or CPU, with N walks per ordered pair and R independent replicates,
4. saves graphs, parameters, r, counts and metrics in a form ready for inverse-problem / ML work,
5. runs unattended for hours. It is resumable, logged, reproducible, and shardable across GPUs and SLURM array jobs.

Reference implementations it was built from (read-only, not imported at runtime):
`../Graphs - First Passage Time Simulation/First Passage Time Simulation (executed on americano).ipynb` and
`Kevin_data_simulation/forward_sim/`. The paper is Gao, Li & Yang, arXiv:2603.20710, Section 5.

```bash
cd "Toggable Simulation of Data Code"
pip install -r requirements.txt
python -m fptsim validate                                  # paper8/paper9 regression + pytest
python -m fptsim run --config configs/example_small.yaml --dry-run
python -m fptsim run --config configs/example_small.yaml
python -m fptsim summarize --run results/example_small
python -m fptsim plot --run results/example_small --graph-id 3
python -m fptsim bench                                     # walks/s on every backend/device
```

---

## 1. Mathematical model

A weighted graph G = (X, E, W, μ): W is symmetric, nonnegative, with zero diagonal; μ > 0.
Observed set B ⊂ X, potential q ≥ 0 per vertex, step size h (`dt`).

| quantity | definition |
|---|---|
| s(x) | Σ_y W[x,y] / μ(x) |
| K[x,y], x ≠ y | h · W[x,y] / μ(x) |
| K[x,x] | 1 − h · (s(x) + q(x)) |
| killing probability | h · q(x) |
| P | (n+1)×(n+1): K on X, killing column to the absorbing cemetery state n. P must be row-stochastic |

**Choosing h.** `dt: auto` gives h = min(1, 1 / maxₓ(s(x)+q(x))). An explicit h must satisfy 0 < h ≤ 1 and
h·(s+q) ≤ 1 everywhere, otherwise it is an error. Rows are never renormalised. At q = 0 and h = 1 this is the
paper's P (Assumption 1(ii): μ(x) ≥ Σ_y W[x,y]).

**First passage.** τ(x,y) = inf{t ≥ 1 : H_t = y}. For x = y this is the first *return*, and a holding step at
t = 1 counts as a return, so r(1,y,y) = K[y,y].

**Exact recursion** (`exact.py`), evaluated over all start vertices and then restricted to B×B:
r(1,x,y) = K[x,y];  r(t,x,y) = Σ_{z≠y} K[x,z] · r(t−1,z,y).
A second, independent forward computation propagates each start distribution with the target made absorbing.
It gives hits, killed and censored masses. The two must agree to 1e-12, and the total mass must be 1 to 1e-11.
Both checks run on every graph. Graphs are padded to a common n_max and run in one batched matmul (torch float64
on GPU, numpy fallback).

**Outcome bins.** Every walk ends in exactly one bin: first hit at t = 1..T, *killed* before hitting, or
*censored* (alive with no hit at step T). Exact: hits + killed + censored = 1. Empirical: counts sum to N exactly.
Both are asserted.

**Empirical distribution.** r′ = counts / N, divided by **all** N trials, not only the walks that hit.

**Time convention.** `T` is the maximum number of steps. The paper's T_paper = 9 corresponds to
T = 2·9 − 1 = 17 steps. `simulation.T_mode: paper` applies T → 2T − 1 (also to `sweep.T`).
Physical time is t·h.

## 2. Metrics (per graph × replicate × sweep point; `metrics.parquet`)

| column | meaning |
|---|---|
| `frne_percent` | 100‖r′ − r‖_F / ‖r‖_F on the (T, \|B\|, \|B\|) tensor (no-arrival bin excluded) |
| `expected_rms_frne_percent` | 100 · sqrt(Σ r(1−r)/N) / ‖r‖_F, the RMS FRNE of an unbiased sampler |
| `pair_frne_percent` | list, row-major over (source index, target index) in B |
| `max_abs_error` | max \|r′ − r\| |
| `tv_mean`, `tv_max` | total variation per pair, including the no-arrival bin (killed + censored) |
| `ci_coverage` | fraction of the T+2 bins per pair whose 95 % Wilson CI covers the exact value |
| `ci_coverage_np5` | the same, restricted to bins with N·p ≥ 5 (close to 0.95 for an unbiased sampler) |
| `hit_mass_*`, `killed_mass_*`, `censored_mass_*` | mean over pairs, `_exact` and `_emp` |
| `mean_fpt_*` | mean over pairs of E[τ \| τ ≤ T] |
| `frne_N{m}`, `exp_frne_N{m}` | convergence from prefix counts at each checkpoint m (like Kevin's `convergence`) |
| `exact_max_disagreement`, `exact_mass_error` | results of the two exact checks |
| `elapsed_s` | exact + Monte Carlo seconds for this unit (shared launches are split pro rata) |
| identity | `graph_id, sweep_index, replicate, family, n, k, num_edges, q_scale, N, T, T_input, dt, dt_mode, backend, device, seed_spawn_key, status` |

`status` is `ok`, or `invalid_kernel` when a sweep's explicit `dt` is too large for that graph (the unit is
recorded and skipped). Mean and std of FRNE across replicates, next to the expected RMS, are reported by
`summarize` (`summary/replicates.csv`).

## 3. Config reference

Every key is validated strictly. Unknown keys, wrong types and out-of-range values are errors that name the
dotted key. YAML 1.1 reads `1e6` as a string, so numeric strings such as `"1e6"` and `1_000_000` are accepted.
The resolved config, with all defaults filled in, is saved as `config.resolved.yaml`.

**Parameter specs.** Family parameters and `B.k` accept a fixed value `5`, an inclusive range `[2, 6]`
(ints uniform on {2..6}, floats U(2, 6)), or a list `{choice: [8, 12, 20]}`.

### top level
| key | default | meaning |
|---|---|---|
| `run_name` | `fptsim_run` | label |
| `output_dir` | `results/<run_name>` | relative to the working directory |
| `master_seed` | 0 | root of the seed tree |
| `num_graphs` | required | graphs 0..num_graphs−1. Can be increased on resume |
| `device` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `backend` | `auto` | `auto` (torch if installed) \| `torch` \| `numpy` (CPU only) |
| `devices` | null | e.g. `[0, 1]`: one worker process per GPU, each taking whole shards |
| `max_batch_walks` | 5e7 | cap on walks per GPU launch. The actual size also follows free memory (`torch.cuda.mem_get_info`) |
| `cpu_batch_walks` | 4e6 | cap on walks per launch on CPU (torch-cpu backend) |
| `shard_size` | 500 | graphs per tensor shard file. Also the unit of `--shard` and multi-GPU splitting |
| `graphs_per_batch` | 64 | graphs generated and solved exactly together |
| `log_interval_s` | 30 | seconds between progress lines |
| `plot_sample` | 5 | graphs plotted when `toggles.make_plots` is on |

### `toggles`
`compute_exact` (true), `run_monte_carlo` (true), `save_counts` (true), `save_r_tensors` (true),
`save_ci` (false), `convergence_checkpoints` (false), `make_plots` (false), `resume` (true).
At least one of `compute_exact` and `run_monte_carlo` must be on. Without exact r, only empirical metrics are
written. Without Monte Carlo, each sweep point has a single unit with replicate 0.

### `graphs`
| key | meaning |
|---|---|
| `families` | `{name: {weight, params...}}`, sampled with probability ∝ `weight` (default 1) |
| `presets` | `[paper8, paper9]` instead of `families`: graph i is `presets[i % len]`. `weights`/`mu`/`B` must then be omitted |
| `weights` | edge weights, see below |
| `mu` | vertex measure, see below |
| `B` | observed set, see below (required with `families`) |
| `q` | potential, see below (default `zero`) |
| `dt` | `auto` or 0 < h ≤ 1 |
| `max_retries` | 100: attempts per graph before giving up (disconnected, μ skip, invalid dt) |

Families (parameters):

| family | params | notes |
|---|---|---|
| `grid` | `rows`, `cols` | vertex = row·cols + col, coordinates kept |
| `grid_holes` | `rows`, `cols`, `hole_frac` | removes round(hole_frac·n) vertices in random order, skipping removals that disconnect |
| `pattern` | `patterns` | one or several "XooX" pictures, as in the notebook, chosen uniformly |
| `ladder` | `length` | 2 × length grid |
| `path`, `cycle`, `star`, `complete`, `random_tree` | `n` | random_tree = uniform labelled tree (Prüfer) |
| `erdos_renyi` | `n`, `p` | |
| `random_geometric` | `n`, `radius` | unit square, coordinates kept |
| `watts_strogatz` | `n`, `k`, `beta` | networkx |
| `barabasi_albert` | `n`, `m` | networkx |
| `random_regular` | `n`, `d` | networkx. n is raised by 1 if n·d is odd |

Disconnected draws are rejected and resampled. Lattice families (grid, grid_holes, pattern, ladder) have a
`boundary` (vertices with fewer than 4 lattice neighbours).

`weights`: `dist` = `constant {value}` \| `uniform {low>0, high}` \| `lognormal {mean, sigma}` \|
`discrete {values, probs}` \| `gamma {shape, scale}`. Optional `rescale: {mode: max|mean, value}`.

`mu`: `mode` = `constant {value}` \| `deg` \| `weighted_deg` \| `uniform {low, high}` \|
`slack {a, b}` (μ = weighted_deg·(1 + U(a,b))). `on_violation` = `fix` (raise μ to the row sum, default) \|
`resample` (redraw a random μ) \| `skip` (discard the graph and draw another).

`B`: `mode` = `random_k {k | frac}` \| `boundary` (lattice families only) \| `leaves` (degree-1 vertices,
else the lowest-degree vertex) \| `low_degree {k}` \| `high_degree {k}` \| `explicit {vertices}`.
Optional `max_k` caps the size with a random subset. B is always nonempty and duplicate-free, and is sorted
except for `explicit`.

`q`: `mode` = `zero` \| `constant {c}` \| `uniform {low, high}` \| `sparse {fraction, low, high}` \|
`localized {center: random|int, decay, amplitude}` (amplitude·exp(−hops/decay)) \| `explicit {values}`.
`on_B_only` / `off_B_only` zero q off / on B.

### `simulation`
| key | default | meaning |
|---|---|---|
| `T` | required | maximum step (see `T_mode`) |
| `T_mode` | `steps` | `steps` \| `paper` (T → 2T−1) |
| `N` | required | walks per ordered pair per replicate |
| `replicates` | 1 | R independent Monte Carlo runs per graph |
| `N_checkpoints` | null | prefix sizes for convergence. With the toggle on and no list, powers of 10 below N are used |
| `chunk_walks` | 1e5 | walks per pair per seeded work item (see seeding). Keep k²·chunk_walks within GPU memory |

### `sweep` (optional)
Cartesian product, in the order `q_scale × N × T × dt`, applied to the same graphs. `q_scale` multiplies q.
`dt` entries are `auto` or numbers. Sweep point s is a column (`sweep_index`) in every table.

## 4. Seeding and reproducibility

The seed tree is rooted at `SeedSequence(master_seed)`:

```
graph i   generation    SeedSequence(master_seed, spawn_key=(i, 0))
          Monte Carlo   SeedSequence(master_seed, spawn_key=(i, 1, sweep, replicate, chunk))
```

Graph i can be regenerated from (master_seed, i) alone (`fptsim.generators.generate_graph`). The Monte Carlo
for a (graph, sweep, replicate) is split into chunks of `chunk_walks` walks per pair, and each chunk is an
independently seeded work item. The torch backend packs many items (from many graphs) into one launch, but each
item draws from its own `torch.Generator`, so **results do not depend on the batch size, the free GPU memory, the
number of GPUs, sharding, or where a run was interrupted**. The numpy backend uses Kevin's layout: one PCG64
stream per (item seed, source, target).

The torch and numpy backends are statistically equivalent but **not bit-identical**, because their random streams
differ. CUDA and torch-CPU streams also differ from each other. Results are reproducible for a fixed
backend + device type. The seed spawn key is stored in `graphs.parquet` (`spawn_key`) and in
`metrics.parquet` (`seed_spawn_key`).

## 5. Monte Carlo engine

* Sparse per-graph sampling tables: for each state, the states reachable in one step (neighbours, the holding
  step, the cemetery) and their float64 cumulative probabilities. A batch holds tables of shape (G, n_max+1, D).
  Padded vertices are unreachable self-loops, and all cemeteries map to index n_max.
* States are int16 when n_max < 32k. Walks are a flat list grouped by item. Finished walks are compacted away
  every step, and the loop exits early when none are left.
* Hits per t, killed and censored are counted in one bincount over (pair, checkpoint bucket, bin). Prefix counts
  for convergence are cumulative sums over the checkpoint buckets.
* Launch size is chosen from free GPU memory (≈ (9·D + 96) bytes per live walk) and capped by `max_batch_walks`.
* Multi-GPU: `devices: [0, 1, ...]` starts one spawn-context worker process per GPU. Workers take disjoint shards
  and write `done.cuda<i>.jsonl`, `run.cuda<i>.log`.

## 6. Command line

| command | does |
|---|---|
| `generate --config X [--limit n] [--shard i/k]` | generate and save graphs only (arrays + `graphs.parquet`) |
| `run --config X [--dry-run] [--resume/--no-resume] [--limit n] [--shard i/k] [--device D] [--backend B] [--output-dir P]` | generate as needed, then exact + Monte Carlo for every pending unit |
| `summarize --run DIR` | merged `metrics.parquet`/`graphs.parquet`, `summary/*.csv`: overall, replicate variability, FRNE by family / n / q_scale / q_mean / N, FRNE vs N, coverage |
| `plot --run DIR --graph-id i [--sweep s] [--replicate r]` | `plots/graph_i.png` (coloured by μ and by q, B ringed, edge width ∝ w) and `plots/graph_i_s{s}_r{r}.png` (r′ bars vs r line) |
| `validate [--N 1e6] [--replicates 3] [--skip-tests]` | paper8/paper9 regression at T = 17, then the pytest suite |
| `bench [--seconds 5]` | Monte Carlo walks/s and walk-steps/s for numpy, torch-cpu and each CUDA device |

`--dry-run` prints the number of graphs and units, the walk count, an upper bound on walk-steps with the
measured fraction still alive, the walks per launch and estimated peak GPU memory, and an estimated runtime
from a short benchmark on sample graphs.

Logging goes to the console and to `output_dir/run.log`. A progress line every `log_interval_s` shows units
done/total, walks/s, walk-steps/s, ETA and peak GPU memory. **Ctrl-C** (or SIGTERM) stops cleanly after the
current batch. A second Ctrl-C aborts at once, and output is still never corrupt.

## 7. Output format (`output_dir/`)

| file | content |
|---|---|
| `config.resolved.yaml` | resolved config |
| `env{.tag}.json` | python, numpy, torch, CUDA, GPU names, networkx/pandas versions, git commit, host, argv, start/end time, interrupted flag |
| `run{.tag}.log` | log |
| `graphs{.tag}.jsonl` / `graphs.parquet` | one row per graph: `graph_id, family, params_json, n, num_edges, k, B, deg_min/mean/max/std, wdeg_mean, mu_*, q_*, dt, dt_mode, lambda_max, lambda_2, lambda_min, spectral_gap, abs_spectral_gap, master_seed, spawn_key, attempts, has_coords` |
| `done{.tag}.jsonl` / `metrics.parquet` | one row per (graph, sweep point, replicate), see §2 |
| `tensors/shard_XXXX.npz` | arrays for graphs [XXXX·shard_size, (XXXX+1)·shard_size) |
| `tensors/parts/*.npz` | arrays of units not yet consolidated into a shard |
| `summary/`, `plots/` | from `summarize` / `plot` / `make_plots` |

`.tag` is empty for a single process. It is `shard{i}of{k}` for `--shard i/k`, and `cuda{d}` (or
`shard..of...cuda{d}`) for multi-GPU workers. The loader merges all of them.

Array keys inside shards (shapes for |B| = k):

```
g{id}__W (n,n)  __mu (n)  __B (k)  __q (n)  __coords (n,2)  __boundary (n)
g{id}__s{s}__K (n,n)  __kill (n)  __dt ()  __r_exact (T,k,k)  __exact_killed (k,k)  __exact_censored (k,k)
g{id}__s{s}__r{r}__counts (T,k,k)  __killed (k,k)  __censored (k,k)
                   __prefix_counts (C,k,k,T+2)  __checkpoints (C)      [convergence_checkpoints]
                   __ci_lower (T,k,k)  __ci_upper (T,k,k)              [save_ci]
```

Axes are (time t−1, source index in B, target index in B). The bins of `prefix_counts` are hits t = 1..T, then
killed, then censored. Lattice coordinates are drawing positions (col, −row).

**Crash safety.** Every file is written to a temp name, fsynced, then renamed. A unit's arrays are written
before its `done` line is appended. Consolidation rewrites the shard atomically and only then deletes the
parts. A truncated last `done` line is skipped and repaired on the next start. **Resume** (`toggles.resume` or
`--resume`) skips units listed in `done*.jsonl` and recomputes anything half-done with identical results. It
refuses to start if the settings that affect results (`master_seed`, `shard_size`, `graphs`, `simulation`,
`sweep`, exact/MC/convergence toggles) differ from `config.resolved.yaml`. Increasing `num_graphs` is
allowed.

**Loading.**

```python
import fptsim
run = fptsim.load_run("results/random_mix_v1")
run.graphs, run.metrics            # pandas DataFrames
t = run.tensors(17)                # lazily loaded: {"W", "mu", "B", "q", ..., "sweeps": {s: {"K", "r_exact", ...,
                                   #                  "replicates": {r: {"counts", "killed", "censored", ...}}}}}
g = run.graph(17)                  # fptsim.graphs.Graph
r_emp = run.r_empirical(17, s=0, r=0)
```

## 8. Server usage

* **nohup**: `bash scripts/run_server.sh configs/server_large.yaml 0` sets `CUDA_VISIBLE_DEVICES`, prints a dry
  run, starts the run in the background with `--resume`, and stores the PID. Stop it with `kill -TERM <pid>`.
  Rerun the same command to continue.
* **SLURM array**: `sbatch scripts/slurm_array.sbatch configs/server_large.yaml`. Task i of k runs
  `--shard i/k`, i.e. the shards with `shard % k == i`. Tasks never touch the same file, and `summarize` merges
  them. `--signal=B:TERM@120` gives a clean stop before the time limit, and `--requeue` plus `--resume`
  continues the work. Choose `shard_size` so that `num_graphs / shard_size ≥` the number of tasks.
* **Several GPUs in one job**: `devices: [0, 1, 2, 3]`.

## 9. Tests

`python -m pytest` (from this folder; `-m "not slow"` skips the N = 1e6 regression):

| file | covers |
|---|---|
| `test_closed_form.py` | 2- and 3-state chains in closed form, first returns, holding steps, killed/censored masses |
| `test_paper_literal.py` | paper8/paper9 vs the notebook's literal recursion (copied into the test) to 1e-14 |
| `test_kevin_agreement.py` | K, P, r, hits/killed/censored vs Kevin's `exact_first_passage` / `exact_outcomes`, q = 0 and q > 0 |
| `test_mass_conservation.py` | exact mass = 1 and counts = N on random graphs with q > 0. Batched = single |
| `test_mc_unbiased.py` | mean FRNE over R = 20 within ±20 % of expected RMS. Wilson coverage ≈ 95 % |
| `test_paper_regression.py` | paper8 ≈ 0.25 %, paper9 ≈ 0.19 % at N = 1e6 (marked `slow`) |
| `test_backends.py` | torch vs numpy statistically equal. Launch packing does not change results. CUDA (skipped without GPU) |
| `test_generators.py` | every family, B and q modes, μ violation handling, dt validity, same seed gives the same graph |
| `test_resume.py` | stop + resume equals an uninterrupted run. Sharded jobs merge. Guards against mismatched resumes |
| `test_config.py` | rejection messages, numeric strings, sweeps, shipped configs |
| `test_pipeline.py` | run → load → summarize → plot, sweeps, exact-only / MC-only, dry run, invalid dt |
