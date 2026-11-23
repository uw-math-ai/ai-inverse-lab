"""Run orchestration: graph batches -> kernels -> exact -> Monte Carlo -> metrics -> atomic output.

A unit of work is one (graph_id, sweep_index, replicate). A unit counts as done once its arrays
are on disk and its line is in a done*.jsonl file. Resume skips done units; anything half-finished
is recomputed, with the same result, because every Monte Carlo chunk has its own seed.

Graph shards (graph_id // shard_size) are the unit of ownership. `--shard i/k` takes the shards
with shard % k == i, and multi-GPU workers split this job's shards among themselves, so no two
processes ever write the same file.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from pathlib import Path

import numpy as np

from . import io
from .config import ConfigError, checkpoints_for, dump_config, science_view, sweep_points
from .exact import exact_batch
from .generators import generate_graph, graph_record
from .kernel import KernelError, build_kernel, transition_table
from .mc_common import (MCItem, bytes_per_walk, chunk_plan, counts_to_outcomes, device_banner,
                        make_engine, mc_seed, resolve_execution)
from .metrics import compute_metrics

log = logging.getLogger("fptsim")


# ---------------------------------------------------------------- logging and signals
def setup_logging(out: Path | None = None, tag: str = "", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("fptsim")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(processName)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out) / f"run{io.suffix(tag)}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def close_file_logging() -> None:
    """Detach and close run.log handlers (keeps the console handler)."""
    logger = logging.getLogger("fptsim")
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()


class StopFlag:
    """Ctrl-C / SIGTERM: the first signal finishes the current batch, a second one aborts."""

    def __init__(self):
        self._set = False
        self._signals = 0
        self._previous = {}
        self.children: list = []

    def is_set(self) -> bool:
        return self._set

    def set(self, reason: str = "") -> None:
        if not self._set:
            log.warning("stopping after the current batch%s", f" ({reason})" if reason else "")
        self._set = True

    def _handler(self, signum, frame):
        self._signals += 1
        if self._signals > 1:
            raise KeyboardInterrupt
        self.set(f"signal {signum}; press Ctrl-C again to abort immediately")
        if signum == getattr(signal, "SIGTERM", None):
            for p in self.children:          # Ctrl-C reaches workers directly; SIGTERM may not
                if p.is_alive():
                    p.terminate()

    def install(self) -> None:
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):    # not the main thread
                pass

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._previous.clear()


# ---------------------------------------------------------------- planning
def job_graph_ids(cfg: dict, shard=(0, 1), limit: int | None = None) -> list[int]:
    i, k = shard
    if not (0 <= i < k):
        raise ConfigError(f"--shard: need 0 <= i < k, got {i}/{k}")
    ss = cfg["shard_size"]
    ids = [g for g in range(cfg["num_graphs"]) if (g // ss) % k == i]
    return ids[:limit] if limit is not None else ids


def _diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            out += _diff(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
        return out
    return [] if a == b else [path or "<root>"]


def prepare_output(out: Path, cfg: dict, resume: bool) -> None:
    import yaml
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "config.resolved.yaml"
    has_results = any(p.stat().st_size > 0 for p in io.done_files(out))
    if has_results:
        if not resume:
            raise ConfigError(f"{out} already contains results; set toggles.resume: true or pass "
                              "--resume to continue, or choose a new output_dir")
        if resolved.exists():
            old = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            diff = _diff(science_view(old), science_view(cfg))
            if diff:
                raise ConfigError("cannot resume: these settings differ from the existing run: "
                                  + ", ".join(diff))
    io.atomic_write_text(resolved, dump_config(cfg))
    removed = io.clean_stale_temps(out)
    if removed:
        log.info("removed %d stale temp files", removed)


def _fmt_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ---------------------------------------------------------------- worker
class Worker:
    """Processes a list of graph ids on one device."""

    def __init__(self, cfg, out, gids, backend, device, tag, stop, stop_after_units=None):
        self.cfg, self.out, self.gids = cfg, Path(out), list(gids)
        self.backend, self.device, self.tag, self.stop = backend, device, tag, stop
        self.stop_after_units = stop_after_units
        t = cfg["toggles"]
        self.do_exact, self.run_mc = t["compute_exact"], t["run_monte_carlo"]
        self.points = sweep_points(cfg)
        self.R = cfg["simulation"]["replicates"] if self.run_mc else 1
        sfx = io.suffix(tag)
        self.done_path = self.out / f"done{sfx}.jsonl"
        self.graphs_path = self.out / f"graphs{sfx}.jsonl"
        io.repair_jsonl_tail(self.done_path)
        io.repair_jsonl_tail(self.graphs_path)
        self.done = io.read_done_keys(self.out)
        self.saved_graphs = {row["graph_id"] for row in io.read_graph_rows(self.out)}
        self.engine = make_engine(backend, device, cfg) if self.run_mc else None
        self.exact_device = device if backend == "torch" else None
        self.units_now = 0
        self.walks_now = 0
        self.started = time.perf_counter()
        self.last_log = 0.0
        self.pending_start = sum(1 for g in self.gids for s in range(len(self.points))
                                 for r in range(self.R) if (g, s, r) not in self.done)

    def _graph_pending(self, g) -> bool:
        return any((g, s, r) not in self.done for s in range(len(self.points)) for r in range(self.R))

    def run(self) -> None:
        total = len(self.gids) * len(self.points) * self.R
        log.info("%d graphs x %d sweep points x %d replicates = %d units; %d pending",
                 len(self.gids), len(self.points), self.R, total, self.pending_start)
        ss, gpb = self.cfg["shard_size"], self.cfg["graphs_per_batch"]
        by_shard: dict[int, list[int]] = {}
        for g in self.gids:
            by_shard.setdefault(g // ss, []).append(g)
        for shard, gids in sorted(by_shard.items()):
            # fixed blocks of graph ids (not "the next pending graphs"), so a resumed run solves
            # the exact problem in the same padded batches and reproduces it bit for bit
            for i in range(0, len(gids), gpb):
                if self.stop.is_set():
                    break
                block = gids[i:i + gpb]
                if any(self._graph_pending(g) for g in block):
                    self._process_batch(block)
            merged = io.consolidate_shard(self.out, shard, ss)
            if merged:
                log.info("shard %04d: consolidated %d part files", shard, merged)
            if self.stop.is_set():
                break
        self.write_tables()
        self._log_progress(force=True)
        if self.stop.is_set():
            log.warning("stopped cleanly; run again with --resume to continue")

    # ------------------------------------------------------------ one batch of graphs
    def _process_batch(self, batch: list[int]) -> None:
        cfg = self.cfg
        graphs = {}
        for g in batch:
            graphs[g] = generate_graph(cfg["graphs"], cfg["master_seed"], g)
            if g not in self.saved_graphs:
                self._save_graph(graphs[g])
        for s, pt in enumerate(self.points):
            if self.stop.is_set():
                return
            need = [g for g in batch if any((g, s, r) not in self.done for r in range(self.R))]
            if not need:
                continue
            kernels, exacts = {}, {}
            for g in batch:
                gr = graphs[g]
                try:
                    kernels[g] = build_kernel(gr.W, gr.mu, gr.q * pt["q_scale"], pt["dt"])
                except KernelError as exc:
                    if g not in need:
                        continue
                    log.warning("graph %d sweep %d: %s; marking its units invalid", g, s, exc)
                    for r in range(self.R):
                        if (g, s, r) not in self.done:
                            self._record(g, s, r, {"status": "invalid_kernel", "message": str(exc)})
            valid = [g for g in need if g in kernels]
            if not valid:
                continue
            if self.do_exact:
                # the whole block, done graphs included, so the padded batches never change
                solved = [g for g in batch if g in kernels]
                t0 = time.perf_counter()
                res = exact_batch([kernels[g] for g in solved], [graphs[g].B for g in solved],
                                  pt["T"], device=self.exact_device)
                share = (time.perf_counter() - t0) / len(solved)
                for g, e in zip(solved, res):
                    e["seconds"] = share
                    exacts[g] = e
            for g in valid:
                self._save_exact(g, s, kernels[g], exacts.get(g))
            if not self.run_mc:
                for g in valid:
                    if (g, s, 0) not in self.done:
                        self._finish_unit(g, s, 0, pt, graphs[g], kernels[g], exacts.get(g), None,
                                          exacts.get(g, {}).get("seconds", 0.0))
                continue
            self._monte_carlo(s, pt, valid, graphs, kernels, exacts)

    def _monte_carlo(self, s, pt, valid, graphs, kernels, exacts) -> None:
        cfg = self.cfg
        cps = np.asarray(checkpoints_for(cfg, pt["N"]), dtype=np.int64)
        plan = chunk_plan(pt["N"], cfg["simulation"]["chunk_walks"])
        items, acc = [], {}
        for g in valid:
            nbr, cdf = transition_table(kernels[g].P)
            for r in range(self.R):
                if (g, s, r) in self.done:
                    continue
                acc[(g, s, r)] = [len(plan), None, 0.0]       # chunks left, summed counts, seconds
                for c, (start, m) in enumerate(plan):
                    items.append(MCItem(key=(g, s, r, c), nbr=nbr, cdf=cdf, B=graphs[g].B, T=pt["T"],
                                        m=m, start=start, checkpoints=cps,
                                        seed=mc_seed(cfg["master_seed"], g, s, r, c)))
        for idx, counts, secs in self.engine.run_items(items, stop=self.stop.is_set):
            g, s_, r, _ = items[idx].key
            a = acc[(g, s_, r)]
            a[0] -= 1
            a[1] = counts if a[1] is None else a[1] + counts
            a[2] += secs
            self.walks_now += items[idx].walks
            if a[0] == 0:
                seconds = a[2] + exacts.get(g, {}).get("seconds", 0.0)
                self._finish_unit(g, s_, r, pt, graphs[g], kernels[g], exacts.get(g), a[1], seconds)
                del acc[(g, s_, r)]
            self._log_progress()
            if self.stop.is_set():
                break

    # ------------------------------------------------------------ saving
    def _save_graph(self, graph) -> None:
        p = f"g{graph.graph_id}__"
        arrays = {p + "W": graph.W, p + "mu": graph.mu, p + "B": graph.B, p + "q": graph.q}
        if graph.coords is not None:
            arrays[p + "coords"] = graph.coords
        if graph.boundary is not None:
            arrays[p + "boundary"] = graph.boundary
        io.atomic_save_npz(io.part_path(self.out, graph.graph_id), arrays)
        io.append_jsonl(self.graphs_path, graph_record(graph, self.cfg["graphs"]["dt"]))
        self.saved_graphs.add(graph.graph_id)

    def _save_exact(self, g, s, kernel, exact) -> None:
        p = f"g{g}__s{s}__"
        arrays = {p + "K": kernel.K, p + "kill": kernel.kill, p + "dt": np.array(kernel.dt)}
        if exact is not None and self.cfg["toggles"]["save_r_tensors"]:
            arrays.update({p + "r_exact": exact["r"], p + "exact_killed": exact["killed"],
                           p + "exact_censored": exact["censored"]})
        io.atomic_save_npz(io.part_path(self.out, g, s), arrays)

    def _base_row(self, g, s, r) -> dict:
        pt = self.points[s]
        return {"graph_id": g, "sweep_index": s, "replicate": r, "q_scale": pt["q_scale"],
                "N": pt["N"], "T": pt["T"], "T_input": pt["T_input"], "dt_requested": str(pt["dt"]),
                "backend": self.backend if self.run_mc else None, "device": self.device,
                "seed_spawn_key": f"{self.cfg['master_seed']}/({g},1,{s},{r},chunk)"}

    def _record(self, g, s, r, fields: dict) -> None:
        row = {**self._base_row(g, s, r), **fields}
        io.append_jsonl(self.done_path, {"key": [g, s, r], "status": row.get("status", "ok"), "metrics": row})
        self.done.add((g, s, r))

    def _finish_unit(self, g, s, r, pt, graph, kernel, exact, raw, seconds) -> None:
        t = self.cfg["toggles"]
        N, T = pt["N"], pt["T"]
        p = f"g{g}__s{s}__r{r}__"
        arrays = {}
        row = {"status": "ok", "family": graph.family, "n": graph.n, "k": int(len(graph.B)),
               "num_edges": graph.num_edges, "dt": kernel.dt, "dt_mode": kernel.dt_mode}
        if exact is not None:
            row["exact_max_disagreement"] = exact["max_disagreement"]
            row["exact_mass_error"] = exact["mass_error"]
        if raw is None:
            metrics, _ = compute_metrics(N, exact)
        else:
            hits, killed, censored, prefix = counts_to_outcomes(raw, T)
            if not np.all(hits.sum(axis=0) + killed + censored == N):
                raise ArithmeticError(f"unit {(g, s, r)}: counts do not sum to N = {N}")
            cps = checkpoints_for(self.cfg, N)
            conv = t["convergence_checkpoints"]
            metrics, extras = compute_metrics(N, exact, hits, killed, censored,
                                              prefix if conv else None, cps if conv else None)
            if t["save_counts"]:
                arrays.update({p + "counts": hits, p + "killed": killed, p + "censored": censored})
                if conv:
                    arrays[p + "prefix_counts"] = prefix
                    arrays[p + "checkpoints"] = np.asarray(cps, dtype=np.int64)
            if t["save_ci"]:
                arrays[p + "ci_lower"], arrays[p + "ci_upper"] = extras["ci_lower"], extras["ci_upper"]
        row.update(metrics)
        row["elapsed_s"] = seconds
        if arrays:
            io.atomic_save_npz(io.part_path(self.out, g, s, r), arrays)
        self._record(g, s, r, row)
        self.units_now += 1
        if self.stop_after_units is not None and self.units_now >= self.stop_after_units:
            self.stop.set("stop_after_units reached")

    def write_tables(self) -> None:
        sfx = io.suffix(self.tag)
        try:
            import pandas as pd
            rows = io.read_jsonl(self.graphs_path)
            if rows:
                df = pd.DataFrame(rows).drop_duplicates("graph_id", keep="last").sort_values("graph_id")
                io.atomic_write_parquet(df, self.out / f"graphs{sfx}.parquet")
            records = io.read_jsonl(self.done_path)
            if records:
                io.atomic_write_parquet(io.metrics_frame(self.out, records), self.out / f"metrics{sfx}.parquet")
        except ImportError as exc:
            log.warning("parquet tables not written (%s); the jsonl files hold the same data", exc)

    def _log_progress(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_log < self.cfg["log_interval_s"]:
            return
        self.last_log = now
        elapsed = now - self.started
        rate = self.walks_now / elapsed if elapsed > 0 else 0.0
        remaining = self.pending_start - self.units_now
        eta = elapsed / self.units_now * remaining if self.units_now else float("nan")
        mem = ""
        if self.engine is not None and str(self.device).startswith("cuda"):
            import torch
            mem = f" | GPU mem peak {torch.cuda.max_memory_allocated(self.device) / 2 ** 30:.2f} GiB"
        steps = getattr(self.engine, "walk_steps", 0) if self.engine else 0
        log.info("units %d/%d | %.3g walks/s, %.3g walk-steps/s | elapsed %s, ETA %s%s",
                 self.units_now, self.pending_start, rate, steps / elapsed if elapsed > 0 else 0.0,
                 _fmt_seconds(elapsed), _fmt_seconds(eta), mem)


# ---------------------------------------------------------------- entry points
def _worker_entry(cfg, out, gids, backend, device, tag):
    setup_logging(Path(out), tag)
    stop = StopFlag()
    stop.install()
    log.info(device_banner(backend, device))
    try:
        Worker(cfg, out, gids, backend, device, tag, stop).run()
    finally:
        close_file_logging()


def run(cfg: dict, *, resume: bool | None = None, limit: int | None = None, shard=(0, 1),
        stop_after_units: int | None = None) -> Path:
    """Generate graphs as needed, then compute exact + Monte Carlo for every pending unit."""
    out = Path(cfg["output_dir"])
    tag = io.shard_tag(shard)
    setup_logging(out, tag)
    resume = cfg["toggles"]["resume"] if resume is None else resume
    prepare_output(out, cfg, resume)
    backend, devices = resolve_execution(cfg)
    for dev in devices:
        log.info(device_banner(backend, dev))
    env = io.environment_info(backend, devices)
    io.write_env(out, tag, env)
    gids = job_graph_ids(cfg, shard, limit)
    stop = StopFlag()
    stop.install()
    try:
        if len(devices) > 1:
            _run_multi(cfg, out, gids, backend, devices, tag, stop)
        else:
            Worker(cfg, out, gids, backend, devices[0], tag, stop, stop_after_units).run()
        if cfg["toggles"]["make_plots"] and not stop.is_set() and cfg["plot_sample"] > 0:
            _sample_plots(out, gids[:cfg["plot_sample"]])
    finally:
        stop.restore()
        env["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        env["interrupted"] = stop.is_set()
        io.write_env(out, tag, env)
        close_file_logging()
    return out


def _run_multi(cfg, out, gids, backend, devices, tag, stop) -> None:
    ss = cfg["shard_size"]
    shards = sorted({g // ss for g in gids})
    ctx = mp.get_context("spawn")
    procs = []
    for w, dev in enumerate(devices):
        mine = set(shards[w::len(devices)])
        my_gids = [g for g in gids if g // ss in mine]
        if not my_gids:
            continue
        wtag = ".".join(x for x in (tag, dev.replace(":", "")) if x)
        p = ctx.Process(target=_worker_entry, args=(cfg, str(out), my_gids, backend, dev, wtag),
                        name=f"worker-{dev}")
        p.start()
        procs.append(p)
        log.info("started %s with %d graphs (%d shards)", p.name, len(my_gids), len(mine))
    stop.children = procs
    while any(p.is_alive() for p in procs):
        for p in procs:
            p.join(timeout=1.0)
    failed = [p.name for p in procs if p.exitcode != 0]
    if failed and not stop.is_set():
        raise RuntimeError(f"workers failed: {failed}; see their run.*.log files")


def _sample_plots(out: Path, gids) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from .plots import plot_graph_report
    for g in gids:
        try:
            plot_graph_report(out, g)
        except Exception as exc:              # plots must never fail a finished run
            log.warning("plot for graph %d failed: %s", g, exc)


def generate_only(cfg: dict, *, limit: int | None = None, shard=(0, 1)) -> Path:
    """Generate and save graphs (arrays + graphs table) without any simulation."""
    out = Path(cfg["output_dir"])
    tag = io.shard_tag(shard)
    setup_logging(out, tag)
    prepare_output(out, cfg, resume=True)
    gids = job_graph_ids(cfg, shard, limit)
    graphs_path = out / f"graphs{io.suffix(tag)}.jsonl"
    io.repair_jsonl_tail(graphs_path)
    saved = {row["graph_id"] for row in io.read_graph_rows(out)}
    ss = cfg["shard_size"]
    started, last = time.perf_counter(), 0.0
    shards_touched = set()
    for i, g in enumerate(gids):
        if g in saved:
            continue
        graph = generate_graph(cfg["graphs"], cfg["master_seed"], g)
        p = f"g{g}__"
        arrays = {p + "W": graph.W, p + "mu": graph.mu, p + "B": graph.B, p + "q": graph.q}
        if graph.coords is not None:
            arrays[p + "coords"] = graph.coords
        if graph.boundary is not None:
            arrays[p + "boundary"] = graph.boundary
        io.atomic_save_npz(io.part_path(out, g), arrays)
        io.append_jsonl(graphs_path, graph_record(graph, cfg["graphs"]["dt"]))
        shards_touched.add(g // ss)
        if time.perf_counter() - last > cfg["log_interval_s"]:
            last = time.perf_counter()
            log.info("generated %d/%d graphs (%.1f graphs/s)", i + 1, len(gids), (i + 1) / (last - started))
    for shard_id in sorted(shards_touched):
        io.consolidate_shard(out, shard_id, ss)
    import pandas as pd
    rows = io.read_jsonl(graphs_path)
    if rows:
        io.atomic_write_parquet(pd.DataFrame(rows).drop_duplicates("graph_id", keep="last")
                                .sort_values("graph_id"), out / f"graphs{io.suffix(tag)}.parquet")
    log.info("generated %d graphs into %s", len(gids), out)
    close_file_logging()
    return out


# ---------------------------------------------------------------- dry run
def dry_run(cfg: dict, *, limit: int | None = None, shard=(0, 1), sample: int = 200) -> dict:
    """Estimate work, memory and runtime from a sample of graphs and a short benchmark."""
    setup_logging(None)
    backend, devices = resolve_execution(cfg)
    for dev in devices:
        log.info(device_banner(backend, dev))
    gids = job_graph_ids(cfg, shard, limit)
    points = sweep_points(cfg)
    run_mc = cfg["toggles"]["run_monte_carlo"]
    R = cfg["simulation"]["replicates"] if run_mc else 1
    sample_ids = gids[:min(sample, len(gids))]
    t0 = time.perf_counter()
    graphs = [generate_graph(cfg["graphs"], cfg["master_seed"], g) for g in sample_ids]
    gen_s = (time.perf_counter() - t0) / max(len(graphs), 1)
    k2 = np.mean([len(g.B) ** 2 for g in graphs])
    ns = np.array([g.n for g in graphs])
    kernels = [build_kernel(g.W, g.mu, g.q * points[0]["q_scale"], points[0]["dt"]) for g in graphs]
    D = max(transition_table(k.P)[0].shape[1] for k in kernels)
    walks = sum(pt["N"] * R * k2 * len(gids) for pt in points) if run_mc else 0
    upper_steps = sum(pt["N"] * R * k2 * pt["T"] * len(gids) for pt in points) if run_mc else 0

    est: dict = {"graphs": len(gids), "sweep_points": len(points), "replicates": R,
                 "units": len(gids) * len(points) * R, "mean_n": float(ns.mean()), "max_n": int(ns.max()),
                 "mean_pairs": float(k2), "walks": float(walks), "walk_steps_upper_bound": float(upper_steps),
                 "generation_s_per_graph": gen_s}
    t0 = time.perf_counter()
    exact_batch(kernels[:64], [g.B for g in graphs[:64]], points[0]["T"],
                device=devices[0] if backend == "torch" else None)
    est["exact_s_estimate"] = (time.perf_counter() - t0) / min(len(kernels), 64) * len(gids) * len(points)

    if run_mc:
        engine = make_engine(backend, devices[0], cfg)
        bpw = bytes_per_walk(D)
        if hasattr(engine, "_limits"):
            cap, budget = engine._limits()
            per_launch = min(cap, budget / bpw)
        else:
            per_launch = cfg["simulation"]["chunk_walks"] * k2
        est["walks_per_launch"] = float(per_launch)
        est["est_peak_memory_gib"] = float(per_launch * bpw / 2 ** 30)
        pt = points[0]
        m = int(min(pt["N"], 20_000))
        cps = np.asarray([pt["N"]], dtype=np.int64)
        items = []
        for i, (g, kern) in enumerate(zip(graphs[:8], kernels[:8])):
            nbr, cdf = transition_table(kern.P)
            items.append(MCItem(key=(i, 0, 0, 0), nbr=nbr, cdf=cdf, B=g.B, T=pt["T"], m=m, start=0,
                                checkpoints=cps, seed=i))
        list(engine.run_items(items[:1]))                 # warm-up (CUDA context, kernels)
        engine.walk_steps = 0
        t0 = time.perf_counter()
        list(engine.run_items(items))
        secs = time.perf_counter() - t0
        bench_walks = sum(it.walks for it in items)
        bench_upper = sum(it.walks * it.T for it in items)
        steps_rate = engine.walk_steps / secs
        alive_ratio = engine.walk_steps / bench_upper
        # the benchmark uses the first sweep point; T and N of later points scale walk-steps linearly
        est.update({"bench_walks_per_s": bench_walks / secs, "bench_walk_steps_per_s": steps_rate,
                    "alive_fraction": alive_ratio,
                    "mc_s_estimate": upper_steps * alive_ratio / steps_rate / max(len(devices), 1)})
    total = est.get("mc_s_estimate", 0.0) + est["exact_s_estimate"] + gen_s * len(gids)
    est["total_s_estimate"] = total
    lines = [f"graphs {est['graphs']:,} | sweep points {len(points)} | replicates {R} | units {est['units']:,}",
             f"graph size: mean n {est['mean_n']:.1f}, max n {est['max_n']}, mean |B|^2 {k2:.1f}"]
    if run_mc:
        lines += [f"walks {walks:.3e} | walk-steps <= {upper_steps:.3e} "
                  f"(measured live fraction {est['alive_fraction']:.2f})",
                  f"per launch ~{est['walks_per_launch']:.3e} walks, est. peak memory "
                  f"{est['est_peak_memory_gib']:.2f} GiB",
                  f"benchmark: {est['bench_walks_per_s']:.3e} walks/s, "
                  f"{est['bench_walk_steps_per_s']:.3e} walk-steps/s on {devices[0]}"]
    lines.append(f"estimated runtime: {_fmt_seconds(total)} on {len(devices)} device(s) "
                 f"(exact {_fmt_seconds(est['exact_s_estimate'])})")
    for line in lines:
        log.info("dry-run: %s", line)
    return est
