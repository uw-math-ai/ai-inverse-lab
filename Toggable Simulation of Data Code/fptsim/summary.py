"""Aggregate tables for a run: overall, replicate variability, FRNE by family / n / q, FRNE vs N, coverage."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import io

log = logging.getLogger("fptsim")


def _agg(df, by):
    import pandas as pd
    if df.empty or by not in df:
        return pd.DataFrame()
    cols = {c: ["mean", "median", "std"] for c in ("frne_percent", "expected_rms_frne_percent")
            if c in df}
    for c in ("ci_coverage", "ci_coverage_np5", "tv_mean", "hit_mass_exact", "killed_mass_exact"):
        if c in df:
            cols[c] = ["mean"]
    if not cols:
        return pd.DataFrame()
    out = df.groupby(by, observed=True).agg(cols)
    out.columns = ["_".join(c) for c in out.columns]
    out.insert(0, "units", df.groupby(by, observed=True).size())
    return out


def summarize(path) -> dict:
    import pandas as pd
    path = Path(path)
    run = io.load_run(path)
    m, g = run.metrics, run.graphs
    out = path / "summary"
    out.mkdir(exist_ok=True)
    tables = {}
    if not g.empty:
        io.atomic_write_parquet(g, path / "graphs.parquet")
    if not m.empty:
        io.atomic_write_parquet(m, path / "metrics.parquet")
    ok = m[m.status == "ok"].copy() if "status" in m else m
    statuses = m.status.value_counts().to_dict() if "status" in m else {}

    overall = {"graphs": int(g.graph_id.nunique()) if not g.empty else 0,
               "units": int(len(m)), "status_counts": statuses}
    if "frne_percent" in ok:
        overall.update({
            "frne_mean": float(ok.frne_percent.mean()), "frne_median": float(ok.frne_percent.median()),
            "expected_rms_mean": float(ok.expected_rms_frne_percent.mean()),
            "frne_over_expected_mean": float((ok.frne_percent / ok.expected_rms_frne_percent).mean()),
            "ci_coverage_mean": float(ok.ci_coverage.mean()),
            "ci_coverage_np5_mean": float(ok.ci_coverage_np5.mean()),
            "max_exact_mass_error": float(ok.get("exact_mass_error", pd.Series([np.nan])).max())})
        # replicate variability per (graph, sweep point)
        rep = ok.groupby(["graph_id", "sweep_index"]).agg(
            replicates=("replicate", "size"), frne_mean=("frne_percent", "mean"),
            frne_std=("frne_percent", "std"), expected_rms=("expected_rms_frne_percent", "first"))
        rep["ratio"] = rep.frne_mean / rep.expected_rms
        tables["replicates"] = rep
        overall["replicate_frne_std_mean"] = float(rep.frne_std.mean())

    if not g.empty and not ok.empty:
        extra = [c for c in ("q_mean", "deg_mean", "spectral_gap") if c in g]
        ok = ok.merge(g[["graph_id", *extra]], on="graph_id", how="left")
    if "family" in ok:
        tables["by_family"] = _agg(ok, "family")
    if "n" in ok:
        ok["n_bin"] = pd.cut(ok.n, [0, 10, 20, 40, 80, 160, 320, np.inf])
        tables["by_n"] = _agg(ok, "n_bin")
    if "q_scale" in ok:
        tables["by_q_scale"] = _agg(ok, "q_scale")
    if "q_mean" in ok and ok.q_mean.notna().any():
        ok["q_mean_bin"] = pd.cut(ok.q_mean, [-np.inf, 0, 0.01, 0.05, 0.1, 0.25, 0.5, np.inf])
        tables["by_q_mean"] = _agg(ok, "q_mean_bin")
    if "N" in ok:
        tables["by_N"] = _agg(ok, "N")
    conv = sorted(int(c[6:]) for c in ok.columns if c.startswith("frne_N"))
    if conv:
        tables["frne_vs_N"] = pd.DataFrame({
            "N": conv, "frne_mean": [ok[f"frne_N{c}"].mean() for c in conv],
            "frne_std": [ok[f"frne_N{c}"].std() for c in conv],
            "expected_rms_mean": [ok[f"exp_frne_N{c}"].mean() for c in conv]})

    for name, table in tables.items():
        if len(table):
            table.to_csv(out / f"{name}.csv")
    io.atomic_write_text(out / "overall.json", __import__("json").dumps(overall, indent=2, default=io.json_default))

    pd.set_option("display.width", 160)
    print(f"run {path}: {overall['graphs']} graphs, {overall['units']} units, status {statuses}")
    for key in ("frne_mean", "frne_median", "expected_rms_mean", "frne_over_expected_mean",
                "replicate_frne_std_mean", "ci_coverage_mean", "ci_coverage_np5_mean", "max_exact_mass_error"):
        if key in overall:
            print(f"  {key:26s} {overall[key]:.6g}")
    for name in ("by_family", "by_n", "by_q_scale", "by_q_mean", "by_N", "frne_vs_N"):
        if name in tables and len(tables[name]):
            print(f"\n== {name} ==")
            print(tables[name].to_string(float_format=lambda v: f"{v:.4g}"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plots import plot_convergence
        if conv:
            ax = plot_convergence(ok)
            ax.figure.tight_layout()
            ax.figure.savefig(out / "frne_vs_N.png", dpi=130)
            plt.close(ax.figure)
    except ImportError:
        pass
    print(f"\ntables written to {out}")
    return {"overall": overall, "tables": tables}
