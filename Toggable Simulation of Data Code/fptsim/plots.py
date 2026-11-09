"""Plots: the graph (coloured by mu or q, B highlighted, edge width by weight) and r vs r' panels.

The notebook's legend label r"$X \\setminus B$" crashes matplotlib's mathtext, which has
no \\setminus. Labels here only use mathtext symbols that exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

B_COLOR, X_COLOR, EDGE_COLOR = "white", "#4cc3e6", "0.2"
EXACT_COLOR, EMP_COLOR = "#2a6fdb", "#e8833a"


def positions(graph) -> np.ndarray:
    """Stored coordinates, or a seeded spring layout when the family has none."""
    if graph.coords is not None:
        return np.asarray(graph.coords, dtype=float)
    try:
        import networkx as nx
        G = nx.from_numpy_array((graph.W > 0).astype(float))
        pos = nx.spring_layout(G, seed=0)
        return np.array([pos[i] for i in range(graph.n)])
    except ImportError:
        angle = 2 * np.pi * np.arange(graph.n) / graph.n
        return np.column_stack([np.cos(angle), np.sin(angle)])


def plot_graph(graph, values=None, label=None, ax=None, title=None, cmap="viridis"):
    """Draw the graph; with `values` (length n) vertices are coloured by value with a colorbar."""
    import matplotlib.pyplot as plt
    xy = positions(graph)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    W = graph.W
    wmax = float(W.max()) if W.max() > 0 else 1.0
    for x, y in graph.edges():
        ax.plot(xy[[x, y], 0], xy[[x, y], 1], color=EDGE_COLOR,
                lw=0.5 + 3.5 * W[x, y] / wmax, zorder=1, solid_capstyle="round")
    inB = np.zeros(graph.n, dtype=bool)
    inB[np.asarray(graph.B)] = True
    if values is None:
        ax.scatter(*xy[~inB].T, s=140, c=X_COLOR, edgecolors="black", linewidths=1, zorder=2,
                   label="vertex not in $B$")
        ax.scatter(*xy[inB].T, s=140, c=B_COLOR, edgecolors="black", linewidths=1, zorder=2,
                   label="vertex in $B$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        sc = ax.scatter(*xy.T, s=180, c=np.asarray(values, dtype=float), cmap=cmap,
                        edgecolors="black", linewidths=1, zorder=2)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cb.set_label(label)
        ax.scatter(*xy[inB].T, s=320, facecolors="none", edgecolors="crimson", linewidths=2.2,
                   zorder=3, label="$B$")
        ax.legend(loc="lower left", frameon=False, fontsize=8)
    if graph.n <= 80:
        for i, (x, y) in enumerate(xy):
            ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_distributions(r_exact=None, r_emp=None, B=None, title="", max_side: int = 6):
    """One panel per ordered pair (B[i], B[j]): r' as bars, exact r as a line."""
    import matplotlib.pyplot as plt
    ref = r_exact if r_exact is not None else r_emp
    T, k, _ = ref.shape
    B = list(range(k)) if B is None else [int(b) for b in B]
    side = min(k, max_side)
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(side, side, figsize=(3.0 * side, 2.3 * side),
                             sharex=True, sharey=True, squeeze=False)
    for i in range(side):
        for j in range(side):
            ax = axes[i, j]
            if r_emp is not None:
                ax.bar(t, r_emp[:, i, j], width=0.7, color=EMP_COLOR, label="empirical $r'$")
            if r_exact is not None:
                ax.plot(t, r_exact[:, i, j], "o-", ms=3, lw=1.3, color=EXACT_COLOR, label="exact $r$")
            ax.set_title(f"$x = {B[i]},\\ y = {B[j]}$", fontsize=9)
            ax.grid(axis="y", color="0.9", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if i == side - 1:
                ax.set_xlabel("$t$")
            if j == 0:
                ax.set_ylabel("probability")
    axes[0, -1].legend(loc="upper right", frameon=False, fontsize=7)
    if side < k:
        title = f"{title} (first {side} of {k} observed vertices)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_convergence(metrics, ax=None):
    """Mean FRNE vs N (from convergence checkpoints) with the expected RMS FRNE."""
    import matplotlib.pyplot as plt
    cols = sorted((int(c[6:]) for c in metrics.columns if c.startswith("frne_N")))
    if not cols:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ok = metrics[metrics.status == "ok"]
    ax.loglog(cols, [ok[f"frne_N{c}"].mean() for c in cols], "o-", color=EMP_COLOR, label="mean FRNE")
    ax.loglog(cols, [ok[f"exp_frne_N{c}"].mean() for c in cols], "--", color=EXACT_COLOR,
              label="expected RMS FRNE")
    ax.set_xlabel("$N$ (walks per pair)")
    ax.set_ylabel("FRNE (%)")
    ax.grid(True, which="both", color="0.92")
    ax.legend(frameon=False)
    return ax


def plot_graph_report(run_path, gid: int, s: int = 0, r: int = 0, out_dir=None) -> list[Path]:
    """Save graph_<id>.png (mu and q panels) and graph_<id>_s<s>_r<r>.png (r vs r')."""
    import matplotlib.pyplot as plt
    from .io import load_run
    run = load_run(run_path)
    graph = run.graph(gid)
    out_dir = Path(out_dir) if out_dir else Path(run_path) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_graph(graph, values=graph.mu, label=r"$\mu$", ax=axes[0],
               title=f"graph {gid} ({graph.family}): $\\mu$, edge width $\\propto w$")
    plot_graph(graph, values=graph.q, label="$q$", ax=axes[1], cmap="magma_r",
               title=f"graph {gid}: potential $q$")
    fig.tight_layout()
    path = out_dir / f"graph_{gid}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(path)

    t = run.tensors(gid)
    sweep = t["sweeps"].get(s, {})
    r_exact = sweep.get("r_exact")
    rep = sweep.get("replicates", {}).get(r, {})
    r_emp = None
    if "counts" in rep:
        rows = run.metrics[(run.metrics.graph_id == gid) & (run.metrics.sweep_index == s)
                           & (run.metrics.replicate == r)]
        r_emp = rep["counts"] / float(rows.N.iloc[0])
    if r_exact is not None or r_emp is not None:
        frne_txt = ""
        if r_exact is not None and r_emp is not None:
            from .metrics import frne
            frne_txt = f", FRNE {frne(r_exact, r_emp):.3f}%"
        fig = plot_distributions(r_exact, r_emp, graph.B,
                                 title=f"graph {gid}, sweep {s}, replicate {r}{frne_txt}")
        path = out_dir / f"graph_{gid}_s{s}_r{r}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)
    return saved
