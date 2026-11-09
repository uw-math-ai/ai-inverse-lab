"""End to end: run -> load_run -> summarize -> plot, with sweeps and toggles."""
import numpy as np

from fptsim import io, load_run
from fptsim.cli import main
from fptsim.runner import dry_run, generate_only, run
from fptsim.summary import summarize


def test_full_pipeline_with_sweep(make_config, capsys):
    cfg = make_config("pipe", num_graphs=4, toggles={"save_ci": True, "make_plots": True},
                      plot_sample=1, sweep={"q_scale": [0.0, 2.0], "N": [500, 1500]})
    out = run(cfg)
    data = load_run(out)
    assert len(data.graphs) == 4
    assert len(data.metrics) == 4 * 4 * 2                     # graphs x sweep points x replicates
    ok = data.metrics[data.metrics.status == "ok"]
    assert (ok.frne_percent > 0).all() and (ok.expected_rms_frne_percent > 0).all()
    # q_scale = 0 means nothing is ever killed
    assert (ok[ok.q_scale == 0].killed_mass_exact == 0).all()
    t = data.tensors(0)
    for s in range(4):
        rep = t["sweeps"][s]["replicates"][0]
        N = int(ok[(ok.graph_id == 0) & (ok.sweep_index == s)].N.iloc[0])
        assert np.all(rep["counts"].sum(axis=0) + rep["killed"] + rep["censored"] == N)
        assert "ci_lower" in rep and "prefix_counts" in rep
        assert t["sweeps"][s]["r_exact"].shape == rep["counts"].shape
    assert (out / "tensors" / "shard_0000.npz").exists()
    assert not list(io.parts_dir(out).glob("*.npz"))          # all parts consolidated
    assert (out / "metrics.parquet").exists() and (out / "graphs.parquet").exists()
    assert (out / "env.json").exists() and (out / "run.log").exists()
    assert list((out / "plots").glob("graph_0*.png"))
    summarize(out)
    assert (out / "summary" / "by_family.csv").exists()
    assert main(["plot", "--run", str(out), "--graph-id", "1", "--sweep", "3"]) == 0
    assert (out / "plots" / "graph_1_s3_r0.png").exists()


def test_exact_only_and_mc_only(make_config):
    exact = make_config("exact_only", num_graphs=3, toggles={"run_monte_carlo": False})
    data = load_run(run(exact))
    assert len(data.metrics) == 3 and data.metrics.hit_mass_exact.notna().all()
    assert "frne_percent" not in data.metrics
    mc = make_config("mc_only", num_graphs=3, toggles={"compute_exact": False})
    data = load_run(run(mc))
    assert len(data.metrics) == 6 and data.metrics.hit_mass_emp.notna().all()
    assert "r_exact" not in data.tensors(0)["sweeps"][0]


def test_generate_then_run_and_dry_run(make_config):
    cfg = make_config("gen", num_graphs=3)
    generate_only(cfg)
    assert len(load_run(cfg["output_dir"]).graphs) == 3
    est = dry_run(cfg)
    assert est["graphs"] == 3 and est["walks"] > 0 and est["total_s_estimate"] > 0
    run(cfg)
    data = load_run(cfg["output_dir"])
    np.testing.assert_array_equal(data.tensors(2)["W"], data.graph(2).W)


def test_invalid_explicit_dt_is_recorded(make_config):
    cfg = make_config("dt", num_graphs=2, graphs={"q": {"mode": "constant", "c": 0.9}},
                      sweep={"dt": ["auto", 1.0]})
    data = load_run(run(cfg))
    assert set(data.metrics.status) == {"ok", "invalid_kernel"}
