"""Test 9: a run stopped midway and resumed gives the same results as an uninterrupted run."""
import numpy as np
import pytest

from fptsim import io, load_run
from fptsim.config import ConfigError
from fptsim.runner import run

IGNORE = {"elapsed_s"}


def _compare(a_path, b_path):
    a, b = load_run(a_path), load_run(b_path)
    assert len(a.metrics) == len(b.metrics) > 0
    cols = [c for c in a.metrics.columns if c not in IGNORE]
    ma = a.metrics[cols].reset_index(drop=True)
    mb = b.metrics[cols].reset_index(drop=True)
    for c in cols:
        assert ma[c].astype(str).tolist() == mb[c].astype(str).tolist(), c
    assert a.graphs.drop(columns=[]).equals(b.graphs)
    for gid in a.graphs.graph_id:
        ta, tb = a.arrays(int(gid)), b.arrays(int(gid))
        assert ta.keys() == tb.keys()
        for key in ta:
            np.testing.assert_array_equal(ta[key], tb[key], err_msg=f"graph {gid} {key}")


def test_stop_and_resume_matches_uninterrupted(make_config):
    full = make_config("full")
    run(full)
    part = make_config("part")
    run(part, stop_after_units=5)                          # behaves like Ctrl-C after 5 units
    done = io.read_done_keys(part["output_dir"])
    assert 0 < len(done) < 6 * 2
    # simulate a crash mid-write: a stray temp file and a part for a unit that is not done
    tensors = io.parts_dir(part["output_dir"])
    tensors.mkdir(parents=True, exist_ok=True)
    (tensors / ".g5__s0__r1.npz.tmp-1-deadbeef").write_bytes(b"partial")
    with open(io.Path(part["output_dir"]) / "done.jsonl", "a", encoding="utf-8") as f:
        f.write('{"key": [5, 0, 1], "status": "ok", "metr')        # truncated line, no newline
    run(part, resume=True)
    _compare(full["output_dir"], part["output_dir"])


def test_resume_twice_is_a_no_op(make_config):
    cfg = make_config("again")
    run(cfg)
    before = io.read_done_records(cfg["output_dir"])
    run(cfg, resume=True)
    assert io.read_done_records(cfg["output_dir"]) == before


def test_refuses_to_overwrite_or_resume_with_changed_settings(make_config):
    cfg = make_config("guard")
    run(cfg, limit=2)
    with pytest.raises(ConfigError, match="already contains results"):
        run(cfg, resume=False)
    changed = make_config("guard", simulation={"N": 999})
    with pytest.raises(ConfigError, match="simulation.N"):
        run(changed, resume=True)
    more = make_config("guard", num_graphs=8)                # extending a run is allowed
    run(more, resume=True)
    assert {k[0] for k in io.read_done_keys(cfg["output_dir"])} == set(range(8))


def test_shards_partition_and_merge(make_config):
    cfg = make_config("sharded")
    run(cfg, shard=(0, 2))
    run(cfg, shard=(1, 2))
    whole = make_config("whole")
    run(whole)
    _compare(whole["output_dir"], cfg["output_dir"])
