"""Test 6: paper8 FRNE ~ 0.25 % and paper9 ~ 0.19 % at T = 17, N = 1e6, within seed-to-seed spread."""
import pytest

from fptsim.validation import PAPER_FRNE, paper_regression, regression_passes


@pytest.mark.slow
def test_paper_regression_1e6():
    rows = paper_regression(N=1_000_000, replicates=2, device="auto")
    for row in rows:
        assert row["exact_vs_literal_max_abs"] < 1e-14
        assert row["exact_mass_error"] < 1e-12
        # expected RMS FRNE alone reproduces the paper's numbers
        assert row["expected_rms_frne"] == pytest.approx(PAPER_FRNE[row["graph"]], rel=0.2)
        assert regression_passes(row), row
        assert 0.6 * PAPER_FRNE[row["graph"]] < row["frne_min"]
        assert row["frne_max"] < 1.4 * PAPER_FRNE[row["graph"]]
