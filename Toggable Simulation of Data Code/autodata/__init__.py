"""autodata: an Autodata-style harness (Meta FAIR, arXiv:2606.25996) on top of fptsim.

A challenger proposes fptsim simulation toggles; a verifier checks the resulting inverse problem is
well posed; a weak and a strong solver try to reconstruct the hidden vertex centrality mu on X \\ B
from noisy first-passage times; an instance is kept only if it separates them. An outer loop
meta-optimizes the challenger's strategy. See AUTODATA.md.
"""

__version__ = "0.1.0"


def load_dataset(run_dir):
    """Accepted items of a run (metadata + arrays); see autodata.loop.load_dataset."""
    from .loop import load_dataset as _load
    return _load(run_dir)
