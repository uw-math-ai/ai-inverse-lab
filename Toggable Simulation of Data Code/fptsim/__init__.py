"""fptsim: config-driven first-passage-time data generation for batches of random weighted graphs."""

__version__ = "0.1.0"


def load_run(path):
    """Load a finished (or partial) run directory; see fptsim.io.load_run."""
    from .io import load_run as _load_run
    return _load_run(path)
