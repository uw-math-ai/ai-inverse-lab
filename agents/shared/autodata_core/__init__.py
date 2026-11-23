"""autodata_core: stdlib-only Autodata harness for the vertex-centrality inverse problem.

Shared by agents/autodata-inverse (free, heuristic challenger) and agents/autodata-inverse-llm
(model challenger). Port of "Toggable Simulation of Data Code/autodata" without NumPy, PyYAML or
fptsim, because the website runtime image has only Python 3.12's standard library.
"""
__version__ = "0.1.0"
