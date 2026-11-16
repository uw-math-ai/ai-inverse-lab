"""mlharness: an autonomous, research-first, multi-agent ML research harness (V2).

Layers
------
1. Research specification   -> ``mlharness.spec``
2. Shared research artifacts -> ``mlharness.artifacts`` (canonical provenance ledger)
3. Agent orchestration       -> ``mlharness.scheduler`` (V1) / ``mlharness.scheduler_v2`` (V2) + agents
4. Research pipeline         -> executor / runner (isolated, budgeted, cached execution)
V2 additions
------------
research phase  -> ``mlharness.research`` (4 researchers, shared critic, loop, portfolio, gate)
literature      -> ``mlharness.tools`` (TheoremSearch adapter + applicability screen)
memory          -> ``mlharness.memory`` (persistent cross-run research memory)
methods         -> ``mlharness.methods_v2`` + ``mlharness.inversion`` + ``mlharness.problem``
safety          -> ``mlharness.safety`` (static checks + runtime audit-hook sandbox)
remote GPUs     -> ``mlharness.remote`` (SSH execution on the AMATH servers)
"""

__version__ = "2.0.0"

from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
