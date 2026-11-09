"""mlharness: an autonomous, multi-agent, compute-efficient ML research harness (V1).

Layers
------
1. Research specification   -> ``mlharness.spec``
2. Shared research artifacts -> ``mlharness.artifacts``
3. Agent orchestration       -> ``mlharness.scheduler`` + ``mlharness.agents``
4. Research pipeline         -> ``mlharness.scheduler`` (stages) + ``mlharness.executor`` / ``mlharness.runner``
"""

__version__ = "1.0.0"

from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
