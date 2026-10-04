"""Smart microscopy: acquisitions that react to their own images.

A *script* -- a Python file defining ``analyze(image, frame, ctx)`` and
optionally ``setup``, ``after_base`` and ``teardown`` hooks -- is called for
frames as they are acquired, and returns what to acquire next: nothing, more
`useq.MDAEvent`s, an `useq.MDASequence`, or `STOP`. The script runs in a
background thread or a separate process, and never touches the core itself.

```python
from pymmcore_plus.smart import SmartRunner

summary = SmartRunner(core).run(useq.MDASequence(channels=["DAPI"]), "script.py")
```

See the "Smart Microscopy" guide for the script contract.

This package must stay free of Qt: scripts import it inside spawned processes.
"""

from ._api import (
    API_VERSION,
    STOP,
    AnalysisContext,
    Analyzer,
    ExecutionMode,
    FrameInfo,
    ParamSpec,
    PixelConfig,
    Response,
    SequencingMode,
    SyncMode,
    SystemInfo,
)
from ._loader import (
    AnalyzeFilter,
    ParamDef,
    ScriptError,
    ScriptSpec,
    inspect_analyzer,
    inspect_script,
)
from ._runner import (
    SmartRunConfig,
    SmartRunError,
    SmartRunner,
    SmartRunStats,
    SmartSignaler,
    dry_run,
)
from ._worker import HookResult

__all__ = [
    "API_VERSION",
    "STOP",
    "AnalysisContext",
    "AnalyzeFilter",
    "Analyzer",
    "ExecutionMode",
    "FrameInfo",
    "HookResult",
    "ParamDef",
    "ParamSpec",
    "PixelConfig",
    "Response",
    "ScriptError",
    "ScriptSpec",
    "SequencingMode",
    "SmartRunConfig",
    "SmartRunError",
    "SmartRunStats",
    "SmartRunner",
    "SmartSignaler",
    "SyncMode",
    "SystemInfo",
    "dry_run",
    "inspect_analyzer",
    "inspect_script",
]
