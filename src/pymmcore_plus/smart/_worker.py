"""Load a smart-microscopy script and call its hooks, in-thread or in a child process.

`_ScriptHost` is the only code that runs user code, and both executors use it
unchanged -- that is what makes a script behave the same in thread and process
mode. This module must stay free of Qt and cheap to import: in process mode it
is the first thing a freshly spawned interpreter imports.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pymmcore_plus.smart._api import (
    AnalysisContext,
    Response,
    SystemInfo,
    normalise_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    import useq

    from pymmcore_plus.smart._api import ExecutionMode, FrameInfo

_MODULE_COUNTER = itertools.count()


@dataclass(frozen=True)
class HostConfig:
    """Everything needed to load a script; picklable (sent to a child process)."""

    path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    """The exact text to run (what was inspected and archived); None: read *path*."""
    run_dir: Path | None = None
    max_events_per_response: int = 1000
    system: SystemInfo = field(default_factory=SystemInfo)
    base_sequence: useq.MDASequence | None = None
    analyzer: Any = None
    """An analysis object, used instead of the script at *path* (see `Analyzer`).

    Last, so the positional order of the other fields is unchanged.
    """
    class_name: str | None = None
    """A class in the script to instantiate (no arguments) and take hooks from."""


@dataclass
class HookResult:
    """Outcome of one hook call; picklable, so it can cross process boundaries."""

    call: str  # "setup" | "analyze" | "after_base" | "teardown"
    ok: bool
    frame_id: int | None = None
    response: Response | None = None
    logs: list[tuple[str, str]] = field(default_factory=list)
    records: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


class _ScriptHost:
    """Imports one script and runs its hooks. Never raises from a hook call.

    Compiled under a fresh, unique module name every time, so edits to the
    file take effect on the next run even in thread mode (where the host
    process is long-lived), and two runs never share module-level state.
    """

    def __init__(self, config: HostConfig, execution: ExecutionMode) -> None:
        self._config = config
        self.ctx = AnalysisContext(
            config.params,
            config.run_dir,
            execution,
            system=config.system,
            base_sequence=config.base_sequence,
        )
        self._module: types.ModuleType | None = None
        self._module_name = ""
        self._sys_path_entry: str | None = None
        self._import_error: str | None = None
        self._analyzer = config.analyzer
        try:
            if self._analyzer is None:
                self._import()
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            self._import_error = _format_exception(e)

    @property
    def import_error(self) -> str | None:
        return self._import_error

    def _import(self) -> None:
        if (path := self._config.path) is None:  # pragma: no cover - guarded above
            raise ValueError("No script path and no analysis object.")
        # Sibling helper modules next to the script must be importable.
        script_dir = str(path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            self._sys_path_entry = script_dir
        self._module_name = (
            f"_pymmplus_smart_{os.getpid()}_{next(_MODULE_COUNTER)}_{path.stem}"
        )
        source = self._config.source
        if source is None:
            source = path.read_text(encoding="utf-8")
        # Compiled from source rather than imported through the normal loader:
        # its bytecode cache keys on mtime in whole seconds plus file size, so
        # a same-length edit saved within a second of the last run would run
        # the *old* code. (It would also write __pycache__ into the user's
        # script folder.)
        code = compile(source, str(path), "exec")
        module = types.ModuleType(self._module_name)
        module.__file__ = str(path)
        sys.modules[self._module_name] = module
        exec(code, module.__dict__)
        self._module = module
        if (class_name := self._config.class_name) is not None:
            # The script defines a class rather than functions: one instance
            # per run, so its attributes are this run's state.
            self._analyzer = getattr(module, class_name)()

    def _hook(self, name: str) -> Callable[..., Any] | None:
        """The named hook: a method of the analysis object, or a module function."""
        source = self._module if self._analyzer is None else self._analyzer
        hook = getattr(source, name, None)
        return hook if callable(hook) else None

    def setup(self) -> HookResult:
        if self._import_error is not None:
            return HookResult("setup", ok=False, error=self._import_error)
        if self._hook("analyze") is None:
            return HookResult(
                "setup", ok=False, error="The script defines no callable analyze()."
            )
        return self._call("setup", None, self._hook("setup"), self.ctx)

    def analyze(self, image: np.ndarray, frame: FrameInfo) -> HookResult:
        return self._call(
            "analyze", frame.frame_id, self._hook("analyze"), image, frame, self.ctx
        )

    def after_base(self) -> HookResult:
        return self._call("after_base", None, self._hook("after_base"), self.ctx)

    def teardown(self) -> HookResult:
        if self._module is None and self._analyzer is None:
            return HookResult("teardown", ok=True)
        return self._call("teardown", None, self._hook("teardown"), self.ctx)

    def close(self) -> None:
        """Forget the module and undo the sys.path change (thread mode)."""
        self._analyzer = None
        sys.modules.pop(self._module_name, None)
        if self._sys_path_entry is not None:
            try:
                sys.path.remove(self._sys_path_entry)
            except ValueError:
                pass
            self._sys_path_entry = None
        self._module = None

    def _call(
        self,
        call: str,
        frame_id: int | None,
        hook: Callable[..., Any] | None,
        *args: Any,
    ) -> HookResult:
        start = time.perf_counter()
        response: Response | None = None
        error: str | None = None
        try:
            if hook is not None:
                value = hook(*args)
                if call in ("analyze", "after_base"):
                    response = normalise_response(
                        value, max_events=self._config.max_events_per_response
                    )
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            error = _format_exception(e)
        duration_ms = (time.perf_counter() - start) * 1000
        logs, records = self.ctx._drain()  # noqa: SLF001 (same package)
        return HookResult(
            call,
            ok=error is None,
            frame_id=frame_id,
            response=response,
            logs=logs,
            records=records,
            duration_ms=duration_ms,
            error=error,
        )


def _format_exception(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()


def read_only_view(image: np.ndarray) -> np.ndarray:
    """A view of *image* that a script cannot write through.

    In thread mode the array handed to ``analyze`` is the very one the data
    sink stores; a script modifying it in place would corrupt saved data.
    """
    view = image.view()
    view.flags.writeable = False
    return view


# ------------------------------------------------- process-mode entry points
# Module-level functions are pickled by reference, so these are what a
# ProcessPoolExecutor sends to its child. The host lives in the child for the
# whole run (one worker per run), keeping `ctx.state` meaningful.

_HOST: _ScriptHost | None = None


def _proc_init(config: HostConfig) -> None:
    global _HOST
    # Never raises: an initializer error would only surface as an opaque
    # BrokenProcessPool, so import errors are reported by _proc_setup instead.
    _HOST = _ScriptHost(config, "process")


def _proc_setup() -> HookResult:
    assert _HOST is not None
    return _HOST.setup()


def _proc_analyze(image: np.ndarray, frame: FrameInfo) -> HookResult:
    assert _HOST is not None
    return _HOST.analyze(image, frame)


def _proc_after_base() -> HookResult:
    assert _HOST is not None
    return _HOST.after_base()


def _proc_teardown() -> HookResult:
    assert _HOST is not None
    result = _HOST.teardown()
    _HOST.close()
    return result
