"""Run one smart acquisition: base events, analysis, and the feedback between them.

Threads involved, and what each one does here:

- **Caller's thread**: `prepare` (blocks until the analysis worker is ready),
  `start`, `cancel`, `request_stop`.
- **Runner thread**: pulls events from `SmartEventIterator`, and calls
  `_on_frame_ready` for each frame. With Qt MDA signals the handler is
  connected with a direct connection -- a normal one would queue it to the Qt
  main thread, tying analysis latency to that thread's event loop (and
  stalling entirely when there is none). It must stay fast.
- **Executor callback thread**: `_on_result`, which turns a script's answer
  into queued events.
- **Finalizer thread**: tears the executor down after the run, so a slow or
  hung ``teardown`` never blocks the runner or the caller.

All signals are emitted from whichever thread produced them; front ends that
need a particular thread (e.g. a GUI) must re-dispatch.
"""

from __future__ import annotations

import itertools
import sys
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from psygnal import Signal, SignalGroup

from pymmcore_plus.smart._api import FrameInfo, SystemInfo
from pymmcore_plus.smart._executors import (
    AnalysisExecutor,
    ExecutorStartError,
    create_executor,
)
from pymmcore_plus.smart._loader import AnalyzeFilter, ScriptSpec, inspect_script
from pymmcore_plus.smart._log import (
    SmartRunLog,
    data_path_for_output,
    event_to_json,
    run_dir_for_output,
)
from pymmcore_plus.smart._scheduler import (
    AFTER_BASE_ID,
    SmartEventIterator,
    StopReason,
    provenance,
)
from pymmcore_plus.smart._worker import HostConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    import numpy as np
    import useq
    from typing_extensions import Self

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.mda import SingleOutput
    from pymmcore_plus.metadata import FrameMetaV1
    from pymmcore_plus.smart._api import ExecutionMode, Response, SyncMode
    from pymmcore_plus.smart._worker import HookResult

OnError = Literal["stop", "skip"]
RunDir = Path | str | Literal["auto"] | None

TEARDOWN_TIMEOUT_S: Final = 10.0


class SmartRunError(RuntimeError):
    """A smart run could not be prepared or started."""


@dataclass(frozen=True)
class SmartRunConfig:
    """Everything that determines how a smart run behaves.

    Build one with `from_script`, which reads the script's own defaults.
    """

    spec: ScriptSpec
    params: dict[str, Any]
    execution: ExecutionMode
    sync: SyncMode
    filter: AnalyzeFilter
    on_error: OnError = "stop"
    analysis_timeout_s: float | None = None
    max_total_events: int = 10_000
    max_events_per_response: int = 1_000
    setup_timeout_s: float = 60.0

    @classmethod
    def from_script(
        cls,
        script: str | Path | ScriptSpec,
        *,
        params: dict[str, Any] | None = None,
        execution: ExecutionMode | None = None,
        sync: SyncMode | None = None,
        filter: AnalyzeFilter | None = None,
        **options: Any,
    ) -> SmartRunConfig:
        """A config for *script*: its declared defaults, updated by the arguments.

        Unknown or invalid *params* are ignored (see `ScriptSpec.resolve_params`).
        """
        spec = script if isinstance(script, ScriptSpec) else inspect_script(script)
        return cls(
            spec=spec,
            params=spec.resolve_params(params),
            execution=execution or spec.execution,
            sync=sync or spec.sync,
            filter=filter or spec.filter,
            **options,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "script": {
                "path": str(self.spec.path),
                "name": self.spec.name,
                "sha256": self.spec.sha256,
                "api_version": self.spec.api_version,
            },
            "params": self.params,
            "execution": self.execution,
            "sync": self.sync,
            "filter": self.filter.to_dict(),
            "on_error": self.on_error,
            "analysis_timeout_s": self.analysis_timeout_s,
            "max_total_events": self.max_total_events,
            "max_events_per_response": self.max_events_per_response,
        }


@dataclass
class SmartRunStats:
    """Running totals of a run."""

    frames: int = 0
    analyses_queued: int = 0
    analyses_done: int = 0
    errors: int = 0
    injected: int = 0
    dropped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class SmartSignaler(SignalGroup):
    """Signals emitted by a `SmartRunner`, from the thread that produced them."""

    runStarted = Signal(object)
    """dict: ``run_dir`` (or None) and the run's settings."""
    frameAcquired = Signal(dict)
    """The ``frames.jsonl`` record of each acquired frame."""
    analysisQueued = Signal(int)
    """frame_id sent to ``analyze``."""
    analysisFinished = Signal(dict)
    """The ``analysis.jsonl`` record of each completed hook call."""
    logMessage = Signal(str, str)
    """(level, message) logged by the script, or by the runner itself."""
    analysisError = Signal(str, bool)
    """(message, fatal). Fatal: the worker died; the run is stopping."""
    runFinished = Signal(dict)
    """Summary: status, run_dir and counts."""


def response_summary(response: Response | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return {
        # normalise_response always leaves a tuple of events
        "n_events": len(response.events)
        if isinstance(response.events, tuple)
        else None,
        "priority": response.priority,
        "timing": response.timing,
        "stop": response.stop,
        "drop_base": response.drop_base,
    }


class SmartRunner:
    """Runs smart acquisitions on *mmcore*, one at a time.

    Examples
    --------
    >>> runner = SmartRunner(core)
    >>> summary = runner.run(useq.MDASequence(channels=["DAPI"]), "script.py")

    or, step by step (e.g. to prepare a slow process worker off a GUI thread):

    >>> config = SmartRunConfig.from_script("script.py", execution="process")
    >>> runner.prepare(sequence, config, output="data.ome.zarr", run_dir="auto")
    >>> runner.start()
    >>> summary = runner.wait()
    """

    def __init__(self, mmcore: CMMCorePlus | None = None) -> None:
        if mmcore is None:
            from pymmcore_plus import CMMCorePlus

            mmcore = CMMCorePlus.instance()
        self._mmc = mmcore
        self.events = SmartSignaler()
        self._lock = threading.Lock()
        self._config: SmartRunConfig | None = None
        self._executor: AnalysisExecutor | None = None
        self._setup_result: HookResult | None = None
        self._base: useq.MDASequence | None = None
        self._output: SingleOutput | None = None
        self._run_info: dict[str, Any] = {}
        self._packages: tuple[str, ...] = ()
        self._iterator: SmartEventIterator | None = None
        self._log: SmartRunLog | None = None
        self._run_dir: Path | None = None
        self._frame_ids = itertools.count()
        self._response_ids = itertools.count()
        self._stats = SmartRunStats()
        self._acquiring = False
        self._finalizing = False
        self._user_cancelled = False
        self._connected = False
        self._done = threading.Event()
        self._done.set()
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------ properties

    @property
    def config(self) -> SmartRunConfig | None:
        """Settings of the prepared or running run."""
        return self._config

    @property
    def run_dir(self) -> Path | None:
        """Folder receiving the run's records (None if not recording)."""
        return self._run_dir

    @property
    def iterator(self) -> SmartEventIterator | None:
        return self._iterator

    @property
    def summary(self) -> dict[str, Any] | None:
        """Summary of the last finished run."""
        return self._summary

    def is_active(self) -> bool:
        """Whether a run is prepared, acquiring, or still finalizing."""
        return self._executor is not None or self._acquiring or self._finalizing

    def stats(self) -> SmartRunStats:
        with self._lock:
            return replace(self._stats, extra=dict(self._stats.extra))

    # ------------------------------------------------------------- lifecycle

    def run(
        self,
        base: useq.MDASequence,
        script: str | Path | ScriptSpec | SmartRunConfig,
        *,
        output: SingleOutput | None = None,
        run_dir: RunDir = None,
        block: bool = True,
        timeout: float | None = None,
        **options: Any,
    ) -> dict[str, Any] | None:
        """Prepare and start a run of *script* over *base*.

        *options* are passed to `SmartRunConfig.from_script` (``params``,
        ``execution``, ``sync``...). With *block*, waits for the run to finish
        and returns its summary; otherwise returns None (see `wait`).
        """
        config = (
            script
            if isinstance(script, SmartRunConfig)
            else SmartRunConfig.from_script(script, **options)
        )
        self.prepare(base, config, output=output, run_dir=run_dir)
        self.start()
        return self.wait(timeout) if block else None

    def prepare(
        self,
        base: useq.MDASequence,
        config: SmartRunConfig,
        *,
        output: SingleOutput | None = None,
        run_dir: RunDir = None,
        packages: tuple[str, ...] = (),
    ) -> HookResult:
        """Start the analysis worker and run the script's ``setup``.

        Blocks until the worker is ready (seconds, in process mode). Raises
        `SmartRunError` when *base* has no events, when the worker does not
        start, or when the script fails to load or set up (the worker is
        stopped again in that case).

        *run_dir*: where to write the run's records (``run.json``,
        ``frames.jsonl``, ``analysis.jsonl``, ``script.py``). None writes
        nothing; "auto" uses a folder next to the data (``<name>_smart``), or
        a new temporary folder when *output* does not write to disk.
        *packages*: extra distributions whose versions ``run.json`` records.
        """
        if self.is_active():
            raise SmartRunError("A smart run is already in progress.")
        if next(iter(base), None) is None:
            # useq yields no events at all for a sequence with no axes; even a
            # purely reactive run needs one event to acquire the first frame.
            raise SmartRunError(
                "The base acquisition contains no events. Add at least one channel "
                "(or position) so there is a first frame to analyze."
            )
        resolved_dir = self._resolve_run_dir(run_dir, output)
        system = SystemInfo.from_core(self._mmc)
        host = HostConfig(
            path=config.spec.path,
            params=dict(config.params),
            source=config.spec.source,
            run_dir=resolved_dir,
            max_events_per_response=config.max_events_per_response,
            system=system,
            base_sequence=base,
        )
        executor = create_executor(config.execution)
        try:
            result = executor.start(host, timeout=config.setup_timeout_s)
        except ExecutorStartError as e:
            executor.stop(timeout=1)
            raise SmartRunError(str(e)) from e
        if not result.ok:
            executor.stop(timeout=TEARDOWN_TIMEOUT_S)
            raise SmartRunError(
                f"The script failed to load or set up:\n\n{result.error}"
            )
        self._config = config
        self._executor = executor
        self._setup_result = result
        self._base = base
        self._output = output
        self._run_dir = resolved_dir
        self._packages = packages
        self._run_info = {
            **config.to_json(),
            "base_sequence": base.model_dump(mode="json"),
            "data_path": _str_or_none(data_path_for_output(output)),
            "system": _system_json(system),
        }
        self._log = SmartRunLog(resolved_dir) if resolved_dir is not None else None
        return result

    def start(self) -> threading.Thread:
        """Start the prepared run (non-blocking); return the acquisition thread."""
        config, executor, base = self._config, self._executor, self._base
        if config is None or executor is None or base is None:
            raise SmartRunError("Call prepare() before start().")
        if self._acquiring or self._finalizing:
            raise SmartRunError("A smart run is already in progress.")

        self._frame_ids = itertools.count()
        self._response_ids = itertools.count()
        self._stats = SmartRunStats()
        self._user_cancelled = False
        self._summary = None
        self._done.clear()
        self._iterator = SmartEventIterator(
            base,
            self._mmc.mda,
            sync=config.sync,
            max_total_events=config.max_total_events,
            analysis_timeout_s=config.analysis_timeout_s,
            on_analysis_timeout=self._on_analysis_timeout,
            on_base_complete=self._on_base_complete
            if config.spec.has_after_base
            else None,
        )
        if (log := self._log) is not None:
            log.open(self._run_info, config.spec.source, packages=self._packages)
        if (setup := self._setup_result) is not None:
            self._record_result(setup, injected=0, dropped=False)

        self._acquiring = True
        self._connect()
        try:
            thread = self._mmc.run_mda(self._iterator, output=self._output)
        except Exception:
            self._acquiring = False
            self._disconnect()
            self._finalize(status="error")
            raise
        self.events.runStarted.emit({"run_dir": self._run_dir, **self._run_info})
        return thread

    def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until the current run has fully finished; return its summary."""
        self._done.wait(timeout)
        return self._summary

    def request_stop(self) -> None:
        """Finish after the event currently running; nothing more is queued."""
        if (iterator := self._iterator) is not None and self._acquiring:
            iterator.stop(StopReason.USER)
            self.events.logMessage.emit("info", "Stopping after the current event.")

    def cancel(self) -> None:
        """Cancel the run now (frames already acquired are kept)."""
        if not self._acquiring:
            return
        self._user_cancelled = True
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.USER)
        self._mmc.mda.cancel()

    def abandon(self) -> None:
        """Release a prepared run that will not be started."""
        if self._executor is not None and not self._acquiring and not self._finalizing:
            self._executor.stop(timeout=TEARDOWN_TIMEOUT_S)
            self._executor = None
            self._config = None
            self._log = None
            self._setup_result = None
            self._base = None

    def shutdown(self) -> None:
        """Stop everything immediately (e.g. the application is closing)."""
        if self._acquiring:
            self.cancel()
        self._disconnect()
        if (executor := self._executor) is not None:
            executor.stop(timeout=2)
            self._executor = None
        if (log := self._log) is not None:
            log.finish("aborted")

    # ---------------------------------------------------------- connections

    def _connect(self) -> None:
        events = self._mmc.mda.events
        for signal, slot in self._slots():
            _connect_direct(events, getattr(events, signal), slot)
        self._connected = True

    def _disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        events = self._mmc.mda.events
        for signal, slot in self._slots():
            try:
                getattr(events, signal).disconnect(slot)
            except (TypeError, RuntimeError, ValueError):
                pass

    def _slots(self) -> tuple[tuple[str, Callable[..., Any]], ...]:
        return (
            ("frameReady", self._on_frame_ready),
            ("sequenceFinished", self._on_sequence_finished),
        )

    # --------------------------------------------------------- runner thread

    def _on_frame_ready(
        self, img: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
    ) -> None:
        config, executor, iterator = self._config, self._executor, self._iterator
        if config is None or executor is None or iterator is None:
            return
        frame_id = next(self._frame_ids)
        info = provenance(event)
        origin = info.get("origin", "base")
        parent = info.get("parent_frame_id")
        record = {
            "frame_id": frame_id,
            "t_index": frame_id,
            "origin": origin,
            "parent_frame_id": parent,
            "response_id": info.get("response_id"),
            "event": event_to_json(event),
            "runner_time_ms": meta.get("runner_time_ms"),
            "camera": meta.get("camera_device"),
            "exposure_ms": meta.get("exposure_ms"),
            "pixel_size_um": meta.get("pixel_size_um"),
            "position": meta.get("position"),
        }
        if (log := self._log) is not None:
            log.write_frame(record)
        with self._lock:
            self._stats.frames += 1
            first_uncalibrated = not meta.get(
                "pixel_size_um"
            ) and not self._stats.extra.get("warned_pixel_size")
            if first_uncalibrated:
                self._stats.extra["warned_pixel_size"] = True
        if first_uncalibrated:
            self.events.logMessage.emit(
                "warning",
                f"Frame {frame_id} has no calibrated pixel size: positions computed "
                "from it in pixels cannot be converted to the stage.",
            )
        self.events.frameAcquired.emit(record)

        if executor.broken or not config.filter.accepts(frame_id, event, origin):
            return
        frame = FrameInfo(
            frame_id=frame_id,
            event=event.model_copy(update={"sequence": None}),
            metadata=dict(meta),
            origin=origin,
            parent_frame_id=parent,
        )
        iterator.analysis_submitted(frame_id)
        try:
            future = executor.submit(img, frame)
        except Exception as e:  # executor shut down / broken pool
            iterator.analysis_finished(frame_id)
            self._fatal(f"Could not send frame {frame_id} to analysis: {e}")
            return
        with self._lock:
            self._stats.analyses_queued += 1
        future.add_done_callback(partial(self._on_result, frame_id))
        self.events.analysisQueued.emit(frame_id)

    def _on_base_complete(self) -> bool:
        """Runner thread (inside the iterator): submit ``after_base``."""
        executor, iterator = self._executor, self._iterator
        if executor is None or iterator is None or executor.broken:
            return False
        iterator.analysis_submitted(AFTER_BASE_ID)
        try:
            future = executor.submit_after_base()
        except Exception as e:
            iterator.analysis_finished(AFTER_BASE_ID)
            self._fatal(f"Could not call after_base(): {e}")
            return False
        future.add_done_callback(partial(self._on_result, AFTER_BASE_ID))
        return True

    # ----------------------------------------------- executor callback thread

    def _on_result(self, frame_id: int, future: Future[HookResult]) -> None:
        iterator, config = self._iterator, self._config
        try:
            if future.cancelled():
                return
            if (exc := future.exception()) is not None:
                what = (
                    "after_base()" if frame_id == AFTER_BASE_ID else f"frame {frame_id}"
                )
                self._fatal(f"The analysis worker died while running {what}: {exc!r}")
                return
            result = future.result()
            injected, dropped = 0, False
            if result.ok and result.response is not None and iterator is not None:
                injected, dropped = self._apply(result, iterator)
            elif not result.ok:
                self._on_script_error(result, config)
            self._record_result(result, injected=injected, dropped=dropped)
        finally:
            if iterator is not None:
                iterator.analysis_finished(frame_id)

    def _apply(
        self, result: HookResult, iterator: SmartEventIterator
    ) -> tuple[int, bool]:
        """Act on a successful response; return (events injected, dropped?)."""
        response = result.response
        assert response is not None
        if response.drop_base:
            iterator.drop_base()
        injected = 0
        dropped = False
        if response.events:
            injected = iterator.inject(
                list(response.events),  # type: ignore[arg-type]
                priority=response.priority,
                parent_frame_id=result.frame_id if result.frame_id is not None else -1,
                response_id=next(self._response_ids),
                relative_timing=response.timing == "relative",
            )
            dropped = injected == 0
        if response.stop:
            iterator.stop(StopReason.SCRIPT)
            who = (
                "after_base()"
                if result.frame_id is None
                else f"Frame {result.frame_id}"
            )
            self.events.logMessage.emit("info", f"{who}: the script stopped the run.")
        with self._lock:
            self._stats.injected += injected
            self._stats.dropped += int(dropped)
        return injected, dropped

    def _on_script_error(
        self, result: HookResult, config: SmartRunConfig | None
    ) -> None:
        with self._lock:
            self._stats.errors += 1
        where = (
            "after_base()"
            if result.call == "after_base"
            else (f"analyze() on frame {result.frame_id}")
        )
        message = f"{where} raised:\n{result.error}"
        if config is not None and config.on_error == "skip":
            self.events.logMessage.emit("error", message)
            return
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.ERROR)
        self.events.analysisError.emit(message, False)

    def _fatal(self, message: str) -> None:
        with self._lock:
            self._stats.errors += 1
        if (iterator := self._iterator) is not None:
            iterator.stop(StopReason.ERROR)
        self.events.analysisError.emit(message, True)

    def _on_analysis_timeout(self, frame_id: int) -> None:
        timeout = self._config.analysis_timeout_s if self._config else None
        what = "after_base()" if frame_id == AFTER_BASE_ID else f"frame {frame_id}"
        self.events.analysisError.emit(
            f"Analysis of {what} took longer than {timeout:g} s; stopping the run.",
            False,
        )

    def _record_result(
        self, result: HookResult, *, injected: int, dropped: bool
    ) -> None:
        record = {
            "frame_id": result.frame_id,
            "call": result.call,
            "ok": result.ok,
            "duration_ms": round(result.duration_ms, 3),
            "records": result.records,
            "logs": result.logs,
            "response": response_summary(result.response),
            "injected": injected,
            "dropped": dropped,
            "error": result.error,
        }
        if (log := self._log) is not None:
            log.write_analysis(record)
        if result.call == "analyze":
            with self._lock:
                self._stats.analyses_done += 1
        for level, message in result.logs:
            self.events.logMessage.emit(level, message)
        self.events.analysisFinished.emit(record)

    # ------------------------------------------------------- end of the run

    def _on_sequence_finished(self, *_: object) -> None:
        """Runner thread: acquisition over; finish off the analysis side."""
        if not self._acquiring:
            return
        self._acquiring = False
        self._disconnect()
        self._finalize(status=self._final_status())

    def _final_status(self) -> str:
        if self._user_cancelled:
            return "cancelled"
        reason = self._iterator.stop_reason if self._iterator else None
        if reason == StopReason.RUNNER:
            # FINISHING without a stop of ours: cancelled from elsewhere.
            return "cancelled"
        finish_reason = str(self._mmc.mda.status.finish_reason or "")
        if finish_reason == "canceled" and reason in (None, StopReason.COMPLETED):
            # Cancelled from elsewhere while an event was being acquired: the
            # runner stopped at the event boundary without asking us again.
            return "cancelled"
        if finish_reason == "errored":
            return "error"
        return reason or StopReason.COMPLETED

    def _finalize(self, *, status: str) -> None:
        executor, log, run_dir = self._executor, self._log, self._run_dir
        self._finalizing = True

        def _run() -> None:
            summary: dict[str, Any] = {
                "status": status,
                "run_dir": _str_or_none(run_dir),
            }
            try:
                teardown = executor.stop(TEARDOWN_TIMEOUT_S) if executor else None
                if teardown is not None:
                    self._record_result(teardown, injected=0, dropped=False)
                stats = self.stats()
                counts = {
                    "frames": stats.frames,
                    "analyses": stats.analyses_done,
                    "errors": stats.errors,
                    "injected_events": stats.injected,
                    "dropped_responses": stats.dropped,
                }
                if log is not None:
                    log.finish(status, counts=counts)
                summary.update(counts)
            finally:
                self._executor = None
                self._setup_result = None
                self._finalizing = False
            self._summary = summary
            self._done.set()
            self.events.runFinished.emit(summary)

        threading.Thread(target=_run, name="smart-finalize", daemon=True).start()

    def _resolve_run_dir(self, run_dir: RunDir, output: object) -> Path | None:
        if run_dir is None:
            return None
        if run_dir == "auto":
            return run_dir_for_output(data_path_for_output(output))
        path = Path(run_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


def _connect_direct(events: object, signal: Any, slot: Callable[..., Any]) -> None:
    """Connect so *slot* runs in the emitting (runner) thread, for any backend.

    psygnal already calls slots synchronously in the emitting thread. Qt
    queues a plain callable to the thread it was connected from, unless asked
    for a direct connection. Qt signals imply qtpy.QtCore is already imported,
    so Qt is never loaded here just to find out it is not in use.
    """
    qt_core = sys.modules.get("qtpy.QtCore")
    if qt_core is not None and isinstance(events, qt_core.QObject):
        signal.connect(slot, qt_core.Qt.ConnectionType.DirectConnection)
    else:
        signal.connect(slot)


def _str_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _system_json(system: SystemInfo) -> dict[str, Any]:
    return {
        "image_width": system.image_width,
        "image_height": system.image_height,
        "pixel_size_um": system.pixel_size_um,
        "pixel_config": system.pixel_config,
        "pixel_configs": [
            {
                "name": c.name,
                "pixel_size_um": c.pixel_size_um,
                "properties": c.properties,
            }
            for c in system.pixel_configs
        ],
    }


def dry_run(
    script: str | Path | ScriptSpec | SmartRunConfig,
    image: np.ndarray,
    frame: FrameInfo | None = None,
    *,
    core: CMMCorePlus | None = None,
    system: SystemInfo | None = None,
    base_sequence: useq.MDASequence | None = None,
    **options: Any,
) -> HookResult:
    """Call a script's ``analyze`` once on *image*; nothing is acquired.

    The script is loaded in a throwaway worker of the configured execution
    mode, and its ``setup``, ``analyze`` and ``teardown`` run once -- the
    quickest way to see what it would request, while writing it.

    Parameters
    ----------
    script : str | Path | ScriptSpec | SmartRunConfig
        The script (*options* go to `SmartRunConfig.from_script`).
    image : np.ndarray
        The image to analyze (e.g. ``core.getImage()`` after a snap).
    frame : FrameInfo, optional
        What the script sees about the image. By default, built from *core*'s
        current state (channel, exposure, stage position, pixel size).
    core : CMMCorePlus, optional
        Source of the default *frame* and *system*.
    system, base_sequence : optional
        What ``ctx.system`` / ``ctx.base_sequence`` hold (default: snapshot of
        *core*, None).

    Returns
    -------
    HookResult
        The ``analyze`` call's result (``response``, ``records``, ``logs``,
        ``error``), or the failed ``setup`` result if the script did not load.
    """
    import tempfile

    config = (
        script
        if isinstance(script, SmartRunConfig)
        else SmartRunConfig.from_script(script, **options)
    )
    if system is None:
        system = SystemInfo.from_core(core) if core is not None else SystemInfo()
    if frame is None:
        frame = _frame_from_core(core) if core is not None else _blank_frame()
    with tempfile.TemporaryDirectory(prefix="pymmcore-smart-dry-run-") as tmp:
        host = HostConfig(
            path=config.spec.path,
            params=dict(config.params),
            source=config.spec.source,
            run_dir=Path(tmp),
            max_events_per_response=config.max_events_per_response,
            system=system,
            base_sequence=base_sequence,
        )
        executor = create_executor(config.execution)
        try:
            setup = executor.start(host, timeout=config.setup_timeout_s)
        except ExecutorStartError as e:
            executor.stop(timeout=1)
            raise SmartRunError(str(e)) from e
        try:
            if not setup.ok:
                return setup
            return executor.submit(image, frame).result(timeout=300)
        finally:
            executor.stop(timeout=TEARDOWN_TIMEOUT_S)


def _blank_frame() -> FrameInfo:
    import useq

    return FrameInfo(frame_id=0, event=useq.MDAEvent(), metadata={})


def _frame_from_core(core: CMMCorePlus) -> FrameInfo:
    """What a frame acquired in *core*'s current state would report."""
    import useq

    channel: dict[str, str] | None = None
    try:
        if (group := core.getChannelGroup()) and (
            preset := core.getCurrentConfig(group)
        ):
            channel = {"config": preset, "group": group}
    except Exception:  # pragma: no cover - no channel group / device error
        channel = None
    x = y = z = None
    try:
        x, y = core.getXPosition(), core.getYPosition()
    except Exception:  # pragma: no cover - no XY stage
        pass
    try:
        z = core.getPosition()
    except Exception:  # pragma: no cover - no focus device
        pass
    # Validated from plain data: an event's channel is its own type, not the
    # `useq.Channel` a sequence takes.
    event = useq.MDAEvent.model_validate(
        {
            "channel": channel,
            "exposure": core.getExposure(),
            "x_pos": x,
            "y_pos": y,
            "z_pos": z,
        }
    )
    metadata = {
        "pixel_size_um": core.getPixelSizeUm(),
        "exposure_ms": core.getExposure(),
        "camera_device": core.getCameraDevice(),
        "position": {"x": x, "y": y, "z": z},
    }
    return FrameInfo(frame_id=0, event=event, metadata=metadata)
