"""Types that smart-microscopy analysis scripts code against (API version 1).

This module must stay free of Qt and cheap to import: scripts import it inside
a spawned analysis process, where only the stdlib, numpy and useq are wanted.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypedDict,
    get_args,
    runtime_checkable,
)

import useq

if TYPE_CHECKING:
    import numpy as np

    from pymmcore_plus import CMMCorePlus

API_VERSION: Final = 1
"""Version of this API. A script declares the version it was written for."""

ExecutionMode = Literal["thread", "process"]
SyncMode = Literal["blocking", "async"]
Origin = Literal["base", "analysis", "external"]
SequencingMode = Literal["off", "safe", "always"]
LogLevel = Literal["debug", "info", "warning", "error"]
Priority = Literal["next", "end"]
Timing = Literal["relative", "absolute"]

EXECUTION_MODES: Final[tuple[str, ...]] = get_args(ExecutionMode)
SYNC_MODES: Final[tuple[str, ...]] = get_args(SyncMode)
ORIGINS: Final[tuple[str, ...]] = get_args(Origin)
SEQUENCING_MODES: Final[tuple[str, ...]] = get_args(SequencingMode)
LOG_LEVELS: Final[tuple[str, ...]] = get_args(LogLevel)

RecordValue = float | int | str | bool | None
"""Types `AnalysisContext.record` accepts (numpy scalars are converted)."""

PropertyState = Mapping[tuple[str, str], str]
"""Values of device properties, keyed by ``(device, property)``."""


# ------------------------------------------------------------- system snapshot


@dataclass(frozen=True, slots=True)
class PixelConfig:
    """A pixel-size configuration: its size and the properties that select it."""

    name: str
    pixel_size_um: float
    properties: tuple[tuple[str, str, str], ...] = ()
    """``(device, property, value)`` settings that make this config active."""

    def matches(self, state: PropertyState) -> bool:
        """Whether every property of this config has its value in *state*."""
        return bool(self.properties) and all(
            state.get((dev, prop)) == value for dev, prop, value in self.properties
        )


@dataclass(frozen=True, slots=True)
class SystemInfo:
    """Read-only snapshot of the microscope, taken when the run starts.

    Scripts never access the core (they may run in another process); this
    holds what they need to plan acquisitions -- above all, the pixel size
    each objective gives, to size grids and convert pixels to stage
    coordinates.
    """

    image_width: int = 0
    image_height: int = 0
    pixel_size_um: float = 0.0
    """Pixel size in effect at the start of the run (0 if not calibrated)."""
    pixel_config: str | None = None
    """Name of the pixel configuration in effect at the start, if any."""
    pixel_configs: tuple[PixelConfig, ...] = ()
    property_state: PropertyState = field(default_factory=dict)
    """Start values of every property used by a pixel configuration."""

    @classmethod
    def from_core(cls, core: CMMCorePlus) -> SystemInfo:
        """Snapshot the current state of *core*."""
        configs: list[PixelConfig] = []
        state: dict[tuple[str, str], str] = {}
        for name in core.getAvailablePixelSizeConfigs():
            settings = tuple(
                (str(dev), str(prop), str(value))
                for dev, prop, value in core.getPixelSizeConfigData(name)
            )
            configs.append(PixelConfig(name, core.getPixelSizeUmByID(name), settings))
            for dev, prop, _ in settings:
                if (dev, prop) not in state:
                    try:
                        state[(dev, prop)] = str(core.getProperty(dev, prop))
                    except Exception:  # pragma: no cover - device unavailable
                        continue
        current = core.getCurrentPixelSizeConfig() or None
        return cls(
            image_width=core.getImageWidth(),
            image_height=core.getImageHeight(),
            pixel_size_um=core.getPixelSizeUm(),
            pixel_config=current,
            pixel_configs=tuple(configs),
            property_state=state,
        )

    def state_after(
        self,
        properties: Iterable[Sequence[Any]],
        state: PropertyState | None = None,
    ) -> dict[tuple[str, str], str]:
        """*state* (default: the start state) after setting *properties*."""
        new = dict(self.property_state if state is None else state)
        for dev, prop, value in properties:
            new[(str(dev), str(prop))] = str(value)
        return new

    def pixel_config_for(
        self, state: PropertyState | None = None
    ) -> PixelConfig | None:
        """The pixel configuration active in *state* (default: start state)."""
        state = self.property_state if state is None else state
        for config in self.pixel_configs:
            if config.matches(state):
                return config
        return None

    def pixel_size_for(self, state: PropertyState | None = None) -> float:
        """Pixel size in *state*; 0.0 when no calibrated configuration matches."""
        if state is None:
            return self.pixel_size_um
        config = self.pixel_config_for(state)
        return config.pixel_size_um if config is not None else 0.0

    def fov_um(self, pixel_config: str | None = None) -> tuple[float, float]:
        """Field of view (width, height) in µm, at *pixel_config* or the start one.

        Raises ``ValueError`` when that configuration has no pixel size.
        """
        if pixel_config is None:
            px = self.pixel_size_um
        else:
            matches = [c for c in self.pixel_configs if c.name == pixel_config]
            if not matches:
                raise ValueError(f"No pixel configuration named {pixel_config!r}")
            px = matches[0].pixel_size_um
        if px <= 0:
            raise ValueError(
                f"Pixel size of {pixel_config or 'the current configuration'!r} is "
                "not set; calibrate it before planning grids at it."
            )
        return self.image_width * px, self.image_height * px


# ------------------------------------------------------------- frame / context


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """Everything known about the frame being analyzed.

    Attributes
    ----------
    frame_id : int
        0-based acquisition order within this run. Also the frame's index along
        the ``t`` axis of the saved data (for a single camera).
    event : useq.MDAEvent
        The event that produced this frame.
    metadata : Mapping[str, Any]
        The frame's metadata as recorded at acquisition time (``FrameMetaV1``):
        ``pixel_size_um``, ``position`` (x/y/z), ``exposure_ms``,
        ``camera_device``, ``runner_time_ms``, ``property_values``...
    origin : "base" | "analysis" | "external"
        Where the event came from: the base acquisition, a previous analysis,
        or a request made from outside the script (`SmartRunner.request`).
    parent_frame_id : int | None
        For an analysis-origin frame, the frame whose analysis requested it.
    """

    frame_id: int
    event: useq.MDAEvent
    metadata: Mapping[str, Any]
    origin: Origin = "base"
    parent_frame_id: int | None = None


class AnalysisContext:
    """Per-run services handed to ``setup``, ``analyze``, ``after_base``, ``teardown``.

    Attributes
    ----------
    params : Mapping[str, Any]
        Resolved values of the script's ``PARAMETERS`` (read-only).
    state : dict[str, Any]
        Free-form storage that persists across calls within one run.
    run_dir : Path | None
        Folder for the script's own outputs (masks, tables...), if the run has
        one.
    execution : "thread" | "process"
        Where the script is running.
    system : SystemInfo
        Snapshot of the microscope at the start of the run.
    base_sequence : useq.MDASequence | None
        The base acquisition being run.
    """

    __slots__ = (
        "_logs",
        "_records",
        "base_sequence",
        "execution",
        "params",
        "run_dir",
        "state",
        "system",
    )

    def __init__(
        self,
        params: Mapping[str, Any],
        run_dir: str | Path | None,
        execution: ExecutionMode,
        *,
        system: SystemInfo | None = None,
        base_sequence: useq.MDASequence | None = None,
    ) -> None:
        self.params: Mapping[str, Any] = MappingProxyType(dict(params))
        self.state: dict[str, Any] = {}
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.execution: ExecutionMode = execution
        self.system = system or SystemInfo()
        self.base_sequence = base_sequence
        self._logs: list[tuple[str, str]] = []
        self._records: dict[str, RecordValue] = {}

    def log(self, message: object, level: LogLevel = "info") -> None:
        """Report *message* (shown by front ends, kept in the run log)."""
        if level not in LOG_LEVELS:
            raise ValueError(f"level must be one of {LOG_LEVELS}, not {level!r}")
        self._logs.append((level, str(message)))

    def record(self, **values: Any) -> None:
        """Attach named scalar results to the current frame.

        Values must be numbers, strings, booleans or None (numpy scalars are
        converted).
        """
        for key, value in values.items():
            if getattr(value, "shape", None) == () and callable(
                item := getattr(value, "item", None)
            ):
                value = item()  # numpy scalar (0-d) -> Python scalar
            if not isinstance(value, (float, int, str, bool, type(None))):
                raise TypeError(
                    f"record({key}=...) needs a number, string, bool or None, "
                    f"not {type(value).__name__}"
                )
            self._records[key] = value

    def _drain(self) -> tuple[list[tuple[str, str]], dict[str, RecordValue]]:
        """Return and clear what the last call logged and recorded."""
        logs, records = self._logs, self._records
        self._logs, self._records = [], {}
        return logs, records


@runtime_checkable
class Analyzer(Protocol):
    """What an analysis object must provide: `analyze`, and optional hooks.

    An alternative to a script file: keep the hooks on a class, with the
    run's state on ``self`` and its settings as ``__init__`` arguments.

    ```python
    class Tracker:
        def __init__(self, threshold: float = 1000.0) -> None:
            self.threshold = threshold
            self.hits = 0

        def analyze(self, image, frame, ctx):
            if float(image.max()) > self.threshold:
                self.hits += 1
                return useq.MDAEvent(exposure=50)


    core.run_smart(sequence, Tracker(threshold=2000))
    ```

    In process mode the object is sent to the worker, so it must be
    picklable: define the class at module level (not inside a function), and
    keep only picklable state on it.

    Optional class attributes mirror a script's constants, and set the
    defaults a front end offers: ``NAME``, ``DESCRIPTION``, ``EXECUTION``,
    ``SYNC``, ``SEQUENCING``, ``ANALYZE``, ``PARAMETERS``.
    """

    def analyze(self, image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext) -> Any:
        """Decide what to acquire after *image*; see `Response`."""


# ------------------------------------------------------------------- response

EventsLike = Sequence[useq.MDAEvent | useq.MDASequence] | useq.MDASequence


@dataclass(frozen=True, slots=True)
class Response:
    """What to do after a frame was analyzed.

    Attributes
    ----------
    events : sequence of MDAEvent / MDASequence, or an MDASequence
        What to acquire, in order. Sequences are expanded into their events;
        a grid plan without a field of view gets one when it is about to run,
        from the pixel size in effect at that moment (so an objective switched
        earlier -- in this response or any previous one -- is accounted for).
    priority : "next" | "end"
        Acquire them before any remaining base events ("next"), or after them.
    timing : "relative" | "absolute"
        "relative" treats each event's ``min_start_time`` as seconds from when
        the response starts executing (so a returned time-lapse starts its own
        clock); "absolute" uses the values as given, on the run's event clock.
    stop : bool
        Finish the run after the event currently running; nothing else that is
        queued is acquired.
    drop_base : bool
        Discard the remaining base events (continue purely reactively).
    """

    events: EventsLike = ()
    priority: Priority = "next"
    timing: Timing = "relative"
    stop: bool = False
    drop_base: bool = False

    def __post_init__(self) -> None:
        if self.priority not in get_args(Priority):
            raise ValueError(f"priority must be 'next' or 'end', not {self.priority!r}")
        if self.timing not in get_args(Timing):
            raise ValueError(
                f"timing must be 'relative' or 'absolute', not {self.timing!r}"
            )


STOP: Final = Response(stop=True)
"""Return this from ``analyze`` to finish the run."""


class ParamSpec(TypedDict, total=False):
    """Documents the dict form of a ``PARAMETERS`` entry (type-hint aid only)."""

    default: Any
    min: float
    max: float
    step: float
    choices: list[Any]
    label: str
    tooltip: str


Item = useq.MDAEvent | useq.MDASequence
"""A normalised response item: an event, or a grid still to be sized."""


def normalise_response(value: object, *, max_events: int = 1000) -> Response:
    """Turn whatever a hook returned into a `Response` with a flat tuple of items.

    Accepts None, an `MDAEvent`, an `MDASequence`, an iterable mixing both, or
    a `Response`. Sequences are expanded into their events, in order -- except
    those with a grid lacking a field of view, which are kept as they are:
    their tiles can only be placed with the pixel size in effect *when they
    run* (see `needs_fov`), which the runner knows only then.

    Raises ``TypeError`` for anything else and ``ValueError`` when more than
    *max_events* events would be produced (grids still to be sized are counted
    when they are expanded, against the run's total).
    """
    if value is None:
        return Response()
    if isinstance(value, Response):
        response = value
    elif isinstance(value, (useq.MDAEvent, useq.MDASequence)):
        response = Response(events=(value,))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        response = Response(events=value)  # type: ignore[arg-type]
    else:
        raise TypeError(
            "Hooks must return None, an MDAEvent, an MDASequence, an iterable of "
            f"those, or a Response; got {type(value).__name__}"
        )
    items = response.events
    if isinstance(items, (useq.MDAEvent, useq.MDASequence)):
        items = (items,)
    if not isinstance(items, Iterable):
        raise TypeError(f"Response.events must be iterable, not {type(items)}")

    expanded = tuple(islice(_expand(items), max_events + 1))
    n_events = sum(isinstance(item, useq.MDAEvent) for item in expanded)
    if len(expanded) > max_events or n_events > max_events:
        raise ValueError(f"more than {max_events} events requested in one response")
    return Response(
        events=expanded,
        priority=response.priority,
        timing=response.timing,
        stop=response.stop,
        drop_base=response.drop_base,
    )


def _expand(items: Iterable[object]) -> Iterator[Item]:
    for item in items:
        if isinstance(item, useq.MDAEvent):
            yield item
        elif isinstance(item, useq.MDASequence):
            if needs_fov(item):
                yield item  # sized and expanded when it runs
            else:
                yield from item
        else:
            raise TypeError(
                f"expected MDAEvent or MDASequence items, got "
                f"{type(item).__name__}: {item!r}"
            )


def needs_fov(seq: useq.MDASequence) -> bool:
    """Whether *seq* has a grid (its own, or a position's) without a field of view.

    useq places grid tiles using ``fov_width``/``fov_height``; without them the
    tiles end up 1 µm apart. The engine fills these in for the base sequence
    only, so the smart runner sizes such grids itself, when they are about to
    run (see `with_fov`).
    """
    sequences = [seq, *(p.sequence for p in _positions(seq) if p.sequence)]
    return any(_grid_needs_fov(s.grid_plan) for s in sequences)


def with_fov(
    seq: useq.MDASequence, fov_width: float, fov_height: float
) -> useq.MDASequence:
    """A copy of *seq* whose grids without a field of view get this one (µm)."""
    fov = {"fov_width": fov_width, "fov_height": fov_height}

    def _fill(s: useq.MDASequence) -> useq.MDASequence:
        update: dict[str, Any] = {}
        if _grid_needs_fov(s.grid_plan):
            update["grid_plan"] = s.grid_plan.model_copy(  # type: ignore[union-attr]
                update={k: v for k, v in fov.items() if getattr(s.grid_plan, k) is None}
            )
        if any(p.sequence for p in _positions(s)):
            update["stage_positions"] = tuple(
                p.model_copy(update={"sequence": _fill(p.sequence)})
                if p.sequence
                else p
                for p in _positions(s)
            )
        return s.model_copy(update=update) if update else s

    return _fill(seq)


def _positions(seq: useq.MDASequence) -> tuple[useq.Position, ...]:
    # A WellPlatePlan has no per-position sub-sequences to look into.
    positions = seq.stage_positions
    return positions if isinstance(positions, tuple) else ()


def _grid_needs_fov(grid: object) -> bool:
    if grid is None or not hasattr(grid, "fov_width"):
        return False
    # Absolute-coordinate plans (e.g. GridFromEdges) also tile by field of view.
    return (
        getattr(grid, "fov_width", None) is None
        or getattr(grid, "fov_height", None) is None
    )
