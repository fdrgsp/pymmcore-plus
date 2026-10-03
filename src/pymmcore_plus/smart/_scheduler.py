"""The event iterator a smart run hands to ``core.run_mda``.

Passing an *iterator* makes the runner pull events one at a time, on its own
thread, with no look-ahead (see ``MDARunner._run``). `SmartEventIterator` uses
that to merge the base acquisition with events requested by analysis:

- In **blocking** mode no event is released while an analysis is pending, so
  every acquisition sees the latest decision.
- In **async** mode base events keep flowing; analysis results are inserted
  as they arrive.

The runner cannot be interrupted once it holds an event (it sleeps until the
event's ``min_start_time``), so base events are held back here until they are
nearly due. That lets an urgent analysis-requested event overtake a base event
scheduled far in the future.

The runner thread spends its idle time blocked inside ``__next__``, where it
cannot see its own cancel flag -- so every wait here is bounded and re-checks
the runner's phase.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import useq

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from pymmcore_plus.smart._api import Origin, Priority, SyncMode

SMART_METADATA_KEY: Final = "pymmcore_plus_smart"
"""Key under ``MDAEvent.metadata`` where each event's provenance is recorded."""

AFTER_BASE_ID: Final = -1
"""Pending-analysis id used for the ``after_base`` call."""


class _RunnerStatus(Protocol):
    @property
    def phase(self) -> Any: ...


class RunnerLike(Protocol):
    """The two things the iterator needs from ``MDARunner``."""

    @property
    def status(self) -> _RunnerStatus: ...
    def event_seconds_elapsed(self) -> float: ...


class StopReason:
    """Why the iterator stopped handing out events."""

    COMPLETED: Final = "completed"
    SCRIPT: Final = "stopped_by_script"
    ERROR: Final = "error"
    TIMEOUT: Final = "analysis_timeout"
    MAX_EVENTS: Final = "max_events_reached"
    USER: Final = "stopped_by_user"
    RUNNER: Final = "runner_finishing"  # cancelled from outside


@dataclass(frozen=True)
class _Grid:
    """A requested sequence whose grid is sized only when it is about to run."""

    sequence: useq.MDASequence
    provenance: dict[str, Any]


QueueItem = useq.MDAEvent | _Grid


def provenance(event: useq.MDAEvent) -> dict[str, Any]:
    """The smart-run provenance recorded on *event* (empty if none)."""
    value = event.metadata.get(SMART_METADATA_KEY)
    return value if isinstance(value, dict) else {}


class SmartEventIterator:
    """Thread-safe merge of base and analysis-requested events."""

    def __init__(
        self,
        base: Iterable[useq.MDAEvent],
        runner: RunnerLike,
        *,
        sync: SyncMode,
        lead_time_s: float = 0.25,
        max_total_events: int = 10_000,
        poll_s: float = 0.05,
        analysis_timeout_s: float | None = None,
        on_analysis_timeout: Callable[[int], None] | None = None,
        on_base_complete: Callable[[], bool] | None = None,
        expand_grid: Callable[[useq.MDASequence], list[useq.MDAEvent]] | None = None,
        on_grid_error: Callable[[str], bool] | None = None,
    ) -> None:
        self._base: Iterator[useq.MDAEvent] = iter(base)
        self._runner = runner
        self._sync = sync
        self._lead_time_s = lead_time_s
        self._max_total_events = max_total_events
        self._poll_s = poll_s
        self._analysis_timeout_s = analysis_timeout_s
        self._on_analysis_timeout = on_analysis_timeout
        # Called once, when the base events are done and nothing is queued or
        # pending; returns whether it submitted work (the after_base hook).
        self._on_base_complete = on_base_complete
        self._base_complete_called = False
        # Sizes a requested grid with the pixel size in effect right now (the
        # runner thread calls __next__ only after the previous event ran), and
        # expands it. Raises ValueError when that pixel size is unknown;
        # on_grid_error then decides: True skips the grid, False stops the run.
        self._expand_grid = expand_grid
        self._on_grid_error = on_grid_error

        self._cond = threading.Condition()
        self._base_head: useq.MDAEvent | None = None
        self._base_exhausted = False
        self._injected_next: deque[QueueItem] = deque()
        self._injected_end: deque[QueueItem] = deque()
        # frame_id -> time.monotonic() at submission
        self._pending: dict[int, float] = {}
        self._yielded = 0
        self._stop_reason: str | None = None
        self._timeout_reported = False
        # response_id -> event-clock time its current timing segment started
        self._anchors: dict[int, float] = {}

    # ----------------------------------------------------------- public API

    @property
    def stop_reason(self) -> str | None:
        """Why iteration ended (a `StopReason` value), or None while running."""
        return self._stop_reason

    @property
    def yielded(self) -> int:
        return self._yielded

    @property
    def queued(self) -> int:
        """Events waiting to be handed out (injected; base excluded)."""
        with self._cond:
            return len(self._injected_next) + len(self._injected_end)

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._pending)

    def analysis_submitted(self, frame_id: int) -> None:
        with self._cond:
            self._pending[frame_id] = time.monotonic()

    def analysis_finished(self, frame_id: int) -> None:
        with self._cond:
            self._pending.pop(frame_id, None)
            self._cond.notify_all()

    def inject(
        self,
        events: Sequence[useq.MDAEvent | useq.MDASequence],
        *,
        priority: Priority,
        parent_frame_id: int,
        response_id: int,
        relative_timing: bool = True,
    ) -> int:
        """Queue analysis-requested items; return how many were accepted.

        Items are events, or sequences whose grids still need a field of view:
        those are sized and expanded when they reach the front of the queue
        (see `expand_grid`). With *relative_timing*, each event's
        ``min_start_time`` counts from the moment its response starts executing
        (see `_rebase`), so a returned time-lapse starts its own clock.
        Otherwise times are taken as given, on the run's event clock. Returns 0
        (dropping them) once stopped.
        """
        info = {
            "origin": "analysis",
            "parent_frame_id": parent_frame_id,
            "response_id": response_id,
            "relative": relative_timing,
        }
        tagged: list[QueueItem] = [
            _Grid(e, info) if isinstance(e, useq.MDASequence) else _tag_with(e, info)
            for e in events
        ]
        with self._cond:
            if self._stop_reason is not None:
                return 0
            if priority == "next":
                # Preserve the response's own order at the front of the queue.
                self._injected_next.extend(tagged)
            else:
                self._injected_end.extend(tagged)
            self._cond.notify_all()
        return len(tagged)

    def drop_base(self) -> None:
        """Discard all remaining base events."""
        with self._cond:
            self._base_head = None
            self._base_exhausted = True
            self._cond.notify_all()

    def stop(self, reason: str) -> None:
        """End iteration at the next ``__next__`` (the running event completes)."""
        with self._cond:
            if self._stop_reason is None:
                self._stop_reason = reason
            self._injected_next.clear()
            self._injected_end.clear()
            self._cond.notify_all()

    # ------------------------------------------------------------- iterator

    def __iter__(self) -> SmartEventIterator:
        return self

    def __next__(self) -> useq.MDAEvent:
        with self._cond:
            while True:
                if self._stop_reason is not None:
                    raise StopIteration
                if str(self._runner.status.phase) == "finishing":
                    self._stop_reason = StopReason.RUNNER
                    raise StopIteration
                if self._yielded >= self._max_total_events:
                    self._stop_reason = StopReason.MAX_EVENTS
                    raise StopIteration

                if self._sync == "blocking" and self._pending:
                    self._check_timeout()
                    self._cond.wait(self._poll_s)
                    continue

                if self._injected_next:
                    if (event := self._take(self._injected_next)) is not None:
                        return self._hand_out(event)
                    continue

                if self._base_head is None and not self._base_exhausted:
                    self._base_head = next(self._base, None)
                    if self._base_head is None:
                        self._base_exhausted = True
                if (head := self._base_head) is not None:
                    due = head.min_start_time
                    remaining = (
                        0.0
                        if due is None
                        else due - self._runner.event_seconds_elapsed()
                    )
                    if remaining <= self._lead_time_s:
                        self._base_head = None
                        return self._hand_out(_tag(head, "base"))
                    self._cond.wait(min(remaining - self._lead_time_s, self._poll_s))
                    continue

                if self._injected_end:
                    if (event := self._take(self._injected_end)) is not None:
                        return self._hand_out(event)
                    continue

                if self._pending:
                    # Nothing to do now, but a pending analysis may still
                    # request more -- the run is not over yet.
                    self._check_timeout()
                    self._cond.wait(self._poll_s)
                    continue

                if (
                    self._on_base_complete is not None
                    and not self._base_complete_called
                ):
                    # Everything requested so far is done: give the script its
                    # one chance to plan a second phase (e.g. survey -> target).
                    self._base_complete_called = True
                    if self._on_base_complete():
                        continue

                self._stop_reason = StopReason.COMPLETED
                raise StopIteration

    def _take(self, queue: deque[QueueItem]) -> useq.MDAEvent | None:
        """Pop the next event of *queue*, expanding a grid first if needed.

        Returns None when a grid could not be expanded (skipped or stopping):
        the caller re-evaluates the queues.
        """
        item = queue.popleft()
        if not isinstance(item, _Grid):
            return item
        try:
            if self._expand_grid is None:
                raise ValueError("No grid expander: cannot size this grid.")
            events = self._expand_grid(item.sequence)
        except ValueError as e:
            skip = self._on_grid_error is not None and self._on_grid_error(str(e))
            if not skip:
                self._stop_reason = StopReason.ERROR
                self._injected_next.clear()
                self._injected_end.clear()
            return None
        tagged = [_tag_with(e, item.provenance) for e in events]
        if not tagged:  # an empty grid: nothing to acquire
            return None
        # The rest go in the grid's place, keeping the response's order.
        queue.extendleft(reversed(tagged[1:]))
        return tagged[0]

    def _hand_out(self, event: useq.MDAEvent) -> useq.MDAEvent:
        self._yielded += 1
        info = provenance(event)
        if info.get("origin") == "analysis" and info.get("relative"):
            return self._rebase(event, info["response_id"])
        return event

    def _rebase(self, event: useq.MDAEvent, response_id: int) -> useq.MDAEvent:
        """Shift a relative-timed event onto the run's event clock.

        Times in a returned sequence count from 0, and useq marks the start of
        each time block (e.g. per position) with ``reset_event_timer``. Letting
        the runner act on that flag would restart the clock the *base* events'
        times are measured against. Instead the flag is consumed here: it (or
        the response's first event) starts a new segment anchored at the
        current event-clock time, and the event's ``min_start_time`` is offset
        by that anchor -- same timing, base clock untouched.
        """
        if event.reset_event_timer or response_id not in self._anchors:
            self._anchors[response_id] = self._runner.event_seconds_elapsed()
        start = event.min_start_time
        return event.model_copy(
            update={
                "reset_event_timer": False,
                "min_start_time": None
                if start is None
                else start + self._anchors[response_id],
            }
        )

    def _check_timeout(self) -> None:
        """Report (once) an analysis that has been pending for too long."""
        limit = self._analysis_timeout_s
        if limit is None or self._timeout_reported or not self._pending:
            return
        frame_id, submitted = min(self._pending.items(), key=lambda kv: kv[1])
        if time.monotonic() - submitted < limit:
            return
        self._timeout_reported = True
        # A single worker runs analyses in order, so nothing behind a hung
        # call can complete either: the run stops rather than skipping it.
        self._stop_reason = StopReason.TIMEOUT
        if self._on_analysis_timeout is not None:
            self._on_analysis_timeout(frame_id)


def _tag(event: useq.MDAEvent, origin: Origin, **extra: Any) -> useq.MDAEvent:
    """Return a copy of *event* recording where it came from."""
    return _tag_with(event, {"origin": origin, **extra})


def _tag_with(event: useq.MDAEvent, info: dict[str, Any]) -> useq.MDAEvent:
    metadata = dict(event.metadata)
    metadata[SMART_METADATA_KEY] = dict(info)
    return event.model_copy(update={"metadata": metadata})
