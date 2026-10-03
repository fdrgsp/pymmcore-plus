# pyright: reportArgumentType=false
# (useq models are built from plain values that pydantic coerces)
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest
import useq

from pymmcore_plus.smart._scheduler import (
    AFTER_BASE_ID,
    SmartEventIterator,
    StopReason,
    provenance,
)


@dataclass
class _Status:
    phase: str = "waiting"


@dataclass
class FakeRunner:
    """Stands in for MDARunner: a phase and a controllable event clock."""

    status: _Status = field(default_factory=_Status)
    clock: float = 0.0

    def event_seconds_elapsed(self) -> float:
        return self.clock


def _ev(i: int, t: float | None = None) -> useq.MDAEvent:
    return useq.MDAEvent(index={"t": i}, min_start_time=t)


def _labels(events: list[useq.MDAEvent]) -> list[tuple[str, int]]:
    return [(provenance(e)["origin"], e.index["t"]) for e in events]


def _next_in_thread(it: SmartEventIterator) -> tuple[threading.Thread, list]:
    out: list = []

    def _run() -> None:
        try:
            out.append(next(it))
        except StopIteration:
            out.append(StopIteration)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, out


def test_base_only_runs_to_completion() -> None:
    it = SmartEventIterator([_ev(0), _ev(1)], FakeRunner(), sync="async")
    assert _labels(list(it)) == [("base", 0), ("base", 1)]
    assert it.stop_reason == StopReason.COMPLETED


def test_blocking_holds_events_until_analysis_finishes() -> None:
    it = SmartEventIterator(
        [_ev(0), _ev(1)], FakeRunner(), sync="blocking", poll_s=0.01
    )
    assert next(it).index["t"] == 0
    it.analysis_submitted(0)
    thread, out = _next_in_thread(it)
    time.sleep(0.1)
    assert out == []  # held back
    it.inject([_ev(10)], priority="next", parent_frame_id=0, response_id=0)
    it.analysis_finished(0)
    thread.join(1)
    assert _labels(out) == [("analysis", 10)]
    assert _labels(list(it)) == [("base", 1)]


def test_async_releases_base_events_while_analysis_pending() -> None:
    it = SmartEventIterator([_ev(0), _ev(1)], FakeRunner(), sync="async", poll_s=0.01)
    next(it)
    it.analysis_submitted(0)
    assert next(it).index["t"] == 1  # not held


def test_run_stays_alive_while_analysis_pending() -> None:
    """Purely reactive runs: the last base event's analysis may request more."""
    it = SmartEventIterator([_ev(0)], FakeRunner(), sync="async", poll_s=0.01)
    next(it)
    it.analysis_submitted(0)
    thread, out = _next_in_thread(it)
    time.sleep(0.1)
    assert out == []
    it.inject([_ev(1)], priority="end", parent_frame_id=0, response_id=0)
    it.analysis_finished(0)
    thread.join(1)
    assert _labels(out) == [("analysis", 1)]
    assert list(it) == []
    assert it.stop_reason == StopReason.COMPLETED


def test_priority_next_overtakes_far_future_base_event() -> None:
    runner = FakeRunner()
    it = SmartEventIterator(
        [_ev(0), _ev(1, t=100.0)], runner, sync="async", poll_s=0.01
    )
    next(it)
    thread, out = _next_in_thread(it)
    time.sleep(0.1)
    assert out == []  # base event 1 is held back (not due for 100 s)
    it.inject([_ev(5)], priority="next", parent_frame_id=0, response_id=0)
    thread.join(1)
    assert _labels(out) == [("analysis", 5)]
    runner.clock = 99.9  # now within the lead time
    assert _labels([next(it)]) == [("base", 1)]


def test_priority_end_waits_for_base_events() -> None:
    it = SmartEventIterator([_ev(0), _ev(1), _ev(2)], FakeRunner(), sync="async")
    next(it)
    it.inject([_ev(9)], priority="end", parent_frame_id=0, response_id=0)
    assert _labels(list(it)) == [("base", 1), ("base", 2), ("analysis", 9)]


def test_drop_base_and_stop() -> None:
    it = SmartEventIterator([_ev(i) for i in range(5)], FakeRunner(), sync="async")
    next(it)
    it.inject([_ev(7)], priority="end", parent_frame_id=0, response_id=0)
    it.drop_base()
    assert _labels(list(it)) == [("analysis", 7)]

    it2 = SmartEventIterator([_ev(i) for i in range(5)], FakeRunner(), sync="async")
    next(it2)
    it2.inject([_ev(7)], priority="next", parent_frame_id=0, response_id=0)
    it2.stop(StopReason.SCRIPT)
    assert list(it2) == []
    assert it2.stop_reason == StopReason.SCRIPT
    assert it2.inject([_ev(8)], priority="next", parent_frame_id=0, response_id=1) == 0


def test_runner_finishing_unblocks_waiting_iterator() -> None:
    runner = FakeRunner()
    it = SmartEventIterator([_ev(0)], runner, sync="blocking", poll_s=0.01)
    next(it)
    it.analysis_submitted(0)
    thread, out = _next_in_thread(it)
    time.sleep(0.05)
    runner.status.phase = "finishing"  # e.g. cancelled
    thread.join(1)
    assert out == [StopIteration]
    assert it.stop_reason == StopReason.RUNNER


def test_max_total_events() -> None:
    it = SmartEventIterator(
        [_ev(i) for i in range(10)], FakeRunner(), sync="async", max_total_events=3
    )
    assert len(list(it)) == 3
    assert it.stop_reason == StopReason.MAX_EVENTS


def test_analysis_timeout_stops_blocking_run() -> None:
    timed_out: list[int] = []
    it = SmartEventIterator(
        [_ev(0), _ev(1)],
        FakeRunner(),
        sync="blocking",
        poll_s=0.01,
        analysis_timeout_s=0.05,
        on_analysis_timeout=timed_out.append,
    )
    next(it)
    it.analysis_submitted(0)
    with pytest.raises(StopIteration):
        next(it)
    assert timed_out == [0]
    assert it.stop_reason == StopReason.TIMEOUT


def test_relative_timing_is_rebased_per_segment() -> None:
    """A returned time-lapse starts its own clock without resetting the run's.

    useq marks each time block (here: per position) with reset_event_timer;
    the flag must be consumed, re-anchoring only that response's events.
    """
    runner = FakeRunner(clock=50.0)
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0), (1, 1, 0)],
        time_plan=useq.TIntervalLoops(interval=1, loops=2),
        axis_order="ptc",
    )
    it = SmartEventIterator([], runner, sync="async", lead_time_s=0)
    it.inject(list(seq), priority="next", parent_frame_id=0, response_id=0)
    times = []
    for _ in range(4):
        event = next(it)
        assert not event.reset_event_timer
        times.append(event.min_start_time)
        runner.clock += 1.0  # each event takes a second
    assert times == [50.0, 51.0, 52.0, 53.0]


def test_absolute_timing_is_untouched() -> None:
    it = SmartEventIterator([], FakeRunner(clock=50.0), sync="async")
    it.inject(
        [useq.MDAEvent(min_start_time=2.0, reset_event_timer=True)],
        priority="next",
        parent_frame_id=0,
        response_id=0,
        relative_timing=False,
    )
    event = next(it)
    assert (event.min_start_time, event.reset_event_timer) == (2.0, True)


def test_provenance_tagging_does_not_mutate_input() -> None:
    original = useq.MDAEvent(metadata={"user": 1})
    it = SmartEventIterator([original], FakeRunner(), sync="async")
    out = next(it)
    assert out.metadata["user"] == 1
    assert provenance(out) == {"origin": "base"}
    assert provenance(original) == {}


def test_after_base_is_triggered_once_when_everything_is_done() -> None:
    calls: list[int] = []
    it: SmartEventIterator

    def _on_base_complete() -> bool:
        calls.append(1)
        it.analysis_submitted(AFTER_BASE_ID)
        it.inject([_ev(9)], priority="next", parent_frame_id=-1, response_id=0)
        it.analysis_finished(AFTER_BASE_ID)
        return True

    it = SmartEventIterator(
        [_ev(0), _ev(1)], FakeRunner(), sync="async", on_base_complete=_on_base_complete
    )
    assert _labels(list(it)) == [("base", 0), ("base", 1), ("analysis", 9)]
    assert calls == [1]
    assert it.stop_reason == StopReason.COMPLETED


def test_after_base_waits_for_pending_analyses() -> None:
    calls: list[int] = []
    it = SmartEventIterator(
        [_ev(0)],
        FakeRunner(),
        sync="async",
        poll_s=0.01,
        on_base_complete=lambda: calls.append(1) or False,
    )
    next(it)
    it.analysis_submitted(0)
    thread, out = _next_in_thread(it)
    time.sleep(0.05)
    assert calls == []  # frame 0 still being analyzed
    it.analysis_finished(0)
    thread.join(1)
    assert out == [StopIteration]
    assert calls == [1]


def _grid() -> useq.MDASequence:
    """A sequence whose grid has no field of view (sized when it runs)."""
    return useq.MDASequence(
        stage_positions=(
            useq.Position(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
        )
    )


def test_grid_is_expanded_when_it_reaches_the_front() -> None:
    """Not when injected: the pixel size may change before it runs."""
    sizes: list[float] = []

    px = [1.0]

    def _expand(_seq: useq.MDASequence) -> list[useq.MDAEvent]:
        sizes.append(px[0])
        return [_ev(100), _ev(101)]

    it = SmartEventIterator(
        [_ev(0), _ev(1)], FakeRunner(), sync="async", expand_grid=_expand
    )
    assert next(it).index["t"] == 0
    it.inject([_grid()], priority="next", parent_frame_id=0, response_id=0)
    px[0] = 0.25  # an objective switch happens before the grid runs
    assert _labels([next(it), next(it)]) == [("analysis", 100), ("analysis", 101)]
    assert sizes == [0.25]  # expanded with the *current* pixel size
    assert _labels(list(it)) == [("base", 1)]


def test_grid_keeps_its_place_in_the_queue() -> None:
    it = SmartEventIterator(
        [],
        FakeRunner(),
        sync="async",
        expand_grid=lambda _s: [_ev(10), _ev(11)],
    )
    it.inject(
        [_ev(1), _grid(), _ev(2)], priority="next", parent_frame_id=0, response_id=0
    )
    assert [e.index["t"] for e in it] == [1, 10, 11, 2]


def test_unsizable_grid_stops_the_run_by_default() -> None:
    def _boom(_seq: useq.MDASequence) -> list[useq.MDAEvent]:
        raise ValueError("pixel size not calibrated")

    errors: list[str] = []
    it = SmartEventIterator(
        [_ev(0)],
        FakeRunner(),
        sync="async",
        expand_grid=_boom,
        on_grid_error=lambda msg: errors.append(msg) or False,  # type: ignore[func-returns-value]
    )
    it.inject([_grid(), _ev(5)], priority="next", parent_frame_id=0, response_id=0)
    assert list(it) == []
    assert it.stop_reason == StopReason.ERROR
    assert errors == ["pixel size not calibrated"]


def test_unsizable_grid_can_be_skipped() -> None:
    def _boom(_seq: useq.MDASequence) -> list[useq.MDAEvent]:
        raise ValueError("pixel size not calibrated")

    it = SmartEventIterator(
        [_ev(0)],
        FakeRunner(),
        sync="async",
        expand_grid=_boom,
        on_grid_error=lambda _msg: True,
    )
    it.inject([_grid(), _ev(5)], priority="next", parent_frame_id=0, response_id=0)
    # the grid is dropped; everything else still runs
    assert _labels(list(it)) == [("analysis", 5), ("base", 0)]
    assert it.stop_reason == StopReason.COMPLETED


def test_empty_grid_is_skipped() -> None:
    it = SmartEventIterator([], FakeRunner(), sync="async", expand_grid=lambda _s: [])
    it.inject([_grid()], priority="next", parent_frame_id=0, response_id=0)
    assert list(it) == []
    assert it.stop_reason == StopReason.COMPLETED


def _combine(events: list[useq.MDAEvent]) -> list[useq.MDAEvent]:
    """Stand-in for hardware sequencing: one 'burst' object per group."""
    return [_Burst(events=tuple(events))] if len(events) > 1 else list(events)


class _Burst(useq.MDAEvent):
    """Minimal stand-in for SequencedEvent (which also carries `events`)."""

    events: tuple[useq.MDAEvent, ...] = ()


def test_response_events_are_grouped_into_one_burst() -> None:
    it = SmartEventIterator([_ev(0)], FakeRunner(), sync="blocking", combine=_combine)
    it.inject(
        [_ev(10), _ev(11), _ev(12)], priority="next", parent_frame_id=0, response_id=0
    )
    first = next(it)
    assert isinstance(first, _Burst)
    assert [e.index["t"] for e in first.events] == [10, 11, 12]
    assert it.yielded == 3  # counted as three acquisitions, not one
    assert [e.index["t"] for e in it] == [0]


def test_separate_responses_are_not_grouped() -> None:
    it = SmartEventIterator([], FakeRunner(), sync="async", combine=_combine)
    it.inject([_ev(10)], priority="next", parent_frame_id=0, response_id=0)
    it.inject([_ev(20)], priority="next", parent_frame_id=1, response_id=1)
    assert [e.index["t"] for e in it] == [10, 20]


def test_base_events_are_grouped_in_async_only() -> None:
    base = [_ev(i) for i in range(4)]
    grouped = next(
        iter(SmartEventIterator(base, FakeRunner(), sync="async", combine=_combine))
    )
    assert isinstance(grouped, _Burst)
    assert len(grouped.events) == 4

    single = next(
        iter(SmartEventIterator(base, FakeRunner(), sync="blocking", combine=_combine))
    )
    assert not isinstance(single, _Burst)

    always = next(
        iter(
            SmartEventIterator(
                base,
                FakeRunner(),
                sync="blocking",
                combine=_combine,
                sequencing="always",
            )
        )
    )
    assert isinstance(always, _Burst)


def test_sequencing_off_never_groups() -> None:
    it = SmartEventIterator(
        [_ev(i) for i in range(4)],
        FakeRunner(),
        sync="async",
        combine=_combine,
        sequencing="off",
    )
    assert all(not isinstance(e, _Burst) for e in it)


def test_base_burst_stops_at_an_event_that_is_not_due() -> None:
    """A timed series keeps its timing; only due events share a burst."""
    runner = FakeRunner()
    it = SmartEventIterator(
        [_ev(0), _ev(1), _ev(2, t=100.0), _ev(3, t=100.0)],
        runner,
        sync="async",
        combine=_combine,
        lead_time_s=0.0,
    )
    first = next(it)
    assert isinstance(first, _Burst)
    assert [e.index["t"] for e in first.events] == [0, 1]
    runner.clock = 100.0
    rest = next(it)
    assert isinstance(rest, _Burst)
    assert [e.index["t"] for e in rest.events] == [2, 3]


def test_burst_is_capped() -> None:
    it = SmartEventIterator(
        [_ev(i) for i in range(10)],
        FakeRunner(),
        sync="async",
        combine=_combine,
        max_burst=4,
    )
    sizes = [len(e.events) if isinstance(e, _Burst) else 1 for e in it]
    assert sizes == [4, 4, 2]


def test_requested_event_runs_after_the_current_burst() -> None:
    """Pre-emption latency is bounded by max_burst, not by the whole base."""
    it = SmartEventIterator(
        [_ev(i) for i in range(10)],
        FakeRunner(),
        sync="async",
        combine=_combine,
        max_burst=3,
    )
    first = next(it)
    assert len(first.events) == 3  # type: ignore[attr-defined]
    it.inject([_ev(99)], priority="next", parent_frame_id=0, response_id=0)
    assert next(it).index["t"] == 99  # before the remaining base events


def test_drop_base_discards_prepared_base_events() -> None:
    it = SmartEventIterator(
        [_ev(i) for i in range(6)],
        FakeRunner(),
        sync="async",
        combine=None,  # no grouping: events are prepared one at a time
    )
    next(it)
    it.drop_base()
    assert list(it) == []
