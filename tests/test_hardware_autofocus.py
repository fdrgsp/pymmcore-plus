"""Hardware autofocus: the Z search for lock, the `autofocusFinished` signal, and
keeping continuous focus locked (`keep_locked`)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import pytest
from useq import HardwareAutofocus, MDAEvent, MDASequence

from pymmcore_plus.autofocus import AutofocusResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pymmcore_plus


def _locking_full_focus(
    core: pymmcore_plus.CMMCorePlus,
    lock_at: float,
    lock_range: float = 2.0,
    attempts: list[float] | None = None,
) -> Callable[[], None]:
    """Return a `fullFocus` that only locks within `lock_range` µm of `lock_at`.

    This is how a hardware autofocus behaves: outside of its search range it cannot
    find the surface at all, no matter how many times it is retried.
    """

    def _full_focus() -> None:
        z = core.getZPosition()
        if attempts is not None:
            attempts.append(z)
        if abs(z - lock_at) > lock_range:
            raise RuntimeError(f"Out of focus search range at z={z}")
        core.setZPosition(lock_at)

    return _full_focus


def _af_event(**kwargs: Any) -> MDAEvent:
    kwargs.setdefault("max_retries", 1)  # keep the attempt counts readable
    return MDAEvent(action=HardwareAutofocus(**kwargs))


def _exec(core: pymmcore_plus.CMMCorePlus, event: MDAEvent) -> None:
    # exec_event is a generator function: autofocus only runs once it is consumed
    assert list(core.mda.engine.exec_event(event)) == []


# ------------------------------------- search ------------------------------------


def test_no_search_by_default(core: pymmcore_plus.CMMCorePlus) -> None:
    """Without an explicit range, autofocus is only attempted where it started."""
    core.setZPosition(100.0)
    attempts: list[float] = []
    with patch.object(
        core, "fullFocus", _locking_full_focus(core, 85.0, attempts=attempts)
    ):
        _exec(core, _af_event())

    assert attempts == [100.0]
    assert core.getZPosition() == 100.0
    assert core.mda.engine._af_succeeded is False


def test_search_finds_lock_below(core: pymmcore_plus.CMMCorePlus) -> None:
    core.setZPosition(100.0)
    attempts: list[float] = []
    with patch.object(
        core, "fullFocus", _locking_full_focus(core, 85.0, attempts=attempts)
    ):
        _exec(core, _af_event(search_below_um=20.0, search_step_um=5.0))

    assert attempts == [100.0, 95.0, 90.0, 85.0]
    assert core.getZPosition() == 85.0
    assert core.mda.engine._af_succeeded is True
    # the search move is part of the correction: the lock is where focus really is
    assert core.mda.engine._z_correction == {None: -15.0}


def test_search_tries_below_before_above(core: pymmcore_plus.CMMCorePlus) -> None:
    """MMStudio's HardwareFocusExtender searches down first, then up."""
    core.setZPosition(100.0)
    attempts: list[float] = []
    with patch.object(
        core, "fullFocus", _locking_full_focus(core, 110.0, attempts=attempts)
    ):
        _exec(
            core,
            _af_event(search_below_um=10.0, search_above_um=10.0, search_step_um=5.0),
        )

    assert attempts == [100.0, 95.0, 90.0, 105.0, 110.0]
    assert core.getZPosition() == 110.0


def test_search_stays_within_requested_range(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """A lock just outside the range is not found, and Z is put back."""
    core.setZPosition(100.0)
    attempts: list[float] = []
    with patch.object(
        core, "fullFocus", _locking_full_focus(core, 70.0, attempts=attempts)
    ):
        _exec(core, _af_event(search_below_um=10.0, search_step_um=5.0))

    assert attempts == [100.0, 95.0, 90.0]
    assert core.getZPosition() == 100.0
    assert core.mda.engine._af_succeeded is False
    assert core.mda.engine._z_correction == {}


def test_search_honors_retries(core: pymmcore_plus.CMMCorePlus) -> None:
    """Each search position gets `max_retries` attempts."""
    core.setZPosition(100.0)
    attempts: list[float] = []
    with patch.object(
        core, "fullFocus", _locking_full_focus(core, 90.0, attempts=attempts)
    ):
        _exec(
            core,
            _af_event(max_retries=2, search_below_um=10.0, search_step_um=10.0),
        )

    # twice at 100 (both fail), then 90 locks on its first attempt
    assert attempts == [100.0, 100.0, 90.0]
    assert core.getZPosition() == 90.0


def test_search_stops_when_canceled(core: pymmcore_plus.CMMCorePlus) -> None:
    core.setZPosition(100.0)
    attempts: list[float] = []
    core.mda._cancel_requested = True
    try:
        with patch.object(
            core, "fullFocus", _locking_full_focus(core, 85.0, attempts=attempts)
        ):
            _exec(core, _af_event(search_below_um=20.0, search_step_um=5.0))
    finally:
        core.mda._cancel_requested = False

    # the first attempt ran, but the search was abandoned rather than stepping Z
    assert attempts == [100.0]
    assert core.getZPosition() == 100.0


def test_search_in_full_mda(core: pymmcore_plus.CMMCorePlus) -> None:
    mda = MDASequence(
        stage_positions=[{"z": 100.0}],
        autofocus_plan={
            "axes": ("p",),
            "autofocus_motor_offset": 25,
            "max_retries": 1,
            "search_below_um": 20.0,
            "search_step_um": 5.0,
        },
    )
    with patch.object(core, "fullFocus", _locking_full_focus(core, 85.0)):
        core.mda.run(mda)

    assert core.mda.engine._z_correction == {0: -15.0}
    assert core.getZPosition() == 85.0


# -------------------------------- the signal --------------------------------


def test_autofocus_finished_on_success(core: pymmcore_plus.CMMCorePlus) -> None:
    results: list[tuple[MDAEvent, AutofocusResult]] = []
    core.mda.events.autofocusFinished.connect(
        lambda event, result: results.append((event, result))
    )

    core.setZPosition(100.0)
    event = _af_event(search_below_um=20.0, search_step_um=5.0)
    with patch.object(core, "fullFocus", _locking_full_focus(core, 85.0)):
        _exec(core, event)

    assert len(results) == 1
    emitted_event, result = results[0]
    assert emitted_event is event
    assert isinstance(result, AutofocusResult)
    assert result.kind == "hardware"
    assert result.method == core.getAutoFocusDevice()
    assert result.focus_device == core.getFocusDevice()
    assert result.succeeded is True
    assert result.message == ""
    assert result.z_before == 100.0
    assert result.z_after == 85.0
    assert result.delta_z == -15.0
    # hardware autofocus exposes no focus score
    assert result.scores == ()
    assert result.n_images == 0
    assert core.mda.engine.last_autofocus_result is result


def test_autofocus_finished_on_failure(core: pymmcore_plus.CMMCorePlus) -> None:
    """A failure must be reported, not swallowed."""
    results: list[AutofocusResult] = []
    core.mda.events.autofocusFinished.connect(
        lambda event, result: results.append(result)
    )

    core.setZPosition(100.0)
    with patch.object(core, "fullFocus", _locking_full_focus(core, 500.0)):
        _exec(core, _af_event())

    assert len(results) == 1
    assert results[0].succeeded is False
    assert "Out of focus search range" in results[0].message
    # the drive was left where autofocus started
    assert results[0].z_before == 100.0
    assert results[0].z_after == 100.0


def test_no_autofocus_finished_without_device(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Nothing is reported if there is no autofocus device to run."""
    results: list[AutofocusResult] = []
    core.mda.events.autofocusFinished.connect(
        lambda event, result: results.append(result)
    )
    with patch.object(core, "getAutoFocusDevice", return_value=""):
        _exec(core, _af_event())
    assert results == []


def test_failed_autofocus_clears_pending_reengagement(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    core.mda.engine._af_was_engaged = True
    with patch.object(core, "fullFocus", side_effect=[None, RuntimeError("no lock")]):
        _exec(core, _af_event())
        assert core.mda.engine._af_needs_reengage
        _exec(core, _af_event())
    assert not core.mda.engine._af_succeeded
    assert not core.mda.engine._af_needs_reengage


# -------------------------------- keep locked --------------------------------

Call = tuple[str, tuple, bool]  # (method, args, continuous focus on at the time)


@contextmanager
def _scope(
    core: pymmcore_plus.CMMCorePlus,
    *,
    on: bool = False,
    locks: Literal["yes", "after_find", "no", "cancel"] = "yes",
    finds: bool = True,
    drift: float = 0.0,
) -> Iterator[list[Call]]:
    """Fake continuous focus, and record in order what the engine asks of it.

    `locks` says when it locks once switched on: always, only after `fullFocus` at
    the current XY position, never, or never and cancelling the acquisition.  Each
    image taken while it is on moves Z by `drift`, as a lock following drift does.
    """
    calls: list[Call] = []
    state = {"on": on, "found": False}

    def record(name: str, *args: Any) -> None:
        calls.append((name, args, state["on"]))

    def enable(on: bool) -> None:
        record("enableContinuousFocus", on)
        state["on"] = on

    def is_locked() -> bool:
        if locks == "cancel" and state["on"]:
            core.mda.cancel()
        return state["on"] and (
            locks == "yes" or (locks == "after_find" and state["found"])
        )

    def full_focus() -> None:
        record("fullFocus")
        if not finds:
            raise RuntimeError("no focus")
        state["found"] = True

    def recorded(name: str, before: Callable[[], None] = lambda: None) -> Any:
        original = getattr(core, name)

        def _method(*args: Any) -> Any:
            record(name, *args)
            before()
            return original(*args)

        return _method

    def new_xy() -> None:
        state["found"] = False

    def drifted() -> None:
        if state["on"] and drift:
            core.setPosition(core.getFocusDevice(), core.getZPosition() + drift)

    core.mda.engine.lock_timeout = 0.05
    with (
        patch.object(core, "enableContinuousFocus", enable),
        patch.object(core, "isContinuousFocusEnabled", lambda: state["on"]),
        patch.object(core, "isContinuousFocusLocked", is_locked),
        patch.object(core, "fullFocus", full_focus),
        patch.object(core, "setXYPosition", recorded("setXYPosition", new_xy)),
        patch.object(core, "setZPosition", recorded("setZPosition")),
        patch.object(core, "setAutoFocusOffset", recorded("setAutoFocusOffset")),
        patch.object(core, "snapImage", recorded("snapImage", drifted)),
    ):
        yield calls


def _called(calls: list[Call], name: str) -> list[tuple[tuple, bool]]:
    return [(args, on) for n, args, on in calls if n == name]


def _locked_seq(axes: tuple[str, ...] = ("p",), z: float | None = 5.0) -> MDASequence:
    """Two positions with their own offsets (1 and 2), two timepoints, `tp` order."""
    plan = {"axes": axes, "keep_locked": True, "max_retries": 1}
    return MDASequence(
        axis_order="tpc",
        time_plan={"interval": 0, "loops": 2},
        stage_positions=[
            {
                "x": x,
                "y": 0,
                "z": z,
                "sequence": {"autofocus_plan": {**plan, "autofocus_motor_offset": x}},
            }
            for x in (1.0, 2.0)
        ],
    )


def _results(core: pymmcore_plus.CMMCorePlus) -> list[tuple[bool, bool, str]]:
    """Collect (find_focus, succeeded, message) of every autofocus result."""
    results: list[tuple[bool, bool, str]] = []
    core.mda.events.autofocusFinished.connect(
        lambda e, r: results.append((e.action.find_focus, r.succeeded, r.message))
    )
    return results


def test_keep_locked_finds_on_its_axes(core: pymmcore_plus.CMMCorePlus) -> None:
    results = _results(core)
    with _scope(core) as calls:
        core.mda.run(_locked_seq())

    assert [r[:2] for r in results] == [(True, True)] * 4
    assert len(_called(calls, "fullFocus")) == 4
    # the lock is off for every move to a new position, and on for every image
    xy = _called(calls, "setXYPosition")
    moves = [
        on for (args, on), (last, _) in zip(xy[1:], xy, strict=False) if args != last
    ]
    assert moves == [False] * 3
    assert all(on for _, on in _called(calls, "snapImage"))
    # Z is only ever moved with the lock off, and the lock is left as it was found
    assert not any(on for _, on in _called(calls, "setZPosition"))
    assert not core.isContinuousFocusEnabled()


@pytest.mark.parametrize("on", [False, True])
def test_keep_locked_holds_without_finding(
    core: pymmcore_plus.CMMCorePlus, on: bool
) -> None:
    results = _results(core)
    with _scope(core, on=on) as calls:
        core.mda.run(_locked_seq(axes=()))
        assert core.isContinuousFocusEnabled() is on

    assert results == [(False, True, "")] * 4
    assert not _called(calls, "fullFocus")
    offsets = [args for args, _ in _called(calls, "setAutoFocusOffset")]
    assert offsets == [(1.0,), (2.0,), (1.0,), (2.0,)]
    assert all(on for _, on in _called(calls, "snapImage"))
    # started off: Z is set once, before locking; started on: never
    assert len(_called(calls, "setZPosition")) == (0 if on else 1)
    switches = [args for args, _ in _called(calls, "enableContinuousFocus")]
    assert switches == ([] if on else [(True,), (False,)])


def test_keep_locked_hold_finds_when_it_cannot_lock(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    results = _results(core)
    with _scope(core, locks="after_find") as calls:
        core.mda.run(_locked_seq(axes=()))

    assert results == [(False, True, "")] * 4
    assert len(_called(calls, "fullFocus")) == 4


def test_keep_locked_gives_up_and_lets_z_move(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    results = _results(core)
    with _scope(core, locks="no", finds=False) as calls:
        core.mda.run(_locked_seq(axes=()))

    assert [r[:2] for r in results] == [(False, False)] * 4
    # one find per hold, and the lock is switched off again after each
    assert len(_called(calls, "fullFocus")) == 4
    assert not any(on for _, on in _called(calls, "snapImage"))
    # Z moves resume for the images (and before each find)
    assert len(_called(calls, "setZPosition")) > 4


def test_keep_locked_finds_when_a_tile_move_loses_the_lock(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Grid tiles get no hold event: the wait after the move finds focus if needed."""
    results = _results(core)
    seq = MDASequence(
        grid_plan={"rows": 1, "columns": 3},
        autofocus_plan={"axes": (), "keep_locked": True, "max_retries": 1},
    )
    with _scope(core, locks="after_find") as calls:
        core.mda.run(seq)

    assert results == [(False, True, ""), (True, True, ""), (True, True, "")]
    assert all(on for _, on in _called(calls, "snapImage"))


def test_keep_locked_find_that_cannot_relock(core: pymmcore_plus.CMMCorePlus) -> None:
    results = _results(core)
    with _scope(core, locks="no") as calls:
        core.mda.run(_locked_seq())

    assert len(_called(calls, "fullFocus")) == 4  # never a second find
    assert all(not ok and "did not lock" in msg for _, ok, msg in results)
    assert not core.isContinuousFocusEnabled()


def test_keep_locked_cancel_never_finds(core: pymmcore_plus.CMMCorePlus) -> None:
    results = _results(core)
    with _scope(core, locks="cancel") as calls:
        core.mda.engine.lock_timeout = 5.0  # cancelling must end the wait, not this
        core.mda.run(_locked_seq(axes=()))

    assert not _called(calls, "fullFocus")
    assert results == []


@pytest.mark.parametrize(
    ("z", "expected"), [(5.0, [5.0, 5.0, 6.0, 6.0]), (None, [1.0, 2.0])]
)
def test_keep_locked_find_starts_from_the_drifted_z(
    core: pymmcore_plus.CMMCorePlus, z: float | None, expected: list[float]
) -> None:
    """Each image moves Z by 1 while locked; a later find there starts from it."""
    core.setZPosition(0.0)
    with _scope(core, drift=1.0) as calls:
        core.mda.run(_locked_seq(z=z))

    assert [args[0] for args, _ in _called(calls, "setZPosition")] == expected


def test_keep_locked_no_burst_across_xy(core: pymmcore_plus.CMMCorePlus) -> None:
    engine = core.mda.engine
    engine.use_hardware_sequencing = True
    seq = MDASequence(
        grid_plan={"rows": 1, "columns": 3},
        autofocus_plan={"axes": (), "keep_locked": True},
    )
    with (
        patch.object(core, "isXYStageSequenceable", return_value=True),
        patch.object(core, "getXYStageSequenceMaxLength", return_value=100),
    ):
        engine.setup_sequence(seq)
        events = list(engine.event_iterator(seq))
        # without keep_locked, the three tiles would be one burst
        engine.setup_sequence(seq.replace(autofocus_plan=None))
        assert len(list(engine.event_iterator(seq.replace(autofocus_plan=None)))) == 1

    assert [len(getattr(e, "events", [e])) for e in events] == [1, 1, 1, 1]


@pytest.mark.parametrize("on", [False, True])
def test_keep_locked_restores_state_with_the_lock_off(
    core: pymmcore_plus.CMMCorePlus, on: bool
) -> None:
    engine = core.mda.engine
    engine.restore_initial_state = True
    with (
        _scope(core, on=on) as calls,
        patch.object(
            engine,
            "_restore_initial_state",
            lambda: calls.append(("restore", (), core.isContinuousFocusEnabled())),
        ),
    ):
        core.mda.run(_locked_seq(axes=()))
        assert core.isContinuousFocusEnabled() is on

    names = [
        (n, args) for n, args, _ in calls if n in ("restore", "enableContinuousFocus")
    ]
    assert names[-2:] == (
        [("enableContinuousFocus", (False,)), ("restore", ())]
        if not on
        else [("restore", ()), ("enableContinuousFocus", (True,))]
    )
    assert _called(calls, "restore") == [((), False)]
