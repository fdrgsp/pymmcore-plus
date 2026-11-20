"""Hardware autofocus: the Z search for lock, the `autofocusFinished` signal, and
continuous focus during acquisition (`keep_locked`, Z excursions)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

import pytest
from useq import HardwareAutofocus, MDAEvent, MDASequence

from pymmcore_plus._logger import logger
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


def test_failed_autofocus_leaves_continuous_focus_off(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    core.mda.engine._af_was_engaged = True
    with patch.object(core, "fullFocus", side_effect=[None, RuntimeError("no lock")]):
        _exec(core, _af_event())
        assert core.isContinuousFocusEnabled()
        _exec(core, _af_event())
    assert not core.mda.engine._af_succeeded
    assert not core.isContinuousFocusEnabled()


# ---------------------------- continuous focus ----------------------------

Call = tuple[str, tuple, bool]  # (method, args, continuous focus on at the time)


@contextmanager
def _scope(
    core: pymmcore_plus.CMMCorePlus,
    *,
    on: bool = False,
    locks: Literal["yes", "after_find", "no", "cancel"] = "yes",
    finds: bool = True,
    drift: float = 0.0,
    focus_drive: bool = False,
) -> Iterator[list[Call]]:
    """Fake continuous focus, and record in order what the engine asks of it.

    `locks` says when it locks once switched on: always, only after `fullFocus` at
    the current XY position, never, or never and cancelling the acquisition.  Each
    image taken while it is on moves Z by `drift`, as a lock following drift does.
    `focus_drive` makes the focus device a continuous-focus drive (a PFS offset).
    """
    calls: list[Call] = []
    state = {"on": on, "found": False}
    set_position = core.setPosition

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
            set_position(core.getFocusDevice(), core.getZPosition() + drift)

    core.mda.engine.lock_timeout = 0.05
    with (
        patch.object(core, "enableContinuousFocus", enable),
        patch.object(core, "isContinuousFocusEnabled", lambda: state["on"]),
        patch.object(core, "isContinuousFocusLocked", is_locked),
        patch.object(core, "isContinuousFocusDrive", lambda label: focus_drive),
        patch.object(core, "fullFocus", full_focus),
        patch.object(core, "setXYPosition", recorded("setXYPosition", new_xy)),
        patch.object(core, "setZPosition", recorded("setZPosition")),
        patch.object(core, "setPosition", recorded("setPosition")),
        patch.object(core, "setAutoFocusOffset", recorded("setAutoFocusOffset")),
        patch.object(core, "snapImage", recorded("snapImage", drifted)),
    ):
        yield calls


def _called(calls: list[Call], name: str) -> list[tuple[tuple, bool]]:
    return [(args, on) for n, args, on in calls if n == name]


def _z_ops(calls: list[Call]) -> list[str | float]:
    """Focus switches, Z moves (`back` to the reference), images (`SNAP`: unlocked)."""
    ops: list[str | float] = []
    for name, args, on in calls:
        if name == "enableContinuousFocus":
            ops.append("on" if args[0] else "off")
        elif name == "setZPosition":
            ops.append(args[0])
        elif name == "setPosition" and len(args) == 2:  # not via setZPosition
            ops.append(f"back {args[-1]}")
        elif name == "snapImage":
            ops.append("snap" if on else "SNAP")
    return ops


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


def _results(core: pymmcore_plus.CMMCorePlus) -> list[tuple[bool, str]]:
    """Collect (succeeded, message) of every autofocus result."""
    results: list[tuple[bool, str]] = []
    core.mda.events.autofocusFinished.connect(
        lambda e, r: results.append((r.succeeded, r.message))
    )
    return results


def test_keep_locked_searches_on_its_axes(core: pymmcore_plus.CMMCorePlus) -> None:
    results = _results(core)
    with _scope(core) as calls:
        core.mda.run(_locked_seq())

    assert results == [(True, "")] * 4
    assert len(_called(calls, "fullFocus")) == 4
    assert all(on for _, on in _called(calls, "snapImage"))
    # the lock holds Z: it is only moved while the lock is off, which it is again
    assert not any(on for _, on in _called(calls, "setZPosition"))
    assert not core.isContinuousFocusEnabled()


@pytest.mark.parametrize("on", [False, True])
def test_keep_locked_holds_without_searching(
    core: pymmcore_plus.CMMCorePlus, on: bool
) -> None:
    results = _results(core)
    with _scope(core, on=on) as calls:
        core.mda.run(_locked_seq(axes=()))
        assert core.isContinuousFocusEnabled() is on

    assert results == []  # no search, so nothing to report
    assert not _called(calls, "fullFocus")
    offsets = [args for args, _ in _called(calls, "setAutoFocusOffset")]
    assert offsets == [(1.0,), (2.0,), (1.0,), (2.0,)]
    assert all(on for _, on in _called(calls, "snapImage"))
    assert not _called(calls, "setZPosition")
    switches = [args for args, _ in _called(calls, "enableContinuousFocus")]
    assert switches == ([] if on else [(True,), (False,)])


def test_keep_locked_searches_once_when_it_cannot_lock(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    results = _results(core)
    with _scope(core, locks="after_find") as calls:
        core.mda.run(_locked_seq(axes=()))

    assert results == [(True, "")] * 4
    assert len(_called(calls, "fullFocus")) == 4
    assert all(on for _, on in _called(calls, "snapImage"))


def test_keep_locked_gives_up_and_lets_z_move(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    results = _results(core)
    with _scope(core, locks="no", finds=False) as calls:
        core.mda.run(_locked_seq(axes=()))

    # one search per position visit, not one per image
    assert results == [(False, "no focus")] * 4
    assert len(_called(calls, "fullFocus")) == 4
    assert not any(on for _, on in _called(calls, "snapImage"))
    # Z moves resume: to the position's Z for the search, back there when it
    # fails, and for the image
    assert [args for args, _ in _called(calls, "setZPosition")] == [(5.0,)] * 12


def test_keep_locked_search_that_cannot_relock(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    results = _results(core)
    with _scope(core, locks="no") as calls:
        core.mda.run(_locked_seq())

    assert len(_called(calls, "fullFocus")) == 4  # never a second search
    assert all(not ok and "did not lock" in msg for ok, msg in results)
    assert not core.isContinuousFocusEnabled()


def test_keep_locked_cancel_never_searches(core: pymmcore_plus.CMMCorePlus) -> None:
    results = _results(core)
    with _scope(core, locks="cancel") as calls:
        core.mda.engine.lock_timeout = 5.0  # cancelling must end the wait, not this
        core.mda.run(_locked_seq(axes=()))

    assert not _called(calls, "fullFocus")
    assert results == []


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


# ------------------------------ Z excursions ------------------------------

STACK = {"range": 2.0, "step": 1.0}  # planes -1, 0, +1
FITC = {"config": "FITC"}
DAPI = {"config": "DAPI", "z_offset": 2.0}
HOLD = {"axes": (), "keep_locked": True}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"channels": [FITC, DAPI]},
            ["snap", "off", 102.0, "SNAP", "back 100.0", "on"],
        ),
        (
            {"z_plan": STACK},
            ["off", 99.0, "SNAP", 100.0, "SNAP", 101.0, "SNAP", "back 100.0", "on"],
        ),
        (
            {"z_plan": STACK, "channels": [FITC, DAPI], "axis_order": "tpcz"},
            [
                "off",
                99.0,
                "SNAP",
                100.0,
                "SNAP",
                101.0,
                "SNAP",
                101.0,
                "SNAP",
                102.0,
                "SNAP",
                103.0,
                "SNAP",
                "back 100.0",
                "on",
            ],
        ),
        (
            {"z_plan": STACK, "channels": [FITC, DAPI], "axis_order": "tpzc"},
            [
                "off",
                99.0,
                "SNAP",
                101.0,
                "SNAP",
                100.0,
                "SNAP",
                102.0,
                "SNAP",
                101.0,
                "SNAP",
                103.0,
                "SNAP",
                "back 100.0",
                "on",
            ],
        ),
    ],
    ids=["channel-offset", "z-stack", "tpcz", "tpzc"],
)
@pytest.mark.parametrize("plan", [HOLD, {"axes": ()}, None], ids=str)
def test_excursion_from_where_focus_holds(
    core: pymmcore_plus.CMMCorePlus,
    kwargs: dict,
    expected: list,
    plan: dict | None,
) -> None:
    """Planes are relative to the Z the lock holds (100), not the saved Z (5).

    The lock is off once for the excursion, then Z goes back and it locks again.
    Without `keep_locked` this is the same when focus was on at the start.
    """
    core.setZPosition(100.0)
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], autofocus_plan=plan, **kwargs)
    with _scope(core, on=True) as calls:
        core.mda.run(seq)
    assert _z_ops(calls) == expected


def test_excursion_follows_drift(core: pymmcore_plus.CMMCorePlus) -> None:
    """Each excursion samples the reference where the lock is when it starts: the
    baseline image between the two lets the lock follow drift (to 101)."""
    core.setZPosition(100.0)
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], channels=[DAPI, FITC, DAPI])
    with _scope(core, on=True, drift=1.0) as calls:
        core.mda.run(seq)
    first = ["off", 102.0, "SNAP", "back 100.0", "on"]
    second = ["off", 103.0, "SNAP", "back 101.0", "on"]
    assert _z_ops(calls) == [*first, "snap", *second]


def test_excursion_on_a_continuous_focus_drive(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """A PFS offset, say, moves with the lock on: it is never switched off, and its
    saved positions are honored."""
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], channels=[FITC, DAPI])
    with _scope(core, on=True, focus_drive=True) as calls:
        core.mda.run(seq)
    assert _z_ops(calls) == [5.0, "snap", 7.0, "snap", "back 5.0"]


def test_no_excursion_handling_with_focus_off(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Focus off at the start and nothing asking for it: Z moves as it always has."""
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], channels=[FITC, DAPI])
    with _scope(core) as calls:
        core.mda.run(seq)
    assert _z_ops(calls) == [5.0, "SNAP", 7.0, "SNAP"]


def test_search_starts_from_each_position_z(core: pymmcore_plus.CMMCorePlus) -> None:
    """A lock held at one position must not keep the next one's search from
    starting at that position's own Z."""
    seq = MDASequence(
        stage_positions=[{"z": 5.0}, {"z": 50.0}],
        autofocus_plan={"axes": ("p",), "keep_locked": True},
    )
    with _scope(core, on=True) as calls:
        core.mda.run(seq)
    search_z, z = [], None
    for name, args, _ in calls:
        if name == "setZPosition":
            z = args[0]
        elif name == "fullFocus":
            search_z.append(z)
    assert search_z == [5.0, 50.0]


def test_recovery_starts_from_the_position_z(core: pymmcore_plus.CMMCorePlus) -> None:
    """Not from the first image's plane: its stack plane and channel offset."""
    core.setZPosition(50.0)
    seq = MDASequence(
        stage_positions=[{"z": 5.0}],
        channels=[DAPI],
        z_plan=STACK,
        autofocus_plan={**HOLD, "max_retries": 1},
    )
    with _scope(core, locks="after_find") as calls:
        core.mda.run(seq)
    # no lock: off, to the position's Z, search (which switches it off), lock;
    # then the excursion begins
    assert _z_ops(calls)[:7] == ["on", "off", 5.0, "off", "on", "off", 6.0]
    assert len(_called(calls, "fullFocus")) == 1


def test_drive_that_cannot_lock_at_a_plane(
    core: pymmcore_plus.CMMCorePlus, caplog: pytest.LogCaptureFixture
) -> None:
    """On a continuous-focus drive the failure is handled before the image: it is
    reported and focus is switched off at this location (warn and continue)."""
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], channels=[FITC, DAPI])
    logger.setLevel("WARNING")
    try:
        with (
            _scope(core, on=True, focus_drive=True) as calls,
            patch.object(
                core, "isContinuousFocusLocked", lambda: core.getZPosition() < 6
            ),
        ):
            core.mda.run(seq)
    finally:
        logger.setLevel("CRITICAL")
    assert _z_ops(calls) == [5.0, "snap", 7.0, "off", "SNAP", "back 5.0", "on"]
    assert "did not lock" in caplog.text


def test_direct_iterator_restores_before_asking_for_more(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """An iterator gets no look-ahead: each plane goes back and relocks before the
    next event is requested, whose producer may be waiting for this frame."""
    core.setZPosition(100.0)
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], z_plan=STACK)
    states: list[tuple[bool, float]] = []

    def events() -> Iterator[MDAEvent]:
        for event in seq:
            yield event
            states.append((core.isContinuousFocusEnabled(), core.getZPosition()))

    with _scope(core, on=True) as calls:
        core.mda.run(events())
    assert states == [(True, 100.0)] * 3
    expected = []
    for z in (99.0, 100.0, 101.0):
        expected += ["off", z, "SNAP", "back 100.0", "on"]
    assert _z_ops(calls) == expected


@pytest.mark.parametrize("stop", ["cancel", "error"])
def test_stopping_mid_excursion_restores_focus(
    core: pymmcore_plus.CMMCorePlus, stop: str
) -> None:
    core.setZPosition(100.0)
    core.mda.engine.restore_initial_state = False
    seq = MDASequence(stage_positions=[{"z": 5.0}], z_plan=STACK)

    def _cancel(*_: Any) -> None:
        core.mda.cancel()

    if stop == "cancel":
        core.mda.events.frameReady.connect(_cancel)
        ctx: Any = nullcontext()
    else:
        ctx = patch.object(
            core.mda.engine, "exec_single_event", side_effect=RuntimeError("boom")
        )
    with (
        _scope(core, on=True) as calls,
        ctx,
        pytest.raises(RuntimeError) if stop == "error" else nullcontext(),
    ):
        core.mda.run(seq)
    assert _z_ops(calls)[-2:] == ["back 100.0", "on"]
    assert core.getZPosition() == 100.0

    # and the engine starts afresh
    if stop == "cancel":
        core.mda.events.frameReady.disconnect(_cancel)
    with _scope(core, on=True) as calls:
        core.mda.run(seq)
    assert _z_ops(calls)[:2] == ["off", 99.0]


def test_changing_focus_device_ends_the_excursion(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """A reference belongs to its device: it is never restored onto another one."""
    core.loadDevice("Z2", "DemoCamera", "DStage")
    core.initializeDevice("Z2")
    core.setPosition("Z", 100.0)
    core.setPosition("Z2", 50.0)
    engine = core.mda.engine
    seq = MDASequence(stage_positions=[{"z": 5.0}], z_plan=STACK)
    first, _, last = list(seq)
    with _scope(core, on=True):
        engine.setup_sequence(seq)
        engine.setup_event(first)  # plane -1 of Z
        core.setFocusDevice("Z2")
        engine.setup_event(last)  # plane +1 of Z2
    assert core.getPosition("Z") == 100.0
    assert core.getPosition("Z2") == 51.0


# ----------------------------- hardware bursts -----------------------------


@contextmanager
def _sequenceable_z(core: pymmcore_plus.CMMCorePlus) -> Iterator[list[tuple]]:
    loaded: list[tuple] = []
    with (
        patch.object(core, "isStageSequenceable", return_value=True),
        patch.object(core, "getStageSequenceMaxLength", return_value=100),
        patch.object(core, "loadStageSequence", lambda d, z: loaded.append(tuple(z))),
        patch.object(core, "startStageSequence"),
        patch.object(core, "stopStageSequence"),
    ):
        yield loaded


def test_bursts_respect_focus_boundaries(core: pymmcore_plus.CMMCorePlus) -> None:
    """A baseline image and an offset image are not one burst while focus may be
    on, but a whole Z stack is, with its planes rebased on the reference."""
    engine = core.mda.engine
    engine.use_hardware_sequencing = True
    engine.restore_initial_state = False
    offset = MDASequence(
        stage_positions=[{"z": 5.0}],
        channels=[FITC, {**FITC, "z_offset": 2.0}],
        autofocus_plan=HOLD,
    )
    stack = MDASequence(stage_positions=[{"z": 5.0}], z_plan=STACK, autofocus_plan=HOLD)
    unlocked = offset.replace(autofocus_plan=None)
    # focus may still be on from the first position at the second, planless one
    mixed = unlocked.replace(
        stage_positions=[{"z": 5.0, "sequence": {"autofocus_plan": HOLD}}, {"z": 5.0}]
    )
    # an event stream switches focus on with its first action
    stream = [
        MDAEvent(action=HardwareAutofocus(keep_locked=True)),
        MDAEvent(index={"p": 0}, z_pos=5.0),
        MDAEvent(index={"p": 1}, z_pos=7.0),
    ]
    two_stacks = stack.replace(time_plan={"interval": 0, "loops": 2})
    time_lapse = offset.replace(channels=[FITC], time_plan={"interval": 0, "loops": 3})

    def burst_sizes(events: Any) -> list[int]:
        engine.setup_sequence(
            events if isinstance(events, MDASequence) else MDASequence()
        )
        return [len(getattr(e, "events", [e])) for e in engine.event_iterator(events)]

    core.setZPosition(100.0)
    with _sequenceable_z(core) as loaded:
        assert burst_sizes(offset) == [1, 1]
        assert burst_sizes(unlocked) == [2]
        assert burst_sizes(mixed) == [1, 1, 1, 1]
        assert burst_sizes(stream) == [1, 1, 1]
        assert burst_sizes(stream[1:]) == [2]
        # events from a sequence carry its plans, whether run as one or as a list
        assert burst_sizes(list(offset)) == burst_sizes(offset)
        assert burst_sizes(two_stacks) == [3, 3]  # relocked between timepoints
        assert burst_sizes(time_lapse) == [3]  # the lock holds across timepoints

        with _scope(core, on=True) as calls:
            core.mda.run(stack)
    assert loaded == [(99.0, 100.0, 101.0)]
    assert _z_ops(calls)[0] == "off"
    assert _z_ops(calls)[-2:] == ["back 100.0", "on"]
