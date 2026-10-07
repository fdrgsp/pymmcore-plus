"""Hardware autofocus: the Z search for lock, and the `autofocusFinished` signal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from useq import HardwareAutofocus, MDAEvent, MDASequence

from pymmcore_plus.autofocus import AutofocusResult

if TYPE_CHECKING:
    from collections.abc import Callable

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
