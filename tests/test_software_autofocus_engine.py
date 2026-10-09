"""Software autofocus inside an MDA: when it runs, and what it corrects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import useq

from pymmcore_plus.autofocus import (
    AutofocusResult,
    available_methods,
    register_software_autofocus,
)
from pymmcore_plus.autofocus._settings import from_dict

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pymmcore_plus

FOUND_Z = 42.0


@pytest.fixture
def fake_method() -> Iterator[list[dict[str, Any]]]:
    """Register a routine that 'finds' focus at FOUND_Z, recording its calls."""
    calls: list[dict[str, Any]] = []

    def _run(core, focus_device, settings=None, *, should_cancel=None):
        z_before = core.getPosition(focus_device)
        calls.append({"settings": settings, "z_before": z_before})
        core.setPosition(focus_device, FOUND_Z)
        core.waitForDevice(focus_device)
        return AutofocusResult(
            kind="software",
            method="fake",
            focus_device=focus_device,
            z_before=z_before,
            z_after=FOUND_Z,
            succeeded=True,
            scores=((FOUND_Z, 1.0),),
            n_images=1,
        )

    register_software_autofocus("fake", _run, dict, description="test double")
    try:
        yield calls
    finally:
        from pymmcore_plus.autofocus import _registry

        _registry._REGISTRY.pop("fake", None)


def _plan(**kwargs: Any) -> dict:
    return {"axes": ("p",), "method": "fake", **kwargs}


def test_runs_on_the_selected_axis(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    mda = useq.MDASequence(
        stage_positions=[{"z": 10.0}, {"z": 20.0}], autofocus_plan=_plan()
    )
    core.mda.run(mda)
    assert len(fake_method) == 2  # once per position, not per image


def test_correction_recenters_the_z_stack(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    """The whole point: the stack that follows is centred on the focus just found."""
    z_positions: list[float] = []
    real_snap = core.snapImage

    def _snap() -> None:
        z_positions.append(core.getZPosition())
        real_snap()

    mda = useq.MDASequence(
        stage_positions=[{"z": 10.0}],
        z_plan={"range": 2.0, "step": 1.0},
        autofocus_plan=_plan(),
    )
    with patch.object(core, "snapImage", _snap):
        core.mda.run(mda)

    # autofocus moved 10 -> 42, so the relative stack is centred on 42
    assert z_positions == pytest.approx([41.0, 42.0, 43.0])
    assert core.mda.engine._z_correction == {0: 32.0}


def test_settings_reach_the_routine(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    mda = useq.MDASequence(
        stage_positions=[{"z": 0.0}],
        autofocus_plan=_plan(settings={"search_range_um": 7.0}),
    )
    core.mda.run(mda)
    assert fake_method[0]["settings"] == {"search_range_um": 7.0}


def test_no_correction_for_another_stage(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    """A routine told to move a different drive has already left it in place.

    Correcting the z plan as well would double the move.  (The demo config has only
    one stage, so the *other* drive is simulated by renaming the default one.)
    """
    drive = core.getFocusDevice()
    event = useq.MDAEvent(
        index={"p": 0},
        action=useq.SoftwareAutofocus(method="fake", focus_device=drive),
    )
    with patch.object(core, "getFocusDevice", return_value="SomeOtherStage"):
        list(core.mda.engine.exec_event(event))

    assert fake_method  # the routine ran, on the drive it was given
    assert core.mda.engine._z_correction == {}


def test_an_unknown_method_is_skipped_with_a_warning(
    core: pymmcore_plus.CMMCorePlus, caplog: pytest.LogCaptureFixture
) -> None:
    from pymmcore_plus._logger import logger

    logger.setLevel("DEBUG")
    try:
        event = useq.MDAEvent(action=useq.SoftwareAutofocus(method="not-registered"))
        list(core.mda.engine.exec_event(event))
    finally:
        logger.setLevel("CRITICAL")
    assert "Unknown software autofocus method" in caplog.text
    assert "not-registered" in caplog.text


def test_a_failing_routine_is_retried_then_reported(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    attempts = 0

    def _flaky(core, focus_device, settings=None, *, should_cancel=None):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("no contrast")

    register_software_autofocus("flaky", _flaky, dict)
    try:
        results: list[AutofocusResult] = []
        core.mda.events.autofocusFinished.connect(lambda e, r: results.append(r))
        event = useq.MDAEvent(
            action=useq.SoftwareAutofocus(method="flaky", max_retries=3)
        )
        list(core.mda.engine.exec_event(event))
    finally:
        from pymmcore_plus.autofocus import _registry

        _registry._REGISTRY.pop("flaky", None)

    assert attempts == 3
    assert len(results) == 1
    assert not results[0].succeeded
    assert "no contrast" in results[0].message


def test_every_retry_starts_from_where_the_first_did(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """A routine that moves and then fails used to leave each retry starting from
    wherever the last gave up, ratcheting the focus away -- 10, 20, 30 -- and the
    frame was then acquired 30 um from where the run had put it."""
    starts: list[float] = []

    def climbs_then_fails(core, focus_device, settings=None, *, should_cancel=None):
        starts.append(core.getPosition(focus_device))
        core.setPosition(focus_device, starts[-1] + 10.0)
        raise RuntimeError("lost it")

    register_software_autofocus("climbs", climbs_then_fails, dict)
    try:
        core.setZPosition(10.0)
        event = useq.MDAEvent(
            action=useq.SoftwareAutofocus(method="climbs", max_retries=3)
        )
        list(core.mda.engine.exec_event(event))
    finally:
        from pymmcore_plus.autofocus import _registry

        _registry._REGISTRY.pop("climbs", None)

    assert starts == pytest.approx([10.0, 10.0, 10.0])
    assert core.getZPosition() == pytest.approx(10.0)


def test_a_setting_that_does_not_exist_is_reported_once_not_retried(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """It would fail every attempt the same way, and cannot fix itself."""

    @dataclass(frozen=True)
    class Settings:
        range_um: float = 1.0

    attempts = 0

    def counted(core, focus_device, settings=None, *, should_cancel=None):
        nonlocal attempts
        attempts += 1
        from_dict(Settings, settings)  # as every routine does, before moving
        raise AssertionError("unreachable")  # pragma: no cover

    register_software_autofocus("counted", counted, Settings)
    results: list[AutofocusResult] = []
    core.mda.events.autofocusFinished.connect(lambda e, r: results.append(r))
    try:
        event = useq.MDAEvent(
            action=useq.SoftwareAutofocus(
                method="counted", settings={"range_umm": 5.0}, max_retries=3
            )
        )
        list(core.mda.engine.exec_event(event))
    finally:
        from pymmcore_plus.autofocus import _registry

        _registry._REGISTRY.pop("counted", None)

    assert attempts == 0
    assert len(results) == 1
    assert not results[0].succeeded
    assert "Unknown setting" in results[0].message


def test_reports_the_focus_curve(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    results: list[AutofocusResult] = []
    core.mda.events.autofocusFinished.connect(lambda e, r: results.append(r))
    core.mda.run(useq.MDASequence(stage_positions=[{"z": 0.0}], autofocus_plan=_plan()))
    assert len(results) == 1
    assert results[0].kind == "software"
    assert results[0].scores == ((FOUND_Z, 1.0),)


def test_continuous_focus_is_switched_off_and_left_off(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    """Re-engaging would pull focus off the position the routine just measured."""
    with (
        patch.object(core, "isContinuousFocusLocked", return_value=True),
        patch.object(core, "enableContinuousFocus") as mock_af,
    ):
        core.mda.run(
            useq.MDASequence(
                stage_positions=[{"z": 0.0}],
                z_plan={"range": 2.0, "step": 1.0},
                autofocus_plan=_plan(),
            )
        )
    assert mock_af.call_args_list
    assert all(call.args == (False,) for call in mock_af.call_args_list)


def test_software_autofocus_events_are_not_sequenced(
    core: pymmcore_plus.CMMCorePlus, fake_method: list
) -> None:
    """An autofocus event must not be merged into a hardware-triggered burst."""
    core.mda.engine.use_hardware_sequencing = True
    try:
        core.mda.run(
            useq.MDASequence(
                stage_positions=[{"z": 0.0}],
                time_plan={"interval": 0, "loops": 3},
                autofocus_plan=_plan(),
            )
        )
    finally:
        core.mda.engine.use_hardware_sequencing = False
    assert len(fake_method) == 1


def test_the_fake_is_not_left_registered(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    assert "fake" not in available_methods()


def test_cancelling_a_run_mid_autofocus_is_not_an_error(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Cancelling is a normal end to a run, not a failed acquisition.

    The routine is told to stop through `should_cancel`, which it reports by raising
    -- but that came from the runner's own cancel request, so letting it propagate
    would end the run with a traceback instead of "canceled".
    """
    from pymmcore_plus.autofocus import AutofocusCancelled
    from pymmcore_plus.mda import FinishReason

    def _cancels(core, focus_device, settings=None, *, should_cancel=None):
        core.mda.cancel()  # as the user would, mid-autofocus
        raise AutofocusCancelled("Autofocus cancelled after 3 image(s).")

    register_software_autofocus("cancels", _cancels, dict)
    try:
        results: list[AutofocusResult] = []
        core.mda.events.autofocusFinished.connect(lambda e, r: results.append(r))
        mda = useq.MDASequence(
            stage_positions=[{"z": 0.0}],
            time_plan={"interval": 0, "loops": 4},
            autofocus_plan=_plan(method="cancels", axes=("t",)),
        )
        core.mda.run(mda)  # must return, not raise
    finally:
        from pymmcore_plus.autofocus import _registry

        _registry._REGISTRY.pop("cancels", None)

    assert core.mda.status.finish_reason == FinishReason.CANCELED
    # and the cancellation is reported rather than silently swallowed
    assert results and not results[0].succeeded
    assert "cancelled" in results[0].message.lower()
