"""Temporary camera state for a focus run must always be put back."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pymmcore_plus._logger import logger
from pymmcore_plus.autofocus._capture import CaptureSettings, capture_state

if TYPE_CHECKING:
    import pymmcore_plus


def test_defaults_change_nothing(core: pymmcore_plus.CMMCorePlus) -> None:
    before = (core.getExposure(), tuple(core.getROI()), core.getAutoShutter())
    with capture_state(core, CaptureSettings()):
        assert (
            core.getExposure(),
            tuple(core.getROI()),
            core.getAutoShutter(),
        ) == before
    assert (core.getExposure(), tuple(core.getROI()), core.getAutoShutter()) == before


def test_exposure_is_applied_and_restored(core: pymmcore_plus.CMMCorePlus) -> None:
    core.setExposure(100.0)
    with capture_state(core, CaptureSettings(exposure_ms=5.0)):
        assert core.getExposure() == 5.0
    assert core.getExposure() == 100.0


def test_channel_is_applied_and_restored(core: pymmcore_plus.CMMCorePlus) -> None:
    group = core.getChannelGroup() or "Channel"
    core.setConfig(group, "DAPI")
    with capture_state(core, CaptureSettings(channel="FITC", channel_group=group)):
        assert core.getCurrentConfig(group) == "FITC"
    assert core.getCurrentConfig(group) == "DAPI"


def test_crop_is_centered_and_restored(core: pymmcore_plus.CMMCorePlus) -> None:
    full = tuple(core.getROI())
    with capture_state(core, CaptureSettings(crop_factor=0.5)):
        x, y, w, h = core.getROI()
        assert (w, h) == (full[2] // 2, full[3] // 2)
        # centred: equal margins on both sides
        assert x == (full[2] - w) // 2
        assert y == (full[3] - h) // 2
    assert tuple(core.getROI()) == full


def test_crop_never_goes_below_a_usable_size(core: pymmcore_plus.CMMCorePlus) -> None:
    """The 3x3 filters need something to work on."""
    with capture_state(core, CaptureSettings(crop_factor=0.001)):
        _, _, w, h = core.getROI()
        assert w >= 8 and h >= 8


def test_shutter_is_held_open_then_restored(core: pymmcore_plus.CMMCorePlus) -> None:
    core.setAutoShutter(True)
    with capture_state(core, CaptureSettings(keep_shutter_open=True)):
        assert core.getAutoShutter() is False
        assert core.getShutterOpen() is True
    assert core.getAutoShutter() is True


def test_state_is_restored_after_an_error(core: pymmcore_plus.CMMCorePlus) -> None:
    """A routine that raises partway must not leave its settings applied."""
    core.setExposure(100.0)
    full = tuple(core.getROI())
    with pytest.raises(RuntimeError, match="boom"):
        with capture_state(core, CaptureSettings(exposure_ms=5.0, crop_factor=0.5)):
            raise RuntimeError("boom")
    assert core.getExposure() == 100.0
    assert tuple(core.getROI()) == full


def test_one_failed_restore_does_not_strand_the_rest(
    core: pymmcore_plus.CMMCorePlus, caplog: pytest.LogCaptureFixture
) -> None:
    """A device that fails on the way out must not block the other restores."""
    core.setExposure(100.0)
    real_set_roi = core.setROI
    calls = 0

    def flaky_set_roi(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:  # succeeds applying the crop, fails putting it back
            raise RuntimeError("stuck camera")
        real_set_roi(*args)  # type: ignore[arg-type]

    logger.setLevel("DEBUG")
    try:
        with patch.object(core, "setROI", side_effect=flaky_set_roi):
            with capture_state(core, CaptureSettings(exposure_ms=5.0, crop_factor=0.5)):
                pass
    finally:
        logger.setLevel("CRITICAL")

    # the ROI could not be put back, but the exposure still was
    assert core.getExposure() == 100.0
    assert "Failed to restore roi" in caplog.text
    core.clearROI()


def test_a_failure_while_applying_rolls_back(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Applying is several device calls; a late failure must undo the early ones."""
    core.setExposure(100.0)
    with patch.object(core, "setROI", side_effect=RuntimeError("stuck camera")):
        with pytest.raises(RuntimeError, match="stuck camera"):
            with capture_state(core, CaptureSettings(exposure_ms=5.0, crop_factor=0.5)):
                pytest.fail("the body must not run")  # pragma: no cover
    assert core.getExposure() == 100.0


@pytest.mark.parametrize("crop", [0.0, -0.5, 1.5])
def test_invalid_crop_factor_is_rejected(crop: float) -> None:
    with pytest.raises(ValueError, match="crop_factor"):
        CaptureSettings(crop_factor=crop)


@pytest.mark.parametrize("exposure", [0.0, -1.0])
def test_invalid_exposure_is_rejected(exposure: float) -> None:
    with pytest.raises(ValueError, match="exposure_ms"):
        CaptureSettings(exposure_ms=exposure)


def test_everything_at_once_round_trips(core: pymmcore_plus.CMMCorePlus) -> None:
    group = core.getChannelGroup() or "Channel"
    core.setConfig(group, "DAPI")
    core.setExposure(100.0)
    core.setAutoShutter(True)
    before = (core.getCurrentConfig(group), core.getExposure(), tuple(core.getROI()))

    settings = CaptureSettings(
        channel="FITC",
        channel_group=group,
        exposure_ms=5.0,
        crop_factor=0.25,
        keep_shutter_open=True,
    )
    with capture_state(core, settings):
        assert core.getCurrentConfig(group) == "FITC"
        assert core.getExposure() == 5.0
        assert core.getROI()[2] < before[2][2]
        assert core.getAutoShutter() is False

    assert (
        core.getCurrentConfig(group),
        core.getExposure(),
        tuple(core.getROI()),
    ) == before
    assert core.getAutoShutter() is True
