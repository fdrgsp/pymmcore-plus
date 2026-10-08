"""The software autofocus routines, driving the demo microscope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from pymmcore_plus.autofocus import (
    AutofocusCancelled,
    AutofocusResult,
    available_methods,
    run_software_autofocus,
    settings_model,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pymmcore_plus

Z_FOCUS = 25.0


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(int(3 * sigma), 1)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, "same"), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, "same"), 1, out)


def _sample(sigma: float, size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = _blur(rng.normal(0.0, 1.0, (size, size)), 1.2)
    base = 400.0 * base / base.std()
    return (_blur(base, sigma) + 500.0).astype(np.uint16)


@pytest.fixture
def focus_sim(core: pymmcore_plus.CMMCorePlus) -> Iterator[float]:
    """Make the camera return an image that is sharpest at `Z_FOCUS`."""
    drive = core.getFocusDevice()

    def fake_image(*_a: Any, **_k: Any) -> np.ndarray:
        # defocus blurs the image, in proportion to the distance from focus
        return _sample(0.3 + abs(core.getPosition(drive) - Z_FOCUS) * 0.6)

    with patch.object(core, "getImage", side_effect=fake_image):
        yield Z_FOCUS


METHODS = [
    pytest.param("oughtafocus", {"search_range_um": 20.0}, id="oughtafocus-brent"),
    pytest.param(
        "oughtafocus",
        {"search_range_um": 20.0, "optimizer": "zstack", "tolerance_um": 2.0},
        id="oughtafocus-zstack",
    ),
    pytest.param(
        "jaf",
        {"coarse_step_um": 4.0, "coarse_steps": 4, "fine_step_um": 1.0, "settle_ms": 0},
        id="jaf",
    ),
]


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_finds_focus(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float, method: str, settings: dict
) -> None:
    core.setZPosition(20.0)
    result = run_software_autofocus(core, method, settings)

    assert result.succeeded, result.message
    assert result.z_after == pytest.approx(focus_sim, abs=2.0)
    # the routine leaves the focus device where it decided
    assert core.getZPosition() == pytest.approx(result.z_after, abs=1e-6)


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_reports_what_it_did(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float, method: str, settings: dict
) -> None:
    core.setZPosition(22.0)
    result = run_software_autofocus(core, method, settings)

    assert isinstance(result, AutofocusResult)
    assert result.kind == "software"
    assert result.method == method
    assert result.focus_device == core.getFocusDevice()
    assert result.z_before == 22.0
    assert result.n_images == len(result.scores) > 2
    assert result.delta_z == pytest.approx(result.z_after - result.z_before)
    # the focus curve, for a plot
    assert all(len(pair) == 2 for pair in result.scores)


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_cancellation_restores_z(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float, method: str, settings: dict
) -> None:
    core.setZPosition(20.0)
    calls = 0

    def should_cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(AutofocusCancelled):
        run_software_autofocus(core, method, settings, should_cancel=should_cancel)
    # a cancelled run must not leave the sample somewhere else
    assert core.getZPosition() == pytest.approx(20.0)


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_failure_restores_z_and_reports(
    core: pymmcore_plus.CMMCorePlus, method: str, settings: dict
) -> None:
    core.setZPosition(20.0)
    with patch.object(core, "getImage", side_effect=RuntimeError("camera died")):
        result = run_software_autofocus(core, method, settings)

    assert not result.succeeded
    assert "camera died" in result.message
    assert result.z_after == 20.0
    assert core.getZPosition() == pytest.approx(20.0)


def test_search_stays_within_its_range(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """The range is a safety limit on how far the objective may travel."""
    core.setZPosition(0.0)  # far from focus, so the search runs to its limit
    result = run_software_autofocus(core, "oughtafocus", {"search_range_um": 6.0})
    for z, _ in result.scores:
        assert -3.0 - 1e-9 <= z <= 3.0 + 1e-9


def test_capture_settings_are_restored(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    group = core.getChannelGroup() or "Channel"
    core.setConfig(group, "DAPI")
    core.setExposure(100.0)
    core.setZPosition(24.0)

    result = run_software_autofocus(
        core,
        "oughtafocus",
        {
            "search_range_um": 6.0,
            "channel": "FITC",
            "channel_group": group,
            "exposure_ms": 5.0,
            "crop_factor": 0.5,
        },
    )
    assert result.succeeded
    assert core.getCurrentConfig(group) == "DAPI"
    assert core.getExposure() == 100.0


def test_duo_runs_both_in_sequence(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """A coarse pass then a precise one reaches focus from further out."""
    core.setZPosition(12.0)
    coarse = {"search_range_um": 40.0, "optimizer": "zstack", "tolerance_um": 4.0}
    fine = {"search_range_um": 6.0, "tolerance_um": 0.2}
    result = run_software_autofocus(
        core,
        "duo",
        {
            "first": {"method": "oughtafocus", "settings": coarse},
            "second": {"method": "oughtafocus", "settings": fine},
        },
    )
    assert result.succeeded, result.message
    assert result.z_after == pytest.approx(focus_sim, abs=1.5)
    # both passes' images are reported
    assert result.n_images > len(coarse) + 2


def test_duo_reports_which_step_failed(core: pymmcore_plus.CMMCorePlus) -> None:
    core.setZPosition(20.0)
    with patch.object(core, "getImage", side_effect=RuntimeError("camera died")):
        result = run_software_autofocus(
            core,
            "duo",
            {
                "first": {"method": "oughtafocus", "settings": {}},
                "second": {"method": "oughtafocus", "settings": {}},
            },
        )
    assert not result.succeeded
    assert result.message.startswith("oughtafocus:")
    assert core.getZPosition() == pytest.approx(20.0)


# -------------------------------- the registry --------------------------------


def test_built_in_methods_are_registered() -> None:
    assert set(available_methods()) >= {"oughtafocus", "jaf", "duo"}


def test_unknown_method_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="Unknown software autofocus method"):
        settings_model("nope")


def test_a_misspelled_setting_is_rejected(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """Loudly, rather than focusing with settings the user did not choose.

    Unlike a hardware condition such as a missing focus device -- which is reported
    as a failed result, so an acquisition can log it and carry on -- a setting that
    does not exist is a mistake in the sequence that will never work.
    """
    core.setZPosition(20.0)
    with pytest.raises(ValueError, match="Unknown setting") as excinfo:
        run_software_autofocus(core, "oughtafocus", {"search_rangee_um": 5.0})
    assert "search_range_um" in str(excinfo.value)  # names the valid ones
    assert core.getZPosition() == pytest.approx(20.0)


def test_without_a_focus_device_it_says_so(core: pymmcore_plus.CMMCorePlus) -> None:
    with patch.object(core, "getFocusDevice", return_value=""):
        result = run_software_autofocus(core, "oughtafocus")
    assert not result.succeeded
    assert "No focus device" in result.message


def test_an_explicit_focus_device_is_used(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    drive = core.getFocusDevice()
    core.setZPosition(23.0)
    result = run_software_autofocus(
        core, "oughtafocus", {"search_range_um": 6.0}, focus_device=drive
    )
    assert result.focus_device == drive
    assert result.succeeded


def test_duo_works_without_being_configured(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """Its defaults are the coarse-then-fine arrangement it exists for."""
    core.setZPosition(10.0)
    result = run_software_autofocus(core, "duo")
    assert result.succeeded, result.message
    assert result.z_after == pytest.approx(focus_sim, abs=1.5)


def test_methods_are_ordered_most_useful_first() -> None:
    """A GUI offering a choice takes the first as its default."""
    methods = available_methods()
    assert methods[0] == "oughtafocus"
    # duo runs two routines, so it costs the most images: last
    assert methods[-1] == "duo"
