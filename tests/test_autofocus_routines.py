"""The software autofocus routines, driving the demo microscope."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from pymmcore_plus.autofocus import (
    AutofocusCancelled,
    AutofocusResult,
    _registry,
    available_methods,
    register_software_autofocus,
    run_software_autofocus,
    settings_model,
)
from pymmcore_plus.autofocus._routines import _scan_pass

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


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_show_images_decides_whether_the_search_is_visible(
    core: pymmcore_plus.CMMCorePlus,
    focus_sim: float,
    method: str,
    settings: dict,
) -> None:
    """Autofocus images are diagnostic, but a user may want to watch them."""
    snapped: list[str] = []
    core.events.imageSnapped.connect(snapped.append)

    core.setZPosition(22.0)
    assert run_software_autofocus(core, method, {**settings}).succeeded
    # by default nothing hears about them, so a live preview does not flicker
    assert snapped == []

    core.setZPosition(22.0)
    result = run_software_autofocus(core, method, {**settings, "show_images": True})
    assert result.succeeded
    assert len(snapped) == result.n_images


def test_duo_passes_show_images_to_the_routine_it_runs(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """`duo` has no cameras of its own: each step carries its own settings."""
    snapped: list[str] = []
    core.events.imageSnapped.connect(snapped.append)

    core.setZPosition(22.0)
    result = run_software_autofocus(
        core,
        "duo",
        {
            "first": {
                "method": "oughtafocus",
                "settings": {"search_range_um": 20.0, "show_images": True},
            },
            "second": {
                "method": "oughtafocus",
                "settings": {"search_range_um": 4.0},
            },
        },
    )
    assert result.succeeded
    # only the first step was asked to show its images
    assert 0 < len(snapped) < result.n_images


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


def test_duo_can_chain_two_different_routines(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """Picking the two steps is the point of it, so a mixed pair must work."""
    core.setZPosition(12.0)
    result = run_software_autofocus(
        core,
        "duo",
        {
            "second": {
                "method": "jaf",
                "settings": {"fine_step_um": 0.5, "settle_ms": 0},
            }
        },
    )
    assert result.succeeded, result.message
    assert result.z_after == pytest.approx(focus_sim, abs=2.0)


def test_duo_inside_duo_terminates(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """Nesting it is wasteful but bounded, since a nested step takes its defaults.

    Worth pinning: a routine that can run other routines is the one place where an
    acquisition could be made to recurse. The GUI does not offer `duo` as a step of
    itself, and nothing here needs to guard against it either.
    """
    core.setZPosition(14.0)
    result = run_software_autofocus(core, "duo", {"first": {"method": "duo"}})
    assert result.succeeded, result.message
    assert result.z_after == pytest.approx(focus_sim, abs=2.0)


# ----------------------- a search that cannot have found focus -----------------------


def _blank(*_a: Any, **_k: Any) -> np.ndarray:
    return np.zeros((48, 48), dtype=np.uint16)


@pytest.mark.parametrize(("method", "settings"), METHODS)
def test_a_blank_image_is_not_mistaken_for_focus(
    core: pymmcore_plus.CMMCorePlus, method: str, settings: dict
) -> None:
    """A lamp left off scores every position the same -- and a search still ends
    *somewhere*, which used to be reported as focus and moved to."""
    core.setZPosition(10.0)
    with patch.object(core, "getImage", side_effect=_blank):
        result = run_software_autofocus(core, method, settings)

    assert not result.succeeded
    assert "scored the same" in result.message
    assert core.getZPosition() == pytest.approx(10.0)
    assert result.z_after == pytest.approx(10.0)
    assert result.n_images > 0  # the curve is still there, to look at


def test_an_unscoreable_image_is_not_mistaken_for_focus(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    core.setZPosition(20.0)
    with patch(
        "pymmcore_plus.autofocus._routines.score_image", return_value=float("nan")
    ):
        result = run_software_autofocus(core, "oughtafocus", {"search_range_um": 6.0})
    assert not result.succeeded
    assert "not a number" in result.message
    assert core.getZPosition() == pytest.approx(20.0)


def test_a_search_that_did_not_converge_is_not_called_focus(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    from pymmcore_plus.autofocus import _routines

    real = _routines.brent_search

    def impatient(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs, max_evaluations=2)

    core.setZPosition(20.0)
    with patch.object(_routines, "brent_search", impatient):
        result = run_software_autofocus(core, "oughtafocus", {"search_range_um": 20.0})
    assert not result.succeeded
    assert "Did not converge" in result.message
    assert core.getZPosition() == pytest.approx(20.0)


# ------------------------------- duo, end to end -------------------------------

WIDE = {
    "method": "oughtafocus",
    "settings": {"search_range_um": 40.0, "optimizer": "zstack", "tolerance_um": 5.0},
}


def test_duo_puts_the_stage_back_when_cancelled_in_its_second_step(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """The second step restores *its* start -- where the first left the stage."""
    core.setZPosition(10.0)
    images = 0

    def cancel_once_the_first_step_is_done() -> bool:
        nonlocal images
        images += 1
        return images > 9  # the first step's scan is 9 images

    with pytest.raises(AutofocusCancelled):
        run_software_autofocus(
            core,
            "duo",
            {"first": WIDE, "second": {"method": "oughtafocus"}},
            should_cancel=cancel_once_the_first_step_is_done,
        )
    assert core.getZPosition() == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("second", "complaint"),
    [
        ({"method": "oughtafocuss"}, "second step: Unknown software autofocus"),
        (
            {"method": "oughtafocus", "settings": {"search_range_um": -1}},
            "second step (oughtafocus): search_range_um must be positive",
        ),
    ],
    ids=["unknown-routine", "invalid-settings"],
)
def test_duo_checks_both_steps_before_moving(
    core: pymmcore_plus.CMMCorePlus,
    focus_sim: float,
    second: dict,
    complaint: str,
) -> None:
    """A typo in the second step used to surface only after the first had moved."""
    core.setZPosition(10.0)
    with patch.object(core, "setPosition", wraps=core.setPosition) as move:
        with pytest.raises((KeyError, ValueError), match=re.escape(complaint)):
            run_software_autofocus(core, "duo", {"first": WIDE, "second": second})
    move.assert_not_called()  # not moved and put back: never moved at all
    assert core.getZPosition() == pytest.approx(10.0)


# -------------------------- jaf: knowing when to stop --------------------------


def _pass(curve: dict[int, float], *, full_scan: bool = False) -> tuple[float, int]:
    samples: list[tuple[float, float]] = []
    best = _scan_pass(
        lambda z: curve[round(z)], 0.0, 1.0, 4, 0.02, samples, None, full_scan=full_scan
    )
    return best, len(samples)


PEAK_AT_ZERO = {
    -4: 1.0,
    -3: 1.5,
    -2: 2.5,
    -1: 4.0,
    0: 5.0,
    1: 4.0,
    2: 2.5,
    3: 1.5,
    4: 1.0,
}


def test_jaf_does_not_stop_at_a_dip_before_the_peak() -> None:
    """One image falling below the best is not proof of having passed the peak --
    least of all when no peak has been seen yet."""
    curve = {**PEAK_AT_ZERO, -4: 1.0, -3: 0.8}
    best, _ = _pass(curve)
    assert best == 0.0


def test_jaf_still_stops_early_once_clearly_past_the_peak() -> None:
    best, images = _pass(PEAK_AT_ZERO)
    assert best == 0.0
    assert images == 7  # the peak, then two images falling away from it


def test_jaf_full_scan_measures_every_position() -> None:
    best, images = _pass(PEAK_AT_ZERO, full_scan=True)
    assert best == 0.0
    assert images == 9


# ------------------ held to the same promises, however it is run ------------------


def test_any_routine_puts_the_stage_back_when_it_raises(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    """Including one registered by a user, which may not know it is meant to."""

    def wanders_off(core, focus_device, settings=None, *, should_cancel=None):
        core.setPosition(focus_device, 99.0)
        raise RuntimeError("lost it")

    register_software_autofocus("wanders_off", wanders_off, dict)
    try:
        core.setZPosition(10.0)
        with pytest.raises(RuntimeError, match="lost it"):
            run_software_autofocus(core, "wanders_off")
        assert core.getZPosition() == pytest.approx(10.0)
    finally:
        _registry._REGISTRY.pop("wanders_off", None)


def test_any_routine_puts_the_stage_back_when_it_fails(
    core: pymmcore_plus.CMMCorePlus,
) -> None:
    def gives_up_where_it_is(core, focus_device, settings=None, *, should_cancel=None):
        z_before = core.getPosition(focus_device)
        core.setPosition(focus_device, 99.0)
        return AutofocusResult(
            kind="software",
            method="gives_up",
            focus_device=focus_device,
            z_before=z_before,
            z_after=99.0,
            succeeded=False,
        )

    register_software_autofocus("gives_up", gives_up_where_it_is, dict)
    try:
        core.setZPosition(10.0)
        assert not run_software_autofocus(core, "gives_up").succeeded
        assert core.getZPosition() == pytest.approx(10.0)
    finally:
        _registry._REGISTRY.pop("gives_up", None)


def test_run_by_hand_it_is_prepared_as_an_acquisition_would_prepare_it(
    core: pymmcore_plus.CMMCorePlus, focus_sim: float
) -> None:
    """A routine run by hand -- a settings dialog's Test button, say -- used to fail
    outright with live running, and to search with the hardware autofocus still
    holding the focus."""
    core.enableContinuousFocus(True)
    core.startContinuousSequenceAcquisition(0)
    core.setZPosition(20.0)

    result = run_software_autofocus(
        core, "oughtafocus", {"search_range_um": 20.0, "optimizer": "zstack"}
    )

    assert result.succeeded, result.message
    assert not core.isSequenceRunning()
    assert not core.isContinuousFocusLocked()
