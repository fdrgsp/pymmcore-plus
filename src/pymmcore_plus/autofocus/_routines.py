"""The software autofocus routines themselves: move the stage, score images, pick a Z.

Each routine restores the focus device to where it started if it fails or is
cancelled, so a failed autofocus leaves the system as it found it.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

import numpy as np

import pymmcore_plus._pymmcore as pymmcore
from pymmcore_plus._logger import logger

from ._capture import capture_state
from ._optimizers import (
    AutofocusCancelled,
    FocusSearchResult,
    brent_search,
    zstack_search,
)
from ._result import AutofocusResult
from ._scoring import jaf_score, score_image
from ._settings import DuoSettings, JAFSettings, OughtaFocusSettings, from_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from pymmcore_plus.core import CMMCorePlus

__all__ = ["duo", "jaf", "oughtafocus"]

# Rec. 709 luminance, for a colour camera
_LUMINANCE = (0.2126, 0.7152, 0.0722)


def _to_2d(image: NDArray) -> NDArray:
    """Reduce a colour image to luminance; focus scores work on one plane."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        lum: NDArray = np.einsum(
            "...c,c->...", image[..., :3].astype(np.float64), _LUMINANCE
        )
        return lum
    raise ValueError(  # pragma: no cover
        f"Cannot score an image of shape {image.shape}."
    )


def _snap(core: CMMCorePlus, show_images: bool = False) -> NDArray:
    """Acquire one image, optionally without announcing it to the rest of the app.

    Autofocus images are diagnostic: they are scored and discarded, never saved. By
    default they are acquired through the base class, so a live preview does not
    flicker through them and no handler mistakes them for acquisition data.
    """
    if show_images:
        core.snapImage()
    else:
        pymmcore.CMMCore.snapImage(core)
    return _to_2d(core.getImage())


def _measurer(
    core: CMMCorePlus,
    focus_device: str,
    score: Callable[[NDArray], float],
    settle_ms: float = 0.0,
    show_images: bool = False,
) -> Callable[[float], float]:
    """Return `measure(z)`: move there, acquire, and score."""

    def measure(z: float) -> float:
        core.setPosition(focus_device, z)
        core.waitForDevice(focus_device)
        if settle_ms:
            time.sleep(settle_ms / 1000.0)
        return score(_snap(core, show_images))

    return measure


def _result(
    method: str,
    focus_device: str,
    z_before: float,
    search: FocusSearchResult,
    succeeded: bool = True,
    message: str = "",
) -> AutofocusResult:
    return AutofocusResult(
        kind="software",
        method=method,
        focus_device=focus_device,
        z_before=z_before,
        z_after=search.z if succeeded else z_before,
        succeeded=succeeded,
        message=message or search.message,
        scores=search.samples,
        n_images=search.n_images,
    )


def _no_focus_information(samples: Sequence[tuple[float, float]]) -> str:
    """Why these `(z, score)` samples cannot place a focus, or "" if they can.

    A search always ends *somewhere*, so a curve with no information in it -- a
    blank or saturated frame, a closed shutter, a lamp left off -- would otherwise
    move the sample to an arbitrary position and call that focus.
    """
    scores = np.array([score for _, score in samples], dtype=np.float64)
    if scores.size == 0:
        return "No images were scored."
    if not np.isfinite(scores).all():
        return "Some images could not be scored: their sharpness was not a number."
    top = float(scores.max())
    if top - float(scores.min()) <= 1e-9 * max(abs(top), 1.0):
        return (
            "Every image scored the same, so there is nothing to focus on. Check "
            "that the light is on, the shutter open, and the images neither black "
            "nor saturated."
        )
    return ""


def _move_to(core: CMMCorePlus, focus_device: str, z: float) -> None:
    core.setPosition(focus_device, z)
    core.waitForDevice(focus_device)


def oughtafocus(
    core: CMMCorePlus,
    focus_device: str,
    settings: Mapping[str, Any] | OughtaFocusSettings | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> AutofocusResult:
    """Search a Z range for the sharpest image.

    The general-purpose routine: pick a sharpness score and a search strategy, and it
    walks the focus device to the peak.  See
    [`OughtaFocusSettings`][pymmcore_plus.autofocus.OughtaFocusSettings].
    """
    cfg = from_dict(OughtaFocusSettings, settings)
    z_before = core.getPosition(focus_device)

    def score(image: NDArray) -> float:
        return score_image(
            image,
            cfg.scoring,
            fft_lower_pct=cfg.fft_lower_pct,
            fft_upper_pct=cfg.fft_upper_pct,
        )

    try:
        with capture_state(core, cfg.capture()):
            measure = _measurer(
                core, focus_device, score, cfg.settle_ms, cfg.show_images
            )
            if cfg.optimizer == "brent":
                search = brent_search(
                    measure,
                    z_before,
                    cfg.search_range_um,
                    tolerance_um=cfg.tolerance_um,
                    should_cancel=should_cancel,
                )
            else:
                search = zstack_search(
                    measure,
                    z_before,
                    cfg.search_range_um,
                    step_um=cfg.tolerance_um,
                    should_cancel=should_cancel,
                )
            reason = _no_focus_information(search.samples) or (
                "" if search.converged else search.message
            )
            if reason:
                _restore(core, focus_device, z_before)
                logger.warning(
                    "Software autofocus (oughtafocus) found no focus. %s", reason
                )
                return _result(
                    "oughtafocus",
                    focus_device,
                    z_before,
                    search,
                    succeeded=False,
                    message=reason,
                )
            _move_to(core, focus_device, search.z)
    except Exception as e:
        _restore(core, focus_device, z_before)
        if isinstance(e, AutofocusCancelled):
            raise
        logger.warning("Software autofocus (oughtafocus) failed. %s", e)
        return AutofocusResult(
            kind="software",
            method="oughtafocus",
            focus_device=focus_device,
            z_before=z_before,
            z_after=z_before,
            succeeded=False,
            message=str(e),
        )
    return _result("oughtafocus", focus_device, z_before, search)


def jaf(
    core: CMMCorePlus,
    focus_device: str,
    settings: Mapping[str, Any] | JAFSettings | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> AutofocusResult:
    """Find focus with a coarse scan followed by a fine one.

    Each pass walks outward from its start, keeping the best position and stopping
    early once the score has fallen well below it. See
    [`JAFSettings`][pymmcore_plus.autofocus.JAFSettings].
    """
    cfg = from_dict(JAFSettings, settings)
    z_before = core.getPosition(focus_device)
    samples: list[tuple[float, float]] = []

    def score(image: NDArray) -> float:
        return jaf_score(image, cfg.crop_ratio)

    try:
        best_z = z_before
        passes = (
            (cfg.channel, cfg.coarse_step_um, cfg.coarse_steps),
            (cfg.fine_channel or cfg.channel, cfg.fine_step_um, cfg.fine_steps),
        )
        for channel, step, n_steps in passes:
            first = len(samples)
            with capture_state(core, cfg.capture(channel)):
                measure = _measurer(
                    core, focus_device, score, cfg.settle_ms, cfg.show_images
                )
                best_z = _scan_pass(
                    measure,
                    best_z,
                    step,
                    n_steps,
                    cfg.threshold,
                    samples,
                    should_cancel,
                    full_scan=cfg.full_scan,
                )
            # checked per pass, so a blank coarse pass does not also pay for a fine one
            if reason := _no_focus_information(samples[first:]):
                _restore(core, focus_device, z_before)
                logger.warning("Software autofocus (jaf) found no focus. %s", reason)
                return AutofocusResult(
                    kind="software",
                    method="jaf",
                    focus_device=focus_device,
                    z_before=z_before,
                    z_after=z_before,
                    succeeded=False,
                    message=reason,
                    scores=tuple(samples),
                    n_images=len(samples),
                )
        _move_to(core, focus_device, best_z)
    except Exception as e:
        _restore(core, focus_device, z_before)
        if isinstance(e, AutofocusCancelled):
            raise
        logger.warning("Software autofocus (jaf) failed. %s", e)
        return AutofocusResult(
            kind="software",
            method="jaf",
            focus_device=focus_device,
            z_before=z_before,
            z_after=z_before,
            succeeded=False,
            message=str(e),
            scores=tuple(samples),
            n_images=len(samples),
        )

    return AutofocusResult(
        kind="software",
        method="jaf",
        focus_device=focus_device,
        z_before=z_before,
        z_after=best_z,
        succeeded=True,
        scores=tuple(samples),
        n_images=len(samples),
    )


# Images in a row that must fall more than `threshold` below the best before a pass
# stops early. Micro-Manager's JAF stops at the first one; see `_scan_pass`.
_FALLS_TO_STOP = 2


def _scan_pass(
    measure: Callable[[float], float],
    center: float,
    step_um: float,
    n_steps: int,
    threshold: float,
    samples: list[tuple[float, float]],
    should_cancel: Callable[[], bool] | None,
    *,
    full_scan: bool = False,
) -> float:
    """One symmetric scan about `center`, stopping early once clearly past the peak.

    Stopping early takes evidence of a peak: the score must first have risen above
    the pass's opening image, then stayed more than `threshold` below its best for
    `_FALLS_TO_STOP` images in a row.  Micro-Manager's JAF stops at the first image
    that falls that far, which far from focus -- where the curve is nearly flat and
    noise alone moves it by more than the threshold -- abandons a pass before it
    reaches the peak.  On a curve with 2% noise this finds a focus 6 um off in 96%
    of passes rather than 39%, for one image more on a clean one.
    """
    best_z, best_score = center, -float("inf")
    rose = False  # has any image beaten the first?
    falls = 0  # images in a row since the best that fell past the threshold
    for i in range(2 * n_steps + 1):
        if should_cancel is not None and should_cancel():
            raise AutofocusCancelled(
                f"Autofocus cancelled after {len(samples)} image(s)."
            )
        z = center - step_um * n_steps + i * step_um
        score = measure(z)
        samples.append((z, score))
        if score > best_score:
            rose = rose or i > 0
            best_z, best_score, falls = z, score, 0
        elif best_score > 0 and best_score - score > threshold * best_score:
            falls += 1
            if rose and falls >= _FALLS_TO_STOP and not full_scan:
                # risen to a peak and clearly falling: the rest cannot win
                break
        else:
            falls = 0
    return best_z


def duo(
    core: CMMCorePlus,
    focus_device: str,
    settings: Mapping[str, Any] | DuoSettings | None = None,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> AutofocusResult:
    """Run two autofocus routines in sequence, each starting where the last ended.

    Usually a coarse method over a wide range followed by a precise one over a narrow
    range, which together find focus from further out than either manages alone.
    """
    cfg = from_dict(DuoSettings, settings)
    z_before = core.getPosition(focus_device)
    samples: list[tuple[float, float]] = []
    messages: list[str] = []

    def failed(message: str) -> AutofocusResult:
        return AutofocusResult(
            kind="software",
            method="duo",
            focus_device=focus_device,
            z_before=z_before,
            z_after=z_before,
            succeeded=False,
            message=message,
            scores=tuple(samples),
            n_images=len(samples),
        )

    # Check both steps before anything moves: a misspelt second routine, or
    # settings it rejects, must not come to light only after the first has
    # already taken the stage somewhere else. Like a misspelt setting of any
    # routine, it is a mistake in the sequence, and raises.
    steps = [_duo_step(label, step) for label, step in _duo_steps(cfg)]

    # Each routine restores its own start when it fails -- but its start is where
    # the previous step left the stage, so the chain as a whole has to put back
    # where *it* started, however a step ends.
    try:
        for name, entry, step_settings in steps:
            result = entry.run(
                core, focus_device, step_settings, should_cancel=should_cancel
            )
            samples.extend(result.scores)
            if not result.succeeded:
                _restore(core, focus_device, z_before)
                return failed(f"{name}: {result.message}")
            if result.message:
                messages.append(f"{name}: {result.message}")
    except Exception as e:
        _restore(core, focus_device, z_before)
        if isinstance(e, AutofocusCancelled):
            raise
        logger.warning("Software autofocus (duo) failed. %s", e)
        return failed(str(e))

    return AutofocusResult(
        kind="software",
        method="duo",
        focus_device=focus_device,
        z_before=z_before,
        z_after=core.getPosition(focus_device),
        succeeded=True,
        message="; ".join(messages),
        scores=tuple(samples),
        n_images=len(samples),
    )


def _duo_steps(cfg: DuoSettings) -> tuple[tuple[str, dict[str, Any]], ...]:
    return (("first", cfg.first), ("second", cfg.second))


def _duo_step(label: str, step: Mapping[str, Any]) -> tuple[str, Any, Any]:
    """Resolve one `duo` step to its routine and validated settings, or raise."""
    from ._registry import get_method

    name = str(step["method"])
    try:
        entry = get_method(name)
    except KeyError as e:
        raise KeyError(f"The {label} step: {e.args[0]}") from None
    step_settings = step.get("settings")
    # a routine registered with some other kind of settings is left to check its own
    if dataclasses.is_dataclass(entry.settings_model):
        try:
            step_settings = from_dict(entry.settings_model, step_settings)
        except (TypeError, ValueError) as e:
            raise ValueError(f"The {label} step ({name}): {e}") from None
    return name, entry, step_settings


def _restore(core: CMMCorePlus, focus_device: str, z: float) -> None:
    """Put the focus device back; a failed autofocus must not move the sample."""
    try:
        _move_to(core, focus_device, z)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to restore Z to %r after autofocus. %s", z, e)
