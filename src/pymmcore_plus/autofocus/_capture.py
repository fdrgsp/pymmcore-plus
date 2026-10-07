"""Temporary camera state for a focus run, restored exactly once.

A focus run usually wants different settings from the acquisition around it: a
brightfield channel that always has contrast, a short exposure because the images are
thrown away, a central crop so scoring is quick, and the shutter held open so it is
not cycled once per image.  All of that has to go back afterwards, including when the
run fails partway through -- otherwise the acquisition continues with the autofocus
channel or exposure still applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_plus._logger import logger

if TYPE_CHECKING:
    from pymmcore_plus.core import CMMCorePlus

__all__ = ["CaptureSettings", "capture_state"]


@dataclass(frozen=True)
class CaptureSettings:
    """Camera state to use while focusing.  Every field defaults to "leave it alone".

    Attributes
    ----------
    channel_group : str | None
        Config group of `channel`.  If `None`, the core's current channel group.
    channel : str | None
        Config preset to apply while focusing, e.g. a brightfield channel.  `None`
        keeps whatever is set.
    exposure_ms : float | None
        Exposure to use while focusing.  `None` keeps the current exposure, which is
        usually what you want inside an MDA, where the channel has already set it.
    crop_factor : float
        Fraction of the frame's width and height to keep, centred, as a camera ROI.
        A smaller frame is faster to acquire and to score.  1.0 (the default) uses the
        whole frame.
    keep_shutter_open : bool
        Hold the shutter open for the whole run instead of letting autoshutter cycle
        it per image.  Faster, and easier on a mechanical shutter, at the cost of
        more light on the sample.
    """

    channel_group: str | None = None
    channel: str | None = None
    exposure_ms: float | None = None
    crop_factor: float = 1.0
    keep_shutter_open: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.crop_factor <= 1.0:
            raise ValueError(
                f"crop_factor must be within (0, 1], got {self.crop_factor}."
            )
        if self.exposure_ms is not None and not self.exposure_ms > 0:
            raise ValueError(f"exposure_ms must be positive, got {self.exposure_ms}.")


def _centered_roi(roi: tuple[int, int, int, int], factor: float) -> list[int]:
    x, y, width, height = roi
    w, h = int(width * factor), int(height * factor)
    # keep at least a few pixels, or the 3x3 filters have nothing to work on
    w, h = max(w, 8), max(h, 8)
    return [x + (width - w) // 2, y + (height - h) // 2, w, h]


class capture_state:
    """Context manager applying `settings` to `core`, then putting everything back.

    Restoration runs on the way out whatever happened, and keeps going across
    individual failures so that one stuck device cannot strand the rest of the state.
    Anything that could not be restored is logged.

    Examples
    --------
    ```python
    with capture_state(core, CaptureSettings(channel="BF", exposure_ms=10)):
        ...  # acquire and score images
    ```
    """

    def __init__(self, core: CMMCorePlus, settings: CaptureSettings) -> None:
        self._core = core
        self._settings = settings
        self._restore: list[tuple[str, Any]] = []

    def __enter__(self) -> capture_state:
        try:
            self._apply()
        except Exception:
            # applying is itself several device calls; if a later one fails, undo the
            # earlier ones rather than leaving the system half-configured
            self._restore_all()
            raise
        return self

    def _apply(self) -> None:
        core, settings = self._core, self._settings
        camera = core.getCameraDevice()

        if settings.channel is not None:
            group = settings.channel_group or core.getChannelGroup()
            if group:
                # remember the whole group's state: a preset may touch several devices
                self._restore.append(("system_state", core.getConfigGroupState(group)))
                core.setConfig(group, settings.channel)
                core.waitForConfig(group, settings.channel)
            else:  # pragma: no cover
                logger.warning(
                    "No channel group available; ignoring autofocus channel %r.",
                    settings.channel,
                )

        if settings.exposure_ms is not None:
            self._restore.append(("exposure", core.getExposure()))
            core.setExposure(settings.exposure_ms)

        if settings.crop_factor < 1.0:
            if camera:
                roi = tuple(core.getROI())
                self._restore.append(("roi", roi))
                core.setROI(*_centered_roi(roi, settings.crop_factor))  # type: ignore[arg-type]
                core.waitForDevice(camera)
            else:  # pragma: no cover
                logger.warning("No camera device; ignoring autofocus crop_factor.")

        if settings.keep_shutter_open:
            self._restore.append(("autoshutter", core.getAutoShutter()))
            self._restore.append(("shutter_open", core.getShutterOpen()))
            core.setAutoShutter(False)
            core.setShutterOpen(True)

    def __exit__(self, *exc_info: object) -> None:
        self._restore_all()

    def _restore_all(self) -> None:
        core = self._core
        camera = core.getCameraDevice()
        # unwind in reverse, so the shutter closes before the channel changes back
        for kind, value in reversed(self._restore):
            try:
                if kind == "system_state":
                    core.setSystemState(value)
                elif kind == "exposure":
                    core.setExposure(value)
                elif kind == "roi":
                    core.setROI(*value)
                    if camera:
                        core.waitForDevice(camera)
                elif kind == "autoshutter":
                    core.setAutoShutter(value)
                elif kind == "shutter_open":
                    core.setShutterOpen(value)
            except Exception as e:
                logger.warning("Failed to restore %s after autofocus. %s", kind, e)
        self._restore.clear()
