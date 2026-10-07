"""Settings for each software autofocus routine.

These are plain dataclasses, so a settings dict carried by a
[`useq.SoftwareAutofocus`][] action round-trips through JSON, and a GUI can build a
form from `dataclasses.fields()`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast

from ._capture import CaptureSettings
from ._scoring import ScoringMethod

__all__ = ["DuoSettings", "JAFSettings", "OughtaFocusSettings", "from_dict"]

_T = TypeVar("_T")


def from_dict(model: type[_T], settings: Mapping[str, Any] | _T | None) -> _T:
    """Build `model` from a settings mapping, rejecting keys it does not define.

    An unknown key is far more likely to be a typo than a feature, and silently
    ignoring it would leave autofocus running with settings the user did not choose.
    """
    if settings is None:
        return model()
    if isinstance(settings, model):  # already built
        return settings
    if not isinstance(settings, Mapping):  # pragma: no cover
        raise TypeError(
            f"Settings for {model.__name__} must be a mapping, got {type(settings)}."
        )
    known = {f.name for f in dataclasses.fields(cast("Any", model))}
    if unknown := set(settings) - known:
        raise ValueError(
            f"Unknown setting(s) for {model.__name__}: {sorted(unknown)}. "
            f"Valid settings are: {sorted(known)}."
        )
    return model(**settings)


@dataclass(frozen=True)
class OughtaFocusSettings:
    """Settings for the general-purpose image-based autofocus.

    Attributes
    ----------
    search_range_um : float
        Total Z range to search, centred on the current position.  By default, 10.0.
    optimizer : Literal["brent", "zstack"]
        `"brent"` walks toward the peak and takes the fewest images; `"zstack"` scans
        the whole range at even spacing, which costs more images but survives noise
        and reports the full focus curve.  By default, `"brent"`.
    tolerance_um : float
        For `"brent"`: stop once the peak is bracketed this tightly.  For `"zstack"`:
        the spacing between images.  By default, 1.0.
    scoring : ScoringMethod
        How to measure sharpness.  By default, `ScoringMethod.EDGES`.
    fft_lower_pct, fft_upper_pct : float
        The frequency band, for `ScoringMethod.FFT_BANDPASS` only.
    channel, channel_group : str | None
        A config preset to focus in, e.g. a brightfield channel.  `None` uses
        whatever is currently set.
    exposure_ms : float | None
        Exposure while focusing.  `None` keeps the current exposure.
    crop_factor : float
        Fraction of the frame to score, centred.  By default, 1.0 (the whole frame).
    keep_shutter_open : bool
        Hold the shutter open for the run rather than cycling it per image.
    settle_ms : float
        Wait this long after each Z move before acquiring.  By default, 0.0.
    """

    search_range_um: float = 10.0
    optimizer: Literal["brent", "zstack"] = "brent"
    tolerance_um: float = 1.0
    scoring: ScoringMethod = ScoringMethod.EDGES
    fft_lower_pct: float = 2.5
    fft_upper_pct: float = 14.0
    channel: str | None = None
    channel_group: str | None = None
    exposure_ms: float | None = None
    crop_factor: float = 1.0
    keep_shutter_open: bool = False
    settle_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.search_range_um > 0:
            raise ValueError(
                f"search_range_um must be positive, got {self.search_range_um}."
            )
        if not self.tolerance_um > 0:
            raise ValueError(f"tolerance_um must be positive, got {self.tolerance_um}.")
        if self.optimizer not in ("brent", "zstack"):
            raise ValueError(
                f"optimizer must be 'brent' or 'zstack', got {self.optimizer!r}."
            )
        if self.settle_ms < 0:
            raise ValueError(f"settle_ms cannot be negative, got {self.settle_ms}.")
        object.__setattr__(self, "scoring", ScoringMethod(self.scoring))
        self.capture()  # validate crop_factor / exposure_ms now, not mid-run

    def capture(self) -> CaptureSettings:
        """The camera state to hold while focusing."""
        return CaptureSettings(
            channel_group=self.channel_group,
            channel=self.channel,
            exposure_ms=self.exposure_ms,
            crop_factor=self.crop_factor,
            keep_shutter_open=self.keep_shutter_open,
        )


@dataclass(frozen=True)
class JAFSettings:
    """Settings for the two-pass coarse/fine autofocus.

    A coarse scan locates the peak roughly, then a fine scan refines it.  Each pass
    stops early once the score has fallen well below the best seen, which saves images
    on the far side of the peak.

    Attributes
    ----------
    coarse_step_um, coarse_steps : float, int
        The coarse pass covers `+/- coarse_step_um * coarse_steps` around the start.
        By default, 2.0 and 1.
    fine_step_um, fine_steps : float, int
        The fine pass covers `+/- fine_step_um * fine_steps` around the coarse best.
        By default, 0.2 and 5.
    threshold : float
        Stop a pass once the score has dropped by more than this fraction of the best
        seen.  By default, 0.02.
    crop_ratio : float
        Fraction of the frame to score, centred.  By default, 0.2.
    channel, channel_group : str | None
        A config preset to focus in.  `None` keeps the current one.
    fine_channel : str | None
        A different preset for the fine pass, e.g. a fluorescence channel once the
        coarse pass has found the sample in brightfield.  `None` keeps using
        `channel`.
    exposure_ms : float | None
        Exposure while focusing.  `None` keeps the current exposure.
    settle_ms : float
        Wait this long after each Z move before acquiring.  By default, 100.0.
    """

    coarse_step_um: float = 2.0
    coarse_steps: int = 1
    fine_step_um: float = 0.2
    fine_steps: int = 5
    threshold: float = 0.02
    crop_ratio: float = 0.2
    channel: str | None = None
    channel_group: str | None = None
    fine_channel: str | None = None
    exposure_ms: float | None = None
    settle_ms: float = 100.0

    def __post_init__(self) -> None:
        for name in ("coarse_step_um", "fine_step_um"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        for name in ("coarse_steps", "fine_steps"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}."
                )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}.")
        if not 0.0 < self.crop_ratio <= 1.0:
            raise ValueError(
                f"crop_ratio must be within (0, 1], got {self.crop_ratio}."
            )
        if self.settle_ms < 0:
            raise ValueError(f"settle_ms cannot be negative, got {self.settle_ms}.")

    def capture(self, channel: str | None) -> CaptureSettings:
        """The camera state to hold during one pass."""
        return CaptureSettings(
            channel_group=self.channel_group,
            channel=channel,
            exposure_ms=self.exposure_ms,
        )


@dataclass(frozen=True)
class DuoSettings:
    """Run two autofocus routines in sequence.

    Typically a coarse method with a wide range, then a precise one with a narrow
    range -- which finds focus from further away than either could alone.

    Attributes
    ----------
    first, second : dict
        Each is `{"method": <name>, "settings": {...}}`.
    """

    first: dict[str, Any] = field(default_factory=dict)
    second: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("first", "second"):
            step = getattr(self, name)
            if not isinstance(step, dict) or not step.get("method"):
                raise ValueError(
                    f"`{name}` must be a dict naming a method, "
                    f'e.g. {{"method": "oughtafocus", "settings": {{}}}}; got {step!r}.'
                )
