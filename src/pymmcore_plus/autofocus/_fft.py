"""Focus scoring from the power in a band of spatial frequencies.

A sharp image carries more energy at high spatial frequencies than a blurred one, so
the power in a mid-frequency band rises and falls with focus.  Restricting to a band
is what makes this useful: the lowest frequencies carry the illumination profile
rather than detail, and the highest carry mostly sensor noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["bandpass_power", "log_power_spectrum"]


def log_power_spectrum(image: ArrayLike) -> NDArray[np.float64]:
    """Return `log(1 + power)` of `image`, with the zero frequency at the centre.

    The transform is unnormalized, so the result scales with image size; a focus score
    only ever compares images of one size.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    # log1p keeps this finite at zero power, where a bare log would diverge
    power: NDArray[np.float64] = np.log1p(np.abs(spectrum) ** 2)
    return power


def bandpass_power(
    image: ArrayLike, lower_pct: float = 2.5, upper_pct: float = 14.0
) -> float:
    """Return the mean log power in an annulus of spatial frequencies.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    lower_pct : float
        Inner radius of the annulus, as a percentage of the Nyquist frequency.
        Frequencies below it are excluded.  By default, 2.5.
    upper_pct : float
        Outer radius, likewise.  Frequencies above it are excluded.  By default, 14.0.

    Returns
    -------
    float
        Mean of `log(1 + power)` over the annulus, or 0.0 if the band is empty (which
        happens when the two radii round to the same pixel on a small image).
    """
    if not 0.0 <= lower_pct <= 100.0:
        raise ValueError(f"lower_pct must be within [0, 100], got {lower_pct}.")
    if not 0.0 <= upper_pct <= 100.0:
        raise ValueError(f"upper_pct must be within [0, 100], got {upper_pct}.")
    if lower_pct >= upper_pct:
        raise ValueError(
            f"lower_pct ({lower_pct}) must be below upper_pct ({upper_pct})."
        )

    power = log_power_spectrum(image)
    h, w = power.shape
    # distance of every pixel from the zero frequency, as a fraction of Nyquist
    cy, cx = h / 2.0, w / 2.0
    yy = (np.arange(h) - cy) / max(cy, 1.0)
    xx = (np.arange(w) - cx) / max(cx, 1.0)
    radius = np.hypot(yy[:, None], xx[None, :]) * 100.0

    band = (radius >= lower_pct) & (radius <= upper_pct)
    if not band.any():
        return 0.0
    return float(power[band].mean())
