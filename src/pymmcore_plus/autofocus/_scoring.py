"""Focus scores: how sharp a single image is.

Every function here returns a number that is **larger when the image is sharper**.
Only the position of the maximum over a Z sweep matters; the absolute values are not
comparable between methods, between cameras, or between samples.

The set of methods mirrors the ones Micro-Manager offers, so that a user who knows
which one works for their sample can keep using it.  No single method is best: edge
and gradient scores need real detail in the image, the intensity-based ones suit
brightfield where focus changes contrast, and the frequency-band score is the most
robust to noise but the slowest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from ._fft import bandpass_power
from ._filters import convolve3x3, median_3x3, sharpen, sobel_magnitude

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

__all__ = ["ScoringMethod", "jaf_score", "score_image"]

# the two diagonal kernels the "median edges" score combines
_DIAGONAL_NE = (2, 1, 0, 1, 0, -1, 0, -1, -2)
_DIAGONAL_NW = (0, 1, 2, -1, 0, 1, -2, -1, 0)
# A Laplacian whose centre weight sits below centre.  This is deliberate: it is how
# the method's source paper defines it, and it is reported to work better for
# brightfield than a symmetric Laplacian.
_REDONDO = (0, 1, 0, -3, 0, 1, 0, 1, 0)


class ScoringMethod(StrEnum):
    """How to measure the sharpness of an image.

    Attributes
    ----------
    EDGES : str
        Mean Sobel gradient, divided by mean intensity.  A good general default.
    SHARP_EDGES : str
        As `EDGES`, after sharpening; more sensitive, and noisier.
    MEAN : str
        Mean intensity.  Only useful where focus changes brightness, e.g. brightfield.
    NORMALIZED_STD_DEV : str
        Standard deviation over mean: contrast, independent of illumination level.
    NORMALIZED_VARIANCE : str
        Variance over mean; like the above but weights strong features more.
    REDONDO : str
        Sum of squared Laplacian responses.
    VOLATH : str
        Volath's autocorrelation: the difference between neighbour and
        next-neighbour correlation along x.
    VOLATH5 : str
        Volath's variant that suppresses noise by dropping one correlation term.
    MEDIAN_EDGES : str
        Diagonal gradients after a median filter.  The median makes this the most
        noise-tolerant of the gradient scores, but it also removes detail only one
        pixel across, so it is blind to a sample of isolated points.
    TENENGRAD : str
        Sum of squared Sobel gradients; reported as the best non-spectral metric for
        light-sheet data.
    FFT_BANDPASS : str
        Mean log power in a band of spatial frequencies, skipping the illumination
        profile at low frequencies and the sensor noise at high ones.
    """

    EDGES = "edges"
    SHARP_EDGES = "sharp_edges"
    MEAN = "mean"
    NORMALIZED_STD_DEV = "normalized_std_dev"
    NORMALIZED_VARIANCE = "normalized_variance"
    REDONDO = "redondo"
    VOLATH = "volath"
    VOLATH5 = "volath5"
    MEDIAN_EDGES = "median_edges"
    TENENGRAD = "tenengrad"
    FFT_BANDPASS = "fft_bandpass"

    def __str__(self) -> str:
        return self.value


def _as_2d_float(image: ArrayLike) -> NDArray[np.float64]:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")
    if img.size == 0:
        raise ValueError("Cannot score an empty image.")
    return img


def _per_mean(value: float, mean: float) -> float:
    """Normalize by mean intensity, which makes a score independent of exposure.

    A mean of zero means there is nothing in the image to focus on.
    """
    return 0.0 if mean == 0.0 else value / mean


def _edges(img: NDArray[np.float64]) -> float:
    return _per_mean(float(sobel_magnitude(img).mean()), float(img.mean()))


def _sharp_edges(img: NDArray[np.float64]) -> float:
    return _per_mean(float(sobel_magnitude(sharpen(img)).mean()), float(img.mean()))


def _normalized_std_dev(img: NDArray[np.float64]) -> float:
    return _per_mean(float(img.std(ddof=1)) if img.size > 1 else 0.0, float(img.mean()))


def _normalized_variance(img: NDArray[np.float64]) -> float:
    var = float(img.var(ddof=1)) if img.size > 1 else 0.0
    return _per_mean(var, float(img.mean()))


def _redondo(img: NDArray[np.float64]) -> float:
    response = convolve3x3(img, _REDONDO)
    # the border reads repeated pixels, so it is excluded from the sum
    interior = response[1:-1, 1:-1] if min(response.shape) > 2 else response
    return float((interior * interior).sum())


def _volath(img: NDArray[np.float64]) -> float:
    if img.shape[1] < 3:
        return 0.0
    neighbor = (img[:, :-1] * img[:, 1:]).sum()
    next_neighbor = (img[:, :-2] * img[:, 2:]).sum()
    return float(neighbor - next_neighbor)


def _volath5(img: NDArray[np.float64]) -> float:
    if img.shape[1] < 2:
        return 0.0
    h, w = img.shape
    neighbor = (img[:, :-1] * img[:, 1:]).sum()
    mean = img.mean()
    return float(neighbor - (w - 1) * h * mean * mean)


def _median_edges(img: NDArray[np.float64]) -> float:
    smoothed = median_3x3(img)
    a = convolve3x3(smoothed, _DIAGONAL_NE)
    b = convolve3x3(smoothed, _DIAGONAL_NW)
    return float(np.hypot(a, b).sum())


def _tenengrad(img: NDArray[np.float64]) -> float:
    gradient = sobel_magnitude(img)
    return float((gradient * gradient).sum())


_SCORES: dict[ScoringMethod, Callable[[NDArray[np.float64]], float]] = {
    ScoringMethod.EDGES: _edges,
    ScoringMethod.SHARP_EDGES: _sharp_edges,
    ScoringMethod.MEAN: lambda img: float(img.mean()),
    ScoringMethod.NORMALIZED_STD_DEV: _normalized_std_dev,
    ScoringMethod.NORMALIZED_VARIANCE: _normalized_variance,
    ScoringMethod.REDONDO: _redondo,
    ScoringMethod.VOLATH: _volath,
    ScoringMethod.VOLATH5: _volath5,
    ScoringMethod.MEDIAN_EDGES: _median_edges,
    ScoringMethod.TENENGRAD: _tenengrad,
}


def score_image(
    image: ArrayLike,
    method: ScoringMethod | str = ScoringMethod.EDGES,
    *,
    fft_lower_pct: float = 2.5,
    fft_upper_pct: float = 14.0,
) -> float:
    """Return a sharpness score for `image`; larger means sharper.

    Parameters
    ----------
    image : ArrayLike
        2D image, of any numeric pixel type.
    method : ScoringMethod | str
        Which score to compute.  By default, [`ScoringMethod.EDGES`][].
    fft_lower_pct : float
        For `FFT_BANDPASS` only: inner radius of the frequency band, as a percentage
        of the Nyquist frequency.  By default, 2.5.
    fft_upper_pct : float
        For `FFT_BANDPASS` only: outer radius, likewise.  By default, 14.0.

    Returns
    -------
    float
        The score.  Comparable only with other scores of the same method.
    """
    img = _as_2d_float(image)
    method = ScoringMethod(method)
    if method is ScoringMethod.FFT_BANDPASS:
        return bandpass_power(img, fft_lower_pct, fft_upper_pct)
    return _SCORES[method](img)


def jaf_score(image: ArrayLike, crop_ratio: float = 0.2) -> float:
    """Return the sharpness score used by Micro-Manager's JAF autofocus plugins.

    A median filter, then a diagonal gradient, then the sum of squares over the
    central `crop_ratio` of the frame.  Scoring only the centre keeps a bright or
    empty edge from dominating, and costs less than scoring the whole frame.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    crop_ratio : float
        Fraction of the frame's width and height to score, centred.  By default, 0.2.
    """
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError(f"crop_ratio must be within (0, 1], got {crop_ratio}.")
    img = _as_2d_float(image)
    # filter the whole frame, then sum over the centre: cropping first would put a
    # filter border inside the region being scored
    response = convolve3x3(median_3x3(img), _DIAGONAL_NE)
    h, w = img.shape
    ch, cw = max(int(h * crop_ratio), 1), max(int(w * crop_ratio), 1)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    centre = response[y0 : y0 + ch, x0 : x0 + cw]
    return float((centre * centre).sum())
