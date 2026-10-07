"""The small image filters that the focus scores are built from.

A focus score is a single number that is *larger* when an image is sharper; only the
position of its maximum matters, never its absolute value.  These filters are the
pieces the scores in `_scoring.py` combine.

Everything works in `float64`, regardless of the camera's pixel type, and nothing is
rounded or clipped along the way.  Micro-Manager's Java implementations write each
intermediate result back into an 8- or 16-bit image, which rounds it and clips
negative values to zero -- so a gradient kernel there keeps only the responses of one
sign and discards the rest.  That changes a score's value but barely moves its peak,
and working in floating point keeps the gradient information the metric is supposed to
measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = ["convolve3x3", "median_3x3", "sharpen", "sobel_magnitude"]

# 3x3 Sobel kernels, as used for edge-based focus scores
_SOBEL_Y = (1, 2, 1, 0, 0, 0, -1, -2, -1)
_SOBEL_X = (1, 0, -1, 2, 0, -2, 1, 0, -1)
# A sharpening kernel: the identity plus a Laplacian, normalized by its sum of 4
_SHARPEN = (-1, -1, -1, -1, 12, -1, -1, -1, -1)


def _as_2d_float(image: ArrayLike) -> NDArray[np.float64]:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")
    return img


def _neighborhoods(image: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """Return the nine shifted copies that make up every 3x3 neighborhood.

    The image is padded by repeating its edge pixels, so the result is the same shape
    as the input and border pixels are filtered like any other.
    """
    padded = np.pad(image, 1, mode="edge")
    h, w = image.shape
    return tuple(
        padded[dy : dy + h, dx : dx + w] for dy in (0, 1, 2) for dx in (0, 1, 2)
    )


def convolve3x3(image: ArrayLike, kernel: Sequence[float]) -> NDArray[np.float64]:
    """Convolve `image` with a 3x3 `kernel` given in row-major order.

    The result is divided by the sum of the kernel, unless that sum is zero (as it is
    for a gradient kernel, whose response is meant to be signed).
    """
    if len(kernel) != 9:
        raise ValueError(f"A 3x3 kernel needs 9 values, got {len(kernel)}.")
    img = _as_2d_float(image)
    out = np.zeros_like(img)
    for weight, neighbor in zip(kernel, _neighborhoods(img), strict=True):
        if weight:
            out += weight * neighbor
    if (scale := float(sum(kernel))) != 0.0:
        out /= scale
    return out


def sobel_magnitude(image: ArrayLike) -> NDArray[np.float64]:
    """Return the 3x3 Sobel gradient magnitude, `sqrt(gx**2 + gy**2)`."""
    img = _as_2d_float(image)
    gx = convolve3x3(img, _SOBEL_X)
    gy = convolve3x3(img, _SOBEL_Y)
    return np.sqrt(gx * gx + gy * gy)


def sharpen(image: ArrayLike) -> NDArray[np.float64]:
    """Sharpen `image` by adding a Laplacian to it."""
    return convolve3x3(image, _SHARPEN)


def median_3x3(image: ArrayLike) -> NDArray[np.float64]:
    """Replace each pixel with the median of its 3x3 neighborhood.

    This suppresses single-pixel noise -- which an edge or gradient score would
    otherwise read as sharpness -- without blurring real edges the way a mean would.
    """
    img = _as_2d_float(image)
    stacked = np.stack(_neighborhoods(img))
    median: NDArray[np.float64] = np.median(stacked, axis=0)
    return median
