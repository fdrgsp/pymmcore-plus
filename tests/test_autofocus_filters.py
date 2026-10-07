"""The image filters the focus scores are built from."""

from __future__ import annotations

import numpy as np
import pytest

from pymmcore_plus.autofocus._filters import (
    convolve3x3,
    median_3x3,
    sharpen,
    sobel_magnitude,
)

IDENTITY = (0, 0, 0, 0, 1, 0, 0, 0, 0)
BOX = (1, 1, 1, 1, 1, 1, 1, 1, 1)
GRADIENT_X = (0, 0, 0, -1, 0, 1, 0, 0, 0)


def test_identity_kernel_returns_the_image() -> None:
    img = np.arange(12, dtype=np.uint16).reshape(3, 4)
    np.testing.assert_allclose(convolve3x3(img, IDENTITY), img)


def test_box_kernel_averages_and_preserves_a_constant() -> None:
    """A normalized kernel must not change a flat image, including at the border."""
    flat = np.full((5, 5), 7.0)
    np.testing.assert_allclose(convolve3x3(flat, BOX), flat)


def test_gradient_kernel_is_signed() -> None:
    """A kernel summing to zero is not normalized, and keeps negative responses."""
    ramp = np.tile(np.arange(5.0), (3, 1))  # increases left to right by 1
    out = convolve3x3(ramp, GRADIENT_X)
    # interior: (x+1) - (x-1) == 2
    np.testing.assert_allclose(out[:, 1:-1], 2.0)
    # a decreasing ramp gives the negative of that, rather than being clipped to 0
    np.testing.assert_allclose(convolve3x3(ramp[:, ::-1], GRADIENT_X)[:, 1:-1], -2.0)


def test_border_repeats_edge_pixels() -> None:
    """With the edge pixel repeated, a horizontal ramp has half the gradient there."""
    ramp = np.tile(np.arange(5.0), (3, 1))
    out = convolve3x3(ramp, GRADIENT_X)
    np.testing.assert_allclose(out[:, 0], 1.0)
    np.testing.assert_allclose(out[:, -1], 1.0)


def test_sobel_magnitude_is_zero_on_a_flat_image() -> None:
    assert not sobel_magnitude(np.full((6, 6), 3.0)).any()


def test_sobel_magnitude_grows_with_contrast() -> None:
    edge = np.zeros((6, 6))
    edge[:, 3:] = 1.0
    assert sobel_magnitude(edge).sum() < sobel_magnitude(edge * 10).sum()


def test_sobel_magnitude_is_rotation_symmetric_for_a_transposed_edge() -> None:
    edge = np.zeros((7, 7))
    edge[:, 4:] = 5.0
    np.testing.assert_allclose(
        sobel_magnitude(edge).T, sobel_magnitude(edge.T), atol=1e-12
    )


def test_sharpen_preserves_a_constant_image() -> None:
    flat = np.full((5, 5), 4.0)
    np.testing.assert_allclose(sharpen(flat), flat)


def test_sharpen_increases_edge_contrast() -> None:
    edge = np.zeros((7, 7))
    edge[:, 4:] = 1.0
    out = sharpen(edge)
    assert out[3, 3] < edge[3, 3]  # dark side pushed darker
    assert out[3, 4] > edge[3, 4]  # bright side pushed brighter


def test_median_removes_a_single_hot_pixel() -> None:
    """The reason the median is there: one bright pixel must not read as sharpness."""
    img = np.full((5, 5), 10.0)
    img[2, 2] = 1000.0
    out = median_3x3(img)
    assert out[2, 2] == 10.0
    np.testing.assert_allclose(out, 10.0)
    # ... and the hot pixel would otherwise dominate an edge score
    assert sobel_magnitude(img).sum() > 50 * sobel_magnitude(out).sum()


def test_median_keeps_a_real_edge() -> None:
    img = np.zeros((6, 6))
    img[:, 3:] = 1.0
    np.testing.assert_allclose(median_3x3(img), img)


def test_median_filters_the_border_too() -> None:
    img = np.full((4, 4), 2.0)
    img[0, 0] = 99.0
    assert median_3x3(img)[0, 0] == 2.0


def test_median_of_a_known_neighborhood() -> None:
    img = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert median_3x3(img)[1, 1] == 5.0


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64]
)
def test_accepts_any_numeric_pixel_type(dtype: np.dtype) -> None:
    """Cameras deliver 8-bit, 16-bit and (after processing) float images."""
    img = (np.arange(16) % 7).reshape(4, 4).astype(dtype)
    for out in (sobel_magnitude(img), median_3x3(img), sharpen(img)):
        assert out.dtype == np.float64
        assert out.shape == img.shape


def test_rejects_non_2d_images() -> None:
    with pytest.raises(ValueError, match="2D image"):
        sobel_magnitude(np.zeros((2, 2, 3)))


def test_rejects_wrong_kernel_size() -> None:
    with pytest.raises(ValueError, match="9 values"):
        convolve3x3(np.zeros((3, 3)), (1, 2, 3))
