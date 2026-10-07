"""Focus scores: every method must peak on the sharpest image."""

from __future__ import annotations

import numpy as np
import pytest

from pymmcore_plus.autofocus._fft import bandpass_power, log_power_spectrum
from pymmcore_plus.autofocus._scoring import ScoringMethod, jaf_score, score_image

# MEAN measures brightness, not sharpness: it only tracks focus on a sample whose
# contrast changes with Z (brightfield).  It is excluded from the sharpness tests.
SHARPNESS_METHODS = [m for m in ScoringMethod if m is not ScoringMethod.MEAN]


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, standing in for defocus."""
    if sigma <= 0:
        return img
    radius = max(int(3 * sigma), 1)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, "same"), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, "same"), 1, out)


def _sample(sigma: float, size: int = 64, seed: int = 0) -> np.ndarray:
    """A defocused image of a textured sample.

    Band-limited noise, so features are a few pixels across, as real sample detail is.
    `sigma` is how far out of focus it is; 0 is in focus.  A background offset is
    added, as every camera has one.
    """
    rng = np.random.default_rng(seed)
    base = _blur(rng.normal(0.0, 1.0, (size, size)), 1.2)
    base = 400.0 * base / base.std()
    return _blur(base, sigma) + 500.0


@pytest.mark.parametrize("method", SHARPNESS_METHODS, ids=str)
def test_score_is_highest_in_focus(method: ScoringMethod) -> None:
    """The defining property: the in-focus image must win."""
    sigmas = [0.0, 0.5, 1.0, 2.0, 4.0]
    scores = [score_image(_sample(s), method) for s in sigmas]
    assert scores[0] == max(scores), dict(zip(sigmas, scores, strict=True))


@pytest.mark.parametrize("method", SHARPNESS_METHODS, ids=str)
def test_score_falls_off_monotonically(method: ScoringMethod) -> None:
    """A focus curve an optimizer can climb: no local maxima on the way out."""
    scores = [score_image(_sample(s), method) for s in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert scores == sorted(scores, reverse=True), scores


def test_jaf_score_is_highest_in_focus() -> None:
    scores = [jaf_score(_sample(s)) for s in (0.0, 1.0, 2.0, 4.0)]
    assert scores[0] == max(scores)


@pytest.mark.parametrize("method", list(ScoringMethod), ids=str)
def test_every_method_returns_a_finite_float(method: ScoringMethod) -> None:
    img = _sample(1.0)
    score = score_image(img, method)
    assert isinstance(score, float)
    assert np.isfinite(score)


@pytest.mark.parametrize("method", list(ScoringMethod), ids=str)
def test_methods_survive_a_blank_image(method: ScoringMethod) -> None:
    """A closed shutter or a dead camera must not raise or return NaN."""
    score = score_image(np.zeros((16, 16)), method)
    assert np.isfinite(score)


def test_method_accepts_its_string_name() -> None:
    img = _sample(1.0)
    assert score_image(img, "tenengrad") == score_image(img, ScoringMethod.TENENGRAD)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid ScoringMethod"):
        score_image(np.zeros((8, 8)), "nope")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_scores_any_pixel_type(dtype: np.dtype) -> None:
    img = _sample(1.0).astype(dtype)
    assert np.isfinite(score_image(img, ScoringMethod.EDGES))


def test_normalized_scores_ignore_exposure() -> None:
    """Doubling the exposure must not change a score that divides by the mean."""
    img = _sample(1.0)
    # mean(gradient)/mean and std/mean are both ratios of linear quantities, so a
    # gain change cancels; variance is quadratic, so it survives as a single factor.
    for method, scale in (
        (ScoringMethod.EDGES, 1.0),
        (ScoringMethod.NORMALIZED_STD_DEV, 1.0),
        (ScoringMethod.NORMALIZED_VARIANCE, 2.0),
    ):
        one = score_image(img, method)
        two = score_image(img * 2.0, method)
        assert two == pytest.approx(one * scale, rel=1e-9)


def test_median_edges_tolerates_hot_pixels_better_than_tenengrad() -> None:
    """Why MEDIAN_EDGES exists: single-pixel noise must not read as sharpness."""
    img = _sample(2.0)
    noisy = img.copy()
    rng = np.random.default_rng(7)
    ys = rng.integers(0, img.shape[0], 40)
    xs = rng.integers(0, img.shape[1], 40)
    noisy[ys, xs] = 5000.0

    def relative_change(method: ScoringMethod) -> float:
        clean = score_image(img, method)
        return abs(score_image(noisy, method) - clean) / clean

    assert relative_change(ScoringMethod.MEDIAN_EDGES) < relative_change(
        ScoringMethod.TENENGRAD
    )


def test_jaf_score_only_looks_at_the_centre() -> None:
    img = _sample(1.0)
    edged = img.copy()
    edged[:4, :] = 5000.0  # bright band along the top edge
    assert jaf_score(img, 0.2) == pytest.approx(jaf_score(edged, 0.2))


def test_jaf_crop_ratio_is_validated() -> None:
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="crop_ratio"):
            jaf_score(np.zeros((8, 8)), bad)


# ---------------------------------- the FFT band ----------------------------------


def test_bandpass_rises_with_detail() -> None:
    assert bandpass_power(_sample(0.0)) > bandpass_power(_sample(4.0))


def test_log_power_spectrum_is_centred() -> None:
    """The zero frequency -- an image's total intensity -- belongs at the centre."""
    img = np.full((16, 16), 5.0)
    spectrum = log_power_spectrum(img)
    assert np.unravel_index(np.argmax(spectrum), spectrum.shape) == (8, 8)


def test_bandpass_rejects_an_inverted_band() -> None:
    with pytest.raises(ValueError, match="must be below"):
        bandpass_power(np.zeros((16, 16)), 20.0, 10.0)


@pytest.mark.parametrize("kwargs", [{"lower_pct": -1.0}, {"upper_pct": 101.0}])
def test_bandpass_rejects_out_of_range_cutoffs(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="within"):
        bandpass_power(np.zeros((16, 16)), **kwargs)


def test_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        score_image(np.zeros((0, 4)), ScoringMethod.EDGES)


def test_median_edges_is_blind_to_single_pixel_detail() -> None:
    """A known limit of MEDIAN_EDGES, inherent to the median filter it is named for.

    Detail only one pixel across is removed along with the noise, so a sample of
    isolated points scores zero however sharp it is.  Real sample detail spans several
    pixels at any usable magnification, but it is why this method should not be the
    default.
    """
    isolated = np.zeros((32, 32))
    isolated[::4, ::4] = 1000.0
    assert score_image(isolated, ScoringMethod.MEDIAN_EDGES) == 0.0
    # a gradient score that does not median-filter still sees them
    assert score_image(isolated, ScoringMethod.TENENGRAD) > 0.0
