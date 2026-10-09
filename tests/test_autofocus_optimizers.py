"""Searching a focus curve for its peak."""

from __future__ import annotations

import numpy as np
import pytest

from pymmcore_plus.autofocus._optimizers import (
    AutofocusCancelled,
    brent_search,
    fit_peak,
    zstack_search,
)

TRUE_FOCUS = 12.3


def _curve(noise: float = 0.0, seed: int = 0, width: float = 4.0):
    """A Gaussian focus curve centred on TRUE_FOCUS, as a `measure` callable."""
    rng = np.random.default_rng(seed)

    def measure(z: float) -> float:
        peak = 1000.0 * np.exp(-((z - TRUE_FOCUS) ** 2) / (2 * width**2))
        return float(peak + 100.0 + (rng.normal(0, noise) if noise else 0.0))

    return measure


SEARCHES = [
    pytest.param(brent_search, {"tolerance_um": 0.2}, id="brent"),
    pytest.param(zstack_search, {"step_um": 1.0}, id="zstack"),
]


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_finds_the_peak(search, kwargs: dict) -> None:
    result = search(_curve(), center=10.0, search_range_um=30.0, **kwargs)
    assert result.z == pytest.approx(TRUE_FOCUS, abs=0.5)
    assert result.converged


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_finds_the_peak_despite_noise(search, kwargs: dict) -> None:
    result = search(_curve(noise=5.0), center=10.0, search_range_um=30.0, **kwargs)
    assert result.z == pytest.approx(TRUE_FOCUS, abs=1.5)


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_records_every_image_it_took(search, kwargs: dict) -> None:
    result = search(_curve(), center=10.0, search_range_um=20.0, **kwargs)
    assert result.n_images == len(result.samples) > 2
    # the samples are the focus curve, in the order measured
    assert all(isinstance(z, float) and isinstance(s, float) for z, s in result.samples)
    assert result.score == max(s for _, s in result.samples)


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_searches_only_within_the_requested_range(search, kwargs: dict) -> None:
    """The range is a safety limit: it must never be exceeded."""
    result = search(_curve(), center=10.0, search_range_um=8.0, **kwargs)
    for z, _ in result.samples:
        assert 10.0 - 4.0 - 1e-9 <= z <= 10.0 + 4.0 + 1e-9


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_cancellation_stops_the_search(search, kwargs: dict) -> None:
    calls = 0

    def measure(z: float) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    with pytest.raises(AutofocusCancelled, match="cancelled"):
        search(
            measure,
            center=10.0,
            search_range_um=30.0,
            should_cancel=lambda: calls >= 3,
            **kwargs,
        )
    assert calls == 3


@pytest.mark.parametrize(("search", "kwargs"), SEARCHES)
def test_rejects_a_non_positive_range(search, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="search_range_um"):
        search(_curve(), center=0.0, search_range_um=0.0, **kwargs)


def test_brent_uses_far_fewer_images_than_a_scan() -> None:
    """The reason to pick Brent: each image tells it where to look next."""
    brent = brent_search(_curve(), 10.0, 30.0, tolerance_um=0.5)
    scan = zstack_search(_curve(), 10.0, 30.0, step_um=0.5)
    assert brent.n_images < scan.n_images / 3


def test_brent_reports_when_it_runs_out_of_images() -> None:
    result = brent_search(_curve(), 10.0, 30.0, tolerance_um=1e-9, max_evaluations=5)
    assert not result.converged
    assert "Did not converge" in result.message
    assert result.n_images <= 5


def test_brent_rejects_a_non_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance_um"):
        brent_search(_curve(), 10.0, 10.0, tolerance_um=0.0)


def test_zstack_rejects_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step_um"):
        zstack_search(_curve(), 10.0, 10.0, step_um=0.0)


def test_zstack_scans_symmetrically_including_both_ends() -> None:
    result = zstack_search(_curve(), center=10.0, search_range_um=10.0, step_um=2.5)
    zs = [z for z, _ in result.samples]
    assert zs == pytest.approx([5.0, 7.5, 10.0, 12.5, 15.0])


@pytest.mark.parametrize(
    ("search_range_um", "step_um"),
    [(10, 6), (10, 4), (10, 3), (20, 7), (0.3, 0.1), (10, 25)],
)
def test_zstack_never_leaves_its_range_whatever_the_step(
    search_range_um: float, step_um: float
) -> None:
    """The range is a travel limit, not a suggestion.

    Stepping out from the lower end used to land past the upper one whenever the
    step did not divide the range -- 10 um at 6 um steps visited -5, 1 and 7 -- or
    to stop short of it.
    """
    result = zstack_search(
        _curve(), center=0.0, search_range_um=search_range_um, step_um=step_um
    )
    zs = np.array([z for z, _ in result.samples])
    half = search_range_um / 2
    # both ends, and nothing beyond them
    assert zs.min() == pytest.approx(-half)
    assert zs.max() == pytest.approx(half)
    # evenly spaced, and never further apart than asked for
    gaps = np.diff(zs)
    assert gaps == pytest.approx(np.full_like(gaps, gaps[0]))
    assert gaps.max() <= step_um + 1e-9


def test_zstack_shrinks_the_step_rather_than_overshooting() -> None:
    result = zstack_search(_curve(), center=0.0, search_range_um=10.0, step_um=6.0)
    assert [z for z, _ in result.samples] == pytest.approx([-5.0, 0.0, 5.0])


def test_zstack_beats_its_step_size() -> None:
    """Fitting the curve locates the peak to better than the sampling."""
    result = zstack_search(_curve(), center=10.0, search_range_um=30.0, step_um=3.0)
    sampled = [z for z, _ in result.samples]
    assert result.z not in sampled
    assert abs(result.z - TRUE_FOCUS) < 3.0 / 2


# ----------------------------------- fit_peak -----------------------------------


def test_fit_peak_interpolates_between_samples() -> None:
    z = np.array([0.0, 1.0, 2.0])
    scores = np.exp(-((z - 1.2) ** 2) / 2.0)
    assert fit_peak(z, scores) == pytest.approx(1.2, abs=1e-6)


def test_fit_peak_falls_back_at_the_edge_of_the_scan() -> None:
    """With the peak at an end there is nothing to interpolate between."""
    z = np.array([0.0, 1.0, 2.0])
    assert fit_peak(z, np.array([9.0, 2.0, 1.0])) == 0.0
    assert fit_peak(z, np.array([1.0, 2.0, 9.0])) == 2.0


def test_fit_peak_handles_too_few_points() -> None:
    assert fit_peak(np.array([3.0]), np.array([1.0])) == 3.0
    assert fit_peak(np.array([3.0, 4.0]), np.array([1.0, 5.0])) == 4.0


def test_fit_peak_handles_unsorted_input() -> None:
    z = np.array([2.0, 0.0, 1.0])
    scores = np.exp(-((z - 1.2) ** 2) / 2.0)
    assert fit_peak(z, scores) == pytest.approx(1.2, abs=1e-6)


def test_fit_peak_survives_a_flat_curve() -> None:
    z = np.array([0.0, 1.0, 2.0])
    assert fit_peak(z, np.zeros(3)) in set(z)


def test_fit_peak_survives_negative_scores() -> None:
    """Some scores sit below zero; the log fit must shift rather than fail."""
    z = np.array([0.0, 1.0, 2.0])
    result = fit_peak(z, np.array([-10.0, -1.0, -8.0]))
    assert 0.0 <= result <= 2.0
