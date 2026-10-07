"""Searching a focus curve for its peak.

Both strategies take a `measure(z) -> score` callable -- in practice, "move the focus
device to z, acquire an image, and score it" -- so they can be reasoned about and
tested without any hardware.

Which to use is a trade-off in the number of images, since an image is by far the most
expensive thing here:

* [`brent_search`][pymmcore_plus.autofocus.brent_search] takes the fewest images,
  because each one tells it where to look next. It needs the focus curve to have a
  single peak within the search range; noise or a second peak can send it to the wrong
  place, and it reports no curve to inspect afterwards.
* [`zstack_search`][pymmcore_plus.autofocus.zstack_search] takes a fixed number of
  images at even spacing and fits the peak. It costs more images, but it sees the whole
  curve, survives noise and local bumps, and hands back the samples to plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "AutofocusCancelled",
    "FocusSearchResult",
    "brent_search",
    "fit_peak",
    "zstack_search",
]

# Brent's method needs a relative tolerance as well as an absolute one, because it is
# written for arbitrary floating point values. Stage positions behave like fixed point
# over their useful range, so only the absolute tolerance really matters; this value is
# small enough to be negligible for any stage (1 nm at a position of 1 m) and large
# enough to stay well clear of double precision's limits.
_RELATIVE_TOLERANCE = 1e-9
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))


class AutofocusCancelled(RuntimeError):
    """Raised when a focus search is stopped before it finished."""


@dataclass(frozen=True)
class FocusSearchResult:
    """Where a focus search ended up, and what it measured getting there.

    Attributes
    ----------
    z : float
        The position of the best focus found.
    score : float
        The focus score there.  For `zstack_search` this is the score of the best
        *measured* position, which may differ slightly from the fitted peak `z`.
    samples : tuple[tuple[float, float], ...]
        Every `(z, score)` measured, in the order measured.  Plot this to see the
        focus curve the search worked from.
    converged : bool
        Whether the search finished on its own terms rather than running out of
        evaluations.
    message : str
        Empty unless something is worth reporting about the search.
    """

    z: float
    score: float
    samples: tuple[tuple[float, float], ...] = field(default=())
    converged: bool = True
    message: str = ""

    @property
    def n_images(self) -> int:
        """How many images the search needed."""
        return len(self.samples)


class _Recorder:
    """Calls `measure`, recording every sample and honouring cancellation."""

    def __init__(
        self,
        measure: Callable[[float], float],
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self._measure = measure
        self._should_cancel = should_cancel
        self.samples: list[tuple[float, float]] = []

    def __call__(self, z: float) -> float:
        if self._should_cancel is not None and self._should_cancel():
            raise AutofocusCancelled(
                f"Autofocus cancelled after {len(self.samples)} image(s)."
            )
        score = float(self._measure(z))
        self.samples.append((z, score))
        return score

    def best(self) -> tuple[float, float]:
        z, score = max(self.samples, key=lambda pair: pair[1])
        return z, score


def _check_range(search_range_um: float) -> None:
    if not search_range_um > 0:
        raise ValueError(f"search_range_um must be positive, got {search_range_um}.")


def brent_search(
    measure: Callable[[float], float],
    center: float,
    search_range_um: float,
    *,
    tolerance_um: float = 1.0,
    max_evaluations: int = 100,
    should_cancel: Callable[[], bool] | None = None,
) -> FocusSearchResult:
    """Find the focus peak with Brent's method, using as few images as possible.

    Brent's method combines a golden-section search with parabolic interpolation: it
    narrows a bracket around the peak, and takes a parabola through its three best
    points whenever that lands inside the bracket.  It assumes a single peak within
    `search_range_um`.

    Parameters
    ----------
    measure : Callable[[float], float]
        Returns the focus score at a position.  Larger means sharper.
    center : float
        Position to search around, normally where the focus device already is.
    search_range_um : float
        Total width of the search, centred on `center`.
    tolerance_um : float
        Stop once the peak is bracketed this tightly.  Set it no smaller than the
        stage can reliably step.  By default, 1.0.
    max_evaluations : int
        Give up after this many images.  By default, 100.
    should_cancel : Callable[[], bool] | None
        Polled before every image; raises `AutofocusCancelled` when it returns True.
    """
    _check_range(search_range_um)
    if not tolerance_um > 0:
        raise ValueError(f"tolerance_um must be positive, got {tolerance_um}.")

    record = _Recorder(measure, should_cancel)
    lower = center - search_range_um / 2.0
    upper = center + search_range_um / 2.0

    # Brent's method as a minimization of the negated score
    def f(z: float) -> float:
        return -record(z)

    a, b = lower, upper
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    converged = False

    for _ in range(max_evaluations - 1):
        midpoint = 0.5 * (a + b)
        tol1 = _RELATIVE_TOLERANCE * abs(x) + tolerance_um
        tol2 = 2.0 * tol1
        if abs(x - midpoint) <= tol2 - 0.5 * (b - a):
            converged = True
            break

        if abs(e) > tol1:
            # try a parabola through the three best points so far
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r, e = e, d
            if abs(p) < abs(0.5 * q * r) and q * (a - x) < p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < midpoint else -tol1
            else:
                e = (b - x) if x < midpoint else (a - x)
                d = _GOLDEN * e
        else:
            e = (b - x) if x < midpoint else (a - x)
            d = _GOLDEN * e

        # never take a step smaller than the tolerance
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0 else -tol1)
        fu = f(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    best_z, best_score = record.best()
    return FocusSearchResult(
        z=best_z,
        score=best_score,
        samples=tuple(record.samples),
        converged=converged,
        message=""
        if converged
        else f"Did not converge within {max_evaluations} images.",
    )


def zstack_search(
    measure: Callable[[float], float],
    center: float,
    search_range_um: float,
    *,
    step_um: float = 1.0,
    should_cancel: Callable[[], bool] | None = None,
) -> FocusSearchResult:
    """Scan a Z range at even spacing and fit the peak of the resulting curve.

    Every position is measured, so the cost is known in advance and the whole focus
    curve is available afterwards.  The returned `z` comes from
    [`fit_peak`][pymmcore_plus.autofocus.fit_peak], so it can fall between samples.

    Parameters
    ----------
    measure : Callable[[float], float]
        Returns the focus score at a position.  Larger means sharper.
    center : float
        Position to scan around.
    search_range_um : float
        Total width of the scan, centred on `center`.
    step_um : float
        Spacing between positions.  By default, 1.0.
    should_cancel : Callable[[], bool] | None
        Polled before every image; raises `AutofocusCancelled` when it returns True.
    """
    _check_range(search_range_um)
    if not step_um > 0:
        raise ValueError(f"step_um must be positive, got {step_um}.")

    record = _Recorder(measure, should_cancel)
    # include both ends of the range, so the scan is symmetric about center
    n_steps = max(round(search_range_um / step_um), 1)
    positions = center - search_range_um / 2.0 + step_um * np.arange(n_steps + 1)
    for z in positions:
        record(float(z))

    best_z, best_score = record.best()
    zs = np.array([z for z, _ in record.samples])
    scores = np.array([s for _, s in record.samples])
    fitted = fit_peak(zs, scores)
    message = ""
    if not (positions[0] <= fitted <= positions[-1]):  # pragma: no cover
        fitted, message = best_z, "Fitted peak fell outside the scan; used the best."
    return FocusSearchResult(
        z=float(fitted),
        score=best_score,
        samples=tuple(record.samples),
        converged=True,
        message=message,
    )


def fit_peak(z: np.ndarray, scores: np.ndarray) -> float:
    """Estimate the position of a focus curve's peak, between samples.

    Fits a parabola to the log of the three points around the highest sample, which is
    the same as fitting a Gaussian to those points, and is what a focus curve looks
    like near its peak.  Falls back to the highest sample when there is nothing to
    interpolate: too few points, a peak at either end of the scan, or a degenerate fit.
    """
    z = np.asarray(z, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if z.size != scores.size:  # pragma: no cover
        raise ValueError("z and scores must be the same length.")
    if z.size == 0:  # pragma: no cover
        raise ValueError("Cannot fit a peak to an empty curve.")

    order = np.argsort(z)
    z, scores = z[order], scores[order]
    peak = int(np.argmax(scores))
    if z.size < 3 or peak == 0 or peak == z.size - 1:
        return float(z[peak])

    z0, z1, z2 = z[peak - 1 : peak + 2]
    y0, y1, y2 = scores[peak - 1 : peak + 2]
    # A Gaussian is a parabola in log space; shift the curve above zero first, since a
    # focus score may sit on a baseline (or dip negative for some methods).
    floor = min(y0, y1, y2)
    offset = 1.0 - floor if floor <= 0 else 0.0
    y0, y1, y2 = math.log(y0 + offset), math.log(y1 + offset), math.log(y2 + offset)

    denominator = (z0 - z1) * (z0 - z2) * (z1 - z2)
    if denominator == 0:  # pragma: no cover
        return float(z[peak])
    a = (z2 * (y1 - y0) + z1 * (y0 - y2) + z0 * (y2 - y1)) / denominator
    b = (z2 * z2 * (y0 - y1) + z1 * z1 * (y2 - y0) + z0 * z0 * (y1 - y2)) / denominator
    if a >= 0:
        # not a peak (the parabola opens upward or is flat): trust the sample
        return float(z[peak])
    vertex = -b / (2.0 * a)
    # the fit is only meaningful between its outer points
    if not z0 <= vertex <= z2:
        return float(z[peak])
    return float(vertex)
