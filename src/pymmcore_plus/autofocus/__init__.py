"""Autofocus: focus scores, focus searches, and the result of an autofocus attempt.

Image-based (software) autofocus routines build on these: a
[`ScoringMethod`][pymmcore_plus.autofocus.ScoringMethod] says how sharp one image is,
a search walks the focus device to the peak of that score, and every attempt reports
an [`AutofocusResult`][pymmcore_plus.autofocus.AutofocusResult].
"""

from ._capture import CaptureSettings, capture_state
from ._fft import bandpass_power
from ._filters import convolve3x3, median_3x3, sharpen, sobel_magnitude
from ._optimizers import (
    AutofocusCancelled,
    FocusSearchResult,
    brent_search,
    fit_peak,
    zstack_search,
)
from ._registry import (
    SoftwareAutofocusMethod,
    available_methods,
    get_method,
    register_software_autofocus,
    run_software_autofocus,
    settings_model,
)
from ._result import AutofocusResult
from ._scoring import ScoringMethod, jaf_score, score_image
from ._settings import DuoSettings, JAFSettings, OughtaFocusSettings

__all__ = [
    "AutofocusCancelled",
    "AutofocusResult",
    "CaptureSettings",
    "DuoSettings",
    "FocusSearchResult",
    "JAFSettings",
    "OughtaFocusSettings",
    "ScoringMethod",
    "SoftwareAutofocusMethod",
    "available_methods",
    "bandpass_power",
    "brent_search",
    "capture_state",
    "convolve3x3",
    "fit_peak",
    "get_method",
    "jaf_score",
    "median_3x3",
    "register_software_autofocus",
    "run_software_autofocus",
    "score_image",
    "settings_model",
    "sharpen",
    "sobel_magnitude",
    "zstack_search",
]
