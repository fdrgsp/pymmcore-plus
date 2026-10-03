"""Adaptive exposure: keep the mean intensity near a target.

A purely reactive run. The base acquisition is a single event (e.g. one
channel, no time plan); every analysis returns the *next* event, with the
exposure scaled to bring the mean intensity toward ``target_mean``. The run
stops after ``n_frames`` frames.

Blocking mode guarantees each new exposure is decided from the previous frame.
"""

import numpy as np
import useq

from pymmcore_plus.smart import STOP, AnalysisContext, FrameInfo, Response

API_VERSION = 1
NAME = "Adaptive exposure"
DESCRIPTION = "Scales the exposure after every frame to reach a target mean."
SYNC = "blocking"

PARAMETERS = {
    "target_mean": {
        "default": 2000.0,
        "min": 1.0,
        "max": 65535.0,
        "step": 100.0,
        "label": "Target mean intensity",
    },
    "min_exposure_ms": {"default": 1.0, "min": 0.01, "max": 10000.0},
    "max_exposure_ms": {"default": 500.0, "min": 0.01, "max": 10000.0},
    "n_frames": {"default": 20, "min": 1, "max": 100000, "label": "Frames"},
}


def analyze(
    image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
) -> useq.MDAEvent | Response:
    """Measure the frame, then return the next event with a new exposure."""
    mean = float(image.mean())
    exposure = frame.event.exposure or frame.metadata.get("exposure_ms") or 10.0
    ctx.record(mean=mean, exposure_ms=exposure)

    if frame.frame_id + 1 >= ctx.params["n_frames"]:
        return STOP

    scale = ctx.params["target_mean"] / mean if mean > 0 else 2.0
    new_exposure = float(
        np.clip(
            exposure * scale,
            ctx.params["min_exposure_ms"],
            ctx.params["max_exposure_ms"],
        )
    )
    # Same event (channel, position...), new exposure, and no index: the
    # repeated event is identified by its frame number in the saved data.
    return frame.event.model_copy(update={"exposure": new_exposure, "index": {}})
