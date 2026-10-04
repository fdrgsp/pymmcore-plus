"""A script file whose hooks are the methods of a class.

A script may define its hooks either as module-level functions (see the other
examples) or as a single class, as here. The runner creates one instance per
run, so ``self`` holds that run's state, and the class attributes are read
exactly like a script's constants.

This is the same shape as ``analyzer_object.py``; the difference is only how
it is delivered. A file can be loaded, edited and reloaded by a front end
(the pymmcore-gui Smart Microscopy tab does that), and is inspected without
being executed. Settings come from ``PARAMETERS`` rather than ``__init__``,
since the runner is the one constructing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from pymmcore_plus.smart import (
    STOP,
    AnalysisContext,
    FrameInfo,
    Response,
    SmartAnalyzer,
)

if TYPE_CHECKING:
    import useq

API_VERSION = 1


class AdaptiveExposure(SmartAnalyzer):
    """Keeps the mean intensity near a target."""

    NAME = "Adaptive exposure (class in a script)"
    DESCRIPTION = "Scales the exposure after every frame to reach a target mean."
    SYNC = "blocking"
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "target_mean": {"default": 2000.0, "min": 1.0, "max": 65535.0},
        "n_frames": {"default": 10, "min": 1, "max": 10000},
    }

    def __init__(self) -> None:
        # No arguments: the runner constructs this. Per-run state goes here,
        # user-settable values come from ctx.params.
        self.means: list[float] = []

    def setup(self, ctx: AnalysisContext) -> None:
        ctx.log(f"Target {ctx.params['target_mean']} on the {ctx.execution}")

    def analyze(
        self, image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
    ) -> useq.MDAEvent | Response | None:
        mean = float(image.mean())
        self.means.append(mean)
        exposure = frame.event.exposure or frame.metadata.get("exposure_ms") or 10.0
        ctx.record(mean=mean, exposure_ms=exposure)

        if len(self.means) >= ctx.params["n_frames"]:
            return STOP
        scale = ctx.params["target_mean"] / mean if mean > 0 else 2.0
        return frame.event.model_copy(
            update={"exposure": float(np.clip(exposure * scale, 1, 500)), "index": {}}
        )
