"""The same experiment as `single_file.py`, written as a class.

Instead of module-level hooks and ``__file__``, the hooks are methods on an
object: the run's state lives on ``self``, its settings are ``__init__``
arguments, and the class can be imported and unit-tested like any other.

Pass an *instance* to the runner::

    core.run_smart(sequence, AdaptiveExposure(target_mean=2000))

In process mode the object is pickled to the worker, so define the class at
module level and keep its state picklable. In thread mode it is used as is,
so the attributes are still yours to read when the run ends.

Run it with::

    python analyzer_object.py            # analysis on a worker thread
    python analyzer_object.py process    # analysis in a separate process
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import useq

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.smart import STOP, AnalysisContext, FrameInfo, Response

if TYPE_CHECKING:
    from collections.abc import Sequence


class AdaptiveExposure:
    """Keeps the mean intensity near a target, and remembers what it saw."""

    NAME = "Adaptive exposure (object)"
    SYNC = "blocking"  # each frame's analysis decides the next one

    def __init__(self, target_mean: float = 2000.0, n_frames: int = 10) -> None:
        self.target_mean = target_mean
        self.n_frames = n_frames
        self.means: list[float] = []

    # --- hooks ------------------------------------------------------------

    def setup(self, ctx: AnalysisContext) -> None:
        ctx.log(f"Starting on the {ctx.execution}; target {self.target_mean}")

    def analyze(
        self, image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
    ) -> useq.MDAEvent | Response | None:
        mean = float(image.mean())
        self.means.append(mean)
        exposure = frame.event.exposure or frame.metadata.get("exposure_ms") or 10.0
        ctx.record(mean=mean, exposure_ms=exposure)

        if len(self.means) >= self.n_frames:
            return STOP
        scale = self.target_mean / mean if mean > 0 else 2.0
        return frame.event.model_copy(
            update={"exposure": float(np.clip(exposure * scale, 1, 500)), "index": {}}
        )

    def teardown(self, ctx: AnalysisContext) -> None:
        ctx.log(f"Saw {len(self.means)} frames")

    # --- ordinary methods, testable without any microscope ----------------

    @property
    def drift(self) -> Sequence[float]:
        """How far each frame was from the target."""
        return [m - self.target_mean for m in self.means]


if __name__ == "__main__":
    execution = "process" if "process" in sys.argv else "thread"

    core = CMMCorePlus()
    core.loadSystemConfiguration()  # the demo configuration

    analyzer = AdaptiveExposure(target_mean=2000.0, n_frames=10)
    runner = core.run_smart(
        useq.MDASequence(channels=["DAPI"]),
        analyzer,
        execution=execution,
        output="memory",
    )
    runner.events.logMessage.connect(lambda level, msg: print(f"[{level}] {msg}"))
    print(runner.wait())

    if execution == "thread":
        # the very object we passed was used, so its state is right here
        print("drift:", [round(d, 1) for d in analyzer.drift])
