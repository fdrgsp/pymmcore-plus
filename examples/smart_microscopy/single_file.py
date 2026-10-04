"""A complete smart acquisition in one file: the hooks and the run that uses them.

Usually a script is a separate file passed to the runner. It does not have to
be: pass ``__file__`` and keep the run itself under ``if __name__ ==
"__main__":``. The worker loads this module under its own name (and a spawned
process imports it as ``__mp_main__``), so the guarded part never runs there --
only the hooks are picked up.

Run it with::

    python single_file.py            # analysis on a worker thread
    python single_file.py process    # analysis in a separate process
"""

from __future__ import annotations

import sys

import numpy as np
import useq

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.smart import STOP, AnalysisContext, FrameInfo, Response

# --------------------------------------------------------------- the script

API_VERSION = 1
NAME = "Adaptive exposure (single file)"
DESCRIPTION = "Scales the exposure after every frame to reach a target mean."
SYNC = "blocking"  # each frame's analysis decides the next one

PARAMETERS = {
    "target_mean": {"default": 2000.0, "min": 1.0, "max": 65535.0},
    "n_frames": {"default": 10, "min": 1, "max": 10000},
}


def setup(ctx: AnalysisContext) -> None:
    """Called once, before the first frame."""
    ctx.log(f"Starting on the {ctx.execution}; target {ctx.params['target_mean']}")


def analyze(
    image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
) -> useq.MDAEvent | Response | None:
    """Called per frame; returns what to acquire next."""
    mean = float(image.mean())
    exposure = frame.event.exposure or frame.metadata.get("exposure_ms") or 10.0
    ctx.record(mean=mean, exposure_ms=exposure)

    if frame.frame_id + 1 >= ctx.params["n_frames"]:
        return STOP
    scale = ctx.params["target_mean"] / mean if mean > 0 else 2.0
    return frame.event.model_copy(
        update={"exposure": float(np.clip(exposure * scale, 1, 500)), "index": {}}
    )


# ------------------------------------------------------- running it, headless

if __name__ == "__main__":
    execution = "process" if "process" in sys.argv else "thread"

    core = CMMCorePlus()
    core.loadSystemConfiguration()  # the demo configuration

    runner = core.run_smart(
        # one event to start the reactive loop; each analysis asks for the next
        useq.MDASequence(channels=["DAPI"]),
        __file__,  # this very file is the analysis script
        execution=execution,
        params={"n_frames": 10},
        output="memory",  # or "experiment.ome.zarr" to save
        run_dir=None,  # or "auto" to write run.json / frames.jsonl
    )
    runner.events.logMessage.connect(lambda level, msg: print(f"[{level}] {msg}"))
    runner.events.analysisFinished.connect(
        lambda rec: (
            print(f"  frame {rec['frame_id']}: {rec['records']}")
            if rec["call"] == "analyze"
            else None
        )
    )

    print(runner.wait())  # blocks until the run has finished
