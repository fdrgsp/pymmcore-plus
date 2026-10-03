"""Run a smart acquisition headless, steered by one of the example scripts.

The base acquisition here is a 3x3 tile scan; ``survey_and_target.py`` stitches
it, segments the mosaic and acquires a 2x2 grid around each object it finds.
Swap in another script (or your own) to change the experiment.
"""

from pathlib import Path

import useq

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.smart import SmartRunner

core = CMMCorePlus()
core.loadSystemConfiguration()  # the demo configuration

survey = useq.MDASequence(
    channels=["DAPI"],
    grid_plan=useq.GridRowsColumns(rows=3, columns=3),
)

runner = SmartRunner(core)
runner.events.logMessage.connect(lambda level, msg: print(f"[{level}] {msg}"))
runner.events.analysisError.connect(lambda msg, fatal: print("ERROR:", msg))

summary = runner.run(
    survey,
    Path(__file__).parent / "survey_and_target.py",
    params={"threshold": 3000.0},
    execution="thread",  # or "process": same script, separate interpreter
    output="memory",  # or "experiment.ome.zarr" to save
    run_dir="auto",  # run.json, frames.jsonl, analysis.jsonl, script.py
)
print(summary)

# The same, non-blocking:
#   runner = core.run_smart(survey, "survey_and_target.py", run_dir="auto")
#   ... runner.cancel() / runner.request_stop() ...
#   summary = runner.wait()
