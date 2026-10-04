# Smart Microscopy

!!! tip "See also"

    [Event-Driven Acquisition](./event_driven_acquisition.md) explains the
    mechanism underneath: passing an iterator of events to the MDA runner.
    `pymmcore_plus.smart` builds a complete, reusable workflow on top of it.

In a smart acquisition, a Python **script** analyzes frames as they arrive
and decides what to acquire next. It can change the exposure, add a z-stack
where something interesting was found, image a region at a higher
magnification, or stop.

```python
import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.smart import SmartRunner

core = CMMCorePlus()
core.loadSystemConfiguration()

survey = useq.MDASequence(
    channels=["DAPI"], grid_plan=useq.GridRowsColumns(rows=3, columns=3)
)
summary = SmartRunner(core).run(
    survey,
    "survey_and_target.py",  # the analysis script
    params={"threshold": 3000},  # values for its PARAMETERS
    execution="process",  # or "thread"
    output="experiment.ome.zarr",
    run_dir="auto",  # write the run's records next to the data
)
```

There's also a non-blocking shortcut: `runner = core.run_smart(survey,
"survey_and_target.py")`, then `runner.wait()`. Runnable scripts are in the
repository's
[`examples/smart_microscopy`](https://github.com/pymmcore-plus/pymmcore-plus/tree/main/examples/smart_microscopy)
folder.

## Everything in one file

The script does not have to be a separate file. Pass ``__file__`` and keep
the run itself under an ``if __name__ == "__main__":`` guard: the worker
loads the module under its own name (a spawned process imports it as
``__mp_main__``), so the guarded part never runs there -- only the hooks are
picked up.

```python
import useq
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.smart import STOP

API_VERSION = 1


def analyze(image, frame, ctx):
    ctx.record(mean=float(image.mean()))
    return STOP if frame.frame_id >= 9 else useq.MDAEvent()


if __name__ == "__main__":
    core = CMMCorePlus()
    core.loadSystemConfiguration()
    runner = core.run_smart(
        useq.MDASequence(channels=["DAPI"]), __file__, output="memory"
    )
    print(runner.wait())
```

A complete version is in
[`examples/smart_microscopy/single_file.py`](https://github.com/pymmcore-plus/pymmcore-plus/tree/main/examples/smart_microscopy/single_file.py).
Without the guard the worker would re-run the acquisition as it loads the
module, so keep it even in thread mode.

## Writing a script

A script is a single Python file. Its settings are read **without running
it**, by parsing the source, so module-level settings must be plain
literals.

```python
import numpy as np
import useq

from pymmcore_plus.smart import STOP, AnalysisContext, FrameInfo, Response

API_VERSION = 1  # required
NAME = "My experiment"  # optional
EXECUTION = "thread"  # default: "thread" | "process"
SYNC = "blocking"  # default: "blocking" | "async"
SEQUENCING = "safe"  # default: "safe" | "off" | "always"
ANALYZE = {"channels": ["FITC"], "every_nth": 1, "origins": ["base"]}
PARAMETERS = {
    "threshold": {"default": 1000.0, "min": 0, "max": 65535},
    "n_max": 10,
}


def setup(ctx: AnalysisContext) -> None:  # optional: before acquiring
    ctx.state["hits"] = 0


def analyze(image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext):
    hit = float(image.max()) > ctx.params["threshold"]
    ctx.record(max=float(image.max()), hit=hit)
    if not hit:
        return None  # nothing to add
    ctx.state["hits"] += 1
    if ctx.state["hits"] > ctx.params["n_max"]:
        return STOP  # finish the run
    return frame.event.model_copy(update={"exposure": 50.0})


def after_base(ctx: AnalysisContext):  # optional: survey done
    return None


def teardown(ctx: AnalysisContext) -> None:  # optional: always called
    ctx.log(f"{ctx.state['hits']} hits")
```

### What the hooks receive

| | |
|---|---|
| `image` | The frame (numpy array, **read-only**). |
| `frame.frame_id` | 0-based acquisition order. It is also the frame's index along the data's `t` axis. |
| `frame.event` | The `useq.MDAEvent` that produced it. |
| `frame.metadata` | Recorded state: `position`, `pixel_size_um`, `exposure_ms`, `camera_device`, `runner_time_ms`... |
| `frame.origin` / `frame.parent_frame_id` | `"base"`, `"analysis"` or `"external"` (see below), and which frame's analysis requested it. |
| `ctx.params` | Resolved `PARAMETERS` (read-only). |
| `ctx.state` | A dict that persists for the run. |
| `ctx.system` | A [`SystemInfo`][pymmcore_plus.smart.SystemInfo] snapshot of the microscope at the start: image size, pixel sizes per pixel configuration... |
| `ctx.base_sequence` | The base `MDASequence`. |
| `ctx.run_dir` | A folder for the script's own outputs (None when the run keeps no records). |
| `ctx.log(msg, level)` / `ctx.record(**values)` | A message, and per-frame scalars. Both are reported in signals and the run log. |

### What `analyze` and `after_base` may return

`None` (no change), an `MDAEvent`, an `MDASequence`, a list mixing both
(acquired in order), `STOP`, or a [`Response`][pymmcore_plus.smart.Response]
for full control:

```python
Response(
    events=[...],  # MDAEvents and MDASequences, in order
    priority="next",  # "next": before remaining base events; "end": after
    timing="relative",  # a returned time-lapse starts its own clock now
    stop=False,  # finish the run
    drop_base=False,  # discard the remaining base events
)
```

### Rules

- **Scripts never touch the core.** They act only by returning events,
  which the engine executes on its own thread. This is what lets the same
  script run in a separate process.
- **Device settings an event applies persist** for later events, including
  the remaining base events. Change them back at the end of a follow-up.
  To change settings without taking an image, use
  `action=useq.CustomAction(...)`. The event still moves the stage and
  applies its `properties`, but acquires nothing.
- **Grids need a field of view.** useq places grid tiles by
  `fov_width`/`fov_height`, and without them tiles end up 1 µm apart. A
  returned grid without those is sized by the runner **when it is about to
  run**, from the pixel size in effect at that moment -- so an objective
  switched earlier counts, whether by this response, an earlier one, or a
  base event. If that pixel size is not calibrated, the run stops with an
  explanation (or the grid is skipped, with `on_error="skip"`). A switch
  into an uncalibrated state that no grid needs only logs a warning.
  `ctx.system` holds every pixel configuration and its pixel size, for
  scripts that would rather size a grid themselves
  (`ctx.system.fov_um("Res40x")`).
- An event's `channel` is a different type from the `useq.Channel` a
  sequence takes. Rebuild it with `useq.Channel(config=c.config,
  group=c.group)`.
- `useq.MDASequence()` with no axes contains no events, and a smart run
  refuses it.

## Threads or processes

| | `execution="thread"` | `execution="process"` |
|---|---|---|
| Start-up | instant | seconds, paid before acquisition starts |
| Per frame | no copy | the frame is pickled to the worker |
| Crash in the script | takes the process down | contained; the run stops |
| Hung analysis | cannot be interrupted | the worker is terminated |

Processes are always *spawned*, never forked. In a frozen (PyInstaller)
application, call `multiprocessing.freeze_support()` first in the entry
point.

## Steering a run from outside the script

`SmartRunner.request()` adds events to a run in progress -- the same thing an
`analyze` hook does by returning them, but callable from a console, a button,
or another thread. It is thread-safe.

```python
runner = core.run_smart(survey, "my_script.py")
...
runner.request(useq.MDAEvent(channel="FITC"))  # acquire next
runner.request(z_stack_sequence, priority="end")  # after the base events
runner.request(drop_base=True, stop=True)  # or just steer the run
```

Such frames are recorded with `origin="external"` (and no
`parent_frame_id`), so they are easy to tell apart afterwards, and a script
can ignore them with `ANALYZE = {"origins": ["base", "analysis"]}`. Each
request is written to `analysis.jsonl` as a ``"request"`` entry.

## Hardware sequencing

Hardware sequencing pre-triggers the camera for a run of events, so they are
acquired as fast as the hardware allows instead of one round-trip at a time.
It needs the events up front, which is in tension with feedback: you cannot
pre-trigger frames whose existence depends on analyzing the previous one.
`SEQUENCING` (or `sequencing=`) decides the balance:

| | Events a script returns in one response | Base events |
|---|---|---|
| `"off"` | one at a time | one at a time |
| `"safe"` (default) | **one burst** | **one burst** in async; one at a time in blocking |
| `"always"` | **one burst** | **one burst** in both modes |

- A **returned batch** (a z-stack, a time-lapse) is sequenced in both modes:
  the script committed to those events as a unit, so nothing is meant to be
  decided between them.
- **Base events** are sequenced in async mode, where nothing gates them. In
  blocking mode each frame's analysis is meant to gate the next acquisition,
  so sequencing them is opt-in (`"always"`): feedback then applies between
  bursts rather than between frames.
- Only events that are **already due** share a burst, so a timed series keeps
  its timing while a 0-interval one runs flat out.
- `max_burst` (default 100) caps a burst, which bounds how long a
  `priority="next"` request waits. Cancelling works inside a burst.
- Sequencing also needs the engine's own switch,
  `core.mda.engine.use_hardware_sequencing` (on by default), and hardware
  that supports it; otherwise events simply run one at a time.

## Blocking or async

- `"blocking"`: no new event is acquired while an analysis is pending. Use
  it for feedback loops.
- `"async"`: base events keep being acquired, and requested events are
  inserted as results arrive. Use it for scanning.

A run ends when the base events are done, nothing requested is queued, no
analysis is pending, and `after_base` (if defined) has run. It also ends on
`STOP`, `request_stop()`, `cancel()`, an analysis error (with the default
`on_error="stop"`), or `analysis_timeout_s`.

## Data and records

A smart run's shape isn't known in advance, so its frames are stored along a
single `t` axis, in acquisition order. Each frame's full event is stored in
its per-frame metadata under `"mda_event"`, as a JSON string. In OME-Zarr
it sits in the attributes, in OME-TIFF in the structured annotations. That
way the data file says which channel, position or z each frame is.

With `run_dir`, the runner also writes:

- `run.json`: settings, parameters, the base sequence, the system snapshot,
  status and counts
- `frames.jsonl`: one line per frame
- `analysis.jsonl`: one line per hook call
- `script.py`: the exact code that ran

Use `run_dir="auto"` for `<data name>_smart/` next to the data, or a
temporary folder when nothing is saved.

## While writing a script

[`dry_run`][pymmcore_plus.smart.dry_run] calls `analyze` once on an image
and shows what it would request, without acquiring anything:

```python
core.snapImage()
result = dry_run("my_script.py", core.getImage(), core=core)
print(result.response, result.records, result.error)
```
