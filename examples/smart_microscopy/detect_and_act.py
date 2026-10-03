"""Detect and act: when something interesting is found, acquire a follow-up there.

The base acquisition scans positions (a well plate, a grid...). Every scan frame
goes through ``detect``; for each hit, ``follow_up`` builds the events to
acquire at that spot, and they are queued ahead of the rest of the scan
(``priority="next"``). Async mode lets the scan keep moving while frames are
analyzed.

``follow_up`` is assembled from small building blocks -- recombine them, or
add your own, to do something else at a hit:

- ``center_on``: move the stage so the hit is in the middle of the field
  (needed before zooming in, or the object may fall outside the smaller field)
- ``switch_objective``: change objective *without taking an image*
- ``snap`` / ``z_stack``: acquire one image, or a stack around the hit

Device settings an event applies stay applied for every event after it. That
is why a follow-up that changes objective ends by switching back to
``scan_objective``: otherwise the rest of the scan would be acquired at the
higher magnification. (If the run is stopped in the middle of a follow-up,
the zoom objective stays in place.)
"""

from typing import Any

import numpy as np
import useq

from pymmcore_plus.smart import AnalysisContext, FrameInfo, Response

API_VERSION = 1
NAME = "Detect and act"
DESCRIPTION = (
    "At every bright spot found during a scan: a z-stack, an image at higher "
    "magnification, or both."
)
SYNC = "async"
# Only look at frames from the scan itself, not at the follow-ups we requested.
ANALYZE = {"origins": ["base"]}

Z_STACK = "Z-stack"
ZOOM = "Higher magnification"
ZOOM_AND_STACK = "Higher magnification + z-stack"

PARAMETERS = {
    "action": {
        "default": "Z-stack",
        "choices": [
            "Z-stack",
            "Higher magnification",
            "Higher magnification + z-stack",
        ],
        "label": "At each hit",
    },
    "threshold": {
        "default": 3000.0,
        "min": 0.0,
        "max": 65535.0,
        "step": 50.0,
        "label": "Hit threshold (max intensity)",
    },
    "max_followups": {"default": 10, "min": 1, "max": 10000, "label": "Max follow-ups"},
    "center_on_hit": {
        "default": True,
        "label": "Center on the hit",
        "tooltip": "Move the stage so the hit is centered before the follow-up. "
        "Assumes image right/down is stage +x/+y; check on your system.",
    },
    "follow_up_exposure_ms": {
        "default": 0.0,
        "min": 0.0,
        "max": 100000.0,
        "label": "Follow-up exposure (ms)",
        "tooltip": "0 keeps the exposure of the scan frame.",
    },
    "z_range_um": {"default": 10.0, "min": 0.1, "max": 500.0, "label": "Z range (µm)"},
    "z_step_um": {"default": 1.0, "min": 0.05, "max": 50.0, "label": "Z step (µm)"},
    "objective_device": {"default": "Objective", "label": "Objective device"},
    "objective_property": {"default": "Label", "label": "Objective property"},
    "scan_objective": {
        "default": "Nikon 10X S Fluor",
        "label": "Scan objective",
        "tooltip": "The objective the base scan uses (restored after each "
        "follow-up). Its value of the property above, as in your configuration.",
    },
    "zoom_objective": {
        "default": "Nikon 40X Plan Fluor ELWD",
        "label": "Zoom objective",
    },
    "zoom_z_offset_um": {
        "default": 0.0,
        "min": -1000.0,
        "max": 1000.0,
        "label": "Zoom focus offset (µm)",
        "tooltip": "Z difference between the two objectives (parfocality).",
    },
}


def setup(ctx: AnalysisContext) -> None:
    """Called once before acquisition starts: initialize per-run state."""
    ctx.state["followups"] = 0


def detect(image: np.ndarray, ctx: AnalysisContext) -> list[tuple[int, int]]:
    """Return the (row, col) pixel of each hit. Replace with your own detection.

    Here: the brightest pixel, if it is above the threshold. A segmentation or
    a classifier would return one entry per object found.
    """
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    if float(image[row, col]) > ctx.params["threshold"]:
        return [(int(row), int(col))]
    return []


def analyze(
    image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext
) -> Response | None:
    """Queue a follow-up acquisition at every hit in this scan frame."""
    hits = detect(image, ctx)
    ctx.record(max=float(image.max()), hits=len(hits))
    events: list[useq.MDAEvent] = []
    for row, col in hits:
        if ctx.state["followups"] >= ctx.params["max_followups"]:
            break
        ctx.state["followups"] += 1
        events.extend(follow_up(image, frame, row, col, ctx))
    return Response(events=events, priority="next") if events else None


def follow_up(
    image: np.ndarray, frame: FrameInfo, row: int, col: int, ctx: AnalysisContext
) -> list[useq.MDAEvent]:
    """The events to acquire for one hit. Edit this to do something else."""
    p = ctx.params
    x0, y0, z0 = position(frame)
    x, y, z = x0, y0, z0
    if p["center_on_hit"]:
        x, y = center_on(image, frame, row, col, x, y)
    ctx.log(f"Frame {frame.frame_id}: hit at pixel ({row}, {col}) -> {p['action']}")

    channel = frame.event.channel
    exposure = p["follow_up_exposure_ms"] or frame.event.exposure
    zoom = p["action"] in (ZOOM, ZOOM_AND_STACK)

    events = []
    if zoom:
        if z is not None:
            z += p["zoom_z_offset_um"]
        events.append(switch_objective(ctx, p["zoom_objective"], x, y, z))
    if p["action"] == ZOOM:
        events.append(snap(channel, exposure, x, y, z))
    else:
        events.extend(z_stack(ctx, channel, exposure, x, y, z))
    if zoom:
        # Back to the scan objective, where the scan left off.
        events.append(switch_objective(ctx, p["scan_objective"], x0, y0, z0))
    return events


# ------------------------------------------------------------ building blocks


def position(frame: FrameInfo) -> tuple[float | None, float | None, float | None]:
    """Stage x, y, z of a frame: from its event, else as recorded."""
    recorded = frame.metadata.get("position") or {}
    event = frame.event
    return (
        event.x_pos if event.x_pos is not None else recorded.get("x"),
        event.y_pos if event.y_pos is not None else recorded.get("y"),
        event.z_pos if event.z_pos is not None else recorded.get("z"),
    )


def center_on(
    image: np.ndarray,
    frame: FrameInfo,
    row: int,
    col: int,
    x: float | None,
    y: float | None,
) -> tuple[float | None, float | None]:
    """Stage x, y that put pixel (row, col) in the middle of the field.

    Assumes image columns grow with stage +x and rows with stage +y (no
    rotation or flip between camera and stage); adjust the signs if yours
    differ.
    """
    px = frame.metadata.get("pixel_size_um") or 0
    if not px or x is None or y is None:
        return x, y
    height, width = image.shape[:2]
    return x + (col - width / 2) * px, y + (row - height / 2) * px


def switch_objective(
    ctx: AnalysisContext,
    objective: str,
    x: float | None,
    y: float | None,
    z: float | None,
) -> useq.MDAEvent:
    """An event that moves the stage and changes objective, taking no image.

    A ``CustomAction`` event is set up like any other (stage, properties) but
    acquires nothing.
    """
    p = ctx.params
    return useq.MDAEvent(
        action=useq.CustomAction(name=f"objective: {objective}"),
        properties=[
            useq.PropertyTuple(
                p["objective_device"], p["objective_property"], objective
            )
        ],
        x_pos=x,
        y_pos=y,
        z_pos=z,
    )


def snap(
    channel: Any,  # an event's channel (its own type, not useq.Channel)
    exposure: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
) -> useq.MDAEvent:
    """One image at (x, y, z)."""
    return useq.MDAEvent(channel=channel, exposure=exposure, x_pos=x, y_pos=y, z_pos=z)


def z_stack(
    ctx: AnalysisContext,
    channel: Any,  # an event's channel (its own type, not useq.Channel)
    exposure: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
) -> list[useq.MDAEvent]:
    """A stack of ``z_range_um`` around z; one image if z is unknown."""
    if z is None:
        ctx.log("No z position known for this frame: one image instead of a stack.")
        return [snap(channel, exposure, x, y, z)]
    half = ctx.params["z_range_um"] / 2
    step = ctx.params["z_step_um"]
    return [
        useq.MDAEvent(
            channel=channel,
            exposure=exposure,
            x_pos=x,
            y_pos=y,
            z_pos=float(z + dz),
        )
        for dz in np.arange(-half, half + step / 2, step)
    ]
