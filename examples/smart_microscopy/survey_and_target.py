"""Survey and target: find objects in a low-magnification scan, then image them.

Phase 1 -- the base acquisition: a tile scan at the survey objective (e.g. a
grid plan). ``analyze`` keeps every tile with its stage position.

Phase 2 -- ``after_base``, once the whole survey is acquired: the tiles are
placed into one mosaic by stage position (no registration), objects are
segmented, and a small grid centred on each object is queued -- optionally at a
higher-magnification objective, switched to before the targets and back after
them.

Grids returned without a field of view get one from the pixel size in effect
where they run: after the objective switch, that of the zoom objective. If that
objective has no calibrated pixel size, the response is refused (the tiles
could not be placed) and the run stops with an explanation.

Replace ``segment`` with your own segmentation (scikit-image, cellpose...).
Mosaic and stage coordinates assume image columns grow with stage +x and rows
with stage +y; adjust the signs in ``to_stage`` if yours differ.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import useq

from pymmcore_plus.smart import AnalysisContext, FrameInfo, Response

API_VERSION = 1
NAME = "Survey and target"
DESCRIPTION = (
    "Segments a stitched low-magnification survey and images a grid around each "
    "object, optionally at a higher magnification."
)
SYNC = "async"
ANALYZE = {"origins": ["base"]}

PARAMETERS = {
    "threshold": {
        "default": 3000.0,
        "min": 0.0,
        "max": 65535.0,
        "step": 50.0,
        "label": "Object threshold (intensity)",
    },
    "min_area_px": {
        "default": 4,
        "min": 1,
        "max": 1000000,
        "label": "Min object area (mosaic px)",
    },
    "max_targets": {"default": 20, "min": 1, "max": 10000, "label": "Max targets"},
    "mosaic_downsample": {
        "default": 4,
        "min": 1,
        "max": 64,
        "label": "Mosaic downsampling",
        "tooltip": "Survey tiles are binned by this factor before segmentation.",
    },
    "grid_rows": {"default": 2, "min": 1, "max": 100, "label": "Target grid rows"},
    "grid_columns": {
        "default": 2,
        "min": 1,
        "max": 100,
        "label": "Target grid columns",
    },
    "grid_overlap_pct": {
        "default": 10.0,
        "min": 0.0,
        "max": 90.0,
        "label": "Overlap (%)",
    },
    "switch_objective": {"default": False, "label": "Image targets at zoom objective"},
    "objective_device": {"default": "Objective", "label": "Objective device"},
    "objective_property": {"default": "Label", "label": "Objective property"},
    "survey_objective": {"default": "Nikon 10X S Fluor", "label": "Survey objective"},
    "zoom_objective": {
        "default": "Nikon 40X Plan Fluor ELWD",
        "label": "Zoom objective",
    },
    "zoom_z_offset_um": {
        "default": 0.0,
        "min": -1000.0,
        "max": 1000.0,
        "label": "Zoom focus offset (µm)",
    },
}


def setup(ctx: AnalysisContext) -> None:
    """Called once before acquisition starts."""
    ctx.state["tiles"] = []


def analyze(image: np.ndarray, frame: FrameInfo, ctx: AnalysisContext) -> None:
    """Phase 1: keep each survey tile (binned) with its position."""
    ds = ctx.params["mosaic_downsample"]
    pos = frame.metadata.get("position") or {}
    x = frame.event.x_pos if frame.event.x_pos is not None else pos.get("x")
    y = frame.event.y_pos if frame.event.y_pos is not None else pos.get("y")
    z = frame.event.z_pos if frame.event.z_pos is not None else pos.get("z")
    px = frame.metadata.get("pixel_size_um") or ctx.system.pixel_size_um
    if x is None or y is None or not px:
        ctx.log(f"Frame {frame.frame_id}: no stage position or pixel size; skipped.")
        return
    ctx.state["tiles"].append(
        {
            "image": bin_image(image, ds),
            "x": x,
            "y": y,
            "z": z,
            "px": px * ds,
            "channel": frame.event.channel,
            "exposure": frame.event.exposure,
        }
    )
    ctx.record(mean=float(image.mean()))


def after_base(ctx: AnalysisContext) -> Response | None:
    """Phase 2: stitch, segment, and queue a grid around each object."""
    tiles = ctx.state["tiles"]
    if not tiles:
        ctx.log("No survey tiles collected; nothing to target.", "warning")
        return None
    mosaic, origin, px = build_mosaic(tiles)
    centroids = segment(mosaic, ctx)[: ctx.params["max_targets"]]
    ctx.record(objects=len(centroids), mosaic_shape=str(mosaic.shape))
    ctx.log(f"Survey: {len(tiles)} tiles, {len(centroids)} objects found.")
    if not centroids:
        return None

    p = ctx.params
    z = tiles[0]["z"]
    if p["switch_objective"] and z is not None:
        z += p["zoom_z_offset_um"]
    grid = useq.GridRowsColumns(
        rows=p["grid_rows"],
        columns=p["grid_columns"],
        overlap=p["grid_overlap_pct"],
    )
    targets = [
        useq.Position(
            x=x,
            y=y,
            z=z,
            name=f"target_{i:03d}",
            sequence=useq.MDASequence(grid_plan=grid),
        )
        for i, (x, y) in enumerate(to_stage(c, origin, px) for c in centroids)
    ]
    sequence = useq.MDASequence(
        stage_positions=tuple(targets), channels=channels_like(tiles[0])
    )
    events: list[Any] = [sequence]
    if p["switch_objective"]:
        first = targets[0]
        events = [
            switch_objective(ctx, p["zoom_objective"], first.x, first.y, z),
            sequence,
            switch_objective(
                ctx, p["survey_objective"], first.x, first.y, tiles[0]["z"]
            ),
        ]
    return Response(events=events)


# ------------------------------------------------------------ building blocks


def bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Average-bin *image* by *factor* (edges that do not fit are dropped)."""
    if factor <= 1:
        return image.astype(np.float32)
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    view = image[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return view.mean(axis=(1, 3), dtype=np.float32)


def build_mosaic(tiles: list[dict]) -> tuple[np.ndarray, tuple[float, float], float]:
    """Place tiles by stage position; return (mosaic, (x0, y0) of pixel 0, px)."""
    px = tiles[0]["px"]
    lefts = [t["x"] - t["image"].shape[1] * px / 2 for t in tiles]
    tops = [t["y"] - t["image"].shape[0] * px / 2 for t in tiles]
    x0, y0 = min(lefts), min(tops)
    width = max(
        round((left - x0) / px) + t["image"].shape[1]
        for left, t in zip(lefts, tiles, strict=False)
    )
    height = max(
        round((top - y0) / px) + t["image"].shape[0]
        for top, t in zip(tops, tiles, strict=False)
    )
    mosaic = np.zeros((height, width), dtype=np.float32)
    for left, top, tile in zip(lefts, tops, tiles, strict=False):
        r, c = round((top - y0) / px), round((left - x0) / px)
        h, w = tile["image"].shape
        mosaic[r : r + h, c : c + w] = tile["image"]
    return mosaic, (x0, y0), px


def segment(mosaic: np.ndarray, ctx: AnalysisContext) -> list[tuple[float, float]]:
    """(row, col) centroids of the objects in *mosaic*, largest first.

    Replace with your own segmentation. Here: threshold, then connected
    components (scipy if installed, otherwise a small built-in labeler).
    """
    mask = mosaic > ctx.params["threshold"]
    labels, n = label(mask)
    objects = []
    for i in range(1, n + 1):
        rows, cols = np.nonzero(labels == i)
        if rows.size >= ctx.params["min_area_px"]:
            objects.append((rows.size, (float(rows.mean()), float(cols.mean()))))
    return [centroid for _, centroid in sorted(objects, reverse=True)]


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of *mask* (4-connectivity)."""
    try:  # optional: scipy is faster, but not required
        ndimage = importlib.import_module("scipy.ndimage")
    except ImportError:
        pass
    else:
        labels, n = ndimage.label(mask)
        return labels, int(n)
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    for start in zip(*np.nonzero(mask), strict=False):
        if labels[start]:
            continue
        n += 1
        stack = [start]
        labels[start] = n
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (
                    0 <= nr < mask.shape[0]
                    and 0 <= nc < mask.shape[1]
                    and mask[nr, nc]
                    and not labels[nr, nc]
                ):
                    labels[nr, nc] = n
                    stack.append((nr, nc))
    return labels, n


def to_stage(
    centroid: tuple[float, float], origin: tuple[float, float], px: float
) -> tuple[float, float]:
    """Mosaic (row, col) -> stage (x, y)."""
    row, col = centroid
    return origin[0] + (col + 0.5) * px, origin[1] + (row + 0.5) * px


def channels_like(tile: dict) -> tuple[useq.Channel, ...]:
    """The survey's channel, as the Channel type a sequence takes."""
    channel = tile["channel"]
    if channel is None:
        return ()
    return (
        useq.Channel(
            config=channel.config, group=channel.group, exposure=tile["exposure"]
        ),
    )


def switch_objective(
    ctx: AnalysisContext,
    objective: str,
    x: float | None,
    y: float | None,
    z: float | None,
) -> useq.MDAEvent:
    """Move the stage and change objective, taking no image."""
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
