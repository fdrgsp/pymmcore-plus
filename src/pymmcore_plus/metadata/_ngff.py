from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import useq
from yaozarrs import v05

from pymmcore_plus.mda._runner import GeneratorMDASequence

if TYPE_CHECKING:
    from pymmcore_plus.metadata.schema import ImageInfo

    from .schema import FrameMetaV1, SummaryMetaV1

__all__ = ["NGFFMetadata", "create_ngff_metadata"]


class NGFFMetadata(NamedTuple):
    """Container for NGFF metadata at different hierarchy levels.

    This structure allows consistent handling of single-position, multi-position,
    and plate acquisitions when writing NGFF/OME-Zarr data to disk.

    Attributes
    ----------
    root : v05.Image | v05.Plate | v05.Bf2Raw
        The top-level group metadata to be written to the root zarr.json file.
        - For single-position: v05.Image (the image itself)
        - For multi-position: v05.Bf2Raw (bioformats2raw layout metadata)
        - For plates: v05.Plate (plate structure with wells)

    images : dict[str, v05.Image]
        Dictionary mapping relative paths to Image metadata objects.
        - For single-position: {} (empty - root IS the image)
        - For multi-position: {"0": Image, "1": Image, ...}
        - For plates: {"A/1/0": Image, "A/1/1": Image, ...} (well/field paths)

    Examples
    --------
    Single position acquisition:
        NGFFMetadata(root=Image(...), images={})

    Multi-position acquisition (2 positions):
        NGFFMetadata(
            root=Bf2Raw(version="0.5", bioformats2raw_layout=3),
            images={"0": Image(...), "1": Image(...)}
        )

    Plate acquisition (well A/1 with 2 fields):
        NGFFMetadata(
            root=Plate(...),
            images={"A/1/0": Image(...), "A/1/1": Image(...)}
        )
    """

    root: v05.Image | v05.Plate | v05.Bf2Raw
    images: dict[str, v05.Image]


def create_ngff_metadata(
    summary_metadata: SummaryMetaV1, frame_metadata_list: list[FrameMetaV1]
) -> NGFFMetadata:
    """Create NGFF (OME-Zarr v0.5) metadata from metadata saved by the core engine.

    Parameters
    ----------
    summary_metadata : SummaryMetaV1
        Summary metadata containing acquisition information.
    frame_metadata_list : list[FrameMetaV1]
        List of frame metadata for each acquired image.

    Returns
    -------
    NGFFMetadata
        A container with `root` metadata (for top-level zarr.json) and `images`
        dict mapping paths to Image metadata (for child zarr.json files).

        - Single position: root=Image, images={}
        - Multi-position: root=Bf2Raw, images={"0": Image, "1": Image, ...}
        - Plate: root=Plate, images={"A/1/0": Image, ...}
    """
    image_infos = summary_metadata.get("image_infos", ())
    if not frame_metadata_list or not image_infos:
        # Return minimal valid Image with empty images dict
        return NGFFMetadata(root=_create_minimal_image(), images={})

    sequence = _extract_mda_sequence(summary_metadata, frame_metadata_list[0])

    # Check if this is a plate acquisition
    if (plate_plan := _extract_plate_plan(sequence)) is not None:
        return _build_ngff_plate(
            plate_plan=plate_plan,
            dimension_info=_extract_dimension_info(image_infos[0]),
            sequence=sequence,
            frame_metadata_list=frame_metadata_list,
        )
    else:
        # Build image metadata
        position_groups = _group_frames_by_position(frame_metadata_list)
        return _build_ngff_image(
            dimension_info=_extract_dimension_info(image_infos[0]),
            sequence=sequence,
            position_groups=position_groups,
        )


# =============================================================================
# Data Structures
# =============================================================================


class _DimensionInfo(NamedTuple):
    pixel_size_um: float
    dtype: str | None
    height: int
    width: int


class _PositionKey(NamedTuple):
    name: str | None
    p_index: int
    g_index: int | None = None

    def __str__(self) -> str:
        if self.g_index is not None:
            # if it has a name, include it in the position string before grid
            # (e.g. name_p0000_g0000)
            if self.name:
                return f"{self.name}_p{self.p_index:04d}_g{self.g_index:04d}"
            # otherwise just use p and g indices (e.g. p0000_g0000)
            return f"p{self.p_index:04d}_g{self.g_index:04d}"
        else:
            # if it has a name, include it in the position string (e.g. name_p0000)
            if self.name:
                return f"{self.name}_p{self.p_index:04d}"
            # otherwise just use p index (e.g. p0000)
            return f"p{self.p_index:04d}"

    @property
    def image_id(self) -> str:
        if self.g_index is not None:
            return f"{self.p_index}:{self.g_index}"
        return f"{self.p_index}"


# =============================================================================
# Metadata Extraction Functions
# =============================================================================


def _extract_dimension_info(
    image_info: ImageInfo,
) -> _DimensionInfo:
    """Extract pixel size (µm), data type, width, and height from image_infos."""
    return _DimensionInfo(
        pixel_size_um=image_info.get("pixel_size_um", 1.0),
        dtype=image_info.get("dtype", None),
        width=image_info.get("width", 0),
        height=image_info.get("height", 0),
    )


def _extract_mda_sequence(
    summary_metadata: SummaryMetaV1, single_frame_metadata: FrameMetaV1
) -> useq.MDASequence | None:
    """Extract the MDA sequence from summary metadata or frame metadata."""
    if (sequence_data := summary_metadata.get("mda_sequence")) is not None:
        return useq.MDASequence.model_validate(sequence_data)
    if (mda_event := _extract_mda_event(single_frame_metadata)) is not None:
        return mda_event.sequence
    return None


def _extract_mda_event(frame_metadata: FrameMetaV1) -> useq.MDAEvent | None:
    """Extract the useq.MDAEvent from frame metadata."""
    if (mda_event_data := frame_metadata.get("mda_event")) is not None:
        return useq.MDAEvent.model_validate(mda_event_data)
    return None  # pragma: no cover


def _extract_plate_plan(
    sequence: useq.MDASequence | None,
) -> useq.WellPlatePlan | None:
    """Extract the plate plan from the MDA sequence if it exists."""
    if sequence is None:  # pragma: no cover
        return None
    stage_positions = sequence.stage_positions
    if isinstance(stage_positions, useq.WellPlatePlan):
        return stage_positions
    return None


# =============================================================================
# Frame Grouping and Processing
# =============================================================================


def _group_frames_by_position(
    frame_metadata_list: list[FrameMetaV1],
) -> dict[_PositionKey, list[FrameMetaV1]]:
    """Reorganize frame metadata by stage position index in a dictionary.

    Handles the 'g' axis (grid) by converting it to separate positions,
    since OME-NGFF doesn't support the 'g' axis. Each grid position becomes
    a separate Image with names like "Pos0000_Grid0000".

    Returns
    -------
    dict[_PositionKey, list[FrameMetaV1]]
        mapping of position identifier to list of `FrameMetaV1`.
    """
    frames_by_position: dict[_PositionKey, list[FrameMetaV1]] = {}
    for frame_metadata in frame_metadata_list:
        if (mda_event := _extract_mda_event(frame_metadata)) is None:
            continue  # pragma: no cover

        p_index = mda_event.index.get(useq.Axis.POSITION, 0) or 0
        g_index = mda_event.index.get(useq.Axis.GRID, None)
        key = _PositionKey(mda_event.pos_name, p_index, g_index)
        frames_by_position.setdefault(key, []).append(frame_metadata)

    return frames_by_position


def _group_frames_by_well_field(
    frame_metadata_list: list[FrameMetaV1],
    plate_plan: useq.WellPlatePlan,
) -> dict[tuple[int, int, int], list[FrameMetaV1]]:
    """Group frame metadata by well (row, col) and field index.

    Returns
    -------
    dict[tuple[int, int, int], list[FrameMetaV1]]
        Mapping of (row_idx, col_idx, field_idx) to list of FrameMetaV1.
    """
    frames_by_field: dict[tuple[int, int, int], list[FrameMetaV1]] = {}

    for frame_metadata in frame_metadata_list:
        if (mda_event := _extract_mda_event(frame_metadata)) is None:
            continue  # pragma: no cover

        p_index = mda_event.index.get(useq.Axis.POSITION, 0) or 0

        # Find which well and field this position belongs to
        well_idx, field_idx = divmod(p_index, plate_plan.num_points_per_well)

        # Get row and column from well index
        if well_idx < len(plate_plan.selected_well_indices):
            row_idx, col_idx = plate_plan.selected_well_indices[well_idx][:2]
            key = (row_idx, col_idx, field_idx)
            frames_by_field.setdefault(key, []).append(frame_metadata)

    return frames_by_field


# =============================================================================
# Axis and Dimension Builders
# =============================================================================


def _build_axes(
    sequence: useq.MDASequence | None,
    dimension_info: _DimensionInfo,
) -> list[v05.TimeAxis | v05.ChannelAxis | v05.SpaceAxis]:
    """Build NGFF axes from sequence information."""
    axes: list[v05.TimeAxis | v05.ChannelAxis | v05.SpaceAxis] = []

    # Determine which axes are present
    has_time = False
    has_channel = False
    has_z = False

    if sequence is not None and not isinstance(sequence, GeneratorMDASequence):
        has_time = sequence.sizes.get("t", 0) > 1
        has_channel = sequence.sizes.get("c", 0) > 1
        has_z = sequence.sizes.get("z", 0) > 1

    # Add axes in standard order: t, c, z, y, x
    if has_time:
        axes.append(v05.TimeAxis(name="t", type="time", unit="millisecond"))
    if has_channel:
        axes.append(v05.ChannelAxis(name="c", type="channel"))
    if has_z:
        axes.append(v05.SpaceAxis(name="z", type="space", unit="micrometer"))

    # Always add y and x
    axes.extend(
        [
            v05.SpaceAxis(name="y", type="space", unit="micrometer"),
            v05.SpaceAxis(name="x", type="space", unit="micrometer"),
        ]
    )

    return axes


def _build_coordinate_transformations(
    dimension_info: _DimensionInfo,
    axes: list[v05.TimeAxis | v05.ChannelAxis | v05.SpaceAxis],
) -> list[v05.ScaleTransformation]:
    """Build coordinate transformations (scale) for the dataset."""
    # Build scale array matching the axes
    scale = []
    for axis in axes:
        if axis.name in ("x", "y"):
            scale.append(dimension_info.pixel_size_um)
        elif axis.name == "z":
            # Use pixel size for z as well (could be customized)
            scale.append(dimension_info.pixel_size_um)
        else:
            # Time and channel axes get scale of 1
            scale.append(1.0)

    return [v05.ScaleTransformation(type="scale", scale=scale)]


def _build_omero_metadata(
    sequence: useq.MDASequence | None,
) -> v05.Omero | None:
    """Build OMERO metadata with channel information."""
    if sequence is None or isinstance(sequence, GeneratorMDASequence):
        return None

    if not sequence.channels:
        return None

    channels = []
    for channel in sequence.channels:
        # Simple color assignment (could be enhanced)
        omero_channel = v05.OmeroChannel(
            label=channel.config,
            window=v05.OmeroWindow(min=0, max=65535, start=0, end=65535),
        )
        channels.append(omero_channel)

    return v05.Omero(channels=channels)


# =============================================================================
# NGFF Object Builders
# =============================================================================


def _create_minimal_image() -> v05.Image:
    """Create a minimal valid NGFF Image for empty metadata."""
    return v05.Image(
        version="0.5",
        multiscales=[
            v05.Multiscale(
                name="minimal",
                axes=[
                    v05.SpaceAxis(name="y", type="space"),
                    v05.SpaceAxis(name="x", type="space"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(type="scale", scale=[1.0, 1.0])
                        ],
                    )
                ],
            )
        ],
    )


def _build_ngff_image(
    dimension_info: _DimensionInfo,
    sequence: useq.MDASequence | None,
    position_groups: dict[_PositionKey, list[FrameMetaV1]],
) -> NGFFMetadata:
    """Build NGFF Image metadata from grouped frame metadata.

    For single-position acquisitions, returns NGFFMetadata with Image root.
    For multi-position acquisitions, returns NGFFMetadata with Bf2Raw root
    and dict of Image objects (following bioformats2raw layout).
    """
    # If we have multiple positions, create separate Image objects with Bf2Raw root
    if len(position_groups) > 1:
        images: dict[str, v05.Image] = {}
        # Sort position keys to ensure consistent ordering
        sorted_positions = sorted(
            position_groups.items(),
            key=lambda x: (x[0].p_index, x[0].g_index or 0)
        )

        for idx, (position_key, _frames) in enumerate(sorted_positions):
            axes = _build_axes(sequence, dimension_info)
            coord_transforms = _build_coordinate_transformations(dimension_info, axes)
            omero = _build_omero_metadata(sequence)

            # Create dataset entry (path to resolution level 0)
            datasets = [
                v05.Dataset(
                    path="0",
                    coordinateTransformations=cast(
                        "list[v05.ScaleTransformation | v05.TranslationTransformation]",
                        coord_transforms,
                    ),
                )
            ]

            # Create multiscale metadata with position-specific name
            multiscale = v05.Multiscale(
                name=str(position_key),
                axes=cast("list[v05.SpaceAxis | v05.TimeAxis | v05.ChannelAxis | v05.CustomAxis]", axes),  # noqa: E501
                datasets=datasets,
            )

            # Build Image for this position
            image = v05.Image(
                version="0.5",
                multiscales=[multiscale],
            )

            if omero is not None:
                image.omero = omero

            images[str(idx)] = image

        # Create Bf2Raw root metadata
        # Note: bioformats2raw_layout field has alias="bioformats2raw.layout"
        bf2raw_root = v05.Bf2Raw(
            version="0.5",
            **{"bioformats2raw.layout": 3},  # type: ignore[arg-type]
        )

        return NGFFMetadata(root=bf2raw_root, images=images)

    # Single position - return NGFFMetadata with Image root and empty images dict
    axes = _build_axes(sequence, dimension_info)
    coord_transforms = _build_coordinate_transformations(dimension_info, axes)
    omero = _build_omero_metadata(sequence)

    # Create dataset entry (path to resolution level 0)
    datasets = [
        v05.Dataset(
            path="0",
            coordinateTransformations=cast(
                "list[v05.ScaleTransformation | v05.TranslationTransformation]",
                coord_transforms,
            ),
        )
    ]

    # Create multiscale metadata
    multiscale = v05.Multiscale(
        name="image",
        axes=cast(
            "list[v05.SpaceAxis | v05.TimeAxis | v05.ChannelAxis | v05.CustomAxis]",
            axes,
        ),
        datasets=datasets,
    )

    # Build final Image
    image = v05.Image(
        version="0.5",
        multiscales=[multiscale],
    )

    if omero is not None:
        image.omero = omero

    return NGFFMetadata(root=image, images={})


def _build_ngff_plate(
    plate_plan: useq.WellPlatePlan,
    dimension_info: _DimensionInfo,
    sequence: useq.MDASequence | None,
    frame_metadata_list: list[FrameMetaV1],
) -> NGFFMetadata:
    """Build NGFF Plate metadata from a WellPlatePlan.
    
    Returns NGFFMetadata with Plate root and dict of Image objects
    for each field (well/field path like "A/1/0").
    """
    # Build plate rows and columns from selected wells
    # Get unique row and column indices from selected wells
    selected_rows = sorted({row for row, _, *_ in plate_plan.selected_well_indices})
    selected_cols = sorted({col for _, col, *_ in plate_plan.selected_well_indices})

    # Build row and column objects using standard naming
    # Rows: A, B, C, ... (using letters)
    # Columns: 01, 02, 03, ... (using zero-padded numbers)
    rows = [
        v05.Row(name=chr(65 + row_idx))  # 65 is ASCII 'A'
        for row_idx in selected_rows
    ]

    columns = [
        v05.Column(name=f"{col_idx + 1:02d}")  # Zero-padded column numbers
        for col_idx in selected_cols
    ]

    # Build wells - map from original indices to new indices
    row_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_rows)}
    col_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_cols)}

    wells: list[v05.PlateWell] = []
    for well_idx in plate_plan.selected_well_indices:
        row_idx, col_idx = well_idx[0], well_idx[1]
        row_name = chr(65 + row_idx)  # Convert to letter (A, B, C, ...)
        col_name = f"{col_idx + 1:02d}"  # Convert to zero-padded number (01, 02, ...)
        well_path = f"{row_name}/{col_name}"

        wells.append(
            v05.PlateWell(
                path=well_path,
                rowIndex=row_mapping[row_idx],
                columnIndex=col_mapping[col_idx],
            )
        )

    # Create plate definition
    plate_def = v05.PlateDef(
        name=plate_plan.plate.name,
        columns=columns,
        rows=rows,
        wells=wells,
        field_count=plate_plan.num_points_per_well,
    )

    plate_root = v05.Plate(
        version="0.5",
        plate=plate_def,
    )

    # Build Image objects for each field
    # Group frames by well and field
    images: dict[str, v05.Image] = {}
    field_groups = _group_frames_by_well_field(frame_metadata_list, plate_plan)

    for (row_idx, col_idx, field_idx), _frames in field_groups.items():
        row_name = chr(65 + row_idx)
        col_name = f"{col_idx + 1:02d}"
        field_path = f"{row_name}/{col_name}/{field_idx}"

        # Build Image metadata for this field
        axes = _build_axes(sequence, dimension_info)
        coord_transforms = _build_coordinate_transformations(dimension_info, axes)
        omero = _build_omero_metadata(sequence)

        datasets = [
            v05.Dataset(
                path="0",
                coordinateTransformations=cast(
                    "list[v05.ScaleTransformation | v05.TranslationTransformation]",
                    coord_transforms,
                ),
            )
        ]

        multiscale = v05.Multiscale(
            name=f"field_{field_idx}",
            axes=cast("list[v05.SpaceAxis | v05.TimeAxis | v05.ChannelAxis | v05.CustomAxis]", axes),  # noqa: E501
            datasets=datasets,
        )

        image = v05.Image(
            version="0.5",
            multiscales=[multiscale],
        )

        if omero is not None:
            image.omero = omero

        images[field_path] = image

    return NGFFMetadata(root=plate_root, images=images)
