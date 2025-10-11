#!/usr/bin/env python3

from datetime import timedelta

import pytest
import useq
from yaozarrs import v05

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.metadata._ngff import create_ngff_metadata

BASIC_SEQ = useq.MDASequence(
    time_plan=useq.TIntervalLoops(interval=timedelta(milliseconds=500), loops=2),
    stage_positions=(
        useq.AbsolutePosition(x=100, y=100, name="FirstPosition"),
        useq.AbsolutePosition(x=200, y=200, name="SecondPosition"),
    ),
    z_plan=useq.ZRangeAround(range=3.0, step=1.0),
    channels=(
        useq.Channel(config="DAPI", exposure=20),
        useq.Channel(config="FITC", exposure=30),
    ),
)

PLATE_SEQ = useq.MDASequence(
    axis_order=("p", "c", "z"),
    stage_positions=useq.WellPlatePlan(
        plate=useq.WellPlate.from_str("96-well"),
        a1_center_xy=(0, 0),
        selected_wells=((0, 0, 0), (0, 1, 2)),
    ),
    z_plan=useq.ZRangeAround(range=3.0, step=1.0),
    channels=(
        useq.Channel(config="DAPI", exposure=20),
        useq.Channel(config="FITC", exposure=30),
    ),
)

GRID_SEQ = useq.MDASequence(
    time_plan=useq.TIntervalLoops(interval=timedelta(milliseconds=500), loops=2),
    stage_positions=(
        useq.AbsolutePosition(x=100, y=100, name="FirstPosition"),
        useq.AbsolutePosition(x=200, y=200),
    ),
    channels=(
        useq.Channel(config="DAPI", exposure=20),
        useq.Channel(config="FITC", exposure=30),
    ),
    grid_plan=useq.GridRowsColumns(rows=2, columns=2),
)


@pytest.mark.parametrize("seq", [BASIC_SEQ, GRID_SEQ])
def test_ngff_image_generation(seq: useq.MDASequence) -> None:
    """Test NGFF Image metadata generation from basic sequences."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")
    mmc.setConfig("Objective", "20X")  # px size 0.5 µm
    mmc.setROI(0, 0, 100, 200)

    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)

    # Should return an Image object for non-plate sequences
    assert isinstance(ngff, v05.Image)
    assert ngff.version == "0.5"
    assert len(ngff.multiscales) > 0

    # Check that we have the expected axes
    multiscale = ngff.multiscales[0]
    assert len(multiscale.axes) >= 2  # At least x and y
    assert multiscale.axes[-1].name == "x"
    assert multiscale.axes[-2].name == "y"

    # Check datasets
    assert len(multiscale.datasets) > 0
    dataset = multiscale.datasets[0]
    assert dataset.path == "0"
    assert len(dataset.coordinateTransformations) > 0


def test_ngff_plate_generation() -> None:
    """Test NGFF Plate metadata generation from plate sequences."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")
    mmc.setConfig("Objective", "20X")  # px size 0.5 µm
    mmc.setROI(0, 0, 100, 200)

    seq = PLATE_SEQ
    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)

    # Should return a Plate object for plate sequences
    assert isinstance(ngff, v05.Plate)
    assert ngff.version == "0.5"
    assert ngff.plate is not None

    # Check plate structure
    plate_def = ngff.plate
    assert len(plate_def.rows) > 0
    assert len(plate_def.columns) > 0
    assert len(plate_def.wells) > 0

    # Validate row and column names are alphanumeric
    for row in plate_def.rows:
        assert row.name.isalnum()
    for col in plate_def.columns:
        assert col.name.replace("_", "").isalnum() or col.name.isdigit()

    # Validate well paths
    for well in plate_def.wells:
        assert "/" in well.path
        assert well.rowIndex >= 0
        assert well.columnIndex >= 0


def test_ngff_empty_metadata() -> None:
    """Test NGFF generation with empty metadata."""
    ngff = create_ngff_metadata({}, [])  # type: ignore

    # Should return a minimal valid Image
    assert isinstance(ngff, v05.Image)
    assert ngff.version == "0.5"
    assert len(ngff.multiscales) > 0


def test_ngff_axes_construction() -> None:
    """Test that NGFF axes are constructed correctly."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")
    mmc.setConfig("Objective", "20X")

    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=timedelta(milliseconds=500), loops=2),
        z_plan=useq.ZRangeAround(range=2.0, step=1.0),
        channels=(
            useq.Channel(config="DAPI", exposure=20),
            useq.Channel(config="FITC", exposure=30),
        ),
    )

    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)
    assert isinstance(ngff, v05.Image)

    multiscale = ngff.multiscales[0]
    axis_names = [axis.name for axis in multiscale.axes]

    # Should have t, c, z, y, x axes for this sequence
    assert "t" in axis_names
    assert "c" in axis_names
    assert "z" in axis_names
    assert "y" in axis_names
    assert "x" in axis_names

    # Check axis order (should be t, c, z, y, x)
    assert axis_names == ["t", "c", "z", "y", "x"]


def test_ngff_omero_metadata() -> None:
    """Test that OMERO metadata is included when channels are present."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")
    mmc.setConfig("Objective", "20X")

    seq = useq.MDASequence(
        channels=(
            useq.Channel(config="DAPI", exposure=20),
            useq.Channel(config="FITC", exposure=30),
        ),
    )

    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)
    assert isinstance(ngff, v05.Image)

    # Check OMERO metadata
    assert ngff.omero is not None
    assert len(ngff.omero.channels) == 2

    # Check channel labels
    channel_labels = [ch.label for ch in ngff.omero.channels]
    assert "DAPI" in channel_labels
    assert "FITC" in channel_labels


def test_ngff_coordinate_transformations() -> None:
    """Test that coordinate transformations include pixel size."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")
    mmc.setConfig("Objective", "20X")  # px size 0.5 µm

    seq = useq.MDASequence(
        channels=(useq.Channel(config="DAPI", exposure=20),),
    )

    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)
    assert isinstance(ngff, v05.Image)

    multiscale = ngff.multiscales[0]
    dataset = multiscale.datasets[0]
    transforms = dataset.coordinateTransformations

    # Should have at least one scale transformation
    assert len(transforms) > 0
    scale_transform = transforms[0]
    assert scale_transform.type == "scale"

    # Check that pixel size is applied to spatial dimensions
    # The scale should match the axes (c, y, x for this simple sequence)
    # Spatial axes (x, y) should have pixel_size_um value
    scale = scale_transform.scale
    assert scale[-1] == 0.5  # x axis (pixel_size_um from 20X objective)
    assert scale[-2] == 0.5  # y axis


def test_ngff_model_validation() -> None:
    """Test that generated NGFF metadata validates against yaozarrs models."""
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration("tests/local_config.cfg")

    seq = useq.MDASequence(
        channels=(useq.Channel(config="DAPI", exposure=20),),
    )

    engine = mmc.mda.engine
    assert engine is not None
    summary_meta = engine.get_summary_metadata(seq)
    frame_meta_list = [
        engine.get_frame_metadata(event, runner_time_ms=idx * 500)
        for idx, event in enumerate(seq)
    ]

    ngff = create_ngff_metadata(summary_meta, frame_meta_list)

    # Should be able to convert to dict and validate
    ngff_dict = ngff.model_dump(by_alias=True, exclude_none=True)
    assert "version" in ngff_dict
    assert ngff_dict["version"] == "0.5"

    # Should be able to re-create from dict
    if isinstance(ngff, v05.Image):
        recreated = v05.Image.model_validate(ngff_dict)
        assert recreated.version == ngff.version
    elif isinstance(ngff, v05.Plate):
        recreated = v05.Plate.model_validate(ngff_dict)
        assert recreated.version == ngff.version
