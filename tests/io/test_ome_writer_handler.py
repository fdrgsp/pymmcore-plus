from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import useq

from pymmcore_plus.mda.handlers import OMEWriterHandler

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


SEQ = useq.MDASequence(
    channels=["DAPI", "FITC"],
    stage_positions=[(222, 1, 1), (111, 0, 0)],
    time_plan={"interval": 0.1, "loops": 2},
    z_plan={"range": 0.2, "step": 0.1},
    axis_order="tpcz",
)


@pytest.mark.parametrize(
    "path",
    [
        "output.tiff",
        "output.zarr",
        "imgseq",  # ImageSequenceWriter
    ],
)
def test_ome_writer_handler_str(
    path: str,
    tmp_path: Path,
    core: CMMCorePlus,
) -> None:
    """Test OMEWriterHandler with imputs as strings and None."""

    output = tmp_path / path

    core.mda.run(SEQ, output=output)

    if output.suffix == ".tiff":
        p0 = tmp_path / "output_p000.tiff"
        p1 = tmp_path / "output_p001.tiff"
        assert p0.is_file()
        assert p1.is_file()

    else:
        assert output.exists()
        assert output.is_dir()

        if output.suffix == ".zarr":
            assert (output / "0").exists()
            assert (output / "1").exists()
            assert (output / "zarr.json").is_file()
            assert (output / "meta.json").is_file()

        elif output.suffix == "":
            assert (output / "_frame_metadata.json").is_file()
            assert (output / "_useq_MDASequence.json").is_file()
            assert len(list(output.glob("*.tif"))) == 24


@pytest.mark.parametrize(
    "path",
    [
        "output.tiff",  # TIFF file
        "output.zarr",  # Zarr directory
        None,  # temp dir
    ],
)
def test_ome_writer_handler_object(
    path: str | None,
    tmp_path: Path,
    core: CMMCorePlus,
) -> None:
    """Test OMEWriterHandler with imputs as strings and None."""

    if path is None:
        writer = OMEWriterHandler(overwrite=True)  # temp dir
    else:
        writer = OMEWriterHandler(str(tmp_path / path), overwrite=True)

    core.mda.run(SEQ, output=writer)

    output_path = Path(writer.path)

    if path is not None and Path(path).suffix == ".tiff":
        p0 = tmp_path / "output_p000.tiff"
        p1 = tmp_path / "output_p001.tiff"
        assert p0.is_file()
        assert p1.is_file()
    else:
        assert output_path.exists()
        assert output_path.is_dir()

        if output_path.suffix == ".zarr":
            assert (output_path / "0").exists()
            assert (output_path / "1").exists()
            assert (output_path / "zarr.json").is_file()
            assert (output_path / "meta.json").is_file()

        elif output_path.suffix == "":
            assert (output_path / "_frame_metadata.json").is_file()
            assert (output_path / "_useq_MDASequence.json").is_file()
            assert len(list(output_path.glob("*.tif"))) == 24
