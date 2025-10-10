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
    ("path", "use_handler_object"),
    [
        ("output.tiff", False),  # TIFF file as string
        ("output.zarr", False),  # Zarr directory as string
        ("imgseq", False),  # ImageSequenceWriter as string
        ("output.tiff", True),  # TIFF file as OMEWriterHandler
        ("output.zarr", True),  # Zarr directory as OMEWriterHandler
        (None, True),  # temp dir as OMEWriterHandler
    ],
)
def test_ome_writer_handler(
    path: str | None,
    use_handler_object: bool,
    tmp_path: Path,
    core: CMMCorePlus,
) -> None:
    """Test OMEWriterHandler with paths as strings or OMEWriterHandler objects."""

    if use_handler_object:
        if path is None:
            output = OMEWriterHandler(overwrite=True)  # temp dir
        else:
            output = OMEWriterHandler(str(tmp_path / path), overwrite=True)
        core.mda.run(SEQ, output=output)
        output_path = Path(output.path)
    else:
        # path is always a string when use_handler_object is False
        assert path is not None
        output_path = tmp_path / path
        core.mda.run(SEQ, output=output_path)

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
