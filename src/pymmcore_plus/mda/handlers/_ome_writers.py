from __future__ import annotations

import atexit
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import useq
from ome_writers import create_stream, dims_from_useq

from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1, create_ome_metadata
from pymmcore_plus.metadata._ome import _extract_dimension_info

# Pre-import tifffile to avoid atexit registration issues during thread execution
try:
    # The tifffile import happens lazily when ome_writers creates TifffileStream
    # instances in threads. Pre-importing it here prevents the "can't register
    # atexit after shutdown" error.
    import tifffile  # noqa: F401
except ImportError:
    # tifffile may not be available depending on optional dependencies
    pass

if TYPE_CHECKING:
    from datetime import timedelta

    import numpy as np
    import useq
    from ome_writers import BackendName

    from pymmcore_plus.metadata.schema import FrameMetaV1, SummaryMetaV1


class OMEWriterHandler:
    """A handler to write images and metadata to OME-Zarr or OME-TIFF format.

    Parameters.
    ----------
    path : str
        Path to the output file or directory.
    backend : Literal["acquire-zarr", "tensorstore", "tiff", "auto"], optional
        The backend to use for writing the data. Options are:
        - "acquire-zarr": Use acquire-zarr backend.
        - "tensorstore": Use tensorstore backend.
        - "tiff": Use tifffile backend.
        - "auto": Automatically determine the backend based on the file extension.
        Default is "auto".
    overwrite : bool, optional
        Whether to overwrite existing files or directories. Default is False.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        backend: Literal[BackendName, "auto"] = "auto",
        overwrite: bool = False,
    ) -> None:
        if path is None:
            path = self._tmp_dir()

        self.path = path
        self.backend: Literal[BackendName, "auto"] = backend
        self.overwrite = overwrite

        self._summary_metadata: SummaryMetaV1 | None = None
        self._frame_metadatas: list[FrameMetaV1] = []

    def sequenceStarted(
        self, sequence: useq.MDASequence, summary_meta: SummaryMetaV1
    ) -> None:
        self._summary_metadata = summary_meta
        self._frame_metadatas.clear()

        z_step = abs(getattr(sequence.z_plan, "step", 1.0))
        if interval := getattr(sequence.time_plan, "interval", None):
            t_step = cast("timedelta", interval).total_seconds()
        else:
            t_step = 1.0

        dim_info = _extract_dimension_info(summary_meta["image_infos"][0])
        self.stream = create_stream(
            self.path,
            dtype=dim_info.dtype,
            dimensions=dims_from_useq(
                sequence,
                image_width=dim_info.width,
                image_height=dim_info.height,
                units={
                    "t": (t_step, "s"),
                    "z": (z_step, "um"),
                    "y": (dim_info.pixel_size_um, "um"),
                    "x": (dim_info.pixel_size_um, "um"),
                },
            ),
            backend=self.backend,
            overwrite=self.overwrite,
        )

    def frameReady(
        self, frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
    ) -> None:
        self.stream.append(frame)
        self._frame_metadatas.append(frame_meta)

    def sequenceFinished(self, sequence: useq.MDASequence) -> None:
        self.stream.flush()

        if self._summary_metadata is None:
            return

        ome = create_ome_metadata(self._summary_metadata, self._frame_metadatas)
        # TODO: update when all backends implement the `update_ome_metadata` method
        if hasattr(self.stream, "update_ome_metadata"):
            self.stream.update_ome_metadata(ome)
        else:
            # since acquire-zarr and tensorstore backends do not have the
            # `update_ome_metadata` method yet, we write the metadata as OME JSON
            # to a separate file for now.
            with contextlib.suppress(Exception):
                out = Path(self.path) / "meta.json"
                out.write_text(ome.model_dump_json(indent=2, exclude_unset=True))

    @staticmethod
    def _tmp_dir() -> str:
        """Create a temporary directory for storing OME files.

        Used when no path is provided.
        """
        path = tempfile.mkdtemp(suffix=".ome.zarr", prefix="pymmcore_zarr_")

        @atexit.register
        def _cleanup_temp_dir() -> None:
            with contextlib.suppress(Exception):
                shutil.rmtree(path)

        return path
