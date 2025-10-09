from pathlib import Path

import numpy as np
import useq
from ome_writers import OMEStream, create_stream, dims_from_useq

from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1, create_ome_metadata


class MMWriter:

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._path = Path(path)

        # TODO: probably better to store them to disk as json files
        self._summary_meta: SummaryMetaV1 = {}  # type: ignore
        self._frame_meta_list: list[FrameMetaV1] = []

        self._stream: OMEStream | None = None

    def sequenceStarted(
        self, sequence: useq.MDASequence, summary_meta: SummaryMetaV1
    ) -> None:
        self._clear()

        self._summary_meta = summary_meta

        # using try/except to be a bit faster than doing individual .get() and checking
        # for None (it's ~1.5x faster)
        try:
            image_infos = summary_meta["image_infos"][0]
            image_w, image_h = image_infos["width"], image_infos["height"]
            dtype = image_infos["dtype"]
        except Exception as e:
            raise ValueError(
                f"Missing image width, height, or dtype information in {summary_meta}:"
                f" {e}"
            ) from e

        dims = dims_from_useq(sequence, image_w, image_h)

        # backend 'auto' with a '.zarr' extension will automatically use 'acquire-zarr'
        # and with a '.tiff' will use 'tifffile'
        self._stream = create_stream(self._path, dtype, dims, overwrite=True)

    def frameReady(
        self, frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
    ) -> None:
        if self._stream is None:
            return
        self._stream.append(frame)
        self._frame_meta_list.append(frame_meta)

    def sequenceFinished(self, sequence: useq.MDASequence) -> None:
        if self._stream is None:
            return
        self._stream.flush()

        # TODO: once implemented in ome-writers, update the metadata
        if hasattr(self._stream, "update_ome_metadata"):
            ome = create_ome_metadata(self._summary_meta, self._frame_meta_list)
            self._stream.update_ome_metadata(ome)

    def _clear(self) -> None:
        self._summary_meta = {}  # type: ignore
        self._frame_meta_list = []
        self._stream = None
