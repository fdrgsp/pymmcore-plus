from pathlib import Path

import numpy as np
import useq
from ome_writers import OMEStream, create_stream, dims_from_useq

from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1


class ZarrNGFFWriter:

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

        if (image_infos := summary_meta.get("image_infos")) is None:
            raise ValueError("No image infos found in summary meta")

        image_w, image_h = image_infos[0].get("width"), image_infos[0].get("height")
        if image_w is None or image_h is None:
            raise ValueError("No width/height found in image infos")

        dtype = image_infos[0].get("dtype")
        if dtype is None:
            raise ValueError("No dtype found in image infos")

        dims = dims_from_useq(sequence, image_w, image_h)

        # backend 'auto' with a '.zarr' extension will automatically use 'acquire-zarr'
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
        # ome = create_ome_metadata(self._summary_meta, self._frame_meta_list)
        # self._stream.update_ome_metadata(ome)

    def _clear(self) -> None:
        self._summary_meta = {}  # type: ignore
        self._frame_meta_list = []
        self._stream = None


from pymmcore_plus import CMMCorePlus

mmc = CMMCorePlus.instance()
mmc.loadSystemConfiguration("/Users/fdrgsp/Desktop/test_config.cfg")

wrt = ZarrNGFFWriter("/Users/fdrgsp/Desktop/t/z.zarr")
mmc.mda.events.sequenceStarted.connect(wrt.sequenceStarted)

seq = useq.MDASequence(
    channels=["DAPI", "FITC"],
)
mmc.mda.run(seq)
