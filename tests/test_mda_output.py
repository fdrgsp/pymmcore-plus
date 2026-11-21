"""Tests for MDA runner output/sink integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import useq
from ome_writers import AcquisitionSettings
from useq import MDAEvent

from pymmcore_plus.mda._runner import MDARunner
from pymmcore_plus.mda._sink import OmeWritersSink
from pymmcore_plus.metadata import summary_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus

SUMMARY_META_KEYS = {
    "config_groups",
    "datetime",
    "devices",
    "format",
    "image_infos",
    "mda_sequence",
    "pixel_size_configs",
    "position",
    "system_info",
    "version",
}


# --- _OmeWritersSink unit tests ---


@pytest.fixture
def zarr_sink(tmp_path: Path) -> OmeWritersSink:
    return OmeWritersSink.from_output(tmp_path / "out.ome.zarr")


def test_from_output_path(tmp_path: Path) -> None:
    dest = tmp_path / "out.ome.zarr"
    sink = OmeWritersSink.from_output(dest)
    assert sink._settings.root_path == str(dest)


def test_from_output_scratch() -> None:
    sink = OmeWritersSink.from_output("scratch")
    assert sink._settings.format.name == "scratch"
    sink = OmeWritersSink.from_output("memory")
    assert sink._settings.format.name == "scratch"


def test_from_output_acquisition_settings() -> None:
    settings = AcquisitionSettings(root_path="/tmp/test.ome.zarr")
    sink = OmeWritersSink.from_output(settings)
    assert sink._settings == settings


def test_sink_get_close_before_setup(zarr_sink: OmeWritersSink) -> None:
    assert zarr_sink.get_view() is None
    zarr_sink.close()  # should not raise


def test_sink_skip_delegates_to_stream() -> None:
    sink = OmeWritersSink(AcquisitionSettings(root_path="/tmp/x.ome.zarr"))
    mock_stream = Mock()
    sink._stream = mock_stream
    sink.skip(frames=3)
    mock_stream.skip.assert_called_once_with(frames=3)


@pytest.mark.parametrize("store_events", [False, True])
def test_sink_frame_metadata_records_appends_and_skips(store_events: bool) -> None:
    sink = OmeWritersSink(AcquisitionSettings(root_path="/tmp/x.ome.zarr"))
    sink._stream = Mock()
    sink._store_events = store_events
    meta: Any = {"runner_time_ms": 1500, "exposure_ms": 10, "mda_event": MDAEvent()}

    sink.append(Mock(), MDAEvent(), meta)
    sink.skip(frames=2)
    sink.append(Mock(), MDAEvent(), {**meta, "runner_time_ms": 2500})

    first, *skipped, last = sink.frame_metadata
    assert skipped == [None, None]
    assert (first["delta_t"], last["delta_t"]) == (1.5, 2.5)
    assert ("mda_event" in first) is store_events
    # each entry is exactly what the stream was given
    written = [c.kwargs["frame_metadata"] for c in sink._stream.append.call_args_list]
    assert written == [first, last]


def test_sink_frame_metadata_skips_failed_writes() -> None:
    sink = OmeWritersSink(AcquisitionSettings(root_path="/tmp/x.ome.zarr"))
    sink._stream = Mock()
    sink._stream.append.side_effect = RuntimeError
    sink._stream.skip.side_effect = RuntimeError
    meta: Any = {"runner_time_ms": 0, "exposure_ms": 10}

    with pytest.raises(RuntimeError):
        sink.append(Mock(), MDAEvent(), meta)
    with pytest.raises(RuntimeError):
        sink.skip()
    assert sink.frame_metadata == []


def test_sink_setup_starts_a_new_frame_metadata_list(core: CMMCorePlus) -> None:
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    core.mda.run(seq, output="memory")
    sink = core.mda.get_sink()
    assert isinstance(sink, OmeWritersSink)
    first_run = sink.frame_metadata
    assert len(first_run) == 2

    sink.setup(seq, summary_metadata(core, mda_sequence=seq))
    assert sink.frame_metadata == []
    assert len(first_run) == 2
    sink.close()


# --- MDARunner integration tests ---


def test_get_view_no_sink() -> None:
    assert MDARunner().get_view() is None


def test_release_sink(core: CMMCorePlus) -> None:
    import gc
    import weakref

    runner = core.mda
    assert not runner.release_sink()  # nothing to release yet

    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    runner.run(seq, output="memory")
    sink = runner.get_sink()
    assert sink is not None

    # a sink that isn't the current one is never released
    assert not runner.release_sink(Mock())
    assert runner.get_sink() is sink

    ref = weakref.ref(sink)
    assert runner.release_sink(sink)
    assert runner.get_sink() is None
    assert runner.get_view() is None
    del sink
    gc.collect()
    assert ref() is None


def test_release_sink_while_running(core: CMMCorePlus) -> None:
    runner = core.mda
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    released: list[bool] = []
    runner.events.frameReady.connect(lambda *_: released.append(runner.release_sink()))
    runner.run(seq, output="memory")
    assert released == [False, False]
    assert runner.get_sink() is not None


def test_run_with_zarr_output(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = core.mda
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))

    out = tmp_path / "test.ome.zarr"
    runner.run(seq, output=out)
    view = runner.get_view()
    assert view is not None
    assert view.shape[:-2] == (2,)
    assert out.exists()

    # Summary metadata is written by ome-writers' set_global_metadata under
    # the "pymmcore_plus" namespace. For single-position the parent root
    # group IS the image group, so it lives in <root>/zarr.json directly.
    # The summary dict is wrapped under a "summary_metadata" sub-key so the
    # namespace stays available for other pymmcore-plus data.
    root_json = json.loads((out / "zarr.json").read_text())
    summary = root_json["attributes"]["pymmcore_plus"]["summary_metadata"]
    assert SUMMARY_META_KEYS <= set(summary)


def test_run_with_zarr_output_multiposition(core: CMMCorePlus, tmp_path: Path) -> None:
    """Multi-position writes summary metadata exactly once at the root group.

    Regression test: previously the sink looped over every per-position group
    and stamped the same summary blob into each one. With set_global_metadata
    the payload lives in the parent root group's zarr.json only.
    """
    runner = core.mda
    seq = useq.MDASequence(
        stage_positions=useq.WellPlatePlan(
            plate="96-well",
            a1_center_xy=(0, 0),
            selected_wells=[(0, 0), (1, 0)],
        ),
        time_plan=useq.TIntervalLoops(interval=0, loops=2),
    )
    out = tmp_path / "multi.ome.zarr"
    runner.run(seq, output=out)
    assert out.exists()

    # Parent root group has the summary, wrapped under summary_metadata.
    root_json = json.loads((out / "zarr.json").read_text())
    summary = root_json["attributes"]["pymmcore_plus"]["summary_metadata"]
    assert SUMMARY_META_KEYS <= set(summary)

    # No per-position group carries the namespace.
    leaked = [
        p
        for p in out.rglob("zarr.json")
        if p != out / "zarr.json"
        and "pymmcore_plus" in json.loads(p.read_text()).get("attributes", {})
    ]
    assert leaked == [], f"summary metadata leaked into: {leaked}"


def test_run_with_tiff_output(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = core.mda
    out = tmp_path / "test.ome.tiff"
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    runner.run(seq, output=out)
    assert out.exists()

    # ome-writers set_global_metadata writes a single MapAnnotation for the
    # namespace, with each top-level key of the payload mapping becoming a
    # `Map` entry whose value is the JSON-encoded value. We wrap the summary
    # under "summary_metadata" so the pymmcore_plus namespace stays available
    # for other pymmcore-plus data.
    import ome_types

    ome = ome_types.from_tiff(str(out))
    sa = ome.structured_annotations
    assert sa is not None
    pmc = [m for m in sa.map_annotations if m.namespace == "pymmcore_plus"]
    assert len(pmc) == 1
    entries = {e.k: e.value for e in pmc[0].value.ms}
    assert "summary_metadata" in entries
    summary = json.loads(entries["summary_metadata"])
    assert SUMMARY_META_KEYS <= set(summary)

    # The summary annotation must not be referenced by any plane: it lives at
    # the OME root, distinguishing it from per-frame metadata.
    summary_id = pmc[0].id
    for image in ome.images:
        for plane in image.pixels.planes:
            assert all(ref.id != summary_id for ref in plane.annotation_refs)


def test_run_with_tiff_output_multiposition(core: CMMCorePlus, tmp_path: Path) -> None:
    """Multi-position OME-TIFF in default 'redundant' mode writes to every file.

    In ``redundant`` mode each per-position TIFF carries a full independent
    OME model by definition, so ome-writers fans the summary MapAnnotation
    out to every file. Each file is independently openable with complete
    acquisition metadata.
    """
    import ome_types

    runner = core.mda
    seq = useq.MDASequence(
        stage_positions=useq.WellPlatePlan(
            plate="96-well",
            a1_center_xy=(0, 0),
            selected_wells=[(0, 0), (1, 0)],
        ),
        time_plan=useq.TIntervalLoops(interval=0, loops=2),
    )
    out = tmp_path / "multi.ome.tiff"
    runner.run(seq, output=out)

    # Multi-file OME-TIFF strips the suffix and writes per-position files
    # into a directory at the stem path.  Plate positions are prefixed with
    # their well; `p###` is the acquisition order (A2 is visited first).
    out_dir = tmp_path / "multi"
    tiffs = sorted(out_dir.glob("*.ome.tiff"))
    assert [p.name for p in tiffs] == [
        "multi_A1_p001.ome.tiff",
        "multi_A2_p000.ome.tiff",
    ]

    # Every per-position file carries the full summary MapAnnotation.
    for tiff_path in tiffs:
        ome = ome_types.from_tiff(str(tiff_path))
        sa = ome.structured_annotations
        assert sa is not None
        pmc = [m for m in sa.map_annotations if m.namespace == "pymmcore_plus"]
        assert len(pmc) == 1, f"expected one summary annotation in {tiff_path.name}"
        entries = {e.k: e.value for e in pmc[0].value.ms}
        assert "summary_metadata" in entries
        summary = json.loads(entries["summary_metadata"])
        assert SUMMARY_META_KEYS <= set(summary)


def test_run_with_acquisition_settings(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = core.mda
    dest = tmp_path / "out.ome.zarr"
    settings = AcquisitionSettings(root_path=str(dest), overwrite=True)
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    runner.run(seq, output=settings)
    assert dest.exists()


def test_coerce_outputs_rejects_multiple_sinks() -> None:
    runner = MDARunner()
    with pytest.raises(NotImplementedError, match="Only one"):
        runner._coerce_outputs(["/a.ome.zarr", "/b.ome.zarr"])


def test_coerce_outputs_rejects_bad_handler() -> None:
    runner = MDARunner()
    with pytest.raises(TypeError, match="frameReady"):
        runner._coerce_outputs([object()])  # type: ignore[list-item]


def test_run_with_custom_handler_and_path(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = core.mda
    out = tmp_path / "test.ome.zarr"

    handler = Mock()
    handler.frameReady = Mock()

    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=2))
    runner.run(seq, output=[out, handler])

    assert handler.frameReady.call_count == 2
    assert out.exists()


@pytest.mark.parametrize("fmt", ["scratch", "ome-zarr", "ome-tiff"])
def test_run_with_ragged_sequence(core: CMMCorePlus, tmp_path: Path, fmt: str) -> None:
    """Ragged sequences (unsupported by ome-writers) fall back to unbounded 3D."""
    # do_stack=False on one channel with a z_plan creates a ragged dimension
    seq = useq.MDASequence(
        channels=[
            {"config": "DAPI", "exposure": 1},
            {"config": "FITC", "exposure": 1, "do_stack": False},
        ],
        z_plan={"range": 3, "step": 1},
    )

    out = AcquisitionSettings(root_path=str(tmp_path / "test"), format=fmt)

    core.mda.run(seq, output=out)
    view = core.mda.get_view()
    assert view is not None
    # unbounded 3D fallback: each event becomes one "time" frame
    assert view.shape[:-2] == (5,)  # (4 planes for one channel + 1 for the other)


def test_run_with_event_iterator(core: CMMCorePlus) -> None:
    """A plain iterator of MDAEvents (non-deterministic) saves data."""
    from useq import MDAEvent

    def event_generator() -> Iterator[MDAEvent]:
        for i in range(3):
            yield MDAEvent(metadata={"frame": i})

    core.mda.run(event_generator(), output="scratch")

    view = core.mda.get_view()
    assert view is not None
    assert view.shape[:-2] == (3,)


def test_run_with_dimension_overrides(core: CMMCorePlus, tmp_path: Path) -> None:
    """dimension_overrides carries through to the written dimensions."""
    seq = useq.MDASequence(z_plan=useq.ZRangeAround(range=4, step=1))
    dest = tmp_path / "test.ome.zarr"
    core.mda.run(seq, output=dest, dimension_overrides={"z": {"chunk_size": 5}})
    metadata = json.loads((dest / "0" / "zarr.json").read_bytes())
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"][0] == 5
