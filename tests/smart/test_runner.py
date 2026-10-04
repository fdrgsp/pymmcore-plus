"""End-to-end smart runs on the demo core, on both signal backends."""

from __future__ import annotations

import glob
import json
import textwrap
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import useq

from pymmcore_plus.smart import SmartRunConfig, SmartRunError, SmartRunner

if TYPE_CHECKING:
    import numpy as np

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1

EXAMPLES = Path(__file__).parents[2] / "examples" / "smart_microscopy"
MODES = ["thread", "process"]
SCAN = useq.MDASequence(
    stage_positions=[(0, 0, 0), (100, 100, 0), (200, 200, 0)],
    channels=["DAPI"],
)
ONE = useq.MDASequence(channels=["DAPI"])
ZOOM = "Nikon 40X Plan Fluor ELWD"
SURVEY = "Nikon 10X S Fluor"


def _script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "script.py"
    path.write_text(textwrap.dedent(source))
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(
    core: CMMCorePlus,
    base: useq.MDASequence,
    script: Path,
    run_dir: Path | None = None,
    **options: Any,
) -> dict[str, Any]:
    runner = SmartRunner(core)
    summary = runner.run(
        base, script, output="memory", run_dir=run_dir, timeout=60, **options
    )
    assert summary is not None, "run did not finish"
    return summary


@pytest.mark.parametrize("mode", MODES)
def test_adaptive_exposure_is_purely_reactive(
    core: CMMCorePlus, tmp_path: Path, mode: str
) -> None:
    summary = _run(
        core,
        ONE,
        EXAMPLES / "adaptive_exposure.py",
        tmp_path / "run",
        execution=mode,
        params={"n_frames": 4, "target_mean": 100.0},
    )
    assert summary["status"] == "stopped_by_script"
    assert summary["frames"] == 4
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    assert [f["origin"] for f in frames] == ["base"] + ["analysis"] * 3
    assert [f["parent_frame_id"] for f in frames] == [None, 0, 1, 2]
    assert len({f["exposure_ms"] for f in frames}) > 1

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "stopped_by_script"
    assert run["execution"] == mode
    assert run["counts"]["frames"] == 4
    assert run["system"]["pixel_configs"]
    assert (tmp_path / "run" / "script.py").read_text() == (
        EXAMPLES / "adaptive_exposure.py"
    ).read_text()
    view = core.mda.get_view()
    assert view is not None and view.shape[0] == 4


def test_no_run_dir_writes_nothing(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    before = set(tmp_path.iterdir())
    runner = SmartRunner(core)
    summary = runner.run(ONE, script, timeout=30)
    assert summary is not None and summary["run_dir"] is None
    assert runner.run_dir is None
    assert set(tmp_path.iterdir()) == before


def test_auto_run_dir_sits_beside_the_data(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    runner = SmartRunner(core)
    summary = runner.run(
        ONE, script, output=str(tmp_path / "exp.ome.zarr"), run_dir="auto", timeout=30
    )
    assert summary is not None
    assert summary["run_dir"] == str(tmp_path / "exp_smart")
    assert (tmp_path / "exp_smart" / "frames.jsonl").exists()


@pytest.mark.parametrize("mode", MODES)
def test_detect_and_act_z_stack(core: CMMCorePlus, tmp_path: Path, mode: str) -> None:
    summary = _run(
        core,
        SCAN,
        EXAMPLES / "detect_and_act.py",
        tmp_path / "run",
        execution=mode,
        params={
            "action": "Z-stack",
            "threshold": 0.0,
            "max_followups": 1,
            "center_on_hit": False,
            "z_range_um": 2.0,
            "z_step_um": 1.0,
        },
    )
    assert summary["status"] == "completed"
    stack = [
        f
        for f in _lines(tmp_path / "run" / "frames.jsonl")
        if f["origin"] == "analysis"
    ]
    assert sorted(f["event"]["z_pos"] for f in stack) == [-1.0, 0.0, 1.0]
    assert {f["parent_frame_id"] for f in stack} == {0}


def test_detect_and_act_zoom_restores_scan_objective(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    core.setProperty("Objective", "Label", SURVEY)
    summary = _run(
        core,
        SCAN,
        EXAMPLES / "detect_and_act.py",
        tmp_path / "run",
        sync="blocking",
        params={
            "action": "Higher magnification + z-stack",
            "threshold": 0.0,
            "max_followups": 1,
            "center_on_hit": False,
            "z_range_um": 2.0,
            "z_step_um": 1.0,
            "scan_objective": SURVEY,
            "zoom_objective": ZOOM,
        },
    )
    assert summary["status"] == "completed"
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    # the two objective switches took no image
    assert [f["origin"] for f in frames] == ["base"] + ["analysis"] * 3 + ["base"] * 2
    assert [f["pixel_size_um"] for f in frames] == [1.0, 0.25, 0.25, 0.25, 1.0, 1.0]
    assert core.getProperty("Objective", "Label") == SURVEY


SURVEY_GRID = useq.MDASequence(
    channels=["DAPI"], grid_plan=useq.GridRowsColumns(rows=2, columns=2)
)


def test_survey_and_target_two_phases(core: CMMCorePlus, tmp_path: Path) -> None:
    core.setProperty("Objective", "Label", SURVEY)
    summary = _run(
        core,
        SURVEY_GRID,
        EXAMPLES / "survey_and_target.py",
        tmp_path / "run",
        params={"threshold": 3000.0, "max_targets": 2, "grid_rows": 1},
    )
    assert summary["status"] == "completed"
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    targets = [f for f in frames if f["origin"] == "analysis"]
    assert [f["origin"] for f in frames[:4]] == ["base"] * 4
    assert len(targets) == 4  # 2 objects x (1 x 2 grid)
    analysis = _lines(tmp_path / "run" / "analysis.jsonl")
    after = [a for a in analysis if a["call"] == "after_base"]
    assert len(after) == 1 and after[0]["records"]["objects"] >= 2
    # the target grid was sized from the survey pixel size: 512 um tiles
    xs = sorted(f["event"]["x_pos"] for f in targets[:2])
    assert xs[1] - xs[0] == pytest.approx(512 * 0.9)  # 10% overlap


def test_survey_and_target_at_zoom_objective(core: CMMCorePlus, tmp_path: Path) -> None:
    core.setProperty("Objective", "Label", SURVEY)
    summary = _run(
        core,
        SURVEY_GRID,
        EXAMPLES / "survey_and_target.py",
        tmp_path / "run",
        params={
            "threshold": 3000.0,
            "max_targets": 1,
            "grid_rows": 1,
            "switch_objective": True,
            "survey_objective": SURVEY,
            "zoom_objective": ZOOM,
        },
    )
    assert summary["status"] == "completed"
    targets = [
        f
        for f in _lines(tmp_path / "run" / "frames.jsonl")
        if f["origin"] == "analysis"
    ]
    assert [f["pixel_size_um"] for f in targets] == [0.25, 0.25]
    xs = sorted(f["event"]["x_pos"] for f in targets)
    assert xs[1] - xs[0] == pytest.approx(512 * 0.25 * 0.9)  # zoom FOV
    assert core.getProperty("Objective", "Label") == SURVEY


def test_target_grid_at_uncalibrated_objective_is_refused(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    core.setProperty("Objective", "Label", SURVEY)
    errors: list[str] = []
    runner = SmartRunner(core)
    runner.events.analysisError.connect(lambda msg, _fatal: errors.append(msg))
    summary = runner.run(
        SURVEY_GRID,
        EXAMPLES / "survey_and_target.py",
        output="memory",
        timeout=60,
        params={
            "threshold": 3000.0,
            "switch_objective": True,
            "zoom_objective": "Objective-2",  # no pixel configuration
        },
    )
    assert summary is not None
    assert summary["status"] == "error"
    assert summary["frames"] == 4  # nothing after the survey was acquired
    assert errors and "not calibrated" in errors[0]


def test_grid_sized_after_a_switch_in_an_earlier_response(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    """The objective is switched by one response; the grid comes in a later one.

    The grid must be sized with the pixel size in effect when it runs (0.25),
    not the one at the start of the run (1.0).
    """
    core.setProperty("Objective", "Label", SURVEY)
    script = _script(
        tmp_path,
        f"""
        import useq
        API_VERSION = 1
        SYNC = "blocking"

        def analyze(image, frame, ctx):
            if frame.frame_id == 0:  # switch, and stay there
                return useq.MDAEvent(
                    properties=[("Objective", "Label", {ZOOM!r})]
                )
            if frame.frame_id == 1:  # a grid, in a *later* response
                return useq.MDASequence(
                    stage_positions=(useq.Position(x=0, y=0),),
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2),
                )
        """,
    )
    summary = _run(core, ONE, script, tmp_path / "run")
    assert summary["status"] == "completed"
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    tiles = [f for f in frames if f["parent_frame_id"] == 1]
    assert len(tiles) == 2
    assert [f["pixel_size_um"] for f in tiles] == [0.25, 0.25]
    # 512 px x 0.25 um/px = 128 um apart, not 512 (the survey objective's)
    xs = sorted(f["event"]["x_pos"] for f in tiles)
    assert xs[1] - xs[0] == pytest.approx(128.0)


def test_grid_at_uncalibrated_objective_stops_the_run(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import useq
        API_VERSION = 1
        SYNC = "blocking"

        def analyze(image, frame, ctx):
            if frame.frame_id == 0:
                return [
                    useq.MDAEvent(
                        action=useq.CustomAction(name="uncalibrated"),
                        properties=[("Objective", "Label", "Objective-2")],
                    ),
                    useq.MDASequence(
                        stage_positions=(useq.Position(x=0, y=0),),
                        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
                    ),
                ]
        """,
    )
    errors: list[str] = []
    runner = SmartRunner(core)
    runner.events.analysisError.connect(lambda msg, _f: errors.append(msg))
    summary = runner.run(ONE, script, output="memory", timeout=60)
    assert summary is not None
    assert summary["status"] == "error"
    assert errors and "not calibrated" in errors[0]
    assert "Objective-2" not in errors[0]  # names the pixel config, not the value


def test_uncalibrated_state_warns_once_per_transition(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    core.setProperty("Objective", "Label", SURVEY)
    script = _script(
        tmp_path,
        """
        import useq
        API_VERSION = 1
        SYNC = "blocking"

        def analyze(image, frame, ctx):
            if frame.frame_id == 0:
                return [
                    useq.MDAEvent(properties=[("Objective", "Label", "Objective-2")]),
                    useq.MDAEvent(),  # still uncalibrated: no second warning
                ]
        """,
    )
    warnings: list[str] = []
    runner = SmartRunner(core)
    runner.events.logMessage.connect(
        lambda level, msg: warnings.append(msg) if level == "warning" else None
    )
    summary = runner.run(ONE, script, output="memory", timeout=60)
    assert summary is not None and summary["status"] == "completed"
    assert len(warnings) == 1
    assert "no calibrated pixel size" in warnings[0]


def test_frame_handler_runs_on_runner_thread(core: CMMCorePlus, tmp_path: Path) -> None:
    threads: list[threading.Thread] = []

    class Probe(SmartRunner):
        def _on_frame_ready(
            self, img: np.ndarray, event: useq.MDAEvent, meta: FrameMetaV1
        ) -> None:
            threads.append(threading.current_thread())
            super()._on_frame_ready(img, event, meta)

    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    summary = Probe(core).run(
        useq.MDASequence(time_plan={"interval": 0, "loops": 2}), script, timeout=30
    )
    assert summary is not None and summary["frames"] == 2
    assert threads and all(t is not threading.main_thread() for t in threads)


def test_returned_time_lapse_keeps_its_own_interval(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        """
        import useq
        API_VERSION = 1
        def analyze(image, frame, ctx):
            if frame.frame_id == 0:
                return useq.MDASequence(
                    time_plan=useq.TIntervalLoops(interval=0.4, loops=2)
                )
        """,
    )
    summary = _run(core, ONE, script, tmp_path / "run")
    assert summary["frames"] == 3
    times = [f["runner_time_ms"] for f in _lines(tmp_path / "run" / "frames.jsonl")]
    assert times[2] - times[1] >= 350
    assert times[1] - times[0] < 350


@pytest.mark.parametrize(
    ("on_error", "status"), [("stop", "error"), ("skip", "completed")]
)
def test_script_error_policy(
    core: CMMCorePlus, tmp_path: Path, on_error: str, status: str
) -> None:
    script = _script(
        tmp_path,
        """
        API_VERSION = 1
        def analyze(image, frame, ctx):
            if frame.frame_id == 1:
                raise ValueError("bad frame")
        """,
    )
    errors: list[tuple[str, bool]] = []
    runner = SmartRunner(core)
    runner.events.analysisError.connect(lambda m, f: errors.append((m, f)))
    config = SmartRunConfig.from_script(script, on_error=on_error)
    summary = runner.run(
        useq.MDASequence(time_plan={"interval": 0, "loops": 5}), config, timeout=30
    )
    assert summary is not None
    assert summary["status"] == status
    assert summary["errors"] == 1
    if on_error == "stop":
        assert summary["frames"] < 5
        assert errors and "bad frame" in errors[0][0] and errors[0][1] is False
    else:
        assert summary["frames"] == 5
        assert not errors


def test_crashing_process_stops_run_cleanly(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        """
        import os
        API_VERSION = 1
        def analyze(image, frame, ctx):
            os._exit(3)
        """,
    )
    fatal: list[bool] = []
    runner = SmartRunner(core)
    runner.events.analysisError.connect(lambda _m, f: fatal.append(f))
    summary = runner.run(
        useq.MDASequence(time_plan={"interval": 0, "loops": 10}),
        script,
        execution="process",
        timeout=60,
    )
    assert summary is not None
    assert summary["status"] == "error"
    assert fatal == [True]
    assert summary["frames"] < 10


def _slow_runner(core: CMMCorePlus, tmp_path: Path, seconds: float) -> SmartRunner:
    script = _script(
        tmp_path,
        f"""
        import time
        API_VERSION = 1
        SYNC = "blocking"
        def analyze(image, frame, ctx):
            time.sleep({seconds})
        """,
    )
    runner = SmartRunner(core)
    queued = threading.Event()
    runner.events.analysisQueued.connect(lambda _fid: queued.set())
    runner.run(
        useq.MDASequence(time_plan={"interval": 0, "loops": 10}), script, block=False
    )
    assert queued.wait(10)
    return runner


def test_cancel_while_blocked_on_analysis(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = _slow_runner(core, tmp_path, 1.5)
    cancelled_at = time.perf_counter()
    runner.cancel()
    deadline = time.perf_counter() + 5
    while core.mda.is_running() and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert time.perf_counter() - cancelled_at < 1.0  # not waiting for analysis
    summary = runner.wait(15)
    assert summary is not None
    assert summary["status"] == "cancelled"
    assert summary["frames"] == 1
    assert core.mda.status.finish_reason == "canceled"


def test_cancel_from_outside_is_reported_as_cancelled(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    runner = _slow_runner(core, tmp_path, 0.5)
    core.mda.cancel()
    summary = runner.wait(15)
    assert summary is not None and summary["status"] == "cancelled"


def test_prepare_reports_setup_failure(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        """
        API_VERSION = 1
        def setup(ctx):
            raise RuntimeError("no GPU")
        def analyze(image, frame, ctx): ...
        """,
    )
    runner = SmartRunner(core)
    with pytest.raises(SmartRunError, match="no GPU"):
        runner.run(ONE, script)
    assert not runner.is_active()


def test_empty_base_sequence_is_refused(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    runner = SmartRunner(core)
    with pytest.raises(SmartRunError, match="no events"):
        runner.run(useq.MDASequence(), script)
    assert not runner.is_active()


def test_stopped_run_drops_late_results(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        """
        import time, useq
        API_VERSION = 1
        SYNC = "async"
        def analyze(image, frame, ctx):
            time.sleep(0.3)
            return useq.MDAEvent()
        """,
    )
    runner = SmartRunner(core)
    queued = threading.Event()
    runner.events.analysisQueued.connect(lambda _fid: queued.set())
    runner.run(
        useq.MDASequence(time_plan={"interval": 0, "loops": 3}), script, block=False
    )
    assert queued.wait(10)
    runner.request_stop()
    summary = runner.wait(15)
    assert summary is not None
    assert summary["status"] == "stopped_by_user"
    assert summary["injected_events"] == 0
    assert summary["dropped_responses"] >= 1


def test_core_run_smart_is_non_blocking(core: CMMCorePlus, tmp_path: Path) -> None:
    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    runner = core.run_smart(ONE, script)
    assert isinstance(runner, SmartRunner)
    summary = runner.wait(30)
    assert summary is not None and summary["status"] == "completed"


def test_frames_carry_their_event_in_the_data_file(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    script = _script(tmp_path, "API_VERSION = 1\ndef analyze(i, f, c): ...\n")
    runner = SmartRunner(core)
    summary = runner.run(
        useq.MDASequence(channels=["DAPI", "FITC"]),
        script,
        output=str(tmp_path / "data.ome.zarr"),
        timeout=30,
    )
    assert summary is not None and summary["frames"] == 2
    text = " ".join(
        Path(f).read_text()
        for f in glob.glob(
            str(tmp_path / "data.ome.zarr" / "**" / "zarr.json"), recursive=True
        )
    )
    assert "mda_event" in text
    assert "FITC" in text
    assert "pymmcore_plus_smart" in text  # provenance travels with the event


def test_dry_run_acquires_nothing(core: CMMCorePlus) -> None:
    from pymmcore_plus.smart import dry_run

    core.snapImage()
    result = dry_run(
        EXAMPLES / "detect_and_act.py",
        core.getImage(),
        core=core,
        params={"threshold": 0.0, "center_on_hit": False},
    )
    assert result.ok, result.error
    assert result.records["hits"] == 1
    assert result.response is not None and len(result.response.events) > 1
    assert not core.mda.is_running()


def test_smart_package_imports_without_qt() -> None:
    """Scripts import pymmcore_plus.smart inside spawned worker processes."""
    import subprocess
    import sys

    code = (
        "import sys, pymmcore_plus.smart\n"
        "heavy = {m.split('.')[0] for m in sys.modules} & {'PyQt5', 'PyQt6', "
        "'PySide2', 'PySide6', 'qtpy'}\n"
        "print(','.join(sorted(heavy)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def _burst_counter(core: CMMCorePlus) -> list[str]:
    """Record the type of every event the engine starts, on either backend.

    Connected directly: with Qt signals a plain callable would be queued to
    the thread it was connected from, which runs no event loop here.
    """
    from pymmcore_plus.smart._runner import _connect_direct

    started: list[str] = []
    events = core.mda.events
    _connect_direct(
        events, events.eventStarted, lambda e: started.append(type(e).__name__)
    )
    return started


FAST_BASE = useq.MDASequence(
    channels=["DAPI"], time_plan=useq.TIntervalLoops(interval=0, loops=20)
)
NOOP = "API_VERSION = 1\ndef analyze(image, frame, ctx): ...\n"


@pytest.mark.parametrize(
    ("sequencing", "sync", "bursts"),
    [
        ("safe", "async", True),  # a 0-interval base runs as one burst
        ("safe", "blocking", False),  # per-frame feedback is preserved
        ("always", "blocking", True),  # opted in explicitly
        ("off", "async", False),
    ],
)
def test_base_sequencing_modes(
    core: CMMCorePlus, tmp_path: Path, sequencing: str, sync: str, bursts: bool
) -> None:
    core.mda.engine.use_hardware_sequencing = True
    started = _burst_counter(core)
    script = _script(tmp_path, NOOP)
    summary = _run(core, FAST_BASE, script, sequencing=sequencing, sync=sync)
    assert summary["status"] == "completed"
    assert summary["frames"] == 20  # every frame arrives, either way
    assert ("SequencedEvent" in started) is bursts
    if bursts:
        assert len(started) < 20


def test_returned_batch_is_sequenced_in_blocking_mode(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    """The script committed to the whole batch: nothing gates inside it."""
    core.mda.engine.use_hardware_sequencing = True
    started = _burst_counter(core)
    script = _script(
        tmp_path,
        """
        import useq
        API_VERSION = 1
        SYNC = "blocking"

        def analyze(image, frame, ctx):
            if frame.frame_id == 0:
                return useq.MDASequence(
                    time_plan=useq.TIntervalLoops(interval=0, loops=8)
                )
        """,
    )
    summary = _run(core, ONE, script, tmp_path / "run")
    assert summary["status"] == "completed"
    assert summary["frames"] == 9
    assert "SequencedEvent" in started
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    # provenance survives sequencing: every burst frame is attributed
    assert [f["origin"] for f in frames] == ["base"] + ["analysis"] * 8
    assert {f["parent_frame_id"] for f in frames[1:]} == {0}


def test_reactive_script_does_not_deadlock_with_sequencing(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    """Each event depends on the previous frame: nothing may be pre-fetched."""
    core.mda.engine.use_hardware_sequencing = True
    script = _script(
        tmp_path,
        """
        import useq
        from pymmcore_plus.smart import STOP
        API_VERSION = 1
        SYNC = "async"

        def analyze(image, frame, ctx):
            if frame.frame_id >= 4:
                return STOP
            return useq.MDAEvent(exposure=1)
        """,
    )
    summary = _run(core, ONE, script, sequencing="always")
    assert summary["status"] == "stopped_by_script"
    assert summary["frames"] == 5


def test_max_burst_bounds_preemption(core: CMMCorePlus, tmp_path: Path) -> None:
    core.mda.engine.use_hardware_sequencing = True
    started = _burst_counter(core)
    script = _script(tmp_path, NOOP)
    config = SmartRunConfig.from_script(script, sync="async", max_burst=5)
    runner = SmartRunner(core)
    runner.prepare(FAST_BASE, config)
    runner.start()
    summary = runner.wait(60)
    assert summary is not None and summary["frames"] == 20
    assert started.count("SequencedEvent") == 4  # 20 frames / 5 per burst


def test_sequencing_is_read_from_the_script(tmp_path: Path) -> None:
    from pymmcore_plus.smart import inspect_script

    script = _script(tmp_path, 'API_VERSION = 1\nSEQUENCING = "off"\n' + NOOP)
    assert inspect_script(script).sequencing == "off"
    assert SmartRunConfig.from_script(script).sequencing == "off"
    # an explicit argument still wins
    assert (
        SmartRunConfig.from_script(script, sequencing="always").sequencing == "always"
    )


SLOW_BASE = useq.MDASequence(
    channels=["DAPI"], time_plan=useq.TIntervalLoops(interval=0.15, loops=8)
)


def _started_runner(core: CMMCorePlus, tmp_path: Path, **options: Any) -> SmartRunner:
    """A run in progress, with one frame already acquired."""
    script = _script(tmp_path, NOOP)
    runner = SmartRunner(core)
    seen = threading.Event()
    runner.events.frameAcquired.connect(lambda _r: seen.set())
    runner.prepare(
        SLOW_BASE,
        SmartRunConfig.from_script(script, **options),
        output="memory",
        run_dir=tmp_path / "run",
    )
    runner.start()
    assert seen.wait(20)
    return runner


def test_request_adds_events_from_outside(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = _started_runner(core, tmp_path, sync="async")
    n = runner.request(useq.MDAEvent(channel={"config": "FITC", "group": "Channel"}))
    assert n == 1
    summary = runner.wait(60)
    assert summary is not None and summary["status"] == "completed"
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    external = [f for f in frames if f["origin"] == "external"]
    assert len(external) == 1
    assert external[0]["event"]["channel"]["config"] == "FITC"
    assert external[0]["parent_frame_id"] is None
    # and it is recorded in the run log
    requests = [
        a for a in _lines(tmp_path / "run" / "analysis.jsonl") if a["call"] == "request"
    ]
    assert len(requests) == 1 and requests[0]["injected"] == 1


def test_request_accepts_sequences_and_priority(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    runner = _started_runner(core, tmp_path, sync="async")
    n = runner.request(
        useq.MDASequence(z_plan=useq.ZRangeAround(range=2, step=1)), priority="end"
    )
    assert n == 3
    summary = runner.wait(60)
    assert summary is not None
    frames = _lines(tmp_path / "run" / "frames.jsonl")
    # priority="end": after every base event
    assert [f["origin"] for f in frames] == ["base"] * 8 + ["external"] * 3


def test_request_can_stop_and_drop_base(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = _started_runner(core, tmp_path, sync="async")
    runner.request(useq.MDAEvent(), drop_base=True)
    summary = runner.wait(60)
    assert summary is not None
    assert summary["frames"] < 8  # the rest of the base was dropped
    assert summary["status"] == "completed"


def test_request_outside_a_run_is_refused(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = SmartRunner(core)
    with pytest.raises(SmartRunError, match="No smart run is in progress"):
        runner.request(useq.MDAEvent())


def test_request_validates_its_input(core: CMMCorePlus, tmp_path: Path) -> None:
    runner = _started_runner(core, tmp_path, sync="async")
    try:
        with pytest.raises(TypeError):
            runner.request(42)
    finally:
        runner.cancel()
        runner.wait(30)


def test_external_frames_can_be_excluded_from_analysis(
    core: CMMCorePlus, tmp_path: Path
) -> None:
    script = _script(
        tmp_path,
        'API_VERSION = 1\nSYNC = "async"\nANALYZE = {"origins": ["base"]}\n'
        "def analyze(image, frame, ctx):\n    ctx.record(origin=frame.origin)\n",
    )
    runner = SmartRunner(core)
    seen = threading.Event()
    runner.events.frameAcquired.connect(lambda _r: seen.set())
    runner.prepare(
        SLOW_BASE,
        SmartRunConfig.from_script(script),
        output="memory",
        run_dir=tmp_path / "run",
    )
    runner.start()
    assert seen.wait(20)
    runner.request(useq.MDAEvent())
    summary = runner.wait(60)
    assert summary is not None
    analysed = [
        a for a in _lines(tmp_path / "run" / "analysis.jsonl") if a["call"] == "analyze"
    ]
    assert {a["records"]["origin"] for a in analysed} == {"base"}


@pytest.mark.parametrize("mode", MODES)
def test_a_script_can_be_the_file_that_runs_it(tmp_path: Path, mode: str) -> None:
    """One file holding both the hooks and the run (guarded by __main__)."""
    import subprocess
    import sys

    script = tmp_path / "single.py"
    script.write_text(
        "import useq\n"
        "from pymmcore_plus import CMMCorePlus\n"
        "from pymmcore_plus.smart import STOP\n"
        "API_VERSION = 1\n"
        "def analyze(image, frame, ctx):\n"
        "    return STOP if frame.frame_id >= 2 else useq.MDAEvent()\n"
        "if __name__ == '__main__':\n"
        "    core = CMMCorePlus()\n"
        "    core.loadSystemConfiguration()\n"
        "    r = core.run_smart(useq.MDASequence(channels=['DAPI']), __file__,\n"
        f"                      execution={mode!r}, output='memory', block=True)\n"
        "    print('FRAMES', r.summary['frames'])\n"
    )
    out = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=180
    )
    assert "FRAMES 3" in out.stdout, out.stderr[-2000:]
