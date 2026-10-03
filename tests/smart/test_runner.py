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
