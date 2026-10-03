"""The same script must behave identically on a thread and in a process."""

from __future__ import annotations

import sys
import textwrap
import threading
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq

from pymmcore_plus.smart import FrameInfo
from pymmcore_plus.smart._executors import (
    ExecutorStartError,
    ProcessAnalysisExecutor,
    create_executor,
)
from pymmcore_plus.smart._worker import HostConfig, _ScriptHost

if TYPE_CHECKING:
    from pathlib import Path

MODES = ["thread", "process"]

SCRIPT = """
import numpy as np
import useq
from helper import double            # sibling module next to the script
from pymmcore_plus.smart import Response

API_VERSION = 1
PARAMETERS = {"factor": 2.0}

def setup(ctx):
    ctx.state["calls"] = 0
    ctx.log("setup done")

def analyze(image, frame, ctx):
    ctx.state["calls"] += 1
    ctx.record(mean=float(image.mean()), calls=ctx.state["calls"],
               where=ctx.execution, writable=bool(image.flags.writeable))
    if frame.frame_id == 1:
        raise RuntimeError("boom")
    return Response(
        events=[useq.MDAEvent(exposure=double(ctx.params["factor"]))],
        priority="end",
    )

def teardown(ctx):
    ctx.log(f"teardown after {ctx.state['calls']} calls")
"""


def _write(tmp_path: Path, source: str, name: str = "script.py") -> Path:
    (tmp_path / "helper.py").write_text("def double(x):\n    return 2 * x\n")
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def _frame(i: int) -> FrameInfo:
    seq = useq.MDASequence(time_plan=useq.TIntervalLoops(interval=0, loops=3))  # pyright: ignore
    event = list(seq)[i]
    return FrameInfo(frame_id=i, event=event, metadata={"exposure_ms": 10.0})


@pytest.mark.parametrize("mode", MODES)
def test_same_results_in_both_modes(tmp_path: Path, mode: str) -> None:
    script = _write(tmp_path, SCRIPT)
    executor = create_executor(mode)  # type: ignore[arg-type]
    setup = executor.start(
        HostConfig(script, {"factor": 3.0}, run_dir=tmp_path), timeout=60
    )
    try:
        assert setup.ok, setup.error
        assert setup.logs == [("info", "setup done")]

        image = np.full((4, 4), 7, dtype=np.uint16)
        first = executor.submit(image, _frame(0)).result(30)
        assert first.ok, first.error
        assert first.records == {
            "mean": 7.0,
            "calls": 1,
            "where": mode,
            # thread mode gets a read-only view of the stored frame; a
            # process gets its own (pickled) copy
            "writable": mode == "process",
        }
        assert first.response is not None
        assert [e.exposure for e in first.response.events] == [6.0]
        assert first.response.priority == "end"

        failed = executor.submit(image, _frame(1)).result(30)
        assert not failed.ok
        assert "RuntimeError: boom" in (failed.error or "")
        assert failed.records["calls"] == 2  # state persisted across calls
    finally:
        teardown = executor.stop(timeout=30)
    assert teardown is not None and teardown.ok
    assert teardown.logs == [("info", "teardown after 2 calls")]
    assert executor.stop() is None  # idempotent


@pytest.mark.parametrize("mode", MODES)
def test_import_error_is_reported_not_raised(tmp_path: Path, mode: str) -> None:
    script = _write(tmp_path, "API_VERSION = 1\nimport not_a_module_xyz\n")
    executor = create_executor(mode)  # type: ignore[arg-type]
    try:
        result = executor.start(HostConfig(script, {}, run_dir=tmp_path), timeout=60)
    finally:
        executor.stop(timeout=10)
    assert not result.ok
    assert "not_a_module_xyz" in (result.error or "")


@pytest.mark.parametrize("mode", MODES)
def test_setup_error_is_reported(tmp_path: Path, mode: str) -> None:
    script = _write(
        tmp_path,
        "API_VERSION = 1\ndef setup(ctx):\n    1 / 0\ndef analyze(i, f, c): ...\n",
    )
    executor = create_executor(mode)  # type: ignore[arg-type]
    try:
        result = executor.start(HostConfig(script, {}, run_dir=tmp_path), timeout=60)
    finally:
        executor.stop(timeout=10)
    assert not result.ok
    assert "ZeroDivisionError" in (result.error or "")


def test_crashing_process_marks_executor_broken(tmp_path: Path) -> None:
    script = _write(
        tmp_path,
        "import os\nAPI_VERSION = 1\ndef analyze(image, frame, ctx):\n"
        "    os._exit(1)\n",
    )
    executor = ProcessAnalysisExecutor()
    assert executor.start(HostConfig(script, {}, run_dir=tmp_path), timeout=60).ok
    future = executor.submit(np.zeros((2, 2)), _frame(0))
    with pytest.raises(BrokenProcessPool):
        future.result(30)
    assert executor.broken
    assert executor.stop(timeout=5) is None


def test_hung_process_is_terminated_on_stop(tmp_path: Path) -> None:
    script = _write(
        tmp_path,
        "import time\nAPI_VERSION = 1\ndef analyze(image, frame, ctx):\n"
        "    time.sleep(600)\n",
    )
    executor = ProcessAnalysisExecutor()
    assert executor.start(HostConfig(script, {}, run_dir=tmp_path), timeout=60).ok
    (worker,) = executor._workers
    executor.submit(np.zeros((2, 2)), _frame(0))
    assert executor.stop(timeout=0.5) is None  # teardown never got to run
    assert not worker.is_alive()


def test_process_start_timeout(tmp_path: Path) -> None:
    script = _write(
        tmp_path,
        "import time\ntime.sleep(600)\nAPI_VERSION = 1\ndef analyze(i, f, c): ...\n",
    )
    executor = ProcessAnalysisExecutor()
    with pytest.raises(ExecutorStartError, match="did not start"):
        executor.start(HostConfig(script, {}, run_dir=tmp_path), timeout=3)
    assert executor._workers
    assert not any(p.is_alive() for p in executor._workers)


def test_thread_mode_runs_off_the_calling_thread(tmp_path: Path) -> None:
    script = _write(
        tmp_path,
        "import threading\nAPI_VERSION = 1\n"
        "def analyze(image, frame, ctx):\n"
        "    ctx.record(thread=threading.current_thread().name)\n",
    )
    executor = create_executor("thread")
    try:
        assert executor.start(HostConfig(script, {}, run_dir=tmp_path)).ok
        result = executor.submit(np.zeros((2, 2)), _frame(0)).result(10)
    finally:
        executor.stop()
    assert result.records["thread"] != threading.current_thread().name
    assert str(result.records["thread"]).startswith("smart-analysis")


def test_host_reimports_fresh_module_each_time(tmp_path: Path) -> None:
    """Edits to a script take effect on the next run, even in-process."""
    script = _write(
        tmp_path,
        "API_VERSION = 1\nVALUE = 1\ndef analyze(image, frame, ctx):\n"
        "    ctx.record(value=VALUE)\n",
    )
    first = _ScriptHost(HostConfig(script, run_dir=tmp_path), "thread")
    a = first.analyze(np.zeros(1), _frame(0))
    first.close()
    script.write_text(script.read_text().replace("VALUE = 1", "VALUE = 2"))
    second = _ScriptHost(HostConfig(script, run_dir=tmp_path), "thread")
    b = second.analyze(np.zeros(1), _frame(0))
    second.close()
    assert (a.records["value"], b.records["value"]) == (1, 2)
    assert str(tmp_path) not in sys.path
