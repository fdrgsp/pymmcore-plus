from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_plus import CMMCorePlus, _logger, configure_logging
from pymmcore_plus._logger import core_logfile, logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def log_file(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "pymmcore-plus.log"
    level = logger.level
    logger.setLevel("DEBUG")
    configure_logging(file=path, log_to_stderr=False)
    yield path
    configure_logging()
    logger.setLevel(level)


def _flush(core: CMMCorePlus) -> None:
    """MMCore writes its log asynchronously: letting go of the file flushes it."""
    core.setPrimaryLogFile("")


def test_core_log_has_its_own_file(log_file: Path) -> None:
    """Two writers appending to one file can overwrite each other on Windows."""
    core = CMMCorePlus()
    assert core.getPrimaryLogFile() == str(log_file.with_name("pymmcore-plus-core.log"))
    core.logMessage("core entry")
    logger.info("python entry")
    _flush(core)
    assert "core entry" in core_logfile(log_file).read_text()
    assert "core entry" not in log_file.read_text()
    assert "python entry" in log_file.read_text()


@pytest.mark.skipif(
    not hasattr(CMMCorePlus, "setPrimaryLogFileRotation"),
    reason="this MMCore cannot rotate its log",
)
def test_core_log_rotates_by_itself(log_file: Path) -> None:
    configure_logging(file=log_file, log_to_stderr=False, file_rotation=1)
    core = CMMCorePlus()
    # Windows cannot rename a file other cores (left by earlier tests) hold open
    for other in list(_logger._CORES):
        if other is not core:
            other.setPrimaryLogFile("")
    for i in range(20_000):  # well over 1 MB
        core.logMessage(f"core entry {i:05d} {'x' * 40}")
    core.logMessage("the last one")
    _flush(core)

    core_file = core_logfile(log_file)
    rotated = list(core_file.parent.glob(f"{core_file.stem}_*{core_file.suffix}"))
    assert rotated
    assert core_file.stat().st_size < 1_000_000
    # MMCore rotates after writing: the last entry may end the newest backup
    assert any("the last one" in f.read_text() for f in [core_file, *rotated])


def test_cores_move_to_a_new_log_file(log_file: Path) -> None:
    core = CMMCorePlus()
    new_file = log_file.with_name("new.log")
    configure_logging(file=new_file, log_to_stderr=False)

    core.logMessage("after the move")
    _flush(core)
    assert "after the move" in core_logfile(new_file).read_text()
    assert "after the move" not in core_logfile(log_file).read_text()


def test_cores_return_when_the_file_is_turned_back_on(log_file: Path) -> None:
    core = CMMCorePlus()
    configure_logging(file=None, log_to_stderr=False)
    assert core.getPrimaryLogFile() == ""
    later = CMMCorePlus()

    configure_logging(file=log_file, log_to_stderr=False)
    expected = str(core_logfile(log_file))
    assert core.getPrimaryLogFile() == later.getPrimaryLogFile() == expected


def test_a_core_with_its_own_log_file_keeps_it(log_file: Path) -> None:
    core = CMMCorePlus()
    own = str(log_file.with_name("own.log"))
    core.setPrimaryLogFile(own)
    configure_logging(file=log_file.with_name("new.log"), log_to_stderr=False)
    assert core.getPrimaryLogFile() == own
