from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_plus import CMMCorePlus, configure_logging
from pymmcore_plus._logger import logger

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


def test_core_log_follows_the_file_as_it_rotates(log_file: Path) -> None:
    core = CMMCorePlus()
    (handler,) = logger.handlers
    handler.maxBytes = 2000  # type: ignore[attr-defined]
    for i in range(300):
        core.logMessage(f"core entry {i}")
        logger.info("python entry %d", i)
    # only Python records rotate the file: this one lands after the last rotation
    core.logMessage("after rotating")
    _flush(core)

    assert list(log_file.parent.glob(f"{log_file.name}.*"))
    text = log_file.read_text()
    assert "after rotating" in text
    assert "python entry 299" in text


def test_cores_move_to_a_new_log_file(log_file: Path) -> None:
    core = CMMCorePlus()
    new_file = log_file.with_name("new.log")
    configure_logging(file=new_file, log_to_stderr=False)

    core.logMessage("after the move")
    _flush(core)
    assert "after the move" in new_file.read_text()
    assert "after the move" not in log_file.read_text()


def test_cores_return_when_the_file_is_turned_back_on(log_file: Path) -> None:
    core = CMMCorePlus()
    configure_logging(file=None, log_to_stderr=False)
    assert core.getPrimaryLogFile() == ""
    later = CMMCorePlus()

    configure_logging(file=log_file, log_to_stderr=False)
    assert core.getPrimaryLogFile() == later.getPrimaryLogFile() == str(log_file)


def test_a_core_with_its_own_log_file_keeps_it(log_file: Path) -> None:
    core = CMMCorePlus()
    own = str(log_file.with_name("own.log"))
    core.setPrimaryLogFile(own)
    (handler,) = logger.handlers
    handler.doRollover()  # type: ignore[attr-defined]
    configure_logging(file=log_file.with_name("new.log"), log_to_stderr=False)
    assert core.getPrimaryLogFile() == own
