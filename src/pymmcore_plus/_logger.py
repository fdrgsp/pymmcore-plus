from __future__ import annotations

import logging
import os
import sys
import threading
import weakref
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymmcore import CMMCore

__all__ = ["logger"]


logger = logging.getLogger("pymmcore-plus")

PYMM_LOG_FILE = os.getenv("PYMM_LOG_FILE", "")
DEFAULT_LOG_LEVEL: str = os.getenv("PYMM_LOG_LEVEL", "WARNING").upper()

if "PYTEST_RUNNING" in os.environ:
    LOG_FILE = None
elif PYMM_LOG_FILE not in ("", "0", "false", "no", "none"):
    LOG_FILE = Path(PYMM_LOG_FILE).expanduser().resolve()
else:
    from ._discovery import USER_DATA_DIR

    LOG_FILE = USER_DATA_DIR / "logs" / "pymmcore-plus.log"


class CustomFormatter(logging.Formatter):
    dark_grey = "\x1b[38;5;240m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    _format: str = (
        "%(asctime)s - %(name)s - %(levelname)s - (%(filename)s:%(lineno)d) %(message)s"
    )

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: dark_grey + _format + reset,
        logging.INFO: grey + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# cores whose CoreLog follows the log file (see `log_core_to_file`)
_CORES: weakref.WeakSet[CMMCore] = weakref.WeakSet()
_CORES_LOCK = threading.Lock()


def _move_cores(old: str, new: str) -> list[CMMCore]:
    """Point the cores' CoreLog at `new` where it is `old` ("" is no file)."""
    with _CORES_LOCK:
        cores = [core for core in _CORES if core.getPrimaryLogFile() == old]
    for core in cores:
        core.setPrimaryLogFile(new)
    return cores


class _CoreSharedFileHandler(RotatingFileHandler):
    """A rotating log file that MMCore's CoreLog writes to as well.

    MMCore holds the file open from C++: Windows cannot rename it then, and
    elsewhere the core would go on writing to the renamed (in time deleted) file.
    So the cores let it go while it rotates, then move to the new file.
    """

    def doRollover(self) -> None:
        cores = _move_cores(self.baseFilename, "")
        try:
            super().doRollover()
        finally:
            for core in cores:
                core.setPrimaryLogFile(self.baseFilename)


_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s.%(msecs)03d    tid0x%(thread)x [%(levelname)s,%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def configure_logging(
    file: str | Path | None = LOG_FILE,
    stderr_level: int | str = DEFAULT_LOG_LEVEL,
    file_level: int | str = logging.DEBUG,
    log_to_stderr: bool = True,
    file_rotation: int = 40,
    file_retention: int = 20,
) -> None:
    r"""Configure logging for pymmcore-plus.

    This function is called automatically once when pymmcore-plus is imported,
    to set up logging to stderr and a log file.  You can call it again to
    change the logging settings.

    You may also configure logging using the following environment variables:

    - `PYMM_LOG_LEVEL` - The log level for `stderr` logging. By default `INFO`.
    - `PYMM_LOG_FILE` - The path to the log file.  If set to `0`, `false`, `no`,
        or `none`, logging to file will be disabled.
    - `PYMM_LOG_RICH` - If set to `1`, `true`, or `yes`, use `rich` for stderr
        logging (requires `rich` to be installed). Note: rich formatting adds
        some overhead; see https://github.com/pymmcore-plus/pymmcore-plus/issues/449.


    !!! note

        This function will clear all existing logging handlers and replace them
        with new ones.  So be sure to pass all the settings you want to use each
        time you call this function.

    Parameters
    ----------
    file : str | Path | None
        Path to logfile. May also be set with MM_LOG_FILE environment variable.
        If `None`, will not log to file.  By default, logs to:
        Mac OS X:   ~/Library/Application Support/pymmcore-plus/logs
        Unix:       ~/.local/share/pymmcore-plus/logs
        Win:        C:\Users\<username>\AppData\Local\pymmcore-plus\pymmcore-plus\logs
    stderr_level : int | str
        Level for stderr logging.
        One of "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        or 5, 10, 20, 30, 40, or 50, respectively.
        by default `"INFO"`.
    file_level : int | str
        Level for logging to file, by default `"TRACE"`
    log_to_stderr : bool
        Whether to log to stderr, by default True
    file_rotation : int
        When to rollover to the next log file, in MegaBytes, by default `40`.
    file_retention : int
        Maximum number of log files to retain, by default `20`
    """
    # logging.basicConfig(level=logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    old_file = current_logfile(logger)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # automatically log to stderr
    if log_to_stderr and sys.stderr:
        # use rich for stderr logging if PYMM_LOG_RICH is set and rich is installed
        stderr_handler: logging.Handler | None = None
        if os.getenv("PYMM_LOG_RICH", "").lower() in ("1", "true", "yes"):
            try:
                from rich.logging import RichHandler

                stderr_handler = RichHandler()
            except ImportError:
                pass

        if stderr_handler is None:
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(CustomFormatter())

        stderr_handler.setLevel(stderr_level)
        logger.addHandler(stderr_handler)

    # automatically log to file
    if file:
        log_file = Path(file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a rotating file handler with a maximum file size and backup count.
        file_handler = _CoreSharedFileHandler(
            log_file, maxBytes=file_rotation * 1_000_000, backupCount=file_retention
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(_FILE_FORMATTER)
        logger.addHandler(file_handler)

    # cores that logged to the previous file (or to none) move to this one
    if (new_file := current_logfile(logger)) != old_file:
        _move_cores(str(old_file or ""), str(new_file or ""))


def log_core_to_file(core: CMMCore) -> None:
    """Send `core`'s CoreLog to the log file, following it as it rotates or moves."""
    with _CORES_LOCK:
        _CORES.add(core)
    if logfile := current_logfile(logger):
        core.setPrimaryLogFile(str(logfile))


def current_logfile(logger: logging.Logger) -> Path | None:
    """Return the first RotatingFileHandler's baseFilename."""
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            return Path(handler.baseFilename)
    return None


@contextmanager
def exceptions_logged() -> Iterator[None]:
    """Context manager to log exceptions."""
    try:
        yield
    except Exception as e:
        logger.error(e)


configure_logging()
