"""The on-disk record of a smart run.

Smart runs are stored along a single ``t`` axis (frames in acquisition order,
whatever their channel or position), so the data store alone cannot say which
event produced which frame. ``frames.jsonl`` is that mapping, and with
``run.json`` (settings, base sequence, parameters), ``analysis.jsonl`` (every
analysis call) and ``script.py`` (the exact code that ran) a run can be
understood and reproduced later.

Written from the runner thread and from executor callback threads: every
write takes one lock and is flushed, so a crash leaves valid, complete lines.
"""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

if TYPE_CHECKING:
    import useq

RUN_FILE: Final = "run.json"
FRAMES_FILE: Final = "frames.jsonl"
ANALYSIS_FILE: Final = "analysis.jsonl"
SCRIPT_FILE: Final = "script.py"

_OME_SUFFIXES: Final = (".ome.zarr", ".ome.tiff", ".ome.tif", ".zarr", ".tiff", ".tif")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def data_path_for_output(output: object) -> Path | None:
    """Where an MDA *output* writes to disk; None for in-memory/scratch or handlers."""
    from ome_writers import AcquisitionSettings, ScratchFormat

    if isinstance(output, (str, Path)):
        if str(output).rstrip("/:").lower() in ("memory", "scratch"):
            return None
        return Path(output)
    if isinstance(output, AcquisitionSettings):
        if isinstance(output.format, ScratchFormat):
            return None
        return Path(output.root_path)
    return None


def run_dir_for_output(data_path: str | Path | None) -> Path:
    """Folder for a run's records: beside the data, or a fresh temp folder.

    ``/data/exp_001.ome.zarr`` -> ``/data/exp_001_smart/``. A name that is
    already taken gets a numeric suffix, so an earlier run is never mixed in.
    """
    if data_path is None:
        return Path(tempfile.mkdtemp(prefix="pymmcore-smart-"))
    data_path = Path(data_path)
    name = data_path.name
    for suffix in _OME_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    candidate = data_path.with_name(f"{name}_smart")
    counter = 1
    while candidate.exists():
        candidate = data_path.with_name(f"{name}_smart_{counter:03d}")
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def event_to_json(event: useq.MDAEvent) -> dict[str, Any]:
    """An event as JSON, without the (large, redundant) parent sequence."""
    return event.model_dump(mode="json", exclude={"sequence"}, exclude_none=True)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class SmartRunLog:
    """Appends a smart run's records under *run_dir*."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._run: dict[str, Any] = {}
        self._frames: TextIO | None = None
        self._analysis: TextIO | None = None

    @property
    def frames_path(self) -> Path:
        return self.run_dir / FRAMES_FILE

    @property
    def analysis_path(self) -> Path:
        return self.run_dir / ANALYSIS_FILE

    def open(
        self,
        run_info: dict[str, Any],
        script_source: str,
        *,
        packages: tuple[str, ...] = (),
    ) -> None:
        """Write ``run.json`` and ``script.py``, and open the line logs.

        *packages* are extra distributions whose versions are recorded (e.g.
        the application running the acquisition).
        """
        with self._lock:
            self._run = {
                "versions": {
                    pkg: _package_version(pkg)
                    for pkg in (
                        *packages,
                        "pymmcore-plus",
                        "useq-schema",
                        "ome-writers",
                    )
                },
                "started": now_iso(),
                "finished": None,
                "status": "running",
                **run_info,
            }
            self._write_run()
            (self.run_dir / SCRIPT_FILE).write_text(script_source, encoding="utf-8")
            self._frames = self.frames_path.open("a", encoding="utf-8")
            self._analysis = self.analysis_path.open("a", encoding="utf-8")

    def write_frame(self, record: dict[str, Any]) -> None:
        self._append("_frames", record)

    def write_analysis(self, record: dict[str, Any]) -> None:
        self._append("_analysis", record)

    def finish(self, status: str, **extra: Any) -> None:
        """Finalize ``run.json`` and close the line logs. Idempotent."""
        with self._lock:
            if self._run.get("finished") is None:
                self._run.update(finished=now_iso(), status=status, **extra)
                self._write_run()
            for attr in ("_frames", "_analysis"):
                if (stream := getattr(self, attr)) is not None:
                    stream.close()
                    setattr(self, attr, None)

    def _append(self, attr: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            if (stream := getattr(self, attr)) is None:
                return  # closed: a late result after the run finished
            stream.write(line + "\n")
            stream.flush()

    def _write_run(self) -> None:
        path = self.run_dir / RUN_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._run, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
