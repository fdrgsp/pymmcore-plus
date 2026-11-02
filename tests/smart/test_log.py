from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from pymmcore_plus.smart._loader import inspect_script
from pymmcore_plus.smart._log import SmartRunLog

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_archived_script_preserves_source_bytes(tmp_path: Path, newline: str) -> None:
    source = newline.join(
        ["API_VERSION = 1", "# Microscopy: µm", "def analyze(i, f, c): ...", ""]
    ).encode("utf-8")
    script = tmp_path / "original.py"
    script.write_bytes(source)
    spec = inspect_script(script)
    log = SmartRunLog(tmp_path / "run")
    try:
        log.open({}, spec.source)
        archived = (log.run_dir / "script.py").read_bytes()
        assert archived == source
        assert hashlib.sha256(archived).hexdigest() == spec.sha256
    finally:
        log.finish("completed")
