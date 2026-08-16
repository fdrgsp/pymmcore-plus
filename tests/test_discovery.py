"""Test Micro-Manager installation discovery."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from pymmcore_plus import _discovery
from pymmcore_plus._discovery import PYMMCORE_DIV, find_micromanager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installation in an empty, private user data dir, discoverable as real."""
    mm_dir = tmp_path / "mm"
    install = mm_dir / "Micro-Manager-2.0.0-20240101"
    install.mkdir(parents=True)

    monkeypatch.setattr(_discovery, "USER_DATA_MM_PATH", mm_dir)
    monkeypatch.setattr(_discovery, "CURRENT_MM_PATH", tmp_path / ".current_mm")
    monkeypatch.setattr(_discovery, "_DISCOVERED_MMS", {})
    monkeypatch.delenv("MICROMANAGER_PATH", raising=False)
    # stand in for actually loading a device adapter library from the directory
    monkeypatch.setattr(
        _discovery, "get_first_device_interface_version", lambda path: PYMMCORE_DIV
    )
    monkeypatch.setattr(_discovery, "_iter_device_paths", lambda path: iter(()))
    return install


def test_uninstalled_mm_is_no_longer_discovered(fake_install: Path) -> None:
    """A deleted installation must stop being reported.

    Discovery results are cached in a module-level dict for the life of the
    process, and `find_micromanager(return_first=False)` used to return that
    dict rather than what the current scan found -- so an installation stayed
    discoverable after being uninstalled, and anything re-listing the installed
    versions (e.g. pymmcore-widgets' InstallWidget) showed the entry it had
    just deleted.
    """
    path = str(fake_install.resolve())
    assert path in find_micromanager(return_first=False)
    assert find_micromanager(return_first=True) == path

    shutil.rmtree(fake_install)

    assert path not in find_micromanager(return_first=False)
    assert find_micromanager(return_first=True) is None
    assert fake_install.resolve() not in _discovery._DISCOVERED_MMS


def test_discovery_reports_what_it_found(fake_install: Path) -> None:
    """The returned list is this scan's result, not every path ever cached."""
    sibling = fake_install.parent / "Micro-Manager-2.0.0-20250202"
    sibling.mkdir()

    found = find_micromanager(return_first=False)
    assert sorted(found) == sorted(
        [str(fake_install.resolve()), str(sibling.resolve())]
    )
    # newest first
    assert find_micromanager(return_first=True) == str(sibling.resolve())
