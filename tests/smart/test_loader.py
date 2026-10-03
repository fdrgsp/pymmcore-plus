# pyright: reportArgumentType=false
# (useq models are built from plain values that pydantic coerces)
from __future__ import annotations

from pathlib import Path

import pytest
import useq

from pymmcore_plus.smart._loader import ScriptError, inspect_script, inspect_source

EXAMPLES = Path(__file__).parents[2] / "examples" / "smart_microscopy"

GOOD = """
API_VERSION = 1
NAME = "Good"
DESCRIPTION = "d"
EXECUTION = "process"
SYNC = "async"
ANALYZE = {"channels": ["DAPI"], "every_nth": 2, "origins": ["base"]}
PARAMETERS = {
    "threshold": {"default": 0.5, "min": 0, "max": 1, "step": 0.1, "label": "T"},
    "count": 3,
    "enabled": True,
    "mode": {"default": "fast", "choices": ["fast", "slow"]},
    "name": "abc",
}

def setup(ctx): ...
def analyze(image, frame, ctx): ...
def teardown(ctx): ...
"""


@pytest.mark.parametrize(
    "path",
    sorted(p for p in EXAMPLES.glob("*.py") if p.name != "run_smart.py"),
    ids=lambda p: p.stem,
)
def test_templates_are_valid(path: Path) -> None:
    spec = inspect_script(path)
    assert spec.api_version == 1
    assert spec.name


def test_good_script() -> None:
    spec = inspect_source(GOOD)
    assert spec.name == "Good"
    assert (spec.execution, spec.sync) == ("process", "async")
    assert spec.has_setup and spec.has_teardown
    kinds = {p.name: p.kind for p in spec.params}
    assert kinds == {
        "threshold": "float",
        "count": "int",
        "enabled": "bool",
        "mode": "choice",
        "name": "str",
    }
    assert spec.default_params()["threshold"] == 0.5
    f = spec.filter
    assert f.channels == ("DAPI",) and f.every_nth == 2 and f.origins == {"base"}
    dapi = useq.MDAEvent(channel="DAPI")
    assert f.accepts(0, dapi, "base")
    assert not f.accepts(1, dapi, "base")  # every 2nd
    assert not f.accepts(0, dapi, "analysis")
    assert not f.accepts(0, useq.MDAEvent(channel="FITC"), "base")


def test_defaults_when_optional_constants_missing() -> None:
    spec = inspect_source("API_VERSION = 1\ndef analyze(image, frame, ctx): ...\n")
    assert (spec.execution, spec.sync) == ("thread", "blocking")
    assert spec.params == ()
    assert spec.filter.accepts(7, useq.MDAEvent(), "analysis")


def test_resolve_params_keeps_only_valid_overrides() -> None:
    spec = inspect_source(GOOD)
    resolved = spec.resolve_params(
        {"threshold": 0.9, "count": "7", "mode": "nope", "gone": 1, "enabled": 1}
    )
    assert resolved["threshold"] == 0.9
    assert resolved["count"] == 7
    assert resolved["mode"] == "fast"  # invalid choice -> default
    assert resolved["enabled"] is True  # not a bool -> default
    assert "gone" not in resolved
    assert spec.resolve_params({"threshold": 5})["threshold"] == 0.5  # out of range


V = "API_VERSION = 1\n"
A = "def analyze(i, f, c): ...\n"


@pytest.mark.parametrize(
    ("source", "match", "line"),
    [
        (V + "def analyze(image, frame, ctx)\n", "Syntax error", 2),
        (V, "must define analyze", None),
        (A, "must declare API_VERSION", None),
        ("API_VERSION = 2\n" + A, "not supported", 1),
        (V + "def analyze(image): ...\n", "exactly 3", 2),
        (V + "async def analyze(i, f, c): ...\n", "async def", 2),
        (V + "def setup(): ...\n" + A, "exactly 1", 2),
        (V + "X = 3\nPARAMETERS = {'a': X}\n" + A, "must be a literal", 3),
        (V + "SYNC = 'later'\n" + A, "SYNC", 2),
        (V + "EXECUTION = 'gpu'\n" + A, "EXECUTION", 2),
        (V + "ANALYZE = {'every_nth': 0}\n" + A, "every_nth", 2),
        (V + "PARAMETERS = {'a': {'min': 1}}\n" + A, "needs a 'default'", 2),
        (V + "PARAMETERS = {'a': {'default': 5, 'max': 1}}\n" + A, "out of range", 2),
        (V + "PARAMETERS = {'a': [1, 2]}\n" + A, "default must be", 2),
    ],
)
def test_script_errors(source: str, match: str, line: int | None) -> None:
    with pytest.raises(ScriptError, match=match) as info:
        inspect_source(source)
    assert info.value.line == line


def test_loading_never_executes_the_script(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    script = tmp_path / "s.py"
    script.write_text(
        f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(3)\n"
        "API_VERSION = 1\ndef analyze(image, frame, ctx): ...\n"
    )
    spec = inspect_script(script)
    assert spec.path == script.resolve()
    assert len(spec.sha256) == 64
    assert not marker.exists()


def test_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="Cannot read"):
        inspect_script(tmp_path / "missing.py")


def test_after_base_hook_detected() -> None:
    spec = inspect_source(V + A + "def after_base(ctx): ...\n")
    assert spec.has_after_base
    assert not inspect_source(V + A).has_after_base
    with pytest.raises(ScriptError, match="exactly 1"):
        inspect_source(V + A + "def after_base(): ...\n")
