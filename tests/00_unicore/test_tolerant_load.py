"""Loading a config with `on_device_error`: broken Python devices are skipped."""

from __future__ import annotations

import sys
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from pymmcore_plus.experimental.unicore import PyDeviceLoadError, UniMMCore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEVICES = dedent("""
    from pymmcore_plus.experimental.unicore import (
        HubDevice,
        ShutterDevice,
        pymm_property,
    )

    SHUTDOWNS = []


    class Shutter(ShutterDevice):
        _open = False
        _power = 0

        @pymm_property
        def power(self) -> int:
            return self._power

        @power.setter
        def _set_power(self, value: int) -> None:
            self._power = value

        def get_open(self):
            return self._open

        def set_open(self, value):
            self._open = value

        def shutdown(self):
            SHUTDOWNS.append(self.get_label())


    class Hub(HubDevice):
        def shutdown(self):
            SHUTDOWNS.append(self.get_label())


    class FailingHub(Hub):
        def initialize(self):
            raise RuntimeError("hub offline")


    class BadInit(Shutter):
        def initialize(self):
            raise RuntimeError("port busy")

        def busy(self):
            raise RuntimeError("not initialized")


    class BadBusy(Shutter):
        def busy(self):
            raise RuntimeError("no reply")


    class BadInitAndShutdown(BadInit):
        def shutdown(self):
            raise RuntimeError("stuck")


    class BadConstruct(Shutter):
        def __init__(self):
            raise ValueError("bad constructor")
""")

HEADER = "Property,Core,Initialize,0\nDevice,Camera,DemoCamera,DCam\n"


@pytest.fixture
def devices(tmp_path: Path) -> Iterator[Path]:
    (tmp_path / "pydevs.py").write_text(DEVICES)
    (tmp_path / "pydevs_missing_dep.py").write_text("import not_a_real_package\n")
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    for name in ("pydevs", "pydevs_missing_dep"):
        sys.modules.pop(name, None)


def _load(
    tmp_path: Path, body: str, core: UniMMCore | None = None
) -> tuple[UniMMCore, list[PyDeviceLoadError]]:
    cfg = tmp_path / "scope.cfg"
    cfg.write_text(HEADER + dedent(body).lstrip())
    core = core or UniMMCore()
    errors: list[PyDeviceLoadError] = []
    core.loadSystemConfiguration(cfg, on_device_error=errors.append)
    return core, errors


def _shutdowns() -> list[str]:
    return sys.modules["pydevs"].SHUTDOWNS  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("module", "cls", "stage", "exc_type"),
    [
        ("pydevs_missing_dep", "Shutter", "import", ModuleNotFoundError),
        ("pydevs", "Missing", "import", AttributeError),
        ("pydevs", "BadConstruct", "construct", ValueError),
        ("pydevs", "BadInit", "initialize", RuntimeError),
        ("pydevs", "BadBusy", "wait", RuntimeError),
    ],
)
def test_failing_device_is_skipped_with_its_lines(
    devices: Path, module: str, cls: str, stage: str, exc_type: type
) -> None:
    core, errors = _load(
        devices,
        f"""
        #py pyDevice,Bad,{module},{cls}
        #py pyDevice,Good,pydevs,Shutter
        Property,Core,Initialize,1
        #py Property,Bad,power,1
        #py Property,Core,Shutter,Bad
        #py ConfigGroup,Channel,On,Bad,power,1
        #py ConfigGroup,Channel,On,Good,power,1
        Property,Core,Camera,Camera
        """,
    )

    (error,) = errors
    assert (error.label, error.stage) == ("Bad", stage)
    assert isinstance(error.exc, exc_type)
    assert "Bad" not in core.getLoadedDevices()
    assert {"Good", "Camera"} <= set(core.getLoadedDevices())
    assert core.getShutterDevice() == ""
    assert core.getCameraDevice() == "Camera"
    assert core.getConfigData("Channel", "On").dict() == {"Good": {"power": "1"}}
    if stage in ("import", "construct"):
        assert error.line_no == 3
        assert error.line.startswith("pyDevice,Bad,")


def test_missing_dependency_keeps_the_original_error(devices: Path) -> None:
    _, (error,) = _load(
        devices,
        "#py pyDevice,Bad,pydevs_missing_dep,Shutter\nProperty,Core,Initialize,1\n",
    )
    assert isinstance(error.exc, ModuleNotFoundError)
    assert error.exc.name == "not_a_real_package"


def test_load_device_names_a_missing_dependency(devices: Path) -> None:
    core = UniMMCore()
    with pytest.raises(ModuleNotFoundError) as info:
        core.loadDevice("Bad", "pydevs_missing_dep", "Shutter")
    assert info.value.name == "not_a_real_package"

    with pytest.raises(ModuleNotFoundError, match="importable python module") as info:
        core.loadDevice("Bad", "no_such_module", "Shutter")
    assert info.value.name == "no_such_module"


def test_without_callback_a_failed_initialize_still_fails_the_load(
    devices: Path,
) -> None:
    cfg = devices / "scope.cfg"
    cfg.write_text(
        HEADER + "#py pyDevice,Bad,pydevs,BadInit\nProperty,Core,Initialize,1\n"
    )
    core = UniMMCore()
    with pytest.raises(RuntimeError, match="not initialized"):
        core.loadSystemConfiguration(cfg)
    assert not core.getLoadedDevices()[1:]  # only Core is left


def test_failed_hub_unloads_its_peripherals_first(devices: Path) -> None:
    core, errors = _load(
        devices,
        """
        #py pyDevice,Hub,pydevs,FailingHub
        #py pyDevice,Child,pydevs,Shutter
        #py pyDevice,Other,pydevs,Shutter
        Property,Core,Initialize,1
        #py Parent,Child,Hub
        #py Property,Child,power,1
        """,
    )

    assert [(e.label, e.stage) for e in errors] == [
        ("Hub", "initialize"),
        ("Child", "initialize"),
    ]
    assert errors[1].exc is errors[0].exc
    assert _shutdowns() == ["Child", "Hub"]
    assert "Other" in core.getLoadedDevices()
    assert not {"Hub", "Child"} & set(core.getLoadedDevices())


def test_peripheral_of_a_hub_that_failed_to_import_is_not_loaded(
    devices: Path,
) -> None:
    core, errors = _load(
        devices,
        """
        #py pyDevice,Hub,pydevs,Missing
        #py pyDevice,Child,pydevs,Shutter
        Property,Core,Initialize,1
        #py Parent,Child,Hub
        """,
    )
    assert [(e.label, e.stage) for e in errors] == [
        ("Hub", "import"),
        ("Child", "import"),
    ]
    assert "Child" not in core.getLoadedDevices()


def test_failed_shutdown_still_removes_the_device(devices: Path) -> None:
    core, errors = _load(
        devices,
        """
        #py pyDevice,Bad,pydevs,BadInitAndShutdown
        Property,Core,Initialize,1
        """,
    )

    assert [(e.label, e.stage) for e in errors] == [
        ("Bad", "initialize"),
        ("Bad", "unload"),
    ]
    assert str(errors[1].exc) == "stuck"
    assert errors[1].line_no is None
    assert "Bad" not in core.getLoadedDevices()
    assert not core.isPyDevice("Bad")


def test_unload_device_completes_when_shutdown_raises(devices: Path) -> None:
    core = UniMMCore()
    core.loadDevice("Bad", "pydevs", "BadInitAndShutdown")
    core.setShutterDevice("Bad")
    core.defineConfig("Channel", "On", "Bad", "power", "1")

    with pytest.raises(RuntimeError, match="stuck"):
        core.unloadDevice("Bad")

    assert not core.isPyDevice("Bad")
    assert core.getShutterDevice() == ""
    assert core.getConfigData("Channel", "On").dict() == {}


def test_failing_settings_on_a_healthy_device_skip_only_that_line(
    devices: Path,
) -> None:
    core, errors = _load(
        devices,
        """
        #py pyDevice,Good,pydevs,Shutter
        Property,Core,Initialize,1
        #py Property,Good,NoSuchProperty,1
        #py Property,Good,power,1
        #py ConfigGroup,System,Startup,Good,AlsoMissing,1
        """,
    )

    assert [(e.label, e.stage, e.line_no) for e in errors] == [
        ("Good", "property", 5),
        ("Good", "startup", None),
    ]
    assert core.getProperty("Good", "power") == "1"


def test_cpp_errors_still_fail_the_load(devices: Path) -> None:
    with pytest.raises(RuntimeError, match="line 3"):
        _load(devices, "Device,Bad,NoSuchAdapter,Nothing\n")


def test_initialization_error_is_available(devices: Path) -> None:
    core = UniMMCore()
    core.loadDevice("Bad", "pydevs", "BadInit")
    core.loadDevice("Good", "pydevs", "Shutter")
    core.initializeAllDevices()
    assert str(core.getPyDeviceInitializationError("Bad")) == "port busy"
    assert core.getPyDeviceInitializationError("Good") is None


def test_iter_properties_filters_python_devices_by_type(devices: Path) -> None:
    from pymmcore_plus import PropertyType

    core = UniMMCore()
    core.loadDevice("Good", "pydevs", "Shutter")
    core.initializeDevice("Good")
    found = core.iterProperties(property_type=PropertyType.Integer, as_object=False)
    assert ("Good", "power") in list(found)


@pytest.mark.parametrize("cls", ["Missing", "BadInit", "BadBusy"])
@pytest.mark.parametrize("before_init", [False, True])
def test_failed_device_drops_its_entire_pixel_calibration(
    devices: Path, cls: str, before_init: bool
) -> None:
    calibration = (
        "#py ConfigPixelSize,BadRes,Bad,power,1\n"
        "ConfigPixelSize,BadRes,Camera,Binning,1\n"
        "PixelSize_um,BadRes,0.5\n"
        "PixelSizeAffine,BadRes,0.5,0,0,0,0.5,0\n"
    )
    body = f"#py pyDevice,Bad,pydevs,{cls}\n"
    if before_init:
        body += calibration
    body += "Property,Core,Initialize,1\n"
    if not before_init:
        body += calibration
    body += "ConfigPixelSize,GoodRes,Camera,Binning,1\nPixelSize_um,GoodRes,1\n"

    core, errors = _load(devices, body)

    assert [e.label for e in errors] == ["Bad"]
    assert "Camera" in core.getLoadedDevices()
    assert set(core.getAvailablePixelSizeConfigs()) == {"GoodRes"}
    assert core.getPixelSizeUmByID("GoodRes") == 1


def test_unload_removes_pixel_calibrations_and_preset_roles(devices: Path) -> None:
    core = UniMMCore()
    core.loadDevice("Bad", "pydevs", "BadInitAndShutdown")
    core.definePixelSizeConfig("BadRes", "Bad", "power", "1")
    core.setPixelSizeUm("BadRes", 0.5)
    core.definePixelSizeConfig("GoodRes")
    core.setPixelSizeUm("GoodRes", 1)
    core.defineConfig("Roles", "Bad", "Core", "Shutter", "Bad")
    core.defineConfig("Roles", "Good", "Core", "AutoShutter", "1")

    with pytest.raises(RuntimeError, match="stuck"):
        core.unloadDevice("Bad")

    assert set(core.getAvailablePixelSizeConfigs()) == {"GoodRes"}
    assert core.getConfigData("Roles", "Bad").dict() == {}
    assert core.getConfigData("Roles", "Good").dict() == {"Core": {"AutoShutter": "1"}}


def test_presets_do_not_keep_roles_naming_a_skipped_device(devices: Path) -> None:
    core, errors = _load(
        devices,
        "#py pyDevice,Bad,pydevs,BadInit\n"
        "Property,Core,Initialize,1\n"
        "#py ConfigGroup,System,Startup,Core,Shutter,Bad\n"
        "#py ConfigGroup,Roles,On,Core,Shutter,Bad\n",
    )
    assert [e.label for e in errors] == ["Bad"]
    assert not core.isConfigDefined("Roles", "On")
    assert "Camera" in core.getLoadedDevices()


def test_startup_problem_identifies_the_failed_setting(devices: Path) -> None:
    core, (error,) = _load(
        devices,
        "#py pyDevice,Good,pydevs,Shutter\n"
        "Property,Core,Initialize,1\n"
        "#py ConfigGroup,System,Startup,Good,Missing,1\n",
    )
    assert error.stage == "startup"
    assert error.line == "Property,Good,Missing,1"
    assert core.isPyDevice("Good")
