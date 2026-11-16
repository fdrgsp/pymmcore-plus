"""Config file loading and saving for UniMMCore.

This module provides Python-owned config file loading/saving that supports
both C++ and Python devices. Python device lines use a `#py ` prefix so they
are treated as comments by upstream C++/pymmcore implementations.

Format example:
    # C++ devices
    Device,Camera,DemoCamera,DCam
    Property,Core,Initialize,1

    # Python devices (hidden from upstream via comment prefix)
    #py pyDevice,PyCamera,mypackage.cameras,MyCameraClass
    #py Property,PyCamera,Exposure,50.0
    #py Property,Core,Camera,PyCamera

    # Config groups can mix both
    ConfigGroup,Channel,DAPI,Dichroic,Label,400DCLP
    #py ConfigGroup,Channel,DAPI,PyFilter,Position,Blue
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pymmcore_plus import CFGCommand, CFGGroup, CMMCorePlus, DeviceType, Keyword
from pymmcore_plus._util import timestamp
from pymmcore_plus.experimental.unicore._device_manager import import_device_class

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ._unicore import UniMMCore

__all__ = [
    "PyDeviceLoadError",
    "load_system_configuration",
    "save_system_configuration",
]

# Prefix for Python device lines (treated as comment by C++/regular pymmcore)
PY_PREFIX = "#py "
# Custom command names (not in CFGCommand enum)
_PY_DEVICE_CMD = "pyDevice"

Stage = Literal[
    "import", "construct", "initialize", "property", "startup", "wait", "unload"
]


# =============================================================================
# Loading
# =============================================================================


@dataclass(frozen=True)
class PyDeviceLoadError:
    """A Python device problem met while loading a config file.

    `stage` is where it failed: `"import"` (module, class lookup), `"construct"`,
    `"initialize"`, `"wait"` (`busy()` raised) or `"unload"` (`shutdown()`
    raised while removing it) mean the device was skipped. `"property"` (a
    config line referencing it) and `"startup"` (a `System/Startup` setting)
    mean only that setting was skipped and the device was kept. The peripherals
    of a hub that failed are reported with the hub's stage and exception.
    """

    label: str
    stage: Stage
    exc: BaseException
    line_no: int | None = None  # None for stages not tied to one line
    line: str = ""


def load_system_configuration(
    core: UniMMCore,
    filename: str | Path,
    on_device_error: Callable[[PyDeviceLoadError], None] | None = None,
) -> None:
    """Load system configuration from a file.

    This is a Python implementation of MMCore::loadSystemConfigurationImpl
    that supports both C++ and Python devices. Lines prefixed with `#py `
    are processed as Python device commands.

    Parameters
    ----------
    core : UniMMCore
        The core instance to configure.
    filename : str | Path
        Path to the configuration file.
    on_device_error : Callable[[PyDeviceLoadError], None] | None
        If given, Python device failures are passed to it and skipped instead of
        aborting the load (see `UniMMCore.loadSystemConfiguration`).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(f"Configuration file not found: {path}")

    load_from_string(core, path.read_text(), str(path), on_device_error)


def load_from_string(
    core: UniMMCore,
    text: str,
    source: str = "<string>",
    on_device_error: Callable[[PyDeviceLoadError], None] | None = None,
) -> None:
    """Load system configuration from a string.

    Parameters
    ----------
    core : UniMMCore
        The core instance to configure.
    text : str
        The configuration text.
    source : str
        Source identifier for error messages.
    on_device_error : Callable[[PyDeviceLoadError], None] | None
        If given, Python device failures are passed to it and skipped instead of
        aborting the load (see `UniMMCore.loadSystemConfiguration`).
    """
    lines = list(_iter_commands(text))
    if on_device_error is not None:
        _TolerantLoader(core, source, on_device_error).run(lines)
        return

    for line_num, line in lines:
        try:
            _run_command(core, line)
        except Exception as e:
            raise _line_error(source, line_num, line, e) from e

    # File parsing finished, apply startup configuration if defined
    if core.isConfigDefined(CFGGroup.System, CFGGroup.System_Startup):
        # Build system state cache before setConfig to avoid failures
        core.waitForSystem()
        core.updateSystemStateCache()
        core.setConfig(CFGGroup.System, CFGGroup.System_Startup)

    # Final sync after all configuration is applied
    core.waitForSystem()
    core.updateSystemStateCache()


def _iter_commands(text: str) -> Iterator[tuple[int, str]]:
    """Yield `(line_number, command)`, with `#py ` stripped and comments dropped."""
    for line_num, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line.startswith(PY_PREFIX):
            line = line[len(PY_PREFIX) :].lstrip()
        if line and not line.startswith("#"):
            yield line_num, line


def _line_error(source: str, line_num: int, line: str, e: Exception) -> RuntimeError:
    return RuntimeError(
        f"Error in configuration file {source!r} at line {line_num}: {line!r}\n{e}"
    )


_ROLES = {
    Keyword.CoreCamera,
    Keyword.CoreShutter,
    Keyword.CoreFocus,
    Keyword.CoreXYStage,
    Keyword.CoreAutoFocus,
    Keyword.CoreSLM,
    Keyword.CoreGalvo,
}


def _referenced_labels(cmd: str, args: Sequence[str]) -> Sequence[str]:
    """Return the device labels that a config command refers to."""
    if cmd in (
        _PY_DEVICE_CMD,
        CFGCommand.Device,
        CFGCommand.Delay,
        CFGCommand.FocusDirection,
        CFGCommand.Label,
    ):
        return args[:1]
    if cmd == CFGCommand.Property:
        if args[:1] != [Keyword.CoreDevice]:
            return args[:1]
        return args[2:3] if args[1:2] and args[1] in _ROLES else ()
    if cmd == CFGCommand.ConfigGroup:
        return _referenced_labels(CFGCommand.Property, args[2:])
    if cmd == CFGCommand.ConfigPixelSize:
        return args[1:2]
    if cmd == CFGCommand.ParentID:
        return args[:2]
    return ()


class _TolerantLoader:
    """Load a config, skipping Python devices that fail and reporting them."""

    def __init__(
        self,
        core: UniMMCore,
        source: str,
        report: Callable[[PyDeviceLoadError], None],
    ) -> None:
        self._core = core
        self._devices = core._pydevices  # noqa: SLF001
        self._source = source
        self._report = report
        self._skipped: set[str] = set()
        self._parents: dict[str, str] = {}
        self._pixel_devices: dict[str, set[str]] = {}
        self._skipped_pixels: set[str] = set()

    def run(self, lines: Sequence[tuple[int, str]]) -> None:
        commands = [(line_num, line, *_split(line)) for line_num, line in lines]
        # read every line first: a hub's peripherals and a calibration's devices
        # must be known when a device fails, even if they are declared later
        for _, _, cmd, args in commands:
            if cmd == CFGCommand.ParentID and len(args) == 2:
                self._parents[args[0]] = args[1]
            elif cmd == CFGCommand.ConfigPixelSize and len(args) == 4:
                self._pixel_devices.setdefault(args[0], set()).add(args[1])

        core = self._core
        for line_num, line, cmd, args in commands:
            if (
                args
                and args[0] in self._skipped_pixels
                and (cmd == CFGCommand.ConfigPixelSize or cmd.startswith("PixelSize"))
            ):
                continue
            labels = _referenced_labels(cmd, args)
            if self._skipped.intersection(labels):
                continue
            if cmd == _PY_DEVICE_CMD and len(args) == 3:
                label, module, cls_name = args
                self._load_device(label, module, cls_name, line_num, line)
                continue
            try:
                _run_command(core, line)
            except Exception as e:
                py_label = next((lbl for lbl in labels if core.isPyDevice(lbl)), None)
                if py_label is None:
                    raise _line_error(self._source, line_num, line, e) from e
                self._report(PyDeviceLoadError(py_label, "property", e, line_num, line))
                continue
            if cmd == CFGCommand.Property and args == [
                Keyword.CoreDevice,
                Keyword.CoreInitialize,
                "1",
            ]:
                self._drop_failed_initializations()

        if core.isConfigDefined(CFGGroup.System, CFGGroup.System_Startup):
            self._wait_for_system()
            core.updateSystemStateCache()
            self._apply_startup()
        self._wait_for_system()
        core.updateSystemStateCache()

    def _load_device(
        self, label: str, module: str, cls_name: str, line_num: int, line: str
    ) -> None:
        stage: Stage = "import"
        try:
            cls = import_device_class(module, cls_name)
            stage = "construct"
            self._core.loadPyDevice(label, cls())
        except Exception as e:
            self._fail(label, stage, e, line_num, line)

    def _fail(
        self,
        label: str,
        stage: Stage,
        exc: BaseException,
        line_num: int | None = None,
        line: str = "",
    ) -> None:
        """Skip `label` and its peripherals (unloaded first), and report each."""
        self._skipped.add(label)
        pixels = {p for p, labels in self._pixel_devices.items() if label in labels}
        self._skipped_pixels.update(pixels)
        for preset in pixels.intersection(self._core.getAvailablePixelSizeConfigs()):
            self._core.deletePixelSizeConfig(preset)
        self._report(PyDeviceLoadError(label, stage, exc, line_num, line))
        peripherals = [p for p, hub in self._parents.items() if hub == label]
        peripherals += self._devices.get_loaded_peripherals(label)
        for peripheral in dict.fromkeys(peripherals):
            if peripheral not in self._skipped:
                self._fail(peripheral, stage, exc, line_num, line)
        if label in self._devices:
            try:
                self._core.unloadDevice(label)
            except Exception as e:
                self._report(PyDeviceLoadError(label, "unload", e))

    def _drop_failed_initializations(self) -> None:
        for label in list(self._devices):
            if label in self._devices and (
                error := self._core.getPyDeviceInitializationError(label)
            ):
                self._fail(label, "initialize", error)

    def _wait_for_system(self) -> None:
        """Wait for C++ devices, then for each Python device on its own."""
        CMMCorePlus.waitForDeviceType(self._core, DeviceType.AnyType)
        for label in list(self._devices):
            if label in self._devices:
                try:
                    self._core.waitForDevice(label)
                except Exception as e:
                    self._fail(label, "wait", e)

    def _apply_startup(self) -> None:
        """Apply `System/Startup`: C++ errors raise, Python ones skip the setting."""
        core = self._core
        group, preset = CFGGroup.System, CFGGroup.System_Startup
        CMMCorePlus.setConfig(core, group, preset)
        settings = core.getConfigData(group, preset, native=True)
        failed = []
        for i in range(settings.size()):
            s = settings.getSetting(i)
            device, prop = s.getDeviceLabel(), s.getPropertyName()
            if core.isPyDevice(device):
                try:
                    core.setProperty(device, prop, s.getPropertyValue())
                except Exception:
                    failed.append((device, prop, s.getPropertyValue()))
        # retry once: a setting may depend on another one applied after it
        for device, prop, value in failed:
            try:
                core.setProperty(device, prop, value)
            except Exception as e:
                self._report(
                    PyDeviceLoadError(
                        device, "startup", e, line=f"Property,{device},{prop},{value}"
                    )
                )


def _split(line: str) -> tuple[str, list[str]]:
    cmd, *args = line.split(CFGCommand.FieldDelimiters)
    return cmd, args


def _run_command(core: UniMMCore, line: str) -> None:
    """Execute a single configuration command.

    Mirrors MMCore::loadSystemConfigurationImpl command processing.
    """
    cmd_name, args = _split(line)

    # Handle custom commands first
    if cmd_name == _PY_DEVICE_CMD:
        _exec_device(core, args)
        return

    try:
        command = CFGCommand(cmd_name)
    except ValueError:  # pragma: no cover
        raise ValueError(f"Unknown configuration command: {cmd_name!r}") from None

    if command == CFGCommand.Configuration:  # pragma: no cover
        warnings.warn(
            f"Obsolete command {cmd_name!r} ignored in configuration file",
            UserWarning,
            stacklevel=3,
        )
        return
    if command == CFGCommand.Equipment:  # pragma: no cover
        raise ValueError("Equipment command has been removed from config format")
    if command == CFGCommand.ImageSynchro:  # pragma: no cover
        raise ValueError("ImageSynchro command has been removed from config format")

    if executor := _COMMAND_EXECUTORS.get(command):
        executor(core, args)


# =============================================================================
# Command Executors
# =============================================================================


def _exec_device(core: UniMMCore, args: Sequence[str]) -> None:
    """Load a device: Device,<label>,<library>,<device_name>."""
    if len(args) != 3:  # pragma: no cover
        raise ValueError(f"Device command requires 3 arguments, got {len(args)}")
    label, library, device_name = args
    core.loadDevice(label, library, device_name)


def _exec_property(core: UniMMCore, args: Sequence[str]) -> None:
    """Set a property: Property,<device>,<property>,<value>."""
    if len(args) not in (2, 3):  # pragma: no cover
        raise ValueError(f"Property command requires 2-3 arguments, got {len(args)}")

    device, prop = args[0], args[1]
    value = args[2] if len(args) > 2 else ""

    # Special case: Core device properties
    if device == Keyword.CoreDevice:
        if prop == Keyword.CoreInitialize:
            try:
                init_val = int(value)
            except (ValueError, TypeError):  # pragma: no cover
                raise ValueError(
                    f"Initialize value must be integer, got {value!r}"
                ) from None
            if init_val == 0:
                # Unload all devices (reset state for fresh config load)
                core.unloadAllDevices()
            elif init_val == 1:
                core.initializeAllDevices()
            return
        elif prop == Keyword.CoreCamera:
            core.setCameraDevice(value)
            return
        elif prop == Keyword.CoreShutter:
            core.setShutterDevice(value)
            return
        elif prop == Keyword.CoreFocus:
            core.setFocusDevice(value)
            return
        elif prop == Keyword.CoreXYStage:
            core.setXYStageDevice(value)
            return
        elif prop == Keyword.CoreAutoFocus:
            core.setAutoFocusDevice(value)
            return
        elif prop == Keyword.CoreSLM:
            core.setSLMDevice(value)
            return
        elif prop == Keyword.CoreGalvo:
            core.setGalvoDevice(value)
            return
        elif prop == Keyword.CoreChannelGroup:
            core.setChannelGroup(value)
            return
        elif prop == Keyword.CoreAutoShutter:
            core.setAutoShutter(bool(int(value)))
            return

    core.setProperty(device, prop, value)


def _exec_delay(core: UniMMCore, args: Sequence[str]) -> None:
    """Set device delay: Delay,<device>,<delay_ms>."""
    if len(args) != 2:  # pragma: no cover
        raise ValueError(f"Delay command requires 2 arguments, got {len(args)}")
    device, delay_str = args
    try:
        delay_ms = float(delay_str)
    except ValueError:  # pragma: no cover
        raise ValueError(f"Delay must be a number, got {delay_str!r}") from None
    core.setDeviceDelayMs(device, delay_ms)


def _exec_focus_direction(core: UniMMCore, args: Sequence[str]) -> None:
    """Set focus direction: FocusDirection,<device>,<direction>."""
    if len(args) != 2:  # pragma: no cover
        raise ValueError(
            f"FocusDirection command requires 2 arguments, got {len(args)}"
        )
    device, direction_str = args
    try:
        direction = int(direction_str)
    except ValueError:  # pragma: no cover
        raise ValueError(
            f"FocusDirection must be an integer, got {direction_str!r}"
        ) from None
    core.setFocusDirection(device, direction)


def _exec_label(core: UniMMCore, args: Sequence[str]) -> None:
    """Define state label: Label,<device>,<state>,<label>."""
    if len(args) != 3:  # pragma: no cover
        raise ValueError(f"Label command requires 3 arguments, got {len(args)}")
    device, state_str, label = args
    try:
        state = int(state_str)
    except ValueError:  # pragma: no cover
        raise ValueError(f"State must be an integer, got {state_str!r}") from None
    core.defineStateLabel(device, state, label)


def _exec_config_group(core: UniMMCore, args: Sequence[str]) -> None:
    """Define config group/preset."""
    if len(args) < 1:
        raise ValueError("ConfigGroup command requires at least 1 argument")

    group_name = args[0]

    if len(args) == 1:
        # Just define an empty group
        core.defineConfigGroup(group_name)
    elif len(args) in (4, 5):
        # Define a config setting
        preset_name = args[1]
        device = args[2]
        prop = args[3]
        value = args[4] if len(args) > 4 else ""
        core.defineConfig(group_name, preset_name, device, prop, value)
    else:  # pragma: no cover
        raise ValueError(
            f"ConfigGroup command requires 1, 4, or 5 arguments, got {len(args)}"
        )


def _exec_config_pixel_size(core: UniMMCore, args: Sequence[str]) -> None:
    """Define pixel size config: ConfigPixelSize,<preset>,<device>,<prop>,<value>."""
    if len(args) != 4:  # pragma: no cover
        raise ValueError(
            f"ConfigPixelSize command requires 4 arguments, got {len(args)}"
        )
    preset_name, device, prop, value = args
    core.definePixelSizeConfig(preset_name, device, prop, value)


def _exec_pixel_size_um(core: UniMMCore, args: Sequence[str]) -> None:
    """Set pixel size: PixelSize_um,<preset>,<size>."""
    if len(args) != 2:  # pragma: no cover
        raise ValueError(f"PixelSize_um command requires 2 arguments, got {len(args)}")
    preset_name, size_str = args
    try:
        size = float(size_str)
    except ValueError:  # pragma: no cover
        raise ValueError(f"Pixel size must be a number, got {size_str!r}") from None
    core.setPixelSizeUm(preset_name, size)


def _exec_pixel_size_affine(core: UniMMCore, args: Sequence[str]) -> None:
    """Set pixel size affine transform."""
    if len(args) != 7:  # pragma: no cover
        raise ValueError(
            f"PixelSizeAffine command requires 7 arguments, got {len(args)}"
        )
    preset_name = args[0]
    try:
        affine = [float(x) for x in args[1:]]
    except ValueError:  # pragma: no cover
        raise ValueError(f"Affine values must be numbers, got {args[1:]!r}") from None
    core.setPixelSizeAffine(preset_name, affine)


def _exec_parent_id(core: UniMMCore, args: Sequence[str]) -> None:
    """Set parent hub: Parent,<device>,<parent_hub>."""
    if len(args) != 2:  # pragma: no cover
        raise ValueError(f"Parent command requires 2 arguments, got {len(args)}")
    device, parent = args
    core.setParentLabel(device, parent)


# Map commands to their executors
# Commands not in this map are silently ignored (Equipment, ImageSynchro, etc.)
_COMMAND_EXECUTORS: dict[CFGCommand, Callable[[UniMMCore, Sequence[str]], None]] = {
    CFGCommand.Device: _exec_device,
    CFGCommand.Property: _exec_property,
    CFGCommand.Delay: _exec_delay,
    CFGCommand.FocusDirection: _exec_focus_direction,
    CFGCommand.Label: _exec_label,
    CFGCommand.ConfigGroup: _exec_config_group,
    CFGCommand.ConfigPixelSize: _exec_config_pixel_size,
    CFGCommand.PixelSize_um: _exec_pixel_size_um,
    CFGCommand.PixelSizeAffine: _exec_pixel_size_affine,
    CFGCommand.ParentID: _exec_parent_id,
}

# Add optional pixel size commands if available
if hasattr(CFGCommand, "PixelSize_dxdz"):

    def _exec_pixel_size_dxdz(core: UniMMCore, args: Sequence[str]) -> None:
        if len(args) != 2:  # pragma: no cover
            raise ValueError(
                f"PixelSize_dxdz command requires 2 arguments, got {len(args)}"
            )
        preset_name, value_str = args
        core.setPixelSizedxdz(preset_name, float(value_str))

    _COMMAND_EXECUTORS[CFGCommand.PixelSize_dxdz] = _exec_pixel_size_dxdz

if hasattr(CFGCommand, "PixelSize_dydz"):

    def _exec_pixel_size_dydz(core: UniMMCore, args: Sequence[str]) -> None:
        if len(args) != 2:  # pragma: no cover
            raise ValueError(
                f"PixelSize_dydz command requires 2 arguments, got {len(args)}"
            )
        preset_name, value_str = args
        core.setPixelSizedydz(preset_name, float(value_str))

    _COMMAND_EXECUTORS[CFGCommand.PixelSize_dydz] = _exec_pixel_size_dydz

if hasattr(CFGCommand, "PixelSize_OptimalZUm"):

    def _exec_pixel_size_optimal_z(core: UniMMCore, args: Sequence[str]) -> None:
        if len(args) != 2:  # pragma: no cover
            raise ValueError(
                f"PixelSize_OptimalZUm command requires 2 arguments, got {len(args)}"
            )
        preset_name, value_str = args
        core.setPixelSizeOptimalZUm(preset_name, float(value_str))

    _COMMAND_EXECUTORS[CFGCommand.PixelSize_OptimalZUm] = _exec_pixel_size_optimal_z


# =============================================================================
# Saving
# =============================================================================


def save_system_configuration(
    core: UniMMCore, filename: str | Path, *, prefix_py_devices: bool = True
) -> None:
    """Save the current system configuration to a file.

    This saves both C++ and Python devices.

    Parameters
    ----------
    core : UniMMCore
        The core instance to save configuration from.
    filename : str | Path
        Path to save the configuration file.
    prefix_py_devices : bool, optional
        If True (default), Python device lines are prefixed with `#py ` so they
        are ignored by upstream C++/pymmcore implementations. If False, Python
        device lines are saved without the prefix (config will only be loadable
        by UniMMCore).
    """
    path = Path(filename).expanduser().resolve()

    with open(path, "w") as f:
        for section, lines in _iter_config_sections(core, prefix_py_devices):
            f.write(f"# {section}\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")


def _serialize(
    *args: Any, py_device: bool = False, prefix_py_devices: bool = True
) -> str:
    """Create a config line from arguments."""
    line = CFGCommand.FieldDelimiters.join(str(a) for a in args)
    return f"{PY_PREFIX}{line}" if (py_device and prefix_py_devices) else line


def _iter_config_sections(
    core: UniMMCore, prefix_py_devices: bool = True
) -> Iterable[tuple[str, Iterable[str]]]:
    """Iterate over config sections, yielding (header, lines) tuples."""
    pfx = prefix_py_devices  # shorthand
    yield f"Generated by pymmcore-plus UniMMCore on {timestamp()}", []

    # Reset command
    yield (
        "Unload all devices",
        [
            _serialize(
                CFGCommand.Property, Keyword.CoreDevice, Keyword.CoreInitialize, 0
            )
        ],
    )

    # Load devices
    yield "Load devices", list(_iter_devices(core, pfx))

    # Pre-initialization properties
    yield "Pre-initialization properties", list(_iter_pre_init_props(core, pfx))

    # Hub references
    yield "Hub references", list(_iter_hub_refs(core))

    # Initialize command
    yield (
        "Initialize",
        [
            _serialize(
                CFGCommand.Property, Keyword.CoreDevice, Keyword.CoreInitialize, 1
            )
        ],
    )

    # Delays
    yield "Delays", list(_iter_delays(core, pfx))

    # Focus directions
    yield "Stage focus directions", list(_iter_focus_directions(core, pfx))

    # Labels
    yield "Labels", list(_iter_labels(core, pfx))

    # Config groups
    yield "Configuration presets", list(_iter_config_groups(core, pfx))

    # Pixel size configs
    yield "PixelSize settings", list(_iter_pixel_size_configs(core, pfx))

    # Core device roles
    yield "Roles", list(_iter_roles(core, pfx))


def _iter_devices(core: UniMMCore, prefix_py_devices: bool = True) -> Iterable[str]:
    """Iterate over device load commands."""
    for label in core.getLoadedDevices():
        if label == Keyword.CoreDevice:  # type: ignore[comparison-overlap]
            continue
        is_py = core.isPyDevice(label)
        library = core.getDeviceLibrary(label)
        device_name = core.getDeviceName(label)
        # Use pyDevice command for Python devices, Device for C++ devices
        cmd = _PY_DEVICE_CMD if is_py else CFGCommand.Device
        yield _serialize(
            cmd,
            label,
            library,
            device_name,
            py_device=is_py,
            prefix_py_devices=prefix_py_devices,
        )


def _iter_pre_init_props(
    core: UniMMCore, prefix_py_devices: bool = True
) -> Iterable[str]:
    """Iterate over pre-initialization property commands."""
    for label in core.getLoadedDevices():
        if label == Keyword.CoreDevice:  # type: ignore[comparison-overlap]
            continue
        is_py = core.isPyDevice(label)
        for prop_name in core.getDevicePropertyNames(label):
            if core.isPropertyPreInit(label, prop_name):
                value = core.getProperty(label, prop_name)
                yield _serialize(
                    CFGCommand.Property,
                    label,
                    prop_name,
                    value,
                    py_device=is_py,
                    prefix_py_devices=prefix_py_devices,
                )


def _iter_hub_refs(core: UniMMCore) -> Iterable[str]:
    """Iterate over parent hub reference commands."""
    for label in core.getLoadedDevices():
        if label == Keyword.CoreDevice:  # type: ignore[comparison-overlap]
            continue
        is_py = core.isPyDevice(label)
        # Python devices don't have parent labels (yet)
        if is_py:
            continue
        try:
            parent = core.getParentLabel(label)
        except RuntimeError:
            continue
        if parent:
            yield _serialize(CFGCommand.ParentID, label, parent, py_device=is_py)


def _iter_delays(core: UniMMCore, prefix_py_devices: bool = True) -> Iterable[str]:
    """Iterate over device delay commands."""
    for label in core.getLoadedDevices():
        if label == Keyword.CoreDevice:  # type: ignore[comparison-overlap]
            continue
        delay = core.getDeviceDelayMs(label)
        if delay > 0:
            yield _serialize(
                CFGCommand.Delay,
                label,
                delay,
                py_device=core.isPyDevice(label),
                prefix_py_devices=prefix_py_devices,
            )


def _iter_focus_directions(
    core: UniMMCore, prefix_py_devices: bool = True
) -> Iterable[str]:
    """Iterate over focus direction commands."""
    for label in core.getLoadedDevicesOfType(DeviceType.Stage):
        is_py = core.isPyDevice(label)
        direction = core.getFocusDirection(label)
        yield _serialize(
            CFGCommand.FocusDirection,
            label,
            direction.value,
            py_device=is_py,
            prefix_py_devices=prefix_py_devices,
        )


def _iter_labels(core: UniMMCore, prefix_py_devices: bool = True) -> Iterable[str]:
    """Iterate over state device label commands."""
    for label in core.getLoadedDevicesOfType(DeviceType.State):
        is_py = core.isPyDevice(label)
        labels = core.getStateLabels(label)
        for state, state_label in enumerate(labels):
            if state_label:
                yield _serialize(
                    CFGCommand.Label,
                    label,
                    state,
                    state_label,
                    py_device=is_py,
                    prefix_py_devices=prefix_py_devices,
                )


def _iter_config_groups(
    core: UniMMCore, prefix_py_devices: bool = True
) -> Iterable[str]:
    """Iterate over config group commands."""
    for group_name in core.getAvailableConfigGroups():
        presets = core.getAvailableConfigs(group_name)
        if not presets:
            # Empty group
            yield _serialize(CFGCommand.ConfigGroup, group_name)
            continue

        for preset_name in presets:
            config = core.getConfigData(group_name, preset_name, native=True)
            for i in range(config.size()):
                setting = config.getSetting(i)
                device = setting.getDeviceLabel()
                prop = setting.getPropertyName()
                value = setting.getPropertyValue()
                is_py = core.isPyDevice(device)
                yield _serialize(
                    CFGCommand.ConfigGroup,
                    group_name,
                    preset_name,
                    device,
                    prop,
                    value,
                    py_device=is_py,
                    prefix_py_devices=prefix_py_devices,
                )


def _iter_pixel_size_configs(
    core: UniMMCore, prefix_py_devices: bool = True
) -> Iterable[str]:
    """Iterate over pixel size config commands."""
    for preset_name in core.getAvailablePixelSizeConfigs():
        config = core.getPixelSizeConfigData(preset_name, native=True)
        for i in range(config.size()):
            setting = config.getSetting(i)
            device = setting.getDeviceLabel()
            prop = setting.getPropertyName()
            value = setting.getPropertyValue()
            # Pixel size configs typically only reference C++ devices
            is_py = core.isPyDevice(device)
            yield _serialize(
                CFGCommand.ConfigPixelSize,
                preset_name,
                device,
                prop,
                value,
                py_device=is_py,
                prefix_py_devices=prefix_py_devices,
            )

        # Pixel size
        size = core.getPixelSizeUmByID(preset_name)
        yield _serialize(CFGCommand.PixelSize_um, preset_name, size)

        # Affine transform
        affine = core.getPixelSizeAffineByID(preset_name)
        if affine and any(v != 0 for v in affine):
            yield _serialize(CFGCommand.PixelSizeAffine, preset_name, *affine)

        # Optional extended pixel size properties
        if hasattr(core, "getPixelSizedxdz"):
            dxdz = core.getPixelSizedxdz(preset_name)
            if dxdz:
                yield _serialize(CFGCommand.PixelSize_dxdz, preset_name, dxdz)
        if hasattr(core, "getPixelSizedydz"):
            dydz = core.getPixelSizedydz(preset_name)
            if dydz:
                yield _serialize(CFGCommand.PixelSize_dydz, preset_name, dydz)
        if hasattr(core, "getPixelSizeOptimalZUm"):
            optimal_z = core.getPixelSizeOptimalZUm(preset_name)
            if optimal_z:
                yield _serialize(
                    CFGCommand.PixelSize_OptimalZUm, preset_name, optimal_z
                )


def _iter_roles(core: UniMMCore, prefix_py_devices: bool = True) -> Iterable[str]:
    """Iterate over core device role commands."""
    roles = [
        (Keyword.CoreCamera, core.getCameraDevice),
        (Keyword.CoreShutter, core.getShutterDevice),
        (Keyword.CoreFocus, core.getFocusDevice),
        (Keyword.CoreXYStage, core.getXYStageDevice),
        (Keyword.CoreAutoFocus, core.getAutoFocusDevice),
        (Keyword.CoreSLM, core.getSLMDevice),
        (Keyword.CoreGalvo, core.getGalvoDevice),
    ]

    for role_keyword, getter in roles:
        try:
            if device := getter():
                yield _serialize(
                    CFGCommand.Property,
                    Keyword.CoreDevice,
                    role_keyword,
                    device,
                    py_device=core.isPyDevice(device),
                    prefix_py_devices=prefix_py_devices,
                )
        except Exception:  # pragma: no cover
            # Some getters may not be available
            pass

    # AutoShutter setting
    yield _serialize(
        CFGCommand.Property,
        Keyword.CoreDevice,
        Keyword.CoreAutoShutter,
        int(core.getAutoShutter()),
    )
