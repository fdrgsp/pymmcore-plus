"""Read a smart-microscopy script's contract without executing it.

Everything a front end or runner needs before a run -- name, parameters (and
enough about them to build a form), default execution/sync modes, the frame
filter, which hooks exist and with what signature -- is read from the source
with `ast`. Inspecting a script therefore never runs user code; that only
happens inside the chosen executor.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pymmcore_plus.smart._api import (
    API_VERSION,
    EXECUTION_MODES,
    ORIGINS,
    SEQUENCING_MODES,
    SYNC_MODES,
    ExecutionMode,
    Origin,
    SequencingMode,
    SyncMode,
)

if TYPE_CHECKING:
    import useq

ParamKind = Literal["float", "int", "bool", "str", "choice"]

_KNOWN_CONSTANTS: Final = frozenset(
    {
        "API_VERSION",
        "NAME",
        "DESCRIPTION",
        "EXECUTION",
        "SYNC",
        "SEQUENCING",
        "ANALYZE",
        "PARAMETERS",
    }
)
_HOOKS: Final[dict[str, int]] = {
    "analyze": 3,
    "setup": 1,
    "after_base": 1,
    "teardown": 1,
}
_SPEC_KEYS: Final = frozenset(
    {"default", "min", "max", "step", "choices", "label", "tooltip"}
)


class ScriptError(Exception):
    """A script that cannot be used, with the offending line when known."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message if line is None else f"line {line}: {message}")
        self.message = message
        self.line = line


@dataclass(frozen=True)
class ParamDef:
    """One entry of a script's ``PARAMETERS``, ready to build a widget from."""

    name: str
    default: Any
    kind: ParamKind
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None
    label: str = ""
    tooltip: str = ""

    def coerce(self, value: Any) -> Any:
        """Return *value* converted to this parameter's type, or raise ValueError."""
        if self.kind == "choice":
            if value not in (self.choices or ()):
                raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
            return value
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name}: expected a bool, got {value!r}")
            return value
        if self.kind == "str":
            return str(value)
        number = int(value) if self.kind == "int" else float(value)
        if self.min is not None and number < self.min:
            raise ValueError(f"{self.name}: {number} is below the minimum {self.min}")
        if self.max is not None and number > self.max:
            raise ValueError(f"{self.name}: {number} is above the maximum {self.max}")
        return number


@dataclass(frozen=True)
class AnalyzeFilter:
    """Which frames are sent to ``analyze`` (the script's ``ANALYZE``)."""

    channels: tuple[str, ...] | None = None
    every_nth: int = 1
    origins: frozenset[Origin] = field(default_factory=lambda: frozenset(ORIGINS))  # type: ignore[arg-type]

    def accepts(self, frame_id: int, event: useq.MDAEvent, origin: Origin) -> bool:
        if origin not in self.origins:
            return False
        if self.channels is not None:
            config = event.channel.config if event.channel is not None else None
            if config not in self.channels:
                return False
        return frame_id % self.every_nth == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": None if self.channels is None else list(self.channels),
            "every_nth": self.every_nth,
            "origins": sorted(self.origins),
        }


@dataclass(frozen=True)
class ScriptSpec:
    """Everything statically known about a script."""

    path: Path
    source: str
    sha256: str
    api_version: int
    name: str
    description: str
    execution: ExecutionMode
    sync: SyncMode
    sequencing: SequencingMode
    filter: AnalyzeFilter
    params: tuple[ParamDef, ...]
    has_setup: bool
    has_teardown: bool
    has_after_base: bool = False

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Defaults updated with valid *overrides*; unknown or invalid ones are dropped.

        Overrides usually come from saved settings, which may predate an edit
        to the script that renamed a parameter or changed its range.
        """
        resolved = self.default_params()
        by_name = {p.name: p for p in self.params}
        for name, value in (overrides or {}).items():
            if (param := by_name.get(name)) is None:
                continue
            try:
                resolved[name] = param.coerce(value)
            except (TypeError, ValueError):
                continue
        return resolved


def inspect_script(path: str | Path) -> ScriptSpec:
    """Read and validate the script at *path*. Raises `ScriptError`."""
    path = Path(path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ScriptError(f"Cannot read {path}: {e.strerror or e}") from e
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ScriptError(f"{path.name} is not UTF-8 text") from e
    return inspect_source(source, path=path, sha256=hashlib.sha256(raw).hexdigest())


def inspect_source(
    source: str, *, path: Path | None = None, sha256: str | None = None
) -> ScriptSpec:
    """Validate script *source* (see `inspect_script`)."""
    filename = str(path) if path else "<script>"
    try:
        tree = ast.parse(source, filename=filename)
        compile(tree, filename, "exec")  # catches errors ast.parse alone accepts
    except SyntaxError as e:
        raise ScriptError(f"Syntax error: {e.msg}", e.lineno) from e

    constants = _read_constants(tree)
    hooks = _read_hooks(tree)
    if "analyze" not in hooks:
        raise ScriptError("The script must define analyze(image, frame, ctx).")

    if "API_VERSION" not in constants:
        raise ScriptError(f"The script must declare API_VERSION = {API_VERSION}.")
    api_version, line = constants["API_VERSION"]
    if api_version != API_VERSION:
        raise ScriptError(
            f"API_VERSION {api_version!r} is not supported (expected {API_VERSION}).",
            line,
        )

    name = _str_constant(constants, "NAME") or (path.stem if path else "script")
    execution = _choice_constant(constants, "EXECUTION", EXECUTION_MODES, "thread")
    sync = _choice_constant(constants, "SYNC", SYNC_MODES, "blocking")
    sequencing = _choice_constant(constants, "SEQUENCING", SEQUENCING_MODES, "safe")
    return ScriptSpec(
        path=path or Path("<script>"),
        source=source,
        sha256=sha256 or hashlib.sha256(source.encode()).hexdigest(),
        api_version=API_VERSION,
        name=name,
        description=_str_constant(constants, "DESCRIPTION"),
        execution=cast("ExecutionMode", execution),
        sync=cast("SyncMode", sync),
        sequencing=cast("SequencingMode", sequencing),
        filter=_read_filter(constants),
        params=_read_params(constants),
        has_setup="setup" in hooks,
        has_teardown="teardown" in hooks,
        has_after_base="after_base" in hooks,
    )


# ---------------------------------------------------------------- internals


def _read_constants(tree: ast.Module) -> dict[str, tuple[Any, int]]:
    """Literal values of the known module-level constants, with their line."""
    found: dict[str, tuple[Any, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id not in _KNOWN_CONSTANTS:
            continue
        try:
            found[target.id] = (ast.literal_eval(value), node.lineno)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
            raise ScriptError(
                f"{target.id} must be a literal value (numbers, strings, lists, "
                "dicts...), so it can be read without running the script.",
                node.lineno,
            ) from e
    return found


def _read_hooks(tree: ast.Module) -> set[str]:
    hooks: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in _HOOKS:
            raise ScriptError(
                f"{node.name}() must be a plain def, not async def.", node.lineno
            )
        if not isinstance(node, ast.FunctionDef) or node.name not in _HOOKS:
            continue
        expected = _HOOKS[node.name]
        args = node.args
        positional = len(args.posonlyargs) + len(args.args)
        required = positional - len(args.defaults)
        if not (required <= expected <= positional or args.vararg is not None):
            signature = "(image, frame, ctx)" if node.name == "analyze" else "(ctx)"
            raise ScriptError(
                f"{node.name}{signature} must accept exactly {expected} positional "
                f"argument{'s' if expected > 1 else ''}.",
                node.lineno,
            )
        hooks.add(node.name)
    return hooks


def _str_constant(constants: dict[str, tuple[Any, int]], key: str) -> str:
    if key not in constants:
        return ""
    value, line = constants[key]
    if not isinstance(value, str):
        raise ScriptError(f"{key} must be a string.", line)
    return value


def _choice_constant(
    constants: dict[str, tuple[Any, int]],
    key: str,
    choices: tuple[str, ...],
    default: str,
) -> str:
    if key not in constants:
        return default
    value, line = constants[key]
    if value not in choices:
        raise ScriptError(f"{key} must be one of {list(choices)}, not {value!r}.", line)
    return str(value)


def _read_filter(constants: dict[str, tuple[Any, int]]) -> AnalyzeFilter:
    if "ANALYZE" not in constants:
        return AnalyzeFilter()
    value, line = constants["ANALYZE"]
    if not isinstance(value, dict):
        raise ScriptError("ANALYZE must be a dict.", line)
    unknown = set(value) - {"channels", "every_nth", "origins"}
    if unknown:
        raise ScriptError(f"ANALYZE has unknown keys: {sorted(unknown)}.", line)

    channels = value.get("channels")
    if channels is not None:
        if isinstance(channels, str):
            channels = [channels]
        if not isinstance(channels, (list, tuple)) or not all(
            isinstance(c, str) for c in channels
        ):
            raise ScriptError("ANALYZE['channels'] must be a list of strings.", line)
        channels = tuple(channels)

    every_nth = value.get("every_nth", 1)
    if isinstance(every_nth, bool) or not isinstance(every_nth, int) or every_nth < 1:
        raise ScriptError("ANALYZE['every_nth'] must be an integer >= 1.", line)

    origins = value.get("origins", list(ORIGINS))
    if isinstance(origins, str):
        origins = [origins]
    if (
        not isinstance(origins, (list, tuple))
        or not origins
        or any(o not in ORIGINS for o in origins)
    ):
        raise ScriptError(
            f"ANALYZE['origins'] must be a subset of {list(ORIGINS)}.", line
        )
    return AnalyzeFilter(
        channels=channels,
        every_nth=every_nth,
        origins=frozenset(cast("list[Origin]", origins)),
    )


def _read_params(constants: dict[str, tuple[Any, int]]) -> tuple[ParamDef, ...]:
    if "PARAMETERS" not in constants:
        return ()
    value, line = constants["PARAMETERS"]
    if not isinstance(value, dict):
        raise ScriptError("PARAMETERS must be a dict of name -> value or spec.", line)
    params: list[ParamDef] = []
    for name, entry in value.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ScriptError(f"PARAMETERS key {name!r} must be an identifier.", line)
        params.append(_param_def(name, entry, line))
    return tuple(params)


def _param_def(name: str, entry: Any, line: int) -> ParamDef:
    spec: dict[str, Any] = entry if isinstance(entry, dict) else {"default": entry}
    if unknown := set(spec) - _SPEC_KEYS:
        raise ScriptError(
            f"PARAMETERS[{name!r}] has unknown keys {sorted(unknown)}.", line
        )
    if "default" not in spec:
        raise ScriptError(f"PARAMETERS[{name!r}] needs a 'default'.", line)
    default = spec["default"]
    label = str(spec.get("label", name.replace("_", " ").capitalize()))
    tooltip = str(spec.get("tooltip", ""))

    if "choices" in spec:
        choices = spec["choices"]
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ScriptError(f"PARAMETERS[{name!r}]['choices'] must be a list.", line)
        if default not in choices:
            raise ScriptError(
                f"PARAMETERS[{name!r}]: default must be one of its choices.", line
            )
        return ParamDef(
            name,
            default,
            "choice",
            choices=tuple(choices),
            label=label,
            tooltip=tooltip,
        )

    # bool before int: bool is an int subclass.
    kind: ParamKind
    if isinstance(default, bool):
        kind = "bool"
    elif isinstance(default, int):
        kind = "int"
    elif isinstance(default, float):
        kind = "float"
    elif isinstance(default, str):
        kind = "str"
    else:
        raise ScriptError(
            f"PARAMETERS[{name!r}]: default must be a number, string or bool "
            f"(or give 'choices'), not {type(default).__name__}.",
            line,
        )

    bounds: dict[str, float | None] = {}
    for key in ("min", "max", "step"):
        bound = spec.get(key)
        if bound is not None:
            if kind not in ("int", "float"):
                raise ScriptError(
                    f"PARAMETERS[{name!r}]: '{key}' only applies to numbers.", line
                )
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ScriptError(
                    f"PARAMETERS[{name!r}]['{key}'] must be a number.", line
                )
        bounds[key] = bound
    param = ParamDef(
        name,
        default,
        kind,
        min=bounds["min"],
        max=bounds["max"],
        step=bounds["step"],
        label=label,
        tooltip=tooltip,
    )
    try:
        param.coerce(default)
    except ValueError as e:
        raise ScriptError(
            f"PARAMETERS[{name!r}]: default is out of range ({e}).", line
        ) from e
    return param
