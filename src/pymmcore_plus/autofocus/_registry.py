"""The catalog of software autofocus routines.

`useq` names a routine but does not define which exist, so this is where a name like
`"oughtafocus"` becomes something that can run.  The built-in routines are registered
here; anything else can be added with
[`register_software_autofocus`][pymmcore_plus.autofocus.register_software_autofocus].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._result import AutofocusResult
from ._routines import duo, jaf, oughtafocus
from ._settings import DuoSettings, JAFSettings, OughtaFocusSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymmcore_plus.core import CMMCorePlus

__all__ = [
    "SoftwareAutofocusMethod",
    "available_methods",
    "get_method",
    "register_software_autofocus",
    "run_software_autofocus",
    "settings_model",
]


@dataclass(frozen=True)
class SoftwareAutofocusMethod:
    """A registered software autofocus routine.

    Attributes
    ----------
    name : str
        How a [`useq.SoftwareAutofocus`][] action refers to this routine.
    run : Callable
        `run(core, focus_device, settings, *, should_cancel) -> AutofocusResult`.
    settings_model : type
        The dataclass describing this routine's settings.  A GUI can build a form
        from `dataclasses.fields()` of it.
    description : str
        One line on what the routine does, for a method picker.
    manages_continuous_focus : bool
        True if the routine drives the hardware autofocus itself, in which case the
        engine must not switch continuous focus off before running it.
    """

    name: str
    run: Callable[..., AutofocusResult]
    settings_model: type
    description: str = ""
    manages_continuous_focus: bool = False


_REGISTRY: dict[str, SoftwareAutofocusMethod] = {}


def register_software_autofocus(
    name: str,
    run: Callable[..., AutofocusResult],
    settings_model: type,
    *,
    description: str = "",
    manages_continuous_focus: bool = False,
) -> None:
    """Add a software autofocus routine, or replace one of the same name."""
    _REGISTRY[name] = SoftwareAutofocusMethod(
        name=name,
        run=run,
        settings_model=settings_model,
        description=description,
        manages_continuous_focus=manages_continuous_focus,
    )


def available_methods() -> list[str]:
    """Return the names of every registered routine, sorted."""
    return sorted(_REGISTRY)


def get_method(name: str) -> SoftwareAutofocusMethod:
    """Return the routine registered as `name`."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown software autofocus method {name!r}. "
            f"Available: {available_methods()}."
        ) from None


def settings_model(name: str) -> type:
    """Return the settings dataclass for the routine registered as `name`."""
    return get_method(name).settings_model


def run_software_autofocus(
    core: CMMCorePlus,
    method: str,
    settings: dict[str, Any] | None = None,
    *,
    focus_device: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AutofocusResult:
    """Run a software autofocus routine now, outside of any acquisition.

    This is the same path an MDA takes, so a "focus now" button and an autofocus
    event behave identically.

    Parameters
    ----------
    core : CMMCorePlus
        The core to drive.
    method : str
        Name of a registered routine; see `available_methods()`.
    settings : dict | None
        Routine-specific settings; see that routine's `settings_model`.
    focus_device : str | None
        Stage to move.  `None` uses the core's current focus device.
    should_cancel : Callable[[], bool] | None
        Polled before every image; raises `AutofocusCancelled` when it returns True.

    Returns
    -------
    AutofocusResult
        Where focus ended up, and the focus curve it measured.
    """
    entry = get_method(method)
    if not (drive := focus_device or core.getFocusDevice()):
        return AutofocusResult(
            kind="software",
            method=method,
            focus_device="",
            z_before=float("nan"),
            z_after=float("nan"),
            succeeded=False,
            message="No focus device available.",
        )
    return entry.run(core, drive, settings, should_cancel=should_cancel)


register_software_autofocus(
    "oughtafocus",
    oughtafocus,
    OughtaFocusSettings,
    description="Search a Z range for the sharpest image.",
)
register_software_autofocus(
    "jaf",
    jaf,
    JAFSettings,
    description="A coarse Z scan followed by a fine one, stopping early past the peak.",
)
register_software_autofocus(
    "duo",
    duo,
    DuoSettings,
    description="Run two routines in sequence, e.g. a coarse one then a precise one.",
)
