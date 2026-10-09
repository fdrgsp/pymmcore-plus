"""The catalog of software autofocus routines.

`useq` names a routine but does not define which exist, so this is where a name like
`"oughtafocus"` becomes something that can run.  The built-in routines are registered
here; anything else can be added with
[`register_software_autofocus`][pymmcore_plus.autofocus.register_software_autofocus].
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._result import AutofocusResult
from ._routines import _restore, duo, jaf, oughtafocus
from ._settings import DuoSettings, JAFSettings, OughtaFocusSettings, from_dict

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
        How this routine finds focus, for a method picker to show. Several lines is
        fine: the point is that someone choosing between routines can tell what
        each will actually do, and what it will cost in images.
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
    """Return the names of every registered routine, most generally useful first.

    A caller offering a choice can take the first as a default: the order is
    registration order, and the built-ins are registered from the one that works on
    the widest range of samples to the one needing the most setting up.
    """
    return list(_REGISTRY)


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
    prepare_microscope(core, entry)
    return run_guarded(entry, core, drive, settings, should_cancel=should_cancel)


# --------------------------------------------------------------------------------
# Shared with the MDA engine, so a routine run on its own -- from a settings
# dialog's Test button, say -- runs under the same conditions as one an
# acquisition runs, and is held to the same promises.


def prepare_microscope(core: CMMCorePlus, entry: SoftwareAutofocusMethod) -> None:
    """Put the microscope in the state a software routine needs.

    - Live, or any other sequence acquisition, is stopped: a routine acquires its
      own images one at a time, and cannot while the camera is streaming.
    - A locked hardware autofocus is switched off -- and left off. It would fight
      the routine for the stage, and re-engaging it afterwards would pull the focus
      straight back off the position the routine just measured. Routines that
      drive the autofocus device themselves opt out.
    """
    if core.isSequenceRunning():
        core.stopSequenceAcquisition()
    if not entry.manages_continuous_focus and core.isContinuousFocusLocked():
        core.enableContinuousFocus(False)


def settings_problem(entry: SoftwareAutofocusMethod, settings: Any) -> str:
    """Why `settings` are not valid for `entry`, or "" if they are.

    A configuration error will not fix itself by retrying, so a caller can report
    it once, before anything moves. A routine registered with settings that are
    not a dataclass is left to check its own.
    """
    if not dataclasses.is_dataclass(entry.settings_model):
        return ""
    try:
        from_dict(entry.settings_model, settings)
    except (TypeError, ValueError) as e:
        return str(e)
    return ""


def run_guarded(
    entry: SoftwareAutofocusMethod,
    core: CMMCorePlus,
    focus_device: str,
    settings: Any,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> AutofocusResult:
    """Run `entry`, holding it to the promise every routine makes.

    A routine that fails, raises or is cancelled leaves the focus device where it
    found it. The built-ins keep that promise themselves, but every routine passes
    through here -- ones registered by users too -- so it is enforced here as well.
    It is also what lets an acquisition's retries each start from the same place,
    rather than from wherever the last attempt gave up.

    Only the stage is dealt with here: an exception is re-raised once it is back,
    because what to make of one is the caller's business -- a mistake in the
    settings should be loud when run by hand, and logged when run by an engine
    that has an acquisition to carry on with.
    """
    z_before = core.getPosition(focus_device)

    def put_back() -> None:
        # only if it moved: a routine that failed before moving -- on a misspelt
        # setting, say -- should not cost a stage move
        if core.getPosition(focus_device) != z_before:
            _restore(core, focus_device, z_before)

    try:
        result = entry.run(core, focus_device, settings, should_cancel=should_cancel)
    except Exception:
        put_back()
        raise
    if not result.succeeded:
        put_back()
    return result


# Registered most generally useful first: `available_methods()` keeps this order, so
# a method picker can take the first as its default.
register_software_autofocus(
    "oughtafocus",
    oughtafocus,
    OughtaFocusSettings,
    description=(
        "Acquires images along Z and moves to the sharpest one.\n"
        "\n"
        "You choose how sharpness is measured, and how it searches: walking toward "
        "the peak takes the fewest images but needs a single clear peak, while "
        "scanning the whole range costs more images and survives noise.\n"
        "\n"
        "The general-purpose choice, and a good place to start."
    ),
)
register_software_autofocus(
    "jaf",
    jaf,
    JAFSettings,
    description=(
        "Scans Z coarsely to find roughly where focus is, then finely around that "
        "point.\n"
        "\n"
        "Each pass walks outward and stops as soon as the image stops improving, so "
        "it wastes few images past the peak. Sharpness is measured on a central crop "
        "after a median filter, which makes it tolerant of noise and of a bright "
        "edge of the frame. The fine pass can use a different channel.\n"
        "\n"
        "Reaches focus from further away than a single search of the same cost."
    ),
)
register_software_autofocus(
    "duo",
    duo,
    DuoSettings,
    description=(
        "Runs two routines one after the other, the second starting where the first "
        "ended.\n"
        "\n"
        "By default a wide coarse scan followed by a narrow precise search, which "
        "finds focus from further out than either manages alone -- at the cost of "
        "both routines' images.\n"
        "\n"
        "Use it when focus can start far off, as after moving to a new well."
    ),
)
