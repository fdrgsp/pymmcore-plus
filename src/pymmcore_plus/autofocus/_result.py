from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Literal


@dataclass(frozen=True)
class AutofocusResult:
    """The outcome of a single autofocus attempt.

    Emitted by
    [`autofocusFinished`][pymmcore_plus.mda.PMDASignaler.autofocusFinished] after
    every autofocus event in an MDA, whether it succeeded or not, and returned by
    software autofocus routines.

    Attributes
    ----------
    kind : Literal["hardware", "software"]
        Whether the focus was found by an autofocus *device* (`"hardware"`) or by an
        image-based routine (`"software"`).
    method : str
        For `"software"`, the name of the routine that ran.  For `"hardware"`, the
        label of the autofocus device.
    focus_device : str
        Label of the stage device that was moved.
    z_before : float
        Position of `focus_device` before autofocus, or `nan` if it could not be read.
    z_after : float
        Position of `focus_device` after autofocus, or `nan` if it could not be read.
        On failure, this is where the drive was left (engines restore `z_before`).
    succeeded : bool
        Whether focus was found.
    message : str
        Empty on success; otherwise why autofocus failed.
    scores : tuple[tuple[float, float], ...]
        `(z, score)` pairs measured by a software routine, in the order they were
        measured.  Always empty for hardware autofocus, which exposes no score.
    n_images : int
        Number of images a software routine acquired.  Always 0 for hardware.
    """

    kind: Literal["hardware", "software"]
    method: str
    focus_device: str
    z_before: float
    z_after: float
    succeeded: bool
    message: str = ""
    scores: tuple[tuple[float, float], ...] = field(default=())
    n_images: int = 0

    @property
    def delta_z(self) -> float:
        """How far `focus_device` moved, or `nan` if either position is unknown."""
        if isnan(self.z_before) or isnan(self.z_after):
            return float("nan")
        return self.z_after - self.z_before
