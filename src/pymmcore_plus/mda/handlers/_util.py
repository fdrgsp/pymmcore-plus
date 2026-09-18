from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import useq


def get_full_sequence_axes(sequence: useq.MDASequence) -> tuple[str, ...]:
    """Get all root and sub-sequence axes in the root acquisition order."""
    axes = set(sequence.used_axes)
    extra_axes: list[str] = []
    positions = list(sequence.stage_positions)

    while positions:
        position = positions.pop(0)
        if subsequence := position.sequence:
            for axis in subsequence.used_axes:
                if axis not in axes:
                    axes.add(axis)
                    extra_axes.append(axis)
            positions.extend(subsequence.stage_positions)

    ordered = [axis for axis in sequence.axis_order if axis in axes]
    ordered.extend(axis for axis in extra_axes if axis not in ordered)
    return tuple(ordered)


def position_sizes(seq: useq.MDASequence) -> list[dict[str, int]]:
    """Return a list of size dicts for each position in the sequence.

    There will be one dict for each position in the sequence. Each dict will contain
    `{dim: size}` pairs for each dimension in the sequence. Dimensions with no size
    will be omitted, though singletons will be included.
    """
    axes = get_full_sequence_axes(seq)
    main_sizes = dict(seq.sizes)
    main_sizes.pop("p", None)  # remove position

    if not seq.stage_positions:
        # this is a simple MDASequence
        return [{axis: main_sizes[axis] for axis in axes if main_sizes.get(axis)}]

    sizes = []
    for position in seq.stage_positions:
        subsequence_sizes = dict(position.sequence.sizes) if position.sequence else {}
        sizes.append(
            {
                axis: size
                for axis in axes
                if axis != "p"
                and (size := subsequence_sizes.get(axis) or main_sizes.get(axis))
            }
        )
    return sizes
