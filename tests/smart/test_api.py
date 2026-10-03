from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq

from pymmcore_plus.smart import STOP, AnalysisContext, PixelConfig, Response, SystemInfo
from pymmcore_plus.smart._api import needs_fov, normalise_response, with_fov

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pymmcore_plus import CMMCorePlus

SYSTEM = SystemInfo(
    image_width=512,
    image_height=256,
    pixel_size_um=1.0,
    pixel_config="Res10x",
    pixel_configs=(
        PixelConfig("Res10x", 1.0, (("Objective", "Label", "10X"),)),
        PixelConfig("Res40x", 0.25, (("Objective", "Label", "40X"),)),
        PixelConfig("Uncal", 0.0, (("Objective", "Label", "Uncal"),)),
    ),
    property_state={("Objective", "Label"): "10X"},
)


def _switch(label: str) -> useq.MDAEvent:
    return useq.MDAEvent(
        action=useq.CustomAction(name="switch"),
        properties=[useq.PropertyTuple("Objective", "Label", label)],
    )


def _subgrid(**grid: float) -> useq.MDASequence:
    return useq.MDASequence(
        stage_positions=(
            useq.Position(
                x=1000,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2, **grid)
                ),
            ),
        )
    )


def test_none_means_no_action() -> None:
    assert normalise_response(None) == Response(events=())


def test_single_event_and_iterables_become_tuples() -> None:
    event = useq.MDAEvent(exposure=5)
    assert normalise_response(event).events == (event,)
    assert normalise_response([event, event]).events == (event, event)
    assert normalise_response(e for e in [event]).events == (event,)


def test_sequences_are_expanded_in_order() -> None:
    seq = useq.MDASequence(z_plan=useq.ZRangeAround(range=2, step=1))
    first, last = useq.MDAEvent(exposure=1), useq.MDAEvent(exposure=2)
    events = normalise_response([first, seq, last]).events
    assert isinstance(events, tuple)
    assert len(events) == 5
    assert events[0] is first and events[-1] is last


def test_response_options_are_kept() -> None:
    event = useq.MDAEvent()
    out = normalise_response(
        Response(events=[event], priority="end", timing="absolute", drop_base=True)
    )
    assert out.events == (event,)
    assert (out.priority, out.timing, out.drop_base) == ("end", "absolute", True)
    assert normalise_response(STOP).stop


@pytest.mark.parametrize("bad", [42, "events", {"a": 1}, [useq.MDAEvent(), 3]])
def test_bad_return_values(bad: object) -> None:
    with pytest.raises(TypeError):
        normalise_response(bad)


def test_event_cap_stops_endless_generators() -> None:
    def endless() -> Iterator[useq.MDAEvent]:
        while True:
            yield useq.MDAEvent()

    with pytest.raises(ValueError, match="more than 10"):
        normalise_response(endless(), max_events=10)


def test_invalid_response_options() -> None:
    with pytest.raises(ValueError):
        Response(priority="soon")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Response(timing="later")  # type: ignore[arg-type]


def test_context_log_and_record(tmp_path: Path) -> None:
    ctx = AnalysisContext({"a": 1}, tmp_path, "thread", system=SYSTEM)
    assert ctx.system is SYSTEM
    with pytest.raises(TypeError):
        ctx.params["a"] = 2  # type: ignore[index]
    ctx.log("hello")
    ctx.record(mean=np.float32(1.5), count=np.int64(3), label="x", hit=True)
    with pytest.raises(TypeError):
        ctx.record(arr=np.zeros(3))
    with pytest.raises(ValueError):
        ctx.log("x", level="loud")  # type: ignore[arg-type]
    logs, records = ctx._drain()
    assert logs == [("info", "hello")]
    assert records == {"mean": 1.5, "count": 3, "label": "x", "hit": True}
    assert type(records["count"]) is int


# ----------------------------------------------------------- grids / FOV


def test_grid_without_fov_is_kept_unexpanded() -> None:
    """It can only be sized when it runs, with the pixel size in effect then."""
    grid = _subgrid()
    assert needs_fov(grid)
    assert normalise_response(grid).events == (grid,)
    event = useq.MDAEvent()
    assert normalise_response([_switch("40X"), grid, event]).events == (
        _switch("40X"),
        grid,
        event,
    )


def test_grid_with_fov_is_expanded_immediately() -> None:
    sized = _subgrid(fov_width=100, fov_height=100)
    assert not needs_fov(sized)
    events = normalise_response(sized).events
    assert [e.x_pos for e in events] == [950.0, 1050.0]


def test_with_fov_sizes_nested_grids() -> None:
    events = list(with_fov(_subgrid(), 512.0, 256.0))
    # 512 um tiles, centered on x=1000
    assert [e.x_pos for e in events] == [744.0, 1256.0]
    # the original is untouched
    assert needs_fov(_subgrid())


def test_with_fov_keeps_explicit_values() -> None:
    seq = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(rows=1, columns=2, fov_width=10)
    )
    sized = with_fov(seq, 512.0, 256.0)
    assert sized.grid_plan is not None
    assert (sized.grid_plan.fov_width, sized.grid_plan.fov_height) == (10.0, 256.0)


def test_system_info_helpers() -> None:
    assert SYSTEM.fov_um() == (512.0, 256.0)
    assert SYSTEM.fov_um("Res40x") == (128.0, 64.0)
    with pytest.raises(ValueError, match="not set"):
        SYSTEM.fov_um("Uncal")
    with pytest.raises(ValueError, match="No pixel configuration"):
        SYSTEM.fov_um("nope")
    state = SYSTEM.state_after([("Objective", "Label", "40X")])
    config = SYSTEM.pixel_config_for(state)
    assert config is not None and config.name == "Res40x"
    assert SYSTEM.pixel_size_for(state) == 0.25
    assert SYSTEM.pixel_size_for({("Objective", "Label"): "?"}) == 0.0


def test_system_info_from_core(core: CMMCorePlus) -> None:
    system = SystemInfo.from_core(core)
    assert system.image_width == core.getImageWidth()
    assert system.pixel_size_um == core.getPixelSizeUm()
    assert system.pixel_config == core.getCurrentPixelSizeConfig()
    assert {c.name for c in system.pixel_configs} == set(
        core.getAvailablePixelSizeConfigs()
    )
    assert system.pixel_config_for() is not None
