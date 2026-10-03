from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import useq

from pymmcore_plus.smart import STOP, AnalysisContext, PixelConfig, Response, SystemInfo
from pymmcore_plus.smart._api import normalise_response

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
    assert normalise_response(None).response == Response(events=())


def test_single_event_and_iterables_become_tuples() -> None:
    event = useq.MDAEvent(exposure=5)
    assert normalise_response(event).response.events == (event,)
    assert normalise_response([event, event]).response.events == (event, event)
    assert normalise_response(e for e in [event]).response.events == (event,)


def test_sequences_are_expanded_in_order() -> None:
    seq = useq.MDASequence(z_plan=useq.ZRangeAround(range=2, step=1))
    first, last = useq.MDAEvent(exposure=1), useq.MDAEvent(exposure=2)
    events = normalise_response([first, seq, last]).response.events
    assert isinstance(events, tuple)
    assert len(events) == 5
    assert events[0] is first and events[-1] is last


def test_response_options_are_kept() -> None:
    event = useq.MDAEvent()
    out = normalise_response(
        Response(events=[event], priority="end", timing="absolute", drop_base=True)
    ).response
    assert out.events == (event,)
    assert (out.priority, out.timing, out.drop_base) == ("end", "absolute", True)
    assert normalise_response(STOP).response.stop


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


# ---------------------------------------------------------------- grids / FOV


def test_grid_fov_filled_from_start_state() -> None:
    events = normalise_response(_subgrid(), system=SYSTEM).response.events
    # 512 px x 1.0 um/px -> tiles 512 um apart, centered on x=1000
    assert [e.x_pos for e in events] == [744.0, 1256.0]


def test_grid_fov_follows_objective_switch_in_response() -> None:
    out = normalise_response(
        [_switch("40X"), _subgrid(), _switch("10X")], system=SYSTEM
    )
    xs = [e.x_pos for e in out.response.events]
    # 512 px x 0.25 um/px -> tiles 128 um apart
    assert xs == [None, 936.0, 1064.0, None]
    assert out.warnings == []


def test_explicit_grid_fov_is_kept() -> None:
    events = normalise_response(
        _subgrid(fov_width=100, fov_height=100), system=SYSTEM
    ).response.events
    assert [e.x_pos for e in events] == [950.0, 1050.0]


@pytest.mark.parametrize("label", ["Uncal", "NotAConfig"])
def test_grid_refused_without_calibrated_pixel_size(label: str) -> None:
    with pytest.raises(ValueError, match="not calibrated"):
        normalise_response([_switch(label), _subgrid()], system=SYSTEM)


def test_grid_refused_without_system_information() -> None:
    with pytest.raises(ValueError, match="not calibrated"):
        normalise_response(_subgrid())


def test_switch_to_uncalibrated_state_only_warns() -> None:
    out = normalise_response([_switch("Uncal"), useq.MDAEvent()], system=SYSTEM)
    assert len(out.response.events) == 2
    assert len(out.warnings) == 1
    assert "no calibrated pixel size" in out.warnings[0]


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
