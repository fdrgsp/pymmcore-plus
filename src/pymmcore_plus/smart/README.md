# `pymmcore_plus.smart`: developer notes

<!-- markdownlint-disable MD013 -->

This is how the smart-microscopy engine is built, and what must stay true
when changing it. **Using** it (the script contract, examples, threads vs
processes, records) is covered in the user guide,
`docs/guides/smart_microscopy.md`. The API reference is
`docs/api/smart.md`.

## What it does

A smart run acquires a *base* `useq.MDASequence` and sends frames to a user
script's `analyze(image, frame, ctx)`. The script returns what to acquire
next. The engine merges those requests into the running acquisition, in
either blocking or asynchronous mode. It runs the script in a thread or a
spawned process, and can record everything to a run folder.

## Modules

| Module | Role | Qt-free | Notes |
|---|---|---|---|
| `_api.py` | The script contract: `FrameInfo`, `AnalysisContext`, `Response`, `STOP`, `SystemInfo`, `PixelConfig`; `normalise_response` | yes | Imported by scripts inside worker processes: keep it cheap. |
| `_loader.py` | `inspect_script`: reads constants and hook signatures with `ast` | yes | **Never executes** the script. |
| `_worker.py` | `_ScriptHost` (imports the script, calls hooks, never raises), `HostConfig`, `HookResult`, process entry points | yes | The only code that runs user code; shared by both executors. |
| `_executors.py` | `ThreadAnalysisExecutor`, `ProcessAnalysisExecutor` | yes | One worker, calls in order. |
| `_scheduler.py` | `SmartEventIterator`: the iterator handed to `core.run_mda` | yes | No core access; unit-testable with a fake runner. |
| `_log.py` | `SmartRunLog`: `run.json`, `frames.jsonl`, `analysis.jsonl`, `script.py` | yes | Opt-in (`run_dir`). |
| `_runner.py` | `SmartRunner`, `SmartRunConfig`, `SmartSignaler`, `dry_run` | yes | Wires everything together. |

Related changes outside this package:

- `CMMCorePlus.run_smart`: a thin wrapper around `SmartRunner.run`.
- `mda/_sink.py`: for stores laid out on one unbounded axis
  (`OmeWritersSink.stores_events`), each frame's event is stored in its
  per-frame metadata (`frame_meta_to_ome(..., include_event=True)`).
- `MDARunner._run`: a cancel that arrives while the iterator is producing
  the next event keeps `FinishReason.CANCELED`.

## Flow of a run

```text
SmartRunner.prepare(base, config, output=, run_dir=)      caller's thread
  ├─ SystemInfo.from_core(core)          snapshot for ctx.system
  ├─ create_executor(mode).start(HostConfig)   blocks until setup() is done
  └─ SmartRunLog(run_dir)                if run_dir is not None
SmartRunner.start()
  └─ core.run_mda(SmartEventIterator(base, ...), output=)   → runner thread

runner thread ── next() ──────────────▶ SmartEventIterator.__next__
runner thread ── frameReady (direct) ─▶ SmartRunner._on_frame_ready
                   ├─ frames.jsonl, frameAcquired
                   └─ filter? → executor.submit(image, FrameInfo)
executor callback thread ─▶ _on_result
                   └─ normalised Response → iterator.inject / stop / drop_base
iterator: base done, nothing queued or pending → after_base() (once)
runner thread ── sequenceFinished (direct) ─▶ _finalize → finalizer thread
                   └─ executor.stop() (teardown), run.json, runFinished
```

## Invariants: easy to break, hard to notice

1. **Pass an iterator, not a sequence.** `MDARunner.run` uses an
   `Iterator` as is, so the engine's `event_iterator` (hardware sequencing)
   is skipped, giving one event per frame and one decision per frame. The
   runner reports an empty `GeneratorMDASequence`, and the sink stores
   frames along one unbounded `t` axis. That's why `frames.jsonl` and the
   per-frame `mda_event` exist.
2. **Runner-side slots must run on the runner thread.** With Qt MDA signals
   (`QMDASignaler`), a plain callable connected from another thread is
   *queued*. In a GUI that ties analysis latency to the GUI event loop, and
   in a script with no event loop it never runs at all.
   `_connect_direct` uses `Qt.DirectConnection` when the signaler is a
   `QObject`. It never imports Qt itself, and checks
   `sys.modules["qtpy.QtCore"]` instead. This applies to `frameReady` and
   to `sequenceFinished`.
3. **Every wait in `SmartEventIterator.__next__` is bounded.** The runner
   thread is blocked inside it and cannot see its own cancel or pause
   flags. Each loop re-checks `runner.status.phase == "finishing"`, and
   waits use `Condition.wait(poll_s)`.
4. **Far-future base events are held back** until they are within
   `lead_time_s` of being due. Once the runner holds an event it sleeps
   until that event's `min_start_time`, so this is the only way a
   `priority="next"` request can overtake it.
5. **Relative timing is rebased at hand-out, per segment.** useq marks each
   time block with `reset_event_timer`. The iterator consumes that flag
   (it never reaches the runner, whose clock the base events rely on) and
   offsets `min_start_time` by the event-clock time at which that
   response's segment started.
6. **Scripts are compiled from the inspected source text** (`exec(compile(...))`),
   not imported with the import system. The bytecode cache keys on
   whole-second mtimes plus size, so a quick same-length edit would
   otherwise run stale code. This also guarantees that `script.py` in the
   run folder is the code that ran.
7. **A script never gets the core.** It acts only through returned events.
   This keeps thread and process modes identical and keeps hardware access
   on the runner thread. What scripts need to know about the hardware goes
   in `SystemInfo`, which must stay picklable.
8. **Grids need a field of view.** useq places tiles 1 µm apart without
   `fov_width`/`fov_height`, and the engine fills these in only for the
   base sequence. `normalise_response` fills them from `SystemInfo`,
   following the `properties` of earlier events in the response (such as
   an objective switch). It refuses the response when the pixel size there
   is not calibrated.
9. **Processes are spawned, never forked**, and are terminated through
   their `multiprocessing` handles from `ProcessPoolExecutor._processes`
   (private, but there is no public API). A worker that hangs while loading
   the script never reports anything, and a surviving worker blocks
   interpreter exit.
10. **This package must stay Qt-free.**
    `tests/smart/test_runner.py::test_smart_package_imports_without_qt`
    enforces it.
11. **Status reporting:** a run's status comes from the runner's own flags
    first (`user_cancelled`, `iterator.stop_reason`), then from
    `core.mda.status.finish_reason`. A cancel from elsewhere can end the
    run at an event boundary without the iterator ever being asked again.

## Tests

`tests/smart/`: `test_api.py` (normalisation, FOV fill and guard,
`SystemInfo`), `test_loader.py`, `test_scheduler.py` (fake runner),
`test_executors.py` (both modes, crash, hang, timeouts), and
`test_runner.py` (end-to-end on the demo configuration, on both signal
backends, with all examples).

## Front ends

pymmcore-gui's *Smart Microscopy* tab is a front end. It wraps
`SmartRunner` in a `QObject` that re-emits `SmartSignaler` as Qt signals,
calls `prepare` off the GUI thread (a process worker takes seconds to
start), and passes `run_dir="auto"`. Anything GUI-specific belongs there,
not here.
