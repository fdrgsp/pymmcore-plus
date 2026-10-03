"""Run a script's hooks on a background thread or in a separate process.

Both executors drive the same `_ScriptHost`, one call at a time and in order,
so a script behaves identically in either mode. They differ only in isolation:

- **thread**: no startup cost and no copying, but the script shares the host
  interpreter -- pure-Python analysis competes for the GIL, and a crash in a C
  extension takes the whole process down. A hung analysis cannot be
  interrupted.
- **process**: a spawned interpreter (never forked: forking a process that
  runs other threads -- the acquisition's, a GUI's -- is unsafe). Startup
  costs seconds and every frame is pickled across, but a crash or hang is
  contained and the process can be terminated.
"""

from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

from pymmcore_plus.smart._worker import (
    HookResult,
    _proc_after_base,
    _proc_analyze,
    _proc_init,
    _proc_setup,
    _proc_teardown,
    _ScriptHost,
    read_only_view,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    import numpy as np

    from pymmcore_plus.smart._api import ExecutionMode, FrameInfo
    from pymmcore_plus.smart._worker import HostConfig


class ExecutorStartError(RuntimeError):
    """The analysis worker could not be started (timeout or crash)."""


class AnalysisExecutor(ABC):
    """Runs one script's hooks for the duration of one run."""

    mode: ExecutionMode

    @abstractmethod
    def start(self, config: HostConfig, *, timeout: float = 60.0) -> HookResult:
        """Load the script and call its ``setup``; return that call's result.

        Raises `ExecutorStartError` if the worker does not come up in time.
        A script error (import or ``setup`` raising) is *not* raised: it is
        reported in the returned result, with ``ok=False``.
        """

    @abstractmethod
    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[HookResult]:
        """Queue ``analyze(image, frame, ctx)``; results arrive in submission order."""

    @abstractmethod
    def submit_after_base(self) -> Future[HookResult]:
        """Queue ``after_base(ctx)``."""

    @abstractmethod
    def stop(self, timeout: float = 10.0) -> HookResult | None:
        """Call ``teardown`` (best effort) and release the worker. Idempotent.

        Returns the teardown result, or None if it could not run (already
        stopped, worker broken, or timed out).
        """

    @property
    @abstractmethod
    def broken(self) -> bool:
        """Whether the worker died; nothing more can be submitted."""


class ThreadAnalysisExecutor(AnalysisExecutor):
    mode: ExecutionMode = "thread"

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="smart-analysis"
        )
        self._host: _ScriptHost | None = None
        self._stopped = False

    def start(self, config: HostConfig, *, timeout: float = 60.0) -> HookResult:
        def _load_and_setup() -> HookResult:
            # On the worker thread: the script's import-time side effects and
            # its setup() must not run on (or block) the calling thread.
            self._host = _ScriptHost(config, "thread")
            return self._host.setup()

        try:
            return self._pool.submit(_load_and_setup).result(timeout)
        except FutureTimeoutError as e:
            raise ExecutorStartError(
                f"The analysis script did not finish loading within {timeout:g} s."
            ) from e

    def _require_host(self) -> _ScriptHost:
        if self._host is None:
            raise RuntimeError("Executor not started.")
        return self._host

    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[HookResult]:
        host = self._require_host()
        return self._pool.submit(host.analyze, read_only_view(image), frame)

    def submit_after_base(self) -> Future[HookResult]:
        return self._pool.submit(self._require_host().after_base)

    def stop(self, timeout: float = 10.0) -> HookResult | None:
        if self._stopped:
            return None
        self._stopped = True
        result: HookResult | None = None
        if (host := self._host) is not None:

            def _teardown() -> HookResult:
                try:
                    return host.teardown()
                finally:
                    host.close()

            with suppress(FutureTimeoutError, CancelledError, RuntimeError):
                result = self._pool.submit(_teardown).result(timeout)
        # A hung analyze() cannot be interrupted: its thread lingers until the
        # call returns, but nothing else will ever be scheduled on it.
        self._pool.shutdown(wait=False, cancel_futures=True)
        return result

    @property
    def broken(self) -> bool:
        return False


class ProcessAnalysisExecutor(AnalysisExecutor):
    mode: ExecutionMode = "process"

    def __init__(self) -> None:
        self._pool: ProcessPoolExecutor | None = None
        self._workers: list[BaseProcess] = []
        self._broken = False
        self._stopped = False

    def start(self, config: HostConfig, *, timeout: float = 60.0) -> HookResult:
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_proc_init,
            initargs=(config,),
        )
        try:
            future = self._pool.submit(_proc_setup)
            # Taken from the pool rather than asked of the child: a child that
            # hangs while importing the script could never answer.
            self._workers = self._worker_processes()
            return future.result(timeout)
        except FutureTimeoutError as e:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._terminate_workers()
            raise ExecutorStartError(
                f"The analysis process did not start within {timeout:g} s."
            ) from e
        except BrokenProcessPool as e:
            self._broken = True
            raise ExecutorStartError(
                "The analysis process exited while loading the script."
            ) from e

    @property
    def worker_pids(self) -> list[int]:
        return [p.pid for p in self._workers if p.pid is not None]

    def _worker_processes(self) -> list[BaseProcess]:
        # ProcessPoolExecutor starts its worker when the first call is
        # submitted and exposes it only privately; there is no public API.
        processes = getattr(self._pool, "_processes", None) or {}
        return list(processes.values())

    def _require_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            raise RuntimeError("Executor not started.")
        return self._pool

    def submit(self, image: np.ndarray, frame: FrameInfo) -> Future[HookResult]:
        # The parent sequence is irrelevant to the script and would be pickled
        # along with every single frame.
        frame = replace(frame, event=frame.event.model_copy(update={"sequence": None}))
        future = self._require_pool().submit(_proc_analyze, image, frame)
        future.add_done_callback(self._note_broken)
        return future

    def submit_after_base(self) -> Future[HookResult]:
        future = self._require_pool().submit(_proc_after_base)
        future.add_done_callback(self._note_broken)
        return future

    def _note_broken(self, future: Future[HookResult]) -> None:
        if not future.cancelled() and isinstance(future.exception(), BrokenProcessPool):
            self._broken = True

    def stop(self, timeout: float = 10.0) -> HookResult | None:
        if self._stopped or self._pool is None:
            return None
        self._stopped = True
        result: HookResult | None = None
        if not self._broken:
            with suppress(
                FutureTimeoutError, BrokenProcessPool, CancelledError, RuntimeError
            ):
                result = self._pool.submit(_proc_teardown).result(timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)
        # An idle worker exits on its own once the pool is shut down; a hung
        # one (stuck in analyze or teardown) is terminated.
        self._terminate_workers(grace=0 if result is None else 2.0)
        return result

    def _terminate_workers(self, grace: float = 0) -> None:
        """Wait up to *grace* s for each worker to exit, then terminate it.

        A worker left running would also block interpreter exit, which joins
        the pool's management thread, which waits for the worker.
        """
        for process in self._workers:
            process.join(grace)
            if process.is_alive():
                process.terminate()
                process.join(2)
            if process.is_alive():  # pragma: no cover - ignored SIGTERM
                process.kill()
                process.join(2)

    @property
    def broken(self) -> bool:
        return self._broken


def create_executor(mode: ExecutionMode) -> AnalysisExecutor:
    """Return a fresh, unstarted executor for *mode*."""
    if mode == "thread":
        return ThreadAnalysisExecutor()
    if mode == "process":
        return ProcessAnalysisExecutor()
    raise ValueError(f"Unknown execution mode {mode!r}")
