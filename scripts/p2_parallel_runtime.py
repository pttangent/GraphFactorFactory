#!/usr/bin/env python3
"""Bounded concurrency and child-process lifecycle helpers for GFF."""
from __future__ import annotations

import concurrent.futures as cf
import ctypes
import multiprocessing as mp
import os
import signal
import subprocess
from collections.abc import Iterable, Iterator
from typing import Any, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(0, value)


def resolve_max_tasks_per_child(value: int | None) -> int | None:
    """Resolve worker recycling without conflating ``None`` and one task.

    ``None`` means use ``GFF_MAX_TASKS_PER_CHILD`` (default 8). ``0`` disables
    recycling for the lifetime of the pool. A positive integer recycles after
    exactly that many completed tasks. This avoids Windows spawn/import storms
    while keeping a bounded option for long-running Pandas/Arrow workers.
    """
    resolved = _nonnegative_int_env("GFF_MAX_TASKS_PER_CHILD", 8) if value is None else max(0, int(value))
    return None if resolved == 0 else resolved


def bounded_thread_map(
    items: Iterable[T],
    workers: int,
    function: Callable[[T], R],
    *,
    max_in_flight: int | None = None,
) -> Iterator[R]:
    """Yield completed thread results with bounded retained inputs."""
    workers = max(1, int(workers))
    limit = max(workers, int(max_in_flight or workers * 2))
    iterator = iter(items)
    executor = cf.ThreadPoolExecutor(max_workers=workers)
    pending: set[cf.Future[R]] = set()

    def submit_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(function, item))
        return True

    try:
        for _ in range(limit):
            if not submit_one():
                break
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                submit_one()
                yield result
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)


def bounded_thread_map_ordered(
    items: Iterable[T],
    workers: int,
    function: Callable[[T], R],
    *,
    max_in_flight: int | None = None,
) -> Iterator[R]:
    """Yield deterministic ordered results with a strictly bounded reorder window.

    ``pending + ready`` never exceeds ``max_in_flight``. Later snapshots may
    finish inside that window, but replacement work is submitted only after the
    next ordered result is yielded. This preserves output order and memory bounds.
    """
    workers = max(1, int(workers))
    limit = max(workers, int(max_in_flight or workers * 2))
    iterator = enumerate(items)
    executor = cf.ThreadPoolExecutor(max_workers=workers)
    pending: dict[cf.Future[R], int] = {}
    ready: dict[int, R] = {}
    next_output = 0

    def submit_one() -> bool:
        try:
            index, item = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(function, item)] = index
        return True

    try:
        for _ in range(limit):
            if not submit_one():
                break
        while pending or ready:
            while next_output in ready:
                result = ready.pop(next_output)
                next_output += 1
                submit_one()
                yield result
            if not pending:
                break
            done, _ = cf.wait(set(pending), return_when=cf.FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                ready[index] = future.result()
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)


def bounded_process_map(
    items: Iterable[T],
    workers: int,
    function: Callable[..., R],
    *shared_args: Any,
    max_in_flight: int | None = None,
    max_tasks_per_child: int | None = None,
) -> Iterator[R]:
    """Yield process results with bounded submissions and controlled recycling."""
    workers = max(1, int(workers))
    limit = max(workers, int(max_in_flight or workers * 2))
    recycle_after = resolve_max_tasks_per_child(max_tasks_per_child)
    iterator = iter(items)
    context = mp.get_context("spawn")
    executor = cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        max_tasks_per_child=recycle_after,
    )
    pending: set[cf.Future[R]] = set()

    def submit_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(function, item, *shared_args))
        return True

    try:
        for _ in range(limit):
            if not submit_one():
                break
        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                submit_one()
                yield result
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)


def collect_process_map(
    items: Iterable[T],
    workers: int,
    function: Callable[..., R],
    *shared_args: Any,
    max_in_flight: int | None = None,
    max_tasks_per_child: int | None = None,
) -> list[R]:
    return list(
        bounded_process_map(
            items,
            workers,
            function,
            *shared_args,
            max_in_flight=max_in_flight,
            max_tasks_per_child=max_tasks_per_child,
        )
    )


if os.name == "nt":
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def _attach_windows_kill_job(process: subprocess.Popen[Any]) -> int | None:
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(information), ctypes.sizeof(information)):
        kernel32.CloseHandle(job)
        return None
    if not kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(process._handle)):  # type: ignore[attr-defined]
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _close_windows_job(job: int | None) -> None:
    if os.name == "nt" and job:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(ctypes.c_void_p(job))


def run_process_tree(command: list[str], *, env: dict[str, str] | None = None) -> int:
    """Run a subprocess whose whole descendant tree dies with this call."""
    kwargs: dict[str, Any] = {"env": env}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    job = _attach_windows_kill_job(process)
    try:
        return process.wait()
    except BaseException:
        if os.name == "nt":
            _close_windows_job(job)
            job = None
            if process.poll() is None:
                process.terminate()
        elif process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=15)
        except Exception:
            if process.poll() is None:
                process.kill()
        raise
    finally:
        _close_windows_job(job)
