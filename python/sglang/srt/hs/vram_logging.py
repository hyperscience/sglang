"""Centralized VRAM logging helpers (Hyperscience).

Designed so that every call site is a one-liner, to minimize merge conflicts
when cherry-picking across upstream rebases.

All logs are gated by the env var ``SGLANG_LOG_VRAM_PEAK`` (default True), and
emitted via the standard ``logging`` module under the ``sglang.hs.vram`` name.

Every emitted line ends with::

    alloc=<absolute MiB> reserved=<absolute MiB>

so the log chain is self-checking: the "alloc=" of one line must match the
"alloc-before=" of the next interesting event. A mismatch tells you you
forgot to log a step in between.

Public API
----------
Handle (low-diff) API — preferred at call sites because the bodies stay
un-indented and the patches are pure insertions:

- ``log_startup(tag, **kv)``          — emit the very first baseline (call as
                                         early as possible in the process)
- ``log_baseline(tag, **kv)``         — point-in-time steady-state snapshot
                                         (adds free/total/activation_budget)
- ``log_static(tag, **kv)``           — checkpoint line for config values
- ``start_peak_tracker()`` / ``finish_peak_tracker(h, tag, **kv)``
                                       — resets peak; for top-level forward
- ``start_snapshot_peak()`` / ``finish_snapshot_peak(h, tag, **kv)``
                                       — no peak reset; for nested blocks
- ``start_alloc_delta()`` / ``finish_alloc_delta(h, tag, **kv)``
                                       — persistent buffer allocations

Context-manager variants are also provided for fresh code:
``peak_tracker(...)``, ``snapshot_peak(...)``, ``alloc_delta(...)``.

Each tag becomes ``VRAM[<tag>]`` in the log line.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from sglang.srt.environ import envs

logger = logging.getLogger("sglang.hs.vram")

_MiB = 1024 * 1024


def enabled() -> bool:
    """Return True iff VRAM logging is on and CUDA is available."""
    try:
        return envs.SGLANG_LOG_VRAM_PEAK.get() and torch.cuda.is_available()
    except Exception:
        return False


def _format_kv(kv: dict[str, Any]) -> str:
    parts = []
    for k, v in kv.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _absolute_suffix() -> str:
    """Always-on absolute state suffix appended to every log line."""
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"alloc={int(alloc / _MiB)}MiB reserved={int(reserved / _MiB)}MiB"


def _emit(tag: str, kv: dict[str, Any], measured: str) -> None:
    extra = _format_kv(kv)
    parts = [f"VRAM[{tag}]"]
    if extra:
        parts.append(extra)
    if measured:
        parts.append(measured)
    parts.append(_absolute_suffix())
    logger.info(" ".join(parts))


# ---------------------------------------------------------------------------
# Point-in-time logs
# ---------------------------------------------------------------------------


def log_static(tag: str, **kv: Any) -> None:
    """Emit a ``VRAM[<tag>] <kv...> alloc=...MiB reserved=...MiB`` checkpoint."""
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        _emit(tag, kv, "")
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_baseline(tag: str = "baseline", **kv: Any) -> None:
    """Detailed steady-state snapshot with free/total VRAM and activation budget.

    Use at major milestones (post-init, after weights, etc.). Also includes
    the standard absolute suffix so it chains with subsequent logs.
    """
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        free, total = torch.cuda.mem_get_info()
        measured = (
            f"free={int(free / _MiB)}MiB "
            f"total={int(total / _MiB)}MiB "
            f"=> activation_budget≈{int((total - alloc) / _MiB)}MiB"
        )
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_startup(tag: str = "startup", **kv: Any) -> None:
    """Emit the very first VRAM log of the process — same content as
    ``log_baseline`` but distinct tag so it stands out at the top of the log.
    """
    log_baseline(tag, **kv)


# ---------------------------------------------------------------------------
# Handle API (preferred — no body indentation, no return path changes)
# ---------------------------------------------------------------------------


def start_peak_tracker(active: bool = True) -> dict[str, Any] | None:
    """Reset peak counters and snapshot baseline. Use for top-level forward.

    Pass ``active=False`` to make this a no-op (returns ``None``). Useful for
    throttled loops (e.g. the decode hot loop) where the caller only wants to
    emit at fixed intervals — the matching ``finish_peak_tracker`` is also a
    no-op when handed a ``None`` handle.

    Note: when called every iteration, the peak counter is reset every call,
    so the reported peak is per-iteration. When throttled (active=False
    between logged iters), the peak counter accumulates across the skipped
    iters so the next logged iter reports the high-water of the whole
    interval — which is usually what you want for VRAM ceiling tuning.
    """
    if not active or not enabled():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "alloc_before": torch.cuda.memory_allocated(),
        "reserved_before": torch.cuda.memory_reserved(),
        "t0": time.perf_counter(),
    }


def finish_peak_tracker(
    handle: dict[str, Any] | None, tag: str, **kv: Any
) -> None:
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        dt_ms = (time.perf_counter() - handle["t0"]) * 1000.0
        measured = (
            f"alloc-before={int(handle['alloc_before'] / _MiB)}MiB "
            f"reserved-before={int(handle['reserved_before'] / _MiB)}MiB "
            f"peak-alloc={int(peak_alloc / _MiB)}MiB "
            f"peak-alloc-increase={int(max(0, peak_alloc - handle['alloc_before']) / _MiB)}MiB "
            f"peak-reserved={int(peak_reserved / _MiB)}MiB "
            f"peak-reserved-increase={int(max(0, peak_reserved - handle['reserved_before']) / _MiB)}MiB "
            f"time={dt_ms:.1f}ms"
        )
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def start_snapshot_peak() -> dict[str, Any] | None:
    """Snapshot running peak WITHOUT resetting it. Use for nested blocks."""
    if not enabled():
        return None
    torch.cuda.synchronize()
    return {
        "peak_before": torch.cuda.max_memory_allocated(),
        "alloc_before": torch.cuda.memory_allocated(),
        "reserved_before": torch.cuda.memory_reserved(),
        "t0": time.perf_counter(),
    }


def finish_snapshot_peak(
    handle: dict[str, Any] | None, tag: str, **kv: Any
) -> None:
    """Reports ``peak-contrib`` — how much this block pushed the peak forward
    (won't disturb an enclosing ``start_peak_tracker``).
    """
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        dt_ms = (time.perf_counter() - handle["t0"]) * 1000.0
        measured = (
            f"alloc-before={int(handle['alloc_before'] / _MiB)}MiB "
            f"reserved-before={int(handle['reserved_before'] / _MiB)}MiB "
            f"peak-alloc={int(peak_alloc / _MiB)}MiB "
            f"peak-alloc-increase={int(max(0, peak_alloc - handle['alloc_before']) / _MiB)}MiB "
            f"peak-contrib={int(max(0, peak_alloc - handle['peak_before']) / _MiB)}MiB "
            f"peak-reserved={int(peak_reserved / _MiB)}MiB "
            f"peak-reserved-increase={int(max(0, peak_reserved - handle['reserved_before']) / _MiB)}MiB "
            f"time={dt_ms:.1f}ms"
        )
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def start_alloc_delta() -> dict[str, Any] | None:
    """Snapshot before a persistent allocation block (no peak tracking)."""
    if not enabled():
        return None
    torch.cuda.synchronize()
    return {
        "alloc_before": torch.cuda.memory_allocated(),
        "reserved_before": torch.cuda.memory_reserved(),
        "t0": time.perf_counter(),
    }


def finish_alloc_delta(
    handle: dict[str, Any] | None, tag: str, **kv: Any
) -> None:
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        dt_ms = (time.perf_counter() - handle["t0"]) * 1000.0
        measured = (
            f"alloc-before={int(handle['alloc_before'] / _MiB)}MiB "
            f"reserved-before={int(handle['reserved_before'] / _MiB)}MiB "
            f"alloc-delta={int((alloc_after - handle['alloc_before']) / _MiB)}MiB "
            f"reserved-delta={int((reserved_after - handle['reserved_before']) / _MiB)}MiB "
            f"time={dt_ms:.1f}ms"
        )
        # _emit will append the absolute alloc=...MiB reserved=...MiB suffix
        # which equals (alloc_after, reserved_after).
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


# ---------------------------------------------------------------------------
# Context-manager API (for new code where indenting the body is acceptable)
# ---------------------------------------------------------------------------


@contextmanager
def peak_tracker(tag: str, active: bool = True, **kv: Any) -> Iterator[None]:
    h = start_peak_tracker(active=active)
    try:
        yield
    finally:
        finish_peak_tracker(h, tag, **kv)


@contextmanager
def snapshot_peak(tag: str, **kv: Any) -> Iterator[None]:
    h = start_snapshot_peak()
    try:
        yield
    finally:
        finish_snapshot_peak(h, tag, **kv)


@contextmanager
def alloc_delta(tag: str, **kv: Any) -> Iterator[None]:
    h = start_alloc_delta()
    try:
        yield
    finally:
        finish_alloc_delta(h, tag, **kv)
