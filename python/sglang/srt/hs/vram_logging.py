"""Centralized VRAM logging helpers (Hyperscience).

Designed so that every call site is a one-liner, to minimize merge conflicts
when cherry-picking across upstream rebases.

All logs are gated by the env var ``SGLANG_LOG_VRAM_PEAK`` (default True), and
emitted via the standard ``logging`` module under the ``sglang.hs.vram`` name.

Line layout
-----------
Every line is grouped into labeled sections separated by `` | ``::

    VRAM[<tag>] <kv...> | <event> | now.alloc=A now.reserved=R | gpu.used=U gpu.free=F gpu.total=T (MiB)

- ``<event>`` (optional): what this block did — e.g. ``peak.alloc=425 (Δ+425)``.
- ``now.*``  : THIS process's torch caching-allocator pool. Process-local and
               **excludes** the ~300-700 MiB driver CUDA context (invisible to
               ``torch.cuda.memory_*``). ``now.reserved`` is sticky — torch
               caches freed blocks, so it only ever grows within a process.
- ``gpu.*``  : physical board totals from ``mem_get_info`` — counts *all*
               processes and *all* contexts (incl. the driver context), so it
               reconciles with ``nvidia-smi``.

Because ``now.reserved`` excludes the driver context, ``gpu.used`` is always
larger than the sum of every process's ``now.reserved``; the difference is the
per-process CUDA contexts plus driver/ECC reserve.

Public API
----------
Handle (low-diff) API — preferred at call sites because the bodies stay
un-indented and the patches are pure insertions:

- ``log_startup(tag, **kv)``          — emit the very first baseline (call as
                                         early as possible in the process)
- ``log_baseline(tag, **kv)``         — point-in-time steady-state snapshot
                                         (adds free/total)
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
import os
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


def _pool() -> tuple[int, int]:
    """This process's torch caching-allocator pool (MiB): (alloc, reserved)."""
    return (
        int(torch.cuda.memory_allocated() / _MiB),
        int(torch.cuda.memory_reserved() / _MiB),
    )


def _gpu() -> tuple[int, int, int]:
    """Physical board totals (MiB): (used, free, total) from mem_get_info."""
    free, total = torch.cuda.mem_get_info()
    free_mib = int(free / _MiB)
    total_mib = int(total / _MiB)
    return (total_mib - free_mib, free_mib, total_mib)


def _state_suffix() -> str:
    """Absolute state appended to every line.

    ``now.*`` = this process's torch pool (excludes the driver CUDA context);
    ``gpu.*`` = physical board via mem_get_info (all processes, matches
    nvidia-smi).
    """
    a, r = _pool()
    used, free, total = _gpu()
    return (
        f"| now.alloc={a} now.reserved={r} "
        f"| gpu.used={used} gpu.free={free} gpu.total={total} (MiB)"
    )


def _emit(tag: str, kv: dict[str, Any], measured: str) -> None:
    extra = _format_kv(kv)
    # PID disambiguates the process-local now.* pool: the tokenizer-manager
    # (main) and scheduler (subprocess) both emit VRAM[...] to the same stream.
    # Not cached — the module may be imported before a fork.
    parts = [f"VRAM[{tag}]", f"pid={os.getpid()}"]
    if extra:
        parts.append(extra)
    if measured:
        parts.append(f"| {measured}")
    parts.append(_state_suffix())
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
    """Steady-state snapshot at a major milestone (post-init, after weights).

    The physical ``gpu.*`` and torch-pool ``now.*`` groups are already part of
    every line via the shared suffix, so this is a plain checkpoint that chains
    with subsequent logs.
    """
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        _emit(tag, kv, "")
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_startup(tag: str = "startup", **kv: Any) -> None:
    """Emit the very first VRAM log of the process, preceded by a one-time
    legend explaining the field groups. Distinct tag so it stands out at the
    top of the log.
    """
    if not enabled():
        return
    logger.info(
        "VRAM[legend] pid=<owner of the now.* pool> | now.*=torch "
        "caching-allocator pool for THAT process (excludes ~300-700MiB driver "
        "context; now.reserved is sticky & only grows) | gpu.*=physical board "
        "via mem_get_info (all processes + contexts, matches nvidia-smi) | "
        "Δ+=increase during the block"
    )
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
    }


def finish_peak_tracker(
    handle: dict[str, Any] | None, tag: str, **kv: Any
) -> None:
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        peak_alloc = int(torch.cuda.max_memory_allocated() / _MiB)
        peak_reserved = int(torch.cuda.max_memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        measured = (
            f"peak.alloc={peak_alloc} (Δ+{max(0, peak_alloc - a0)}) "
            f"peak.reserved={peak_reserved} (Δ+{max(0, peak_reserved - r0)})"
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
        peak_alloc = int(torch.cuda.max_memory_allocated() / _MiB)
        peak_reserved = int(torch.cuda.max_memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        peak0 = int(handle["peak_before"] / _MiB)
        measured = (
            f"peak.alloc={peak_alloc} (Δ+{max(0, peak_alloc - a0)}) "
            f"peak.contrib=+{max(0, peak_alloc - peak0)} "
            f"peak.reserved={peak_reserved} (Δ+{max(0, peak_reserved - r0)})"
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
    }


def finish_alloc_delta(
    handle: dict[str, Any] | None, tag: str, **kv: Any
) -> None:
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        alloc_after = int(torch.cuda.memory_allocated() / _MiB)
        reserved_after = int(torch.cuda.memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        measured = (
            f"delta.alloc={alloc_after - a0:+d} "
            f"delta.reserved={reserved_after - r0:+d}"
        )
        # _emit appends the now.*/gpu.* suffix; now.alloc == alloc_after.
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
