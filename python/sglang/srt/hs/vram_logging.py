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
- ``log_baseline(tag, **kv)``         — point-in-time steady-state snapshot;
                                         also marks the reserved watermark that
                                         ``log_budget`` uses to split init vs
                                         runtime
- ``log_budget(tag, only_on_growth)`` — decompose reserved.total into named
                                         contributors (the tuning artifact);
                                         self-throttles on growth so it is safe
                                         to call every hot-loop iteration
- ``log_processes(tag, **kv)``        — NVML per-process board decomposition
                                         (ctx.self + procs=[host_pid:used])
- ``log_static(tag, **kv)``           — checkpoint line for config values
- ``start_peak_tracker()`` / ``finish_peak_tracker(h, tag, only_on_growth=..)``
                                       — resets peak; for top-level forward.
                                         ``only_on_growth`` mutes flat repeats
                                         (still attributes to the budget).
- ``start_snapshot_peak()`` / ``finish_snapshot_peak(h, tag, **kv)``
                                       — no peak reset; for nested blocks
- ``start_alloc_delta()`` / ``finish_alloc_delta(h, tag, net_of_nested=..)``
                                       — persistent buffer allocations
                                         (weights/kv-pool/mamba-pool/lora-pool);
                                         ``net_of_nested`` credits only the
                                         residual when a block wraps other
                                         tracked allocations (kv-pool ⊃ mamba)

Context-manager variants are also provided for fresh code:
``peak_tracker(...)``, ``snapshot_peak(...)``, ``alloc_delta(...)``.

Each tag becomes ``VRAM[<tag>]`` in the log line.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from sglang.srt.environ import envs

logger = logging.getLogger("sglang.hs.vram")

logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

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


# ---------------------------------------------------------------------------
# NVML per-process CUDA-context accounting
# ---------------------------------------------------------------------------
# torch.cuda.memory_* can't see the driver CUDA context (~300-700 MiB of
# kernels/cublas/driver modules). NVML reports each PID's *total* committed
# GPU memory, so:  context = nvml_used[pid] - torch.memory_reserved().
_nvml_handle: Any = None
_nvml_failed = False


def _nvml_dev_handle() -> Any:
    global _nvml_handle, _nvml_failed
    if _nvml_failed:
        return None
    if _nvml_handle is None:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                torch.cuda.current_device()
            )
        except Exception:
            _nvml_failed = True
            return None
    return _nvml_handle


def _host_pid() -> int:
    """PID as the driver/NVML sees it (outermost namespace).

    Inside a PID namespace (containers) NVML reports host PIDs while
    ``os.getpid()`` returns the namespaced PID; ``/proc/self/status`` NSpid
    lists the host PID first.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return os.getpid()


def _nvml_proc_table() -> dict[int, int] | None:
    """{host_pid: used_MiB} for every compute process on this GPU, or None."""
    h = _nvml_dev_handle()
    if h is None:
        return None
    try:
        import pynvml  # noqa: PLC0415

        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    except Exception:
        return None
    table: dict[int, int] = {}
    for p in procs:
        used = getattr(p, "usedGpuMemory", None)
        if used is not None:
            table[int(p.pid)] = int(used / _MiB)
    return table


def _context_group() -> str:
    """Best-effort NVML context accounting: ``ctx.self`` + ``procs`` list.

    ``ctx.self`` = this process's NVML used memory minus its torch reserved
    pool = the CUDA context (+ any non-torch cudaMalloc). ``procs`` lists every
    compute PID's used MiB so both processes' totals are visible on one line.
    Returns "" when NVML is unavailable or self-attribution fails.
    """
    table = _nvml_proc_table()
    if not table:
        return ""
    reserved = int(torch.cuda.memory_reserved() / _MiB)
    parts = []
    self_used = table.get(_host_pid())
    if self_used is not None:
        parts.append(f"ctx.self={max(0, self_used - reserved)}")
    procs = ",".join(f"{pid}:{used}" for pid, used in sorted(table.items()))
    parts.append(f"procs=[{procs}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Reserved-memory attribution ledger (per process)
# ---------------------------------------------------------------------------
# End goal: decompose each process's ``reserved.total`` into named contributors
# so KV-cache size and batch size can be tuned against the OOM ceiling.
#
# torch ``reserved`` is sticky (only grows), so we attribute its growth:
#   - persistent one-shot allocations (weights/kv-pool/lora) -> exact ledger
#   - transient forward phases (prefill/decode/heatmap/vlm-embed) share the
#     caching pool and overlap, so we can't split the sticky lump cleanly; we
#     record each phase's worst single-block reserved bump as *indicative*.
_init_ledger: "OrderedDict[str, int]" = OrderedDict()  # exact persistent MiB
_phase_hi: dict[str, int] = {}  # indicative per-phase hi-water MiB
_baseline_reserved: int | None = None  # reserved.total at post-init baseline
_last_budget_reserved: int = -1  # throttle for log_budget(only_on_growth=True)


def _ledger_add(tag: str, delta_mib: int) -> None:
    """Attribute an exact, persistent reserved delta to ``tag`` (accumulates)."""
    _init_ledger[tag] = _init_ledger.get(tag, 0) + int(delta_mib)


def _phase_record(tag: str, delta_mib: int) -> None:
    """Record ``tag``'s worst single-block reserved bump (indicative hi-water)."""
    if delta_mib > _phase_hi.get(tag, 0):
        _phase_hi[tag] = int(delta_mib)


def _state_suffix(with_context: bool = False) -> str:
    """Absolute state appended to every line.

    ``now.*`` = this process's torch pool (excludes the driver CUDA context);
    ``ctx.*``/``procs`` = NVML per-process totals (only when
    ``with_context``); ``gpu.*`` = physical board via mem_get_info (all
    processes, matches nvidia-smi).
    """
    a, r = _pool()
    used, free, total = _gpu()
    ctx = f"| {_context_group()} " if with_context else ""
    ctx = ctx if ctx.strip() != "|" else ""
    return (
        f"| now.alloc={a} now.reserved={r} "
        f"{ctx}"
        f"| gpu.used={used} gpu.free={free} gpu.total={total} (MiB)"
    )


def _emit(
    tag: str, kv: dict[str, Any], measured: str, with_context: bool = False
) -> None:
    extra = _format_kv(kv)
    # PID disambiguates the process-local now.* pool: the tokenizer-manager
    # (main) and scheduler (subprocess) both emit VRAM[...] to the same stream.
    # Not cached — the module may be imported before a fork.
    parts = [f"VRAM[{tag}]", f"pid={os.getpid()}"]
    if extra:
        parts.append(extra)
    if measured:
        parts.append(f"| {measured}")
    parts.append(_state_suffix(with_context=with_context))
    logger.info(" ".join(parts))


# ---------------------------------------------------------------------------
# Point-in-time logs
# ---------------------------------------------------------------------------


def log_static(tag: str, **kv: Any) -> None:
    """Emit a ``VRAM[<tag>] <kv...>`` checkpoint (with NVML context group)."""
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        _emit(tag, kv, "", with_context=True)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_baseline(tag: str = "baseline", **kv: Any) -> None:
    """Steady-state snapshot at a major milestone (post-init, after weights).

    The physical ``gpu.*``, torch-pool ``now.*`` and NVML ``ctx.*``/``procs``
    groups are all part of the line, so this is a plain checkpoint that chains
    with subsequent logs. Also records the reserved watermark used by
    ``log_budget`` to split init (weights/kv/graph) from runtime growth.
    """
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        global _baseline_reserved
        _baseline_reserved = int(torch.cuda.memory_reserved() / _MiB)
        _emit(tag, kv, "", with_context=True)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_processes(tag: str = "gpu-procs", **kv: Any) -> None:
    """Snapshot the full NVML compute-process table + this process's context.

    Cheap cross-process board decomposition: emits ``ctx.self`` (this PID's
    CUDA context) and ``procs=[host_pid:used,...]`` for every process on the
    GPU. Call at will (e.g. around a peak) to see both contexts at once.
    """
    if not enabled():
        return
    try:
        torch.cuda.synchronize()
        _emit(tag, kv, "", with_context=True)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def log_budget(tag: str = "budget", only_on_growth: bool = False, **kv: Any) -> None:
    """Decompose this process's ``reserved.total`` into named contributors.

    This is the tuning artifact: it answers "where did the reserved MiB go?" so
    KV-cache size and batch size can be set against the OOM ceiling.

    Layout::

        reserved.total=R = weights=.. + kv-pool=.. + lora-pool=.. \
                           + init-other=.. + runtime=.. \
                         | runtime-hi[prefill≤.. heatmap≤.. ..] \
                         | ctx.self=.. procs=[..] | gpu.used/free/total

    - ``weights``/``kv-pool``/``lora-pool``: exact persistent allocations.
    - ``init-other``: residual up to the post-init baseline (CUDA-graph capture,
      fragmentation, misc) — only shown once ``log_baseline`` has run.
    - ``runtime``: sticky reserved grown *after* baseline = the combined
      transient working set (prefill/decode/heatmap/vlm-embed share the pool and
      overlap, so this lump can't be split cleanly).
    - ``runtime-hi[...]``: *indicative* per-phase worst single-block bump — these
      overlap, so they do NOT sum to ``runtime``; they rank the phases.
    - ``ctx.self`` (NVML): the driver CUDA context, on top of ``reserved.total``.

    ``only_on_growth=True`` self-throttles: it returns immediately (no sync/NVML)
    unless ``reserved`` climbed since the last budget line — so it can be called
    every iteration in the hot loop and only prints when the ceiling moves.
    """
    if not enabled():
        return
    global _last_budget_reserved
    reserved = int(torch.cuda.memory_reserved() / _MiB)
    if only_on_growth and reserved <= _last_budget_reserved:
        return
    _last_budget_reserved = reserved
    try:
        torch.cuda.synchronize()
        reserved = int(torch.cuda.memory_reserved() / _MiB)
        terms: list[str] = [f"{k}={v}" for k, v in _init_ledger.items()]
        runtime_hi = ""
        if _baseline_reserved is not None:
            init_other = _baseline_reserved - sum(_init_ledger.values())
            if abs(init_other) >= 1:
                terms.append(f"init-other={init_other}")
            terms.append(f"runtime={reserved - _baseline_reserved}")
            if _phase_hi:
                ranked = sorted(_phase_hi.items(), key=lambda x: -x[1])
                runtime_hi = " | runtime-hi[" + " ".join(
                    f"{k}≤{v}" for k, v in ranked
                ) + "]"
        else:
            # No baseline (e.g. tokenizer process): decompose by phase directly.
            for k, v in sorted(_phase_hi.items(), key=lambda x: -x[1]):
                terms.append(f"{k}={v}")
            other = reserved - sum(_phase_hi.values())
            if abs(other) >= 1:
                terms.append(f"other={other}")
        decomp = " + ".join(terms) if terms else "(nothing attributed yet)"
        ctx = _context_group()
        ctx_str = f" | {ctx}" if ctx else ""
        used, free, total = _gpu()
        head = " ".join(
            p for p in [f"VRAM[{tag}]", f"pid={os.getpid()}", _format_kv(kv)] if p
        )
        logger.info(
            f"{head} | reserved.total={reserved} = {decomp}"
            f"{runtime_hi}{ctx_str} "
            f"| gpu.used={used} gpu.free={free} gpu.total={total} (MiB)"
        )
    except Exception as e:
        logger.warning(f"VRAM[{tag}] budget failed: {e}")


def log_startup(tag: str = "startup", **kv: Any) -> None:
    """Emit the very first VRAM log of the process, preceded by a one-time
    legend explaining the field groups. Distinct tag so it stands out at the
    top of the log.
    """
    if not enabled():
        return
    logger.info(
        "VRAM[legend] pid=<owner of the now.* pool> | now.*=torch "
        "caching-allocator pool for THAT process (excludes the driver context; "
        "now.reserved is sticky & only grows) | ctx.self=this PID's CUDA "
        "context (NVML used − torch reserved) | procs=[host_pid:used] per "
        "process | gpu.*=physical board via mem_get_info (all processes + "
        "contexts, matches nvidia-smi) | Δ+=increase during the block | "
        "VRAM[budget]=reserved.total decomposed into contributors; runtime-hi[] "
        "is indicative per-phase hi-water (overlaps, does NOT sum)"
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
    handle: dict[str, Any] | None,
    tag: str,
    only_on_growth: bool = False,
    **kv: Any,
) -> None:
    """Emit a per-block peak line and attribute reserved growth to ``tag``.

    ``only_on_growth=True`` suppresses the log line when this block didn't push
    ``reserved`` any higher (the hot loop's flat repeats) — the phase hi-water
    is still recorded either way, so the budget stays complete.
    """
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        peak_alloc = int(torch.cuda.max_memory_allocated() / _MiB)
        peak_reserved = int(torch.cuda.max_memory_reserved() / _MiB)
        now_reserved = int(torch.cuda.memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        _phase_record(tag, now_reserved - r0)
        if only_on_growth and now_reserved <= r0:
            return
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
    handle: dict[str, Any] | None,
    tag: str,
    only_on_growth: bool = False,
    **kv: Any,
) -> None:
    """Reports ``peak-contrib`` — how much this block pushed the peak forward
    (won't disturb an enclosing ``start_peak_tracker``). Attributes reserved
    growth to ``tag``; ``only_on_growth`` suppresses flat repeats (see
    ``finish_peak_tracker``).
    """
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        peak_alloc = int(torch.cuda.max_memory_allocated() / _MiB)
        peak_reserved = int(torch.cuda.max_memory_reserved() / _MiB)
        now_reserved = int(torch.cuda.memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        peak0 = int(handle["peak_before"] / _MiB)
        _phase_record(tag, now_reserved - r0)
        if only_on_growth and now_reserved <= r0:
            return
        measured = (
            f"peak.alloc={peak_alloc} (Δ+{max(0, peak_alloc - a0)}) "
            f"peak.contrib=+{max(0, peak_alloc - peak0)} "
            f"peak.reserved={peak_reserved} (Δ+{max(0, peak_reserved - r0)})"
        )
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


def start_alloc_delta() -> dict[str, Any] | None:
    """Snapshot before a persistent allocation block (no peak tracking).

    Also snapshots the running ledger total so an outer block can record its
    contribution *net of* any nested ``finish_alloc_delta`` calls (see
    ``net_of_nested``), avoiding double-counting when e.g. ``kv-pool`` wraps a
    nested ``mamba-pool``.
    """
    if not enabled():
        return None
    torch.cuda.synchronize()
    return {
        "alloc_before": torch.cuda.memory_allocated(),
        "reserved_before": torch.cuda.memory_reserved(),
        "ledger_before": sum(_init_ledger.values()),
    }


def finish_alloc_delta(
    handle: dict[str, Any] | None,
    tag: str,
    net_of_nested: bool = False,
    **kv: Any,
) -> None:
    """Attribute a persistent reserved delta to ``tag`` in the budget ledger.

    ``net_of_nested=True``: this block wraps other tracked allocations that
    already recorded their own ledger entries (e.g. ``kv-pool`` wraps
    ``mamba-pool``). The ledger gets ``delta − nested`` so the sub-components
    and this residual sum to the true total without overlap. The logged
    ``delta.*`` still shows the full measured delta; ``ledger`` annotates the
    net figure when they differ.
    """
    if handle is None or not enabled():
        return
    try:
        torch.cuda.synchronize()
        alloc_after = int(torch.cuda.memory_allocated() / _MiB)
        reserved_after = int(torch.cuda.memory_reserved() / _MiB)
        a0 = int(handle["alloc_before"] / _MiB)
        r0 = int(handle["reserved_before"] / _MiB)
        delta_reserved = reserved_after - r0
        nested = 0
        if net_of_nested:
            nested = sum(_init_ledger.values()) - handle.get("ledger_before", 0)
        ledger_val = delta_reserved - nested
        _ledger_add(tag, ledger_val)
        measured = (
            f"delta.alloc={alloc_after - a0:+d} "
            f"delta.reserved={delta_reserved:+d}"
        )
        if nested:
            measured += f" ledger={ledger_val:+d} (net of nested {nested})"
        # _emit appends the now.*/gpu.* suffix; now.alloc == alloc_after.
        _emit(tag, kv, measured)
    except Exception as e:
        logger.warning(f"VRAM[{tag}] log failed: {e}")


# ---------------------------------------------------------------------------
# Context-manager API (for new code where indenting the body is acceptable)
# ---------------------------------------------------------------------------


@contextmanager
def peak_tracker(
    tag: str, active: bool = True, only_on_growth: bool = False, **kv: Any
) -> Iterator[None]:
    h = start_peak_tracker(active=active)
    try:
        yield
    finally:
        finish_peak_tracker(h, tag, only_on_growth=only_on_growth, **kv)


@contextmanager
def snapshot_peak(
    tag: str, only_on_growth: bool = False, **kv: Any
) -> Iterator[None]:
    h = start_snapshot_peak()
    try:
        yield
    finally:
        finish_snapshot_peak(h, tag, only_on_growth=only_on_growth, **kv)


@contextmanager
def alloc_delta(tag: str, **kv: Any) -> Iterator[None]:
    h = start_alloc_delta()
    try:
        yield
    finally:
        finish_alloc_delta(h, tag, **kv)
