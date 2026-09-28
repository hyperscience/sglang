from __future__ import annotations

import functools

import torch

from sglang.srt.utils.custom_op import register_custom_op

# HS (AML-4240 / AML-4268): bypass the JIT store_cache kernel. It compiles with
# nvcc at runtime and our GPU runtime images ship no compiler. Use the
# precompiled sgl_kernel op when it exists, else let the caller fall back to
# plain tensor indexing (memory_pool._set_kv_buffer_impl).


@functools.cache
def _has_store_kv_cache() -> bool:
    try:
        import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)
    except ImportError:
        return False
    return hasattr(torch.ops, "sgl_kernel") and hasattr(
        torch.ops.sgl_kernel, "store_kv_cache"
    )


def can_use_store_cache(k_row_bytes: int, v_row_bytes: int = 0) -> bool:
    """Whether the precompiled store_kv_cache op can serve these row widths.
    v_row_bytes=0 means symmetric, i.e. it defaults to k_row_bytes. The
    precompiled op takes a single row width, so asymmetric KV is not served."""
    v_row_bytes = v_row_bytes or k_row_bytes
    return (
        k_row_bytes == v_row_bytes
        and k_row_bytes % 4 == 0
        and _has_store_kv_cache()
    )


@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    v_row_bytes: int = 0,
    num_split: int = 0,  # can be tuned for performance
    size_limit: int = 0,
    reserved_skip_index: int = 0,
) -> None:
    """Store key and value tensors into KV cache at specified indices.

    Args:
        k (torch.Tensor): Key tensor of shape (batch_size, H * D).
        v (torch.Tensor): Value tensor of shape (batch_size, H * Dv).
        k_cache (torch.Tensor): Key cache tensor of shape (num_pages, H * D).
        v_cache (torch.Tensor): Value cache tensor of shape (num_pages, H * Dv).
        indices (torch.Tensor): Indices tensor of shape (batch_size,).
        row_bytes (int): Key row width in bytes. Inferred from k when 0.
        v_row_bytes (int): Value row width in bytes; differs from row_bytes for
            asymmetric KV (head_dim != v_head_dim). Inferred from v when 0.
        size_limit (int): Valid slot bound (cache row count = real slots + the
            reserved padding slot); an index outside [0, size_limit) fails fast
            (device assert) instead of an illegal memory access. Defaults to the
            cache row count when 0.
        reserved_skip_index (int): If nonnegative, writes targeting this index
            are skipped. Defaults to the reserved CUDA-graph padding slot 0;
            pass -1 to disable skipping.
    """
    # size_limit / reserved_skip_index / num_split are JIT-kernel tuning knobs;
    # the precompiled op writes every index (slot 0 is the padding slot).
    torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, indices, k, v)
