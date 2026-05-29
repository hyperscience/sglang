"""JIT kvcache module — patched to bypass JIT compilation."""

from __future__ import annotations

import torch

from sglang.srt.utils.custom_op import register_custom_op

# Check if the pre-compiled C++ store_kv_cache op exists in sgl_kernel
_HAS_STORE_KV_CACHE = hasattr(torch.ops, "sgl_kernel") and hasattr(
    torch.ops.sgl_kernel, "store_kv_cache"
)


def can_use_store_cache(size: int) -> bool:
    """Return True if size is valid and pre-compiled kernel is available."""
    return size % 4 == 0 and _HAS_STORE_KV_CACHE


@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,
) -> None:
    """Store KV cache using pre-compiled sgl_kernel op or naive fallback."""
    if _HAS_STORE_KV_CACHE:
        torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, indices, k, v)
    else:
        k_cache[indices] = k
        v_cache[indices] = v
