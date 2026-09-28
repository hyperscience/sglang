# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the attention heatmap (Hyperscience).

See sglang.srt.hs.attention_heatmap. The heatmap query recorder assumes one
query per request per forward step, recorded for the last token of each
request:

- Prefill CUDA graph (breakable by default since v0.5.12) captures the
  recorder's per-request loop over the extend lengths once, at capture time,
  so every replay would record the query of a padding token. Prefill CUDA
  graph is therefore disabled when attention_heatmap_layer_ids is set.
- Speculative decoding verifies several draft tokens per step, discards the
  rejected ones, and runs separate draft-model passes, so the recorded queries
  would not line up with the output tokens. The combination is rejected.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

logger = logging.getLogger(__name__)


def validate_attention_heatmap_args(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    layer_ids = cfg.attention_heatmap_layer_ids
    if layer_ids is not None:
        if not isinstance(layer_ids, list) or not all(
            isinstance(x, int) for x in layer_ids
        ):
            raise ValueError("attention_heatmap_layer_ids must be a list of ints")
        if any(x < 0 for x in layer_ids):
            raise ValueError(
                "attention_heatmap_layer_ids must contain only non-negative ints"
            )
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("attention_heatmap_layer_ids must not contain duplicates")
    chunk_size = cfg.chunked_attention_heatmap_size
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunked_attention_heatmap_size must be a positive int")


def disable_prefill_cuda_graph_for_attention_heatmap(server_args: Any) -> None:
    """Runs right after handle_cuda_graph_config, so memory budgeting sees the
    final prefill backend."""
    cfg = resolving_view(server_args)
    if cfg.attention_heatmap_layer_ids is None:
        return
    prefill_backend = cfg.cuda_graph_config.prefill.backend
    if prefill_backend == Backend.DISABLED:
        return
    logger.warning(
        "Disabling prefill CUDA graph (%s) because attention_heatmap_layer_ids is "
        "set: the attention heatmap query recorder records one query per request "
        "per prefill step, which a captured prefill graph would replay with "
        "capture-time shapes.",
        prefill_backend,
    )
    declare_resolution(
        server_args,
        "_disable_prefill_cuda_graph_for_attention_heatmap",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )


def reject_speculative_decoding_for_attention_heatmap(server_args: Any) -> None:
    """Runs after handle_speculative_decoding, so speculative_algorithm is final."""
    cfg = resolving_view(server_args)
    if cfg.attention_heatmap_layer_ids is None or cfg.speculative_algorithm is None:
        return
    # Future direction: compute the heatmap after generation with one extra
    # forward pass over prompt + generated tokens, which would lift this.
    raise ValueError(
        "Attention heatmap (attention_heatmap_layer_ids) combined with speculative "
        f"decoding (speculative_algorithm={cfg.speculative_algorithm}) is not "
        "supported yet: the heatmap query recorder assumes one query per request "
        "per step, while speculative decoding verifies several draft tokens per "
        "step, discards rejected ones, and runs separate draft-model passes."
    )
