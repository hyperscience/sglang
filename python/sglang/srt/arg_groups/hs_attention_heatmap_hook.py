# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the attention heatmap (Hyperscience).

See sglang.srt.hs.attention_heatmap. The heatmap query recorder assumes one
query per request per forward step, recorded for the last token of each
request:

- Prefill CUDA graph (breakable by default since v0.5.12) captures the
  recorder's per-request loop over the extend lengths once, at capture time,
  so every replay would record the query of a padding token. Prefill CUDA
  graph is therefore disabled whenever the heatmap is configured, i.e.
  attention_heatmap_layer_ids or chunked_attention_heatmap_size is set
  (layer ids may be None, meaning all layers).
- Speculative decoding verifies several draft tokens per step and discards the
  rejected ones. The recorder keeps the query of every verified token and the
  scheduler keeps the ones of the accepted tokens, which relies on the EAGLE V2
  verify layout: one contiguous block of verified tokens per request, accepted
  tokens first (a single draft chain, speculative_eagle_topk == 1). Other
  speculative setups are rejected.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

logger = logging.getLogger(__name__)


def attention_heatmap_configured(cfg: Any) -> bool:
    """Whether the server is set up to serve attention heatmaps. forms sets
    chunked_attention_heatmap_size for heatmap models and may leave
    attention_heatmap_layer_ids as None (all layers)."""
    return (
        cfg.attention_heatmap_layer_ids is not None
        or cfg.chunked_attention_heatmap_size is not None
    )


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
    if not attention_heatmap_configured(cfg):
        return
    prefill_backend = cfg.cuda_graph_config.prefill.backend
    if prefill_backend == Backend.DISABLED:
        return
    logger.warning(
        "Disabling prefill CUDA graph (%s) because the attention heatmap is "
        "configured (attention_heatmap_layer_ids=%s, "
        "chunked_attention_heatmap_size=%s): its query recorder records one "
        "query per request "
        "per prefill step, which a captured prefill graph would replay with "
        "capture-time shapes.",
        prefill_backend,
        cfg.attention_heatmap_layer_ids,
        cfg.chunked_attention_heatmap_size,
    )
    declare_resolution(
        server_args,
        "_disable_prefill_cuda_graph_for_attention_heatmap",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )


def validate_speculative_decoding_for_attention_heatmap(server_args: Any) -> None:
    """Runs after handle_speculative_decoding, so speculative_algorithm is final
    (NEXTN is resolved to EAGLE by then)."""
    cfg = resolving_view(server_args)
    if not attention_heatmap_configured(cfg) or cfg.speculative_algorithm is None:
        return
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algorithm = SpeculativeAlgorithm.from_string(cfg.speculative_algorithm)
    supported_algorithm = algorithm in (
        SpeculativeAlgorithm.EAGLE,
        SpeculativeAlgorithm.EAGLE3,
    ) and not cfg.enable_multi_layer_eagle
    topk = cfg.speculative_eagle_topk
    if supported_algorithm and topk in (None, 1):
        return
    raise ValueError(
        "Attention heatmap (attention_heatmap_layer_ids / "
        "chunked_attention_heatmap_size) with speculative decoding requires the "
        "EAGLE V2 worker (speculative_algorithm EAGLE, EAGLE3 or NEXTN, without "
        "enable_multi_layer_eagle) and a single draft chain "
        f"(speculative_eagle_topk=1); got speculative_algorithm="
        f"{cfg.speculative_algorithm}, enable_multi_layer_eagle="
        f"{cfg.enable_multi_layer_eagle}, speculative_eagle_topk={topk}."
    )
