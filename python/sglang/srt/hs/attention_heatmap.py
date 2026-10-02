"""Attention heatmap (Hyperscience).

Returns, per generated token, the attention of that token's query over the
prompt tokens, averaged over heads and the selected layers
(``meta_info["attention_weights"]``).

How it works:

1. Models that support the heatmap inherit ``AttentionHeatmapQueryRecorderMixin``.
   During every forward pass, the recorder copies the post-RoPE query of the
   last token of each request into ``query_buffer`` for the selected layers
   (``attention_heatmap_layer_ids``). The recorder is attached to each selected
   attention module, which calls it right before ``RadixAttention``.
2. After each forward pass, ``fill_output_token_query_buffer_for_batch`` appends
   the recorded queries of the requests that asked for the heatmap
   (``output_attention_weights``) to a per-request buffer.
3. When a request finishes, ``compute_attention_weights_for_finished_req``
   recomputes softmax(Q K^T) between the recorded queries and the prompt keys
   still in the KV cache (``compute_attn_weights_for_request``).

Constraints: page_size == 1, no mixed chunked prefill, no speculative decoding
(the recorder assumes one query per request per forward step), and no prefill
CUDA graph (the per-request extend loop would be replayed with capture-time
shapes). The server-arg resolution disables prefill CUDA graph and rejects
speculative decoding when the heatmap is configured (``attention_heatmap_layer_ids``
or ``chunked_attention_heatmap_size`` set;
``sglang.srt.arg_groups.hs_attention_heatmap_hook``).
"""

from __future__ import annotations

import functools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_schedule

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

MiB = 1024**2


class AttentionHeatmapQueryRecorderMixin:
    """Records per-layer attention queries into a `query_buffer` for attention
    heatmap computation.

    Call `_init_attention_heatmap_query_buffer` from the model's __init__, then
    `_attach_attention_heatmap_recorders` with the attention module of each
    layer. An attached attention module calls its `attention_heatmap_recorder`
    once its post-RoPE `q` is available.

    Layer ids are taken verbatim from ``attention_heatmap_layer_ids`` (defaulting
    to all layers, or to `valid_layer_ids` when the model restricts them).
    """

    attention_heatmap_layer_ids: list[int]
    _heatmap_layer_id_to_buffer_idx: tuple[Optional[int], ...]
    query_buffer: torch.Tensor

    # Softmax scaling applied to `Q @ K^T` during heatmap recomputation.
    # Must match the model's `RadixAttention(scaling=...)`. `None` ⇒
    # default `head_dim**-0.5` (Qwen). Override in subclasses whose
    # attention uses a different scaling (e.g. Gemma 4 uses `1.0`).
    attention_score_scaling: Optional[float] = None

    def _init_attention_heatmap_query_buffer(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        torch_dtype: torch.dtype,
        valid_layer_ids: Optional[Iterable[int]] = None,
    ) -> None:
        max_batch_size: Optional[int] = get_schedule().max_running_requests
        assert max_batch_size is not None, (
            "Expecting max_running_requests to be set for query buffer initialization."
        )

        # Optional per-model restriction (e.g. Gemma 4 only records
        # full-attention non-KV-shared layers).
        valid_set: Optional[set[int]] = (
            set(valid_layer_ids) if valid_layer_ids is not None else None
        )
        requested = get_exec().features.attention_heatmap_layer_ids

        if requested is not None:
            requested = list(requested)
            out_of_range = [i for i in requested if i >= num_hidden_layers]
            if out_of_range:
                raise ValueError(
                    f"attention_heatmap_layer_ids contains layer ids "
                    f"{out_of_range} but the model has {num_hidden_layers} layers."
                )
            if valid_set is not None:
                invalid = [i for i in requested if i not in valid_set]
                if invalid:
                    raise ValueError(
                        f"attention_heatmap_layer_ids contains layer ids "
                        f"{invalid} that are not supported by this model "
                        f"(supported ids: {sorted(valid_set)})."
                    )
            self.attention_heatmap_layer_ids = requested
        elif valid_set is not None:
            self.attention_heatmap_layer_ids = sorted(valid_set)
        else:
            self.attention_heatmap_layer_ids = list(range(num_hidden_layers))
        # Tuple indexed by layer_id (not a dict) so the lookup in
        # `_record_query_for_layer` stays constant-foldable under
        # torch.compile / CUDA graph capture.
        layer_id_to_buffer_idx: list[Optional[int]] = [None] * num_hidden_layers
        for buffer_idx, layer_id in enumerate(self.attention_heatmap_layer_ids):
            layer_id_to_buffer_idx[layer_id] = buffer_idx
        self._heatmap_layer_id_to_buffer_idx = tuple(layer_id_to_buffer_idx)

        num_query_buffer_layers = max(len(self.attention_heatmap_layer_ids), 1)
        self.register_buffer(
            "query_buffer",
            torch.zeros(
                (
                    num_query_buffer_layers,
                    max_batch_size,  # capacity for queries across all requests in the batch
                    hidden_size,  # num_q_heads * head_dim
                ),
                dtype=torch_dtype,
                device=torch.get_device_module().current_device(),
            ),
            persistent=False,
        )

    def _attach_attention_heatmap_recorders(
        self, attention_modules: Mapping[int, Optional["nn.Module"]]
    ) -> None:
        """Give the attention module of each selected layer a recorder.

        `attention_modules` maps layer ids to the module that calls
        `self.attention_heatmap_recorder(q, forward_batch)` in its forward (it
        must define `attention_heatmap_recorder = None` as a class attribute).
        Layers that are not selected keep `None`, so they pay nothing. `None`
        modules (e.g. layers of a subclass without such an attention module)
        are skipped.
        """
        for layer_id, module in attention_modules.items():
            if module is None or self._heatmap_layer_id_to_buffer_idx[layer_id] is None:
                continue
            # Plain attribute, not a submodule: nn.Module.__setattr__ only
            # registers Modules / Parameters / buffers.
            module.attention_heatmap_recorder = functools.partial(
                self._record_query_for_layer, layer_id
            )

    @torch._dynamo.disable
    def _record_query_for_layer(
        self,
        layer_id: int,
        q: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        # @torch._dynamo.disable: tracing this per-layer recording into the
        # compiled forward increased CUDA-graph capture time by ~1.5x
        # (AML-4595). Running it eager still records the in-place query_buffer
        # writes under sglang's outer CUDA-graph capture, so behaviour is
        # unchanged (forward is compiled without fullgraph, so the graph break
        # is allowed).
        buffer_idx = self._heatmap_layer_id_to_buffer_idx[layer_id]
        if buffer_idx is None:
            return

        assert not forward_batch.forward_mode.is_mixed(), (
            "MIXED forward mode, which mixes prefilling and decoding, is not supported for query buffer capture."
        )

        batch_size = forward_batch.batch_size
        hidden_size = self.query_buffer.shape[-1]
        assert batch_size <= self.query_buffer.shape[1], (
            "Batch size exceeds query buffer capacity."
        )
        # Some attention paths hand over q as [tokens, num_heads, head_dim].
        q = q.reshape(q.shape[0], -1)

        if forward_batch.forward_mode.is_decode():
            # Decode mode: one new token per request. Record the query used to
            # generate that token for each request in the batch.
            # q: [batch_size, hidden_size]
            assert q.shape == (batch_size, hidden_size)
            self.query_buffer[buffer_idx][:batch_size] = q
            return

        if forward_batch.forward_mode.is_extend():
            # Extend mode: prefilling multiple requests together. Record the
            # query for the last token of each request, since that query will
            # be used to generate the first output token.
            # q: [total_extend_tokens, hidden_size]
            extend_seq_lens = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens is not None
            assert len(extend_seq_lens) == batch_size
            assert q.shape == (sum(extend_seq_lens), hidden_size)

            req_last_token_idx = -1
            for req_idx, extend_len in enumerate(extend_seq_lens):
                req_last_token_idx += extend_len
                self.query_buffer[buffer_idx, req_idx] = q[req_last_token_idx]


@dataclass
class RequestOutputTokenQueryBuffer:
    # the mode for the last forward pass involving this request: either EXTEND or DECODE.
    last_forward_pass_mode: Optional[ForwardMode] = None
    # Stores the query used to generate each output token, for each running request.
    # A new token is generated for the last prefill and each decode forward pass.
    queries: list[torch.Tensor] = field(
        default_factory=list
    )  # list[torch.Tensor((num_selected_layers, hidden_size))]]


OUTPUT_TOKEN_QUERY_BUFFER: dict[str, RequestOutputTokenQueryBuffer] = defaultdict(
    RequestOutputTokenQueryBuffer  # key is Req.rid
)


def validate_attention_heatmap_request() -> Optional[str]:
    """Error message if a request with output_attention_weights cannot be served
    by this server, else None."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )

    if not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
        # Safety net: configuring the heatmap already turns prefill CUDA graph
        # off (hs_attention_heatmap_hook); this fires only for a server started
        # without any heatmap args whose prefill graph was not auto-disabled.
        return (
            "output_attention_weights requires prefill CUDA graph to be disabled: "
            "a captured prefill graph replays the attention heatmap query "
            "recorder with capture-time shapes. Set chunked_attention_heatmap_size "
            "or attention_heatmap_layer_ids (or disable_prefill_cuda_graph=True) "
            "when starting the server."
        )
    return None


def maybe_drop_extra_query_due_to_overlap_scheduling(
    req_rid: str,
    req_num_output_tokens: int,
) -> None:
    """With Overlap Scheduling, we may have an extra query in the buffer due to a forward pass
    that hasn't been post-processed: request was already finished or retracted."""
    req_buffer = OUTPUT_TOKEN_QUERY_BUFFER[req_rid]
    assert (
        len(req_buffer.queries) == req_num_output_tokens
        or len(req_buffer.queries) == req_num_output_tokens + 1
    ), (
        f"Query buffer out of sync for {req_rid=}: {len(req_buffer.queries)=}, "
        f"{req_num_output_tokens=}, {req_buffer.last_forward_pass_mode=}"
    )
    if len(req_buffer.queries) == req_num_output_tokens + 1:
        req_buffer.queries.pop()
    # The buffer now holds exactly one query per output token. A retracted request
    # is prefilled again next, and that prefill's first chunk must append a new query,
    # even if the request was retracted right after a prefill (mode EXTEND, where
    # the next EXTEND would otherwise replace queries[-1]). Harmless for a finished
    # request, whose buffer is popped right after.
    req_buffer.last_forward_pass_mode = ForwardMode.DECODE


def fill_output_token_query_buffer_for_batch(
    query_buffer: torch.Tensor,  # (num_layers, max_num_requests, hidden_size)
    forward_batch: ForwardBatch,
) -> None:
    """Populates the buffer with the queries of the current forward batch."""
    mode = forward_batch.forward_mode
    assert mode in (ForwardMode.EXTEND, ForwardMode.DECODE), (
        f"{forward_batch.forward_mode=} is not supported."
    )
    assert query_buffer.ndim == 3

    batch_size = forward_batch.batch_size
    req_rids = forward_batch.rids
    output_attention_weights = forward_batch.output_attention_weights
    assert req_rids is not None and output_attention_weights is not None
    assert batch_size == len(req_rids) == len(output_attention_weights)
    assert batch_size <= query_buffer.shape[1]

    if mode == ForwardMode.DECODE:
        for req_idx, req_rid in enumerate(req_rids):
            if not output_attention_weights[req_idx]:
                continue
            request_query_buffer = OUTPUT_TOKEN_QUERY_BUFFER[req_rid]
            # As we are generating one new token in this forward pass, we append the query.
            request_query_buffer.queries.append(query_buffer[:, req_idx, :].clone())
            request_query_buffer.last_forward_pass_mode = ForwardMode.DECODE
    else:
        for req_idx, req_rid in enumerate(req_rids):
            if not output_attention_weights[req_idx]:
                continue
            request_query_buffer = OUTPUT_TOKEN_QUERY_BUFFER[req_rid]
            if request_query_buffer.last_forward_pass_mode == ForwardMode.EXTEND:
                # Replace the query for the last forward pass, as the pass did not generate any new token.
                request_query_buffer.queries[-1] = query_buffer[:, req_idx, :].clone()
            elif request_query_buffer.last_forward_pass_mode == ForwardMode.DECODE:
                # This means the request was retracted due to lack of KV cache capacity: either during decoding,
                # or right after a prefill (maybe_drop_extra_query_due_to_overlap_scheduling resets the mode to
                # DECODE on retraction, after leaving exactly one query per output token).
                # Now, we are doing the first prefill forward pass, which may contain both prompt and already decoded tokens.
                # We keep the existing queries corresponding to the already decoded tokens and append the query
                # for the current forward pass, which may be generating a new token if there are no more remaining tokens to prefill.
                request_query_buffer.queries.append(query_buffer[:, req_idx, :].clone())
            else:
                # First prefill forward pass for this request.
                assert request_query_buffer.last_forward_pass_mode is None
                request_query_buffer.queries.append(query_buffer[:, req_idx, :].clone())
            request_query_buffer.last_forward_pass_mode = ForwardMode.EXTEND


def get_req_query_buffer_mb(
    req_query_buffer: list[
        list[torch.Tensor]
    ],  # [num_output_tokens, num_layers, (num_q_heads * head_dim)]
) -> float:
    if not req_query_buffer:
        return 0.0

    # req_query_buffer[token_idx][layer_idx] is a torch.Tensor
    # We count elements in one token's worth of layers
    total_elements = 0
    for layer_tensor in req_query_buffer[0]:
        total_elements += layer_tensor.numel()

    # Multiply by number of output tokens
    total_elements *= len(req_query_buffer)

    # Get bytes per element (e.g., 2 for float16)
    bytes_per_element = req_query_buffer[0][0].element_size()

    # Convert to Megabytes (1024^2)
    return (total_elements * bytes_per_element) / MiB


def compute_attn_weights_for_request(
    selected_key_cache: list[
        torch.Tensor
    ],  # [num_selected_layers, KV cache size, num_k_heads, head_dim]
    req_query_buffer: list[
        list[torch.Tensor]
    ],  # [num_output_tokens, num_selected_layers, (num_q_heads * head_dim)]
    req_prompt_token_indices: list[int],  # indices of prompt tokens in the KV cache
    page_size: int,
    chunked_attention_heatmap_size: Optional[int],
    attention_score_scaling: Optional[float] = None,
) -> list[list[torch.Tensor]]:
    """Compute per-(output-token, prompt-token) attention weights for each
    layer recorded in the query buffer.

    Depends only on the keys and queries passed in, not on any decode-step
    state, so it can also serve a post-generation pass that records the queries
    of prompt + generated tokens in one extra forward pass.

    `selected_key_cache[buffer_idx]` must hold the key cache for the same
    model layer as query-buffer slot `buffer_idx`. The caller is
    responsible for filtering / remapping the underlying KV pool (e.g.
    `HybridLinearKVPool` only stores keys for full-attention layers).

    `attention_score_scaling` is the scalar applied to ``Q @ K^T`` before
    the softmax — pass the model's `RadixAttention(scaling=...)` value
    (e.g. `1.0` for Gemma 4). `None` defaults to `head_dim**-0.5`.
    """
    assert page_size == 1, "Implemented only for page_size == 1"

    num_selected_layers = len(selected_key_cache)
    assert num_selected_layers == len(req_query_buffer[0]), (
        f"Expecting query buffer to have {num_selected_layers} layers "
        f"(matching selected_key_cache), but got {len(req_query_buffer[0])}."
    )

    num_output_tokens = len(req_query_buffer)
    num_prompt_tokens = len(req_prompt_token_indices)

    layers_attn_weights_per_token: list[list[torch.Tensor]] | None = None

    for buffer_idx in range(num_selected_layers):
        # Prepare Keys [num_prompt_tokens, num_k_heads, head_dim]
        keys_base = selected_key_cache[buffer_idx][req_prompt_token_indices, :, :]
        num_k_heads, head_dim = keys_base.shape[-2:]

        layer_scaling = (
            attention_score_scaling
            if attention_score_scaling is not None
            else head_dim**-0.5
        )

        # Reconstruct Queries [num_output_tokens, num_q_heads, head_dim]
        query_last_dimension = req_query_buffer[0][buffer_idx].shape[-1]
        num_q_heads = query_last_dimension // head_dim

        all_querys = torch.concat(
            [
                query[buffer_idx].reshape(1, num_q_heads, head_dim)
                for query in req_query_buffer
            ],
            axis=0,
        )

        # Handle GQA & Transpose Keys once
        repeat_factor = num_q_heads // num_k_heads
        if repeat_factor > 1:
            keys_base = keys_base.repeat_interleave(repeat_factor, dim=1)

        # Move to float32 for stable softmax and prepare for BMM: [num_q_heads, head_dim, num_prompt_tokens]
        keys_bmm = keys_base.permute(1, 2, 0).to(torch.float32)

        # Process attention scores in chunks (or all at once if chunk_size is None)
        chunk_size = chunked_attention_heatmap_size or num_output_tokens

        # Pre-allocate the result tensor for this layer on CPU
        layer_scores_cpu = torch.empty(
            (num_output_tokens, num_prompt_tokens), dtype=torch.float16
        )

        for start_idx in range(0, num_output_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, num_output_tokens)

            # Extract chunk and reshape for BMM: [num_q_heads, chunk_len, head_dim]
            query_chunk = (
                all_querys[start_idx:end_idx].transpose(0, 1).to(torch.float32)
            )

            # [num_q_heads, chunk_len, head_dim] @ [num_q_heads, head_dim, num_prompt_tokens]
            # Result: [num_q_heads, chunk_len, num_prompt_tokens]
            chunk_scores = torch.bmm(query_chunk, keys_bmm) * layer_scaling

            # Stable Softmax in float32
            chunk_scores = torch.softmax(chunk_scores, dim=-1)

            # Average over heads and offload: [chunk_len, num_prompt_tokens]
            layer_scores_cpu[start_idx:end_idx] = chunk_scores.mean(dim=0).half().cpu()

            # Optional: explicitly clear large GPU tensors
            del query_chunk, chunk_scores

        # Populate result structure
        if layers_attn_weights_per_token is None:
            layers_attn_weights_per_token = [
                [token_attn] for token_attn in layer_scores_cpu
            ]
        else:
            for i, token_attn in enumerate(layer_scores_cpu):
                layers_attn_weights_per_token[i].append(token_attn)

    return layers_attn_weights_per_token


def aggregate_attentions(attentions: list[torch.Tensor]) -> np.ndarray:
    """
    Extract mean attentions over all layers and heads
    """
    layer_attentions = []
    for layer_attention in attentions:
        attention_device = layer_attention.device
        layer_attention_head_avg = torch.concat(
            (
                # The attention to the first prompt token is called null attention
                # (https://aclanthology.org/W19-4808.pdf)
                # Usually it is very large compared to other attention values
                # Replacing it with 0 instead
                torch.tensor(
                    [0.0], device=attention_device, dtype=layer_attention.dtype
                ),
                # Use [-1] here to only keep the attention from the newly generated token
                layer_attention[1:],
            )
        )
        layer_attentions.append(
            layer_attention_head_avg / layer_attention_head_avg.sum()
        )
    return torch.stack(layer_attentions).mean(dim=0).cpu().numpy()


def select_heatmap_key_cache(kvcache, layer_ids: list[int]) -> list[torch.Tensor]:
    """Key cache of each layer in `layer_ids`, in that order.

    `get_key_buffer(layer_id)` is the polymorphic accessor that handles MHA /
    MLA / SWA pools. `HybridLinearKVPool` needs an explicit remap because it
    only stores keys for full-attention layers.
    """
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if isinstance(kvcache, HybridLinearKVPool):
        return [
            kvcache.full_kv_pool.k_buffer[
                kvcache.full_attention_layer_id_mapping[layer_id]
            ]
            for layer_id in layer_ids
        ]
    return [kvcache.get_key_buffer(layer_id) for layer_id in layer_ids]


def compute_attention_weights_for_finished_req(
    req: "Req",
    *,
    heatmap_model: AttentionHeatmapQueryRecorderMixin,
    kvcache,
    req_to_token: torch.Tensor,
    page_size: int,
    chunked_attention_heatmap_size: Optional[int],
) -> None:
    """Compute `req.attention_weights` from the recorded queries and the prompt
    keys still in the KV cache. Call before the request's KV cache is freed."""
    start_time = time.perf_counter()
    maybe_drop_extra_query_due_to_overlap_scheduling(
        req_rid=req.rid, req_num_output_tokens=len(req.output_ids)
    )

    # Track GPU memory before computation
    track_memory = torch.cuda.is_available()
    if track_memory:
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated() / MiB
        reserved_before = torch.cuda.memory_reserved() / MiB

    req_query_buffer = OUTPUT_TOKEN_QUERY_BUFFER.pop(req.rid).queries
    assert len(req_query_buffer) == len(req.output_ids)
    query_buffer_mb = get_req_query_buffer_mb(req_query_buffer)

    # Indices of request prompt tokens in the KV cache
    req_prompt_token_indices = req_to_token[
        req.kv.req_pool_idx, : len(req.origin_input_ids)
    ].tolist()

    # The model owns the canonical list of layer ids whose queries are recorded
    # in the query buffer (in buffer order).
    selected_layer_ids: list[int] = list(heatmap_model.attention_heatmap_layer_ids)
    layers_attn_weights = compute_attn_weights_for_request(
        selected_key_cache=select_heatmap_key_cache(kvcache, selected_layer_ids),
        req_query_buffer=req_query_buffer,
        req_prompt_token_indices=req_prompt_token_indices,
        page_size=page_size,
        chunked_attention_heatmap_size=chunked_attention_heatmap_size,
        attention_score_scaling=getattr(heatmap_model, "attention_score_scaling", None),
    )
    req.attention_weights = list(map(aggregate_attentions, layers_attn_weights))

    msg = (
        f"Compute attention weights, "
        f"#input-token: {len(req_prompt_token_indices)}, "
        f"#output-token: {len(req.output_ids)}, "
        f"selected-layer-ids: {selected_layer_ids}, "
        f"chunk-size: {chunked_attention_heatmap_size}, "
        f"query-buffer: {int(query_buffer_mb)} MiB, "
    )
    if track_memory:
        # Track peak GPU memory usage
        peak_alloc = torch.cuda.max_memory_allocated() / MiB
        peak_reserved = torch.cuda.max_memory_reserved() / MiB
        msg += (
            f"VRAM-alloc-peak-increase: {int(max(0.0, peak_alloc - alloc_before))} MiB, "
            f"VRAM-reserved-peak-increase: {int(max(0.0, peak_reserved - reserved_before))} MiB, "
        )
    msg += f"time: {time.perf_counter() - start_time:.3f} s, "
    logger.info(msg)
