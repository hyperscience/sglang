"""Attention heatmap (Hyperscience).

Returns, per generated token, the attention of that token's query over the
prompt tokens, averaged over heads and the selected layers
(``meta_info["attention_weights"]``).

How it works:

1. Models that support the heatmap inherit ``AttentionHeatmapQueryRecorderMixin``.
   During every target forward pass, the recorder copies post-RoPE queries of
   the selected layers (``attention_heatmap_layer_ids``) into ``query_buffer``:
   the query of every request in decode, the query of the last token of every
   request in prefill, and the query of every verified token in speculative
   decoding's TARGET_VERIFY. The recorder is attached to each selected attention
   module, which calls it right before ``RadixAttention``.
2. ``ModelRunner.forward`` snapshots those rows onto the step's
   ``LogitsProcessorOutput`` (``snapshot_attention_heatmap_queries``), and
   ``GenerationBatchResult.copy_to_cpu`` copies the snapshot to pinned host
   memory together with the rest of the step's outputs.
3. When the scheduler commits a step's tokens to a request, it keeps one query
   per committed token on the host (``append_attention_heatmap_queries``), so
   the queries cost no GPU memory while the request runs. With speculative
   decoding and a single draft chain (``speculative_eagle_topk == 1``), verify
   row j of a request produced its j-th emitted token, so keeping the first
   ``len(committed tokens)`` rows drops the rejected drafts.
4. When the request finishes, ``compute_attention_weights_for_finished_req``
   uploads its queries layer by layer and computes softmax(Q K^T) against the
   prompt keys still in the KV cache entirely on the GPU, with one
   device-to-host copy of the result (``compute_attn_weights_for_request``).

Constraints: page_size == 1, no mixed chunked prefill, and no prefill CUDA graph
(the per-request extend loop would be replayed with capture-time shapes): the
server-arg resolution disables prefill CUDA graph when the heatmap is configured
(``attention_heatmap_layer_ids`` or ``chunked_attention_heatmap_size`` set), and
restricts speculative decoding to the EAGLE V2 worker with
``speculative_eagle_topk == 1`` (``sglang.srt.arg_groups.hs_attention_heatmap_hook``).
When the heatmap is not configured, ``_init_attention_heatmap_query_buffer``
selects no layers, so models that call it unconditionally attach no recorder and
the constraints do not apply. Draft models (e.g. the Qwen3.5 MTP head) record
nothing.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numpy as np
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_schedule, get_spec

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

MiB = 1024**2


def _max_recorded_rows_per_request() -> int:
    """Query rows one request records in one forward pass: one, or the number of
    verified tokens per request with speculative decoding."""
    spec = get_spec()
    for name in ("max_speculative_num_draft_tokens", "speculative_num_draft_tokens"):
        try:
            value = getattr(spec, name, None)
        except Exception:  # noqa: BLE001 - config bag without the field
            value = None
        if value:
            return int(value)
    return 1


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
        enabled: bool = True,
    ) -> None:
        """`enabled=False` records nothing; draft models (e.g. an MTP head built
        from the target's model class) pass it, since their forward passes do not
        produce output tokens."""
        max_batch_size: Optional[int] = get_schedule().max_running_requests
        assert max_batch_size is not None, (
            "Expecting max_running_requests to be set for query buffer initialization."
        )

        # Optional per-model restriction (e.g. Gemma 4 only records
        # full-attention non-KV-shared layers).
        valid_set: Optional[set[int]] = (
            set(valid_layer_ids) if valid_layer_ids is not None else None
        )
        features = get_exec().features
        requested = features.attention_heatmap_layer_ids

        if not enabled or (
            requested is None and features.chunked_attention_heatmap_size is None
        ):
            # Draft model, or heatmap not configured (same test as
            # hs_attention_heatmap_hook.attention_heatmap_configured): record
            # nothing, so no attention module gets a recorder.
            self.attention_heatmap_layer_ids = []
        elif requested is not None:
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
        # Without selected layers the buffer is never written: keep it minimal.
        max_rows = (
            max_batch_size * _max_recorded_rows_per_request()
            if self.attention_heatmap_layer_ids
            else 1
        )
        self.register_buffer(
            "query_buffer",
            torch.zeros(
                (
                    num_query_buffer_layers,
                    # One row per request (decode, prefill), or per verified
                    # token (speculative decoding's TARGET_VERIFY).
                    max_rows,
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

        mode = forward_batch.forward_mode
        assert not mode.is_mixed(), (
            "MIXED forward mode, which mixes prefilling and decoding, is not supported for query buffer capture."
        )

        hidden_size = self.query_buffer.shape[-1]
        # Some attention paths hand over q as [tokens, num_heads, head_dim].
        q = q.reshape(q.shape[0], -1)

        if mode.is_decode() or mode.is_target_verify():
            # Decode: one new token per request, q: [batch_size, hidden_size].
            # Target verify (speculative decoding): every verified token, one
            # contiguous block per request, q: [batch_size * num_draft_tokens,
            # hidden_size]. Row j of a request's block is the query that
            # produces its j-th emitted token if that token is accepted; the
            # scheduler keeps only the accepted ones. Under CUDA graph replay
            # q is padded to the captured batch size; padding rows are never read.
            num_rows = q.shape[0]
            assert q.shape[1] == hidden_size
            assert num_rows <= self.query_buffer.shape[1], (
                f"{num_rows} query rows exceed the query buffer capacity "
                f"{self.query_buffer.shape[1]}."
            )
            self.query_buffer[buffer_idx, :num_rows] = q
            return

        if mode == ForwardMode.EXTEND:
            # Extend mode: prefilling multiple requests together. Record the
            # query for the last token of each request, since that query will
            # be used to generate the first output token.
            # q: [total_extend_tokens, hidden_size]
            batch_size = forward_batch.batch_size
            assert batch_size <= self.query_buffer.shape[1], (
                "Batch size exceeds query buffer capacity."
            )
            extend_seq_lens = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens is not None
            assert len(extend_seq_lens) == batch_size
            assert q.shape == (sum(extend_seq_lens), hidden_size)

            req_last_token_idx = -1
            for req_idx, extend_len in enumerate(extend_seq_lens):
                req_last_token_idx += extend_len
                self.query_buffer[buffer_idx, req_idx] = q[req_last_token_idx]
        # Other modes (idle, draft extend, split prefill) produce no output
        # token on this model: nothing to record.


def snapshot_attention_heatmap_queries(
    query_buffer: torch.Tensor,  # (num_layers, max_rows, hidden_size)
    forward_batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    """Copy of the queries recorded during this forward pass, shaped
    (num_layers, batch_size, rows_per_request, hidden_size): one row per request
    in decode and prefill, one per verified token in TARGET_VERIFY. None for
    modes that record nothing.

    The copy is needed because the next forward pass overwrites query_buffer
    before an overlapped scheduler processes this pass's results."""
    mode = forward_batch.forward_mode
    if mode.is_target_verify():
        rows_per_request = forward_batch.spec_info.draft_token_num
    elif mode.is_decode() or mode == ForwardMode.EXTEND:
        rows_per_request = 1
    else:
        return None
    batch_size = forward_batch.batch_size
    num_layers, _, hidden_size = query_buffer.shape
    num_rows = batch_size * rows_per_request
    return (
        query_buffer[:, :num_rows]
        .view(num_layers, batch_size, rows_per_request, hidden_size)
        .clone()
    )


def host_attention_heatmap_queries(logits_output) -> Optional[torch.Tensor]:
    """The step's query snapshot on the host. It is already there when
    `GenerationBatchResult.copy_to_cpu` ran (overlap scheduling)."""
    queries = getattr(logits_output, "hs_attention_heatmap_queries", None)
    if queries is not None and queries.device.type != "cpu":
        queries = queries.cpu()
    return queries


def append_attention_heatmap_queries(
    req: "Req",
    queries: Optional[torch.Tensor],  # (num_layers, batch_size, rows, hidden) on host
    req_idx: int,
    num_tokens: int,
) -> None:
    """Keep the queries that produced the `num_tokens` tokens just committed to
    `req` (rows 0..num_tokens-1 of its block)."""
    if num_tokens <= 0:
        return
    assert queries is not None, (
        f"No attention heatmap queries were recorded for request {req.rid}."
    )
    assert num_tokens <= queries.shape[2], (
        f"Request {req.rid} committed {num_tokens} tokens but only "
        f"{queries.shape[2]} query rows were recorded for it."
    )
    # Clone: a view would keep the whole step's pinned snapshot alive until
    # the request finishes.
    req.attention_heatmap_queries.append(queries[:, req_idx, :num_tokens].clone())


def trim_attention_heatmap_queries(req: "Req", num_tokens: int) -> None:
    """Keep the queries of the first `num_tokens` output tokens of `req`."""
    kept: list[torch.Tensor] = []
    total = 0
    for chunk in req.attention_heatmap_queries:
        if total >= num_tokens:
            break
        take = min(chunk.shape[1], num_tokens - total)
        kept.append(chunk if take == chunk.shape[1] else chunk[:, :take].clone())
        total += take
    req.attention_heatmap_queries = kept


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


@torch.no_grad()
def compute_attn_weights_for_request(
    selected_key_cache: list[
        torch.Tensor
    ],  # [num_selected_layers][KV cache size, num_k_heads, head_dim]
    req_queries: torch.Tensor,  # [num_selected_layers, num_output_tokens, num_q_heads * head_dim]
    req_prompt_token_indices: torch.Tensor,  # [num_prompt_tokens] KV cache indices
    page_size: int,
    chunked_attention_heatmap_size: Optional[int],
    attention_score_scaling: Optional[float] = None,
) -> np.ndarray:
    """Attention of each output token over the prompt tokens, averaged over
    heads and over the selected layers: float16 [num_output_tokens,
    num_prompt_tokens].

    Per layer: softmax(Q K^T * scaling) over the prompt keys in float32,
    averaged over heads; the first prompt token (null attention,
    https://aclanthology.org/W19-4808.pdf, usually very large) is zeroed and
    each row renormalized; then the layers are averaged.

    Runs on the KV cache device without host synchronization until the end:
    each chunk of output tokens is reduced over all layers on the GPU and
    copied asynchronously into one pinned host array. Grouped-query attention
    is handled by grouping the queries of each KV head instead of repeating the
    keys. Peak GPU memory is the float32 keys of one layer plus the float32
    scores of one chunk of `chunked_attention_heatmap_size` output tokens,
    independent of the number of output tokens. `req_queries` may live on the
    host (pinned for asynchronous uploads).

    `selected_key_cache[i]` must hold the key cache of the model layer whose
    queries are `req_queries[i]`. `attention_score_scaling` is the model's
    `RadixAttention(scaling=...)` (e.g. `1.0` for Gemma 4); `None` means
    `head_dim**-0.5`.
    """
    assert page_size == 1, "Implemented only for page_size == 1"

    num_selected_layers, num_output_tokens, query_dim = req_queries.shape
    assert num_selected_layers == len(selected_key_cache), (
        f"Expecting queries for {len(selected_key_cache)} layers "
        f"(matching selected_key_cache), but got {num_selected_layers}."
    )
    device = selected_key_cache[0].device
    req_prompt_token_indices = req_prompt_token_indices.to(
        device=device, dtype=torch.int64
    )
    num_prompt_tokens = req_prompt_token_indices.shape[0]
    chunk_size = chunked_attention_heatmap_size or num_output_tokens

    attn_weights = torch.empty(
        (num_output_tokens, num_prompt_tokens),
        dtype=torch.float16,
        pin_memory=device.type == "cuda",
    )
    for start_idx in range(0, num_output_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, num_output_tokens)
        chunk_len = end_idx - start_idx
        chunk_attn_sum = torch.zeros(
            (chunk_len, num_prompt_tokens), dtype=torch.float32, device=device
        )
        for buffer_idx in range(num_selected_layers):
            # [num_prompt_tokens, num_k_heads, head_dim]
            keys = selected_key_cache[buffer_idx].index_select(
                0, req_prompt_token_indices
            )
            num_k_heads, head_dim = keys.shape[-2:]
            num_q_heads = query_dim // head_dim
            group_size = num_q_heads // num_k_heads
            layer_scaling = (
                attention_score_scaling
                if attention_score_scaling is not None
                else head_dim**-0.5
            )
            # [num_k_heads, head_dim, num_prompt_tokens], float32 for a stable softmax.
            keys_t = keys.permute(1, 2, 0).to(torch.float32)
            del keys

            # Query head h attends with KV head h // group_size:
            # [num_k_heads, group_size * chunk_len, head_dim]
            query_chunk = (
                req_queries[buffer_idx, start_idx:end_idx]
                .to(device, non_blocking=True)
                .view(chunk_len, num_k_heads, group_size, head_dim)
                .permute(1, 2, 0, 3)
                .reshape(num_k_heads, group_size * chunk_len, head_dim)
                .to(torch.float32)
            )
            query_chunk.mul_(layer_scaling)

            # [num_q_heads, chunk_len, num_prompt_tokens]
            scores = torch.bmm(query_chunk, keys_t).view(
                num_q_heads, chunk_len, num_prompt_tokens
            )
            del query_chunk, keys_t
            # In-place softmax over the prompt tokens.
            scores.sub_(scores.amax(dim=-1, keepdim=True)).exp_()
            scores.div_(scores.sum(dim=-1, keepdim=True))

            # Average over heads, drop the null attention, renormalize.
            layer_attn = scores.mean(dim=0)  # [chunk_len, num_prompt_tokens]
            del scores
            layer_attn[:, 0] = 0.0
            layer_attn.div_(layer_attn.sum(dim=-1, keepdim=True))
            chunk_attn_sum.add_(layer_attn)
            del layer_attn

        chunk_attn_sum.div_(num_selected_layers)
        # Stream-ordered: the source stays valid until the copy has run.
        attn_weights[start_idx:end_idx].copy_(
            chunk_attn_sum.to(torch.float16), non_blocking=True
        )
        del chunk_attn_sum

    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return attn_weights.numpy()


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
    """Compute `req.attention_weights` (one float16 array over the prompt per
    output token) from the queries kept on the host and the prompt keys still
    in the KV cache. Call before the request's KV cache is freed."""
    start_time = time.perf_counter()

    # Speculative decoding can accept tokens past the stop condition; the
    # response stops at finished_len (Req.output_ids_through_stop).
    num_output_tokens = len(req.output_ids_through_stop)
    trim_attention_heatmap_queries(req, num_output_tokens)
    chunks = req.attention_heatmap_queries
    req.attention_heatmap_queries = []
    num_recorded_tokens = sum(chunk.shape[1] for chunk in chunks)
    assert num_recorded_tokens == num_output_tokens, (
        f"Attention heatmap queries out of sync for {req.rid=}: "
        f"{num_recorded_tokens=}, {num_output_tokens=}"
    )
    if num_output_tokens == 0:
        req.attention_weights = []
        return

    # Track GPU memory before computation
    track_memory = torch.cuda.is_available()
    if track_memory:
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated() / MiB
        reserved_before = torch.cuda.memory_reserved() / MiB

    # One host copy into pinned memory, so the per-layer uploads are asynchronous.
    num_layers, _, query_dim = chunks[0].shape
    req_queries = torch.empty(
        (num_layers, num_output_tokens, query_dim),
        dtype=chunks[0].dtype,
        pin_memory=track_memory,
    )
    torch.cat(chunks, dim=1, out=req_queries)
    del chunks
    query_buffer_mb = req_queries.numel() * req_queries.element_size() / MiB

    # Indices of request prompt tokens in the KV cache
    req_prompt_token_indices = req_to_token[
        req.kv.req_pool_idx, : len(req.origin_input_ids)
    ]

    # The model owns the canonical list of layer ids whose queries are recorded
    # in the query buffer (in buffer order).
    selected_layer_ids: list[int] = list(heatmap_model.attention_heatmap_layer_ids)
    attn_weights = compute_attn_weights_for_request(
        selected_key_cache=select_heatmap_key_cache(kvcache, selected_layer_ids),
        req_queries=req_queries,
        req_prompt_token_indices=req_prompt_token_indices,
        page_size=page_size,
        chunked_attention_heatmap_size=chunked_attention_heatmap_size,
        attention_score_scaling=getattr(heatmap_model, "attention_score_scaling", None),
    )
    req.attention_weights = list(attn_weights)

    msg = (
        f"Compute attention weights, "
        f"#input-token: {len(req_prompt_token_indices)}, "
        f"#output-token: {num_output_tokens}, "
        f"selected-layer-ids: {selected_layer_ids}, "
        f"chunk-size: {chunked_attention_heatmap_size}, "
        f"query-buffer (host): {int(query_buffer_mb)} MiB, "
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
