from typing import Optional

from dataclasses import dataclass, field
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from collections import defaultdict


import numpy as np
import torch

MiB = 1024**2


class AttentionHeatmapQueryRecorderMixin:
    """Records per-layer attention queries into a `query_buffer` for attention
    heatmap computation.

    Call `_init_attention_heatmap_query_buffer` from the model's __init__, and
    `_record_query_for_layer` from its forward loop once `q` is available.

    Layer ids are taken verbatim from ``server_args.attention_heatmap_layer_ids``
    (defaulting to all layers).
    """

    attention_heatmap_layer_ids: list[int]
    _heatmap_layer_id_to_buffer_idx: tuple[Optional[int], ...]
    query_buffer: torch.Tensor

    def _init_attention_heatmap_query_buffer(
        self,
        *,
        num_hidden_layers: int,
        hidden_size: int,
        torch_dtype: torch.dtype,
    ) -> None:
        server_args = get_global_server_args()
        max_batch_size: Optional[int] = server_args.max_running_requests
        assert max_batch_size is not None, (
            "Expecting max_running_requests to be set for query buffer initialization."
        )

        self.attention_heatmap_layer_ids = (
            list(server_args.attention_heatmap_layer_ids)
            if server_args.attention_heatmap_layer_ids is not None
            else list(range(num_hidden_layers))
        )
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
                device=torch.cuda.current_device(),
            ),
            persistent=False,
        )

    def _record_query_for_layer(
        self,
        layer_id: int,
        q: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
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

        if forward_batch.forward_mode.is_decode():
            # Decode mode: one new token per request. Record the query used to
            # generate that token for each request in the batch.
            # q: [batch_size, hidden_size]
            assert q.ndim == 2 and q.shape == (batch_size, hidden_size)
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
            assert q.ndim == 2 and q.shape == (sum(extend_seq_lens), hidden_size)

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

def maybe_drop_extra_query_due_to_overlap_scheduling(
    req_rid: str,
    req_num_output_tokens: int,
) -> None:
    """With Overlap Scheduling, we may have an extra query in the buffer due to a forward pass
    that hasn't been post-processed: request was already finished or retracted."""
    req_buffer = OUTPUT_TOKEN_QUERY_BUFFER[req_rid]
    assert len(req_buffer.queries) == req_num_output_tokens or len(req_buffer.queries) == req_num_output_tokens + 1
    if len(req_buffer.queries) == req_num_output_tokens + 1:
        req_buffer.queries.pop()

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
    req_rids = forward_batch.req_rids
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
                # This means the request was retracted during decoding due to lack of KV cache capacity.
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
) -> list[list[torch.Tensor]]:
    """Compute per-(output-token, prompt-token) attention weights for each
    layer recorded in the query buffer.

    `selected_key_cache[buffer_idx]` must hold the key cache for the same
    model layer as query-buffer slot `buffer_idx`. The caller is
    responsible for filtering / remapping the underlying KV pool (e.g.
    `HybridLinearKVPool` only stores keys for full-attention layers).
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
            chunk_scores = torch.bmm(query_chunk, keys_bmm) / (head_dim**0.5)

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
