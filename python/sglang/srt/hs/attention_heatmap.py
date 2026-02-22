from typing import Optional

from dataclasses import dataclass, field
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from collections import defaultdict


import numpy as np
import torch

MiB = 1024**2


@dataclass
class RequestOutputTokenQueryBuffer:
    # the mode for the last forward pass involving this request: either EXTEND or DECODE.
    last_forward_pass_mode: Optional[ForwardMode] = None
    # Stores the query used to generate each output token, for each running request.
    # A new token is generated for the last prefill and each decode forward pass.
    queries: list[torch.Tensor] = field(
        default_factory=list
    )  # list[torch.Tensor((num_hidden_layers, hidden_size))]]



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
    key_cache_buffer: list[
        torch.Tensor
    ],  # [num_layers, KV cache size, num_k_heads, head_dim]
    req_query_buffer: list[
        list[torch.Tensor]
    ],  # [num_output_tokens, num_layers, (num_q_heads * head_dim)]
    req_prompt_token_indices: list[int],  # indices of prompt tokens in the KV cache
    page_size: int,
    chunked_attention_compute_size: Optional[int],
) -> list[list[torch.Tensor]]:
    assert page_size == 1, "Implemented only for page_size == 1"
    assert len(key_cache_buffer) == len(req_query_buffer[0]), (
        "Expecting same number of layers."
    )

    num_layers = len(key_cache_buffer)
    num_output_tokens = len(req_query_buffer)
    num_prompt_tokens = len(req_prompt_token_indices)

    layers_attn_weights_per_token: list[list[torch.Tensor]] | None = None

    for layer_id in range(num_layers):
        # Prepare Keys [num_prompt_tokens, num_k_heads, head_dim]
        keys_base = key_cache_buffer[layer_id][req_prompt_token_indices, :, :]
        num_k_heads, head_dim = keys_base.shape[-2:]

        # Reconstruct Queries [num_output_tokens, num_q_heads, head_dim]
        query_last_dimension = req_query_buffer[0][layer_id].shape[-1]
        num_q_heads = query_last_dimension // head_dim

        all_querys = torch.concat(
            [
                query[layer_id].reshape(1, num_q_heads, head_dim)
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
        chunk_size = chunked_attention_compute_size or num_output_tokens

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
