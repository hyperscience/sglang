from typing import Optional

import numpy as np
import torch

GB = 1024**3


def get_query_buffer_gb(
    query_buffer: list[
        list[torch.Tensor]
    ],  # [num_output_tokens, num_layers, 1, (num_q_heads * head_dim)]
) -> float:
    if not query_buffer:
        return 0.0

    # query_buffer[token_idx][layer_idx] is a torch.Tensor
    # We count elements in one token's worth of layers
    total_elements = 0
    for layer_tensor in query_buffer[0]:
        total_elements += layer_tensor.numel()

    # Multiply by number of output tokens
    total_elements *= len(query_buffer)

    # Get bytes per element (e.g., 2 for float16)
    bytes_per_element = query_buffer[0][0].element_size()

    # Convert to Gigabytes (1024^3)
    return (total_elements * bytes_per_element) / GB


def compute_attn_weights(
    key_cache_buffer: list[
        torch.Tensor
    ],  # [num_layers, context_len, num_k_heads, head_dim]
    query_buffer: list[
        list[torch.Tensor]
    ],  # [num_output_tokens, num_layers, 1, (num_q_heads * head_dim)]
    prompt_token_indices: list[int],
    page_size: int,
    chunked_attention_compute_size: Optional[int],
) -> list[list[torch.Tensor]]:
    assert page_size == 1, "Implemented only for page_size == 1"
    assert len(key_cache_buffer) == len(query_buffer[0]), (
        "Expecting same number of layers."
    )

    num_layers = len(key_cache_buffer)
    num_output_tokens = len(query_buffer)
    num_prompt_tokens = len(prompt_token_indices)

    layers_attn_weights_per_token: list[list[torch.Tensor]] | None = None

    for layer_id in range(num_layers):
        # Prepare Keys [num_prompt_tokens, num_k_heads, head_dim]
        keys_base = key_cache_buffer[layer_id][prompt_token_indices, :, :]
        num_k_heads, head_dim = keys_base.shape[-2:]

        # Reconstruct Queries [num_output_tokens, num_q_heads, head_dim]
        query_last_dimension = query_buffer[0][layer_id].shape[-1]
        num_q_heads = query_last_dimension // head_dim

        all_querys = torch.concat(
            [
                query[layer_id].reshape(1, num_q_heads, head_dim)
                for query in query_buffer
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
                # The attention to the first token is called null attetion
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
