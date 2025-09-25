import numpy as np
import torch


def compute_attn_weights(
    key_cache_buffer: list[torch.Tensor],  # [num_layers, context_len, num_k_heads, head_dim]
    query_buffer: list[list[torch.Tensor]],  # [num_output_tokens, num_layers, 1, (num_q_heads * head_dim)]
    prompt_token_indices: list[int],
    page_size: int,
) -> list[list[torch.Tensor]]:
    assert page_size == 1, "Implemented only for page_size == 1"
    assert len(key_cache_buffer) == len(
        query_buffer[0]
    ), "Expecting same number of layers."
    num_layers = len(key_cache_buffer)

    layers_attn_weights_per_token: list[list[torch.Tensor]] | None = None
    for layer_id in range(num_layers):
        # keys in [num_prompt_tokens, num_k_heads, head_dim]
        keys = key_cache_buffer[layer_id][prompt_token_indices, :, :]
        num_k_heads, head_dim = keys.shape[-2:]

        query_last_dimension = query_buffer[0][layer_id].shape[-1]
        num_q_heads = query_last_dimension // head_dim

        assert num_q_heads * head_dim == query_last_dimension

        # querys in [num_output_tokens, num_q_heads, head_dim]
        querys = torch.concat(
            [
                query[layer_id].reshape(1, num_q_heads, head_dim)
                for query in query_buffer
            ],
            axis=0,
        )

        repeat_factor = num_q_heads // num_k_heads
        keys = keys.repeat_interleave(repeat_factor, dim=1)
        keys = keys.to(torch.float32)
        querys = querys.to(torch.float32)

        # reshape querys to [num_q_heads, num_output_tokens, head_dim]
        # reshape keys to [num_q_heads, head_dim, num_prompt_tokens]
        # scroes will be in [num_q_heads, num_output_tokens, num_prompt_tokens]
        scores = torch.bmm(querys.transpose(0, 1), keys.permute(1, 2, 0)) / (
            head_dim**0.5
        )
        scores = torch.softmax(scores, dim=-1)
        # average over num_q_heads which generate attn_weights in
        # [num_output_tokens, num_prompt_tokens]
        # Move it to cpu to save gpu memory
        scores = scores.mean(dim=0).half().cpu()

        # layers_attn_weights_per_token is a list of lists, where each element contains the
        # attention weights for a specific token across all layers
        if layers_attn_weights_per_token is None:
            layers_attn_weights_per_token = [[token_attn] for token_attn in scores]
        else:
            for i, token_attn in enumerate(scores):
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
