import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_embedding_add_layernorm_kernel(
    input_embeds_ptr, token_type_embeds_ptr, position_embeds_ptr, position_ids_ptr,
    layernorm_weight_ptr, layernorm_bias_ptr,
    output_ptr,
    seq_len, hidden_size, embedding_dim,
    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for:
    tmp_4 = in_4 + in_6  # input_embeds + token_type_embeds
    tmp_5 = embedding(in_5, position_embeds)  # position embedding lookup
    tmp_4 = tmp_4 + tmp_5  # add position embedding
    tmp_7 = layer_norm(tmp_4, normalized_shape, weight, bias, eps)
    """
    # Each program processes one position in the sequence
    row_idx = tl.program_id(0)
    
    # Compute the offset for this row
    row_offset = row_idx * hidden_size
    
    # Load position_ids for this row
    position_id = tl.load(position_ids_ptr + row_idx).to(tl.int64)
    
    # Load and add input_embeds + token_type_embeds
    # Then add position embedding (gathered by position_id)
    sum_vals = tl.zeros((BLOCK_SIZE,), tl.float32)
    
    # Process in blocks
    for col_offset in range(0, hidden_size, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < hidden_size
        
        # Load input_embeds[row, col]
        input_offset = row_offset + col_offsets
        input_val = tl.load(input_embeds_ptr + input_offset, mask=mask, other=0.0)
        
        # Load token_type_embeds[0, col] (broadcast over batch)
        token_type_offset = col_offsets
        token_type_val = tl.load(token_type_embeds_ptr + token_type_offset, mask=mask, other=0.0)
        
        # Load position_embeds[position_id, col]
        position_embed_offset = position_id * embedding_dim + col_offsets
        position_val = tl.load(position_embeds_ptr + position_embed_offset, mask=mask, other=0.0)
        
        # Sum: input + token_type + position
        sum_vals = input_val + token_type_val + position_val
        
        # Compute layer norm statistics for this block
        # First, compute mean
        col_sum = tl.sum(sum_vals, axis=0)
        col_mean = col_sum / hidden_size
        
        # Compute variance
        col_diff = sum_vals - col_mean
        col_var = tl.sum(col_diff * col_diff, axis=0) / hidden_size
        col_std = tl.sqrt(col_var + eps)
        
        # Normalize
        # Load layernorm weight and bias
        ln_weight = tl.load(layernorm_weight_ptr + col_offsets, mask=mask, other=1.0)
        ln_bias = tl.load(layernorm_bias_ptr + col_offsets, mask=mask, other=0.0)
        
        normalized = (sum_vals - col_mean) / col_std * ln_weight + ln_bias
        
        # Store result
        output_offset = row_offset + col_offsets
        tl.store(output_ptr + output_offset, normalized, mask=mask)


@torch.fx.wrap
def fused_embedding_add_layernorm(
    input_embeds, token_type_embeds, position_embeds, position_ids,
    layernorm_weight, layernorm_bias, eps
):
    """
    Fused function that combines:
    1. input_embeds + token_type_embeds
    2. + position_embeddings[position_ids]
    3. LayerNorm
    """
    batch_size, seq_len, hidden_size = input_embeds.shape
    
    # position_embeds shape: [max_position, hidden_size]
    # We need to handle the case where hidden_size might differ from embedding_dim
    embedding_dim = position_embeds.shape[1]
    
    output = torch.empty_like(input_embeds)
    
    # Launch kernel - one program per sequence position
    grid = (batch_size * seq_len,)
    
    fused_embedding_add_layernorm_kernel[grid](
        input_embeds, token_type_embeds, position_embeds, position_ids,
        layernorm_weight, layernorm_bias,
        output,
        seq_len, hidden_size, embedding_dim,
        eps,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the pattern:
    tmp_4 = in_4 + in_6  # inputs_embeds + token_type_embeddings
    tmp_5 = torch.nn.functional.embedding(in_5, tmp_3, ...)
    tmp_4 += tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_4, ..., tmp_2, tmp_1, eps)
    tmp_8 = torch.nn.functional.dropout(tmp_7, ...)
    
    Returns both tmp_8 (the normalized output) and tmp_10 (the expand output)
    """
    tmp_4 = in_4 + in_6
    tmp_5 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_4 = tmp_4 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_4, (32,), in_2, in_1, 1e-12)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.1, False, False)
    tmp_9 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_10 = tmp_9.expand(1, 1, 64, 64)
    return tmp_8, tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Extract arguments for replacement.
    - in_0: attention_mask (for expand)
    - in_1: layernorm bias
    - in_2: layernorm weight
    - in_3: position_embeddings
    - in_4: input_embeds
    - in_5: position_ids
    - in_6: token_type_embeddings
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return fused_embedding_add_layernorm