import torch
from torch import device
import triton
import triton.language as tl


# Pattern: Match exact computation from forward()
def pattern(in_0, in_1, in_2, in_3):
    # Must match exactly what's in model.py
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda'))
    tmp_4 = tmp_3.expand(1, -1)
    tmp_5 = tmp_4 + 2
    tmp_6 = torch.nn.functional.embedding(tmp_5, tmp_0, None, None, 2.0, False, False)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), tmp_2, tmp_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, p=0.1, training=False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized Triton kernel: fused embedding lookup + add + layer_norm
@triton.jit
def fused_embedding_layernorm_kernel(
    embedding_ptr,      # embedding table [514, 1024]
    input_ptr,          # input [1, 1, 1024]
    weight_ptr,         # layernorm weight [1024]
    bias_ptr,           # layernorm bias [1024]
    output_ptr,         # output [1, 1, 1024]
    embedding_row_idx: tl.constexpr,  # Fixed index 2
    norm_dim: tl.constexpr,           # 1024
    eps: tl.constexpr,                # 1e-05
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the embedding (1024 elements)
    row_offset = tl.program_id(0) * BLOCK_SIZE
    col_offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < norm_dim
    
    # Load embedding row at index 2
    embedding_base = embedding_ptr + embedding_row_idx * norm_dim
    embedding_vals = tl.load(embedding_base + col_offsets, mask=mask, other=0.0)
    
    # Load input
    input_vals = tl.load(input_ptr + col_offsets, mask=mask, other=0.0)
    
    # Add: input + embedding
    added_vals = input_vals + embedding_vals
    
    # Store to compute mean and variance
    temp_storage = tl.cast(added_vals, tl.float32)
    
    # Compute mean
    sum_vals = tl.sum(temp_storage, axis=0) / norm_dim
    mean = sum_vals
    
    # Compute variance
    diff = temp_storage - mean
    variance = tl.sum(diff * diff, axis=0) / norm_dim
    std = tl.sqrt(variance + eps)
    
    # Normalize
    # Load weight and bias
    weight_vals = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    normalized = (temp_storage - mean) / std * weight_vals + bias_vals
    
    # Store output (dropout with training=False is no-op)
    tl.store(output_ptr + col_offsets, normalized, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(embedding_table, layernorm_bias, layernorm_weight, input_embeds):
    # Ensure inputs are on cuda
    embedding_table = embedding_table.cuda()
    input_embeds = input_embeds.cuda()
    
    # Output shape: [1, 1, 1024]
    output = torch.empty_like(input_embeds)
    
    # Launch kernel - each program handles one row (1024 elements)
    BLOCK_SIZE = 1024
    norm_dim = 1024
    embedding_row_idx = 2  # Fixed index from the pattern
    eps = 1e-05
    
    # For [1, 1, 1024], we process the last dimension
    # Grid: 1 program for the sequence dimension, 1 for batch, 1 for embedding
    # Actually we need to process all elements in the last dimension
    num_programs = (norm_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_layernorm_kernel[(num_programs,)](
        embedding_ptr=embedding_table,
        input_ptr=input_embeds,
        weight_ptr=layernorm_weight,
        bias_ptr=layernorm_bias,
        output_ptr=output,
        embedding_row_idx=embedding_row_idx,
        norm_dim=norm_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper