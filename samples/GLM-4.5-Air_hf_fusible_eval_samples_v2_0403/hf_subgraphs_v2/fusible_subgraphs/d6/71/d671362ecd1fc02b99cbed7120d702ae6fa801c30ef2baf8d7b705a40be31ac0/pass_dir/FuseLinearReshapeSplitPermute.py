import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_2, in_1):
    """
    Pattern matching for the exact computation sequence.
    """
    # Linear transformation as in original
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    
    # Reshape to 4D tensor - using a generic pattern
    tmp_4 = linear.reshape(-1, 49, 8, -1)  # Use -1 for batch dimension
    
    # Split into three components
    split = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = split[0]
    tmp_7 = split[1] 
    tmp_8 = split[2]
    
    # Permute each component  
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    
    return tmp_9, tmp_10, tmp_11

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

@triton.jit
def fused_linear_split_permute_kernel(
    input_ptr, weight_ptr, bias_ptr,
    query_out_ptr, key_out_ptr, value_out_ptr,
    batch_size: tl.constexpr, seq_len: tl.constexpr, 
    n_heads: tl.constexpr, d_head_qk: tl.constexpr, d_head_v: tl.constexpr,
    feat_total: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for Linear + Reshape + Split + Permute operations.
    Processes Q, K, V components efficiently in a single kernel.
    """
    pid = tl.program_id(0)
    n_elements = batch_size * seq_len * n_heads * feat_total
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute indices for each tensor
    batch_idx = offsets // (seq_len * n_heads * feat_total)
    seq_idx = (offsets // (n_heads * feat_total)) % seq_len
    head_idx = (offsets // feat_total) % n_heads
    feat_idx = offsets % feat_total
    
    # Determine which component (Q, K, V) this offset corresponds to
    q_mask = feat_idx < d_head_qk
    k_mask = (feat_idx >= d_head_qk) & (feat_idx < (d_head_qk + d_head_qk))
    v_mask = feat_idx >= (d_head_qk + d_head_qk)
    
    # Load input and compute linear operation
    input_offset = batch_idx * seq_len * feat_total + seq_idx * feat_total + feat_idx
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Linear operation: input @ weight.T + bias
    weight_offset = feat_idx * feat_total + batch_idx * seq_len * n_heads * feat_total
    bias_offset = batch_idx * seq_len * n_heads * feat_total
    
    weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
    
    linear_output = input_val * weight_val + bias_val
    
    # Split and permute operations by storing directly to output locations
    query_offset = batch_idx * n_heads * seq_len * d_head_qk + head_idx * seq_len * d_head_qk + seq_idx * d_head_qk + (feat_idx if feat_idx < d_head_qk else 0)
    key_offset = batch_idx * n_heads * seq_len * d_head_qk + head_idx * seq_len * d_head_qk + seq_idx * d_head_qk + (feat_idx - d_head_qk if q_mask and k_mask else 0)
    value_offset = batch_idx * n_heads * seq_len * d_head_v + head_idx * seq_len * d_head_v + seq_idx * d_head_v + (feat_idx - (d_head_qk * 2) if v_mask else 0)
    
    # Store results directly to permuted output locations
    tl.store(query_out_ptr + query_offset, linear_output, mask=q_mask & mask)
    tl.store(key_out_ptr + key_offset, linear_output, mask=k_mask & mask)
    tl.store(value_out_ptr + value_offset, linear_output, mask=v_mask & mask)

@torch.fx.wrap
def fused_linear_split_permute(in_3, in_2, in_1):
    """
    Wrapper function for the fused linear + reshape + split + permute kernel.
    This fuses multiple operations into a single efficient GPU kernel.
    """
    batch_size, seq_len, _ = linear_input.shape
    n_heads = 8
    d_head_qk = 32
    d_head_v = 128
    feat_total = n_heads * (d_head_qk + d_head_qk + d_head_v)  # 8 * (32 + 32 + 128) = 1536
    
    # Output shapes after permutation
    query_shape = (batch_size, n_heads, seq_len, d_head_qk)
    key_shape = (batch_size, n_heads, seq_len, d_head_qk)
    value_shape = (batch_size, n_heads, seq_len, d_head_v)
    
    # Allocate output tensors
    query_out = torch.empty(query_shape, dtype=linear_input.dtype, device=linear_input.device)
    key_out = torch.empty(key_shape, dtype=linear_input.dtype, device=linear_input.device)
    value_out = torch.empty(value_shape, dtype=linear_input.dtype, device=linear_input.device)
    
    # Calculate optimal block size and grid size
    total_elements = batch_size * seq_len * feat_total
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    fused_linear_split_permute_kernel[grid_size](
        in_3, in_2, in_1,  # linear_input, weight, bias
        query_out, key_out, value_out,
        batch_size, seq_len, n_heads, d_head_qk, d_head_v,
        feat_total, BLOCK_SIZE
    )
    
    return query_out, key_out, value_out

def replacement_func():
    return fused_linear_split_permute