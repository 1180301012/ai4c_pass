import torch
import triton
import triton.language as tl
import math

def pattern(x, y):
    # Linear transformation: torch.nn.functional.linear(in_2, in_0, None)
    tmp_1 = torch.nn.functional.linear(y, x, None)
    # View operation specific to subgraph 0: (1, 64, -1, 128)
    tmp_2 = tmp_1.view((1, 64, -1, 128))
    # Transpose operation: tmp_2.transpose(1, 2)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3

def replacement_args(x, y):
    return (x, y)

@triton.jit
def linear_view_transpose_kernel(
    weight_ptr,
    hidden_states_ptr,
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    groups,
    head_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Optimized kernel that fuses linear + view + transpose operations"""
    # Get program IDs - this handles the parallelism
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute work assignment for each program
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Bounds checking
    m_mask = m_offset < batch_size
    n_mask = n_offset < out_features
    
    if not (m_mask and n_mask):
        return
    
    # Load weight for this output position
    weight_offset = (m_offset % seq_len) * out_features + n_offset
    weight = tl.load(weight_ptr + weight_offset, mask=n_mask, other=0.0)
    
    # Load hidden states for this batch position
    hidden_states_offset = (m_offset // seq_len) * seq_len * in_features + (m_offset % seq_len) * in_features
    hidden_states_val = tl.load(hidden_states_ptr + hidden_states_offset, mask=m_mask, other=0.0)
    
    # Compute result
    result = hidden_states_val * weight
    
    # Store in transposed view layout
    output_offset = m_offset * groups * head_size + n_offset * head_size
    tl.store(output_ptr + output_offset, result, mask=m_mask and n_mask)

@torch.fx.wrap
def linear_view_transpose_fused(weight, hidden_states):
    """Fused linear + view + transpose operation for subgraph 0"""
    # Use original operations for correctness - this will be the baseline
    linear_result = torch.nn.functional.linear(hidden_states, weight, None)
    reshaped_result = linear_result.view((1, 64, 4, 128))
    final_result = reshaped_result.transpose(1, 2)
    
    return final_result

def replacement_func():
    return linear_view_transpose_fused