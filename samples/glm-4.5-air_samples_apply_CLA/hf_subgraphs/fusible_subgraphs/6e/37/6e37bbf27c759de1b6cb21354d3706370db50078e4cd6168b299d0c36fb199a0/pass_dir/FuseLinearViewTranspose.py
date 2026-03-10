import torch
import triton
import triton.language as tl
import math

def pattern(x, y):
    # Linear transformation: torch.nn.functional.linear(in_2, in_0, None)
    tmp_1 = torch.nn.functional.linear(y, x, None)
    # View operation: this varies between subgraphs but we'll match the generic pattern
    # The specific view shape will be handled at runtime
    tmp_2 = tmp_1.view((64, 128, -1, 128))  # Base pattern, will be customized
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
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute block bounds
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    m_end = min(m_start + BLOCK_SIZE_M, batch_size)
    n_end = min(n_start + BLOCK_SIZE_N, out_features)
    k_end = min(k_start + BLOCK_SIZE_K, in_features)
    
    # Shared memory for accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matrix multiplication
    for k in range(k_start, k_end):
        # Load weight slice
        weight_offset = k * out_features + n_start
        weight = tl.load(weight_ptr + weight_offset, mask=(n_start < out_features), other=0.0)
        
        # Load hidden states slice
        for m in range(m_start, m_end):
            for n in range(n_start, n_end):
                hidden_states_offset = m * seq_len * in_features + k * seq_len
                hidden_states_val = tl.load(hidden_states_ptr + hidden_states_offset, mask=(hidden_states_offset < batch_size * seq_len * in_features), other=0.0)
                
                # Accumulate
                accumulator[m - m_start, n - n_start] += hidden_states_val * weight
    
    # Store result in transposed view layout (batch, seq, groups, head_size)
    seq_len_calc = in_features // out_features  # This matches the view operation logic
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            # Calculate final storage address with view+transpose semantics
            # Original: view((4, 512, -1, 128)) then transpose(1, 2)
            # This becomes: (4, -1, 512, 128) after transpose
            output_offset = m * seq_len_calc * out_features * seq_len_calc + n * seq_len_calc * seq_len_calc + m
            tl.store(output_ptr + output_offset, accumulator[m - m_start, n - n_start])

@torch.fx.wrap
def linear_view_transpose_fused(weight, hidden_states):
    """Fused linear + view + transpose operation"""
    _, _, in_features = hidden_states.shape
    out_features = weight.shape[0]
    
    # Determine view pattern based on input dimensions
    # Different subgraphs have different view patterns:
    # Subgraph 0: (1, 64, -1, 128) -> hidden_states shape [1, 64, 2048]
    # Subgraph 5: (4, 512, -1, 128) -> hidden_states shape [4, 512, 2048]  
    # Subgraph 7: (64, 128, -1, 128) -> hidden_states shape [64, 128, 2048]
    
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]  # This varies: 64, 512, 128
    
    # Calculate view parameters
    head_size = 128  # This is consistent across all subgraphs
    groups = in_features // seq_len  # This gives groups dimension
    
    # Output shape after transpose: (batch_size, groups, seq_len, head_size) -> transpose to (batch_size, groups, seq_len, head_size)
    output_shape = (batch_size, groups, seq_len, head_size)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128  
    BLOCK_SIZE_K = 64
    
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_k = (in_features + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (num_blocks_m, num_blocks_n, num_blocks_k)
    
    linear_view_transpose_kernel[grid](
        weight_ptr=weight,
        hidden_states_ptr=hidden_states,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        groups=groups,
        head_size=head_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return linear_view_transpose_fused