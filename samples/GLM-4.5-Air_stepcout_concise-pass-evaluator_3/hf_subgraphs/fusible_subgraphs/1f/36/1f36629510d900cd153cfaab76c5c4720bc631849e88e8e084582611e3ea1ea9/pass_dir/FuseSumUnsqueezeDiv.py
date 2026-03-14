import torch
import triton
import triton.language as tl

def pattern(x, sum_val):
    # Pattern: divide input by a sum value that was computed by summing along last dimension
    # This matches the final operation: in_0 / tmp_1 where tmp_1 was the sum result
    result = x / sum_val
    return result

def replacement_args(x, sum_val):
    return (x, sum_val)

@triton.jit
def fused_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Fused kernel that performs sum along last dimension, unsqueeze, and division
    efficiently using Triton.
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate grid boundaries
    num_m = x_ptr.shape[0] * x_ptr.shape[1] * x_ptr.shape[2]  # 1 * 16 * 196
    num_n = x_ptr.shape[3]  # 196
    
    # Calculate offsets
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create masks
    m_mask = m_offset + tl.arange(0, BLOCK_SIZE_M) < num_m
    n_mask = n_offset + tl.range(0, BLOCK_SIZE_N) < num_n
    
    # Reshape for processing - treat as (num_m, num_n) where num_m = 1*16*196, num_n = 196
    x_flat = x_ptr.reshape((num_m, num_n))
    out_flat = out_ptr.reshape((num_m, num_n))
    
    # Load input data for this block
    x_block = x_flat[m_offset:m_offset + BLOCK_SIZE_M, n_offset:n_offset + BLOCK_SIZE_N]
    
    # Compute sum along last dimension (dim=-1, which is dimension 1 in reshaped view)
    sum_result = tl.sum(x_block, axis=1, keepdim=True)
    
    # Perform division (broadcast sum across all columns)
    out_block = x_block / sum_result
    
    # Store result
    out_flat[m_offset:m_offset + BLOCK_SIZE_M, n_offset:n_offset + BLOCK_SIZE_N] = out_block

@torch.fx.wrap
def fused_sum_unsqueeze_div(x):
    """
    Fused operation: sum along last dimension, unsqueeze, and divide
    """
    # Get input shape and total elements
    orig_shape = x.shape  # [1, 16, 196, 196]
    N = x.numel()
    
    # Set block sizes for good performance
    BLOCK_SIZE_M = 64   # Process multiple combined dimensions
    BLOCK_SIZE_N = 32   # Process last dimension
    
    # Calculate grid dimensions
    num_m = orig_shape[0] * orig_shape[1] * orig_shape[2]  # 1 * 16 * 196 = 3136
    num_n = orig_shape[3]  # 196
    
    grid_m = (num_m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (num_n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch fused kernel
    fused_kernel[(grid_m, grid_n)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_sum_unsqueeze_div