import torch
import triton
import triton.language as tl
import math

def pattern(dividend, divisor):
    # The pattern: unsqueeze dividend and broadcast division
    dividend_reshaped = dividend.unsqueeze(-1)
    result = dividend_reshaped / divisor
    return result

def replacement_args(dividend, divisor):
    return (dividend, divisor)

@triton.jit
def broadcast_div_kernel(
    dividend_ptr, divisor_ptr,
    output_ptr,
    m_size, n_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Program grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this block
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for boundary conditions
    m_mask = m_range < m_size
    n_mask = n_range < n_size
    
    # Create 2D indices
    m_idx = m_range[:, None]
    n_idx = n_range[None, :]
    
    # Load dividend (broadcasted over columns)
    dividend_mask = m_mask[:, None] & n_mask[None, :]
    dividend_val = tl.load(dividend_ptr + m_idx, mask=m_mask[:, None])
    
    # Load divisor (broadcasted over rows)  
    divisor_val = tl.load(divisor_ptr + n_idx, mask=n_mask[None, :])
    
    # Perform division
    result = dividend_val / divisor_val
    
    # Store result
    output_idx = m_idx * n_size + n_idx
    tl.store(output_ptr + output_idx, result, mask=dividend_mask)

@torch.fx.wrap
def optimized_broadcast_div(dividend, divisor):
    # Get original sizes
    m_size = dividend.shape[0]
    n_size = divisor.shape[0]
    
    # Calculate output shape
    output_shape = (m_size, n_size)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=dividend.dtype, device=dividend.device)
    
    # Optimal block sizes for GPU
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    
    # Calculate grid dimensions
    grid_m = (m_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    broadcast_div_kernel[(grid_m, grid_n)](
        dividend, divisor, output,
        m_size, n_size,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return optimized_broadcast_div