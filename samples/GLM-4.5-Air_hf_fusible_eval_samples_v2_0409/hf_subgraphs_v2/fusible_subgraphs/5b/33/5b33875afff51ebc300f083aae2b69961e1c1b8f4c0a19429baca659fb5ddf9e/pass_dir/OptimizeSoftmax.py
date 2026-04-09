import torch
import triton
import triton.language as tl
import math

@triton.jit
def optimized_softmax_kernel(
    input_ptr, output_ptr,
    rows, cols,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for this program
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create masks for valid indices
    mask = (m_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] < rows
    n_mask = (n_offset + tl.arange(0, BLOCK_SIZE_N)) < cols
    mask = mask & n_mask[:, None]
    
    # Load input data
    input_data = tl.load(
        input_ptr + (m_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] * cols + (n_offset + tl.arange(0, BLOCK_SIZE_N))[None, :],
        mask=mask,
        other=-float('inf')
    )
    
    # Compute max along the last dimension
    max_val = tl.maximum(input_data, tl.zeros_like(input_data))
    if cols > 32:  # For larger dimensions, use a multi-step reduction
        # First reduce within blocks
        max_val = tl.maximum(max_val, tl.max(max_val, axis=1))
        max_val = tl.max(max_val, axis=1)
        # Broadcast max values back
        max_val = max_val[:, None] * tl.ones((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=max_val.dtype)
    else:  # For smaller dimensions, use direct reduction
        max_val = tl.max(input_data, axis=1, keepdim=True)
    
    # Subtract max and exponentiate
    exp_scores = tl.exp(input_data - max_val)
    
    # Sum exponentials
    sum_exp = tl.sum(exp_scores, axis=1, keepdim=True)
    
    # Normalize
    softmax_output = exp_scores / sum_exp
    
    # Store output
    tl.store(
        output_ptr + (m_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] * cols + (n_offset + tl.arange(0, BLOCK_SIZE_N))[None, :],
        softmax_output,
        mask=mask
    )

@torch.fx.wrap
def optimized_softmax(x, dim=-1):
    # Optimized softmax implementation using Triton
    if dim != -1 and dim != x.dim() - 1:
        # Convert to last dimension for easier optimization
        x = x.transpose(dim, -1)
        last_dim = True
    else:
        last_dim = False
    
    rows, cols = x.shape
    
    # Choose optimal block sizes based on tensor dimensions
    BLOCK_SIZE_M = min(128, rows)
    BLOCK_SIZE_N = min(128, cols)
    
    # Calculate grid size
    grid_m = (rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = max(1, (cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    if grid_n == 1:
        optimized_softmax_kernel[(grid_m,)](
            x, output,
            rows, cols,
            BLOCK_SIZE_M, cols
        )
    else:
        optimized_softmax_kernel[(grid_m, grid_n)](
            x, output,
            rows, cols,
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    # Transpose back if needed
    if last_dim:
        output = output.transpose(-1, dim)
    
    return output

def pattern(x):
    # Simple softmax pattern matches the attention softmax operation
    return torch.nn.functional.softmax(x, dim=-1)

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_softmax