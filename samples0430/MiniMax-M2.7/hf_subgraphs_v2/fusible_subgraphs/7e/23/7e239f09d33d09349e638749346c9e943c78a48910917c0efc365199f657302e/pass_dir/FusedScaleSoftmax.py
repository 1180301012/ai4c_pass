import torch
import triton
import triton.language as tl


# Optimized softmax kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def triton_softmax_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for softmax over the last dimension.
    Processes one row per program for better GPU utilization.
    """
    row_idx = tl.program_id(0)
    
    # Boundary check
    if row_idx >= n_rows:
        return
    
    row_offset = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load all values for this row
    x = tl.load(x_ptr + row_offset + col_offsets, mask=mask, other=float('-inf'))
    
    # Compute row-wise max for numerical stability
    row_max = tl.max(x, axis=0)
    x = x - row_max
    x = tl.exp(x)
    
    # Compute sum for normalization
    row_sum = tl.sum(x, axis=0)
    x = x / row_sum
    
    # Store result
    tl.store(output_ptr + row_offset + col_offsets, x, mask=mask)


@torch.fx.wrap
def triton_softmax(x):
    """
    Triton-optimized softmax over the last dimension.
    Uses a 1D grid where each program handles one row.
    """
    shape = x.shape
    n_cols = shape[-1]
    n_rows = 1
    for dim in range(len(shape) - 1):
        n_rows *= shape[dim]
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Use 1D grid for rows - each row is processed by one program
    grid = (n_rows,)
    triton_softmax_kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
    )
    
    return output


def pattern(in_0):
    """
    Match: tensor.softmax(dim=-1)
    This is the entry point for pattern matching.
    """
    return in_0.softmax(dim=-1)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_softmax