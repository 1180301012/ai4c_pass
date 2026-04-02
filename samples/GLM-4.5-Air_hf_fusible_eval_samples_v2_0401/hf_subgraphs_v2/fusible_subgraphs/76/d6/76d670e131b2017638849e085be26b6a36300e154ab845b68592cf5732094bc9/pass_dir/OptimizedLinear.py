import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    n_rows,
    n_cols,
    n_features_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for linear operation with better memory access"""
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for bounds checking
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_features_out
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Better loop ordering: loop over K dimension with configurable blocks for better cache utilization
    for k in range(0, n_cols, BLOCK_SIZE_K):  # Configurable K stride for better cache performance
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < n_cols
        
        # Load input data with coalesced memory access
        x_local = tl.load(x_ptr + row_offsets[:, None] * n_cols + k_offsets[None, :], 
                         mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight data with stride optimization
        weight_local = tl.load(weight_ptr + k_offsets[:, None] * n_features_out + col_offsets[None, :], 
                              mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        # Optimized matrix multiply
        x_fp32 = x_local.to(tl.float32)
        weight_fp32 = weight_local.to(tl.float32)
        accumulator += tl.dot(x_fp32, weight_fp32)
    
    # Load bias with vectorized access
    bias_local = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    accumulator += bias_local[None, :]
    
    # Store with stride optimization
    out_ptrs = out_ptr + row_offsets[:, None] * n_features_out + col_offsets[None, :]
    tl.store(out_ptrs, accumulator, mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """Wrapper function that launches the optimized linear kernel"""
    
    # Linear operation setup
    n_rows, n_cols = x.shape
    n_features_out = weight.shape[0]
    
    # Optimized block sizes for matrix multiplication
    BLOCK_SIZE_M = 16   # Work per row
    BLOCK_SIZE_N = 64   # Work per column
    BLOCK_SIZE_K = 128  # Work for accumulation loop
    
    # Calculate grid size
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_features_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Output tensor
    out = torch.empty((n_rows, n_features_out), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    linear_kernel[(grid_m, grid_n)](
        x, weight, bias, out,
        n_rows, n_cols, n_features_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

def pattern(in_6, in_5, in_4):
    """
    Pattern matching for linear operation: torch.nn.functional.linear(in_6, in_5, in_4)
    """
    return torch.nn.functional.linear(in_6, in_5, in_4)

def replacement_args(in_6, in_5, in_4):
    """Extract arguments needed for the replacement"""
    return (in_6, in_5, in_4)

def replacement_func():
    """Return the replacement function"""
    return optimized_linear