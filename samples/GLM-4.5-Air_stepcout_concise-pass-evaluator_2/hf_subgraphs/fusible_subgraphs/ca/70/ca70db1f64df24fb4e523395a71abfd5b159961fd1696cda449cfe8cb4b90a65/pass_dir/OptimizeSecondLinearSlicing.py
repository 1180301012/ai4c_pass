import torch
import triton
import triton.language as tl

def pattern(x):
    # Second slice pattern - take last 256 elements of dimension 1
    result = x[:, -256:]
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def slice_last_kernel(
    x_ptr, 
    out_ptr,
    n_batch, n_in, n_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs for batch and output dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block boundaries
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create indices
    m_indices = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    n_indices = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    m_mask = m_indices < n_batch
    n_mask = n_indices < min(n_out, 256)  # Only take last 256 elements
    
    # Calculate source indices for last 256 elements
    src_start = max(0, n_in - 256)  # Start index for the last 256 elements
    src_n_indices = src_start + n_indices[None, :]
    
    # Load from the last 256 elements
    x_ptrs = x_ptr + (m_indices[:, None] * n_in + src_n_indices)
    x_vals = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Store to output - output has shape [n_batch, 256]
    out_ptrs = out_ptr + (m_indices[:, None] * 256 + n_indices[None, :])
    tl.store(out_ptrs, x_vals, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def optimized_slice_last(x):
    n_batch, n_in = x.shape
    n_out = 256  # We're slicing to last 256 elements
    
    # Triton parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 256
    
    # Grid size
    grid_m = (n_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty((n_batch, n_out), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    slice_last_kernel[(grid_m, grid_n)](
        x, out, n_batch, n_in, n_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return optimized_slice_last