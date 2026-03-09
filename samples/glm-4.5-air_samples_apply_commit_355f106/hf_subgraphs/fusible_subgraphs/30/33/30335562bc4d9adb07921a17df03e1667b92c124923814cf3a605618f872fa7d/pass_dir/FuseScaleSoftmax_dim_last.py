import torch
import triton
import triton.language as tl

# Pattern: simple scalar multiplication for testing
def pattern(x):
    return 0.0625 * x

# Arguments for replacement - just need the input tensor
def replacement_args(x):
    return (x,)

# Optimized kernel that fuses scalar multiplication with softmax along last dimension
@triton.jit
def fused_scale_softmax_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_first_dim,
    n_last_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # For softmax along last dimension (dim=-1), we need to handle the tensor as [batch, first_dim, last_dim]
    # Each program handles one "row" across the last dimension for all batches and first_dim positions
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # First dimension (e.g., 8192)
    pid_k_offset = tl.program_id(2) * BLOCK_SIZE_N  # Within last dimension (e.g., 19)
    
    # Calculate pointers for this work item
    batch_offset = pid_m * n_first_dim * n_last_dim
    first_dim_offset = pid_n * n_last_dim
    k_offset = pid_k_offset
    
    # Create range for the block in the last dimension
    k_range = k_offset + tl.arange(0, BLOCK_SIZE_N)
    k_mask = k_range < n_last_dim
    
    # Load the entire row across last dimension for current batch and first_dim position
    x_ptrs = (batch_offset + first_dim_offset + k_range).to(tl.int64)
    x = tl.load(x_ptr + x_ptrs, mask=k_mask, other=-float('inf'))
    
    # Apply scaling
    scaled_x = 0.0625 * x
    
    # Compute max for numerical stability (needed for softmax)
    max_val = tl.max(scaled_x)
    
    # Compute exponential and sum for softmax
    exp_x = tl.exp(scaled_x - max_val)
    # For sum with mask in Triton, we need to handle it differently
    sum_exp = tl.sum(exp_x * k_mask.to(tl.float32))
    
    # Compute softmax
    softmax_val = exp_x / (sum_exp + (sum_exp == 0).to(tl.float32))
    
    # Store result
    out_ptrs = (batch_offset + first_dim_offset + k_range).to(tl.int64)
    tl.store(out_ptr + out_ptrs, softmax_val * k_mask.to(tl.float32), mask=k_mask)

@torch.fx.wrap
def fused_scale_softmax(x):
    # Get tensor dimensions - shape is [batch, first_dim, last_dim]
    n_batch = x.shape[0]
    n_first_dim = x.shape[1]
    n_last_dim = x.shape[2]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 1      # We'll process one batch at a time
    BLOCK_SIZE_N = 64     # Block size for the last dimension (softmax dim)
    
    # Calculate grid dimensions
    # Grid: (batch, first_dim, chunks_of_last_dim)
    grid = (
        n_batch,
        n_first_dim,
        (n_last_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    
    # Launch kernel
    fused_scale_softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        n_first_dim=n_first_dim,
        n_last_dim=n_last_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_scale_softmax