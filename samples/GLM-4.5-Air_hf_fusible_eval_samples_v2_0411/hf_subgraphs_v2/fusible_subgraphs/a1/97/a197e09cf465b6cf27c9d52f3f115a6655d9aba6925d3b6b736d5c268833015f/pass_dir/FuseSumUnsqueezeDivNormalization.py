import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sum operation
def pattern(x):
    tmp_0 = x.sum(dim=-1)
    return tmp_0

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for complete normalization computation
@triton.jit
def fused_norm_kernel(
    x_ptr,
    out_ptr,
    n_channels,       # 16 in our case
    h_dim,           # 196 in our case  
    w_dim,           # 196 in our case
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program identifiers
    pid_m = tl.program_id(0)
    
    # Calculate which height position and channel this program handles
    channel_idx = pid_m // h_dim
    h_pos = pid_m % h_dim
    
    # Step 1: Sum along the width dimension for this specific channel position
    sum_val = 0.0
    for w_offset in range(0, w_dim, BLOCK_SIZE_N):
        mask = w_offset + tl.arange(0, BLOCK_SIZE_N) < w_dim
        x_block = tl.load(
            x_ptr + channel_idx * h_dim * w_dim + h_pos * w_dim + w_offset + tl.arange(0, BLOCK_SIZE_N), 
            mask=mask
        )
        sum_val += tl.sum(x_block)
    
    # Step 2: Compute mean (divide sum by width count)
    # This replicates the computation: result = x / sum_val with broadcasting
    # where sum_val becomes a scalar that gets broadcasted across all positions
    mean_val = sum_val / w_dim
    
    # Step 3: Perform normalization by dividing each element by the sum (with broadcasting)
    for w_offset in range(0, w_dim, BLOCK_SIZE_N):
        mask = w_offset + tl.arange(0, BLOCK_SIZE_N) < w_dim
        x_block = tl.load(
            x_ptr + channel_idx * h_dim * w_dim + h_pos * w_dim + w_offset + tl.arange(0, BLOCK_SIZE_N), 
            mask=mask, other=0.0
        )
        
        # Divide each element by the sum (which is constant for all elements in the row)
        # This replicates the broadcasting behavior of x /= tmp_1
        out_block = x_block / mean_val
        
        # Store the result to output tensor
        tl.store(
            out_ptr + channel_idx * h_dim * w_dim + h_pos * w_dim + w_offset + tl.arange(0, BLOCK_SIZE_N), 
            out_block, 
            mask=mask
        )

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
def fused_normalization(x):
    # Get input dimensions
    batch_size, n_channels, h_dim, w_dim = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Optimized block sizes for better GPU utilization
    BLOCK_SIZE_M = 64   # Number of height positions to process per program
    BLOCK_SIZE_N = 256   # Number of width elements to process per iteration
    
    # Calculate grid size: one program per (channel, height_position)
    total_programs = n_channels * h_dim
    
    # Launch the kernel
    fused_norm_kernel[(total_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_channels=n_channels,
        h_dim=h_dim,
        w_dim=w_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_normalization