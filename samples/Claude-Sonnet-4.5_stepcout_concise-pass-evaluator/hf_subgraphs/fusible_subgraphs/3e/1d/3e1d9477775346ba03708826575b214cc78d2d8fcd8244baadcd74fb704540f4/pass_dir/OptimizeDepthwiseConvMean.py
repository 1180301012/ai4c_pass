import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Match the spatial mean pattern.
    """
    result = x.mean((2, 3), keepdim=True)
    return result

def replacement_args(x):
    """
    Extract arguments needed for the replacement function.
    """
    return (x,)

@triton.jit
def spatial_mean_kernel_optimized(
    input_ptr,
    output_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized spatial mean kernel for 56x56 spatial size.
    """
    bc_idx = tl.program_id(0)
    spatial_size = H * W
    base_offset = bc_idx * spatial_size
    
    # Unrolled reduction for better performance
    sum_val = 0.0
    
    num_iters = tl.cdiv(spatial_size, BLOCK_SIZE)
    for i in range(num_iters):
        start_idx = i * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Store mean
    tl.store(output_ptr + bc_idx, sum_val / spatial_size)

@torch.fx.wrap
def optimized_spatial_mean(x):
    """
    Optimized spatial mean with minimal overhead.
    """
    B, C, H, W = x.shape
    
    # Allocate output
    mean_out = torch.empty((B, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Use fixed block size tuned for 56x56
    BLOCK_SIZE = 1024
    grid = (B * C,)
    
    spatial_mean_kernel_optimized[grid](
        x, mean_out, B, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return mean_out

def replacement_func():
    """
    Return the replacement function (not a call to it).
    """
    return optimized_spatial_mean