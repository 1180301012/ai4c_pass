import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern: adaptive_avg_pool2d followed by flatten
    tmp_pool = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    result = tmp_pool.flatten(1, -1)
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def fused_pool_flatten_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for adaptive_avg_pool2d(1) + flatten(1, -1)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C
    
    # Each output element corresponds to average of one HxW spatial region
    # For adaptive_avg_pool2d(1), we're averaging entire spatial dimensions
    total_elements_per_channel = H * W
    
    # Calculate (N, C) indices from flattened output
    c = offsets % C
    n = offsets // C
    
    # Load all spatial elements for this channel and batch
    spatial_offsets = n * C * H * W + c * H * W + tl.arange(0, H * W)
    spatial_mask = spatial_offsets < N * C * H * W
    
    # Load spatial data and sum
    spatial_x = tl.load(x_ptr + spatial_offsets, mask=spatial_mask, other=0.0)
    sum_val = tl.sum(spatial_x)
    count = tl.sum(spatial_mask.to(tl.float32))
    
    # Compute average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    tl.store(out_ptr + offsets, avg_val, mask=mask)

@torch.fx.wrap
def fused_pool_flatten(x):
    """Fused adaptive_avg_pool2d(1) + flatten operation"""
    if x.numel() == 0:
        return x
    
    N, C, H, W = x.shape
    out_elements = N * C
    
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_pool_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_pool_flatten