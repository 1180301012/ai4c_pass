import torch
import triton
import triton.language as tl

def pattern(x, dim, keepdim):
    # Pattern: mean reduction over spatial dimensions with keepdim=True
    return x.mean(dim, keepdim=keepdim)

def replacement_args(x, dim, keepdim):
    return (x,)

@triton.jit
def optimized_mean_kernel(
    x_ptr, out_ptr, 
    N, C, H, W,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Process this batch and channel
    n = pid_n
    c = pid_c
    
    # Initialize sum for this channel
    channel_sum = tl.zeros([], dtype=tl.float32)
    count = 0
    
    # Process all spatial positions
    for hw in range(0, H * W, BLOCK_SIZE_HW):
        # Load block of spatial data
        hw_end = min(hw + BLOCK_SIZE_HW, H * W)
        
        # Sum spatial values
        for hw_idx in range(hw, hw_end):
            x_idx = n * C * H * W + c * H * W + hw_idx
            val = tl.load(x_ptr + x_idx, mask=True, other=0.0)
            channel_sum += val
            count += 1
    
    # Compute mean (divide by H*W)
    if count > 0:
        channel_mean = channel_sum / float(H * W)
    else:
        channel_mean = 0.0
    
    # Store result at spatial position [0,0] since keepdim=True
    out_idx = n * C * 1 * 1 + c * 1 * 1 + 0  # H=1, W=1 for keepdim output
    tl.store(out_ptr + out_idx, channel_mean)

def optimized_mean_wrapper(x, dim=(2,3), keepdim=True):
    N, C, H, W = x.shape
    
    # Output shape depends on keepdim
    if keepdim:
        out_shape = (N, C, 1, 1)
    else:
        out_shape = (N, C)
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Optimize block size based on spatial dimensions
    BLOCK_SIZE_HW = min(1024, H * W)
    
    # Calculate grid - one program per batch and channel
    grid_n = N
    grid_c = C
    
    # Launch mean reduction kernel
    optimized_mean_kernel[(
        grid_n,
        grid_c
    )](
        x_ptr=x,
        out_ptr=out.view(-1),  # Flatten for easier indexing
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

@torch.fx.wrap  
def triton_mean(x, dim=(2,3), keepdim=True):
    return optimized_mean_wrapper(x, dim, keepdim)

def replacement_func():
    return triton_mean