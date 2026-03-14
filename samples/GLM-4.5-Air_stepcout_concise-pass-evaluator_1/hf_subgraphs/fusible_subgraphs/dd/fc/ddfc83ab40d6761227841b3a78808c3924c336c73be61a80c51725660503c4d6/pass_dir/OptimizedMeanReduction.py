import torch
import triton
import triton.language as tl

def pattern(x):
    """Mean reduction pattern over spatial dimensions"""
    return x.mean((2, 3), keepdim=True)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_mean_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one channel in one batch element for mean reduction
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C)
    
    # Reshape to batch and channel dimensions
    offset_n = offsets % N
    offset_c = offsets // N
    
    # Initialize sum for each batch and channel
    if pid == 0:
        # Initialize mean accumulator
        for i in range(N * C):
            tl.store(out_ptr + i, 0.0)
    
    # Each thread processes one spatial element
    # We need parallel reduction over H x W elements per (N, C)
    
    for hw_offset in range(H * W):
        hw_idx = offset_n * C * H * W + offset_c * H * W + hw_offset
        if hw_idx < N * C * H * W:
            # Load element
            val = tl.load(x_ptr + hw_idx, other=0.0)
            # Atomic add to accumulator
            tl.atomic_add(out_ptr + offset_n * C + offset_c, val)

@torch.fx.wrap  
def optimized_mean(x):
    """Optimized mean reduction with parallel reduction"""
    N, C, H, W = x.shape
    
    # Output is [N, C, 1, 1]
    mean_out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    
    # Flatten mean for easier atomic operations
    mean_flat = mean_out.squeeze()  # [N, C]
    
    total_spatial_elements = H * W
    
    BLOCK_SIZE = 256
    # Grid is over N * C (batch * channels)  
    num_programs = (N * C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_mean_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=mean_flat,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Divide sum by number of elements to get mean
    mean_flat = mean_flat / (H * W)
    
    # Reshape back to [N, C, 1, 1]
    mean_out = mean_flat.reshape(N, C, 1, 1)
    
    return mean_out

def replacement_func():
    return optimized_mean