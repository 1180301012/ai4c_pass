import torch
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    n_height,
    BLOCK_SIZE: tl.constexpr,
):
    """L1 normalization kernel that computes sum along last dimension and divides"""
    # Each program handles one channel and one height position
    channel_id = tl.program_id(0)
    height_id = tl.program_id(1)
    
    # Compute the offset for this channel and height
    base_offset = channel_id * n_height + height_id
    
    # Load elements for reduction
    x = tl.load(x_ptr + base_offset * 8, mask=range(BLOCK_SIZE))  # 8 is the width (last dim size)
    
    # Parallel reduction for sum
    partial_sum = tl.sum(x)
    
    # Store the sum (this approach stores in a separate output, but for simplicity we'll do the division in a separate step)
    # Actually, let's do a simpler approach where each thread computes its local sum and then we divide
    start_idx = base_offset * 8
    for i in range(BLOCK_SIZE):
        idx = start_idx + i
        if idx < x_ptr.shape[0] if hasattr(x_ptr, 'shape') else True:
            val = tl.load(x_ptr + idx)
            tl.store(out_ptr + base_offset, val / partial_sum if partial_sum != 0 else 0.0)

@triton.jit
def l1_norm_kernel_single_pass(
    x_ptr,
    out_ptr,
    n_elements,
    n_channels,
    n_height,
    width: tl.constexpr,
):
    """Single-pass L1 normalization with explicit grid management"""
    # Each program handles one channel and one height position
    channel_id = tl.program_id(0)
    height_id = tl.program_id(1)
    
    # Compute the base offset for this channel and height position
    base_offset = (channel_id * n_height + height_id) * width
    
    # Initialize sum for reduction
    slice_sum = 0.0
    
    # Manual reduction for the slice
    for i in range(width):
        idx = base_offset + i
        if idx < n_elements:
            val = tl.load(x_ptr + idx)
            slice_sum += val
    
    # Normalize each element and store
    for i in range(width):
        idx = base_offset + i
        if idx < n_elements:
            x_val = tl.load(x_ptr + idx)
            if slice_sum != 0:
                tl.store(out_ptr + idx, x_val / slice_sum)
            else:
                tl.store(out_ptr + idx, 0.0)

@torch.fx.wrap
def l1_normalization(x):
    """L1 normalization along the last dimension"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor")
    
    n_channels, n_height, width = x.shape[1], x.shape[2], x.shape[3]
    n_elements = x.numel()
    
    out = torch.empty_like(x)
    
    # Grid: (channels, height)
    grid = (n_channels, n_height)
    
    l1_norm_kernel_single_pass[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        n_channels=n_channels,
        n_height=n_height,
        width=width,
    )
    
    return out

def pattern(x):
    """Match: sum(dim=3, keepdim=True) followed by division"""
    tmp_5 = x.sum(dim=3, keepdim=True)
    tmp_6 = x / tmp_5
    return tmp_6

def replacement_args(x):
    """Extract arguments for replacement - just the input tensor"""
    return (x,)

def replacement_func():
    """Return the optimized normalization function"""
    return l1_normalization