import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_5 = torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_pad_2d_kernel(
    x_ptr,
    out_ptr,
    n_batch, n_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements per spatial position
    elements_per_spatial = n_batch * n_channels
    
    # Handle spatial dimensions
    h_idx = tl.program_id(0)
    w_idx = tl.program_id(1)
    
    # Calculate offset for this spatial position
    spatial_offset = h_idx * (width + 1) + w_idx
    
    # Handle elements within this spatial position (batch and channel dimensions)
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = spatial_offset * elements_per_spatial + block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_start + tl.arange(0, BLOCK_SIZE) < elements_per_spatial
    
    # Load input data
    x = tl.load(x_ptr + offsets / elements_per_spatial, mask=mask, other=0.0)
    
    # Determine output position
    if w_idx < width and h_idx < height:
        # Interior pixel - just copy
        tl.store(out_ptr + offsets, x, mask=mask)
    elif w_idx == width and h_idx < height:
        # Right border - pad with zero
        tl.store(out_ptr + offsets, 0.0, mask=mask)
    elif w_idx < width and h_idx == height:
        # Bottom border - pad with zero  
        tl.store(out_ptr + offsets, 0.0, mask=mask)
    else:  # w_idx == width and h_idx == height
        # Bottom-right corner - pad with zero
        tl.store(out_ptr + offsets, 0.0, mask=mask)

@torch.fx.wrap 
def optimized_pad_2d(x):
    # Get input dimensions
    n_batch, n_channels, height, width = x.shape
    
    # Output dimensions after padding
    out_height = height + 1
    out_width = width + 1
    
    # Calculate grid dimensions
    grid_x = out_height
    grid_y = out_width  
    grid_z = (n_batch * n_channels + 1023) // 1024  # Handle batch/channel blocks
    
    # Create output tensor
    out = torch.empty((n_batch, n_channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    optimized_pad_2d_kernel[(grid_x, grid_y, grid_z)](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels, 
        height=height,
        width=width,
        BLOCK_SIZE=1024
    )
    
    return out

def replacement_func():
    return optimized_pad_2d