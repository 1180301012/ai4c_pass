import torch
import triton
import triton.language as tl

def pattern(x):
    # Three identical max_pool2d operations
    tmp_1 = torch.nn.functional.max_pool2d(x, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(x, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(x, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    # Return all three results for fusion
    return tmp_1, tmp_2, tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_max_pool_kernel(
    x_ptr,
    out1_ptr, out2_ptr, out3_ptr,
    n_channels, height, width,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate program indices
    c = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = tl.program_id(2) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create coordinate grid
    h_grid, w_grid = tl.meshgrid(h, w)
    
    # Calculate x indices for max pooling (kernel 5x5, padding 2 on each side)
    x_h = h_grid + tl.arange(-2, 3)
    x_w = w_grid + tl.arange(-2, 3)
    x_h_grid, x_w_grid = tl.meshgrid(x_h, x_w)
    
    # Create masks
    h_mask = (x_h_grid >= 0) & (x_h_grid < height)
    w_mask = (x_w_grid >= 0) & (x_w_grid < width)
    mask = h_mask & w_mask
    
    # Load input data for all positions
    x_grid = x_h_grid * width + x_w_grid + c * height * width
    x_val = tl.load(x_ptr + x_grid, mask=mask, other=-float('inf'))
    
    # Find max values (resulting shape is the same as input due to padding)
    max_val = tl.max(x_val, axis=0)
    
    # Calculate output indices
    out_idx = c * height * width + h_grid * width + w_grid
    
    # Store results to all three output locations
    tl.store(out1_ptr + out_idx, max_val, mask=(h_grid < height) & (w_grid < width))
    tl.store(out2_ptr + out_idx, max_val, mask=(h_grid < height) & (w_grid < width))
    tl.store(out3_ptr + out_idx, max_val, mask=(h_grid < height) & (w_grid < width))

@torch.fx.wrap
def fused_max_pool(x):
    n_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
    
    # Create output tensors
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    out3 = torch.empty_like(x)
    
    # Set up block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Calculate grid dimensions
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (
        n_channels,
        grid_h,
        grid_w
    )
    
    # Launch kernel
    fused_max_pool_kernel[grid](
        x_ptr=x,
        out1_ptr=out1,
        out2_ptr=out2,
        out3_ptr=out3,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out1, out2, out3

def replacement_func():
    return fused_max_pool