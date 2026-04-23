import torch
import triton
import triton.language as tl

# The constant from the model
DIV_CONSTANT = 11.313708498984761


@triton.jit
def fuse_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: square(relu(x / c))
    Optimized by:
    1. Using multiplication by inverse instead of division
    2. Fusing relu + square into single pass
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute relu(x) = max(0, x)
    relu_x = tl.where(x > 0, x, 0.0)
    
    # Compute square(relu(x)) = relu(x)^2
    squared = relu_x * relu_x
    
    # Multiply by scale = 1/c^2 (use multiplication instead of division)
    out = squared * scale
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def fuse_div_relu_square(x):
    """Fused implementation of: square(relu(x / c))"""
    n_elements = x.numel()
    
    # Precompute scale = 1/c^2 using multiplication instead of division
    scale = 1.0 / (DIV_CONSTANT * DIV_CONSTANT)
    
    # Use 2D tiling for better cache utilization on 2D/3D tensors
    # Get tensor shape
    shape = x.shape
    rank = len(shape)
    
    if rank == 3:
        # Shape: [B, H, W]
        B, H, W = shape
        # Use 2D grid with tile size 32x32
        block_h = 32
        block_w = 32
        grid_h = (H + block_h - 1) // block_h
        grid_w = (W + block_w - 1) // block_w
        
        out = torch.empty_like(x)
        
        # Launch kernel with 3D grid: (batch, height_tiles, width_tiles)
        fuse_div_relu_square_kernel[(B, grid_h, grid_w)](
            x,
            out,
            H * W,  # elements per batch
            scale,
            block_h * block_w,
        )
    else:
        # Fallback to 1D for other ranks
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        
        fuse_div_relu_square_kernel[(num_programs,)](
            x,
            out,
            n_elements,
            scale,
            BLOCK_SIZE,
        )
    
    return out


@torch.fx.wrap
def fuse_div_relu_square_wrapper(x):
    return fuse_div_relu_square(x)


def pattern(in_0):
    """
    Match the pattern: div -> relu -> square
    """
    tmp_0 = in_0 / DIV_CONSTANT
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fuse_div_relu_square_wrapper