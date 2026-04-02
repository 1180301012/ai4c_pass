import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(x):
    """Match Interpolate pattern exactly as in model.py"""
    # Match the exact pattern: torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    return torch.nn.functional.interpolate(x, (64, 128), None, 'bilinear', False)

# Argument extraction function
def replacement_args(x):
    # Extract arguments needed for the fused interpolate + multiply operation
    return (x, (64, 128), None, 'bilinear', False)

# Triton kernel for fused bilinear interpolate + element-wise multiplication
@triton.jit
def fused_interpolate_multiply_kernel(
    x_ptr, multiplier_ptr, out_ptr,
    x_batch, x_channels, x_height, x_width,
    out_height, out_width,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    # Get program IDs for 2D grid
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Calculate output spatial coordinates
    out_c = tl.program_id(2)  # channel dimension
    out_h = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    out_w = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Calculate bounds
    mask_h = out_h < out_height
    mask_w = out_w < out_width
    mask_c = out_c < x_channels
    mask_b = True  # single batch
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    if (mask_c and mask_b):
        # Calculate input coordinates using bilinear interpolation
        for i in range(BLOCK_SIZE_Y):
            for j in range(BLOCK_SIZE_X):
                if (mask_h[i] and mask_w[j]):
                    # Normalize coordinates to [0, 1] range
                    norm_h = tl.cast(out_h[i], tl.float32) / max(tl.cast(out_height, tl.float32) - 1, 1)
                    norm_w = tl.cast(out_w[j], tl.float32) / max(tl.cast(out_width, tl.float32) - 1, 1)
                    
                    # Scale to input dimensions
                    src_h = norm_h * max(tl.cast(x_height, tl.float32) - 1, 1)
                    src_w = norm_w * max(tl.cast(x_width, tl.float32) - 1, 1)
                    
                    # Get integer and fractional parts
                    src_h_floor = tl.floor(src_h).to(tl.int32)
                    src_h_frac = src_h - tl.cast(src_h_floor, tl.float32)
                    src_w_floor = tl.floor(src_w).to(tl.int32)
                    src_w_frac = src_w - tl.cast(src_w_floor, tl.float32)
                    
                    # Clamp to valid range
                    src_h_floor = tl.maximum(tl.minimum(src_h_floor, x_height - 1), 0)
                    src_w_floor = tl.maximum(tl.minimum(src_w_floor, x_width - 1), 0)
                    
                    # Get neighbors
                    src_h_ceil = tl.minimum(src_h_floor + 1, x_height - 1)
                    src_w_ceil = tl.minimum(src_w_floor + 1, x_width - 1)
                    
                    # Load 4 neighbors with bounds checking
                    idx_00 = out_c * x_height * x_width + src_h_floor * x_width + src_w_floor
                    idx_01 = out_c * x_height * x_width + src_h_floor * x_width + src_w_ceil
                    idx_10 = out_c * x_height * x_width + src_h_ceil * x_width + src_w_floor
                    idx_11 = out_c * x_height * x_width + src_h_ceil * x_width + src_w_ceil
                    
                    # Load interpolated values
                    val_00 = tl.load(x_ptr + idx_00, mask=True, other=0.0).to(tl.float32)
                    val_01 = tl.load(x_ptr + idx_01, mask=True, other=0.0).to(tl.float32)
                    val_10 = tl.load(x_ptr + idx_10, mask=True, other=0.0).to(tl.float32)
                    val_11 = tl.load(x_ptr + idx_11, mask=True, other=0.0).to(tl.float32)
                    
                    # Bilinear interpolation
                    top = val_00 * (1 - src_w_frac) + val_01 * src_w_frac
                    bottom = val_10 * (1 - src_w_frac) + val_11 * src_w_frac
                    interpolated = top * (1 - src_h_frac) + bottom * src_h_frac
                    
                    # Load multiplier value
                    mult_idx = out_c * out_height * out_width + out_h[i] * out_width + out_w[j]
                    multiplier_val = tl.load(multiplier_ptr + mult_idx, mask=True, other=1.0).to(tl.float32)
                    
                    # Apply multiplication
                    acc[i, j] = interpolated * multiplier_val
                else:
                    acc[i, j] = 0.0
    
    # Store result if valid
    if mask_c and mask_b:
        out_idx = out_c * out_height * out_width + out_h * out_width + out_w
        tl.store(out_ptr + out_idx.reshape((BLOCK_SIZE_Y, BLOCK_SIZE_X)), 
                acc.reshape((BLOCK_SIZE_Y, BLOCK_SIZE_X)), 
                mask=mask_h.reshape((BLOCK_SIZE_Y, 1)) & mask_w.reshape((1, BLOCK_SIZE_X)))

# Optimized kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap  
def fused_interpolate_multiply(x, multiplier_tensor, size=(64, 128), mode='bilinear', align_corners=False):
    """Fused bilinear interpolate + element-wise multiplication using Triton"""
    
    # Get input dimensions
    x_batch, x_channels, x_height, x_width = x.shape
    
    # Use target size from size argument
    out_height, out_width = size
    
    # Create output tensor
    out = torch.empty((x_batch, x_channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions (2D spatial + 1D channel)
    grid_x = (out_width + 31) // 32  # BLOCK_SIZE_X = 32
    grid_y = (out_height + 31) // 32  # BLOCK_SIZE_Y = 32
    grid_c = x_channels
    
    # Launch kernel with 3D grid
    fused_interpolate_multiply_kernel[(grid_x, grid_y, grid_c)](
        x_ptr=x,
        multiplier_ptr=multiplier,
        out_ptr=out,
        x_batch=x_batch,
        x_channels=x_channels,
        x_height=x_height,
        x_width=x_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE_X=32,
        BLOCK_SIZE_Y=32
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_interpolate_multiply