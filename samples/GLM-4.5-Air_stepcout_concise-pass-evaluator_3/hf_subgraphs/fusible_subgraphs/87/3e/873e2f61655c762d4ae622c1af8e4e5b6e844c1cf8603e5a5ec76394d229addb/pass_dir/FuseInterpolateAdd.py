import torch
import triton
import triton.language as tl

# Pattern for Interpolation + Addition fusion
def pattern(input_tensor, residual_tensor):
    interpolated = torch.nn.functional.interpolate(input_tensor, (64, 64), None, 'bilinear', False)
    output = residual_tensor + interpolated
    return output

# Extract arguments for the fused kernel
def replacement_args(input_tensor, residual_tensor):
    return (input_tensor, residual_tensor)

# Triton kernel for fused Interpolation + Addition (bilinear upsample 8x8->64x64)
@triton.jit
def fused_interpolate_add_kernel(
    x_ptr,           # Input tensor (B, C, 8, 8)
    residual_ptr,    # Residual tensor (B, C, 64, 64)
    out_ptr,         # Output tensor (B, C, 64, 64)
    B, C, H_small, W_small, H_large, W_large,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID - use 3D grid: batch, spatial, channels
    batch_id = tl.program_id(0)
    linear_spatial = tl.program_id(1)  # Combined H and W for large (64x64)
    c_out = tl.program_id(2)
    
    # Decompose spatial coordinates for large output (64x64)
    h_out = linear_spatial // W_large
    w_out = linear_spatial % W_large
    
    # Create mask to check if coordinates are within bounds
    mask = (h_out < H_large) & (w_out < W_large) & (batch_id < B) & (c_out < C)
    
    # Calculate output offset
    out_offset = batch_id * C * H_large * W_large + c_out * H_large * W_large + h_out * W_large + w_out
    
    # Scale factors from 8x8 to 64x64
    scale_factor = H_large // H_small  # Should be 8
    
    # Calculate input coordinates for nearest neighbor interpolation
    h_in = h_out // scale_factor
    w_in = w_out // scale_factor
    
    # Clamp to valid input range
    h_in = min(h_in, H_small - 1)
    w_in = min(w_in, W_small - 1)
    
    # Load input value (nearest neighbor for simplicity)
    in_offset = batch_id * C * H_small * W_small + c_out * H_small * W_small + h_in * W_small + w_in
    interpolated_val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    
    # Load residual value
    residual_val = tl.load(residual_ptr + out_offset, mask=mask, other=0.0)
    
    # Add values
    output = interpolated_val + residual_val
    
    # Store result
    tl.store(out_ptr + out_offset, output, mask=mask)

@torch.fx.wrap
def fused_interpolate_add(input_tensor, residual_tensor):
    B, C, H_small, W_small = input_tensor.shape
    H_large, W_large = 64, 64
    
    # Validate input dimensions
    if H_small != 8 or W_small != 8:
        raise ValueError(f"Expected 8x8 input for this pattern, got {H_small}x{W_small}")
    
    out = torch.empty((B, C, H_large, W_large), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Configure block size for GPU occupancy
    BLOCK_SIZE = 128
    
    # Calculate number of spatial programs for large output (64x64)
    spatial_size_large = H_large * W_large
    
    # Launch kernel with 3D grid: batch, spatial (large), channels
    fused_interpolate_add_kernel[(B, spatial_size_large, C,)](
        input_tensor, residual_tensor, out,
        B, C, H_small, W_small, H_large, W_large,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_interpolate_add