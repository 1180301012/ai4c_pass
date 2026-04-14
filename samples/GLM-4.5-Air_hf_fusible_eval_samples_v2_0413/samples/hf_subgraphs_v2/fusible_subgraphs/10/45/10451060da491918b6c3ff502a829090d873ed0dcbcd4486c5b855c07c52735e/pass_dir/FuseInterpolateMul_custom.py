import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_2,)  # We only need in_2 for the multiplication part

# Optimized fused interpolate + multiplication kernel
@triton.jit
def fused_interpolate_mul_kernel(
    in_ptr,                          # Input tensor pointer [1, 128, 1, 4]
    mul_ptr,                         # Multiplication tensor pointer [1, 128, 64, 128]
    out_ptr,                         # Output tensor pointer [1, 128, 64, 128]
    batch,                           # Batch size = 1
    channels,                        # Channels = 128
    in_height,                       # Input height = 1
    in_width,                        # Input width = 4
    out_height,                      # Output height = 64
    out_width,                       # Output width = 128
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program processes one output position (H, W)
    if pid >= out_height * out_width:
        return
    
    # Calculate output position
    out_pos = pid
    out_h = out_pos // out_width
    out_w = out_pos % out_width
    
    # Process all channels
    for c in range(channels):
        # Calculate input coordinates for bilinear interpolation
        # Map output [0, 63] x [0, 127] to input [0, 0] x [0, 3]
        in_h = 0.0  # Since input height is 1, we always interpolate at height 0
        in_w = (out_w * (in_width - 1)) / (out_width - 1) if out_width > 1 else 0.0
        
        # Get integer coordinates and weights
        in_h0 = int(in_h)
        in_h1 = min(in_h0 + 1, in_height - 1)
        in_w0 = int(in_w)
        in_w1 = min(in_w0 + 1, in_width - 1)
        
        # Calculate interpolation weights
        if in_height > 1:
            h_weight_in = in_h - in_h0
            h_weight_out = 1.0 - h_weight_in
        else:
            h_weight_in = 0.0
            h_weight_out = 1.0
        
        w_weight_in = in_w - in_w0
        w_weight_out = 1.0 - w_weight_in
        
        # Load input values for bilinear interpolation
        # Input layout: [batch, channels, height, width]
        base_idx = c * (batch * in_height * in_width)
        
        # Four corners for bilinear interpolation
        val_00 = tl.load(in_ptr + base_idx + (in_h0 * in_width + in_w0))
        val_01 = tl.load(in_ptr + base_idx + (in_h0 * in_width + in_w1)) if in_w1 < in_width else val_00
        val_10 = tl.load(in_ptr + base_idx + (in_h1 * in_width + in_w0)) if in_h1 < in_height else val_00
        val_11 = tl.load(in_ptr + base_idx + (in_h1 * in_width + in_w1)) if (in_h1 < in_height and in_w1 < in_width) else val_00
        
        # Bilinear interpolation
        interpolated_val = (h_weight_out * w_weight_out * val_00 +
                           h_weight_out * w_weight_in * val_01 +
                           h_weight_in * w_weight_out * val_10 +
                           h_weight_in * w_weight_in * val_11)
        
        # Load multiplication value
        out_idx = c * (batch * out_height * out_width) + (out_h * out_width + out_w)
        mul_val = tl.load(mul_ptr + out_idx)
        
        # Apply multiplication and store result
        result = mul_val * interpolated_val
        tl.store(out_ptr + out_idx, result)

@torch.fx.wrap
def fused_interpolate_mul(in_2, tmp_2):
    # Get tensor shapes
    in_shape = tmp_2.shape  # [1, 128, 1, 4]
    mul_shape = in_2.shape  # [1, 128, 64, 128]
    
    # Output shape: same as multiplication input [1, 128, 64, 128]
    out_shape = mul_shape
    out = torch.empty(out_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Kernel launch parameters
    batch = in_shape[0]
    channels = in_shape[1]
    in_height = in_shape[2]
    in_width = in_shape[3]
    out_height = mul_shape[2]
    out_width = mul_shape[3]
    
    BLOCK_SIZE = 256  # Number of output pixels to process per program
    
    # Launch kernel
    grid = (out_height * out_width,)  # One program per output position
    
    fused_interpolate_mul_kernel[grid](
        tmp_2,
        in_2,
        out,
        batch,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_interpolate_mul