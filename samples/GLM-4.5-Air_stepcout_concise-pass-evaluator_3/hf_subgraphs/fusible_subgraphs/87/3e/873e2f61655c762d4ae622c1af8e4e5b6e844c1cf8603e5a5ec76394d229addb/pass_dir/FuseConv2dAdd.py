import torch
import triton
import triton.language as tl

# Pattern for Conv2D + Addition fusion
def pattern(conv_input, conv_weight, add_input):
    conv_output = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    output = add_input + conv_output
    return output

# Extract arguments for the fused kernel
def replacement_args(conv_input, conv_weight, add_input):
    return (conv_input, conv_weight, add_input)

# Triton kernel for fused Conv2D + Addition
@triton.jit
def fused_conv2d_add_kernel(
    x_ptr,           # Input to conv2d (B, C_in, H, W)
    weight_ptr,      # Conv2d weight (C_out, C_in/k, KH, KW)
    residual_ptr,    # Residual to add (B, C_out, H, W)
    out_ptr,         # Output (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride, padding, dilation, groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID - use 3D grid: batch, spatial, channel
    batch_id = tl.program_id(0)
    linear_spatial = tl.program_id(1)  # Combined H and W
    c_out = tl.program_id(2)
    
    # Decompose spatial coordinates
    h_out = linear_spatial // W
    w_out = linear_spatial % W
    
    # Create mask to check if coordinates are within bounds
    mask = (h_out < H) & (w_out < W) & (batch_id < B) & (c_out < C_out)
    
    # Calculate offsets for weight
    weight_offset = c_out * (C_in // groups) * 1 * 1
    weight = tl.load(weight_ptr + weight_offset)
    
    # Calculate output spatial offset and convolution result
    base_offset = batch_id * C_out * H * W + c_out * H * W + h_out * W + w_out
    conv_val = 0.0
    
    # Perform 1x1 convolution - just multiply with weight and accumulate
    for c_in in range(C_in // groups):
        input_offset = batch_id * C_in * H * W + (c_in * groups + c_out % groups) * H * W + h_out * W + w_out
        val = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
        conv_val += val * weight
    
    # Add residual
    residual_val = tl.load(residual_ptr + base_offset, mask=mask, other=0.0)
    output = conv_val + residual_val
    
    # Store result
    tl.store(out_ptr + base_offset, output, mask=mask)

@torch.fx.wrap
def fused_conv2d_add(conv_input, conv_weight, residual):
    B, C_in, H_in, W_in = conv_input.shape
    C_out, _, _, _ = conv_weight.shape
    
    out = torch.empty((B, C_out, H_in, W_in), device=conv_input.device, dtype=conv_input.dtype)
    
    # Configure block sizes for GPU occupancy
    BLOCK_SIZE = 128  # Adjust based on performance testing
    
    # Calculate number of spatial programs (H * W)
    spatial_size = H_in * W_in
    
    # Launch kernel with 3D grid: batch, spatial, channels
    fused_conv2d_add_kernel[(B, spatial_size, C_out,)](
        conv_input, conv_weight, residual, out,
        B, C_in, C_out, H_in, W_in,
        1, 0, 1, 1,  # stride, padding, dilation, groups
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_conv2d_add